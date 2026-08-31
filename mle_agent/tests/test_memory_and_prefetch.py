"""Offline checks for phase-aware memory, bootstrap prefetch, schemas, and blobs."""
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from mle_agent.harness.agent_tools import AGENT_TOOLS, AgentToolRuntime, BootstrapState
from mle_agent.harness.blobstore import INLINE_MAX_CHARS, BlobStore
from mle_agent.harness.bootstrap_prefetch import (
    BootstrapDigest,
    SourceSummary,
    curate_markdown,
    curate_python,
)
from mle_agent.harness.config import load_config

_CHANGED_MODEL_SOURCE = "value = 2\nprint('changed candidate')\n"
from mle_agent.harness.memory import (
    PhaseAwareMemory,
    approx_tokens,
    mark_experiment_start,
)
from mle_agent.harness.provider import (
    PROVIDER_CAPABILITIES,
    LangChainModelClient,
    UnsupportedProviderError,
    resolve_langchain_settings,
)
from mle_agent.harness.tool_schemas import (
    TARGET_COMPONENTS,
    ReflectionResult,
    RunModelArgs,
    tool_parameters,
)


# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

def test_every_tool_schema_is_derived_and_provider_safe():
    for tool in AGENT_TOOLS:
        parameters = tool["parameters"]
        assert parameters["type"] == "object", tool["name"]
        assert tool["description"].strip(), tool["name"]
        # $ref/$defs are inlined so no provider has to resolve them.
        encoded = json.dumps(parameters)
        assert "$ref" not in encoded, tool["name"]
        assert "$defs" not in encoded, tool["name"]
        # Pydantic's per-field titles are pure restatement and are resent on
        # every call in a persistent conversation.
        assert '"title"' not in encoded, tool["name"]


def test_nested_models_inline_into_a_self_contained_schema():
    schema = next(t for t in AGENT_TOOLS if t["name"] == "record_task_context")["parameters"]
    splits = schema["properties"]["data_splits"]
    assert set(splits["required"]) == {"train", "validation", "test"}
    feature = schema["properties"]["feature_engineering_context"]
    assert "baseline_fields" in feature["properties"]


def test_target_components_have_exactly_one_definition():
    run_model = next(t for t in AGENT_TOOLS if t["name"] == "run_model")["parameters"]
    backlog = next(
        t for t in AGENT_TOOLS if t["name"] == "record_research_backlog"
    )["parameters"]
    from mle_agent.harness.agent_tools import RUN_MODEL_COMPONENTS

    assert tuple(run_model["properties"]["target_component"]["enum"]) == TARGET_COMPONENTS
    backlog_enum = backlog["properties"]["candidates"]["items"]["properties"][
        "target_component"
    ]["enum"]
    assert tuple(backlog_enum) == TARGET_COMPONENTS
    assert RUN_MODEL_COMPONENTS == TARGET_COMPONENTS


def test_invalid_tool_arguments_are_rejected_with_field_level_detail():
    config = load_config()
    with tempfile.TemporaryDirectory() as temp:
        trial = Path(temp)
        (trial / "model.py").write_text("value = 1\n", encoding="utf-8")
        runtime = AgentToolRuntime(trial, config, BootstrapState(required=False))
        result = json.loads(runtime.dispatch("run_model", {
            "hypothesis": "x", "reasoning": "y", "target_component": "not-a-component",
        }))
        assert result["error"] == "TOOL_ARGUMENT_INVALID"
        assert result["invalid_fields"][0]["field"] == "target_component"


def test_gates_are_reported_before_argument_validation():
    """A malformed payload sent too early must still report the prerequisite.

    Telling the agent its enum is wrong, when the real problem is that it has not
    read the sources yet, sends it to fix the wrong thing.
    """
    config = load_config()
    with tempfile.TemporaryDirectory() as temp:
        trial = Path(temp)
        (trial / "model.py").write_text("value = 1\n", encoding="utf-8")
        runtime = AgentToolRuntime(trial, config, BootstrapState())
        result = json.loads(runtime.dispatch("run_model", {"target_component": "bogus"}))
        assert result["error"] == "BOOTSTRAP_REQUIRED"


def test_run_model_defaults_materialise_from_the_schema():
    parsed = RunModelArgs.model_validate({
        "hypothesis": "h", "reasoning": "r", "target_component": "loss",
    }).model_dump()
    assert parsed["rollback_plan"] == "retain incumbent"
    assert parsed["execution_class"] == "normal"
    assert parsed["seed"] is None


# ---------------------------------------------------------------------------
# Phase-aware memory
# ---------------------------------------------------------------------------

def _bootstrap_state() -> BootstrapState:
    state = BootstrapState(required=False)
    state.fully_read_paths = {"/kit/README.md", "/kit/evaluate.py"}
    state.baseline_reproduced = True
    state.baseline_metrics = {"primary": 0.6016}
    state.literature_queries = ["bpr loss"]
    state.data_inspected = True
    state.environment_inventory = {"frameworks": ["torch"]}
    state.task_context = {"target_label": "long_view", "metrics": ["GAUC", "nDCG@5"]}
    state.research_backlog = [{"hypothesis": "BPR", "target_component": "loss"}]
    state.scored_experiments = [{"target_component": "loss"}]
    return state


def _long_history() -> list:
    messages: list = [{"role": "user", "content": "SYSTEM PROMPT"}]
    for index in range(6):
        messages.append(AIMessage(
            content="",
            tool_calls=[{
                "id": f"c{index}", "name": "read_file",
                "args": {"path": "/kit/README.md"}, "type": "tool_call",
            }],
        ))
        messages.append(ToolMessage(
            content="X" * 30_000, tool_call_id=f"c{index}", name="read_file"
        ))
    for iteration in (1, 2):
        messages.append(mark_experiment_start(f"Begin experiment {iteration} now."))
        messages.append(AIMessage(
            content="",
            tool_calls=[{
                "id": f"r{iteration}", "name": "run_model",
                "args": {"hypothesis": "h"}, "type": "tool_call",
            }],
        ))
        messages.append(ToolMessage(
            content="primary=0.60", tool_call_id=f"r{iteration}", name="run_model"
        ))
    return messages


def test_bootstrap_is_never_compacted_while_reads_are_still_being_credited():
    memory = PhaseAwareMemory(
        token_budget=10, experiment_tail_budget=10, ledger_limit=8
    )
    _, outcome = memory.compact(
        _long_history(), phase="bootstrap", bootstrap_state=_bootstrap_state(),
        incumbent={}, ledger=[],
    )
    assert not outcome.compacted
    assert "bootstrap in progress" in outcome.reason


def test_compaction_keeps_the_system_prompt_and_replaces_bootstrap_reads():
    memory = PhaseAwareMemory(
        token_budget=100, experiment_tail_budget=8000, ledger_limit=8
    )
    messages, outcome = memory.compact(
        _long_history(), phase="experiment", bootstrap_state=_bootstrap_state(),
        incumbent={"incumbent_validation_primary": 0.6016},
        ledger=[{"iteration": 1, "primary": 0.60}],
    )
    assert outcome.compacted
    assert outcome.tokens_after < outcome.tokens_before / 5

    text = [str(m.content) for m in messages]
    # Rule 1: the system prompt survives verbatim, at the front.
    assert text[0] == "SYSTEM PROMPT"
    # Rules 2 and 3: the 180k characters of raw reads are gone, replaced by a
    # checkpoint that still names what was read.
    assert not any("X" * 1000 in entry for entry in text)
    assert any("/kit/README.md" in entry for entry in text)
    # Rule 5: incumbent and backlog come back from trusted state.
    briefing = next(entry for entry in text if "authoritative" in entry)
    assert "0.6016" in briefing
    assert "long_view" in briefing
    assert "BPR" in briefing


def test_compaction_never_orphans_a_tool_result_from_its_call():
    memory = PhaseAwareMemory(
        token_budget=10, experiment_tail_budget=200, ledger_limit=8
    )
    messages, outcome = memory.compact(
        _long_history(), phase="experiment", bootstrap_state=_bootstrap_state(),
        incumbent={}, ledger=[],
    )
    assert outcome.compacted
    pending: set[str] = set()
    for message in messages:
        if isinstance(message, ToolMessage):
            assert message.tool_call_id in pending, "tool result without its call"
        pending |= {c["id"] for c in (getattr(message, "tool_calls", None) or [])}


def test_compaction_is_skipped_while_within_budget():
    memory = PhaseAwareMemory(
        token_budget=10_000_000, experiment_tail_budget=1000, ledger_limit=8
    )
    original = _long_history()
    _, outcome = memory.compact(
        original, phase="experiment", bootstrap_state=_bootstrap_state(),
        incumbent={}, ledger=[],
    )
    assert not outcome.compacted
    assert "within budget" in outcome.reason


def test_compaction_emits_an_auditable_event():
    memory = PhaseAwareMemory(
        token_budget=100, experiment_tail_budget=8000, ledger_limit=8
    )
    _, outcome = memory.compact(
        _long_history(), phase="experiment", bootstrap_state=_bootstrap_state(),
        incumbent={}, ledger=[],
    )
    event = outcome.as_event()
    assert event["event_type"] == "memory_compaction"
    assert event["tokens_saved"] > 0
    assert event["dropped_kinds"]["ToolMessage"] >= 1


# ---------------------------------------------------------------------------
# Bootstrap prefetch and curation
# ---------------------------------------------------------------------------

def test_curate_python_keeps_the_contract_and_drops_bodies():
    source = '''"""Module docstring."""
FIELDS = ["user_id", "video_id"]

def score(user_ids, labels, k=5) -> dict:
    """Return the official metrics."""
    total = 0
    for value in labels:
        total += value * 12345
    return {"primary": total}
'''
    curated = curate_python(source)
    assert "Module docstring." in curated
    assert 'FIELDS = [' in curated
    assert "def score(user_ids, labels, k=5) -> dict:" in curated
    assert "Return the official metrics." in curated
    assert "12345" not in curated, "function bodies must be elided"


def test_curate_markdown_keeps_every_numeric_line():
    source = (
        "# Task\n"
        "Some prose with no numbers at all here.\n"
        "The published primary score is 0.5946 over 5 seeds.\n"
        "- convergence epsilon 0.002\n"
    )
    curated = curate_markdown(source)
    assert "0.5946" in curated
    assert "0.002" in curated
    assert "# Task" in curated
    assert "no numbers at all" not in curated


def test_digest_provenance_reports_the_curation_ratio():
    digest = BootstrapDigest(
        sources=[SourceSummary(path="/kit/a.py", kind="python", total_chars=1000,
                               summary="x" * 100)],
        baseline={"success": True},
    )
    provenance = digest.provenance()
    assert provenance["prefetched_by"] == "harness"
    assert provenance["raw_chars"] == 1000
    assert provenance["curated_chars"] == 100
    assert provenance["baseline_reproduced"] is True
    assert "read_file" in digest.render()


# ---------------------------------------------------------------------------
# Blob sidecar
# ---------------------------------------------------------------------------

def test_small_payloads_stay_inline_and_large_ones_offload():
    with tempfile.TemporaryDirectory() as temp:
        store = BlobStore(Path(temp))
        small = store.reference("short output", field="output")
        assert small["output"] == "short output"
        assert small["output_storage"] == "inline"

        big = "Y" * (INLINE_MAX_CHARS + 5000)
        large = store.reference(big, field="output")
        assert large["output_storage"] == "blob"
        assert large["output_chars"] == len(big)
        assert store.get(large["output_sha256"]) == big
        assert len(large["output_preview"]) < len(big)


def test_identical_payloads_are_stored_once():
    with tempfile.TemporaryDirectory() as temp:
        store = BlobStore(Path(temp))
        text = "Z" * 5000
        first, second = store.put(text), store.put(text)
        assert first == second
        assert len(list(Path(temp).rglob("*"))) == 2  # one fan-out dir, one blob


# ---------------------------------------------------------------------------
# Provider capability registry
# ---------------------------------------------------------------------------

def test_unregistered_provider_is_refused_rather_than_silently_degraded():
    try:
        LangChainModelClient(
            model="gpt-5.6-terra", api_key="k", provider="totally-unknown",
            model_instance=object(),
        )
        raise AssertionError("unknown provider must be refused")
    except UnsupportedProviderError as exc:
        assert "capability registry" in str(exc)


def test_provider_without_cache_reporting_bills_every_input_token():
    """Gemini does not report cache reads; billing them as cached would under-count."""
    google = PROVIDER_CAPABILITIES["google_genai"]
    assert google.reports_cache_read_tokens is False
    assert google.structured_output_method() == "json_schema"
    anthropic = PROVIDER_CAPABILITIES["anthropic"]
    assert anthropic.strict_tools is False
    assert anthropic.structured_output_method() == "function_calling"


def test_agent_model_accepts_any_registered_provider():
    import os

    saved = {key: os.environ.get(key) for key in ("AGENT_MODEL", "ANTHROPIC_API_KEY")}
    try:
        os.environ["AGENT_MODEL"] = "anthropic:gpt-5.4"
        os.environ["ANTHROPIC_API_KEY"] = "test-key"
        settings = resolve_langchain_settings()
        assert settings["provider"] == "anthropic"
        assert settings["capabilities"].api_key_env == "ANTHROPIC_API_KEY"

        os.environ["AGENT_MODEL"] = "cohere:gpt-5.4"
        try:
            resolve_langchain_settings()
            raise AssertionError("unregistered provider must be refused")
        except UnsupportedProviderError:
            pass
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


# ---------------------------------------------------------------------------
# Structured reflection
# ---------------------------------------------------------------------------

class _FakeStructuredModel:
    """Minimal stand-in for a LangChain model supporting structured output."""

    def __init__(self, parsed, parsing_error=None):
        self._parsed = parsed
        self._parsing_error = parsing_error
        self.method = None
        self.include_raw = None

    def with_structured_output(self, schema, method=None, include_raw=False):
        self.method = method
        self.include_raw = include_raw
        return self

    def invoke(self, messages, **kwargs):
        raw = AIMessage(content="reflection text")
        raw.usage_metadata = {"input_tokens": 100, "output_tokens": 20}
        raw.response_metadata = {"finish_reason": "stop", "id": "resp_1"}
        return {"raw": raw, "parsed": self._parsed, "parsing_error": self._parsing_error}


def test_structured_reflection_uses_json_schema_and_keeps_the_raw_response():
    parsed = ReflectionResult(
        reflection="BPR did not help.", hypothesis_status="not_supported",
        suggested_next="Try a sequence feature.",
    )
    fake = _FakeStructuredModel(parsed)
    client = LangChainModelClient(
        model="gpt-5.6-terra", api_key="k", max_cost_usd=10.0, model_instance=fake,
    )
    result, response = client.complete_structured([], ReflectionResult, max_tokens=256)
    assert fake.method == "json_schema"
    assert fake.include_raw is True
    assert result.hypothesis_status == "not_supported"
    assert result.hypothesis_supported is False
    # The raw response still carries usage and cost, so the audit trail is intact
    # even though the model's text was never appended to the conversation.
    assert response.input_tokens == 100
    assert response.provider_metadata["call_cost_usd"] > 0
    assert client.spent_usd > 0


def test_structured_reflection_parse_failure_returns_none_without_raising():
    """A malformed reflection is a research event, not a provider outage.

    Raising would trip the retry path and buy a second billed call for something
    a retry cannot fix.
    """
    fake = _FakeStructuredModel(None, parsing_error=ValueError("bad json"))
    client = LangChainModelClient(
        model="gpt-5.6-terra", api_key="k", max_cost_usd=10.0, model_instance=fake,
    )
    result, response = client.complete_structured([], ReflectionResult)
    assert result is None
    assert "bad json" in response.provider_metadata["structured_output_error"]


def test_structured_output_still_respects_the_cost_cap():
    from mle_agent.harness.provider import CostLimitError

    fake = _FakeStructuredModel(None)
    client = LangChainModelClient(
        model="gpt-5.6-terra", api_key="k", max_cost_usd=0.0000001,
        model_instance=fake,
    )
    try:
        client.complete_structured([{"role": "user", "content": "x" * 10000}],
                                   ReflectionResult)
        raise AssertionError("cost cap must be enforced on the structured path too")
    except CostLimitError:
        pass



# ---------------------------------------------------------------------------
# End-to-end: prefetched bootstrap
# ---------------------------------------------------------------------------

class _PrefetchedBootstrapClient:
    """Records the whole conversation and answers with the two remaining calls."""

    def __init__(self, paths):
        self.paths = paths
        self.calls = 0
        self.seen_messages = []

    def complete(self, messages, tools=None, max_tokens=4096):
        from mle_agent.harness.provider import LLMResponse, ToolCall
        from mle_agent.tests.test_agent_sdk import (
            _task_context_payload, _valid_research_backlog,
        )

        self.calls += 1
        self.seen_messages.append(list(messages))
        if self.calls == 1:
            return LLMResponse(
                text="Recording context and backlog from the harness digest.",
                tool_calls=[
                    ToolCall(id="ctx", name="record_task_context",
                             input=_task_context_payload(self.paths)),
                    ToolCall(id="backlog", name="record_research_backlog",
                             input={"candidates": _valid_research_backlog()}),
                ],
                stop_reason="tool_use", input_tokens=10, output_tokens=4,
            )
        return LLMResponse(
            text="Bootstrap complete.", tool_calls=[], stop_reason="end_turn",
            input_tokens=5, output_tokens=2,
        )

    def add_response_to_history(self, messages, response):
        messages.append({"role": "assistant", "content": response.text or ""})

    def add_tool_results_to_history(self, messages, tool_calls, outputs):
        for call, output in zip(tool_calls, outputs):
            messages.append({"role": "user", "content": output})


def test_prefetched_bootstrap_completes_without_the_model_reading_anything():
    """The mechanical bootstrap costs zero model calls and still satisfies the gate."""
    import mle_agent.harness.agent_tools as agent_tools_module
    from mle_agent.research_agent.agent import ResearchAgent

    config = load_config()
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        starter, trial = root / "starter", root / "trial"
        starter.mkdir()
        trial.mkdir()
        (starter / "README.md").write_text(
            "# Task\nPrimary score is 0.6016 on validation.\nProse without digits.\n",
            encoding="utf-8",
        )
        (starter / "baseline.py").write_text(
            '"""Official FM baseline."""\nMODEL = "fm"\n\n'
            "def run_fm(seed):\n    hidden = 987654\n    return hidden\n",
            encoding="utf-8",
        )
        (starter / "evaluate.py").write_text("metric = 'primary'\n", encoding="utf-8")
        (starter / "data.py").write_text(
            "FIELDS = ['user_id', 'video_id', 'author_id', 'tab', 'dur_bucket']\n",
            encoding="utf-8",
        )
        (starter / "ablation_features.py").write_text(
            "RESULT = '13 static fields: no gain'\n", encoding="utf-8"
        )
        (trial / "model.py").write_text(
            "import argparse, json\n"
            "ap = argparse.ArgumentParser()\n"
            "ap.add_argument('--data_dir')\n"
            "ap.add_argument('--seed')\n"
            "ap.add_argument('--prediction-path')\n"
            "ap.add_argument('--trial-config')\n"
            "a = ap.parse_args()\n"
            "open(a.prediction_path, 'w').write('row_id,user_id,video_id,score\\n')\n"
            "print(json.dumps({'GAUC': 0.6674, 'nDCG@5': 0.5358, 'primary': 0.6016}))\n",
            encoding="utf-8",
        )
        config.BASELINE_ROOT = starter
        config.AGENT_BOOTSTRAP_PREFETCH = True

        paths = [str((starter / name).resolve()) for name in (
            "README.md", "baseline.py", "evaluate.py", "data.py", "ablation_features.py",
        )] + [str((trial / "model.py").resolve())]

        original = agent_tools_module.inspect_train_valid_data
        original_scorer = agent_tools_module.score_validation_predictions
        agent_tools_module.inspect_train_valid_data = lambda _c: {"train": {"rows": 1}}
        agent_tools_module.score_validation_predictions = (
            lambda *a, **k: __import__(
                "mle_agent.harness.evaluation", fromlist=["ScoredPredictions"]
            ).ScoredPredictions(
                metrics={"GAUC": 0.6674, "nDCG@5": 0.5358, "primary": 0.6016}, rows=1
            )
        )
        try:
            client = _PrefetchedBootstrapClient(paths)
            agent = ResearchAgent(
                config, client=client, provider_retry_delay_s=0,
                bootstrap_state=BootstrapState(),
            )
            result = agent.run_bootstrap(trial, max_turns=4)
        finally:
            agent_tools_module.inspect_train_valid_data = original
            agent_tools_module.score_validation_predictions = original_scorer

        assert result.success, result.error
        # The whole read/EDA/environment/baseline phase cost no model calls; the
        # agent spent its turns only on the judgement the harness cannot make.
        assert client.calls <= 2, client.calls

        state = agent.bootstrap_state
        assert state.complete
        assert state.baseline_reproduced
        # The read gate is satisfied by harness provenance for every required source.
        assert len(state.fully_read_paths) >= 6

        digest_text = "\n".join(
            str(m.get("content") if isinstance(m, dict) else getattr(m, "content", ""))
            for m in client.seen_messages[0]
        )
        assert "Harness bootstrap digest" in digest_text
        assert "0.6016" in digest_text
        assert "Official FM baseline." in digest_text
        # Curation, not truncation: the contract survives, the body does not.
        assert "def run_fm(seed):" in digest_text
        assert "987654" not in digest_text
        assert "Prose without digits." not in digest_text



def test_compaction_never_grows_the_context_even_when_forced():
    """A prefetched bootstrap is already compact; rebuilding it would cost more."""
    memory = PhaseAwareMemory(
        token_budget=10, experiment_tail_budget=10, ledger_limit=8
    )
    short = [
        {"role": "user", "content": "SYSTEM PROMPT"},
        {"role": "user", "content": "digest"},
    ]
    messages, outcome = memory.compact(
        short, phase="bootstrap_complete", bootstrap_state=_bootstrap_state(),
        incumbent={}, ledger=[], force=True,
    )
    assert not outcome.compacted
    assert "would grow the context" in outcome.reason
    assert len(messages) == 2



def test_blob_sidecar_ships_with_the_log_that_references_it():
    """finalize_run copies only the run directory.

    A blob store outside it leaves every reference in the delivered log dangling,
    which is precisely the audit trail the run log exists to provide.
    """
    from mle_agent.harness.logger import RunLogger
    from mle_agent.research_agent.agent import ResearchAgent

    config = load_config()
    with tempfile.TemporaryDirectory() as temp:
        logger = RunLogger(Path(temp), "testrun")
        agent = ResearchAgent(
            config, client=object(), bootstrap_state=BootstrapState(required=False),
            blob_dir=logger.logs_dir / "blobs",
        )
        digest = agent._blobs.put("Q" * 5000)
        blob = logger.logs_dir / "blobs" / digest[:2] / digest
        assert blob.is_file()
        # The reference recorded in the event resolves relative to the log file.
        assert (logger.llm_events_path.parent / f"blobs/{digest[:2]}/{digest}").is_file()



# ---------------------------------------------------------------------------
# Cross-run experiment history
# ---------------------------------------------------------------------------

def _history_record(**kw):
    from mle_agent.research_agent.experiment_history import ExperimentRecord
    base = dict(
        run_id="r1", iteration=1, hypothesis="BPR pairwise loss", target_component="loss",
        outcome="rejected", primary=0.4994, delta_vs_incumbent=-0.1047,
        incumbent_primary=0.6041, candidate_fingerprint="deadbeef",
    )
    base.update(kw)
    return ExperimentRecord(**base)


def test_history_round_trips_and_survives_a_missing_file():
    from mle_agent.research_agent.experiment_history import append_record, load_history

    with tempfile.TemporaryDirectory() as temp:
        path = Path(temp) / "history.json"
        # Fails open: a run must not be blocked by an absent operational log.
        assert load_history(path)["rows"] == []
        append_record(_history_record(), path)
        append_record(_history_record(iteration=2, outcome="improved", primary=0.6041), path)
        rows = load_history(path)["rows"]
        assert len(rows) == 2
        assert rows[0]["outcome"] == "rejected"


def test_corrupt_history_fails_open_rather_than_stopping_a_run():
    from mle_agent.research_agent.experiment_history import load_history

    with tempfile.TemporaryDirectory() as temp:
        path = Path(temp) / "history.json"
        path.write_text("{not json", encoding="utf-8")
        history = load_history(path)
        assert history["rows"] == []
        assert history["corrupt"] is True


def test_only_non_improving_fingerprints_become_a_gate():
    from mle_agent.research_agent.experiment_history import (
        append_record, failed_fingerprints,
    )

    with tempfile.TemporaryDirectory() as temp:
        path = Path(temp) / "history.json"
        append_record(_history_record(candidate_fingerprint="bad1"), path)
        append_record(_history_record(
            outcome="improved", candidate_fingerprint="good1"), path)
        append_record(_history_record(
            outcome="failed", candidate_fingerprint="bad2"), path)
        gate = failed_fingerprints(path)
        assert gate == {"bad1", "bad2"}, gate


def test_prompt_rendering_never_crowds_out_the_negative_results():
    """Failures are the point of this file; a tail of wins must not bury them."""
    from mle_agent.research_agent.experiment_history import append_record, compact_for_prompt

    with tempfile.TemporaryDirectory() as temp:
        path = Path(temp) / "history.json"
        append_record(_history_record(
            hypothesis="SAME-USER BPR IS A DEAD END", candidate_fingerprint="x"), path)
        for index in range(40):
            append_record(_history_record(
                outcome="improved", hypothesis=f"win {index}",
                candidate_fingerprint=f"w{index}"), path)
        rendered = compact_for_prompt(path, limit=5)
        assert "SAME-USER BPR IS A DEAD END" in rendered
        assert '"total_recorded":41' in rendered


def test_compact_prompt_surfaces_a_suspect_implementation_diagnosis():
    """A retry must inherit the known bug, not merely a generic suspect label."""
    from mle_agent.research_agent.experiment_history import (
        SUSPECT, append_record, compact_for_prompt,
    )

    with tempfile.TemporaryDirectory() as temp:
        path = Path(temp) / "history.json"
        append_record(_history_record(
            outcome=SUSPECT,
            error="avalues() bound p to the negatives slot, creating negative-negative pairs",
        ), path)

        rendered = json.loads(compact_for_prompt(path))
        row = rendered["rows"][0]
        assert row["outcome"] == SUSPECT
        assert "negative-negative pairs" in row["implementation_diagnosis"]


def test_checked_in_legacy_history_has_backfilled_candidate_fingerprints():
    """Legacy rows participate in the exact-repeat gate after the migration."""
    from mle_agent.research_agent.experiment_history import (
        HISTORY_PATH, failed_fingerprints, load_history,
    )

    rows = load_history(HISTORY_PATH)["rows"]
    assert rows, "the checked-in migration fixture unexpectedly has no rows"
    for row in rows:
        fingerprint = row.get("candidate_fingerprint")
        assert isinstance(fingerprint, str) and len(fingerprint) == 64
        assert all(char in "0123456789abcdef" for char in fingerprint)

    gated = failed_fingerprints(HISTORY_PATH)
    assert rows[1]["candidate_fingerprint"] in gated  # suspect BPR implementation
    assert rows[2]["candidate_fingerprint"] in gated  # rejected DIN implementation


def test_an_exact_repeat_from_an_earlier_run_is_refused():
    """Identical code cannot score differently, so re-running it is pure waste."""
    from mle_agent.harness.agent_tools import semantic_model_fingerprint

    config = load_config()
    candidate = "value = 2\nprint('candidate')\n"
    with tempfile.TemporaryDirectory() as temp:
        # Fingerprint the candidate an earlier run would have measured.
        scratch = Path(temp) / "scratch"
        scratch.mkdir()
        (scratch / "model.py").write_text(candidate, encoding="utf-8")
        already_failed = semantic_model_fingerprint(scratch / "model.py")

        # This run starts from the unchanged inherited model, then writes that
        # same candidate — the realistic shape of a cross-run repeat.
        trial = Path(temp) / "trial"
        trial.mkdir()
        (trial / "model.py").write_text("value = 1\n", encoding="utf-8")
        runtime = AgentToolRuntime(
            trial, config, BootstrapState(required=False), iteration=1,
            failed_fingerprints={already_failed},
        )
        runtime.dispatch("write_file", {"path": "model.py", "content": candidate})
        result = json.loads(runtime.dispatch("run_model", {
            "hypothesis": "retry the same thing", "reasoning": "because",
            "target_component": "loss",
        }))
        assert result["success"] is False
        assert "already scored in an earlier" in result["error"], result["error"]


def test_history_is_written_from_validated_state_not_agent_claims():
    """The outcome label is derived from the measured score, not from the model."""
    from mle_agent.research_agent.experiment_history import load_history
    from mle_agent.research_agent.agent import ResearchAgent
    import mle_agent.research_agent.agent as agent_module

    config = load_config()
    with tempfile.TemporaryDirectory() as temp:
        path = Path(temp) / "history.json"
        try:
            agent = ResearchAgent(
                config, client=object(),
                bootstrap_state=BootstrapState(required=False), run_id="testrun",
                history_path=path,
            )
            # The agent "claims" success; the metrics say otherwise. Metrics win.
            # A competent-but-losing score, above the popularity rung, so this
            # exercises the "idea was tested and lost" label rather than the
            # broken-build one.
            agent._record_cross_run_history(
                iteration=4,
                chosen={"hypothesis": "great idea", "proposal": {"target_component": "loss"},
                        "candidate_fingerprint": "abc"},
                metrics={"primary": 0.59},
                reflection=None,
                incumbent_primary=0.604,
                error=None,
            )
        finally:
            pass
        row = load_history(path)["rows"][0]
        assert row["outcome"] == "rejected"
        assert row["delta_vs_incumbent"] == -0.014
        assert row["run_id"] == "testrun"


def test_in_run_ledger_retains_the_nested_proposal_component():
    """Compaction must not turn a scored component into an empty ledger field."""
    from mle_agent.research_agent.agent import ResearchAgent

    agent = ResearchAgent(
        load_config(), client=object(), bootstrap_state=BootstrapState(required=False),
    )
    agent._record_ledger_entry(
        iteration=2,
        chosen={
            "hypothesis": "Correct same-user BPR sampling",
            "proposal": {"target_component": "loss"},
        },
        metrics={"primary": 0.603},
        reflection=None,
        incumbent_primary=0.604,
    )

    assert agent._ledger[-1]["target_component"] == "loss"



def test_a_run_that_did_not_opt_in_never_writes_cross_run_memory():
    """Offline suites and smoke tests must not append to the shipped ledger."""
    from mle_agent.research_agent.agent import ResearchAgent
    from mle_agent.research_agent.experiment_history import HISTORY_PATH

    before = HISTORY_PATH.read_bytes() if HISTORY_PATH.is_file() else None
    config = load_config()
    agent = ResearchAgent(
        config, client=object(), bootstrap_state=BootstrapState(required=False),
    )
    assert agent._history_path is None
    agent._record_cross_run_history(
        iteration=1, chosen={"hypothesis": "junk", "proposal": {}},
        metrics={"primary": 0.9}, reflection=None, incumbent_primary=0.1, error=None,
    )
    after = HISTORY_PATH.read_bytes() if HISTORY_PATH.is_file() else None
    assert after == before, "a non-opted-in agent wrote to the real history file"



def test_a_near_chance_score_is_a_broken_build_not_a_refuted_idea():
    """The real failure mode this guards against.

    The 20260831_134234 run's same-user BPR scored GAUC 0.522 against 0.499 for
    random scoring — it learned essentially nothing. Recorded as "rejected" that
    would retire BPR, one of the README's own named unexplored directions, for
    every later run on the strength of a single bug.
    """
    from mle_agent.research_agent.experiment_history import SUSPECT, classify_outcome

    config = load_config()
    popularity = config.POPULARITY_PRIMARY
    incumbent = 0.604062

    # Below the trivial popularity ranker while holding every baseline feature.
    assert classify_outcome(0.499380, incumbent, popularity) == SUSPECT
    # A competent result that simply lost stays a settled negative.
    assert classify_outcome(0.596819, incumbent, popularity) == "rejected"
    assert classify_outcome(0.604900, incumbent, popularity) == "improved"
    assert classify_outcome(None, incumbent, popularity) == "failed"
    # The boundary itself counts as suspect: matching a popularity heuristic is
    # not evidence that a learned model works.
    assert classify_outcome(popularity, incumbent, popularity) == SUSPECT


def test_suspect_rows_are_flagged_as_open_but_still_gate_identical_code():
    from mle_agent.research_agent.experiment_history import (
        SUSPECT, append_record, compact_for_prompt, failed_fingerprints,
    )

    with tempfile.TemporaryDirectory() as temp:
        path = Path(temp) / "history.json"
        append_record(_history_record(
            hypothesis="same-user BPR pairwise loss", outcome=SUSPECT,
            primary=0.4994, candidate_fingerprint="bpr1"), path)
        rendered = compact_for_prompt(path)
        # The direction is advertised as still open...
        assert "still open" in rendered
        assert "implementation failure" in rendered
        # ...but identical code is still refused, because it cannot score
        # differently no matter why it failed the first time.
        assert "bpr1" in failed_fingerprints(path)


def test_reference_rungs_come_from_the_starter_kit_not_hardcoded():
    config = load_config()
    import json
    scores = json.loads(config.BASELINE_JSON.read_text())["scores"]
    assert config.POPULARITY_PRIMARY == scores["item_popularity"]["valid"]["primary"]
    assert config.RANDOM_PRIMARY == scores["random"]["valid"]["primary"]
    assert config.POPULARITY_PRIMARY < config.BASELINE_PRIMARY



def test_in_loop_reflection_still_populates_the_structured_fields():
    """The common case: the agent volunteers its reflection during the loop.

    The closing structured call only fires when it has not. If the in-loop text
    is not parsed, hypothesis_supported and suggested_next are lost on every
    well-behaved experiment -- and they are what the ledger and the cross-run
    history carry forward.
    """
    from mle_agent.research_agent.agent import AgentIterationResult

    import mle_agent.tests.recovery_demo as demo
    evidence = demo.run_recovery_scenario()
    row = evidence["event"]
    assert row["hypothesis_supported"] is True
    assert row["suggested_next"]



def test_the_experiment_boundary_does_not_depend_on_prompt_wording():
    """The boundary decides what is safe to drop, so it must be structural.

    It previously matched the literal opening line of iteration.md. Rewording
    that line would have silently reclassified every experiment message as
    discardable bootstrap traffic — no crash, no warning, total memory loss.
    """
    memory = PhaseAwareMemory(
        token_budget=100, experiment_tail_budget=8000, ledger_limit=8
    )

    def history(opening: str) -> list:
        out: list = [
            {"role": "user", "content": "SYSTEM PROMPT"},
            {"role": "user", "content": "bootstrap " + "b" * 30000},
        ]
        for index in (1, 2):
            out.append(mark_experiment_start(f"{opening} {index} now."))
            out.append(AIMessage(content="", tool_calls=[{
                "id": f"r{index}", "name": "run_model",
                "args": {}, "type": "tool_call"}]))
            out.append(ToolMessage(
                content="R" * 6000, tool_call_id=f"r{index}", name="run_model"))
        return out

    # Wording is now irrelevant; the marker carries the meaning.
    for opening in ("Begin experiment", "Start experiment", "Kick off run"):
        kept, outcome = memory.compact(
            history(opening), phase="experiment", bootstrap_state=_bootstrap_state(),
            incumbent={}, ledger=[],
        )
        assert outcome.compacted
        survivors = [m for m in kept if isinstance(m, (AIMessage, ToolMessage))]
        assert survivors, f"experiment work dropped for opening {opening!r}"


def test_unmarked_history_is_retained_rather_than_discarded():
    """With no marker at all, fail toward keeping work, not deleting it."""
    memory = PhaseAwareMemory(
        token_budget=100, experiment_tail_budget=8000, ledger_limit=8
    )
    messages: list = [
        {"role": "user", "content": "SYSTEM PROMPT"},
        {"role": "user", "content": "old bulk " + "p" * 30000},
        # The in-flight experiment is always the most recent traffic.
        {"role": "user", "content": "an unmarked experiment prompt"},
        AIMessage(content="", tool_calls=[{
            "id": "r1", "name": "run_model", "args": {}, "type": "tool_call"}]),
        ToolMessage(content="scored primary=0.604", tool_call_id="r1", name="run_model"),
    ]
    kept, _ = memory.compact(
        messages, phase="experiment", bootstrap_state=_bootstrap_state(),
        incumbent={}, ledger=[], force=True,
    )
    assert any("primary=0.604" in str(m.content) for m in kept)


def test_repeated_compaction_sawtooths_without_stacking_checkpoints():
    """Twelve iterations of growth: checkpoints must be replaced, not accumulated."""
    from mle_agent.harness.memory import _is_checkpoint

    memory = PhaseAwareMemory(
        token_budget=20000, experiment_tail_budget=8000, ledger_limit=8
    )
    messages: list = [
        {"role": "user", "content": "SYSTEM PROMPT"},
        {"role": "user", "content": "digest " + "d" * 20000},
    ]
    compactions = 0
    for iteration in range(1, 13):
        messages, outcome = memory.compact(
            messages, phase="experiment", bootstrap_state=_bootstrap_state(),
            incumbent={}, ledger=[{"iteration": iteration}],
        )
        compactions += int(outcome.compacted)
        messages = list(messages)
        messages.append(mark_experiment_start(f"Begin experiment {iteration} now."))
        messages.append(AIMessage(content="", tool_calls=[{
            "id": f"r{iteration}", "name": "run_model", "args": {},
            "type": "tool_call"}]))
        messages.append(ToolMessage(
            content="T" * 9000, tool_call_id=f"r{iteration}", name="run_model"))
        checkpoints = sum(
            1 for m in messages
            if isinstance(m, BaseMessage) and _is_checkpoint(m)
            and not _is_checkpoint(m, "experiment_start")
        )
        assert checkpoints <= 2, f"checkpoints stacked to {checkpoints}"
    assert compactions >= 2, "a 12-iteration run must cross the budget more than once"



def test_a_near_chance_candidate_is_flagged_to_the_agent_during_the_run():
    """The live run must get the diagnostic, not just the cross-run history.

    In run 20260831_134234 the BPR candidate trained on (negative, negative)
    pairs and never saw a positive, scoring GAUC 0.522 against 0.499 for random.
    The agent was told only the score, so it concluded "the BPR objective is not
    competitive" and retired a named research direction over an indexing bug.
    """
    from mle_agent.harness.agent_tools import sanity_check_metrics

    config = load_config()
    broken = sanity_check_metrics(
        {"primary": 0.499380, "GAUC": 0.522289, "nDCG@5": 0.476472}, config
    )
    assert broken is not None
    assert broken["verdict"] == "SUSPECT_IMPLEMENTATION"
    assert "not evidence the hypothesis is wrong" in broken["why"]
    assert "inverted index" in " ".join(broken["common_causes"])
    # Reference rungs travel with the verdict so the agent can see the gap itself.
    assert broken["reference_rungs"]["item_popularity"] == round(
        config.POPULARITY_PRIMARY, 6
    )

    # A competent-but-losing result is not flagged: it really did test the idea.
    assert sanity_check_metrics({"primary": 0.596819, "GAUC": 0.660479}, config) is None
    assert sanity_check_metrics({"primary": 0.604062, "GAUC": 0.671090}, config) is None
    assert sanity_check_metrics(None, config) is None


def test_a_broken_candidate_does_not_consume_a_diversity_slot():
    """A bug must not spend the run's one budgeted attempt at a component."""
    import mle_agent.harness.agent_tools as tools

    config = load_config()
    with tempfile.TemporaryDirectory() as temp:
        trial = Path(temp)
        (trial / "model.py").write_text("value = 1\n", encoding="utf-8")
        state = BootstrapState(required=False)
        runtime = AgentToolRuntime(trial, config, state, iteration=1)
        runtime.dispatch("write_file", {
            "path": "model.py", "content": _CHANGED_MODEL_SOURCE,
        })

        original = tools.execute_model
        tools.execute_model = lambda *a, **k: SimpleNamespace(
            success=True,
            metrics={"GAUC": 0.5223, "nDCG@5": 0.4765, "primary": 0.4994},
            error=None, wall_seconds=1.0, seed=42, trial_config={},
            prediction_path="p", prediction_sha256="s", resource_usage={},
        )
        try:
            result = json.loads(runtime.dispatch("run_model", {
                "hypothesis": "BPR pairwise loss", "reasoning": "ranking aligned",
                "target_component": "loss",
            }))
        finally:
            tools.execute_model = original

        assert result["success"] is True
        assert result["sanity_check"]["verdict"] == "SUSPECT_IMPLEMENTATION"
        # 'loss' stays available for a correct implementation to test properly.
        assert state.scored_experiments == []



def test_a_broken_build_is_never_recorded_as_evidence_against_the_idea():
    """'not tested' and 'tested and lost' must stay distinguishable end to end.

    A two-valued hypothesis_supported forced a broken implementation to be logged
    as evidence that the hypothesis failed. That is how one indexing bug came to
    retire BPR — a direction the starter-kit README names as unexplored.
    """
    from mle_agent.harness.tool_schemas import ReflectionResult
    from mle_agent.research_agent.experiment_history import SUSPECT, classify_outcome

    config = load_config()
    not_tested = ReflectionResult(
        reflection="The candidate never saw a positive example.",
        hypothesis_status="not_tested",
        implementation_diagnosis="Training pairs were built from an inverted index.",
        suggested_next="Rebuild the pair sampler and retry BPR.",
    )
    # Not collapsed to False: the idea was never put to the test.
    assert not_tested.hypothesis_supported is None

    # A subtle bug can score above the popularity rung, so the agent's own
    # diagnosis has to be honoured as well as the score heuristic.
    assert classify_outcome(
        0.590, 0.604, config.POPULARITY_PRIMARY, "not_tested"
    ) == SUSPECT
    assert classify_outcome(
        0.590, 0.604, config.POPULARITY_PRIMARY, "not_supported"
    ) == "rejected"
    # The score still wins where it is unambiguous: an improvement is an
    # improvement whatever the agent says about it.
    assert classify_outcome(
        0.610, 0.604, config.POPULARITY_PRIMARY, "not_tested"
    ) == "improved"



def test_change_scope_is_reported_so_a_bundled_result_is_not_mistaken_for_one_idea():
    """Run 20260831_134234's first experiment replaced model.py in one write.

    It bundled three feature families, and its own reflection conceded the result
    supported them only "as a bundle". The agent should learn that while it can
    still act on it, not the reader of the log afterwards.
    """
    from mle_agent.harness.agent_tools import summarize_change

    parent = "\n".join(f"line {index}" for index in range(60))

    targeted = summarize_change(parent, parent.replace("line 7\n", "line 7 tweaked\n"))
    assert targeted["is_rewrite"] is False
    assert "attribution_warning" not in targeted

    rewrite = summarize_change(parent, "\n".join(f"new {i}" for i in range(60)))
    assert rewrite["is_rewrite"] is True
    assert "cannot be attributed" in rewrite["attribution_warning"]

    scattered = parent
    for index in range(0, 60, 6):
        scattered = scattered.replace(f"line {index}\n", f"line {index} x\n")
    many = summarize_change(parent, scattered)
    assert many["hunks"] > 6
    assert "separate regions" in many["attribution_warning"]

    # A first candidate with no meaningful parent is not called a rewrite.
    assert summarize_change("", "anything\n")["is_rewrite"] is False


def test_run_model_reports_the_change_against_the_inherited_parent():
    import mle_agent.harness.agent_tools as tools

    config = load_config()
    with tempfile.TemporaryDirectory() as temp:
        trial = Path(temp)
        (trial / "model.py").write_text(
            "\n".join(f"x = {i}" for i in range(40)) + "\n", encoding="utf-8"
        )
        runtime = AgentToolRuntime(
            trial, config, BootstrapState(required=False), iteration=1,
        )
        runtime.dispatch("write_file", {
            "path": "model.py",
            "content": "\n".join(f"y = {i}" for i in range(40)) + "\n",
        })
        original = tools.execute_model
        tools.execute_model = lambda *a, **k: SimpleNamespace(
            success=True, metrics={"GAUC": 0.67, "nDCG@5": 0.54, "primary": 0.605},
            error=None, wall_seconds=1.0, seed=42, trial_config={},
            prediction_path="p", prediction_sha256="s", resource_usage={},
        )
        try:
            result = json.loads(runtime.dispatch("run_model", {
                "hypothesis": "h", "reasoning": "r", "target_component": "model",
            }))
        finally:
            tools.execute_model = original
        change = result["change_summary"]
        assert change["is_rewrite"] is True
        assert change["parent_lines"] == 40
        assert "attribution_warning" in change



def test_every_global_name_the_harness_modules_reference_actually_resolves():
    """Catch a NameError on a code path the offline suite never executes.

    Removing the Google ADK runtime deleted an import block that also supplied
    ``AgentIterationResult`` to agent_main. Nothing failed at import time and no
    test called ``main()``, so the break only surfaced in a live run, after
    bootstrap had already been paid for. Every module-level name a function
    references is checked here rather than only the ones the tests happen to hit.
    """
    import ast
    import builtins
    import importlib

    modules = [
        "mle_agent.harness.agent_main",
        "mle_agent.harness.agent_tools",
        "mle_agent.harness.memory",
        "mle_agent.harness.provider",
        "mle_agent.harness.bootstrap_prefetch",
        "mle_agent.harness.blobstore",
        "mle_agent.harness.tool_schemas",
        "mle_agent.research_agent.agent",
        "mle_agent.research_agent.experiment_history",
        "mle_agent.tests.recovery_demo",
    ]
    problems: list[str] = []
    for name in modules:
        module = importlib.import_module(name)
        tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
        known = set(vars(module)) | set(dir(builtins))

        # Only top-level functions and methods. Walking one of those already
        # covers the bodies of any functions nested inside it, and analysing a
        # nested function on its own would flag every closure variable it reads
        # from the enclosing scope.
        roots: list[ast.AST] = []
        for top in tree.body:
            if isinstance(top, (ast.FunctionDef, ast.AsyncFunctionDef)):
                roots.append(top)
            elif isinstance(top, ast.ClassDef):
                roots.extend(
                    child for child in top.body
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
                )

        for node in roots:
            # Names bound anywhere inside the function, however indirectly.
            local: set[str] = set()
            for inner in ast.walk(node):
                if isinstance(inner, ast.Name) and isinstance(inner.ctx, ast.Store):
                    local.add(inner.id)
                elif isinstance(inner, ast.arg):
                    local.add(inner.arg)
                elif isinstance(inner, (ast.Import, ast.ImportFrom)):
                    local |= {(a.asname or a.name).split(".")[0] for a in inner.names}
                elif isinstance(inner, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    local.add(inner.name)
                elif isinstance(inner, ast.ExceptHandler) and inner.name:
                    local.add(inner.name)
                elif isinstance(inner, ast.comprehension):
                    for target in ast.walk(inner.target):
                        if isinstance(target, ast.Name):
                            local.add(target.id)
                elif isinstance(inner, (ast.Global, ast.Nonlocal)):
                    local |= set(inner.names)

            for inner in ast.walk(node):
                if (
                    isinstance(inner, ast.Name)
                    and isinstance(inner.ctx, ast.Load)
                    and inner.id not in known
                    and inner.id not in local
                ):
                    problems.append(
                        f"{name}.{node.name} references undefined '{inner.id}' "
                        f"(line {inner.lineno})"
                    )
    assert not problems, "unresolved names:\n  " + "\n  ".join(sorted(set(problems)))


def main() -> None:
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"  PASS  {test.__name__}")
    print(f"\n{len(tests)} tests passed.")


if __name__ == "__main__":
    main()
