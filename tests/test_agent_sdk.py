"""Offline checks for the single-agent runner and its safety boundaries."""
import json
import sys
import tempfile
from io import StringIO
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness.agent_main import _converged, _log_row
import harness.agent_tools as agent_tools_module
from harness.agent_tools import AgentToolRuntime, BootstrapState
from harness.config import load_config
from harness.console import RunConsole
from harness.data_view import (
    EXPECTED_TEST_ROWS,
    EXPECTED_TRAIN_ROWS,
    EXPECTED_VALID_ROWS,
    classify_date,
)
from harness.hooks import PostFileSaveHook, run_post_file_save_hooks
from harness.logger import RunLogger
from harness.provider import (
    LLMClient,
    LLMResponse,
    OpenAICompatClient,
    ToolCall,
)
from harness.tools import exec_write_file
from harness.validator import scan_candidate_source, validate_row
from research_agent.agent import ResearchAgent, _console_reasoning_line
from research_agent.prompts import render_prompt


def test_convergence_requires_three_completed_no_gain_experiments():
    assert not _converged([0.6016, 0.6016, 0.6016], 0.002, 3)
    assert _converged([0.6016, 0.6016, 0.6017, 0.6017], 0.002, 3)
    assert not _converged([0.6016, 0.6017, 0.6050, 0.6050], 0.002, 3)


def test_test_split_scanner_catches_aliases_and_computed_keys():
    assert scan_candidate_source("data = load(path)\nx = data['test']")
    assert scan_candidate_source("x = splits['te' + 'st']")
    assert scan_candidate_source("x = splits.get('test')")
    assert not scan_candidate_source("x = splits['train']\ny = splits['valid']")


def test_candidate_write_cannot_escape_to_prefix_sibling():
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp) / "trial"
        root.mkdir()
        result = exec_write_file("../trial_evil/escaped.py", "bad", root)
        assert result.startswith("ERROR")
        assert not (Path(temp) / "trial_evil" / "escaped.py").exists()


def test_general_post_file_save_hook_uses_path_matcher_and_file_placeholder():
    with tempfile.TemporaryDirectory() as temp:
        target = Path(temp) / "generated" / "nested" / "example.txt"
        target.parent.mkdir(parents=True)
        target.write_text("saved", encoding="utf-8")
        hook = PostFileSaveHook(
            name="read-saved-file",
            path_glob="generated/**/*.txt",
            command=(
                sys.executable,
                "-c",
                "import pathlib,sys; print(pathlib.Path(sys.argv[1]).read_text())",
                "{file}",
            ),
        )

        results = run_post_file_save_hooks(target, (hook,))

        assert len(results) == 1
        assert results[0].success
        assert results[0].output == "saved"


def test_model_post_file_save_hook_reports_syntax_error_immediately():
    with tempfile.TemporaryDirectory() as temp:
        trial = Path(temp) / "experiment_workspace" / "run_001" / "trial_001"
        trial.mkdir(parents=True)

        result = exec_write_file("model.py", "def broken(:\n    pass\n", trial)

        assert result.startswith("FAILED:")
        assert "PostFileSave model-py-compile" in result
        assert "SyntaxError" in result
        assert trial.joinpath("model.py").exists()


def test_model_post_file_save_hook_stays_silent_after_valid_save():
    with tempfile.TemporaryDirectory() as temp:
        trial = Path(temp) / "experiment_workspace" / "run_001" / "trial_001"
        trial.mkdir(parents=True)

        result = exec_write_file("model.py", "value = 1\n", trial)

        assert result == "OK: wrote 10 bytes to model.py"


def test_run_model_skips_after_failed_save_until_model_is_repaired():
    config = load_config()
    with tempfile.TemporaryDirectory() as temp:
        trial = Path(temp) / "experiment_workspace" / "run_001" / "trial_001"
        trial.mkdir(parents=True)
        trial.joinpath("model.py").write_text("value = 1\n", encoding="utf-8")
        runtime = AgentToolRuntime(trial, config, BootstrapState(required=False))
        execute_calls: list[Path] = []
        original_execute_model = agent_tools_module.execute_model

        def fake_execute_model(candidate_dir, _config):
            execute_calls.append(candidate_dir)
            return agent_tools_module.ModelExecution(
                True,
                {"GAUC": 0.61, "nDCG@5": 0.60, "primary": 0.605},
                "",
                None,
                0.01,
            )

        agent_tools_module.execute_model = fake_execute_model
        try:
            failed_save = runtime.dispatch(
                "write_file", {"path": "model.py", "content": "def broken(:\n"}
            )
            skipped = json.loads(runtime.dispatch("run_model", {
                "hypothesis": "Exercise the failed-save gate",
                "reasoning": "A queued execution must not run invalid Python.",
            }))

            assert failed_save.startswith("FAILED:")
            assert not skipped["success"]
            assert skipped["skipped"]
            assert skipped["wall_seconds"] == 0.0
            assert skipped["error"].startswith("SKIPPED:")
            assert execute_calls == []

            repaired_save = runtime.dispatch(
                "write_file", {"path": "model.py", "content": "value = 2\n"}
            )
            executed = json.loads(runtime.dispatch("run_model", {
                "hypothesis": "Exercise the repaired-save gate",
                "reasoning": "A successful save must re-enable execution.",
            }))

            assert repaired_save.startswith("OK:")
            assert executed["success"]
            assert execute_calls == [trial]
        finally:
            agent_tools_module.execute_model = original_execute_model


def test_fixed_organizer_date_splits_and_counts_are_pinned():
    assert classify_date(20220408) == "train"
    assert classify_date(20220421) == "train"
    assert classify_date(20220422) == "valid"
    assert classify_date(20220428) == "valid"
    assert classify_date(20220429) == "test"
    assert classify_date(20220508) == "test"
    assert classify_date(20220509) == "outside"
    assert (EXPECTED_TRAIN_ROWS, EXPECTED_VALID_ROWS, EXPECTED_TEST_ROWS) == (
        1_141_112, 124_909, 170_588
    )


_CHANGED_SCORING_MODEL = """import argparse, json
ap = argparse.ArgumentParser()
ap.add_argument('--data_dir')
ap.parse_args()
experiment_variant = 'semantic-change'
print(json.dumps({'GAUC': 0.61, 'nDCG@5': 0.60, 'primary': 0.605}))
"""


def _write_and_run_calls(call_prefix: str, hypothesis: str, reasoning: str):
    return [
        ToolCall(
            id=f"{call_prefix}_write",
            name="write_file",
            input={"path": "model.py", "content": _CHANGED_SCORING_MODEL},
        ),
        ToolCall(
            id=f"{call_prefix}_run",
            name="run_model",
            input={"hypothesis": hypothesis, "reasoning": reasoning},
        ),
    ]


def test_run_model_rejects_unchanged_and_comment_only_candidates():
    config = load_config()
    with tempfile.TemporaryDirectory() as temp:
        trial = Path(temp)
        initial = "value = 1\n"
        trial.joinpath("model.py").write_text(initial, encoding="utf-8")
        runtime = AgentToolRuntime(trial, config, BootstrapState(required=False))
        payload = {"hypothesis": "Change behavior", "reasoning": "Test the guard."}

        unchanged = json.loads(runtime.dispatch("run_model", payload))
        assert not unchanged["success"]
        assert not unchanged["candidate_changed"]
        assert "semantically unchanged" in unchanged["error"]

        runtime.dispatch("write_file", {
            "path": "model.py",
            "content": "# only a comment changed\nvalue = 1\n",
        })
        comment_only = json.loads(runtime.dispatch("run_model", payload))
        assert not comment_only["success"]
        assert not comment_only["candidate_changed"]


def test_run_model_accepts_a_semantically_changed_candidate():
    config = load_config()
    with tempfile.TemporaryDirectory() as temp:
        trial = Path(temp)
        trial.joinpath("model.py").write_text("value = 1\n", encoding="utf-8")
        runtime = AgentToolRuntime(trial, config, BootstrapState(required=False))
        runtime.dispatch("write_file", {
            "path": "model.py",
            "content": _CHANGED_SCORING_MODEL,
        })
        result = json.loads(runtime.dispatch("run_model", {
            "hypothesis": "Change executable model code",
            "reasoning": "A semantic edit must be measured.",
        }))
        assert result["success"]
        assert result["candidate_changed"]
        assert result["metrics"]["primary"] == 0.605


def _task_context_payload(source_paths: list[str]) -> dict:
    return {
        "task_objective": "Improve within-user long-view ranking on KuaiRand.",
        "target_label": "long_view",
        "metrics": ["GAUC", "nDCG@5", "primary"],
        "data_splits": {
            "train": "20220408-20220421",
            "validation": "20220422-20220428",
            "test": "20220429-20220508; hidden and unavailable during development",
        },
        "baseline": "Official factorization-machine validation baseline.",
        "evaluation_protocol": ["Train on train and score validation only."],
        "hard_constraints": ["Never access the hidden test split."],
        "known_dead_ends": ["Comment-only edits."],
        "promising_directions": ["Evidence-backed ranking objectives."],
        "candidate_contract": ["Accept --data_dir and print JSON metrics."],
        "source_paths": source_paths,
    }


def test_bootstrap_gate_rejects_premature_context_edits_and_runs():
    config = load_config()
    with tempfile.TemporaryDirectory() as temp:
        trial = Path(temp)
        trial.joinpath("model.py").write_text("value = 1\n", encoding="utf-8")
        state = BootstrapState()
        runtime = AgentToolRuntime(trial, config, state)

        early_context = json.loads(runtime.dispatch(
            "record_task_context", _task_context_payload([])
        ))
        assert early_context["error"] == "BOOTSTRAP_SOURCES_INCOMPLETE"

        early_write = json.loads(runtime.dispatch(
            "write_file", {"path": "model.py", "content": "value = 2\n"}
        ))
        assert early_write["error"] == "BOOTSTRAP_REQUIRED"
        assert trial.joinpath("model.py").read_text(encoding="utf-8") == "value = 1\n"

        early_run = json.loads(runtime.dispatch("run_model", {
            "hypothesis": "Premature change",
            "reasoning": "Exercise the gate.",
        }))
        assert early_run["error"] == "BOOTSTRAP_REQUIRED"
        assert runtime.executions == []
        assert len(state.rejected_actions) == 3


def test_bootstrap_requires_a_literature_query_with_retrieved_evidence():
    config = load_config()
    with tempfile.TemporaryDirectory() as temp:
        trial = Path(temp)
        trial.joinpath("model.py").write_text("value = 1\n", encoding="utf-8")
        state = BootstrapState()
        runtime = AgentToolRuntime(trial, config, state)

        no_match = json.loads(runtime.dispatch("search_ml_literature", {
            "query": "zzzz_nonexistent_method_zzzz",
        }))
        assert not no_match["bootstrap_counted"]
        assert state.literature_queries == []

        matched = json.loads(runtime.dispatch("search_ml_literature", {
            "query": "within-user pairwise ranking loss",
        }))
        assert matched["bootstrap_counted"]
        assert matched["results"]
        assert state.literature_queries == ["within-user pairwise ranking loss"]


def test_task_docs_are_discovered_and_long_reads_are_explicitly_paginated():
    config = load_config()
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        starter = root / "starter"
        trial = root / "trial"
        starter.mkdir()
        trial.mkdir()
        readme_text = "0123456789" * 5
        starter.joinpath("README.md").write_text(readme_text, encoding="utf-8")
        starter.joinpath("evaluate.py").write_text("metric = 'primary'\n", encoding="utf-8")
        starter.joinpath("data.py").write_text("SPLITS = {}\n", encoding="utf-8")
        root.joinpath("README-outside.md").write_text("not discoverable", encoding="utf-8")
        trial.joinpath("model.py").write_text("value = 1\n", encoding="utf-8")
        config.BASELINE_ROOT = starter
        state = BootstrapState()
        runtime = AgentToolRuntime(trial, config, state)

        discovery = json.loads(runtime.dispatch("discover_task_docs", {}))
        assert discovery["primary_readme_path"] == str(starter.joinpath("README.md").resolve())
        discovered = {item["relative_path"] for item in discovery["documents"]}
        assert discovered == {"README.md", "evaluate.py", "data.py"}

        first = json.loads(runtime.dispatch("read_file", {
            "path": discovery["primary_readme_path"], "offset": 0, "max_chars": 17,
        }))
        assert first["content"] == readme_text[:17]
        assert not first["complete"]
        assert first["next_offset"] == 17

        offset = first["next_offset"]
        while offset is not None:
            page = json.loads(runtime.dispatch("read_file", {
                "path": discovery["primary_readme_path"],
                "offset": offset,
                "max_chars": 17,
            }))
            offset = page["next_offset"]
        assert str(starter.joinpath("README.md").resolve()) in state.fully_read_paths


def test_completed_task_context_persists_across_iteration_runtimes():
    config = load_config()
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        starter = root / "starter"
        first_trial = root / "trial_001"
        second_trial = root / "trial_002"
        starter.mkdir()
        first_trial.mkdir()
        second_trial.mkdir()
        starter.joinpath("README.md").write_text("Task documentation.\n", encoding="utf-8")
        starter.joinpath("evaluate.py").write_text("metric = 'primary'\n", encoding="utf-8")
        starter.joinpath("baseline.py").write_text("MODEL = 'fm'\n", encoding="utf-8")
        unread_source = starter / "data.py"
        unread_source.write_text("SPLITS = {}\n", encoding="utf-8")
        first_trial.joinpath("model.py").write_text("value = 1\n", encoding="utf-8")
        second_trial.joinpath("model.py").write_text("value = 1\n", encoding="utf-8")
        config.BASELINE_ROOT = starter
        state = BootstrapState()
        first_runtime = AgentToolRuntime(first_trial, config, state)

        discovery = json.loads(first_runtime.dispatch("discover_task_docs", {}))
        required_paths = [
            discovery["primary_readme_path"],
            str(starter.joinpath("baseline.py").resolve()),
            str(starter.joinpath("evaluate.py").resolve()),
            str(first_trial.joinpath("model.py").resolve()),
        ]
        for path in required_paths:
            response = json.loads(first_runtime.dispatch("read_file", {"path": path}))
            assert response["complete"]
        state.data_inspected = True
        state.literature_queries.append("ranking losses for recommender systems")
        state.baseline_reproduced = True
        state.baseline_metrics = {"GAUC": 0.6674, "nDCG@5": 0.5358, "primary": 0.6016}

        rejected_context = json.loads(first_runtime.dispatch(
            "record_task_context",
            _task_context_payload(required_paths + [str(unread_source.resolve())]),
        ))
        assert rejected_context["error"] == "TASK_CONTEXT_INVALID"
        assert rejected_context["cited_sources_not_fully_read"] == [
            str(unread_source.resolve())
        ]

        recorded = json.loads(first_runtime.dispatch(
            "record_task_context", _task_context_payload(required_paths)
        ))
        assert recorded["success"]
        assert state.complete

        second_runtime = AgentToolRuntime(second_trial, config, state)
        write_result = second_runtime.dispatch(
            "write_file", {"path": "model.py", "content": "value = 2\n"}
        )
        assert write_result.startswith("OK")
        assert second_runtime.bootstrap_state.task_context == state.task_context
        evidence = state.evidence()
        assert evidence["complete"]
        assert evidence["task_context"]["target_label"] == "long_view"


def test_openai_compat_round_trips_gemini_thought_signature_opaquely():
    signature = "encrypted-provider-value"
    native_tool_call = SimpleNamespace(
        id="function-call-1",
        function=SimpleNamespace(name="inspect_data", arguments="{}"),
        extra_content={"google": {"thought_signature": signature}},
    )
    native_message = SimpleNamespace(
        content="Inspect the fixed splits.",
        tool_calls=[native_tool_call],
    )
    completion = SimpleNamespace(
        choices=[SimpleNamespace(message=native_message, finish_reason="tool_calls")],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
    )
    completions = SimpleNamespace(create=lambda **kwargs: completion)
    client = object.__new__(OpenAICompatClient)
    client._client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    client._model = "gemini-3.5-flash-lite"

    response = client.complete([], tools=[{
        "name": "inspect_data",
        "description": "Inspect data.",
        "parameters": {"type": "object", "properties": {}},
    }])
    assert response.tool_calls[0].provider_metadata == {
        "extra_content": {"google": {"thought_signature": signature}}
    }

    history = []
    client.add_response_to_history(history, response)
    client.add_tool_results_to_history(history, response.tool_calls, ["result"])
    assert history[0]["tool_calls"][0]["extra_content"]["google"] == {
        "thought_signature": signature
    }
    assert history[1] == {
        "role": "tool",
        "tool_call_id": "function-call-1",
        "content": "result",
        "name": "inspect_data",
    }


def test_console_reasoning_prefers_summary_parses_reflection_and_has_fallback():
    summarized = LLMResponse(
        text=None,
        tool_calls=[ToolCall(id="1", name="inspect_data", input={})],
        stop_reason="tool_use",
        input_tokens=1,
        output_tokens=1,
        reasoning_summary="Inspect fixed splits before modeling.",
    )
    assert _console_reasoning_line(summarized) == "Inspect fixed splits before modeling."

    reflected = LLMResponse(
        text='{"reflection":"Validation improved enough to retain the candidate."}',
        tool_calls=[],
        stop_reason="end_turn",
        input_tokens=1,
        output_tokens=1,
    )
    assert _console_reasoning_line(reflected) == (
        "Validation improved enough to retain the candidate."
    )

    fallback = LLMResponse(
        text=None,
        tool_calls=[ToolCall(id="2", name="run_model", input={})],
        stop_reason="tool_use",
        input_tokens=1,
        output_tokens=1,
    )
    assert _console_reasoning_line(fallback) == "Calling local tool: run_model."


def test_structured_console_separates_agent_and_harness_without_dumping_code():
    stream = StringIO()
    run_console = RunConsole(stream=stream, use_color=False)
    run_console.harness("Run setup", provider="gemini / test-model")
    run_console.agent_reasoning(
        "Testing a ranking-aware loss because pointwise FM under-optimizes ordering.",
        phase="Experiment 1, turn 1/10",
        progress="Execution attempts=0",
        response_text="line one\nline two\nline three\nline four\nline five\nline six hidden",
        tool_names=["write_file"],
        stop_reason="tool_use",
        input_tokens=120,
        output_tokens=40,
        call_number=3,
        response_event_id="llm_000003",
        provider="gemini / test-model",
        latency_seconds=1.25,
        tool_summaries=["write_file(path=model.py, chars=35)"],
    )
    secret_code = "API_KEY_SHOULD_NOT_APPEAR = 'value'"
    run_console.agent_tool_call(
        "write_file", {"path": "model.py", "content": secret_code}
    )
    run_console.agent_tool_result("write_file", "OK: wrote 35 bytes to model.py")
    rendered = stream.getvalue()

    assert "--- Harness ---" in rendered
    assert "--- Agent ---" in rendered
    assert "Reasoning: Testing a ranking-aware loss" in rendered
    assert "LLM Call: #3 | llm_000003 | gemini / test-model | 1.25s" in rendered
    assert "LLM Response: line one" in rendered
    assert "Model Event: stop=tool_use; tokens in=120, out=40" in rendered
    assert "Actions Requested: write_file(path=model.py, chars=35)" in rendered
    assert "line five … [response truncated]" in rendered
    assert "line six hidden" not in rendered
    assert f"Tool Calling: write_file(path=model.py, chars={len(secret_code)})" in rendered
    assert "Tool Called: write_file" in rendered
    assert "Returned Result: OK: wrote 35 bytes to model.py" in rendered
    assert secret_code not in rendered
    assert "\033[" not in rendered

    colour_stream = StringIO()
    RunConsole(stream=colour_stream, use_color=True).harness("Colour check")
    assert "\033[" in colour_stream.getvalue()


class _FakeClient(LLMClient):
    def __init__(self):
        self.calls = 0

    def complete(self, messages, tools=None, max_tokens=4096):
        self.calls += 1
        if self.calls == 1:
            return LLMResponse(
                text=None,
                tool_calls=_write_and_run_calls(
                    "call_1",
                    "Smoke-test the unified execution loop",
                    "Verify plumbing without a provider call.",
                ),
                stop_reason="tool_use",
                input_tokens=10,
                output_tokens=5,
            )
        return LLMResponse(
            text='{"reflection":"The unified loop returned metrics.","hypothesis_supported":true,"suggested_next":"real model"}',
            tool_calls=[],
            stop_reason="end_turn",
            input_tokens=12,
            output_tokens=7,
        )

    def add_response_to_history(self, messages, response):
        messages.append({"role": "assistant", "content": response.text or ""})

    def add_tool_results_to_history(self, messages, tool_calls, outputs):
        messages.append({"role": "user", "content": "\n".join(outputs)})


class _BootstrapThenExecuteClient(_FakeClient):
    def __init__(self, readme: Path, baseline: Path, evaluate: Path, model: Path):
        super().__init__()
        self.readme = str(readme.resolve())
        self.baseline = str(baseline.resolve())
        self.evaluate = str(evaluate.resolve())
        self.model = str(model.resolve())

    def complete(self, messages, tools=None, max_tokens=4096):
        self.calls += 1
        if self.calls == 1:
            return LLMResponse(
                text="Discovering and reading the task sources.",
                tool_calls=[
                    ToolCall(id="discover", name="discover_task_docs", input={}),
                    ToolCall(
                        id="read_readme_1", name="read_file",
                        input={"path": self.readme, "offset": 0},
                    ),
                    ToolCall(
                        id="read_baseline", name="read_file",
                        input={"path": self.baseline, "offset": 0},
                    ),
                    ToolCall(
                        id="read_evaluate", name="read_file",
                        input={"path": self.evaluate, "offset": 0},
                    ),
                    ToolCall(
                        id="read_model", name="read_file",
                        input={"path": self.model, "offset": 0},
                    ),
                    ToolCall(id="inspect", name="inspect_data", input={}),
                    ToolCall(
                        id="literature", name="search_ml_literature",
                        input={"query": "within-user pairwise ranking loss"},
                    ),
                ],
                stop_reason="tool_use", input_tokens=10, output_tokens=4,
            )
        if self.calls == 2:
            return LLMResponse(
                text="Reading the remaining README page.",
                tool_calls=[ToolCall(
                        id="read_readme_2", name="read_file",
                    input={"path": self.readme, "offset": 200},
                )],
                stop_reason="tool_use", input_tokens=10, output_tokens=3,
            )
        if self.calls == 3:
            return LLMResponse(
                text="Reproducing the unchanged official baseline.",
                tool_calls=[ToolCall(
                    id="reproduce_baseline", name="reproduce_baseline", input={},
                )],
                stop_reason="tool_use", input_tokens=10, output_tokens=3,
            )
        if self.calls == 4:
            return LLMResponse(
                text="Recording the retained task context after baseline reproduction.",
                tool_calls=[ToolCall(
                    id="record_context", name="record_task_context",
                    input=_task_context_payload([
                        self.readme, self.baseline, self.evaluate, self.model,
                    ]),
                )],
                stop_reason="tool_use", input_tokens=10, output_tokens=4,
            )
        if self.calls == 5:
            return LLMResponse(
                text="Implementing and measuring the first hypothesis.",
                tool_calls=_write_and_run_calls(
                    "bootstrap_run",
                    "Test a semantic model change after bootstrap",
                    "The retained task context and ranking literature justify measuring it.",
                ),
                stop_reason="tool_use", input_tokens=10, output_tokens=5,
            )
        return LLMResponse(
            text='{"reflection":"Bootstrap and execution completed.","hypothesis_supported":true,"suggested_next":"continue"}',
            tool_calls=[], stop_reason="end_turn", input_tokens=8, output_tokens=4,
        )


def test_research_agent_completes_and_retains_the_bootstrap_before_running():
    config = load_config()
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        starter = root / "starter"
        trial = root / "trial"
        starter.mkdir()
        trial.mkdir()
        readme = starter / "README.md"
        baseline = starter / "baseline.py"
        evaluate = starter / "evaluate.py"
        model = trial / "model.py"
        readme.write_text("A" * 300, encoding="utf-8")
        baseline.write_text("MODEL = 'fm'\n", encoding="utf-8")
        evaluate.write_text("metric = 'primary'\n", encoding="utf-8")
        model.write_text(
            """import argparse, json
ap = argparse.ArgumentParser()
ap.add_argument('--data_dir')
ap.parse_args()
print(json.dumps({'GAUC': 0.6674, 'nDCG@5': 0.5358, 'primary': 0.6016}))
""",
            encoding="utf-8",
        )
        config.BASELINE_ROOT = starter
        config.AGENT_READ_MAX_CHARS = 200
        client = _BootstrapThenExecuteClient(readme, baseline, evaluate, model)
        original_inspector = agent_tools_module.inspect_train_valid_data
        agent_tools_module.inspect_train_valid_data = lambda _config: {
            "policy": "train and validation only",
            "train": {"rows": 1_141_112},
            "valid": {"rows": 124_909},
        }
        try:
            agent = ResearchAgent(
                config, client=client, provider_retry_delay_s=0,
                bootstrap_state=BootstrapState(),
            )
            bootstrap = agent.run_bootstrap(trial, max_turns=4)
            result = agent.run_iteration(1, trial, 0.6016, 0.6016, max_turns=1)
        finally:
            agent_tools_module.inspect_train_valid_data = original_inspector

        assert bootstrap.success
        assert bootstrap.metrics["primary"] == 0.6016
        assert result.success
        assert result.metrics["primary"] == 0.605
        assert agent.bootstrap_evidence["complete"]
        assert agent.bootstrap_evidence["task_context"]["target_label"] == "long_view"
        assert any(
            "TASK_CONTEXT_RECORDED_AND_RETAINED" in str(message.get("content"))
            for message in agent.messages
        )
        assert client.calls == 6


def test_single_agent_owns_a_persistent_tool_loop():
    config = load_config()
    source = """import argparse, json
ap = argparse.ArgumentParser()
ap.add_argument('--data_dir')
ap.parse_args()
print(json.dumps({'GAUC': 0.61, 'nDCG@5': 0.60, 'primary': 0.605}))
"""
    with tempfile.TemporaryDirectory() as temp:
        trial = Path(temp)
        (trial / "model.py").write_text(source, encoding="utf-8")
        client = _FakeClient()
        agent = ResearchAgent(
            config,
            client=client,
            provider_retry_delay_s=0,
            bootstrap_state=BootstrapState(required=False),
        )
        result = agent.run_iteration(1, trial, 0.6016, 0.6016, max_turns=1)
        assert result.success
        assert result.metrics["primary"] == 0.605
        assert result.hypothesis == "Smoke-test the unified execution loop"
        assert result.token_counts == {"input": 22, "output": 12}
        assert client.calls == 2
        assert result.reflection == "The unified loop returned metrics."
        assert {item["name"] for item in agent.prompt_evidence} == {
            "single_agent.md", "iteration.md"
        }
        row = _log_row(
            1, 0, result, "success", True, 0.6016, 0.2468,
            "--- diff", trial / "model.py",
        )
        assert validate_row(row) == []


def test_llm_trace_records_provider_responses_and_safe_tool_results():
    config = load_config()
    source = """import argparse, json
ap = argparse.ArgumentParser()
ap.add_argument('--data_dir')
ap.parse_args()
print(json.dumps({'GAUC': 0.61, 'nDCG@5': 0.60, 'primary': 0.605}))
"""
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        trial = root / "trial"
        trial.mkdir()
        (trial / "model.py").write_text(source, encoding="utf-8")
        logger = RunLogger(root, "trace-test")
        client = _FakeClient()
        agent = ResearchAgent(
            config,
            client=client,
            provider_retry_delay_s=0,
            bootstrap_state=BootstrapState(required=False),
            event_writer=logger.write_llm_event,
            provider_label="fake / test-model",
        )
        result = agent.run_iteration(1, trial, 0.6016, 0.6016, max_turns=1)
        totals = logger.running_totals()["llm_trace"]
        trace_path = logger.llm_events_path
        logger.close()

        assert result.success
        rows = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
        responses = [row for row in rows if row["event_type"] == "llm_response"]
        tool_results = [row for row in rows if row["event_type"] == "tool_result"]
        assert len(responses) == 2
        assert len(tool_results) == 2
        assert totals["llm_response"] == 2
        assert totals["tool_result"] == 2
        assert responses[0]["phase"] == "experiment_1_turn_1"
        assert responses[0]["stop_reason"] == "tool_use"
        assert responses[0]["provider"] == "fake / test-model"
        assert responses[1]["phase"] == "experiment_1_closing_reflection"
        assert "unified loop returned metrics" in responses[1]["response_text"]
        write_call = next(
            call for call in responses[0]["tool_calls"] if call["name"] == "write_file"
        )
        assert "content" not in write_call["input"]
        assert write_call["input"]["content_chars"] == len(_CHANGED_SCORING_MODEL)
        assert len(write_call["input"]["content_sha256"]) == 64
        assert _CHANGED_SCORING_MODEL not in trace_path.read_text(encoding="utf-8")
        assert all(row["llm_response_id"] == "llm_000001" for row in tool_results)


class _EndEarlyThenExecuteClient(_FakeClient):
    def complete(self, messages, tools=None, max_tokens=4096):
        self.calls += 1
        if self.calls == 1:
            return LLMResponse(
                text="I am done.", tool_calls=[], stop_reason="end_turn",
                input_tokens=3, output_tokens=2,
            )
        if self.calls == 2:
            return LLMResponse(
                text=None,
                tool_calls=_write_and_run_calls(
                    "call_2", "Execute after correction", "Required."
                ),
                stop_reason="tool_use", input_tokens=4, output_tokens=2,
            )
        return LLMResponse(
            text='{"reflection":"Execution happened after protocol recovery.","hypothesis_supported":true,"suggested_next":"continue"}',
            tool_calls=[], stop_reason="end_turn", input_tokens=5, output_tokens=3,
        )


def _write_fake_scoring_model(trial: Path) -> None:
    trial.joinpath("model.py").write_text(
        """import argparse, json
ap = argparse.ArgumentParser()
ap.add_argument('--data_dir')
ap.parse_args()
print(json.dumps({'GAUC': 0.61, 'nDCG@5': 0.60, 'primary': 0.605}))
""",
        encoding="utf-8",
    )


def test_agent_cannot_end_before_run_model_when_turns_remain():
    config = load_config()
    with tempfile.TemporaryDirectory() as temp:
        trial = Path(temp)
        _write_fake_scoring_model(trial)
        client = _EndEarlyThenExecuteClient()
        agent = ResearchAgent(
            config,
            client=client,
            provider_retry_delay_s=0,
            bootstrap_state=BootstrapState(required=False),
        )
        result = agent.run_iteration(1, trial, 0.6016, 0.6016, max_turns=2)
        assert result.success
        assert client.calls == 3
        assert any(
            event["action"] == "continue_with_execution_requirement"
            for event in result.recovery_events
        )


class _RetryOnceClient(_FakeClient):
    def complete(self, messages, tools=None, max_tokens=4096):
        self.calls += 1
        if self.calls == 1:
            raise ConnectionError("temporary provider failure")
        if self.calls == 2:
            return LLMResponse(
                text=None,
                tool_calls=_write_and_run_calls(
                    "call_retry", "Retry provider once", "Transient failure."
                ),
                stop_reason="tool_use", input_tokens=6, output_tokens=3,
            )
        return LLMResponse(
            text='{"reflection":"The single retry recovered.","hypothesis_supported":true,"suggested_next":"continue"}',
            tool_calls=[], stop_reason="end_turn", input_tokens=7, output_tokens=4,
        )


def test_provider_errors_receive_exactly_one_retry():
    config = load_config()
    with tempfile.TemporaryDirectory() as temp:
        trial = Path(temp)
        _write_fake_scoring_model(trial)
        client = _RetryOnceClient()
        trace_events = []
        agent = ResearchAgent(
            config,
            client=client,
            provider_retry_delay_s=0,
            bootstrap_state=BootstrapState(required=False),
            event_writer=trace_events.append,
            provider_label="fake / retry-model",
        )
        result = agent.run_iteration(1, trial, 0.6016, 0.6016, max_turns=1)
        assert result.success
        assert client.calls == 3  # failed call + retry + closing reflection
        provider_events = [e for e in result.recovery_events if e["type"] == "provider_error"]
        assert len(provider_events) == 1
        assert provider_events[0]["action"] == "retry_once"
        trace_errors = [e for e in trace_events if e["event_type"] == "provider_error"]
        trace_responses = [e for e in trace_events if e["event_type"] == "llm_response"]
        assert len(trace_errors) == 1
        assert trace_errors[0]["will_retry"]
        assert len(trace_responses) == 2
        assert trace_responses[0]["provider_call_attempt"] == 2


def test_prompt_templates_render_and_have_a_stable_hash():
    first = render_prompt(
        "iteration.md",
        iteration=1,
        candidate_dir="trial_001",
        parent_primary="0.601600",
        best_primary="0.601600",
        max_turns=8,
        stage_instruction="Perform EDA.",
    )
    second = render_prompt(
        "iteration.md",
        iteration=2,
        candidate_dir="trial_002",
        parent_primary="0.602000",
        best_primary="0.602000",
        max_turns=10,
        stage_instruction="Use prior evidence.",
    )
    assert "experiment `1`" in first.content
    assert first.template_sha256 == second.template_sha256
    assert len(first.template_sha256) == 64


def main() -> None:
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"  PASS  {test.__name__}")
    print(f"\n{len(tests)} tests passed.")


if __name__ == "__main__":
    main()
