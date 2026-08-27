"""Offline checks for the single-agent runner and its safety boundaries."""
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness.agent_main import _converged, _log_row
from harness.agent_tools import AgentToolRuntime
from harness.config import load_config
from harness.data_view import (
    EXPECTED_TEST_ROWS,
    EXPECTED_TRAIN_ROWS,
    EXPECTED_VALID_ROWS,
    classify_date,
)
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
        runtime = AgentToolRuntime(trial, config)
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
        runtime = AgentToolRuntime(trial, config)
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
        agent = ResearchAgent(config, client=client, provider_retry_delay_s=0)
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
        agent = ResearchAgent(config, client=client, provider_retry_delay_s=0)
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
        agent = ResearchAgent(config, client=client, provider_retry_delay_s=0)
        result = agent.run_iteration(1, trial, 0.6016, 0.6016, max_turns=1)
        assert result.success
        assert client.calls == 3  # failed call + retry + closing reflection
        provider_events = [e for e in result.recovery_events if e["type"] == "provider_error"]
        assert len(provider_events) == 1
        assert provider_events[0]["action"] == "retry_once"


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
