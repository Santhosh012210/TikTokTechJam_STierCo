"""Offline checks for the active Google ADK research-agent path."""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from typing import AsyncGenerator

from google.adk.models import BaseLlm, LlmRequest, LlmResponse
from google.adk.tools import FunctionTool
from google.genai import types

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mle_agent.harness.adk_config import configure_google_adk_environment
from mle_agent.harness.agent_tools import BootstrapState
from mle_agent.harness.config import Config
from mle_agent.research_agent.adk_agent import ResearchAgent
from mle_agent.tests.recovery_demo import run_recovery_scenario


_CHANGED_MODEL = """import argparse, json
ap = argparse.ArgumentParser()
ap.add_argument('--data_dir')
ap.parse_args()
print(json.dumps({'GAUC': 0.61, 'nDCG@5': 0.60, 'primary': 0.605}))
"""


class _FakeADKModel(BaseLlm):
    """Three model responses: edit, execute, then reflect."""

    calls: int = 0
    request_content_counts: list[int] = []

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        self.calls += 1
        self.request_content_counts.append(len(llm_request.contents))
        if self.calls == 1:
            parts = [
                types.Part.from_text(text="Writing the candidate through ADK."),
                types.Part.from_function_call(
                    name="write_file",
                    args={"path": "model.py", "content": _CHANGED_MODEL},
                ),
            ]
        elif self.calls == 2:
            parts = [
                types.Part.from_text(text="Executing the changed candidate through ADK."),
                types.Part.from_function_call(
                    name="run_model",
                    args={
                        "hypothesis": "Exercise the native Google ADK tool loop",
                        "reasoning": "Verify ADK orchestration while retaining harness execution.",
                        "literature_chunk_ids": [],
                    },
                ),
            ]
        else:
            parts = [types.Part.from_text(text=(
                '{"reflection":"ADK retained the tool history and returned metrics.",'
                '"hypothesis_supported":true,"suggested_next":"continue"}'
            ))]
        yield LlmResponse(
            content=types.Content(role="model", parts=parts),
            finish_reason=types.FinishReason.STOP,
            usage_metadata=types.GenerateContentResponseUsageMetadata(
                prompt_token_count=10,
                candidates_token_count=4,
                total_token_count=14,
            ),
        )


class _StallingBootstrapModel(BaseLlm):
    calls: int = 0

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        self.calls += 1
        yield LlmResponse(
            content=types.Content(
                role="model",
                parts=[types.Part.from_function_call(name="discover_task_docs", args={})],
            ),
            finish_reason=types.FinishReason.STOP,
        )


class _FinalBootstrapModel(BaseLlm):
    calls: int = 0
    payload: dict

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        self.calls += 1
        yield LlmResponse(
            content=types.Content(
                role="model",
                parts=[types.Part.from_function_call(
                    name="record_task_context", args=self.payload
                )],
            ),
            finish_reason=types.FinishReason.STOP,
        )


class _QuotaThenSuccessModel(BaseLlm):
    calls: int = 0

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        self.calls += 1
        if self.calls <= 2:
            raise RuntimeError(
                "ClientError: 429 RESOURCE_EXHAUSTED: quota exceeded. "
                "Please retry in 0s."
            )
        yield LlmResponse(
            content=types.Content(
                role="model", parts=[types.Part.from_text(text="resumed")]
            ),
            finish_reason=types.FinishReason.STOP,
        )


def test_google_adk_owns_the_persistent_tool_loop() -> None:
    config = Config()
    config.PYTHON_EXE = sys.executable
    with tempfile.TemporaryDirectory() as temp:
        trial = Path(temp)
        trial.joinpath("model.py").write_text("print('inherited')\n", encoding="utf-8")
        events: list[dict] = []
        model = _FakeADKModel(model="fake-google-adk")
        agent = ResearchAgent(
            config,
            model=model,
            bootstrap_state=BootstrapState(required=False),
            event_writer=events.append,
            provider_label="google-adk / fake-google-adk",
        )

        result = agent.run_iteration(
            1, trial, parent_primary=0.6016, best_primary=0.6016, max_turns=2
        )

        assert result.success
        assert result.metrics == {"GAUC": 0.61, "nDCG@5": 0.60, "primary": 0.605}
        assert result.hypothesis == "Exercise the native Google ADK tool loop"
        assert result.reflection == "ADK retained the tool history and returned metrics."
        assert result.hypothesis_supported is True
        assert result.suggested_next == "continue"
        assert agent.experiment_memory == [{
            "iteration": 1,
            "status": "success",
            "hypothesis": "Exercise the native Google ADK tool loop",
            "target_component": "agent_selected",
            "primary": 0.605,
            "hypothesis_supported": True,
            "finding": "ADK retained the tool history and returned metrics.",
            "suggested_next": "continue",
        }]
        assert result.token_counts == {"input": 30, "output": 12}
        assert model.calls == 3
        assert model.request_content_counts[0] < model.request_content_counts[-1]
        assert trial.joinpath("model.py").read_text(encoding="utf-8") == _CHANGED_MODEL

        responses = [event for event in events if event["event_type"] == "llm_response"]
        tool_results = [event for event in events if event["event_type"] == "tool_result"]
        assert len(responses) == 3
        assert len(tool_results) == 2
        assert [event["stop_reason"] for event in responses] == [
            "tool_use", "tool_use", "end_turn"
        ]
        write_call = responses[0]["tool_calls"][0]
        assert write_call["name"] == "write_file"
        assert "content" not in write_call["input"]
        assert write_call["input"]["content_chars"] == len(_CHANGED_MODEL)
        serialized_events = str(events)
        assert _CHANGED_MODEL not in serialized_events


def test_persistent_agent_recovers_from_syntax_and_runtime_errors() -> None:
    evidence = run_recovery_scenario()

    assert evidence["success"]
    assert evidence["failed_execution_attempts"] == 2
    assert evidence["successful_execution_attempts"] == 1
    assert evidence["manual_interventions"] == 0
    assert evidence["validation_errors"] == []


def test_bootstrap_budget_ends_without_adk_limit_exception() -> None:
    config = Config()
    with tempfile.TemporaryDirectory() as temp:
        trial = Path(temp)
        trial.joinpath("model.py").write_text("print('baseline')\n", encoding="utf-8")
        model = _StallingBootstrapModel(model="fake-stalling-adk")
        agent = ResearchAgent(config, model=model, bootstrap_state=BootstrapState())

        result = agent.run_bootstrap(trial, max_turns=1)

        assert not result.success
        assert model.calls == 1
        assert "exhausted 1 bootstrap model calls" in result.error
        assert "LlmCallsLimitExceededError" not in result.error


def test_task_context_tool_requires_validation_split_key() -> None:
    agent = ResearchAgent(
        Config(),
        model=_StallingBootstrapModel(model="fake-schema-adk"),
        bootstrap_state=BootstrapState(required=False),
    )
    function = next(
        tool for tool in agent._agent.tools if tool.__name__ == "record_task_context"
    )
    schema = FunctionTool(func=function)._get_declaration().parameters_json_schema
    split_schema = schema["$defs"]["_TaskDataSplits"]

    assert split_schema["required"] == ["train", "validation", "test"]
    assert "valid" not in split_schema["properties"]
    assert split_schema["additionalProperties"] is False


def test_successful_context_on_last_call_completes_without_summary_call() -> None:
    config = Config()
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        starter = root / "starter"
        trial = root / "trial"
        starter.mkdir()
        trial.mkdir()
        readme = starter / "README.md"
        baseline = starter / "baseline.py"
        evaluate = starter / "evaluate.py"
        candidate = trial / "model.py"
        for path in (readme, baseline, evaluate, candidate):
            path.write_text("fully read\n", encoding="utf-8")
        config.BASELINE_ROOT = starter
        paths = [str(path.resolve()) for path in (readme, baseline, evaluate, candidate)]
        state = BootstrapState(
            discovery_completed=True,
            primary_readme_path=paths[0],
            required_baseline_path=paths[1],
            required_evaluation_path=paths[2],
            required_candidate_model_path=paths[3],
            fully_read_paths=set(paths),
            data_inspected=True,
            literature_queries=["within-user ranking"],
            baseline_reproduced=True,
            baseline_metrics={"GAUC": 0.6674, "nDCG@5": 0.5358, "primary": 0.6016},
        )
        payload = {
            "task_objective": "Improve within-user ranking.",
            "target_label": "long_view",
            "metrics": ["GAUC", "nDCG@5", "primary"],
            "data_splits": {
                "train": "20220408-20220421",
                "validation": "20220422-20220428",
                "test": "hidden",
            },
            "baseline": "Official factorization-machine baseline.",
            "evaluation_protocol": ["Train on train and score validation."],
            "hard_constraints": ["Never access hidden test data."],
            "known_dead_ends": ["Comment-only changes."],
            "promising_directions": ["Ranking objectives."],
            "candidate_contract": ["Accept --data_dir and print JSON metrics."],
            "source_paths": paths,
        }
        model = _FinalBootstrapModel(model="fake-final-bootstrap", payload=payload)
        agent = ResearchAgent(config, model=model, bootstrap_state=state)

        result = agent.run_bootstrap(trial, max_turns=1)

        assert result.success
        assert model.calls == 1
        assert result.error is None
        assert state.task_context["data_splits"]["validation"] == "20220422-20220428"


def test_quota_approval_is_asked_once_then_all_resumes_are_automatic() -> None:
    answers: list[str] = []
    sleeps: list[float] = []
    events: list[dict] = []

    def approve(prompt: str) -> str:
        answers.append(prompt)
        return "y"

    model = _QuotaThenSuccessModel(model="fake-quota-adk")
    agent = ResearchAgent(
        Config(),
        model=model,
        bootstrap_state=BootstrapState(required=False),
        event_writer=events.append,
        quota_input=approve,
        quota_sleep=sleeps.append,
    )

    result = agent._run_invocation(
        "Do work.", phase_kind="quota_test", max_llm_calls=0, progress=lambda: ""
    )

    assert result.error is None
    assert result.text == "resumed"
    assert model.calls == 3
    assert len(answers) == 1
    assert answers[0].endswith("[y/n]: ")
    assert sleeps == [1.0, 1.0]
    assert [event["action"] for event in result.recovery_events] == [
        "wait_and_resume", "wait_and_resume"
    ]
    assert [event["human_intervention"] for event in result.recovery_events] == [
        True, False
    ]
    assert len([event for event in events if event["event_type"] == "provider_error"]) == 2
    assert len([event for event in events if event["event_type"] == "quota_pause"]) == 2
    assert len([event for event in events if event["event_type"] == "llm_response"]) == 1


def test_quota_decline_stops_without_waiting() -> None:
    answers: list[str] = []
    sleeps: list[float] = []

    def decline(prompt: str) -> str:
        answers.append(prompt)
        return "n"

    model = _QuotaThenSuccessModel(model="fake-quota-decline-adk")
    agent = ResearchAgent(
        Config(),
        model=model,
        bootstrap_state=BootstrapState(required=False),
        quota_input=decline,
        quota_sleep=sleeps.append,
    )

    result = agent._run_invocation(
        "Do work.", phase_kind="quota_test", max_llm_calls=0, progress=lambda: ""
    )

    assert result.error is not None
    assert "429 RESOURCE_EXHAUSTED" in result.error
    assert model.calls == 1
    assert len(answers) == 1
    assert sleeps == []
    assert result.recovery_events[0]["action"] == "user_declined_resume"
    assert result.recovery_events[0]["human_intervention"] is True


def test_adk_configuration_prefers_native_names_and_keeps_gemini_compatibility() -> None:
    names = ("GOOGLE_API_KEY", "ADK_MODEL", "LLM_PROVIDER", "LLM_API_KEY", "LLM_MODEL")
    original = {name: os.environ.get(name) for name in names}
    try:
        os.environ["GOOGLE_API_KEY"] = "native-test-key"
        os.environ["ADK_MODEL"] = "test-gemini-model"
        settings = configure_google_adk_environment()
        assert settings.model == "test-gemini-model"

        os.environ.pop("GOOGLE_API_KEY", None)
        os.environ.pop("ADK_MODEL", None)
        os.environ["LLM_PROVIDER"] = "gemini"
        os.environ["LLM_API_KEY"] = "legacy-gemini-test-key"
        os.environ["LLM_MODEL"] = "legacy-gemini-model"
        settings = configure_google_adk_environment()
        assert settings.model == "legacy-gemini-model"
        assert os.environ["GOOGLE_API_KEY"] == "legacy-gemini-test-key"
    finally:
        for name, value in original.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def main() -> None:
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"  PASS  {test.__name__}")
    print(f"\n{len(tests)} Google ADK tests passed.")


if __name__ == "__main__":
    main()
