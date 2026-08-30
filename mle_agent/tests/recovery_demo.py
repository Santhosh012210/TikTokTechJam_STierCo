"""Deterministic end-to-end syntax and runtime recovery demonstration."""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import AsyncGenerator

from google.adk.models import BaseLlm, LlmRequest, LlmResponse
from google.genai import types

from mle_agent.harness.agent_main import _diff, _log_row
from mle_agent.harness.agent_tools import BootstrapState
from mle_agent.harness.config import Config
from mle_agent.harness.validator import validate_row
from mle_agent.research_agent.adk_agent import ResearchAgent


_INHERITED_MODEL = "print('inherited candidate')\n"
_SYNTAX_FAILURE = "def broken(:\n    pass\n"
_RUNTIME_FAILURE = """import argparse
ap = argparse.ArgumentParser()
ap.add_argument('--data_dir')
ap.add_argument('--seed')
ap.add_argument('--prediction-path')
ap.add_argument('--trial-config')
ap.parse_args()
raise RuntimeError('deliberate recovery demonstration')
"""
_RECOVERED_MODEL = """import argparse, json
ap = argparse.ArgumentParser()
ap.add_argument('--data_dir')
ap.add_argument('--seed')
ap.add_argument('--prediction-path')
ap.add_argument('--trial-config')
a = ap.parse_args()
open(a.prediction_path, 'w').write('row_id,user_id,video_id,score\\n')
print(json.dumps({'GAUC': 0.61, 'nDCG@5': 0.60, 'primary': 0.605}))
"""


def _response(parts: list[types.Part]) -> LlmResponse:
    return LlmResponse(
        content=types.Content(role="model", parts=parts),
        finish_reason=types.FinishReason.STOP,
        usage_metadata=types.GenerateContentResponseUsageMetadata(
            prompt_token_count=8,
            candidates_token_count=4,
            total_token_count=12,
        ),
    )


class RecoveryDemoModel(BaseLlm):
    """Scripted model that repairs one syntax and one runtime failure."""

    calls: int = 0

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        self.calls += 1
        if self.calls == 1:
            parts = [
                types.Part.from_text(text="Testing a deliberately malformed candidate."),
                types.Part.from_function_call(
                    name="write_file",
                    args={"path": "model.py", "content": _SYNTAX_FAILURE},
                ),
            ]
        elif self.calls == 2:
            parts = [types.Part.from_function_call(
                name="run_model",
                args={
                    "hypothesis": "Demonstrate bounded automatic candidate recovery",
                    "reasoning": "First verify that the syntax gate blocks invalid Python.",
                    "target_component": "model",
                    "literature_chunk_ids": [],
                },
            )]
        elif self.calls == 3:
            parts = [
                types.Part.from_text(text="Repairing syntax, then exposing a runtime failure."),
                types.Part.from_function_call(
                    name="write_file",
                    args={"path": "model.py", "content": _RUNTIME_FAILURE},
                ),
            ]
        elif self.calls == 4:
            parts = [types.Part.from_function_call(
                name="run_model",
                args={
                    "hypothesis": "Demonstrate bounded automatic candidate recovery",
                    "reasoning": "Run the syntax-valid candidate and inspect its traceback.",
                    "target_component": "model",
                    "literature_chunk_ids": [],
                },
            )]
        elif self.calls == 5:
            parts = [
                types.Part.from_text(text="Applying the smallest runtime repair."),
                types.Part.from_function_call(
                    name="write_file",
                    args={"path": "model.py", "content": _RECOVERED_MODEL},
                ),
            ]
        elif self.calls == 6:
            parts = [types.Part.from_function_call(
                name="run_model",
                args={
                    "hypothesis": "Demonstrate bounded automatic candidate recovery",
                    "reasoning": "Verify that the repaired candidate completes and scores.",
                    "target_component": "model",
                    "literature_chunk_ids": [],
                },
            )]
        else:
            parts = [types.Part.from_text(text=json.dumps({
                "reflection": (
                    "The syntax gate prevented invalid execution, the runtime traceback "
                    "guided a targeted repair, and the final candidate scored successfully."
                ),
                "hypothesis_supported": True,
                "suggested_next": "Apply the same bounded repair protocol to real experiments.",
            }))]
        yield _response(parts)


def run_recovery_scenario(output_dir: Path | None = None) -> dict:
    """Run the real ADK tool loop with a fake provider and optionally save evidence."""
    with tempfile.TemporaryDirectory() as temp:
        temp_root = Path(temp)
        trial = temp_root / "experiment_workspace" / "recovery_demo" / "trial_001"
        trial.mkdir(parents=True)
        model_path = trial / "model.py"
        model_path.write_text(_INHERITED_MODEL, encoding="utf-8")

        config = Config()
        config.PYTHON_EXE = sys.executable
        trace_events: list[dict] = []
        model = RecoveryDemoModel(model="fake-recovery-adk")
        agent = ResearchAgent(
            config,
            model=model,
            bootstrap_state=BootstrapState(required=False),
            event_writer=trace_events.append,
            provider_label="google-adk / fake-recovery-adk",
        )
        result = agent.run_iteration(
            1, trial, parent_primary=0.6016, best_primary=0.6016, max_turns=8
        )
        final_source = model_path.read_text(encoding="utf-8")
        row = _log_row(
            1,
            0,
            result,
            "success" if result.success else "failed",
            bool(result.success),
            0.6016,
            0.2468,
            _diff(_INHERITED_MODEL, final_source, "trial_000/model.py", "trial_001/model.py"),
            model_path,
        )
        validation_errors = validate_row(row)
        evidence = {
            "scenario": "automatic_syntax_and_runtime_recovery",
            "success": result.success,
            "manual_interventions": 0,
            "model_calls": model.calls,
            "failed_execution_attempts": sum(
                not attempt["success"] for attempt in result.executions
            ),
            "successful_execution_attempts": sum(
                attempt["success"] for attempt in result.executions
            ),
            "validation_errors": validation_errors,
            "event": row,
            "llm_trace": trace_events,
        }

        assert result.success
        assert [attempt["success"] for attempt in result.executions] == [False, False, True]
        assert len(result.recovery_events) == 2
        assert all(event["outcome"] == "recovered" for event in result.recovery_events)
        assert result.hypothesis_supported is True
        assert result.suggested_next
        assert validation_errors == []

        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            serialized = json.dumps(evidence, ensure_ascii=False, indent=2, default=str)
            serialized = serialized.replace(str(temp_root), "<DEMO_WORKSPACE>")
            output_dir.joinpath("recovery-evidence.json").write_text(
                serialized + "\n", encoding="utf-8"
            )
            output_dir.joinpath("README.md").write_text(
                "# Automatic recovery demo\n\n"
                "A deterministic fake provider drives the real persistent Google ADK tool "
                "loop through a syntax failure, a runtime failure, and a successful repair. "
                "Run `./mle_agent/scripts/demo_recovery.sh` to regenerate the evidence.\n",
                encoding="utf-8",
            )
        return evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    evidence = run_recovery_scenario(args.output)
    print(json.dumps({
        "success": evidence["success"],
        "model_calls": evidence["model_calls"],
        "failed_execution_attempts": evidence["failed_execution_attempts"],
        "successful_execution_attempts": evidence["successful_execution_attempts"],
        "manual_interventions": evidence["manual_interventions"],
    }, indent=2))


if __name__ == "__main__":
    main()
