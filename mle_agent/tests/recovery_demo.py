"""Deterministic end-to-end syntax and runtime recovery demonstration.

A scripted fake provider drives the *real* agent loop — real tool dispatch, real
syntax gate, real subprocess execution, real repair prompts — through a syntax
failure, a runtime failure, and a successful repair, with no human in the loop.

The point is that nothing here is stubbed except the model's choices. The
recovery events, the execution attempts, and the log row are produced by the same
code path a scored run uses, so the evidence is about the harness rather than
about the demo.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

from mle_agent.harness.agent_main import _diff, _log_row
from mle_agent.harness.agent_tools import BootstrapState
from mle_agent.harness.config import Config
from mle_agent.harness.provider import LLMResponse, ToolCall
from mle_agent.harness.validator import validate_row
from mle_agent.research_agent.agent import ResearchAgent
from mle_agent.harness.tool_schemas import ReflectionResult


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
# The repair is the real root FM candidate rather than a script that merely prints
# plausible metrics. The trusted evaluator scores the prediction file this writes,
# so the demo's final "recovered" step is a genuine scored execution -- the earlier
# version of this demo printed its own metrics and would not survive that check.
def _recovered_model(config: Config) -> str:
    from mle_agent.harness.root_model import make_root_model_py

    return make_root_model_py(config)

_RUN_MODEL_ARGS = {
    "hypothesis": "Demonstrate bounded automatic candidate recovery",
    "target_component": "model",
    "literature_chunk_ids": [],
}


class RecoveryDemoClient:
    """Scripted client that repairs one syntax and one runtime failure."""

    def __init__(self, recovered_model: str) -> None:
        self.calls = 0
        self._recovered = recovered_model

    @staticmethod
    def _write(content: str, note: str, call_id: str) -> LLMResponse:
        return LLMResponse(
            text=note,
            tool_calls=[ToolCall(
                id=call_id, name="write_file",
                input={"path": "model.py", "content": content},
            )],
            stop_reason="tool_use", input_tokens=8, output_tokens=4,
        )

    @staticmethod
    def _run(reasoning: str, call_id: str) -> LLMResponse:
        return LLMResponse(
            text=None,
            tool_calls=[ToolCall(
                id=call_id, name="run_model",
                input={**_RUN_MODEL_ARGS, "reasoning": reasoning},
            )],
            stop_reason="tool_use", input_tokens=8, output_tokens=4,
        )

    def complete(self, messages, tools=None, max_tokens=4096) -> LLMResponse:
        self.calls += 1
        if self.calls == 1:
            return self._write(
                _SYNTAX_FAILURE, "Testing a deliberately malformed candidate.", "w1"
            )
        if self.calls == 2:
            return self._run(
                "First verify that the syntax gate blocks invalid Python.", "r1"
            )
        if self.calls == 3:
            return self._write(
                _RUNTIME_FAILURE,
                "Repairing syntax, then exposing a runtime failure.", "w2",
            )
        if self.calls == 4:
            return self._run(
                "Run the syntax-valid candidate and inspect its traceback.", "r2"
            )
        if self.calls == 5:
            return self._write(
                self._recovered, "Applying the smallest runtime repair.", "w3"
            )
        if self.calls == 6:
            return self._run(
                "Verify that the repaired candidate completes and scores.", "r3"
            )
        return LLMResponse(
            text=_REFLECTION.model_dump_json(), tool_calls=[],
            stop_reason="end_turn", input_tokens=8, output_tokens=4,
        )

    def complete_structured(self, messages, schema, max_tokens=4096):
        """Mirror the real structured-output contract used for the reflection."""
        self.calls += 1
        return _REFLECTION, LLMResponse(
            text=_REFLECTION.model_dump_json(), tool_calls=[],
            stop_reason="end_turn", input_tokens=8, output_tokens=4,
        )

    def add_response_to_history(self, messages, response: LLMResponse) -> None:
        messages.append({"role": "assistant", "content": response.text or ""})

    def add_tool_results_to_history(self, messages, tool_calls, outputs) -> None:
        for call, output in zip(tool_calls, outputs):
            messages.append({"role": "user", "content": output})


_REFLECTION = ReflectionResult(
    reflection=(
        "The syntax gate prevented invalid execution, the runtime traceback guided "
        "a targeted repair, and the final candidate scored successfully."
    ),
    hypothesis_status="supported",
    suggested_next="Apply the same bounded repair protocol to real experiments.",
)


def run_recovery_scenario(output_dir: Path | None = None) -> dict:
    """Drive the real agent loop with a fake provider; optionally save evidence."""
    with tempfile.TemporaryDirectory() as temp:
        temp_root = Path(temp)
        trial = temp_root / "experiment_workspace" / "recovery_demo" / "trial_001"
        trial.mkdir(parents=True)
        model_path = trial / "model.py"
        model_path.write_text(_INHERITED_MODEL, encoding="utf-8")

        config = Config()
        config.PYTHON_EXE = sys.executable
        trace_events: list[dict] = []
        client = RecoveryDemoClient(_recovered_model(config))
        agent = ResearchAgent(
            config,
            client=client,
            bootstrap_state=BootstrapState(required=False),
            event_writer=trace_events.append,
            provider_label="langchain / scripted-recovery-client",
            blob_dir=temp_root / "blobs",
            # A demo must never contribute to the agent's cross-run memory.
            history_path=None,
        )
        result = agent.run_iteration(
            1, trial, parent_primary=0.6016, best_primary=0.6016, max_turns=8
        )
        final_source = model_path.read_text(encoding="utf-8")
        row = _log_row(
            1, 0, result,
            "success" if result.success else "failed",
            bool(result.success), 0.6016, 0.2468,
            _diff(_INHERITED_MODEL, final_source, "trial_000/model.py", "trial_001/model.py"),
            model_path,
        )
        validation_errors = validate_row(row)
        evidence = {
            "scenario": "automatic_syntax_and_runtime_recovery",
            "success": result.success,
            "manual_interventions": 0,
            "model_calls": client.calls,
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
                "A deterministic scripted provider drives the real agent tool loop "
                "through a syntax failure, a runtime failure, and a successful repair, "
                "with zero manual interventions. Everything except the model's choices "
                "is the same code a scored run uses: the syntax gate, tool dispatch, "
                "subprocess execution, repair prompts, and log-row validation.\n\n"
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
