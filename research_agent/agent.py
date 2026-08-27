"""One persistent agent that owns the complete MLE research loop.

There is deliberately no Strategist/Builder handoff. One conversation keeps
the task understanding, EDA, literature, code changes, failures, metrics, and
reflections in context across every experiment in a run.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

from harness.agent_tools import AGENT_TOOLS, AgentToolRuntime
from harness.config import Config
from harness.provider import LLMClient, LLMResponse, make_client
from research_agent.prompts import RenderedPrompt, render_prompt


@dataclass
class AgentIterationResult:
    success: bool
    hypothesis: str
    reasoning: str
    reflection: str
    metrics: dict | None
    executions: list[dict]
    recovery_events: list[dict]
    token_counts: dict[str, int]
    wall_seconds: float
    error: str | None
    final_code: str | None


_CONSOLE_REASONING_MAX_CHARS = 240


def _console_reasoning_line(response: LLMResponse) -> str:
    """Return one safe, compact status line for a completed LLM stage."""
    value: object = response.reasoning_summary or ""
    if not value and response.text:
        value = response.text
        try:
            parsed = json.loads(response.text)
            if isinstance(parsed, dict):
                value = next(
                    (
                        parsed[key]
                        for key in ("reasoning", "status", "reflection", "message")
                        if parsed.get(key)
                    ),
                    response.text,
                )
        except json.JSONDecodeError:
            pass
    if not value and response.tool_calls:
        names = ", ".join(call.name for call in response.tool_calls)
        value = f"Calling local tool: {names}."
    if not value:
        value = f"LLM stage completed with stop reason {response.stop_reason}."

    line = " ".join(str(value).split())
    if len(line) > _CONSOLE_REASONING_MAX_CHARS:
        line = line[:_CONSOLE_REASONING_MAX_CHARS - 1].rstrip() + "…"
    return line


class ResearchAgent:
    def __init__(
        self,
        config: Config,
        client: LLMClient | None = None,
        provider_retry_delay_s: float | None = None,
        rate_limit_retry_delay_s: float | None = None,
    ) -> None:
        self.config = config
        self.client = client or make_client()
        self.provider_retry_delay_s = (
            config.PROVIDER_RETRY_DELAY_S
            if provider_retry_delay_s is None else provider_retry_delay_s
        )
        self.rate_limit_retry_delay_s = (
            config.RATE_LIMIT_RETRY_DELAY_S
            if rate_limit_retry_delay_s is None else rate_limit_retry_delay_s
        )
        system_prompt = render_prompt(
            "single_agent.md",
            starter_kit_root=config.BASELINE_ROOT,
            baseline_primary=f"{config.BASELINE_PRIMARY:.4f}",
            convergence_epsilon=f"{config.CONVERGENCE_EPSILON:.4f}",
        )
        self._prompt_records: dict[str, RenderedPrompt] = {system_prompt.name: system_prompt}
        self.messages: list[dict] = [{"role": "user", "content": system_prompt.content}]
        self._bootstrapped = False

    @property
    def prompt_evidence(self) -> list[dict[str, str]]:
        return [record.evidence() for record in self._prompt_records.values()]

    def _complete_with_one_retry(
        self,
        tools: list[dict] | None,
        max_tokens: int,
        recovery_events: list[dict],
        phase: str,
    ) -> LLMResponse:
        """Call the provider with exactly one retry after an exception."""
        for attempt in range(2):
            try:
                return self.client.complete(
                    self.messages, tools=tools, max_tokens=max_tokens
                )
            except Exception as exc:
                will_retry = attempt == 0
                error_text = f"{type(exc).__name__}: {exc}"
                is_rate_limit = any(
                    marker in error_text.lower()
                    for marker in ("rate_limit", "rate limit", "error code: 429", "error code: 413")
                )
                retry_delay = (
                    self.rate_limit_retry_delay_s
                    if is_rate_limit else self.provider_retry_delay_s
                )
                recovery_events.append({
                    "type": "provider_error",
                    "phase": phase,
                    "attempt": attempt + 1,
                    "error": error_text,
                    "action": "retry_once" if will_retry else "retry_exhausted",
                    "retry_delay_seconds": retry_delay if will_retry else 0,
                })
                if not will_retry:
                    raise
                if retry_delay > 0:
                    time.sleep(retry_delay)
        raise AssertionError("provider retry loop terminated unexpectedly")

    def run_iteration(
        self,
        iteration: int,
        candidate_dir: Path,
        parent_primary: float,
        best_primary: float,
        max_turns: int,
    ) -> AgentIterationResult:
        started = time.time()
        runtime = AgentToolRuntime(candidate_dir, self.config)
        total_input = 0
        total_output = 0
        last_text = ""
        reflection_after_metrics = ""
        last_error: str | None = None
        recovery_events: list[dict] = []
        needs_reflection = False

        iteration_prompt = render_prompt(
            "iteration.md",
            iteration=iteration,
            candidate_dir=candidate_dir,
            parent_primary=f"{parent_primary:.6f}",
            best_primary=f"{best_primary:.6f}",
            max_turns=max_turns,
            stage_instruction=(
                "This is the first experiment: perform the required benchmark reading "
                "and EDA before choosing the change."
                if not self._bootstrapped
                else "Use the full prior conversation and previous result to choose the next change."
            ),
        )
        self._prompt_records[iteration_prompt.name] = iteration_prompt
        instruction = iteration_prompt.content
        if not self._bootstrapped:
            self.messages[-1]["content"] += "\n\n" + instruction
        else:
            self.messages.append({"role": "user", "content": instruction})

        for turn in range(max_turns):
            try:
                response = self._complete_with_one_retry(
                    tools=AGENT_TOOLS,
                    max_tokens=self.config.AGENT_MAX_OUTPUT_TOKENS,
                    recovery_events=recovery_events,
                    phase=f"experiment_{iteration}_turn_{turn + 1}",
                )
            except Exception as exc:
                last_error = f"API error after one retry: {exc}"
                break

            total_input += response.input_tokens
            total_output += response.output_tokens
            print(f"[agent/reasoning] {_console_reasoning_line(response)}", flush=True)
            if response.text:
                last_text = response.text
                if needs_reflection and response.stop_reason == "end_turn":
                    reflection_after_metrics = response.text
                    needs_reflection = False
            self.client.add_response_to_history(self.messages, response)

            if response.stop_reason == "tool_use":
                outputs = [runtime.dispatch(tc.name, tc.input) for tc in response.tool_calls]
                self.client.add_tool_results_to_history(self.messages, response.tool_calls, outputs)
                if any(
                    tc.name == "run_model" and runtime.executions and runtime.executions[-1]["success"]
                    for tc in response.tool_calls
                ):
                    needs_reflection = True
                continue
            if response.stop_reason == "end_turn":
                turns_remain = turn + 1 < max_turns
                if not runtime.executions and turns_remain:
                    recovery_events.append({
                        "type": "agent_protocol_recovery",
                        "phase": f"experiment_{iteration}_turn_{turn + 1}",
                        "error": "agent ended before calling run_model",
                        "action": "continue_with_execution_requirement",
                    })
                    self.messages.append({
                        "role": "user",
                        "content": (
                            "The experiment is not complete because you have not called run_model. "
                            "Continue now: inspect or edit as needed, then execute the candidate."
                        ),
                    })
                    continue
                if runtime.executions and not any(e["success"] for e in runtime.executions) and turns_remain:
                    recovery_events.append({
                        "type": "agent_protocol_recovery",
                        "phase": f"experiment_{iteration}_turn_{turn + 1}",
                        "error": "agent ended after a failed model execution",
                        "action": "continue_with_repair_requirement",
                    })
                    self.messages.append({
                        "role": "user",
                        "content": (
                            "The latest model execution failed. Read the recorded error, apply a "
                            "specific repair, and call run_model again before ending."
                        ),
                    })
                    continue
                if needs_reflection and turns_remain:
                    self.messages.append({
                        "role": "user",
                        "content": (
                            "Interpret the metrics now. Return only the required JSON reflection "
                            "with reflection, hypothesis_supported, and suggested_next."
                        ),
                    })
                    continue
                break
            if response.stop_reason == "length":
                self.messages.append({
                    "role": "user",
                    "content": "Continue from the exact point you stopped. Complete and run the experiment.",
                })
                continue
            last_error = f"unexpected stop reason: {response.stop_reason}"
            break

        self._bootstrapped = True
        best_execution = next(
            (execution for execution in reversed(runtime.executions) if execution["success"]),
            None,
        )
        last_execution = runtime.executions[-1] if runtime.executions else None
        chosen = best_execution or last_execution
        if chosen is None and last_error is None:
            last_error = f"agent ended after {max_turns} turns without calling run_model"
        elif chosen is not None and not chosen["success"] and last_error is None:
            last_error = chosen["error"] or "candidate execution failed"

        # A successful run_model call on the final work turn still receives one
        # short, tools-disabled closing turn so its metrics inform the next iteration.
        if best_execution is not None and needs_reflection:
            self.messages.append({
                "role": "user",
                "content": (
                    "The candidate has now produced validation metrics. Return only a concise "
                    "JSON reflection with reflection, hypothesis_supported, and suggested_next."
                ),
            })
            try:
                closing = self._complete_with_one_retry(
                    tools=None,
                    max_tokens=self.config.AGENT_REFLECTION_MAX_TOKENS,
                    recovery_events=recovery_events,
                    phase=f"experiment_{iteration}_closing_reflection",
                )
                total_input += closing.input_tokens
                total_output += closing.output_tokens
                print(f"[agent/reasoning] {_console_reasoning_line(closing)}", flush=True)
                self.client.add_response_to_history(self.messages, closing)
                if closing.text:
                    reflection_after_metrics = closing.text
                    last_text = closing.text
                    needs_reflection = False
            except Exception as exc:
                recovery_events.append({
                    "type": "reflection_failure",
                    "phase": f"experiment_{iteration}_closing_reflection",
                    "error": f"{type(exc).__name__}: {exc}",
                    "action": "record_explicit_fallback_reflection",
                })

        model_path = candidate_dir / "model.py"
        final_code = model_path.read_text(encoding="utf-8") if model_path.exists() else None
        reflection = (reflection_after_metrics or last_text).strip()
        if best_execution is not None and not reflection:
            reflection = "Agent produced metrics but no reflection after the closing retry."
        if reflection:
            try:
                parsed = json.loads(reflection)
                reflection = str(parsed.get("reflection", reflection))
            except (json.JSONDecodeError, AttributeError):
                pass

        return AgentIterationResult(
            success=best_execution is not None,
            hypothesis=str(chosen["hypothesis"]) if chosen else "Agent produced no experiment",
            reasoning=str(chosen["reasoning"]) if chosen else "",
            reflection=reflection,
            metrics=best_execution["metrics"] if best_execution else None,
            executions=runtime.executions,
            recovery_events=recovery_events,
            token_counts={"input": total_input, "output": total_output},
            wall_seconds=time.time() - started,
            error=None if best_execution else last_error,
            final_code=final_code,
        )
