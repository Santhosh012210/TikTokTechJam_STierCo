"""One persistent agent that owns the complete MLE research loop.

There is deliberately no Strategist/Builder handoff. One conversation keeps
the task understanding, EDA, literature, code changes, failures, metrics, and
reflections in context across every experiment in a run.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from mle_agent.harness.agent_tools import AGENT_TOOLS, AgentToolRuntime, BootstrapState
from mle_agent.harness.config import Config
from mle_agent.harness.console import console
from mle_agent.harness.provider import LLMClient, LLMResponse, make_client
from mle_agent.harness.tools import redact_secrets
from mle_agent.research_agent.prompts import RenderedPrompt, render_prompt


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


@dataclass
class AgentBootstrapResult:
    success: bool
    metrics: dict | None
    recovery_events: list[dict]
    token_counts: dict[str, int]
    wall_seconds: float
    error: str | None


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
        bootstrap_state: BootstrapState | None = None,
        event_writer: Callable[[dict], None] | None = None,
        provider_label: str | None = None,
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
            convergence_epsilon=f"{config.CONVERGENCE_EPSILON:.4f}",
        )
        self._prompt_records: dict[str, RenderedPrompt] = {system_prompt.name: system_prompt}
        self.messages: list[dict] = [{"role": "user", "content": system_prompt.content}]
        self.bootstrap_state = bootstrap_state or BootstrapState()
        self._bootstrapped = self.bootstrap_state.complete
        self._event_writer = event_writer
        self._provider_label = provider_label or type(self.client).__name__
        self._provider_call_attempts = 0
        self._llm_response_count = 0
        self._last_response_trace: dict[str, object] = {}

    @property
    def prompt_evidence(self) -> list[dict[str, str]]:
        return [record.evidence() for record in self._prompt_records.values()]

    @property
    def bootstrap_evidence(self) -> dict[str, object]:
        return self.bootstrap_state.evidence()

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    @staticmethod
    def _digest(value: str) -> str:
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    @staticmethod
    def _redacted_json_value(value: object) -> object:
        encoded = json.dumps(value, ensure_ascii=False, default=str)
        return json.loads(redact_secrets(encoded))

    def _emit_trace_event(self, event: dict[str, object]) -> None:
        if self._event_writer is not None:
            self._event_writer(event)

    def _safe_tool_call(self, tool_call) -> dict[str, object]:
        payload = dict(tool_call.input)
        if tool_call.name == "write_file" and "content" in payload:
            content = redact_secrets(str(payload.pop("content")))
            payload["content_chars"] = len(content)
            payload["content_sha256"] = self._digest(content)
        return {
            "tool_call_id": tool_call.id,
            "name": tool_call.name,
            "summary": console.tool_call_summary(tool_call.name, tool_call.input),
            "input": self._redacted_json_value(payload),
        }

    def _record_provider_response(
        self,
        response: LLMResponse,
        *,
        phase: str,
        provider_attempt: int,
        retry_attempt: int,
        latency_seconds: float,
        messages_before_call: int,
    ) -> None:
        self._llm_response_count += 1
        response_id = f"llm_{self._llm_response_count:06d}"
        response_text = redact_secrets(response.text or "")
        trace = {
            "event_type": "llm_response",
            "timestamp": self._now(),
            "llm_response_id": response_id,
            "llm_response_number": self._llm_response_count,
            "provider_call_attempt": provider_attempt,
            "retry_attempt": retry_attempt,
            "provider": self._provider_label,
            "phase": phase,
            "latency_seconds": latency_seconds,
            "messages_before_call": messages_before_call,
            "stop_reason": response.stop_reason,
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
            "decision_summary": _console_reasoning_line(response),
            "response_text": response_text,
            "response_text_chars": len(response_text),
            "response_text_sha256": self._digest(response_text),
            "tool_calls": [self._safe_tool_call(call) for call in response.tool_calls],
        }
        self._last_response_trace = trace
        self._emit_trace_event(trace)

    def _console_trace_fields(self, response: LLMResponse) -> dict[str, object]:
        trace = self._last_response_trace
        return {
            "call_number": trace.get("llm_response_number"),
            "response_event_id": trace.get("llm_response_id"),
            "provider": trace.get("provider"),
            "latency_seconds": trace.get("latency_seconds"),
            "tool_summaries": [
                console.tool_call_summary(call.name, call.input)
                for call in response.tool_calls
            ],
        }

    def _bootstrap_progress(self) -> str:
        state = self.bootstrap_state
        checks = [
            state.discovery_completed,
            bool(state.primary_readme_path and state.primary_readme_path in state.fully_read_paths),
            bool(
                state.required_evaluation_path
                and state.required_evaluation_path in state.fully_read_paths
            ),
            bool(
                state.required_baseline_path
                and state.required_baseline_path in state.fully_read_paths
            ),
            bool(
                state.required_candidate_model_path
                and state.required_candidate_model_path in state.fully_read_paths
            ),
            state.data_inspected,
            bool(state.literature_queries),
            state.baseline_reproduced,
            state.task_context is not None,
        ]
        missing = state.missing_requirements()
        pending = "; ".join(missing[:3])
        if len(missing) > 3:
            pending += f"; +{len(missing) - 3} more"
        return f"{sum(checks)}/{len(checks)} complete" + (
            f" — pending: {pending}" if pending else ""
        )

    def _dispatch_tool_calls(
        self,
        runtime: AgentToolRuntime,
        response: LLMResponse,
        *,
        phase: str,
    ) -> list[str]:
        outputs: list[str] = []
        for tool_call in response.tool_calls:
            console.agent_tool_call(tool_call.name, tool_call.input)
            output = runtime.dispatch(tool_call.name, tool_call.input)
            outputs.append(output)
            console.agent_tool_result(tool_call.name, output)
            output_safe = redact_secrets(output)
            output_summary, success = console.tool_result_summary(tool_call.name, output_safe)
            self._emit_trace_event({
                "event_type": "tool_result",
                "timestamp": self._now(),
                "llm_response_id": self._last_response_trace.get("llm_response_id"),
                "provider": self._provider_label,
                "phase": phase,
                "tool_call": self._safe_tool_call(tool_call),
                "success": success,
                "output_summary": output_summary,
                "output_chars": len(output_safe),
                "output_sha256": self._digest(output_safe),
                "output_preview": output_safe[:1200],
                "output_truncated": len(output_safe) > 1200,
            })
        self.client.add_tool_results_to_history(self.messages, response.tool_calls, outputs)
        return outputs

    def _complete_with_one_retry(
        self,
        tools: list[dict] | None,
        max_tokens: int,
        recovery_events: list[dict],
        phase: str,
    ) -> LLMResponse:
        """Call the provider with exactly one retry after an exception."""
        for attempt in range(2):
            self._provider_call_attempts += 1
            provider_attempt = self._provider_call_attempts
            messages_before_call = len(self.messages)
            call_started = time.time()
            try:
                response = self.client.complete(
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
                self._emit_trace_event({
                    "event_type": "provider_error",
                    "timestamp": self._now(),
                    "provider": self._provider_label,
                    "phase": phase,
                    "provider_call_attempt": provider_attempt,
                    "retry_attempt": attempt + 1,
                    "latency_seconds": time.time() - call_started,
                    "messages_before_call": messages_before_call,
                    "error": redact_secrets(error_text),
                    "will_retry": will_retry,
                    "retry_delay_seconds": retry_delay if will_retry else 0,
                })
                console.harness(
                    "Provider recovery",
                    phase=phase,
                    attempt=f"{attempt + 1}/2",
                    action="Retrying once" if will_retry else "Retry exhausted",
                    delay_seconds=retry_delay if will_retry else 0,
                    error=redact_secrets(error_text)[:500],
                )
                if not will_retry:
                    raise
                if retry_delay > 0:
                    time.sleep(retry_delay)
                continue

            # Persist the successful response outside the provider-exception
            # handler. A disk/logging failure must not be mistaken for a
            # provider failure and trigger a duplicate billed request.
            self._record_provider_response(
                response,
                phase=phase,
                provider_attempt=provider_attempt,
                retry_attempt=attempt + 1,
                latency_seconds=time.time() - call_started,
                messages_before_call=messages_before_call,
            )
            return response
        raise AssertionError("provider retry loop terminated unexpectedly")

    def run_bootstrap(
        self,
        candidate_dir: Path,
        max_turns: int,
    ) -> AgentBootstrapResult:
        """Let the agent understand the task and reproduce the official baseline once."""
        started = time.time()
        if self.bootstrap_state.complete:
            return AgentBootstrapResult(
                success=True,
                metrics=self.bootstrap_state.baseline_metrics,
                recovery_events=[],
                token_counts={"input": 0, "output": 0},
                wall_seconds=0.0,
                error=None,
            )

        runtime = AgentToolRuntime(candidate_dir, self.config, self.bootstrap_state)
        total_input = 0
        total_output = 0
        recovery_events: list[dict] = []
        last_error: str | None = None
        bootstrap_prompt = render_prompt(
            "bootstrap.md",
            candidate_dir=candidate_dir,
            max_turns=max_turns,
        )
        self._prompt_records[bootstrap_prompt.name] = bootstrap_prompt
        if (
            len(self.messages) == 1
            and self.messages[-1].get("role") == "user"
            and isinstance(self.messages[-1].get("content"), str)
        ):
            self.messages[-1]["content"] += "\n\n" + bootstrap_prompt.content
        else:
            self.messages.append({"role": "user", "content": bootstrap_prompt.content})

        for turn in range(max_turns):
            phase = f"bootstrap_turn_{turn + 1}"
            try:
                response = self._complete_with_one_retry(
                    tools=AGENT_TOOLS,
                    max_tokens=self.config.AGENT_MAX_OUTPUT_TOKENS,
                    recovery_events=recovery_events,
                    phase=phase,
                )
            except Exception as exc:
                last_error = f"API error after one retry: {exc}"
                break

            total_input += response.input_tokens
            total_output += response.output_tokens
            console.agent_reasoning(
                _console_reasoning_line(response),
                phase=f"Bootstrap turn {turn + 1}/{max_turns}",
                progress=self._bootstrap_progress(),
                response_text=response.text,
                tool_names=[call.name for call in response.tool_calls],
                stop_reason=response.stop_reason,
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
                **self._console_trace_fields(response),
            )
            self.client.add_response_to_history(self.messages, response)

            if response.stop_reason == "tool_use":
                self._dispatch_tool_calls(runtime, response, phase=phase)
                if self.bootstrap_state.complete:
                    break
                continue
            if response.stop_reason == "end_turn":
                if self.bootstrap_state.complete:
                    break
                missing = self.bootstrap_state.missing_requirements()
                recovery_events.append({
                    "type": "agent_protocol_recovery",
                    "phase": f"bootstrap_turn_{turn + 1}",
                    "error": "agent ended before completing task bootstrap",
                    "action": "continue_with_bootstrap_requirement",
                    "missing_requirements": missing,
                })
                console.harness(
                    "Agent protocol recovery",
                    issue="Agent ended its response before bootstrap completed",
                    action="Requesting the remaining bootstrap requirements",
                    missing="; ".join(missing),
                )
                self.messages.append({
                    "role": "user",
                    "content": (
                        "Bootstrap is incomplete. Continue with these requirements: "
                        + "; ".join(missing)
                        + ". Do not edit the candidate yet."
                    ),
                })
                continue
            if response.stop_reason == "length":
                recovery_events.append({
                    "type": "agent_protocol_recovery",
                    "phase": f"bootstrap_turn_{turn + 1}",
                    "error": "LLM response reached its output-token limit",
                    "action": "continue_bootstrap_after_length_stop",
                })
                console.harness(
                    "Agent protocol recovery",
                    issue="LLM response reached its output-token limit during bootstrap",
                    action="Continuing from the pending bootstrap state",
                )
                self.messages.append({
                    "role": "user",
                    "content": (
                        "Continue the bootstrap from the exact point you stopped. Pending: "
                        + "; ".join(self.bootstrap_state.missing_requirements())
                    ),
                })
                continue
            last_error = f"unexpected stop reason: {response.stop_reason}"
            break

        self._bootstrapped = self.bootstrap_state.complete
        if not self.bootstrap_state.complete and last_error is None:
            last_error = (
                f"agent ended after {max_turns} bootstrap turns with incomplete requirements: "
                + ", ".join(self.bootstrap_state.missing_requirements())
            )
        return AgentBootstrapResult(
            success=self.bootstrap_state.complete,
            metrics=self.bootstrap_state.baseline_metrics,
            recovery_events=recovery_events,
            token_counts={"input": total_input, "output": total_output},
            wall_seconds=time.time() - started,
            error=None if self.bootstrap_state.complete else last_error,
        )

    def run_iteration(
        self,
        iteration: int,
        candidate_dir: Path,
        parent_primary: float,
        best_primary: float,
        max_turns: int,
    ) -> AgentIterationResult:
        started = time.time()
        if not self.bootstrap_state.complete:
            model_path = candidate_dir / "model.py"
            return AgentIterationResult(
                success=False,
                hypothesis="Agent experiment blocked by incomplete bootstrap",
                reasoning="The task and official baseline must be understood first.",
                reflection="",
                metrics=None,
                executions=[],
                recovery_events=[],
                token_counts={"input": 0, "output": 0},
                wall_seconds=time.time() - started,
                error=(
                    "task bootstrap incomplete; call run_bootstrap first: "
                    + ", ".join(self.bootstrap_state.missing_requirements())
                ),
                final_code=(
                    model_path.read_text(encoding="utf-8") if model_path.exists() else None
                ),
            )
        runtime = AgentToolRuntime(candidate_dir, self.config, self.bootstrap_state)
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
            experiment_ledger="[]",
            stage_instruction=(
                "Use the retained task context, reproduced baseline result, literature evidence, "
                "and full prior conversation to choose the next change."
            ),
        )
        self._prompt_records[iteration_prompt.name] = iteration_prompt
        instruction = iteration_prompt.content
        self.messages.append({"role": "user", "content": instruction})

        for turn in range(max_turns):
            phase = f"experiment_{iteration}_turn_{turn + 1}"
            try:
                response = self._complete_with_one_retry(
                    tools=AGENT_TOOLS,
                    max_tokens=self.config.AGENT_MAX_OUTPUT_TOKENS,
                    recovery_events=recovery_events,
                    phase=phase,
                )
            except Exception as exc:
                last_error = f"API error after one retry: {exc}"
                break

            total_input += response.input_tokens
            total_output += response.output_tokens
            console.agent_reasoning(
                _console_reasoning_line(response),
                phase=f"Experiment {iteration}, turn {turn + 1}/{max_turns}",
                progress=(
                    f"Execution attempts={len(runtime.executions)}; "
                    f"parent primary={parent_primary:.6f}; best primary={best_primary:.6f}"
                ),
                response_text=response.text,
                tool_names=[call.name for call in response.tool_calls],
                stop_reason=response.stop_reason,
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
                **self._console_trace_fields(response),
            )
            if response.text:
                last_text = response.text
                if needs_reflection and response.stop_reason == "end_turn":
                    reflection_after_metrics = response.text
                    needs_reflection = False
            self.client.add_response_to_history(self.messages, response)

            if response.stop_reason == "tool_use":
                self._dispatch_tool_calls(runtime, response, phase=phase)
                if any(
                    tc.name == "run_model" and runtime.executions and runtime.executions[-1]["success"]
                    for tc in response.tool_calls
                ):
                    needs_reflection = True
                continue
            if response.stop_reason == "end_turn":
                turns_remain = turn + 1 < max_turns
                if not runtime.executions and turns_remain:
                    bootstrap_missing = self.bootstrap_state.missing_requirements()
                    recovery_events.append({
                        "type": "agent_protocol_recovery",
                        "phase": f"experiment_{iteration}_turn_{turn + 1}",
                        "error": (
                            "agent ended before completing task bootstrap"
                            if bootstrap_missing
                            else "agent ended before calling run_model"
                        ),
                        "action": (
                            "continue_with_bootstrap_requirement"
                            if bootstrap_missing
                            else "continue_with_execution_requirement"
                        ),
                    })
                    console.harness(
                        "Agent protocol recovery",
                        issue=(
                            "Agent ended before completing task bootstrap"
                            if bootstrap_missing
                            else "Agent ended without emitting a run_model tool call"
                        ),
                        action=(
                            "Requesting remaining bootstrap work"
                            if bootstrap_missing
                            else "Requesting implementation and model execution"
                        ),
                        missing="; ".join(bootstrap_missing) if bootstrap_missing else "run_model",
                    )
                    self.messages.append({
                        "role": "user",
                        "content": (
                            "The experiment is not complete. Finish these bootstrap requirements: "
                            + "; ".join(bootstrap_missing)
                            + ". Then implement the hypothesis and call run_model."
                            if bootstrap_missing
                            else (
                                "The experiment is not complete because you have not called run_model. "
                                "Continue now: inspect or edit as needed, then execute the candidate."
                            )
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
                    console.harness(
                        "Agent protocol recovery",
                        issue="Latest model execution failed",
                        action="Requesting a targeted repair and another run_model call",
                        error=runtime.executions[-1].get("error", "unknown execution error"),
                    )
                    self.messages.append({
                        "role": "user",
                        "content": (
                            "The latest model execution failed. Read the recorded error, apply a "
                            "specific repair, and call run_model again before ending."
                        ),
                    })
                    continue
                if needs_reflection and turns_remain:
                    console.harness(
                        "Agent protocol recovery",
                        issue="Model produced metrics but the agent did not reflect on them",
                        action="Requesting the required structured reflection",
                    )
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
                recovery_events.append({
                    "type": "agent_protocol_recovery",
                    "phase": f"experiment_{iteration}_turn_{turn + 1}",
                    "error": "LLM response reached its output-token limit",
                    "action": "continue_experiment_after_length_stop",
                })
                console.harness(
                    "Agent protocol recovery",
                    issue="LLM response reached its output-token limit",
                    action="Continuing the same experiment from the saved conversation",
                )
                self.messages.append({
                    "role": "user",
                    "content": "Continue from the exact point you stopped. Complete and run the experiment.",
                })
                continue
            last_error = f"unexpected stop reason: {response.stop_reason}"
            break

        self._bootstrapped = self.bootstrap_state.complete
        best_execution = next(
            (execution for execution in reversed(runtime.executions) if execution["success"]),
            None,
        )
        last_execution = runtime.executions[-1] if runtime.executions else None
        chosen = best_execution or last_execution
        if chosen is None and last_error is None:
            if not self.bootstrap_state.complete:
                missing = ", ".join(self.bootstrap_state.missing_requirements())
                last_error = (
                    f"agent ended after {max_turns} turns with incomplete task bootstrap: {missing}"
                )
            else:
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
                console.agent_reasoning(
                    _console_reasoning_line(closing),
                    phase=f"Experiment {iteration} reflection",
                    progress=(
                        f"Scored primary={best_execution['metrics']['primary']:.6f}; "
                        f"best before experiment={best_primary:.6f}"
                    ),
                    response_text=closing.text,
                    tool_names=[call.name for call in closing.tool_calls],
                    stop_reason=closing.stop_reason,
                    input_tokens=closing.input_tokens,
                    output_tokens=closing.output_tokens,
                    **self._console_trace_fields(closing),
                )
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
