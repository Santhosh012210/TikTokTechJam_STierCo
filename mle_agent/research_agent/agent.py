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
from typing import Callable, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from mle_agent.harness.agent_tools import (
    AGENT_TOOLS,
    OPTIONAL_AGENT_TOOLS,
    AgentToolRuntime,
    BootstrapState,
)
from mle_agent.harness.blobstore import BlobStore
from mle_agent.harness.config import Config
from mle_agent.harness.console import console
from mle_agent.harness.memory import PhaseAwareMemory, mark_experiment_start
from mle_agent.harness.bootstrap_prefetch import prefetch_bootstrap
from mle_agent.harness.provider import LLMClient, LLMResponse, make_client
from mle_agent.harness.tool_schemas import ReflectionResult
from mle_agent.harness.tools import redact_secrets
from mle_agent.research_agent.prompts import RenderedPrompt, render_prompt
from mle_agent.research_agent.experiment_history import (
    ExperimentRecord,
    append_record,
    classify_outcome,
    compact_for_prompt,
    failed_fingerprints,
)
from mle_agent.research_agent.prior_evidence import (
    compact_prior_experiment_evidence,
    snapshot_prior_experiment_evidence,
)


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
    hypothesis_supported: bool | None = None
    # Three-valued: 'not_tested' keeps a broken build from being recorded as
    # evidence against the idea it never actually tested.
    hypothesis_status: str = ""
    implementation_diagnosis: str = ""
    suggested_next: str = ""
    # Populated only when the agent ran a bounded sweep: one row per scored
    # config. agent_main reads this to emit one evidence row per config while
    # still counting the sweep as a single convergence turn.
    sweep_members: list[dict] | None = None


@dataclass
class AgentBootstrapResult:
    success: bool
    metrics: dict | None
    recovery_events: list[dict]
    token_counts: dict[str, int]
    wall_seconds: float
    error: str | None


class AgentLoopState(BaseModel):
    """Small, code-owned state machine; conversation text is not authority."""

    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    phase: Literal["bootstrap", "experiment", "reflection", "complete", "failed"]
    messages: list[object] = Field(default_factory=list)
    experiment_index: int = 0
    model_calls_used: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    estimated_cost_usd: float = 0.0




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
        dependency_input: Callable[[str], str] = input,
        blob_dir: Path | None = None,
        run_id: str | None = None,
        history_path: Path | None = None,
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
            "agent.md",
            starter_kit_root=config.BASELINE_ROOT,
            convergence_epsilon=f"{config.CONVERGENCE_EPSILON:.4f}",
        )
        self._prompt_records: dict[str, RenderedPrompt] = {system_prompt.name: system_prompt}
        self.loop_state = AgentLoopState(
            phase="experiment" if (bootstrap_state and bootstrap_state.complete) else "bootstrap",
            messages=[{"role": "user", "content": system_prompt.content}],
        )
        self.messages = self.loop_state.messages
        self.bootstrap_state = bootstrap_state or BootstrapState()
        self._bootstrapped = self.bootstrap_state.complete
        self._event_writer = event_writer
        self._provider_label = provider_label or type(self.client).__name__
        self._dependency_input = dependency_input
        self._provider_call_attempts = 0
        self._llm_response_count = 0
        self._last_response_trace: dict[str, object] = {}
        self._prior_experiment_evidence = snapshot_prior_experiment_evidence(
            config.RUN_RESEARCH_DIR
        )
        self._prior_experiment_prompt = compact_prior_experiment_evidence()
        self._memory = PhaseAwareMemory(
            token_budget=config.AGENT_CONTEXT_TOKEN_BUDGET,
            experiment_tail_budget=config.AGENT_EXPERIMENT_TAIL_TOKEN_BUDGET,
            ledger_limit=config.AGENT_EXPERIMENT_MEMORY_LIMIT,
        )
        # Bounded, harness-owned record of what every experiment tested and scored.
        # Previously the iteration prompt rendered a hard-coded empty ledger and the
        # reflect->revise edge relied entirely on the conversation surviving intact;
        # this survives compaction because it is state, not chat.
        self._ledger: list[dict[str, object]] = []
        # The sidecar must ship with the JSONL that references it. finalize_run
        # copies only the run directory, so a blob store under the gitignored
        # experiment workspace would leave every reference in the delivered log
        # dangling.
        self._blobs = BlobStore(
            blob_dir
            or (config.RUN_RESEARCH_DIR or config.ARTIFACTS_DIR) / "blobs"
        )
        self._bootstrap_digest = None
        self._run_id = run_id or "unknown_run"
        # Cross-run memory. Reading is always safe and always useful, so it is on
        # by default. Writing is opt-in: a caller must name the file. Anything
        # that constructs a ResearchAgent without meaning to contribute knowledge
        # -- the offline suites, a smoke test, an embedded experiment -- would
        # otherwise silently append junk to the ledger the agent later trusts.
        self._history_path = history_path
        self._failed_fingerprints = failed_fingerprints()
        self._history_prompt = compact_for_prompt()

    @property
    def prompt_evidence(self) -> list[dict[str, str]]:
        return [record.evidence() for record in self._prompt_records.values()]

    @property
    def bootstrap_evidence(self) -> dict[str, object]:
        return self.bootstrap_state.evidence()

    @property
    def prior_experiment_evidence(self) -> dict[str, object]:
        return dict(self._prior_experiment_evidence)

    @property
    def provider_cost(self) -> dict[str, float] | None:
        spent = getattr(self.client, "spent_usd", None)
        cap = getattr(self.client, "max_cost_usd", None)
        if spent is None or cap is None:
            return None
        return {"estimated_spend_usd": float(spent), "cap_usd": float(cap)}

    def _tools_for_phase(self, phase: str) -> list[dict] | None:
        if phase == "reflection":
            return None
        if phase == "bootstrap":
            forbidden = {"write_file", "edit_file", "run_model", "request_dependency_install"}
            return [tool for tool in AGENT_TOOLS if tool["name"] not in forbidden]
        if self.config.AGENT_ENABLE_SWEEPS:
            return AGENT_TOOLS + OPTIONAL_AGENT_TOOLS
        return AGENT_TOOLS

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

    def _approve_dependency_install(
        self, requirements: list[str], justification: str
    ) -> bool:
        console.harness(
            "Dependency installation requested",
            packages=", ".join(requirements),
            reason=justification,
            environment=self.config.PYTHON_EXE,
        )
        prompt = (
            "These packages are outside the auto-install allowlist. Allow binary-wheel "
            "installation into this run's dedicated venv: "
            + ", ".join(requirements)
            + "? [y/n]: "
        )
        while True:
            try:
                answer = self._dependency_input(prompt).strip().lower()
            except (EOFError, KeyboardInterrupt):
                answer = "n"
            if answer in {"y", "yes"}:
                return True
            if answer in {"n", "no"}:
                return False
            print("Please answer y or n.")

    def _safe_tool_call(self, tool_call) -> dict[str, object]:
        payload = dict(tool_call.input)
        if tool_call.name == "write_file" and "content" in payload:
            content = redact_secrets(str(payload.pop("content")))
            payload["content_chars"] = len(content)
            payload["content_sha256"] = self._blobs.put(content)
            payload["content_blob"] = f"blobs/{payload['content_sha256'][:2]}/{payload['content_sha256']}"
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
            # Long reasoning goes to the blob sidecar; the event keeps a preview
            # and the digest so the JSONL stays readable and still verifiable.
            **self._blobs.reference(response_text, field="response_text"),
            "tool_calls": [self._safe_tool_call(call) for call in response.tool_calls],
        }
        for key in (
            "model", "response_id", "cached_input_tokens", "reasoning_tokens",
            "call_cost_usd", "cumulative_cost_usd", "max_cost_usd",
        ):
            if key in response.provider_metadata:
                trace[key] = response.provider_metadata[key]
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
                state.required_data_path
                and state.required_data_path in state.fully_read_paths
            ),
            bool(
                state.required_feature_ablation_path
                and state.required_feature_ablation_path in state.fully_read_paths
            ),
            bool(
                state.required_candidate_model_path
                and state.required_candidate_model_path in state.fully_read_paths
            ),
            state.data_inspected,
            state.environment_inspected,
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
                # File reads and training stdout dominate the log by volume and
                # repeat across repair loops, so they are stored once by content
                # digest rather than inlined per event.
                **self._blobs.reference(output_safe, field="output"),
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
                will_retry = attempt == 0 and not bool(
                    getattr(exc, "non_retryable", False)
                )
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
            self.loop_state.model_calls_used += 1
            self.loop_state.input_tokens += response.input_tokens
            self.loop_state.output_tokens += response.output_tokens
            self.loop_state.estimated_cost_usd = float(
                getattr(self.client, "spent_usd", 0.0) or 0.0
            )
            return response
        raise AssertionError("provider retry loop terminated unexpectedly")

    def _reflect(
        self,
        *,
        iteration: int,
        best_execution: dict,
        best_primary: float,
        recovery_events: list[dict],
    ) -> tuple[ReflectionResult | None, LLMResponse]:
        """Ask for the closing reflection under a provider-enforced schema.

        Where the provider supports it, ``with_structured_output`` constrains the
        response to ReflectionResult, so the previous hand-parse-then-validate
        path (and its schema-failure recovery event) is no longer the mechanism
        that produces the structured fields — it is only the fallback for clients
        that cannot enforce a schema at all.
        """
        phase = f"experiment_{iteration}_closing_reflection"
        structured = getattr(self.client, "complete_structured", None)

        if structured is None:
            # Provider without structured output: keep the original behaviour of
            # asking for JSON in the prompt and validating what comes back.
            self.messages.append({
                "role": "user",
                "content": (
                    "Return only a JSON object with reflection, hypothesis_status "
                    "(supported / not_supported / not_tested), implementation_diagnosis, "
                    "and suggested_next."
                ),
            })
            closing = self._complete_with_one_retry(
                tools=self._tools_for_phase("reflection"),
                max_tokens=self.config.AGENT_REFLECTION_MAX_TOKENS,
                recovery_events=recovery_events,
                phase=phase,
            )
            self.client.add_response_to_history(self.messages, closing)
            parsed: ReflectionResult | None = None
            try:
                parsed = ReflectionResult.model_validate_json(closing.text or "")
            except ValidationError as exc:
                recovery_events.append({
                    "type": "reflection_schema_failure",
                    "phase": phase,
                    "error": str(exc),
                    "action": "retain_text_but_do_not_trust_structured_fields",
                })
            self._log_reflection(closing, iteration, best_execution, best_primary)
            return parsed, closing

        self._provider_call_attempts += 1
        started = time.time()
        messages_before = len(self.messages)
        parsed, closing = structured(
            self.messages,
            ReflectionResult,
            max_tokens=self.config.AGENT_REFLECTION_MAX_TOKENS,
        )
        self._record_provider_response(
            closing,
            phase=phase,
            provider_attempt=self._provider_call_attempts,
            retry_attempt=1,
            latency_seconds=time.time() - started,
            messages_before_call=messages_before,
        )
        self.loop_state.model_calls_used += 1
        self.loop_state.input_tokens += closing.input_tokens
        self.loop_state.output_tokens += closing.output_tokens
        self.loop_state.estimated_cost_usd = float(
            getattr(self.client, "spent_usd", 0.0) or 0.0
        )
        if parsed is None:
            recovery_events.append({
                "type": "reflection_schema_failure",
                "phase": phase,
                "error": str(
                    closing.provider_metadata.get(
                        "structured_output_error", "structured output returned no value"
                    )
                ),
                "action": "retain_text_but_do_not_trust_structured_fields",
            })
        else:
            # The structured result is text the conversation never saw, because the
            # constrained call is made against the history rather than appended to
            # it. Put the agent's own conclusion back so the next experiment can
            # reason from it directly.
            self.messages.append({
                "role": "assistant",
                "content": parsed.model_dump_json(),
            })
        self._log_reflection(closing, iteration, best_execution, best_primary)
        return parsed, closing

    def _log_reflection(
        self,
        closing: LLMResponse,
        iteration: int,
        best_execution: dict,
        best_primary: float,
    ) -> None:
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

    def _record_cross_run_history(
        self,
        *,
        iteration: int,
        chosen: dict | None,
        metrics: dict | None,
        reflection: ReflectionResult | None,
        incumbent_primary: float,
        error: str | None,
    ) -> None:
        """Append this experiment to the cross-run history.

        Written by the harness from the validated execution rather than offered to
        the agent as a tool: a hypothesis must never be recordable as measured
        unless it actually ran and was scored.

        A history write must never take down a run that has already produced a
        result, so a filesystem failure here is logged and swallowed.
        """
        primary = float(metrics["primary"]) if metrics else None
        # A candidate that scores at or below the item-popularity rung is a broken
        # build, not a refuted hypothesis. Labelling it separately stops one bad
        # implementation from retiring an entire research direction for every
        # later run.
        outcome = classify_outcome(
            primary,
            incumbent_primary,
            self.config.POPULARITY_PRIMARY,
            reflection.hypothesis_status if reflection else "",
        )
        record = ExperimentRecord(
            run_id=self._run_id,
            iteration=iteration,
            hypothesis=str(chosen["hypothesis"]) if chosen else "",
            target_component=(
                str((chosen.get("proposal") or {}).get("target_component", ""))
                if chosen else ""
            ),
            outcome=outcome,
            primary=primary,
            delta_vs_incumbent=(
                round(primary - incumbent_primary, 6) if primary is not None else None
            ),
            incumbent_primary=round(incumbent_primary, 6),
            candidate_fingerprint=(
                str(chosen.get("candidate_fingerprint")) if chosen else None
            ),
            reflection=reflection.reflection if reflection else "",
            error=(
                reflection.implementation_diagnosis
                if reflection and reflection.implementation_diagnosis
                else (error or "")
            ),
        )
        if self._history_path is None:
            return
        try:
            append_record(record, self._history_path)
        except OSError as exc:
            self._emit_trace_event({
                "event_type": "experiment_history_write_failed",
                "timestamp": self._now(),
                "iteration": iteration,
                "error": f"{type(exc).__name__}: {exc}",
            })

    def _record_ledger_entry(
        self,
        *,
        iteration: int,
        chosen: dict | None,
        metrics: dict | None,
        reflection: ReflectionResult | None,
        incumbent_primary: float,
    ) -> None:
        """Append one compact, harness-owned row describing this experiment.

        This is what the iteration prompt renders as the experiment ledger. It is
        deliberately small and derived from validated state, so it survives memory
        compaction and stays honest even when the conversation no longer contains
        the experiment it describes.
        """
        primary = float(metrics["primary"]) if metrics else None
        self._ledger.append({
            "iteration": iteration,
            "hypothesis": (str(chosen["hypothesis"])[:280] if chosen else ""),
            "target_component": (
                str((chosen.get("proposal") or {}).get("target_component", ""))
                if chosen else ""
            ),
            "primary": primary,
            "delta_vs_incumbent": (
                round(primary - incumbent_primary, 6) if primary is not None else None
            ),
            "outcome": (
                "no_result" if primary is None
                else "improved" if primary > incumbent_primary
                else "not_improved"
            ),
            "hypothesis_status": (
                reflection.hypothesis_status if reflection else ""
            ),
            "suggested_next": (reflection.suggested_next[:280] if reflection else ""),
        })
        del self._ledger[: max(0, len(self._ledger) - self.config.AGENT_EXPERIMENT_MEMORY_LIMIT)]

    def _compact_memory(self, *, phase: str, incumbent: dict, force: bool = False) -> None:
        """Apply the phase-aware memory policy in place, and log what it did."""
        compacted, outcome = self._memory.compact(
            self.messages,
            phase=phase,
            bootstrap_state=self.bootstrap_state,
            incumbent=incumbent,
            ledger=self._ledger,
            force=force,
        )
        if outcome.compacted:
            self.messages[:] = compacted
            console.harness(
                "Context compacted",
                phase=phase,
                messages=f"{outcome.messages_before} -> {outcome.messages_after}",
                approx_tokens=f"{outcome.tokens_before} -> {outcome.tokens_after}",
                reason=outcome.reason,
            )
        event = outcome.as_event()
        event["timestamp"] = self._now()
        self._emit_trace_event(event)


    def run_bootstrap(
        self,
        candidate_dir: Path,
        max_turns: int,
    ) -> AgentBootstrapResult:
        """Let the agent understand the task and reproduce the official baseline once."""
        started = time.time()
        self.loop_state.phase = "bootstrap"
        if self.bootstrap_state.complete:
            self.loop_state.phase = "experiment"
            return AgentBootstrapResult(
                success=True,
                metrics=self.bootstrap_state.baseline_metrics,
                recovery_events=[],
                token_counts={"input": 0, "output": 0},
                wall_seconds=0.0,
                error=None,
            )

        runtime = AgentToolRuntime(
            candidate_dir,
            self.config,
            self.bootstrap_state,
            dependency_approver=self._approve_dependency_install,
            failed_fingerprints=self._failed_fingerprints,
        )
        total_input = 0
        total_output = 0
        recovery_events: list[dict] = []
        last_error: str | None = None

        # Every mechanical bootstrap step is fully determined by the repository,
        # so it runs here with no model in the loop. The agent is then handed one
        # curated digest instead of paging six files through read_file.
        digest_section = ""
        if self.config.AGENT_BOOTSTRAP_PREFETCH:
            console.harness(
                "Bootstrap prefetch",
                action="Running discovery, required reads, EDA, environment, and baseline",
                model_calls="0 (deterministic)",
            )
            digest = prefetch_bootstrap(
                runtime,
                self.config,
                self.bootstrap_state,
                on_step=lambda name, payload, output: self._emit_trace_event({
                    "event_type": "bootstrap_prefetch_step",
                    "timestamp": self._now(),
                    "phase": "bootstrap_prefetch",
                    "tool": name,
                    "input": self._redacted_json_value(payload),
                    **self._blobs.reference(
                        redact_secrets(output), field="output"
                    ),
                }),
            )
            self._bootstrap_digest = digest
            provenance = digest.provenance()
            self._emit_trace_event({
                "event_type": "bootstrap_prefetch",
                "timestamp": self._now(),
                **provenance,
            })
            console.harness(
                "Bootstrap prefetch complete",
                sources=len(digest.sources),
                raw_chars=provenance["raw_chars"],
                curated_chars=provenance["curated_chars"],
                baseline_reproduced=str(provenance["baseline_reproduced"]).lower(),
                problems="; ".join(digest.errors) or "none",
            )
            digest_section = digest.render()

        bootstrap_prompt = render_prompt(
            "bootstrap.md",
            candidate_dir=candidate_dir,
            max_turns=max_turns,
            prior_experiment_evidence=self._prior_experiment_prompt,
            cross_run_history=self._history_prompt,
            bootstrap_digest=digest_section,
            remaining_work=(
                "The harness has completed discovery, the required reads, the EDA, the "
                "environment inventory, and the baseline reproduction. Your remaining "
                "work is to interpret that evidence, call record_task_context, then "
                "call record_research_backlog."
                if digest_section else
                "Complete every bootstrap requirement yourself using the tools below."
            ),
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
                    tools=self._tools_for_phase("bootstrap"),
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
        self.loop_state.phase = "experiment" if self._bootstrapped else "failed"
        if self._bootstrapped:
            # The bootstrap/experiment boundary is where compaction is both most
            # valuable and unambiguously safe: the task context and backlog have
            # just been validated, so the structured state now holds everything
            # the raw source reads were carrying.
            self._compact_memory(
                phase="bootstrap_complete",
                incumbent={
                    "iteration": 0,
                    "incumbent_validation_primary": (
                        (self.bootstrap_state.baseline_metrics or {}).get("primary")
                    ),
                    "official_baseline_primary": round(self.config.BASELINE_PRIMARY, 6),
                    "reproduced_baseline_metrics": self.bootstrap_state.baseline_metrics,
                },
                force=True,
            )
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
        experiments_remaining: int | None = None,
    ) -> AgentIterationResult:
        started = time.time()
        self.loop_state.phase = "experiment"
        self.loop_state.experiment_index = iteration
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
        runtime = AgentToolRuntime(
            candidate_dir,
            self.config,
            self.bootstrap_state,
            dependency_approver=self._approve_dependency_install,
            failed_fingerprints=self._failed_fingerprints,
        )
        total_input = 0
        total_output = 0
        last_text = ""
        reflection_after_metrics = ""
        last_error: str | None = None
        recovery_events: list[dict] = []
        needs_reflection = False

        self._compact_memory(
            phase="experiment",
            incumbent={
                "iteration": iteration,
                "incumbent_validation_primary": round(best_primary, 6),
                "parent_validation_primary": round(parent_primary, 6),
                "official_baseline_primary": round(self.config.BASELINE_PRIMARY, 6),
                "reproduced_baseline_metrics": self.bootstrap_state.baseline_metrics,
            },
        )
        iteration_prompt = render_prompt(
            "iteration.md",
            iteration=iteration,
            candidate_dir=candidate_dir,
            parent_primary=f"{parent_primary:.6f}",
            best_primary=f"{best_primary:.6f}",
            max_turns=max_turns,
            experiment_ledger=json.dumps(
                self._ledger, ensure_ascii=False, indent=2
            ),
            research_plan=json.dumps(
                self.bootstrap_state.research_backlog or [], ensure_ascii=False
            ),
            prior_experiment_evidence=self._prior_experiment_prompt,
            cross_run_history=self._history_prompt,
            stage_instruction=(
                "Use the retained task context, reproduced baseline result, literature evidence, "
                "and full prior conversation to choose the next change."
            ),
        )
        self._prompt_records[iteration_prompt.name] = iteration_prompt
        instruction = iteration_prompt.content
        # Tagged rather than plain, so the memory policy locates the
        # bootstrap/experiment boundary structurally instead of by matching this
        # prompt's opening prose.
        self.messages.append(mark_experiment_start(instruction))

        for turn in range(max_turns):
            phase = f"experiment_{iteration}_turn_{turn + 1}"
            try:
                response = self._complete_with_one_retry(
                    tools=self._tools_for_phase("experiment"),
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
                            "with reflection, hypothesis_status, implementation_diagnosis, "
                            "and suggested_next."
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
        structured_reflection: ReflectionResult | None = None
        if best_execution is not None and needs_reflection:
            self.loop_state.phase = "reflection"
            self.messages.append({
                "role": "user",
                "content": (
                    "The candidate has now produced validation metrics. Interpret them "
                    "against the hypothesis you set out to test."
                ),
            })
            try:
                structured_reflection, closing = self._reflect(
                    iteration=iteration,
                    best_execution=best_execution,
                    best_primary=best_primary,
                    recovery_events=recovery_events,
                )
                total_input += closing.input_tokens
                total_output += closing.output_tokens
                if closing.text:
                    reflection_after_metrics = closing.text
                    last_text = closing.text
                if structured_reflection is not None:
                    needs_reflection = False
            except Exception as exc:
                recovery_events.append({
                    "type": "reflection_failure",
                    "phase": f"experiment_{iteration}_closing_reflection",
                    "error": f"{type(exc).__name__}: {exc}",
                    "action": "record_explicit_fallback_reflection",
                })
            finally:
                self.loop_state.phase = "experiment"

        model_path = candidate_dir / "model.py"
        final_code = model_path.read_text(encoding="utf-8") if model_path.exists() else None
        hypothesis_supported: bool | None = None
        hypothesis_status = ""
        implementation_diagnosis = ""
        suggested_next = ""
        if structured_reflection is not None:
            reflection = structured_reflection.reflection
            hypothesis_supported = structured_reflection.hypothesis_supported
            hypothesis_status = structured_reflection.hypothesis_status
            implementation_diagnosis = structured_reflection.implementation_diagnosis
            suggested_next = structured_reflection.suggested_next
        else:
            # The closing structured call runs only when the agent has not already
            # volunteered its reflection during the loop. When it has -- which is
            # the common case, since the iteration prompt asks for one -- the text
            # is all we have, so it is still parsed here. Without this the ledger
            # and the cross-run history lose hypothesis_supported and
            # suggested_next on every well-behaved experiment.
            reflection = (reflection_after_metrics or last_text).strip()
            if best_execution is not None and not reflection:
                reflection = (
                    "Agent produced metrics but no reflection after the closing retry."
                )
            if reflection:
                try:
                    parsed = ReflectionResult.model_validate_json(reflection)
                    reflection = parsed.reflection
                    hypothesis_supported = parsed.hypothesis_supported
                    hypothesis_status = parsed.hypothesis_status
                    implementation_diagnosis = parsed.implementation_diagnosis
                    suggested_next = parsed.suggested_next
                    structured_reflection = parsed
                except ValidationError as exc:
                    recovery_events.append({
                        "type": "reflection_schema_failure",
                        "phase": f"experiment_{iteration}_reflection",
                        "error": str(exc),
                        "action": "retain_text_but_do_not_trust_structured_fields",
                    })

        sweep_members = [
            item for item in runtime.executions if item.get("sweep_id")
        ] or None

        recovery_events.extend(runtime.dependency_events)
        self._record_ledger_entry(
            iteration=iteration,
            chosen=chosen,
            metrics=best_execution["metrics"] if best_execution else None,
            reflection=structured_reflection,
            incumbent_primary=best_primary,
        )
        self._record_cross_run_history(
            iteration=iteration,
            chosen=chosen,
            metrics=best_execution["metrics"] if best_execution else None,
            reflection=structured_reflection,
            incumbent_primary=best_primary,
            error=last_error,
        )
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
            hypothesis_supported=hypothesis_supported,
            hypothesis_status=hypothesis_status,
            implementation_diagnosis=implementation_diagnosis,
            suggested_next=suggested_next,
            sweep_members=sweep_members,
        )
