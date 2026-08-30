"""Google ADK implementation of the persistent KuaiRand MLE research agent.

Google ADK owns the model/tool event loop and persistent conversation session.
The repository's existing ``AgentToolRuntime`` remains the deterministic safety
boundary for file access, bootstrap gates, data isolation, and model execution.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import math
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from google.adk.agents import Agent
from google.adk.agents.run_config import RunConfig
from google.adk.models import BaseLlm, LlmRequest, LlmResponse
from google.adk.runners import InMemoryRunner
from google.adk.tools import ToolContext
from google.genai import types
from pydantic import BaseModel, ConfigDict

from mle_agent.harness.adk_config import configure_google_adk_environment
from mle_agent.harness.agent_tools import AgentToolRuntime, BootstrapState
from mle_agent.harness.config import Config
from mle_agent.harness.console import console
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
    hypothesis_supported: bool | None = None
    suggested_next: str = ""


@dataclass
class AgentBootstrapResult:
    success: bool
    metrics: dict | None
    recovery_events: list[dict]
    token_counts: dict[str, int]
    wall_seconds: float
    error: str | None


@dataclass
class _InvocationResult:
    text: str
    model_calls: int
    token_counts: dict[str, int]
    budget_exhausted: bool
    recovery_events: list[dict]
    error: str | None


_CONSOLE_REASONING_MAX_CHARS = 240


class _TaskDataSplits(BaseModel):
    """Exact nested schema expected by ``BootstrapState.record_task_context``."""

    model_config = ConfigDict(extra="forbid")

    train: str
    validation: str
    test: str


class _FeatureEngineeringContext(BaseModel):
    """Structured evidence that the agent understood the organizer feature baseline."""

    model_config = ConfigDict(extra="forbid")

    baseline_fields: list[str]
    measured_dead_ends: list[str]
    promising_feature_families: list[str]
    leakage_controls: list[str]
    implementation_boundary: str


class ResearchAgent:
    """One persistent Google ADK agent for the complete research run."""

    _APP_NAME = "kuairand_mle_research"
    _USER_ID = "autonomous_researcher"
    _SESSION_ID = "persistent_research_session"

    def __init__(
        self,
        config: Config,
        *,
        model: str | BaseLlm | None = None,
        bootstrap_state: BootstrapState | None = None,
        event_writer: Callable[[dict], None] | None = None,
        provider_label: str | None = None,
        quota_input: Callable[[str], str] = input,
        quota_sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.config = config
        if model is None:
            settings = configure_google_adk_environment()
            model = settings.model
            provider_label = provider_label or f"google-adk / {settings.model}"
        self.model = model
        model_name = model if isinstance(model, str) else model.model
        self._provider_label = provider_label or f"google-adk / {model_name}"
        self.bootstrap_state = bootstrap_state or BootstrapState()
        self._event_writer = event_writer
        self._quota_input = quota_input
        self._quota_sleep = quota_sleep
        self._quota_resume_approved: bool | None = None
        self._quota_pause_count = 0
        self._experiment_memory: list[dict[str, object]] = []

        system_prompt = render_prompt(
            "single_agent.md",
            starter_kit_root=config.BASELINE_ROOT,
            convergence_epsilon=f"{config.CONVERGENCE_EPSILON:.4f}",
        )
        self._prompt_records: dict[str, RenderedPrompt] = {
            system_prompt.name: system_prompt
        }

        self._runtime: AgentToolRuntime | None = None
        self._phase_kind = "idle"
        self._iteration = 0
        self._max_calls = 0
        self._current_phase = "idle"
        self._invocation_model_calls = 0
        self._invocation_tokens = {"input": 0, "output": 0}
        self._invocation_text = ""
        self._budget_short_circuit = False
        self._quota_short_circuit = False
        self._pending_quota_error: str | None = None
        self._invocation_budget_exhausted = False
        self._model_call_started = 0.0
        self._messages_before_call = 0
        self._llm_response_count = 0
        self._last_response_trace: dict[str, object] = {}
        self._progress: Callable[[], str] = lambda: ""

        self._agent = Agent(
            name="kuairand_mle_agent",
            description="Autonomous machine-learning engineer for the KuaiRand benchmark.",
            model=model,
            instruction=system_prompt.content,
            tools=self._build_adk_tools(),
            generate_content_config=types.GenerateContentConfig(
                max_output_tokens=config.AGENT_MAX_OUTPUT_TOKENS,
            ),
            before_model_callback=self._before_model_callback,
            after_model_callback=self._after_model_callback,
            on_model_error_callback=self._on_model_error_callback,
            before_tool_callback=self._before_tool_callback,
            after_tool_callback=self._after_tool_callback,
        )
        self._runner = InMemoryRunner(agent=self._agent, app_name=self._APP_NAME)
        asyncio.run(
            self._runner.session_service.create_session(
                app_name=self._APP_NAME,
                user_id=self._USER_ID,
                session_id=self._SESSION_ID,
                state={"task": "kuairand_mle_research"},
            )
        )

    @property
    def prompt_evidence(self) -> list[dict[str, str]]:
        return [record.evidence() for record in self._prompt_records.values()]

    @property
    def bootstrap_evidence(self) -> dict[str, object]:
        return self.bootstrap_state.evidence()

    @property
    def experiment_memory(self) -> list[dict[str, object]]:
        return [dict(item) for item in self._experiment_memory]

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

    @staticmethod
    def _is_provider_limit_error(error_text: str) -> bool:
        lowered = error_text.lower()
        return any(marker in lowered for marker in (
            "resource_exhausted",
            "quota exceeded",
            "exceeded your current quota",
            "rate limit",
            "ratelimit",
            "error code: 429",
            "clienterror: 429",
            "status code 429",
        ))

    def _provider_retry_delay(self, error_text: str) -> float:
        """Extract Gemini's advertised retry/reset delay, with a safe fallback."""
        patterns = (
            r"please\s+retry\s+in\s+(\d+(?:\.\d+)?)\s*(ms|s|m|h)",
            r"retry[_ ]?delay['\"\s:=>]+(\d+(?:\.\d+)?)\s*(ms|s|m|h)",
            r"retry\s+after\s+(\d+(?:\.\d+)?)\s*(ms|s|m|h)",
        )
        multipliers = {"ms": 0.001, "s": 1.0, "m": 60.0, "h": 3600.0}
        lowered = error_text.lower()
        for pattern in patterns:
            match = re.search(pattern, lowered)
            if match:
                seconds = float(match.group(1)) * multipliers[match.group(2)]
                return max(1.0, float(math.ceil(seconds)) + 1.0)
        return max(1.0, self.config.RATE_LIMIT_RETRY_DELAY_S)

    def _ask_to_resume_after_quota(self) -> bool:
        """Ask once; an approval applies to every later quota pause in this run."""
        if self._quota_resume_approved is not None:
            return self._quota_resume_approved
        prompt = (
            "You have hit your LLM limit. Would you like to resume when the limit "
            "has reset? [y/n]: "
        )
        while True:
            try:
                answer = self._quota_input(prompt).strip().lower()
            except (EOFError, KeyboardInterrupt):
                answer = "n"
            if answer in {"y", "yes"}:
                self._quota_resume_approved = True
                return True
            if answer in {"n", "no"}:
                self._quota_resume_approved = False
                return False
            print("Please answer y or n.")

    def _pause_for_provider_limit(self, error_text: str) -> tuple[bool, dict]:
        asked_user = self._quota_resume_approved is None
        approved = self._ask_to_resume_after_quota()
        delay = self._provider_retry_delay(error_text)
        self._quota_pause_count += 1
        event = {
            "type": "provider_quota_pause",
            "phase": self._current_phase,
            "quota_pause_number": self._quota_pause_count,
            "retry_delay_seconds": delay,
            "action": "wait_and_resume" if approved else "user_declined_resume",
            "human_intervention": asked_user,
            "error": redact_secrets(error_text),
        }
        self._emit_trace_event({
            "event_type": "quota_pause",
            "timestamp": self._now(),
            "provider": self._provider_label,
            **event,
        })
        if not approved:
            console.harness(
                "LLM quota recovery declined",
                status="Stopping the autonomous run at the retained ADK session state",
            )
            return False, event

        console.harness(
            "LLM quota pause",
            status="Resume approved; no further human confirmation will be requested",
            retry_in=f"{delay:.0f}s",
        )
        remaining = delay
        while remaining > 0:
            step = min(60.0, remaining)
            self._quota_sleep(step)
            remaining = max(0.0, remaining - step)
            if remaining > 0:
                console.harness(
                    "LLM quota pause",
                    status="Waiting for provider reset",
                    retry_in=f"{remaining:.0f}s",
                )
        console.harness(
            "LLM quota reset wait complete",
            status="Automatically resuming the retained Google ADK session",
        )
        return True, event

    def _phase_for_call(self, number: int) -> str:
        if self._phase_kind == "bootstrap":
            return f"bootstrap_turn_{number}"
        if self._phase_kind == "experiment":
            return f"experiment_{self._iteration}_turn_{number}"
        return self._phase_kind

    def _console_phase_for_call(self, number: int) -> str:
        limit = str(self._max_calls) if self._max_calls > 0 else "∞"
        if self._phase_kind == "bootstrap":
            return f"Bootstrap model call {number}/{limit}"
        if self._phase_kind == "experiment":
            return f"Experiment {self._iteration}, ADK model call {number}/{limit}"
        return self._phase_kind

    @staticmethod
    def _response_parts(response: LlmResponse) -> tuple[str, list[dict[str, Any]]]:
        texts: list[str] = []
        calls: list[dict[str, Any]] = []
        if response.content and response.content.parts:
            for part in response.content.parts:
                if part.text and not getattr(part, "thought", False):
                    texts.append(part.text)
                call = part.function_call
                if call is not None:
                    calls.append({
                        "id": call.id or "",
                        "name": call.name or "unknown_tool",
                        "input": dict(call.args or {}),
                    })
        return "\n".join(texts).strip(), calls

    @staticmethod
    def _usage(response: LlmResponse) -> tuple[int, int]:
        usage = response.usage_metadata
        if usage is None:
            return 0, 0
        return (
            int(usage.prompt_token_count or 0),
            int(usage.candidates_token_count or 0),
        )

    @staticmethod
    def _stop_reason(response: LlmResponse, has_tools: bool) -> str:
        if has_tools:
            return "tool_use"
        reason = response.finish_reason
        if reason is None:
            return "end_turn"
        value = getattr(reason, "value", str(reason))
        return "length" if str(value).upper() == "MAX_TOKENS" else "end_turn"

    @staticmethod
    def _decision_summary(text: str, tool_names: list[str], stop_reason: str) -> str:
        value: object = text
        if text:
            try:
                parsed = json.loads(text)
                if isinstance(parsed, dict):
                    value = next(
                        (
                            parsed[key]
                            for key in ("reasoning", "status", "reflection", "message")
                            if parsed.get(key)
                        ),
                        text,
                    )
            except json.JSONDecodeError:
                pass
        elif tool_names:
            value = f"Calling local tool: {', '.join(tool_names)}."
        else:
            value = f"ADK model call completed with stop reason {stop_reason}."
        line = " ".join(str(value).split())
        if len(line) > _CONSOLE_REASONING_MAX_CHARS:
            line = line[: _CONSOLE_REASONING_MAX_CHARS - 1].rstrip() + "…"
        return line

    def _safe_tool_call(
        self, name: str, payload: dict[str, Any], call_id: str | None = None
    ) -> dict[str, object]:
        safe_payload = dict(payload)
        if name == "write_file" and "content" in safe_payload:
            content = redact_secrets(str(safe_payload.pop("content")))
            safe_payload["content_chars"] = len(content)
            safe_payload["content_sha256"] = self._digest(content)
        return {
            "tool_call_id": call_id or "",
            "name": name,
            "summary": console.tool_call_summary(name, payload),
            "input": self._redacted_json_value(safe_payload),
        }

    def _before_model_callback(
        self, callback_context: Any, llm_request: LlmRequest
    ) -> LlmResponse | None:
        if self._max_calls > 0 and self._invocation_model_calls >= self._max_calls:
            # ADK's max_llm_calls raises when a final tool call lands exactly
            # on the cap and the runtime prepares the next model step. End the
            # invocation before another provider request instead.
            self._budget_short_circuit = True
            self._invocation_budget_exhausted = True
            callback_context.actions.end_of_agent = True
            return LlmResponse(
                content=types.Content(
                    role="model",
                    parts=[types.Part.from_text(text="Model-call budget reached.")],
                ),
                finish_reason=types.FinishReason.STOP,
            )
        self._model_call_started = time.time()
        self._messages_before_call = len(llm_request.contents)
        return None

    def _after_model_callback(
        self, callback_context: Any, llm_response: LlmResponse
    ) -> LlmResponse | None:
        if self._budget_short_circuit:
            self._budget_short_circuit = False
            return None
        if self._quota_short_circuit:
            # The model-error callback converted an expected provider quota
            # exception into an ADK response so the runner can unwind cleanly.
            # It is not a real provider response and must not affect usage or
            # the retained answer text.
            self._quota_short_circuit = False
            return None
        response = llm_response
        self._invocation_model_calls += 1
        self._llm_response_count += 1
        call_number = self._invocation_model_calls
        self._current_phase = self._phase_for_call(call_number)
        response_id = f"llm_{self._llm_response_count:06d}"
        text, tool_calls = self._response_parts(response)
        if text:
            self._invocation_text = text
        tool_names = [str(call["name"]) for call in tool_calls]
        input_tokens, output_tokens = self._usage(response)
        self._invocation_tokens["input"] += input_tokens
        self._invocation_tokens["output"] += output_tokens
        stop_reason = self._stop_reason(response, bool(tool_calls))
        summary = self._decision_summary(text, tool_names, stop_reason)
        safe_text = redact_secrets(text)
        latency_seconds = max(0.0, time.time() - self._model_call_started)
        trace = {
            "event_type": "llm_response",
            "timestamp": self._now(),
            "llm_response_id": response_id,
            "llm_response_number": self._llm_response_count,
            "provider_call_attempt": self._llm_response_count,
            "retry_attempt": 1,
            "provider": self._provider_label,
            "phase": self._current_phase,
            "adk_invocation_id": getattr(callback_context, "invocation_id", None),
            "adk_invocation_model_call": call_number,
            "latency_seconds": latency_seconds,
            "messages_before_call": self._messages_before_call,
            "stop_reason": stop_reason,
            "finish_reason": (
                getattr(response.finish_reason, "value", str(response.finish_reason))
                if response.finish_reason is not None else None
            ),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "decision_summary": summary,
            "response_text": safe_text,
            "response_text_chars": len(safe_text),
            "response_text_sha256": self._digest(safe_text),
            "tool_calls": [
                self._safe_tool_call(
                    str(call["name"]), dict(call["input"]), str(call["id"])
                )
                for call in tool_calls
            ],
        }
        self._last_response_trace = trace
        self._emit_trace_event(trace)
        console.agent_reasoning(
            summary,
            phase=self._console_phase_for_call(call_number),
            progress=self._progress(),
            response_text=text,
            tool_names=tool_names,
            stop_reason=stop_reason,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            call_number=self._llm_response_count,
            response_event_id=response_id,
            provider=self._provider_label,
            latency_seconds=latency_seconds,
            tool_summaries=[
                console.tool_call_summary(str(call["name"]), dict(call["input"]))
                for call in tool_calls
            ],
        )
        return None

    def _on_model_error_callback(
        self, callback_context: Any, llm_request: LlmRequest, error: Exception
    ) -> LlmResponse | None:
        error_text = redact_secrets(f"{type(error).__name__}: {error}")
        provider_limit = self._is_provider_limit_error(error_text)
        retry_delay = self._provider_retry_delay(error_text) if provider_limit else 0
        self._emit_trace_event({
            "event_type": "provider_error",
            "timestamp": self._now(),
            "provider": self._provider_label,
            "phase": self._current_phase,
            "adk_invocation_id": getattr(callback_context, "invocation_id", None),
            "provider_call_attempt": self._llm_response_count + 1,
            "retry_attempt": 1,
            "latency_seconds": max(0.0, time.time() - self._model_call_started),
            "messages_before_call": self._messages_before_call,
            "error": error_text,
            "will_retry": provider_limit,
            "retry_delay_seconds": retry_delay,
        })
        if provider_limit:
            # Handle an expected quota condition inside ADK's callback path.
            # Returning a response prevents ADK from printing a full exception
            # traceback; _run_invocation prompts/waits after the runner exits.
            self._pending_quota_error = error_text
            self._quota_short_circuit = True
            callback_context.actions.end_of_agent = True
            return LlmResponse(
                content=types.Content(
                    role="model",
                    parts=[types.Part.from_text(
                        text="Provider quota reached; waiting for reset approval."
                    )],
                ),
                finish_reason=types.FinishReason.STOP,
            )
        return None

    def _before_tool_callback(
        self, tool: Any, args: dict[str, Any], tool_context: Any
    ) -> dict[str, Any] | None:
        console.agent_tool_call(tool.name, args)
        return None

    def _after_tool_callback(
        self,
        tool: Any,
        args: dict[str, Any],
        tool_context: Any,
        tool_response: dict[str, Any],
    ) -> dict[str, Any] | None:
        result = tool_response
        output = json.dumps(result, ensure_ascii=False, indent=2, default=str)
        console.agent_tool_result(tool.name, output)
        output_safe = redact_secrets(output)
        output_summary, success = console.tool_result_summary(tool.name, output_safe)
        self._emit_trace_event({
            "event_type": "tool_result",
            "timestamp": self._now(),
            "llm_response_id": self._last_response_trace.get("llm_response_id"),
            "provider": self._provider_label,
            "phase": self._current_phase,
            "adk_invocation_id": getattr(tool_context, "invocation_id", None),
            "tool_call": self._safe_tool_call(
                tool.name, args, getattr(tool_context, "function_call_id", "")
            ),
            "success": success,
            "output_summary": output_summary,
            "output_chars": len(output_safe),
            "output_sha256": self._digest(output_safe),
            "output_preview": output_safe[:1200],
            "output_truncated": len(output_safe) > 1200,
        })
        return None

    def _dispatch(self, name: str, payload: dict[str, Any]) -> dict[str, Any]:
        if self._runtime is None:
            return {"success": False, "error": "ADK_TOOL_RUNTIME_NOT_ACTIVE"}
        output = self._runtime.dispatch(name, payload)
        try:
            parsed = json.loads(output)
        except json.JSONDecodeError:
            # ``write_file`` intentionally returns a compact human-readable OK
            # string in the retained runtime.  ADK tools return dictionaries,
            # so preserve that output while adding an explicit status field.
            failed = output.lstrip().upper().startswith(("ERROR", "FAILED"))
            return {"success": not failed, "output": output}
        return parsed if isinstance(parsed, dict) else {"result": parsed}

    def _build_adk_tools(self) -> list[Callable[..., dict[str, Any]]]:
        dispatch = self._dispatch

        def discover_task_docs(tool_context: ToolContext = None) -> dict[str, Any]:
            """Discover the official task README and benchmark-support files."""
            return dispatch("discover_task_docs", {})

        def read_file(
            path: str,
            offset: int = 0,
            max_chars: int | None = None,
            tool_context: ToolContext = None,
        ) -> dict[str, Any]:
            """Read one explicit page from the candidate or official starter kit."""
            payload: dict[str, Any] = {"path": path, "offset": offset}
            if max_chars is not None:
                payload["max_chars"] = max_chars
            return dispatch("read_file", payload)

        def write_file(
            path: str, content: str, tool_context: ToolContext = None
        ) -> dict[str, Any]:
            """Create or replace a file inside the current candidate directory."""
            return dispatch("write_file", {"path": path, "content": content})

        def inspect_data(tool_context: ToolContext = None) -> dict[str, Any]:
            """Inspect a deterministic train/validation-only data summary."""
            return dispatch("inspect_data", {})

        def search_ml_literature(
            query: str,
            k: int = 3,
            tool_context: ToolContext = None,
        ) -> dict[str, Any]:
            """Search the local recommender-systems method corpus for evidence."""
            return dispatch("search_ml_literature", {"query": query, "k": k})

        def reproduce_baseline(tool_context: ToolContext = None) -> dict[str, Any]:
            """Run the unchanged baseline after all required task and feature sources are read."""
            return dispatch("reproduce_baseline", {})

        def record_task_context(
            task_objective: str,
            target_label: str,
            metrics: list[str],
            data_splits: _TaskDataSplits,
            baseline: str,
            evaluation_protocol: list[str],
            hard_constraints: list[str],
            known_dead_ends: list[str],
            promising_directions: list[str],
            feature_engineering_context: _FeatureEngineeringContext,
            candidate_contract: list[str],
            source_paths: list[str],
            tool_context: ToolContext = None,
        ) -> dict[str, Any]:
            """Persist the structured benchmark understanding for later experiments."""
            result = dispatch("record_task_context", {
                "task_objective": task_objective,
                "target_label": target_label,
                "metrics": metrics,
                "data_splits": data_splits.model_dump(),
                "baseline": baseline,
                "evaluation_protocol": evaluation_protocol,
                "hard_constraints": hard_constraints,
                "known_dead_ends": known_dead_ends,
                "promising_directions": promising_directions,
                "feature_engineering_context": feature_engineering_context.model_dump(),
                "candidate_contract": candidate_contract,
                "source_paths": source_paths,
            })
            if result.get("success") and tool_context is not None:
                # Bootstrap completion is itself the useful final output. Do
                # not spend another provider call merely summarizing it.
                tool_context.actions.skip_summarization = True
                tool_context.actions.end_of_agent = True
            return result

        def run_model(
            hypothesis: str,
            reasoning: str,
            literature_chunk_ids: list[str] | None = None,
            target_component: str = "agent_selected",
            expected_effect: str = "",
            falsification_criterion: str = "",
            rollback_plan: str = "retain incumbent",
            feature_sources: list[str] | None = None,
            feature_transformations: list[str] | None = None,
            leakage_controls: list[str] | None = None,
            tool_context: ToolContext = None,
        ) -> dict[str, Any]:
            """Execute model.py against train and validation and return metrics/errors."""
            return dispatch("run_model", {
                "hypothesis": hypothesis,
                "reasoning": reasoning,
                "literature_chunk_ids": literature_chunk_ids or [],
                "target_component": target_component,
                "expected_effect": expected_effect,
                "falsification_criterion": falsification_criterion,
                "rollback_plan": rollback_plan,
                "feature_sources": feature_sources or [],
                "feature_transformations": feature_transformations or [],
                "leakage_controls": leakage_controls or [],
            })

        return [
            discover_task_docs,
            read_file,
            write_file,
            inspect_data,
            search_ml_literature,
            reproduce_baseline,
            record_task_context,
            run_model,
        ]

    def _bootstrap_progress(self) -> str:
        state = self.bootstrap_state
        checks = [
            state.discovery_completed,
            bool(state.primary_readme_path and state.primary_readme_path in state.fully_read_paths),
            bool(state.required_evaluation_path and state.required_evaluation_path in state.fully_read_paths),
            bool(state.required_baseline_path and state.required_baseline_path in state.fully_read_paths),
            bool(state.required_data_path and state.required_data_path in state.fully_read_paths),
            bool(
                state.required_feature_ablation_path
                and state.required_feature_ablation_path in state.fully_read_paths
            ),
            bool(state.required_candidate_model_path and state.required_candidate_model_path in state.fully_read_paths),
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

    def _run_invocation(
        self,
        prompt: str,
        *,
        phase_kind: str,
        max_llm_calls: int,
        progress: Callable[[], str],
    ) -> _InvocationResult:
        self._phase_kind = phase_kind
        self._max_calls = max_llm_calls
        self._current_phase = phase_kind
        self._invocation_model_calls = 0
        self._invocation_tokens = {"input": 0, "output": 0}
        self._invocation_text = ""
        self._budget_short_circuit = False
        self._quota_short_circuit = False
        self._pending_quota_error = None
        self._invocation_budget_exhausted = False
        self._progress = progress
        error: str | None = None
        recovery_events: list[dict] = []
        sdk_call_limit = max_llm_calls + 1 if max_llm_calls > 0 else 0
        run_config = RunConfig(
            # Zero means unlimited in Google ADK. For an explicitly configured
            # positive limit, keep one SDK step for the harness's synthetic
            # non-provider completion response.
            max_llm_calls=sdk_call_limit,
            custom_metadata={"phase": phase_kind, "iteration": self._iteration},
            http_options=types.HttpOptions(
                retry_options=types.HttpRetryOptions(
                    attempts=2,
                    initial_delay=self.config.PROVIDER_RETRY_DELAY_S,
                    max_delay=self.config.RATE_LIMIT_RETRY_DELAY_S,
                )
            ),
        )
        next_message = prompt
        while True:
            attempt_error: str | None = None
            self._quota_short_circuit = False
            self._pending_quota_error = None
            try:
                for event in self._runner.run(
                    user_id=self._USER_ID,
                    session_id=self._SESSION_ID,
                    new_message=types.Content(
                        role="user", parts=[types.Part.from_text(text=next_message)]
                    ),
                    run_config=run_config,
                ):
                    if event.error_code or event.error_message:
                        attempt_error = (
                            f"{event.error_code or 'ADK_ERROR'}: "
                            f"{event.error_message or ''}"
                        ).strip()
            except Exception as exc:
                attempt_error = f"{type(exc).__name__}: {exc}"

            if self._pending_quota_error is not None:
                attempt_error = self._pending_quota_error

            if attempt_error and self._is_provider_limit_error(attempt_error):
                should_resume, recovery = self._pause_for_provider_limit(attempt_error)
                recovery_events.append(recovery)
                if should_resume:
                    next_message = (
                        "The provider quota has reset. Resume the interrupted task from "
                        "this retained ADK session. Do not repeat completed tool work."
                    )
                    continue
            error = attempt_error
            break
        return _InvocationResult(
            text=self._invocation_text,
            model_calls=self._invocation_model_calls,
            token_counts=dict(self._invocation_tokens),
            budget_exhausted=self._invocation_budget_exhausted,
            recovery_events=recovery_events,
            error=error,
        )

    @staticmethod
    def _add_tokens(total: dict[str, int], addition: dict[str, int]) -> None:
        total["input"] += addition["input"]
        total["output"] += addition["output"]

    def run_bootstrap(
        self, candidate_dir: Path, max_turns: int
    ) -> AgentBootstrapResult:
        """Use one ADK session to understand the task and reproduce the baseline."""
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
        call_budget_label = max_turns if max_turns > 0 else "unlimited"
        bootstrap_prompt = render_prompt(
            "bootstrap.md", candidate_dir=candidate_dir, max_turns=call_budget_label
        )
        self._prompt_records[bootstrap_prompt.name] = bootstrap_prompt
        recovery_events: list[dict] = []
        totals = {"input": 0, "output": 0}
        unlimited = max_turns <= 0
        remaining = max_turns
        prompt = bootstrap_prompt.content
        last_error: str | None = None
        self._runtime = runtime
        try:
            while (unlimited or remaining > 0) and not self.bootstrap_state.complete:
                invocation = self._run_invocation(
                    prompt,
                    phase_kind="bootstrap",
                    max_llm_calls=0 if unlimited else remaining,
                    progress=self._bootstrap_progress,
                )
                self._add_tokens(totals, invocation.token_counts)
                recovery_events.extend(invocation.recovery_events)
                if not unlimited:
                    remaining -= invocation.model_calls
                if invocation.error:
                    last_error = invocation.error
                    break
                if self.bootstrap_state.complete:
                    break
                if invocation.model_calls == 0:
                    break
                missing = self.bootstrap_state.missing_requirements()
                recovery_events.append({
                    "type": "agent_protocol_recovery",
                    "phase": "bootstrap",
                    "error": "ADK agent ended before completing task bootstrap",
                    "action": "continue_with_bootstrap_requirement",
                    "missing_requirements": missing,
                })
                prompt = (
                    "Bootstrap is incomplete. Continue using the retained ADK session. "
                    "Complete these requirements without editing the candidate: "
                    + "; ".join(missing)
                )
        finally:
            self._runtime = None

        if not self.bootstrap_state.complete and last_error is None:
            if unlimited:
                last_error = (
                    "ADK agent stopped without completing bootstrap; missing: "
                    + ", ".join(self.bootstrap_state.missing_requirements())
                )
            else:
                last_error = (
                    f"ADK agent exhausted {max_turns} bootstrap model calls; missing: "
                    + ", ".join(self.bootstrap_state.missing_requirements())
                )
        return AgentBootstrapResult(
            success=self.bootstrap_state.complete,
            metrics=self.bootstrap_state.baseline_metrics,
            recovery_events=recovery_events,
            token_counts=totals,
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
        """Run one experiment through ADK's native model/tool event loop."""
        started = time.time()
        model_path = candidate_dir / "model.py"
        if not self.bootstrap_state.complete:
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
                final_code=model_path.read_text(encoding="utf-8") if model_path.exists() else None,
            )

        runtime = AgentToolRuntime(candidate_dir, self.config, self.bootstrap_state)
        call_budget_label = max_turns if max_turns > 0 else "unlimited"
        iteration_prompt = render_prompt(
            "iteration.md",
            iteration=iteration,
            candidate_dir=candidate_dir,
            parent_primary=f"{parent_primary:.6f}",
            best_primary=f"{best_primary:.6f}",
            max_turns=call_budget_label,
            experiment_ledger=json.dumps(
                self._experiment_memory[-self.config.AGENT_EXPERIMENT_MEMORY_LIMIT:],
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            stage_instruction=(
                "Use the retained ADK session, task context, baseline result, literature "
                "evidence, and prior experiment events to choose the next change."
            ),
        )
        self._prompt_records[iteration_prompt.name] = iteration_prompt
        self._iteration = iteration
        recovery_events: list[dict] = []
        totals = {"input": 0, "output": 0}
        unlimited = max_turns <= 0
        # Positive values remain supported as an explicit override. The
        # default zero delegates termination to the agent/provider and the
        # outer experiment and wall-clock budgets.
        remaining = max_turns + 1 if not unlimited else 0
        prompt = iteration_prompt.content
        final_text = ""
        last_error: str | None = None

        def progress() -> str:
            return (
                f"Execution attempts={len(runtime.executions)}; "
                f"parent primary={parent_primary:.6f}; best primary={best_primary:.6f}"
            )

        self._runtime = runtime
        try:
            while unlimited or remaining > 0:
                invocation = self._run_invocation(
                    prompt,
                    phase_kind="experiment",
                    max_llm_calls=0 if unlimited else remaining,
                    progress=progress,
                )
                self._add_tokens(totals, invocation.token_counts)
                recovery_events.extend(invocation.recovery_events)
                if not unlimited:
                    remaining -= invocation.model_calls
                if invocation.text:
                    final_text = invocation.text
                if invocation.error:
                    last_error = invocation.error
                    break
                successful = next(
                    (item for item in reversed(runtime.executions) if item["success"]),
                    None,
                )
                if successful is not None and final_text:
                    break
                if (not unlimited and remaining <= 0) or invocation.model_calls == 0:
                    break
                if not runtime.executions:
                    action = "continue_with_execution_requirement"
                    issue = "ADK agent ended before calling run_model"
                    prompt = (
                        "The experiment is incomplete because run_model has not been called. "
                        "Continue in this retained ADK session: implement the hypothesis, then "
                        "execute the candidate."
                    )
                elif successful is None:
                    action = "continue_with_repair_requirement"
                    issue = "ADK agent ended after a failed model execution"
                    prompt = (
                        "The latest model execution failed. Read its recorded error, make a "
                        "targeted repair, and call run_model again."
                    )
                else:
                    action = "continue_with_reflection_requirement"
                    issue = "ADK agent produced metrics without the required reflection"
                    prompt = (
                        "Interpret the latest validation metrics. Return only JSON with "
                        "reflection, hypothesis_supported, and suggested_next."
                    )
                recovery_events.append({
                    "type": "agent_protocol_recovery",
                    "phase": f"experiment_{iteration}",
                    "error": issue,
                    "action": action,
                })
        finally:
            self._runtime = None

        best_execution = next(
            (item for item in reversed(runtime.executions) if item["success"]), None
        )
        last_execution = runtime.executions[-1] if runtime.executions else None
        chosen = best_execution or last_execution
        if chosen is None and last_error is None:
            last_error = (
                "ADK agent ended without calling run_model"
                if unlimited
                else f"ADK agent ended without calling run_model within {max_turns} work calls"
            )
        elif chosen is not None and not chosen["success"] and last_error is None:
            last_error = chosen.get("error") or "candidate execution failed"

        reflection = final_text.strip()
        hypothesis_supported: bool | None = None
        suggested_next = ""
        if reflection:
            try:
                parsed = json.loads(reflection)
                if isinstance(parsed, dict):
                    reflection = str(parsed.get("reflection", reflection))
                    supported = parsed.get("hypothesis_supported")
                    if isinstance(supported, bool):
                        hypothesis_supported = supported
                    suggested_next = str(parsed.get("suggested_next", ""))
            except json.JSONDecodeError:
                pass
        if best_execution is not None and not reflection:
            reflection = "ADK produced validation metrics but no closing reflection."

        if best_execution is not None:
            for index, execution in enumerate(runtime.executions):
                if execution.get("success"):
                    continue
                if not any(
                    later.get("success") for later in runtime.executions[index + 1:]
                ):
                    continue
                recovery_events.append({
                    "type": "model_execution_recovery",
                    "phase": f"experiment_{iteration}",
                    "failed_attempt": index + 1,
                    "error": execution.get("error") or "candidate execution failed",
                    "action": "repair_candidate_and_rerun",
                    "outcome": "recovered",
                })

        final_code = model_path.read_text(encoding="utf-8") if model_path.exists() else None
        result = AgentIterationResult(
            success=best_execution is not None,
            hypothesis=str(chosen["hypothesis"]) if chosen else "Agent produced no experiment",
            reasoning=str(chosen["reasoning"]) if chosen else "",
            reflection=reflection,
            metrics=best_execution["metrics"] if best_execution else None,
            executions=runtime.executions,
            recovery_events=recovery_events,
            token_counts=totals,
            wall_seconds=time.time() - started,
            error=None if best_execution else last_error,
            final_code=final_code,
            hypothesis_supported=hypothesis_supported,
            suggested_next=suggested_next,
        )
        chosen_proposal = chosen.get("proposal", {}) if chosen else {}
        self._experiment_memory.append({
            "iteration": iteration,
            "status": "success" if result.success else "failed",
            "hypothesis": result.hypothesis,
            "target_component": chosen_proposal.get(
                "target_component", "agent_selected"
            ),
            "primary": (
                result.metrics.get("primary") if result.metrics is not None else None
            ),
            "hypothesis_supported": result.hypothesis_supported,
            "finding": result.reflection[:500],
            "suggested_next": result.suggested_next[:300],
        })
        return result
