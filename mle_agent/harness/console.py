"""Structured, colour-aware console output for agent runs.

The console is intentionally concise: it exposes decisions, tool intent, and
safe result summaries without dumping source files, full datasets, or secrets.
Colours are enabled on interactive terminals, disabled by ``NO_COLOR``, and
can be forced with ``FORCE_COLOR=1``.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import TextIO

from mle_agent.harness.tools import redact_secrets


_RESET = "\033[0m"
_BOLD = "\033[1m"
_CYAN = "\033[36m"
_BLUE = "\033[34m"
_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_RED = "\033[31m"


def _make_stream_encoding_tolerant(stream: TextIO) -> None:
    """Stop a decorative glyph from aborting a run on a non-UTF-8 stdout.

    Windows consoles and redirected pipes often default to cp1252, which cannot
    encode the box-drawing characters this console prints. Prefer UTF-8; fall
    back to replacing unencodable characters rather than raising.
    """
    reconfigure = getattr(stream, "reconfigure", None)
    if reconfigure is None:
        return
    current = (getattr(stream, "encoding", "") or "").lower().replace("-", "")
    try:
        if current not in {"utf8", "utf16", "utf32"}:
            reconfigure(encoding="utf-8", errors="replace")
        else:
            reconfigure(errors="replace")
    except (ValueError, OSError):  # pragma: no cover - platform-dependent
        pass


class RunConsole:
    def __init__(self, stream: TextIO | None = None, use_color: bool | None = None) -> None:
        self.stream = stream or sys.stdout
        _make_stream_encoding_tolerant(self.stream)
        if use_color is None:
            use_color = (
                "NO_COLOR" not in os.environ
                and (os.environ.get("FORCE_COLOR") == "1" or self.stream.isatty())
            )
        self.use_color = use_color

    def _style(self, text: str, *codes: str) -> str:
        if not self.use_color:
            return text
        return "".join(codes) + text + _RESET

    def _line(self, label: str, value: object, colour: str = "") -> None:
        styled_label = self._style(label, _BOLD, colour) if colour else self._style(label, _BOLD)
        rendered = " ".join(redact_secrets(str(value)).split())
        if len(rendered) > 800:
            rendered = rendered[:799].rstrip() + "…"
        print(f"{styled_label}: {rendered}", file=self.stream, flush=True)

    def harness(self, stage: str, **fields: object) -> None:
        print(file=self.stream)
        print(self._style("--- Harness ---", _BOLD, _BLUE), file=self.stream)
        self._line("Stage", stage, _BLUE)
        for label, value in fields.items():
            self._line(label.replace("_", " ").title(), value)
        self.stream.flush()

    def agent_reasoning(
        self,
        reasoning: str,
        *,
        phase: str,
        progress: str | None = None,
        response_text: str | None = None,
        tool_names: list[str] | None = None,
        stop_reason: str | None = None,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        call_number: int | None = None,
        response_event_id: str | None = None,
        provider: str | None = None,
        latency_seconds: float | None = None,
        tool_summaries: list[str] | None = None,
    ) -> None:
        print(file=self.stream)
        print(self._style("--- Agent ---", _BOLD, _CYAN), file=self.stream)
        self._line("Phase", phase, _CYAN)
        if call_number is not None:
            call_parts = [f"#{call_number}"]
            if response_event_id:
                call_parts.append(response_event_id)
            if provider:
                call_parts.append(provider)
            if latency_seconds is not None:
                call_parts.append(f"{latency_seconds:.2f}s")
            self._line("LLM Call", " | ".join(call_parts), _CYAN)
        self._line("Reasoning", reasoning)
        if progress:
            self._line("Progress", progress, _YELLOW)
        if stop_reason:
            usage = ""
            if input_tokens is not None and output_tokens is not None:
                usage = f"; tokens in={input_tokens}, out={output_tokens}"
            self._line("Model Event", f"stop={stop_reason}{usage}", _CYAN)
        if tool_summaries:
            self._line("Actions Requested", " | ".join(tool_summaries), _YELLOW)
        self._response_preview(response_text, tool_names or [])
        self.stream.flush()

    def _response_preview(self, response_text: str | None, tool_names: list[str]) -> None:
        if response_text and response_text.strip():
            raw_lines = redact_secrets(response_text).strip().splitlines()
            lines = raw_lines[:5]
            if len(raw_lines) > 5:
                lines[-1] = lines[-1][:380].rstrip() + " … [response truncated]"
            lines = [
                line[:400].rstrip() + ("…" if len(line) > 400 else "")
                for line in lines
            ]
        elif tool_names:
            lines = [
                "(No assistant text returned; function-call-only response for "
                + ", ".join(tool_names)
                + ".)"
            ]
        else:
            lines = ["(No assistant text or function call returned.)"]

        self._line("LLM Response", lines[0], _CYAN)
        for line in lines[1:]:
            print(f"  {self._style('│', _CYAN)} {line}", file=self.stream, flush=True)

    def agent_tool_call(self, name: str, payload: dict) -> None:
        self._line("Tool Calling", self.tool_call_summary(name, payload), _YELLOW)

    def agent_tool_result(self, name: str, output: str) -> None:
        self._line("Tool Called", name, _GREEN)
        summary, success = self.tool_result_summary(name, output)
        self._line("Returned Result", summary, _GREEN if success else _RED)

    @staticmethod
    def tool_call_summary(name: str, payload: dict) -> str:
        if name == "read_file":
            path = Path(str(payload.get("path", "?"))).name
            return f"read_file(path={path}, offset={payload.get('offset', 0)})"
        if name == "write_file":
            content = str(payload.get("content", ""))
            return f"write_file(path={payload.get('path', '?')}, chars={len(content)})"
        if name == "search_ml_literature":
            return f"search_ml_literature(query={payload.get('query', '')!r})"
        if name == "request_dependency_install":
            packages = ", ".join(str(item) for item in payload.get("packages", []))
            return f"request_dependency_install(packages={packages!r})"
        if name == "record_task_context":
            return f"record_task_context(objective={payload.get('task_objective', '')!r})"
        if name == "run_model":
            hypothesis = " ".join(str(payload.get("hypothesis", "")).split())
            reasoning = " ".join(str(payload.get("reasoning", "")).split())
            if len(reasoning) > 220:
                reasoning = reasoning[:219].rstrip() + "…"
            return f"run_model(hypothesis={hypothesis!r}; rationale={reasoning!r})"
        return f"{name}()"

    @staticmethod
    def tool_result_summary(name: str, output: str) -> tuple[str, bool]:
        safe_output = redact_secrets(output)
        try:
            payload = json.loads(safe_output)
        except json.JSONDecodeError:
            compact = " ".join(safe_output.split())
            success = not compact.startswith(("ERROR", "REJECTED", "TIMEOUT"))
            return (compact[:360] + ("…" if len(compact) > 360 else ""), success)

        success = bool(payload.get("success", True)) and not payload.get("error")
        if name == "discover_task_docs":
            docs = payload.get("documents", [])
            primary = Path(str(payload.get("primary_readme_path", "?"))).name
            return f"Found {len(docs)} task files; primary README={primary}.", success
        if name == "read_file" and "total_chars" in payload:
            path = Path(str(payload.get("path", "?"))).name
            start = payload.get("offset", 0)
            end = int(start) + int(payload.get("chars_returned", 0))
            return (
                f"Read {path} chars {start}-{end}/{payload['total_chars']}; "
                f"complete={payload.get('complete')}; next_offset={payload.get('next_offset')}.",
                success,
            )
        if name == "inspect_data":
            train = payload.get("train", {})
            valid = payload.get("valid", {})
            return (
                f"Train rows={train.get('rows', '?')}; validation rows={valid.get('rows', '?')}; "
                f"policy={payload.get('policy', '?')}.",
                success,
            )
        if name == "inspect_environment":
            packages = payload.get("packages", {})
            installed = [
                name for name, details in packages.items()
                if isinstance(details, dict) and details.get("installed")
            ]
            return f"Installed ML packages: {', '.join(installed) or 'none detected'}.", success
        if name == "request_dependency_install":
            requirements = ", ".join(str(item) for item in payload.get("requirements", []))
            operations = payload.get("operations", [])
            operation_outcomes = [
                f"{operation.get('authorization', '?')}:{operation.get('outcome', '?')}"
                for operation in operations
                if isinstance(operation, dict)
            ]
            outcome = (
                ", ".join(operation_outcomes)
                or payload.get("outcome")
                or payload.get("status")
                or payload.get("error")
            )
            return f"Dependencies={requirements or '?'}; outcome={outcome}.", success
        if name == "search_ml_literature":
            hits = payload.get("results", [])
            chunk_ids = [str(hit.get("chunk_id", "?")) for hit in hits[:3]]
            return (
                f"Retrieved {len(hits)} passages"
                + (f": {', '.join(chunk_ids)}" if chunk_ids else "")
                + f"; bootstrap_counted={payload.get('bootstrap_counted', False)}.",
                success and bool(payload.get("bootstrap_counted", False)),
            )
        if name == "record_task_context":
            if success:
                context = payload.get("task_context", {})
                return (
                    f"Retained task context for target={context.get('target_label', '?')} "
                    f"with {len(context.get('source_paths', []))} cited sources.",
                    True,
                )
            details = list(payload.get("missing_requirements", []))
            details.extend(
                f"unread citation: {Path(str(path)).name}"
                for path in payload.get("cited_sources_not_fully_read", [])
            )
            details.extend(
                f"missing citation: {Path(str(path)).name}"
                for path in payload.get("required_sources_not_cited", [])
            )
            return (
                f"{payload.get('error', 'context rejected')}; missing="
                f"{', '.join(str(item) for item in details) or 'see artifact'}.",
                False,
            )
        if name in {"reproduce_baseline", "run_model"}:
            metrics = payload.get("metrics") or {}
            if success:
                def metric_value(key: str) -> str:
                    value = metrics.get(key)
                    return f"{float(value):.6f}" if isinstance(value, (int, float)) else "?"

                wall = payload.get("wall_seconds")
                wall_text = f"{float(wall):.2f}s" if isinstance(wall, (int, float)) else "?"
                return (
                    f"GAUC={metric_value('GAUC')}; nDCG@5={metric_value('nDCG@5')}; "
                    f"primary={metric_value('primary')}; wall={wall_text}.",
                    True,
                )
            missing = payload.get("missing_requirements", [])
            detail = str(payload.get("error", "execution failed"))
            if missing:
                detail += "; missing=" + ", ".join(str(item) for item in missing)
            return detail[:360], False

        compact = " ".join(safe_output.split())
        return compact[:360] + ("…" if len(compact) > 360 else ""), success


console = RunConsole()
