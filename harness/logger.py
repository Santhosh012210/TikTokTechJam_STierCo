"""Append-before-decide JSONL logger.

The write-before-decide guarantee: every log entry is flushed + fsynced to
disk BEFORE main.py updates the search tree or reads the result. A crash
between builder return and tree update therefore never loses history.
"""
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from harness.validator import validate_row


class _NpEncoder(json.JSONEncoder):
    """Handle numpy float32/int64 values that may appear in metrics dicts."""
    def default(self, obj: Any) -> Any:
        try:
            import numpy as np
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.integer):
                return int(obj)
        except ImportError:
            pass
        return super().default(obj)


class RunLogger:
    def __init__(self, artifacts_dir: Path, run_id: str) -> None:
        self._run_dir = artifacts_dir / "runs" / run_id
        self._logs_dir = self._run_dir / "logs"
        self._results_dir = self._run_dir / "results"
        self._reports_dir = self._run_dir / "reports"
        self._path = self._logs_dir / "events.jsonl"
        self._llm_events_path = self._logs_dir / "llm_events.jsonl"
        self._token_total: dict[str, int] = {"input": 0, "output": 0}
        self._intervention_count: int = 0
        self._llm_event_sequence: int = 0
        self._llm_event_counts: dict[str, int] = {
            "llm_response": 0,
            "tool_result": 0,
            "provider_error": 0,
            "quota_pause": 0,
        }
        for directory in (self._logs_dir, self._results_dir, self._reports_dir):
            directory.mkdir(parents=True, exist_ok=True)
        # Open in append mode and keep handle open for the lifetime of the run
        self._file = open(self._path, "a", encoding="utf-8")
        self._llm_events_file = open(self._llm_events_path, "a", encoding="utf-8")

    @property
    def path(self) -> Path:
        return self._path

    @property
    def llm_events_path(self) -> Path:
        return self._llm_events_path

    @property
    def run_dir(self) -> Path:
        return self._run_dir

    def write(self, row: dict) -> None:
        """Write one log entry. Guaranteed to fsync before returning."""
        # Stamp timestamp if missing
        if "timestamp" not in row:
            row["timestamp"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

        # Accumulate token totals
        tokens = row.get("tokens") or {}
        self._token_total["input"]  += tokens.get("input",  0)
        self._token_total["output"] += tokens.get("output", 0)

        # Count manual interventions
        if row.get("human_intervention"):
            self._intervention_count += 1

        # Serialize and write
        line = json.dumps(row, cls=_NpEncoder, ensure_ascii=False)
        self._file.write(line + "\n")
        self._file.flush()
        try:
            os.fsync(self._file.fileno())
        except OSError:
            pass  # some platforms don't support fsync on text files; flush is enough

        # Sanity-check schema (warn only; never fatal)
        errors = validate_row(row)
        if errors:
            for e in errors:
                print(f"[logger] schema warning: {e}", file=sys.stderr)

    def write_llm_event(self, event: dict) -> None:
        """Append one provider/tool trace event and fsync it before returning.

        ``events.jsonl`` remains the compact experiment-level record. This
        separate stream preserves the chronological agent trace without
        changing the stable experiment schema.
        """
        row = dict(event)
        self._llm_event_sequence += 1
        row.setdefault("event_sequence", self._llm_event_sequence)
        row.setdefault(
            "timestamp", datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        )
        event_type = str(row.get("event_type", "unknown"))
        self._llm_event_counts[event_type] = self._llm_event_counts.get(event_type, 0) + 1

        self._llm_events_file.write(
            json.dumps(row, cls=_NpEncoder, ensure_ascii=False) + "\n"
        )
        self._llm_events_file.flush()
        try:
            os.fsync(self._llm_events_file.fileno())
        except OSError:
            pass

    def running_totals(self) -> dict:
        return {
            "tokens": dict(self._token_total),
            "interventions": self._intervention_count,
            "llm_trace": {
                "path": str(self._llm_events_path),
                "events": self._llm_event_sequence,
                **dict(self._llm_event_counts),
            },
        }

    def write_results(self, results: dict) -> Path:
        """Write the machine-readable run summary."""
        path = self._results_dir / "metrics.json"
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps(results, cls=_NpEncoder, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        tmp.replace(path)
        return path

    def write_report(self, report: str) -> Path:
        """Write the human-readable run report."""
        path = self._reports_dir / "summary.md"
        tmp = path.with_suffix(".md.tmp")
        tmp.write_text(report.rstrip() + "\n", encoding="utf-8")
        tmp.replace(path)
        return path

    def close(self) -> None:
        for handle in (self._file, self._llm_events_file):
            try:
                handle.flush()
                os.fsync(handle.fileno())
            except OSError:
                pass
            handle.close()
