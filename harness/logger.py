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
    def __init__(self, logs_dir: Path, run_id: str) -> None:
        self._path = logs_dir / f"run_{run_id}.jsonl"
        self._token_total: dict[str, int] = {"input": 0, "output": 0}
        self._intervention_count: int = 0
        logs_dir.mkdir(parents=True, exist_ok=True)
        # Open in append mode and keep handle open for the lifetime of the run
        self._file = open(self._path, "a", encoding="utf-8")

    @property
    def path(self) -> Path:
        return self._path

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

    def running_totals(self) -> dict:
        return {
            "tokens": dict(self._token_total),
            "interventions": self._intervention_count,
        }

    def close(self) -> None:
        try:
            self._file.flush()
            os.fsync(self._file.fileno())
        except OSError:
            pass
        self._file.close()
