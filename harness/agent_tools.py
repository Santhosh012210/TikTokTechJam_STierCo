"""Constrained tools for the single autonomous research agent.

Unlike the legacy Builder tool surface, this module does not expose an
arbitrary shell. The agent can read its current candidate and the official
starter kit, write candidate files, inspect a train/validation-only summary,
search the local method corpus, and execute exactly one candidate model.
"""
from __future__ import annotations

import ast
import csv
import hashlib
import json
import os
import subprocess
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from harness.config import Config
from harness.metrics import try_extract_metrics
from harness.tools import exec_read_file, exec_write_file, redact_secrets
from harness.validator import scan_candidate_source
from research_agent.knowledge import SEARCH_ML_LITERATURE_TOOL, search_ml_literature


AGENT_TOOLS = [
    {
        "name": "read_file",
        "description": (
            "Read the current candidate or an official starter-kit file. "
            "Read model.py before editing and consult the starter-kit README/data/evaluation code as needed."
        ),
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description": "Create or replace a file inside the current candidate directory.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "inspect_data",
        "description": (
            "Return a deterministic EDA summary made only from the training and validation dates. "
            "The hidden test dates are never summarized."
        ),
        "parameters": {"type": "object", "properties": {}},
    },
    SEARCH_ML_LITERATURE_TOOL,
    {
        "name": "run_model",
        "description": (
            "Execute the current model.py against train and validation only. "
            "Provide the hypothesis and evidence behind this experiment for the run log. "
            "Returns validation metrics or a traceback to repair."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "hypothesis": {
                    "type": "string",
                    "description": "The precise model or pipeline change being tested.",
                },
                "reasoning": {
                    "type": "string",
                    "description": "Why the change may improve the ranking metrics, grounded in data/history/literature.",
                },
                "literature_chunk_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Relevant chunk IDs returned by search_ml_literature.",
                },
            },
            "required": ["hypothesis", "reasoning"],
        },
    },
]


def semantic_model_fingerprint(path: Path) -> str | None:
    """Fingerprint Python meaning while ignoring comments and formatting."""
    if not path.exists():
        return None
    source = path.read_text(encoding="utf-8", errors="replace")
    try:
        normalized = ast.dump(
            ast.parse(source), annotate_fields=True, include_attributes=False
        )
        payload = f"ast:{normalized}"
    except SyntaxError:
        # A malformed edit is still a real change; execution will return the
        # syntax error so the agent can repair it.
        payload = f"invalid-python:{source}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass
class ModelExecution:
    success: bool
    metrics: dict | None
    output: str
    error: str | None
    wall_seconds: float


def _safe_float_metrics(metrics: dict) -> dict:
    return {
        key: float(value) if isinstance(value, (int, float)) or hasattr(value, "item") else value
        for key, value in metrics.items()
    }


def execute_model(candidate_dir: Path, config: Config, timeout_seconds: int = 300) -> ModelExecution:
    """Run exactly ``model.py`` without allowing agent-authored shell commands."""
    import time

    started = time.time()
    model_path = candidate_dir / "model.py"
    if not model_path.exists():
        return ModelExecution(False, None, "", "model.py does not exist", time.time() - started)

    source = model_path.read_text(encoding="utf-8", errors="replace")
    violations = scan_candidate_source(source)
    if violations:
        message = f"REJECTED: validation-only policy violations: {violations}"
        return ModelExecution(False, None, message, message, time.time() - started)

    env = {
        key: value
        for key, value in os.environ.items()
        if key not in {
            "LLM_API_KEY", "ANTHROPIC_API_KEY", "GROQ_API_KEY",
            "GEMINI_API_KEY", "OPENAI_API_KEY",
        }
    }
    env["PYTHONPATH"] = str(config.BASELINE_ROOT)
    command = [
        config.PYTHON_EXE,
        str(model_path),
        "--data_dir",
        str(config.DATA_DIR),
    ]
    try:
        result = subprocess.run(
            command,
            cwd=str(candidate_dir),
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        output = redact_secrets((exc.stdout or "") + (exc.stderr or ""))
        return ModelExecution(
            False, None, output, f"TIMEOUT after {timeout_seconds}s", time.time() - started
        )
    except Exception as exc:
        return ModelExecution(False, None, "", f"ERROR: {exc}", time.time() - started)

    output = redact_secrets(result.stdout + result.stderr)
    if len(output) > 12_000:
        output = output[-12_000:]
    metrics = try_extract_metrics(output)
    if result.returncode != 0:
        error = f"model.py exited {result.returncode}\n{output[-6000:]}"
        return ModelExecution(False, None, output, error, time.time() - started)
    if metrics is None:
        error = "model.py exited successfully but emitted no parseable validation metrics"
        return ModelExecution(False, None, output, error, time.time() - started)
    return ModelExecution(
        True, _safe_float_metrics(metrics), output, None, time.time() - started
    )


_EDA_CACHE: dict[str, object] | None = None


def inspect_train_valid_data(config: Config) -> dict[str, object]:
    """Summarize only dates through the end of validation (2022-04-28)."""
    global _EDA_CACHE
    if _EDA_CACHE is not None:
        return _EDA_CACHE

    split_stats = {
        "train": {"rows": 0, "positives": 0, "users": Counter()},
        "valid": {"rows": 0, "positives": 0, "users": Counter()},
    }
    auxiliary = Counter()
    auxiliary_total = 0
    files = (
        config.DATA_DIR / "log_standard_4_08_to_4_21_pure.csv",
        config.DATA_DIR / "log_standard_4_22_to_5_08_pure.csv",
    )
    aux_fields = ("is_click", "is_like", "is_follow", "is_comment", "is_forward")

    for path in files:
        with path.open(encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                date = int(row["date"])
                if 20220408 <= date <= 20220421:
                    split = "train"
                elif 20220422 <= date <= 20220428:
                    split = "valid"
                else:
                    continue
                stats = split_stats[split]
                stats["rows"] += 1
                stats["positives"] += int(row.get("long_view", "0") != "0")
                stats["users"][row["user_id"]] += 1
                if split == "train":
                    auxiliary_total += 1
                    for field in aux_fields:
                        auxiliary[field] += int(row.get(field, "0") not in ("", "0"))

    summary: dict[str, object] = {"policy": "train and validation dates only; test excluded"}
    for split, stats in split_stats.items():
        counts = sorted(stats["users"].values())
        rows = int(stats["rows"])
        summary[split] = {
            "rows": rows,
            "users": len(counts),
            "long_view_rate": round(int(stats["positives"]) / rows, 6) if rows else 0.0,
            "impressions_per_user_p50": counts[len(counts) // 2] if counts else 0,
            "impressions_per_user_p90": counts[int(0.9 * (len(counts) - 1))] if counts else 0,
        }
    summary["train_auxiliary_positive_rates"] = {
        field: round(auxiliary[field] / auxiliary_total, 6) if auxiliary_total else 0.0
        for field in aux_fields
    }
    _EDA_CACHE = summary
    return summary


class AgentToolRuntime:
    """Per-iteration dispatcher with captured experiment evidence."""

    def __init__(self, candidate_dir: Path, config: Config) -> None:
        self.candidate_dir = candidate_dir
        self.config = config
        self.executions: list[dict] = []
        self.inherited_model_fingerprint = semantic_model_fingerprint(
            candidate_dir / "model.py"
        )

    def dispatch(self, name: str, payload: dict) -> str:
        print(f"[agent/tool] {name}", flush=True)
        if name == "read_file":
            content = exec_read_file(
                payload["path"], self.candidate_dir, self.config.BASELINE_ROOT
            )
            if len(content) > self.config.AGENT_READ_MAX_CHARS:
                content = content[:self.config.AGENT_READ_MAX_CHARS] + "\n... [agent read truncated]"
            return content
        if name == "write_file":
            return exec_write_file(payload["path"], payload["content"], self.candidate_dir)
        if name == "inspect_data":
            return json.dumps(inspect_train_valid_data(self.config), indent=2)
        if name == "search_ml_literature":
            return json.dumps(
                search_ml_literature(payload["query"], int(payload.get("k", 3))),
                ensure_ascii=False,
                indent=2,
            )
        if name == "run_model":
            candidate_changed = (
                semantic_model_fingerprint(self.candidate_dir / "model.py")
                != self.inherited_model_fingerprint
            )
            if not candidate_changed:
                error = (
                    "REJECTED: model.py is semantically unchanged from the inherited "
                    "candidate. Implement the proposed experiment with write_file before "
                    "calling run_model; comments or formatting alone do not count."
                )
                record = {
                    "hypothesis": str(payload.get("hypothesis", "")),
                    "reasoning": str(payload.get("reasoning", "")),
                    "literature_chunk_ids": list(payload.get("literature_chunk_ids", [])),
                    "success": False,
                    "metrics": None,
                    "error": error,
                    "wall_seconds": 0.0,
                    "candidate_changed": False,
                }
                self.executions.append(record)
                print(f"[agent/tool] model rejected: {error}", flush=True)
                return json.dumps(record, ensure_ascii=False, indent=2)

            execution = execute_model(self.candidate_dir, self.config)
            record = {
                "hypothesis": str(payload.get("hypothesis", "")),
                "reasoning": str(payload.get("reasoning", "")),
                "literature_chunk_ids": list(payload.get("literature_chunk_ids", [])),
                "success": execution.success,
                "metrics": execution.metrics,
                "error": execution.error,
                "wall_seconds": execution.wall_seconds,
                "candidate_changed": True,
            }
            self.executions.append(record)
            if execution.success:
                print(
                    f"[agent/tool] validation primary={execution.metrics['primary']:.6f}",
                    flush=True,
                )
            else:
                print(f"[agent/tool] model failed: {execution.error}", flush=True)
            return json.dumps(record, ensure_ascii=False, indent=2)
        return f"ERROR: unknown tool {name!r}"
