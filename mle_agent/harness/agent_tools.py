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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name

from mle_agent.harness.config import Config
from mle_agent.harness.metrics import try_extract_metrics
from mle_agent.harness.run_environment import (
    AUTO_INSTALL_ALLOWLIST,
    snapshot_run_environment,
)
from mle_agent.harness.tools import exec_write_file, redact_secrets
from mle_agent.harness.validator import scan_candidate_source
from mle_agent.research_agent.knowledge import SEARCH_ML_LITERATURE_TOOL, search_ml_literature


AGENT_TOOLS = [
    {
        "name": "discover_task_docs",
        "description": (
            "Discover the task README and important benchmark-support files in the official "
            "starter kit. Call this once at the start of the first experiment."
        ),
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "read_file",
        "description": (
            "Read one explicit page from the current candidate or an official starter-kit file. "
            "Continue from next_offset until complete=true; content is never silently truncated."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "offset": {
                    "type": "integer",
                    "minimum": 0,
                    "description": "Character offset to start at (default 0).",
                },
                "max_chars": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "Requested page size, capped by the harness read limit.",
                },
            },
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
            "Return a deterministic EDA summary and candidate-data column inventory made only "
            "from the training and validation view. The hidden test dates are never summarized."
        ),
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "inspect_environment",
        "description": (
            "Inspect the Python runtime and available open-source ML frameworks without importing "
            "or installing them. Required during bootstrap so framework choice is evidence-based."
        ),
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "request_dependency_install",
        "description": (
            "Request installation of missing PyPI packages into the run's Python environment. "
            "Allowlisted ML packages install automatically as binary wheels in the dedicated run "
            "venv. Off-allowlist packages require explicit user approval. Every outcome is logged; "
            "URLs, extras, markers, and pip flags are rejected."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "packages": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "One to six PyPI requirement specifiers, such as lightgbm or torch==2.8.0.",
                },
                "justification": {
                    "type": "string",
                    "description": "Why these packages are necessary for the current research hypothesis.",
                },
            },
            "required": ["packages", "justification"],
        },
    },
    SEARCH_ML_LITERATURE_TOOL,
    {
        "name": "reproduce_baseline",
        "description": (
            "Execute the unchanged inherited baseline candidate after reading the task README, "
            "baseline, evaluation, feature/data, feature-ablation, and model code. Verifies the "
            "official validation score before research."
        ),
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "record_task_context",
        "description": (
            "Record the structured benchmark understanding after fully reading the required "
            "sources, inspecting data, and searching the literature. The stored summary remains "
            "available to every later experiment."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "task_objective": {"type": "string"},
                "target_label": {"type": "string"},
                "metrics": {"type": "array", "items": {"type": "string"}},
                "data_splits": {
                    "type": "object",
                    "properties": {
                        "train": {"type": "string"},
                        "validation": {"type": "string"},
                        "test": {"type": "string"},
                    },
                    "required": ["train", "validation", "test"],
                },
                "baseline": {"type": "string"},
                "evaluation_protocol": {"type": "array", "items": {"type": "string"}},
                "hard_constraints": {"type": "array", "items": {"type": "string"}},
                "known_dead_ends": {"type": "array", "items": {"type": "string"}},
                "promising_directions": {"type": "array", "items": {"type": "string"}},
                "feature_engineering_context": {
                    "type": "object",
                    "properties": {
                        "baseline_fields": {"type": "array", "items": {"type": "string"}},
                        "measured_dead_ends": {"type": "array", "items": {"type": "string"}},
                        "promising_feature_families": {"type": "array", "items": {"type": "string"}},
                        "leakage_controls": {"type": "array", "items": {"type": "string"}},
                        "implementation_boundary": {"type": "string"},
                    },
                    "required": [
                        "baseline_fields",
                        "measured_dead_ends",
                        "promising_feature_families",
                        "leakage_controls",
                        "implementation_boundary",
                    ],
                },
                "candidate_contract": {"type": "array", "items": {"type": "string"}},
                "source_paths": {"type": "array", "items": {"type": "string"}},
            },
            "required": [
                "task_objective",
                "target_label",
                "metrics",
                "data_splits",
                "baseline",
                "evaluation_protocol",
                "hard_constraints",
                "known_dead_ends",
                "promising_directions",
                "feature_engineering_context",
                "candidate_contract",
                "source_paths",
            ],
        },
    },
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
                "target_component": {
                    "type": "string",
                    "enum": [
                        "loss",
                        "sampling",
                        "features",
                        "sequence",
                        "auxiliary-task",
                        "model",
                        "training",
                        "evaluation",
                    ],
                    "description": "Primary component changed: loss, sampling, features, sequence, auxiliary-task, model, or training.",
                },
                "expected_effect": {
                    "type": "string",
                    "description": "Expected effect on the official metrics and why.",
                },
                "falsification_criterion": {
                    "type": "string",
                    "description": "Metric result that would reject the hypothesis.",
                },
                "rollback_plan": {
                    "type": "string",
                    "description": "How to return to the incumbent if the experiment fails.",
                },
                "feature_sources": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Candidate-data columns or train-derived histories used by a feature-oriented experiment.",
                },
                "feature_transformations": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Exact joins, buckets, crosses, aggregates, or sequence transformations being tested.",
                },
                "leakage_controls": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "How every fitted statistic avoids validation labels and future information.",
                },
            },
            "required": ["hypothesis", "reasoning", "target_component"],
        },
    },
]


@dataclass
class BootstrapState:
    """Machine-checkable task understanding shared across all experiments."""

    required: bool = True
    discovery_completed: bool = False
    discovered_documents: list[dict[str, object]] = field(default_factory=list)
    primary_readme_path: str | None = None
    required_evaluation_path: str | None = None
    required_baseline_path: str | None = None
    required_data_path: str | None = None
    required_feature_ablation_path: str | None = None
    required_candidate_model_path: str | None = None
    read_coverage: dict[str, list[tuple[int, int]]] = field(default_factory=dict)
    fully_read_paths: set[str] = field(default_factory=set)
    data_inspected: bool = False
    environment_inspected: bool = False
    environment_inventory: dict[str, object] | None = None
    literature_queries: list[str] = field(default_factory=list)
    baseline_reproduced: bool = False
    baseline_metrics: dict[str, object] | None = None
    baseline_execution: dict[str, object] | None = None
    task_context: dict[str, object] | None = None
    rejected_actions: list[dict[str, object]] = field(default_factory=list)
    seen_candidate_fingerprints: set[str] = field(default_factory=set)

    @property
    def complete(self) -> bool:
        return not self.required or not self.missing_requirements()

    def missing_requirements(self, before_context: bool = False) -> list[str]:
        if not self.required:
            return []
        missing: list[str] = []
        if not self.discovery_completed:
            missing.append("discover task documents")
        if not self.primary_readme_path:
            missing.append("identify the primary task README")
        elif self.primary_readme_path not in self.fully_read_paths:
            missing.append("fully read the primary task README")
        if not self.required_evaluation_path:
            missing.append("identify the official evaluation code")
        elif self.required_evaluation_path not in self.fully_read_paths:
            missing.append("fully read the official evaluation code")
        if not self.required_baseline_path:
            missing.append("identify the official baseline implementation")
        elif self.required_baseline_path not in self.fully_read_paths:
            missing.append("fully read the official baseline implementation")
        if not self.required_data_path:
            missing.append("identify the official feature/data implementation")
        elif self.required_data_path not in self.fully_read_paths:
            missing.append("fully read the official feature/data implementation")
        if not self.required_feature_ablation_path:
            missing.append("identify the organizer feature ablation")
        elif self.required_feature_ablation_path not in self.fully_read_paths:
            missing.append("fully read the organizer feature ablation")
        if not self.required_candidate_model_path:
            missing.append("identify the inherited candidate model.py")
        elif self.required_candidate_model_path not in self.fully_read_paths:
            missing.append("fully read the inherited candidate model.py")
        if not self.data_inspected:
            missing.append("inspect the train/validation data")
        if not self.environment_inspected:
            missing.append("inspect the available ML environment")
        if not self.literature_queries:
            missing.append("search the ML literature corpus")
        if not self.baseline_reproduced:
            missing.append("reproduce the official baseline")
        if not before_context and self.task_context is None:
            missing.append("record the structured task context")
        return missing

    def missing_baseline_prerequisites(self) -> list[str]:
        """Reading required before the agent may reproduce the baseline."""
        if not self.required:
            return []
        missing: list[str] = []
        if not self.discovery_completed:
            missing.append("discover task documents")
        required_reads = (
            (self.primary_readme_path, "fully read the primary task README"),
            (self.required_evaluation_path, "fully read the official evaluation code"),
            (self.required_baseline_path, "fully read the official baseline implementation"),
            (self.required_data_path, "fully read the official feature/data implementation"),
            (self.required_feature_ablation_path, "fully read the organizer feature ablation"),
            (self.required_candidate_model_path, "fully read the inherited candidate model.py"),
        )
        for path, label in required_reads:
            if path is None or path not in self.fully_read_paths:
                missing.append(label)
        return missing

    def record_read(self, path: Path, start: int, end: int, total: int) -> None:
        key = str(path.resolve())
        ranges = sorted(self.read_coverage.get(key, []) + [(start, end)])
        merged: list[tuple[int, int]] = []
        for range_start, range_end in ranges:
            if not merged or range_start > merged[-1][1]:
                merged.append((range_start, range_end))
            else:
                merged[-1] = (merged[-1][0], max(merged[-1][1], range_end))
        self.read_coverage[key] = merged
        if merged and merged[0][0] == 0 and merged[0][1] >= total:
            self.fully_read_paths.add(key)

    def evidence(self) -> dict[str, object]:
        return {
            "required": self.required,
            "complete": self.complete,
            "missing_requirements": self.missing_requirements(),
            "discovery_completed": self.discovery_completed,
            "discovered_documents": self.discovered_documents,
            "primary_readme_path": self.primary_readme_path,
            "required_evaluation_path": self.required_evaluation_path,
            "required_baseline_path": self.required_baseline_path,
            "required_data_path": self.required_data_path,
            "required_feature_ablation_path": self.required_feature_ablation_path,
            "required_candidate_model_path": self.required_candidate_model_path,
            "fully_read_paths": sorted(self.fully_read_paths),
            "read_coverage": {
                path: [[start, end] for start, end in ranges]
                for path, ranges in sorted(self.read_coverage.items())
            },
            "data_inspected": self.data_inspected,
            "environment_inspected": self.environment_inspected,
            "environment_inventory": self.environment_inventory,
            "literature_queries": self.literature_queries,
            "baseline_reproduced": self.baseline_reproduced,
            "baseline_metrics": self.baseline_metrics,
            "baseline_execution": self.baseline_execution,
            "task_context": self.task_context,
            "rejected_actions": self.rejected_actions,
            "successful_candidate_fingerprints": len(self.seen_candidate_fingerprints),
        }


def _resolve_read_path(path: str, candidate_dir: Path, starter_kit_root: Path) -> Path:
    requested = Path(path)
    if requested.is_absolute():
        resolved = requested.resolve()
    else:
        candidate_path = (candidate_dir / requested).resolve()
        starter_path = (starter_kit_root / requested).resolve()
        resolved = candidate_path if candidate_path.exists() else starter_path
    allowed = (
        resolved.is_relative_to(candidate_dir.resolve())
        or resolved.is_relative_to(starter_kit_root.resolve())
    )
    if not allowed:
        raise ValueError(
            f"'{path}' is outside allowed directories; only the candidate and starter kit are readable"
        )
    if not resolved.exists():
        raise FileNotFoundError(f"file not found: {resolved}")
    if not resolved.is_file():
        raise ValueError(f"not a file: {resolved}")
    return resolved


def _discover_task_documents(starter_kit_root: Path) -> tuple[list[dict[str, object]], Path | None]:
    readmes = sorted(
        (
            path for path in starter_kit_root.rglob("*")
            if path.is_file()
            and (path.name.lower().startswith("readme"))
            and path.suffix.lower() in {".md", ".txt", ".rst"}
        ),
        key=lambda path: (
            0 if path.parent == starter_kit_root and path.name.lower() == "readme.md" else 1,
            len(path.relative_to(starter_kit_root).parts),
            str(path).lower(),
        ),
    )
    support_names = (
        "baseline.py",
        "evaluate.py",
        "data.py",
        "ablation_features.py",
        "baseline_scores.json",
    )
    support = [starter_kit_root / name for name in support_names if (starter_kit_root / name).is_file()]
    ordered = readmes + [path for path in support if path not in readmes]
    documents: list[dict[str, object]] = []
    for path in ordered:
        text = path.read_text(encoding="utf-8", errors="replace")
        documents.append({
            "path": str(path.resolve()),
            "relative_path": str(path.relative_to(starter_kit_root)),
            "kind": "task_readme" if path in readmes else "benchmark_support",
            "chars": len(text),
            "lines": len(text.splitlines()),
        })
    return documents, readmes[0].resolve() if readmes else None


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
            "GEMINI_API_KEY", "GOOGLE_API_KEY", "OPENAI_API_KEY",
        }
    }
    env["PYTHONPATH"] = os.pathsep.join((
        str(config.BASELINE_ROOT),
        str(config.PROJECT_ROOT),
    ))
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


_EDA_CACHE: dict[str, dict[str, object]] = {}


def _candidate_data_inventory(data_dir: Path) -> dict[str, object]:
    """Describe usable columns without returning raw rows or hidden-split information."""
    roles = {
        "log_standard_4_08_to_4_21_pure.csv": "training interactions",
        "log_standard_4_22_to_5_08_pure.csv": "validation interactions only in this filtered view",
        "user_features_pure.csv": "static user-side features keyed by user_id",
        "video_features_basic_pure.csv": "static item metadata keyed by video_id",
        "video_features_statistic_pure.csv": "organizer-provided item statistics keyed by video_id",
    }
    files: dict[str, object] = {}
    for name, role in roles.items():
        path = data_dir / name
        with path.open(encoding="utf-8", newline="") as handle:
            columns = csv.DictReader(handle).fieldnames or []
        files[name] = {"role": role, "columns": columns}
    return {
        "root": str(data_dir.resolve()),
        "raw_files_are_immutable_inputs": True,
        "baseline_fields": ["user_id", "video_id", "author_id", "tab", "dur_bucket"],
        "files": files,
        "implementation_boundary": (
            "Read these CSVs through --data_dir, but implement all feature joins, train-fitted "
            "encoders, histories, and transformations in the self-contained candidate model.py."
        ),
    }


def inspect_train_valid_data(config: Config) -> dict[str, object]:
    """Summarize only dates through the end of validation (2022-04-28)."""
    cache_key = str(config.DATA_DIR.resolve())
    if cache_key in _EDA_CACHE:
        return _EDA_CACHE[cache_key]

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
    summary["candidate_data"] = _candidate_data_inventory(config.DATA_DIR)
    _EDA_CACHE[cache_key] = summary
    return summary


_ML_DISTRIBUTIONS = (
    "catboost",
    "imbalanced-learn",
    "lightgbm",
    "numpy",
    "optuna",
    "pandas",
    "polars",
    "pyarrow",
    "recbole",
    "scikit-learn",
    "scipy",
    "statsmodels",
    "torch",
    "torchaudio",
    "torchvision",
    "torchrec",
    "transformers",
    "xgboost",
    "tensorflow",
)


def _probe_python_environment(
    config: Config, distributions: list[str] | tuple[str, ...]
) -> dict[str, object]:
    """Query the exact interpreter used for candidate execution."""
    names = list(dict.fromkeys(distributions))
    probe = (
        "import importlib.metadata, json, platform, sys\n"
        "names = json.loads(sys.argv[1])\n"
        "packages = {}\n"
        "for name in names:\n"
        "    try:\n"
        "        version = importlib.metadata.version(name)\n"
        "    except importlib.metadata.PackageNotFoundError:\n"
        "        version = None\n"
        "    packages[name] = {'installed': version is not None, 'version': version}\n"
        "print(json.dumps({\n"
        "    'success': True,\n"
        "    'python_executable': sys.executable,\n"
        "    'python_version': platform.python_version(),\n"
        "    'platform': platform.platform(),\n"
        "    'packages': packages,\n"
        "}))\n"
    )
    try:
        result = subprocess.run(
            [config.PYTHON_EXE, "-c", probe, json.dumps(names)],
            cwd=str(config.PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=30,
        )
    except Exception as exc:
        return {
            "success": False,
            "python_executable": config.PYTHON_EXE,
            "python_version": None,
            "platform": None,
            "packages": {
                name: {"installed": False, "version": None} for name in names
            },
            "error": f"{type(exc).__name__}: {exc}",
        }
    if result.returncode != 0:
        return {
            "success": False,
            "python_executable": config.PYTHON_EXE,
            "python_version": None,
            "platform": None,
            "packages": {
                name: {"installed": False, "version": None} for name in names
            },
            "error": redact_secrets((result.stdout + result.stderr)[-2000:]),
        }
    try:
        parsed = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        return {
            "success": False,
            "python_executable": config.PYTHON_EXE,
            "python_version": None,
            "platform": None,
            "packages": {
                name: {"installed": False, "version": None} for name in names
            },
            "error": f"environment probe returned invalid JSON: {exc}",
        }
    if not isinstance(parsed, dict) or not isinstance(parsed.get("packages"), dict):
        return {
            "success": False,
            "python_executable": config.PYTHON_EXE,
            "python_version": None,
            "platform": None,
            "packages": {
                name: {"installed": False, "version": None} for name in names
            },
            "error": "environment probe returned an invalid object",
        }
    return parsed


def inspect_ml_environment(config: Config) -> dict[str, object]:
    """Return package availability without importing heavyweight frameworks."""
    inventory = _probe_python_environment(config, _ML_DISTRIBUTIONS)
    inventory["policy"] = (
        "The hackathon permits open-source frameworks. Missing allowlisted ML packages "
        "auto-install as binary wheels in this run's dedicated venv; off-allowlist "
        "packages require explicit approval. Every resolution is logged and frozen."
    )
    inventory["environment_dir"] = (
        str(config.RUN_ENV_DIR) if config.RUN_ENV_DIR is not None else None
    )
    inventory["auto_install_allowlist"] = sorted(AUTO_INSTALL_ALLOWLIST)
    inventory["binary_only_installs"] = True
    return inventory


def _validated_requirements(packages: object) -> list[tuple[str, Requirement]]:
    if not isinstance(packages, list) or not 1 <= len(packages) <= 6:
        raise ValueError("packages must contain between one and six PyPI requirements")
    validated: list[tuple[str, Requirement]] = []
    for raw in packages:
        if not isinstance(raw, str) or not raw.strip():
            raise ValueError("every package requirement must be a non-empty string")
        try:
            requirement = Requirement(raw.strip())
        except InvalidRequirement as exc:
            raise ValueError(f"invalid PyPI requirement {raw!r}: {exc}") from exc
        if requirement.url is not None or requirement.marker is not None:
            raise ValueError("dependency URLs and environment markers are not allowed")
        if requirement.extras:
            raise ValueError("dependency extras are not allowed; request the base package")
        validated.append((str(requirement), requirement))
    return validated


def _requirement_is_satisfied(
    requirement: Requirement, package_inventory: dict[str, object]
) -> bool:
    details = package_inventory.get(requirement.name, {})
    if not isinstance(details, dict) or not details.get("installed"):
        return False
    installed = str(details.get("version", ""))
    if requirement.extras:
        return False
    return not requirement.specifier or requirement.specifier.contains(
        installed, prereleases=True
    )


def install_python_dependencies(
    config: Config, requirements: list[str], timeout_seconds: int = 1200
) -> dict[str, object]:
    """Install binary wheels only into the dedicated per-run venv."""
    if config.RUN_ENV_DIR is None:
        return {
            "success": False,
            "requirements": requirements,
            "installed_versions": {},
            "error": "dedicated per-run Python environment is not configured",
            "output_tail": "",
        }
    expected_python = config.RUN_ENV_DIR / (
        "Scripts/python.exe" if os.name == "nt" else "bin/python"
    )
    configured_python = Path(config.PYTHON_EXE)
    if not configured_python.is_absolute():
        configured_python = config.PROJECT_ROOT / configured_python
    if configured_python.absolute() != expected_python.absolute():
        return {
            "success": False,
            "requirements": requirements,
            "installed_versions": {},
            "error": "refusing to install outside the dedicated per-run venv",
            "output_tail": "",
        }
    command = [
        config.PYTHON_EXE,
        "-m",
        "pip",
        "--isolated",
        "install",
        "--disable-pip-version-check",
        "--no-input",
        "--only-binary=:all:",
        *requirements,
    ]
    env = dict(os.environ)
    env["PIP_REQUIRE_VIRTUALENV"] = "true"
    env["PYTHONNOUSERSITE"] = "1"
    try:
        result = subprocess.run(
            command,
            cwd=str(config.PROJECT_ROOT),
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
        output = redact_secrets(result.stdout + result.stderr)
        requested_names = [Requirement(spec).name for spec in requirements]
        installed_packages = _probe_python_environment(
            config, requested_names
        )["packages"]
        installed_versions = {
            name: (
                details.get("version")
                if isinstance(details, dict) and details.get("installed")
                else None
            )
            for name, details in installed_packages.items()
        }
        snapshot = snapshot_run_environment(
            config, phase="dependency_install"
        ) if result.returncode == 0 else None
        return {
            "success": result.returncode == 0,
            "returncode": result.returncode,
            "requirements": requirements,
            "installed_versions": installed_versions,
            "output_tail": output[-4000:],
            "binary_only": True,
            "environment_snapshot": snapshot,
        }
    except subprocess.TimeoutExpired as exc:
        output = redact_secrets((exc.stdout or "") + (exc.stderr or ""))
        return {
            "success": False,
            "requirements": requirements,
            "installed_versions": {},
            "error": f"dependency installation timed out after {timeout_seconds}s",
            "output_tail": output[-4000:],
        }
    except Exception as exc:
        return {
            "success": False,
            "requirements": requirements,
            "installed_versions": {},
            "error": f"{type(exc).__name__}: {exc}",
            "output_tail": "",
        }


class AgentToolRuntime:
    """Per-iteration dispatcher with captured experiment evidence."""

    def __init__(
        self,
        candidate_dir: Path,
        config: Config,
        bootstrap_state: BootstrapState | None = None,
        dependency_approver: Callable[[list[str], str], bool] | None = None,
        dependency_installer: Callable[[Config, list[str]], dict[str, object]] | None = None,
    ) -> None:
        self.candidate_dir = candidate_dir
        self.config = config
        self.bootstrap_state = bootstrap_state or BootstrapState()
        self.dependency_approver = dependency_approver
        self.dependency_installer = dependency_installer or install_python_dependencies
        self.dependency_events: list[dict[str, object]] = []
        self.executions: list[dict] = []
        self.model_post_save_failed = False
        self.inherited_model_fingerprint = semantic_model_fingerprint(
            candidate_dir / "model.py"
        )
        if self.bootstrap_state.required_candidate_model_path is None:
            self.bootstrap_state.required_candidate_model_path = str(
                (candidate_dir / "model.py").resolve()
            )

    def _bootstrap_rejection(self, action: str) -> str:
        missing = self.bootstrap_state.missing_requirements()
        record: dict[str, object] = {
            "action": action,
            "error": "BOOTSTRAP_REQUIRED",
            "missing_requirements": missing,
        }
        self.bootstrap_state.rejected_actions.append(record)
        return json.dumps({"success": False, **record}, ensure_ascii=False, indent=2)

    def dispatch(self, name: str, payload: dict) -> str:
        if name == "discover_task_docs":
            documents, primary_readme = _discover_task_documents(self.config.BASELINE_ROOT)
            self.bootstrap_state.discovery_completed = True
            self.bootstrap_state.discovered_documents = documents
            self.bootstrap_state.primary_readme_path = (
                str(primary_readme) if primary_readme is not None else None
            )
            evaluation_path = (self.config.BASELINE_ROOT / "evaluate.py").resolve()
            self.bootstrap_state.required_evaluation_path = (
                str(evaluation_path) if evaluation_path.is_file() else None
            )
            baseline_path = (self.config.BASELINE_ROOT / "baseline.py").resolve()
            self.bootstrap_state.required_baseline_path = (
                str(baseline_path) if baseline_path.is_file() else None
            )
            data_path = (self.config.BASELINE_ROOT / "data.py").resolve()
            self.bootstrap_state.required_data_path = (
                str(data_path) if data_path.is_file() else None
            )
            feature_ablation_path = (
                self.config.BASELINE_ROOT / "ablation_features.py"
            ).resolve()
            self.bootstrap_state.required_feature_ablation_path = (
                str(feature_ablation_path) if feature_ablation_path.is_file() else None
            )
            return json.dumps({
                "starter_kit_root": str(self.config.BASELINE_ROOT.resolve()),
                "primary_readme_path": self.bootstrap_state.primary_readme_path,
                "required_support_paths": [
                    path for path in [
                        self.bootstrap_state.required_baseline_path,
                        self.bootstrap_state.required_evaluation_path,
                        self.bootstrap_state.required_data_path,
                        self.bootstrap_state.required_feature_ablation_path,
                    ]
                    if path is not None
                ],
                "documents": documents,
                "instruction": (
                    "Read the primary README, official baseline, evaluation code, data.py, "
                    "feature ablation, and inherited model.py until every response is complete."
                ),
            }, ensure_ascii=False, indent=2)
        if name == "read_file":
            try:
                path = _resolve_read_path(
                    str(payload["path"]), self.candidate_dir, self.config.BASELINE_ROOT
                )
                content = path.read_text(encoding="utf-8", errors="replace")
                offset = max(0, int(payload.get("offset", 0)))
                requested_chars = int(
                    payload.get("max_chars", self.config.AGENT_READ_MAX_CHARS)
                )
                page_size = max(1, min(requested_chars, self.config.AGENT_READ_MAX_CHARS))
                if offset > len(content):
                    raise ValueError(
                        f"offset {offset} exceeds file length {len(content)} for {path}"
                    )
                end = min(len(content), offset + page_size)
                self.bootstrap_state.record_read(path, offset, end, len(content))
                return json.dumps({
                    "path": str(path),
                    "offset": offset,
                    "chars_returned": end - offset,
                    "total_chars": len(content),
                    "next_offset": None if end >= len(content) else end,
                    "complete": end >= len(content),
                    "content": content[offset:end],
                }, ensure_ascii=False, indent=2)
            except (KeyError, TypeError, ValueError, OSError) as exc:
                return json.dumps({
                    "success": False,
                    "error": f"READ_ERROR: {exc}",
                }, ensure_ascii=False, indent=2)
        if name == "write_file":
            if not self.bootstrap_state.complete:
                return self._bootstrap_rejection("write_file")
            result = exec_write_file(
                payload["path"], payload["content"], self.candidate_dir
            )
            target = (self.candidate_dir / str(payload["path"])).resolve()
            if target == (self.candidate_dir / "model.py").resolve():
                if result.startswith("FAILED:"):
                    self.model_post_save_failed = True
                elif result.startswith("OK:"):
                    self.model_post_save_failed = False
            return result
        if name == "inspect_data":
            summary = inspect_train_valid_data(self.config)
            self.bootstrap_state.data_inspected = True
            return json.dumps(summary, indent=2)
        if name == "inspect_environment":
            inventory = inspect_ml_environment(self.config)
            self.bootstrap_state.environment_inspected = True
            self.bootstrap_state.environment_inventory = inventory
            return json.dumps(inventory, ensure_ascii=False, indent=2)
        if name == "request_dependency_install":
            if not self.bootstrap_state.complete:
                return self._bootstrap_rejection("request_dependency_install")
            justification = str(payload.get("justification", "")).strip()
            if not justification:
                event = {
                    "type": "dependency_request",
                    "requirements": list(payload.get("packages") or []),
                    "justification": "",
                    "success": False,
                    "human_intervention": False,
                    "outcome": "justification_required",
                    "error": "DEPENDENCY_JUSTIFICATION_REQUIRED",
                }
                self.dependency_events.append(event)
                return json.dumps({
                    **event,
                }, ensure_ascii=False, indent=2)
            try:
                validated = _validated_requirements(payload.get("packages"))
            except ValueError as exc:
                event = {
                    "type": "dependency_request",
                    "requirements": list(payload.get("packages") or []),
                    "justification": justification,
                    "success": False,
                    "human_intervention": False,
                    "outcome": "invalid_request",
                    "error": "INVALID_DEPENDENCY_REQUEST",
                    "detail": str(exc),
                }
                self.dependency_events.append(event)
                return json.dumps({
                    **event,
                }, ensure_ascii=False, indent=2)
            requested_environment = _probe_python_environment(
                self.config, [requirement.name for _, requirement in validated]
            )
            pending = [
                (spec, requirement)
                for spec, requirement in validated
                if not _requirement_is_satisfied(
                    requirement, requested_environment["packages"]
                )
            ]
            if not pending:
                installed_versions = {
                    requirement.name: requested_environment["packages"][
                        requirement.name
                    ].get("version")
                    for _, requirement in validated
                }
                event = {
                    "type": "dependency_request",
                    "requirements": [spec for spec, _ in validated],
                    "justification": justification,
                    "authorization": "already_available",
                    "approval_required": False,
                    "success": True,
                    "human_intervention": False,
                    "outcome": "already_available",
                    "installed_versions": installed_versions,
                    "environment": self.config.PYTHON_EXE,
                }
                self.dependency_events.append(event)
                return json.dumps({
                    **event,
                    "status": "ALREADY_AVAILABLE",
                }, ensure_ascii=False, indent=2)

            auto_pending = [
                spec for spec, requirement in pending
                if canonicalize_name(requirement.name) in AUTO_INSTALL_ALLOWLIST
            ]
            restricted_pending = [
                spec for spec, requirement in pending
                if canonicalize_name(requirement.name) not in AUTO_INSTALL_ALLOWLIST
            ]
            operations: list[dict[str, object]] = []

            if auto_pending:
                install_result = self.dependency_installer(
                    self.config, auto_pending
                )
                event = {
                    "type": "dependency_install",
                    "requirements": auto_pending,
                    "justification": justification,
                    "authorization": "auto_allowlist",
                    "approval_required": False,
                    "approved": True,
                    "binary_only": True,
                    "success": bool(install_result.get("success")),
                    "human_intervention": False,
                    "outcome": (
                        "installed" if install_result.get("success")
                        else "install_failed"
                    ),
                    "installed_versions": install_result.get(
                        "installed_versions", {}
                    ),
                    "environment": self.config.PYTHON_EXE,
                    "environment_snapshot": install_result.get(
                        "environment_snapshot"
                    ),
                    "error": install_result.get("error"),
                    "output_tail": str(
                        install_result.get("output_tail", "")
                    )[-1000:],
                }
                self.dependency_events.append(event)
                operations.append(event)

            if restricted_pending and self.dependency_approver is None:
                event = {
                    "type": "dependency_install",
                    "requirements": restricted_pending,
                    "justification": justification,
                    "authorization": "off_allowlist",
                    "approval_required": True,
                    "approved": False,
                    "binary_only": True,
                    "success": False,
                    "human_intervention": False,
                    "outcome": "approval_unavailable",
                    "environment": self.config.PYTHON_EXE,
                    "error": "DEPENDENCY_APPROVAL_UNAVAILABLE",
                }
                self.dependency_events.append(event)
                operations.append(event)
            elif restricted_pending:
                try:
                    approved = bool(self.dependency_approver(
                        restricted_pending, justification
                    ))
                except (EOFError, KeyboardInterrupt):
                    approved = False
                if approved:
                    install_result = self.dependency_installer(
                        self.config, restricted_pending
                    )
                    event = {
                        "type": "dependency_install",
                        "requirements": restricted_pending,
                        "justification": justification,
                        "authorization": "user_approved_off_allowlist",
                        "approval_required": True,
                        "approved": True,
                        "binary_only": True,
                        "success": bool(install_result.get("success")),
                        "human_intervention": True,
                        "outcome": (
                            "installed" if install_result.get("success")
                            else "install_failed"
                        ),
                        "installed_versions": install_result.get(
                            "installed_versions", {}
                        ),
                        "environment": self.config.PYTHON_EXE,
                        "environment_snapshot": install_result.get(
                            "environment_snapshot"
                        ),
                        "error": install_result.get("error"),
                        "output_tail": str(
                            install_result.get("output_tail", "")
                        )[-1000:],
                    }
                else:
                    event = {
                        "type": "dependency_install",
                        "requirements": restricted_pending,
                        "justification": justification,
                        "authorization": "user_declined_off_allowlist",
                        "approval_required": True,
                        "approved": False,
                        "binary_only": True,
                        "success": False,
                        "human_intervention": True,
                        "outcome": "user_declined",
                        "environment": self.config.PYTHON_EXE,
                        "error": "DEPENDENCY_INSTALL_DECLINED",
                    }
                self.dependency_events.append(event)
                operations.append(event)

            if any(operation.get("success") for operation in operations):
                self.bootstrap_state.environment_inventory = inspect_ml_environment(self.config)
            success = bool(operations) and all(
                bool(operation.get("success")) for operation in operations
            )
            first_error = next(
                (
                    str(operation["error"])
                    for operation in operations
                    if operation.get("error")
                ),
                None,
            )
            return json.dumps({
                "success": success,
                "requirements": [spec for spec, _ in validated],
                "operations": operations,
                "error": first_error,
                "instruction": (
                    "Dependencies are available; implement and evaluate the experiment."
                    if success
                    else "Use the successful installs if relevant, then choose an installed "
                    "alternative for every denied or failed dependency."
                ),
            }, ensure_ascii=False, indent=2)
        if name == "search_ml_literature":
            query = str(payload["query"])
            results = search_ml_literature(query, int(payload.get("k", 3)))
            counted = bool(query.strip() and results.get("results"))
            if counted:
                self.bootstrap_state.literature_queries.append(query)
            results["bootstrap_counted"] = counted
            if not counted:
                results["bootstrap_instruction"] = (
                    "Search again with a non-empty query that retrieves at least one passage."
                )
            return json.dumps(
                results,
                ensure_ascii=False,
                indent=2,
            )
        if name == "reproduce_baseline":
            missing = self.bootstrap_state.missing_baseline_prerequisites()
            if missing:
                record: dict[str, object] = {
                    "success": False,
                    "error": "BASELINE_CONTEXT_REQUIRED",
                    "missing_requirements": missing,
                    "metrics": None,
                    "wall_seconds": 0.0,
                }
                self.bootstrap_state.rejected_actions.append({
                    "action": "reproduce_baseline",
                    "error": record["error"],
                    "missing_requirements": missing,
                })
                return json.dumps(record, ensure_ascii=False, indent=2)

            execution = execute_model(self.candidate_dir, self.config)
            score_matches = bool(
                execution.success
                and execution.metrics is not None
                and abs(
                    float(execution.metrics["primary"]) - self.config.BASELINE_PRIMARY
                ) <= self.config.CONVERGENCE_EPSILON
            )
            error = execution.error
            if execution.success and not score_matches:
                error = (
                    "BASELINE_MISMATCH: validation primary "
                    f"{execution.metrics['primary']:.6f} differs from official "
                    f"{self.config.BASELINE_PRIMARY:.6f} by more than "
                    f"{self.config.CONVERGENCE_EPSILON:.6f}"
                )
            record = {
                "success": execution.success and score_matches,
                "metrics": execution.metrics,
                "expected_primary": self.config.BASELINE_PRIMARY,
                "tolerance": self.config.CONVERGENCE_EPSILON,
                "error": error,
                "wall_seconds": execution.wall_seconds,
            }
            self.bootstrap_state.baseline_execution = record
            if record["success"]:
                self.bootstrap_state.baseline_reproduced = True
                self.bootstrap_state.baseline_metrics = execution.metrics
            return json.dumps(record, ensure_ascii=False, indent=2)
        if name == "record_task_context":
            missing = self.bootstrap_state.missing_requirements(before_context=True)
            if missing:
                record: dict[str, object] = {
                    "action": "record_task_context",
                    "error": "BOOTSTRAP_SOURCES_INCOMPLETE",
                    "missing_requirements": missing,
                }
                self.bootstrap_state.rejected_actions.append(record)
                return json.dumps({"success": False, **record}, ensure_ascii=False, indent=2)

            scalar_fields = ("task_objective", "target_label", "baseline")
            list_fields = (
                "metrics",
                "evaluation_protocol",
                "hard_constraints",
                "known_dead_ends",
                "promising_directions",
                "candidate_contract",
                "source_paths",
            )
            invalid_fields = [
                field_name for field_name in scalar_fields
                if not isinstance(payload.get(field_name), str)
                or not str(payload[field_name]).strip()
            ]
            invalid_fields.extend(
                field_name for field_name in list_fields
                if not isinstance(payload.get(field_name), list)
                or not payload[field_name]
            )
            splits = payload.get("data_splits")
            if not isinstance(splits, dict) or any(
                not isinstance(splits.get(split), str) or not splits[split].strip()
                for split in ("train", "validation", "test")
            ):
                invalid_fields.append("data_splits")

            feature_context = payload.get("feature_engineering_context")
            feature_context_errors: list[str] = []
            if not isinstance(feature_context, dict):
                invalid_fields.append("feature_engineering_context")
            else:
                expected_fields = [
                    "user_id", "video_id", "author_id", "tab", "dur_bucket",
                ]
                if feature_context.get("baseline_fields") != expected_fields:
                    feature_context_errors.append(
                        "baseline_fields must match the five fields defined by official data.py"
                    )
                for name in (
                    "measured_dead_ends",
                    "promising_feature_families",
                    "leakage_controls",
                ):
                    value = feature_context.get(name)
                    if not isinstance(value, list) or not value or not all(
                        isinstance(item, str) and item.strip() for item in value
                    ):
                        feature_context_errors.append(f"{name} must be a non-empty string list")
                dead_end_text = " ".join(
                    str(item) for item in feature_context.get("measured_dead_ends", [])
                ).lower()
                if not (
                    "13" in dead_end_text
                    and "static" in dead_end_text
                    and any(term in dead_end_text for term in (
                        "no gain", "did not help", "didn't help", "lower", "0.594",
                    ))
                ):
                    feature_context_errors.append(
                        "measured_dead_ends must record that the organizer's 13 static fields did not improve the baseline"
                    )
                boundary = str(feature_context.get("implementation_boundary", "")).lower()
                if "candidate_data" not in boundary or "model.py" not in boundary:
                    feature_context_errors.append(
                        "implementation_boundary must place feature code in model.py over candidate_data"
                    )

            resolved_sources: set[str] = set()
            source_errors: list[str] = []
            for source in payload.get("source_paths", []):
                try:
                    resolved_sources.add(str(_resolve_read_path(
                        str(source), self.candidate_dir, self.config.BASELINE_ROOT
                    )))
                except (TypeError, ValueError, OSError) as exc:
                    source_errors.append(str(exc))
            required_sources = {
                path for path in (
                    self.bootstrap_state.primary_readme_path,
                    self.bootstrap_state.required_evaluation_path,
                    self.bootstrap_state.required_baseline_path,
                    self.bootstrap_state.required_data_path,
                    self.bootstrap_state.required_feature_ablation_path,
                    self.bootstrap_state.required_candidate_model_path,
                )
                if path is not None
            }
            uncited_sources = sorted(required_sources - resolved_sources)
            unread_cited_sources = sorted(
                resolved_sources - self.bootstrap_state.fully_read_paths
            )
            if (
                invalid_fields
                or feature_context_errors
                or source_errors
                or uncited_sources
                or unread_cited_sources
            ):
                record = {
                    "action": "record_task_context",
                    "error": "TASK_CONTEXT_INVALID",
                    "invalid_or_empty_fields": sorted(set(invalid_fields)),
                    "feature_context_errors": feature_context_errors,
                    "source_errors": source_errors,
                    "required_sources_not_cited": uncited_sources,
                    "cited_sources_not_fully_read": unread_cited_sources,
                }
                self.bootstrap_state.rejected_actions.append(record)
                return json.dumps({"success": False, **record}, ensure_ascii=False, indent=2)

            self.bootstrap_state.task_context = json.loads(json.dumps(payload))
            return json.dumps({
                "success": True,
                "status": "TASK_CONTEXT_RECORDED_AND_RETAINED",
                "task_context": self.bootstrap_state.task_context,
                "instruction": (
                    "Use this retained benchmark context together with literature evidence "
                    "for this and every later experiment."
                ),
            }, ensure_ascii=False, indent=2)
        if name == "run_model":
            if not self.bootstrap_state.complete:
                return self._bootstrap_rejection("run_model")
            candidate_fingerprint = semantic_model_fingerprint(
                self.candidate_dir / "model.py"
            )
            candidate_changed = candidate_fingerprint != self.inherited_model_fingerprint
            allowed_components = {
                "loss", "sampling", "features", "sequence", "auxiliary-task",
                "model", "training", "evaluation",
            }
            target_component = str(payload.get("target_component", "")).strip().lower()
            if target_component not in allowed_components:
                record = {
                    "hypothesis": str(payload.get("hypothesis", "")),
                    "reasoning": str(payload.get("reasoning", "")),
                    "literature_chunk_ids": list(payload.get("literature_chunk_ids", [])),
                    "proposal": {"target_component": target_component},
                    "success": False,
                    "metrics": None,
                    "error": "INVALID_TARGET_COMPONENT: declare one supported pipeline component",
                    "wall_seconds": 0.0,
                    "candidate_changed": candidate_changed,
                }
                self.executions.append(record)
                return json.dumps(record, ensure_ascii=False, indent=2)
            proposal = {
                "target_component": target_component,
                "expected_effect": str(payload.get("expected_effect", "")),
                "falsification_criterion": str(
                    payload.get("falsification_criterion", "")
                ),
                "rollback_plan": str(payload.get("rollback_plan", "retain incumbent")),
                "feature_sources": list(payload.get("feature_sources") or []),
                "feature_transformations": list(
                    payload.get("feature_transformations") or []
                ),
                "leakage_controls": list(payload.get("leakage_controls") or []),
            }
            if self.model_post_save_failed:
                error = (
                    "SKIPPED: model.py failed its PostFileSave check. Fix and save "
                    "model.py successfully before calling run_model again."
                )
                record = {
                    "hypothesis": str(payload.get("hypothesis", "")),
                    "reasoning": str(payload.get("reasoning", "")),
                    "literature_chunk_ids": list(payload.get("literature_chunk_ids", [])),
                    "proposal": proposal,
                    "success": False,
                    "metrics": None,
                    "error": error,
                    "wall_seconds": 0.0,
                    "candidate_changed": candidate_changed,
                    "skipped": True,
                }
                self.executions.append(record)
                return json.dumps(record, ensure_ascii=False, indent=2)
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
                    "proposal": proposal,
                    "success": False,
                    "metrics": None,
                    "error": error,
                    "wall_seconds": 0.0,
                    "candidate_changed": False,
                }
                self.executions.append(record)
                return json.dumps(record, ensure_ascii=False, indent=2)

            if candidate_fingerprint in self.bootstrap_state.seen_candidate_fingerprints:
                record = {
                    "hypothesis": str(payload.get("hypothesis", "")),
                    "reasoning": str(payload.get("reasoning", "")),
                    "literature_chunk_ids": list(payload.get("literature_chunk_ids", [])),
                    "proposal": proposal,
                    "success": False,
                    "metrics": None,
                    "error": (
                        "REJECTED: this semantic candidate was already scored successfully "
                        "in the current autonomous run. Test a distinct change."
                    ),
                    "wall_seconds": 0.0,
                    "candidate_changed": True,
                }
                self.executions.append(record)
                return json.dumps(record, ensure_ascii=False, indent=2)

            feature_components = {
                "feature",
                "features",
                "sequence",
                "history",
                "time",
                "temporal",
                "watch-time",
                "auxiliary-task",
            }
            if target_component in feature_components and any(
                not proposal[field_name]
                for field_name in (
                    "feature_sources",
                    "feature_transformations",
                    "leakage_controls",
                )
            ):
                record = {
                    "hypothesis": str(payload.get("hypothesis", "")),
                    "reasoning": str(payload.get("reasoning", "")),
                    "literature_chunk_ids": list(payload.get("literature_chunk_ids", [])),
                    "proposal": proposal,
                    "success": False,
                    "metrics": None,
                    "error": (
                        "FEATURE_EVIDENCE_REQUIRED: feature-oriented experiments must declare "
                        "feature_sources, feature_transformations, and leakage_controls before execution"
                    ),
                    "wall_seconds": 0.0,
                    "candidate_changed": True,
                }
                self.executions.append(record)
                return json.dumps(record, ensure_ascii=False, indent=2)

            execution = execute_model(self.candidate_dir, self.config)
            record = {
                "hypothesis": str(payload.get("hypothesis", "")),
                "reasoning": str(payload.get("reasoning", "")),
                "literature_chunk_ids": list(payload.get("literature_chunk_ids", [])),
                "proposal": proposal,
                "success": execution.success,
                "metrics": execution.metrics,
                "error": execution.error,
                "wall_seconds": execution.wall_seconds,
                "candidate_changed": True,
            }
            self.executions.append(record)
            if execution.success and candidate_fingerprint is not None:
                self.bootstrap_state.seen_candidate_fingerprints.add(
                    candidate_fingerprint
                )
            return json.dumps(record, ensure_ascii=False, indent=2)
        return f"ERROR: unknown tool {name!r}"
