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
import re
import subprocess
import uuid
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

try:  # POSIX-only; absent on Windows, where os.times() is used instead.
    import resource
except ImportError:  # pragma: no cover - platform-dependent
    resource = None

from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name

from mle_agent.harness.config import Config
from mle_agent.harness.eda import (
    build_eda_report,
    query_aggregates,
    summarize_eda_for_model,
)
from mle_agent.harness.evaluation import score_validation_predictions
from mle_agent.harness.run_environment import (
    AUTO_INSTALL_ALLOWLIST,
    snapshot_run_environment,
)
from mle_agent.harness.tools import exec_edit_file, exec_write_file, redact_secrets
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
        "name": "edit_file",
        "description": (
            "Make one exact old-text replacement in a candidate file. Prefer this over a full "
            "rewrite for focused changes; the old text must match exactly once."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "old_text": {"type": "string"},
                "new_text": {"type": "string"},
            },
            "required": ["path", "old_text", "new_text"],
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
        "name": "query_data",
        "description": (
            "Run one bounded aggregate query over train or validation. Returns at most 20 "
            "grouped rows and never returns raw interaction rows. Use only to resolve a "
            "specific uncertainty left by inspect_data."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "split": {"type": "string", "enum": ["train", "validation"]},
                "group_by": {
                    "type": "array", "maxItems": 2,
                    "items": {"type": "string"},
                },
                "metrics": {
                    "type": "array", "minItems": 1, "maxItems": 5,
                    "items": {"type": "string"},
                },
                "filters": {"type": "array", "maxItems": 3, "items": {"type": "object"}},
                "limit": {"type": "integer", "minimum": 1, "maximum": 20},
            },
            "required": ["split", "metrics"],
        },
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
                "seed": {
                    "type": "integer",
                    "description": "Reproducible training seed; defaults to the official seed.",
                },
                "trial_config": {
                    "type": "object",
                    "description": "Candidate-declared data-only hyperparameter configuration.",
                },
                "execution_class": {
                    "type": "string",
                    "enum": ["quick", "normal", "substantial"],
                    "description": (
                        "Propose an execution budget: 'quick' for a diagnostic, 'normal' for "
                        "an ordinary candidate, 'substantial' for a framework-backed run. The "
                        "harness decides the actual limit and may clamp it to the wall budget."
                    ),
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
    # Ranked evidence-backed research plan produced right after task context.
    research_backlog: list[dict[str, object]] | None = None
    backlog_reject_count: int = 0
    # Every successfully scored experiment's target component, across iterations.
    # Persists because the same BootstrapState is reused for every run_iteration.
    scored_experiments: list[dict[str, object]] = field(default_factory=list)

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
        if not before_context and self.research_backlog is None:
            missing.append("record the ranked research backlog")
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
            "research_backlog": self.research_backlog,
            "backlog_reject_count": self.backlog_reject_count,
            "scored_experiments": self.scored_experiments,
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


# The starter-kit README (baseline_kuairand-starter-kit/README.md:122-133) records
# two directions the organizers measured and that add nothing. Prior autonomous
# runs proposed both anyway. Reject a run_model / run_sweep proposal that repeats one.
_CAPACITY_NOUNS = (
    "embedding dim", "embedding dimension", "latent dim", "latent dimension",
    "n_factors", "num_factors", "factorization dimension", "hidden size",
)
_INCREASE_VERBS = (
    "increase", "raise", "rais", "bigger", "larger", "higher", "grow",
    "more capacity", "scale up", "widen", "expand",
)
_ITEM_SIDE_TOKENS = (
    "video_id", "video_", "item_id", "item_", "author_id", "author_", "music_id",
    "tag", "category", "play_progress", "show_cnt", "upload_", "video_type",
    "video_duration", "dur_bucket", "item_pop", "item pop",
)


def _repeats_measured_dead_end(payload: dict, proposal: dict) -> str | None:
    """Return a README-quoting rejection reason if the proposal is a known dead end."""
    text = f"{payload.get('hypothesis', '')} {payload.get('reasoning', '')}".lower()
    capacity_noun = (
        any(noun in text for noun in _CAPACITY_NOUNS)
        or re.search(r"\bk\s*=\s*\d", text) is not None
        or "capacity" in text
    )
    capacity_verb = (
        any(verb in text for verb in _INCREASE_VERBS)
        or "->" in text
        or "→" in text
    )
    if capacity_noun and capacity_verb:
        return (
            "REJECTED: the starter-kit README records embedding dimension k=8/16/32 as "
            "flat (0.5895/0.5902/0.5887) -- capacity is not the bottleneck and 1.14M rows "
            "do not support more. Test a different pipeline component."
        )
    if str(proposal.get("target_component", "")).lower() in {"features", "feature"}:
        sources = " ".join(str(s) for s in proposal.get("feature_sources", [])).lower()
        transforms = " ".join(
            str(t) for t in proposal.get("feature_transformations", [])
        ).lower()
        touches_user = bool(sources) and (
            "user_id" in sources or re.search(r"\buser[_ ]", sources) is not None
        )
        has_item_side = any(
            token in sources or token in transforms for token in _ITEM_SIDE_TOKENS
        )
        if touches_user and not has_item_side:
            return (
                "REJECTED: the starter-kit README records that first-order terms from "
                "purely user-side features contribute exactly zero to within-user ranking. "
                "A user-side feature helps only through an interaction with an item-side "
                "field -- add an item-side cross to feature_transformations."
            )
    return None


RUN_MODEL_COMPONENTS = (
    "loss", "sampling", "features", "sequence", "auxiliary-task",
    "model", "training", "evaluation",
)


def _validate_research_backlog(candidates: object) -> list[str]:
    """Check the ranked research plan the agent must produce before experimenting."""
    if not isinstance(candidates, list) or not 6 <= len(candidates) <= 10:
        return ["research_backlog must be a list of 6 to 10 ranked candidate families"]
    errors: list[str] = []
    for index, entry in enumerate(candidates):
        if not isinstance(entry, dict):
            errors.append(f"candidate {index} must be an object")
            continue
        for field_name in (
            "hypothesis", "evidence_id", "estimated_cost", "falsification_criterion",
        ):
            value = entry.get(field_name)
            if not isinstance(value, str) or not value.strip():
                errors.append(f"candidate {index} needs a non-empty {field_name}")
        if not isinstance(entry.get("expected_primary_delta"), (int, float)):
            errors.append(
                f"candidate {index} needs a numeric expected_primary_delta"
            )
        if str(entry.get("target_component", "")).lower() not in RUN_MODEL_COMPONENTS:
            errors.append(
                f"candidate {index} target_component must be one of {RUN_MODEL_COMPONENTS}"
            )
    first_three = [
        str(entry.get("target_component", "")).lower()
        for entry in candidates[:3]
        if isinstance(entry, dict)
    ]
    if len(set(first_three)) < 3:
        errors.append(
            "the first three backlog entries must cover three distinct target_components "
            "(e.g. loss, features, model)"
        )
    if candidates and all(
        isinstance(entry, dict)
        and _repeats_measured_dead_end(
            {
                "hypothesis": entry.get("hypothesis", ""),
                "reasoning": entry.get("falsification_criterion", ""),
            },
            {"target_component": entry.get("target_component", "")},
        )
        is not None
        for entry in candidates
    ):
        errors.append("every backlog entry merely restates a measured dead end")
    return errors


@dataclass
class ModelExecution:
    success: bool
    metrics: dict | None
    output: str
    error: str | None
    wall_seconds: float
    prediction_path: str | None = None
    prediction_sha256: str | None = None
    seed: int | None = None
    trial_config: dict[str, object] | None = None
    resource_usage: dict[str, object] = field(default_factory=dict)


def _safe_float_metrics(metrics: dict) -> dict:
    return {
        key: float(value) if isinstance(value, (int, float)) or hasattr(value, "item") else value
        for key, value in metrics.items()
    }


def _sample_child_usage():
    """Snapshot cumulative child-process resource usage (POSIX or Windows)."""
    if resource is not None:
        return resource.getrusage(resource.RUSAGE_CHILDREN)
    return os.times()


def _child_usage_delta(before, after, timeout_seconds: int) -> dict:
    """Resource reading for one candidate execution.

    On POSIX this is ``RUSAGE_CHILDREN`` (CPU deltas + peak RSS). On Windows
    ``os.times().children_*`` is always ``0.0`` and there is no child RSS, so the
    CPU fields read ~0 and ``peak_rss_platform_units`` is ``None``. The field is
    diagnostic only -- ``ModelExecution.resource_usage`` is never schema-checked.
    """
    if resource is not None:
        return {
            "cpu_user_seconds": max(0.0, after.ru_utime - before.ru_utime),
            "cpu_system_seconds": max(0.0, after.ru_stime - before.ru_stime),
            "peak_rss_platform_units": after.ru_maxrss,
            "timeout_seconds": timeout_seconds,
            "accelerator": "not_measured",
        }
    return {
        "cpu_user_seconds": max(0.0, after.children_user - before.children_user),
        "cpu_system_seconds": max(0.0, after.children_system - before.children_system),
        "peak_rss_platform_units": None,
        "timeout_seconds": timeout_seconds,
        "accelerator": "not_measured",
    }


def execute_model(
    candidate_dir: Path,
    config: Config,
    timeout_seconds: int | None = None,
    *,
    seed: int | None = None,
    trial_config: dict[str, object] | None = None,
) -> ModelExecution:
    """Run ``model.py`` and score its aligned predictions in trusted code."""
    import time

    started = time.time()
    timeout_seconds = timeout_seconds or config.AGENT_NORMAL_EXECUTION_TIMEOUT_S
    seed = config.SEED if seed is None else int(seed)
    model_path = candidate_dir / "model.py"
    if not model_path.exists():
        return ModelExecution(
            False, None, "", "model.py does not exist", time.time() - started,
            seed=seed, trial_config=trial_config,
        )

    source = model_path.read_text(encoding="utf-8", errors="replace")
    violations = scan_candidate_source(source)
    if violations:
        message = f"REJECTED: validation-only policy violations: {violations}"
        return ModelExecution(
            False, None, message, message, time.time() - started,
            seed=seed, trial_config=trial_config,
        )

    env = {
        key: value
        for key, value in os.environ.items()
        if key not in {
            "LLM_API_KEY", "ANTHROPIC_API_KEY", "GROQ_API_KEY",
            "GEMINI_API_KEY", "GOOGLE_API_KEY", "OPENAI_API_KEY",
        }
    }
    # Only the organiser starter kit is importable. The project root is deliberately
    # absent so candidate code cannot import trusted harness modules across the
    # boundary; the prediction and submission writers are inlined into the candidate.
    env["PYTHONPATH"] = str(config.BASELINE_ROOT)
    harness_dir = candidate_dir / ".harness"
    harness_dir.mkdir(exist_ok=True)
    prediction_path = harness_dir / f"validation_predictions_seed_{seed}.csv"
    prediction_path.unlink(missing_ok=True)
    command = [
        config.PYTHON_EXE,
        str(model_path),
        "--data_dir",
        str(config.DATA_DIR),
        "--seed",
        str(seed),
        "--prediction-path",
        str(prediction_path),
    ]
    if trial_config is not None:
        trial_config_path = harness_dir / f"trial_config_seed_{seed}.json"
        trial_config_path.write_text(
            json.dumps(trial_config, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        command.extend(["--trial-config", str(trial_config_path)])
    before_usage = _sample_child_usage()
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
            False, None, output, f"TIMEOUT after {timeout_seconds}s", time.time() - started,
            seed=seed, trial_config=trial_config,
            resource_usage={"timeout_seconds": timeout_seconds},
        )
    except Exception as exc:
        return ModelExecution(
            False, None, "", f"ERROR: {exc}", time.time() - started,
            seed=seed, trial_config=trial_config,
        )

    output = redact_secrets(result.stdout + result.stderr)
    if len(output) > 12_000:
        output = output[-12_000:]
    after_usage = _sample_child_usage()
    usage = _child_usage_delta(before_usage, after_usage, timeout_seconds)
    if result.returncode != 0:
        error = f"model.py exited {result.returncode}\n{output[-6000:]}"
        return ModelExecution(
            False, None, output, error, time.time() - started,
            seed=seed, trial_config=trial_config, resource_usage=usage,
        )
    if not prediction_path.is_file():
        error = (
            "model.py exited successfully but did not write the required aligned "
            "validation prediction file"
        )
        return ModelExecution(
            False, None, output, error, time.time() - started,
            seed=seed, trial_config=trial_config, resource_usage=usage,
        )
    try:
        scored = score_validation_predictions(
            prediction_path,
            config.DATA_DIR,
            config.BASELINE_ROOT / "evaluate.py",
        )
    except (OSError, TypeError, ValueError, RuntimeError) as exc:
        error = f"PREDICTION_VALIDATION_ERROR: {exc}"
        return ModelExecution(
            False, None, output, error, time.time() - started,
            prediction_path=str(prediction_path), seed=seed,
            trial_config=trial_config, resource_usage=usage,
        )
    prediction_sha256 = hashlib.sha256(prediction_path.read_bytes()).hexdigest()
    return ModelExecution(
        True,
        _safe_float_metrics(scored.metrics),
        output,
        None,
        time.time() - started,
        prediction_path=str(prediction_path),
        prediction_sha256=prediction_sha256,
        seed=seed,
        trial_config=trial_config,
        resource_usage=usage,
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
    """Return the cached rich EDA report for train and validation only."""
    cache_key = str(config.DATA_DIR.resolve())
    if cache_key in _EDA_CACHE:
        return _EDA_CACHE[cache_key]
    summary = build_eda_report(config.DATA_DIR)
    if config.RUN_RESEARCH_DIR is not None:
        config.RUN_RESEARCH_DIR.mkdir(parents=True, exist_ok=True)
        destination = config.RUN_RESEARCH_DIR / "eda.json"
        destination.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        summary["artifact_path"] = str(destination.resolve())
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
        run_deadline: float | None = None,
        iteration: int = 0,
        experiments_remaining: int | None = None,
    ) -> None:
        self.candidate_dir = candidate_dir
        self.config = config
        # Epoch seconds at which the run's wall budget expires, or None when the
        # caller does not enforce one (offline tests, single-shot tools).
        self.run_deadline = run_deadline
        # Experiment number this runtime serves; 0 during bootstrap. Used by the
        # first-three diversity gate, which persists onto bootstrap_state.
        self.iteration = int(iteration)
        # How many of the 50 scored variants the run has left. Bounds a sweep so it
        # cannot overspend the official budget. None means no caller-enforced cap.
        self.experiments_remaining = experiments_remaining
        self.bootstrap_state = bootstrap_state or BootstrapState()
        self.dependency_approver = dependency_approver
        self.dependency_installer = dependency_installer or install_python_dependencies
        self.dependency_events: list[dict[str, object]] = []
        self.data_query_events: list[dict[str, object]] = []
        self._data_query_cache: dict[str, dict[str, object]] = {}
        self.executions: list[dict] = []
        self.model_post_save_failed = False
        self.inherited_model_fingerprint = semantic_model_fingerprint(
            candidate_dir / "model.py"
        )
        if self.bootstrap_state.required_candidate_model_path is None:
            self.bootstrap_state.required_candidate_model_path = str(
                (candidate_dir / "model.py").resolve()
            )

    def grant_execution_timeout(self, requested_class: str) -> dict[str, object]:
        """Decide the allowed execution timeout; the agent only proposes a class.

        The agent names an execution class, but the harness owns the number. A
        request is capped by the class ceiling and then clamped again to whatever
        wall-clock the run has left, so a long execution can never push the run
        past its budget. When too little time remains to finish even a quick
        diagnostic, the execution is refused rather than started and killed.
        """
        ceilings = {
            "quick": self.config.AGENT_QUICK_EXECUTION_TIMEOUT_S,
            "normal": self.config.AGENT_NORMAL_EXECUTION_TIMEOUT_S,
            "substantial": self.config.AGENT_SUBSTANTIAL_EXECUTION_TIMEOUT_S,
        }
        granted_class = requested_class if requested_class in ceilings else "normal"
        timeout = ceilings[granted_class]
        reason = f"requested class '{requested_class}' granted at {timeout}s"
        if requested_class not in ceilings:
            reason = f"unknown class '{requested_class}'; downgraded to normal ({timeout}s)"

        if self.run_deadline is not None:
            import time

            remaining = self.run_deadline - time.time() - self.config.AGENT_WALL_RESERVE_S
            if remaining < ceilings["quick"]:
                return {
                    "granted": False,
                    "requested_class": requested_class,
                    "granted_class": None,
                    "timeout_seconds": 0,
                    "remaining_wall_seconds": round(max(0.0, remaining), 1),
                    "reason": (
                        "WALL_BUDGET_EXHAUSTED: not enough wall-clock remains to run and "
                        "score another candidate within the run budget"
                    ),
                }
            if remaining < timeout:
                timeout = int(remaining)
                granted_class = "clamped"
                reason = (
                    f"requested class '{requested_class}' clamped to {timeout}s by the "
                    "remaining wall budget"
                )
            return {
                "granted": True,
                "requested_class": requested_class,
                "granted_class": granted_class,
                "timeout_seconds": timeout,
                "remaining_wall_seconds": round(remaining, 1),
                "reason": reason,
            }
        return {
            "granted": True,
            "requested_class": requested_class,
            "granted_class": granted_class,
            "timeout_seconds": timeout,
            "remaining_wall_seconds": None,
            "reason": reason,
        }

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
        if name == "edit_file":
            if not self.bootstrap_state.complete:
                return self._bootstrap_rejection("edit_file")
            result = exec_edit_file(
                str(payload["path"]),
                str(payload["old_text"]),
                str(payload["new_text"]),
                self.candidate_dir,
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
            # Return the digest, not the full report: this session is persistent, so
            # whatever goes here is resent on every later model call.
            return json.dumps(
                summarize_eda_for_model(summary), ensure_ascii=False, indent=2
            )
        if name == "query_data":
            if not self.bootstrap_state.data_inspected:
                return json.dumps({
                    "success": False,
                    "error": "EDA_REQUIRED: call inspect_data before query_data",
                }, indent=2)
            key = hashlib.sha256(
                json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
            ).hexdigest()
            cached = key in self._data_query_cache
            try:
                if cached:
                    result = self._data_query_cache[key]
                else:
                    result = query_aggregates(self.config.DATA_DIR, payload)
                    if len(self._data_query_cache) < self.config.EDA_QUERY_CACHE_LIMIT:
                        self._data_query_cache[key] = result
                event = {
                    "type": "data_query",
                    "query_sha256": key,
                    "result_sha256": result["result_sha256"],
                    "cached": cached,
                    "success": True,
                    "human_intervention": False,
                }
                self.data_query_events.append(event)
                return json.dumps({**result, "cached": cached}, ensure_ascii=False, indent=2)
            except (TypeError, ValueError, OSError) as exc:
                event = {
                    "type": "data_query",
                    "query_sha256": key,
                    "cached": False,
                    "success": False,
                    "error": f"INVALID_DATA_QUERY: {exc}",
                    "human_intervention": False,
                }
                self.data_query_events.append(event)
                return json.dumps(event, ensure_ascii=False, indent=2)
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
                # Two further dead ends the starter-kit README (lines 122-133)
                # rules out and prior runs repeated anyway.
                if not (
                    ("capacity" in dead_end_text or "embedding dim" in dead_end_text)
                    and any(term in dead_end_text for term in (
                        "k=8", "k = 8", "8/16/32", "8 / 16 / 32", "0.589", "0.590",
                    ))
                ):
                    feature_context_errors.append(
                        "measured_dead_ends must record that raising embedding dimension "
                        "k=8/16/32 stayed flat (~0.589) and capacity is not the bottleneck "
                        "(starter-kit README)"
                    )
                if not (
                    ("user-side" in dead_end_text or "user side" in dead_end_text)
                    and any(term in dead_end_text for term in (
                        "first-order", "first order", "within-user", "within user",
                        "contribute zero", "contributes zero", "constant within",
                    ))
                ):
                    feature_context_errors.append(
                        "measured_dead_ends must record that purely user-side first-order "
                        "features contribute zero to within-user ranking and help only "
                        "through item-side interactions (starter-kit README)"
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
        if name == "record_research_backlog":
            if self.bootstrap_state.task_context is None:
                return json.dumps({
                    "success": False,
                    "error": "BACKLOG_REQUIRES_TASK_CONTEXT",
                    "instruction": "Call record_task_context first.",
                }, ensure_ascii=False, indent=2)
            candidates = payload.get("candidates")
            errors = _validate_research_backlog(candidates)
            degraded = False
            if errors and self.bootstrap_state.backlog_reject_count >= 3:
                # Do not let backlog validation deadlock the bootstrap phase.
                degraded, errors = True, []
            if errors:
                self.bootstrap_state.backlog_reject_count += 1
                record = {
                    "action": "record_research_backlog",
                    "error": "RESEARCH_BACKLOG_INVALID",
                    "validation_errors": errors,
                    "attempt": self.bootstrap_state.backlog_reject_count,
                }
                self.bootstrap_state.rejected_actions.append(record)
                return json.dumps({"success": False, **record}, ensure_ascii=False, indent=2)
            normalized = json.loads(json.dumps(candidates))
            self.bootstrap_state.research_backlog = normalized
            if self.config.RUN_RESEARCH_DIR is not None:
                self.config.RUN_RESEARCH_DIR.mkdir(parents=True, exist_ok=True)
                (self.config.RUN_RESEARCH_DIR / "research_plan.json").write_text(
                    json.dumps(
                        {"candidates": normalized, "backlog_degraded": degraded},
                        ensure_ascii=False, indent=2,
                    ) + "\n",
                    encoding="utf-8",
                )
            return json.dumps({
                "success": True,
                "status": "RESEARCH_BACKLOG_RECORDED",
                "entries": len(normalized),
                "backlog_degraded": degraded,
                "instruction": (
                    "Experiments 1-3 must cover three distinct target_components. The "
                    "diversity gate refuses a repeat before then unless you pass "
                    "diversity_override."
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

            dead_end_reason = _repeats_measured_dead_end(payload, proposal)
            if dead_end_reason is not None:
                record = {
                    "hypothesis": str(payload.get("hypothesis", "")),
                    "reasoning": str(payload.get("reasoning", "")),
                    "literature_chunk_ids": list(payload.get("literature_chunk_ids", [])),
                    "proposal": proposal,
                    "success": False,
                    "metrics": None,
                    "error": dead_end_reason,
                    "wall_seconds": 0.0,
                    "candidate_changed": True,
                }
                self.executions.append(record)
                return json.dumps(record, ensure_ascii=False, indent=2)

            # First-three diversity gate: until three distinct components have been
            # scored, refuse a repeat unless the agent justifies it in writing.
            scored_components = {
                str(item.get("target_component"))
                for item in self.bootstrap_state.scored_experiments
            }
            if (
                len(scored_components) < 3
                and target_component in scored_components
                and not str(payload.get("diversity_override", "")).strip()
            ):
                record = {
                    "hypothesis": str(payload.get("hypothesis", "")),
                    "reasoning": str(payload.get("reasoning", "")),
                    "literature_chunk_ids": list(payload.get("literature_chunk_ids", [])),
                    "proposal": proposal,
                    "success": False,
                    "metrics": None,
                    "error": (
                        "DIVERSITY_REQUIRED: score three distinct target_components before "
                        f"repeating '{target_component}'. The research backlog's first three "
                        "families must cover distinct components. Pass diversity_override with "
                        "a written evidence-based justification to proceed anyway."
                    ),
                    "wall_seconds": 0.0,
                    "candidate_changed": True,
                }
                self.executions.append(record)
                return json.dumps(record, ensure_ascii=False, indent=2)

            execution_class = str(payload.get("execution_class", "normal"))
            requested_seed = int(payload.get("seed", self.config.SEED))
            trial_config = payload.get("trial_config")
            if trial_config is not None and not isinstance(trial_config, dict):
                record = {
                    "hypothesis": str(payload.get("hypothesis", "")),
                    "reasoning": str(payload.get("reasoning", "")),
                    "proposal": proposal,
                    "success": False,
                    "metrics": None,
                    "error": "INVALID_TRIAL_CONFIG: trial_config must be an object",
                    "wall_seconds": 0.0,
                    "candidate_changed": True,
                }
                self.executions.append(record)
                return json.dumps(record, ensure_ascii=False, indent=2)
            grant = self.grant_execution_timeout(execution_class)
            if not grant["granted"]:
                record = {
                    "hypothesis": str(payload.get("hypothesis", "")),
                    "reasoning": str(payload.get("reasoning", "")),
                    "proposal": proposal,
                    "success": False,
                    "metrics": None,
                    "error": str(grant["reason"]),
                    "wall_seconds": 0.0,
                    "candidate_changed": True,
                    "execution_grant": grant,
                }
                self.executions.append(record)
                return json.dumps(record, ensure_ascii=False, indent=2)
            execution = execute_model(
                self.candidate_dir,
                self.config,
                timeout_seconds=int(grant["timeout_seconds"]),
                seed=requested_seed,
                trial_config=trial_config,
            )
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
                "seed": execution.seed,
                "trial_config": execution.trial_config,
                "prediction_path": execution.prediction_path,
                "prediction_sha256": execution.prediction_sha256,
                "resource_usage": execution.resource_usage,
                "execution_class": execution_class,
                "execution_grant": grant,
            }
            self.executions.append(record)
            if execution.success and candidate_fingerprint is not None:
                self.bootstrap_state.seen_candidate_fingerprints.add(
                    candidate_fingerprint
                )
                self.bootstrap_state.scored_experiments.append({
                    "iteration": self.iteration,
                    "target_component": target_component,
                })
            return json.dumps(record, ensure_ascii=False, indent=2)
        if name == "run_sweep":
            return self._dispatch_run_sweep(payload)
        return f"ERROR: unknown tool {name!r}"

    def _sweep_rejection(self, payload: dict, error: str) -> str:
        record = {
            "hypothesis": str(payload.get("hypothesis", "")),
            "reasoning": str(payload.get("reasoning", "")),
            "literature_chunk_ids": list(payload.get("literature_chunk_ids", [])),
            "proposal": {"target_component": str(payload.get("target_component", ""))},
            "success": False,
            "metrics": None,
            "error": error,
            "wall_seconds": 0.0,
            "candidate_changed": True,
        }
        self.executions.append(record)
        return json.dumps(record, ensure_ascii=False, indent=2)

    def _dispatch_run_sweep(self, payload: dict) -> str:
        """Score 2-6 trial configs of the same candidate; each is one logged variant."""
        if not self.bootstrap_state.complete:
            return self._bootstrap_rejection("run_sweep")
        target_component = str(payload.get("target_component", "")).strip().lower()
        if target_component not in set(RUN_MODEL_COMPONENTS):
            return self._sweep_rejection(
                payload, "INVALID_TARGET_COMPONENT: declare one supported pipeline component"
            )
        proposal = {
            "target_component": target_component,
            "expected_effect": str(payload.get("expected_effect", "")),
            "falsification_criterion": str(payload.get("falsification_criterion", "")),
            "rollback_plan": str(payload.get("rollback_plan", "retain incumbent")),
            "feature_sources": list(payload.get("feature_sources") or []),
            "feature_transformations": list(payload.get("feature_transformations") or []),
            "leakage_controls": list(payload.get("leakage_controls") or []),
        }
        if self.model_post_save_failed:
            return self._sweep_rejection(
                payload, "SKIPPED: model.py failed its PostFileSave check; fix and save first."
            )
        fingerprint = semantic_model_fingerprint(self.candidate_dir / "model.py")
        if fingerprint == self.inherited_model_fingerprint:
            return self._sweep_rejection(
                payload, "REJECTED: model.py is semantically unchanged from the inherited candidate."
            )
        dead_end = _repeats_measured_dead_end(payload, proposal)
        if dead_end is not None:
            return self._sweep_rejection(payload, dead_end)
        scored_components = {
            str(item.get("target_component"))
            for item in self.bootstrap_state.scored_experiments
        }
        if (
            len(scored_components) < 3
            and target_component in scored_components
            and not str(payload.get("diversity_override", "")).strip()
        ):
            return self._sweep_rejection(
                payload,
                f"DIVERSITY_REQUIRED: score three distinct target_components before repeating "
                f"'{target_component}'. Pass diversity_override with a written justification.",
            )
        configs = payload.get("trial_configs")
        allowed_keys = {"k", "lr", "l2", "epochs", "batch_size", "patience"}
        if (
            not isinstance(configs, list)
            or not 2 <= len(configs) <= self.config.AGENT_SWEEP_MAX_CONFIGS
        ):
            return self._sweep_rejection(
                payload,
                f"INVALID_SWEEP: trial_configs must be a list of 2 to "
                f"{self.config.AGENT_SWEEP_MAX_CONFIGS} objects",
            )
        for index, cfg in enumerate(configs):
            if not isinstance(cfg, dict) or set(cfg) - allowed_keys:
                return self._sweep_rejection(
                    payload,
                    f"INVALID_SWEEP: config {index} must be an object using only "
                    f"{sorted(allowed_keys)}",
                )
        cap = len(configs)
        if self.experiments_remaining is not None:
            cap = min(cap, max(0, self.experiments_remaining))
        if cap < 2:
            return self._sweep_rejection(
                payload, "WALL_BUDGET_EXHAUSTED: not enough scored-variant budget left for a sweep"
            )
        sweep_id = uuid.uuid4().hex
        requested_seed = int(payload.get("seed", self.config.SEED))
        execution_class = str(payload.get("execution_class", "normal"))
        members: list[dict] = []
        for index in range(cap):
            grant = self.grant_execution_timeout(execution_class)
            if not grant["granted"]:
                break
            execution = execute_model(
                self.candidate_dir,
                self.config,
                timeout_seconds=int(grant["timeout_seconds"]),
                seed=requested_seed,
                trial_config=configs[index],
            )
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
                "seed": execution.seed,
                "trial_config": execution.trial_config,
                "prediction_path": execution.prediction_path,
                "prediction_sha256": execution.prediction_sha256,
                "resource_usage": execution.resource_usage,
                "execution_class": execution_class,
                "execution_grant": grant,
                "sweep_id": sweep_id,
                "sweep_member": index,
                "sweep_size": cap,
            }
            self.executions.append(record)
            members.append(record)
        successful = [m for m in members if m["success"]]
        if successful and fingerprint is not None:
            self.bootstrap_state.seen_candidate_fingerprints.add(fingerprint)
            self.bootstrap_state.scored_experiments.append({
                "iteration": self.iteration,
                "target_component": target_component,
            })
        best = max(
            successful, key=lambda m: float(m["metrics"]["primary"]), default=None
        )
        return json.dumps({
            "success": bool(successful),
            "sweep_id": sweep_id,
            "configs_run": len(members),
            "configs_requested": len(configs),
            "best_member": best["sweep_member"] if best else None,
            "best_primary": float(best["metrics"]["primary"]) if best else None,
            "members": [
                {
                    "member": m["sweep_member"],
                    "success": m["success"],
                    "trial_config": m["trial_config"],
                    "primary": (
                        float(m["metrics"]["primary"]) if m["success"] else None
                    ),
                    "error": m["error"],
                }
                for m in members
            ],
        }, ensure_ascii=False, indent=2)
