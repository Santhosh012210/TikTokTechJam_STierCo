"""Versioned, audited evidence from earlier autonomous development runs."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


PRIOR_EXPERIMENTS_PATH = Path(__file__).with_name("prior_experiments.json")


def load_prior_experiment_evidence() -> dict[str, object]:
    """Load the compact prior-run ledger and fail closed if its shape drifts."""
    raw = PRIOR_EXPERIMENTS_PATH.read_bytes()
    payload = json.loads(raw)
    required = {"schema_version", "evidence_id", "status", "trials", "interpretation"}
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"prior experiment evidence is missing fields: {missing}")
    trials = payload["trials"]
    if not isinstance(trials, list) or not trials:
        raise ValueError("prior experiment evidence must contain at least one trial")
    for index, trial in enumerate(trials):
        if not isinstance(trial, dict) or not {
            "trial", "GAUC", "nDCG@5", "primary", "model_sha256", "prediction_sha256"
        }.issubset(trial):
            raise ValueError(f"prior experiment trial {index} is incomplete")
    return {
        "path": str(PRIOR_EXPERIMENTS_PATH),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "payload": payload,
    }


def compact_prior_experiment_evidence() -> str:
    """Render evidence for a persistent prompt without whitespace-heavy replay."""
    return json.dumps(
        load_prior_experiment_evidence(), ensure_ascii=False, separators=(",", ":")
    )


def snapshot_prior_experiment_evidence(run_research_dir: Path | None) -> dict[str, object]:
    """Copy the exact evidence ledger into a run's durable research directory."""
    evidence = load_prior_experiment_evidence()
    if run_research_dir is not None:
        run_research_dir.mkdir(parents=True, exist_ok=True)
        target = run_research_dir / "prior_experiments.json"
        target.write_bytes(PRIOR_EXPERIMENTS_PATH.read_bytes())
        evidence = {**evidence, "run_snapshot_path": str(target)}
    return evidence


__all__ = [
    "PRIOR_EXPERIMENTS_PATH",
    "compact_prior_experiment_evidence",
    "load_prior_experiment_evidence",
    "snapshot_prior_experiment_evidence",
]
