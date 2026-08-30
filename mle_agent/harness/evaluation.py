"""Trusted validation-prediction format and organiser scoring."""
from __future__ import annotations

import csv
import importlib.util
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from mle_agent.harness.data_view import VALID_DATES, VALID_TEST_LOG


PREDICTION_HEADER = ["row_id", "user_id", "video_id", "score"]

# Copying the validation labels into the score column yields GAUC exactly 1.0 --
# the published ``oracle_ceiling`` in baseline_scores.json. Real models on this
# benchmark sit near 0.667 (official FM) and 0.639 (item popularity), so nothing
# legitimate approaches this bound. The candidate process can read the validation
# labels (the organiser baseline early-stops on them), so this is detection of the
# catastrophic case rather than prevention. Subtler leakage scores far lower and is
# governed by the feature-manifest leakage declarations, not by this check.
MAX_PLAUSIBLE_VALIDATION_GAUC = 0.95


@dataclass(frozen=True)
class ScoredPredictions:
    metrics: dict[str, float]
    rows: int


def write_validation_predictions(
    path: str | Path,
    rows: Sequence[tuple],
    scores: Iterable[float],
) -> None:
    """Write scores aligned to the candidate's validation rows."""
    destination = Path(path)
    values = list(scores)
    if len(values) != len(rows):
        raise ValueError(f"prediction count {len(values)} != validation rows {len(rows)}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(PREDICTION_HEADER)
        for row_id, (row, score) in enumerate(zip(rows, values)):
            value = float(score)
            if not math.isfinite(value):
                raise ValueError(f"non-finite score at row {row_id}")
            writer.writerow([row_id, row[1], row[2], repr(value)])


def _load_expected_validation(data_dir: Path) -> tuple[list[str], list[int], list[str]]:
    users: list[str] = []
    labels: list[int] = []
    items: list[str] = []
    with (data_dir / VALID_TEST_LOG).open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            date = int(row["date"])
            if VALID_DATES[0] <= date <= VALID_DATES[1]:
                users.append(row["user_id"])
                items.append(row["video_id"])
                labels.append(int(row["long_view"] not in ("", "0", "0.0")))
    return users, labels, items


def _load_organizer_evaluator(path: Path):
    spec = importlib.util.spec_from_file_location("_trusted_kuairand_evaluate", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load organiser evaluator: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.evaluate


def score_validation_predictions(
    path: Path,
    data_dir: Path,
    evaluator_path: Path,
) -> ScoredPredictions:
    """Validate exact row alignment and score with the unchanged evaluator."""
    users, labels, items = _load_expected_validation(data_dir)
    scores: list[float] = []
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != PREDICTION_HEADER:
            raise ValueError(
                f"prediction header must be {PREDICTION_HEADER}, got {reader.fieldnames}"
            )
        for expected_row_id, row in enumerate(reader):
            if expected_row_id >= len(users):
                raise ValueError("prediction file contains too many rows")
            if int(row["row_id"]) != expected_row_id:
                raise ValueError(f"row_id mismatch at row {expected_row_id}")
            if row["user_id"] != users[expected_row_id]:
                raise ValueError(f"user_id mismatch at row {expected_row_id}")
            if row["video_id"] != items[expected_row_id]:
                raise ValueError(f"video_id mismatch at row {expected_row_id}")
            try:
                score = float(row["score"])
            except ValueError as exc:
                raise ValueError(f"non-numeric score at row {expected_row_id}") from exc
            if not math.isfinite(score):
                raise ValueError(f"non-finite score at row {expected_row_id}")
            scores.append(score)
    if len(scores) != len(users):
        raise ValueError(f"prediction row count {len(scores)} != expected {len(users)}")
    evaluate = _load_organizer_evaluator(evaluator_path)
    result = evaluate(users, labels, scores)
    gauc = float(result["GAUC"])
    if gauc > MAX_PLAUSIBLE_VALIDATION_GAUC:
        raise ValueError(
            f"IMPLAUSIBLE_VALIDATION_GAUC: {gauc:.6f} exceeds the plausibility ceiling "
            f"{MAX_PLAUSIBLE_VALIDATION_GAUC}. Scores this good indicate the validation "
            "labels reached the prediction column (directly, or through a feature fitted "
            "on them) rather than a model that generalizes. Rebuild the candidate so every "
            "feature and target is derived from the training split only."
        )
    return ScoredPredictions(
        metrics={key: float(value) for key, value in result.items()},
        rows=len(scores),
    )


__all__ = [
    "PREDICTION_HEADER",
    "ScoredPredictions",
    "score_validation_predictions",
    "write_validation_predictions",
]
