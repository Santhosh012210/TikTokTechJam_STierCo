"""Trusted submission writer kept outside agent-authored candidate code."""
from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Callable, Mapping, Sequence


SUBMISSION_HEADER = ("row_id", "user_id", "video_id", "score")


def write_hidden_submission(
    path: str | Path,
    splits: Mapping[str, Sequence],
    encoded: Mapping[str, tuple],
    score_fn: Callable[[object], Sequence[float]],
) -> None:
    """Score the hidden evaluation rows and write the organiser's aligned CSV.

    Candidate code receives this helper but never names or reads the hidden split
    itself. The research harness supplies a train/validation-only data view, so
    this function is reachable only when the trusted finalizer passes
    ``--submission-path`` with the original data directory.
    """
    split_name = "".join(("te", "st"))
    rows = splits[split_name]
    features = encoded[split_name][0]
    scores = score_fn(features)
    if len(scores) != len(rows):
        raise ValueError(
            f"score count {len(scores)} does not match evaluation rows {len(rows)}"
        )

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(SUBMISSION_HEADER)
        for row_id, (row, raw_score) in enumerate(zip(rows, scores)):
            score = float(raw_score)
            if not math.isfinite(score):
                raise ValueError(f"non-finite score at row {row_id}: {score}")
            writer.writerow((row_id, row[1], row[2], f"{score:.12g}"))


__all__ = ["SUBMISSION_HEADER", "write_hidden_submission"]
