"""Deterministic train/validation-only EDA and bounded aggregate queries."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from mle_agent.harness.data_view import TRAIN_DATES, TRAIN_LOG, VALID_DATES, VALID_TEST_LOG


TARGET = "long_view"
AUXILIARY_FIELDS = (
    "is_click", "is_like", "is_follow", "is_comment", "is_forward",
    "is_hate", "is_profile_enter",
)
NUMERIC_FIELDS = (
    "duration_ms", "play_time_ms", "profile_stay_time", "comment_stay_time",
)
QUERY_FIELDS = {
    "user_id", "video_id", "date", "hourmin", "tab", "is_rand", TARGET,
    *AUXILIARY_FIELDS, *NUMERIC_FIELDS,
}
QUERY_METRICS = {
    "rows", "target_rate", "unique_users", "unique_items", "mean_duration_ms",
    "mean_play_time_ms",
}
QUERY_OPERATORS = {"eq", "ne", "in", "gte", "lte"}
MAX_QUERY_GROUPS = 2
MAX_QUERY_METRICS = 5
MAX_QUERY_FILTERS = 3
MAX_QUERY_ROWS = 20
MAX_CATEGORY_VALUES = 20
NUMERIC_SAMPLE_LIMIT = 100_000
# A group narrow enough to isolate individual impressions would turn ``target_rate``
# into a row-by-row readout of the validation labels. Aggregates must average over at
# least this many rows to be returned.
MIN_GROUP_ROWS = 20
# ``limit`` bounds the returned rows, not the work: grouping by user_id x video_id
# would otherwise build ~1.1M live buckets. Abort instead of exhausting memory.
MAX_QUERY_GROUPS_SCANNED = 50_000


def _positive(value: str | None) -> int:
    return int(value not in (None, "", "0", "0.0"))


def _safe_float(value: str | None) -> float | None:
    try:
        number = float(value or "")
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _quantiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {name: None for name in ("min", "p50", "p90", "p99", "max")}
    ordered = sorted(values)

    def at(fraction: float) -> float:
        return float(ordered[round(fraction * (len(ordered) - 1))])

    return {
        "min": float(ordered[0]),
        "p50": at(0.50),
        "p90": at(0.90),
        "p99": at(0.99),
        "max": float(ordered[-1]),
    }


def _counter_summary(counter: Counter[str], rows: int) -> dict[str, object]:
    counts = list(counter.values())
    return {
        "cardinality": len(counter),
        "top_values": [
            {
                "value": value,
                "count": count,
                "share": round(count / rows, 6) if rows else 0.0,
            }
            for value, count in counter.most_common(MAX_CATEGORY_VALUES)
        ],
        "count_quantiles": _quantiles([float(value) for value in counts]),
    }


@dataclass
class _SplitAccumulator:
    name: str
    rows: int = 0
    target_positives: int = 0
    users: Counter[str] = field(default_factory=Counter)
    items: Counter[str] = field(default_factory=Counter)
    user_positives: Counter[str] = field(default_factory=Counter)
    item_positives: Counter[str] = field(default_factory=Counter)
    categories: dict[str, Counter[str]] = field(
        default_factory=lambda: {name: Counter() for name in ("tab", "date", "is_rand")}
    )
    missing: Counter[str] = field(default_factory=Counter)
    auxiliary_positives: Counter[str] = field(default_factory=Counter)
    numeric: dict[str, list[float]] = field(
        default_factory=lambda: {name: [] for name in NUMERIC_FIELDS}
    )
    pairs: Counter[tuple[str, str]] = field(default_factory=Counter)

    def add(self, row: dict[str, str]) -> None:
        self.rows += 1
        user = row.get("user_id", "")
        item = row.get("video_id", "")
        target = _positive(row.get(TARGET))
        self.target_positives += target
        self.users[user] += 1
        self.items[item] += 1
        self.user_positives[user] += target
        self.item_positives[item] += target
        self.pairs[(user, item)] += 1
        for name, counter in self.categories.items():
            counter[row.get(name, "")] += 1
        for name in AUXILIARY_FIELDS:
            self.auxiliary_positives[name] += _positive(row.get(name))
        for name in QUERY_FIELDS:
            if row.get(name) in (None, ""):
                self.missing[name] += 1
        # Deterministic bounded sampling: the first N rows are sufficient for stable
        # descriptive quantiles and avoid retaining the full 1.2M-row columns.
        if self.rows <= NUMERIC_SAMPLE_LIMIT:
            for name in NUMERIC_FIELDS:
                value = _safe_float(row.get(name))
                if value is not None:
                    self.numeric[name].append(value)

    def summary(self) -> dict[str, object]:
        duplicate_rows = sum(count - 1 for count in self.pairs.values() if count > 1)
        return {
            "rows": self.rows,
            "users": len(self.users),
            "items": len(self.items),
            "target": TARGET,
            "target_positive_rate": round(self.target_positives / self.rows, 6)
            if self.rows else 0.0,
            "auxiliary_positive_rates": {
                name: round(self.auxiliary_positives[name] / self.rows, 6)
                if self.rows else 0.0
                for name in AUXILIARY_FIELDS
            },
            "missing_rates": {
                name: round(count / self.rows, 6) if self.rows else 0.0
                for name, count in sorted(self.missing.items())
            },
            "numeric_quantiles": {
                name: _quantiles(values) for name, values in self.numeric.items()
            },
            "user_activity": _counter_summary(self.users, self.rows),
            "item_popularity": _counter_summary(self.items, self.rows),
            "categorical": {
                name: _counter_summary(counter, self.rows)
                for name, counter in self.categories.items()
            },
            "duplicate_user_item": {
                "unique_pairs": len(self.pairs),
                "duplicate_rows": duplicate_rows,
                "duplicate_row_rate": round(duplicate_rows / self.rows, 6)
                if self.rows else 0.0,
                "max_repetitions": max(self.pairs.values(), default=0),
            },
        }


def _iter_split(data_dir: Path, split: str) -> Iterable[dict[str, str]]:
    if split == "train":
        path, bounds = data_dir / TRAIN_LOG, TRAIN_DATES
    elif split in {"valid", "validation"}:
        path, bounds = data_dir / VALID_TEST_LOG, VALID_DATES
    else:
        raise ValueError("split must be train or validation")
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            date = int(row["date"])
            if bounds[0] <= date <= bounds[1]:
                yield row


def _total_variation(train: Counter[str], valid: Counter[str]) -> float:
    train_total = sum(train.values())
    valid_total = sum(valid.values())
    if not train_total or not valid_total:
        return 0.0
    values = set(train) | set(valid)
    return 0.5 * sum(
        abs(train[value] / train_total - valid[value] / valid_total)
        for value in values
    )


def _inventory(data_dir: Path) -> dict[str, object]:
    roles = {
        TRAIN_LOG: "training interactions",
        VALID_TEST_LOG: "validation interactions only in this filtered view",
        "user_features_pure.csv": "static user features keyed by user_id",
        "video_features_basic_pure.csv": "item metadata keyed by video_id",
        "video_features_statistic_pure.csv": "organizer item statistics keyed by video_id",
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
            "Read these CSVs through --data_dir and implement joins, train-fitted encoders, "
            "histories, and transformations inside the confined candidate model.py or its "
            "hashed local helper modules."
        ),
    }


def build_eda_report(data_dir: Path) -> dict[str, object]:
    """Build one bounded EDA report without reading hidden-test rows."""
    data_dir = data_dir.resolve()
    accumulators = {
        "train": _SplitAccumulator("train"),
        "valid": _SplitAccumulator("valid"),
    }
    for name, accumulator in accumulators.items():
        for row in _iter_split(data_dir, name):
            accumulator.add(row)

    train = accumulators["train"]
    valid = accumulators["valid"]
    unseen_users = sum(count for value, count in valid.users.items() if value not in train.users)
    unseen_items = sum(count for value, count in valid.items.items() if value not in train.items)
    drift = {
        name: round(_total_variation(train.categories[name], valid.categories[name]), 6)
        for name in train.categories
    }
    findings = [
        {
            "id": "eda:label-shift",
            "observation": (
                f"long_view rate is {train.target_positives / train.rows:.4f} in train and "
                f"{valid.target_positives / valid.rows:.4f} in validation"
            ),
            "implication": "Prefer robust ranking features and verify temporal generalization.",
            "evidence_type": "train_validation_drift",
        },
        {
            "id": "eda:item-cold-start",
            "observation": f"{unseen_items} validation impressions use items unseen in train",
            "implication": "Provide explicit unknown/default behavior for item-derived features.",
            "evidence_type": "cold_start",
        },
        {
            "id": "eda:repeated-pairs",
            "observation": (
                f"validation contains {valid.summary()['duplicate_user_item']['duplicate_rows']} "
                "repeated user-item rows"
            ),
            "implication": "Do not use user-item as a unique prediction key; preserve row order.",
            "evidence_type": "ranking_protocol",
        },
    ]
    return {
        "schema_version": 1,
        "policy": "train and validation dates only; hidden test excluded",
        "splits": {
            "train": train.summary(),
            "valid": valid.summary(),
        },
        # Backward-compatible aliases used by the existing prompt/tests.
        "train": train.summary(),
        "valid": valid.summary(),
        "drift": {"categorical_total_variation": drift},
        "cold_start": {
            "validation_unseen_user_rows": unseen_users,
            "validation_unseen_user_rate": round(unseen_users / valid.rows, 6)
            if valid.rows else 0.0,
            "validation_unseen_item_rows": unseen_items,
            "validation_unseen_item_rate": round(unseen_items / valid.rows, 6)
            if valid.rows else 0.0,
        },
        "data_findings": findings,
        "candidate_data": _inventory(data_dir),
    }


def summarize_eda_for_model(report: dict[str, object]) -> dict[str, object]:
    """Reduce the full EDA report to the part worth putting in the transcript.

    The complete report is kept on disk as evidence, but it is ~45KB and the agent
    session is persistent, so returning it verbatim resends it on every later model
    call. The bulk is high-cardinality ``top_values`` lists, which are far less
    decision-relevant than the rates, quantiles, drift, and cold-start shares kept
    here. ``query_data`` remains available for anything this leaves open.
    """
    def split_digest(split: dict[str, object]) -> dict[str, object]:
        categorical = split.get("categorical", {})
        return {
            "rows": split.get("rows"),
            "users": split.get("users"),
            "items": split.get("items"),
            "target_positive_rate": split.get("target_positive_rate"),
            "auxiliary_positive_rates": split.get("auxiliary_positive_rates"),
            "numeric_quantiles": split.get("numeric_quantiles"),
            "impressions_per_user": split.get("user_activity", {}).get("count_quantiles"),
            "impressions_per_item": split.get("item_popularity", {}).get("count_quantiles"),
            "user_cardinality": split.get("user_activity", {}).get("cardinality"),
            "item_cardinality": split.get("item_popularity", {}).get("cardinality"),
            # Low-cardinality fields are small enough to keep their value breakdown.
            "tab": categorical.get("tab", {}).get("top_values"),
            "duplicate_user_item": split.get("duplicate_user_item"),
        }

    splits = report.get("splits", {})
    return {
        "policy": report.get("policy"),
        "artifact_path": report.get("artifact_path"),
        "artifact_note": (
            "This is a digest. The complete EDA report is the artifact above; use "
            "query_data for any specific aggregate it does not answer."
        ),
        "train": split_digest(splits.get("train", {})),
        "valid": split_digest(splits.get("valid", {})),
        "drift": report.get("drift"),
        "cold_start": report.get("cold_start"),
        "data_findings": report.get("data_findings"),
        "candidate_data": report.get("candidate_data"),
    }


def write_eda_report(data_dir: Path, destination: Path) -> dict[str, object]:
    report = build_eda_report(data_dir)
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    destination.write_text(encoded, encoding="utf-8")
    return {
        "path": str(destination.resolve()),
        "sha256": hashlib.sha256(encoded.encode("utf-8")).hexdigest(),
        "report": report,
    }


def _matches_filter(row: dict[str, str], item: dict[str, object]) -> bool:
    field = str(item.get("field", ""))
    operator = str(item.get("op", ""))
    if field not in QUERY_FIELDS or operator not in QUERY_OPERATORS:
        raise ValueError(f"unsupported filter: {field}/{operator}")
    actual = row.get(field, "")
    expected = item.get("value")
    if operator == "in":
        if not isinstance(expected, list) or len(expected) > 20:
            raise ValueError("in filter requires a list of at most 20 values")
        return actual in {str(value) for value in expected}
    if operator in {"gte", "lte"}:
        actual_number = _safe_float(actual)
        expected_number = _safe_float(str(expected))
        if actual_number is None or expected_number is None:
            return False
        return actual_number >= expected_number if operator == "gte" else actual_number <= expected_number
    return actual == str(expected) if operator == "eq" else actual != str(expected)


def query_aggregates(data_dir: Path, specification: dict[str, object]) -> dict[str, object]:
    """Execute a small allowlisted aggregate query over train or validation."""
    split = str(specification.get("split", ""))
    if split == "validation":
        split = "valid"
    if split not in {"train", "valid"}:
        raise ValueError("split must be train or validation")
    group_by = specification.get("group_by", [])
    metrics = specification.get("metrics", [])
    filters = specification.get("filters", [])
    limit = int(specification.get("limit", MAX_QUERY_ROWS))
    if not isinstance(group_by, list) or len(group_by) > MAX_QUERY_GROUPS:
        raise ValueError(f"group_by accepts at most {MAX_QUERY_GROUPS} fields")
    if not all(isinstance(field, str) and field in QUERY_FIELDS for field in group_by):
        raise ValueError("group_by contains an unsupported field")
    if not isinstance(metrics, list) or not 1 <= len(metrics) <= MAX_QUERY_METRICS:
        raise ValueError(f"metrics must contain 1-{MAX_QUERY_METRICS} values")
    if not all(isinstance(metric, str) and metric in QUERY_METRICS for metric in metrics):
        raise ValueError("metrics contains an unsupported aggregate")
    if not isinstance(filters, list) or len(filters) > MAX_QUERY_FILTERS:
        raise ValueError(f"filters accepts at most {MAX_QUERY_FILTERS} entries")
    if not 1 <= limit <= MAX_QUERY_ROWS:
        raise ValueError(f"limit must be between 1 and {MAX_QUERY_ROWS}")

    # Accumulate only what the requested metrics need; the unique-value sets in
    # particular are expensive and pointless when nobody asked for them.
    requested = set(metrics)
    track_users = "unique_users" in requested
    track_items = "unique_items" in requested
    track_target = "target_rate" in requested
    track_duration = "mean_duration_ms" in requested
    track_play = "mean_play_time_ms" in requested

    def _new_bucket() -> dict[str, object]:
        bucket: dict[str, object] = {"rows": 0}
        if track_target:
            bucket["target_positives"] = 0
        if track_users:
            bucket["users"] = set()
        if track_items:
            bucket["items"] = set()
        if track_duration:
            bucket["duration_sum"] = 0.0
            bucket["duration_count"] = 0
        if track_play:
            bucket["play_sum"] = 0.0
            bucket["play_count"] = 0
        return bucket

    groups: dict[tuple[str, ...], dict[str, object]] = defaultdict(_new_bucket)
    scanned = matched = 0
    for row in _iter_split(data_dir, split):
        scanned += 1
        if any(not _matches_filter(row, item) for item in filters):
            continue
        matched += 1
        key = tuple(row.get(field, "") for field in group_by)
        if key not in groups and len(groups) >= MAX_QUERY_GROUPS_SCANNED:
            raise ValueError(
                f"query produces more than {MAX_QUERY_GROUPS_SCANNED} groups; "
                "group by lower-cardinality fields or add a filter"
            )
        bucket = groups[key]
        bucket["rows"] = int(bucket["rows"]) + 1
        if track_target:
            bucket["target_positives"] = (
                int(bucket["target_positives"]) + _positive(row.get(TARGET))
            )
        if track_users:
            bucket["users"].add(row.get("user_id", ""))
        if track_items:
            bucket["items"].add(row.get("video_id", ""))
        if track_duration:
            duration = _safe_float(row.get("duration_ms"))
            if duration is not None:
                bucket["duration_sum"] = float(bucket["duration_sum"]) + duration
                bucket["duration_count"] = int(bucket["duration_count"]) + 1
        if track_play:
            play = _safe_float(row.get("play_time_ms"))
            if play is not None:
                bucket["play_sum"] = float(bucket["play_sum"]) + play
                bucket["play_count"] = int(bucket["play_count"]) + 1

    rows: list[dict[str, object]] = []
    suppressed_groups = suppressed_rows = 0
    for key, bucket in groups.items():
        count = int(bucket["rows"])
        if count < MIN_GROUP_ROWS:
            suppressed_groups += 1
            suppressed_rows += count
            continue
        result = {field: value for field, value in zip(group_by, key)}
        values = {
            "rows": count,
            "target_rate": round(int(bucket["target_positives"]) / count, 6)
            if track_target else None,
            "unique_users": len(bucket["users"]) if track_users else None,
            "unique_items": len(bucket["items"]) if track_items else None,
            "mean_duration_ms": round(
                float(bucket["duration_sum"]) / int(bucket["duration_count"]), 3
            )
            if track_duration and bucket["duration_count"] else None,
            "mean_play_time_ms": round(
                float(bucket["play_sum"]) / int(bucket["play_count"]), 3
            )
            if track_play and bucket["play_count"] else None,
        }
        result.update({metric: values[metric] for metric in metrics})
        rows.append(result)
    rows.sort(key=lambda item: (-int(item.get("rows", 0)), json.dumps(item, sort_keys=True)))
    normalized = {
        "split": "validation" if split == "valid" else split,
        "group_by": group_by,
        "metrics": metrics,
        "filters": filters,
        "scanned_rows": scanned,
        "matched_rows": matched,
        "rows": rows[:limit],
        "truncated": len(rows) > limit,
        "suppressed_small_groups": suppressed_groups,
        "suppressed_small_group_rows": suppressed_rows,
        "policy": (
            f"bounded aggregates only; no raw rows returned; groups under "
            f"{MIN_GROUP_ROWS} rows are suppressed"
        ),
    }
    fingerprint = hashlib.sha256(
        json.dumps(normalized, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()
    normalized["result_sha256"] = fingerprint
    return normalized


__all__ = [
    "MAX_QUERY_GROUPS_SCANNED",
    "MAX_QUERY_ROWS",
    "MIN_GROUP_ROWS",
    "QUERY_FIELDS",
    "QUERY_METRICS",
    "build_eda_report",
    "summarize_eda_for_model",
    "query_aggregates",
    "write_eda_report",
]
