"""Create a count-verified train/validation-only view for candidate processes."""
from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path


TRAIN_LOG = "log_standard_4_08_to_4_21_pure.csv"
VALID_TEST_LOG = "log_standard_4_22_to_5_08_pure.csv"
FEATURE_FILES = (
    "user_features_pure.csv",
    "video_features_basic_pure.csv",
    "video_features_statistic_pure.csv",
)

TRAIN_DATES = (20220408, 20220421)
VALID_DATES = (20220422, 20220428)
TEST_DATES = (20220429, 20220508)

EXPECTED_TRAIN_ROWS = 1_141_112
EXPECTED_VALID_ROWS = 124_909
EXPECTED_TEST_ROWS = 170_588


def classify_date(date: int) -> str:
    if TRAIN_DATES[0] <= date <= TRAIN_DATES[1]:
        return "train"
    if VALID_DATES[0] <= date <= VALID_DATES[1]:
        return "valid"
    if TEST_DATES[0] <= date <= TEST_DATES[1]:
        return "test"
    return "outside"


def _filter_log(
    source: Path,
    destination: Path,
    allowed_source_splits: set[str],
    emitted_split: str,
    counts: dict[str, int],
) -> None:
    with source.open(encoding="utf-8", newline="") as src:
        reader = csv.DictReader(src)
        if not reader.fieldnames or "date" not in reader.fieldnames:
            raise ValueError(f"{source.name} has no date column")
        with destination.open("w", encoding="utf-8", newline="") as dst:
            writer = csv.DictWriter(dst, fieldnames=reader.fieldnames)
            writer.writeheader()
            for row in reader:
                split = classify_date(int(row["date"]))
                if split not in allowed_source_splits:
                    raise ValueError(
                        f"{source.name} contains unexpected {split} row dated {row['date']}"
                    )
                counts[split] += 1
                if split == emitted_split:
                    writer.writerow(row)


def prepare_train_valid_view(source_dir: Path, view_dir: Path) -> dict:
    """Build a candidate data directory with exact train/valid rows and no test rows.

    The function validates all rows in both organizer standard logs, including
    the expected hidden-test count, but emits only the fixed train and validation
    date ranges. Candidate subprocesses receive ``view_dir`` rather than
    ``source_dir``.
    """
    source_dir = source_dir.resolve()
    view_dir = view_dir.resolve()
    if view_dir.exists():
        raise FileExistsError(f"candidate data view already exists: {view_dir}")

    required = (TRAIN_LOG, VALID_TEST_LOG, *FEATURE_FILES)
    missing = [name for name in required if not (source_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"source dataset is missing required files: {missing}")

    building = view_dir.with_name(view_dir.name + ".building")
    if building.exists():
        raise FileExistsError(f"stale candidate data view build exists: {building}")
    building.mkdir(parents=True)
    counts = {"train": 0, "valid": 0, "test": 0, "outside": 0}

    try:
        _filter_log(
            source_dir / TRAIN_LOG,
            building / TRAIN_LOG,
            allowed_source_splits={"train"},
            emitted_split="train",
            counts=counts,
        )
        _filter_log(
            source_dir / VALID_TEST_LOG,
            building / VALID_TEST_LOG,
            allowed_source_splits={"valid", "test"},
            emitted_split="valid",
            counts=counts,
        )

        expected = {
            "train": EXPECTED_TRAIN_ROWS,
            "valid": EXPECTED_VALID_ROWS,
            "test": EXPECTED_TEST_ROWS,
            "outside": 0,
        }
        if counts != expected:
            raise ValueError(f"organizer split counts mismatch: got {counts}, expected {expected}")

        for name in FEATURE_FILES:
            shutil.copy2(source_dir / name, building / name)

        manifest = {
            "source_logs": [TRAIN_LOG, VALID_TEST_LOG],
            "date_splits": {
                "train": list(TRAIN_DATES),
                "valid": list(VALID_DATES),
                "test_excluded": list(TEST_DATES),
            },
            "verified_source_counts": counts,
            "emitted_candidate_counts": {
                "train": EXPECTED_TRAIN_ROWS,
                "valid": EXPECTED_VALID_ROWS,
                "test": 0,
            },
            "excluded_files": ["log_random_4_22_to_5_08_pure.csv"],
            "policy": "candidate processes receive train and validation rows only",
        }
        (building / "split_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        building.replace(view_dir)
        return manifest
    except Exception:
        shutil.rmtree(building, ignore_errors=True)
        raise


__all__ = [
    "EXPECTED_TEST_ROWS",
    "EXPECTED_TRAIN_ROWS",
    "EXPECTED_VALID_ROWS",
    "TEST_DATES",
    "TRAIN_DATES",
    "VALID_DATES",
    "classify_date",
    "prepare_train_valid_view",
]
