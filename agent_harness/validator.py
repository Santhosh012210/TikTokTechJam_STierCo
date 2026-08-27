"""Two responsibilities:
1. scan_candidate_source() — reject model.py files that access the test split.
2. validate_row() / validate_file() — enforce the JSONL log schema.

Run standalone: python -m agent_harness.validator logs/run_TIMESTAMP.jsonl
"""
import json
import re
import sys
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# 1. Candidate source scanner — test-set prohibition
# ---------------------------------------------------------------------------

FORBIDDEN_PATTERNS: list[tuple[str, str]] = [
    (r"""splits\s*\[\s*['"]test['"]\s*\]""",  "splits['test'] access"),
    (r"""enc\s*\[\s*['"]test['"]\s*\]""",     "enc['test'] access"),
    (r"""['"]test['"]\s*:\s*\(20220429""",    "hardcoded test date range"),
    (r"""SPLITS\s*=.*20220429""",             "hardcoded SPLITS dict with test range"),
    (r"""out\s*\[\s*['"]test['"]\s*\]""",     "out['test'] access"),
]


def scan_candidate_source(source: str) -> list[str]:
    """Return list of violation descriptions. Empty = clean."""
    violations = []
    for pattern, description in FORBIDDEN_PATTERNS:
        if re.search(pattern, source, re.IGNORECASE):
            violations.append(f"{description} (pattern: {pattern})")
    return violations


# ---------------------------------------------------------------------------
# 2. Log schema validator
# ---------------------------------------------------------------------------

REQUIRED_FIELDS: dict[str, type | tuple] = {
    "iteration":         int,
    "session_type":      str,
    "timestamp":         str,
    "parent_iteration":  (int, type(None)),
    "hypothesis":        str,
    "hypothesis_source": str,
    "target_component":  str,
    "code_path":         str,
    "code_diff":         str,
    "status":            str,
    "internal_repairs":  int,
    "metrics":           (dict, type(None)),
    "delta_vs_baseline": (float, int, type(None)),
    "pct_of_headroom":   (float, int, type(None)),
    "is_new_best":       bool,
    "error":             (str, type(None)),
    "tokens":            dict,
    "wall_seconds":      (float, int),
    "human_intervention": bool,
}

STRATEGIST_EXTRA: dict[str, type | tuple] = {
    "reasoning":           str,
    "proposed_hypotheses": list,
}

VALID_SESSION_TYPES = {"builder", "strategist"}
VALID_STATUSES      = {"success", "failed", "rejected"}


def validate_row(row: dict, lineno: int = 0) -> list[str]:
    """Return list of error strings (empty = valid)."""
    errors: list[str] = []
    prefix = f"line {lineno}: " if lineno else ""

    # Required fields + types
    for field, expected_type in REQUIRED_FIELDS.items():
        if field not in row:
            errors.append(f"{prefix}missing field '{field}'")
            continue
        val = row[field]
        if not isinstance(val, expected_type):
            errors.append(
                f"{prefix}field '{field}' has type {type(val).__name__}, "
                f"expected {expected_type}"
            )

    # Enum checks
    if row.get("session_type") not in VALID_SESSION_TYPES:
        errors.append(f"{prefix}session_type must be one of {VALID_SESSION_TYPES}")
    if row.get("status") not in VALID_STATUSES:
        errors.append(f"{prefix}status must be one of {VALID_STATUSES}")

    # Timestamp parseable
    ts = row.get("timestamp", "")
    if ts:
        try:
            datetime.fromisoformat(ts.rstrip("Z"))
        except ValueError:
            errors.append(f"{prefix}timestamp '{ts}' is not valid ISO8601")

    # tokens sub-dict
    tokens = row.get("tokens")
    if isinstance(tokens, dict):
        for k in ("input", "output"):
            if k not in tokens:
                errors.append(f"{prefix}tokens.{k} is missing")
            elif not isinstance(tokens[k], int):
                errors.append(f"{prefix}tokens.{k} must be int")

    # metrics sub-dict (when not None)
    metrics = row.get("metrics")
    if metrics is not None and isinstance(metrics, dict):
        for k in ("GAUC", "nDCG@5", "primary"):
            if k not in metrics:
                errors.append(f"{prefix}metrics.{k} is missing")
            elif not isinstance(metrics[k], (float, int)):
                errors.append(f"{prefix}metrics.{k} must be numeric")
        primary = metrics.get("primary")
        if isinstance(primary, (float, int)) and not (0.0 <= primary <= 1.0):
            errors.append(f"{prefix}metrics.primary={primary} is outside [0,1]")

    # Strategist extras
    if row.get("session_type") == "strategist":
        for field, expected_type in STRATEGIST_EXTRA.items():
            if field not in row:
                errors.append(f"{prefix}strategist row missing field '{field}'")
            elif not isinstance(row[field], expected_type):
                errors.append(
                    f"{prefix}strategist field '{field}' has wrong type "
                    f"(got {type(row[field]).__name__})"
                )

    return errors


def validate_file(path: str | Path) -> tuple[int, int, list[str]]:
    """Parse every line of a JSONL file and validate each row.

    Returns (rows_ok, rows_error, all_error_messages).
    """
    path = Path(path)
    if not path.exists():
        return 0, 0, [f"File not found: {path}"]

    ok = err = 0
    all_errors: list[str] = []

    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                all_errors.append(f"line {lineno}: JSON parse error: {e}")
                err += 1
                continue
            row_errors = validate_row(row, lineno)
            if row_errors:
                all_errors.extend(row_errors)
                err += 1
            else:
                ok += 1

    return ok, err, all_errors


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Validate a harness JSONL log file")
    ap.add_argument("path", help="Path to the .jsonl log file")
    args = ap.parse_args()

    ok, err, errors = validate_file(args.path)
    total = ok + err
    print(f"Validated {total} rows: {ok} ok, {err} with errors")
    for e in errors:
        print(f"  ERROR: {e}", file=sys.stderr)
    sys.exit(1 if err else 0)


if __name__ == "__main__":
    main()
