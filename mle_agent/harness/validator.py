"""Two responsibilities:
1. scan_candidate_source() — reject hidden-test access and package-manager bypasses.
2. validate_row() / validate_file() — enforce the JSONL log schema.

Run standalone:
python -m mle_agent.harness.validator artifacts/runs/<run_id>/logs/events.jsonl
"""
import json
import ast
import math
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

    # Catch aliases (data['test']) and simple computed keys
    # (splits['te' + 'st']) that the regex layer cannot recognize.
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return violations

    def constant_string(node: ast.AST) -> str | None:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            left = constant_string(node.left)
            right = constant_string(node.right)
            if left is not None and right is not None:
                return left + right
        return None

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            imported = (
                [alias.name for alias in node.names]
                if isinstance(node, ast.Import)
                else [node.module or ""]
            )
            for module in imported:
                if module.split(".", 1)[0] in {"ensurepip", "pip", "subprocess"}:
                    message = f"candidate process-launch/package-manager import: {module}"
                    if message not in violations:
                        violations.append(message)
            if isinstance(node, ast.ImportFrom) and node.module == "os":
                for alias in node.names:
                    if alias.name in {"system", "popen"} or alias.name.startswith(
                        ("exec", "spawn")
                    ):
                        message = f"candidate process-launch import: os.{alias.name}"
                        if message not in violations:
                            violations.append(message)
        if isinstance(node, ast.Subscript) and constant_string(node.slice) == "test":
            message = "subscript access to the test split"
            if message not in violations:
                violations.append(message)
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and (
                node.func.id in {"system", "popen"}
                or node.func.id.startswith(("exec", "spawn"))
            ):
                message = f"candidate process launch via {node.func.id}"
                if message not in violations:
                    violations.append(message)
            if isinstance(node.func, ast.Attribute):
                owner = node.func.value
                if (
                    isinstance(owner, ast.Name)
                    and owner.id == "os"
                    and (
                        node.func.attr in {"system", "popen"}
                        or node.func.attr.startswith(("exec", "spawn"))
                    )
                ):
                    message = f"candidate process launch via os.{node.func.attr}"
                    if message not in violations:
                        violations.append(message)
            for arg in node.args:
                if constant_string(arg) == "test":
                    message = "function call referencing the test split"
                    if message not in violations:
                        violations.append(message)
                value = constant_string(arg)
                if (
                    value
                    and isinstance(node.func, ast.Name)
                    and node.func.id == "__import__"
                    and value.split(".", 1)[0] in {"ensurepip", "pip", "subprocess"}
                ):
                    message = f"dynamic candidate package/process import: {value}"
                    if message not in violations:
                        violations.append(message)
                if value and re.search(
                    r"(?:^|\s)(?:pip(?:3)?\s+install|python\s+-m\s+pip|"
                    r"conda\s+install|uv\s+pip\s+install)(?:\s|$)",
                    value,
                    re.IGNORECASE,
                ):
                    message = "candidate package-manager command"
                    if message not in violations:
                        violations.append(message)
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

AGENT_EXTRA: dict[str, type | tuple] = {
    "reasoning": str,
    "reflection": str,
    "execution_attempts": list,
    "recovery_events": list,
}

AGENT_V2_EXTRA: dict[str, type | tuple] = {
    "schema_version": int,
    "code_diff_reason": (str, type(None)),
    "hypothesis_supported": (bool, type(None)),
    "hypothesis_status": (str,),
    "implementation_diagnosis": (str,),
    "suggested_next": str,
}

VALID_SESSION_TYPES = {"builder", "strategist", "agent"}
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
            elif tokens[k] < 0:
                errors.append(f"{prefix}tokens.{k} must be non-negative")

    # metrics sub-dict (when not None)
    metrics = row.get("metrics")
    if metrics is not None and isinstance(metrics, dict):
        for k in ("GAUC", "nDCG@5", "primary"):
            if k not in metrics:
                errors.append(f"{prefix}metrics.{k} is missing")
            elif not isinstance(metrics[k], (float, int)):
                errors.append(f"{prefix}metrics.{k} must be numeric")
            elif not math.isfinite(float(metrics[k])):
                errors.append(f"{prefix}metrics.{k} must be finite")
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

    if row.get("session_type") == "agent":
        for field, expected_type in AGENT_EXTRA.items():
            if field not in row:
                errors.append(f"{prefix}agent row missing field '{field}'")
            elif not isinstance(row[field], expected_type):
                errors.append(
                    f"{prefix}agent field '{field}' has wrong type "
                    f"(got {type(row[field]).__name__})"
                )

        if int(row.get("schema_version", 1)) >= 2:
            for field, expected_type in AGENT_V2_EXTRA.items():
                if field not in row:
                    errors.append(f"{prefix}agent v2 row missing field '{field}'")
                elif not isinstance(row[field], expected_type):
                    errors.append(
                        f"{prefix}agent v2 field '{field}' has wrong type "
                        f"(got {type(row[field]).__name__})"
                    )

            iteration = row.get("iteration")
            status = row.get("status")
            hypothesis = row.get("hypothesis", "")
            reasoning = row.get("reasoning", "")
            reflection = row.get("reflection", "")
            code_diff = row.get("code_diff", "")
            diff_reason = row.get("code_diff_reason")
            attempts = row.get("execution_attempts", [])
            recoveries = row.get("recovery_events", [])
            intervention_count = row.get("manual_intervention_count")
            if intervention_count is not None and (
                not isinstance(intervention_count, int) or intervention_count < 0
            ):
                errors.append(
                    f"{prefix}manual_intervention_count must be a non-negative int"
                )
            if not str(hypothesis).strip():
                errors.append(f"{prefix}agent v2 hypothesis must be non-empty")
            if not str(reasoning).strip():
                errors.append(f"{prefix}agent v2 reasoning must be non-empty")
            if status == "success" and not str(reflection).strip():
                errors.append(f"{prefix}successful agent v2 row requires reflection")
            if isinstance(iteration, int) and iteration > 0 and status == "success":
                if not str(code_diff).strip():
                    errors.append(
                        f"{prefix}successful experiment requires a substantive code diff"
                    )
                if not attempts:
                    errors.append(
                        f"{prefix}successful experiment requires execution attempts"
                    )
            if not str(code_diff).strip() and not str(diff_reason or "").strip():
                errors.append(
                    f"{prefix}empty code diff requires code_diff_reason"
                )
            if status == "failed" and not row.get("error") and not recoveries:
                errors.append(
                    f"{prefix}failed experiment requires an error or recovery event"
                )
            for index, attempt in enumerate(attempts):
                if not isinstance(attempt, dict):
                    errors.append(
                        f"{prefix}execution_attempts[{index}] must be an object"
                    )
                    continue
                if not isinstance(attempt.get("success"), bool):
                    errors.append(
                        f"{prefix}execution_attempts[{index}].success must be bool"
                    )
                wall = attempt.get("wall_seconds")
                if not isinstance(wall, (int, float)) or not math.isfinite(float(wall)):
                    errors.append(
                        f"{prefix}execution_attempts[{index}].wall_seconds must be finite"
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
