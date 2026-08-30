"""Provider- and agent-neutral parsing helpers for candidate metrics."""
from __future__ import annotations

import json


def try_extract_metrics(text: str) -> dict | None:
    """Return the last valid JSON line containing benchmark metrics."""
    for line in reversed(text.splitlines()):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict) or "GAUC" not in obj or "primary" not in obj:
            continue
        primary = obj.get("primary", -1)
        if isinstance(primary, (float, int)) and 0.0 <= float(primary) <= 1.0:
            return obj
    return None


__all__ = ["try_extract_metrics"]
