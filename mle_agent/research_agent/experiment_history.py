"""Operational history of every scored experiment, written by the harness.

Within a run the agent already cannot repeat itself: ``seen_candidate_fingerprints``
catches identical candidate code, ``_repeats_measured_dead_end`` catches the two
directions the README rules out, and the diversity gate forces the first three
experiments onto distinct components. All of that dies with the run.

So the second run cheerfully re-tests what the first run disproved. In the
20260831_134234 run the agent spent two of its three experiments establishing
that same-user BPR scores 0.499 and a causal-exposure DIN scores 0.597 — both
decisive, both expensive, and both of which the next run would have had to
rediscover from scratch.

This module is that missing memory. Two deliberate constraints:

* **The harness writes it, not the agent.** Every row is built from a validated
  execution result, so a hypothesis can never be recorded as measured unless it
  actually ran and was scored by the trusted evaluator. Handing the agent a
  write tool would turn this from evidence into assertion.
* **It is separate from ``prior_experiments.json``.** That file is a curated,
  re-scored frontier with per-trial content digests, cited by the agent as
  verified evidence. This one is an append-only operational log that includes
  failures. Mixing them would let an unverified row be cited as a verified one.

Rows are appended per experiment rather than per run, so a run that crashes at
iteration 20 still contributes the 19 results it earned.

The agent-facing interpretation lives in ``prompts/prior_findings.md``. This raw
ledger supplies audit history and exact candidate fingerprints; it is deliberately
not injected wholesale into the model's context.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path


HISTORY_PATH = Path(__file__).with_name("experiment_history.json")

SCHEMA_VERSION = 1

#: How many rows the prompt renders. The full file is kept on disk; only the
#: most informative slice is resent on every model call.
PROMPT_ROW_LIMIT = 24

# Outcome vocabulary.
#
#   improved              scored above the incumbent
#   rejected              a competent result that lost; the idea was tested
#   suspect_implementation  scored at or below the item-popularity rung, i.e. worse
#                         than ranking by global popularity while having every
#                         baseline feature and more. That is a broken candidate,
#                         not a refuted hypothesis, and recording it as a refutation
#                         would retire a whole research direction on the strength of
#                         one bug. The 20260831_134234 run produced exactly this:
#                         same-user BPR at GAUC 0.522 against 0.499 for random.
#   failed                never produced a score at all
#
#: Outcomes worth warning the agent about before it spends an experiment.
_NEGATIVE = {"rejected", "failed", "suspect_implementation"}

#: Negative outcomes where the *idea* is settled and should not be re-tested.
_SETTLED = {"rejected"}

SUSPECT = "suspect_implementation"


def classify_outcome(
    primary: float | None,
    incumbent_primary: float,
    popularity_primary: float,
    hypothesis_status: str = "",
) -> str:
    """Label one experiment from its measured score and the agent's own diagnosis.

    The score is the primary signal and is never overridden: an improvement is an
    improvement whatever the agent claims. But the popularity rung only catches
    implementations broken badly enough to score near chance. A subtler bug can
    still land at 0.59 and look like a fair test that lost, so an explicit
    ``not_tested`` diagnosis from the agent is honoured too -- it is the only
    thing that can distinguish "we tried it and it lost" from "we never
    actually tried it".
    """
    if primary is None:
        return "failed"
    if primary > incumbent_primary:
        return "improved"
    if primary <= popularity_primary:
        return SUSPECT
    if hypothesis_status == "not_tested":
        return SUSPECT
    return "rejected"


@dataclass
class ExperimentRecord:
    """One scored (or attempted) experiment, from validated harness state."""

    run_id: str
    iteration: int
    hypothesis: str
    target_component: str
    outcome: str  # improved | rejected | failed
    primary: float | None
    delta_vs_incumbent: float | None
    incumbent_primary: float | None
    candidate_fingerprint: str | None = None
    reflection: str = ""
    error: str = ""
    recorded_at: str = ""

    def as_row(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "iteration": self.iteration,
            "hypothesis": self.hypothesis[:400],
            "target_component": self.target_component,
            "outcome": self.outcome,
            "primary": self.primary,
            "delta_vs_incumbent": self.delta_vs_incumbent,
            "incumbent_primary": self.incumbent_primary,
            "candidate_fingerprint": self.candidate_fingerprint,
            "reflection": self.reflection[:400],
            "error": self.error[:300],
            "recorded_at": self.recorded_at or _now(),
        }


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def load_history(path: Path | None = None) -> dict[str, object]:
    """Read the history, returning an empty ledger when there is none yet.

    Unlike ``prior_experiments.json`` this fails *open*: a missing or corrupt
    operational log must not stop a run, because it is an optimisation, not the
    evidence base. A corrupt file is reported rather than silently replaced.
    """
    target = path or HISTORY_PATH
    if not target.is_file():
        return {"schema_version": SCHEMA_VERSION, "rows": [], "corrupt": False}
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
        rows = payload.get("rows", [])
        if not isinstance(rows, list):
            raise ValueError("rows must be a list")
        return {
            "schema_version": payload.get("schema_version", SCHEMA_VERSION),
            "rows": rows,
            "corrupt": False,
        }
    except (json.JSONDecodeError, ValueError, OSError):
        return {"schema_version": SCHEMA_VERSION, "rows": [], "corrupt": True}


def append_record(record: ExperimentRecord, path: Path | None = None) -> None:
    """Append one row durably.

    Rewrites the whole file through a temporary name so an interrupted write can
    never leave a truncated ledger that ``load_history`` would then discard.
    """
    target = path or HISTORY_PATH
    history = load_history(target)
    rows = list(history["rows"])
    rows.append(record.as_row())
    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "harness_written_operational_log_not_verified_evidence",
        "note": (
            "Every row was produced by the harness from a validated execution. "
            "Cite prior_experiments.json, not this file, for verified scores."
        ),
        "rows": rows,
    }
    temporary = target.with_suffix(".partial")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(temporary, target)


def failed_fingerprints(path: Path | None = None) -> set[str]:
    """Candidate code digests that were already measured and did not improve.

    Used as a hard gate: an identical candidate cannot produce a different score,
    so re-running one is pure waste. This is intentionally narrow — it catches
    exact repeats only, never a re-implementation of the same idea.
    """
    return {
        str(row["candidate_fingerprint"])
        for row in load_history(path)["rows"]
        if row.get("candidate_fingerprint") and row.get("outcome") in _NEGATIVE
    }


def compact_for_prompt(path: Path | None = None, limit: int = PROMPT_ROW_LIMIT) -> str:
    """Render the history for a prompt: all negatives first, then recent wins.

    Negative results are the point of this file — a direction already disproved is
    what the agent most needs to not spend an experiment on — so they are never
    crowded out by a long tail of successes.
    """
    rows = load_history(path)["rows"]
    if not rows:
        return json.dumps(
            {"rows": [], "note": "No cross-run experiment history yet."},
            ensure_ascii=False, separators=(",", ":"),
        )
    negatives = [row for row in rows if row.get("outcome") in _NEGATIVE]
    positives = [row for row in rows if row.get("outcome") not in _NEGATIVE]
    selected = negatives[-limit:] + positives[-max(0, limit - len(negatives[-limit:])):]

    def slim(row: dict) -> dict:
        entry = {
            "run": row.get("run_id"),
            "component": row.get("target_component"),
            "hypothesis": str(row.get("hypothesis", ""))[:220],
            "outcome": row.get("outcome"),
            "primary": row.get("primary"),
            "delta": row.get("delta_vs_incumbent"),
        }
        if row.get("outcome") == SUSPECT:
            entry["retry_guidance"] = (
                "Direction still open: this scored no better than a trivial "
                "popularity ranker, so treat it as an implementation failure to "
                "diagnose, not a settled negative result."
            )
            diagnosis = str(row.get("error") or row.get("reflection") or "").strip()
            if diagnosis:
                # A generic "suspect" label prevents a false refutation, but it does
                # not prevent the next run from re-deriving the same implementation
                # bug. Carry the concrete harness-reviewed diagnosis in the compact
                # prompt so a retry starts from the known failure mechanism.
                entry["implementation_diagnosis"] = diagnosis[:300]
        return entry

    return json.dumps(
        {
            "note": (
                "Experiments already measured in earlier runs of this agent. "
                "'rejected' means the idea was tested competently and lost: do not "
                "re-test it unless you can state specifically what the earlier "
                "attempt got wrong. 'suspect_implementation' means the candidate "
                "scored at or below ranking by global item popularity, which is "
                "evidence that the code was broken rather than that the idea is "
                "wrong -- these directions are still open and worth retrying with a "
                "correct implementation, so diagnose the earlier failure first. "
                "'failed' means it never produced a score. The harness refuses an "
                "identical candidate outright in every one of these cases, because "
                "the same code cannot score differently."
            ),
            "total_recorded": len(rows),
            "rows": [slim(row) for row in selected],
        },
        ensure_ascii=False, separators=(",", ":"),
    )


__all__ = [
    "HISTORY_PATH",
    "PROMPT_ROW_LIMIT",
    "ExperimentRecord",
    "append_record",
    "SUSPECT",
    "classify_outcome",
    "compact_for_prompt",
    "failed_fingerprints",
    "load_history",
]
