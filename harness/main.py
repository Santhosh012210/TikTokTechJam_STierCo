"""Harness orchestration: tree search + builder/strategist dispatch + convergence.

Usage:
  # Dev run (fast feedback, 2 builder turns)
  python -m harness.main --max-iter 10 --wall-hours 0.5 --builder-turns 2

  # Production run (after the agent harness is verified)
  python -m harness.main --max-iter 50 --wall-hours 4 --builder-turns 10
"""
import argparse
import difflib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Ensure the harness package is importable when this file is run directly.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from harness.builder import BuilderResult, run_builder_session
from harness.config import Config, load_config
from harness.logger import RunLogger
from harness.provider import validate_provider_environment
from harness.strategist import run_strategist_session
from harness.tree import SearchTree
from harness.validator import scan_candidate_source

# ---------------------------------------------------------------------------
# Seed knowledge (from research brief)
# ---------------------------------------------------------------------------

KNOWN_DEAD_ENDS: list[str] = [
    "Adding CWM 13 feature fields (music_id, video_type, upload_type, user buckets): no gain (0.5940 vs 0.5950)",
    "Embedding dim k=8/16/32: flat (0.5895/0.5902/0.5887) — capacity is not the bottleneck",
    "Pure user-side first-order terms: contribute zero (ranking is within-user; user features constant per user)",
]

INITIAL_UNTRIED_DIRECTIONS: list[str] = [
    "1. Loss: BPR pairwise loss or listwise softmax-per-user (aligns objective with GAUC/nDCG ranking metrics)",
    "2. User history sequences: DIN/SIM-style attention over user's past interactions (completely blank slate)",
    "3. Multi-task: is_click/is_like/is_follow/is_comment/is_forward/play_time_ms as auxiliary tasks for long_view",
    "4. Watch-time censored regression: one-sided loss for truncated play_time_ms (CWM-style)",
    "5. Model architecture: DeepFM or DCN (lower priority; capacity shown not to be bottleneck)",
    "6. Temporal features: hourmin, date, and modelling train/valid distribution drift",
    "7. Unbiased validation: log_random_4_22_to_5_08_pure.csv as debiased check",
]

# ---------------------------------------------------------------------------
# Root model.py template (FM baseline adapted for the harness contract)
# ---------------------------------------------------------------------------

def make_root_model_py(config: Config) -> str:
    return f'''"""Root candidate: FM baseline (k=16, pointwise BCE) adapted for harness contract."""
import sys
sys.path.insert(0, r'{config.BASELINE_ROOT}')

import argparse, json, time, collections
import numpy as np
from data import load, encode, FIELDS
from evaluate import evaluate


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


class FM:
    def __init__(self, dim, k=16, lr=0.001, l2=1e-6, seed={config.SEED}):
        rng = np.random.default_rng(seed)
        self.V = rng.normal(0, 0.01, (dim, k)).astype(np.float32)
        self.W = np.zeros(dim, dtype=np.float32)
        self.b = np.float32(0.0)
        self.lr, self.l2 = lr, l2
        self.mV = np.zeros_like(self.V); self.vV = np.zeros_like(self.V)
        self.mW = np.zeros_like(self.W); self.vW = np.zeros_like(self.W)
        self.t = 0

    def logits(self, X):
        E = self.V[X]
        S = E.sum(1)
        inter = 0.5 * ((S ** 2).sum(1) - (E ** 2).sum((1, 2)))
        return self.b + self.W[X].sum(1) + inter, E, S

    def step(self, X, y):
        B = len(y)
        z, E, S = self.logits(X)
        g = ((sigmoid(z) - y) / B).astype(np.float32)
        gV = np.zeros_like(self.V); gW = np.zeros_like(self.W)
        np.add.at(gW, X, g[:, None])
        np.add.at(gV, X, g[:, None, None] * (S[:, None, :] - E))
        gV += self.l2 * self.V; gW += self.l2 * self.W
        self.t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        for P, G, M, Vv in ((self.V, gV, self.mV, self.vV), (self.W, gW, self.mW, self.vW)):
            M *= b1; M += (1 - b1) * G
            Vv *= b2; Vv += (1 - b2) * (G * G)
            P -= self.lr * (M / (1 - b1 ** self.t)) / (np.sqrt(Vv / (1 - b2 ** self.t)) + eps)
        self.b -= self.lr * g.sum()

    def predict(self, X, bs=200_000):
        return np.concatenate([self.logits(X[i:i + bs])[0] for i in range(0, len(X), bs)])


ap = argparse.ArgumentParser()
ap.add_argument('--data_dir', default=r'{config.DATA_DIR}')
a = ap.parse_args()

splits = load(a.data_dir)
enc, dim = encode(splits)
Xtr, ytr, _ = enc['train']
Xva, yva, uva = enc['valid']

m = FM(dim)
rng = np.random.default_rng({config.SEED})
best, best_state, bad = -1, None, 0
for ep in range(40):
    idx = rng.permutation(len(ytr))
    for i in range(0, len(idx), 8192):
        m.step(Xtr[idx[i:i + 8192]], ytr[idx[i:i + 8192]])
    va = evaluate(uva, yva, m.predict(Xva))
    if va['primary'] > best + 1e-5:
        best, bad = va['primary'], 0
        best_state = (m.V.copy(), m.W.copy(), np.float32(m.b))
    else:
        bad += 1
        if bad >= 4:
            break

m.V, m.W, m.b = best_state
result = evaluate(uva, yva, m.predict(Xva))
print(json.dumps(result))
'''


# ---------------------------------------------------------------------------
# Convergence check (deterministic Python — not in any agent)
# ---------------------------------------------------------------------------

def check_convergence(history: list[float], epsilon: float, N: int) -> bool:
    if len(history) < N:
        return False
    recent = history[-N:]
    improvements = [max(0.0, recent[i] - recent[i - 1]) for i in range(1, len(recent))]
    return max(improvements, default=0.0) <= epsilon


# ---------------------------------------------------------------------------
# Target component classifier (keyword heuristic)
# ---------------------------------------------------------------------------

_COMPONENT_KEYWORDS: list[tuple[str, str]] = [
    ("bpr",       "loss"),
    ("listwise",  "loss"),
    ("softmax",   "loss"),
    ("pairwise",  "loss"),
    ("din",       "architecture"),
    ("sim",       "architecture"),
    ("attention", "architecture"),
    ("deepfm",    "architecture"),
    ("dcn",       "architecture"),
    ("multi-task","multi-task"),
    ("auxiliary", "multi-task"),
    ("multitask", "multi-task"),
    ("watch",     "features"),
    ("censored",  "features"),
    ("temporal",  "features"),
    ("hourmin",   "features"),
    ("sequence",  "features"),
]


def classify_target(hypothesis: str) -> str:
    h = hypothesis.lower()
    for kw, label in _COMPONENT_KEYWORDS:
        if kw in h:
            return label
    return "other"


# ---------------------------------------------------------------------------
# Log row assemblers
# ---------------------------------------------------------------------------

def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def assemble_builder_log(
    iteration: int,
    parent_node_id: int | None,
    hypothesis: str,
    hypothesis_source: str,
    candidate_dir: Path,
    result: BuilderResult,
    status: str,
    is_new_best: bool,
    baseline_primary: float,
    headroom: float,
    parent_code: str,
) -> dict:
    new_code = result.final_code or ""
    diff_lines = list(difflib.unified_diff(
        parent_code.splitlines(),
        new_code.splitlines(),
        fromfile=f"trial_{parent_node_id:03d}/model.py" if parent_node_id is not None else "baseline",
        tofile=f"{candidate_dir.name}/model.py",
        lineterm="",
    ))
    code_diff = "\n".join(diff_lines)

    metrics = result.metrics
    delta = (metrics["primary"] - baseline_primary) if metrics else None
    pct   = (delta / headroom * 100) if (delta is not None and headroom > 0) else None

    return {
        "iteration":         iteration,
        "session_type":      "builder",
        "timestamp":         _iso_now(),
        "parent_iteration":  parent_node_id,
        "hypothesis":        hypothesis,
        "hypothesis_source": hypothesis_source,
        "target_component":  classify_target(hypothesis),
        "code_path":         str(candidate_dir / "model.py"),
        "code_diff":         code_diff,
        "status":            status,
        "internal_repairs":  result.repair_count,
        "metrics":           metrics,
        "delta_vs_baseline": delta,
        "pct_of_headroom":   pct,
        "is_new_best":       is_new_best,
        "error":             result.error,
        "tokens":            result.token_counts,
        "wall_seconds":      result.wall_seconds,
        "human_intervention": False,
    }


def assemble_strategist_log(iteration: int, strat, config: Config) -> dict:
    from harness.strategist import StrategistResult
    return {
        "iteration":          iteration,
        "session_type":       "strategist",
        "timestamp":          _iso_now(),
        "parent_iteration":   None,
        "hypothesis":         strat.direction,
        "hypothesis_source":  "strategist",
        "target_component":   classify_target(strat.direction),
        "code_path":          "",
        "code_diff":          "",
        "status":             "success" if strat.error is None else "failed",
        "internal_repairs":   0,
        "metrics":            None,
        "delta_vs_baseline":  None,
        "pct_of_headroom":    None,
        "is_new_best":        False,
        "error":              strat.error,
        "tokens":             strat.token_counts,
        "wall_seconds":       strat.wall_seconds,
        "human_intervention": False,
        "reasoning":          strat.reasoning,
        "proposed_hypotheses": strat.hypotheses,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="KuaiRand autonomous ML research harness")
    ap.add_argument("--max-iter",      type=int,   default=10,   help="Iteration budget")
    ap.add_argument("--wall-hours",    type=float, default=0.5,  help="Wall-clock budget in hours")
    ap.add_argument("--builder-turns", type=int,   default=2,    help="Builder max_turns per session")
    ap.add_argument("--data-dir",      type=str,   default=None, help="Override data directory")
    args = ap.parse_args()

    config = load_config()

    # Apply CLI overrides
    config.BUILDER_MAX_TURNS = args.builder_turns
    config.ITERATION_BUDGET  = args.max_iter
    config.WALL_BUDGET_S     = args.wall_hours * 3600.0
    if args.data_dir:
        config.DATA_DIR = Path(args.data_dir)

    # Fail before creating run files if the selected provider is misconfigured.
    provider_info = validate_provider_environment()

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    logger = RunLogger(config.ARTIFACTS_DIR, run_id)
    run_workspace = config.EXPERIMENT_WORKSPACE_DIR / run_id
    run_workspace.mkdir(parents=True, exist_ok=True)
    print(f"[harness] run_id={run_id}  log={logger.path}")
    print(f"[harness] workspace={run_workspace}")
    print(f"[harness] provider={provider_info['provider']}  model={provider_info['model']}")
    print(f"[harness] budget: {config.ITERATION_BUDGET} iters / {args.wall_hours}h / builder_turns={config.BUILDER_MAX_TURNS}")
    print(f"[harness] baseline primary={config.BASELINE_PRIMARY:.4f}  oracle={config.ORACLE_PRIMARY:.4f}  headroom={config.HEADROOM:.4f}")

    # ------------------------------------------------------------------
    # Root candidate: FM baseline
    # ------------------------------------------------------------------
    root_candidate_dir = run_workspace / "trial_000"
    root_candidate_dir.mkdir(parents=True, exist_ok=True)
    root_model_py = root_candidate_dir / "model.py"
    root_model_py.write_text(make_root_model_py(config), encoding="utf-8")

    tree = SearchTree(
        root_code_path=str(root_model_py),
        root_hypothesis="FM baseline (k=16, pointwise BCE, Adam) — no change",
    )

    # Run root to establish a real primary score in the tree
    print("[harness] Running root node (FM baseline)...")
    root_result = run_builder_session(
        hypothesis="Run the FM baseline as-is to establish the reference score",
        parent_code_path=str(root_model_py),
        candidate_dir=root_candidate_dir,
        config=config,
        node_id=0,
    )
    root_primary = root_result.metrics["primary"] if root_result.metrics else config.BASELINE_PRIMARY
    root_status  = "success" if root_result.success else "failed"
    tree.update(0, primary=root_primary, status=root_status)

    root_log = assemble_builder_log(
        iteration=0,
        parent_node_id=None,
        hypothesis="FM baseline (k=16, pointwise BCE, Adam) — no change",
        hypothesis_source="harness_root",
        candidate_dir=root_candidate_dir,
        result=root_result,
        status=root_status,
        is_new_best=root_result.success,
        baseline_primary=config.BASELINE_PRIMARY,
        headroom=config.HEADROOM,
        parent_code="",
    )
    logger.write(root_log)  # WRITE BEFORE DECIDE
    print(f"[harness] Root: status={root_status}  primary={root_primary:.4f}")

    primary_history: list[float] = [root_primary]
    hypothesis_history: list[dict] = [{
        "iteration": 0,
        "hypothesis": "FM baseline",
        "primary": root_primary,
        "status": root_status,
    }]
    pending_strategist_hypotheses: list[str] = []
    untried_directions = list(INITIAL_UNTRIED_DIRECTIONS)
    direction_idx = 0   # round-robin index through untried directions

    run_start = time.time()

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    for iteration in range(1, config.ITERATION_BUDGET + 1):

        # Budget checks
        elapsed = time.time() - run_start
        if elapsed > config.WALL_BUDGET_S:
            print(f"[harness] Wall-clock budget reached at iteration {iteration} ({elapsed/3600:.2f}h)")
            break

        if check_convergence(primary_history, config.CONVERGENCE_EPSILON, config.CONVERGENCE_N):
            print(f"[harness] Convergence detected at iteration {iteration}")
            break

        # Strategist trigger
        consecutive_no_improve = (
            len(primary_history) >= 3
            and primary_history[-1] <= tree._best_primary
            and primary_history[-2] <= tree._best_primary
        )
        need_strategist = (
            iteration % config.STRATEGIST_EVERY_N == 0
            or (consecutive_no_improve and not pending_strategist_hypotheses)
        )

        if need_strategist:
            print(f"[harness] iter {iteration}: running Strategist session...")
            strat = run_strategist_session(
                leaderboard=tree.leaderboard(top_k=10),
                hypothesis_history=hypothesis_history[-20:],
                dead_ends=KNOWN_DEAD_ENDS,
                untried_directions=untried_directions,
                current_best=tree._best_primary,
                config=config,
            )
            strat_log = assemble_strategist_log(iteration, strat, config)
            logger.write(strat_log)      # WRITE BEFORE DECIDE
            pending_strategist_hypotheses = list(strat.hypotheses)
            print(f"[harness]   Strategist direction: {strat.direction[:80]}")

        # Pick hypothesis
        if pending_strategist_hypotheses:
            hypothesis = pending_strategist_hypotheses.pop(0)
            hypothesis_source = f"strategist_iter_{iteration:03d}"
        else:
            # Round-robin through untried directions as seed
            hypothesis = untried_directions[direction_idx % len(untried_directions)]
            hypothesis_source = "tree_round_robin"
            direction_idx += 1

        # Select parent via UCB
        parent_node = tree.select(C=config.UCB_C)
        parent_code_path = parent_node.code_path
        parent_code = (
            Path(parent_code_path).read_text(encoding="utf-8")
            if Path(parent_code_path).exists() else ""
        )

        # Create candidate directory and add tree node
        candidate_dir = run_workspace / f"trial_{iteration:03d}"
        candidate_dir.mkdir(parents=True, exist_ok=True)
        child_node = tree.add_child(
            parent_id=parent_node.id,
            code_path=str(candidate_dir / "model.py"),
            hypothesis=hypothesis,
        )

        print(f"[harness] iter {iteration}/{config.ITERATION_BUDGET}: {hypothesis[:70]}...")

        # Run Builder session
        result = run_builder_session(
            hypothesis=hypothesis,
            parent_code_path=parent_code_path,
            candidate_dir=candidate_dir,
            config=config,
            node_id=child_node.id,
        )

        # Defense-in-depth: re-scan final model.py
        status = "success" if result.success else "failed"
        if result.final_code:
            violations = scan_candidate_source(result.final_code)
            if violations:
                status = "rejected"
                result = BuilderResult(
                    success=False,
                    final_code=result.final_code,
                    metrics=None,
                    repair_count=result.repair_count,
                    token_counts=result.token_counts,
                    wall_seconds=result.wall_seconds,
                    error=f"Post-hoc scan violations: {violations}",
                    stdout_raw=result.stdout_raw,
                )

        new_primary = result.metrics["primary"] if result.metrics else None
        is_new_best = new_primary is not None and new_primary > tree._best_primary

        # Assemble log row
        log_row = assemble_builder_log(
            iteration=iteration,
            parent_node_id=parent_node.id,
            hypothesis=hypothesis,
            hypothesis_source=hypothesis_source,
            candidate_dir=candidate_dir,
            result=result,
            status=status,
            is_new_best=is_new_best,
            baseline_primary=config.BASELINE_PRIMARY,
            headroom=config.HEADROOM,
            parent_code=parent_code,
        )

        # WRITE BEFORE DECIDE
        logger.write(log_row)

        # Update tree and history AFTER write
        tree.update(child_node.id, primary=new_primary or -1.0, status=status)
        primary_history.append(new_primary if new_primary is not None else primary_history[-1])
        hypothesis_history.append({
            "iteration": iteration,
            "hypothesis": hypothesis,
            "primary": new_primary,
            "status": status,
        })

        if is_new_best:
            print(f"[harness]   NEW BEST primary={new_primary:.4f} (+{new_primary - config.BASELINE_PRIMARY:+.4f} vs baseline, {(new_primary - config.BASELINE_PRIMARY)/config.HEADROOM*100:.1f}% of headroom)")
        elif new_primary is not None:
            print(f"[harness]   primary={new_primary:.4f}  repairs={result.repair_count}  {result.wall_seconds:.1f}s")
        else:
            print(f"[harness]   FAILED  repairs={result.repair_count}  error={str(result.error)[:60]}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    totals = logger.running_totals()
    best_nodes = tree.leaderboard(top_k=1)
    best_node  = best_nodes[0] if best_nodes else None

    results_payload = {
        "run_id": run_id,
        "baseline": {
            "GAUC": config.BASELINE_GAUC,
            "nDCG@5": config.BASELINE_NDCG,
            "primary": config.BASELINE_PRIMARY,
        },
        "best_trial": (
            {
                "trial": best_node.id,
                "primary": best_node.primary,
                "hypothesis": best_node.hypothesis,
                "workspace_code_path": best_node.code_path,
            }
            if best_node else None
        ),
        "trials_completed": len(primary_history) - 1,
        "tokens": totals["tokens"],
        "manual_interventions": totals["interventions"],
        "tree": tree.to_dict(),
    }
    results_path = logger.write_results(results_payload)

    best_summary = (
        f"trial_{best_node.id:03d} with validation primary "
        f"{best_node.primary:.4f}\n\n"
        f"Hypothesis: {best_node.hypothesis}\n"
        if best_node else "No successful trial was produced.\n"
    )
    report_path = logger.write_report(
        f"""# Autonomous research run {run_id}

## Outcome

{best_summary}
## Run totals

- Trials completed: {len(primary_history) - 1}
- Input tokens: {totals['tokens']['input']}
- Output tokens: {totals['tokens']['output']}
- Manual interventions: {totals['interventions']}
- Baseline validation primary: {config.BASELINE_PRIMARY:.4f}

## Evidence

- Raw event log: `../logs/events.jsonl`
- Machine-readable results: `../results/metrics.json`

Trial implementations were generated in the Git-ignored `experiment_workspace/`.
Promote the selected final implementation into `artifacts/final/` before submission.
"""
    )
    logger.close()

    print("\n" + "=" * 60)
    print(f"Run complete. Log: {logger.path}")
    print(f"Trials: {len(primary_history) - 1} (root + {len(primary_history) - 1} experiments)")
    print(f"Tokens: input={totals['tokens']['input']:,}  output={totals['tokens']['output']:,}")
    print(f"Manual interventions: {totals['interventions']}")
    if best_node:
        print(f"Best trial: trial_{best_node.id:03d}  primary={best_node.primary:.4f}")
        print(f"  Code: {best_node.code_path}")
        print(f"  Hypothesis: {best_node.hypothesis}")
    print("=" * 60)
    print(f"Results: {results_path}")
    print(f"Report: {report_path}")
    print("\nNext steps:")
    print(f"  1. Validate the log: python -m harness.validator {logger.path}")
    print("  2. Run test eval on best candidate (ONE TIME ONLY)")
    print("  3. Copy the selected model and submission evidence into artifacts/final/")


if __name__ == "__main__":
    main()
