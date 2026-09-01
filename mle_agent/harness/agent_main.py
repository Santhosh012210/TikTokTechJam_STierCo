"""CLI for the autonomous MLE loop.

Usage:
    python -m mle_agent.harness.agent_main --max-iter 3 --wall-hours 0.5 \
        --bootstrap-turns 24 --agent-turns 16
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import math
import shutil
import statistics
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from mle_agent.harness.agent_tools import execute_model
from mle_agent.harness.config import Config, load_config
from mle_agent.harness.console import console
from mle_agent.harness.data_view import prepare_train_valid_view
from mle_agent.harness.logger import RunLogger
from mle_agent.harness.provider import make_langchain_client, resolve_langchain_settings
from mle_agent.harness.root_model import assert_organizer_fm_equivalence, make_root_model_py
from mle_agent.harness.run_environment import (
    create_run_environment,
    snapshot_run_environment,
)
from mle_agent.research_agent.agent import AgentIterationResult, ResearchAgent
from mle_agent.research_agent.experiment_history import HISTORY_PATH
from mle_agent.research_agent.search.frontier import CandidateFrontier, CandidateNode


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _diff(before: str, after: str, old_name: str, new_name: str) -> str:
    return "\n".join(difflib.unified_diff(
        before.splitlines(), after.splitlines(), fromfile=old_name, tofile=new_name, lineterm=""
    ))


def _log_row(
    iteration: int,
    parent_iteration: int | None,
    result: AgentIterationResult,
    status: str,
    is_new_best: bool,
    baseline: float,
    headroom: float,
    code_diff: str,
    code_path: Path,
    stability: dict | None = None,
) -> dict:
    primary = result.metrics.get("primary") if result.metrics else None
    delta = primary - baseline if primary is not None else None
    chosen_attempt = result.selected_execution or next(
        (attempt for attempt in reversed(result.executions) if attempt.get("success")),
        result.executions[-1] if result.executions else {},
    )
    proposal = chosen_attempt.get("proposal", {})
    # An iteration can end before the agent ever produces a proposal -- most often
    # when the provider becomes unavailable on the first model call. That is an
    # abort, not an experiment, but it still has to be logged, and the strict
    # schema requires a non-empty hypothesis and reasoning. Say plainly what
    # happened instead of letting the run crash on its own log row.
    aborted_before_proposal = not result.success and not result.executions
    reason = result.error or "the agent returned no hypothesis, reasoning, or tool call"
    hypothesis = result.hypothesis
    reasoning = result.reasoning
    if aborted_before_proposal:
        # Fill each field independently: the agent may have produced one and not the
        # other. Anything it did say is kept verbatim.
        if not str(hypothesis).strip():
            hypothesis = "No hypothesis: the iteration ended before the agent proposed one."
        if not str(reasoning).strip():
            reasoning = f"Iteration aborted before any candidate was proposed or run: {reason}"
    recovery_outcome = "recovered" if result.success else "unresolved"
    recovery_events = [
        {**event, "outcome": event.get("outcome", recovery_outcome)}
        for event in result.recovery_events
    ]
    return {
        "schema_version": 2,
        "iteration": iteration,
        "session_type": "agent",
        "timestamp": _now(),
        "parent_iteration": parent_iteration,
        "hypothesis": hypothesis,
        "hypothesis_source": "single_agent",
        "aborted_before_proposal": aborted_before_proposal,
        "target_component": str(proposal.get("target_component", "unclassified")),
        "feature_engineering": {
            "sources": list(proposal.get("feature_sources", [])),
            "transformations": list(proposal.get("feature_transformations", [])),
            "leakage_controls": list(proposal.get("leakage_controls", [])),
        },
        "code_path": str(code_path),
        "code_diff": code_diff,
        "code_diff_reason": (
            None
            if code_diff.strip()
            else "official baseline" if iteration == 0 else "no final source change"
        ),
        "status": status,
        "internal_repairs": max(0, len(result.executions) - 1),
        "metrics": result.metrics,
        "delta_vs_baseline": delta,
        "pct_of_headroom": (delta / headroom * 100) if delta is not None and headroom else None,
        "is_new_best": is_new_best,
        "error": result.error,
        "tokens": result.token_counts,
        "wall_seconds": result.wall_seconds,
        "human_intervention": any(
            bool(event.get("human_intervention"))
            for event in result.recovery_events
        ),
        "manual_intervention_count": sum(
            1 for event in result.recovery_events
            if bool(event.get("human_intervention"))
        ),
        "reasoning": reasoning,
        "reflection": result.reflection,
        "hypothesis_supported": getattr(result, "hypothesis_supported", None),
        "hypothesis_status": getattr(result, "hypothesis_status", ""),
        "implementation_diagnosis": getattr(result, "implementation_diagnosis", ""),
        "suggested_next": getattr(result, "suggested_next", ""),
        "execution_attempts": result.executions,
        "recovery_events": recovery_events,
        # Present only on a new-best row: the winning candidate re-scored on two
        # extra fixed seeds. The convergence trajectory still uses seed 0 only.
        "stability": stability,
    }


def _stability_across_seeds(
    candidate_dir: Path,
    config: Config,
    primary_metrics: dict,
    *,
    trial_config: dict[str, object] | None = None,
    run_deadline: float | None = None,
) -> dict:
    """Re-score one candidate on the extra fixed seeds and summarise the spread.

    The official seed's metrics are passed in (already computed by the iteration);
    the other seeds in ``config.AGENT_STABILITY_SEEDS`` are run here. Each run is a
    full training pass (~13s) but only fires when a candidate becomes the new best.
    """
    seeds = list(config.AGENT_STABILITY_SEEDS)
    per_seed: dict[int, dict | None] = {seeds[0]: dict(primary_metrics)}
    for seed in seeds[1:]:
        timeout_seconds = config.AGENT_NORMAL_EXECUTION_TIMEOUT_S
        if run_deadline is not None:
            remaining = run_deadline - time.time() - config.AGENT_WALL_RESERVE_S
            if remaining < config.AGENT_QUICK_EXECUTION_TIMEOUT_S:
                per_seed[seed] = None
                continue
            timeout_seconds = max(1, min(timeout_seconds, int(remaining)))
        execution = execute_model(
            candidate_dir,
            config,
            timeout_seconds=timeout_seconds,
            seed=seed,
            trial_config=trial_config,
        )
        per_seed[seed] = dict(execution.metrics) if execution.success else None
    primaries = [
        float(metrics["primary"])
        for metrics in per_seed.values()
        if metrics is not None
    ]
    return {
        "seeds": seeds,
        "primary_mean": statistics.fmean(primaries) if primaries else None,
        "primary_std": statistics.pstdev(primaries) if len(primaries) > 1 else 0.0,
        "successful_seed_count": len(primaries),
        "per_seed": {str(seed): metrics for seed, metrics in per_seed.items()},
    }


# Published FM standard deviation across five seeds (starter kit README line 74).
PUBLISHED_SEED_STD = 0.0008


def render_stability_section(stability: dict | None, best_iteration: int) -> str:
    """Markdown block reporting the winner's per-seed spread (empty string if none)."""
    if not stability or best_iteration == 0:
        return (
            "## Multi-seed stability\n\n"
            "No candidate beat the reproduced baseline, so no multi-seed check was run.\n\n"
        )
    rows = []
    for seed, metrics in stability["per_seed"].items():
        if metrics is None:
            rows.append(f"| {seed} | failed | failed | failed |")
        else:
            rows.append(
                f"| {seed} | {float(metrics.get('GAUC', 0.0)):.6f} | "
                f"{float(metrics.get('nDCG@5', 0.0)):.6f} | "
                f"{float(metrics['primary']):.6f} |"
            )
    mean = stability.get("primary_mean")
    std = stability.get("primary_std") or 0.0
    header = (
        "## Multi-seed stability\n\n"
        f"Winning candidate `node_{best_iteration:03d}` re-scored on fixed seeds "
        f"{stability['seeds']} (seed {stability['seeds'][0]} is the sole convergence "
        "observation).\n\n"
        "| Seed | GAUC | nDCG@5 | primary |\n"
        "|---:|---:|---:|---:|\n"
        + "\n".join(rows)
        + "\n\n"
    )
    if mean is None:
        return header + "primary mean unavailable (extra-seed runs failed).\n\n"
    ratio = f"{std / PUBLISHED_SEED_STD:.1f}x" if std else "0.0x"
    return header + (
        f"primary mean **{mean:.6f}** ± std **{std:.6f}** "
        f"({ratio} the published FM 5-seed std of {PUBLISHED_SEED_STD}).\n\n"
    )


def _emit_sweep_rows(
    logger: RunLogger,
    sweep_members: list[dict],
    *,
    iteration: int,
    best_iteration: int,
    base_result: AgentIterationResult,
    member_diff: str,
    trial_path: Path,
    baseline_primary: float,
    headroom: float,
    incumbent_primary: float,
) -> dict:
    """Write one evidence row per swept config and return the accounting deltas.

    Each config is one scored variant: one row, one budget charge. Tokens and
    interventions are counted once (RunLogger accumulates them on every write),
    so members after the first carry zeroed totals.
    """
    for member_index, member in enumerate(sweep_members):
        m_metrics = member.get("metrics") if member.get("success") else None
        m_primary = float(m_metrics["primary"]) if m_metrics else None
        member_result = replace(
            base_result,
            executions=[member],
            metrics=m_metrics,
            success=bool(member.get("success")),
            error=member.get("error"),
            sweep_members=None,
            selected_execution=member if member.get("success") else None,
        )
        row = _log_row(
            iteration, best_iteration, member_result,
            "success" if member.get("success") else "failed",
            m_primary is not None and m_primary > incumbent_primary,
            baseline_primary,
            headroom,
            member_diff,
            Path(str(member.get("source_snapshot_path") or trial_path)),
        )
        row["sweep_id"] = member["sweep_id"]
        row["sweep_member"] = member["sweep_member"]
        if member_index > 0:
            row["tokens"] = {"input": 0, "output": 0}
            row["manual_intervention_count"] = 0
        logger.write(row)

    successful = [m for m in sweep_members if m.get("success")]
    best_member = max(
        successful, key=lambda m: float(m["metrics"]["primary"]), default=None
    )
    return {
        # member 0 was already charged before run_iteration; charge the rest here.
        "attempted_delta": len(sweep_members) - 1,
        "successful": len(successful),
        "failed": len(sweep_members) - len(successful),
        "best_member": best_member,
        "sweep_primary": (
            float(best_member["metrics"]["primary"]) if best_member else None
        ),
    }


def _converged(best_history: list[float], epsilon: float, consecutive: int) -> bool:
    if len(best_history) < consecutive + 1:
        return False
    gains = [best_history[i] - best_history[i - 1] for i in range(1, len(best_history))]
    return all(gain <= epsilon for gain in gains[-consecutive:])


def _frontier_converged(
    history: list[tuple[float, ...]],
    *,
    epsilon: float,
    consecutive: int,
    completed_experiments: int,
    minimum_experiments: int,
) -> bool:
    """Stop only when the complete top frontier has stalled over a full window."""
    if completed_experiments < minimum_experiments or len(history) < consecutive + 1:
        return False
    old = history[-consecutive - 1]
    current = history[-1]
    width = max(len(old), len(current))
    old_pad = old + (-float("inf"),) * (width - len(old))
    current_pad = current + (-float("inf"),) * (width - len(current))
    improvements = [
        now - before
        for before, now in zip(old_pad, current_pad)
        if math.isfinite(before) and math.isfinite(now)
    ]
    return bool(improvements) and max(improvements) <= epsilon


def _frontier_scores(frontier: CandidateFrontier, top_k: int) -> tuple[float, ...]:
    return tuple(
        node.conservative_primary(PUBLISHED_SEED_STD)
        for node in frontier.leaderboard(
            top_k=top_k, conservative=True, seed_std=PUBLISHED_SEED_STD
        )
    )


def _freeze_candidate_bundle(
    workspace: Path,
    node_id: int,
    source: str,
    *,
    parent_id: int | None,
    execution: dict[str, object],
) -> dict[str, object]:
    """Persist the exact scored source/config/seed as an immutable frontier node."""
    bundle_dir = workspace / "frontier" / f"node_{node_id:03d}"
    if bundle_dir.exists():
        raise FileExistsError(f"frontier bundle already exists: {bundle_dir}")
    bundle_dir.mkdir(parents=True)
    model_path = bundle_dir / "model.py"
    model_path.write_text(source, encoding="utf-8")
    source_sha256 = hashlib.sha256(model_path.read_bytes()).hexdigest()
    expected_sha256 = execution.get("source_sha256")
    if expected_sha256 and source_sha256 != expected_sha256:
        raise ValueError(
            f"frozen source hash {source_sha256} != executed source hash {expected_sha256}"
        )
    trial_config = dict(execution.get("trial_config") or {})
    config_path = bundle_dir / "trial_config.json"
    config_path.write_text(
        json.dumps(trial_config, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "schema_version": 1,
        "node_id": node_id,
        "parent_id": parent_id,
        "model_path": str(model_path),
        "source_sha256": source_sha256,
        "trial_config_path": str(config_path),
        "trial_config": trial_config,
        "seed": int(0 if execution.get("seed") is None else execution["seed"]),
        "prediction_sha256": execution.get("prediction_sha256"),
        "execution_id": execution.get("execution_id"),
    }
    manifest_path = bundle_dir / "bundle.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return {**manifest, "manifest_path": str(manifest_path)}


def _inherit_cross_run_champion(
    config: Config,
    workspace: Path,
    frontier: CandidateFrontier,
) -> dict[str, object] | None:
    """Attach the last frozen champion as a verified branch of this run's root."""
    index_path = config.EXPERIMENT_WORKSPACE_DIR / "champions" / "latest.json"
    if not index_path.is_file():
        return None
    try:
        pointer = json.loads(index_path.read_text(encoding="utf-8"))
        champion_dir = Path(str(pointer["champion_dir"])).resolve()
        champions_root = (config.EXPERIMENT_WORKSPACE_DIR / "champions").resolve()
        if not champion_dir.is_relative_to(champions_root):
            raise ValueError("champion path escapes the local champion archive")
        manifest = json.loads(
            (champion_dir / "champion.json").read_text(encoding="utf-8")
        )
        source_path = champion_dir / "model.py"
        source = source_path.read_text(encoding="utf-8")
        source_sha256 = hashlib.sha256(source.encode("utf-8")).hexdigest()
        if source_sha256 != manifest["source_sha256"]:
            raise ValueError("champion source hash mismatch")
        root = frontier.get_node(0)
        if (
            source_sha256 == root.source_sha256
            and dict(manifest.get("trial_config") or {}) == root.trial_config
            and int(manifest.get("seed", config.SEED)) == root.seed
        ):
            return None
        node = frontier.add_child(
            0,
            str(source_path),
            f"Inherited champion from run {manifest.get('source_run_id', 'unknown')}",
            target_component=str(manifest.get("target_component", "cross_run_champion")),
        )
        bundle = _freeze_candidate_bundle(
            workspace,
            node.id,
            source,
            parent_id=0,
            execution={
                "source_sha256": source_sha256,
                "trial_config": dict(manifest.get("trial_config") or {}),
                "seed": int(manifest.get("seed", config.SEED)),
                "prediction_sha256": manifest.get("prediction_sha256"),
                "execution_id": manifest.get("execution_id"),
            },
        )
        frontier.freeze_result(
            node.id,
            primary=float(manifest["metrics"]["primary"]),
            status="success",
            code_path=str(bundle["model_path"]),
            hypothesis=str(manifest.get("hypothesis") or node.hypothesis),
            target_component=str(manifest.get("target_component", "cross_run_champion")),
            source_sha256=source_sha256,
            trial_config=dict(manifest.get("trial_config") or {}),
            seed=int(manifest.get("seed", config.SEED)),
            metrics=dict(manifest["metrics"]),
            stability=dict(manifest.get("stability") or {}) or None,
        )
        return {
            "source_run_id": manifest.get("source_run_id"),
            "node_id": node.id,
            "source_sha256": source_sha256,
            "champion_dir": str(champion_dir),
        }
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        console.harness("Cross-run champion ignored", error=str(exc))
        return None


def _archive_cross_run_champion(
    config: Config,
    run_id: str,
    node: CandidateNode,
) -> Path:
    """Write a versioned immutable champion and update a small latest pointer."""
    champion_dir = config.EXPERIMENT_WORKSPACE_DIR / "champions" / run_id
    champion_dir.mkdir(parents=True, exist_ok=False)
    source_path = champion_dir / "model.py"
    shutil.copy2(node.code_path, source_path)
    source_sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()
    if source_sha256 != node.source_sha256:
        raise ValueError("refusing to archive a champion whose source hash changed")
    (champion_dir / "trial_config.json").write_text(
        json.dumps(node.trial_config, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    champion_manifest = {
        "schema_version": 1,
        "source_run_id": run_id,
        "source_node_id": node.id,
        "source_sha256": source_sha256,
        "hypothesis": node.hypothesis,
        "target_component": node.target_component,
        "trial_config": node.trial_config,
        "seed": node.seed,
        "metrics": node.metrics,
        "stability": node.stability,
    }
    (champion_dir / "champion.json").write_text(
        json.dumps(champion_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    latest_path = champion_dir.parent / "latest.json"
    latest_path.write_text(
        json.dumps({"champion_dir": str(champion_dir)}, indent=2) + "\n",
        encoding="utf-8",
    )
    return champion_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Configurable KuaiRand MLE research loop"
    )
    parser.add_argument("--max-iter", type=int, default=3)
    parser.add_argument("--wall-hours", type=float, default=0.5)
    parser.add_argument(
        "--agent-turns", type=int, default=Config.AGENT_MAX_TURNS,
        help=(
            "Positive model-call cap per experiment "
            f"(default: {Config.AGENT_MAX_TURNS})."
        ),
    )
    parser.add_argument(
        "--bootstrap-turns", type=int, default=Config.AGENT_BOOTSTRAP_MAX_TURNS,
        help=(
            "Positive bootstrap model-call cap "
            f"(default: {Config.AGENT_BOOTSTRAP_MAX_TURNS})."
        ),
    )
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument(
        "--gpu-hours",
        type=float,
        default=0.0,
        help="Measured GPU-hours for this run; leave at 0 for CPU-only runs.",
    )
    parser.add_argument(
        "--task-definition-confirmed",
        action="store_true",
        help="Record that the checked-in Starter Kit label and metrics were confirmed.",
    )
    args = parser.parse_args()
    if args.agent_turns <= 0:
        parser.error("--agent-turns must be positive; unlimited agent loops are disabled")
    if args.bootstrap_turns <= 0:
        parser.error(
            "--bootstrap-turns must be positive; unlimited bootstrap loops are disabled"
        )
    started = time.time()
    run_deadline = started + args.wall_hours * 3600

    config = load_config()
    if args.data_dir:
        config.DATA_DIR = Path(args.data_dir).resolve()
    source_data_dir = config.DATA_DIR
    langchain_settings = resolve_langchain_settings()
    provider_label = (
        f"langchain / {langchain_settings['provider']} / {langchain_settings['model']}"
    )
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    workspace = config.EXPERIMENT_WORKSPACE_DIR / run_id
    workspace.mkdir(parents=True, exist_ok=True)
    config.RUN_RESEARCH_DIR = workspace / "research"
    config.RUN_RESEARCH_DIR.mkdir(parents=True, exist_ok=True)
    logger = RunLogger(config.ARTIFACTS_DIR, run_id, strict_validation=True)
    initial_environment = create_run_environment(
        config, workspace, logger.run_dir
    )
    if not initial_environment.get("success"):
        logger.close()
        raise RuntimeError(
            "could not snapshot the dedicated run environment: "
            + str(initial_environment.get("error", "unknown error"))
        )
    console.harness(
        "Run Python environment",
        status="Created dedicated install target for this run",
        python=config.PYTHON_EXE,
        binary_only="true",
        lock=initial_environment.get("requirements_lock"),
    )

    console.harness(
        "Data preparation",
        status="Preparing count-verified train/validation-only data view",
    )
    split_manifest = prepare_train_valid_view(source_data_dir, workspace / "candidate_data")
    config.DATA_DIR = workspace / "candidate_data"
    experiment_calls = args.agent_turns
    bootstrap_calls = args.bootstrap_turns

    console.harness(
        "Run setup",
        run_id=run_id,
        provider=provider_label,
        experiment_budget=(
            f"{args.max_iter} experiments, {args.wall_hours}h, "
            f"{experiment_calls} model calls/experiment"
        ),
        bootstrap_budget=f"{bootstrap_calls} model calls (separate from experiments)",
        llm_limits=(
            f"Model output={config.AGENT_MAX_OUTPUT_TOKENS} tokens/call; "
            f"read page={config.AGENT_READ_MAX_CHARS} chars"
        ),
        candidate_rows=(
            f"train={split_manifest['emitted_candidate_counts']['train']:,}; "
            f"validation={split_manifest['emitted_candidate_counts']['valid']:,}; test=0"
        ),
        llm_trace=logger.llm_events_path,
    )

    root_dir = workspace / "trial_000"
    root_dir.mkdir(parents=True, exist_ok=True)
    root_path = root_dir / "model.py"
    root_code = make_root_model_py(config)
    assert_organizer_fm_equivalence(config, root_code)
    root_path.write_text(root_code, encoding="utf-8")
    agent = ResearchAgent(
        config,
        client=make_langchain_client(),
        event_writer=logger.write_llm_event,
        provider_label=provider_label,
        blob_dir=logger.logs_dir / "blobs",
        run_id=run_id,
        # A real scored run is the only thing allowed to write cross-run memory.
        history_path=HISTORY_PATH,
        run_deadline=run_deadline,
    )
    console.harness(
        "Agent bootstrap",
        status=(
            "Agent will discover the task, inspect data, consult literature, and reproduce "
            "the official FM baseline before experiment 1"
        ),
    )
    bootstrap_result = agent.run_bootstrap(root_dir, args.bootstrap_turns)
    if not bootstrap_result.success or not bootstrap_result.metrics:
        failed_bootstrap = AgentIterationResult(
            success=False,
            hypothesis="Reproduce the official FM baseline after understanding the benchmark",
            reasoning=(
                "The autonomous bootstrap attempted task discovery, constrained EDA, "
                "literature retrieval, and official baseline reproduction."
            ),
            reflection="",
            metrics=None,
            executions=[],
            recovery_events=bootstrap_result.recovery_events,
            token_counts=bootstrap_result.token_counts,
            wall_seconds=bootstrap_result.wall_seconds,
            error=bootstrap_result.error or "bootstrap did not return baseline metrics",
            final_code=root_code,
        )
        logger.write(_log_row(
            0, None, failed_bootstrap, "failed", False, config.BASELINE_PRIMARY,
            config.HEADROOM, "", root_path,
        ))
        totals = logger.running_totals()
        final_environment = snapshot_run_environment(
            config, phase="bootstrap_failed"
        )
        failed_results = {
            "run_id": run_id,
            "architecture": "immutable_candidate_frontier",
            "provider": provider_label,
            "task_definition_confirmed": args.task_definition_confirmed,
            "stop_reason": "bootstrap_failed",
            "converged": False,
            "error": failed_bootstrap.error,
            "run_config": {
                "max_iterations": args.max_iter,
                "wall_hours": args.wall_hours,
                "bootstrap_model_calls": args.bootstrap_turns,
                "experiment_model_calls": args.agent_turns,
                "max_quota_resumes_per_invocation": config.AGENT_MAX_QUOTA_RESUMES,
                "max_quota_wait_seconds": config.AGENT_MAX_QUOTA_WAIT_S,
                "frontier_branch_trials": config.AGENT_FRONTIER_BRANCH_TRIALS,
                "frontier_top_k": config.AGENT_FRONTIER_TOP_K,
                "minimum_experiments_before_convergence": (
                    config.AGENT_MIN_EXPERIMENTS_BEFORE_CONVERGENCE
                ),
            },
            "attempted_agent_experiments": 0,
            "successful_agent_experiments": 0,
            "failed_agent_experiments": 0,
            "tokens": totals["tokens"],
            "gpu_hours": args.gpu_hours,
            "manual_interventions": totals["interventions"],
            "total_wall_seconds": time.time() - started,
            "llm_trace": totals["llm_trace"],
            "prompt_templates": agent.prompt_evidence,
            "task_context_bootstrap": agent.bootstrap_evidence,
            "data_view_manifest": split_manifest,
            "python_environment": final_environment,
        }
        logger.write_results(failed_results)
        logger.write_report(f"""# Single-agent research run {run_id}

## Outcome

- Stop reason: `bootstrap_failed`
- Converged: `false`
- Error: {failed_bootstrap.error}
- Input/output tokens: {totals['tokens']['input']} / {totals['tokens']['output']}
- Manual interventions: {totals['interventions']}
- Python environment lock: `{final_environment.get('requirements_lock', 'unavailable')}`

The candidate loop did not start because the autonomous bootstrap failed. See
`logs/events.jsonl` and `logs/llm_events.jsonl` for the failure and recovery evidence.
""")
        console.harness(
            "Agent bootstrap failed",
            error=bootstrap_result.error,
            action=(
                "Review logs/llm_events.jsonl. Provider quota declines and non-quota "
                "provider failures preserve the trace for diagnosis."
            ),
        )
        logger.close()
        raise SystemExit(1)
    root_primary = float(bootstrap_result.metrics["primary"])
    official_baseline_reference = dict(
        (agent.bootstrap_state.baseline_execution or {}).get("organizer_reference") or {}
    )
    console.harness(
        "Baseline accepted",
        validation_primary=f"{root_primary:.6f}",
        official_primary=f"{config.BASELINE_PRIMARY:.6f}",
        status="Agent reproduced the official baseline; research experiments may begin",
    )

    root_result = AgentIterationResult(
        success=True,
        hypothesis="Reproduce the official FM baseline after understanding the benchmark",
        reasoning=(
            "The agent discovered and read the task sources, inspected the allowed data, "
            "and invoked the constrained baseline reproduction tool before proposing changes."
        ),
        reflection="The official baseline was reproduced and retained as experiment context.",
        metrics=bootstrap_result.metrics,
        executions=[],
        recovery_events=bootstrap_result.recovery_events,
        token_counts=bootstrap_result.token_counts,
        wall_seconds=bootstrap_result.wall_seconds,
        error=None,
        final_code=root_code,
    )
    logger.write(_log_row(
        0, None, root_result, "success", True, config.BASELINE_PRIMARY,
        config.HEADROOM, "", root_path,
    ))

    # Seed the immutable frontier with the exact baseline execution bundle.  The
    # extra fixed seeds make the root comparable to later conservative scores.
    root_execution = dict(agent.bootstrap_state.baseline_execution or {})
    root_execution.setdefault("seed", config.SEED)
    root_execution.setdefault("trial_config", {})
    root_execution.setdefault(
        "source_sha256", hashlib.sha256(root_code.encode("utf-8")).hexdigest()
    )
    frontier = CandidateFrontier(
        str(root_path), "Reproduce the official FM baseline"
    )
    root_bundle = _freeze_candidate_bundle(
        workspace, 0, root_code,
        parent_id=None,
        execution=root_execution,
    )
    root_bundle_dir = Path(str(root_bundle["model_path"])).parent
    root_stability = _stability_across_seeds(
        root_bundle_dir, config, dict(bootstrap_result.metrics), trial_config={},
        run_deadline=run_deadline,
    )
    frontier.freeze_result(
        0,
        primary=root_primary,
        status="success",
        code_path=str(root_bundle["model_path"]),
        hypothesis="Reproduce the official FM baseline",
        target_component="baseline",
        source_sha256=str(root_bundle["source_sha256"]),
        trial_config={},
        seed=config.SEED,
        metrics=dict(bootstrap_result.metrics),
        stability=root_stability,
    )
    inherited_champion = _inherit_cross_run_champion(config, workspace, frontier)
    if inherited_champion:
        console.harness(
            "Cross-run champion inherited",
            source_run=inherited_champion["source_run_id"],
            frontier_node=inherited_champion["node_id"],
        )
    best_node = frontier.best_node(seed_std=PUBLISHED_SEED_STD)
    assert best_node is not None
    best_primary = best_node.primary
    best_metrics = dict(best_node.metrics)
    best_iteration = best_node.id
    best_path = Path(best_node.code_path)
    best_stability: dict | None = best_node.stability
    best_history = [best_primary]
    frontier_history = [_frontier_scores(frontier, config.AGENT_FRONTIER_TOP_K)]
    successful_agent_experiments = 0
    attempted_agent_experiments = 0
    failed_agent_experiments = 0
    aborted_agent_iterations = 0
    stop_reason = "max_iterations"
    trajectory: list[dict[str, object]] = [{
        "iteration": 0,
        "status": "success",
        "primary": root_primary,
        "incumbent_primary": root_primary,
        "is_new_best": True,
    }]

    for iteration in range(1, args.max_iter + 1):
        if attempted_agent_experiments >= args.max_iter:
            stop_reason = "max_iterations"
            break
        if time.time() - started >= args.wall_hours * 3600:
            stop_reason = "wall_clock"
            console.harness("Run stopped", reason="Wall-clock budget reached")
            break
        if _frontier_converged(
            frontier_history,
            epsilon=config.CONVERGENCE_EPSILON,
            consecutive=config.CONVERGENCE_N,
            completed_experiments=attempted_agent_experiments,
            minimum_experiments=config.AGENT_MIN_EXPERIMENTS_BEFORE_CONVERGENCE,
        ):
            stop_reason = "converged"
            console.harness("Run stopped", reason="Frontier convergence rule reached")
            break

        # The initial breadth phase deliberately branches independent components
        # from the baseline. Afterwards exploit/explore among the strongest few
        # frozen nodes rather than extending one irreversible greedy chain.
        parent_node = (
            frontier.get_node(0)
            if iteration <= config.AGENT_FRONTIER_BRANCH_TRIALS
            else frontier.select_parent(
                top_k=config.AGENT_FRONTIER_TOP_K,
                exploration_c=config.UCB_C,
                exploration_scale=config.CONVERGENCE_EPSILON,
                seed_std=PUBLISHED_SEED_STD,
            )
        )
        trial_dir = workspace / f"trial_{iteration:03d}"
        trial_dir.mkdir(parents=True, exist_ok=True)
        trial_path = trial_dir / "model.py"
        shutil.copy2(parent_node.code_path, trial_path)
        parent_code = trial_path.read_text(encoding="utf-8")
        pending_node = frontier.add_child(
            parent_node.id,
            str(trial_path),
            f"Experiment {iteration} pending",
        )
        console.harness(
            "Experiment start",
            experiment=f"{iteration}/{args.max_iter}",
            parent_node=parent_node.id,
            inherited_primary=f"{parent_node.primary:.6f}",
            frontier_best=f"{best_primary:.6f}",
        )
        experiments_remaining = args.max_iter - attempted_agent_experiments
        required_target_component = (
            config.AGENT_FRONTIER_BRANCH_COMPONENTS[iteration - 1]
            if iteration <= min(
                config.AGENT_FRONTIER_BRANCH_TRIALS,
                len(config.AGENT_FRONTIER_BRANCH_COMPONENTS),
            )
            else None
        )
        result = agent.run_iteration(
            iteration, trial_dir, parent_node.primary, best_primary, args.agent_turns,
            experiments_remaining=experiments_remaining,
            required_target_component=required_target_component,
        )
        actual_executions = [
            execution for execution in result.executions
            if execution.get("execution_attempted")
        ]
        attempted_agent_experiments += len(actual_executions)
        successful_agent_experiments += sum(
            1 for execution in actual_executions if execution.get("success")
        )
        failed_agent_experiments += sum(
            1 for execution in actual_executions if not execution.get("success")
        )
        if not actual_executions:
            aborted_agent_iterations += 1
        final_code = result.final_code or parent_code
        member_diff = _diff(
            parent_code, final_code,
            f"node_{parent_node.id:03d}/model.py", f"node_{pending_node.id:03d}/model.py",
        )

        sweep_members = getattr(result, "sweep_members", None)
        sweep_outcome = None
        if sweep_members:
            # One agent turn ran a bounded sweep: one evidence row and one
            # scored-variant charge per config, but exactly one best_history
            # append so the convergence rule still counts turns, not configs.
            sweep_outcome = _emit_sweep_rows(
                logger, sweep_members,
                iteration=iteration, best_iteration=parent_node.id,
                base_result=result, member_diff=member_diff, trial_path=trial_path,
                baseline_primary=config.BASELINE_PRIMARY, headroom=config.HEADROOM,
                incumbent_primary=best_primary,
            )
            represented_member = result.selected_execution or sweep_members[-1]
            represented_index = represented_member.get("sweep_member")
            for member in sweep_members:
                if member.get("sweep_member") == represented_index:
                    continue
                sibling = frontier.add_child(
                    parent_node.id,
                    str(trial_path),
                    str(member.get("hypothesis") or result.hypothesis),
                    target_component=str(
                        (member.get("proposal") or {}).get(
                            "target_component", "unclassified"
                        )
                    ),
                )
                member_snapshot = (
                    Path(str(member["source_snapshot_path"]))
                    if member.get("source_snapshot_path") else None
                )
                member_source = (
                    member_snapshot.read_text(encoding="utf-8")
                    if member_snapshot is not None and member_snapshot.is_file()
                    else final_code
                )
                member_execution = dict(member)
                if not member_execution.get("source_sha256"):
                    member_execution["source_sha256"] = hashlib.sha256(
                        member_source.encode("utf-8")
                    ).hexdigest()
                member_bundle = _freeze_candidate_bundle(
                    workspace,
                    sibling.id,
                    member_source,
                    parent_id=parent_node.id,
                    execution=member_execution,
                )
                member_metrics = (
                    dict(member["metrics"])
                    if member.get("success") and member.get("metrics") else None
                )
                frontier.freeze_result(
                    sibling.id,
                    primary=(
                        float(member_metrics["primary"])
                        if member_metrics is not None else -1.0
                    ),
                    status="success" if member_metrics is not None else "failed",
                    code_path=str(member_bundle["model_path"]),
                    hypothesis=str(member.get("hypothesis") or result.hypothesis),
                    target_component=str(
                        (member.get("proposal") or {}).get(
                            "target_component", "unclassified"
                        )
                    ),
                    source_sha256=str(member_bundle["source_sha256"]),
                    trial_config=dict(member.get("trial_config") or {}),
                    seed=int(
                        config.SEED if member.get("seed") is None else member["seed"]
                    ),
                    metrics=member_metrics,
                    stability=None,
                )

        represented_execution = (
            result.selected_execution
            or (actual_executions[-1] if actual_executions else None)
        )
        for execution_record in actual_executions:
            if execution_record is represented_execution or execution_record.get("sweep_id"):
                continue
            auxiliary = frontier.add_child(
                parent_node.id,
                str(trial_path),
                str(execution_record.get("hypothesis") or result.hypothesis),
                target_component=str(
                    (execution_record.get("proposal") or {}).get(
                        "target_component", "unclassified"
                    )
                ),
            )
            auxiliary_snapshot = (
                Path(str(execution_record["source_snapshot_path"]))
                if execution_record.get("source_snapshot_path") else trial_path
            )
            auxiliary_source = auxiliary_snapshot.read_text(encoding="utf-8")
            auxiliary_execution = dict(execution_record)
            if not auxiliary_execution.get("source_sha256"):
                auxiliary_execution["source_sha256"] = hashlib.sha256(
                    auxiliary_source.encode("utf-8")
                ).hexdigest()
            auxiliary_bundle = _freeze_candidate_bundle(
                workspace,
                auxiliary.id,
                auxiliary_source,
                parent_id=parent_node.id,
                execution=auxiliary_execution,
            )
            auxiliary_metrics = (
                dict(execution_record["metrics"])
                if execution_record.get("success") and execution_record.get("metrics")
                else None
            )
            frontier.freeze_result(
                auxiliary.id,
                primary=(
                    float(auxiliary_metrics["primary"])
                    if auxiliary_metrics is not None else -1.0
                ),
                status="success" if auxiliary_metrics is not None else "failed",
                code_path=str(auxiliary_bundle["model_path"]),
                hypothesis=str(execution_record.get("hypothesis") or result.hypothesis),
                target_component=str(
                    (execution_record.get("proposal") or {}).get(
                        "target_component", "unclassified"
                    )
                ),
                source_sha256=str(auxiliary_bundle["source_sha256"]),
                trial_config=dict(execution_record.get("trial_config") or {}),
                seed=int(
                    config.SEED if execution_record.get("seed") is None
                    else execution_record["seed"]
                ),
                metrics=auxiliary_metrics,
                stability=None,
            )

        primary = float(result.metrics["primary"]) if result.metrics else None
        previous_raw_best = max(
            node.primary for node in frontier.leaderboard(top_k=max(1, len(frontier)))
        )
        is_new_best = primary is not None and primary > previous_raw_best
        status = "success" if result.success else "failed"
        stability = None
        frozen_path = trial_path
        if result.success:
            selected = dict(result.selected_execution or {})
            if not selected:
                raise RuntimeError("successful iteration has no selected immutable execution")
            bundle = _freeze_candidate_bundle(
                workspace,
                pending_node.id,
                final_code,
                parent_id=parent_node.id,
                execution=selected,
            )
            frozen_path = Path(str(bundle["model_path"]))
            current_best = frontier.best_node(seed_std=PUBLISHED_SEED_STD)
            should_stabilize = (
                current_best is None
                or primary >= current_best.conservative_primary(PUBLISHED_SEED_STD)
                - config.CONVERGENCE_EPSILON
            )
            if should_stabilize:
                stability = _stability_across_seeds(
                    frozen_path.parent,
                    config,
                    dict(result.metrics or {}),
                    trial_config=dict(selected.get("trial_config") or {}),
                    run_deadline=run_deadline,
                )
            proposal = selected.get("proposal") or {}
            frontier.freeze_result(
                pending_node.id,
                primary=float(primary),
                status="success",
                code_path=str(frozen_path),
                hypothesis=result.hypothesis,
                target_component=str(proposal.get("target_component", "unclassified")),
                source_sha256=str(bundle["source_sha256"]),
                trial_config=dict(selected.get("trial_config") or {}),
                seed=int(selected.get("seed") or config.SEED),
                metrics=dict(result.metrics or {}),
                stability=stability,
            )
        else:
            failed_execution = dict(
                represented_execution
                or (result.executions[-1] if result.executions else {})
            )
            failed_snapshot = (
                Path(str(failed_execution["source_snapshot_path"]))
                if failed_execution.get("source_snapshot_path") else None
            )
            failed_source = (
                failed_snapshot.read_text(encoding="utf-8")
                if failed_snapshot is not None and failed_snapshot.is_file()
                else trial_path.read_text(encoding="utf-8")
            )
            failed_execution.setdefault(
                "source_sha256",
                hashlib.sha256(failed_source.encode("utf-8")).hexdigest(),
            )
            failed_execution.setdefault("trial_config", {})
            failed_execution.setdefault("seed", config.SEED)
            failed_bundle = _freeze_candidate_bundle(
                workspace,
                pending_node.id,
                failed_source,
                parent_id=parent_node.id,
                execution=failed_execution,
            )
            frozen_path = Path(str(failed_bundle["model_path"]))
            frontier.freeze_result(
                pending_node.id,
                primary=-1.0,
                status="failed",
                code_path=str(frozen_path),
                hypothesis=result.hypothesis,
                target_component="unclassified",
                source_sha256=str(failed_bundle["source_sha256"]),
                trial_config=dict(failed_execution.get("trial_config") or {}),
                seed=int(
                    config.SEED if failed_execution.get("seed") is None
                    else failed_execution["seed"]
                ),
                metrics=None,
                stability=None,
            )

        if not sweep_members:
            logger.write(_log_row(
                iteration,
                parent_node.id,
                result,
                status,
                is_new_best,
                config.BASELINE_PRIMARY,
                config.HEADROOM,
                member_diff,
                frozen_path,
                stability,
            ))

        if result.success:
            best_node = frontier.best_node(seed_std=PUBLISHED_SEED_STD)
            assert best_node is not None
            best_primary = best_node.primary
            best_metrics = dict(best_node.metrics)
            best_iteration = best_node.id
            best_path = Path(best_node.code_path)
            best_stability = best_node.stability
            best_history.append(best_primary)
            console.harness(
                "Experiment result",
                status=("New raw best" if is_new_best else "Frontier candidate scored"),
                validation_primary=f"{primary:.6f}",
                conservative_frontier_best=f"{best_node.conservative_primary():.6f}",
                selected_node=best_node.id,
            )
        else:
            console.harness("Experiment failed", error=result.error)
            user_declined = any(
                event.get("action") == "user_declined_resume"
                for event in result.recovery_events
            )
            quota_resume_exhausted = any(
                event.get("action") == "quota_resume_limit_exhausted"
                for event in result.recovery_events
            )
            cost_limit = any(
                event.get("action") == "cost_limit_exhausted"
                for event in result.recovery_events
            )
            if user_declined or quota_resume_exhausted or cost_limit:
                trajectory.append({
                    "iteration": iteration,
                    "status": status,
                    "primary": primary,
                    "incumbent_primary": best_primary,
                    "is_new_best": is_new_best,
                })
                console.harness(
                    "Run stopped",
                    reason=(
                        "User declined automatic provider-quota recovery"
                        if user_declined
                        else "Provider remained unavailable after bounded quota recovery"
                    ),
                )
                stop_reason = (
                    "user_stopped" if user_declined
                    else "cost_limit" if cost_limit
                    else "provider_unavailable"
                )
                break

        frontier_history.append(_frontier_scores(frontier, config.AGENT_FRONTIER_TOP_K))
        trajectory.append({
            "iteration": iteration,
            "node_id": pending_node.id,
            "parent_node_id": parent_node.id,
            "status": status,
            "primary": primary,
            "incumbent_primary": best_primary,
            "is_new_best": is_new_best,
        })

    converged = _frontier_converged(
        frontier_history,
        epsilon=config.CONVERGENCE_EPSILON,
        consecutive=config.CONVERGENCE_N,
        completed_experiments=attempted_agent_experiments,
        minimum_experiments=config.AGENT_MIN_EXPERIMENTS_BEFORE_CONVERGENCE,
    )
    if converged and stop_reason == "max_iterations":
        stop_reason = "converged"
    totals = logger.running_totals()
    final_environment = snapshot_run_environment(config, phase="run_complete")
    total_wall_seconds = time.time() - started
    baseline_metrics = {
        "GAUC": config.BASELINE_GAUC,
        "nDCG@5": config.BASELINE_NDCG,
        "primary": config.BASELINE_PRIMARY,
    }
    metric_deltas = {
        name: float(best_metrics[name]) - float(value)
        for name, value in baseline_metrics.items()
        if name in best_metrics
    }
    champion_archive = _archive_cross_run_champion(config, run_id, best_node)
    results = {
        "run_id": run_id,
        "architecture": "immutable_candidate_frontier",
        "provider": provider_label,
        "task_definition_confirmed": args.task_definition_confirmed,
        "stop_reason": stop_reason,
        "converged": converged,
        "run_config": {
            "max_iterations": args.max_iter,
            "wall_hours": args.wall_hours,
            "bootstrap_model_calls": args.bootstrap_turns,
            "experiment_model_calls": args.agent_turns,
            "max_quota_resumes_per_invocation": config.AGENT_MAX_QUOTA_RESUMES,
            "max_quota_wait_seconds": config.AGENT_MAX_QUOTA_WAIT_S,
            "frontier_branch_trials": config.AGENT_FRONTIER_BRANCH_TRIALS,
            "frontier_top_k": config.AGENT_FRONTIER_TOP_K,
            "minimum_experiments_before_convergence": (
                config.AGENT_MIN_EXPERIMENTS_BEFORE_CONVERGENCE
            ),
        },
        "convergence": {
            "epsilon": config.CONVERGENCE_EPSILON,
            "patience": config.CONVERGENCE_N,
            "incumbent_history": best_history,
            "frontier_history": [list(scores) for scores in frontier_history],
        },
        "baseline_valid_metrics": baseline_metrics,
        "official_baseline_reference": official_baseline_reference,
        "official_hidden_test_baseline_metrics": {
            "GAUC": config.BASELINE_TEST_GAUC,
            "nDCG@5": config.BASELINE_TEST_NDCG,
            "primary": config.BASELINE_TEST_PRIMARY,
        },
        "baseline_valid_primary": config.BASELINE_PRIMARY,
        "reproduced_baseline_valid_primary": root_primary,
        "best_trial": best_iteration,
        "best_node_id": best_node.id,
        "best_valid_metrics": best_metrics,
        "best_valid_primary": best_primary,
        "best_conservative_primary": best_node.conservative_primary(PUBLISHED_SEED_STD),
        "best_trial_stability": best_stability,
        "best_trial_config": best_node.trial_config,
        "best_seed": best_node.seed,
        "best_source_sha256": best_node.source_sha256,
        "best_bundle_manifest": str(best_path.parent / "bundle.json"),
        "stability_seeds": list(config.AGENT_STABILITY_SEEDS),
        "metric_deltas_vs_baseline": metric_deltas,
        "delta_vs_baseline": best_primary - config.BASELINE_PRIMARY,
        "validation_beats_official_baseline": best_primary > root_primary,
        "hidden_test_comparison_status": (
            "unmeasured; only the organizer can determine whether the final submission "
            "beats published hidden-test primary 0.5946"
        ),
        "attempted_agent_experiments": attempted_agent_experiments,
        "successful_agent_experiments": successful_agent_experiments,
        "failed_agent_experiments": failed_agent_experiments,
        "aborted_agent_iterations": aborted_agent_iterations,
        "trajectory": trajectory,
        "tokens": totals["tokens"],
        "gpu_hours": args.gpu_hours,
        "total_wall_seconds": total_wall_seconds,
        "manual_interventions": totals["interventions"],
        "llm_trace": totals["llm_trace"],
        "best_workspace_code_path": str(best_path),
        "candidate_frontier": frontier.to_dict(),
        "inherited_champion": inherited_champion,
        "champion_archive": str(champion_archive),
        "provider_cost": agent.provider_cost,
        "prompt_templates": agent.prompt_evidence,
        "task_context_bootstrap": agent.bootstrap_evidence,
        "data_view_manifest": split_manifest,
        "python_environment": final_environment,
    }
    results_path = logger.write_results(results)
    trajectory_table = "\n".join(
        "| {iteration} | {status} | {primary} | {incumbent} | {new_best} |".format(
            iteration=item["iteration"],
            status=item["status"],
            primary=(
                "—" if item["primary"] is None else f"{float(item['primary']):.6f}"
            ),
            incumbent=f"{float(item['incumbent_primary']):.6f}",
            new_best="yes" if item["is_new_best"] else "no",
        )
        for item in trajectory
    )
    stability_section = render_stability_section(best_stability, best_iteration)
    report_path = logger.write_report(f"""# Single-agent research run {run_id}

## Outcome

- Stop reason: {stop_reason}
- Converged: {str(converged).lower()}
- Starter Kit task definition confirmed: {str(args.task_definition_confirmed).lower()}
- Reproduced FM baseline validation primary: {root_primary:.6f}
- Best validation primary: {best_primary:.6f}
- Best conservative primary: {best_node.conservative_primary(PUBLISHED_SEED_STD):.6f}
- Delta versus published validation baseline: {best_primary - config.BASELINE_PRIMARY:+.6f}
- Published hidden-test baseline primary: {config.BASELINE_TEST_PRIMARY:.6f}
- Hidden-test win status: unmeasured (no hidden labels are exposed to the agent)
- Best frozen frontier node: node_{best_iteration:03d}
- Attempted autonomous experiments: {attempted_agent_experiments}
- Successful autonomous experiments: {successful_agent_experiments}
- Failed autonomous experiments: {failed_agent_experiments}

## Validation metrics

| Metric | Official baseline | Best validation | Delta |
|---|---:|---:|---:|
| GAUC | {config.BASELINE_GAUC:.6f} | {float(best_metrics.get('GAUC', 0.0)):.6f} | {metric_deltas.get('GAUC', 0.0):+.6f} |
| nDCG@5 | {config.BASELINE_NDCG:.6f} | {float(best_metrics.get('nDCG@5', 0.0)):.6f} | {metric_deltas.get('nDCG@5', 0.0):+.6f} |
| primary | {config.BASELINE_PRIMARY:.6f} | {best_primary:.6f} | {best_primary - config.BASELINE_PRIMARY:+.6f} |

{stability_section}
## Experiment trajectory

| Iteration | Status | Candidate primary | Incumbent primary | New best |
|---:|---|---:|---:|---|
{trajectory_table}

## Resource usage

- Input tokens: {totals['tokens']['input']}
- Output tokens: {totals['tokens']['output']}
- GPU-hours: {args.gpu_hours:.6f}
- Total wall seconds: {total_wall_seconds:.2f}
- Manual interventions: {totals['interventions']}
- LLM responses: {totals['llm_trace']['llm_response']}
- Tool results: {totals['llm_trace']['tool_result']}
- Provider errors: {totals['llm_trace']['provider_error']}
- Quota pauses: {totals['llm_trace']['quota_pause']}
- Maximum automatic quota resumes per invocation: {config.AGENT_MAX_QUOTA_RESUMES}
- Maximum wait per quota pause: {config.AGENT_MAX_QUOTA_WAIT_S:.0f}s
- Detailed LLM trace: `{totals['llm_trace']['path']}`
- Dedicated run Python: `{final_environment.get('python_executable', 'unavailable')}`
- Resolved dependency lock: `{final_environment.get('requirements_lock', 'unavailable')}`
- Resolved distributions: {final_environment.get('resolved_distribution_count', 0)}

## Prompt templates

```json
{json.dumps(agent.prompt_evidence, indent=2)}
```

## Retained task context bootstrap

```json
{json.dumps(agent.bootstrap_evidence, indent=2)}
```

## Architecture

The deterministic Python loop owns an immutable candidate tree. Early experiments branch
from the baseline, later experiments choose among the conservative top frontier with a
noise-scaled UCB score, and rewards/visits propagate through each candidate's lineage.
Every scored node freezes the exact executed source, trial configuration, seed, metrics,
and parent. Final selection is the strongest conservative frozen node, never the latest
working file. LangChain supplies the model adapter and structured tool calls; the retained
Python harness enforces budgets, validation-only execution, baseline verification, and
evidence logging.
""")
    logger.close()
    console.harness(
        "Run artifacts",
        results=results_path,
        report=report_path,
        llm_trace=logger.llm_events_path,
    )


if __name__ == "__main__":
    main()
