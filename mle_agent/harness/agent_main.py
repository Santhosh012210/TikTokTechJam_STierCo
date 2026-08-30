"""CLI for the Google ADK-backed autonomous MLE loop.

Usage:
    python -m mle_agent.harness.agent_main --max-iter 3 --wall-hours 0.5 \
        --bootstrap-turns 0 --agent-turns 0
"""
from __future__ import annotations

import argparse
import difflib
import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

from mle_agent.harness.adk_config import configure_google_adk_environment
from mle_agent.harness.config import load_config
from mle_agent.harness.console import console
from mle_agent.harness.data_view import prepare_train_valid_view
from mle_agent.harness.logger import RunLogger
from mle_agent.harness.root_model import make_root_model_py
from mle_agent.research_agent.adk_agent import AgentIterationResult, ResearchAgent


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
) -> dict:
    primary = result.metrics.get("primary") if result.metrics else None
    delta = primary - baseline if primary is not None else None
    chosen_attempt = next(
        (attempt for attempt in reversed(result.executions) if attempt.get("success")),
        result.executions[-1] if result.executions else {},
    )
    proposal = chosen_attempt.get("proposal", {})
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
        "hypothesis": result.hypothesis,
        "hypothesis_source": "single_agent",
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
        "reasoning": result.reasoning,
        "reflection": result.reflection,
        "hypothesis_supported": getattr(result, "hypothesis_supported", None),
        "suggested_next": getattr(result, "suggested_next", ""),
        "execution_attempts": result.executions,
        "recovery_events": recovery_events,
    }


def _converged(best_history: list[float], epsilon: float, consecutive: int) -> bool:
    if len(best_history) < consecutive + 1:
        return False
    gains = [best_history[i] - best_history[i - 1] for i in range(1, len(best_history))]
    return all(gain <= epsilon for gain in gains[-consecutive:])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Google ADK-backed KuaiRand MLE research loop"
    )
    parser.add_argument("--max-iter", type=int, default=3)
    parser.add_argument("--wall-hours", type=float, default=0.5)
    parser.add_argument(
        "--agent-turns", type=int, default=0,
        help="Optional model-call cap per experiment; 0 means unlimited (default).",
    )
    parser.add_argument(
        "--bootstrap-turns", type=int, default=0,
        help="Optional bootstrap model-call cap; 0 means unlimited (default).",
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
    started = time.time()

    config = load_config()
    if args.data_dir:
        config.DATA_DIR = Path(args.data_dir).resolve()
    source_data_dir = config.DATA_DIR
    adk_settings = configure_google_adk_environment()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    workspace = config.EXPERIMENT_WORKSPACE_DIR / run_id
    workspace.mkdir(parents=True, exist_ok=True)

    console.harness(
        "Data preparation",
        status="Preparing count-verified train/validation-only data view",
    )
    split_manifest = prepare_train_valid_view(source_data_dir, workspace / "candidate_data")
    config.DATA_DIR = workspace / "candidate_data"
    logger = RunLogger(config.ARTIFACTS_DIR, run_id, strict_validation=True)
    experiment_calls = args.agent_turns if args.agent_turns > 0 else "unlimited"
    bootstrap_calls = args.bootstrap_turns if args.bootstrap_turns > 0 else "unlimited"

    console.harness(
        "Run setup",
        run_id=run_id,
        provider=f"google-adk / {adk_settings.model}",
        experiment_budget=(
            f"{args.max_iter} experiments, {args.wall_hours}h, "
            f"{experiment_calls} model calls/experiment"
        ),
        bootstrap_budget=f"{bootstrap_calls} model calls (separate from experiments)",
        llm_limits=(
            f"ADK model output={config.AGENT_MAX_OUTPUT_TOKENS} tokens/call; "
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
    root_path.write_text(root_code, encoding="utf-8")
    provider_label = f"google-adk / {adk_settings.model}"
    agent = ResearchAgent(
        config,
        model=adk_settings.model,
        event_writer=logger.write_llm_event,
        provider_label=provider_label,
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
        failed_results = {
            "run_id": run_id,
            "architecture": "google_adk_persistent_agent",
            "provider": provider_label,
            "task_definition_confirmed": args.task_definition_confirmed,
            "stop_reason": "bootstrap_failed",
            "converged": False,
            "error": failed_bootstrap.error,
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
        }
        logger.write_results(failed_results)
        logger.write_report(f"""# Single-agent research run {run_id}

## Outcome

- Stop reason: `bootstrap_failed`
- Converged: `false`
- Error: {failed_bootstrap.error}
- Input/output tokens: {totals['tokens']['input']} / {totals['tokens']['output']}
- Manual interventions: {totals['interventions']}

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

    best_primary = root_primary
    best_metrics = dict(bootstrap_result.metrics)
    best_iteration = 0
    best_path = root_path
    best_history = [best_primary]
    successful_agent_experiments = 0
    attempted_agent_experiments = 0
    failed_agent_experiments = 0
    stop_reason = "max_iterations"
    trajectory: list[dict[str, object]] = [{
        "iteration": 0,
        "status": "success",
        "primary": root_primary,
        "incumbent_primary": root_primary,
        "is_new_best": True,
    }]

    for iteration in range(1, args.max_iter + 1):
        if time.time() - started >= args.wall_hours * 3600:
            stop_reason = "wall_clock"
            console.harness("Run stopped", reason="Wall-clock budget reached")
            break
        if _converged(best_history, config.CONVERGENCE_EPSILON, config.CONVERGENCE_N):
            stop_reason = "converged"
            console.harness("Run stopped", reason="Convergence rule reached")
            break

        attempted_agent_experiments += 1
        trial_dir = workspace / f"trial_{iteration:03d}"
        trial_dir.mkdir(parents=True, exist_ok=True)
        trial_path = trial_dir / "model.py"
        shutil.copy2(best_path, trial_path)
        parent_code = trial_path.read_text(encoding="utf-8")
        console.harness(
            "Experiment start",
            experiment=f"{iteration}/{args.max_iter}",
            inherited_primary=f"{best_primary:.6f}",
        )
        result = agent.run_iteration(
            iteration, trial_dir, best_primary, best_primary, args.agent_turns
        )
        final_code = result.final_code or parent_code
        primary = float(result.metrics["primary"]) if result.metrics else None
        is_new_best = primary is not None and primary > best_primary
        status = "success" if result.success else "failed"
        logger.write(_log_row(
            iteration,
            best_iteration,
            result,
            status,
            is_new_best,
            config.BASELINE_PRIMARY,
            config.HEADROOM,
            _diff(parent_code, final_code, f"trial_{best_iteration:03d}/model.py", f"trial_{iteration:03d}/model.py"),
            trial_path,
        ))

        if result.success:
            successful_agent_experiments += 1
            if is_new_best:
                best_primary = primary
                best_metrics = dict(result.metrics or {})
                best_iteration = iteration
                best_path = trial_path
                console.harness(
                    "Experiment result",
                    status="New best candidate",
                    validation_primary=f"{primary:.6f}",
                )
            else:
                console.harness(
                    "Experiment result",
                    status="Candidate scored; incumbent retained",
                    validation_primary=f"{primary:.6f}",
                    best_primary=f"{best_primary:.6f}",
                )
            best_history.append(best_primary)
        else:
            failed_agent_experiments += 1
            console.harness("Experiment failed", error=result.error)
            if any(
                event.get("action") == "user_declined_resume"
                for event in result.recovery_events
            ):
                trajectory.append({
                    "iteration": iteration,
                    "status": status,
                    "primary": primary,
                    "incumbent_primary": best_primary,
                    "is_new_best": is_new_best,
                })
                console.harness(
                    "Run stopped",
                    reason="User declined automatic provider-quota recovery",
                )
                stop_reason = "user_stopped"
                break

        trajectory.append({
            "iteration": iteration,
            "status": status,
            "primary": primary,
            "incumbent_primary": best_primary,
            "is_new_best": is_new_best,
        })

    converged = _converged(
        best_history, config.CONVERGENCE_EPSILON, config.CONVERGENCE_N
    )
    if converged and stop_reason == "max_iterations":
        stop_reason = "converged"
    totals = logger.running_totals()
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
    results = {
        "run_id": run_id,
        "architecture": "google_adk_persistent_agent",
        "provider": provider_label,
        "task_definition_confirmed": args.task_definition_confirmed,
        "stop_reason": stop_reason,
        "converged": converged,
        "run_config": {
            "max_iterations": args.max_iter,
            "wall_hours": args.wall_hours,
            "bootstrap_model_calls": args.bootstrap_turns,
            "experiment_model_calls": args.agent_turns,
        },
        "convergence": {
            "epsilon": config.CONVERGENCE_EPSILON,
            "patience": config.CONVERGENCE_N,
            "incumbent_history": best_history,
        },
        "baseline_valid_metrics": baseline_metrics,
        "baseline_valid_primary": config.BASELINE_PRIMARY,
        "reproduced_baseline_valid_primary": root_primary,
        "best_trial": best_iteration,
        "best_valid_metrics": best_metrics,
        "best_valid_primary": best_primary,
        "metric_deltas_vs_baseline": metric_deltas,
        "delta_vs_baseline": best_primary - config.BASELINE_PRIMARY,
        "attempted_agent_experiments": attempted_agent_experiments,
        "successful_agent_experiments": successful_agent_experiments,
        "failed_agent_experiments": failed_agent_experiments,
        "trajectory": trajectory,
        "tokens": totals["tokens"],
        "gpu_hours": args.gpu_hours,
        "total_wall_seconds": total_wall_seconds,
        "manual_interventions": totals["interventions"],
        "llm_trace": totals["llm_trace"],
        "best_workspace_code_path": str(best_path),
        "prompt_templates": agent.prompt_evidence,
        "task_context_bootstrap": agent.bootstrap_evidence,
        "data_view_manifest": split_manifest,
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
    report_path = logger.write_report(f"""# Single-agent research run {run_id}

## Outcome

- Stop reason: {stop_reason}
- Converged: {str(converged).lower()}
- Starter Kit task definition confirmed: {str(args.task_definition_confirmed).lower()}
- Reproduced FM baseline validation primary: {root_primary:.6f}
- Best validation primary: {best_primary:.6f}
- Delta versus published baseline: {best_primary - config.BASELINE_PRIMARY:+.6f}
- Best trial: trial_{best_iteration:03d}
- Attempted autonomous experiments: {attempted_agent_experiments}
- Successful autonomous experiments: {successful_agent_experiments}
- Failed autonomous experiments: {failed_agent_experiments}

## Validation metrics

| Metric | Official baseline | Best validation | Delta |
|---|---:|---:|---:|
| GAUC | {config.BASELINE_GAUC:.6f} | {float(best_metrics.get('GAUC', 0.0)):.6f} | {metric_deltas.get('GAUC', 0.0):+.6f} |
| nDCG@5 | {config.BASELINE_NDCG:.6f} | {float(best_metrics.get('nDCG@5', 0.0)):.6f} | {metric_deltas.get('nDCG@5', 0.0):+.6f} |
| primary | {config.BASELINE_PRIMARY:.6f} | {best_primary:.6f} | {best_primary - config.BASELINE_PRIMARY:+.6f} |

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
- Detailed LLM trace: `{totals['llm_trace']['path']}`

## Prompt templates

```json
{json.dumps(agent.prompt_evidence, indent=2)}
```

## Retained task context bootstrap

```json
{json.dumps(agent.bootstrap_evidence, indent=2)}
```

## Architecture

Google ADK owned the persistent session and model/tool event loop across EDA, research,
hypothesis selection, code, execution, repair, and reflection. The retained Python
harness enforced budgets, validation-only execution, baseline verification, and evidence
logging.
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
