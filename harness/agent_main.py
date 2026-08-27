"""CLI for the single-agent autonomous MLE loop.

Usage:
    python -m harness.agent_main --max-iter 3 --wall-hours 0.5 --agent-turns 8
"""
from __future__ import annotations

import argparse
import difflib
import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

from harness.agent_tools import execute_model
from harness.config import load_config
from harness.data_view import prepare_train_valid_view
from harness.logger import RunLogger
from harness.provider import validate_provider_environment
from harness.root_model import make_root_model_py
from research_agent.agent import AgentIterationResult, ResearchAgent


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
    return {
        "iteration": iteration,
        "session_type": "agent",
        "timestamp": _now(),
        "parent_iteration": parent_iteration,
        "hypothesis": result.hypothesis,
        "hypothesis_source": "single_agent",
        "target_component": "agent_selected",
        "code_path": str(code_path),
        "code_diff": code_diff,
        "status": status,
        "internal_repairs": max(0, len(result.executions) - 1),
        "metrics": result.metrics,
        "delta_vs_baseline": delta,
        "pct_of_headroom": (delta / headroom * 100) if delta is not None and headroom else None,
        "is_new_best": is_new_best,
        "error": result.error,
        "tokens": result.token_counts,
        "wall_seconds": result.wall_seconds,
        "human_intervention": False,
        "reasoning": result.reasoning,
        "reflection": result.reflection,
        "execution_attempts": result.executions,
        "recovery_events": result.recovery_events,
    }


def _converged(best_history: list[float], epsilon: float, consecutive: int) -> bool:
    if len(best_history) < consecutive + 1:
        return False
    gains = [best_history[i] - best_history[i - 1] for i in range(1, len(best_history))]
    return all(gain <= epsilon for gain in gains[-consecutive:])


def main() -> None:
    parser = argparse.ArgumentParser(description="Single-agent KuaiRand MLE research loop")
    parser.add_argument("--max-iter", type=int, default=3)
    parser.add_argument("--wall-hours", type=float, default=0.5)
    parser.add_argument("--agent-turns", type=int, default=8)
    parser.add_argument("--data-dir", type=str, default=None)
    args = parser.parse_args()

    config = load_config()
    if args.data_dir:
        config.DATA_DIR = Path(args.data_dir).resolve()
    source_data_dir = config.DATA_DIR
    provider = validate_provider_environment()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    workspace = config.EXPERIMENT_WORKSPACE_DIR / run_id
    workspace.mkdir(parents=True, exist_ok=True)

    print("[agent] preparing count-verified train/validation-only data view...")
    split_manifest = prepare_train_valid_view(source_data_dir, workspace / "candidate_data")
    config.DATA_DIR = workspace / "candidate_data"
    logger = RunLogger(config.ARTIFACTS_DIR, run_id)

    print(f"[agent] run_id={run_id}")
    print(f"[agent] provider={provider['provider']} model={provider['model']}")
    print(f"[agent] budget={args.max_iter} experiments, {args.wall_hours}h, {args.agent_turns} turns/experiment")
    print(
        "[agent] LLM budgets: "
        f"work_output={config.AGENT_MAX_OUTPUT_TOKENS} "
        f"reflection_output={config.AGENT_REFLECTION_MAX_TOKENS} tokens "
        f"read_result={config.AGENT_READ_MAX_CHARS} chars"
    )
    print(
        "[agent] candidate rows: "
        f"train={split_manifest['emitted_candidate_counts']['train']:,} "
        f"valid={split_manifest['emitted_candidate_counts']['valid']:,} test=0"
    )

    root_dir = workspace / "trial_000"
    root_dir.mkdir(parents=True, exist_ok=True)
    root_path = root_dir / "model.py"
    root_code = make_root_model_py(config)
    root_path.write_text(root_code, encoding="utf-8")
    print("[agent] reproducing FM baseline deterministically...")
    root_execution = execute_model(root_dir, config)
    if not root_execution.success:
        logger.close()
        raise RuntimeError(f"FM baseline reproduction failed: {root_execution.error}")
    root_primary = float(root_execution.metrics["primary"])
    if abs(root_primary - config.BASELINE_PRIMARY) > config.CONVERGENCE_EPSILON:
        logger.close()
        raise RuntimeError(
            f"FM baseline mismatch: got {root_primary:.6f}, expected {config.BASELINE_PRIMARY:.6f}"
        )
    print(f"[agent] baseline valid primary={root_primary:.6f}")

    root_result = AgentIterationResult(
        success=True,
        hypothesis="Reproduce the official FM baseline",
        reasoning="Deterministic preflight before autonomous research.",
        reflection="Baseline reproduced successfully.",
        metrics=root_execution.metrics,
        executions=[],
        recovery_events=[],
        token_counts={"input": 0, "output": 0},
        wall_seconds=root_execution.wall_seconds,
        error=None,
        final_code=root_code,
    )
    logger.write(_log_row(
        0, None, root_result, "success", True, config.BASELINE_PRIMARY,
        config.HEADROOM, "", root_path,
    ))

    agent = ResearchAgent(config)
    best_primary = root_primary
    best_iteration = 0
    best_path = root_path
    best_history = [best_primary]
    successful_agent_experiments = 0
    started = time.time()

    for iteration in range(1, args.max_iter + 1):
        if time.time() - started >= args.wall_hours * 3600:
            print("[agent] wall-clock budget reached")
            break
        if _converged(best_history, config.CONVERGENCE_EPSILON, config.CONVERGENCE_N):
            print("[agent] convergence reached")
            break

        trial_dir = workspace / f"trial_{iteration:03d}"
        trial_dir.mkdir(parents=True, exist_ok=True)
        trial_path = trial_dir / "model.py"
        shutil.copy2(best_path, trial_path)
        parent_code = trial_path.read_text(encoding="utf-8")
        print(f"[agent] experiment {iteration}/{args.max_iter}")
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
                best_iteration = iteration
                best_path = trial_path
                print(f"[agent] NEW BEST primary={primary:.6f}")
            else:
                print(f"[agent] scored primary={primary:.6f}; best remains {best_primary:.6f}")
            best_history.append(best_primary)
        else:
            print(f"[agent] failed: {result.error}")

    totals = logger.running_totals()
    results = {
        "run_id": run_id,
        "architecture": "single_persistent_agent",
        "baseline_valid_primary": config.BASELINE_PRIMARY,
        "reproduced_baseline_valid_primary": root_primary,
        "best_trial": best_iteration,
        "best_valid_primary": best_primary,
        "delta_vs_baseline": best_primary - config.BASELINE_PRIMARY,
        "successful_agent_experiments": successful_agent_experiments,
        "tokens": totals["tokens"],
        "manual_interventions": totals["interventions"],
        "best_workspace_code_path": str(best_path),
        "prompt_templates": agent.prompt_evidence,
        "data_view_manifest": split_manifest,
    }
    results_path = logger.write_results(results)
    report_path = logger.write_report(f"""# Single-agent research run {run_id}

## Outcome

- Reproduced FM baseline validation primary: {root_primary:.6f}
- Best validation primary: {best_primary:.6f}
- Delta versus published baseline: {best_primary - config.BASELINE_PRIMARY:+.6f}
- Best trial: trial_{best_iteration:03d}
- Successful autonomous experiments: {successful_agent_experiments}

## Resource usage

- Input tokens: {totals['tokens']['input']}
- Output tokens: {totals['tokens']['output']}
- Manual interventions: {totals['interventions']}

## Prompt templates

```json
{json.dumps(agent.prompt_evidence, indent=2)}
```

## Architecture

One persistent agent conversation owned EDA, research, hypothesis selection, code,
execution, repair, and reflection. Python retained budgets, validation-only execution,
baseline verification, and evidence logging.
""")
    logger.close()
    print(f"[agent] results={results_path}")
    print(f"[agent] report={report_path}")


if __name__ == "__main__":
    main()
