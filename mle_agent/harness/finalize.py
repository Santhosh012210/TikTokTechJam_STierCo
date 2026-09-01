"""Promote one frozen autonomous run into validated final submission artifacts."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from mle_agent.harness.config import Config
from mle_agent.harness.evaluation import score_validation_predictions
from mle_agent.harness.tools import redact_secrets
from mle_agent.harness.validator import scan_candidate_source


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run(command: list[str], *, cwd: Path, env: dict[str, str], timeout: int) -> str:
    completed = subprocess.run(
        command,
        cwd=str(cwd),
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    output = redact_secrets(completed.stdout + completed.stderr)
    if completed.returncode != 0:
        raise RuntimeError(
            f"command failed with exit {completed.returncode}: {command!r}\n{output[-6000:]}"
        )
    return output


def _submission_candidates(run_metrics: dict) -> list[dict]:
    """Return frozen successful nodes in conservative-score order.

    Submission compatibility is an eligibility constraint, not a validation
    metric.  Keeping the full ordered frontier lets finalization fall back to
    the strongest immutable node that both reproduces and can score the hidden
    rows, without rewriting the research winner after the run.
    """
    if run_metrics.get("architecture") != "immutable_candidate_frontier":
        return [{
            "node_id": run_metrics.get("best_node_id", run_metrics.get("best_trial")),
            "code_path": run_metrics["best_workspace_code_path"],
            "source_sha256": run_metrics.get("best_source_sha256"),
            "trial_config": dict(run_metrics.get("best_trial_config") or {}),
            "seed": int(run_metrics.get("best_seed", Config.SEED)),
            "metrics": dict(run_metrics.get("best_valid_metrics") or {}),
            "stability": run_metrics.get("best_trial_stability"),
            "conservative_primary": run_metrics.get("best_conservative_primary"),
        }]

    frontier = run_metrics.get("candidate_frontier") or {}
    raw_nodes = frontier.get("nodes") if isinstance(frontier, dict) else None
    if not isinstance(raw_nodes, dict):
        raise ValueError("immutable run does not contain a candidate frontier")
    candidates = [
        dict(node)
        for node in raw_nodes.values()
        if isinstance(node, dict)
        and node.get("status") == "success"
        and node.get("frozen") is True
        and node.get("code_path")
        and node.get("source_sha256")
    ]
    repairs = run_metrics.get("submission_repairs") or []
    candidates.extend(
        dict(node)
        for node in repairs
        if isinstance(node, dict)
        and node.get("status") == "success"
        and node.get("frozen") is True
        and node.get("code_path")
        and node.get("source_sha256")
    )
    candidates.sort(
        key=lambda node: (
            float(node.get("conservative_primary", -float("inf"))),
            bool(node.get("submission_ready_verified")),
        ),
        reverse=True,
    )
    if not candidates:
        raise ValueError("immutable run has no frozen successful candidate")
    return candidates


def finalize_run(
    run_id: str,
    *,
    data_dir: Path,
    task_definition_confirmed: bool,
    allow_budget_stop: bool = False,
    timeout_seconds: int = 1800,
) -> dict:
    """Create final artifacts from the validation-best candidate of one run."""
    if not task_definition_confirmed:
        raise ValueError(
            "finalization requires explicit confirmation that the checked-in Starter Kit "
            "defines the authoritative label and metrics"
        )

    project_root = Config.PROJECT_ROOT
    artifacts_root = Config.ARTIFACTS_DIR
    run_dir = (artifacts_root / "runs" / run_id).resolve()
    run_metrics_path = run_dir / "results" / "metrics.json"
    if not run_metrics_path.is_file():
        raise FileNotFoundError(f"run metrics not found: {run_metrics_path}")
    run_metrics = json.loads(run_metrics_path.read_text(encoding="utf-8"))
    repair_manifest_path = run_dir / "results" / "submission_repairs.json"
    if repair_manifest_path.is_file():
        repair_manifest = json.loads(repair_manifest_path.read_text(encoding="utf-8"))
        repairs = repair_manifest.get("repairs")
        if not isinstance(repairs, list):
            raise ValueError("submission repair manifest has invalid repairs")
        run_metrics = dict(run_metrics)
        run_metrics["submission_repairs"] = repairs
    if not run_metrics.get("task_definition_confirmed"):
        raise ValueError(
            f"run {run_id} did not record task-definition confirmation; "
            "do not promote it as the official submission run"
        )
    stop_reason = str(run_metrics.get("stop_reason", "unknown"))
    converged = bool(run_metrics.get("converged"))
    if not converged and not (
        allow_budget_stop and stop_reason in {"max_iterations", "wall_clock"}
    ):
        raise ValueError(
            f"run {run_id} is not converged (stop_reason={stop_reason}); "
            "use --allow-budget-stop only for an official hard-budget stop"
        )

    workspace_root = Config.EXPERIMENT_WORKSPACE_DIR.resolve()
    immutable_frontier = run_metrics.get("architecture") == "immutable_candidate_frontier"
    candidate_options = _submission_candidates(run_metrics)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"data directory not found: {data_dir}")

    environment_evidence = run_metrics.get("python_environment") or {}
    recorded_python = environment_evidence.get("python_executable")
    python_exe = Path(recorded_python).absolute() if recorded_python else Path(
        Config.PROJECT_ROOT / ".venv" / (
            "Scripts/python.exe" if os.name == "nt" else "bin/python"
        )
    ).absolute()
    if not python_exe.is_file():
        raise FileNotFoundError(f"recorded Python environment not found: {python_exe}")
    if recorded_python and not python_exe.parent.resolve().is_relative_to(workspace_root):
        raise ValueError(
            f"recorded run Python is outside the experiment workspace: {python_exe}"
        )
    final_env_names = {
        "PATH", "LANG", "LC_ALL", "LC_CTYPE", "TZ", "SYSTEMROOT", "WINDIR",
        "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
    }
    env = {
        key: value for key, value in os.environ.items() if key in final_env_names
    }
    # Same import boundary as research execution: only the organiser starter kit.
    # Candidates inline their own prediction and submission writers, so nothing
    # here needs the trusted harness on the path.
    env["PYTHONPATH"] = str(Config.BASELINE_ROOT)

    final_dir = artifacts_root / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    finalization_attempts: list[dict] = []
    selected: dict | None = None
    candidate: Path | None = None
    trial_config: dict = {}
    best_seed = Config.SEED
    validation_output = ""
    candidate_output = ""
    check_output = ""
    with tempfile.TemporaryDirectory() as temp:
        temp_dir = Path(temp)
        for option in candidate_options:
            option_candidate = Path(str(option["code_path"])).resolve()
            node_id = option.get("id", option.get("node_id"))
            if (
                not option_candidate.is_relative_to(workspace_root)
                or option_candidate.name != "model.py"
            ):
                raise ValueError(
                    f"frozen candidate is outside the experiment workspace: {option_candidate}"
                )
            if not option_candidate.is_file():
                raise FileNotFoundError(
                    f"frozen candidate no longer exists: {option_candidate}"
                )
            option_hash = str(option.get("source_sha256") or "")
            if immutable_frontier and (
                not option_hash or _sha256(option_candidate) != option_hash
            ):
                raise ValueError(
                    f"frozen node {node_id} source no longer matches its scored hash"
                )
            option_config = dict(option.get("trial_config") or {})
            option_seed = int(option.get("seed", Config.SEED))
            if immutable_frontier:
                manifest_path = (option_candidate.parent / "bundle.json").resolve()
                if not manifest_path.is_file():
                    raise FileNotFoundError(
                        f"frozen node {node_id} bundle manifest is missing"
                    )
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                if (
                    manifest.get("source_sha256") != option_hash
                    or int(manifest.get("seed", -1)) != option_seed
                    or dict(manifest.get("trial_config") or {}) != option_config
                ):
                    raise ValueError(
                        f"frozen node {node_id} bundle identity does not match the frontier"
                    )
            source = option_candidate.read_text(encoding="utf-8", errors="replace")
            violations = scan_candidate_source(source)
            if violations:
                raise ValueError(
                    f"frozen node {node_id} violates hidden-test policy: {violations}"
                )
            if "--submission-path" not in source or "write_hidden_submission" not in source:
                finalization_attempts.append({
                    "node_id": node_id,
                    "status": "submission_incompatible",
                    "error": "missing trusted submission contract",
                })
                continue

            option_temp = temp_dir / f"node_{int(node_id):03d}"
            option_temp.mkdir()
            trial_config_path = option_temp / "trial_config.json"
            trial_config_path.write_text(
                json.dumps(option_config, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
                encoding="utf-8",
            )
            trial_config_args = (
                ["--trial-config", str(trial_config_path)] if immutable_frontier else []
            )
            option_validation_output = "legacy run: frozen-bundle revalidation not required"
            if immutable_frontier:
                candidate_data = option_candidate.parents[2] / "candidate_data"
                if not candidate_data.is_dir():
                    raise FileNotFoundError(
                        f"frozen run validation data no longer exists: {candidate_data}"
                    )
                validation_predictions = option_temp / "validation_predictions.csv"
                option_validation_output = _run(
                    [
                        str(python_exe), str(option_candidate),
                        "--data_dir", str(candidate_data),
                        "--seed", str(option_seed),
                        "--prediction-path", str(validation_predictions),
                        *trial_config_args,
                    ],
                    cwd=option_temp,
                    env=env,
                    timeout=timeout_seconds,
                )
                rescored = score_validation_predictions(
                    validation_predictions,
                    candidate_data,
                    Config.BASELINE_ROOT / "evaluate.py",
                ).metrics
                recorded_metrics = dict(option.get("metrics") or {})
                for metric in ("GAUC", "nDCG@5", "primary"):
                    if metric not in recorded_metrics or abs(
                        float(rescored[metric]) - float(recorded_metrics[metric])
                    ) > 1e-10:
                        raise ValueError(
                            f"frozen node {node_id} re-score mismatch for {metric}: "
                            f"recorded={recorded_metrics.get(metric)!r}, "
                            f"rescored={rescored[metric]!r}"
                        )

            temporary_submission = option_temp / "submission.csv"
            try:
                option_candidate_output = _run(
                    [
                        str(python_exe), str(option_candidate),
                        "--data_dir", str(data_dir.resolve()),
                        "--seed", str(option_seed),
                        "--submission-path", str(temporary_submission),
                        *trial_config_args,
                    ],
                    cwd=option_temp,
                    env=env,
                    timeout=timeout_seconds,
                )
                option_check_output = _run(
                    [
                        str(python_exe), str(Config.BASELINE_ROOT / "submit.py"),
                        str(temporary_submission),
                        "--data_dir", str(data_dir.resolve()),
                        "--split", "test",
                        "--check",
                    ],
                    cwd=project_root,
                    env=env,
                    timeout=timeout_seconds,
                )
            except RuntimeError as exc:
                finalization_attempts.append({
                    "node_id": node_id,
                    "status": "submission_incompatible",
                    "error": str(exc)[-2000:],
                })
                continue

            selected = option
            candidate = option_candidate
            trial_config = option_config
            best_seed = option_seed
            validation_output = option_validation_output
            candidate_output = option_candidate_output
            check_output = option_check_output
            finalization_attempts.append({
                "node_id": node_id,
                "status": "selected",
                "conservative_primary": option.get("conservative_primary"),
            })
            shutil.copy2(temporary_submission, final_dir / "submission.csv")
            break

    if selected is None or candidate is None:
        raise RuntimeError(
            "no frozen frontier candidate could generate a valid submission: "
            + json.dumps(finalization_attempts, ensure_ascii=False)
        )

    shutil.copy2(candidate, final_dir / "model.py")

    # Per-run evidence is local-only; only the promoted run ships. Copy its whole
    # directory in, so the hypothesis / diff / metrics / recovery logs the brief
    # asks for travel with the submission they belong to.
    final_run_dir = final_dir / "run"
    if final_run_dir.exists():
        shutil.rmtree(final_run_dir)
    shutil.copytree(run_dir, final_run_dir)
    requirements_lock = environment_evidence.get("requirements_lock")
    final_requirements_lock: Path | None = None
    if requirements_lock:
        source_lock = Path(str(requirements_lock)).resolve()
        if not source_lock.is_relative_to(run_dir) or not source_lock.is_file():
            raise ValueError(
                f"recorded dependency lock is invalid for run {run_id}: {source_lock}"
            )
        final_requirements_lock = final_dir / "requirements.lock.txt"
        shutil.copy2(source_lock, final_requirements_lock)
    source_hash = _sha256(final_dir / "model.py")
    submission_hash = _sha256(final_dir / "submission.csv")
    selected_node_id = selected.get("id", selected.get("node_id"))
    selected_metrics = dict(selected.get("metrics") or {})
    selected_stability = selected.get("stability")
    baseline_metrics = dict(run_metrics.get("baseline_valid_metrics") or {})
    selected_deltas = {
        metric: float(selected_metrics.get(metric, 0.0)) - float(baseline_metrics.get(metric, 0.0))
        for metric in ("GAUC", "nDCG@5", "primary")
    }
    final_metrics = {
        "source_run_id": run_id,
        "run_evidence_dir": str(final_run_dir),
        "source_trial": selected_node_id,
        "source_node_id": selected_node_id,
        "research_best_node_id": run_metrics.get("best_node_id"),
        "submission_repair_parent_node_id": selected.get("repair_parent_id"),
        "source_model_sha256": source_hash,
        "source_trial_config": trial_config,
        "source_seed": best_seed,
        "submission_sha256": submission_hash,
        "task_definition_confirmed": True,
        "stop_reason": stop_reason,
        "converged": converged,
        "baseline_valid_metrics": baseline_metrics,
        "official_baseline_reference": run_metrics.get("official_baseline_reference"),
        "official_hidden_test_baseline_metrics": run_metrics.get(
            "official_hidden_test_baseline_metrics"
        ),
        "best_valid_metrics": selected_metrics,
        "best_trial_stability": selected_stability,
        "stability_seeds": (
            selected_stability.get("seeds") if selected_stability
            else run_metrics.get("stability_seeds")
        ),
        "metric_deltas_vs_baseline": selected_deltas,
        "tokens": run_metrics.get("tokens"),
        "gpu_hours": run_metrics.get("gpu_hours", 0.0),
        "total_wall_seconds": run_metrics.get("total_wall_seconds"),
        "manual_interventions": run_metrics.get("manual_interventions", 0),
        "python_environment": environment_evidence,
        "requirements_lock": (
            str(final_requirements_lock) if final_requirements_lock else None
        ),
        "prompt_templates": run_metrics.get("prompt_templates"),
        "submission_validation": check_output.strip(),
        "frozen_candidate_revalidation": validation_output[-2000:],
        "candidate_output_tail": candidate_output[-2000:],
        "finalization_candidate_attempts": finalization_attempts,
        "hidden_test_comparison_status": (
            "unmeasured; submission alignment was checked without scoring hidden labels"
        ),
    }
    (final_dir / "metrics.json").write_text(
        json.dumps(final_metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    best = final_metrics.get("best_valid_metrics") or {}
    baseline = final_metrics.get("baseline_valid_metrics") or {}
    deltas = final_metrics.get("metric_deltas_vs_baseline") or {}
    stability = final_metrics.get("best_trial_stability")
    if stability and stability.get("primary_mean") is not None:
        per_seed = "; ".join(
            f"seed {seed}: {float(m['primary']):.6f}" if m else f"seed {seed}: failed"
            for seed, m in stability["per_seed"].items()
        )
        stability_line = (
            f"- Multi-seed primary: mean `{stability['primary_mean']:.6f}` "
            f"± std `{stability.get('primary_std', 0.0):.6f}` "
            f"(published FM 5-seed std `0.0008`); {per_seed}\n"
        )
    else:
        stability_line = "- Multi-seed stability: not available for this run\n"
    fallback_nodes = [
        item["node_id"] for item in finalization_attempts
        if item.get("status") == "submission_incompatible"
    ]
    fallback_line = (
        f"- Submission compatibility fallback: rejected frozen node(s) "
        f"{fallback_nodes}; selected the next-best reproducible frozen node\n"
        if fallback_nodes else
        "- Submission compatibility fallback: not required\n"
    )
    repair_parent = final_metrics.get("submission_repair_parent_node_id")
    repair_line = (
        f"- Submission-contract repair: selected frozen child of `node_{int(repair_parent):03d}`; "
        "the child reproduced all validation seeds before promotion\n"
        if repair_parent is not None else
        "- Submission-contract repair: not required\n"
    )
    report = f"""# Final submission

- Source autonomous run: `{run_id}`
- Source frozen node: `node_{int(final_metrics.get('source_node_id', 0)):03d}`
- Research frontier best node: `node_{int(run_metrics.get('best_node_id', run_metrics.get('best_trial', 0))):03d}`
- Stop reason: `{stop_reason}`
- Converged: `{str(converged).lower()}`
- Manual interventions during autonomous research: `{run_metrics.get('manual_interventions', 0)}`
- Input/output tokens: `{(run_metrics.get('tokens') or {}).get('input', 0)}` / `{(run_metrics.get('tokens') or {}).get('output', 0)}`
- GPU-hours: `{run_metrics.get('gpu_hours', 0.0)}`
- Python environment: `{python_exe}`
- Resolved dependency lock: `{final_requirements_lock or 'not recorded by legacy run'}`
- Per-iteration run logs: `{final_run_dir}`
- Published organizer hidden-test baseline primary: `{float((run_metrics.get('official_hidden_test_baseline_metrics') or {}).get('primary', 0.5946)):.6f}`
- Hidden-test win status: `unmeasured`; organizer evaluation is required
{stability_line}
{fallback_line}
{repair_line}
| Validation metric | Official baseline | Final candidate | Delta |
|---|---:|---:|---:|
| GAUC | {float(baseline.get('GAUC', 0.0)):.6f} | {float(best.get('GAUC', 0.0)):.6f} | {float(deltas.get('GAUC', 0.0)):+.6f} |
| nDCG@5 | {float(baseline.get('nDCG@5', 0.0)):.6f} | {float(best.get('nDCG@5', 0.0)):.6f} | {float(deltas.get('nDCG@5', 0.0)):+.6f} |
| primary | {float(baseline.get('primary', 0.0)):.6f} | {float(best.get('primary', 0.0)):.6f} | {float(deltas.get('primary', 0.0)):+.6f} |

Submission validation: `{check_output.strip()}`

The trusted finalizer generated row-aligned hidden-split predictions but did not compute or expose
hidden-test metrics to the research agent.

## Task definition

The brief's judging criteria mention `click` / NDCG@10 / Recall@50, while the checked-in Starter
Kit (`evaluate.py`, `baseline_scores.json`, its README, and the brief's own line 74) defines the
target as `long_view` scored by GAUC and nDCG@5. This entry was built to the **Starter Kit** as the
executable authority the submission is checked against -- a deliberate reading of an internally
inconsistent brief, not an oversight. `--task-definition-confirmed` records that this reading was
made intentionally.
"""
    (final_dir / "final-report.md").write_text(report, encoding="utf-8")
    return final_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--data-dir", type=Path,
        default=Config.PROJECT_ROOT / "datasets" / "KuaiRand-Pure" / "data",
    )
    parser.add_argument("--task-definition-confirmed", action="store_true")
    parser.add_argument("--allow-budget-stop", action="store_true")
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    args = parser.parse_args()
    try:
        result = finalize_run(
            args.run_id,
            data_dir=args.data_dir,
            task_definition_confirmed=args.task_definition_confirmed,
            allow_budget_stop=args.allow_budget_stop,
            timeout_seconds=args.timeout_seconds,
        )
    except (FileNotFoundError, KeyError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
