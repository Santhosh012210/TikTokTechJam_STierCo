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

    candidate = Path(str(run_metrics["best_workspace_code_path"])).resolve()
    workspace_root = Config.EXPERIMENT_WORKSPACE_DIR.resolve()
    if not candidate.is_relative_to(workspace_root) or candidate.name != "model.py":
        raise ValueError(f"best candidate is outside the experiment workspace: {candidate}")
    if not candidate.is_file():
        raise FileNotFoundError(f"best candidate no longer exists: {candidate}")
    source = candidate.read_text(encoding="utf-8", errors="replace")
    violations = scan_candidate_source(source)
    if violations:
        raise ValueError(f"candidate violates hidden-test policy: {violations}")
    if "--submission-path" not in source or "write_hidden_submission" not in source:
        raise ValueError(
            "candidate does not preserve the trusted finalization contract "
            "(--submission-path and write_hidden_submission)"
        )
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
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in {
            "LLM_API_KEY", "ANTHROPIC_API_KEY", "GROQ_API_KEY",
            "GEMINI_API_KEY", "OPENAI_API_KEY", "GOOGLE_API_KEY",
        }
    }
    # Same import boundary as research execution: only the organiser starter kit.
    # Candidates inline their own prediction and submission writers, so nothing
    # here needs the trusted harness on the path.
    env["PYTHONPATH"] = str(Config.BASELINE_ROOT)

    final_dir = artifacts_root / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as temp:
        temporary_submission = Path(temp) / "submission.csv"
        candidate_output = _run(
            [
                str(python_exe), str(candidate),
                "--data_dir", str(data_dir.resolve()),
                "--submission-path", str(temporary_submission),
            ],
            cwd=candidate.parent,
            env=env,
            timeout=timeout_seconds,
        )
        check_output = _run(
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
        shutil.copy2(temporary_submission, final_dir / "submission.csv")

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
    final_metrics = {
        "source_run_id": run_id,
        "run_evidence_dir": str(final_run_dir),
        "source_trial": run_metrics.get("best_trial"),
        "source_model_sha256": source_hash,
        "submission_sha256": submission_hash,
        "task_definition_confirmed": True,
        "stop_reason": stop_reason,
        "converged": converged,
        "baseline_valid_metrics": run_metrics.get("baseline_valid_metrics"),
        "best_valid_metrics": run_metrics.get("best_valid_metrics"),
        "best_trial_stability": run_metrics.get("best_trial_stability"),
        "stability_seeds": run_metrics.get("stability_seeds"),
        "metric_deltas_vs_baseline": run_metrics.get("metric_deltas_vs_baseline"),
        "tokens": run_metrics.get("tokens"),
        "gpu_hours": run_metrics.get("gpu_hours", 0.0),
        "total_wall_seconds": run_metrics.get("total_wall_seconds"),
        "manual_interventions": run_metrics.get("manual_interventions"),
        "python_environment": environment_evidence,
        "requirements_lock": (
            str(final_requirements_lock) if final_requirements_lock else None
        ),
        "prompt_templates": run_metrics.get("prompt_templates"),
        "submission_validation": check_output.strip(),
        "candidate_output_tail": candidate_output[-2000:],
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
    report = f"""# Final submission

- Source autonomous run: `{run_id}`
- Source trial: `trial_{int(run_metrics.get('best_trial', 0)):03d}`
- Stop reason: `{stop_reason}`
- Converged: `{str(converged).lower()}`
- Manual interventions: `{run_metrics.get('manual_interventions', 0)}`
- Input/output tokens: `{(run_metrics.get('tokens') or {}).get('input', 0)}` / `{(run_metrics.get('tokens') or {}).get('output', 0)}`
- GPU-hours: `{run_metrics.get('gpu_hours', 0.0)}`
- Python environment: `{python_exe}`
- Resolved dependency lock: `{final_requirements_lock or 'not recorded by legacy run'}`
- Per-iteration run logs: `{final_run_dir}`
{stability_line}
| Validation metric | Official baseline | Final candidate | Delta |
|---|---:|---:|---:|
| GAUC | {float(baseline.get('GAUC', 0.0)):.6f} | {float(best.get('GAUC', 0.0)):.6f} | {float(deltas.get('GAUC', 0.0)):+.6f} |
| nDCG@5 | {float(baseline.get('nDCG@5', 0.0)):.6f} | {float(best.get('nDCG@5', 0.0)):.6f} | {float(deltas.get('nDCG@5', 0.0)):+.6f} |
| primary | {float(baseline.get('primary', 0.0)):.6f} | {float(best.get('primary', 0.0)):.6f} | {float(deltas.get('primary', 0.0)):+.6f} |

Submission validation: `{check_output.strip()}`

The trusted finalizer generated row-aligned hidden-split predictions but did not compute or expose
hidden-test metrics to the research agent.
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
