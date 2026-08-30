"""Per-run Python environments and reproducible dependency snapshots."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from mle_agent.harness.config import Config
from mle_agent.harness.tools import redact_secrets


AUTO_INSTALL_ALLOWLIST = frozenset({
    "catboost",
    "imbalanced-learn",
    "lightgbm",
    "numpy",
    "optuna",
    "pandas",
    "polars",
    "pyarrow",
    "recbole",
    "scikit-learn",
    "scipy",
    "statsmodels",
    "torch",
    "torchaudio",
    "torchrec",
    "torchvision",
    "transformers",
    "xgboost",
})


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _run(
    command: list[str], *, cwd: Path, timeout: int = 120
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def _site_packages(python_executable: str, project_root: Path) -> list[str]:
    probe = (
        "import json, site\n"
        "print(json.dumps(site.getsitepackages()))\n"
    )
    result = _run(
        [python_executable, "-c", probe], cwd=project_root, timeout=30
    )
    if result.returncode != 0:
        raise RuntimeError(
            "could not inspect Python site-packages: "
            + redact_secrets((result.stdout + result.stderr)[-2000:])
        )
    paths = json.loads(result.stdout)
    if not isinstance(paths, list) or not all(isinstance(path, str) for path in paths):
        raise RuntimeError("Python site-packages probe returned an invalid result")
    return paths


def create_run_environment(
    config: Config, workspace: Path, artifact_run_dir: Path
) -> dict[str, object]:
    """Create one install-isolated venv and make it the candidate interpreter.

    The repository environment remains a read-only package base through a ``.pth``
    file. New or upgraded distributions are installed only into the run venv.
    """
    workspace = workspace.resolve()
    environment_dir = (workspace / ".venv").resolve()
    base_python = str(config.PYTHON_EXE)
    result = _run(
        [base_python, "-m", "venv", str(environment_dir)],
        cwd=config.PROJECT_ROOT,
        timeout=180,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "could not create the per-run Python environment: "
            + redact_secrets((result.stdout + result.stderr)[-4000:])
        )

    run_python = environment_dir / (
        "Scripts/python.exe" if os.name == "nt" else "bin/python"
    )
    if not run_python.is_file():
        raise RuntimeError(f"run environment Python was not created: {run_python}")

    base_sites = [
        str(Path(path).resolve())
        for path in _site_packages(base_python, config.PROJECT_ROOT)
        if Path(path).is_dir()
    ]
    run_sites = _site_packages(str(run_python), config.PROJECT_ROOT)
    if not run_sites:
        raise RuntimeError("run environment has no site-packages directory")
    inheritance_file = Path(run_sites[0]) / "mle_agent_base_environment.pth"
    inheritance_file.write_text(
        "".join(f"{path}\n" for path in base_sites), encoding="utf-8"
    )

    environment_artifacts = artifact_run_dir / "environment"
    environment_artifacts.mkdir(parents=True, exist_ok=True)
    config.BASE_PYTHON_EXE = base_python
    config.RUN_ENV_DIR = environment_dir
    config.RUN_ENV_ARTIFACT_DIR = environment_artifacts
    config.PYTHON_EXE = str(run_python)

    manifest = {
        "schema_version": 1,
        "created_at": _now(),
        "environment_dir": str(environment_dir),
        "python_executable": str(run_python),
        "base_python_executable": base_python,
        "base_package_paths": base_sites,
        "install_isolation": "all pip writes target the dedicated per-run venv",
        "auto_install_allowlist": sorted(AUTO_INSTALL_ALLOWLIST),
        "binary_only_installs": True,
        "pip_isolated_mode": True,
        "snapshots": [],
    }
    (environment_artifacts / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return snapshot_run_environment(config, phase="created")


def snapshot_run_environment(config: Config, *, phase: str) -> dict[str, object]:
    """Write a resolved ``pip freeze --all`` lock for the current run."""
    artifact_dir = config.RUN_ENV_ARTIFACT_DIR
    if artifact_dir is None:
        return {
            "success": False,
            "phase": phase,
            "error": "run environment artifact directory is not configured",
        }
    artifact_dir.mkdir(parents=True, exist_ok=True)
    result = _run(
        [config.PYTHON_EXE, "-m", "pip", "freeze", "--all"],
        cwd=config.PROJECT_ROOT,
        timeout=120,
    )
    output = redact_secrets(result.stdout + result.stderr)
    if result.returncode != 0:
        return {
            "success": False,
            "phase": phase,
            "error": output[-2000:],
        }
    freeze = result.stdout.rstrip() + "\n"
    lock_path = artifact_dir / "requirements.lock.txt"
    lock_path.write_text(freeze, encoding="utf-8")
    digest = hashlib.sha256(freeze.encode("utf-8")).hexdigest()

    manifest_path = artifact_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    snapshot = {
        "phase": phase,
        "timestamp": _now(),
        "requirements_lock": str(lock_path),
        "requirements_lock_sha256": digest,
        "resolved_distribution_count": len(
            [line for line in freeze.splitlines() if line.strip()]
        ),
    }
    manifest.setdefault("snapshots", []).append(snapshot)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return {
        "success": True,
        "environment_manifest": str(manifest_path),
        **snapshot,
        "python_executable": config.PYTHON_EXE,
        "environment_dir": str(config.RUN_ENV_DIR),
    }
