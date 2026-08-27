"""Single authoritative source for all runtime constants.

Loads baseline_scores.json, resolves absolute paths, reads .env file.
No logic beyond loading and validation.
"""
import json
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# .env loader (minimal, no third-party deps)
# ---------------------------------------------------------------------------

def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        if k not in os.environ:          # env var takes precedence over .env
            os.environ[k] = v


# Repository root (contains agent_harness/, baseline_kuairand-starter-kit/,
# datasets/, candidates/, and logs/).
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load .env before anything reads os.environ
_load_dotenv(_PROJECT_ROOT / ".env")


# ---------------------------------------------------------------------------
# Config class
# ---------------------------------------------------------------------------

class Config:
    # Paths
    PROJECT_ROOT:   Path = _PROJECT_ROOT
    BASELINE_ROOT:  Path = _PROJECT_ROOT / "baseline_kuairand-starter-kit"
    DATA_DIR:       Path = _PROJECT_ROOT / "datasets" / "KuaiRand-Pure" / "data"
    CANDIDATES_DIR: Path = _PROJECT_ROOT / "candidates"
    LOGS_DIR:       Path = _PROJECT_ROOT / "logs"
    HARNESS_DIR:    Path = _PROJECT_ROOT / "agent_harness"
    BASELINE_JSON:  Path = BASELINE_ROOT / "baseline_scores.json"

    # Loaded from baseline_scores.json
    BASELINE_PRIMARY:     float = 0.6016
    BASELINE_GAUC:        float = 0.6674
    BASELINE_NDCG:        float = 0.5357
    ORACLE_PRIMARY:       float = 0.8484
    HEADROOM:             float = 0.2468
    CONVERGENCE_EPSILON:  float = 0.002
    CONVERGENCE_N:        int   = 3

    # Agent session budgets (defaults = dev mode; overridden by CLI in main.py)
    BUILDER_MAX_TURNS:        int   = 2        # dev default; prod = 10
    STRATEGIST_MAX_TURNS:     int   = 6
    BUILDER_WALL_TIMEOUT_S:   float = 300.0    # 5 min per builder session in dev
    ITERATION_BUDGET:         int   = 10       # dev default; prod = 50
    WALL_BUDGET_S:            float = 1800.0   # 30 min dev; prod = 14400
    STRATEGIST_EVERY_N:       int   = 8

    SEED:    int   = 42
    UCB_C:   float = 1.414
    MODEL_ID: str  = "claude-haiku-4-5-20251001"
    PYTHON_EXE: str = sys.executable

    @property
    def api_key(self) -> str:
        key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not key:
            raise EnvironmentError(
                "ANTHROPIC_API_KEY is not set. "
                "Either export it or add it to .env at the project root."
            )
        return key


def load_config() -> Config:
    cfg = Config()

    # Verify data directory exists
    if not cfg.DATA_DIR.exists():
        raise FileNotFoundError(
            f"Data directory not found: {cfg.DATA_DIR}\n"
            "Follow datasets/README.md to install KuaiRand-Pure at "
            f"{cfg.PROJECT_ROOT / 'datasets' / 'KuaiRand-Pure'}"
        )

    # Load authoritative scores from baseline_scores.json
    if cfg.BASELINE_JSON.exists():
        with open(cfg.BASELINE_JSON, encoding="utf-8") as fh:
            data = json.load(fh)
        scores = data.get("scores", {})
        fm = scores.get("fm_official", {})
        oracle = scores.get("oracle_ceiling", {})
        conv = data.get("convergence_rule", {})

        cfg.BASELINE_PRIMARY = fm.get("valid", {}).get("primary", cfg.BASELINE_PRIMARY)
        cfg.BASELINE_GAUC    = fm.get("valid", {}).get("GAUC",    cfg.BASELINE_GAUC)
        cfg.BASELINE_NDCG    = fm.get("valid", {}).get("nDCG@5",  cfg.BASELINE_NDCG)
        cfg.ORACLE_PRIMARY   = oracle.get("valid", {}).get("primary", cfg.ORACLE_PRIMARY)
        cfg.HEADROOM         = cfg.ORACLE_PRIMARY - cfg.BASELINE_PRIMARY
        cfg.CONVERGENCE_EPSILON = float(conv.get("epsilon", cfg.CONVERGENCE_EPSILON))
        cfg.CONVERGENCE_N       = int(conv.get("N",       cfg.CONVERGENCE_N))

    # Ensure output directories exist
    cfg.CANDIDATES_DIR.mkdir(parents=True, exist_ok=True)
    cfg.LOGS_DIR.mkdir(parents=True, exist_ok=True)

    # Resolve Python executable (prefer the repository-level virtual environment)
    venv_py = cfg.PROJECT_ROOT / ".venv" / (
        "Scripts/python.exe" if os.name == "nt" else "bin/python"
    )
    if venv_py.exists():
        cfg.PYTHON_EXE = str(venv_py)

    # Override model from env if set
    cfg.MODEL_ID = os.environ.get("ANTHROPIC_MODEL", cfg.MODEL_ID)

    return cfg


# Singleton — import and call once per process
_cfg: Config | None = None

def get_config() -> Config:
    global _cfg
    if _cfg is None:
        _cfg = load_config()
    return _cfg
