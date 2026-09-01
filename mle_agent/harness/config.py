"""Single authoritative source for all runtime constants.

Loads baseline_scores.json, resolves absolute paths, reads .env file.
No logic beyond loading and validation.
"""
import json
import os
import shlex
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
        try:
            # Support shell-style quoted values and ignore unquoted inline comments.
            parts = shlex.split(v.strip(), comments=True, posix=True)
        except ValueError as exc:
            raise ValueError(f"Invalid .env value for {k}: {exc}") from exc
        if not parts:
            continue
        v = " ".join(parts)
        if k not in os.environ:          # env var takes precedence over .env
            os.environ[k] = v


# Repository root (contains mle_agent/, baseline_kuairand-starter-kit/,
# datasets/, experiment_workspace/, and artifacts/).
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Load .env before anything reads os.environ
_load_dotenv(_PROJECT_ROOT / ".env")

# ---------------------------------------------------------------------------
# Config class
# ---------------------------------------------------------------------------

class Config:
    # Paths
    PROJECT_ROOT:             Path = _PROJECT_ROOT
    BASELINE_ROOT:            Path = _PROJECT_ROOT / "baseline_kuairand-starter-kit"
    DATA_DIR:                 Path = _PROJECT_ROOT / "datasets" / "KuaiRand-Pure" / "data"
    EXPERIMENT_WORKSPACE_DIR: Path = _PROJECT_ROOT / "experiment_workspace"
    ARTIFACTS_DIR:            Path = _PROJECT_ROOT / "artifacts"
    HARNESS_DIR:              Path = _PROJECT_ROOT / "mle_agent" / "harness"
    BASELINE_JSON:            Path = BASELINE_ROOT / "baseline_scores.json"

    # Loaded from baseline_scores.json
    BASELINE_PRIMARY:     float = 0.6016
    BASELINE_GAUC:        float = 0.6674
    BASELINE_NDCG:        float = 0.5357
    BASELINE_TEST_PRIMARY: float = 0.5946
    BASELINE_TEST_GAUC:    float = 0.6610
    BASELINE_TEST_NDCG:    float = 0.5282
    ORACLE_PRIMARY:       float = 0.8484
    # Reference rungs from the starter kit, used to tell an idea that lost from an
    # implementation that never worked. A candidate with the baseline's features
    # plus more that cannot beat "rank by global item popularity" has a bug, not a
    # refuted hypothesis.
    POPULARITY_PRIMARY:   float = 0.5807
    RANDOM_PRIMARY:       float = 0.4834
    POPULARITY_TEST_PRIMARY: float = 0.5715
    RANDOM_TEST_PRIMARY:     float = 0.4753
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

    # Conservative provider-agnostic request and read limits.
    # On thinking models max_output_tokens covers thinking tokens too, so a small
    # cap can be spent before the model emits a tool call.
    AGENT_MAX_OUTPUT_TOKENS:      int   = int(os.environ.get("AGENT_MAX_OUTPUT_TOKENS", "8192"))
    AGENT_REFLECTION_MAX_TOKENS:  int   = int(os.environ.get("AGENT_REFLECTION_MAX_TOKENS", "768"))
    AGENT_READ_MAX_CHARS:         int   = int(os.environ.get("AGENT_READ_MAX_CHARS", "6000"))
    AGENT_BOOTSTRAP_MAX_TURNS:    int   = int(
        os.environ.get("AGENT_BOOTSTRAP_MAX_TURNS", "24")
    )
    AGENT_MAX_TURNS:              int   = int(
        os.environ.get("AGENT_MAX_TURNS", "16")
    )
    # Pre-approve bounded quota recovery for unattended runs. Without it an
    # unattended run stops at the first per-minute rate limit, and the interactive
    # prompt itself counts as a manual intervention.
    AGENT_AUTO_RESUME_QUOTA: bool = os.environ.get(
        "AGENT_AUTO_RESUME_QUOTA", ""
    ).strip().lower() in {"1", "true", "yes"}
    AGENT_MAX_QUOTA_RESUMES:      int   = int(
        os.environ.get("AGENT_MAX_QUOTA_RESUMES", "3")
    )
    AGENT_MAX_QUOTA_WAIT_S:       float = float(
        os.environ.get("AGENT_MAX_QUOTA_WAIT_S", "300")
    )
    AGENT_EXPERIMENT_MEMORY_LIMIT: int  = int(
        os.environ.get("AGENT_EXPERIMENT_MEMORY_LIMIT", "8")
    )
    AGENT_MODEL: str = os.environ.get(
        "AGENT_MODEL", "openai:gpt-5.6-terra"
    ).strip()
    AGENT_REASONING_EFFORT: str = os.environ.get(
        "AGENT_REASONING_EFFORT", "low"
    ).strip().lower()
    AGENT_MODEL_TIMEOUT_S: float = float(
        os.environ.get("AGENT_MODEL_TIMEOUT_S", "120")
    )
    # The harness owns retry (ResearchAgent._complete_with_one_retry), which logs
    # every attempt and bills it against AGENT_MAX_RUN_COST_USD. The provider
    # client's own retry is disabled, so this is deliberately not passed through.
    AGENT_MAX_RUN_COST_USD: float = float(
        os.environ.get("AGENT_MAX_RUN_COST_USD", "1.00")
    )
    PROVIDER_RETRY_DELAY_S:       float = float(os.environ.get("PROVIDER_RETRY_DELAY_S", "1"))
    RATE_LIMIT_RETRY_DELAY_S:     float = float(os.environ.get("RATE_LIMIT_RETRY_DELAY_S", "60"))

    SEED:    int   = 0
    UCB_C:   float = 1.414
    PYTHON_EXE: str = sys.executable
    BASE_PYTHON_EXE: str | None = None
    RUN_ENV_DIR: Path | None = None
    RUN_ENV_ARTIFACT_DIR: Path | None = None
    RUN_RESEARCH_DIR: Path | None = None

    # Phase-aware conversation memory. The budget is the point at which the
    # harness compacts bootstrap traffic into structured state; the tail budget
    # bounds how much recent experiment conversation survives that compaction.
    # Both are estimates in the cost gate's conservative 3-chars-per-token units,
    # not exact tokenizer counts.
    AGENT_CONTEXT_TOKEN_BUDGET: int = int(
        os.environ.get("AGENT_CONTEXT_TOKEN_BUDGET", "60000")
    )
    AGENT_EXPERIMENT_TAIL_TOKEN_BUDGET: int = int(
        os.environ.get("AGENT_EXPERIMENT_TAIL_TOKEN_BUDGET", "24000")
    )
    # Run the deterministic bootstrap (discovery, required reads, EDA, environment
    # inventory, baseline reproduction) in Python before the first model call.
    AGENT_BOOTSTRAP_PREFETCH: bool = os.environ.get(
        "AGENT_BOOTSTRAP_PREFETCH", "1"
    ).strip().lower() not in {"0", "false", "no"}

    # Trusted research-quality controls.
    EDA_QUERY_CACHE_LIMIT: int = int(os.environ.get("EDA_QUERY_CACHE_LIMIT", "64"))
    AGENT_NORMAL_EXECUTION_TIMEOUT_S: int = int(
        os.environ.get("AGENT_NORMAL_EXECUTION_TIMEOUT_S", "600")
    )
    AGENT_SUBSTANTIAL_EXECUTION_TIMEOUT_S: int = int(
        os.environ.get("AGENT_SUBSTANTIAL_EXECUTION_TIMEOUT_S", "1800")
    )
    AGENT_QUICK_EXECUTION_TIMEOUT_S: int = int(
        os.environ.get("AGENT_QUICK_EXECUTION_TIMEOUT_S", "120")
    )
    # Wall-clock kept back from the last execution so finalization can still run.
    AGENT_WALL_RESERVE_S: int = int(os.environ.get("AGENT_WALL_RESERVE_S", "300"))
    AGENT_SWEEP_MAX_CONFIGS: int = int(
        os.environ.get("AGENT_SWEEP_MAX_CONFIGS", "6")
    )
    # Bounded sweeps are off unless explicitly enabled for a run. When unset,
    # run_sweep is never registered as a tool and the accounting bridge is dead.
    AGENT_ENABLE_SWEEPS: bool = os.environ.get(
        "AGENT_ENABLE_SWEEPS", ""
    ).strip().lower() in {"1", "true", "yes"}
    AGENT_CANDIDATE_ARCHIVE_LIMIT: int = int(
        os.environ.get("AGENT_CANDIDATE_ARCHIVE_LIMIT", "3")
    )
    # Match the organizer's published five-seed panel exactly.
    AGENT_STABILITY_SEEDS: tuple[int, ...] = (0, 1, 2, 3, 4)
    # Research-frontier policy. The first four trials branch from the reproduced
    # baseline across independent components; later trials select among the
    # strongest few frozen nodes with a noise-scaled UCB bonus.
    AGENT_FRONTIER_BRANCH_TRIALS: int = int(
        os.environ.get("AGENT_FRONTIER_BRANCH_TRIALS", "4")
    )
    AGENT_FRONTIER_TOP_K: int = int(os.environ.get("AGENT_FRONTIER_TOP_K", "3"))
    AGENT_FRONTIER_BRANCH_COMPONENTS: tuple[str, ...] = (
        "loss", "features", "model", "sequence"
    )
    AGENT_MIN_EXPERIMENTS_BEFORE_CONVERGENCE: int = int(
        os.environ.get("AGENT_MIN_EXPERIMENTS_BEFORE_CONVERGENCE", "8")
    )

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
        cfg.BASELINE_TEST_PRIMARY = fm.get("test", {}).get(
            "primary", cfg.BASELINE_TEST_PRIMARY
        )
        cfg.BASELINE_TEST_GAUC = fm.get("test", {}).get(
            "GAUC", cfg.BASELINE_TEST_GAUC
        )
        cfg.BASELINE_TEST_NDCG = fm.get("test", {}).get(
            "nDCG@5", cfg.BASELINE_TEST_NDCG
        )
        cfg.ORACLE_PRIMARY   = oracle.get("valid", {}).get("primary", cfg.ORACLE_PRIMARY)
        popularity = scores.get("item_popularity", {})
        random_rung = scores.get("random", {})
        cfg.POPULARITY_PRIMARY = popularity.get("valid", {}).get(
            "primary", cfg.POPULARITY_PRIMARY
        )
        cfg.RANDOM_PRIMARY = random_rung.get("valid", {}).get(
            "primary", cfg.RANDOM_PRIMARY
        )
        cfg.POPULARITY_TEST_PRIMARY = popularity.get("test", {}).get(
            "primary", cfg.POPULARITY_TEST_PRIMARY
        )
        cfg.RANDOM_TEST_PRIMARY = random_rung.get("test", {}).get(
            "primary", cfg.RANDOM_TEST_PRIMARY
        )
        cfg.HEADROOM         = cfg.ORACLE_PRIMARY - cfg.BASELINE_PRIMARY
        cfg.CONVERGENCE_EPSILON = float(conv.get("epsilon", cfg.CONVERGENCE_EPSILON))
        cfg.CONVERGENCE_N       = int(conv.get("N",       cfg.CONVERGENCE_N))

    # Ensure generated workspace and durable artifact roots exist.
    cfg.EXPERIMENT_WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)
    cfg.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    # Resolve Python executable (prefer the repository-level virtual environment)
    venv_py = cfg.PROJECT_ROOT / ".venv" / (
        "Scripts/python.exe" if os.name == "nt" else "bin/python"
    )
    if venv_py.exists():
        cfg.PYTHON_EXE = str(venv_py)

    return cfg


# Singleton — import and call once per process
_cfg: Config | None = None

def get_config() -> Config:
    global _cfg
    if _cfg is None:
        _cfg = load_config()
    return _cfg
