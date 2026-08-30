"""Live smoke test: one Builder session end-to-end against a real provider.

NOT a unit test. This makes real LLM calls and trains FM twice, so it costs
tokens and a few minutes. Named without a ``test_`` prefix so it is never
collected by a test runner — run it by hand before a long autonomous run.

Requires a configured provider in ``.env`` and the KuaiRand-Pure dataset.

Checks:
  1. Broken hypothesis  -> repair_count >= 1 (the session recovers from an error)
  2. Trivial hypothesis -> success=True, 0.55 < primary < 0.65

Run from the project root:
  python -m mle_agent.tests.live_builder_smoke
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mle_agent.harness.config import load_config
from mle_agent.research_agent.builder import run_builder_session

def main():
    config = load_config()
    config.BUILDER_MAX_TURNS = 2        # dev: 1 attempt + 1 repair
    config.BUILDER_WALL_TIMEOUT_S = 300.0

    print("=" * 60)
    print("TEST 1: Broken hypothesis (expect repair_count >= 1)")
    print("=" * 60)
    test_workspace = config.EXPERIMENT_WORKSPACE_DIR / "dev_tests"
    broken_dir = test_workspace / "test_broken"
    broken_dir.mkdir(parents=True, exist_ok=True)

    # Give it the root model.py as parent
    root_mp = test_workspace / "trial_000" / "model.py"
    parent = str(root_mp) if root_mp.exists() else str(config.BASELINE_ROOT / "baseline.py")

    result1 = run_builder_session(
        hypothesis=(
            "Intentionally broken implementation: call an undefined_function() "
            "in the training loop to trigger a NameError. "
            "Then fix it by removing the call and using the FM baseline loss."
        ),
        parent_code_path=parent,
        candidate_dir=broken_dir,
        config=config,
        node_id=999,
    )
    print(f"  success={result1.success}")
    print(f"  repair_count={result1.repair_count}")
    print(f"  metrics={result1.metrics}")
    print(f"  error={result1.error}")
    assert result1.repair_count >= 1 or result1.success, (
        f"FAIL: repair_count={result1.repair_count} and success={result1.success}. "
        "Expected at least one repair attempt or self-repair."
    )
    print("PASS: repair_count >= 1 OR model self-repaired to success\n")

    print("=" * 60)
    print("TEST 2: Trivial hypothesis — change k from 16 to 8")
    print("=" * 60)
    trivial_dir = test_workspace / "test_trivial"
    trivial_dir.mkdir(parents=True, exist_ok=True)

    result2 = run_builder_session(
        hypothesis=(
            "Change FM embedding dimension from k=16 to k=8. "
            "This is a trivial change — just modify the k parameter in the FM constructor. "
            "Everything else stays the same as the parent code."
        ),
        parent_code_path=parent,
        candidate_dir=trivial_dir,
        config=config,
        node_id=998,
    )
    print(f"  success={result2.success}")
    print(f"  repair_count={result2.repair_count}")
    print(f"  metrics={result2.metrics}")
    print(f"  error={result2.error}")

    assert result2.success, f"FAIL: trivial hypothesis failed. error={result2.error}"
    primary = result2.metrics["primary"]
    assert 0.55 < primary < 0.65, f"FAIL: primary={primary:.4f} outside expected [0.55, 0.65]"
    print(f"PASS: success=True, primary={primary:.4f} is in [0.55, 0.65]\n")

    print("=" * 60)
    print("ALL STEP 2 TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
