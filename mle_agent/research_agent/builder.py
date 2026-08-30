"""Builder session: one LLM multi-turn loop per candidate.

Provider-agnostic — all SDK calls go through harness.provider.LLMClient.

Given a hypothesis and a parent code path, the builder:
  1. Reads the parent code.
  2. Writes experiment_workspace/<run_id>/trial_NNN/model.py via write_file tool.
  3. Runs it via run_bash tool.
  4. Self-repairs on error (up to max_turns).
  5. Returns structured BuilderResult.
"""
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from dataclasses import dataclass
from pathlib import Path

from mle_agent.harness.config import Config
from mle_agent.harness.metrics import try_extract_metrics
from mle_agent.harness.provider import LLMClient, LLMResponse, make_client
from mle_agent.harness.tools import BUILDER_TOOLS, dispatch_tool_call, redact_secrets

# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class BuilderResult:
    success:      bool
    final_code:   str | None
    metrics:      dict | None
    repair_count: int
    token_counts: dict
    wall_seconds: float
    error:        str | None
    stdout_raw:   str | None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ERROR_MARKERS = (
    "Traceback (most recent call last)",
    "Error:",
    "TIMEOUT after",
    "REJECTED:",
    "ModuleNotFoundError",
    "SyntaxError",
    "IndentationError",
    "ImportError",
)


def is_error_output(text: str) -> bool:
    return any(marker in text for marker in _ERROR_MARKERS)


def _builder_prompt(config: Config, hypothesis: str, parent_code_path: str, node_id: int) -> str:
    return f"""You are an ML research engineer implementing a hypothesis for the KuaiRand-Pure ranking benchmark.

HYPOTHESIS TO IMPLEMENT:
{hypothesis}

PARENT CODE (your starting point):
Read it from: {parent_code_path}

TRIAL DIRECTORY: trial_{node_id:03d}/
You will write your implementation there using write_file, then run it using run_bash.

════════════════════════════════════════════════════════════
MANDATORY model.py CONTRACT — violating any rule = failure
════════════════════════════════════════════════════════════
1. Start with EXACTLY these lines (before any other imports):
   import sys
   sys.path.insert(0, r'{config.BASELINE_ROOT}')
   from data import load, encode, FIELDS
   from evaluate import evaluate
   import json, argparse

2. Accept --data_dir argument:
   ap = argparse.ArgumentParser()
   ap.add_argument('--data_dir', default=r'{config.DATA_DIR}')
   a = ap.parse_args()
   splits = load(a.data_dir)

3. TRAIN ON splits['train'] ONLY.
   SCORE splits['valid'] ONLY.
   NEVER access splits['test'] — this will be detected and rejected.

4. Print EXACTLY ONE JSON line to stdout at the end:
   print(json.dumps(evaluate(users_valid, labels_valid, scores_valid)))

5. Use seed={config.SEED} for all randomness (numpy default_rng({config.SEED})).
6. Must complete in under 5 minutes.

════════════════════════════════════════════════
CONTEXT
════════════════════════════════════════════════
Current valid baseline primary : {config.BASELINE_PRIMARY:.4f}
Oracle ceiling (valid)          : {config.ORACLE_PRIMARY:.4f}
Remaining headroom              : {config.HEADROOM:.4f}

Current FM config: k=16, lr=0.001, batch=8192, max_epochs=40, patience=4, pointwise BCE loss
Current features : user_id, video_id, author_id, tab, dur_bucket

Known dead ends (do NOT retry):
- Adding more features (CWM 13-field set): no gain
- Increasing embedding dim k=8/16/32: flat
- Pure user-side first-order terms: zero contribution (ranking is within-user)

════════════════════════════════════════════════
WORKFLOW
════════════════════════════════════════════════
1. read_file the parent code.
2. write_file model.py with your implementation.
3. run_bash: {config.PYTHON_EXE} model.py --data_dir "{config.DATA_DIR}"
4. If it errors, read the traceback, fix model.py, run again.
5. When you get a clean JSON line in stdout, you are done.
"""


# ---------------------------------------------------------------------------
# Core session runner (runs in a thread, enforced by wall-clock timeout)
# ---------------------------------------------------------------------------

def _run_session(
    hypothesis: str,
    parent_code_path: str,
    candidate_dir: Path,
    config: Config,
    node_id: int,
) -> BuilderResult:
    t0 = time.time()
    client: LLMClient = make_client()

    messages: list[dict] = [
        {"role": "user", "content": _builder_prompt(config, hypothesis, parent_code_path, node_id)},
    ]

    total_input = total_output = 0
    repair_count = 0
    saw_error = False
    final_code: str | None = None
    metrics: dict | None = None
    last_error: str | None = None
    last_stdout: str | None = None

    for _turn in range(config.BUILDER_MAX_TURNS):
        try:
            response: LLMResponse = client.complete(
                messages, tools=BUILDER_TOOLS, max_tokens=4096
            )
        except Exception as e:
            last_error = f"API error: {e}"
            break

        total_input  += response.input_tokens
        total_output += response.output_tokens

        client.add_response_to_history(messages, response)

        if response.stop_reason == "end_turn":
            break

        if response.stop_reason == "tool_use":
            outputs: list[str] = []
            for tc in response.tool_calls:
                raw_output = dispatch_tool_call(
                    tc.name, tc.input, candidate_dir, config.BASELINE_ROOT
                )

                if tc.name == "run_bash":
                    if saw_error:
                        repair_count += 1       # re-run after a prior error = repair
                    if is_error_output(raw_output):
                        saw_error = True
                        last_error = raw_output[:500]
                    else:
                        last_stdout = raw_output
                        found = try_extract_metrics(raw_output)
                        if found:
                            metrics = found
                            mp = candidate_dir / "model.py"
                            if mp.exists():
                                final_code = mp.read_text(encoding="utf-8")

                elif tc.name == "write_file":
                    if "model.py" in tc.input.get("path", ""):
                        final_code = tc.input.get("content")

                outputs.append(redact_secrets(raw_output))

            client.add_tool_results_to_history(messages, response.tool_calls, outputs)
        else:
            last_error = f"Unexpected stop_reason: {response.stop_reason}"
            break

    return BuilderResult(
        success=metrics is not None,
        final_code=final_code,
        metrics=metrics,
        repair_count=repair_count,
        token_counts={"input": total_input, "output": total_output},
        wall_seconds=time.time() - t0,
        error=last_error if metrics is None else None,
        stdout_raw=last_stdout,
    )


# ---------------------------------------------------------------------------
# Public entry point (with wall-clock timeout)
# ---------------------------------------------------------------------------

def run_builder_session(
    hypothesis: str,
    parent_code_path: str,
    candidate_dir: Path,
    config: Config,
    node_id: int,
) -> BuilderResult:
    candidate_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(
            _run_session, hypothesis, parent_code_path, candidate_dir, config, node_id,
        )
        try:
            return future.result(timeout=config.BUILDER_WALL_TIMEOUT_S)
        except FuturesTimeout:
            future.cancel()
            return BuilderResult(
                success=False, final_code=None, metrics=None, repair_count=0,
                token_counts={"input": 0, "output": 0},
                wall_seconds=time.time() - t0,
                error=f"wall-clock timeout ({config.BUILDER_WALL_TIMEOUT_S}s)",
                stdout_raw=None,
            )
