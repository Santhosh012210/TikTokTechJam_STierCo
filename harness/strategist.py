"""Strategist session: single-turn LLM call that proposes research direction.

Provider-agnostic — all SDK calls go through harness.provider.LLMClient.
No tool calls — all context injected as text.
The `reasoning` field is logged verbatim (graded deliverable).
"""
import json
import re
import time
from dataclasses import dataclass

from harness.config import Config
from harness.provider import make_client
from harness.tree import Node

# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class StrategistResult:
    direction:    str
    reasoning:    str
    hypotheses:   list[str]    # exactly 3
    token_counts: dict
    wall_seconds: float
    raw_response: str
    error:        str | None


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _format_leaderboard(nodes: list[Node]) -> str:
    if not nodes:
        return "  (no successful trials yet)"
    return "\n".join(
        f"  {rank}. trial_{n.id:03d} | primary={n.primary:.4f} | {n.hypothesis[:80]}"
        for rank, n in enumerate(nodes, 1)
    )


def _format_history(history: list[dict]) -> str:
    if not history:
        return "  (no iterations yet)"
    return "\n".join(
        f"  trial_{h['iteration']:03d} [{h['status']:8s}] "
        f"primary={h['primary']:.4f if h.get('primary') is not None else 'failed'} "
        f"| {h['hypothesis'][:80]}"
        for h in reversed(history)
    )


# ---------------------------------------------------------------------------
# JSON extraction
# ---------------------------------------------------------------------------

def _extract_json(text: str) -> dict | None:
    fenced = re.search(r"```(?:json)?\s*([\s\S]+?)```", text)
    candidate = fenced.group(1).strip() if fenced else text.strip()
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{[\s\S]+\}", candidate)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    return None


# ---------------------------------------------------------------------------
# Session runner
# ---------------------------------------------------------------------------

def run_strategist_session(
    leaderboard: list[Node],
    hypothesis_history: list[dict],
    dead_ends: list[str],
    untried_directions: list[str],
    current_best: float,
    config: Config,
) -> StrategistResult:
    t0 = time.time()
    client = make_client()

    headroom_consumed = (
        (current_best - config.BASELINE_PRIMARY) / config.HEADROOM * 100
        if config.HEADROOM > 0 else 0.0
    )

    user_prompt = f"""You are a research strategist for an autonomous ML experiment search system.
Analyze the current search state and output a JSON object with the next research direction.

## Current state

Best valid primary so far : {current_best:.4f}
Baseline (FM)             : {config.BASELINE_PRIMARY:.4f}
Oracle ceiling            : {config.ORACLE_PRIMARY:.4f}
Headroom remaining        : {config.ORACLE_PRIMARY - current_best:.4f}
Headroom consumed         : {headroom_consumed:.1f}%

Note: nDCG@5 cannot exceed ~0.70 on valid (27.1% all-negative users score 0; 9.2% all-positive score 1).
Effective ceiling: nDCG=0.6968, primary=0.8484.

## Leaderboard (top {len(leaderboard)})
{_format_leaderboard(leaderboard)}

## Recent hypothesis history (last {len(hypothesis_history)}, newest first)
{_format_history(hypothesis_history)}

## Known dead ends — do NOT revisit
{chr(10).join(f'  - {d}' for d in dead_ends)}

## Untried directions (priority order)
{chr(10).join(f'  {d}' for d in untried_directions)}

## Current FM baseline config
  Fields: user_id, video_id, author_id, tab, dur_bucket
  Model : FM  k=16, lr=0.001, batch=8192, max_epochs=40, patience=4, pointwise BCE

## Output
Respond with ONLY a JSON object (no prose, no fences):

{{
  "direction": "<one sentence>",
  "reasoning": "<2-4 sentences: why this direction, what alternatives considered, what signal influenced the choice>",
  "hypotheses": ["<specific change 1>", "<specific change 2>", "<specific change 3>"]
}}

Hypotheses must be implementable in numpy only (no torch required).
"""

    messages: list[dict] = [{"role": "user", "content": user_prompt}]
    total_input = total_output = 0
    last_raw = ""
    last_error: str | None = None

    for attempt in range(config.STRATEGIST_MAX_TURNS):
        try:
            response = client.complete(messages, max_tokens=2048)
        except Exception as e:
            last_error = f"API error: {e}"
            break

        total_input  += response.input_tokens
        total_output += response.output_tokens
        raw = response.text or ""
        last_raw = raw

        parsed = _extract_json(raw)
        if parsed and all(k in parsed for k in ("direction", "reasoning", "hypotheses")):
            hypotheses = parsed.get("hypotheses", [])
            if not isinstance(hypotheses, list):
                hypotheses = [str(hypotheses)]
            while len(hypotheses) < 3:
                hypotheses.append(f"Variant of: {parsed.get('direction', '')}")
            return StrategistResult(
                direction=str(parsed.get("direction", "")),
                reasoning=str(parsed.get("reasoning", "")),
                hypotheses=hypotheses[:3],
                token_counts={"input": total_input, "output": total_output},
                wall_seconds=time.time() - t0,
                raw_response=raw,
                error=None,
            )

        last_error = f"Failed to parse JSON on attempt {attempt + 1}"
        client.add_response_to_history(messages, response)
        messages.append({
            "role": "user",
            "content": (
                "Your response could not be parsed as JSON. "
                "Reply with ONLY a valid JSON object matching the schema. No other text."
            ),
        })

    return StrategistResult(
        direction="Fallback: BPR pairwise loss (strategist parse failed)",
        reasoning=f"Strategist failed after {config.STRATEGIST_MAX_TURNS} attempts. Error: {last_error}",
        hypotheses=[
            "Replace pointwise BCE with BPR pairwise loss, sampling negatives within each user's batch",
            "Replace pointwise BCE with listwise softmax over each user's impressions per mini-batch",
            "Add DIN-style attention over user's recent 50 training interactions as sequence features",
        ],
        token_counts={"input": total_input, "output": total_output},
        wall_seconds=time.time() - t0,
        raw_response=last_raw,
        error=last_error,
    )
