# TikTokTechJam_STierCo

An **autonomous ML research agent** that tries to beat the official Factorization Machine
baseline on the [KuaiRand-Pure](https://kuairand.com) within-user ranking benchmark.

The harness runs an LLM-driven loop: it proposes a hypothesis, implements it as a
self-contained candidate model, trains and scores it against the fixed evaluation
protocol, logs the result, and uses tree search to decide what to try next — until it
converges or runs out of budget.

---

## Task

Fixed by the organiser starter kit (`baseline_kuairand-starter-kit/`) — the protocol is not ours to change:

| | |
|---|---|
| **Problem** | Within-user ranking — rank each user's own impressions, no full-corpus retrieval |
| **Label** | `long_view` (native 0/1 column) |
| **Metric** | mean of `GAUC` and `nDCG@5` ("primary") |
| **Splits** | train `04-08–04-21` / valid `04-22–04-28` / test `04-29–05-08` |
| **Baseline to beat** | FM — test primary **0.5946** (valid **0.6016**) |
| **Ceiling** | oracle primary **0.8645** — real headroom is ~0.27, not ~0.41 |

Full conventions live in `baseline_kuairand-starter-kit/evaluate.py` and its README.

---

## Repository layout

```
agent_harness/                     autonomous research agent (orchestration, builder, strategist, tree search)
baseline_kuairand-starter-kit/     organiser starter kit — read-only reference
datasets/                          dataset instructions; downloaded data is gitignored
candidates/               one folder per experiment (auto-created)
logs/                     JSONL run logs (auto-created)
requirements.txt          Python dependencies for all supported LLM providers
SETUP.md                  full first-time setup guide
```

---

## Harness at a glance

| Module | Role |
|---|---|
| `agent_harness/main.py` | Orchestration loop — tree search, convergence check, budget enforcement |
| `agent_harness/builder.py` | LLM session that implements one hypothesis as a candidate `model.py` and runs it |
| `agent_harness/strategist.py` | Periodic LLM session that reviews progress and proposes new research directions |
| `agent_harness/tree.py` | UCB tree search over hypotheses |
| `agent_harness/provider.py` | Provider-agnostic LLM client (Anthropic / Groq / Gemini / Ollama / OpenAI) — switch via `.env` |
| `agent_harness/logger.py` + `agent_harness/validator.py` | Structured JSONL logging and schema validation |
| `agent_harness/config.py` | Single source of runtime constants and paths |

Convergence rule (from the starter kit's 5-seed variance): ε = 0.002, N = 3 — three
consecutive iterations with ≤0.002 validation gain means stop.

---

## Quick start

See **[SETUP.md](SETUP.md)** for the full walkthrough (Windows + Mac/Linux). In short:

```bash
python3 -m venv .venv
source .venv/bin/activate                            # .venv\Scripts\Activate.ps1 on Windows
python -m pip install -r requirements.txt
cp .env.example .env                                 # then add your API key

# verify the baseline reproduces (run from repo root)
python baseline_kuairand-starter-kit/baseline.py \
  --model fm --data_dir datasets/KuaiRand-Pure/data

# dev run of the harness (~30 min)
python -m agent_harness.main --max-iter 10 --wall-hours 0.5 --builder-turns 2
```

---

## Known dead ends (don't re-test)

- Adding CWM's 13 static feature fields — no gain (0.5940 vs 0.5950)
- Larger embedding dim (k = 8/16/32) — flat; capacity is not the bottleneck
- Pure user-side first-order terms — contribute exactly zero (ranking is within-user)

## Untried directions (where the headroom should be)

Loss function alignment (BPR / listwise softmax-per-user) · user history sequence
modelling (DIN/SIM) · multi-task auxiliary heads · watch-time censored regression ·
temporal / distribution-drift features · unbiased validation on the random-exposure log.
