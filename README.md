# TikTokTechJam_STierCo

An **autonomous ML research agent** that tries to beat the official Factorization Machine
baseline on the [KuaiRand-Pure](https://kuairand.com) within-user ranking benchmark.

It runs an LLM-driven loop: propose a hypothesis, implement it as a self-contained
candidate model, train and score it against the fixed evaluation protocol, log the
result, and use tree search to decide what to try next — until it converges or runs out
of budget.

The code splits along one line: `research_agent/` decides what to investigate and try
next; `harness/` executes those decisions deterministically and records the evidence.

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
research_agent/                    agent reasoning — builder, strategist, tree search, method corpus
harness/                           deterministic runtime — orchestration, provider, tools, logging
tests/                             offline tests + the live builder smoke check
baseline_kuairand-starter-kit/     organiser starter kit — read-only reference
datasets/                          dataset instructions; downloaded data is gitignored
experiment_workspace/              generated trial code; local and gitignored
artifacts/                          tracked run evidence, results, reports, and final submission
requirements.txt                   Python dependencies for all supported LLM providers
SETUP.md                           full first-time setup guide
```

Each autonomous run writes disposable trial implementations to
`experiment_workspace/<run_id>/trial_NNN/`. Durable evidence is grouped under
`artifacts/runs/<run_id>/`, with raw JSONL events, machine-readable metrics, and a
human-readable summary. See `artifacts/README.md` for the promotion workflow.

---

## Harness at a glance

| Module | Role |
|---|---|
| `harness/main.py` | Orchestration loop — tree search, convergence check, budget enforcement |
| `research_agent/builder.py` | LLM session that implements one hypothesis as a candidate `model.py` and runs it |
| `research_agent/strategist.py` | Periodic LLM session that reviews progress and proposes new research directions |
| `research_agent/search/tree.py` | UCB tree search over hypotheses |
| `research_agent/knowledge/` | Local method corpus + offline BM25 `search_ml_literature` tool |
| `harness/provider.py` | Provider-agnostic LLM client (Anthropic / Groq / Gemini / Ollama / OpenAI) — switch via `.env` |
| `harness/logger.py` + `harness/validator.py` | Structured JSONL logging and schema validation |
| `harness/config.py` | Single source of runtime constants and paths |

Convergence rule (from the starter kit's 5-seed variance): ε = 0.002, N = 3 — three
consecutive iterations with ≤0.002 validation gain means stop.

---

## Quick start

See **[SETUP.md](SETUP.md)** for the full walkthrough (Windows + Mac/Linux), and
[datasets/README.md](datasets/README.md) for dataset details. In short:

```bash
python3 -m venv .venv
source .venv/bin/activate                            # .venv\Scripts\Activate.ps1 on Windows
python -m pip install -r requirements.txt
cp .env.example .env                                 # then add your API key

# download KuaiRand-Pure (~300MB) into datasets/KuaiRand-Pure/
curl -L https://zenodo.org/records/10439422/files/KuaiRand-Pure.tar.gz \
  -o datasets/KuaiRand-Pure.tar.gz
tar -xzf datasets/KuaiRand-Pure.tar.gz -C datasets

# offline checks — no API key, no dataset
python tests/test_knowledge.py

# verify the baseline reproduces (needs the dataset, ~40s)
python baseline_kuairand-starter-kit/baseline.py \
  --model fm --data_dir datasets/KuaiRand-Pure/data

# dev run of the harness (~30 min)
python -m harness.main --max-iter 10 --wall-hours 0.5 --builder-turns 2
```

---

## Known dead ends (measured by the organisers — don't re-test)

- Adding CWM's 13 static feature fields — no gain (0.5940 vs 0.5950)
- Larger embedding dim (k = 8/16/32) — flat; capacity is not the bottleneck
- Pure user-side first-order terms — contribute exactly zero (ranking is within-user)

## Where the headroom is

The agent is not handed a ranked list of things to try — that would make the harness the
researcher and the agent the typist. Instead `research_agent/knowledge/` ships a corpus of
19 established methods (ranking losses, sequence models, multi-task architectures,
watch-time and debiasing methods, GBDT ranking), each covering what it is, why it helps a
ranking metric, how to implement it, and **when it will not help**. The Strategist sees the
catalogue and chooses; its reasoning lands in the run log.

```bash
python -m research_agent.knowledge --list
python -m research_agent.knowledge "within-user negative sampling"
```

Retrieval is BM25 over Markdown sections — offline, deterministic, no embeddings or vector
store, so any passage cited in a run log can be reproduced exactly. See
`research_agent/knowledge/corpus/README.md` to add a method.
