# TikTokTechJam_STierCo

An **autonomous ML research agent** that tries to beat the official Factorization Machine
baseline on the [KuaiRand-Pure](https://kuairand.com) within-user ranking benchmark.

It runs one persistent Google ADK research session across the complete MLE loop: understand
the benchmark, inspect train/validation data, research and propose a hypothesis,
implement a self-contained candidate model, train and score it, repair failures, and
reflect before the next experiment — until it converges or runs out of budget.

All agent code is grouped under the local `mle_agent/` namespace; no installation is
required. `mle_agent/research_agent/` decides what to investigate and try next, while
`mle_agent/harness/` executes those decisions deterministically and records the evidence.

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
mle_agent/
  research_agent/                  persistent reasoning, prompts, and method corpus
  harness/                         deterministic runtime, safety, convergence, evidence
  tests/                           offline, ADK, and recovery tests
  scripts/                         run, verify, recovery-demo, and finalization commands
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
Before any agent call, the runner creates a count-verified, run-local data view containing
exactly the official train and validation date ranges; hidden-test rows and the randomized
log are not exposed to candidate processes.

---

## Harness at a glance

| Module | Role |
|---|---|
| `mle_agent/harness/agent_main.py` | Single-agent run entrypoint — baseline, budgets, convergence, selection, evidence |
| `mle_agent/research_agent/adk_agent.py` | Persistent Google ADK session, recovery loop, compact experiment memory |
| `mle_agent/harness/agent_tools.py` | Constrained tools — train/valid EDA, literature, file editing, model execution |
| `mle_agent/harness/hooks.py` | Immediate syntax check; failed saves gate execution until repaired |
| `mle_agent/harness/main.py` + Builder/Strategist | Legacy comparison path; not used for the submission run |
| `mle_agent/research_agent/knowledge/` | Offline BM25 method corpus |
| `mle_agent/harness/logger.py` + `validator.py` | Strict v2 experiment evidence and validation |
| `mle_agent/harness/finalize.py` | Trusted final promotion and submission-alignment check |

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

# offline checks + Starter Kit integrity — no API key, no dataset
./mle_agent/scripts/test_offline.sh

# deterministic syntax/runtime recovery demonstration
./mle_agent/scripts/demo_recovery.sh

# verify the baseline reproduces (needs the dataset, ~40s)
python baseline_kuairand-starter-kit/baseline.py \
  --model fm --data_dir datasets/KuaiRand-Pure/data

# single-agent dev run (3 experiments, up to 30 min)
./mle_agent/scripts/run_agent.sh

# one-experiment smoke run (unlimited ADK calls, up to 30 min)
./mle_agent/scripts/run_agent_once.sh

# optionally impose explicit call caps when diagnosing a runaway prompt/tool loop
AGENT_MAX_ITER=5 AGENT_MAX_TURNS=20 AGENT_BOOTSTRAP_MAX_TURNS=20 ./mle_agent/scripts/run_agent.sh

# official-budget run after smoke verification and metric confirmation
TASK_DEFINITION_CONFIRMED=1 ./mle_agent/scripts/run_official.sh
```

Before experiment 1, the ADK agent completes a separate uncapped bootstrap phase. It discovers the
starter-kit task documentation, reads the complete README, `baseline.py`, `evaluate.py`,
`data.py`, `ablation_features.py`, and inherited candidate through explicit pages, inspects
the train/validation-only data view and its available columns, searches the local literature
corpus, explicitly reproduces the official baseline, and records one structured task summary.
That summary must record the five baseline fields and the organizer's measured no-gain result
for all 13 static fields. The harness blocks edits until this bootstrap is complete; the summary and
baseline result then stay in the same conversation for every later experiment. Model-call
caps default to zero (unlimited); positive `AGENT_BOOTSTRAP_MAX_TURNS` or
`AGENT_MAX_TURNS` values are optional diagnostic safeguards.

Feature engineering reads the filtered `experiment_workspace/<run_id>/candidate_data` through
`--data_dir`; it never rewrites those CSVs or the organizer starter kit. The agent implements the
complete feature pipeline in the trial's self-contained `model.py`. Feature-oriented runs must log
their source columns, exact transformations, and train-only leakage controls before the harness will
execute them. This supports history/sequence, temporal, auxiliary-signal, watch-time, aggregate,
and user-item-cross experiments without repeating the measured static-field ablation.

When Gemini returns a real quota/rate-limit response, an interactive run asks once whether
to resume after reset. Choosing `y` waits for the provider-advertised retry interval and
automatically resumes the same ADK session. Later quota pauses in that run need no further
human confirmation.

---

## Known dead ends (measured by the organisers — don't re-test)

- Adding CWM's 13 static feature fields — no gain (0.5940 vs 0.5950)
- Larger embedding dim (k = 8/16/32) — flat; capacity is not the bottleneck
- Pure user-side first-order terms — contribute exactly zero (ranking is within-user)

## Where the headroom is

The agent is not handed a ranked list of things to try — that would make the harness the
researcher and the agent the typist. Instead `mle_agent/research_agent/knowledge/` ships a corpus of
19 established methods (ranking losses, sequence models, multi-task architectures,
watch-time and debiasing methods, GBDT ranking), each covering what it is, why it helps a
ranking metric, how to implement it, and **when it will not help**. The persistent single
agent searches this catalogue, cites retrieved chunks in its experiment reasoning, and
keeps that evidence beside the task summary in the run log.

```bash
python -m mle_agent.research_agent.knowledge --list
python -m mle_agent.research_agent.knowledge "within-user negative sampling"
```

Retrieval is BM25 over Markdown sections — offline, deterministic, no embeddings or vector
store, so any passage cited in a run log can be reproduced exactly. See
`mle_agent/research_agent/knowledge/corpus/README.md` to add a method.

## Evidence, recovery, and finalization

Each schema-v2 experiment row records the hypothesis, reasoning, exact diff, metrics,
execution attempts, errors, recovery outcomes, reflection, next direction, tokens, wall
time, and intervention flag. Run summaries add the explicit stop reason, convergence
history, per-metric deltas, provider, GPU-hours, and validation-best candidate.

`./mle_agent/scripts/demo_recovery.sh` reproducibly drives the real persistent ADK tool
loop through an invalid save, a blocked execution, a runtime traceback, and a successful
repair using a deterministic fake provider. Its sanitized evidence is written to
`artifacts/demo/recovery/`.

After an official run converges (or honestly reaches the published hard budget), promote
its frozen validation-best candidate with:

```bash
./mle_agent/scripts/finalize.sh \
  --run-id <run_id> \
  --task-definition-confirmed
```

The confirmation flag is deliberate because the supplied prose conflicts with the
checked-in Starter Kit about the label and metrics. The trusted finalizer generates test
predictions, runs `submit.py --check` for schema/alignment, and never scores hidden labels.
