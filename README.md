# TikTokTechJam_STierCo

An **autonomous ML research agent** that tries to beat the official baseline on the [KuaiRand-Pure](https://kuairand.com) within-user ranking benchmark.

It runs one persistent LangChain-backed research session across the complete MLE loop:
1. understand the benchmark
2. inspect train/validation data
3. research and propose a hypothesis
4. implement a self-contained candidate model, train and score it, repair failures, and
reflect before the next experiment — until it converges or runs out of budget.

All agent code is grouped under the local `mle_agent/` namespace. `mle_agent/research_agent/` decides what to investigate and try next, while `mle_agent/harness/` executes those decisions deterministically and records the evidence.

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

## Results

One autonomous run (`20260901_012832`) produced the submitted model. It stopped on the
convergence rule rather than on budget at ~21 minutes wall clock, 2.84M input / 29.6K output tokens.

**Winning candidate** — a DCN-V2 (embedding dim 16, 2 cross layers, 96-unit deep tower,
AdamW, early stopping on validation primary) over the starter-kit fields plus one
**hour-of-day context field** derived from the log's `hourmin` column, with its vocabulary
fitted on train only. The hour field is the reason it can help: it varies *within* a user's
impression list, so unlike the static user fields the organisers already measured as flat, it
can change intra-user order.

| Validation metric | Official FM baseline | Final candidate | Delta |
|---|---:|---:|---:|
| GAUC | 0.6674 | 0.6706 | **+0.0032** |
| nDCG@5 | 0.5357 | 0.5376 | **+0.0019** |
| primary | 0.6016 | 0.6041 | **+0.0025** |

Re-scored on five fixed seeds, the winner holds at primary **0.604293 ± 0.000198** — the
spread is 0.2x the published FM 5-seed std of 0.0008, so the +0.0025 gain is roughly 3x seed
noise rather than a lucky draw.

**Hidden test is deliberately unmeasured.** The agent never sees test rows or test metrics;
the trusted finalizer generated row-aligned predictions (170,588 rows, format and alignment
validated) without scoring them. The published FM hidden-test primary is 0.5946; whether we
beat it is for the organisers' evaluation to say, and we do not claim it here.

Everything above is reproduced from tracked evidence in `artifacts/final/` — `model.py`
(the exact frozen source), `metrics.json`, `submission.csv`, `final-report.md`, the resolved
`requirements.lock.txt`, and the complete per-iteration run log under `artifacts/final/run/`.

### Reproducing the reported result

Two levels, depending on how much you want to re-derive.

**Re-score the submitted model** (deterministic, ~minutes; needs the dataset and the install
from *Setup and run* below):

```bash
python artifacts/final/model.py \
  --data_dir datasets/KuaiRand-Pure/data \
  --seed 0 \
  --prediction-path valid_preds.csv
```

It prints the validation primary it reached; that should be 0.6041 for seed 0, and the five
values listed in `artifacts/final/final-report.md` for seeds 0–4. The frozen file's built-in
defaults are the exact promoted config (also recorded as `source_trial_config` in
`artifacts/final/metrics.json`), so no `--trial-config` is needed. It is a frozen run artifact
and carries an absolute path to this repository's starter kit on line 3 — point that at your
own checkout if you cloned elsewhere.

**Re-run the autonomous search end to end** (needs an API key; a new run explores its own
trajectory and will not land on an identical model — the harness is reproducible, the
research is not):

```bash
./mle_agent/scripts/test_offline.sh                        # offline checks, no key or data
python baseline_kuairand-starter-kit/baseline.py \
  --model fm --data_dir datasets/KuaiRand-Pure/data        # must be within ±0.002 of 0.6016
./mle_agent/scripts/run_agent_once.sh                      # one-experiment provider smoke run
TASK_DEFINITION_CONFIRMED=1 ./mle_agent/scripts/run_official.sh
./mle_agent/scripts/finalize.sh --run-id <run_id> --task-definition-confirmed
```

---

## Repository layout

```
mle_agent/
  research_agent/                  persistent reasoning, prompts, and method corpus
  harness/                         deterministic runtime, safety, convergence, evidence
  tests/                           offline, memory/prefetch, and recovery tests
  scripts/                         run, verify, recovery-demo, and finalization commands
baseline_kuairand-starter-kit/     organiser starter kit — read-only reference
datasets/                          dataset instructions; downloaded data is gitignored
experiment_workspace/              frozen candidate trees/champions; local and gitignored
artifacts/                          local run evidence plus tracked promoted final submission
requirements.txt                   Python dependencies for all supported LLM providers
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
| `mle_agent/research_agent/agent.py` | Persistent research session, recovery loop, phase-aware memory |
| `mle_agent/research_agent/prompts/prior_findings.md` | Reviewed agent-facing knowledge: improvements and scoped negative results |
| `mle_agent/research_agent/experiment_history.py` | Operational score ledger and exact-fingerprint repeat gate |
| `mle_agent/harness/memory.py` | Phase-aware context compaction policy |
| `mle_agent/harness/bootstrap_prefetch.py` | Deterministic bootstrap and source curation |
| `mle_agent/harness/tool_schemas.py` | Pydantic argument models; the single tool-schema source |
| `mle_agent/harness/agent_tools.py` | Constrained tools — train/valid EDA, literature, file editing, model execution |
| `mle_agent/harness/hooks.py` | Immediate syntax check; failed saves gate execution until repaired |
| `mle_agent/harness/main.py` + Builder/Strategist | Legacy comparison path; not used for the submission run |
| `mle_agent/research_agent/knowledge/` | Offline BM25 method corpus |
| `mle_agent/harness/logger.py` + `validator.py` | Strict v2 experiment evidence and validation |
| `mle_agent/harness/finalize.py` | Trusted final promotion and submission-alignment check |

Candidate search is an immutable tree: the first four trials branch from baseline across loss,
features, model, and sequence; later trials choose among the conservative top three with a
noise-scaled UCB score. Every node freezes source, config, seed, metrics, and parent, and visit/
reward updates propagate through its lineage. Convergence (ε = 0.002, N = 3) is evaluated over
the whole top frontier only after at least eight scored variants. Final selection uses multi-seed
mean minus standard deviation and can never fall through to the latest working file.

---

## Setup and run

Prerequisites: Python 3.10+, internet access for initial package/data downloads, and an API
key for one audited model provider. Run all commands from the repository root.

### 1. Install

```bash
python3 -m venv .venv
source .venv/bin/activate                 # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
cp .env.example .env
```

If PowerShell blocks activation, run
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once for your user account.

### 2. Configure a provider

Edit `.env` and uncomment one audited autonomous-runner configuration:

| Provider | `AGENT_MODEL` | API-key variable |
|---|---|---|
| OpenAI (recommended) | `openai:gpt-5.6-terra` | `OPENAI_API_KEY` |
| Anthropic | `anthropic:claude-sonnet-4-5-20250929` | `ANTHROPIC_API_KEY` |
| Google GenAI | `google_genai:gemini-2.5-pro` | `GOOGLE_API_KEY` |

Only providers and model IDs with audited capabilities and token prices in
`mle_agent/harness/provider.py` are accepted. The autonomous runner deliberately does not treat
an arbitrary OpenAI-compatible endpoint as equivalent. The separate legacy comparison runner can
use the configured OpenAI, Groq, Gemini, and Ollama compatibility endpoints documented in
`.env.example`. `.env` is gitignored; never commit API keys.

### 3. Download the data

KuaiRand-Pure is about 300 MB and requires no registration:

```bash
curl -L https://zenodo.org/records/10439422/files/KuaiRand-Pure.tar.gz \
  -o datasets/KuaiRand-Pure.tar.gz
tar -xzf datasets/KuaiRand-Pure.tar.gz -C datasets
ls datasets/KuaiRand-Pure/data/*.csv | wc -l        # expect 6
```

Windows download instructions and the expected filenames are in
[datasets/README.md](datasets/README.md). The archive and extracted CSVs are gitignored.

### 4. Verify and run

```bash
# offline checks + Starter Kit integrity; no API key or dataset needed
./mle_agent/scripts/test_offline.sh

# optional deterministic syntax/runtime recovery demonstration
./mle_agent/scripts/demo_recovery.sh

# reproduce the official validation baseline; needs the dataset, about 40 seconds
python baseline_kuairand-starter-kit/baseline.py \
  --model fm --data_dir datasets/KuaiRand-Pure/data

# one-experiment provider smoke run
./mle_agent/scripts/run_agent_once.sh

# normal autonomous run: 12 experiments, 2 hours, $6 model-cost ceiling
./mle_agent/scripts/run_agent.sh

# short development run: 3 experiments, up to 30 minutes
./mle_agent/scripts/run_agent_dev.sh

# official-budget run after the smoke run and baseline check succeed
TASK_DEFINITION_CONFIRMED=1 ./mle_agent/scripts/run_official.sh
```

The reproduced validation primary should be within ±0.002 of `0.6016`; do not start an official
run if it is not. The normal profile can reach the eight scored variants required before frontier
convergence is allowed and enables bounded quota recovery. Its $6 ceiling, like every profile
budget, can be adjusted with environment variables, for example:

```bash
AGENT_MAX_ITER=10 AGENT_MAX_RUN_COST_USD=4.00 AGENT_MAX_TURNS=20 \
  ./mle_agent/scripts/run_agent.sh
```

Before experiment 1, the harness completes a deterministic bootstrap phase in Python. It discovers the
starter-kit task documentation, reads the complete README, `baseline.py`, `evaluate.py`,
`data.py`, `ablation_features.py`, and inherited candidate through explicit pages, inspects
the train/validation-only data view and its available columns, searches the local literature
corpus, explicitly reproduces the official baseline, and records one structured task summary.
That summary must record the five baseline fields and the organizer's measured no-gain result
for all 13 static fields. The harness blocks edits until this bootstrap is complete; the summary and
baseline result then stay in the same conversation for every later experiment. Bootstrap defaults
to 24 model calls and each experiment defaults to 16. Both values must remain positive; exhausting a
cap ends that phase with an explicit evidence event while preserving completed tools and metrics.
Provider-quota recovery is bounded to three automatic resumes per invocation. Official unattended
runs pre-authorize those bounded resumes; development runs can opt in with
`AGENT_AUTO_RESUME_QUOTA=1`.
Each quota wait is capped at five minutes; exhausting the resume limit stops the run as
`provider_unavailable` instead of starting another experiment against the same unavailable provider.

Feature engineering reads the filtered `experiment_workspace/<run_id>/candidate_data` through
`--data_dir`; it never rewrites those CSVs or the organizer starter kit. The agent implements the
complete feature pipeline in the trial's self-contained `model.py`. Feature-oriented runs must log
their source columns, exact transformations, and train-only leakage controls before the harness will
execute them. This supports history/sequence, temporal, auxiliary-signal, watch-time, aggregate,
and user-item-cross experiments without repeating the measured static-field ablation.

The organizer's NumPy FM is a reference, not an implementation ceiling. Bootstrap also inventories
the active Python environment without importing heavyweight packages. Experiments may replace the
model or pipeline with any justified open-source framework allowed by the hackathon brief. Every run
gets a dedicated venv. Missing packages on the curated ML allowlist—including sklearn, XGBoost,
LightGBM, PyTorch, Polars, and Optuna—install automatically as binary wheels. Off-allowlist requests
show their exact specifiers and justification and wait for `y/n`. URLs, extras, environment markers,
pip flags, source distributions, system installs, and `--user` installs are rejected. Every request
and outcome is logged, and the resolved environment is written to `environment/requirements.lock.txt`.

When a provider returns a real quota/rate-limit response, an opted-in run waits up to five minutes
and resumes the same retained session. The hard resume count and wall deadline prevent an
unattended retry loop from running forever.

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
time, and intervention flag. Run summaries add the explicit stop reason, full frontier and lineage
statistics, conservative convergence history, per-metric deltas, provider cost, GPU-hours, and
the frozen winning bundle.

`./mle_agent/scripts/demo_recovery.sh` reproducibly drives the real agent tool
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

The finalizer verifies the frozen source/config/seed manifest, reproduces the recorded validation
metrics exactly, and only then creates the hidden-split submission. The confirmation flag is
deliberate because the supplied prose conflicts with the
checked-in Starter Kit about the label and metrics. The trusted finalizer generates test
predictions, runs `submit.py --check` for schema/alignment, and never scores hidden labels.

---

## Reflection: limitations and what we would improve

**The honest size of the win.** +0.0025 validation primary is a real gain — about 3x the
published seed noise, stable across five seeds — but it is small, and it is a *validation*
gain. Across experiments we selected repeatedly against the same validation window, so
some of it is selection bias; the hidden-test transfer is unverified by design. We would not
describe this as having solved the benchmark, and the oracle ceiling (0.8645) says most of the
headroom is still there.

**The search stopped early, and the convergence rule let it.** The run declared convergence 
due to ε = 0.002 with N = 3 is derived from seed variance and is defensible, but it means any genuinely different architecture whose *first* untuned attempt lands within noise gets read as "no gain" and the branch dies. Most of the 19-method corpus — sequence models (SASRec, DIN, SIM), multi-task heads (ESMM, MMoE, PLE), listwise/LambdaRank losses, GBDT ranking — got at most one shot each, at hand-picked hyperparameters. A good idea can lose to a bad learning rate here, and did at least once.

**Scoring during search is single-seed.** Only the final winner is re-scored on five seeds.
Every intermediate keep/revert decision therefore runs at roughly the noise floor, which is
exactly the failure mode the project notes warn about. 

**The agent's knowledge is a frozen snapshot.** Retrieval is offline BM25 over 19 hand-written
method cards. That buys determinism and reproducible citations, and it costs novelty: the agent
cannot read a paper published after the corpus was written, and it is anchored toward the
methods we chose to include.

**Budgets bind.** 16 turns per experiment caps how long a candidate can be debugged before it is
abandoned. A harder architecture is more likely to be dropped for a fixable bug than an easy one is.

**With more time, in priority order:**

1. **Multi-seed or sequential-test scoring inside the search loop**, not just at the end — so
   keep/revert decisions stop being made at noise level. This is the single highest-value fix.
2. **A second held-out validation window** carved from the training range, used only to confirm
   the final pick, to measure how much of the gain is selection bias.
3. **A short automatic hyperparameter pass on any candidate that lands near the incumbent**,
   so architectures are compared at a fair configuration instead of a first guess.
4. **Parallel trial execution.** Trials are independent and cheap (~2 min each); running four at
   once turns a 9-experiment run into a 36-experiment one for the same wall clock.
5. **Cross-run frontier memory** — today each run re-derives the search from the baseline. Seeding
   a new run with the previous run's frozen frontier compounds progress across runs.
6. **Live literature retrieval** behind the same citation discipline, so the corpus stops being a
   ceiling on what the agent can propose.
7. **Push into the ranking-loss and sequence directions properly** — within-user listwise softmax
   and a short user-history encoder are the two places the benchmark's structure most obviously
   rewards, and neither got a fair trial.

---

## Team member contributions

| Member | Contribution |
|---|---|
| **Sabitha Jayakumar** | Agent architecture and the implementation — the research loop, search, and documentation. |
| **Santhosh Kumar** | Harness scaffold, the PyTorch model path, stability evidence, and cross-platform hardening. |
| **Sriivatsav** | Literature research and the offline method corpus the agent searches and cites. |
| **Balasubramani Viveka** | Starter-kit integration, baseline reproduction, and training-path fixes. |
