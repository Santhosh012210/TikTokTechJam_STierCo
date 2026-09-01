# TikTokTechJam_STierCo

An **autonomous ML research agent** that tries to beat the official baseline on the [KuaiRand-Pure](https://kuairand.com) within-user ranking benchmark.

It runs one persistent LangChain-backed research session across the complete MLE loop:
1. understand the benchmark
2. inspect train/validation data
3. research and propose a hypothesis
4. implement a self-contained candidate model, train and score it, repair failures, and
reflect before the next experiment — until it converges or runs out of budget.

All agent code is grouped under the local `mle_agent/` namespace. `mle_agent/research_agent/` decides what to investigate and try next, while `mle_agent/harness/` executes those decisions deterministically and records the evidence.

For the narrative version — what inspired this, how it was built, and what broke along the way —
see **[PROJECT_STORY.md](PROJECT_STORY.md)**.

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
PROJECT_STORY.md                   submission write-up: inspiration, build, challenges
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

One LangChain session spans the whole diagram below — bootstrap and every experiment share a
single conversation, so the agent still has the task summary and its own last reflection in
context at experiment 9. The harness owns every arrow.

```text
  BOOTSTRAP  (once, deterministic; edits are blocked until it completes)
    read starter kit -> inspect train/valid view -> search corpus -> reproduce baseline
                                |
                                v
                       one retained task summary
                                |
  EXPERIMENT LOOP               v
    1. harness picks a parent node from the frontier
                                |
                                v
    2. agent proposes a hypothesis   <--- cites ---   offline BM25 method corpus
                                |
                                v
    3. agent writes model.py  -->  syntax hook  --fail-->  repair (run_model gated)
                                |  pass
                                v
    4. harness runs it in a subprocess, against the train/valid-only data view
                                |
                                v
    5. harness validates metrics, drops any test key, freezes an immutable node
                                |
                                v
    6. agent reflects and names the next direction
                                |
                +---------------+----------------+
                |                                |
        budget left, no convergence          converged
        -> back to step 1                    (e = 0.002, N = 3, >= 8 scored variants)
                                                 |
                                                 v
  FINALIZE  multi-seed re-score -> conservative pick (mean - std) -> submission.csv
```

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

The reproduced validation primary must be within ±0.002 of `0.6016`; don't start an official run
if it isn't. The normal profile is sized to reach the eight scored variants that frontier
convergence requires. Every budget is an environment variable:

```bash
AGENT_MAX_ITER=10 AGENT_MAX_RUN_COST_USD=4.00 AGENT_MAX_TURNS=20 \
  ./mle_agent/scripts/run_agent.sh
```

What a run does, in brief:

- **Bootstrap.** Before experiment 1, a deterministic Python phase reads the starter kit in full,
  inspects the train/validation-only data view, searches the method corpus, reproduces the
  baseline, and records one task summary that stays in context for every later experiment. Edits
  are blocked until it completes. Caps are 24 model calls for bootstrap and 16 per experiment;
  exhausting one ends that phase with an explicit evidence event, keeping completed work.
- **Environment.** Each run gets its own venv. Allowlisted ML packages — sklearn, XGBoost,
  LightGBM, PyTorch, Polars, Optuna — install as binary wheels automatically; anything else shows
  its specifier and justification and waits for `y/n`. URLs, extras, pip flags, source
  distributions, and system or `--user` installs are rejected. The resolved set is written to
  `environment/requirements.lock.txt`. The organiser's NumPy FM is a reference, not a ceiling.
- **Feature work.** Candidates read the filtered `candidate_data` view through `--data_dir` and
  implement the whole pipeline in their own `model.py`, never rewriting the CSVs or the starter
  kit. A feature run must log its source columns, transformations, and train-only leakage
  controls before the harness will execute it.
- **Quota recovery.** On a real rate-limit response an opted-in run waits up to five minutes and
  resumes the same session, at most three times, then stops as `provider_unavailable` rather than
  retrying forever. Official runs pre-authorise this; dev runs opt in with
  `AGENT_AUTO_RESUME_QUOTA=1`.

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

**Limitations.**

- The +0.0025 gain is real (3x seed noise, stable over five seeds) but small, and it is a
  *validation* gain — selected repeatedly against one window, with hidden-test transfer
  unverified by design. The oracle ceiling of 0.8645 says most of the headroom is untouched.
- Convergence fired shortly past the eight-variant floor. ε = 0.002 / N = 3 is derived from seed
  variance and defensible, but it kills any branch whose first untuned attempt lands within noise.
- Scoring during the search is single-seed; only the winner is re-scored on five. Intermediate
  keep/revert decisions therefore run at the noise floor.
- Most of the 19-method corpus got at most one shot at hand-picked hyperparameters, and the
  corpus itself is a frozen snapshot — the agent cannot propose what we did not write down.
- 16 turns per experiment means a harder architecture is likelier to be dropped for a fixable bug
  than an easy one.

**With more time, in priority order:** multi-seed scoring inside the search loop; a second
held-out window to size the selection bias; a short hyperparameter pass for candidates near the
incumbent; parallel trials (independent, ~2 min each); cross-run frontier memory; and a fair
trial for within-user listwise losses and a user-history encoder.

The longer version — how this was built and what broke — is in
[PROJECT_STORY.md](PROJECT_STORY.md).

---

## Team member contributions

| Member | Contribution |
|---|---|
| **Sabitha Jayakumar** | Agent architecture and the implementation — the research loop, search, and documentation. |
| **Santhosh Kumar** | Harness scaffold, the PyTorch model path, stability evidence, and Agent architecture refinements. |
| **Sriivatsav** | Literature research and the offline method corpus the agent searches and cites. Agent architecture refinements |
| **Balasubramani Viveka** | Starter-kit integration, baseline reproduction, and training-path fixes. Agent architecture refinements |
