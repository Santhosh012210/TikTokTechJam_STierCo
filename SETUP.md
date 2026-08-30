# Setup

First-time setup, from the repository root. For what the project *is*, see [README.md](README.md).

## 1. Prerequisites

Python 3.10+ (`python3 --version`) and internet access for packages, data, and a hosted
LLM provider.

## 2. Virtual environment

One venv at the repository root. All later commands assume it is active.

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

If PowerShell blocks activation: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.

## 3. Dataset

KuaiRand-Pure, ~300MB from Zenodo — no registration:

```bash
curl -L https://zenodo.org/records/10439422/files/KuaiRand-Pure.tar.gz \
  -o datasets/KuaiRand-Pure.tar.gz
tar -xzf datasets/KuaiRand-Pure.tar.gz -C datasets
```

Windows PowerShell uses `Invoke-WebRequest` instead of `curl`; the exact command, the six
expected CSV filenames, and the bonus datasets are in
[datasets/README.md](datasets/README.md).

Verify the six CSVs landed in the right place:

```bash
ls datasets/KuaiRand-Pure/data/*.csv | wc -l    # expect 6
```

Downloaded data is gitignored — never commit the archive or the CSVs.

## 4. Google ADK and Gemini

```bash
cp .env.example .env
```

Set `GOOGLE_API_KEY` to a Gemini API key from Google AI Studio. The optional `ADK_MODEL`
defaults to the low-cost `gemini-3.5-flash-lite` model.

For a non-disruptive migration, an existing Gemini block using `LLM_PROVIDER=gemini`,
`LLM_API_KEY`, and optional `LLM_MODEL` is also accepted by the ADK runner. The other
provider blocks remain only for the legacy Builder/Strategist comparison runner.

`.env` is gitignored. Never commit API keys.

Google ADK now owns the persistent conversation, Gemini function-calling protocol,
automatic tool loop, and lifecycle callbacks. The Python harness still owns file and data
boundaries, experiment execution, metric validation, and durable redacted logs.

## 5. Verify

Three checks, cheapest first. Do them in order — each one rules out a class of problem
that would otherwise show up as a confusing failure later.

**Imports and paths** (instant, no data needed):

```bash
python -c "from mle_agent.harness.config import load_config; c=load_config(); print(c.DATA_DIR, c.BASELINE_PRIMARY)"
```

Prints the dataset path and `0.6016`.

**Offline tests** (seconds, no API key, no data):

```bash
./mle_agent/scripts/test_offline.sh
```

**Optional manual baseline preflight** (~40s, needs the dataset):

```bash
python baseline_kuairand-starter-kit/baseline.py --model fm --data_dir datasets/KuaiRand-Pure/data
```

Validation primary must land within ±0.002 of `0.6016`. The autonomous runner performs
this reproduction again through an explicit agent tool call after the agent has read the
task documentation. If it does not match, the bootstrap stops before experiment 1.

## 6. Run the harness

```bash
# single-agent dev run: one persistent Google ADK session owns the complete MLE loop
./mle_agent/scripts/run_agent.sh

# one-experiment smoke run: uncapped ADK calls, 30-minute outer wall budget
./mle_agent/scripts/run_agent_once.sh

# full run, only after a dev run succeeds
TASK_DEFINITION_CONFIRMED=1 ./mle_agent/scripts/run_official.sh
```

The ADK agent defaults to at most 2,048 output tokens per model call, but does not cap the
number of calls. Override the output size with
`AGENT_MAX_OUTPUT_TOKENS`; use `AGENT_READ_MAX_CHARS` to change the constrained file page
size. Each ADK invocation receives one SDK-managed HTTP retry with exponential backoff
bounded by `PROVIDER_RETRY_DELAY_S` and `RATE_LIMIT_RETRY_DELAY_S`.
`AGENT_BOOTSTRAP_MAX_TURNS` and `AGENT_MAX_TURNS` default to `0`, meaning unlimited;
positive values remain available as optional diagnostic caps.

If the provider returns a quota/rate-limit error, the terminal asks: `You have hit your
LLM limit. Would you like to resume when the limit has reset? [y/n]:`. After `y`, the
harness waits for the advertised retry/reset delay and resumes the retained ADK session.
Any later quota pauses in the same run wait and resume automatically without asking again.
Interactive terminals receive colour-coded Agent/Harness debug output; set `NO_COLOR=1`
to disable it or `FORCE_COLOR=1` to force ANSI colour output. Each Agent block includes
up to five sanitized lines of the provider's assistant text. Function-call-only responses
are labelled explicitly instead of being presented as hidden reasoning.

The agent may also pause when a concrete experiment needs a missing open-source dependency. It
prints the exact validated PyPI requirements and its justification, then asks whether to install
them into `.venv`. Answering `y` permits only that request; answering `n` instructs the agent to use
an installed alternative. Each decision is logged as a manual intervention. The agent cannot pass
pip flags or URLs and is instructed never to invoke a package manager from generated model code.

Trial code goes to the gitignored `experiment_workspace/<run_id>/trial_NNN/`. Durable
evidence goes to `artifacts/runs/<run_id>/` as `logs/events.jsonl`,
`logs/llm_events.jsonl`, `results/metrics.json`,
and `reports/summary.md`. See `artifacts/README.md` for promoting a run to `artifacts/final/`.

Each run also creates `experiment_workspace/<run_id>/candidate_data`, a filtered copy containing
training and validation rows but no test rows. The bootstrap exposes its CSV column inventory after
the agent has read the official feature loader and organizer ablation. Raw CSVs remain immutable;
all feature joins, histories, buckets, crosses, and train-fitted encoders belong in the candidate's
self-contained `model.py`. Feature experiments are rejected unless they declare their input columns,
transformations, and leakage controls.

Check a run's log schema:

```bash
python -m mle_agent.harness.validator artifacts/runs/<run_id>/logs/events.jsonl
```

Exit `0` means every row passed.

## 7. Submissions

After the authoritative task definition is confirmed and an autonomous run converges,
promote its validation-best candidate and validate the final CSV in one command:

```bash
./mle_agent/scripts/finalize.sh --run-id <run_id> --task-definition-confirmed
```

For an official run that truthfully stopped at the hard iteration/wall budget, add
`--allow-budget-stop`. The finalizer does not compute hidden-test metrics.

The organiser utility can also be used directly. `submit.py` takes the path first, then a mode flag:

```bash
SK=baseline_kuairand-starter-kit
python $SK/submit.py submission.csv --make  --split test  --data_dir datasets/KuaiRand-Pure/data
python $SK/submit.py submission.csv --check --split test  --data_dir datasets/KuaiRand-Pure/data
```

`--make` generates an example from the official FM baseline; `--check` validates header,
row count, `row_id` continuity, and alignment. Always run `--check` before submitting.
`--score` also scores, and is only valid on `--split valid`.

## Other commands

```bash
python -m mle_agent.research_agent.knowledge --list          # method corpus contents
python -m mle_agent.research_agent.knowledge "bpr gradient"  # query it (offline)
python -m mle_agent.tests.live_builder_smoke                 # legacy live check — costs tokens
./mle_agent/scripts/demo_recovery.sh                         # recovery evidence
```

## Troubleshooting

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'mle_agent'` | Run from the repository root; no installation is required |
| `FileNotFoundError: Data directory not found` | `datasets/KuaiRand-Pure/data/` is missing the CSVs — see `datasets/README.md` |
| Baseline primary outside ±0.002 | Dataset is incomplete or wrong. Re-download; do not proceed |
| Google ADK API key missing | Set `GOOGLE_API_KEY` in `.env` (or retain the old Gemini `LLM_PROVIDER=gemini` + `LLM_API_KEY` block) |
| Gemini free-tier rate limit | Answer `y` at the recovery prompt to wait and resume automatically, or `n` to stop with the ADK trace preserved |
