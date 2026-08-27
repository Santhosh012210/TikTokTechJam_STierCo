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

## 4. LLM provider

```bash
cp .env.example .env
```

Uncomment exactly one provider block in `.env` — every provider uses the same variable
names (`LLM_PROVIDER`, `LLM_API_KEY`, optionally `LLM_MODEL` / `LLM_BASE_URL`), and each
block in the template carries its own default model. Ollama needs no API key.

Switching providers later is a `.env` edit, not a code change; provider-specific code is
confined to `harness/provider.py`.

`.env` is gitignored. Never commit API keys.

Gemini 3.x function calls include opaque thought signatures. The provider layer preserves
and returns this metadata automatically; agent code and run logs never interpret it.

## 5. Verify

Three checks, cheapest first. Do them in order — each one rules out a class of problem
that would otherwise show up as a confusing failure later.

**Imports and paths** (instant, no data needed):

```bash
python -c "from harness.config import load_config; c=load_config(); print(c.DATA_DIR, c.BASELINE_PRIMARY)"
```

Prints the dataset path and `0.6016`.

**Offline tests** (seconds, no API key, no data):

```bash
python tests/test_knowledge.py
```

**Baseline reproduction** (~40s, needs the dataset):

```bash
python baseline_kuairand-starter-kit/baseline.py --model fm --data_dir datasets/KuaiRand-Pure/data
```

Validation primary must land within ±0.002 of `0.6016`. If it does not, stop and fix the
dataset before running anything else — every downstream comparison is against this number.

## 6. Run the harness

```bash
# single-agent dev run: one persistent agent owns the complete MLE loop
./scripts/run_agent.sh

# one-experiment smoke run: 1 experiment, 10 turns, 30-minute wall budget
./scripts/run_agent_once.sh

# full run, only after a dev run succeeds
AGENT_MAX_ITER=50 AGENT_WALL_HOURS=4 AGENT_MAX_TURNS=10 ./scripts/run_agent.sh
```

The single-agent defaults reserve at most 2,048 output tokens for work turns and 768
for the closing reflection. Override with `AGENT_MAX_OUTPUT_TOKENS` and
`AGENT_REFLECTION_MAX_TOKENS` and `AGENT_READ_MAX_CHARS` for a higher-limit provider or
model.
Rate-limit failures receive one retry after `RATE_LIMIT_RETRY_DELAY_S` (default 60 seconds).

Trial code goes to the gitignored `experiment_workspace/<run_id>/trial_NNN/`. Durable
evidence goes to `artifacts/runs/<run_id>/` as `logs/events.jsonl`, `results/metrics.json`,
and `reports/summary.md`. See `artifacts/README.md` for promoting a run to `artifacts/final/`.

Check a run's log schema:

```bash
python -m harness.validator artifacts/runs/<run_id>/logs/events.jsonl
```

Exit `0` means every row passed.

## 7. Submissions

`submit.py` takes the path first, then a mode flag:

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
python -m research_agent.knowledge --list          # method corpus contents
python -m research_agent.knowledge "bpr gradient"  # query it (offline)
python tests/live_builder_smoke.py                 # live Builder check — costs tokens
```

## Troubleshooting

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'harness'` | Run from the repo root, and invoke as `python -m harness.main` |
| `FileNotFoundError: Data directory not found` | `datasets/KuaiRand-Pure/data/` is missing the CSVs — see `datasets/README.md` |
| Baseline primary outside ±0.002 | Dataset is incomplete or wrong. Re-download; do not proceed |
| Provider API key missing | `.env` needs `LLM_PROVIDER` and `LLM_API_KEY` (Ollama excepted) |
| Ollama connection refused | Start Ollama and confirm its OpenAI-compatible endpoint is up |
