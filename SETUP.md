# KuaiRand Autonomous ML Research Agent — Setup Guide

Follow this guide from the repository root. Commands use the canonical repository layout and a single root virtual environment.

## 1. Prerequisites

| Requirement | Minimum | Check |
|---|---:|---|
| Python | 3.10 | `python --version` or `python3 --version` |
| pip | Recent | `python -m pip --version` |
| Internet access | — | Required for packages, data, and hosted LLM providers |

## 2. Repository layout

```text
TikTokTechJam_STierCo/
├── .venv/                              local Python environment, gitignored
├── harness/                           autonomous research agent package
├── baseline_kuairand-starter-kit/     fixed organizer reference
├── datasets/
│   ├── README.md                       dataset download instructions
│   └── KuaiRand-Pure/data/            downloaded CSV files, gitignored
├── experiment_workspace/             disposable generated trial code, gitignored
├── artifacts/
│   ├── runs/<run_id>/                 logs, results, and reports for one agent run
│   └── final/                         selected model and submission evidence
├── requirements.txt
└── .env                               local provider configuration, gitignored
```

Keep `baseline_kuairand-starter-kit/` unchanged. It is the versioned source of truth for the official baseline, split logic, evaluator, and submission format.

## 3. Create the root virtual environment

### Windows PowerShell

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

### macOS and Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

All remaining commands assume `.venv` is active and are run from the repository root.

## 4. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The single requirements file supports Anthropic and the OpenAI-compatible providers: OpenAI, Groq, Gemini, and Ollama.

## 5. Download KuaiRand-Pure

Follow [datasets/README.md](datasets/README.md). The required CSV files must end up at:

```text
datasets/KuaiRand-Pure/data/
```

KuaiRand-1k and KuaiRand-27k are optional bonus datasets and are not required for the primary benchmark.

## 6. Configure an LLM provider

Copy the environment template:

### Windows PowerShell

```powershell
Copy-Item .env.example .env
notepad .env
```

### macOS and Linux

```bash
cp .env.example .env
```

Enable exactly one provider block in `.env`. Every hosted provider uses the same
variable names. For example, Groq uses:

```dotenv
LLM_PROVIDER=groq
LLM_API_KEY=gsk_your_key_here
```

Current provider examples—copy one only:

Anthropic:

```dotenv
LLM_PROVIDER=anthropic
LLM_API_KEY=sk-ant-your-key-here
```

Default model: `claude-haiku-4-5-20251001`.

Groq:

```dotenv
LLM_PROVIDER=groq
LLM_API_KEY=gsk_your_key_here
```

Default model: `llama-3.3-70b-versatile`.

Google Gemini:

```dotenv
LLM_PROVIDER=gemini
LLM_API_KEY=your-gemini-key-here
```

Default model: `gemini-2.0-flash`.

OpenAI:

```dotenv
LLM_PROVIDER=openai
LLM_API_KEY=sk-your-openai-key-here
```

Default model: `gpt-4o-mini`.

Ollama:

```dotenv
LLM_PROVIDER=ollama
```

Default model: `llama3.2`; no API key is required.

`LLM_MODEL` is optional. When omitted, the harness chooses the default shown for the
selected provider. Set it only to override that default. `LLM_BASE_URL` is also optional;
it is mainly useful for a custom Ollama endpoint or another compatible gateway.

The `.env` file is ignored by Git. Never commit API keys.

## 7. Verify paths and imports

```bash
python -c "from harness.config import load_config; c=load_config(); print(c.DATA_DIR); print(c.BASELINE_ROOT); print(c.BASELINE_PRIMARY)"
```

The command should print the dataset path, the baseline directory, and baseline primary score `0.6016` without raising an exception.

## 8. Reproduce the official baseline

```bash
python baseline_kuairand-starter-kit/baseline.py \
  --model fm \
  --data_dir datasets/KuaiRand-Pure/data
```

The validation primary score should be within `±0.002` of the published value `0.6016`. If it is not, stop and verify the dataset files and path before running agent experiments.

## 9. Run the agent harness

Development run:

```bash
python -m harness.main \
  --max-iter 10 \
  --wall-hours 0.5 \
  --builder-turns 2
```

Full run, after the development run succeeds:

```bash
python -m harness.main \
  --max-iter 50 \
  --wall-hours 4 \
  --builder-turns 10
```

Generated trial implementations are written to the Git-ignored
`experiment_workspace/<run_id>/`. Durable run evidence is written to:

```text
artifacts/runs/<run_id>/
├── logs/events.jsonl
├── results/metrics.json
└── reports/summary.md
```

Review a run there, then copy its selected model and submission evidence into
`artifacts/final/` when it is designated for submission.

## 10. Validate a run log

```bash
python -m harness.validator artifacts/runs/<run_id>/logs/events.jsonl
```

Exit code `0` means every row passed schema validation. Exit code `1` means validation errors were written to stderr.

## 11. Generate or validate a submission

Generate an example submission with the official baseline:

```bash
python baseline_kuairand-starter-kit/submit.py submission.csv \
  --make \
  --split test \
  --data_dir datasets/KuaiRand-Pure/data
```

Validate a submission before uploading it:

```bash
python baseline_kuairand-starter-kit/submit.py submission.csv \
  --check \
  --split test \
  --data_dir datasets/KuaiRand-Pure/data
```

## Provider switching

Switch providers by changing `LLM_PROVIDER` and `LLM_API_KEY`; no source changes are
needed. Optionally set `LLM_MODEL` or `LLM_BASE_URL` to override the selected provider's
defaults. Provider-specific code is isolated in `harness/provider.py`.

## Troubleshooting

### `ModuleNotFoundError: No module named 'harness'`

Run commands from the repository root and invoke the package with `python -m harness.main`.

### `FileNotFoundError: Data directory not found`

Confirm that `datasets/KuaiRand-Pure/data/` contains the six CSV files listed in `datasets/README.md`.

### Baseline primary is outside the expected range

Check the dataset location and integrity. Re-download the archive if files are missing or incomplete.

### Provider API key is missing

Confirm that `.env` exists at the repository root and contains `LLM_PROVIDER` plus
`LLM_API_KEY`. Ollama is the only provider that does not require `LLM_API_KEY`.

### Ollama connection refused

Start Ollama and ensure its OpenAI-compatible endpoint is available before launching the agent harness.
