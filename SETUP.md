# KuaiRand Autonomous ML Research Agent — Setup Guide

This document is the single source of truth for first-time setup. Follow it top to bottom. Every command is shown for both **Windows (PowerShell)** and **Mac/Linux (bash)**.

---

## 1. Prerequisites

| Requirement | Minimum version | Check |
|-------------|----------------|-------|
| Python | 3.10+ | `python --version` or `python3 --version` |
| pip | any recent | `pip --version` |
| Internet access | — | needed to download packages and call LLM APIs |

> **Mac:** if `python3` opens the App Store, install Python from [python.org](https://python.org) first.
> **Windows:** use `py` (Python Launcher) if `python` is not on your PATH.

---

## 2. Project layout

The git repository root **is** the project root. After setup it should look like this:

```
TikTokTechJam_STierCo/         ← project root (this repo)
  kuairand-starter-kit/        ← organiser starter kit (read-only reference)
      baseline.py
      data.py
      evaluate.py
      submit.py
      ablation_features.py
      baseline_scores.json
      venv/                    ← virtual environment (created in step 3)
  KuaiRand-Pure/
      KuaiRand-Pure/
          data/                ← dataset CSVs (downloaded separately)
  harness/                     ← autonomous research agent code
  candidates/                  ← auto-created: one folder per experiment
  logs/                        ← auto-created: JSONL run logs
  requirements.txt             ← Python dependencies for all providers
  .env                         ← your API key (created in step 5, never committed)
  .env.example
  SETUP.md                     ← this file
```

All commands below are written relative to one of two locations:

- **project root** — the outer `TikTokTechJam_STierCo/` folder
- **starter kit** — `kuairand-starter-kit/` (one level down)

---

## 3. Create and activate the virtual environment

The venv lives inside `kuairand-starter-kit/` so it is co-located with the Python scripts it needs to call. The harness auto-detects it at `kuairand-starter-kit/venv/`.

**Windows (PowerShell):**
```powershell
cd kuairand-starter-kit
python -m venv venv
venv\Scripts\Activate.ps1
```

> If you get a script-execution error, run this first:
> `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

**Mac / Linux:**
```bash
cd kuairand-starter-kit
python3 -m venv venv
source venv/bin/activate
```

Your prompt will now show `(venv)` when the environment is active.

---

## 4. Install dependencies

Run this from **inside `kuairand-starter-kit/`** with the venv active. The requirements file lives one level up, at the project root, and supports every LLM provider.

**Windows:**
```powershell
pip install -r ..\requirements.txt
```
**Mac/Linux:**
```bash
pip install -r ../requirements.txt
```

---

## 5. Configure your API key

Go back to the **project root**:

**Windows:**
```powershell
cd ..
```
**Mac/Linux:**
```bash
cd ..
```

Copy the example file and fill it in:

**Windows:**
```powershell
Copy-Item .env.example .env
notepad .env
```
**Mac/Linux:**
```bash
cp .env.example .env
nano .env          # or: open .env, vim .env, code .env
```

Edit `.env` to match your chosen provider — uncomment the relevant block and paste your key. The other blocks can be left commented out.

```
# Example for Groq (free)
LLM_PROVIDER=groq
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxx
```

> **`.env` is gitignored** — it will never be committed. `.env.example` is the tracked template.

---

## 6. Verify the dataset

The dataset must be at, relative to the project root:
```
KuaiRand-Pure/KuaiRand-Pure/data/
```

It should contain these files:
```
log_standard_4_08_to_4_21_pure.csv
log_standard_4_22_to_5_08_pure.csv
log_random_4_22_to_5_08_pure.csv
video_features_basic_pure.csv
video_features_statistic_pure.csv
user_features_pure.csv
```

If the `data/` folder is missing, download from [kuairand.com](https://kuairand.com) (Zenodo, no login needed):

**Mac/Linux:**
```bash
# From the project root
wget https://zenodo.org/records/10439422/files/KuaiRand-Pure.tar.gz
tar xzf KuaiRand-Pure.tar.gz
```

**Windows:** Download the `.tar.gz` from the URL above and extract with 7-Zip or Windows Explorer (right-click → Extract All).

> The extracted archive nests as `KuaiRand-Pure/KuaiRand-Pure/data/` — that double folder is expected, and is what `harness/config.py` looks for.

---

## 7. Run the baseline verification (Step 1)

From **inside `kuairand-starter-kit/`** with the venv active:

**Windows:**
```powershell
cd kuairand-starter-kit
venv\Scripts\Activate.ps1
python baseline.py --model fm --data_dir "..\KuaiRand-Pure\KuaiRand-Pure\data"
```

**Mac/Linux:**
```bash
cd kuairand-starter-kit
source venv/bin/activate
python baseline.py --model fm --data_dir "../KuaiRand-Pure/KuaiRand-Pure/data"
```

Expected output (last 3 lines):
```
=== fm (seed=0) ===
  valid  GAUC 0.6671 | nDCG@5 0.5358 | primary 0.6015
  test   GAUC 0.6621 | nDCG@5 0.5286 | primary 0.5953
```

Valid primary must be within ±0.002 of **0.6016**. If it is not, stop and check the data path.

---

## 8. Verify the harness imports

From the **project root**:

**Windows:**
```powershell
kuairand-starter-kit\venv\Scripts\python.exe -c "
import sys; sys.path.insert(0, '.')
from harness.config import load_config
cfg = load_config()
print('Data dir exists:', cfg.DATA_DIR.exists())
print('Starter kit exists:', cfg.STARTER_KIT_ROOT.exists())
print('Baseline primary:', cfg.BASELINE_PRIMARY)
"
```

**Mac/Linux:**
```bash
kuairand-starter-kit/venv/bin/python -c "
import sys; sys.path.insert(0, '.')
from harness.config import load_config
cfg = load_config()
print('Data dir exists:', cfg.DATA_DIR.exists())
print('Starter kit exists:', cfg.STARTER_KIT_ROOT.exists())
print('Baseline primary:', cfg.BASELINE_PRIMARY)
"
```

All three lines should print `True` / `True` / `0.6016`.

---

## 9. Run the harness (dev mode — 10 iterations)

From the **project root**:

**Windows:**
```powershell
kuairand-starter-kit\venv\Scripts\python.exe harness\main.py `
  --max-iter 10 `
  --wall-hours 0.5 `
  --builder-turns 2
```

**Mac/Linux:**
```bash
kuairand-starter-kit/venv/bin/python harness/main.py \
  --max-iter 10 \
  --wall-hours 0.5 \
  --builder-turns 2
```

This runs 10 experiments (~30 min wall time) using the cheap dev settings. Verify a log file appears in `logs/` and contains valid JSON rows before running the full budget.

---

## 10. Full autonomous run (Step 6 — only after dev run verified)

**Windows:**
```powershell
kuairand-starter-kit\venv\Scripts\python.exe harness\main.py `
  --max-iter 50 `
  --wall-hours 4 `
  --builder-turns 10
```

**Mac/Linux:**
```bash
kuairand-starter-kit/venv/bin/python harness/main.py \
  --max-iter 50 \
  --wall-hours 4 \
  --builder-turns 10
```

---

## 11. Validate the log

After any run, check the JSONL log is well-formed:

**Windows:**
```powershell
kuairand-starter-kit\venv\Scripts\python.exe -m harness.validator logs\<run_file>.jsonl
```

**Mac/Linux:**
```bash
kuairand-starter-kit/venv/bin/python -m harness.validator logs/<run_file>.jsonl
```

Exit code 0 = all rows valid. Exit code 1 = schema errors printed to stderr.

---

## 12. Generate a submission

From the **project root**, once you have a candidate you want to submit:

**Windows:**
```powershell
kuairand-starter-kit\venv\Scripts\python.exe kuairand-starter-kit\submit.py --make --split test submission.csv
```

**Mac/Linux:**
```bash
kuairand-starter-kit/venv/bin/python kuairand-starter-kit/submit.py --make --split test submission.csv
```

---

## Provider switching (after initial setup)

To switch from one LLM provider to another at any time, edit `.env` only:

```
# From Anthropic to Groq — change these two lines, nothing else
LLM_PROVIDER=groq
GROQ_API_KEY=gsk_...
```

Optionally set `LLM_MODEL=...` to override the per-provider default. No code changes needed — all provider-specific logic is isolated in `harness/provider.py`.

---

## Quick-reference command table

All paths are relative to the **project root** unless noted. `PY` = `kuairand-starter-kit\venv\Scripts\python.exe` (Windows) or `kuairand-starter-kit/venv/bin/python` (Mac/Linux).

| Task | Windows | Mac/Linux |
|------|---------|-----------|
| Activate venv (from `kuairand-starter-kit/`) | `venv\Scripts\Activate.ps1` | `source venv/bin/activate` |
| Install deps (from `kuairand-starter-kit/`) | `pip install -r ..\requirements.txt` | `pip install -r ../requirements.txt` |
| Run baseline (from `kuairand-starter-kit/`) | `python baseline.py --model fm --data_dir "..\KuaiRand-Pure\KuaiRand-Pure\data"` | `python baseline.py --model fm --data_dir "../KuaiRand-Pure/KuaiRand-Pure/data"` |
| Dev run | `PY harness\main.py --max-iter 10 --wall-hours 0.5 --builder-turns 2` | `PY harness/main.py --max-iter 10 --wall-hours 0.5 --builder-turns 2` |
| Full run | `PY harness\main.py --max-iter 50 --wall-hours 4 --builder-turns 10` | `PY harness/main.py --max-iter 50 --wall-hours 4 --builder-turns 10` |
| Validate log | `PY -m harness.validator logs\<file>.jsonl` | `PY -m harness.validator logs/<file>.jsonl` |
| Make submission | `PY kuairand-starter-kit\submit.py --make --split test submission.csv` | `PY kuairand-starter-kit/submit.py --make --split test submission.csv` |

> All bare `python` commands assume the venv is active **or** you prefix them with the full `PY` path shown above.

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'harness'`**
→ Run from the project root (the repo root), not from inside `kuairand-starter-kit/`.

**`FileNotFoundError: Data directory not found`**
→ Check step 6. The data path must be `KuaiRand-Pure/KuaiRand-Pure/data` relative to the project root (note the doubled folder name).

**`EnvironmentError: ANTHROPIC_API_KEY is not set`** (or the matching key for your provider)
→ Check step 5. Make sure `.env` exists at the project root and `LLM_PROVIDER` + the matching key are uncommented.

**`Set-ExecutionPolicy` error on Windows**
→ Open PowerShell as Administrator and run `Set-ExecutionPolicy RemoteSigned`.

**Ollama connection refused**
→ Make sure the Ollama app is running before starting the harness. On Mac it lives in the menu bar; on Windows check the system tray.

**Baseline primary is way off (< 0.58 or > 0.62)**
→ Wrong data path, or data files are corrupted/incomplete. Re-download and re-extract.

**Harness runs but `cfg.STARTER_KIT_ROOT.exists()` is `False`**
→ The `kuairand-starter-kit/` folder must sit directly under the project root, next to `harness/`. Do not nest it inside another folder.
