# Experiment artifacts

This directory is the durable, Git-tracked record of autonomous research runs. It is
both the agent's experiment memory and the primary evidence surface for judges.

## Layout

```text
artifacts/
├── runs/
│   └── <run_id>/
│       ├── logs/events.jsonl
│       ├── environment/manifest.json
│       ├── environment/requirements.lock.txt
│       ├── results/metrics.json
│       └── reports/summary.md
└── final/
    ├── model.py
    ├── metrics.json
    ├── requirements.lock.txt
    ├── submission.csv
    └── final-report.md
```

- `logs/events.jsonl` is the append-only source of truth for hypotheses, code diffs,
  metrics, failures, recoveries, token usage, runtime, and interventions.
- `environment/manifest.json` records the dedicated run venv and dependency policy;
  `requirements.lock.txt` pins the final resolved distributions.
- `results/metrics.json` is a machine-readable run summary.
- `reports/summary.md` is the human-readable view generated from the run state.
- `final/` contains only the explicitly selected submission and its evidence.

The agent writes temporary trial implementations to the Git-ignored
`experiment_workspace/`. Promote the validation-best candidate with the trusted
finalizer instead of copying a peak trial by hand:

```bash
./mle_agent/scripts/finalize.sh --run-id <run_id> --task-definition-confirmed
```

It accepts a converged run (or an explicitly acknowledged official hard-budget stop),
copies the selected model and dependency lock, generates the aligned prediction CSV, runs
the organiser's format/alignment check, and writes the final metrics and report. It does
not score the hidden split.

Commit official run artifacts used to support reported results. Do not promote secrets,
raw datasets, or disposable development runs.
