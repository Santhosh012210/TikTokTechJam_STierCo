# Experiment artifacts

This directory is the durable, Git-tracked record of autonomous research runs. It is
both the agent's experiment memory and the primary evidence surface for judges.

## Layout

```text
artifacts/
├── runs/
│   └── <run_id>/
│       ├── logs/events.jsonl
│       ├── results/metrics.json
│       └── reports/summary.md
└── final/
    ├── model.py
    ├── metrics.json
    ├── submission.csv
    └── final-report.md
```

- `logs/events.jsonl` is the append-only source of truth for hypotheses, code diffs,
  metrics, failures, recoveries, token usage, runtime, and interventions.
- `results/metrics.json` is a machine-readable run summary.
- `reports/summary.md` is the human-readable view generated from the run state.
- `final/` contains only the explicitly selected submission and its evidence.

The agent writes temporary trial implementations to the Git-ignored
`experiment_workspace/`. When a trial is selected for submission, copy its runnable
model into `artifacts/final/` so the final implementation is preserved in Git.

Commit official run artifacts used to support reported results. Do not promote secrets,
raw datasets, or disposable development runs.
