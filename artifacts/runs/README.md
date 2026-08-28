# Runs

Each subdirectory is one autonomous-agent run, named by its UTC run ID. The harness
creates its `logs/`, `results/`, and `reports/` subdirectories automatically.

- `logs/events.jsonl` contains one consolidated row per experiment.
- `logs/llm_events.jsonl` contains the chronological provider-response and tool-result trace.
- `results/metrics.json` contains the machine-readable run outcome and trace counts.
- `reports/summary.md` contains the human-readable run summary.
