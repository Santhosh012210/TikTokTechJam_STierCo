# Runs

Each subdirectory is one autonomous-agent run, named by its UTC run ID. The harness
creates its `logs/`, `results/`, `reports/`, and `environment/` subdirectories
automatically, so nothing here needs to exist before a run starts.

- `logs/events.jsonl` contains one consolidated row per experiment: the hypothesis,
  the code diff, the resulting metrics, and any error or recovery events.
- `logs/llm_events.jsonl` contains the chronological provider-response and tool-result trace.
- `results/metrics.json` contains the machine-readable run outcome and trace counts.
- `reports/summary.md` contains the human-readable run summary.
- `environment/` contains the resolved dependency lock and manifest for the run's
  dedicated virtual environment.

Run directories are gitignored; only this README is tracked. Per-run evidence is local
by default because most runs are development runs. The run promoted to the final
submission is different: `mle_agent.harness.finalize` copies its whole directory to
`artifacts/final/run/`, which is what ships as the per-iteration log deliverable.
