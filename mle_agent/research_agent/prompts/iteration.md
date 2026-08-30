Begin experiment `${iteration}` now.

- Current candidate directory: `${candidate_dir}`
- Parent validation primary: `${parent_primary}`
- Best validation primary so far: `${best_primary}`
- Remaining agent turns for this experiment: `${max_turns}`

`${stage_instruction}`

Compact prior experiment ledger (empty on experiment 1):

```json
${experiment_ledger}
```

Choose a non-redundant, evidence-backed change. Before editing, form one compact proposal with:
the observation, precise hypothesis, primary target component, literature or prior-run evidence,
expected metric effect, falsification criterion, implementation plan, and rollback plan. The plan
must be sufficiently complete to test the hypothesis faithfully; implementation size is not an
optimization target.
Pass every structured proposal field to `run_model`, including one required `target_component`
category. Prefer one attributable change. Do not repeat
a measured dead end, prior hypothesis, or already-scored semantic candidate. Search across loss,
sampling, sequence/history, auxiliary signals, watch-time, leakage-safe aggregate/cross features,
time/shift, and model structure rather than falling back to arbitrary capacity or learning-rate
changes.

The official NumPy FM is a starting point, not the solution boundary. Consider complete pipeline
or model replacements and established open-source frameworks when they offer a better test of the
hypothesis. Use `inspect_environment` evidence retained from bootstrap. If a required package is
missing, call `request_dependency_install` with the narrow package list and a concrete justification;
continue only after the harness reports it installed. Never invoke pip from `model.py`.

Feature engineering uses the filtered `candidate_data` passed through `--data_dir`; never rewrite
the raw CSVs. Keep every scored feature pipeline self-contained in `model.py`. Do not repeat the
organizer's measured all-13-static-field ablation. Prefer within-user-varying item/context signals,
train-history sequences or aggregates, temporal signals, and meaningful user-item crosses. Fit all
vocabularies, buckets, aggregates, and any label-derived statistics on training rows only, then apply
the frozen transform to validation. For these experiments, pass `feature_sources`,
`feature_transformations`, and `leakage_controls` to `run_model`.

Measure validation accuracy and runtime/complexity tradeoffs. Do not end the experiment until you
have called `run_model` or exhausted a genuine repair path. Keep tool-facing explanations concise;
full stdout, tracebacks, code, and diffs are already retained by the harness.
