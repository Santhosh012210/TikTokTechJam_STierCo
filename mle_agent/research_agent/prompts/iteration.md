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
expected metric effect, falsification criterion, smallest implementation, and rollback plan.
Pass the structured proposal fields to `run_model`. Prefer one attributable change. Do not repeat
a measured dead end, prior hypothesis, or already-scored semantic candidate. Search across loss,
sampling, sequence/history, auxiliary signals, time/shift, and model structure rather than falling
back to arbitrary capacity or learning-rate changes.

Measure validation accuracy and runtime/complexity tradeoffs. Do not end the experiment until you
have called `run_model` or exhausted a genuine repair path. Keep tool-facing explanations concise;
full stdout, tracebacks, code, and diffs are already retained by the harness.
