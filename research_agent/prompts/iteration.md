Begin experiment `${iteration}` now.

- Current candidate directory: `${candidate_dir}`
- Parent validation primary: `${parent_primary}`
- Best validation primary so far: `${best_primary}`
- Remaining agent turns for this experiment: `${max_turns}`

`${stage_instruction}`

Choose a non-redundant, evidence-backed change. Measure validation accuracy and consider
runtime/complexity tradeoffs. Do not end the experiment until you have called `run_model`
or exhausted a genuine repair path.
