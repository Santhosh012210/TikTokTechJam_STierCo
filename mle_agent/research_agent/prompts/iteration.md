Begin experiment `${iteration}` now.

- Current candidate directory: `${candidate_dir}`
- Parent validation primary: `${parent_primary}`
- Best validation primary so far: `${best_primary}`
- Remaining agent turns for this experiment: `${max_turns}`

`${stage_instruction}`

Compact prior experiment ledger, built by the harness from validated results (empty on experiment 1). This survives context compaction, so trust it over your recollection of earlier turns:

```json
${experiment_ledger}
```

Ranked research backlog recorded during bootstrap (follow it; revise priorities only with new
evidence):

```json
${research_plan}
```

Reviewed research knowledge from earlier runs. Follow its scoped `do not retry` rules;
the harness separately refuses identical candidate fingerprints:

${prior_findings}

Use these findings as supplied knowledge, not as an official incumbent. Do not repeat an
exact experiment merely to rediscover its score; isolate or extend a mechanism through an
attributable change and fresh trusted scoring.

Experiments 1-3 must each target a distinct `target_component`. The harness refuses a repeat before
three distinct components have been scored unless you pass `diversity_override` with a written,
evidence-based justification. Work through the backlog's first three families first.

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
hypothesis: a tuned DeepFM/DCN/xDeepFM (PyTorch) or a LightGBM ranker is a valid `model` experiment
-- set `execution_class: "substantial"`, seed every RNG from `--seed`, bound epochs, early-stop on
the official validation primary, and restore the best checkpoint. Use `inspect_environment`
evidence retained from bootstrap. If a required package is missing, call `request_dependency_install`
with the narrow package list and a concrete justification.
Known-good ML packages resolve automatically as binary wheels inside the dedicated run venv;
off-allowlist packages require user approval. Continue only after the harness reports the dependency
installed. Never invoke pip from `model.py`.

Work in diffs, not rewrites. You inherit the incumbent `model.py`; change it with
`edit_file` so the experiment is a targeted, attributable delta against that parent.
`run_model` reports a `change_summary` with the hunk count and whether your change
replaced the file, and the harness records the resulting diff as the experiment's
evidence. A wholesale rewrite, or a change scattered across many regions, bundles
several ideas into one score and makes the result unattributable to any of them --
if the metric moves you will not know which part moved it. A full replacement is
legitimate when the hypothesis genuinely requires one (a different model family, a
framework-backed candidate); in that case say so, and report the result as evidence
about the bundle rather than about a single change.

If `run_model` returns a `sanity_check`, your candidate scored at or below a trivial
item-popularity ranker. Treat that as a broken implementation until you have shown
otherwise: diagnose it, and if you cannot rule out a bug, set `hypothesis_status` to
`not_tested` rather than `not_supported`. An untested idea must not be recorded as a
refuted one -- that retires a research direction nobody actually tried.

Feature engineering uses the filtered `candidate_data` passed through `--data_dir`; never rewrite
the raw CSVs. Keep every scored feature pipeline self-contained in `model.py`. Do not repeat the
organizer's measured all-13-static-field ablation. Prefer within-user-varying item/context signals,
train-history sequences or aggregates, temporal signals, and meaningful user-item crosses. Fit all
vocabularies, buckets, aggregates, and any label-derived statistics on training rows only, then apply
the frozen transform to validation. Implement the identical frozen transform and field order for the
`--submission-path` inference branch as part of the same edit, even though research cannot execute
that branch. A candidate that augments train/validation but sends raw submission encoding to the
model violates the candidate contract. For these experiments, pass `feature_sources`,
`feature_transformations`, and `leakage_controls` to `run_model`.

Measure validation accuracy and runtime/complexity tradeoffs. Do not end the experiment until you
have called `run_model` or exhausted a genuine repair path. Keep tool-facing explanations concise;
full stdout, tracebacks, code, and diffs are already retained by the harness.
