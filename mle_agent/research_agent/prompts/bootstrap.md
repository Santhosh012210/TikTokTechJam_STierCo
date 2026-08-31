Bootstrap the autonomous MLE run before beginning experiment 1.

- Baseline candidate directory: `${candidate_dir}`
- Available bootstrap turns: `${max_turns}`

${remaining_work}

${bootstrap_digest}

Experiments already measured in earlier runs of this agent (harness-written
operational log, includes failures — treat settled outcomes as settled):

```json
${cross_run_history}
```

Verified prior autonomous-run evidence (audited, read-only):

```json
${prior_experiment_evidence}
```

Interpret the reproduced baseline score, then call `record_task_context`.
Its `data_splits` object must use exactly the keys `train`, `validation`, and `test`.
Its `source_paths` must include the primary README, official baseline, official
evaluation code, official `data.py`, organizer `ablation_features.py`, and inherited candidate
`model.py`. Its feature context must explicitly preserve the five baseline fields as the reference,
record that adding all 13 static fields was measured and did not help, distinguish promising
feature families from that dead end, and state train-only leakage controls. `measured_dead_ends`
must also record the two other directions the README rules out: raising embedding dimension
k=8/16/32 stayed flat (~0.589, capacity is not the bottleneck), and purely user-side first-order
features contribute zero to within-user ranking.

The digest above is curated, not lossy by accident: function bodies are elided and non-numeric
README prose is dropped. If a claim you are about to record depends on an exact implementation
detail rather than on the contract, call `read_file` on that path first and read the region that
settles it. Those reads are logged as your own decisions, so read what you actually need — and
do not re-read what the digest already states.

After `record_task_context`, call `record_research_backlog` once with 6-10 ranked candidate
research families before any experiment. Draw them from the README's "Unexplored" directions 1-7:
a ranking-aligned loss (BPR / listwise softmax), user-history sequences (DIN / SIM), multiple
objectives (`is_click` / `is_like` auxiliary tasks), censored watch-time modelling, a different
model (DeepFM / DCN / xDeepFM), time / distribution-shift features. Each entry needs a hypothesis,
a `target_component`, an `evidence_id` (a `search_ml_literature` chunk id or an EDA finding id),
a numeric `expected_primary_delta`, an `estimated_cost`, and a `falsification_criterion`. The
first three entries must target three distinct components; none may merely restate a measured
dead end. At least one of the first three must cite the supplied `prior-run:` evidence and propose
a controlled consolidation, isolation, or extension of the verified 0.603718 frontier. Preserve
the attribution warning: the retained source is pointwise BCE, not BPR, and its feature effects are
confounded. Do not spend an experiment blindly rediscovering the same sequence.

The organizer's NumPy baseline is a reference, not a restriction. The environment inventory in the
digest is what later experiments may choose from — PyTorch, sklearn, pandas, or another installed
framework. Do not request installs during bootstrap; request a missing dependency only for a
concrete experiment after the task context is complete.

This is a context-and-baseline phase only. Do not edit `model.py` or propose the first model change
yet. Use independent tool calls together when practical, and do not end until the retained task
context and the ranked research backlog are both accepted.
