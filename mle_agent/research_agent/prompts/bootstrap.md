Bootstrap the autonomous MLE run before beginning experiment 1.

- Baseline candidate directory: `${candidate_dir}`
- Available bootstrap turns: `${max_turns}`

Discover and completely read the task sources, including the official baseline, evaluation,
feature/data implementation, and organizer feature-ablation code; inspect the allowed data and
the available ML frameworks, retrieve relevant literature, and
explicitly reproduce the official baseline using `reproduce_baseline`.
Interpret that score, then call `record_task_context` once all prerequisites are complete.
Its `data_splits` object must use exactly the keys `train`, `validation`, and `test`.
Its `source_paths` must include the fully read primary README, official baseline, official
evaluation code, official `data.py`, organizer `ablation_features.py`, and inherited candidate
`model.py`. Its feature context must explicitly preserve the five baseline fields as the reference,
record that adding all 13 static fields was measured and did not help, distinguish promising
feature families from that dead end, and state train-only leakage controls. `measured_dead_ends`
must also record the two other directions the README rules out: raising embedding dimension
k=8/16/32 stayed flat (~0.589, capacity is not the bottleneck), and purely user-side first-order
features contribute zero to within-user ranking. Do not cite extra discovered files unless you
also read them completely.

After `record_task_context`, call `record_research_backlog` once with 6-10 ranked candidate
research families before any experiment. Draw them from the README's "Unexplored" directions 1-7:
a ranking-aligned loss (BPR / listwise softmax), user-history sequences (DIN / SIM), multiple
objectives (`is_click` / `is_like` auxiliary tasks), censored watch-time modelling, a different
model (DeepFM / DCN / xDeepFM), time / distribution-shift features. Each entry needs a hypothesis,
a `target_component`, an `evidence_id` (a `search_ml_literature` chunk id or an EDA finding id),
a numeric `expected_primary_delta`, an `estimated_cost`, and a `falsification_criterion`. The
first three entries must target three distinct components; none may merely restate a measured
dead end.

The organizer's NumPy baseline is a reference, not a restriction. Retain the environment inventory
from the dedicated per-run venv so later experiments can choose PyTorch, sklearn, pandas, or another
installed framework. Do not request installs during bootstrap; request a missing dependency only for
a concrete experiment after the task context is complete.

This is a context-and-baseline phase only. Do not edit `model.py` or propose the first model change yet. Use independent tool calls together when practical, follow every `next_offset`,
and do not end until the retained task context and the ranked research backlog are both accepted.
