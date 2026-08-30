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
feature families from that dead end, and state train-only leakage controls. Do not cite extra
discovered files unless you also read them completely.

The organizer's NumPy baseline is a reference, not a restriction. Retain the environment inventory
so later experiments can choose PyTorch, sklearn, pandas, or another installed framework. Do not
request installs during bootstrap; request a missing dependency only for a concrete experiment after
the task context is complete.

This is a context-and-baseline phase only. Do not edit `model.py` or propose the first model change yet. Use independent tool calls together when practical, follow every `next_offset`,
and do not end until the retained task context is accepted.
