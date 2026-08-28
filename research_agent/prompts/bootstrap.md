Bootstrap the autonomous MLE run before beginning experiment 1.

- Baseline candidate directory: `${candidate_dir}`
- Available bootstrap turns: `${max_turns}`

Discover and completely read the task sources, including the official baseline and
evaluation implementations; inspect the allowed data, retrieve relevant literature, and
explicitly reproduce the official baseline using `reproduce_baseline`.
Interpret that score, then call `record_task_context` once all prerequisites are complete.

This is a context-and-baseline phase only. Do not edit `model.py` or propose the first model
change yet. Use independent tool calls together when practical, follow every `next_offset`,
and do not end until the retained task context is accepted.
