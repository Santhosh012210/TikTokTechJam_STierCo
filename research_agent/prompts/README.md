# Agent prompt templates

One Markdown template per agent stage, rendered with `string.Template` (`${placeholder}`
substitution) so JSON examples and ordinary braces need no escaping.

| Template | Stage |
|---|---|
| `bootstrap.md` | Problem reading, task extraction, initial EDA |
| `builder.md` | Inspect, hypothesize, edit, execute, repair |
| `strategist.md` | Tree state, evidence, dead ends, research direction |
| `reflection.md` | Result interpretation, noise-floor comparison, next step |

Templates are resolved relative to the installed `research_agent` package, never the current working directory. The loader records each template's filename and content hash in run evidence so prompt changes are auditable.

Prompts are still inline in `builder.py` and `strategist.py`; extracting them is a later step (see `temp/plan/agent-harness-architecture-review.md`).
