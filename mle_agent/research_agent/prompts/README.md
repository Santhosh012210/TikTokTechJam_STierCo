# Agent prompt templates

One Markdown template per agent stage, rendered with `string.Template` (`${placeholder}`
substitution) so JSON examples and ordinary braces need no escaping.

| Template | Stage |
|---|---|
| `agent.md` | Persistent identity, scientific principles, task rails, complete MLE loop |
| `bootstrap.md` | Deterministic bootstrap digest and retained-context requirements |
| `iteration.md` | Per-experiment state, bootstrap/continuation instruction, execution requirement |
| `prior_findings.md` | Reviewed cross-run findings injected into bootstrap and every experiment |

Templates are resolved relative to the installed `research_agent` package, never the
current working directory. The loader records each template's filename and SHA-256 hash
in run evidence so prompt changes are auditable.

The previous Builder and Strategist prompts remain inline in their legacy modules; the
single-agent runner does not use them.
