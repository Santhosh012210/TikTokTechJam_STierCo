# Single-agent research run 20260827_174200

## Outcome

- Reproduced FM baseline validation primary: 0.601855
- Best validation primary: 0.602654
- Delta versus published baseline: +0.001054
- Best trial: trial_001
- Successful autonomous experiments: 1

## Resource usage

- Input tokens: 28977
- Output tokens: 1803
- Manual interventions: 0

## Prompt templates

```json
[
  {
    "name": "single_agent.md",
    "path": "research_agent/prompts/single_agent.md",
    "template_sha256": "b1e254d8a17152138a8b546d7c18dec7135af6866d41c8c24c7683f9d1649636"
  },
  {
    "name": "iteration.md",
    "path": "research_agent/prompts/iteration.md",
    "template_sha256": "f04352ee3e7f9702c1c7ffa7e35b291f5afd85aa77249f378d1271e121d2d354"
  }
]
```

## Architecture

One persistent agent conversation owned EDA, research, hypothesis selection, code,
execution, repair, and reflection. Python retained budgets, validation-only execution,
baseline verification, and evidence logging.
