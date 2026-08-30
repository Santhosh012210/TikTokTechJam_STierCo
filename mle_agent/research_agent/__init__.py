"""Autonomous research judgment: what to investigate, change, and try next.

This package owns the agent's reasoning — hypothesis formation, code authoring
and repair, research direction, and search over candidates. It is deliberately
provider-agnostic: no Anthropic / Groq / OpenAI / Gemini / Ollama SDK may be
imported from here. The deterministic runtime that executes and records its
decisions lives in ``harness``.

Migration note: ``builder`` and ``strategist`` still import ``harness.config``,
``harness.provider``, and ``harness.tools``. That is the mechanical-move state.
The target is dependency inversion — the runtime injects client, tools, and
config values, and this package imports nothing from ``harness``. See
``temp/plan/agent-harness-architecture-review.md`` (A2).
"""
