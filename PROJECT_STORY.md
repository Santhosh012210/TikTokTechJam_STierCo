# The Porject Story

---

## About the project

### Inspiration

Every ML competition ends the same way: a human sits in a loop of *try something → train →
read the number → decide what to try next*, for hours. The interesting part is the decision.
The rest is typing.

So the narrower question rather than "can an AI do ML research" was **can an agent run that loop
unattended on a benchmark where the honest gains are smaller than the noise?** 

KuaiRand-Pure's within-user ranking task is a good adversary for this. The published seed variance of the official Factorization Machine baseline is ±0.0008 primary, and a genuinely good idea might move the metric by 0.003. An agent that can't tell signal from seed noise will confidently chase
nothing for hours and report a win. That failure mode — not code generation — is the real
problem.

### What it does

One persistent LLM research session owns the complete MLE loop on KuaiRand-Pure: read the
benchmark, inspect the training and validation data, search a literature corpus, form a
hypothesis, write a self-contained model, train it, score it, repair its own crashes, reflect,
and choose what to try next — until it converges or exhausts its budget.

The submitted run converged on its own after 9 experiments in ~21 minutes with **zero manual
interventions**. It landed on a DCN-V2 with one added hour-of-day context field derived from the
log's `hourmin` column:

| Validation metric | Official FM baseline | Agent's final candidate | Delta |
|---|---:|---:|---:|
| GAUC | 0.6674 | 0.6706 | **+0.0032** |
| nDCG@5 | 0.5357 | 0.5376 | **+0.0019** |
| primary | 0.6016 | 0.6041 | **+0.0025** |

Re-scored on five fixed seeds it holds at $0.604293 \pm 0.000198$ — a spread 0.2x the published
FM 5-seed std of $0.0008$, so the $+0.0025$ gain is about three times seed noise rather than a
lucky draw.

What we like most is *why* the agent's idea works, because it shows the reasoning was real
rather than decorative. Scoring happens strictly inside each user's own impression list, so any
term constant within a user cannot change the intra-user order — the organisers had already
measured that adding 13 static user fields does nothing. Hour-of-day varies *within* a user's
impressions. The agent found the one class of cheap feature that the metric can actually see.

### How we built it

The architecture is one deliberate split: **the agent decides, the harness enforces.**

`research_agent/` holds the reasoning — a single persistent LangChain-backed session (not a fresh
prompt per experiment, so it accumulates context across the whole run), phase-aware memory
compaction, prompts, and an offline BM25 corpus of 19 method cards covering ranking losses,
sequence models, multi-task heads, feature-interaction models, and debiasing. Retrieval is BM25
over Markdown, not embeddings, so every passage the agent cites in a run log can be reproduced
exactly.

`harness/` is deterministic Python and owns everything that must not be left to a language
model's good intentions: budgets, data boundaries, subprocess execution, metric validation,
convergence, final selection, and durable evidence. The pieces that mattered most:

- **A count-verified, run-local data view.** Before the agent is called at all, the harness
  materialises exactly the official train and validation date ranges. Hidden-test rows never
  reach the candidate process. This is a filesystem fact, not a prompt instruction.
- **A search tree, not a chain.** The first four trials branch from baseline across loss,
  features, model, and sequence; later trials pick among the conservative top three by a UCB
  score whose exploration term is scaled by *measured* noise. Every node freezes its source,
  config, seed, metrics, and parent, and reward propagates through the lineage.
- **Convergence tied to published variance.** $\epsilon = 0.002$, $N = 3$, evaluated over the
  whole frontier and only after at least eight scored variants.
- **Conservative final selection** — multi-seed mean minus standard deviation. It can never fall
  through to "the last file that ran".
- **Self-repair with teeth.** A post-save hook syntax-checks every write immediately; a failed
  save *gates execution* until it's repaired, so a broken candidate can't burn a training run.
- **A dedicated venv per run**, with an ML package allowlist that installs binary wheels
  automatically and stops for a human `y/n` on anything off-list. The resolved environment is
  written to a lockfile beside the evidence.
- **A trusted finalizer.** It verifies the frozen manifest, reproduces the recorded validation
  metrics exactly, then generates hidden-split predictions — without ever computing a test score.

Every experiment writes a schema-validated record: hypothesis, reasoning, exact diff, metrics,
execution attempts, errors, recovery outcome, reflection, tokens, wall time, and an intervention
flag. The claim "zero manual interventions" is a field in the log, not a sentence in a README.

### Challenges we ran into

**`baseline.py` returns test metrics on every call.** The organisers' helper computes validation
*and* test and hands back both. A prompt telling an agent "don't look at test" is not a control.
The harness drops the test key at the boundary, and the data view the agent sees doesn't contain
the rows in the first place.

**Noise is the actual adversary.** Our first instinct — keep the candidate if its number went up
— is indefensible when the seed std is 0.0008. Comparisons had to become noise-aware everywhere:
the convergence threshold, the UCB exploration term, and a final selection rule that subtracts
the standard deviation instead of trusting the peak.

### What we learned

That the hard part of an autonomous research agent isn't getting a model to write code — it's
building the surrounding system that makes the agent's claims *checkable*. Nearly every design
decision we're proud of is a place where we moved a guarantee out of the prompt and into Python:
the data boundary, the test-metric drop, the syntax gate, the convergence rule, the final
selection. The agent got more autonomy precisely because the harness got less trusting.

And on the ML side: the benchmark's structure told us more than any model choice. Once you
internalise that ranking is *within-user*, a whole class of features is provably useless and one
narrow class is promising — which is exactly the conclusion the agent reached on its own.

### What's next

Multi-seed scoring *inside* the search loop, so intermediate keep/revert decisions stop being
made at the noise floor. A second held-out window to measure how much of the gain is selection
bias from repeatedly picking against one validation set. Parallel trials — they're independent
and ~2 minutes each, so four at a time turns a 9-experiment run into a 36-experiment one for the
same wall clock. And a fair shot for the two directions the benchmark most obviously rewards:
within-user listwise losses and a short user-history encoder.

---

## Built with

```
python
langchain
pytorch
numpy
openai
anthropic-claude
google-gemini
llm-agents
autonomous-agents
tool-calling
learning-to-rank
dcn-v2
factorization-machines
feature-engineering
bm25
kuairand
tree-search
ucb
pydantic
subprocess-sandboxing
```

---

## "Try it out" links

- **Code:** https://github.com/Santhosh012210/TikTokTechJam_STierCo
- **Reproduce the submitted model** (frozen source, config, and seeds):
  https://github.com/Santhosh012210/TikTokTechJam_STierCo/tree/main/artifacts/final
- **Full run evidence for the submitted run** — per-iteration hypotheses, diffs, metrics, and
  the convergence trace: https://github.com/Santhosh012210/TikTokTechJam_STierCo/tree/main/artifacts/final/run
