# Method corpus

One Markdown file per method. Not indexed: this README.

## Format

Every file uses the same four `##` sections, because sections are the retrieval unit —
a query about implementation detail should return an implementation section, not a whole
paper summary.

```markdown
# Method name

Tags: comma, separated, tags

Source: Author, "Title", Venue Year. arXiv:NNNN.NNNNN

## What it is
## Why it helps a ranking metric
## How to implement
## When it will not help
```

`Tags:` and the H1 are appended to every chunk's searchable text, so a query naming a
method reaches all of its sections.

The `When it will not help` section is not padding. An agent that reads a method, decides
it does not fit the evidence, and says why is demonstrating judgment — and that reasoning
lands in the run log where judges can see it.

## Scope

Methods, not library documentation. The model already knows the XGBoost and PyTorch APIs
from pretraining; retrieving API reference spends tokens to say what it already knows.
What it does not reliably know is how a method interacts with *this* task — a within-user
ranking metric on a logged short-video feed — which is what these files cover.

Deliberately includes methods that are unlikely to work here. Breadth is what makes the
agent's selection meaningful rather than a lookup of a pre-filtered shortlist.

## Adding a method

Drop in a new `.md` file following the format above. The index is built from the
directory at import time, so no registration step and no rebuild command.

Check retrieval afterwards:

    python -m mle_agent.research_agent.knowledge "your query" -k 3
    python -m mle_agent.research_agent.knowledge --list

## TODO: tool-use rails (write these when the Builder prompt is written)

Retrieval is BM25, so it matches on shared vocabulary. It is strong when the caller uses
precise technical terms and weak on paraphrase — a query like "my score went up on dev but
I do not trust it" returns nothing useful, while "unbiased validation leakage" finds the
right passage. The caller is an LLM, so this is fixable with instructions rather than by
changing the retrieval backend.

Rails to add to the `search_ml_literature` tool description and to `builder.md`:

- Query with method names and technical terms, not natural-language questions.
- Use the `[slug]` from the catalogue when expanding a specific method.
- Cite the returned `chunk_id` in the hypothesis, so the run log records provenance.

The stronger half of the fix is structural: at this corpus size the **whole catalogue fits
in the prompt**. Inject `list_methods()` (19 lines) so the agent sees every available method
by name, then let it call the tool only to expand one it has already chosen. That turns
retrieval from *search* — which needs semantics BM25 does not have — into *fetch*, which is
exact-match and what BM25 is best at. It also removes the residual failure that rails alone
cannot fix: lexical search can only find what the caller already knows to name.

`mle_agent/harness/main.py` already injects the catalogue into the Strategist prompt. The Builder does
not yet have the tool registered at all.

If paraphrase gaps still appear after that, the cheap fix is adding synonyms to a file's
`Tags:` line — tags are indexed into every chunk of that file — not swapping in embeddings.
