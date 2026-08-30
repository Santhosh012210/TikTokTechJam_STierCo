"""The ``search_ml_literature`` tool: query the local method corpus.

Public surface is one function plus one tool schema. The function is pure and
offline, so it can be exercised from the CLI with no API key and no cost:

    python -m mle_agent.research_agent.knowledge "pairwise ranking loss gradient"
"""
from __future__ import annotations

import argparse
import json
import sys

from mle_agent.research_agent.knowledge.index import get_index

DEFAULT_K = 3
MAX_K = 10

# Chunks scoring at or below this are not returned at all. BM25 scores are
# unbounded, but a query sharing no meaningful term with a chunk scores 0, and
# returning zero-score chunks would pad the context with noise the agent would
# then have to reason about.
_MIN_SCORE = 1e-9


def search_ml_literature(query: str, k: int = DEFAULT_K) -> dict:
    """Search the local recommender-systems method corpus.

    Args:
        query: Free-text query. Method names, metric names, and implementation
            terms all work ("BPR gradient", "within-user negative sampling",
            "multi-task auxiliary loss").
        k: Number of passages to return (1..10).

    Returns:
        A dict with the echoed query and a ``results`` list, each entry holding
        ``chunk_id``, ``method``, ``title``, ``section``, ``score``, and
        ``text``. ``chunk_id`` is stable across runs and is what should be
        recorded in the run log as the provenance of a hypothesis.
    """
    if not isinstance(query, str) or not query.strip():
        return {"query": query, "results": [], "error": "query must be a non-empty string"}

    k = max(1, min(int(k), MAX_K))
    scored = get_index().score(query)

    results = [
        {
            "chunk_id": chunk.chunk_id,
            "method": chunk.method,
            "title": chunk.title,
            "section": chunk.section,
            "score": round(score, 4),
            "text": chunk.text,
        }
        for score, chunk in scored[:k]
        if score > _MIN_SCORE
    ]

    return {"query": query, "results": results}


def list_methods() -> list[dict]:
    """Every method in the corpus: ``method`` slug, ``title``, and ``tags``.

    Used to tell an agent what the corpus contains without spending context on
    the contents. Deliberately unordered by merit — the corpus is a library to
    select from, not a ranked plan to work through, and alphabetical order
    avoids implying a priority the evidence has not established.
    """
    seen: dict[str, dict] = {}
    for chunk in get_index().chunks:
        if chunk.method not in seen:
            seen[chunk.method] = {
                "method": chunk.method,
                "title": chunk.title,
                "tags": list(chunk.tags),
            }
    return [seen[k] for k in sorted(seen)]


# ---------------------------------------------------------------------------
# Tool schema (canonical format; mle_agent/harness/provider.py converts to vendor format)
# ---------------------------------------------------------------------------
#
# Declared here so it is versioned next to the implementation. It is NOT yet
# registered in any tool list — wiring it into a specific agent stage is a
# separate decision about the agent loop.

SEARCH_ML_LITERATURE_TOOL = {
    "name": "search_ml_literature",
    "description": (
        "Search a local corpus of established recommender-systems methods "
        "(ranking losses, sequence models, multi-task architectures, watch-time "
        "and debiasing methods). Returns passages describing what a method is, "
        "why it helps a ranking metric, and how to implement it. Use this before "
        "proposing a change, and cite the returned chunk_id in your hypothesis."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "What you want to know. Method names, metric names, or "
                    "implementation questions all work."
                ),
            },
            "k": {
                "type": "integer",
                "description": f"Number of passages to return (1-{MAX_K}, default {DEFAULT_K}).",
                "default": DEFAULT_K,
            },
        },
        "required": ["query"],
    },
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Query the local method corpus (offline; no API key needed)."
    )
    ap.add_argument("query", nargs="*", help="Query text.")
    ap.add_argument("-k", type=int, default=DEFAULT_K, help=f"Passages to return (default {DEFAULT_K}).")
    ap.add_argument("--json", action="store_true", help="Emit raw JSON instead of formatted text.")
    ap.add_argument("--list", action="store_true", help="List every indexed chunk and exit.")
    args = ap.parse_args()

    if args.list:
        index = get_index()
        for chunk in sorted(index.chunks, key=lambda c: c.chunk_id):
            print(f"{chunk.chunk_id:<48} {chunk.title}")
        print(f"\n{len(index)} chunks indexed.")
        return

    if not args.query:
        ap.error("provide a query, or use --list")

    payload = search_ml_literature(" ".join(args.query), k=args.k)

    if args.json:
        print(json.dumps(payload, indent=2))
        return

    if not payload["results"]:
        print(f"No matches for {payload['query']!r}.", file=sys.stderr)
        sys.exit(1)

    for hit in payload["results"]:
        print("=" * 72)
        print(f"{hit['chunk_id']}  (score {hit['score']})")
        print(f"{hit['title']} — {hit['section']}")
        print("=" * 72)
        print(hit["text"])
        print()


if __name__ == "__main__":
    main()
