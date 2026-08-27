"""Local method knowledge base.

A pre-seeded corpus of established recommender-systems methods, searchable
offline. It exists so the agent can ground a hypothesis in published work
instead of inventing one, and so the run log can show *which* passages a
hypothesis was drawn from.

Retrieval is BM25 over Markdown files in ``corpus/``. No embeddings, no vector
store, no network: the corpus is small enough that lexical search is both
sufficient and deterministic, and determinism means a judge can reproduce any
retrieval in the run log exactly.
"""

from research_agent.knowledge.search import (
    SEARCH_ML_LITERATURE_TOOL,
    list_methods,
    search_ml_literature,
)

__all__ = ["search_ml_literature", "list_methods", "SEARCH_ML_LITERATURE_TOOL"]
