"""Offline tests for the method corpus and its retrieval.

No API key, no network, no dataset. Run either way:

    python tests/test_knowledge.py
    pytest tests/test_knowledge.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from research_agent.knowledge import list_methods, search_ml_literature
from research_agent.knowledge.index import CORPUS_DIR, build_index, parse_document

REQUIRED_SECTIONS = {
    "What it is",
    "Why it helps a ranking metric",
    "How to implement",
    "When it will not help",
}


def test_every_document_has_the_required_sections():
    """The four-section format is what makes section-level retrieval useful."""
    for path in sorted(CORPUS_DIR.glob("*.md")):
        if path.name == "README.md":
            continue
        sections = {c.section for c in parse_document(path)}
        missing = REQUIRED_SECTIONS - sections
        assert not missing, f"{path.name} is missing sections: {sorted(missing)}"


def test_every_document_declares_tags_and_a_source():
    for path in sorted(CORPUS_DIR.glob("*.md")):
        if path.name == "README.md":
            continue
        text = path.read_text(encoding="utf-8")
        assert "\nTags:" in text, f"{path.name} has no Tags: line"
        assert "\nSource:" in text, f"{path.name} has no Source: line"
        chunks = parse_document(path)
        assert chunks and chunks[0].tags, f"{path.name} tags did not parse"


def test_readme_is_not_indexed():
    methods = {m["method"] for m in list_methods()}
    assert "README" not in methods and "readme" not in methods


def test_retrieval_finds_the_expected_method():
    """Queries a hypothesis would plausibly ask, and the method that should win."""
    expectations = [
        ("sample negatives from the same user", "negative-sampling"),
        ("attention over user behaviour history", "din"),
        ("watch time duration bias censoring", "censored-regression"),
        ("pairwise loss bayesian personalized ranking", "bpr"),
        ("user side features cannot reorder within a user", "within-user-ranking"),
        ("auxiliary task entire space multi task", "esmm"),
        ("xgboost lambdarank group by user", "gbdt-ranking"),
    ]
    for query, expected in expectations:
        hits = search_ml_literature(query, k=3)["results"]
        assert hits, f"no results for {query!r}"
        methods = [h["method"] for h in hits]
        assert expected in methods, f"{query!r} returned {methods}, expected {expected}"


def test_results_are_deterministic():
    """A judge re-running a logged query must get the logged passages back."""
    query = "listwise softmax over impressions"
    first = search_ml_literature(query, k=5)
    for _ in range(3):
        assert search_ml_literature(query, k=5) == first


def test_k_is_clamped_and_empty_queries_are_rejected():
    assert len(search_ml_literature("ranking", k=100)["results"]) <= 10
    assert len(search_ml_literature("ranking", k=0)["results"]) == 1
    assert search_ml_literature("  ", k=3)["results"] == []
    assert "error" in search_ml_literature("", k=3)


def test_a_query_matching_nothing_returns_nothing():
    """Zero-score chunks are dropped rather than padding the agent's context."""
    assert search_ml_literature("zzzzq nonexistentterm", k=3)["results"] == []


def test_index_covers_every_corpus_file():
    files = {p.stem for p in CORPUS_DIR.glob("*.md") if p.name != "README.md"}
    assert {m["method"] for m in list_methods()} == files
    assert len(build_index()) >= 4 * len(files)


def main() -> None:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print(f"  PASS  {test.__name__}")
    print(f"\n{len(tests)} tests passed.")


if __name__ == "__main__":
    main()
