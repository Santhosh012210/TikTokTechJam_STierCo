"""Chunking and BM25 ranking over the local method corpus.

Pure standard library. The index is built once per process on first
use and cached; the corpus is a few dozen files, so build time is milliseconds
and persistence would only add a staleness bug.

Chunking is by Markdown ``##`` section. Sections are the natural retrieval unit
here because every corpus file uses the same four headings (What it is / Why it
helps / How to implement / When it will not help), so a query about
implementation detail retrieves an implementation section rather than a whole
paper summary.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path

CORPUS_DIR = Path(__file__).resolve().parent / "corpus"

# BM25 parameters. k1 controls term-frequency saturation, b controls how much
# document length normalises the score. These are the standard defaults.
_K1 = 1.5
_B = 0.75

# Split on anything that is not a word character or hyphen, so "multi-task" and
# "long_view" survive tokenisation as single terms.
_TOKEN_RE = re.compile(r"[a-z0-9_\-]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


@dataclass(frozen=True)
class Chunk:
    """One ``##`` section of one corpus file."""

    chunk_id: str      # "bpr#how-to-implement"
    method: str        # "bpr" (the filename stem)
    title: str         # H1 of the file
    section: str       # H2 of this chunk
    text: str          # section body, heading excluded
    tags: tuple[str, ...]


_HEADING_SLUG_RE = re.compile(r"[^a-z0-9]+")


def _slug(text: str) -> str:
    return _HEADING_SLUG_RE.sub("-", text.lower()).strip("-")


def parse_document(path: Path) -> list[Chunk]:
    """Split one Markdown file into per-section chunks.

    Recognises an optional ``Tags:`` line in the preamble (before the first
    ``##``); tags are appended to every chunk's searchable text so that a query
    naming a method category reaches all of that category's sections.
    """
    lines = path.read_text(encoding="utf-8").splitlines()
    title = path.stem
    tags: tuple[str, ...] = ()
    chunks: list[Chunk] = []

    current_section: str | None = None
    buffer: list[str] = []

    def flush() -> None:
        if current_section is None:
            return
        body = "\n".join(buffer).strip()
        if not body:
            return
        chunks.append(
            Chunk(
                chunk_id=f"{path.stem}#{_slug(current_section)}",
                method=path.stem,
                title=title,
                section=current_section,
                text=body,
                tags=tags,
            )
        )

    for line in lines:
        if line.startswith("# "):
            title = line[2:].strip()
        elif line.startswith("## "):
            flush()
            current_section = line[3:].strip()
            buffer = []
        elif current_section is None and line.lower().startswith("tags:"):
            tags = tuple(t.strip() for t in line.split(":", 1)[1].split(",") if t.strip())
        else:
            buffer.append(line)
    flush()

    return chunks


class BM25Index:
    """Okapi BM25 over a fixed list of chunks.

    Deterministic: ties are broken by ``chunk_id`` so repeated queries against
    an unchanged corpus always return the same ordering.
    """

    def __init__(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks
        # Each chunk is indexed over its own text plus its title and tags, so a
        # query naming the method reaches every section of that method's file.
        self._docs: list[list[str]] = [
            tokenize(f"{c.title} {c.section} {' '.join(c.tags)} {c.text}") for c in chunks
        ]
        self._lengths = [len(d) for d in self._docs]
        self._avg_len = (sum(self._lengths) / len(self._docs)) if self._docs else 0.0

        self._term_freqs: list[dict[str, int]] = []
        doc_freq: dict[str, int] = {}
        for doc in self._docs:
            tf: dict[str, int] = {}
            for term in doc:
                tf[term] = tf.get(term, 0) + 1
            self._term_freqs.append(tf)
            for term in tf:
                doc_freq[term] = doc_freq.get(term, 0) + 1

        n = len(self._docs)
        # Standard BM25 idf with the +1 smoothing that keeps values positive for
        # terms appearing in more than half the corpus.
        self._idf = {
            term: math.log(1.0 + (n - df + 0.5) / (df + 0.5)) for term, df in doc_freq.items()
        }

    def score(self, query: str) -> list[tuple[float, Chunk]]:
        """Return every chunk with its BM25 score, best first."""
        terms = tokenize(query)
        scored: list[tuple[float, Chunk]] = []

        for i, chunk in enumerate(self.chunks):
            tf = self._term_freqs[i]
            length = self._lengths[i]
            total = 0.0
            for term in terms:
                freq = tf.get(term)
                if not freq:
                    continue
                denom = freq + _K1 * (1 - _B + _B * length / self._avg_len)
                total += self._idf[term] * (freq * (_K1 + 1)) / denom
            scored.append((total, chunk))

        scored.sort(key=lambda pair: (-pair[0], pair[1].chunk_id))
        return scored

    def __len__(self) -> int:
        return len(self.chunks)


_index: BM25Index | None = None


def build_index(corpus_dir: Path | None = None) -> BM25Index:
    """Build a fresh index. Used by tests; normal callers use ``get_index``."""
    directory = corpus_dir or CORPUS_DIR
    if not directory.is_dir():
        raise FileNotFoundError(f"Corpus directory not found: {directory}")

    chunks: list[Chunk] = []
    for path in sorted(directory.glob("*.md")):
        if path.name.upper() == "README.MD":
            continue
        chunks.extend(parse_document(path))

    if not chunks:
        raise ValueError(f"Corpus at {directory} produced no chunks")
    return BM25Index(chunks)


def get_index() -> BM25Index:
    """Process-cached index over the packaged corpus."""
    global _index
    if _index is None:
        _index = build_index()
    return _index
