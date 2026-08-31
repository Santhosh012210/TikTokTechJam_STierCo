"""Content-addressed sidecar for bulky, reconstructible trace payloads.

The run log has two jobs that pull in opposite directions. Judges read it to
assess autonomy and robustness, which wants a small, legible JSONL. Reproducing
exactly what the model saw at turn 37 wants every byte of every tool output,
which is megabytes of file text and training stdout.

The split here: the event stays in the JSONL with its summary, sizes, and digest;
the payload itself is written once to ``blobs/<sha256>`` and referenced. Identical
content — the same file read twice, the same traceback repeated across a repair
loop — is stored once, which is common enough in these runs to matter.

The digest in the event is the digest of the stored bytes, so the reference can be
verified rather than trusted.
"""
from __future__ import annotations

import hashlib
from pathlib import Path


#: Payloads at or below this size stay inline in the JSONL. Below roughly this
#: length a blob reference costs more bytes than the content it replaces, and
#: short payloads are exactly the ones a reader wants to see without a lookup.
INLINE_MAX_CHARS = 2000

#: How much of a large payload is still shown inline, so the event remains
#: readable on its own without resolving the blob.
PREVIEW_CHARS = 800


class BlobStore:
    """Write-once, deduplicating store keyed by content digest."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self._known: set[str] = set()

    def _path(self, digest: str) -> Path:
        # One level of fan-out keeps directory listings usable on a long run.
        return self.root / digest[:2] / digest

    def put(self, text: str) -> str:
        """Store ``text`` and return its sha256, writing only on first sight."""
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if digest in self._known:
            return digest
        path = self._path(digest)
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            # Write to a temporary name and rename, so a crash mid-write can
            # never leave a truncated blob that still matches its digest path.
            temporary = path.with_suffix(".partial")
            temporary.write_text(text, encoding="utf-8")
            temporary.replace(path)
        self._known.add(digest)
        return digest

    def get(self, digest: str) -> str | None:
        path = self._path(digest)
        return path.read_text(encoding="utf-8") if path.is_file() else None

    def reference(self, text: str, *, field: str) -> dict[str, object]:
        """Return the event fields describing ``text``, offloading it when large.

        Small payloads come back inline and untouched. Large ones come back as a
        preview plus a verifiable blob reference.
        """
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if len(text) <= INLINE_MAX_CHARS:
            return {
                field: text,
                f"{field}_chars": len(text),
                f"{field}_sha256": digest,
                f"{field}_storage": "inline",
            }
        self.put(text)
        return {
            f"{field}_preview": text[:PREVIEW_CHARS],
            f"{field}_chars": len(text),
            f"{field}_sha256": digest,
            f"{field}_storage": "blob",
            f"{field}_blob": f"blobs/{digest[:2]}/{digest}",
            f"{field}_truncated": True,
        }
