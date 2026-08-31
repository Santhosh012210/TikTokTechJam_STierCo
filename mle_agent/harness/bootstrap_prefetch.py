"""Deterministic bootstrap: do the mechanical work in Python, then curate for the LLM.

Bootstrap used to cost 15-24 model turns in which the agent paged six files
through ``read_file`` a few thousand characters at a time, ran the EDA, inventoried
the environment, and reproduced the baseline — every step a billed round trip whose
outcome was fully determined by the repository. None of it required a model.

So the harness does all of it before the first model call and hands the agent one
curated digest. Curation is structural rather than lossy summarisation: for Python
sources it keeps the module docstring, the API surface, and the constants that
encode the task's rules, and drops function bodies; for the README it keeps the
headings and every line carrying a number, because on this benchmark the numbers
*are* the content — the published scores, the convergence rule, the measured
ablations.

The agent keeps ``read_file`` throughout, so anything the curation dropped is one
call away, and each such call is recorded. That distinction matters for the run
log: the read gate is satisfied by harness provenance, and what the agent chose to
look at in full is separately visible as its own decision.
"""
from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass, field
from pathlib import Path


#: README lines mentioning a number are kept verbatim; on this benchmark those
#: carry the scores, the split dates, the tolerance, and the ablation results.
_NUMBER = re.compile(r"\d")
_HEADING = re.compile(r"^\s{0,3}#{1,6}\s+\S")
_TABLE_ROW = re.compile(r"^\s*\|")


@dataclass
class SourceSummary:
    """One curated source file."""

    path: str
    kind: str
    total_chars: int
    summary: str
    curated_chars: int = 0
    note: str = ""

    def __post_init__(self) -> None:
        self.curated_chars = len(self.summary)


@dataclass
class BootstrapDigest:
    """Everything the agent needs to start reasoning, with nothing it can derive."""

    sources: list[SourceSummary] = field(default_factory=list)
    baseline: dict[str, object] = field(default_factory=dict)
    data: dict[str, object] = field(default_factory=dict)
    environment: dict[str, object] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    def provenance(self) -> dict[str, object]:
        """Audit record: what the harness read, and how much survived curation."""
        return {
            "prefetched_by": "harness",
            "sources": [
                {
                    "path": source.path,
                    "kind": source.kind,
                    "total_chars": source.total_chars,
                    "curated_chars": source.curated_chars,
                }
                for source in self.sources
            ],
            "raw_chars": sum(source.total_chars for source in self.sources),
            "curated_chars": sum(source.curated_chars for source in self.sources),
            "baseline_reproduced": bool(self.baseline.get("success")),
            "errors": self.errors,
        }

    def render(self) -> str:
        """The digest as one message, in the order the agent should read it."""
        parts = [
            "## Harness bootstrap digest",
            "",
            "The harness has already discovered and read every required source, run the "
            "train/validation EDA, inventoried the ML environment, and reproduced the "
            "official baseline. Those steps are complete and recorded; do not repeat them.",
            "",
            "Below is a *curated* view: full docstrings, signatures, constants, and every "
            "numeric line, with function bodies elided. If you need an exact implementation "
            "detail, call `read_file` on the path — the full text is still available and "
            "your reads are logged as your own decisions.",
            "",
            "### Reproduced official baseline",
            "```json",
            json.dumps(self.baseline, ensure_ascii=False, indent=2, default=str),
            "```",
            "",
            "### Train/validation EDA",
            "```json",
            json.dumps(self.data, ensure_ascii=False, indent=2, default=str),
            "```",
            "",
            "### Available ML environment",
            "```json",
            json.dumps(self.environment, ensure_ascii=False, indent=2, default=str),
            "```",
            "",
            "### Curated sources",
        ]
        for source in self.sources:
            parts.extend([
                "",
                f"#### `{source.path}` ({source.kind}, {source.total_chars} chars total)",
                source.note or "",
                "```",
                source.summary,
                "```",
            ])
        if self.errors:
            parts.extend([
                "",
                "### Prefetch problems",
                "These steps did not complete; handle them yourself before recording context.",
                "```json",
                json.dumps(self.errors, ensure_ascii=False, indent=2),
                "```",
            ])
        return "\n".join(parts)


# ---------------------------------------------------------------------------
# Curation
# ---------------------------------------------------------------------------

def curate_python(source: str) -> str:
    """Keep the module docstring, module constants, and the annotated API surface.

    Function bodies are the bulk of these files and the part the agent least needs
    up front: what it must know to write a compatible candidate is the contract —
    what exists, what it takes, what it returns, and which module-level constants
    encode the task's fixed rules (split dates, field lists, metric names).
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return source[:4000]

    lines: list[str] = []
    docstring = ast.get_docstring(tree)
    if docstring:
        lines.append(f'"""{docstring}"""')
        lines.append("")

    def signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
        prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
        try:
            args = ast.unparse(node.args)
        except Exception:  # pragma: no cover - defensive on exotic signatures
            args = "..."
        returns = f" -> {ast.unparse(node.returns)}" if node.returns else ""
        return f"{prefix} {node.name}({args}){returns}:"

    def emit(node: ast.AST, indent: str = "") -> None:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            lines.append(indent + signature(node))
            inner = ast.get_docstring(node)
            if inner:
                first = inner.strip().splitlines()
                lines.append(f'{indent}    """{first[0]}"""')
            lines.append(f"{indent}    ...")
            lines.append("")
        elif isinstance(node, ast.ClassDef):
            bases = ", ".join(ast.unparse(base) for base in node.bases)
            lines.append(f"{indent}class {node.name}({bases}):" if bases
                         else f"{indent}class {node.name}:")
            inner = ast.get_docstring(node)
            if inner:
                lines.append(f'{indent}    """{inner.strip().splitlines()[0]}"""')
            body = [
                child for child in node.body
                if not (isinstance(child, ast.Expr) and isinstance(child.value, ast.Constant))
            ]
            if not body:
                lines.append(f"{indent}    ...")
            for child in body:
                emit(child, indent + "    ")
            lines.append("")
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            # Module- and class-level constants carry the task's fixed rules.
            try:
                lines.append(indent + ast.unparse(node))
            except Exception:  # pragma: no cover
                pass

    for node in tree.body:
        emit(node)
    return "\n".join(lines).strip()


def curate_markdown(source: str, *, max_chars: int = 12000) -> str:
    """Keep headings, list items, table rows, and every line containing a digit.

    The starter-kit README is the task definition, and its load-bearing content is
    numeric: published scores, seed variance, the convergence epsilon, the split
    row counts, and the measured dead ends. Prose that carries no number is the
    part that can be dropped without losing a decision the agent has to make.
    """
    kept: list[str] = []
    in_code = False
    for line in source.splitlines():
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code = not in_code
            kept.append(line)
            continue
        if (
            in_code
            or _HEADING.match(line)
            or _TABLE_ROW.match(line)
            or _NUMBER.search(line)
            or stripped.startswith(("-", "*", ">"))
        ):
            kept.append(line)
    text = "\n".join(kept)
    if len(text) > max_chars:
        text = text[:max_chars] + "\n... [curation truncated; call read_file for the rest]"
    return text


def curate(path: Path, source: str) -> tuple[str, str, str]:
    """Return ``(kind, curated_text, note)`` for one source file."""
    if path.suffix == ".py":
        return (
            "python",
            curate_python(source),
            "Signatures, docstrings, and module constants; function bodies elided.",
        )
    if path.suffix in {".md", ".markdown", ".txt"}:
        return (
            "markdown",
            curate_markdown(source),
            "Headings, tables, lists, and every line containing a number.",
        )
    return ("raw", source[:6000], "First 6000 characters.")


# ---------------------------------------------------------------------------
# The prefetch itself
# ---------------------------------------------------------------------------

def prefetch_bootstrap(
    runtime,
    config,
    bootstrap_state,
    *,
    literature_queries: tuple[str, ...] = (
        "ranking loss BPR listwise softmax recommendation",
        "user behaviour sequence modelling DIN SIM click-through rate",
        "multi-task learning auxiliary objectives recommender",
    ),
    on_step: object = None,
) -> BootstrapDigest:
    """Run every deterministic bootstrap step, then build the curated digest.

    Each step goes through the same ``runtime.dispatch`` the agent would have
    called, so the bootstrap-state bookkeeping, the completeness gate, and the
    baseline tolerance check all behave identically to an agent-driven bootstrap.
    Only the caller differs — and that difference is recorded in the provenance.
    """
    digest = BootstrapDigest()

    def step(name: str, payload: dict | None = None) -> dict | None:
        raw = runtime.dispatch(name, payload or {})
        if callable(on_step):
            on_step(name, payload or {}, raw)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            digest.errors.append(f"{name}: non-JSON tool output")
            return None

    discovery = step("discover_task_docs")

    # Reading the required sources is what the completeness gate checks, so it
    # goes through read_file page by page exactly as the agent would have done.
    required_paths: list[str] = []
    if discovery:
        if discovery.get("primary_readme_path"):
            required_paths.append(str(discovery["primary_readme_path"]))
        required_paths.extend(
            str(path) for path in discovery.get("required_support_paths", [])
        )
    candidate_model = (runtime.candidate_dir / "model.py").resolve()
    if candidate_model.is_file():
        required_paths.append(str(candidate_model))

    for path_str in required_paths:
        path = Path(path_str)
        offset = 0
        chunks: list[str] = []
        while True:
            page = step("read_file", {"path": path_str, "offset": offset})
            if page is None or page.get("success") is False:
                digest.errors.append(f"read_file failed for {path_str}")
                break
            chunks.append(str(page.get("content", "")))
            if page.get("complete") or page.get("next_offset") is None:
                break
            offset = int(page["next_offset"])
        if not chunks:
            continue
        source = "".join(chunks)
        kind, curated, note = curate(path, source)
        digest.sources.append(SourceSummary(
            path=path_str, kind=kind, total_chars=len(source),
            summary=curated, note=note,
        ))

    eda = step("inspect_data")
    if eda is not None:
        digest.data = eda

    environment = step("inspect_environment")
    if environment is not None:
        digest.environment = environment

    # The literature corpus is local and deterministic, so seeding the standard
    # queries here costs nothing and satisfies the gate. The agent is free to run
    # its own queries once it has a hypothesis worth grounding.
    for query in literature_queries:
        step("search_ml_literature", {"query": query, "k": 3})

    baseline = step("reproduce_baseline")
    if baseline is not None:
        digest.baseline = baseline
        if not baseline.get("success"):
            digest.errors.append(
                f"reproduce_baseline did not match: {baseline.get('error')}"
            )

    return digest
