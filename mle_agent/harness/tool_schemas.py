"""Pydantic argument models for every agent tool — the single schema source.

The tool surface used to be hand-written JSON Schema dicts in ``agent_tools.py``
that were then re-encoded into OpenAI function format inside the provider. The
schema therefore existed twice, in two shapes, with nothing checking they agreed.

Here each tool's arguments are one Pydantic model, and ``AGENT_TOOLS`` is derived
from it by ``model_json_schema()``. The models are also the natural place to put
the validation that ``AgentToolRuntime.dispatch`` would otherwise repeat by hand,
so a malformed tool call can be rejected with a precise field-level message
instead of a KeyError.

``TARGET_COMPONENTS`` is defined once and shared by ``run_model`` and the research
backlog; those two enums previously drifted apart by construction.
"""
from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field


# The pipeline stages an experiment may target. Shared by run_model and the
# research backlog so the diversity gate compares like with like.
#: A hypothesis can lose a fair test, or never get one. Collapsing those two into
#: a single boolean is how a one-line indexing bug came to be recorded as evidence
#: that BPR does not work for within-user ranking.
HypothesisStatus = Literal["supported", "not_supported", "not_tested"]

TargetComponent = Literal[
    "loss",
    "sampling",
    "features",
    "sequence",
    "auxiliary-task",
    "model",
    "training",
    "evaluation",
]

TARGET_COMPONENTS: tuple[str, ...] = tuple(TargetComponent.__args__)


class ToolArgs(BaseModel):
    """Base for every tool argument model.

    ``extra="forbid"`` emits ``additionalProperties: false``, which the OpenAI
    strict-tools path requires and which turns a hallucinated argument into a
    validation error the agent can see and correct.
    """

    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------------------
# Bootstrap / inspection
# ---------------------------------------------------------------------------

class DiscoverTaskDocsArgs(ToolArgs):
    """No arguments — discovery is fully determined by the starter-kit root."""


class ReadFileArgs(ToolArgs):
    path: str = Field(description="Candidate-relative, starter-kit-relative, or absolute file path.")
    offset: int = Field(
        default=0, ge=0, description="Character offset to start at (default 0)."
    )
    max_chars: int | None = Field(
        default=None,
        ge=1,
        description="Requested page size, capped by the harness read limit.",
    )


class InspectDataArgs(ToolArgs):
    """No arguments — the EDA view is fixed to the train/validation split."""


class QueryDataFilter(BaseModel):
    model_config = ConfigDict(extra="forbid")

    column: str
    op: Literal["==", "!=", "<", "<=", ">", ">="] = "=="
    value: str


class QueryDataArgs(ToolArgs):
    split: Literal["train", "validation"]
    metrics: Annotated[list[str], Field(min_length=1, max_length=5)]
    group_by: Annotated[list[str], Field(max_length=2)] = []
    filters: Annotated[list[QueryDataFilter], Field(max_length=3)] = []
    limit: int = Field(default=20, ge=1, le=20)


class InspectEnvironmentArgs(ToolArgs):
    """No arguments — the inventory is taken from the run's own interpreter."""


class RequestDependencyInstallArgs(ToolArgs):
    packages: Annotated[list[str], Field(min_length=1, max_length=6)] = Field(
        description="One to six PyPI requirement specifiers, such as lightgbm or torch==2.8.0."
    )
    justification: str = Field(
        min_length=1,
        description="Why these packages are necessary for the current research hypothesis.",
    )


class ReproduceBaselineArgs(ToolArgs):
    """No arguments — the inherited candidate is executed unchanged."""


# ---------------------------------------------------------------------------
# Candidate editing
# ---------------------------------------------------------------------------

class WriteFileArgs(ToolArgs):
    path: str
    content: str


class EditFileArgs(ToolArgs):
    path: str
    old_text: str = Field(min_length=1)
    new_text: str


# ---------------------------------------------------------------------------
# Structured task understanding
# ---------------------------------------------------------------------------

class DataSplits(BaseModel):
    model_config = ConfigDict(extra="forbid")

    train: str
    validation: str
    test: str


class FeatureEngineeringContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    baseline_fields: Annotated[list[str], Field(min_length=1)]
    measured_dead_ends: Annotated[list[str], Field(min_length=1)]
    promising_feature_families: Annotated[list[str], Field(min_length=1)]
    leakage_controls: Annotated[list[str], Field(min_length=1)]
    implementation_boundary: str = Field(min_length=1)


class RecordTaskContextArgs(ToolArgs):
    task_objective: str = Field(min_length=1)
    target_label: str = Field(min_length=1)
    metrics: Annotated[list[str], Field(min_length=1)]
    data_splits: DataSplits
    baseline: str = Field(min_length=1)
    evaluation_protocol: Annotated[list[str], Field(min_length=1)]
    hard_constraints: Annotated[list[str], Field(min_length=1)]
    known_dead_ends: Annotated[list[str], Field(min_length=1)]
    promising_directions: Annotated[list[str], Field(min_length=1)]
    feature_engineering_context: FeatureEngineeringContext
    candidate_contract: Annotated[list[str], Field(min_length=1)]
    source_paths: Annotated[list[str], Field(min_length=1)]


class ResearchCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hypothesis: str = Field(min_length=1)
    target_component: TargetComponent
    evidence_id: str = Field(
        min_length=1,
        description="A search_ml_literature chunk id, an EDA finding id, or a prior-run: id.",
    )
    expected_primary_delta: float
    estimated_cost: str = Field(min_length=1)
    falsification_criterion: str = Field(min_length=1)


class RecordResearchBacklogArgs(ToolArgs):
    candidates: Annotated[list[ResearchCandidate], Field(min_length=6, max_length=10)]


# ---------------------------------------------------------------------------
# Experiment execution
# ---------------------------------------------------------------------------

class RunModelArgs(ToolArgs):
    hypothesis: str = Field(
        min_length=1, description="The precise model or pipeline change being tested."
    )
    reasoning: str = Field(
        min_length=1,
        description=(
            "Why the change may improve the ranking metrics, grounded in "
            "data/history/literature."
        ),
    )
    target_component: TargetComponent = Field(
        description="Primary pipeline component changed by this experiment."
    )
    literature_chunk_ids: list[str] = Field(
        default=[], description="Relevant chunk IDs returned by search_ml_literature."
    )
    expected_effect: str = Field(
        default="", description="Expected effect on the official metrics and why."
    )
    falsification_criterion: str = Field(
        default="", description="Metric result that would reject the hypothesis."
    )
    rollback_plan: str = Field(
        default="retain incumbent",
        description="How to return to the incumbent if the experiment fails.",
    )
    seed: int | None = Field(
        default=None, description="Reproducible training seed; defaults to the official seed."
    )
    trial_config: dict[str, Any] = Field(
        default={}, description="Candidate-declared data-only hyperparameter configuration."
    )
    execution_class: Literal["quick", "normal", "substantial"] = Field(
        default="normal",
        description=(
            "Propose an execution budget: 'quick' for a diagnostic, 'normal' for an "
            "ordinary candidate, 'substantial' for a framework-backed run. The harness "
            "decides the actual limit and may clamp it to the wall budget."
        ),
    )
    feature_sources: list[str] = Field(
        default=[],
        description="Candidate-data columns or train-derived histories used by a feature experiment.",
    )
    feature_transformations: list[str] = Field(
        default=[],
        description="Exact joins, buckets, crosses, aggregates, or sequence transformations tested.",
    )
    leakage_controls: list[str] = Field(
        default=[],
        description="How every fitted statistic avoids validation labels and future information.",
    )
    diversity_override: str = Field(
        default="",
        description=(
            "Written, evidence-based justification for repeating a target_component "
            "before three distinct components have been scored."
        ),
    )


class RunSweepArgs(RunModelArgs):
    """One bounded hyperparameter sweep: the same candidate, 2-6 configs.

    Scored as one experiment for the convergence rule but as one variant per
    config against the run's scored-variant budget, so a sweep cannot buy extra
    convergence turns.
    """

    trial_configs: Annotated[list[dict[str, Any]], Field(min_length=2, max_length=6)] = Field(
        description=(
            "Two to six data-only hyperparameter configurations of the same "
            "candidate. Each is executed and scored separately."
        )
    )


# ---------------------------------------------------------------------------
# Reflection (structured output, not a tool)
# ---------------------------------------------------------------------------

class ReflectionResult(BaseModel):
    """Closing reflection for one experiment.

    Used with ``with_structured_output(..., method="json_schema")`` so the
    provider enforces the shape instead of the harness parsing free text.
    """

    model_config = ConfigDict(extra="forbid")

    reflection: str = Field(
        min_length=1,
        description="What the metrics show about the hypothesis, in two or three sentences.",
    )
    hypothesis_status: HypothesisStatus = Field(
        description=(
            "'supported' if the metrics back the hypothesis; 'not_supported' if the "
            "experiment was a fair test and the hypothesis lost; 'not_tested' if the "
            "implementation failed, so the result says nothing about the hypothesis."
        )
    )
    implementation_diagnosis: str = Field(
        default="",
        description=(
            "Required when hypothesis_status is 'not_tested': what specifically was "
            "wrong with the implementation, and what a corrected attempt would change."
        ),
    )

    @property
    def hypothesis_supported(self) -> bool | None:
        """Legacy boolean view. None where the hypothesis was never actually tested.

        A two-valued field forced a broken build to be recorded as evidence
        against the idea. 'not_tested' has to stay distinguishable from
        'not_supported' all the way into the run log and the cross-run history,
        so this deliberately returns None rather than collapsing to False.
        """
        if self.hypothesis_status == "not_tested":
            return None
        return self.hypothesis_status == "supported"
    suggested_next: str = Field(
        min_length=1,
        description="The single most informative next experiment given this result.",
    )


# ---------------------------------------------------------------------------
# JSON Schema derivation
# ---------------------------------------------------------------------------

def _inline_defs(schema: dict[str, Any]) -> dict[str, Any]:
    """Resolve ``$ref``/``$defs`` into a self-contained schema.

    Nested models make Pydantic emit ``$defs`` plus ``$ref`` pointers. Several
    provider tool endpoints reject or silently mishandle those, so the refs are
    expanded once here rather than being trusted to survive the round trip.
    """
    defs = schema.pop("$defs", {})

    def resolve(node: Any, seen: frozenset[str]) -> Any:
        if isinstance(node, list):
            return [resolve(item, seen) for item in node]
        if not isinstance(node, dict):
            return node
        ref = node.get("$ref")
        if isinstance(ref, str) and ref.startswith("#/$defs/"):
            name = ref.split("/")[-1]
            if name in seen:
                # A self-referential model cannot be inlined; leave a permissive
                # object rather than recursing forever.
                return {"type": "object"}
            target = defs.get(name, {})
            merged = dict(resolve(target, seen | {name}))
            # Sibling keys of a $ref (title, description, default) win.
            merged.update({
                key: resolve(value, seen)
                for key, value in node.items()
                if key != "$ref"
            })
            return merged
        return {key: resolve(value, seen) for key, value in node.items()}

    return resolve(schema, frozenset())


def _strip_noise(node: Any) -> Any:
    """Drop keys that add tokens to every request without changing validation.

    Pydantic emits a ``title`` for every model and every field, which is pure
    restatement of the field name. The tool block is resent on every model call
    in a persistent conversation, so this is not a cosmetic saving.
    """
    if isinstance(node, list):
        return [_strip_noise(item) for item in node]
    if not isinstance(node, dict):
        return node
    return {
        key: _strip_noise(value)
        for key, value in node.items()
        if key != "title"
    }


def tool_parameters(model: type[BaseModel]) -> dict[str, Any]:
    """Return the OpenAI-compatible ``parameters`` schema for an argument model."""
    schema = _strip_noise(_inline_defs(model.model_json_schema()))
    schema.setdefault("type", "object")
    schema.setdefault("properties", {})
    schema.setdefault("additionalProperties", False)
    return schema


def tool_spec(
    name: str, model: type[BaseModel], description: str
) -> dict[str, Any]:
    """Build one canonical tool entry from its Pydantic argument model."""
    return {
        "name": name,
        "description": description,
        "parameters": tool_parameters(model),
        # Retained so dispatch can validate the payload against the same model
        # that produced the schema the model was shown.
        "args_model": model,
    }


#: Every tool the research agent may call, keyed by name.
TOOL_ARGS_MODELS: dict[str, type[BaseModel]] = {
    "discover_task_docs": DiscoverTaskDocsArgs,
    "read_file": ReadFileArgs,
    "write_file": WriteFileArgs,
    "edit_file": EditFileArgs,
    "inspect_data": InspectDataArgs,
    "query_data": QueryDataArgs,
    "inspect_environment": InspectEnvironmentArgs,
    "request_dependency_install": RequestDependencyInstallArgs,
    "reproduce_baseline": ReproduceBaselineArgs,
    "record_task_context": RecordTaskContextArgs,
    "record_research_backlog": RecordResearchBacklogArgs,
    "run_model": RunModelArgs,
}

#: Registered only when AGENT_ENABLE_SWEEPS is set, so the default tool surface
#: stays as narrow as the run log claims it is.
OPTIONAL_TOOL_ARGS_MODELS: dict[str, type[BaseModel]] = {
    "run_sweep": RunSweepArgs,
}
