"""Phase-aware conversation memory for the persistent research agent.

The agent holds one conversation for a whole run — bootstrap plus up to 50
experiments. Left alone that list only grows, and the expensive part is not the
recent reasoning but the bootstrap: six files read in full, an EDA report, an
environment inventory, and literature passages, all of which are resent on every
later model call for the rest of the run.

Trimming that by token count alone is not safe here. The oldest messages are
exactly the ones carrying the task definition, and a generic sliding window would
drop the leakage rules and the measured dead ends while keeping the most recent
traceback. So the harness decides *what* is semantically safe to remove, and
``trim_messages`` is used only as the primitive that enforces the resulting token
budget on the one segment where recency genuinely is the right ordering.

The policy, in order:

1. The system prompt is always retained, verbatim.
2. Completed bootstrap reads are dropped once their content exists as structured
   Python state — ``BootstrapState.task_context`` and ``research_backlog`` are the
   agent's own distillation of those files, already validated by the harness.
3. Whatever bootstrap evidence is not captured by that structured state (which
   files were read, the reproduced baseline, the environment inventory, the
   literature queries) is re-emitted as one compact checkpoint message.
4. The current experiment is kept in full, including tool-call/result pairs, so
   in-flight repair work is never truncated mid-sequence.
5. Incumbent and frontier facts are re-injected from trusted harness state rather
   than left to be re-read out of old assistant text, which may be stale or wrong.

Nothing here touches the audit trail: every raw tool output still reaches the
JSONL via ``_emit_trace_event``. This governs only what the model is shown.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    convert_to_messages,
    trim_messages,
)


#: Marker on a harness-authored message so a later compaction can recognise and
#: replace its own output instead of stacking checkpoints every iteration.
_CHECKPOINT_KEY = "mle_checkpoint"

BOOTSTRAP_CHECKPOINT = "bootstrap"
STATE_BRIEFING = "state_briefing"

#: Marks the message that opens an experiment. The bootstrap/experiment boundary
#: decides what is safe to drop, so it must not depend on prose: matching the
#: literal opening line of iteration.md meant that rewording that line would
#: silently reclassify every experiment message as discardable bootstrap traffic,
#: with no crash and no warning. ``mark_experiment_start`` owns this.
EXPERIMENT_START = "experiment_start"


@dataclass
class CompactionOutcome:
    """What one compaction did — emitted as its own trace event."""

    compacted: bool
    phase: str
    messages_before: int
    messages_after: int
    tokens_before: int
    tokens_after: int
    dropped_kinds: dict[str, int] = field(default_factory=dict)
    reason: str = ""

    def as_event(self) -> dict[str, object]:
        return {
            "event_type": "memory_compaction",
            "compacted": self.compacted,
            "phase": self.phase,
            "messages_before": self.messages_before,
            "messages_after": self.messages_after,
            "tokens_before": self.tokens_before,
            "tokens_after": self.tokens_after,
            "tokens_saved": max(0, self.tokens_before - self.tokens_after),
            "dropped_kinds": dict(self.dropped_kinds),
            "reason": self.reason,
        }


def _is_checkpoint(message: BaseMessage, kind: str | None = None) -> bool:
    marker = (message.additional_kwargs or {}).get(_CHECKPOINT_KEY)
    return marker is not None and (kind is None or marker == kind)


def _checkpoint(kind: str, content: str) -> HumanMessage:
    return HumanMessage(content=content, additional_kwargs={_CHECKPOINT_KEY: kind})


def mark_experiment_start(content: str) -> HumanMessage:
    """Build the message that opens an experiment, tagged for the memory policy."""
    return HumanMessage(
        content=content, additional_kwargs={_CHECKPOINT_KEY: EXPERIMENT_START}
    )


def approx_tokens(messages: Sequence[BaseMessage]) -> int:
    """Cheap, provider-independent token estimate.

    Deliberately not a tokenizer call: this runs on every turn purely to decide
    whether compaction is worth doing, and an estimate that is consistently in
    the right ballpark is enough for a threshold. Three characters per token
    matches the conservative estimate the cost gate already uses for code and
    JSON-heavy content.
    """
    total = 0
    for message in messages:
        content = message.content
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False, default=str)
        total += len(content) // 3 + 8
        for call in getattr(message, "tool_calls", None) or []:
            total += len(json.dumps(call.get("args", {}), default=str)) // 3 + 8
    return total


def _normalise(messages: Iterable[Any]) -> list[BaseMessage]:
    """Accept the mixed dict / BaseMessage history the agent actually keeps."""
    out: list[BaseMessage] = []
    for message in messages:
        if isinstance(message, BaseMessage):
            out.append(message)
        else:
            out.extend(convert_to_messages([message]))
    return out


def _tool_call_ids(message: BaseMessage) -> set[str]:
    return {
        str(call.get("id", ""))
        for call in (getattr(message, "tool_calls", None) or [])
    }


def _split_into_turns(messages: Sequence[BaseMessage]) -> list[list[BaseMessage]]:
    """Group messages so an AIMessage stays with the ToolMessages answering it.

    Dropping an assistant message that requested tools while keeping its
    ToolMessage replies (or the reverse) produces a history the provider rejects.
    Grouping first means every later filter operates on units that are always
    valid to keep or drop as a whole.
    """
    turns: list[list[BaseMessage]] = []
    pending_ids: set[str] = set()
    for message in messages:
        if isinstance(message, ToolMessage) and message.tool_call_id in pending_ids:
            turns[-1].append(message)
            continue
        turns.append([message])
        pending_ids = _tool_call_ids(message) if isinstance(message, AIMessage) else set()
    return turns


class PhaseAwareMemory:
    """Decides what the agent is shown, given trusted harness state."""

    def __init__(
        self,
        *,
        token_budget: int,
        experiment_tail_budget: int,
        ledger_limit: int,
        keep_experiments: int = 1,
    ) -> None:
        self.token_budget = token_budget
        self.experiment_tail_budget = experiment_tail_budget
        self.ledger_limit = ledger_limit
        self.keep_experiments = keep_experiments

    # -- checkpoint construction -------------------------------------------

    @staticmethod
    def bootstrap_checkpoint(bootstrap_state) -> str:
        """The residue of bootstrap that the structured state does not already hold.

        Task context and the research backlog are re-injected separately by
        ``state_briefing`` because they change meaning as the run progresses
        (the backlog gets consumed). This covers only the fixed provenance:
        what was read, what the environment offers, what the baseline scored.
        """
        environment = bootstrap_state.environment_inventory or {}
        payload = {
            "note": (
                "Bootstrap is complete. The full text of the sources below was read "
                "during bootstrap and has been compacted out of this conversation. "
                "Call read_file on any path here to pull an exact region back."
            ),
            "fully_read_sources": sorted(bootstrap_state.fully_read_paths),
            "baseline_reproduced": bootstrap_state.baseline_reproduced,
            "reproduced_baseline_metrics": bootstrap_state.baseline_metrics,
            "literature_queries_run": bootstrap_state.literature_queries,
            "data_inspected": bootstrap_state.data_inspected,
            "available_frameworks": (
                environment.get("frameworks")
                or environment.get("installed")
                or environment
            ),
        }
        return (
            "## Bootstrap checkpoint (harness-authored from verified state)\n\n"
            "```json\n"
            + json.dumps(payload, ensure_ascii=False, indent=2, default=str)
            + "\n```"
        )

    def state_briefing(
        self,
        *,
        bootstrap_state,
        incumbent: dict[str, object],
        ledger: Sequence[dict[str, object]],
    ) -> str:
        """Re-assert the facts the agent must not misremember.

        Everything here comes from harness state that was validated when it was
        recorded, never from earlier assistant text. That is the point: after
        compaction the agent's belief about the incumbent score, the remaining
        backlog, and what has already been scored is refreshed from the source of
        truth rather than reconstructed from a conversation that may no longer
        contain it.
        """
        scored = bootstrap_state.scored_experiments or []
        payload = {
            "task_context": bootstrap_state.task_context,
            "research_backlog": bootstrap_state.research_backlog,
            "incumbent": incumbent,
            "scored_target_components": sorted(
                {
                    str(entry.get("target_component"))
                    for entry in scored
                    if entry.get("target_component")
                }
            ),
            "experiment_ledger": list(ledger)[-self.ledger_limit:],
        }
        return (
            "## Current run state (harness-authored; authoritative)\n\n"
            "These values come from validated harness state, not from earlier messages "
            "in this conversation. Where they disagree with anything above, these win.\n\n"
            "```json\n"
            + json.dumps(payload, ensure_ascii=False, indent=2, default=str)
            + "\n```"
        )

    # -- the compaction itself ---------------------------------------------

    def compact(
        self,
        messages: list[Any],
        *,
        phase: str,
        bootstrap_state,
        incumbent: dict[str, object],
        ledger: Sequence[dict[str, object]],
        force: bool = False,
        token_counter: Callable[[Sequence[BaseMessage]], int] | None = None,
    ) -> tuple[list[BaseMessage], CompactionOutcome]:
        """Return the compacted history and a record of what changed.

        ``force`` compacts regardless of budget; the agent uses it once at the
        bootstrap/experiment boundary, where the saving is largest and entirely
        safe because the structured task context has just been accepted.
        """
        count = token_counter or approx_tokens
        normalised = _normalise(messages)
        tokens_before = count(normalised)
        before = len(normalised)

        if phase == "bootstrap" and not force:
            # Bootstrap is where the required reads happen. Compacting mid-phase
            # could drop a file the completeness gate has not yet credited.
            return normalised, CompactionOutcome(
                False, phase, before, before, tokens_before, tokens_before,
                reason="bootstrap in progress; reads are still being credited",
            )
        if not bootstrap_state.complete:
            return normalised, CompactionOutcome(
                False, phase, before, before, tokens_before, tokens_before,
                reason="bootstrap incomplete; structured state cannot replace the reads",
            )
        if not force and tokens_before <= self.token_budget:
            return normalised, CompactionOutcome(
                False, phase, before, before, tokens_before, tokens_before,
                reason=f"within budget ({tokens_before} <= {self.token_budget} tokens)",
            )

        turns = _split_into_turns(normalised)

        # Rule 1: the system prompt is the first message and is never dropped.
        system_turns = [
            turn for turn in turns[:1]
            if isinstance(turn[0], (SystemMessage, HumanMessage))
        ]
        rest = turns[len(system_turns):]

        # Rules 2 + 3: everything from bootstrap becomes one checkpoint, because
        # its content now exists as validated structured state.
        boundary = self._experiment_boundary(rest)
        bootstrap_turns, experiment_turns = rest[:boundary], rest[boundary:]

        # Rule 4: keep whole recent experiments, then let trim_messages enforce
        # the byte budget on that tail without ever splitting a tool sequence.
        kept_tail = self._recent_experiments(experiment_turns)
        tail_messages = [message for turn in kept_tail for message in turn]
        tail_messages = trim_messages(
            tail_messages,
            max_tokens=self.experiment_tail_budget,
            token_counter=count,
            strategy="last",
            # Tool-call integrity: never begin the kept window on a ToolMessage
            # whose requesting AIMessage was trimmed away.
            start_on="human",
            include_system=False,
            allow_partial=False,
        )

        rebuilt: list[BaseMessage] = [message for turn in system_turns for message in turn]
        rebuilt.append(_checkpoint(
            BOOTSTRAP_CHECKPOINT, self.bootstrap_checkpoint(bootstrap_state)
        ))
        # Rule 5: trusted state last, so it is the most recent thing the model
        # reads before its own working context.
        rebuilt.append(_checkpoint(STATE_BRIEFING, self.state_briefing(
            bootstrap_state=bootstrap_state, incumbent=incumbent, ledger=ledger,
        )))
        rebuilt.extend(tail_messages)

        dropped_kinds: dict[str, int] = {}
        kept = {id(message) for message in tail_messages}
        for turn in bootstrap_turns + experiment_turns:
            for message in turn:
                if id(message) in kept or _is_checkpoint(message):
                    continue
                key = type(message).__name__
                dropped_kinds[key] = dropped_kinds.get(key, 0) + 1

        tokens_after = count(rebuilt)
        if tokens_after >= tokens_before:
            # The checkpoint and the state briefing are not free. On a prefetched
            # bootstrap the conversation is already compact — one curated digest
            # rather than dozens of raw read pages — so rebuilding it would cost
            # more than it saves. Compaction that grows the context is never
            # right, force or not.
            return normalised, CompactionOutcome(
                False, phase, before, before, tokens_before, tokens_before,
                reason=(
                    f"skipped: rebuild would grow the context "
                    f"({tokens_before} -> {tokens_after} tokens)"
                ),
            )
        return rebuilt, CompactionOutcome(
            True, phase, before, len(rebuilt), tokens_before, tokens_after,
            dropped_kinds=dropped_kinds,
            reason=(
                "bootstrap sources replaced by structured task context; "
                f"kept the last {len(kept_tail)} experiment segment(s)"
            ),
        )

    @staticmethod
    def _experiment_starts(turns: Sequence[list[BaseMessage]]) -> list[int]:
        """Indices of the turns that open an experiment, by structural marker."""
        return [
            index for index, turn in enumerate(turns)
            if _is_checkpoint(turn[0], EXPERIMENT_START)
        ]

    @classmethod
    def _experiment_boundary(cls, turns: Sequence[list[BaseMessage]]) -> int:
        """Index of the first message belonging to experiment work.

        Everything before it is bootstrap traffic whose content the structured
        state now carries. When no marker is present -- an older history, or a
        caller that did not use ``mark_experiment_start`` -- this returns 0 so
        every turn is treated as experiment work and kept subject to the token
        budget. Failing toward retention matters: the opposite default silently
        discards the entire experiment history instead of merely being untidy.
        """
        starts = cls._experiment_starts(turns)
        return starts[0] if starts else 0

    def _recent_experiments(
        self, turns: Sequence[list[BaseMessage]]
    ) -> list[list[BaseMessage]]:
        """Keep the last ``keep_experiments`` experiment segments intact."""
        starts = self._experiment_starts(turns)
        if not starts:
            return list(turns)
        return list(turns[starts[-self.keep_experiments]:])
