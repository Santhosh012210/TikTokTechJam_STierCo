"""Immutable-candidate research frontier with lineage-aware UCB selection."""
from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class CandidateNode:
    id: int
    code_path: str
    hypothesis: str
    primary: float
    parent_id: int | None
    depth: int
    status: str = "pending"
    target_component: str = "unclassified"
    source_sha256: str | None = None
    trial_config: dict[str, object] = field(default_factory=dict)
    seed: int = 0
    metrics: dict[str, float] = field(default_factory=dict)
    stability: dict[str, object] | None = None
    children: list[int] = field(default_factory=list)
    visits: int = 0
    reward_sum: float = 0.0
    frozen: bool = False

    def conservative_primary(self, seed_std: float = 0.0008) -> float:
        if self.primary <= -1e9:
            return -float("inf")
        if (
            self.stability
            and self.stability.get("primary_mean") is not None
            and int(self.stability.get("successful_seed_count", 2)) >= 2
        ):
            mean = float(self.stability["primary_mean"])
            std = float(self.stability.get("primary_std") or 0.0)
            return mean - std
        return self.primary - 2.0 * seed_std

    @property
    def mean_reward(self) -> float:
        return self.reward_sum / self.visits if self.visits else 0.0


class CandidateFrontier:
    """Tree of frozen candidates with top-k, noise-scaled UCB parent selection."""

    def __init__(self, root_code_path: str, root_hypothesis: str) -> None:
        self._nodes: dict[int, CandidateNode] = {
            0: CandidateNode(
                id=0,
                code_path=root_code_path,
                hypothesis=root_hypothesis,
                primary=-float("inf"),
                parent_id=None,
                depth=0,
            )
        }
        self._next_id = 1
        self._total_iters = 0

    def add_child(
        self,
        parent_id: int,
        code_path: str,
        hypothesis: str,
        *,
        target_component: str = "unclassified",
    ) -> CandidateNode:
        parent = self._nodes[parent_id]
        child = CandidateNode(
            id=self._next_id,
            code_path=code_path,
            hypothesis=hypothesis,
            primary=-float("inf"),
            parent_id=parent_id,
            depth=parent.depth + 1,
            target_component=target_component,
        )
        self._nodes[child.id] = child
        parent.children.append(child.id)
        self._next_id += 1
        return child

    def freeze_result(
        self,
        node_id: int,
        *,
        primary: float,
        status: str,
        code_path: str,
        hypothesis: str,
        target_component: str,
        source_sha256: str | None,
        trial_config: dict[str, object] | None,
        seed: int,
        metrics: dict[str, float] | None,
        stability: dict[str, object] | None,
    ) -> None:
        node = self._nodes[node_id]
        if node.frozen:
            raise ValueError(f"frontier node {node_id} is already frozen")
        parent_primary = (
            self._nodes[node.parent_id].primary
            if node.parent_id is not None and self._nodes[node.parent_id].primary > -1e9
            else primary
        )
        reward = primary - parent_primary if status == "success" else 0.0
        node.primary = primary
        node.status = status
        node.code_path = code_path
        node.hypothesis = hypothesis
        node.target_component = target_component
        node.source_sha256 = source_sha256
        node.trial_config = dict(trial_config or {})
        node.seed = int(seed)
        node.metrics = dict(metrics or {})
        node.stability = dict(stability) if stability else None
        node.frozen = True

        current: int | None = node_id
        while current is not None:
            ancestor = self._nodes[current]
            ancestor.visits += 1
            ancestor.reward_sum += reward
            current = ancestor.parent_id
        self._total_iters += 1

    def set_stability(self, node_id: int, stability: dict[str, object]) -> None:
        node = self._nodes[node_id]
        if not node.frozen or node.status != "success":
            raise ValueError("stability requires a frozen successful node")
        node.stability = dict(stability)

    def ucb_score(
        self,
        node: CandidateNode,
        *,
        exploration_c: float = 1.414,
        exploration_scale: float = 0.002,
        seed_std: float = 0.0008,
    ) -> float:
        # Conservative score represents the node itself; back-propagated mean
        # reward represents whether experiments below this branch keep improving.
        # Both are in primary-score units, so no arbitrary normalization is needed.
        exploitation = node.conservative_primary(seed_std) + node.mean_reward
        exploration = exploration_c * exploration_scale * math.sqrt(
            math.log(max(2, self._total_iters + 1)) / max(1, node.visits)
        )
        return exploitation + exploration

    def select_parent(
        self,
        *,
        top_k: int = 3,
        exploration_c: float = 1.414,
        exploration_scale: float = 0.002,
        seed_std: float = 0.0008,
    ) -> CandidateNode:
        candidates = self.leaderboard(
            top_k=max(1, top_k), conservative=True, seed_std=seed_std
        )
        if not candidates:
            raise RuntimeError("no successful candidate is available as a parent")
        return max(
            candidates,
            key=lambda node: self.ucb_score(
                node,
                exploration_c=exploration_c,
                exploration_scale=exploration_scale,
                seed_std=seed_std,
            ),
        )

    def leaderboard(
        self,
        *,
        top_k: int = 10,
        conservative: bool = False,
        seed_std: float = 0.0008,
    ) -> list[CandidateNode]:
        candidates = [
            node for node in self._nodes.values()
            if node.status == "success" and node.primary > -1e9
        ]
        key = (
            (lambda node: node.conservative_primary(seed_std))
            if conservative else (lambda node: node.primary)
        )
        return sorted(candidates, key=key, reverse=True)[:top_k]

    def best_node(self, *, seed_std: float = 0.0008) -> CandidateNode | None:
        candidates = self.leaderboard(
            top_k=1, conservative=True, seed_std=seed_std
        )
        return candidates[0] if candidates else None

    def get_node(self, node_id: int) -> CandidateNode:
        return self._nodes[node_id]

    def lineage(self, node_id: int) -> list[CandidateNode]:
        lineage: list[CandidateNode] = []
        current: int | None = node_id
        while current is not None:
            node = self._nodes[current]
            lineage.append(node)
            current = node.parent_id
        return list(reversed(lineage))

    def frontier_signature(self, *, top_k: int = 3) -> tuple[tuple[int, float], ...]:
        """Stable convergence signature of the strongest conservative frontier."""
        return tuple(
            (node.id, round(node.conservative_primary(), 8))
            for node in self.leaderboard(top_k=top_k, conservative=True)
        )

    def to_dict(self) -> dict[str, object]:
        best = self.best_node()
        return {
            "total_iters": self._total_iters,
            "conservative_best_node_id": best.id if best else None,
            "nodes": {
                str(node_id): {
                    "id": node.id,
                    "code_path": node.code_path,
                    "hypothesis": node.hypothesis,
                    "primary": node.primary,
                    "parent_id": node.parent_id,
                    "depth": node.depth,
                    "status": node.status,
                    "target_component": node.target_component,
                    "source_sha256": node.source_sha256,
                    "trial_config": node.trial_config,
                    "seed": node.seed,
                    "metrics": node.metrics,
                    "stability": node.stability,
                    "children": node.children,
                    "visits": node.visits,
                    "reward_sum": node.reward_sum,
                    "mean_reward": node.mean_reward,
                    "conservative_primary": (
                        node.conservative_primary() if node.status == "success" else None
                    ),
                    "frozen": node.frozen,
                }
                for node_id, node in self._nodes.items()
            },
        }

    def __len__(self) -> int:
        return len(self._nodes)


__all__ = ["CandidateFrontier", "CandidateNode"]
