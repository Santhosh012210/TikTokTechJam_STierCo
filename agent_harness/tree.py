"""UCB search tree. Pure data structure — no I/O, no API calls."""
import math
from dataclasses import dataclass, field


@dataclass
class Node:
    id:          int
    code_path:   str
    hypothesis:  str
    primary:     float          # -inf until first run
    parent_id:   int | None
    visits:      int
    depth:       int
    children:    list[int] = field(default_factory=list)
    status:      str = "pending"   # pending | success | failed | rejected


class SearchTree:
    def __init__(self, root_code_path: str, root_hypothesis: str) -> None:
        self._nodes: dict[int, Node] = {}
        self._next_id: int = 0
        self._total_iters: int = 0
        self._best_primary: float = -float("inf")
        self._best_node_id: int | None = None

        root = Node(
            id=0,
            code_path=root_code_path,
            hypothesis=root_hypothesis,
            primary=-float("inf"),
            parent_id=None,
            visits=0,
            depth=0,
        )
        self._nodes[0] = root
        self._next_id = 1

    # ------------------------------------------------------------------
    # UCB
    # ------------------------------------------------------------------

    def ucb_score(self, node: Node, C: float = 1.414) -> float:
        # Completely unvisited nodes get inf so they are always explored before
        # any revisit — this enforces the "favour breadth" requirement.
        if node.visits == 0:
            return float("inf")
        if self._total_iters == 0:
            return node.primary if node.primary > -1e9 else 0.0
        exploration = C * math.sqrt(math.log(self._total_iters + 1) / node.visits)
        base = node.primary if node.primary > -1e9 else 0.0
        return base + exploration

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------

    def select(self, C: float = 1.414) -> Node:
        """Return the node with the highest UCB score (excludes rejected nodes)."""
        candidates = [n for n in self._nodes.values() if n.status != "rejected"]
        if not candidates:
            raise RuntimeError("All nodes are rejected — cannot select")
        return max(candidates, key=lambda n: self.ucb_score(n, C))

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add_child(self, parent_id: int, code_path: str, hypothesis: str) -> Node:
        parent = self._nodes[parent_id]
        child = Node(
            id=self._next_id,
            code_path=code_path,
            hypothesis=hypothesis,
            primary=-float("inf"),
            parent_id=parent_id,
            visits=0,
            depth=parent.depth + 1,
        )
        self._nodes[self._next_id] = child
        parent.children.append(self._next_id)
        self._next_id += 1
        return child

    def update(self, node_id: int, primary: float, status: str) -> None:
        node = self._nodes[node_id]
        node.primary = primary
        node.visits += 1
        node.status = status
        self._total_iters += 1
        if primary > self._best_primary:
            self._best_primary = primary
            self._best_node_id = node_id

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def leaderboard(self, top_k: int = 10) -> list[Node]:
        """Top-k successful nodes sorted by primary descending."""
        successful = [
            n for n in self._nodes.values()
            if n.status == "success" and n.primary > -1e9
        ]
        return sorted(successful, key=lambda n: n.primary, reverse=True)[:top_k]

    def get_node(self, node_id: int) -> Node:
        return self._nodes[node_id]

    def lineage(self, node_id: int) -> list[Node]:
        """Return path from root to this node (inclusive)."""
        path: list[Node] = []
        current: int | None = node_id
        while current is not None:
            n = self._nodes[current]
            path.append(n)
            current = n.parent_id
        return list(reversed(path))

    def to_dict(self) -> dict:
        """Serialize for checkpointing."""
        return {
            "total_iters": self._total_iters,
            "best_primary": self._best_primary,
            "best_node_id": self._best_node_id,
            "nodes": {
                str(nid): {
                    "id": n.id, "code_path": n.code_path, "hypothesis": n.hypothesis,
                    "primary": n.primary, "parent_id": n.parent_id,
                    "visits": n.visits, "depth": n.depth,
                    "children": n.children, "status": n.status,
                }
                for nid, n in self._nodes.items()
            },
        }

    def __len__(self) -> int:
        return len(self._nodes)
