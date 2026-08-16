from __future__ import annotations

import math

from arca.memory import MemoryStore


class PairBandit:
    """Transparent UCB1 router keyed by (task class, operator)."""

    def __init__(self, store: MemoryStore, exploration: float = 1.2) -> None:
        self.store = store
        self.exploration = exploration

    def choose(self, context: str, operators: list[str]) -> str:
        known = {row["operator"]: row for row in self.store.bandit_rows(context)}
        for operator in operators:
            if operator not in known or known[operator]["trials"] == 0:
                return operator
        total = sum(known[o]["trials"] for o in operators)
        return max(operators, key=lambda o: known[o]["reward"] + self.exploration * math.sqrt(math.log(total) / known[o]["trials"]))

    def update(self, context: str, operator: str, correct: bool, latency_ms: float, rss_mb: float) -> None:
        reward = (1.0 if correct else -1.0) - min(latency_ms / 10_000, 0.25) - min(rss_mb / 1024, 0.25)
        self.store.reward(context, operator, reward)
