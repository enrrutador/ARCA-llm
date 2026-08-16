from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from arca.model import utc_now


@dataclass(frozen=True, slots=True)
class Event:
    type: str
    payload: Any
    producer: str
    confidence: float = 1.0
    timestamp: str = field(default_factory=utc_now)


class Blackboard:
    def __init__(self, capacity: int = 256) -> None:
        self.capacity = capacity
        self._events: list[Event] = []

    def publish(self, event: Event) -> None:
        self._events.append(event)
        if len(self._events) > self.capacity:
            del self._events[: len(self._events) - self.capacity]

    def events(self, event_type: str | None = None) -> tuple[Event, ...]:
        items = self._events if event_type is None else [e for e in self._events if e.type == event_type]
        return tuple(items)
