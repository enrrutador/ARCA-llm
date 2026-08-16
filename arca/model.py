from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True, slots=True)
class TraceStep:
    operator: str
    detail: str
    timestamp: str = field(default_factory=utc_now)
    evidence: tuple[str, ...] = ()


@dataclass(slots=True)
class Task:
    kind: str
    payload: dict[str, Any]
    objective: str = ""


@dataclass(slots=True)
class ReasonResult:
    answer: Any
    success: bool
    trace: list[TraceStep] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Expediente:
    objective: str
    task_kind: str
    id: str = field(default_factory=lambda: str(uuid4()))
    created_at: str = field(default_factory=utc_now)
    assertions: list[dict[str, Any]] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    hypotheses: list[str] = field(default_factory=list)
    evidence: list[dict[str, Any]] = field(default_factory=list)
    plan: list[str] = field(default_factory=list)
    trace: list[TraceStep] = field(default_factory=list)
    result: Any = None
    telemetry: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
