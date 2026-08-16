from __future__ import annotations

import time
import tracemalloc
from typing import Protocol

from arca.kernel.blackboard import Blackboard, Event
from arca.kernel.budget import Budget, rss_mb
from arca.model import Expediente, ReasonResult, Task, TraceStep


class Reasoner(Protocol):
    kind: str

    def solve(self, task: Task, budget: Budget) -> ReasonResult: ...


class Executive:
    def __init__(self, reasoners: list[Reasoner], blackboard: Blackboard | None = None) -> None:
        self.reasoners = {r.kind: r for r in reasoners}
        self.blackboard = blackboard or Blackboard()

    def execute(self, task: Task, budget: Budget | None = None) -> Expediente:
        budget = budget or Budget()
        expediente = Expediente(objective=task.objective or task.kind, task_kind=task.kind)
        expediente.plan.append(f"route to specialist:{task.kind}")
        reasoner = self.reasoners.get(task.kind)
        if reasoner is None:
            raise ValueError(f"no reasoner registered for {task.kind!r}")

        self.blackboard.publish(Event("task.accepted", task.objective, "executive"))
        before_rss = rss_mb()
        tracemalloc.start()
        started = time.perf_counter()
        try:
            budget.check_time()
            result = reasoner.solve(task, budget)
            budget.check_time()
            current, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        elapsed = time.perf_counter() - started
        after_rss = rss_mb()
        peak_rss = max(before_rss, after_rss)
        if peak_rss > budget.max_rss_mb:
            result.success = False
            result.trace.append(TraceStep("budget", f"RSS {peak_rss:.1f} MB exceeded limit"))

        expediente.result = result.answer
        expediente.trace.extend(result.trace)
        expediente.telemetry = {
            "success": result.success,
            "latency_ms": round(elapsed * 1000, 3),
            "python_current_mb": round(current / 1_048_576, 3),
            "python_peak_mb": round(peak / 1_048_576, 3),
            "rss_peak_mb": round(peak_rss, 3),
            **result.metadata,
        }
        self.blackboard.publish(Event("task.completed", expediente.telemetry, reasoner.kind))
        return expediente
