from __future__ import annotations

import os
import resource
import time
from dataclasses import dataclass, field


@dataclass(slots=True)
class Budget:
    max_seconds: float = 1.5
    max_rss_mb: float = 256.0
    started: float = field(default_factory=time.perf_counter)

    def elapsed(self) -> float:
        return time.perf_counter() - self.started

    def check_time(self) -> None:
        if self.elapsed() > self.max_seconds:
            raise TimeoutError(f"cognitive budget exceeded: {self.elapsed():.3f}s")


def rss_mb() -> float:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes; Linux and most BSD tooling report KiB.
    if os.uname().sysname == "Darwin":
        return value / (1024 * 1024)
    return value / 1024
