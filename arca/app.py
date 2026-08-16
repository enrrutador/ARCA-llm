from __future__ import annotations

from arca.kernel.executive import Executive
from arca.reasoners import AStarReasoner, CASReasoner, DatalogReasoner


def build_executive() -> Executive:
    return Executive([DatalogReasoner(), AStarReasoner(), CASReasoner()])
