from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class Intent:
    kind: str
    payload: dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    ambiguity: str | None = None


_MATH = re.compile(r"^[\d\s+\-*/().%^]+$")
_REMEMBER = re.compile(r"^(?:remember|recuerda|memoriza)\s+(?:that\s+|que\s+)?(.+?)\s+(?:is|es|son|means|significa)\s+(.+)$", re.I)
_WHAT = re.compile(r"^(?:what (?:is|are)|qué es|que es|quién es|quien es)\s+(.+?)[?¿]*$", re.I)


def compile_text(text: str) -> Intent:
    clean = " ".join(text.strip().split())
    if not clean:
        return Intent("clarify", ambiguity="The request is empty.", confidence=0.0)
    match = _REMEMBER.match(clean)
    if match:
        return Intent("remember", {"subject": match.group(1), "predicate": "is", "object": match.group(2)})
    lowered = clean.casefold()
    if lowered in {"help", "ayuda", "?"}:
        return Intent("help")
    if lowered in {"memory", "memoria", "qué recuerdas", "que recuerdas"}:
        return Intent("recall", {"query": "*"})
    expression = clean.replace("^", "**")
    if _MATH.fullmatch(clean) and any(op in clean for op in "+-*/%^()"):
        return Intent("cas", {"expression": expression})
    match = _WHAT.match(clean)
    if match:
        return Intent("recall", {"query": match.group(1)})
    return Intent("recall", {"query": clean}, confidence=0.55, ambiguity="No exact operator matched; searching memory.")
