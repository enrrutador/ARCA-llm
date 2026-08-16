from __future__ import annotations

from typing import Any


def render(kind: str, value: Any, trace_count: int = 0) -> str:
    if kind == "remember":
        return f"Stored with provenance. Trace: {trace_count} step(s)."
    if kind == "cas":
        return f"Result: {value}. Trace: {trace_count} step(s)."
    if kind in {"web_search", "web_fetch"}:
        if not value:
            return "No web evidence acquired. The request failed or returned no results."
        return "\n".join(f"{row['title']}: {row['url']}\n  {row['snippet']}" for row in value)
    if kind == "recall":
        if not value:
            return "I don't have reliable local evidence for that yet."
        return "\n".join(f"{row['subject']} {row['predicate']} {row['object']} (source: {row['source']}, confidence: {row['confidence']:.2f})" for row in value)
    if kind == "clarify":
        return str(value)
    return str(value)
