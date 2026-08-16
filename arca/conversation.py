from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class Turn:
    role: str
    text: str
    timestamp: float = field(default_factory=time.time)


@dataclass(slots=True)
class ConversationState:
    session_id: str
    turns: list[Turn] = field(default_factory=list)
    summary: str = ""
    salient_facts: list[str] = field(default_factory=list)
    user_preferences: dict[str, str] = field(default_factory=dict)


class ConversationMemory:
    """Bounded, serializable working memory for multi-turn agent use."""

    def __init__(self, path: str | Path | None = None, max_turns: int = 12) -> None:
        self.path = Path(path) if path else None
        self.max_turns = max_turns

    def load(self, session_id: str) -> ConversationState:
        if not self.path or not self.path.exists():
            return ConversationState(session_id)
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            raw = payload.get(session_id, {})
            return ConversationState(session_id, [Turn(**turn) for turn in raw.get("turns", [])], raw.get("summary", ""), raw.get("salient_facts", []), raw.get("user_preferences", {}))
        except (OSError, ValueError, TypeError):
            return ConversationState(session_id)

    def save(self, state: ConversationState) -> None:
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {}
        if self.path.exists():
            try: payload = json.loads(self.path.read_text(encoding="utf-8"))
            except (OSError, ValueError): payload = {}
        state.turns = state.turns[-self.max_turns:]
        payload[state.session_id] = asdict(state)
        self.path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def add(self, state: ConversationState, role: str, text: str) -> None:
        state.turns.append(Turn(role, text))
        state.turns = state.turns[-self.max_turns:]
        self._extract_salience(state, text)
        self.save(state)

    def _extract_salience(self, state: ConversationState, text: str) -> None:
        lower = text.casefold()
        for marker in ("remember that ", "recuerda que ", "mi nombre es ", "my name is ", "prefiero ", "i prefer "):
            if marker in lower:
                fact = text[lower.index(marker) + len(marker):].strip(" .")
                if fact and fact not in state.salient_facts:
                    state.salient_facts.append(fact[:240])
        state.salient_facts = state.salient_facts[-20:]

    def context(self, state: ConversationState) -> str:
        lines = []
        if state.summary: lines.append(f"Conversation summary: {state.summary}")
        if state.salient_facts: lines.append("Salient facts: " + "; ".join(state.salient_facts[-8:]))
        lines.extend(f"{turn.role}: {turn.text}" for turn in state.turns[-8:])
        return "\n".join(lines)

    def compress(self, state: ConversationState) -> None:
        if len(state.turns) <= 8:
            return
        content = " | ".join(f"{turn.role}: {turn.text}" for turn in state.turns[:-6])
        state.summary = (state.summary + " | " + content)[-1600:]
        state.turns = state.turns[-6:]
        self.save(state)


class ConversationIntelligence:
    """Deterministic pre/post-processing that makes a tiny model agent-usable."""

    _REFERENCE = re.compile(r"\b(eso|esa|ese|ello|él|ella|it|that|this|they|them)\b", re.I)

    @staticmethod
    def resolve_references(text: str, state: ConversationState) -> str:
        if not state.turns:
            return text
        previous = state.turns[-1].text
        candidates = re.findall(r"\b[A-ZÁÉÍÓÚÑ][\wÁÉÍÓÚÑ-]{2,}\b", previous)
        subject = candidates[-1] if candidates else previous[:80]
        return ConversationIntelligence._REFERENCE.sub(lambda match: f"{match.group(0)} ({subject})", text)

    @staticmethod
    def classify(text: str) -> str:
        lower = text.casefold().strip()
        if any(word in lower for word in ("remember", "recuerda", "memoriza")): return "memory_write"
        if any(word in lower for word in ("what", "qué", "que es", "who", "quién", "recall")): return "memory_read"
        if any(word in lower for word in ("search", "buscar", "investiga", "web")): return "web_research"
        if any(word in lower for word in ("calculate", "calcula", "cuánto", "cuanto")): return "calculation"
        if any(word in lower for word in ("plan", "steps", "pasos", "how do i", "cómo")): return "planning"
        if any(word in lower for word in ("hello", "hola", "hey")): return "social"
        return "open_generation"

    @staticmethod
    def build_prompt(message: str, state: ConversationState, context: str) -> str:
        return ("You are ARCA, a local cognitive language model.\n"
                "Use the conversation context, but do not invent facts. State uncertainty.\n"
                f"Intent: {ConversationIntelligence.classify(message)}\n"
                f"Context:\n{context[-5000:]}\nUser: {message}\nARCA:")


class ResponseQuality:
    """Cheap guardrails for empty, runaway and repetitive generations."""

    @staticmethod
    def clean(text: str, prompt: str = "", max_chars: int = 4000) -> str:
        if prompt and text.startswith(prompt): text = text[len(prompt):]
        text = text.strip()
        if not text: return "I could not produce a response from the available model state."
        sentences = re.split(r"(?<=[.!?])\s+", text)
        output: list[str] = []
        for sentence in sentences:
            if sentence and sentence not in output: output.append(sentence)
        return " ".join(output)[:max_chars].strip()

    @staticmethod
    def telemetry(text: str, prompt: str) -> dict[str, Any]:
        words = text.split()
        unique = len(set(words)) / max(len(words), 1)
        return {"output_chars": len(text), "output_words": len(words), "lexical_diversity": round(unique, 3), "quality_warning": unique < 0.25 or len(text) < 3}
