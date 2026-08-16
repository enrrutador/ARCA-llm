from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from arca.assistant import CognitiveAssistant
from arca.conversation import ConversationIntelligence, ConversationMemory, ConversationState, ResponseQuality
from arca.native_lm import ARCALanguageModel


@dataclass(frozen=True, slots=True)
class AgentResponse:
    text: str
    success: bool
    trace: tuple[dict[str, Any], ...]
    telemetry: dict[str, Any]


class AgentBackend(Protocol):
    def respond(self, message: str, session_id: str = "default") -> AgentResponse: ...


class ARCAAgentBackend:
    """Stable multi-turn adapter and OpenCode-compatible backend."""
    def __init__(self, model_path: str | Path, db_path: str | Path = "arca.db", max_turns: int = 12) -> None:
        self.model_path = Path(model_path)
        self.memory_path = self.model_path.with_suffix(".sessions.json")
        self.sessions = ConversationMemory(self.memory_path, max_turns=max_turns)
        self.assistant = CognitiveAssistant(db_path, model=ARCALanguageModel.load(self.model_path))

    def respond(self, message: str, session_id: str = "default") -> AgentResponse:
        state: ConversationState = self.sessions.load(session_id)
        normalized = ConversationIntelligence.resolve_references(message.strip(), state)
        context = self.sessions.context(state)
        prompt = ConversationIntelligence.build_prompt(normalized, state, context)
        self.sessions.add(state, "user", normalized)
        result = self.assistant.ask(prompt)
        text = ResponseQuality.clean(result["answer"], prompt)
        self.sessions.add(state, "assistant", text)
        self.sessions.compress(state)
        telemetry = dict(result["expediente"].get("telemetry", {}))
        telemetry.update({"session_id": session_id, "intent": ConversationIntelligence.classify(normalized), "context_turns": len(state.turns), "salient_facts": len(state.salient_facts), "model": "ARCA-native-recurrent", "external_weights": False})
        telemetry.update(ResponseQuality.telemetry(text, prompt))
        trace = tuple(result["expediente"].get("trace", []))
        return AgentResponse(text, bool(telemetry.get("success", True)) and not telemetry["quality_warning"], trace, telemetry)
