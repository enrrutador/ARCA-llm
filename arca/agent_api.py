from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from arca.assistant import CognitiveAssistant
from arca.conversation import ConversationIntelligence, ConversationMemory, ConversationState, ResponseQuality
from arca.native_lm import ARCALanguageModel
from arca.model import Expediente, TraceStep


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
        intent = ConversationIntelligence.classify(normalized)
        context = self.sessions.context(state)
        self.sessions.add(state, "user", normalized)

        # Route structured intents through ARCA's memory/reasoners. Only open
        # generation uses the native LM directly, avoiding prompt text being
        # misclassified as a memory command.
        if intent in {"memory_write", "memory_read", "web_research", "calculation"}:
            result = self.assistant.ask(normalized)
            text = ResponseQuality.clean(result["answer"], normalized)
            expediente = result["expediente"]
            trace = tuple(expediente.get("trace", []))
            telemetry = dict(expediente.get("telemetry", {}))
        else:
            prompt = ConversationIntelligence.build_prompt(normalized, state, context)
            generated = self.assistant.model.generate(prompt, max_tokens=160, seed=42)
            text = ResponseQuality.clean(generated, prompt)
            trace = (TraceStep("native_lm.generate", "ARCA recurrent model with bounded session context").__dict__,)
            telemetry = {"success": bool(text), "external_weights": False, "model": "ARCA-native-recurrent"}
            expediente = Expediente(normalized, "native_llm", result=text, trace=[TraceStep("native_lm.generate", "bounded session generation")], telemetry=telemetry)
            self.assistant.memory.save_episode(expediente)

        self.sessions.add(state, "assistant", text)
        self.sessions.compress(state)
        telemetry.update({"session_id": session_id, "intent": intent, "context_turns": len(state.turns), "salient_facts": len(state.salient_facts), "model": "ARCA-native-recurrent", "external_weights": False})
        telemetry.update(ResponseQuality.telemetry(text, normalized))
        success = bool(telemetry.get("success", True)) and not bool(telemetry.get("quality_warning", False))
        return AgentResponse(text, success, trace, telemetry)
