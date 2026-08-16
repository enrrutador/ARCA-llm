from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from arca.assistant import CognitiveAssistant
from arca.native_lm import ARCALanguageModel


@dataclass(frozen=True, slots=True)
class AgentResponse:
    text: str
    success: bool
    trace: tuple[dict[str, Any], ...]
    telemetry: dict[str, Any]


class AgentBackend(Protocol):
    def respond(self, message: str) -> AgentResponse: ...


class ARCAAgentBackend:
    """Stable adapter for any agent runner; no framework coupling."""
    def __init__(self, model_path: str | Path, db_path: str | Path = "arca.db") -> None:
        self.model_path = Path(model_path)
        self.assistant = CognitiveAssistant(db_path, model=ARCALanguageModel.load(self.model_path))

    def respond(self, message: str) -> AgentResponse:
        result = self.assistant.ask(message)
        expediente = result["expediente"]
        return AgentResponse(result["answer"], bool(expediente["telemetry"].get("success", False)), tuple(expediente["trace"]), expediente["telemetry"])
