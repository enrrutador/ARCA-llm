from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from arca.agent_api import ARCAAgentBackend, AgentResponse


@dataclass(slots=True)
class LoopResult:
    responses: list[AgentResponse] = field(default_factory=list)
    completed: bool = False


class ARCAAgentLoop:
    """Small framework-neutral loop suitable for embedding in an agent."""

    def __init__(self, backend: ARCAAgentBackend, max_turns: int = 8) -> None:
        self.backend, self.max_turns = backend, max_turns

    def run(self, goal: str, stepper: Callable[[str, AgentResponse | None], str], session_id: str = "agent") -> LoopResult:
        result = LoopResult()
        message = goal
        previous = None
        for _ in range(self.max_turns):
            response = self.backend.respond(message, session_id=session_id)
            result.responses.append(response)
            if not response.success:
                break
            message = stepper(message, response)
            previous = response
            if not message or message.casefold() in {"done", "finish", "terminado"}:
                result.completed = True
                break
        return result
