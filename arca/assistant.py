from __future__ import annotations

from pathlib import Path
from typing import Any

from arca.app import build_executive
from arca.language import compile_text, render
from arca.memory import MemoryStore
from arca.model import Expediente, TraceStep


class CognitiveAssistant:
    def __init__(self, db_path: str | Path = "arca.db") -> None:
        self.memory = MemoryStore(db_path)
        self.executive = build_executive()

    def ask(self, text: str) -> dict[str, Any]:
        intent = compile_text(text)
        if intent.kind == "remember":
            self.memory.remember(intent.payload["subject"], intent.payload["predicate"], intent.payload["object"])
            record = Expediente(text, "remember")
            record.result = intent.payload
            record.trace.append(TraceStep("persist", "assertion written to versioned SQLite memory", evidence=("user",)))
            record.telemetry = {"success": True, "confidence": intent.confidence}
        elif intent.kind == "recall":
            query = intent.payload["query"]
            rows = self._all_recent() if query == "*" else self.memory.recall(query)
            record = Expediente(text, "recall")
            record.result = rows
            record.trace.append(TraceStep("retrieve", f"retrieved {len(rows)} active assertions"))
            record.telemetry = {"success": bool(rows), "confidence": intent.confidence, "ambiguity": intent.ambiguity}
        elif intent.kind == "cas":
            from arca.model import Task
            record = self.executive.execute(Task("cas", intent.payload, text))
        elif intent.kind == "help":
            record = Expediente(text, "help")
            record.result = "Commands: arithmetic, 'remember that X is Y', 'what is X?', 'memory', 'trace'."
            record.trace.append(TraceStep("help", "reported local capabilities"))
            record.telemetry = {"success": True}
        else:
            record = Expediente(text, "clarify")
            record.result = intent.ambiguity
            record.trace.append(TraceStep("clarify", "insufficiently specified request"))
            record.telemetry = {"success": False, "confidence": intent.confidence}
        self.memory.save_episode(record)
        return {"answer": render(record.task_kind, record.result, len(record.trace)), "expediente": record.to_dict()}

    def _all_recent(self) -> list[dict[str, Any]]:
        with self.memory.connect() as db:
            return [dict(row) for row in db.execute("SELECT subject,predicate,object,confidence,source,valid_from FROM assertions WHERE superseded_at IS NULL ORDER BY id DESC LIMIT 20")]
