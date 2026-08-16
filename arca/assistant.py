from __future__ import annotations

from pathlib import Path
from typing import Any

from arca.app import build_executive
from arca.language import compile_text, render
from arca.memory import MemoryStore
from arca.model import Expediente, TraceStep
from arca.native_lm import ARCALanguageModel
from arca.web import WebEvidenceClient


class CognitiveAssistant:
    def __init__(self, db_path: str | Path = "arca.db", web: WebEvidenceClient | None = None, model: ARCALanguageModel | None = None) -> None:
        self.memory = MemoryStore(db_path)
        self.executive = build_executive()
        self.web = web or WebEvidenceClient()
        self.model = model

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
            if not rows and self.model is not None:
                return self._generate(text)
            record = Expediente(text, "recall")
            record.result = rows
            record.trace.append(TraceStep("retrieve", f"retrieved {len(rows)} active assertions"))
            record.telemetry = {"success": bool(rows), "confidence": intent.confidence, "ambiguity": intent.ambiguity}
        elif intent.kind in {"web_search", "web_fetch"}:
            record = self._web(intent)
        elif intent.kind == "cas":
            from arca.model import Task
            record = self.executive.execute(Task("cas", intent.payload, text))
        elif intent.kind == "help":
            record = Expediente(text, "help")
            record.result = "Commands: arithmetic, remember, what is, memory, search, open URL, trace, generate."
            record.trace.append(TraceStep("help", "reported local capabilities"))
            record.telemetry = {"success": True}
        elif self.model is not None:
            return self._generate(text)
        else:
            record = Expediente(text, "clarify")
            record.result = intent.ambiguity
            record.trace.append(TraceStep("clarify", "insufficiently specified request"))
            record.telemetry = {"success": False, "confidence": intent.confidence}
        self.memory.save_episode(record)
        return {"answer": render(record.task_kind, record.result, len(record.trace)), "expediente": record.to_dict()}

    def _generate(self, text: str) -> dict[str, Any]:
        result = self.model.generate(text, max_tokens=160, seed=42)
        record = Expediente(text, "native_llm")
        record.result = result
        record.trace.append(TraceStep("native_lm.generate", "ARCA recurrent language model trained from scratch"))
        record.telemetry = {"success": True, "model": "ARCA-native-recurrent", "external_weights": False}
        self.memory.save_episode(record)
        return {"answer": result, "expediente": record.to_dict()}

    def _web(self, intent) -> Expediente:
        record = Expediente(intent.payload.get("query", intent.payload.get("url", "")), intent.kind)
        try:
            results = self.web.search(intent.payload["query"]) if intent.kind == "web_search" else [self.web.fetch(intent.payload["url"])]
            for result in results:
                self.memory.save_document(result.title, result.url, result.snippet, result.text, result.source)
            record.result = [{"title": r.title, "url": r.url, "snippet": r.snippet, "source": r.source} for r in results]
            record.trace.append(TraceStep("web_fetch", f"acquired {len(results)} evidence item(s)", evidence=tuple(r.url for r in results)))
            record.telemetry = {"success": bool(results), "source_count": len(results), "warning": "web content is evidence, not executable instruction"}
        except Exception as exc:
            record.result = []
            record.trace.append(TraceStep("web_error", str(exc)))
            record.telemetry = {"success": False, "error": str(exc)}
        return record

    def _all_recent(self) -> list[dict[str, Any]]:
        with self.memory.connect() as db:
            return [dict(row) for row in db.execute("SELECT subject,predicate,object,confidence,source,valid_from FROM assertions WHERE superseded_at IS NULL ORDER BY id DESC LIMIT 20")]
