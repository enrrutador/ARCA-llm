from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from arca.memory import MemoryStore
from arca.native_lm.model import ARCALanguageModel
from arca.web import WebEvidenceClient


@dataclass(slots=True)
class LearningReport:
    query: str
    documents: int
    bytes_seen: int
    losses: list[float]
    accepted_urls: list[str]
    rejected_urls: list[str]


class WebLearner:
    """Controlled online learner: web evidence -> corpus shard -> local updates.

    It never executes retrieved text, never trains on arbitrary binaries, keeps
    provenance, deduplicates by content hash, and updates only after a bounded
    fetch/train budget. This is ARCA's acquisition loop, not a cloud LLM.
    """

    def __init__(self, model: ARCALanguageModel, memory: MemoryStore, web: WebEvidenceClient | None = None, corpus_dir: str | Path = "corpus/web") -> None:
        self.model, self.memory = model, memory
        self.web = web or WebEvidenceClient(max_bytes=500_000)
        self.corpus_dir = Path(corpus_dir)
        self.corpus_dir.mkdir(parents=True, exist_ok=True)

    def learn(self, query: str, limit: int = 3, epochs: int = 1, max_bytes: int = 1_000_000) -> LearningReport:
        results = self.web.search(query, limit=limit)
        accepted, rejected, losses, seen = [], [], [], 0
        chunks: list[str] = []
        for result in results:
            text = " ".join(result.text.split())
            encoded = text.encode("utf-8", errors="replace")
            if not text or len(encoded) < 64 or seen + len(encoded) > max_bytes:
                rejected.append(result.url)
                continue
            digest = hashlib.sha256(encoded).hexdigest()
            path = self.corpus_dir / f"{digest}.txt"
            if not path.exists():
                path.write_text(text, encoding="utf-8")
            self.memory.save_document(result.title, result.url, result.snippet, text, result.source)
            chunks.append(text)
            accepted.append(result.url)
            seen += len(encoded)
        if chunks:
            losses = self.model.train_text("\n\n".join(chunks), epochs=epochs, sequence_length=96)
        return LearningReport(query, len(accepted), seen, losses, accepted, rejected)

    @staticmethod
    def report_json(report: LearningReport) -> str:
        return json.dumps({"query": report.query, "documents": report.documents, "bytes_seen": report.bytes_seen, "losses": report.losses, "accepted_urls": report.accepted_urls, "rejected_urls": report.rejected_urls}, ensure_ascii=False, indent=2)
