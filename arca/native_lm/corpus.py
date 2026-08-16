from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable


class CorpusStore:
    """Deduplicated, provenance-preserving UTF-8 corpus for local training."""

    def __init__(self, root: str | Path = "corpus") -> None:
        self.root = Path(root); self.root.mkdir(parents=True, exist_ok=True)
        self.manifest = self.root / "manifest.jsonl"

    def add(self, text: str, source: str, title: str = "", min_chars: int = 40) -> bool:
        text = " ".join(text.split())
        if len(text) < min_chars or not source.startswith(("http://", "https://", "file://", "user:")):
            return False
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        path = self.root / f"{digest}.txt"
        if path.exists(): return False
        path.write_text(text, encoding="utf-8")
        with self.manifest.open("a", encoding="utf-8") as output:
            output.write(json.dumps({"path": path.name, "source": source, "title": title, "sha256": digest}, ensure_ascii=False) + "\n")
        return True

    def read_all(self, max_bytes: int = 10_000_000) -> str:
        chunks, total = [], 0
        for path in sorted(self.root.glob("*.txt")):
            data = path.read_text(encoding="utf-8")
            if total + len(data.encode("utf-8")) > max_bytes: break
            chunks.append(data); total += len(data.encode("utf-8"))
        return "\n\n".join(chunks)
