from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from arca.model import Expediente, utc_now


class MemoryStore:
    """Portable SQLite memory for assertions, evidence documents and episodes."""
    def __init__(self, path: str | Path = "arca.db") -> None:
        self.path = str(path); self._initialize()

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        db = sqlite3.connect(self.path); db.row_factory = sqlite3.Row
        try:
            yield db; db.commit()
        finally: db.close()

    def _initialize(self) -> None:
        with self.connect() as db:
            db.executescript("""
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS assertions(id INTEGER PRIMARY KEY, subject TEXT NOT NULL, predicate TEXT NOT NULL, object TEXT NOT NULL, confidence REAL NOT NULL DEFAULT 1.0, source TEXT NOT NULL, valid_from TEXT NOT NULL, superseded_at TEXT);
            CREATE INDEX IF NOT EXISTS idx_assertion_sp ON assertions(subject,predicate,superseded_at);
            CREATE TABLE IF NOT EXISTS documents(id INTEGER PRIMARY KEY, title TEXT NOT NULL, url TEXT NOT NULL UNIQUE, snippet TEXT NOT NULL, content TEXT NOT NULL, source TEXT NOT NULL, fetched_at TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS episodes(id TEXT PRIMARY KEY, created_at TEXT NOT NULL, objective TEXT NOT NULL, task_kind TEXT NOT NULL, payload TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS bandit(context TEXT NOT NULL, operator TEXT NOT NULL, trials INTEGER NOT NULL DEFAULT 0, reward REAL NOT NULL DEFAULT 0, PRIMARY KEY(context,operator));
            """)

    def remember(self, subject: str, predicate: str, object_: str, source: str = "user", confidence: float = 1.0) -> int:
        values = (subject.strip().casefold(), predicate.strip().casefold(), object_.strip(), max(0.0, min(1.0, confidence)), source, utc_now())
        if not all(values[:3]): raise ValueError("subject, predicate and object are required")
        with self.connect() as db:
            return int(db.execute("INSERT INTO assertions(subject,predicate,object,confidence,source,valid_from) VALUES(?,?,?,?,?,?)", values).lastrowid)

    def recall(self, query: str, limit: int = 8) -> list[dict[str, Any]]:
        tokens = [t.casefold() for t in query.split() if len(t) > 1][:8]
        if not tokens: return []
        clauses, params = [], []
        for token in tokens:
            clauses.append("(subject LIKE ? OR predicate LIKE ? OR object LIKE ?)"); value = f"%{token}%"; params += [value, value, value]
        params.append(limit)
        with self.connect() as db:
            return [dict(row) for row in db.execute(f"SELECT subject,predicate,object,confidence,source,valid_from FROM assertions WHERE superseded_at IS NULL AND ({' OR '.join(clauses)}) ORDER BY confidence DESC,id DESC LIMIT ?", params)]

    def save_document(self, title: str, url: str, snippet: str, content: str, source: str = "web") -> None:
        with self.connect() as db:
            db.execute("""INSERT INTO documents(title,url,snippet,content,source,fetched_at) VALUES(?,?,?,?,?,?)
            ON CONFLICT(url) DO UPDATE SET title=excluded.title,snippet=excluded.snippet,content=excluded.content,source=excluded.source,fetched_at=excluded.fetched_at""", (title, url, snippet, content, source, utc_now()))

    def search_documents(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        tokens = [t.casefold() for t in query.split() if len(t) > 1][:8]
        if not tokens: return []
        clauses, params = [], []
        for token in tokens:
            clauses.append("(title LIKE ? OR snippet LIKE ? OR content LIKE ?)"); value = f"%{token}%"; params += [value, value, value]
        params.append(limit)
        with self.connect() as db:
            return [dict(row) for row in db.execute(f"SELECT title,url,snippet,source,fetched_at FROM documents WHERE {' OR '.join(clauses)} ORDER BY id DESC LIMIT ?", params)]

    def save_episode(self, expediente: Expediente) -> None:
        with self.connect() as db: db.execute("INSERT OR REPLACE INTO episodes VALUES(?,?,?,?,?)", (expediente.id, expediente.created_at, expediente.objective, expediente.task_kind, json.dumps(expediente.to_dict(), ensure_ascii=False)))
    def bandit_rows(self, context: str) -> list[dict[str, Any]]:
        with self.connect() as db: return [dict(row) for row in db.execute("SELECT * FROM bandit WHERE context=?", (context,))]
    def reward(self, context: str, operator: str, value: float) -> None:
        with self.connect() as db: db.execute("""INSERT INTO bandit(context,operator,trials,reward) VALUES(?,?,1,?) ON CONFLICT(context,operator) DO UPDATE SET reward=(bandit.reward*bandit.trials+excluded.reward)/(bandit.trials+1),trials=bandit.trials+1""", (context, operator, value))
