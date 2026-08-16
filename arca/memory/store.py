from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from arca.model import Expediente, utc_now


class MemoryStore:
    """Versioned SQLite memory with provenance and full-text-free portable search."""

    def __init__(self, path: str | Path = "arca.db") -> None:
        self.path = str(path)
        self._initialize()

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        db = sqlite3.connect(self.path)
        db.row_factory = sqlite3.Row
        try:
            yield db
            db.commit()
        finally:
            db.close()

    def _initialize(self) -> None:
        with self.connect() as db:
            db.executescript("""
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS assertions (
                    id INTEGER PRIMARY KEY,
                    subject TEXT NOT NULL,
                    predicate TEXT NOT NULL,
                    object TEXT NOT NULL,
                    confidence REAL NOT NULL DEFAULT 1.0,
                    source TEXT NOT NULL,
                    valid_from TEXT NOT NULL,
                    superseded_at TEXT,
                    UNIQUE(subject, predicate, object, source, valid_from)
                );
                CREATE INDEX IF NOT EXISTS idx_assertion_sp
                    ON assertions(subject, predicate, superseded_at);
                CREATE TABLE IF NOT EXISTS episodes (
                    id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    objective TEXT NOT NULL,
                    task_kind TEXT NOT NULL,
                    payload TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS bandit (
                    context TEXT NOT NULL,
                    operator TEXT NOT NULL,
                    trials INTEGER NOT NULL DEFAULT 0,
                    reward REAL NOT NULL DEFAULT 0,
                    PRIMARY KEY(context, operator)
                );
            """)

    def remember(self, subject: str, predicate: str, object_: str, source: str = "user", confidence: float = 1.0) -> int:
        subject, predicate, object_ = subject.strip(), predicate.strip(), object_.strip()
        if not all((subject, predicate, object_)):
            raise ValueError("subject, predicate and object are required")
        with self.connect() as db:
            cursor = db.execute(
                "INSERT INTO assertions(subject,predicate,object,confidence,source,valid_from) VALUES(?,?,?,?,?,?)",
                (subject.casefold(), predicate.casefold(), object_, max(0.0, min(1.0, confidence)), source, utc_now()),
            )
            return int(cursor.lastrowid)

    def recall(self, query: str, limit: int = 8) -> list[dict[str, Any]]:
        tokens = [t.casefold() for t in query.split() if len(t) > 1][:8]
        if not tokens:
            return []
        clauses = []
        params: list[Any] = []
        for token in tokens:
            clauses.append("(subject LIKE ? OR predicate LIKE ? OR object LIKE ?)")
            value = f"%{token}%"
            params.extend([value, value, value])
        params.append(limit)
        sql = f"""SELECT subject,predicate,object,confidence,source,valid_from
                  FROM assertions WHERE superseded_at IS NULL AND ({' OR '.join(clauses)})
                  ORDER BY confidence DESC, id DESC LIMIT ?"""
        with self.connect() as db:
            return [dict(row) for row in db.execute(sql, params)]

    def facts(self, subject: str, predicate: str | None = None) -> list[dict[str, Any]]:
        sql = "SELECT * FROM assertions WHERE subject=? AND superseded_at IS NULL"
        params: list[Any] = [subject.casefold()]
        if predicate:
            sql += " AND predicate=?"
            params.append(predicate.casefold())
        sql += " ORDER BY confidence DESC, id DESC"
        with self.connect() as db:
            return [dict(row) for row in db.execute(sql, params)]

    def save_episode(self, expediente: Expediente) -> None:
        with self.connect() as db:
            db.execute(
                "INSERT OR REPLACE INTO episodes(id,created_at,objective,task_kind,payload) VALUES(?,?,?,?,?)",
                (expediente.id, expediente.created_at, expediente.objective, expediente.task_kind,
                 json.dumps(expediente.to_dict(), ensure_ascii=False)),
            )

    def bandit_rows(self, context: str) -> list[dict[str, Any]]:
        with self.connect() as db:
            return [dict(row) for row in db.execute("SELECT * FROM bandit WHERE context=?", (context,))]

    def reward(self, context: str, operator: str, value: float) -> None:
        with self.connect() as db:
            db.execute("""INSERT INTO bandit(context,operator,trials,reward) VALUES(?,?,1,?)
                ON CONFLICT(context,operator) DO UPDATE SET
                reward=(bandit.reward*bandit.trials+excluded.reward)/(bandit.trials+1),
                trials=bandit.trials+1""", (context, operator, value))
