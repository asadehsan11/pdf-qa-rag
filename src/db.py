"""SQLite log of every question and answer."""
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path("data/qa_log.db")


def _connect():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """CREATE TABLE IF NOT EXISTS qa_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            question TEXT NOT NULL,
            answer TEXT NOT NULL,
            sources TEXT NOT NULL,
            latency_sec REAL
        )"""
    )
    return conn


def log_qa(question, answer, sources, latency_sec=None):
    """Save one question/answer pair."""
    conn = _connect()
    try:
        with conn:
            conn.execute(
                "INSERT INTO qa_log (created_at, question, answer, sources, latency_sec) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    datetime.now(timezone.utc).isoformat(),
                    question,
                    answer,
                    json.dumps(sources),
                    latency_sec,
                ),
            )
    finally:
        conn.close()


def get_history(limit=20):
    """Return the latest rows, newest first."""
    conn = _connect()
    try:
        return conn.execute(
            "SELECT created_at, question, answer FROM qa_log ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
    finally:
        conn.close()