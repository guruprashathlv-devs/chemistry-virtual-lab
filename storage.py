"""Local SQLite persistence for the Cheminformatics Virtual Laboratory.

The student identifier is stored as a SHA-256 digest. The raw registration number
is not written to the database. For multi-user production deployment, use an
external managed database and real authentication; Streamlit Cloud local storage
may be ephemeral.
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any

DEFAULT_DB = Path(os.getenv("CHEM_LAB_DB_PATH", "data/chem_lab.db"))


def _connect() -> sqlite3.Connection:
    DEFAULT_DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DEFAULT_DB), timeout=15)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


@contextmanager
def _db():
    """Commit or roll back a short database operation and always close the connection."""
    conn = _connect()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    with _db() as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS activities (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_key TEXT NOT NULL,
            student_label TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            activity TEXT NOT NULL,
            details TEXT NOT NULL DEFAULT ''
        );
        CREATE INDEX IF NOT EXISTS idx_activities_student_time
            ON activities(student_key, timestamp DESC);
        CREATE TABLE IF NOT EXISTS quiz_attempts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_key TEXT NOT NULL,
            student_label TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            label TEXT NOT NULL,
            correct INTEGER NOT NULL,
            total INTEGER NOT NULL,
            score_percent INTEGER NOT NULL,
            review_json TEXT NOT NULL DEFAULT '[]'
        );
        CREATE INDEX IF NOT EXISTS idx_quiz_student_time
            ON quiz_attempts(student_key, timestamp DESC);
        """)


def make_student_key(name: str, registration_number: str, session_fallback: str = "guest") -> str:
    """Create a stable pseudonymous key without storing the raw registration number."""
    # Registration number is the stable pseudonymous identifier. Without one,
    # keep progress scoped to this browser session rather than merging students
    # who happen to share a name.
    identity = (registration_number or "").strip().casefold() or session_fallback
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def save_activity(student_key: str, student_label: str, timestamp: str, activity: str, details: str = "") -> None:
    with _db() as conn:
        conn.execute(
            "INSERT INTO activities(student_key, student_label, timestamp, activity, details) VALUES (?, ?, ?, ?, ?)",
            (student_key, student_label or "Student", timestamp, activity, details or ""),
        )


def save_quiz_attempt(student_key: str, student_label: str, attempt: dict[str, Any], review: list[dict[str, Any]]) -> None:
    with _db() as conn:
        conn.execute(
            "INSERT INTO quiz_attempts(student_key, student_label, timestamp, label, correct, total, score_percent, review_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                student_key, student_label or "Student", attempt["timestamp"],
                attempt.get("label", "Practice attempt"), int(attempt["correct"]),
                int(attempt["total"]), int(attempt["score_percent"]),
                json.dumps(review, ensure_ascii=False),
            ),
        )


def get_activities(student_key: str, limit: int = 500) -> list[dict[str, Any]]:
    with _db() as conn:
        rows = conn.execute(
            "SELECT timestamp, student_label AS student, activity, details FROM activities WHERE student_key=? ORDER BY id DESC LIMIT ?",
            (student_key, max(1, min(int(limit), 5000))),
        ).fetchall()
    return [dict(row) for row in rows]


def get_quiz_attempts(student_key: str, limit: int = 500) -> list[dict[str, Any]]:
    with _db() as conn:
        rows = conn.execute(
            "SELECT timestamp, student_label AS student, label, correct, total, score_percent FROM quiz_attempts WHERE student_key=? ORDER BY id DESC LIMIT ?",
            (student_key, max(1, min(int(limit), 5000))),
        ).fetchall()
    return [dict(row) for row in rows]


def get_progress_summary(student_key: str) -> dict[str, Any]:
    with _db() as conn:
        activity_count = conn.execute("SELECT COUNT(*) FROM activities WHERE student_key=?", (student_key,)).fetchone()[0]
        quiz_count, best_score, average_score = conn.execute(
            "SELECT COUNT(*), COALESCE(MAX(score_percent), 0), COALESCE(AVG(score_percent), 0) FROM quiz_attempts WHERE student_key=?",
            (student_key,),
        ).fetchone()
        last_activity = conn.execute(
            "SELECT timestamp FROM activities WHERE student_key=? ORDER BY id DESC LIMIT 1", (student_key,)
        ).fetchone()
    return {
        "activity_count": int(activity_count), "quiz_count": int(quiz_count),
        "best_score": int(best_score), "average_score": round(float(average_score), 1),
        "last_activity": last_activity[0] if last_activity else None,
    }


init_db()
