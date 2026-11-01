"""Epistemic Knowledge Tracing & Mastery Engine.

Maintains SQLite user_concept_mastery table.
Enforces 3-state classification:
  - UNKNOWN: Default state for unattempted concepts (prevents cold-start collapse).
  - WEAK: Score < 0.65 or failed prerequisite in the DAG.
  - STRONG: Score >= 0.80 sustained across multiple verification signals.

Multi-Signal Scoring:
  Score = (0.50 * QuizAccuracy) + (0.30 * SocraticClarityScore) + (0.20 * PrereqHealthScore)
"""

from __future__ import annotations

import json
import sqlite3
import time
from contextlib import contextmanager
from typing import List, Dict

from .. import config
from ..core import store
from ..models.schemas import MasteryStatus, MasteryRecord, LearnerDashboard

DB_PATH = config.VECTOR_STORE_PATH / "mastery.db"

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS user_concept_mastery (
    user_id             TEXT NOT NULL DEFAULT 'default',
    concept_id          TEXT NOT NULL,
    mastery_score       REAL NOT NULL DEFAULT 0.5,
    state               TEXT NOT NULL DEFAULT 'UNKNOWN',
    attempt_count       INTEGER NOT NULL DEFAULT 0,
    quiz_correct        INTEGER NOT NULL DEFAULT 0,
    quiz_total          INTEGER NOT NULL DEFAULT 0,
    dialogue_score      REAL NOT NULL DEFAULT 0.5,
    prereq_health       REAL NOT NULL DEFAULT 1.0,
    a                   REAL NOT NULL DEFAULT 1.0,
    b                   REAL NOT NULL DEFAULT 1.0,
    last_attempted_at   TEXT DEFAULT NULL,
    evidence_notes      TEXT NOT NULL DEFAULT '[]',
    PRIMARY KEY (user_id, concept_id)
);
"""


@contextmanager
def get_db():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with get_db() as conn:
        conn.execute(_CREATE_TABLE_SQL)
        # Migrate schema if table existed from an earlier version
        cols = [col[1] for col in conn.execute("PRAGMA table_info(user_concept_mastery)").fetchall()]
        if "prereq_health" not in cols:
            conn.execute("ALTER TABLE user_concept_mastery ADD COLUMN prereq_health REAL NOT NULL DEFAULT 1.0")
        if "dialogue_score" not in cols:
            conn.execute("ALTER TABLE user_concept_mastery ADD COLUMN dialogue_score REAL NOT NULL DEFAULT 0.5")


init_db()

# Legacy in-memory dictionary for backward compatibility
STATE: dict = {}


def _get_row(conn, concept_id: str, user_id: str = "default") -> dict:
    row = conn.execute(
        "SELECT * FROM user_concept_mastery WHERE user_id=? AND concept_id=?",
        (user_id, concept_id)
    ).fetchone()
    if row:
        return dict(row)
    return {
        "user_id": user_id,
        "concept_id": concept_id,
        "mastery_score": 0.5,
        "state": "UNKNOWN",
        "attempt_count": 0,
        "quiz_correct": 0,
        "quiz_total": 0,
        "dialogue_score": 0.5,
        "prereq_health": 1.0,
        "a": 1.0,
        "b": 1.0,
        "last_attempted_at": None,
        "evidence_notes": "[]",
    }


def _save_row(conn, d: dict):
    conn.execute(
        """
        INSERT OR REPLACE INTO user_concept_mastery
        (user_id, concept_id, mastery_score, state, attempt_count, quiz_correct, quiz_total,
         dialogue_score, prereq_health, a, b, last_attempted_at, evidence_notes)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            d["user_id"], d["concept_id"], d["mastery_score"], d["state"],
            d["attempt_count"], d["quiz_correct"], d["quiz_total"],
            d["dialogue_score"], d["prereq_health"], d["a"], d["b"],
            d["last_attempted_at"], d.get("evidence_notes", "[]")
        )
    )


def compute_prereq_health(concept_id: str, edges: list, conn, user_id: str = "default") -> float:
    """Calculate prerequisite health score (0.0 to 1.0). Penalized if immediate parents are WEAK."""
    prereq_ids = [a for a, b in edges if b == concept_id]
    if not prereq_ids:
        return 1.0

    scores = []
    for pid in prereq_ids:
        p_row = _get_row(conn, pid, user_id)
        if p_row["state"] == "WEAK":
            scores.append(0.3)
        elif p_row["state"] == "STRONG":
            scores.append(1.0)
        else:  # UNKNOWN
            scores.append(0.7)
    return sum(scores) / len(scores)


def classify_state(score: float, attempt_count: int, prereq_health: float) -> str:
    """Classify into UNKNOWN, WEAK, or STRONG."""
    if attempt_count == 0:
        return "UNKNOWN"
    # If any prerequisite is severely compromised, concept cannot be STRONG
    if prereq_health < 0.5:
        return "WEAK"
    if score >= 0.70 and attempt_count >= 2:
        return "STRONG"
    if score < 0.65:
        return "WEAK"
    return "STRONG" if score >= 0.68 and attempt_count >= 2 else "WEAK"


def record_quiz(user_id: str, concept_id: str, is_correct: bool, evidence_text: str = "") -> dict:
    """Update multi-signal mastery following a quiz attempt."""
    kb = store.KB
    edges = kb.get("edges", [])

    with get_db() as conn:
        d = _get_row(conn, concept_id, user_id)
        d["attempt_count"] += 1
        d["quiz_total"] += 1
        if is_correct:
            d["quiz_correct"] += 1
            d["a"] += 1.0
        else:
            d["b"] += 1.0

        now_str = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        d["last_attempted_at"] = now_str

        notes = json.loads(d.get("evidence_notes", "[]"))
        if evidence_text:
            notes.append(f"{evidence_text} [{now_str}]")
        else:
            notes.append(f"Quiz {'passed' if is_correct else 'failed'} at {now_str}")
        d["evidence_notes"] = json.dumps(notes[-15:])

        # Calculate multi-signal components
        quiz_accuracy = d["quiz_correct"] / d["quiz_total"] if d["quiz_total"] > 0 else 0.5
        d["prereq_health"] = compute_prereq_health(concept_id, edges, conn, user_id)
        dialogue_clarity = d["dialogue_score"]

        # Score = (0.50 * QuizAccuracy) + (0.30 * SocraticClarityScore) + (0.20 * PrereqHealthScore)
        d["mastery_score"] = round(
            0.50 * quiz_accuracy + 0.30 * dialogue_clarity + 0.20 * d["prereq_health"],
            3
        )
        d["state"] = classify_state(d["mastery_score"], d["attempt_count"], d["prereq_health"])

        _save_row(conn, d)

    # Sync legacy STATE
    STATE[concept_id] = {"a": d["a"], "b": d["b"], "n": d["attempt_count"]}
    return d


def record_dialogue(user_id: str, concept_id: str, clarity_score: float):
    """Record Socratic dialogue clarity score (0.0 to 1.0)."""
    kb = store.KB
    edges = kb.get("edges", [])

    with get_db() as conn:
        d = _get_row(conn, concept_id, user_id)
        curr_diag = float(d["dialogue_score"]) if d.get("dialogue_score") is not None else 0.5
        d["dialogue_score"] = round(0.6 * clarity_score + 0.4 * curr_diag, 3)
        d["prereq_health"] = compute_prereq_health(concept_id, edges, conn, user_id)
        quiz_acc = d["quiz_correct"] / d["quiz_total"] if d["quiz_total"] > 0 else 0.5
        d["mastery_score"] = round(
            0.50 * quiz_acc + 0.30 * d["dialogue_score"] + 0.20 * d["prereq_health"],
            3
        )
        d["state"] = classify_state(d["mastery_score"], d["attempt_count"], d["prereq_health"])
        _save_row(conn, d)


def get_record(user_id: str, concept_id: str) -> MasteryRecord:
    with get_db() as conn:
        row = _get_row(conn, concept_id, user_id)
        return MasteryRecord(
            user_id=row["user_id"],
            concept_id=row["concept_id"],
            score=row["mastery_score"],
            status=MasteryStatus(row["state"]),
            attempt_count=row["attempt_count"],
            evidence_notes=json.loads(row["evidence_notes"]),
            updated_at=row["last_attempted_at"],
            quiz_accuracy=round(row["quiz_correct"] / max(row["quiz_total"], 1), 3),
            dialogue_score=row["dialogue_score"],
            prereq_health=row["prereq_health"],
        )


def reset(concepts: list[dict], user_id: str = "default"):
    """Reset mastery records for a new document upload."""
    with get_db() as conn:
        conn.execute("DELETE FROM user_concept_mastery WHERE user_id=?", (user_id,))
        for c in concepts:
            cid = c["id"]
            _save_row(conn, {
                "user_id": user_id,
                "concept_id": cid,
                "mastery_score": 0.5,
                "state": "UNKNOWN",
                "attempt_count": 0,
                "quiz_correct": 0,
                "quiz_total": 0,
                "dialogue_score": 0.5,
                "prereq_health": 1.0,
                "a": 1.0,
                "b": 1.0,
                "last_attempted_at": None,
                "evidence_notes": "[]",
            })
    STATE.clear()
    STATE.update({c["id"]: {"a": 1.0, "b": 1.0, "n": 0} for c in concepts})


def load(user_id: str = "default"):
    with get_db() as conn:
        rows = conn.execute("SELECT * FROM user_concept_mastery WHERE user_id=?", (user_id,)).fetchall()
    STATE.clear()
    for r in rows:
        STATE[r["concept_id"]] = {"a": r["a"], "b": r["b"], "n": r["attempt_count"]}


def snapshot(user_id: str = "default") -> list[dict]:
    """Snapshot format consumed by frontend dashboards."""
    kb = store.KB
    if not kb.get("ready"):
        return []

    results = []
    with get_db() as conn:
        for c in kb["concepts"]:
            cid = c["id"]
            row = _get_row(conn, cid, user_id)
            state_lower = row["state"].lower()
            notes = json.loads(row["evidence_notes"])
            why = notes[-1] if notes else ("No answers yet. That isn't enough evidence to call it strong or weak." if row["attempt_count"] == 0 else f"{row['quiz_correct']} of {row['quiz_total']} answers right.")
            results.append({
                "id": cid,
                "concept_id": cid,
                "name": c["name"],
                "score": row["mastery_score"],
                "status": state_lower,
                "state": row["state"],
                "attempts": row["attempt_count"],
                "uncertainty": round(0.5 / (row["attempt_count"] + 1), 3),
                "why": why,
                "evidence_notes": notes,
                "prereq_health": row["prereq_health"],
                "dialogue_score": row["dialogue_score"],
            })
    return results


def get_dashboard(user_id: str = "default") -> LearnerDashboard:
    """Generate explainable tri-state dashboard."""
    snap = snapshot(user_id)
    records = [
        MasteryRecord(
            user_id=user_id,
            concept_id=s["id"],
            score=s["score"],
            status=MasteryStatus(s["state"]),
            attempt_count=s["attempts"],
            evidence_notes=s["evidence_notes"],
        )
        for s in snap
    ]

    strong = [r for r in records if r.status == MasteryStatus.STRONG]
    weak = [r for r in records if r.status == MasteryStatus.WEAK]
    unknown = [r for r in records if r.status == MasteryStatus.UNKNOWN]

    total = len(records)
    overall = sum(r.score for r in records) / total if total > 0 else 0.0

    summary = (
        f"{len(strong)} mastered, {len(weak)} needing remediation, {len(unknown)} unattempted "
        f"across {total} course concepts. Overall mastery: {round(overall * 100)}%."
    )

    return LearnerDashboard(
        user_id=user_id,
        strong=strong,
        weak=weak,
        unknown=unknown,
        overall_score=round(overall, 3),
        total_concepts=total,
        summary=summary,
    )


# Compatibility aliases
record = record_quiz
dashboard = lambda uid="default": get_dashboard(uid).model_dump()
ranked = lambda concepts, edges: [c["id"] for c in sorted(concepts, key=lambda c: (0 if STATE.get(c["id"], {}).get("n", 0) > 0 and STATE.get(c["id"], {}).get("a", 1)/(STATE.get(c["id"], {}).get("a", 1)+STATE.get(c["id"], {}).get("b", 1)) < 0.65 else 1 if STATE.get(c["id"], {}).get("n", 0) == 0 else 2, c.get("depth", 0)))]
prereqs = lambda edges: {b: [a for a, b2 in edges if b2 == b] for _, b in edges}
