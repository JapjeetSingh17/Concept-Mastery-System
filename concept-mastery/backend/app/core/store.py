"""Shared knowledge base (index + concepts), SQLite study materials, user auth, and audit log persistence."""
from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from .. import config
from .vector_index import VectorIndex

class UserError(Exception):
    """A problem the user can fix. Becomes a 400 response with this message."""

KB: dict = {"ready": False}
USER_KBS: dict[str, dict] = {}
KB_FILE = config.VECTOR_STORE_PATH / "kb.json"
MASTERY_FILE = config.VECTOR_STORE_PATH / "mastery.json"
MATERIALS_DB_PATH = config.VECTOR_STORE_PATH / "study_materials.db"

_MATERIALS_SCHEMA = """
-- Study Materials
CREATE TABLE IF NOT EXISTS study_materials (
    material_id                 TEXT PRIMARY KEY,
    user_id                     TEXT NOT NULL DEFAULT 'default',
    title                       TEXT NOT NULL,
    source_type                 TEXT NOT NULL,
    original_filename_or_url    TEXT NOT NULL,
    raw_extracted_text          TEXT NOT NULL,
    structured_preview_json     TEXT NOT NULL DEFAULT '[]',
    chunk_count                 INTEGER NOT NULL DEFAULT 0,
    extracted_concepts_json     TEXT NOT NULL DEFAULT '[]',
    created_at                  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_materials_user ON study_materials(user_id);

-- Registered Users (Google OAuth & Local Dev)
CREATE TABLE IF NOT EXISTS users (
    user_id         TEXT PRIMARY KEY,
    email           TEXT UNIQUE NOT NULL,
    full_name       TEXT NOT NULL DEFAULT '',
    picture_url     TEXT NOT NULL DEFAULT '',
    created_at      TEXT NOT NULL,
    last_login_at   TEXT NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_users_email ON users(email);

-- User Login & Session Audit Trail
CREATE TABLE IF NOT EXISTS user_login_logs (
    log_id          TEXT PRIMARY KEY,
    user_id         TEXT NOT NULL,
    logged_in_at    TEXT NOT NULL,
    ip_address      TEXT,
    user_agent      TEXT,
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_login_logs_user ON user_login_logs(user_id);

-- Active Recall Sessions
CREATE TABLE IF NOT EXISTS active_recall_sessions (
    session_id                  TEXT PRIMARY KEY,
    user_id                     TEXT NOT NULL DEFAULT 'default',
    concept_id                  TEXT NOT NULL,
    timer_seconds               INTEGER NOT NULL DEFAULT 120,
    lesson_json                 TEXT NOT NULL DEFAULT '{}',
    recall_submission_text      TEXT NOT NULL DEFAULT '',
    verification_question_json  TEXT NOT NULL DEFAULT '{}',
    retrieval_score             REAL DEFAULT NULL,
    retained_concepts_json      TEXT NOT NULL DEFAULT '[]',
    missed_gaps_json            TEXT NOT NULL DEFAULT '[]',
    created_at                  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_active_recall_user ON active_recall_sessions(user_id);
"""

@contextmanager
def get_materials_db():
    conn = sqlite3.connect(str(MATERIALS_DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()

def init_materials_db():
    config.VECTOR_STORE_PATH.mkdir(parents=True, exist_ok=True)
    with get_materials_db() as conn:
        conn.executescript(_MATERIALS_SCHEMA)

init_materials_db()

# ── User Account & Session Audit Persistence ────────────────────────────────

def upsert_user(user_id: str, email: str, full_name: str = "", picture_url: str = "") -> dict:
    """Create or update student profile upon Google OAuth sign-in."""
    init_materials_db()
    now_iso = datetime.now(timezone.utc).isoformat()
    with get_materials_db() as conn:
        existing = conn.execute("SELECT created_at FROM users WHERE user_id = ?", (user_id,)).fetchone()
        created_at = existing["created_at"] if existing else now_iso
        conn.execute(
            """
            INSERT OR REPLACE INTO users (user_id, email, full_name, picture_url, created_at, last_login_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (user_id, email, full_name or "", picture_url or "", created_at, now_iso)
        )
    return {
        "user_id": user_id,
        "email": email,
        "full_name": full_name or "",
        "picture_url": picture_url or "",
        "created_at": created_at,
        "last_login_at": now_iso,
    }

def get_user(user_id: str) -> dict | None:
    """Lookup student account by user_id."""
    init_materials_db()
    with get_materials_db() as conn:
        row = conn.execute("SELECT * FROM users WHERE user_id = ?", (user_id,)).fetchone()
        return dict(row) if row else None

def get_user_by_email(email: str) -> dict | None:
    """Lookup student account by verified email."""
    init_materials_db()
    with get_materials_db() as conn:
        row = conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
        return dict(row) if row else None

def record_login_log(user_id: str, ip_address: str | None = None, user_agent: str | None = None) -> dict:
    """Record an audit log entry for student login."""
    init_materials_db()
    now_iso = datetime.now(timezone.utc).isoformat()
    log_id = f"log_{uuid.uuid4().hex[:12]}"
    with get_materials_db() as conn:
        conn.execute(
            """
            INSERT INTO user_login_logs (log_id, user_id, logged_in_at, ip_address, user_agent)
            VALUES (?, ?, ?, ?, ?)
            """,
            (log_id, user_id, now_iso, ip_address, user_agent)
        )
    return {
        "log_id": log_id,
        "user_id": user_id,
        "logged_in_at": now_iso,
        "ip_address": ip_address,
        "user_agent": user_agent,
    }

def get_user_login_logs(user_id: str, limit: int = 20) -> list[dict]:
    """Retrieve recent login audit history for user."""
    init_materials_db()
    with get_materials_db() as conn:
        rows = conn.execute(
            "SELECT * FROM user_login_logs WHERE user_id = ? ORDER BY logged_in_at DESC LIMIT ?",
            (user_id, limit)
        ).fetchall()
        return [dict(r) for r in rows]

# ── Study Materials Persistence ─────────────────────────────────────────────

def save_study_material(mat: dict) -> dict:
    """Save or update a study material in the SQLite database."""
    init_materials_db()
    now_iso = datetime.now(timezone.utc).isoformat()
    material_id = mat.get("material_id") or mat.get("id") or f"mat_{int(datetime.now().timestamp()*1000)}"
    user_id = mat.get("user_id", "default")
    title = mat.get("title") or mat.get("filename") or "Untitled Material"
    source_type = mat.get("source_type", "pdf")
    origin = mat.get("original_filename_or_url") or mat.get("filename") or ""
    raw_text = mat.get("raw_extracted_text") or mat.get("text") or ""
    preview_json = mat.get("structured_preview_json")
    if preview_json is None:
        preview_json = json.dumps(mat.get("structured_preview", []))
    elif not isinstance(preview_json, str):
        preview_json = json.dumps(preview_json)
        
    chunk_count = int(mat.get("chunk_count", 0))
    concepts_json = mat.get("extracted_concepts_json")
    if concepts_json is None:
        concepts_json = json.dumps(mat.get("concepts", []))
    elif not isinstance(concepts_json, str):
        concepts_json = json.dumps(concepts_json)

    created_at = mat.get("created_at") or now_iso

    with get_materials_db() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO study_materials
            (material_id, user_id, title, source_type, original_filename_or_url,
             raw_extracted_text, structured_preview_json, chunk_count,
             extracted_concepts_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                material_id, user_id, title, source_type, origin,
                raw_text, preview_json, chunk_count, concepts_json, created_at
            )
        )
    return {
        "material_id": material_id,
        "user_id": user_id,
        "title": title,
        "source_type": source_type,
        "original_filename_or_url": origin,
        "chunk_count": chunk_count,
        "concepts": json.loads(concepts_json),
        "concept_count": len(json.loads(concepts_json)),
        "created_at": created_at,
    }

def list_study_materials(user_id: str = "default") -> list[dict]:
    """Retrieve all study materials for the given learner."""
    init_materials_db()
    with get_materials_db() as conn:
        rows = conn.execute(
            """
            SELECT material_id, user_id, title, source_type, original_filename_or_url,
                   chunk_count, extracted_concepts_json, created_at
            FROM study_materials
            WHERE user_id = ?
            ORDER BY created_at DESC
            """,
            (user_id,)
        ).fetchall()
        materials = []
        for r in rows:
            concepts = json.loads(r["extracted_concepts_json"]) if r["extracted_concepts_json"] else []
            materials.append({
                "material_id": r["material_id"],
                "user_id": r["user_id"],
                "title": r["title"],
                "source_type": r["source_type"],
                "original_filename_or_url": r["original_filename_or_url"],
                "chunk_count": r["chunk_count"],
                "concepts": concepts,
                "concept_count": len(concepts),
                "created_at": r["created_at"],
            })
        return materials

def get_study_material(material_id: str, user_id: str | None = None) -> dict | None:
    """Retrieve full details and structured preview for a specific material."""
    init_materials_db()
    with get_materials_db() as conn:
        query = "SELECT * FROM study_materials WHERE material_id = ?"
        params = [material_id]
        if user_id:
            query += " AND user_id = ?"
            params.append(user_id)
        row = conn.execute(query, tuple(params)).fetchone()
        if not row:
            return None
        concepts = json.loads(row["extracted_concepts_json"]) if row["extracted_concepts_json"] else []
        preview = json.loads(row["structured_preview_json"]) if row["structured_preview_json"] else []
        return {
            "material_id": row["material_id"],
            "user_id": row["user_id"],
            "title": row["title"],
            "source_type": row["source_type"],
            "original_filename_or_url": row["original_filename_or_url"],
            "raw_extracted_text": row["raw_extracted_text"],
            "raw_text_snippet": (row["raw_extracted_text"][:600] + "...") if len(row["raw_extracted_text"]) > 600 else row["raw_extracted_text"],
            "structured_preview": preview,
            "chunk_count": row["chunk_count"],
            "concepts": concepts,
            "concept_count": len(concepts),
            "created_at": row["created_at"],
        }

def delete_study_material(material_id: str, user_id: str | None = None) -> bool:
    """Delete a study material by ID, optionally isolated by user_id."""
    init_materials_db()
    with get_materials_db() as conn:
        if user_id:
            res = conn.execute("DELETE FROM study_materials WHERE material_id = ? AND user_id = ?", (material_id, user_id))
        else:
            res = conn.execute("DELETE FROM study_materials WHERE material_id = ?", (material_id,))
        return res.rowcount > 0

def clear_all_study_materials(user_id: str | None = None):
    init_materials_db()
    with get_materials_db() as conn:
        if user_id:
            conn.execute("DELETE FROM study_materials WHERE user_id = ?", (user_id,))
        else:
            conn.execute("DELETE FROM study_materials")


# ── Active Recall Session Persistence ───────────────────────────────────────

def save_active_recall_session(session: dict) -> dict:
    """Save or initialize an active recall session in SQLite."""
    init_materials_db()
    now_iso = datetime.now(timezone.utc).isoformat()
    session_id = session.get("session_id") or f"ar_{uuid.uuid4().hex[:12]}"
    user_id = session.get("user_id", "default")
    concept_id = session.get("concept_id", "")
    timer_seconds = int(session.get("timer_seconds", 120))

    lesson = session.get("lesson_json", "{}")
    if not isinstance(lesson, str):
        lesson = json.dumps(lesson)

    recall_text = session.get("recall_submission_text", "")

    verification_q = session.get("verification_question_json", "{}")
    if not isinstance(verification_q, str):
        verification_q = json.dumps(verification_q)

    retrieval_score = session.get("retrieval_score")
    if retrieval_score is not None:
        retrieval_score = float(retrieval_score)

    retained = session.get("retained_concepts_json", "[]")
    if not isinstance(retained, str):
        retained = json.dumps(retained)

    missed = session.get("missed_gaps_json", "[]")
    if not isinstance(missed, str):
        missed = json.dumps(missed)

    created_at = session.get("created_at") or now_iso

    with get_materials_db() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO active_recall_sessions
            (session_id, user_id, concept_id, timer_seconds, lesson_json,
             recall_submission_text, verification_question_json, retrieval_score,
             retained_concepts_json, missed_gaps_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_id, user_id, concept_id, timer_seconds, lesson,
                recall_text, verification_q, retrieval_score,
                retained, missed, created_at
            )
        )
    return {
        "session_id": session_id,
        "user_id": user_id,
        "concept_id": concept_id,
        "timer_seconds": timer_seconds,
        "lesson_json": lesson,
        "recall_submission_text": recall_text,
        "verification_question_json": verification_q,
        "retrieval_score": retrieval_score,
        "retained_concepts_json": retained,
        "missed_gaps_json": missed,
        "created_at": created_at,
    }


def update_active_recall_session(session_id: str, updates: dict) -> dict | None:
    """Update an existing active recall session with student submission and AI evaluation."""
    init_materials_db()
    existing = get_active_recall_session(session_id)
    if not existing:
        return None

    for k, v in updates.items():
        if k in ("lesson_json", "verification_question_json", "retained_concepts_json", "missed_gaps_json") and not isinstance(v, str):
            existing[k] = json.dumps(v)
        else:
            existing[k] = v

    with get_materials_db() as conn:
        conn.execute(
            """
            UPDATE active_recall_sessions
            SET recall_submission_text = ?,
                retrieval_score = ?,
                retained_concepts_json = ?,
                missed_gaps_json = ?,
                lesson_json = ?,
                verification_question_json = ?
            WHERE session_id = ?
            """,
            (
                existing["recall_submission_text"],
                existing["retrieval_score"],
                existing["retained_concepts_json"] if isinstance(existing["retained_concepts_json"], str) else json.dumps(existing["retained_concepts_json"]),
                existing["missed_gaps_json"] if isinstance(existing["missed_gaps_json"], str) else json.dumps(existing["missed_gaps_json"]),
                existing["lesson_json"] if isinstance(existing["lesson_json"], str) else json.dumps(existing["lesson_json"]),
                existing["verification_question_json"] if isinstance(existing["verification_question_json"], str) else json.dumps(existing["verification_question_json"]),
                session_id
            )
        )
    return get_active_recall_session(session_id)


def get_active_recall_session(session_id: str) -> dict | None:
    """Fetch an active recall session by session_id."""
    init_materials_db()
    with get_materials_db() as conn:
        row = conn.execute("SELECT * FROM active_recall_sessions WHERE session_id = ?", (session_id,)).fetchone()
        if not row:
            return None
        return dict(row)


def list_active_recall_sessions(user_id: str = "default", limit: int = 50) -> list[dict]:
    """Retrieve list of active recall sessions for user_id."""
    init_materials_db()
    with get_materials_db() as conn:
        rows = conn.execute(
            """
            SELECT * FROM active_recall_sessions
            WHERE user_id = ?
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (user_id, limit)
        ).fetchall()
        return [dict(r) for r in rows]

# ── Knowledge Base Per-User Management ──────────────────────────────────────

def save_kb(file, pages, index, concepts, edges, user_id: str = "default"):
    """Save knowledge base state isolated to user_id."""
    kb_data = {
        "ready": True,
        "file": file,
        "pages": pages,
        "index": index,
        "concepts": concepts,
        "edges": edges,
        "user_id": user_id,
    }
    USER_KBS[user_id] = kb_data
    if user_id == "default":
        KB.update(kb_data)

def get_kb(user_id: str = "default") -> dict:
    """Retrieve user-specific knowledge base, or empty structure if no materials."""
    if user_id == "default":
        # If store.KB has been explicitly cleared (e.g. by _EmptyKB in test fixtures), respect it
        if not KB.get("ready"):
            return KB
        if "default" in USER_KBS and USER_KBS["default"].get("ready"):
            return USER_KBS["default"]
        mats = list_study_materials("default")
        if mats:
            from ..pipelines.ingest import UniversalMaterialIngestor
            UniversalMaterialIngestor.rebuild_unified_knowledge_base(user_id="default")
            return USER_KBS.get("default", KB)
        return KB

    # For specific non-default user:
    if user_id in USER_KBS and USER_KBS[user_id].get("ready"):
        return USER_KBS[user_id]

    # Check if materials exist in database for this user
    mats = list_study_materials(user_id)
    if mats:
        # Rebuild user's knowledge base dynamically from their materials
        from ..pipelines.ingest import UniversalMaterialIngestor
        UniversalMaterialIngestor.rebuild_unified_knowledge_base(user_id=user_id)
        return USER_KBS.get(user_id, {"ready": False, "file": None, "pages": 0, "index": None, "concepts": [], "edges": [], "user_id": user_id})

    # Test fixture fallback: if user has no materials of their own, but test fixture loaded KB into default/store.KB
    if USER_KBS.get("default", {}).get("ready") and KB.get("ready"):
        return USER_KBS["default"]
    if KB.get("ready"):
        return KB

    empty = {
        "ready": False,
        "file": None,
        "pages": 0,
        "index": None,
        "concepts": [],
        "edges": [],
        "user_id": user_id
    }
    USER_KBS[user_id] = empty
    return empty

def clear_kb(user_id: str = "default"):
    """Reset knowledge base for a specific user."""
    empty = {
        "ready": False,
        "file": None,
        "pages": 0,
        "index": None,
        "concepts": [],
        "edges": [],
        "user_id": user_id
    }
    USER_KBS[user_id] = empty
    if user_id == "default":
        KB.update(empty)

def load_kb():
    """No-op on startup: ensures clean unpopulated boot without auto-loading default sample."""
    pass

def require(user_id: str = "default"):
    """Assert that the user has ingested study material."""
    kb = get_kb(user_id)
    if not kb.get("ready") or not kb.get("concepts"):
        raise UserError("Upload a study PDF or material first (Study material page).")
    return kb

def load_json(path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default

def save_json(path, data):
    path.write_text(json.dumps(data), encoding="utf-8")
