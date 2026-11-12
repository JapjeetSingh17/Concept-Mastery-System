"""Vectorless Inverted Index + SQLite BM25 Retrieval Engine with Multi-Tenant Isolation.

STRICT VECTORLESS ARCHITECTURE:
  - NO dense embeddings (NO ChromaDB, FAISS, Pinecone, OpenAI embeddings).
  - Storage & retrieval strictly use:
      1. Canonical concept tag mapping via inverted table `concept_chunks` scoped by `user_id`.
      2. Inverted full-text index with native BM25 ranking (`fts5`) filtered by `user_id`.
      3. Exact 1-indexed page metadata and section boundaries.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import sqlite3
import threading
from typing import List, Tuple

from .. import config
from ..models.schemas import ChunkMetadata
from .lru_cache import DEFAULT_LRU_MAXSIZE, DiskBackedLRUCache

MAX_ACTIVE_CONCEPT_CACHES = DEFAULT_LRU_MAXSIZE
INDEX_DB_PATH = config.SQLITE_INDEX_DB_PATH

_SCHEMA_SQL = """
-- Main chunks storage table
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id                 TEXT PRIMARY KEY,
    user_id                  TEXT NOT NULL DEFAULT 'default',
    page_number              INTEGER NOT NULL,
    section_title            TEXT NOT NULL,
    text                     TEXT NOT NULL,
    canonical_concepts       TEXT NOT NULL DEFAULT '[]',
    source_filename          TEXT NOT NULL,
    token_count              INTEGER NOT NULL DEFAULT 0,
    locator_label            TEXT NOT NULL DEFAULT '',
    source_id                TEXT NOT NULL DEFAULT '',
    source_type              TEXT NOT NULL DEFAULT 'pdf',
    source_title             TEXT NOT NULL DEFAULT '',
    origin_url_or_filename   TEXT NOT NULL DEFAULT ''
);

-- Inverted index: concept_id -> chunk mapping
CREATE TABLE IF NOT EXISTS concept_chunks (
    user_id             TEXT NOT NULL DEFAULT 'default',
    concept_id          TEXT NOT NULL,
    chunk_id            TEXT NOT NULL,
    page_number         INTEGER NOT NULL,
    content             TEXT NOT NULL,
    locator_label       TEXT NOT NULL DEFAULT '',
    source_type         TEXT NOT NULL DEFAULT 'pdf',
    PRIMARY KEY (user_id, concept_id, chunk_id),
    FOREIGN KEY (chunk_id) REFERENCES chunks(chunk_id) ON DELETE CASCADE
);

-- Inverted full-text search index with native BM25
CREATE VIRTUAL TABLE IF NOT EXISTS chunk_fts USING fts5(
    chunk_id UNINDEXED,
    section_title,
    content,
    tokenize='porter unicode61'
);
"""


_INITIALIZED_DB_PATHS: set[str] = set()
_INIT_LOCK = threading.Lock()


def _ensure_db():
    current_path = str(INDEX_DB_PATH)
    if current_path not in _INITIALIZED_DB_PATHS:
        with _INIT_LOCK:
            if current_path not in _INITIALIZED_DB_PATHS:
                config.VECTOR_STORE_PATH.mkdir(parents=True, exist_ok=True)
                Path(INDEX_DB_PATH).parent.mkdir(parents=True, exist_ok=True)
                conn = sqlite3.connect(str(INDEX_DB_PATH))
                conn.row_factory = sqlite3.Row
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("PRAGMA synchronous=NORMAL")
                conn.execute("PRAGMA foreign_keys=ON")
                conn.execute("PRAGMA busy_timeout=10000")
                try:
                    conn.executescript(_SCHEMA_SQL)
                    # Migrate existing tables if upgraded without DROP
                    cols_chunks = [r[1] for r in conn.execute("PRAGMA table_info(chunks)").fetchall()]
                    for col, col_type, default_val in [
                        ("user_id", "TEXT", "'default'"),
                        ("locator_label", "TEXT", "''"),
                        ("source_id", "TEXT", "''"),
                        ("source_type", "TEXT", "'pdf'"),
                        ("source_title", "TEXT", "''"),
                        ("origin_url_or_filename", "TEXT", "''")
                    ]:
                        if col not in cols_chunks:
                            try:
                                conn.execute(f"ALTER TABLE chunks ADD COLUMN {col} {col_type} DEFAULT {default_val}")
                            except Exception:
                                pass
                    cols_concept_chunks = [r[1] for r in conn.execute("PRAGMA table_info(concept_chunks)").fetchall()]
                    for col, col_type, default_val in [
                        ("user_id", "TEXT", "'default'"),
                        ("locator_label", "TEXT", "''"),
                        ("source_type", "TEXT", "'pdf'")
                    ]:
                        if col not in cols_concept_chunks:
                            try:
                                conn.execute(f"ALTER TABLE concept_chunks ADD COLUMN {col} {col_type} DEFAULT {default_val}")
                            except Exception:
                                pass

                    try:
                        conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_user ON chunks(user_id)")
                        conn.execute("CREATE INDEX IF NOT EXISTS idx_concept_chunks_user ON concept_chunks(user_id)")
                    except Exception:
                        pass
                    conn.commit()
                finally:
                    conn.close()
                _INITIALIZED_DB_PATHS.add(current_path)


def init_db():
    _ensure_db()


@contextmanager
def get_db():
    _ensure_db()
    Path(INDEX_DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(INDEX_DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=10000")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


class ConceptIndex:
    """Vectorless Index managing document chunks, canonical concept links, and BM25 search."""

    def __init__(self):
        # Lazy DB initialization: do not connect to SQLite on instance creation
        self._lock = threading.RLock()
        self._index_cache = self._build_cache(MAX_ACTIVE_CONCEPT_CACHES)
        self._retrieval_ops_count = 0

    @property
    def retrieval_ops_count(self) -> int:
        return self._retrieval_ops_count

    @staticmethod
    def _build_cache(
        maxsize: int = MAX_ACTIVE_CONCEPT_CACHES,
        cache_dir: Path | Callable[[], Path] | None = None,
    ) -> DiskBackedLRUCache:
        resolved_dir = cache_dir if cache_dir is not None else (lambda: config.VECTOR_STORE_PATH / "concept_cache")
        return DiskBackedLRUCache(cache_dir=resolved_dir, maxsize=maxsize)

    def configure_cache(
        self,
        maxsize: int = MAX_ACTIVE_CONCEPT_CACHES,
        cache_dir: Path | Callable[[], Path] | None = None,
    ) -> None:
        """Reconfigure the bounded LRU cache (primarily for tests or tuning)."""
        with self._lock:
            self._index_cache.clear()
            self._index_cache = self._build_cache(maxsize=maxsize, cache_dir=cache_dir)

    @staticmethod
    def _db_namespace() -> str:
        return hashlib.sha256(str(INDEX_DB_PATH).encode("utf-8")).hexdigest()[:8]

    def _cache_key(self, user_id: str, op: str, *parts: object) -> str:
        suffix = "|".join(str(p) for p in parts)
        return f"u:{user_id}|{self._db_namespace()}|{op}|{suffix}"

    def clear(self, user_id: str | None = None):
        """Clear indexed chunks. If user_id provided, clear only for that learner."""
        with self._lock:
            with get_db() as conn:
                if user_id:
                    conn.execute("DELETE FROM concept_chunks WHERE user_id = ?", (user_id,))
                    conn.execute(
                        "DELETE FROM chunk_fts WHERE chunk_id IN (SELECT chunk_id FROM chunks WHERE user_id = ?)",
                        (user_id,)
                    )
                    conn.execute("DELETE FROM chunks WHERE user_id = ?", (user_id,))
                else:
                    conn.execute("DELETE FROM concept_chunks")
                    conn.execute("DELETE FROM chunks")
                    conn.execute("DELETE FROM chunk_fts")
            if user_id:
                self._index_cache.invalidate_prefix(f"u:{user_id}|")
            else:
                self._index_cache.clear()

    def index_chunks(self, chunks: list[ChunkMetadata], user_id: str = "default"):
        """Index a list of chunks into SQLite and the inverted index tables scoped by user_id."""
        if not chunks:
            return

        with self._lock:
            touched_users: set[str] = {user_id}
            with get_db() as conn:
                for c in chunks:
                    chunk_user = getattr(c, "user_id", None) or user_id
                    touched_users.add(chunk_user)
                    # 1. Insert into main chunks table
                    concepts_json = json.dumps(c.canonical_concepts)
                    token_count = len(c.text.split())
                    loc_label = c.locator_label or (f"Page {c.page_number}" if c.source_type == "pdf" else (f"Slide {c.page_number}" if c.source_type == "pptx" else c.section_title))
                    conn.execute(
                        """
                        INSERT OR REPLACE INTO chunks 
                        (chunk_id, user_id, page_number, section_title, text, canonical_concepts, source_filename,
                         token_count, locator_label, source_id, source_type, source_title, origin_url_or_filename)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            c.chunk_id, chunk_user, c.page_number, c.section_title, c.text, concepts_json, c.source_filename,
                            token_count, loc_label, c.source_id, c.source_type, c.source_title, c.origin_url_or_filename
                        )
                    )

                    # 2. Insert into FTS5 for BM25 ranking
                    conn.execute("DELETE FROM chunk_fts WHERE chunk_id = ?", (c.chunk_id,))
                    conn.execute(
                        "INSERT INTO chunk_fts (chunk_id, section_title, content) VALUES (?, ?, ?)",
                        (c.chunk_id, c.section_title, c.text)
                    )

                    # 3. Populate inverted index for every canonical concept tag
                    for cid in c.canonical_concepts:
                        conn.execute(
                            """
                            INSERT OR IGNORE INTO concept_chunks 
                            (user_id, concept_id, chunk_id, page_number, content, locator_label, source_type)
                            VALUES (?, ?, ?, ?, ?, ?, ?)
                            """,
                            (chunk_user, cid.lower().strip(), c.chunk_id, c.page_number, c.text, loc_label, c.source_type)
                        )
            for uid in touched_users:
                self._index_cache.invalidate_prefix(f"u:{uid}|")

    def _row_to_chunk(self, r: sqlite3.Row) -> ChunkMetadata:
        keys = r.keys()
        return ChunkMetadata(
            chunk_id=r["chunk_id"],
            page_number=r["page_number"],
            section_title=r["section_title"],
            text=r["text"],
            canonical_concepts=json.loads(r["canonical_concepts"]) if r["canonical_concepts"] else [],
            source_filename=r["source_filename"],
            locator_label=r["locator_label"] if "locator_label" in keys and r["locator_label"] else f"Page {r['page_number']}",
            source_id=r["source_id"] if "source_id" in keys and r["source_id"] else "",
            source_type=r["source_type"] if "source_type" in keys and r["source_type"] else "pdf",
            source_title=r["source_title"] if "source_title" in keys and r["source_title"] else "",
            origin_url_or_filename=r["origin_url_or_filename"] if "origin_url_or_filename" in keys and r["origin_url_or_filename"] else r["source_filename"],
        )

    def retrieve_chunks(self, concept_id: str, max_chunks: int = 3, user_id: str = "default") -> list[ChunkMetadata]:
        """Strict vectorless retrieval: pull chunks by canonical concept ID scoped to user_id."""
        self._retrieval_ops_count += 1
        cid = concept_id.lower().strip()
        cache_key = self._cache_key(user_id, "concept", cid, max_chunks)
        with self._lock:
            cached = self._index_cache.get(cache_key)
            if cached is not None:
                return [ChunkMetadata(**item) for item in cached]

            with get_db() as conn:
                rows = conn.execute(
                    """
                    SELECT c.chunk_id, c.page_number, c.section_title, c.text, c.canonical_concepts,
                           c.source_filename, c.locator_label, c.source_id, c.source_type, c.source_title, c.origin_url_or_filename
                    FROM concept_chunks cc
                    JOIN chunks c ON cc.chunk_id = c.chunk_id
                    WHERE cc.concept_id = ? AND cc.user_id = ?
                    ORDER BY c.page_number ASC
                    LIMIT ?
                    """,
                    (cid, user_id, max_chunks)
                ).fetchall()

            if rows:
                chunks = [self._row_to_chunk(r) for r in rows]
            else:
                # Fallback to BM25 query scoped to user_id if canonical tag has no direct hits
                bm25_hits = self.search_bm25(concept_id, max_chunks=max_chunks, user_id=user_id)
                chunks = [hit[1] for hit in bm25_hits]

            self._index_cache.put(cache_key, [c.model_dump() for c in chunks])
            return chunks

    def search_bm25(self, query: str, max_chunks: int = 4, user_id: str = "default") -> list[tuple[float, ChunkMetadata]]:
        """Vectorless BM25 ranking over inverted full-text index scoped to user_id."""
        self._retrieval_ops_count += 1
        import re
        from .text import STOP, GENERIC_NON_CONCEPTS
        cleaned_query = re.sub(r"[^\w\s]", " ", query).strip()
        non_generic_terms = [t for t in cleaned_query.split() if len(t) > 2 and t.lower() not in STOP and t.lower() not in GENERIC_NON_CONCEPTS]
        terms = non_generic_terms or [t for t in cleaned_query.split() if len(t) > 2]
        if not terms:
            return []

        cache_key = self._cache_key(user_id, "bm25", " ".join(t.lower() for t in terms[:6]), max_chunks)
        with self._lock:
            cached = self._index_cache.get(cache_key)
            if cached is not None:
                return [(float(score), ChunkMetadata(**item)) for score, item in cached]

            # For multi-word queries, prioritize exact phrase and AND intersection to prevent cross-contamination
            if len(terms) >= 2:
                exact_phrase = " ".join(terms)
                and_clause = " AND ".join(f'"{t}"' for t in terms[:6])
                fts_match_query = f'"{exact_phrase}" OR ({and_clause})'
            else:
                fts_match_query = f'"{terms[0]}"'

            with get_db() as conn:
                try:
                    rows = conn.execute(
                        """
                        SELECT c.chunk_id, c.page_number, c.section_title, c.text, c.canonical_concepts,
                               c.source_filename, c.locator_label, c.source_id, c.source_type, c.source_title, c.origin_url_or_filename,
                               bm25(chunk_fts) as rank_score
                        FROM chunk_fts
                        JOIN chunks c ON chunk_fts.chunk_id = c.chunk_id
                        WHERE chunk_fts MATCH ? AND c.user_id = ?
                        ORDER BY rank_score ASC
                        LIMIT ?
                        """,
                        (fts_match_query, user_id, max_chunks)
                    ).fetchall()

                    results = []
                    for r in rows:
                        raw_score = float(r["rank_score"])
                        score = round(1.0 / (1.0 + max(0.0, raw_score + 10.0)), 3)
                        results.append((score, self._row_to_chunk(r)))
                except Exception:
                    like_rows = conn.execute(
                        """
                        SELECT chunk_id, page_number, section_title, text, canonical_concepts,
                               source_filename, locator_label, source_id, source_type, source_title, origin_url_or_filename
                        FROM chunks
                        WHERE text LIKE ? AND user_id = ?
                        LIMIT ?
                        """,
                        (f"%{terms[0]}%", user_id, max_chunks)
                    ).fetchall()
                    results = [(0.5, self._row_to_chunk(r)) for r in like_rows]

            self._index_cache.put(cache_key, [[score, chunk.model_dump()] for score, chunk in results])
            return results

    def get_all_chunks(self, user_id: str = "default") -> list[ChunkMetadata]:
        cache_key = self._cache_key(user_id, "all")
        with self._lock:
            cached = self._index_cache.get(cache_key)
            if cached is not None:
                return [ChunkMetadata(**item) for item in cached]

            with get_db() as conn:
                rows = conn.execute(
                    """
                    SELECT chunk_id, page_number, section_title, text, canonical_concepts,
                           source_filename, locator_label, source_id, source_type, source_title, origin_url_or_filename
                    FROM chunks WHERE user_id = ? ORDER BY page_number ASC
                    """,
                    (user_id,)
                ).fetchall()
                chunks = [self._row_to_chunk(r) for r in rows]

            self._index_cache.put(cache_key, [c.model_dump() for c in chunks])
            return chunks

    def count(self, user_id: str | None = "default") -> int:
        with get_db() as conn:
            if user_id:
                return conn.execute("SELECT count(*) FROM chunks WHERE user_id = ?", (user_id,)).fetchone()[0]
            return conn.execute("SELECT count(*) FROM chunks").fetchone()[0]


# Global singleton instance
GLOBAL_INDEX = ConceptIndex()
