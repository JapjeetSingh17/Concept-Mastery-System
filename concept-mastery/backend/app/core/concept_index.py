"""Vectorless Inverted Index + SQLite BM25 Retrieval Engine with Multi-Tenant Isolation.

STRICT VECTORLESS ARCHITECTURE:
  - NO dense embeddings (NO ChromaDB, FAISS, Pinecone, OpenAI embeddings).
  - Storage & retrieval strictly use:
      1. Canonical concept tag mapping via inverted table `concept_chunks` scoped by `user_id`.
      2. Inverted full-text index with native BM25 ranking (`fts5`) filtered by `user_id`.
      3. Exact 1-indexed page metadata and section boundaries.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import List, Tuple

from .. import config
from ..models.schemas import ChunkMetadata

INDEX_DB_PATH = config.VECTOR_STORE_PATH / "vectorless_index.db"

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


@contextmanager
def get_db():
    conn = sqlite3.connect(str(INDEX_DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    config.VECTOR_STORE_PATH.mkdir(parents=True, exist_ok=True)
    with get_db() as conn:
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


init_db()


class ConceptIndex:
    """Vectorless Index managing document chunks, canonical concept links, and BM25 search."""

    def __init__(self):
        init_db()

    def clear(self, user_id: str | None = None):
        """Clear indexed chunks. If user_id provided, clear only for that learner."""
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

    def index_chunks(self, chunks: list[ChunkMetadata], user_id: str = "default"):
        """Index a list of chunks into SQLite and the inverted index tables scoped by user_id."""
        if not chunks:
            return

        with get_db() as conn:
            for c in chunks:
                chunk_user = getattr(c, "user_id", None) or user_id
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
        cid = concept_id.lower().strip()
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
                return [self._row_to_chunk(r) for r in rows]

            # Fallback to BM25 query scoped to user_id if canonical tag has no direct hits
            bm25_hits = self.search_bm25(concept_id, max_chunks=max_chunks, user_id=user_id)
            return [hit[1] for hit in bm25_hits]

    def search_bm25(self, query: str, max_chunks: int = 4, user_id: str = "default") -> list[tuple[float, ChunkMetadata]]:
        """Vectorless BM25 ranking over inverted full-text index scoped to user_id."""
        import re
        cleaned_query = re.sub(r"[^\w\s]", " ", query).strip()
        terms = [t for t in cleaned_query.split() if len(t) > 2]
        if not terms:
            return []

        fts_match_query = " OR ".join(f'"{t}"' for t in terms[:10])

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
                return results
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
                return [(0.5, self._row_to_chunk(r)) for r in like_rows]

    def get_all_chunks(self, user_id: str = "default") -> list[ChunkMetadata]:
        with get_db() as conn:
            rows = conn.execute(
                """
                SELECT chunk_id, page_number, section_title, text, canonical_concepts,
                       source_filename, locator_label, source_id, source_type, source_title, origin_url_or_filename
                FROM chunks WHERE user_id = ? ORDER BY page_number ASC
                """,
                (user_id,)
            ).fetchall()
            return [self._row_to_chunk(r) for r in rows]

    def count(self, user_id: str | None = None) -> int:
        with get_db() as conn:
            if user_id:
                return conn.execute("SELECT count(*) FROM chunks WHERE user_id = ?", (user_id,)).fetchone()[0]
            return conn.execute("SELECT count(*) FROM chunks").fetchone()[0]


# Global singleton instance
GLOBAL_INDEX = ConceptIndex()
