"""Vectorless Inverted Index + SQLite BM25 Retrieval Engine.

STRICT VECTORLESS ARCHITECTURE:
  - NO dense embeddings (NO ChromaDB, FAISS, Pinecone, OpenAI embeddings).
  - Storage & retrieval strictly use:
      1. Canonical concept tag mapping via inverted table `concept_chunks`.
      2. Inverted full-text index with native BM25 ranking (`fts5`).
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
    chunk_id            TEXT PRIMARY KEY,
    page_number         INTEGER NOT NULL,
    section_title       TEXT NOT NULL,
    text                TEXT NOT NULL,
    canonical_concepts  TEXT NOT NULL DEFAULT '[]',
    source_filename     TEXT NOT NULL,
    token_count         INTEGER NOT NULL DEFAULT 0
);

-- Inverted index: concept_id -> chunk mapping
CREATE TABLE IF NOT EXISTS concept_chunks (
    concept_id          TEXT NOT NULL,
    chunk_id            TEXT NOT NULL,
    page_number         INTEGER NOT NULL,
    content             TEXT NOT NULL,
    PRIMARY KEY (concept_id, chunk_id),
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
    with get_db() as conn:
        conn.executescript(_SCHEMA_SQL)


init_db()


class ConceptIndex:
    """Vectorless Index managing document chunks, canonical concept links, and BM25 search."""

    def __init__(self):
        init_db()

    def clear(self):
        """Clear all indexed chunks."""
        with get_db() as conn:
            conn.execute("DELETE FROM concept_chunks")
            conn.execute("DELETE FROM chunks")
            conn.execute("DELETE FROM chunk_fts")

    def index_chunks(self, chunks: list[ChunkMetadata]):
        """Index a list of chunks into SQLite and the inverted index tables."""
        if not chunks:
            return

        with get_db() as conn:
            for c in chunks:
                # 1. Insert into main chunks table
                concepts_json = json.dumps(c.canonical_concepts)
                token_count = len(c.text.split())
                conn.execute(
                    """
                    INSERT OR REPLACE INTO chunks 
                    (chunk_id, page_number, section_title, text, canonical_concepts, source_filename, token_count)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (c.chunk_id, c.page_number, c.section_title, c.text, concepts_json, c.source_filename, token_count)
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
                        (concept_id, chunk_id, page_number, content)
                        VALUES (?, ?, ?, ?)
                        """,
                        (cid.lower().strip(), c.chunk_id, c.page_number, c.text)
                    )

    def retrieve_chunks(self, concept_id: str, max_chunks: int = 3) -> list[ChunkMetadata]:
        """Strict vectorless retrieval: pull chunks by canonical concept ID from inverted index."""
        cid = concept_id.lower().strip()
        with get_db() as conn:
            rows = conn.execute(
                """
                SELECT c.chunk_id, c.page_number, c.section_title, c.text, c.canonical_concepts, c.source_filename
                FROM concept_chunks cc
                JOIN chunks c ON cc.chunk_id = c.chunk_id
                WHERE cc.concept_id = ?
                ORDER BY c.page_number ASC
                LIMIT ?
                """,
                (cid, max_chunks)
            ).fetchall()

            if rows:
                return [
                    ChunkMetadata(
                        chunk_id=r["chunk_id"],
                        page_number=r["page_number"],
                        section_title=r["section_title"],
                        text=r["text"],
                        canonical_concepts=json.loads(r["canonical_concepts"]),
                        source_filename=r["source_filename"],
                    )
                    for r in rows
                ]

            # Fallback to BM25 query if canonical tag has no direct hits
            bm25_hits = self.search_bm25(concept_id, max_chunks=max_chunks)
            return [hit[1] for hit in bm25_hits]

    def search_bm25(self, query: str, max_chunks: int = 4) -> list[tuple[float, ChunkMetadata]]:
        """Vectorless BM25 ranking over inverted full-text index."""
        # Sanitize query for FTS5 (remove special characters that break FTS syntax)
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
                    SELECT c.chunk_id, c.page_number, c.section_title, c.text, c.canonical_concepts, c.source_filename,
                           bm25(chunk_fts) as rank_score
                    FROM chunk_fts
                    JOIN chunks c ON chunk_fts.chunk_id = c.chunk_id
                    WHERE chunk_fts MATCH ?
                    ORDER BY rank_score ASC
                    LIMIT ?
                    """,
                    (fts_match_query, max_chunks)
                ).fetchall()

                results = []
                for r in rows:
                    # FTS5 bm25 returns negative values (lower is better); convert to positive normalized relevance
                    raw_score = float(r["rank_score"])
                    score = round(1.0 / (1.0 + max(0.0, raw_score + 10.0)), 3)
                    chunk = ChunkMetadata(
                        chunk_id=r["chunk_id"],
                        page_number=r["page_number"],
                        section_title=r["section_title"],
                        text=r["text"],
                        canonical_concepts=json.loads(r["canonical_concepts"]),
                        source_filename=r["source_filename"],
                    )
                    results.append((score, chunk))
                return results
            except Exception as e:
                # Fallback to LIKE if FTS fails
                like_rows = conn.execute(
                    """
                    SELECT chunk_id, page_number, section_title, text, canonical_concepts, source_filename
                    FROM chunks
                    WHERE text LIKE ?
                    LIMIT ?
                    """,
                    (f"%{terms[0]}%", max_chunks)
                ).fetchall()
                return [
                    (
                        0.5,
                        ChunkMetadata(
                            chunk_id=r["chunk_id"],
                            page_number=r["page_number"],
                            section_title=r["section_title"],
                            text=r["text"],
                            canonical_concepts=json.loads(r["canonical_concepts"]),
                            source_filename=r["source_filename"],
                        )
                    )
                    for r in like_rows
                ]

    def get_all_chunks(self) -> list[ChunkMetadata]:
        with get_db() as conn:
            rows = conn.execute(
                "SELECT chunk_id, page_number, section_title, text, canonical_concepts, source_filename FROM chunks ORDER BY page_number ASC"
            ).fetchall()
            return [
                ChunkMetadata(
                    chunk_id=r["chunk_id"],
                    page_number=r["page_number"],
                    section_title=r["section_title"],
                    text=r["text"],
                    canonical_concepts=json.loads(r["canonical_concepts"]),
                    source_filename=r["source_filename"],
                )
                for r in rows
            ]

    def count(self) -> int:
        with get_db() as conn:
            return conn.execute("SELECT count(*) FROM chunks").fetchone()[0]


# Global singleton instance
GLOBAL_INDEX = ConceptIndex()

