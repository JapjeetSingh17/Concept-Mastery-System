"""Vectorless Document Ingestion Pipeline.

Extracts text from PDF at semantic section boundaries with exact 1-indexed page numbers,
normalizes domain concepts with fuzzy matching, and populates the SQLite inverted index.
"""

from __future__ import annotations

import asyncio
import re
import uuid
from pathlib import Path

from ..core import store
from ..core.concept_index import GLOBAL_INDEX
from ..core.text import sentences, NORMALIZER
from ..models.schemas import ChunkMetadata, UploadResponse
from . import concepts, mastery

CHUNK_WORDS = 110
_PAGE_NO = re.compile(r"^(page\s+)?\d+(\s*(of|/)\s*\d+)?$", re.I)
_NUMBERING = re.compile(r"^(chapter|section|unit)?\s*\d+(\.\d+)*[.)]?\s+", re.I)
_SMALL = {"of", "and", "the", "in", "to", "for", "a", "vs", "with", "on", "an"}


def extract_pages(path: Path) -> list[tuple[int, str]]:
    """Extract pages from PDF returning [(1-based page_num, text), ...]."""
    from pypdf import PdfReader
    reader = PdfReader(str(path))
    pages = []
    for i, p in enumerate(reader.pages):
        text = p.extract_text() or ""
        pages.append((i + 1, text))
    return pages


def is_heading(line: str) -> bool:
    """Detect if line is a semantic heading / section title."""
    words = line.split()
    if not (1 <= len(words) <= 9) or len(line) > 70 or line[-1] in ".,;:!?":
        return False
    if _NUMBERING.match(line):
        return True
    alpha = [w for w in words if w[0].isalpha()]
    return bool(alpha) and (line.isupper() or all(w[0].isupper() or w.lower() in _SMALL for w in alpha))


def clean_heading(line: str) -> str:
    h = _NUMBERING.sub("", line).strip()
    return h.title() if h.isupper() else h


def _emit(chunks: list[dict], page: int, section: str, buf: list[str], source_filename: str):
    """Emit semantic chunks respecting sentence boundaries."""
    body = re.sub(r"\s+", " ", " ".join(buf)).strip()
    buf.clear()
    cur, n = [], 0
    for s in sentences(body) + [None]:
        if s is None or (n + len(s.split()) > CHUNK_WORDS and cur):
            text = " ".join(cur)
            words = text.split()
            if len(words) >= 8:
                chunks.append({
                    "id": len(chunks),
                    "chunk_id": str(uuid.uuid4()),
                    "page": page,
                    "section": section,
                    "text": text,
                    "source_filename": source_filename,
                    "token_count": len(words),
                    "canonical_concepts": [],
                })
            cur, n = [], 0
        if s:
            cur.append(s)
            n += len(s.split())


def chunk_pages(pages: list[tuple[int, str]], source_filename: str) -> list[dict]:
    """Segment pages into section-based chunks."""
    chunks: list[dict] = []
    section = "Introduction"
    for page, text in pages:
        buf: list[str] = []
        for line in (l.strip() for l in text.splitlines()):
            if not line or _PAGE_NO.match(line):
                continue
            if is_heading(line):
                _emit(chunks, page, section, buf, source_filename)
                section = clean_heading(line)
            else:
                buf.append(line)
        _emit(chunks, page, section, buf, source_filename)
    return chunks


def run_ingest(path: Path) -> dict:
    """Run full vectorless ingestion pipeline for a single PDF synchronously."""
    return run_ingest_multiple([path])


def run_ingest_multiple(paths: list[Path]) -> dict:
    """Run vectorless ingestion pipeline across one or more study PDFs."""
    if not paths:
        raise store.UserError("No PDF files provided for ingestion.")

    all_raw_chunks: list[dict] = []
    total_pages = 0
    valid_filenames: list[str] = []

    for path in paths:
        if not path.exists():
            continue
        valid_filenames.append(path.name)
        pages = extract_pages(path)
        total_pages += len(pages)
        raw_chunks = chunk_pages(pages, source_filename=path.name)
        all_raw_chunks.extend(raw_chunks)

    if not all_raw_chunks:
        raise store.UserError("No readable text found in uploaded PDF(s). Scanned PDFs need OCR first.")

    # 1. Build Prerequisite Concept DAG across all combined material
    cs, edges = concepts.build(all_raw_chunks)
    concept_names = [c["name"] for c in cs]

    # 2. Tag each chunk with canonical concept IDs
    for chunk in all_raw_chunks:
        detected = NORMALIZER.detect_canonical_concepts(chunk["text"], concept_names)
        section_norm = NORMALIZER.canonicalize(chunk["section"])
        if section_norm and section_norm not in detected:
            detected.append(section_norm)
        chunk["canonical_concepts"] = detected

    # 3. Populate Vectorless SQLite Inverted Index
    GLOBAL_INDEX.clear()
    chunk_models = [
        ChunkMetadata(
            chunk_id=c["chunk_id"],
            page_number=c["page"],
            section_title=c["section"],
            text=c["text"],
            canonical_concepts=c["canonical_concepts"],
            source_filename=c.get("source_filename", valid_filenames[0]),
        )
        for c in all_raw_chunks
    ]
    GLOBAL_INDEX.index_chunks(chunk_models)

    # 4. Save to KB for frontend and API compatibility
    combined_name = ", ".join(valid_filenames)
    from ..core.vector_index import VectorIndex
    index = VectorIndex.build(all_raw_chunks)
    store.save_kb(combined_name, total_pages, index, cs, edges)
    mastery.reset(cs)

    return {
        "job_id": combined_name,
        "file": combined_name,
        "filename": combined_name,
        "filenames": valid_filenames,
        "files_count": len(valid_filenames),
        "pages": total_pages,
        "chunks": len(all_raw_chunks),
        "chunk_count": len(all_raw_chunks),
        "chunks_indexed": len(all_raw_chunks),
        "concepts": [c["name"] for c in cs],
        "concepts_found": [c["name"] for c in cs],
        "graph_nodes": len(cs),
        "graph_edges": len(edges),
    }


async def run_ingest_async(path_or_paths: Path | list[Path]) -> dict:
    """Run ingest in an executor so it never blocks the async event loop."""
    paths = [path_or_paths] if isinstance(path_or_paths, Path) else path_or_paths
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, run_ingest_multiple, paths)
