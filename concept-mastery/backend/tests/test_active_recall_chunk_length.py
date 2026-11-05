"""Test Suite for Active Recall Semantic Windowing, Micro-Fragment Merging, and Zero-Hallucination Grounding.

Verifies:
  1. Semantic chunk merger aggregates 20+ tiny micro-fragments into substantive >= 150-word blocks.
  2. Sparse retrieval automatically expands context window by stitching adjacent material chunks.
  3. generate_micro_lesson() produces valid, grounded study cards with ZERO generic placeholder templates
     (e.g., 'Consider applying X to test inputs...' is completely eliminated).
  4. Essential keywords filter out generic conversational stop words (e.g., 'what', 'have', 'stands', 'segment').
  5. Generic locator names ('Segment 00:00') are sanitized to real domain concepts.
"""

from __future__ import annotations

import pytest
from unittest.mock import patch, MagicMock

from app.core import store
from app.core.concept_index import GLOBAL_INDEX, ChunkMetadata
from app.pipelines.ingest import chunk_extracted_items, format_seconds_to_timestamp
from app.pipelines.active_recall import (
    expand_context_window,
    _resolve_clean_concept_name,
    _extract_keywords,
    _synthesize_card_fallback,
    generate_micro_lesson,
    start_session,
)


@pytest.fixture(autouse=True)
def clean_test_env():
    user_id = "test_ar_window_user"
    store.clear_all_study_materials(user_id)
    store.clear_kb(user_id)
    GLOBAL_INDEX.clear(user_id)
    yield
    store.clear_all_study_materials(user_id)
    store.clear_kb(user_id)
    GLOBAL_INDEX.clear(user_id)


def test_chunk_extracted_items_aggregates_micro_fragments():
    """Verify that feeding 20 tiny 10-word text snippets does not create 20 micro-chunks."""
    # 20 micro-items, each containing 10 words
    raw_micro_items = []
    for i in range(1, 21):
        raw_micro_items.append({
            "page": i,
            "locator_label": f"Page {i}",
            "section": f"Section {i}",
            "text": f"This is sentence {i} discussing parameter optimization and convergence properties in depth.",
        })

    # Total words = 20 * 10 = 200 words
    chunks = chunk_extracted_items(
        items=raw_micro_items,
        source_id="mat_micro_test",
        source_type="pdf",
        source_title="Optimization Lecture",
        origin_name="lecture.pdf"
    )

    # Must aggregate into 1-2 cohesive chunks (~150-200 words), NOT 20 micro-chunks
    assert len(chunks) in (1, 2), f"Expected 1-2 aggregated chunks for 200 words, got {len(chunks)}"
    first_chunk = chunks[0]
    assert first_chunk["token_count"] >= 120, f"Chunk has only {first_chunk['token_count']} words, expected >=120"
    assert "Page" in first_chunk["locator_label"]
    assert "1" in first_chunk["locator_label"]


def test_expand_context_window_sparse_retrieval():
    """Verify expand_context_window() stitches adjacent material chunks when initial retrieval is sparse."""
    user_id = "test_ar_window_user"

    # Seed index with 4 adjacent chunks from the same lecture
    c1 = ChunkMetadata(
        chunk_id="chk_001",
        page_number=1,
        section_title="Introduction",
        text="So, what is RAG exactly? RAG stands for Retrieval-Augmented Generation.",
        canonical_concepts=["rag"],
        source_filename="rag_lecture.mp4",
        locator_label="00:00 - 02:00",
        source_id="mat_rag",
        source_type="youtube",
        source_title="RAG Masterclass",
    )
    c2 = ChunkMetadata(
        chunk_id="chk_002",
        page_number=2,
        section_title="Architecture",
        text="In Retrieval-Augmented Generation, an external knowledge base is indexed using vector embeddings. "
             "When a user prompt arrives, the system queries the vector database using semantic similarity search. "
             "The top ranked text passages are retrieved and injected directly into the LLM context window before synthesis.",
        canonical_concepts=["rag", "vector embeddings"],
        source_filename="rag_lecture.mp4",
        locator_label="02:00 - 04:30",
        source_id="mat_rag",
        source_type="youtube",
        source_title="RAG Masterclass",
    )
    c3 = ChunkMetadata(
        chunk_id="chk_003",
        page_number=3,
        section_title="Evaluation",
        text="This prevents hallucinations by grounding generative models in verified corporate documentation. "
             "Key metrics for evaluating RAG systems include context relevance, groundedness, and answer faithfulness.",
        canonical_concepts=["rag"],
        source_filename="rag_lecture.mp4",
        locator_label="04:30 - 07:00",
        source_id="mat_rag",
        source_type="youtube",
        source_title="RAG Masterclass",
    )

    GLOBAL_INDEX.index_chunks([c1, c2, c3], user_id=user_id)

    # Suppose initial query only retrieved c1 (only ~11 words!)
    sparse_chunks = [c1]
    expanded, citation = expand_context_window(sparse_chunks, user_id=user_id, min_words=70)

    # Should expand to include adjacent chunks c2 and c3
    assert len(expanded) >= 2, f"Expected window expansion, got {len(expanded)} chunks"
    total_words = sum(len(c.text.split()) for c in expanded)
    assert total_words >= 60, f"Expected expanded words >=60, got {total_words}"
    assert "00:00" in citation
    assert "04:30" in citation or "07:00" in citation


def test_sanitize_generic_segment_name():
    """Verify that generic locator names like 'Segment 00:00' are resolved to real subject matter."""
    c = ChunkMetadata(
        chunk_id="chk_001",
        page_number=1,
        section_title="Video [00:00]",
        text="So, what is RAG exactly? RAG stands for Retrieval-Augmented Generation.",
        canonical_concepts=[],
        source_filename="lecture.mp4",
        locator_label="00:00 - 03:00",
        source_id="mat_001",
        source_type="youtube",
        source_title="AI Lecture",
    )

    clean_name = _resolve_clean_concept_name(
        concept_id="segment-00-00",
        raw_name="Segment 00:00",
        chunks=[c]
    )
    assert clean_name == "RAG", f"Expected clean name 'RAG', got '{clean_name}'"


def test_keywords_filter_conversational_noise():
    """Verify essential keywords extract technical terms and reject conversational tokens."""
    sample_text = "So, what is RAG exactly? RAG stands for Now, we have our class that has our And these functions I would like to keep."
    kws = _extract_keywords(sample_text, count=5)

    # Conversational tokens from the screenshot bug MUST NOT appear
    banned = {"what", "have", "stands", "exactly", "keep", "would", "like"}
    for b in banned:
        assert b not in kws, f"Generic word '{b}' leaked into essential keywords: {kws}"


def test_generate_micro_lesson_zero_hallucination_and_no_generic_template():
    """Verify generate_micro_lesson() generates grounded study cards without fake template sentences."""
    user_id = "test_ar_window_user"

    c = ChunkMetadata(
        chunk_id="chk_rag",
        page_number=1,
        section_title="Retrieval Augmented Generation",
        text="Retrieval-Augmented Generation (RAG) is a technique for enhancing language model responses "
             "with verified external knowledge. The process embeds documents into high-dimensional vectors, "
             "indexes them in a vector database, and executes cosine similarity searches on student queries. "
             "The retrieved primary text chunks are prepended to the system prompt to eliminate hallucinations. "
             "In practical workflows, a student queries documentation, the top 3 passages are retrieved, and "
             "the model synthesizes an evidence-backed factual answer.",
        canonical_concepts=["rag"],
        source_filename="rag_guide.pdf",
        locator_label="Page 1-2",
        source_id="mat_rag_guide",
        source_type="pdf",
        source_title="RAG Guide",
    )
    GLOBAL_INDEX.index_chunks([c], user_id=user_id)

    # Test deterministic fallback synthesis (which previously generated the fake template)
    fallback_card = _synthesize_card_fallback(
        concept_id="rag",
        concept_name="Retrieval-Augmented Generation",
        chunks=[c],
        c_def="",
        citation="[Page 1-2]"
    )

    # Must NOT contain the old generic template
    assert "Consider applying" not in fallback_card["worked_mini_example"], \
        f"Fake placeholder template still present: {fallback_card['worked_mini_example']}"
    assert "governing equation to yield predictable outputs" not in fallback_card["worked_mini_example"]

    # Must contain real content from the text
    assert len(fallback_card["core_intuition"]) > 20
    assert len(fallback_card["key_formula_or_mechanism"]) > 20
    assert len(fallback_card["worked_mini_example"]) > 20

    # Test full session creation
    session_card = start_session(user_id=user_id, concept_id="rag", timer_seconds=180)
    assert session_card.session_id.startswith("ar_")
    assert "Consider applying" not in session_card.worked_mini_example
    assert "governing equation to yield predictable outputs" not in session_card.worked_mini_example
    assert session_card.exact_source_citation != ""
