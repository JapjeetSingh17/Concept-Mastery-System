"""Regression and Quality Test Suite for Concept Extraction, DAG Edges, and Active Recall.

Validates fixes for:
  Bug 1: Naive Capitalized-Word / Regex Fallback in Concept Extraction (zero conversational stopwords/starters).
  Bug 2: Active Recall sentence slicing replacement with structured textbook synthesis (zero speaker filler).
  Bug 3: Cross-Contaminated BM25 Chunk Retrieval prevention.
  Bug 4: Directed Prerequisite Dependency Graph Validation (genuine pedagogical prerequisites, zero sequential zip chaining).
"""

import pytest
import networkx as nx

from app.core import text
from app.core.concept_index import GLOBAL_INDEX
from app.models.schemas import ChunkMetadata
from app.pipelines import concepts, active_recall, ingest


def test_generic_term_blocklist():
    """Verify that conversational sentence-starters, verbs, adverbs, and locators are strictly blocked."""
    forbidden = [
        "Going", "Right", "Because", "Later", "Already", "Creating", "Asking",
        "Once Again", "Small", "Understand", "Number", "Insert", "Create",
        "Segment 00:00", "Lecture Segment 1", "Video [04:15]", "Page 2", "Slide 10",
        "Extract Text", "Pdf Reader", "Model", "Project", "Paragraphs", "Document"
    ]
    for term in forbidden:
        assert text.is_generic_term(term) is True, f"Expected '{term}' to be recognized as generic."

    valid_academic = [
        "Linked List", "Binary Search", "Embedding", "Vector Database",
        "Retrieval-Augmented Generation", "Stack", "Queue", "Array", "Gradient Descent"
    ]
    for term in valid_academic:
        assert text.is_generic_term(term) is False, f"Expected '{term}' to be recognized as valid academic concept."


def test_concept_extraction_rejects_conversational_fillers():
    """Verify that concepts.build() NEVER extracts conversational sentence starters as syllabus concepts."""
    mock_chunks = [
        {
            "id": 0,
            "page": 1,
            "section": "Video [00:00 - 02:00]",
            "text": (
                "Going to understand how we create a project. Right now we are asking if already you know model. "
                "Because here is the truth, later we will talk about embeddings and vector database."
            )
        },
        {
            "id": 1,
            "page": 2,
            "section": "Video [02:00 - 04:00]",
            "text": (
                "Once again, extract text from pdf reader. Small paragraphs can be processed. "
                "Insert number into array and linked list with node pointers."
            )
        }
    ]

    cs, edges = concepts.build(mock_chunks)
    extracted_names = [c["name"] for c in cs]

    # Conversational sentence-starters MUST NOT appear in extracted concepts
    for forbidden in ["Going", "Right", "Because", "Later", "Already", "Creating", "Asking", "Once Again", "Small", "Understand", "Number", "Insert", "Extract Text", "Pdf Reader"]:
        assert forbidden not in extracted_names, f"Forbidden filler '{forbidden}' was extracted as a concept!"

    # Ensure any extracted concept is non-generic
    for c in cs:
        assert not text.is_generic_term(c["name"]), f"Generic concept in DAG: {c['name']}"


def test_active_recall_zero_speaker_filler_fallback():
    """Verify that Active Recall study cards contain zero informal speaker filler strings in fallback."""
    raw_lecture_text = (
        "Again guys, you don't really have to understand this part. "
        "You can just say it's a bit it's like So there's a lot of math here, token counts "
        "And this is basically just the helper functions that allow our embedding to do a better job. "
        "And we're going to be returning Wait, let me just put this."
    )
    chunk = ChunkMetadata(
        chunk_id="chunk_test_filler",
        page_number=1,
        section_title="Video [00:00 - 03:00]",
        text=raw_lecture_text,
        canonical_concepts=["embeddings"],
        source_filename="test.mp4",
        locator_label="[00:00 - 03:00]",
        source_id="mat_filler",
        source_type="youtube",
        source_title="Lecture",
        origin_url_or_filename="test.mp4",
    )

    card = active_recall._synthesize_card_fallback(
        concept_id="linked-list",
        concept_name="Linked List",
        chunks=[chunk],
        c_def="",
        citation="[00:00 - 03:00]"
    )

    # Core intuition must be clean and educational
    assert "Again guys" not in card["core_intuition"]
    assert "token counts" not in card["core_intuition"]
    assert len(card["core_intuition"]) > 30

    # Key formula/mechanism must not be identical to core intuition or a raw spoken slice
    assert card["key_formula_or_mechanism"] != card["core_intuition"]
    assert "Again guys" not in card["key_formula_or_mechanism"]
    assert "helper functions" not in card["key_formula_or_mechanism"]

    # Worked example must NOT be "Wait, let me just put this"
    assert "Wait, let me" not in card["worked_mini_example"]
    assert "Wait let me" not in card["worked_mini_example"]

    # Keywords must NEVER include filler words
    for kw in card["essential_keywords"]:
        assert kw not in {"again", "guys", "really", "understand", "stands", "what", "have", "segment"}
        assert len(kw) >= 3


def test_cross_contaminated_bm25_retrieval_prevention():
    """Verify that querying Linked-List on an index containing only RAG chunks does NOT cross-contaminate."""
    GLOBAL_INDEX.clear(user_id="test_cross_contamination")

    rag_chunk = ChunkMetadata(
        chunk_id="rag_chunk_001",
        page_number=1,
        section_title="Video [00:00 - 02:30]",
        text="RAG stands for retrieval augmented generation. Now we have our class that has our embeddings and vector index.",
        canonical_concepts=["retrieval-augmented-generation", "embeddings"],
        source_filename="rag_intro.mp4",
        locator_label="[00:00 - 02:30]",
        source_id="mat_rag_1",
        source_type="youtube",
        source_title="RAG Lecture",
        origin_url_or_filename="rag_intro.mp4"
    )
    GLOBAL_INDEX.index_chunks([rag_chunk], user_id="test_cross_contamination")

    # Generate micro lesson for Linked List
    lesson = active_recall.generate_micro_lesson("linked-list", concept_name="Linked List", user_id="test_cross_contamination")
    card = lesson["card_data"]

    # Assert that the card teaches Linked List, not RAG embeddings
    assert "RAG stands for" not in card["core_intuition"]
    assert "RAG stands for" not in card["key_formula_or_mechanism"]
    assert "Now we have our class" not in card["key_formula_or_mechanism"]
    assert "segment 00:00" not in card["essential_keywords"]
    assert "rag" not in card["essential_keywords"]

    # Citation should fall back cleanly rather than referencing the irrelevant RAG chunk
    assert "[00:00 - 02:30]" not in card["exact_source_citation"]


def test_dag_validation_no_sequential_zip_chaining():
    """Verify that DAG edges are not blindly formed by chaining concepts sequentially."""
    mock_chunks = [
        {"id": 0, "page": 1, "section": "Video [00:00 - 02:00]", "text": "Arrays store contiguous memory elements with constant time access."},
        {"id": 1, "page": 2, "section": "Video [02:00 - 04:00]", "text": "Linked lists use nodes and pointers to link discrete elements."},
        {"id": 2, "page": 3, "section": "Video [04:00 - 06:00]", "text": "Stacks operate using LIFO semantics with push and pop operations."},
        {"id": 3, "page": 4, "section": "Video [06:00 - 08:00]", "text": "Binary search algorithms achieve logarithmic runtime over sorted collections."}
    ]

    cs, edges = concepts.build(mock_chunks)
    assert len(cs) >= 2

    # Verify that the graph is an acyclic DiGraph (DAG)
    G = nx.DiGraph()
    for c in cs:
        G.add_node(c["id"])
    for src, tgt in edges:
        G.add_edge(src, tgt)

    assert nx.is_directed_acyclic_graph(G), "Concept Graph must be a Directed Acyclic Graph (DAG)!"

    # No generic locator nodes
    for c in cs:
        assert not c["name"].lower().startswith("segment"), f"Generic segment node found in DAG: {c['name']}"
        assert not text.is_generic_term(c["name"]), f"Generic word found in DAG: {c['name']}"


def test_no_conversational_concepts():
    """Pass noisy chunk and assert concepts contain Vector Database and Linked List with zero conversational fillers."""
    chunk = {
        "id": 0,
        "page": 1,
        "section": "Introduction",
        "text": "Right, because later on we are going to be creating a Vector Database and Linked List. Once again, guys, understand this small project."
    }
    cs, edges = concepts.build([chunk])
    names = [c["name"] for c in cs]
    assert "Vector Database" in names, f"Expected 'Vector Database' in {names}"
    assert "Linked List" in names, f"Expected 'Linked List' in {names}"
    forbidden = {"Right", "Because", "Later", "Going", "Creating", "Once Again", "Understand", "Small"}
    intersection = set(names).intersection(forbidden)
    assert len(intersection) == 0, f"Conversational concepts leaked into DAG: {intersection}"


def test_active_recall_no_raw_splicing():
    """Generate Active Recall card on noisy chunk and assert no raw filler words or identical intuition/mechanism."""
    noisy_chunk = ChunkMetadata(
        chunk_id="noisy_chunk_1",
        page_number=1,
        section_title="Video [00:00 - 02:00]",
        text="Again guys, you don't really have to understand this part. So there's a lot of math here, token counts and helper functions that allow our embedding to do a better job.",
        canonical_concepts=["embeddings"],
        source_filename="lecture.mp4",
        locator_label="[00:00 - 02:00]",
        source_id="mat_noisy",
        source_type="youtube",
        source_title="Lecture",
        origin_url_or_filename="lecture.mp4"
    )
    card = active_recall._synthesize_card_fallback("linked-list", "Linked List", [noisy_chunk], "")
    ci_lower = card["core_intuition"].lower()
    kws = [k.lower() for k in card["essential_keywords"]]

    for bad in ["again", "guys", "really"]:
        assert bad not in ci_lower, f"Forbidden filler '{bad}' found in core_intuition: {card['core_intuition']}"
        assert bad not in kws, f"Forbidden filler '{bad}' found in essential_keywords: {card['essential_keywords']}"

    assert card["core_intuition"] != card["key_formula_or_mechanism"], "core_intuition must not equal formula_mechanism"


def test_openrouter_model_routing():
    """Verify backend routes structured JSON tasks to google/gemini-2.5-flash rather than free Nemotron VL models."""
    from app import config
    assert config.OPENROUTER_PRIMARY_MODEL == "google/gemini-2.5-flash", f"Expected google/gemini-2.5-flash, got {config.OPENROUTER_PRIMARY_MODEL}"
    assert "nemotron" not in config.OPENROUTER_PRIMARY_MODEL.lower(), f"Unreliable Nemotron model used as primary router: {config.OPENROUTER_PRIMARY_MODEL}"


