"""Test Suite for Pedagogical Upgrades, Quiz Anti-Bias, Socratic Tutor, and Smart Notes Exports."""

import json
import pytest
from pathlib import Path

from app import config
from app.core import text, store
from app.pipelines import concepts, quiz, tutor, notes, tools
from app.models.schemas import SmartNoteRequest, QuizSubmission


@pytest.fixture(autouse=True)
def clean_test_env():
    """Ensure clean test environment."""
    user_id = "test_pedagogy_user"
    yield user_id


def test_generic_non_concepts_filtered():
    """Verify that stop words, segment locators, and generic non-concepts are rejected."""
    # Check stop words & conversational noise
    assert text.is_generic_term("the") is True
    assert text.is_generic_term("is") is True
    assert text.is_generic_term("what") is True
    assert text.is_generic_term("stands") is True
    assert text.is_generic_term("have") is True
    assert text.is_generic_term("update") is True
    assert text.is_generic_term("logical") is True
    assert text.is_generic_term("primary") is True
    assert text.is_generic_term("Segment 00:00") is True
    assert text.is_generic_term("Video [04:15]") is True
    assert text.is_generic_term("Lecture Segment 2") is True
    assert text.is_generic_term("Voice Note (01:23)") is True

    # Real domain concepts must NOT be generic
    assert text.is_generic_term("Binary Search Tree") is False
    assert text.is_generic_term("Dynamic Programming") is False
    assert text.is_generic_term("Vector Database") is False
    assert text.is_generic_term("Retrieval Augmented Generation") is False
    assert text.is_generic_term("Recursion") is False
    assert text.is_generic_term("Array") is False


def test_candidate_terms_excludes_junk():
    """Verify that _candidate_terms extracts authentic domain concepts rather than segment locators or noise."""
    mock_chunks = [
        {
            "id": "c1",
            "section": "Segment 00:00",
            "text": "In this video segment we introduce Retrieval Augmented Generation. What is RAG exactly? Have we looked at vector databases?",
            "page": 1,
        },
        {
            "id": "c2",
            "section": "Segment 01:15",
            "text": "Retrieval Augmented Generation combines dense vector retrieval with large language models to ground generation in external knowledge.",
            "page": 2,
        },
        {
            "id": "c3",
            "section": "Segment 03:45",
            "text": "Vector databases index embeddings using hierarchical navigable small world graphs or inverted file indexes for low latency search.",
            "page": 3,
        }
    ]

    terms = concepts._candidate_terms(mock_chunks, limit=10)
    terms_lower = [t.lower() for t in terms]

    # Must NOT contain segment or conversational filler
    for junk in ["segment", "segment 00:00", "what", "exactly", "stands", "have", "video"]:
        assert junk not in terms_lower, f"Junk '{junk}' found in extracted candidate terms: {terms}"

    # Build DAG concepts from these chunks
    c_list, _ = concepts.build(mock_chunks)
    c_names = [c["name"].lower() for c in c_list]
    for c in c_names:
        assert not concepts.is_generic_section_heading(c), f"Generic heading '{c}' became a concept node!"
        assert not text.is_generic_term(c), f"Generic term '{c}' became a concept node!"


def test_quiz_mcq_no_topic_name_bias():
    """Verify that when testing a topic (e.g. 'Array'), the correct option is NEVER simply 'Array'."""
    concept = {
        "id": "array",
        "name": "Array",
        "key": "array",
        "definition": "An array allocates elements in contiguous memory locations with O(1) random access."
    }

    chunks = [
        type("ChunkMeta", (), {
            "chunk_id": "ch_arr_1",
            "page_number": 1,
            "section_title": "Array Data Structure",
            "text": "An array allocates elements in contiguous memory locations, enabling constant-time O(1) random access via index arithmetic.",
            "locator_label": "Page 1"
        })(),
        type("ChunkMeta", (), {
            "chunk_id": "ch_arr_2",
            "page_number": 2,
            "section_title": "Array Operations",
            "text": "Inserting into a full static array requires linear O(N) reallocation and copying of existing elements.",
            "locator_label": "Page 2"
        })()
    ]

    kb = {"ready": True, "concepts": [concept]}
    store.KB = kb

    questions = quiz.generate_diagnostic_quiz(
        concept_id="array",
        user_id="test_quiz_user",
        count=4,
        question_types=["mcq", "fill_in_the_blanks", "mcq", "fill_in_the_blanks"],
        offset=0,
        chunks=chunks,
        kb=kb
    )

    assert len(questions) == 4

    mcqs = [q for q in questions if q.question_type == "mcq"]
    assert len(mcqs) >= 1

    for mcq in mcqs:
        # 1. The correct option must NEVER be the concept name itself!
        assert 0 <= mcq.correct_option_idx < len(mcq.options)
        correct_answer = mcq.options[mcq.correct_option_idx]
        assert correct_answer.lower().strip() != "array", "Answer bias trap! Correct answer is just 'Array'!"

        # 2. None of the options should be just the isolated concept name
        for opt in mcq.options:
            assert opt.lower().strip() != "array", "Option is just bare topic name!"
            # Options must be substantive statements
            assert len(opt.split()) >= 3, f"Option too short: '{opt}'"

        # 3. Question text should test operational mechanism
        assert "operational mechanism" in mcq.question_text or "statement" in mcq.question_text

    # Verify Fill-in-the-blanks doesn't blank out 'Array'
    fitbs = [q for q in questions if q.question_type == "fill_in_the_blanks"]
    assert len(fitbs) >= 1
    for fitb in fitbs:
        assert fitb.target_blank_answer.lower() != "array", "FITB blanks out topic name instead of mechanism term!"
        assert len(fitb.target_blank_answer) >= 4


def test_socratic_tutor_four_pillars():
    """Verify that Socratic Tutor produces a structured 4-pillar educational response."""
    concept = {
        "id": "vector_embeddings",
        "name": "Vector Embeddings",
        "key": "vector_embeddings",
        "definition": "Vector embeddings represent high-dimensional semantic relationships as dense vectors in Euclidean space."
    }

    chunks = [
        type("ChunkMeta", (), {
            "chunk_id": "ch_emb_1",
            "page_number": 12,
            "section_title": "Dense Vector Embeddings",
            "text": "Vector embeddings map discrete tokens into continuous Euclidean vector spaces where semantic similarity corresponds to geometric cosine distance.",
            "locator_label": "[12:30]"
        })()
    ]

    kb = {"ready": True, "concepts": [concept]}
    store.KB = kb

    res = tutor.chat(
        message="Can you explain how vector embeddings work and why cosine similarity is used?",
        concept_id="vector_embeddings",
        user_id="test_tutor_user",
        chunks=chunks,
        kb=kb
    )

    answer = res.answer
    assert answer is not None
    assert len(answer.split()) >= 50, "Answer too short!"

    # Must contain 4 pedagogical pillars
    assert "Mental Model" in answer or "Analogy" in answer or "1." in answer
    assert "Architecture" in answer or "Mechanism" in answer or "2." in answer
    assert "Example" in answer or "Walkthrough" in answer or "3." in answer
    assert "Challenge" in answer or "Socratic" in answer or "4." in answer

    # Citations must be present
    assert len(res.citations) >= 1
    assert res.citations[0].locator_label == "[12:30]"


def test_smart_notes_synthesis_no_fake_bellman():
    """Verify that Smart Notes synthesis does not hardcode fake Bellman equations for arbitrary subjects."""
    chunks = [
        type("ChunkMeta", (), {
            "chunk_id": "ch_note_1",
            "page_number": 5,
            "section_title": "Binary Search Trees",
            "text": "A binary search tree enforces the binary search invariant: for any node, all left subtree keys are smaller and right keys are larger.",
            "locator_label": "Page 5"
        })(),
        type("ChunkMeta", (), {
            "chunk_id": "ch_note_2",
            "page_number": 6,
            "section_title": "Tree Invariants",
            "text": "Balanced trees guarantee logarithmic O(log N) lookup overhead, whereas degenerate trees degrade into linear O(N) linked lists.",
            "locator_label": "Page 6"
        })()
    ]

    req = SmartNoteRequest(
        concept_id="binary_search_tree",
        task_type="exam",
        user_id="test_note_user",
        custom_instructions=""
    )

    res = notes.generate_smart_note(req)
    assert res.note_id.startswith("sn_")
    assert "Binary Search Tree" in res.title
    md = res.markdown_content

    # Assert NO fake Bellman equations or RL templates
    assert "V(s)" not in md
    assert "R(s, a)" not in md
    assert "gamma \\in" not in md
    assert "equilibrium response evaluates to" not in md

    # Assert genuine grounded content from chunks
    assert "binary search" in md.lower() or "invariant" in md.lower()
    assert len(res.key_takeaways) >= 3


def test_notes_pdf_and_pptx_export_creates_verified_files():
    """Verify that export_note_artifact produces real .pdf and .pptx files on disk without 404s."""
    note_record = {
        "note_id": "sn_test_export_99",
        "user_id": "test_export_user",
        "concept_id": "retrieval_augmented_generation",
        "material_id": "mat_1",
        "task_type": "exam",
        "custom_instructions": "",
        "title": "High-Yield Exam Revision: Retrieval Augmented Generation",
        "markdown_content": """# High-Yield Exam Revision: Retrieval Augmented Generation

## 1. High-Yield Principles & Grounded Definitions
- **Foundational Definition**: Retrieval Augmented Generation combines neural retrieval with generative language models [Page 1].
- **Core Mechanism**: Queries are converted into dense vector embeddings and matched via cosine similarity [Page 2].

## 2. Step-by-Step Governing Mechanisms
- Step 1: Pre-index textbook chunks using dense vector embeddings [Page 1].
- Step 2: Retrieve top-k semantic matches using approximate nearest neighbors [Page 2].
- Step 3: Augment LLM prompt context with retrieved passages [Page 3].

## 3. Common Exam Traps & Boundary Pitfalls
- Conflating retrieval precision with generation hallucination [Page 2].
- Overlooking chunk boundary segmentation fragmentation [Page 3].

## 4. Worked Walkthrough & Verification
- Input query: 'Explain the Bellman optimality principle' [Page 1].
- Retrieved passage: Section 4.2 definition with citations [Page 2].
- Generated output: Grounded derivation citing exact pages [Page 3].
""",
        "key_takeaways": [
            "Dense retrieval grounds generative language models [Page 1].",
            "Cosine similarity matches semantic vector embeddings [Page 2].",
            "Context augmentation prevents factual hallucinations [Page 3]."
        ],
        "citations": ["Page 1", "Page 2", "Page 3"],
        "created_at": "2026-11-09T00:00:00Z"
    }

    store.save_smart_note(note_record)

    # 1. Test PDF Export
    pdf_res = notes.export_note_artifact(
        note_id="sn_test_export_99",
        user_id="test_export_user",
        export_format="pdf"
    )
    assert "pdf_url" in pdf_res
    pdf_url = pdf_res["pdf_url"]
    assert pdf_url.endswith(".pdf")

    pdf_filename = Path(pdf_url).name
    pdf_path = config.GENERATED_ARTIFACTS_DIR / pdf_filename
    assert pdf_path.exists(), f"PDF artifact {pdf_path} was not created on disk!"
    assert pdf_path.stat().st_size > 100, "PDF file is empty!"

    # 2. Test PPTX Export
    pptx_res = notes.export_note_artifact(
        note_id="sn_test_export_99",
        user_id="test_export_user",
        export_format="pptx"
    )
    assert "pptx_url" in pptx_res
    pptx_url = pptx_res["pptx_url"]
    assert pptx_url.endswith(".pptx")

    pptx_filename = Path(pptx_url).name
    pptx_path = config.GENERATED_ARTIFACTS_DIR / pptx_filename
    assert pptx_path.exists(), f"PPTX artifact {pptx_path} was not created on disk!"
    assert pptx_path.stat().st_size > 1000, "PPTX file is empty!"

    # Clean up test note
    store.delete_smart_note("sn_test_export_99", user_id="test_export_user")
