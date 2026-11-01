"""Grounded Socratic Tutor Engine with Two-Layer Anti-Hallucination Guardrails.

Guardrail 1: Refuses direct answers/solutions; responds with Socratic scaffolding and analogies.
Guardrail 2: Strictly grounded in vectorless SQLite chunks for the active concept, citing [Page X].
"""

from __future__ import annotations

import re
from ..core import store, llm
from ..core.concept_index import GLOBAL_INDEX
from ..core.text import NORMALIZER, tokens
from ..models.schemas import TutorChatResponse, Citation
from . import mastery

SOCRATIC_GUARDRAIL_SYSTEM = """You are an expert Socratic tutor for an adaptive concept mastery platform.

STRICT TWO-LAYER GUARDRAILS:
1. NEVER provide direct quiz answers, full solution steps, or completed homework.
2. If asked directly for an answer (e.g. "What is the answer to question 2?"), REFUSE politely and provide a scaffolding conceptual hint instead.
3. Base your explanations ONLY on the provided textbook excerpt. DO NOT hallucinate facts outside the context.
4. MANDATORY: You MUST cite the source page for every key fact, e.g. [Page 3] or (p. 3).
5. Guide the student with one intuitive real-world analogy and one targeted follow-up question to test their understanding.
6. Keep explanations concise, clear, and pedagogically sound (under 120 words)."""

REMEDIAL_SYSTEM = """You are a patient remedial tutor helping a student overcome a diagnosed misconception.

RULES:
1. Explain the target concept step-by-step using simpler analogies.
2. Directly address why the prerequisite foundational concept matters.
3. Cite the exact textbook page using [Page X].
4. DO NOT give direct quiz solutions.
5. End with a simple check-in question."""


def chat(
    message: str,
    concept_id: str | None = None,
    user_id: str = "default",
    mode: str = "socratic"
) -> TutorChatResponse:
    """Socratic chat response grounded in vectorless chunks."""
    kb = store.require()

    # 1. Vectorless Retrieval: pull chunks for active concept or via BM25
    retrieved_chunks = []
    if concept_id:
        retrieved_chunks = GLOBAL_INDEX.retrieve_chunks(concept_id, max_chunks=3)
    if not retrieved_chunks:
        bm25_hits = GLOBAL_INDEX.search_bm25(message, max_chunks=3)
        retrieved_chunks = [hit[1] for hit in bm25_hits]

    if not retrieved_chunks:
        # Fallback to KB index if SQLite is fresh
        from_kb = kb.get("index")
        if from_kb and hasattr(from_kb, "search"):
            hits = from_kb.search(message, k=3)
            retrieved_chunks = [
                type("ChunkMeta", (), {
                    "chunk_id": str(c["id"]),
                    "page_number": c["page"],
                    "section_title": c["section"],
                    "text": c["text"],
                    "canonical_concepts": [concept_id] if concept_id else []
                })()
                for _, c in hits
            ]

    if not retrieved_chunks:
        return TutorChatResponse(
            answer="I couldn't locate this topic in your study material. Please check that the PDF covers it.",
            citations=[],
            detected_concepts=[],
            dialogue_score=0.5,
        )

    # 2. Build context with explicit page labels
    context_text = "\n\n".join(
        f"[Page {c.page_number} · {c.section_title}]\n{c.text}"
        for c in retrieved_chunks
    )

    system_prompt = REMEDIAL_SYSTEM if mode == "remedial" else SOCRATIC_GUARDRAIL_SYSTEM
    user_prompt = (
        f"Context from uploaded study material:\n{context_text}\n\n"
        f"Target Concept: {concept_id or 'General'}\n"
        f"Student Query: {message}"
    )

    raw_answer = llm.complete(system_prompt, user_prompt, max_tokens=600)
    if not raw_answer:
        # Vectorless extractive offline fallback with page citation
        top_chunk = retrieved_chunks[0]
        raw_answer = (
            f"Based on the study material on [Page {top_chunk.page_number}], {top_chunk.section_title} "
            f"states: \"{top_chunk.text[:220]}...\"\n\n"
            f"How does this relate to your current question?"
        )

    # 3. Build structured citations
    citations: list[Citation] = []
    seen_pages = set()
    for c in retrieved_chunks[:3]:
        if c.page_number not in seen_pages:
            seen_pages.add(c.page_number)
            citations.append(Citation(
                page=c.page_number,
                snippet=c.section_title,
                text=c.text
            ))

    # 4. Detect canonical concepts in the conversation
    all_concept_names = [cn["name"] for cn in kb.get("concepts", [])]
    detected = NORMALIZER.detect_canonical_concepts(message + " " + raw_answer, all_concept_names)

    # 5. Measure dialogue clarity signal for Knowledge Tracing
    clarity_score = _score_dialogue_clarity(message, raw_answer, retrieved_chunks)
    if concept_id:
        mastery.record_dialogue(user_id, concept_id, clarity_score)

    return TutorChatResponse(
        answer=raw_answer,
        citations=citations,
        detected_concepts=detected,
        dialogue_score=clarity_score,
    )


def reteach(concept_id: str, user_id: str = "default") -> str:
    """Generate a remedial Socratic intervention for a weak concept."""
    res = chat(
        message=f"I am struggling with this concept. Can you re-explain the fundamentals with an analogy?",
        concept_id=concept_id,
        user_id=user_id,
        mode="remedial"
    )
    return res.answer


def _score_dialogue_clarity(query: str, answer: str, chunks: list) -> float:
    """Heuristic dialogue clarity score (0.0 to 1.0) measuring technical grounding."""
    score = 0.5
    if len(query.split()) >= 6:
        score += 0.1
    if re.search(r"\[Page\s+\d+\]|\(p\.\s*\d+\)", answer, re.I):
        score += 0.2
    # Check vocabulary grounding
    chunk_words = set(tokens(" ".join(c.text for c in chunks)))
    query_words = set(tokens(query))
    if chunk_words and query_words:
        overlap = len(chunk_words & query_words) / len(query_words)
        score += min(0.2, overlap * 0.3)
    return min(1.0, round(score, 3))


# Compatibility alias
ask = lambda q, docId=None, concept_id=None: chat(q, concept_id=concept_id).model_dump()
