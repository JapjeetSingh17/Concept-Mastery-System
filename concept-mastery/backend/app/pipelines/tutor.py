"""Grounded Socratic Tutor Engine with Two-Layer Anti-Hallucination Guardrails and Live Web Search.

Guardrail 1: Refuses direct answers/solutions; responds with Socratic scaffolding and analogies.
Guardrail 2: Strictly grounded in vectorless SQLite chunks for the active concept citing [Page X],
             or live TinyFish web search results with source citations.
"""

from __future__ import annotations

import re
from ..core import store, llm, web_search
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

WEB_SOCRATIC_SYSTEM = """You are an expert Socratic tutor for an adaptive concept mastery platform.
The student's question was researched using live web search (TinyFish Search) because it extends beyond the local study PDF or was requested.

STRICT SOCRATIC GUARDRAILS:
1. NEVER provide direct quiz answers or completed homework.
2. Ground your explanation in the provided web search excerpts.
3. MANDATORY: Cite web sources in your explanation, e.g. [Source: Site Name] or [Web: Title].
4. Guide the student with conceptual intuition, an analogy, and one targeted check-in question.
5. Keep explanations concise, clear, and pedagogical (under 130 words)."""

GREETING_PATTERN = re.compile(
    r"^(hi|hello|hey|heya|howdy|greetings|good\s+(morning|afternoon|evening|day)|yo|sup|help|who\s+are\s+you)[!?.\s]*$",
    re.IGNORECASE
)

WEB_SEARCH_TRIGGER = re.compile(
    r"\b(search\s+(the\s+)?web|look\s+up|search\s+online|google|internet|web\s+search|latest|recent|real\s*world)\b",
    re.IGNORECASE
)


def is_greeting(msg: str) -> bool:
    """Detect if student message is a conversational opener or greeting."""
    cleaned = msg.strip().lower()
    return bool(GREETING_PATTERN.match(cleaned)) or cleaned in {"hi", "hello", "hey", "help", "who are you"}


def chat(
    message: str,
    concept_id: str | None = None,
    user_id: str = "default",
    mode: str = "socratic"
) -> TutorChatResponse:
    """Socratic chat response grounded in vectorless chunks or live web search."""
    clean_message = message.strip()
    kb = store.KB
    kb_ready = kb.get("ready", False)

    # 0. Conversational Greeting Handler
    if is_greeting(clean_message):
        if kb_ready:
            doc_name = kb.get("file", "uploaded course material")
            concepts = [c["name"] for c in kb.get("concepts", [])[:4]]
            concept_hint = f" We are studying **{doc_name}** covering concepts like {', '.join(concepts)}." if concepts else ""
            reply = (
                f"Hello! I am your Socratic AI Tutor.{concept_hint}\n\n"
                f"What topic, concept, or question would you like to explore today? "
                f"You can also ask me to search the web if you need external examples or broader context!"
            )
        else:
            reply = (
                "Hello! Welcome to the Concept Mastery Platform. I am your Socratic AI Tutor.\n\n"
                "To begin personalized study on your course material, please upload your PDF in the **Study Material** tab. "
                "You can also ask me any question right now, and I can search the web using TinyFish Search to help you learn!"
            )
        return TutorChatResponse(
            answer=reply,
            citations=[],
            detected_concepts=[],
            dialogue_score=0.5,
        )

    wants_web_search = bool(WEB_SEARCH_TRIGGER.search(clean_message))

    # 1. Vectorless Retrieval from Local SQLite / KB (unless explicitly asked to search the web)
    retrieved_chunks = []
    if not wants_web_search:
        if concept_id:
            retrieved_chunks = GLOBAL_INDEX.retrieve_chunks(concept_id, max_chunks=3)
        if not retrieved_chunks:
            bm25_hits = GLOBAL_INDEX.search_bm25(clean_message, max_chunks=3)
            retrieved_chunks = [hit[1] for hit in bm25_hits]

        if not retrieved_chunks and kb_ready:
            from_kb = kb.get("index")
            if from_kb and hasattr(from_kb, "search"):
                hits = from_kb.search(clean_message, k=3)
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

    # 2. Live Web Search Fallback (if explicitly requested or not found in local material)
    if wants_web_search or not retrieved_chunks:
        search_query = WEB_SEARCH_TRIGGER.sub("", clean_message).strip() or clean_message
        web_results = web_search.search(search_query, max_results=3)

        if web_results:
            context_text = "\n\n".join(
                f"[Source: {r.get('site_name') or 'Web'} · {r['title']}]\n{r['snippet']}\nURL: {r['url']}"
                for r in web_results
            )
            user_prompt = (
                f"Live Web Search Excerpts (TinyFish Search):\n{context_text}\n\n"
                f"Student Question: {clean_message}"
            )
            raw_answer = llm.complete(WEB_SOCRATIC_SYSTEM, user_prompt, max_tokens=600)
            if not raw_answer:
                top = web_results[0]
                raw_answer = (
                    f"According to {top.get('site_name') or 'web sources'} ([Source: {top['title']}]): "
                    f"\"{top['snippet'][:220]}...\"\n\n"
                    f"How does this connect with what you are trying to understand?"
                )

            web_citations = [
                Citation(
                    page=None,
                    snippet=f"[{r.get('site_name') or 'Web'}] {r['title']}",
                    text=r["snippet"],
                    source_type="web",
                    url=r.get("url"),
                    title=r.get("title")
                )
                for r in web_results
            ]

            return TutorChatResponse(
                answer=raw_answer,
                citations=web_citations,
                detected_concepts=[],
                dialogue_score=0.6,
            )

    # 3. If no local chunks and no web results found
    if not retrieved_chunks:
        return TutorChatResponse(
            answer="I couldn't locate this topic in your study material or on the web. Could you rephrase your question or specify the concept?",
            citations=[],
            detected_concepts=[],
            dialogue_score=0.5,
        )

    # 4. Grounded Response from Local Textbook Material
    context_text = "\n\n".join(
        f"[Page {c.page_number} · {c.section_title}]\n{c.text}"
        for c in retrieved_chunks
    )

    system_prompt = REMEDIAL_SYSTEM if mode == "remedial" else SOCRATIC_GUARDRAIL_SYSTEM
    user_prompt = (
        f"Context from uploaded study material:\n{context_text}\n\n"
        f"Target Concept: {concept_id or 'General'}\n"
        f"Student Query: {clean_message}"
    )

    raw_answer = llm.complete(system_prompt, user_prompt, max_tokens=600)
    if not raw_answer:
        top_chunk = retrieved_chunks[0]
        raw_answer = (
            f"Based on the study material on [Page {top_chunk.page_number}], {top_chunk.section_title} "
            f"states: \"{top_chunk.text[:220]}...\"\n\n"
            f"How does this relate to your current question?"
        )

    citations: list[Citation] = []
    seen_pages = set()
    for c in retrieved_chunks[:3]:
        if c.page_number not in seen_pages:
            seen_pages.add(c.page_number)
            citations.append(Citation(
                page=c.page_number,
                snippet=c.section_title,
                text=c.text,
                source_type="document"
            ))

    all_concept_names = [cn["name"] for cn in kb.get("concepts", [])] if kb_ready else []
    detected = NORMALIZER.detect_canonical_concepts(clean_message + " " + raw_answer, all_concept_names)

    clarity_score = _score_dialogue_clarity(clean_message, raw_answer, retrieved_chunks)
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
        message="I am struggling with this concept. Can you re-explain the fundamentals with an analogy?",
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
    if re.search(r"\[Page\s+\d+\]|\(p\.\s*\d+\)|\[Source:", answer, re.I):
        score += 0.2
    chunk_words = set(tokens(" ".join(c.text for c in chunks)))
    query_words = set(tokens(query))
    if chunk_words and query_words:
        overlap = len(chunk_words & query_words) / len(query_words)
        score += min(0.2, overlap * 0.3)
    return min(1.0, round(score, 3))


# Compatibility alias
ask = lambda q, docId=None, concept_id=None: chat(q, concept_id=concept_id).model_dump()
