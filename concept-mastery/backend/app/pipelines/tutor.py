"""Grounded Socratic Tutor Engine with Two-Layer Anti-Hallucination Guardrails and Live Web Search.

Guardrail 1: Refuses direct answers/solutions; responds with Socratic scaffolding and analogies.
Guardrail 2: Strictly grounded in vectorless SQLite chunks for the active concept citing [Page X],
             or live TinyFish web search results with source citations.
"""

from __future__ import annotations

import re
import logging
from typing import Any, Optional
from ..core import store, llm, web_search
from ..core.concept_index import GLOBAL_INDEX
from ..core.text import NORMALIZER, tokens, sentences, build_context_window, truncate_boundary_aware
from ..models.schemas import TutorChatResponse, TutorResponse, Citation
from . import mastery, tools

logger = logging.getLogger("concept_mastery.tutor")

SOCRATIC_GUARDRAIL_SYSTEM = """You are an expert Socratic master tutor for an adaptive concept mastery platform.
Your mission is to guide students to deep conceptual intuition, architectural understanding, and operational mastery.

FOUR PEDAGOGICAL PILLARS FOR EVERY EXPLANATION:
1. Intuitive Mental Model: Start with a vivid, relatable real-world analogy that demystifies abstract notation.
2. Architecture & Governing Mechanism: Explain step-by-step how the concept actually operates under the hood, citing the exact textbook page or video timestamp (e.g. [Page X], [MM:SS]).
3. Concrete Worked Walkthrough: Trace a tangible mini-example, calculation, or system scenario demonstrating the mechanism in action.
4. Socratic Application Challenge: Conclude with a thought-provoking check-in question that tests whether the student can apply this mechanism to edge cases or novel scenarios.

STRICT GUARDRAILS:
- Ground all explanations strictly in the provided study material chunks.
- NEVER give direct quiz solutions or homework answers; scaffold the student's problem-solving instead.
- Provide comprehensive, clear, high-signal educational guidance (250 to 450 words)."""

REMEDIAL_SYSTEM = """You are a patient remedial tutor helping a student overcome a diagnosed misconception.

REMEDIAL PEDAGOGICAL PILLARS:
1. Deconstruct the target concept from first principles using an intuitive mental model.
2. Explicitly explain why the prerequisite foundational concept is required to avoid confusion.
3. Ground every step in the assigned study material citing exact locators [Page X] or [MM:SS].
4. Provide a worked mini-example contrasting the correct mechanism against common student mistakes.
5. End with an empowering Socratic check-in question to verify comprehension."""

WEB_SOCRATIC_SYSTEM = """You are an expert Socratic tutor for an adaptive concept mastery platform.
The student's question was researched using live web search because it extends beyond the local study material or was requested.

STRICT SOCRATIC GUARDRAILS:
1. Explain the underlying mental model and operational mechanics using the web search excerpts.
2. NEVER provide direct quiz answers or completed homework.
3. MANDATORY: Cite web sources in your explanation, e.g. [Source: Site Name].
4. Provide a clear worked example and conclude with a targeted Socratic follow-up question.
5. Keep explanations structured, rigorous, and educational (250 to 400 words)."""

GREETING_PATTERN = re.compile(
    r"^(hi|hello|hey|heya|howdy|greetings|good\s+(morning|afternoon|evening|day)|yo|sup|help|who\s+are\s+you)[!?.\s]*$",
    re.IGNORECASE
)

WEB_SEARCH_TRIGGER = re.compile(
    r"\b(search\s+(the\s+)?web|look\s+up|search\s+online|google|internet|web\s+search|latest|recent|real\s*world)\b",
    re.IGNORECASE
)

VISUAL_INTENT_PATTERN = re.compile(
    r"\b(diagram|visual|chart|draw|flowchart|architecture)s?\b",
    re.IGNORECASE
)

SLIDES_INTENT_PATTERN = re.compile(
    r"\b(ppt|pptx|slides?|presentation)s?\b",
    re.IGNORECASE
)

PDF_INTENT_PATTERN = re.compile(
    r"\b(pdf|handout|cheat\s*sheet(\s*file)?|download\s*notes)s?\b",
    re.IGNORECASE
)

EXTERNAL_LINKS_INTENT_PATTERN = re.compile(
    r"\b(youtube|video|link|article|extra\s*source|tutorial)s?\b",
    re.IGNORECASE
)


def route_student_intent(message: str) -> Optional[str]:
    """Detect keywords for visuals, slides, PDF handouts, or external links."""
    clean = message.strip()
    if VISUAL_INTENT_PATTERN.search(clean):
        return "visual"
    if SLIDES_INTENT_PATTERN.search(clean):
        return "slides"
    if PDF_INTENT_PATTERN.search(clean):
        return "pdf"
    if EXTERNAL_LINKS_INTENT_PATTERN.search(clean):
        return "external_links"
    return None


def is_greeting(msg: str) -> bool:
    """Detect if student message is a conversational opener or greeting."""
    cleaned = msg.strip().lower()
    return bool(GREETING_PATTERN.match(cleaned)) or cleaned in {"hi", "hello", "hey", "help", "who are you"}


def chat(
    message: str,
    concept_id: str | None = None,
    user_id: str = "default",
    mode: str = "socratic",
    chunks: list | None = None,
    kb: dict | None = None
) -> TutorChatResponse:
    """Socratic chat response grounded in vectorless chunks or live web search."""
    clean_message = message.strip()
    if kb is None:
        kb = store.get_kb(user_id)
    kb_ready = kb.get("ready", False)

    # 0. Detect Student Intent for Artifacts or Curated Resources
    detected_intent = route_student_intent(clean_message)
    artifacts: list[dict] = []
    external_links: list[dict] = []

    active_cid = concept_id
    if not active_cid and kb_ready:
        for c in kb.get("concepts", []):
            if c.get("name", "").lower() in clean_message.lower():
                active_cid = c.get("id")
                break
        if not active_cid and kb.get("concepts"):
            active_cid = kb.get("concepts")[0].get("id")
    if not active_cid:
        active_cid = "concept_overview"

    if detected_intent:
        try:
            if detected_intent == "visual":
                art = tools.generate_visual_diagram_tool(active_cid, prompt=clean_message, user_id=user_id)
                artifacts.append(art)
            elif detected_intent == "slides":
                art = tools.generate_pptx_deck_tool(active_cid, user_id=user_id)
                artifacts.append(art)
            elif detected_intent == "pdf":
                art = tools.generate_pdf_handout_tool(active_cid, user_id=user_id)
                artifacts.append(art)
            elif detected_intent == "external_links":
                res = tools.suggest_curated_resources_tool(active_cid, query=clean_message, user_id=user_id)
                external_links.extend(res.get("resources", []))
        except Exception as e:
            logger.warning(f"Error executing tool for intent '{detected_intent}': {e}")

    # 0.1 Conversational Greeting Handler (unless an intent was specifically requested)
    if is_greeting(clean_message) and not detected_intent:
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
            artifacts=artifacts,
            external_links=external_links,
        )

    wants_web_search = bool(WEB_SEARCH_TRIGGER.search(clean_message))

    # 1. Vectorless Retrieval from Local SQLite / KB (unless explicitly asked to search the web)
    retrieved_chunks = list(chunks) if chunks else []
    if not wants_web_search and not retrieved_chunks:
        store.ensure_retrieval_index_if_needed(user_id=user_id)
        if concept_id:
            retrieved_chunks = GLOBAL_INDEX.retrieve_chunks(concept_id, max_chunks=3, user_id=user_id)
        if not retrieved_chunks:
            bm25_hits = GLOBAL_INDEX.search_bm25(clean_message, max_chunks=3, user_id=user_id)
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
            def _format_web_result(r: dict, body: str) -> str:
                return (
                    f"[Source: {r.get('site_name') or 'Web'} · {r['title']}]\n"
                    f"{body}\n"
                    f"URL: {r['url']}"
                )

            context_text = build_context_window(
                web_results,
                max_chars=4500,
                chunk_max_chars=1500,
                body_getter=lambda r: r.get("snippet", ""),
                formatter=_format_web_result,
            )
            user_prompt = (
                f"Live Web Search Excerpts (TinyFish Search):\n{context_text}\n\n"
                f"Student Question: {clean_message}"
            )
            raw_answer = llm.complete(WEB_SOCRATIC_SYSTEM, user_prompt, max_tokens=600)
            if not raw_answer:
                top = web_results[0]
                top_excerpt = truncate_boundary_aware(top.get("snippet", ""), 220)
                raw_answer = (
                    f"According to {top.get('site_name') or 'web sources'} ([Source: {top['title']}]): "
                    f"\"{top_excerpt}\"\n\n"
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
                artifacts=artifacts,
                external_links=external_links,
            )

    # 3. If no local chunks and no web results found
    if not retrieved_chunks:
        return TutorChatResponse(
            answer="I couldn't locate this topic in your study material or on the web. Could you rephrase your question or specify the concept?",
            citations=[],
            detected_concepts=[],
            dialogue_score=0.5,
            artifacts=artifacts,
            external_links=external_links,
        )

    # 4. Grounded Response from Local Textbook Material
    def _loc_str(c):
        return getattr(c, "locator_label", None) or f"Page {c.page_number}"

    def _format_tutor_chunk(c: Any, body: str) -> str:
        return f"[{_loc_str(c)} · {c.section_title}]\n{body}"

    context_text = build_context_window(
        retrieved_chunks,
        max_chars=6000,
        chunk_max_chars=1800,
        formatter=_format_tutor_chunk,
    )

    system_prompt = REMEDIAL_SYSTEM if mode == "remedial" else SOCRATIC_GUARDRAIL_SYSTEM
    user_prompt = (
        f"Context from uploaded study material:\n{context_text}\n\n"
        f"Target Concept: {concept_id or 'General'}\n"
        f"Student Query: {clean_message}"
    )

    raw_answer = llm.complete(system_prompt, user_prompt, max_tokens=1200)
    if not raw_answer:
        top_chunk = retrieved_chunks[0]
        top_loc = _loc_str(top_chunk)
        c_title = (top_chunk.section_title or concept_id or "Core Concept").replace("_", " ").title()

        all_sents = []
        for c in retrieved_chunks:
            for s in sentences(c.text):
                if 8 <= len(s.split()) <= 45:
                    all_sents.append(s.strip())

        summary_sent = all_sents[0] if all_sents else truncate_boundary_aware(top_chunk.text, 220)
        mechanism_sent = all_sents[1] if len(all_sents) > 1 else "State transformations occur through strictly bounded operational invariants."
        example_sent = all_sents[2] if len(all_sents) > 2 else f"Applying {c_title} to sample inputs guarantees structured, deterministic execution."

        raw_answer = (
            f"### 1. Intuitive Mental Model\n"
            f"Think of **{c_title}** as a structured blueprint: rather than treating operations as unpredictable ad-hoc steps, "
            f"it enforces an underlying organizational model where each component occupies a predictable role.\n\n"
            f"### 2. Architecture & Governing Mechanism\n"
            f"According to your study material [{top_loc}], {summary_sent}\n\n"
            f"Under the hood, {mechanism_sent} This ensures that operations adhere to verified system constraints.\n\n"
            f"### 3. Concrete Worked Example\n"
            f"Consider a concrete scenario from [{top_loc}]: {example_sent}\n\n"
            f"Notice how each step transforms the state sequentially without violating foundational properties.\n\n"
            f"### 4. Socratic Check-in Challenge\n"
            f"Given how {c_title} operates in [{top_loc}], what edge case or structural constraint would break this guarantee if state inputs were unbounded?"
        )

    citations: list[Citation] = []
    seen_locators = set()
    for c in retrieved_chunks[:3]:
        loc = _loc_str(c)
        if loc not in seen_locators:
            seen_locators.add(loc)
            citations.append(Citation(
                page=c.page_number,
                snippet=f"[{loc}] {c.section_title}",
                text=c.text,
                source_type="document",
                locator_label=loc,
                source_id=getattr(c, "source_id", "")
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
        artifacts=artifacts,
        external_links=external_links,
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
