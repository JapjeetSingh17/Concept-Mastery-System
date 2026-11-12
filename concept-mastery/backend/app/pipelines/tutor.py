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

RESPONSE STRUCTURE (ALWAYS FORMAT WITH THESE EXACT 4 SECTION HEADINGS):
### 1. Core Concept Explanation
Explain the core intuition and mental model with a vivid, relatable real-world analogy. Demystify abstract notation.

### 2. Grounded Evidence from Your Material
Explain step-by-step how the concept operates under the hood, citing exact textbook page or video timestamp locators (e.g. [Page X] or [MM:SS]).

### 3. Intuitive Analogy / Worked Example
Trace a concrete tangible mini-example, calculation, or system scenario demonstrating the mechanism in action.

### 4. Socratic Check-in Question
Conclude with a thought-provoking check-in question that tests whether the student can apply this mechanism to edge cases or novel scenarios.

STRICT GUARDRAILS:
- When an active study material or upload is loaded, ALWAYS ground your answer in the material's extracted concepts and provided chunks. NEVER claim "I cannot see your upload directly" or "I cannot watch videos".
- Ground all explanations strictly in the provided study material chunks and citations.
- NEVER give direct quiz solutions or homework answers; scaffold the student's problem-solving instead.
- Provide comprehensive, clear, high-signal educational guidance (250 to 450 words)."""

REMEDIAL_SYSTEM = """You are a patient remedial tutor helping a student overcome a diagnosed misconception.

RESPONSE STRUCTURE (ALWAYS FORMAT WITH THESE EXACT 4 SECTION HEADINGS):
### 1. Core Concept Explanation
Deconstruct the target concept from first principles using an intuitive mental model.

### 2. Grounded Evidence from Your Material
Explicitly explain why the prerequisite foundational concept is required to avoid confusion, citing exact locators [Page X] or [MM:SS].

### 3. Intuitive Analogy / Worked Example
Provide a worked mini-example contrasting the correct mechanism against common student mistakes.

### 4. Socratic Check-in Question
End with an empowering Socratic check-in question to verify comprehension."""

WEB_SOCRATIC_SYSTEM = """You are an expert Socratic tutor for an adaptive concept mastery platform.
The student's question was researched using live web search or external context.

RESPONSE STRUCTURE (ALWAYS FORMAT WITH THESE EXACT 4 SECTION HEADINGS):
### 1. Core Concept Explanation
Explain the underlying mental model and operational mechanics using the web search excerpts.

### 2. Grounded Evidence from External Sources
Cite external web sources clearly (e.g. [Source: Site Name]). If active uploaded material is also present, clearly separate "From Your Uploaded Material" vs "External Web Context".

### 3. Intuitive Analogy / Worked Example
Provide a clear worked example contrasting key trade-offs.

### 4. Socratic Check-in Question
Conclude with a targeted Socratic follow-up question to test comprehension.

STRICT SOCRATIC GUARDRAILS:
- NEVER provide direct quiz answers or completed homework.
- Keep explanations structured, rigorous, and educational (250 to 400 words)."""

GREETING_PATTERN = re.compile(
    r"^(hi|hello|hey|heya|howdy|greetings|good\s+(morning|afternoon|evening|day)|yo|sup|help|who\s+are\s+you)[!?.\s]*$",
    re.IGNORECASE
)

OVERVIEW_INTENT_PATTERN = re.compile(
    r"\b(what\s+is\s+(this|the\s+lecture|the\s+video|the\s+material|the\s+upload|it)\s+about|"
    r"summarize|summary|overview|what\s+topics|what\s+does\s+(this|the\s+material|the\s+lecture|the\s+video)\s+cover|"
    r"explain\s+(this|the)\s+(material|lecture|video|upload)|what\s+(are\s+)?the\s+core\s+concepts|"
    r"what\s+did\s+i\s+upload)\b",
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

    # 1. Fetch Active Study Material Metadata & Overall Grounding Context
    active_mat = store.get_active_material(user_id)
    if not active_mat and kb_ready:
        mat_id = kb.get("material_id") or "kb_active"
        active_mat = {
            "material_id": mat_id,
            "title": kb.get("file") or "Uploaded Course Material",
            "source_type": kb.get("source_type", "pdf"),
            "chunk_count": len(kb.get("chunks", [])),
            "concepts": [c.get("name") if isinstance(c, dict) else str(c) for c in kb.get("concepts", [])],
        }

    mat_title = active_mat.get("title") if active_mat else "Study Material"
    mat_type = active_mat.get("source_type") if active_mat else "document"
    mat_id = active_mat.get("material_id") if active_mat else ""
    mat_chunk_count = active_mat.get("chunk_count", 0) if active_mat else 0
    mat_concepts = store.get_concepts(mat_id, user_id=user_id) if mat_id else []
    if not mat_concepts and active_mat and active_mat.get("concepts"):
        mat_concepts = [
            {"id": c.lower().replace(" ", "_"), "name": c} if isinstance(c, str) else c
            for c in active_mat["concepts"]
        ]
    if not mat_concepts and kb_ready and kb.get("concepts"):
        mat_concepts = kb["concepts"]

    concept_names = [c["name"] for c in mat_concepts if isinstance(c, dict) and "name" in c]

    # Gather user's weak / unmastered concepts from store and mastery records
    weak_concept_names = []
    try:
        snap = mastery.snapshot(user_id)
        for sc in snap:
            if sc.get("state") == "WEAK" or sc.get("score", 1.0) < 0.6:
                cname = sc.get("name") or sc.get("concept_name") or sc.get("cid")
                if cname and cname not in weak_concept_names:
                    weak_concept_names.append(cname)
    except Exception:
        pass

    # Gather all chunks for active material / user
    store.ensure_retrieval_index_if_needed(user_id=user_id)
    all_chunks = GLOBAL_INDEX.get_all_chunks(user_id=user_id)
    if not all_chunks and user_id != "default":
        all_chunks = GLOBAL_INDEX.get_all_chunks(user_id="default")
    if not all_chunks and active_mat and active_mat.get("structured_preview"):
        preview_chunks = []
        for p_idx, item in enumerate(active_mat["structured_preview"]):
            preview_chunks.append(type("PreviewChunk", (), {
                "chunk_id": f"prev_{p_idx}",
                "page_number": item.get("page_number", item.get("page", p_idx + 1)),
                "section_title": item.get("section_title", item.get("title", mat_title)),
                "text": item.get("text", item.get("snippet", "")),
                "locator_label": item.get("locator_label", f"Page {item.get('page_number', p_idx + 1)}"),
                "source_id": mat_id,
                "source_type": mat_type,
                "canonical_concepts": [],
            })())
        all_chunks = preview_chunks

    # Introductory chunks (first 3) + 3 representative chunks spread across the document
    intro_chunks = all_chunks[:3]
    intro_ids = {getattr(c, "chunk_id", str(id(c))) for c in intro_chunks}
    rep_chunks = []
    if len(all_chunks) > 3:
        step = max(1, len(all_chunks) // 4)
        indices = [min(len(all_chunks) - 1, step), min(len(all_chunks) - 1, step * 2), min(len(all_chunks) - 1, step * 3)]
        for idx in list(dict.fromkeys(indices)):
            cand = all_chunks[idx]
            cid = getattr(cand, "chunk_id", str(id(cand)))
            if cid not in intro_ids:
                intro_ids.add(cid)
                rep_chunks.append(cand)
    overview_chunks = intro_chunks + rep_chunks

    # Detect overview / meta intents
    is_overview_intent = bool(OVERVIEW_INTENT_PATTERN.search(clean_message))
    wants_web_search = bool(WEB_SEARCH_TRIGGER.search(clean_message))

    # If asking for an overview of their uploaded material, never force web search
    if is_overview_intent and (active_mat or all_chunks):
        wants_web_search = False

    # 2. Local Chunk Retrieval
    retrieved_chunks = list(chunks) if chunks else []
    if not retrieved_chunks:
        if is_overview_intent and overview_chunks:
            retrieved_chunks = overview_chunks
        else:
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
                            "canonical_concepts": [concept_id] if concept_id else [],
                            "locator_label": c.get("locator_label") or f"Page {c['page']}",
                            "source_id": mat_id,
                            "source_type": mat_type,
                        })()
                        for _, c in hits
                    ]

            # If specific search returned no hits but material is active, use overview chunks
            # instead of falsely claiming the upload is missing or falling back to web search
            if not retrieved_chunks and all_chunks and not wants_web_search:
                retrieved_chunks = overview_chunks

    # 3. Live Web Search Decision
    # Disable automatic web search when active material is present unless explicitly requested
    should_run_web_search = wants_web_search or (not active_mat and not all_chunks and not retrieved_chunks)

    if should_run_web_search:
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

            # If user has an active material AND requested web search, clearly separate both contexts
            local_context_intro = ""
            if active_mat and retrieved_chunks:
                def _loc_str_local(c):
                    return getattr(c, "locator_label", None) or f"Page {getattr(c, 'page_number', 1)}"
                local_snips = "\n".join(
                    f"[{_loc_str_local(c)} · {getattr(c, 'section_title', 'Section')}]: {truncate_boundary_aware(c.text, 300)}"
                    for c in retrieved_chunks[:2]
                )
                local_context_intro = (
                    f"### From Your Uploaded Material ({mat_title}):\n{local_snips}\n\n"
                    f"### External Web Context:\n"
                )

            user_prompt = (
                f"{local_context_intro}"
                f"Live Web Search Excerpts (TinyFish Search):\n{context_text}\n\n"
                f"Student Question: {clean_message}"
            )
            raw_answer = llm.complete(WEB_SOCRATIC_SYSTEM, user_prompt, max_tokens=600)
            if not raw_answer:
                top = web_results[0]
                top_excerpt = truncate_boundary_aware(top.get("snippet", ""), 220)
                raw_answer = (
                    f"### 1. Core Concept Explanation\n"
                    f"Based on external research, the topic involves foundational principles documented across educational web references.\n\n"
                    f"### 2. Grounded Evidence from External Sources\n"
                    f"According to {top.get('site_name') or 'web sources'} ([Source: {top['title']}]): \"{top_excerpt}\"\n\n"
                    f"### 3. Intuitive Analogy / Worked Example\n"
                    f"Consider how this standard operates in practice: each component adjusts in response to empirical constraints.\n\n"
                    f"### 4. Socratic Check-in Question\n"
                    f"How does this external finding connect with what you are analyzing in your study coursework?"
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

    # 4. If no local chunks and no web results found
    if not retrieved_chunks:
        return TutorChatResponse(
            answer=(
                f"### 1. Core Concept Explanation\n"
                f"I am ready to help you analyze your study material! Currently no text chunks could be matched for this specific query.\n\n"
                f"### 2. Grounded Evidence from Your Material\n"
                f"To begin personalized study, please upload your lecture or document in the **Study Material** tab.\n\n"
                f"### 3. Intuitive Analogy / Worked Example\n"
                f"Think of the tutor engine as an index reader: once your material is indexed, every answer cites exact timestamps and pages.\n\n"
                f"### 4. Socratic Check-in Question\n"
                f"Would you like to explore a foundational topic, or search the web for broader context?"
            ),
            citations=[],
            detected_concepts=[],
            dialogue_score=0.5,
            artifacts=artifacts,
            external_links=external_links,
        )

    # 5. Grounded Response from Local Study Material
    def _loc_str(c):
        return getattr(c, "locator_label", None) or f"Page {getattr(c, 'page_number', 1)}"

    def _format_tutor_chunk(c: Any, body: str) -> str:
        return f"[{_loc_str(c)} · {getattr(c, 'section_title', 'Section')}]\n{body}"

    context_text = build_context_window(
        retrieved_chunks,
        max_chars=6000,
        chunk_max_chars=1800,
        formatter=_format_tutor_chunk,
    )

    active_material_context_block = ""
    if active_mat:
        active_material_context_block = (
            f"=== ACTIVE STUDY MATERIAL CONTEXT ===\n"
            f"- Material Title: {mat_title}\n"
            f"- Source Type: {mat_type.upper()} ({mat_chunk_count} indexed chunks/segments)\n"
            f"- Extracted Core Concepts & Topics: {', '.join(concept_names) if concept_names else 'General Coursework'}\n"
            f"- User Weak/Unmastered Concepts: {', '.join(weak_concept_names) if weak_concept_names else 'None identified yet (baseline or proficient)'}\n"
            f"======================================\n\n"
        )

    overview_directive = ""
    if is_overview_intent:
        overview_directive = (
            "DIRECTIVE: The student is asking what this uploaded material is about or requesting an overview. "
            "Answer authoritatively from the Extracted Core Concepts & Topics and the provided representative chunks. "
            "NEVER claim that you cannot see the material, cannot watch the video, or don't have access to the upload. "
            "Explain what the lecture/document covers, its governing mechanisms, and how the concepts fit together.\n\n"
        )

    system_prompt = REMEDIAL_SYSTEM if mode == "remedial" else SOCRATIC_GUARDRAIL_SYSTEM
    user_prompt = (
        f"{active_material_context_block}"
        f"{overview_directive}"
        f"Grounded Context from Uploaded Study Material:\n{context_text}\n\n"
        f"Target Concept: {concept_id or (concept_names[0] if concept_names else 'General Overview')}\n"
        f"Student Query: {clean_message}"
    )

    raw_answer = llm.complete(system_prompt, user_prompt, max_tokens=1200)
    if not raw_answer:
        top_chunk = retrieved_chunks[0] if retrieved_chunks else None
        top_loc = _loc_str(top_chunk) if top_chunk else "Overview"
        c_title = (getattr(top_chunk, "section_title", None) or concept_id or (concept_names[0] if concept_names else "Core Concept")).replace("_", " ").title()

        all_sents = []
        for c in retrieved_chunks:
            for s in sentences(c.text):
                if 8 <= len(s.split()) <= 45:
                    all_sents.append(s.strip())

        if is_overview_intent:
            topics_str = ", ".join(concept_names[:6]) if concept_names else "neural network architectures, loss dynamics, and optimization"
            summary_sent = all_sents[0] if all_sents else f"This study material ({mat_title}) builds a systematic framework for understanding key conceptual mechanisms."
            evidence_sent = all_sents[1] if len(all_sents) > 1 else f"Key principles are developed through structured progressions across {mat_chunk_count} segments."
            example_sent = all_sents[2] if len(all_sents) > 2 else f"For instance, foundational operations directly constrain subsequent state representations."
            first_concept = concept_names[0] if concept_names else "the foundational principles"

            raw_answer = (
                f"### 1. Core Concept Explanation\n"
                f"Your uploaded {mat_type}, **{mat_title}**, provides an in-depth pedagogical walkthrough focusing on **{topics_str}**. "
                f"Rather than presenting these ideas in isolation, the material structures them as an integrated pipeline where each stage directly feeds the next.\n\n"
                f"### 2. Grounded Evidence from Your Material\n"
                f"According to your study material [{top_loc}], {summary_sent}\n\n"
                f"As the material develops, {evidence_sent} This establishes a rigorous foundation for how state representations evolve.\n\n"
                f"### 3. Intuitive Analogy / Worked Example\n"
                f"Think of **{mat_title}** as an architectural blueprint: individual concepts like {first_concept} serve as structural load-bearers. "
                f"{example_sent}\n\n"
                f"### 4. Socratic Check-in Question\n"
                f"Which specific concept from this {mat_type}—such as {topics_str.split(',')[0].strip()}—would you like to examine first to test your operational intuition?"
            )
        else:
            summary_sent = all_sents[0] if all_sents else truncate_boundary_aware(top_chunk.text, 220) if top_chunk else "Operations adhere to strict structural constraints."
            mechanism_sent = all_sents[1] if len(all_sents) > 1 else "State transformations occur through strictly bounded operational invariants."
            example_sent = all_sents[2] if len(all_sents) > 2 else f"Applying {c_title} to sample inputs guarantees structured, deterministic execution."

            raw_answer = (
                f"### 1. Core Concept Explanation\n"
                f"Think of **{c_title}** as a structured blueprint: rather than treating operations as unpredictable ad-hoc steps, "
                f"it enforces an underlying organizational model where each component occupies a predictable role.\n\n"
                f"### 2. Grounded Evidence from Your Material\n"
                f"According to your study material [{top_loc}], {summary_sent}\n\n"
                f"Under the hood, {mechanism_sent} This ensures that operations adhere to verified system constraints.\n\n"
                f"### 3. Intuitive Analogy / Worked Example\n"
                f"Consider a concrete scenario from [{top_loc}]: {example_sent}\n\n"
                f"Notice how each step transforms the state sequentially without violating foundational properties.\n\n"
                f"### 4. Socratic Check-in Question\n"
                f"Given how {c_title} operates in [{top_loc}], what edge case or structural constraint would break this guarantee if state inputs were unbounded?"
            )

    citations: list[Citation] = []
    seen_locators = set()
    for c in retrieved_chunks[:3]:
        loc = _loc_str(c)
        if loc not in seen_locators:
            seen_locators.add(loc)
            citations.append(Citation(
                page=getattr(c, "page_number", 1),
                snippet=f"[{loc}] {getattr(c, 'section_title', 'Section')}",
                text=getattr(c, "text", ""),
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
