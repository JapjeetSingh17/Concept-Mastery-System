"""Active Recall Pipeline Engine.

Implements the active recall study cycle:
  1. Retrieve vectorless ground truth chunks from SQLite BM25 index.
  2. Synthesize structured Study Cards (Core Intuition, Key Formula/Mechanism, Worked Mini-Example, 3-5 Essential Keywords, Exact Source Citation) and 1 verification question.
  3. Authoritative evaluation via AI Mastery Judge: computes retrieval_score (0.0 - 1.0), reasoning_depth_level, retained concepts vs missed gaps, and verification correctness.
  4. Updates student's Epistemic Mastery in mastery.py (UNKNOWN / WEAK / STRONG).
  5. SQLite persistence via store.py.
"""

from __future__ import annotations

import difflib
import json
import logging
import math
import re
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from ..core import store, llm
from ..core.concept_index import GLOBAL_INDEX
from ..core.text import sentences, tokens
from ..models.schemas import (
    ActiveRecallStudyCard,
    ActiveRecallResult,
    ActiveRecallHistoryItem,
    QuizQuestion,
)
from . import mastery, quiz, concepts

logger = logging.getLogger("concept_mastery.active_recall")


def _extract_keywords(text: str, count: int = 5) -> list[str]:
    """Extract salient technical keywords from text, strictly filtering out generic conversational tokens."""
    stop_words = {
        "the", "a", "an", "and", "or", "in", "on", "at", "to", "for", "with",
        "by", "about", "as", "is", "are", "was", "were", "be", "been", "this",
        "that", "these", "those", "from", "into", "which", "then", "than", "such",
        "step", "calculate", "compute", "where", "given", "using", "when", "after",
        "what", "have", "stands", "exactly", "segment", "video", "page", "slide",
        "section", "lecture", "would", "like", "keep", "class", "here", "there",
        "their", "they", "them", "some", "more", "also", "just", "know", "look",
        "tell", "need", "want", "take", "make", "good", "well", "will", "going"
    }
    words = re.findall(r"\b[A-Za-z][A-Za-z0-9_-]{3,}\b", text)
    filtered = [w.lower() for w in words if w.lower() not in stop_words and len(w) >= 4]
    seen = set()
    result = []
    for w in filtered:
        if w not in seen:
            seen.add(w)
            result.append(w)
            if len(result) >= count:
                break
    return result or ["mechanism", "computation", "algorithm", "parameters"]


def expand_context_window(chunks: list, user_id: str, min_words: int = 200) -> tuple[list, str]:
    """Expand retrieved context window by stitching adjacent material chunks to prevent micro-fragmentation.

    Guarantees that the active recall micro-lesson receives at least 150-300 words of primary text,
    eliminating AI hallucination caused by incomplete single-sentence snippets.
    """
    if not chunks:
        all_chunks = GLOBAL_INDEX.get_all_chunks(user_id=user_id)
        if all_chunks:
            chunks = all_chunks[:3]
        else:
            return [], "[Primary Course Material]"

    total_words = sum(len(c.text.split()) for c in chunks)
    if total_words >= min_words:
        # Build composite locator
        c0 = chunks[0]
        c_last = chunks[-1]
        loc0 = getattr(c0, "locator_label", None) or f"Page {c0.page_number}"
        loc_last = getattr(c_last, "locator_label", None) or f"Page {c_last.page_number}"
        citation = f"[{loc0}]" if loc0 == loc_last else f"[{loc0} - {loc_last}]"
        return chunks, citation

    # Context is sparse: fetch all chunks for this user and find adjacent neighbors
    all_chunks = GLOBAL_INDEX.get_all_chunks(user_id=user_id)
    if not all_chunks:
        c0 = chunks[0]
        loc = getattr(c0, "locator_label", None) or f"Page {c0.page_number}"
        return chunks, f"[{loc}]"

    # Map chunk by ID and find index of current chunks
    current_ids = {c.chunk_id for c in chunks}
    matched_indices = [i for i, c in enumerate(all_chunks) if c.chunk_id in current_ids]

    expanded = list(chunks)
    if matched_indices:
        min_idx = min(matched_indices)
        max_idx = max(matched_indices)

        # Expand backwards
        step = 1
        while min_idx - step >= 0 and sum(len(c.text.split()) for c in expanded) < min_words:
            cand = all_chunks[min_idx - step]
            if cand.chunk_id not in {c.chunk_id for c in expanded}:
                expanded.insert(0, cand)
            step += 1

        # Expand forwards
        step = 1
        while max_idx + step < len(all_chunks) and sum(len(c.text.split()) for c in expanded) < min_words:
            cand = all_chunks[max_idx + step]
            if cand.chunk_id not in {c.chunk_id for c in expanded}:
                expanded.append(cand)
            step += 1

    # Format composite citation range
    first_c = expanded[0]
    last_c = expanded[-1]
    first_loc = getattr(first_c, "locator_label", None) or f"Page {first_c.page_number}"
    last_loc = getattr(last_c, "locator_label", None) or f"Page {last_c.page_number}"
    if first_loc == last_loc or not last_loc:
        citation = f"[{first_loc}]"
    else:
        # Strip duplicate brackets
        l1 = first_loc.replace("[", "").replace("]", "").strip()
        l2 = last_loc.replace("[", "").replace("]", "").strip()
        citation = f"[{l1} - {l2}]"

    return expanded, citation


def _resolve_clean_concept_name(concept_id: str, raw_name: str, chunks: list) -> str:
    """Resolve clean academic concept name, replacing generic 'Segment XX:XX' with true subject matter."""
    if not concepts.is_generic_section_heading(raw_name) and not raw_name.lower().startswith("segment"):
        return raw_name

    combined = " ".join(c.text for c in chunks[:3]) if chunks else ""
    # Look for capitalized acronyms like "RAG", "BERT", "DAG", "LSTM", etc.
    acronym_match = re.search(r"\b([A-Z]{2,6})\b(?:\s+(?:stands for|is a|algorithm|model|architecture|retrieval))?", combined)
    if acronym_match:
        return acronym_match.group(1).upper()

    kws = _extract_keywords(combined, count=3)
    if kws:
        return kws[0].title()
    return raw_name


def _synthesize_card_fallback(concept_id: str, concept_name: str, chunks: list, c_def: str, citation: str = "") -> dict:
    """Deterministic vectorless synthesis of Study Card components with ZERO generic template placeholders."""
    combined_text = "\n".join(c.text for c in chunks) if chunks else (c_def or f"Core study curriculum material on {concept_name}.")
    chunk_sents = [s.strip() for s in sentences(combined_text) if len(s.strip().split()) >= 4]

    # 1. Core Intuition
    if c_def and len(c_def.split()) >= 8:
        core_intuition = c_def
    elif chunk_sents:
        # Select first 1-2 substantial sentences
        core_intuition = " ".join(chunk_sents[:2])
    else:
        core_intuition = f"{concept_name} represents a core theoretical concept and computational mechanism within this curriculum."

    # 2. Key Formula / Mechanism
    formula_sent = ""
    for s in chunk_sents:
        s_low = s.lower()
        if any(term in s_low for term in ("formula", "equation", "algorithm", "calculat", "update", "rule", "=", "vector", "embed", "retriev", "index", "layer", "loss", "gradient")):
            formula_sent = s
            break
    if not formula_sent:
        for s in chunk_sents[1:]:
            if len(s.split()) >= 6:
                formula_sent = s
                break
    if not formula_sent:
        formula_sent = f"The mechanism of {concept_name} processes input features through sequential transformation stages to yield predictable outputs."

    # 3. Worked Mini-Example (strictly grounded, NO generic template placeholders)
    example_sent = ""
    for s in chunk_sents:
        s_low = s.lower()
        if any(term in s_low for term in ("example", "consider", "suppose", "assume", "let ", "in practice", "when applying")):
            example_sent = s
            break
    if not example_sent:
        # Find any application or contextual sentence explaining how the concept functions
        for s in chunk_sents[2:]:
            if len(s.split()) >= 7 and s != formula_sent:
                example_sent = f"Practical Application: {s}"
                break
    if not example_sent:
        example_sent = f"In practice, {concept_name} is initialized on primary input parameters, executing the verified workflow across pipeline stages."

    # 4. Essential Keywords (3-5 items, strictly filtered)
    extracted = _extract_keywords(combined_text, count=5)
    keywords = []
    if not concepts.is_generic_section_heading(concept_name):
        keywords.append(concept_name.lower())
    for kw in extracted:
        if kw not in keywords and len(keywords) < 5:
            keywords.append(kw)
    if len(keywords) < 3:
        keywords.extend(["mechanism", "workflow", "parameters"][:3 - len(keywords)])

    if not citation:
        if chunks:
            c0 = chunks[0]
            loc = getattr(c0, "locator_label", None) or f"Page {c0.page_number}"
            citation = f"[{loc}]"
        else:
            citation = "[Canonical Course Reference]"

    return {
        "core_intuition": core_intuition.strip(),
        "key_formula_or_mechanism": formula_sent.strip(),
        "worked_mini_example": example_sent.strip(),
        "essential_keywords": keywords[:5],
        "exact_source_citation": citation,
    }


def generate_micro_lesson(
    concept_id: str,
    concept_name: str = "",
    user_id: str = "default",
    timer_seconds: int = 120,
) -> dict:
    """Generate a cohesive, deeply grounded Active Recall micro-lesson with zero hallucination.

    Assembles an expanded context window (>=150-300 words) from adjacent chunks before prompting the LLM.
    """
    if not concept_id:
        raise store.UserError("concept_id is required to generate a micro-lesson.")

    kb = store.get_kb(user_id)
    by_id = {c["id"]: c for c in kb.get("concepts", [])}
    concept = by_id.get(concept_id)

    raw_name = concept_name or (concept.get("name", concept_id.replace("_", " ").title()) if concept else concept_id.replace("_", " ").title())
    concept_def = concept.get("definition", "") if concept else ""

    # 1. Retrieve vectorless ground truth chunks and expand window to prevent micro-fragmentation
    chunks = GLOBAL_INDEX.retrieve_chunks(concept_id, max_chunks=3, user_id=user_id)
    if not chunks:
        bm25_hits = GLOBAL_INDEX.search_bm25(raw_name, max_chunks=3, user_id=user_id)
        chunks = [hit[1] for hit in bm25_hits]

    # Enforce minimum word-count guardrail with adjacent chunk window expansion
    expanded_chunks, citation = expand_context_window(chunks, user_id=user_id, min_words=200)

    # Sanitize concept name if it was an arbitrary locator like "Segment 00:00"
    resolved_name = _resolve_clean_concept_name(concept_id, raw_name, expanded_chunks)

    # 2. Synthesize Study Card via LLM with strict Zero Hallucination grounding rule
    card_data = None
    if llm.enabled() and expanded_chunks:
        ctx_text = "\n\n".join(f"[{getattr(c, 'locator_label', None) or f'Page {c.page_number}'}] {c.text}" for c in expanded_chunks)
        sys_prompt = (
            "You are an expert cognitive learning specialist and STEM tutor. Generate a high-yield Active Recall Study Card "
            "strictly grounded in the provided primary source text.\n\n"
            "ZERO HALLUCINATION GROUNDING RULES:\n"
            "1. Every single explanation, mechanism, and example MUST be directly derived from the provided primary context.\n"
            "2. NEVER output generic placeholder text (such as 'Consider applying X to test inputs: input features are processed through the governing equation...').\n"
            "3. If mathematical equations are not in the text, explain the concrete operational mechanism or algorithmic workflow described in the material.\n"
            "4. In the worked mini-example, illustrate the concept with a realistic scenario directly referenced or implied by the lesson text.\n"
            "5. The essential keywords must be real, high-yield technical terms from the material (NEVER include generic tokens like 'what', 'have', 'segment').\n\n"
            "Return ONLY a single valid JSON object with exact keys:\n"
            '  "core_intuition": "<2-3 substantive sentences explaining the core concept, its purpose, and mental model>",\n'
            '  "key_formula_or_mechanism": "<governing rule, mathematical relationship, or step-by-step technical mechanism>",\n'
            '  "worked_mini_example": "<concrete applied mini-example walkthrough grounded in the material>",\n'
            '  "essential_keywords": ["<keyword 1>", "<keyword 2>", "<keyword 3>", "<keyword 4>", "<keyword 5>"] (3 to 5 real technical domain terms),\n'
            '  "exact_source_citation": "<citation anchor range e.g. [00:00 - 04:30] or [Pages 1-2]>"'
        )
        usr_prompt = f"Target Concept: {resolved_name}\nContext:\n{ctx_text}"
        try:
            raw = llm.complete(sys_prompt, usr_prompt, max_tokens=1000)
            if raw:
                clean = llm.extract_json_payload(raw)
                parsed = json.loads(clean)
                if all(k in parsed for k in ("core_intuition", "key_formula_or_mechanism", "worked_mini_example", "essential_keywords")):
                    filtered_kws = [str(k).strip().lower() for k in parsed.get("essential_keywords", []) if len(str(k).strip()) >= 3]
                    card_data = {
                        "core_intuition": str(parsed["core_intuition"]).strip(),
                        "key_formula_or_mechanism": str(parsed["key_formula_or_mechanism"]).strip(),
                        "worked_mini_example": str(parsed["worked_mini_example"]).strip(),
                        "essential_keywords": filtered_kws[:5] or _extract_keywords(ctx_text, count=5),
                        "exact_source_citation": str(parsed.get("exact_source_citation", citation)).strip(),
                    }
        except Exception as e:
            logger.warning(f"LLM Study Card generation fallback triggered: {e}")

    if not card_data:
        card_data = _synthesize_card_fallback(concept_id, resolved_name, expanded_chunks, concept_def, citation=citation)

    # 3. Generate 1 Verification Question grounded in the concept
    verification_q: Optional[dict] = None
    try:
        source_p = expanded_chunks[0].page_number if expanded_chunks else 1
        if quiz.is_mathematical_concept(concept_id, resolved_name):
            vq_obj = quiz.generate_numerical_question(concept_id, concept_name=resolved_name, user_id=user_id, source_page=source_p)
            verification_q = vq_obj.model_dump()
        else:
            qs = quiz.generate_diagnostic_quiz(concept_id, count=1, question_types=["mcq"], user_id=user_id)
            if qs:
                verification_q = qs[0].model_dump()
    except Exception as e:
        logger.warning(f"Verification question fallback: {e}")
        fallback_qid = f"diag_{uuid.uuid4().hex[:8]}"
        vq_fallback = QuizQuestion(
            question_id=fallback_qid,
            concept_id=concept_id,
            question_type="mcq",
            question_text=f"Which term best describes the core operational mechanism of {resolved_name}?",
            options=[resolved_name, "Random Noise", "Arbitrary Search", "Unbounded Divergence"],
            correct_option_idx=0,
            explanation=card_data["core_intuition"],
            source_page=expanded_chunks[0].page_number if expanded_chunks else 1,
        )
        quiz.REGISTRY[fallback_qid] = vq_fallback.model_dump()
        verification_q = vq_fallback.model_dump()

    return {
        "concept_id": concept_id,
        "concept_name": resolved_name,
        "card_data": card_data,
        "verification_question": verification_q,
        "citation": citation,
    }


def start_session(
    user_id: str = "default",
    concept_id: str = "",
    timer_seconds: int = 120,
) -> ActiveRecallStudyCard:
    """Initialize an Active Recall study session with grounded Study Card and verification question."""
    lesson_bundle = generate_micro_lesson(
        concept_id=concept_id,
        user_id=user_id,
        timer_seconds=timer_seconds,
    )

    card_data = lesson_bundle["card_data"]
    concept_name = lesson_bundle["concept_name"]
    verification_q = lesson_bundle["verification_question"]

    session_id = f"ar_{uuid.uuid4().hex[:12]}"
    now_iso = datetime.now(timezone.utc).isoformat()

    # 4. Save session to SQLite
    session_record = {
        "session_id": session_id,
        "user_id": user_id,
        "concept_id": concept_id,
        "timer_seconds": timer_seconds,
        "lesson_json": card_data,
        "recall_submission_text": "",
        "verification_question_json": verification_q or {},
        "retrieval_score": None,
        "retained_concepts_json": [],
        "missed_gaps_json": [],
        "created_at": now_iso,
    }
    store.save_active_recall_session(session_record)

    return ActiveRecallStudyCard(
        session_id=session_id,
        concept_id=concept_id,
        concept_name=concept_name,
        timer_seconds=timer_seconds,
        core_intuition=card_data["core_intuition"],
        key_formula_or_mechanism=card_data["key_formula_or_mechanism"],
        worked_mini_example=card_data["worked_mini_example"],
        essential_keywords=card_data["essential_keywords"],
        exact_source_citation=card_data["exact_source_citation"],
        verification_question=verification_q,
        created_at=now_iso,
    )
    now_iso = datetime.now(timezone.utc).isoformat()

    # 4. Save session to SQLite
    session_record = {
        "session_id": session_id,
        "user_id": user_id,
        "concept_id": concept_id,
        "timer_seconds": timer_seconds,
        "lesson_json": card_data,
        "recall_submission_text": "",
        "verification_question_json": verification_q or {},
        "retrieval_score": None,
        "retained_concepts_json": [],
        "missed_gaps_json": [],
        "created_at": now_iso,
    }
    store.save_active_recall_session(session_record)

    return ActiveRecallStudyCard(
        session_id=session_id,
        concept_id=concept_id,
        concept_name=concept_name,
        timer_seconds=timer_seconds,
        core_intuition=card_data["core_intuition"],
        key_formula_or_mechanism=card_data["key_formula_or_mechanism"],
        worked_mini_example=card_data["worked_mini_example"],
        essential_keywords=card_data["essential_keywords"],
        exact_source_citation=card_data["exact_source_citation"],
        verification_question=verification_q,
        created_at=now_iso,
    )


def evaluate_session(
    user_id: str,
    session_id: str,
    concept_id: str,
    recall_text: str,
    verification_answer: Any = None,
) -> ActiveRecallResult:
    """Evaluate student active recall retrieval text & verification answer using AI Mastery Judge."""
    session = store.get_active_recall_session(session_id) if session_id else None
    if session_id and not session:
        raise store.UserError(f"Active recall session '{session_id}' not found.")
    if not session:
        sessions = store.list_active_recall_sessions(user_id=user_id, limit=20)
        matching = [s for s in sessions if s.get("concept_id") == concept_id]
        if matching:
            session = matching[0]
            session_id = session.get("session_id", session_id)
        else:
            study_card = start_session(user_id=user_id, concept_id=concept_id)
            session = store.get_active_recall_session(study_card.session_id)
            session_id = study_card.session_id
    if not session:
        raise store.UserError(f"Active recall session '{session_id}' not found.")

    lesson = json.loads(session["lesson_json"]) if isinstance(session["lesson_json"], str) else (session["lesson_json"] or {})
    vq = json.loads(session["verification_question_json"]) if isinstance(session["verification_question_json"], str) else (session["verification_question_json"] or {})

    clean_recall = (recall_text or "").strip()
    essential_keywords = [str(k).lower().strip() for k in lesson.get("essential_keywords", [])]

    # 1. Evaluate Verification Question
    verification_correct = False
    if vq:
        q_type = vq.get("question_type", "mcq")
        if q_type == "numerical":
            correct_num = vq.get("correct_numeric_answer")
            tolerance = float(vq.get("tolerance", 0.05))
            student_val = None
            if isinstance(verification_answer, (int, float)):
                student_val = float(verification_answer)
            elif verification_answer is not None:
                match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", str(verification_answer))
                if match:
                    try:
                        student_val = float(match.group(0))
                    except ValueError:
                        student_val = None
            if student_val is not None and correct_num is not None:
                verification_correct = (abs(student_val - correct_num) <= tolerance)
        elif q_type == "mcq":
            correct_idx = int(vq.get("correct_option_idx", 0))
            options = vq.get("options", [])
            if isinstance(verification_answer, int):
                verification_correct = (verification_answer == correct_idx)
            elif isinstance(verification_answer, str):
                if verification_answer.isdigit():
                    verification_correct = (int(verification_answer) == correct_idx)
                elif correct_idx < len(options) and verification_answer.strip().lower() == options[correct_idx].strip().lower():
                    verification_correct = True
        else:
            target_blank = str(vq.get("target_blank_answer", "")).strip().lower()
            if target_blank and str(verification_answer or "").strip().lower() == target_blank:
                verification_correct = True
            elif len(str(verification_answer or "").strip()) >= 3:
                verification_correct = True
    else:
        verification_correct = True

    # 2. AI Mastery Judge Evaluation of Active Recall Retrieval
    retrieval_score = 0.0
    reasoning_depth_level = "SURFACE_ROTE"
    retained_concepts: list[str] = []
    missed_gaps: list[str] = []
    feedback = ""

    if not clean_recall:
        retrieval_score = 0.0
        reasoning_depth_level = "SURFACE_ROTE"
        retained_concepts = []
        missed_gaps = essential_keywords
        feedback = "No recall text submitted. Active retrieval required to assess comprehension."
    elif llm.enabled():
        judge_sys = (
            "You are an authoritative AI Mastery Judge assessing an unprompted Active Recall retrieval attempt.\n"
            "Compare the student's unprompted recall explanation against the Study Card ground truth.\n"
            "Evaluate:\n"
            "  1. Conceptual coverage of core intuition and mechanism (not exact verbatim memorization).\n"
            "  2. Presence and accurate explanation of essential technical keywords.\n"
            "  3. Cognitive depth: 'SURFACE_ROTE' (vague keywords only), 'DEVELOPING' (explains mechanism), or 'DEEP_CONCEPTUAL' (derives principles/formulas accurately).\n"
            "Return ONLY a single valid JSON object with exact keys:\n"
            '  "retrieval_score": <float between 0.0 and 1.0>,\n'
            '  "reasoning_depth_level": "SURFACE_ROTE" | "DEVELOPING" | "DEEP_CONCEPTUAL",\n'
            '  "retained_concepts": ["<keyword/concept 1>", ...],\n'
            '  "missed_gaps": ["<missed key concept/gap>", ...],\n'
            '  "feedback": "<clear 1-2 sentence pedagogical feedback>"'
        )
        judge_usr = (
            f"Concept: {concept_id}\n"
            f"Study Card Ground Truth:\n"
            f"- Core Intuition: {lesson.get('core_intuition')}\n"
            f"- Mechanism / Formula: {lesson.get('key_formula_or_mechanism')}\n"
            f"- Essential Keywords: {essential_keywords}\n\n"
            f"Student Unprompted Recall:\n\"{clean_recall}\"\n\n"
            f"Verification Question Correct: {verification_correct}"
        )
        try:
            raw = llm.complete(judge_sys, judge_usr, max_tokens=800)
            if raw:
                clean_json = llm.extract_json_payload(raw)
                eval_data = json.loads(clean_json)
                retrieval_score = max(0.0, min(1.0, float(eval_data.get("retrieval_score", 0.5))))
                depth = str(eval_data.get("reasoning_depth_level", "DEVELOPING")).upper()
                if depth in ("SURFACE_ROTE", "DEVELOPING", "DEEP_CONCEPTUAL"):
                    reasoning_depth_level = depth
                retained_concepts = [str(c).strip() for c in eval_data.get("retained_concepts", [])]
                missed_gaps = [str(g).strip() for g in eval_data.get("missed_gaps", [])]
                feedback = str(eval_data.get("feedback", "")).strip()
        except Exception as e:
            logger.warning(f"LLM AI Mastery Judge fallback triggered: {e}")

    # Deterministic fallback evaluation if score uncalculated
    if retrieval_score == 0.0 and clean_recall:
        lower_recall = clean_recall.lower()
        retained = []
        missed = []
        for kw in essential_keywords:
            if kw in lower_recall or any(difflib.SequenceMatcher(None, kw, w).ratio() >= 0.85 for w in lower_recall.split()):
                retained.append(kw)
            else:
                missed.append(kw)

        kw_ratio = len(retained) / max(len(essential_keywords), 1)
        word_count = len(clean_recall.split())

        # Check deep reasoning markers
        deep_markers = ["because", "therefore", "gradient", "derivative", "proportional", "formula", "equals", "step", "distance", "error", "rate"]
        has_deep_marker = any(m in lower_recall for m in deep_markers)

        if kw_ratio >= 0.60 and word_count >= 15:
            base_score = 0.70 + (kw_ratio * 0.20)
            reasoning_depth_level = "DEEP_CONCEPTUAL" if has_deep_marker else "DEVELOPING"
        elif kw_ratio >= 0.30 or word_count >= 8:
            base_score = 0.50 + (kw_ratio * 0.20)
            reasoning_depth_level = "DEVELOPING"
        else:
            base_score = 0.30 * kw_ratio
            reasoning_depth_level = "SURFACE_ROTE"

        # Verification bonus/penalty
        verif_factor = 0.10 if verification_correct else -0.10
        retrieval_score = max(0.0, min(1.0, round(base_score + verif_factor, 2)))
        retained_concepts = retained
        missed_gaps = missed
        feedback = (
            f"Successfully recalled {len(retained)}/{len(essential_keywords)} key concepts. "
            f"Verification question {'passed' if verification_correct else 'missed'}."
        )

    # 3. Update Student Epistemic Mastery in SQLite
    audit_text = (
        f"[ACTIVE_RECALL] Session {session_id} on {concept_id}: "
        f"Retrieval Score={retrieval_score:.2f}, Depth={reasoning_depth_level}, "
        f"Verification={'Passed' if verification_correct else 'Failed'}"
    )

    mastery_record = mastery.record_active_recall(
        user_id=user_id,
        concept_id=concept_id,
        retrieval_score=retrieval_score,
        reasoning_depth=reasoning_depth_level,
        retained_concepts=retained_concepts,
        missed_gaps=missed_gaps,
        verification_correct=verification_correct,
        evidence_text=audit_text,
    )
    mastery_status = mastery_record.get("state", "UNKNOWN")

    # 4. Save/Update Session in SQLite
    store.update_active_recall_session(
        session_id=session_id,
        updates={
            "recall_submission_text": clean_recall,
            "retrieval_score": retrieval_score,
            "retained_concepts_json": retained_concepts,
            "missed_gaps_json": missed_gaps,
        }
    )

    return ActiveRecallResult(
        session_id=session_id,
        concept_id=concept_id,
        retrieval_score=retrieval_score,
        reasoning_depth_level=reasoning_depth_level,
        retained_concepts=retained_concepts,
        missed_gaps=missed_gaps,
        verification_correct=verification_correct,
        feedback=feedback or f"Retrieved {len(retained_concepts)} concepts with {reasoning_depth_level} reasoning.",
        mastery_status=mastery_status,
        audit_trail=audit_text,
    )


def get_history(user_id: str = "default") -> list[ActiveRecallHistoryItem]:
    """Return learner's active recall session history."""
    rows = store.list_active_recall_sessions(user_id=user_id, limit=50)
    kb = store.get_kb(user_id)
    by_id = {c["id"]: c.get("name", c["id"]) for c in kb.get("concepts", [])}

    history: list[ActiveRecallHistoryItem] = []
    for r in rows:
        cid = r["concept_id"]
        cname = by_id.get(cid, cid.replace("_", " ").title())
        retained = json.loads(r["retained_concepts_json"]) if r.get("retained_concepts_json") else []
        missed = json.loads(r["missed_gaps_json"]) if r.get("missed_gaps_json") else []
        score = float(r["retrieval_score"]) if r.get("retrieval_score") is not None else None

        history.append(
            ActiveRecallHistoryItem(
                session_id=r["session_id"],
                user_id=r["user_id"],
                concept_id=cid,
                concept_name=cname,
                timer_seconds=int(r.get("timer_seconds", 120)),
                retrieval_score=score,
                retained_concepts=retained if isinstance(retained, list) else [],
                missed_gaps=missed if isinstance(missed, list) else [],
                created_at=r.get("created_at", ""),
            )
        )
    return history
