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
from . import mastery, quiz

logger = logging.getLogger("concept_mastery.active_recall")


def _extract_keywords(text: str, count: int = 5) -> list[str]:
    """Extract salient technical keywords from text."""
    stop_words = {
        "the", "a", "an", "and", "or", "in", "on", "at", "to", "for", "with",
        "by", "about", "as", "is", "are", "was", "were", "be", "been", "this",
        "that", "these", "those", "from", "into", "which", "then", "than", "such",
        "step", "calculate", "compute", "where", "given", "using", "when", "after"
    }
    words = re.findall(r"\b[A-Za-z][A-Za-z0-9_-]{3,}\b", text)
    filtered = [w.lower() for w in words if w.lower() not in stop_words]
    seen = set()
    result = []
    for w in filtered:
        if w not in seen:
            seen.add(w)
            result.append(w)
            if len(result) >= count:
                break
    return result or ["concept", "mechanism", "formula"]


def _synthesize_card_fallback(concept_id: str, concept_name: str, chunks: list, c_def: str) -> dict:
    """Deterministic vectorless synthesis of Study Card components."""
    combined_text = "\n".join(c.text for c in chunks) if chunks else c_def or f"Conceptual material on {concept_name}."
    chunk_sents = sentences(combined_text)

    # 1. Core Intuition
    if c_def:
        core_intuition = c_def
    elif chunk_sents:
        core_intuition = chunk_sents[0]
    else:
        core_intuition = f"{concept_name} provides a foundational algorithmic and theoretical approach in this domain."

    # 2. Key Formula / Mechanism
    # Look for equations or sentences with mathematical/algorithmic terms
    formula_sent = ""
    for s in chunk_sents:
        if any(term in s.lower() for term in ("formula", "equation", "algorithm", "calculat", "update", "rule", "=", "sum", "rate")):
            formula_sent = s
            break
    if not formula_sent:
        formula_sent = chunk_sents[1] if len(chunk_sents) > 1 else f"Mechanisms of {concept_name} operate through sequential computation and optimization."

    # 3. Worked Mini-Example
    example_sent = ""
    for s in chunk_sents:
        if any(term in s.lower() for term in ("example", "consider", "suppose", "assume", "let ", "yields", "given")):
            example_sent = s
            break
    if not example_sent:
        example_sent = f"Consider applying {concept_name} to test inputs: input features are processed through the governing equation to yield predictable outputs."

    # 4. Essential Keywords (3-5 items)
    keywords = [concept_name.lower()]
    extracted = _extract_keywords(combined_text, count=5)
    for kw in extracted:
        if kw not in keywords and len(keywords) < 5:
            keywords.append(kw)
    if len(keywords) < 3:
        keywords.extend(["optimization", "computation", "parameters"][:3 - len(keywords)])

    # 5. Citation
    if chunks:
        c0 = chunks[0]
        loc = getattr(c0, "locator_label", None) or f"Page {c0.page_number}"
        sec = getattr(c0, "section_title", "")
        citation = f"[{loc}{f', Section: {sec}' if sec else ''}]"
    else:
        citation = "[Canonical Syllabus Reference]"

    return {
        "core_intuition": core_intuition.strip(),
        "key_formula_or_mechanism": formula_sent.strip(),
        "worked_mini_example": example_sent.strip(),
        "essential_keywords": keywords[:5],
        "exact_source_citation": citation,
    }


def start_session(
    user_id: str = "default",
    concept_id: str = "",
    timer_seconds: int = 120,
) -> ActiveRecallStudyCard:
    """Initialize an Active Recall study session with grounded Study Card and verification question."""
    if not concept_id:
        raise store.UserError("concept_id is required to start an active recall session.")

    kb = store.get_kb(user_id)
    by_id = {c["id"]: c for c in kb.get("concepts", [])}
    concept = by_id.get(concept_id)

    concept_name = concept.get("name", concept_id.replace("_", " ").title()) if concept else concept_id.replace("_", " ").title()
    concept_def = concept.get("definition", "") if concept else ""

    # 1. Retrieve vectorless ground truth chunks from SQLite BM25 index
    chunks = GLOBAL_INDEX.retrieve_chunks(concept_id, max_chunks=3, user_id=user_id)
    if not chunks:
        bm25_hits = GLOBAL_INDEX.search_bm25(concept_name, max_chunks=2, user_id=user_id)
        chunks = [hit[1] for hit in bm25_hits]

    # 2. Synthesize Study Card
    card_data = None
    if llm.enabled() and chunks:
        ctx_text = "\n\n".join(f"[{getattr(c, 'locator_label', None) or f'Page {c.page_number}'}] {c.text}" for c in chunks)
        sys_prompt = (
            "You are an expert cognitive learning specialist and STEM tutor. Generate a high-yield Active Recall Study Card "
            "strictly grounded in the provided textbook context.\n"
            "Return ONLY a single valid JSON object with exact keys:\n"
            '  "core_intuition": "<concise executive summary / mental model>",\n'
            '  "key_formula_or_mechanism": "<governing mathematical equation, rule, or algorithm>",\n'
            '  "worked_mini_example": "<concrete applied mini-example demonstrating the mechanism>",\n'
            '  "essential_keywords": ["<keyword 1>", "<keyword 2>", "<keyword 3>", "<keyword 4>", "<keyword 5>"] (3 to 5 terms),\n'
            '  "exact_source_citation": "<citation anchor e.g. [Page 4, Section: Optimization]>"'
        )
        usr_prompt = f"Target Concept: {concept_name}\nContext:\n{ctx_text}"
        try:
            raw = llm.complete(sys_prompt, usr_prompt, max_tokens=1000)
            if raw:
                clean = llm.extract_json_payload(raw)
                parsed = json.loads(clean)
                if all(k in parsed for k in ("core_intuition", "key_formula_or_mechanism", "worked_mini_example", "essential_keywords")):
                    card_data = {
                        "core_intuition": str(parsed["core_intuition"]).strip(),
                        "key_formula_or_mechanism": str(parsed["key_formula_or_mechanism"]).strip(),
                        "worked_mini_example": str(parsed["worked_mini_example"]).strip(),
                        "essential_keywords": [str(k).strip().lower() for k in parsed.get("essential_keywords", [])][:5],
                        "exact_source_citation": str(parsed.get("exact_source_citation", f"[Page {chunks[0].page_number}]")).strip(),
                    }
        except Exception as e:
            logger.warning(f"LLM Study Card generation fallback triggered: {e}")

    if not card_data:
        card_data = _synthesize_card_fallback(concept_id, concept_name, chunks, concept_def)

    # 3. Generate 1 Verification Question grounded in the concept
    verification_q: Optional[dict] = None
    try:
        source_p = chunks[0].page_number if chunks else 1
        if quiz.is_mathematical_concept(concept_id, concept_name):
            vq_obj = quiz.generate_numerical_question(concept_id, concept_name=concept_name, user_id=user_id, source_page=source_p)
            verification_q = vq_obj.model_dump()
        else:
            qs = quiz.generate_diagnostic_quiz(concept_id, count=1, question_types=["mcq"], user_id=user_id)
            if qs:
                verification_q = qs[0].model_dump()
    except Exception as e:
        logger.warning(f"Verification question fallback: {e}")
        # Build simple fallback verification question
        fallback_qid = f"diag_{uuid.uuid4().hex[:8]}"
        vq_fallback = QuizQuestion(
            question_id=fallback_qid,
            concept_id=concept_id,
            question_type="mcq",
            question_text=f"Which term best describes the core operational mechanism of {concept_name}?",
            options=[concept_name, "Random Noise", "Arbitrary Search", "Unbounded Divergence"],
            correct_option_idx=0,
            explanation=card_data["core_intuition"],
            source_page=chunks[0].page_number if chunks else 1,
        )
        quiz.REGISTRY[fallback_qid] = vq_fallback.model_dump()
        verification_q = vq_fallback.model_dump()

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
