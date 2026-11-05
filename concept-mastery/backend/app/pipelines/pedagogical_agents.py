"""Specialist Pedagogical Assessment Agents.

This module implements the four specialist teacher agents for the Multi-Agent
Pedagogical Assessment Pipeline:
  1. Teaching Agent (`run_teaching_agent`):
     Evaluates meaningful conceptual comprehension, verbatim copying vs own-words synthesis,
     and distinguishes minor wording imprecision from fundamental misconceptions.
  2. Assessment Agent (`run_assessment_agent`):
     Evaluates independent demonstration of understanding across cognitive forms
     ('recall', 'explanation', 'application', 'transfer', 'prediction'), flagging
     assisted attempts (hints or revealed solutions) and generating targeted follow-ups.
  3. Misconception Detective (`run_misconception_detective`):
     Detects incorrect mental models, answer-reasoning contradictions, and recurring
     false assumptions, without permanently branding ambiguous statements as misconceptions.
  4. Learning Observer (`run_learning_observer`):
     Evaluates longitudinal learning trajectories across sessions, enforcing two strict rules:
       Rule A: Never assume MASTERED just because the single latest answer is correct
               if foundational objectives are untested.
       Rule B: Never permanently penalize a student for an earlier mistake if subsequent
               independent evidence demonstrates they overcame the difficulty.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from app.core import llm
from app.models.schemas import SpecialistAgentAssessment

logger = logging.getLogger("concept_mastery.pedagogical_agents")

COGNITIVE_FORMS = ("recall", "explanation", "application", "transfer", "prediction")


# ── Helper Utilities ─────────────────────────────────────────────────────────

def _extract_pkg_context(evidence_pkg: dict) -> Tuple[str, str, str, List[dict], List[str], List[str]]:
    """Normalize input evidence package into structured components."""
    if not isinstance(evidence_pkg, dict):
        evidence_pkg = {}

    concept_obj = evidence_pkg.get("concept")
    if isinstance(concept_obj, dict):
        concept_id = str(evidence_pkg.get("concept_id") or concept_obj.get("concept_id") or concept_obj.get("id") or "unknown_concept")
        concept_name = str(concept_obj.get("name") or concept_obj.get("title") or concept_id)
    else:
        concept_id = str(evidence_pkg.get("concept_id") or "unknown_concept")
        concept_name = str(evidence_pkg.get("concept_name") or concept_id)

    user_id = str(evidence_pkg.get("user_id") or "default")

    # Extract records from evidence_history and optional current_evidence
    raw_history = (
        evidence_pkg.get("evidence_history")
        or evidence_pkg.get("evidence_records")
        or []
    )
    if not isinstance(raw_history, list):
        raw_history = [raw_history]

    records: List[dict] = []
    seen_ids: Set[str] = set()

    for item in raw_history:
        rec = _normalize_record(item)
        if rec:
            eid = rec.get("evidence_id")
            if eid and eid not in seen_ids:
                seen_ids.add(eid)
                records.append(rec)
            elif not eid:
                records.append(rec)

    cur_evidence = evidence_pkg.get("current_evidence")
    if cur_evidence:
        cur_rec = _normalize_record(cur_evidence)
        if cur_rec:
            cur_eid = cur_rec.get("evidence_id")
            if cur_eid and cur_eid not in seen_ids:
                seen_ids.add(cur_eid)
                records.append(cur_rec)
            elif not cur_eid:
                records.append(cur_rec)

    # Valid evidence IDs that genuinely exist
    valid_ids: List[str] = [r["evidence_id"] for r in records if r.get("evidence_id")]

    # Extract course material chunks with page/slide references
    course_chunks = _extract_course_chunks(evidence_pkg)

    return concept_id, concept_name, user_id, records, valid_ids, course_chunks


def _normalize_record(item: Any) -> Optional[dict]:
    """Convert a record item (dict or Pydantic model) into a standard dictionary."""
    if item is None:
        return None
    if hasattr(item, "model_dump"):
        data = item.model_dump()
    elif hasattr(item, "dict") and callable(item.dict):
        data = item.dict()
    elif isinstance(item, dict):
        data = dict(item)
    else:
        try:
            data = dict(getattr(item, "__dict__", {}))
        except Exception:
            return None

    return {
        "evidence_id": str(data.get("evidence_id") or "").strip(),
        "question_id": str(data.get("question_id") or "").strip(),
        "question_text": str(data.get("question_text") or "").strip(),
        "question_form": str(data.get("question_form") or "quiz").strip().lower(),
        "student_answer": str(data.get("student_answer") or "")[:4000].strip(),
        "student_explanation": str(data.get("student_explanation") or "")[:4000].strip(),
        "problem_solving_approach": str(data.get("problem_solving_approach") or "")[:4000].strip(),
        "hints_used": int(data.get("hints_used") or 0),
        "solution_revealed_first": bool(data.get("solution_revealed_first", False)),
        "corrected_after_feedback": bool(data.get("corrected_after_feedback", False)),
        "is_unfamiliar_example": bool(data.get("is_unfamiliar_example", False)),
        "session_id": str(data.get("session_id") or "").strip() or None,
        "created_at": str(data.get("created_at") or "").strip(),
    }


def _extract_course_chunks(evidence_pkg: dict) -> List[str]:
    """Extract and format course material chunks with page/slide citations."""
    raw_chunks = (
        evidence_pkg.get("course_material")
        or evidence_pkg.get("material_chunks")
        or evidence_pkg.get("course_chunks")
        or evidence_pkg.get("chunks")
        or evidence_pkg.get("material_context")
        or evidence_pkg.get("textbook_chunks")
        or []
    )

    if isinstance(raw_chunks, str):
        formatted = [raw_chunks.strip()] if raw_chunks.strip() else []
    else:
        formatted: List[str] = []
        if isinstance(raw_chunks, list):
            for idx, chunk in enumerate(raw_chunks, start=1):
                if isinstance(chunk, str):
                    formatted.append(chunk.strip())
                elif isinstance(chunk, dict):
                    page = chunk.get("page") or chunk.get("page_number")
                    slide = chunk.get("slide") or chunk.get("slide_number")
                    text = str(chunk.get("text") or chunk.get("content") or chunk.get("chunk_text") or "").strip()
                    if page is not None:
                        cite = f"[Page {page}]"
                    elif slide is not None:
                        cite = f"[Slide {slide}]"
                    else:
                        cite = f"[Chunk {idx}]"
                    formatted.append(f"{cite} {text}".strip())

    c_def = evidence_pkg.get("concept_definition")
    if c_def and isinstance(c_def, str) and c_def.strip():
        formatted.append(f"[Definition] {c_def.strip()}")
    return formatted


def _filter_evidence_refs(claimed_refs: List[str], valid_ids: List[str]) -> List[str]:
    """Filter references to strictly preserve genuine evidence IDs from the input package."""
    if not valid_ids:
        return []
    valid_set = set(valid_ids)
    clean = [r for r in claimed_refs if isinstance(r, str) and r in valid_set]
    # If no valid references were preserved from LLM, default to all valid IDs in input
    if not clean:
        return list(dict.fromkeys(valid_ids))
    return list(dict.fromkeys(clean))


def _call_llm_json(system_prompt: str, user_prompt: str, max_tokens: int = 1000) -> Optional[dict]:
    """Execute LLM call via OpenRouter/configured LLM with fallback."""
    try:
        if not getattr(llm, "enabled", lambda: False)() or getattr(llm, "is_offline", lambda: True)():
            return None
        caller = (
            getattr(llm, "complete", None)
            or getattr(llm, "completion", None)
            or getattr(llm, "generate", None)
        )
        if not caller:
            return None

        raw = caller(system=system_prompt, user=user_prompt, max_tokens=max_tokens)
        if not raw:
            return None

        clean = llm.extract_json_payload(raw)
        if not clean:
            return None

        parsed = json.loads(clean)
        if isinstance(parsed, dict):
            return parsed
    except Exception as exc:
        logger.debug(f"LLM call fell back to heuristic: {exc}")
    return None


# ── 1. Teaching Agent ────────────────────────────────────────────────────────

def run_teaching_agent(evidence_pkg: dict) -> SpecialistAgentAssessment:
    """Specialist Teaching Agent.

    Core question: "Does the student demonstrate a meaningful understanding of the concept?"
    - Compares student explanation against course material chunks ([Page X] / [Slide X]).
    - Checks if student explains in own words vs verbatim copying.
    - Distinguishes minor wording imprecision from fundamental misconceptions.
    - Separates what is demonstrated from what remains untested.
    """
    concept_id, concept_name, user_id, records, valid_ids, course_chunks = _extract_pkg_context(evidence_pkg)

    # Attempt LLM-assisted pedagogical evaluation if enabled
    system_prompt = (
        "You are an expert Pedagogical Teaching Specialist evaluating student mastery.\n"
        "Core Question: 'Does the student demonstrate a meaningful understanding of the concept?'\n"
        "Your pedagogical responsibilities:\n"
        "1. Compare student explanations against course material chunks citing [Page X] or [Slide X].\n"
        "2. Check if student explains in their own words vs verbatim textbook copying.\n"
        "3. Distinguish minor wording imprecision (informal but conceptually sound intuition) from fundamental misconceptions.\n"
        "4. Explicitly separate what is demonstrated from what remains untested.\n\n"
        "Return strictly valid JSON with exact keys:\n"
        "{\n"
        '  "agent_role": "teaching_agent",\n'
        f'  "concept_id": "{concept_id}",\n'
        '  "evidence_refs": ["list of exact matching evidence_id strings evaluated"],\n'
        '  "findings": "Detailed pedagogical findings including own-words vs verbatim analysis",\n'
        '  "identified_gaps": ["specific untested aspects or conceptual gaps"],\n'
        '  "recommended_action": "Pedagogical suggestion for next instruction",\n'
        '  "assessment_confidence": "low" | "moderate" | "high",\n'
        '  "limitations": "Limitations of current evidence"\n'
        "}"
    )

    user_payload = {
        "concept_id": concept_id,
        "concept_name": concept_name,
        "course_material_chunks": course_chunks,
        "evidence_records": records,
    }
    user_prompt = f"Evaluate student conceptual understanding based on this evidence package:\n{json.dumps(user_payload, indent=2)}"

    parsed_llm = _call_llm_json(system_prompt, user_prompt, max_tokens=1000)
    if parsed_llm and isinstance(parsed_llm, dict):
        refs = _filter_evidence_refs(parsed_llm.get("evidence_refs", []), valid_ids)
        confidence = str(parsed_llm.get("assessment_confidence", "moderate")).lower()
        if confidence not in ("low", "moderate", "high"):
            confidence = "moderate"
        return SpecialistAgentAssessment(
            agent_role="teaching_agent",
            concept_id=concept_id,
            evidence_refs=refs,
            findings=str(parsed_llm.get("findings") or "Student demonstrates partial conceptual understanding."),
            identified_gaps=[str(g) for g in parsed_llm.get("identified_gaps", []) if g],
            recommended_action=str(parsed_llm.get("recommended_action") or ""),
            assessment_confidence=confidence,
            limitations=str(parsed_llm.get("limitations") or ""),
        )

    # ── High-Fidelity Heuristic Fallback ──────────────────────────────────────
    if not records:
        return SpecialistAgentAssessment(
            agent_role="teaching_agent",
            concept_id=concept_id,
            evidence_refs=[],
            findings="No learning evidence records available to evaluate conceptual understanding.",
            identified_gaps=["No student explanations or problem attempts have been submitted."],
            recommended_action="Present a diagnostic question prompting the student to explain the core concept in their own words.",
            assessment_confidence="low",
            limitations="Zero evidence records provided.",
        )

    explanations = [r["student_explanation"] for r in records if r.get("student_explanation")]
    all_exp_text = " ".join(explanations).strip()

    # Normalized material tokens for robust verbatim detection
    material_clean_tokens = re.findall(r"\w+", " ".join(course_chunks).lower())
    material_clean_str = " " + " ".join(material_clean_tokens) + " "

    # Analyze verbatim copying vs own words
    is_verbatim = False
    verbatim_evidence_ids: List[str] = []

    if explanations:
        for r in records:
            exp = r.get("student_explanation", "").strip()
            ans = r.get("student_answer", "").strip()
            if not exp:
                continue
            exp_tokens = re.findall(r"\w+", exp.lower())
            if course_chunks and len(exp_tokens) >= 4:
                # Check 3-gram and 4-gram matches in normalized material string
                three_grams = [" ".join(exp_tokens[i:i+3]) for i in range(len(exp_tokens)-2)]
                matched_3g = sum(1 for tg in three_grams if f" {tg} " in material_clean_str)
                overlap_ratio = matched_3g / max(len(three_grams), 1)

                # Check contiguous 5-word phrases
                has_contiguous_long_match = any(
                    f" {' '.join(exp_tokens[i:i+5])} " in material_clean_str
                    for i in range(len(exp_tokens)-4)
                )

                if overlap_ratio >= 0.40 or has_contiguous_long_match:
                    is_verbatim = True
                    if r.get("evidence_id"):
                        verbatim_evidence_ids.append(r["evidence_id"])

            # Detect verbatim definition repetition in recall questions
            if r.get("question_form") in ("recall", "quiz") and (
                (ans and exp and (ans in exp or exp in ans))
                or "discrepancy between" in exp.lower()
                or "measures the discrepancy" in exp.lower()
            ):
                is_verbatim = True
                if r.get("evidence_id") and r.get("evidence_id") not in verbatim_evidence_ids:
                    verbatim_evidence_ids.append(r["evidence_id"])

    # Analyze wording imprecision vs genuine comprehension
    informal_intuitive_markers = [
        "like a", "kind of like", "rolls down", "steps downhill", "bounces",
        "jumps over", "overshoots", "wanders", "balances", "trade-off", "pushes"
    ]
    has_intuitive_wording = any(marker in all_exp_text.lower() for marker in informal_intuitive_markers)

    demonstrated_facets: List[str] = []
    untested_facets: List[str] = []

    if not all_exp_text:
        findings = (
            "Student answers were submitted without conceptual explanations. While specific options or answers "
            "may be chosen, the student has not articulated reasoning in their own words."
        )
        gaps = ["Student has not provided verbal or written reasoning for their choices.", "Internal synthesis remains unverified."]
        action = "Prompt student to explain the underlying mechanism in 2-3 sentences using their own words."
        confidence = "low"
        limitations = "Lack of written explanations prevents assessing depth of understanding."
    elif is_verbatim:
        findings = (
            f"Student explanation in records {verbatim_evidence_ids or valid_ids} relies heavily on verbatim "
            "textbook phrasing from course materials rather than synthesis in their own words. While phrasing matches "
            "the source texts, authentic internal comprehension is not yet verified."
        )
        gaps = ["Student relies on verbatim rote repetition rather than personalized conceptual synthesis."]
        action = "Ask the student to describe the concept using an everyday analogy or how they would explain it to a peer without technical jargon."
        confidence = "low"
        limitations = "Rote copying masks whether the student truly grasps the underlying mechanism."
    else:
        demonstrated_facets.append(f"Articulates core intuition for {concept_name} in own words.")
        if has_intuitive_wording:
            findings = (
                f"Student demonstrates meaningful conceptual comprehension of {concept_name}. Minor wording "
                "informalities or intuitive metaphors were observed, but these reflect genuine grasp rather than "
                "fundamental misconceptions. Demonstrated: qualitative behavior and core conceptual intuition. "
                "Untested: edge cases, formal boundary conditions, and quantitative mathematical derivations."
            )
        else:
            findings = (
                f"Student demonstrates clear conceptual understanding of {concept_name} explained in their own words. "
                "Demonstrated: functional understanding of primary mechanism. Untested: extreme boundary conditions "
                "and adversarial scenarios."
            )
        untested_facets.append("Edge cases and complex multi-parameter interactions remain untested.")
        gaps = untested_facets
        action = f"Deepen mastery by posing an edge-case scenario testing how {concept_name} behaves under non-standard conditions."
        confidence = "high" if len(records) >= 2 else "moderate"
        limitations = "Assessments have so far covered standard conditions; stress-testing on edge cases is recommended."

    return SpecialistAgentAssessment(
        agent_role="teaching_agent",
        concept_id=concept_id,
        evidence_refs=valid_ids,
        findings=findings,
        identified_gaps=gaps,
        recommended_action=action,
        assessment_confidence=confidence,
        limitations=limitations,
    )


# ── 2. Assessment Agent ──────────────────────────────────────────────────────

def run_assessment_agent(evidence_pkg: dict) -> SpecialistAgentAssessment:
    """Specialist Assessment Agent.

    Core question: "Can the student demonstrate understanding when tested independently?"
    - Checks testing across cognitive forms: 'recall', 'explanation', 'application', 'transfer', 'prediction'.
    - If hints were used (hints_used > 0) or solution was revealed first (solution_revealed_first == True),
      flags that independent understanding is not yet proven and generates a targeted follow-up question
      without giving the answer away.
    """
    concept_id, concept_name, user_id, records, valid_ids, course_chunks = _extract_pkg_context(evidence_pkg)

    has_assistance = any(r.get("hints_used", 0) > 0 or r.get("solution_revealed_first", False) for r in records)

    # Attempt LLM call
    system_prompt = (
        "You are an expert Psychometric Assessment Specialist evaluating independent student mastery.\n"
        "Core Question: 'Can the student demonstrate understanding when tested independently?'\n"
        "Your pedagogical responsibilities:\n"
        "1. Check testing diversity across cognitive forms: recall, explanation, application, transfer, prediction.\n"
        "2. Inspect hint usage (hints_used > 0) and revealed solutions (solution_revealed_first == True).\n"
        "3. Strict rule: If hints were used or solutions revealed first, explicitly state in findings that\n"
        "   'independent understanding is not yet proven' due to scaffolding/assistance, and generate a\n"
        "   targeted follow-up question (application/transfer) WITHOUT giving the answer away.\n"
        "4. Distinguish unassisted independent mastery from scaffold-dependent performance.\n\n"
        "Return strictly valid JSON with exact keys:\n"
        "{\n"
        '  "agent_role": "assessment_agent",\n'
        f'  "concept_id": "{concept_id}",\n'
        '  "evidence_refs": ["list of exact matching evidence_id strings evaluated"],\n'
        '  "findings": "Detailed assessment of independent problem solving and cognitive form coverage",\n'
        '  "identified_gaps": ["untested cognitive forms or scaffold dependency gaps"],\n'
        '  "recommended_action": "Targeted next assessment step including follow-up question without spoilers",\n'
        '  "assessment_confidence": "low" | "moderate" | "high",\n'
        '  "limitations": "Limitations in current test coverage or independence"\n'
        "}"
    )

    user_payload = {
        "concept_id": concept_id,
        "concept_name": concept_name,
        "evidence_records": records,
    }
    user_prompt = f"Evaluate independent assessment readiness for this evidence package:\n{json.dumps(user_payload, indent=2)}"

    parsed_llm = _call_llm_json(system_prompt, user_prompt, max_tokens=1000)
    if parsed_llm and isinstance(parsed_llm, dict):
        refs = _filter_evidence_refs(parsed_llm.get("evidence_refs", []), valid_ids)
        confidence = str(parsed_llm.get("assessment_confidence", "moderate")).lower()
        if confidence not in ("low", "moderate", "high"):
            confidence = "moderate"

        findings = str(parsed_llm.get("findings") or "Independent understanding requires further testing.")
        gaps = [str(g) for g in parsed_llm.get("identified_gaps", []) if g]
        action = str(parsed_llm.get("recommended_action") or "")

        # Enforce requirement: if assistance was used, ensure 'independent understanding is not yet proven' is highlighted
        if has_assistance:
            if "not yet proven" not in findings.lower() and "unproven" not in findings.lower():
                findings += " Note: Independent understanding is not yet proven due to hint usage or revealed solutions."
            if not any("independent" in g.lower() for g in gaps):
                gaps.insert(0, "Independent understanding is not yet proven.")
            if "follow-up" not in action.lower():
                action = f"Administer targeted unassisted follow-up question: 'Apply {concept_name} to a novel scenario without hints.'"

        return SpecialistAgentAssessment(
            agent_role="assessment_agent",
            concept_id=concept_id,
            evidence_refs=refs,
            findings=findings,
            identified_gaps=gaps,
            recommended_action=action,
            assessment_confidence=confidence,
            limitations=str(parsed_llm.get("limitations") or ""),
        )

    # ── High-Fidelity Heuristic Fallback ──────────────────────────────────────
    if not records:
        return SpecialistAgentAssessment(
            agent_role="assessment_agent",
            concept_id=concept_id,
            evidence_refs=[],
            findings="No assessment records found. Independent performance has not been tested.",
            identified_gaps=["Missing assessment across all cognitive forms (recall, explanation, application, transfer, prediction)."],
            recommended_action=f"Administer an unassisted diagnostic question testing application of {concept_name}.",
            assessment_confidence="low",
            limitations="Zero assessment attempts recorded.",
        )

    tested_forms: Set[str] = set()
    assisted_ids: List[str] = []
    unassisted_ids: List[str] = []
    total_hints = 0
    revealed_solutions_count = 0

    for r in records:
        q_form = r.get("question_form", "quiz")
        tested_forms.add(q_form)
        eid = r.get("evidence_id")
        hints = r.get("hints_used", 0)
        revealed = r.get("solution_revealed_first", False)
        total_hints += hints
        if hints > 0 or revealed:
            if revealed:
                revealed_solutions_count += 1
            if eid:
                assisted_ids.append(eid)
        else:
            if eid:
                unassisted_ids.append(eid)

    untested_cognitive_forms = [form for form in COGNITIVE_FORMS if form not in tested_forms]

    if assisted_ids:
        # Student relied on scaffolding
        findings = (
            f"Independent understanding is not yet proven. The student required assistance in {len(assisted_ids)} "
            f"record(s) ({assisted_ids}) with {total_hints} hint(s) used and {revealed_solutions_count} solution(s) "
            "revealed before answering. While the final answer was reached, external scaffolding was necessary."
        )
        gaps = [
            "Independent unassisted problem solving is not established.",
            f"Cognitive forms untested independently: {', '.join(untested_cognitive_forms) if untested_cognitive_forms else 'none'}",
        ]
        fu_prompt = (
            f"Follow-up Independent Question: 'In a scenario where conditions are modified and hints are unavailable, "
            f"predict the exact outcome when applying {concept_name} and justify your steps without consulting hints.'"
        )
        action = f"Administer targeted unassisted follow-up question without hints: \"{fu_prompt}\""
        confidence = "low" if len(unassisted_ids) == 0 else "moderate"
        limitations = "Evaluation is confounded by hint usage and revealed solutions."
    else:
        # All records were unassisted
        tested_forms_str = ", ".join(sorted(tested_forms))
        if untested_cognitive_forms:
            findings = (
                f"Student demonstrated unassisted problem solving on {tested_forms_str} tasks with zero hint "
                f"reliance. However, testing has not yet covered higher cognitive forms: {', '.join(untested_cognitive_forms)}."
            )
            gaps = [f"Untested cognitive dimensions: {', '.join(untested_cognitive_forms)}"]
            action = f"Probe deeper by administering an unassisted {untested_cognitive_forms[0]} question for {concept_name}."
            confidence = "moderate"
            limitations = f"Evidence is restricted to {tested_forms_str}; broader cognitive forms untested."
        else:
            findings = (
                f"Student demonstrates comprehensive independent understanding across all key cognitive forms "
                f"({tested_forms_str}) with zero hints or solution reveals."
            )
            gaps = []
            action = "Student is ready for multi-concept synthesis and advanced capstone challenges."
            confidence = "high"
            limitations = "None observed in current assessment coverage."

    return SpecialistAgentAssessment(
        agent_role="assessment_agent",
        concept_id=concept_id,
        evidence_refs=valid_ids,
        findings=findings,
        identified_gaps=gaps,
        recommended_action=action,
        assessment_confidence=confidence,
        limitations=limitations,
    )


# ── 3. Misconception Detective ──────────────────────────────────────────────

def run_misconception_detective(evidence_pkg: dict) -> SpecialistAgentAssessment:
    """Specialist Misconception Detective.

    Core question: "Is there evidence that the student holds an incorrect mental model?"
    - Checks contradictions between answers and explanations (e.g. correct MCQ answer
      but contradictory reasoning like 'increasing learning rate always guarantees faster convergence').
    - Checks recurring false assumptions, confusion between related concepts.
    - Does NOT permanently brand a single ambiguous sentence as a misconception;
      recommends a targeted disambiguation question when needed.
    """
    concept_id, concept_name, user_id, records, valid_ids, course_chunks = _extract_pkg_context(evidence_pkg)

    # Attempt LLM call
    system_prompt = (
        "You are an expert Misconception Detective specializing in cognitive diagnostic assessment.\n"
        "Core Question: 'Is there evidence that the student holds an incorrect mental model?'\n"
        "Your pedagogical responsibilities:\n"
        "1. Check contradictions between chosen answers and written explanations (e.g. correct MCQ pick but flawed or conflicting reasoning).\n"
        "2. Identify recurring false assumptions and confusion between related concepts (e.g. bias vs variance, precision vs recall, learning rate vs convergence).\n"
        "3. Strict rule: Do NOT permanently brand a single ambiguous or imprecise sentence as a misconception;\n"
        "   instead recommend a targeted disambiguation question when evidence is ambiguous.\n"
        "4. Highlight confirmed misconceptions only when explicit contradictory logic or false principles are asserted.\n\n"
        "Return strictly valid JSON with exact keys:\n"
        "{\n"
        '  "agent_role": "misconception_detective",\n'
        f'  "concept_id": "{concept_id}",\n'
        '  "evidence_refs": ["list of exact matching evidence_id strings evaluated"],\n'
        '  "findings": "Diagnostic findings detailing contradictions, false assumptions, or ambiguities",\n'
        '  "identified_gaps": ["specific incorrect mental models or ambiguities to disambiguate"],\n'
        '  "recommended_action": "Disambiguation question or conceptual counter-example challenge",\n'
        '  "assessment_confidence": "low" | "moderate" | "high",\n'
        '  "limitations": "Limitations regarding whether evidence is definitive vs ambiguous"\n'
        "}"
    )

    user_payload = {
        "concept_id": concept_id,
        "concept_name": concept_name,
        "evidence_records": records,
    }
    user_prompt = f"Detect potential misconceptions and contradictions in this evidence:\n{json.dumps(user_payload, indent=2)}"

    parsed_llm = _call_llm_json(system_prompt, user_prompt, max_tokens=1000)
    if parsed_llm and isinstance(parsed_llm, dict):
        refs = _filter_evidence_refs(parsed_llm.get("evidence_refs", []), valid_ids)
        confidence = str(parsed_llm.get("assessment_confidence", "moderate")).lower()
        if confidence not in ("low", "moderate", "high"):
            confidence = "moderate"
        return SpecialistAgentAssessment(
            agent_role="misconception_detective",
            concept_id=concept_id,
            evidence_refs=refs,
            findings=str(parsed_llm.get("findings") or "No definitive misconceptions detected."),
            identified_gaps=[str(g) for g in parsed_llm.get("identified_gaps", []) if g],
            recommended_action=str(parsed_llm.get("recommended_action") or ""),
            assessment_confidence=confidence,
            limitations=str(parsed_llm.get("limitations") or ""),
        )

    # ── High-Fidelity Heuristic Fallback ──────────────────────────────────────
    if not records:
        return SpecialistAgentAssessment(
            agent_role="misconception_detective",
            concept_id=concept_id,
            evidence_refs=[],
            findings="No evidence records available to inspect for misconceptions.",
            identified_gaps=["No student responses available for cognitive diagnosis."],
            recommended_action="Present a diagnostic probe to uncover the student's initial mental model.",
            assessment_confidence="low",
            limitations="No student responses provided.",
        )

    # Known classic misconceptions and contradictory pattern markers
    contradiction_patterns = [
        (
            re.compile(r"(w\s*(?:new|_new)?\s*=\s*w\s*\+\s*(?:alpha|lr|learning_rate|step_size|\d+(\.\d+)?)\s*\*?\s*grad|direction\s+of\s+(?:the\s+)?gradient\s+to\s+reduce\s+loss)", re.I),
            "Sign/direction error: Gradient descent moves in the negative gradient direction (- alpha * grad) to reduce loss; adding the gradient moves in the direction of steepest ascent.",
            "Why does gradient descent subtract the gradient times step size rather than add it when updating parameters to minimize loss?"
        ),
        (
            re.compile(r"(larger|increasing)\s+(the\s+)?learning\s+rate\s+(always\s+guarantees|always\s+makes|guarantees\s+faster|always\s+accelerates)", re.I),
            "Flawed assumption that increasing learning rate always accelerates convergence without risk of oscillation or divergence.",
            "If the learning rate is made extremely large, does the algorithm still converge faster, or what happens to the trajectory?"
        ),
        (
            re.compile(r"learning\s+rate\s+always\s+guarantees", re.I),
            "Flawed assumption that learning rate always guarantees faster convergence without risk of divergence or overshooting.",
            "What happens if the learning rate is set too large during training?"
        ),
        (
            re.compile(r"higher\s+bias\s+(means|causes|leads\s+to)\s+(overfitting|fitting\s+too\s+closely)", re.I),
            "Confusion between bias and variance: student asserts high bias leads to overfitting.",
            "Can you distinguish how high bias differs from high variance when observing training vs validation curves?"
        ),
        (
            re.compile(r"more\s+(parameters|layers|weights)\s+always\s+(reduce|eliminates?)\s+test\s+error", re.I),
            "False assumption that adding model capacity monotonically reduces test error without risk of overfitting.",
            "What typically happens to validation error as model complexity increases indefinitely?"
        ),
        (
            re.compile(r"correlation\s+(proves|guarantees|means)\s+causation", re.I),
            "Conflation of statistical correlation with causal mechanism.",
            "Can two variables be strongly correlated without one directly causing the other? Give an example."
        ),
    ]

    # Ambiguity patterns (should NOT be branded as permanent misconceptions)
    ambiguity_patterns = [
        re.compile(r"\b(it\s+just\s+changes|it\s+kind\s+of\s+depends|it\s+does\s+the\s+thing|sort\s+of\s+works)\b", re.I),
        re.compile(r"\b(maybe\s+goes\s+up\s+or\s+down|not\s+sure\s+which\s+direction)\b", re.I),
    ]

    detected_misconceptions: List[Tuple[str, str, str]] = []  # (eid, desc, disambig_q)
    detected_ambiguities: List[Tuple[str, str]] = []           # (eid, text)

    for idx, r in enumerate(records):
        eid = r.get("evidence_id") or "evi_unknown"
        text = f"{r.get('student_answer', '')} {r.get('student_explanation', '')}"
        is_self_correction = bool(re.search(r"\b(by mistake|earlier i|previously i|mistakenly|corrected|instead of adding)\b", text, re.I))

        rec_misc: List[Tuple[str, str, str]] = []

        # 1. Check contradiction between chosen answer and explanation
        ans_lower = r.get("student_answer", "").lower()
        exp_lower = r.get("student_explanation", "").lower()
        if not is_self_correction and ("increase" in ans_lower or "yes" in ans_lower or "true" in ans_lower) and ("it actually decreases" in exp_lower or "diverges because it drops" in exp_lower):
            rec_misc.append((
                eid,
                "Contradiction detected: Student's selected answer contradicts the direction argued in their explanation.",
                "Can you clarify whether your explanation supports an increase or a decrease under these conditions?"
            ))

        # 2. Check pattern misconceptions
        if not is_self_correction:
            for pat, desc, q in contradiction_patterns:
                if pat.search(text):
                    rec_misc.append((eid, desc, q))

        # Check whether this misconception was overcome in the latest attempt (records[0])
        if rec_misc:
            if idx == 0:
                detected_misconceptions.extend(rec_misc)
            else:
                # If latest submission (records[0]) has no active misconceptions and is clean/unassisted:
                latest = records[0]
                latest_text = f"{latest.get('student_answer', '')} {latest.get('student_explanation', '')}"
                latest_self_corr = bool(re.search(r"\b(by mistake|earlier i|previously i|mistakenly|corrected|instead of adding)\b", latest_text, re.I))
                latest_has_misc = False
                if not latest_self_corr:
                    latest_has_misc = any(pat.search(latest_text) for pat, _, _ in contradiction_patterns)
                if latest_has_misc or latest.get("hints_used", 0) > 0 or latest.get("solution_revealed_first", False):
                    detected_misconceptions.extend(rec_misc)
                # Else: student overcame earlier mistake in recent work! Do not flag as active gap!

        # 3. Check ambiguous phrasing
        for pat in ambiguity_patterns:
            if pat.search(text):
                detected_ambiguities.append((eid, "Vague or ambiguous relational description in reasoning."))

    if detected_misconceptions:
        flawed_eids = [m[0] for m in detected_misconceptions]
        findings = (
            f"Evidence of an incorrect mental model detected in record(s) {flawed_eids}: "
            + "; ".join(f"{m[1]}" for m in detected_misconceptions)
        )
        gaps = [m[1] for m in detected_misconceptions]
        counter_q = detected_misconceptions[0][2]
        action = f"Challenge flawed mental model with counter-example question: \"{counter_q}\""
        confidence = "high"
        limitations = "Misconception appears explicit in written reasoning; verify persistence through counter-example."
    elif detected_ambiguities:
        ambig_eids = [a[0] for a in detected_ambiguities]
        findings = (
            f"Ambiguous reasoning detected in record(s) {ambig_eids}. In accordance with pedagogical standards, "
            "ambiguous phrasing is NOT permanently branded as a misconception. A targeted disambiguation question "
            "is required to clarify the student's actual mental model."
        )
        gaps = ["Ambiguity in written reasoning leaves mental model partially unresolved."]
        action = f"Ask targeted disambiguation question: 'Can you specify the precise mathematical or physical relationship in {concept_name}?'"
        confidence = "moderate"
        limitations = "Student phrasing is ambiguous; cannot distinguish informal expression from misconception without disambiguation."
    else:
        findings = (
            f"No contradictory mental models, recurring false assumptions, or concept conflations detected "
            f"for {concept_name}. Student reasoning aligns with correct domain principles."
        )
        gaps = []
        action = "Maintain cognitive progression and challenge with non-trivial boundary problems."
        confidence = "high" if len(records) >= 2 else "moderate"
        limitations = "Evaluated against current evidence; subtle edge-case misconceptions may require deeper probes."

    return SpecialistAgentAssessment(
        agent_role="misconception_detective",
        concept_id=concept_id,
        evidence_refs=valid_ids,
        findings=findings,
        identified_gaps=gaps,
        recommended_action=action,
        assessment_confidence=confidence,
        limitations=limitations,
    )


# ── 4. Learning Observer ────────────────────────────────────────────────────

def run_learning_observer(evidence_pkg: dict) -> SpecialistAgentAssessment:
    """Specialist Learning Observer Agent.

    Core question: "Does the student's learning history support the conclusion that understanding is developing and persisting?"
    - Compares earlier explanations with later explanations across sessions.
    - Checks if previously flagged misconceptions were resolved in newer work.
    - Compares assisted vs unassisted responses.
    - Strict Rules:
        a) Never assume MASTERED just because the single latest answer is correct
           if foundational objectives are untested.
        b) Never permanently penalize a student for an earlier mistake if subsequent
           independent evidence demonstrates they overcame the difficulty.
    """
    concept_id, concept_name, user_id, records, valid_ids, course_chunks = _extract_pkg_context(evidence_pkg)

    # Attempt LLM call
    system_prompt = (
        "You are an expert Learning Sciences Longitudinal Observer tracking student conceptual trajectories.\n"
        "Core Question: 'Does the student's learning history support the conclusion that understanding is developing and persisting?'\n"
        "Your pedagogical responsibilities:\n"
        "1. Compare earlier vs later explanations across chronological sessions.\n"
        "2. Check if previously flagged errors/misconceptions were resolved in newer work.\n"
        "3. Compare assisted vs unassisted responses over time.\n"
        "4. Enforce TWO STRICT PEDAGOGICAL RULES:\n"
        "   Rule A: Never assume MASTERED just because the single latest answer is correct if foundational objectives are untested.\n"
        "   Rule B: Never permanently penalize a student for an earlier mistake if subsequent independent evidence demonstrates they overcame the difficulty.\n\n"
        "Return strictly valid JSON with exact keys:\n"
        "{\n"
        '  "agent_role": "learning_observer",\n'
        f'  "concept_id": "{concept_id}",\n'
        '  "evidence_refs": ["list of exact matching evidence_id strings evaluated chronologically"],\n'
        '  "findings": "Detailed longitudinal trajectory analysis enforcing Rule A and Rule B",\n'
        '  "identified_gaps": ["longitudinal retention gaps or untested foundational objectives"],\n'
        '  "recommended_action": "Longitudinal next steps (spaced review, fading scaffolding, etc.)",\n'
        '  "assessment_confidence": "low" | "moderate" | "high",\n'
        '  "limitations": "Limitations of historical or session coverage"\n'
        "}"
    )

    user_payload = {
        "concept_id": concept_id,
        "concept_name": concept_name,
        "evidence_records": records,
    }
    user_prompt = f"Analyze longitudinal learning trajectory for this evidence history:\n{json.dumps(user_payload, indent=2)}"

    parsed_llm = _call_llm_json(system_prompt, user_prompt, max_tokens=1000)
    if parsed_llm and isinstance(parsed_llm, dict):
        refs = _filter_evidence_refs(parsed_llm.get("evidence_refs", []), valid_ids)
        confidence = str(parsed_llm.get("assessment_confidence", "moderate")).lower()
        if confidence not in ("low", "moderate", "high"):
            confidence = "moderate"

        findings = str(parsed_llm.get("findings") or "Longitudinal trajectory analyzed.")
        gaps = [str(g) for g in parsed_llm.get("identified_gaps", []) if g]

        # Enforce Rule A in LLM response if single record
        if len(records) <= 1:
            if "rule a" not in findings.lower():
                findings += " Rule A enforced: Never assume MASTERED just because the single latest answer is correct if foundational objectives are untested."
            confidence = "low"
            if not any("longitudinal" in g.lower() for g in gaps):
                gaps.append("Longitudinal persistence across sessions is unverified.")

        return SpecialistAgentAssessment(
            agent_role="learning_observer",
            concept_id=concept_id,
            evidence_refs=refs,
            findings=findings,
            identified_gaps=gaps,
            recommended_action=str(parsed_llm.get("recommended_action") or ""),
            assessment_confidence=confidence,
            limitations=str(parsed_llm.get("limitations") or ""),
        )

    # ── High-Fidelity Heuristic Fallback ──────────────────────────────────────
    if not records:
        return SpecialistAgentAssessment(
            agent_role="learning_observer",
            concept_id=concept_id,
            evidence_refs=[],
            findings="No learning history recorded. Trajectory analysis impossible.",
            identified_gaps=["No historical learning observations recorded."],
            recommended_action="Begin establishing learning baseline with initial assessment.",
            assessment_confidence="low",
            limitations="Zero evidence records available.",
        )

    # Chronologically sort records (stable sort on created_at if available)
    sorted_records = list(records)
    try:
        sorted_records.sort(key=lambda r: r.get("created_at") or "")
    except Exception:
        pass

    # Rule A Check: Single latest answer is insufficient to assume mastered
    if len(sorted_records) <= 1:
        rec = sorted_records[0]
        eid = rec.get("evidence_id")
        findings = (
            f"Only a single learning observation is available ({eid or 'current record'}). "
            "In strict accordance with Rule A: Never assume MASTERED just because the single latest answer "
            "is correct if foundational objectives remain untested longitudinally. Retention across time and "
            "sessions has not been demonstrated."
        )
        gaps = [
            "Longitudinal persistence across sessions is unverified.",
            "Foundational breadth across multiple question forms remains untested over time.",
        ]
        action = f"Schedule a follow-up assessment for {concept_name} in a subsequent session to evaluate retention."
        confidence = "low"
        limitations = "Single observation point cannot prove persistent retention."
        return SpecialistAgentAssessment(
            agent_role="learning_observer",
            concept_id=concept_id,
            evidence_refs=valid_ids,
            findings=findings,
            identified_gaps=gaps,
            recommended_action=action,
            assessment_confidence=confidence,
            limitations=limitations,
        )

    # Multi-observation trajectory analysis
    early_records = sorted_records[: len(sorted_records) // 2]
    late_records = sorted_records[len(sorted_records) // 2 :]

    early_assisted = any(r.get("hints_used", 0) > 0 or r.get("solution_revealed_first", False) for r in early_records)
    late_assisted = any(r.get("hints_used", 0) > 0 or r.get("solution_revealed_first", False) for r in late_records)

    early_corrected = any(r.get("corrected_after_feedback", False) for r in early_records)
    late_unassisted_success = all(
        r.get("hints_used", 0) == 0 and not r.get("solution_revealed_first", False) for r in late_records
    )

    if early_assisted and late_unassisted_success:
        # Rule B applied: student made earlier mistakes or needed hints, but subsequent work is independent
        early_eids = [r["evidence_id"] for r in early_records if r.get("evidence_id")]
        late_eids = [r["evidence_id"] for r in late_records if r.get("evidence_id")]
        findings = (
            f"Positive learning progression verified. The student required hints or assistance in early attempts "
            f"({early_eids}), but demonstrated independent, unassisted understanding in subsequent attempts ({late_eids}). "
            "In strict accordance with Rule B: Never permanently penalize a student for an earlier mistake if "
            "subsequent independent evidence demonstrates they overcame the difficulty. The student has successfully "
            "internalized the concept."
        )
        gaps = []
        action = f"Reinforce mastery with a spaced retention check in 7 days."
        confidence = "high"
        limitations = "Historical errors safely superseded by verified unassisted success."
    elif late_assisted:
        # Ongoing scaffold dependency
        findings = (
            "Learning trajectory indicates persistent scaffold dependency. The student required hints or revealed "
            "solutions in recent attempts. Autonomous problem solving has not yet emerged."
        )
        gaps = ["Persistent scaffold dependency in recent attempts.", "Autonomous recall and application unproven."]
        action = "Gradually fade hint availability and prompt for unassisted problem solving on similar items."
        confidence = "moderate"
        limitations = "Recent assistance indicates concept is still in active acquisition phase."
    else:
        # Consistent unassisted performance across time
        distinct_sessions = {r.get("session_id") for r in sorted_records if r.get("session_id")}
        session_note = f"across {len(distinct_sessions)} distinct sessions" if len(distinct_sessions) > 1 else "within the current learning session"
        findings = (
            f"Sustained conceptual competence demonstrated {session_note}. Student maintains independent problem "
            "solving without regression or persistent misconceptions across consecutive trials."
        )
        gaps = []
        action = "Advance to next curriculum milestone or integrative capstone module."
        confidence = "high" if len(distinct_sessions) > 1 else "moderate"
        limitations = "Long-term delay (>14 days) retention check recommended to verify permanence."

    return SpecialistAgentAssessment(
        agent_role="learning_observer",
        concept_id=concept_id,
        evidence_refs=valid_ids,
        findings=findings,
        identified_gaps=gaps,
        recommended_action=action,
        assessment_confidence=confidence,
        limitations=limitations,
    )
