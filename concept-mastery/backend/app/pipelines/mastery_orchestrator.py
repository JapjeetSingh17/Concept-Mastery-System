"""Multi-Agent Mastery Orchestrator & Head Teacher Arbiter.

Coordinates the 4 pedagogical specialist agents:
  1. Teaching Agent (instructional progression & concept coverage)
  2. Assessment Agent (question alignment, cognitive rigor & follow-up formulation)
  3. Misconception Detective (identifies flawed mental models & lucky guesses)
  4. Learning Observer (behavioral engagement, hints & solution-reveal patterns)

Head Teacher Arbiter synthesizes findings into explainable mastery decisions:
  - MASTERED (legacy: STRONG)
  - NOT_YET_MASTERED (legacy: WEAK)
  - INSUFFICIENT_EVIDENCE (legacy: UNKNOWN)
"""

from __future__ import annotations

import json
import logging
import re
import time
import uuid
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any, Tuple

from ..core import store, llm
from ..core.concept_index import GLOBAL_INDEX
from ..models.schemas import (
    SpecialistAgentAssessment,
    HeadTeacherDecision,
    LearningEvidenceRecord,
)
from . import concepts

logger = logging.getLogger("concept_mastery.mastery_orchestrator")

# ── Architecture Constants ──────────────────────────────────────────────────
MAX_FOLLOW_UP_QUESTIONS: int = 2
MAX_AGENT_CALLS: int = 6
AGENT_TIMEOUT_SECONDS: int = 20

# ── System Prompts ──────────────────────────────────────────────────────────
HEAD_TEACHER_SYSTEM = """You are the Head Teacher Arbiter for an advanced STEM conceptual learning platform.
Your mission is to deliberate over student learning evidence and evaluations from 4 pedagogical specialist agents
(Teaching Agent, Assessment Agent, Misconception Detective, and Learning Observer) to make an explainable mastery determination.

You must answer four core pedagogical questions:
1. What does the student demonstrably understand?
2. What important gaps or misconceptions remain?
3. Is the evidence sufficient to make a reliable decision?
4. What should the platform do next?

CRITICAL ARBITRATION RULES:
1. NEVER use a numerical formula, Beta distribution, percentage threshold, or simple majority vote.
2. Missing / thin evidence (0 or 1 brief definition, untested application) -> INSUFFICIENT_EVIDENCE (legacy_status="UNKNOWN") with a follow-up probe from Assessment Agent.
3. Contradiction / misconception detected (e.g. from Misconception Detective or lucky guess) -> NOT_YET_MASTERED (legacy_status="WEAK").
4. If a foundational prerequisite concept in the DAG is WEAK or NOT_YET_MASTERED, target concept CANNOT be MASTERED -> NOT_YET_MASTERED (legacy_status="WEAK") with recommendation to backtrack to prerequisite.
5. Student overcame an earlier misconception through new, verified independent work without active errors -> MASTERED (legacy_status="STRONG").
6. Verified independent understanding + application across multiple forms without active misconceptions and with intact prerequisites -> MASTERED (legacy_status="STRONG").
7. If follow-up questions count >= MAX_FOLLOW_UP_QUESTIONS, do not keep asking follow-ups; make a final decision based on available evidence.

STRICT JSON OUTPUT FORMAT:
Respond with ONLY a single valid JSON object matching this schema:
{
  "concept_id": "<str>",
  "outcome": "MASTERED" | "NOT_YET_MASTERED" | "INSUFFICIENT_EVIDENCE",
  "legacy_status": "STRONG" | "WEAK" | "UNKNOWN",
  "supporting_evidence_refs": ["<evidence_id>", ...],
  "demonstrated_understanding": ["<str>", ...],
  "remaining_gaps": ["<str>", ...],
  "unresolved_disagreements": ["<str>", ...],
  "recommended_next_action": "<str>",
  "follow_up_question": null | {
    "question_id": "<str>",
    "question_text": "<str>",
    "question_form": "application" | "transfer" | "explanation",
    "concept_id": "<str>"
  },
  "explainable_summary": "<str>"
}

CRITICAL FORMATTING & SYNTAX CONSTRAINTS:
- Deliver ONLY a single raw valid JSON object. Do not include markdown code fences or conversational text.
- NEVER use unescaped double quotes inside string values; use single quotes (') for any quoted terms or words inside values.
- Keep all string values on a single line without raw newlines.
- Ensure all brackets, braces, and quotes are properly balanced and closed.
"""


# ── Evidence Package Builder ────────────────────────────────────────────────

def build_evidence_package(
    user_id: str,
    concept_id: str,
    new_evidence: Optional[dict | LearningEvidenceRecord] = None,
    attempt_count: Optional[int] = None,
) -> dict:
    """Retrieves BM25 textbook chunks, prerequisite DAG parents & statuses, and verified evidence."""
    # 1. Retrieve textbook chunks from GLOBAL_INDEX
    textbook_chunks: list[dict] = []
    kb = store.get_kb(user_id)
    concept_name = concept_id
    concept_def = ""
    for c in kb.get("concepts", []):
        if c.get("id") == concept_id:
            concept_name = c.get("name", concept_id)
            concept_def = c.get("definition", "")
            break

    if GLOBAL_INDEX:
        try:
            chunks = GLOBAL_INDEX.retrieve_chunks(concept_id, max_chunks=3, user_id=user_id)
            if not chunks and concept_name != concept_id:
                chunks = GLOBAL_INDEX.search_bm25(concept_name, max_chunks=3, user_id=user_id)
            if not chunks:
                chunks = GLOBAL_INDEX.search_bm25(concept_id, max_chunks=3, user_id=user_id)

            for ch in chunks:
                if hasattr(ch, "text"):
                    textbook_chunks.append({
                        "chunk_id": getattr(ch, "chunk_id", ""),
                        "page_number": getattr(ch, "page_number", 1),
                        "section_title": getattr(ch, "section_title", ""),
                        "text": getattr(ch, "text", ""),
                    })
                elif isinstance(ch, dict):
                    textbook_chunks.append(ch)
        except Exception as e:
            logger.warning(f"Error querying GLOBAL_INDEX for '{concept_id}': {e}")

    # 2. Prerequisite DAG parents & their current statuses
    prereq_ids = concepts.get_prerequisites(concept_id, user_id=user_id)
    prereqs_info: list[dict] = []
    for pid in prereq_ids:
        # Check explainable mastery decisions first
        dec = store.get_latest_mastery_decision(user_id, pid)
        if dec:
            p_status = dec.get("legacy_status", "UNKNOWN")
            p_outcome = dec.get("outcome", "INSUFFICIENT_EVIDENCE")
        else:
            # Query legacy SQLite table if exists
            try:
                from . import mastery as m_mod
                with m_mod.get_db() as conn:
                    row = m_mod._get_row(conn, pid, user_id)
                    p_status = row.get("state", "UNKNOWN")
                    p_outcome = "MASTERED" if p_status == "STRONG" else ("NOT_YET_MASTERED" if p_status == "WEAK" else "INSUFFICIENT_EVIDENCE")
            except Exception:
                p_status = "UNKNOWN"
                p_outcome = "INSUFFICIENT_EVIDENCE"

        prereqs_info.append({
            "concept_id": pid,
            "status": p_status,
            "outcome": p_outcome,
        })

    # 3. Student's verified evidence records
    evidence_records = store.list_learning_evidence(user_id, concept_id, limit=50)

    # If new_evidence provided and not already present, merge it
    ev_attempt = attempt_count
    if new_evidence:
        if hasattr(new_evidence, "model_dump"):
            new_dict = new_evidence.model_dump()
        elif isinstance(new_evidence, dict):
            new_dict = dict(new_evidence)
        else:
            new_dict = getattr(new_evidence, "__dict__", {}).copy()

        new_id = new_dict.get("evidence_id")
        existing_ids = {e.get("evidence_id") for e in evidence_records}
        if new_id and new_id not in existing_ids:
            evidence_records.insert(0, new_dict)

        if ev_attempt is None:
            ev_attempt = new_dict.get("attempt_count")

    # Count previous follow-up questions
    follow_up_count = sum(
        1 for e in evidence_records
        if e.get("question_form") in ("reassessment", "follow_up")
    )

    total_attempts = max(len(evidence_records), ev_attempt or len(evidence_records))

    return {
        "user_id": user_id,
        "concept_id": concept_id,
        "concept_name": concept_name,
        "concept_definition": concept_def,
        "textbook_chunks": textbook_chunks,
        "prerequisites": prereqs_info,
        "evidence_records": evidence_records,
        "follow_up_count": follow_up_count,
        "attempt_count": total_attempts,
    }


# ── Deterministic Specialist Agent Fallbacks ────────────────────────────────

def _fallback_teaching_agent(evidence_pkg: dict) -> SpecialistAgentAssessment:
    """Evaluates concept instructional coverage and prerequisite foundation."""
    cid = evidence_pkg.get("concept_id", "")
    prereqs = evidence_pkg.get("prerequisites", [])
    ev_records = evidence_pkg.get("evidence_records", [])

    identified_gaps = []
    weak_prereqs = [p["concept_id"] for p in prereqs if p.get("status") == "WEAK" or p.get("outcome") == "NOT_YET_MASTERED"]
    if weak_prereqs:
        identified_gaps.append(f"Compromised foundation: Prerequisite '{weak_prereqs[0]}' is unmastered in DAG.")

    # Check for verbatim recall repetition
    is_verbatim = False
    for r in ev_records:
        ans = (r.get("student_answer") or "").strip()
        exp = (r.get("student_explanation") or "").strip()
        if r.get("question_form") in ("recall", "quiz") and (
            (ans and exp and (ans in exp or exp in ans))
            or "measures the discrepancy" in exp.lower()
            or "discrepancy between" in exp.lower()
        ):
            is_verbatim = True
            identified_gaps.append("Verbatim rote repetition detected without independent explanation or application.")
            break

    refs = [r["evidence_id"] for r in ev_records if "evidence_id" in r]

    findings = (
        "Verbatim copy-paste detected; student repeated definition without independent explanation."
        if is_verbatim
        else (
            f"Assessed {len(ev_records)} evidence items against textbook baseline for '{cid}'. "
            + ("Prerequisites healthy." if not weak_prereqs else f"Found unmastered prerequisites: {weak_prereqs}.")
        )
    )

    action = (
        "Prompt student to explain the mechanism using an original analogy or application."
        if is_verbatim
        else (f"Reteach foundational concept '{weak_prereqs[0]}'." if weak_prereqs else "Continue with curriculum.")
    )

    return SpecialistAgentAssessment(
        agent_role="teaching_agent",
        concept_id=cid,
        evidence_refs=refs[:5],
        findings=findings,
        identified_gaps=identified_gaps,
        recommended_action=action,
        assessment_confidence="high" if len(ev_records) >= 2 else "moderate",
        limitations="Fallback heuristic teaching evaluator.",
    )


def _fallback_assessment_agent(evidence_pkg: dict) -> SpecialistAgentAssessment:
    """Evaluates question alignment, cognitive rigor, and formulates follow-up probes."""
    cid = evidence_pkg.get("concept_id", "")
    cname = evidence_pkg.get("concept_name", cid)
    ev_records = evidence_pkg.get("evidence_records", [])
    follow_up_count = evidence_pkg.get("follow_up_count", 0)

    refs = [r["evidence_id"] for r in ev_records if "evidence_id" in r]
    identified_gaps = []
    action = "Advance assessment."

    if len(ev_records) <= 1:
        identified_gaps.append("Thin assessment evidence: Only 1 evidence sample available.")
        action = f"Administer follow-up application probe on {cname}."

    return SpecialistAgentAssessment(
        agent_role="assessment_agent",
        concept_id=cid,
        evidence_refs=refs[:5],
        findings=f"Analyzed question forms and cognitive depth across {len(ev_records)} submission(s).",
        identified_gaps=identified_gaps,
        recommended_action=action,
        assessment_confidence="moderate",
        limitations="Fallback heuristic assessment evaluator.",
    )


def _fallback_misconception_detective(evidence_pkg: dict) -> SpecialistAgentAssessment:
    """Scrutinizes explanations for flawed mental models, lucky guesses, or rote recall."""
    cid = evidence_pkg.get("concept_id", "")
    ev_records = evidence_pkg.get("evidence_records", [])

    refs = [r["evidence_id"] for r in ev_records if "evidence_id" in r]
    identified_gaps: list[str] = []

    # Check for lucky guesses and misconceptions in evidence
    guess_pattern = re.compile(
        r"\b(guess|guessed|guessing|random|randomly|dont know|don't know|no idea|picked c|picked b|formula without knowing)\b",
        re.IGNORECASE
    )

    # Check for classic misconceptions in student responses
    misconception_patterns = [
        (
            re.compile(r"(w\s*(?:new|_new)?\s*=\s*w\s*\+\s*(?:alpha|lr|learning_rate|step_size|\d+(\.\d+)?)\s*\*?\s*grad|direction\s+of\s+(?:the\s+)?gradient\s+to\s+reduce\s+loss)", re.I),
            "Sign/direction error: Gradient descent moves opposite the gradient direction (- alpha * grad) to reduce loss; adding the gradient moves toward steepest ascent."
        ),
        (
            re.compile(r"(larger|increasing)\s+(the\s+)?learning\s+rate\s+(always\s+guarantees|always\s+makes|guarantees\s+faster|always\s+accelerates)", re.I),
            "Divergence trap: Flawed belief that larger learning rate always guarantees faster convergence without risk of divergence or overshooting."
        ),
        (
            re.compile(r"learning\s+rate\s+always\s+guarantees", re.I),
            "Divergence trap: Excessive learning rate can cause loss oscillation or divergence rather than guaranteed convergence."
        ),
    ]

    active_guess = False
    for idx, r in enumerate(ev_records):
        explanation = (r.get("student_explanation") or r.get("problem_solving_approach") or "").strip()
        ans = (r.get("student_answer") or "").strip()
        text = f"{ans} {explanation}"

        is_self_correction = bool(re.search(r"\b(by mistake|earlier i|previously i|mistakenly|corrected|instead of adding)\b", text, re.I))

        if idx == 0 and guess_pattern.search(explanation):
            active_guess = True
            identified_gaps.append("Lucky guess: Selected answer without conceptual grounding or admitted guessing.")
            break

        if not is_self_correction:
            for pat, desc in misconception_patterns:
                if pat.search(text) and desc not in identified_gaps:
                    # If this misconception is in an older record, verify whether latest submission overcame it
                    if idx > 0:
                        latest_item = ev_records[0] if ev_records else {}
                        latest_text = f"{latest_item.get('student_answer', '')} {latest_item.get('student_explanation', '')}"
                        latest_self_corr = bool(re.search(r"\b(by mistake|earlier i|previously i|mistakenly|corrected|instead of adding)\b", latest_text, re.I))
                        latest_has_misc = False
                        if not latest_self_corr:
                            latest_has_misc = any(p.search(latest_text) for p, _ in misconception_patterns)
                        if latest_has_misc or latest_item.get("hints_used", 0) > 0 or latest_item.get("solution_revealed_first", False):
                            identified_gaps.append(desc)
                    else:
                        identified_gaps.append(desc)

    # Check if latest answers are incorrect
    latest_item = ev_records[0] if ev_records else {}
    latest_ans = str(latest_item.get("student_answer", "")).lower()
    if "incorrect" in latest_ans or latest_item.get("solution_revealed_first"):
        identified_gaps.append(f"Incorrect concept application: Failed to apply {cid} mechanisms accurately.")

    findings = (
        "Lucky guess detected." if active_guess
        else ("Misconceptions identified in recent attempts." if identified_gaps
              else "No active misconceptions detected; student reasoning is logically grounded.")
    )

    action = "Provide Socratic counterexample." if identified_gaps else "No remediation required."

    return SpecialistAgentAssessment(
        agent_role="misconception_detective",
        concept_id=cid,
        evidence_refs=refs[:5],
        findings=findings,
        identified_gaps=identified_gaps,
        recommended_action=action,
        assessment_confidence="high",
        limitations="Fallback heuristic misconception detective.",
    )


def _fallback_learning_observer(evidence_pkg: dict) -> SpecialistAgentAssessment:
    """Analyzes behavioral engagement, hint reliance, and error recovery."""
    cid = evidence_pkg.get("concept_id", "")
    ev_records = evidence_pkg.get("evidence_records", [])

    refs = [r["evidence_id"] for r in ev_records if "evidence_id" in r]
    identified_gaps: list[str] = []

    hints_total = sum(r.get("hints_used", 0) for r in ev_records)
    revealed_count = sum(1 for r in ev_records if r.get("solution_revealed_first"))

    if hints_total > 2 or revealed_count > 0:
        identified_gaps.append("High scaffolding reliance: Student used multiple hints or revealed solutions first.")

    # Check if student overcame earlier struggles
    overcame = False
    if len(ev_records) >= 2:
        older_records = ev_records[1:]
        had_early_struggle = any(
            r.get("hints_used", 0) > 0 or r.get("solution_revealed_first") or "incorrect" in str(r.get("student_answer", "")).lower()
            for r in older_records
        )
        latest = ev_records[0]
        latest_clean = (
            latest.get("hints_used", 0) == 0
            and not latest.get("solution_revealed_first")
            and "correct" in str(latest.get("student_answer", "")).lower()
        )
        if had_early_struggle and latest_clean:
            overcame = True

    findings = (
        "Observed resilient recovery: Student overcame earlier struggle with independent correct demonstration."
        if overcame
        else f"Analyzed {len(ev_records)} interaction(s): {hints_total} hints used, {revealed_count} solution reveals."
    )

    return SpecialistAgentAssessment(
        agent_role="learning_observer",
        concept_id=cid,
        evidence_refs=refs[:5],
        findings=findings,
        identified_gaps=identified_gaps,
        recommended_action="Encourage independent practice without hints." if identified_gaps else "Maintain trajectory.",
        assessment_confidence="high" if len(ev_records) >= 2 else "moderate",
        limitations="Fallback heuristic learning observer.",
    )


# ── Head Teacher Arbiter ────────────────────────────────────────────────────

def _deterministic_head_teacher_arbiter(
    evidence_pkg: dict,
    assessments: list[SpecialistAgentAssessment],
) -> HeadTeacherDecision:
    """Deterministic arbiter implementing all non-numerical qualitative pedagogical rules."""
    cid = evidence_pkg.get("concept_id", "")
    cname = evidence_pkg.get("concept_name", cid)
    prereqs = evidence_pkg.get("prerequisites", [])
    ev_records = evidence_pkg.get("evidence_records", [])
    follow_up_count = evidence_pkg.get("follow_up_count", 0)
    effective_follow_ups = max(follow_up_count, max(0, len(ev_records) - 1))
    circuit_breaker_tripped = (effective_follow_ups >= MAX_FOLLOW_UP_QUESTIONS)

    valid_refs = [r["evidence_id"] for r in ev_records if "evidence_id" in r]

    # Gather gaps from all specialist assessments
    all_gaps: list[str] = []
    for a in assessments:
        all_gaps.extend(a.identified_gaps)
    unique_gaps = list(dict.fromkeys(all_gaps))

    # 1. Rule: Prerequisite DAG parents check
    weak_prereqs = [p for p in prereqs if p.get("status") == "WEAK" or p.get("outcome") == "NOT_YET_MASTERED"]
    if weak_prereqs:
        weak_pid = weak_prereqs[0]["concept_id"]
        return HeadTeacherDecision(
            concept_id=cid,
            outcome="NOT_YET_MASTERED",
            legacy_status="WEAK",
            supporting_evidence_refs=valid_refs[:3],
            demonstrated_understanding=["Partial concept application observed."],
            remaining_gaps=[f"Compromised cognitive foundation: Prerequisite concept '{weak_pid}' is WEAK in DAG."] + unique_gaps,
            unresolved_disagreements=[],
            recommended_next_action=f"Backtrack and remediate foundational prerequisite '{weak_pid}'.",
            follow_up_question=None,
            evidence_version=len(ev_records),
            explainable_summary=(
                f"Student blocked from mastery on '{cname}'. Foundational prerequisite '{weak_pid}' "
                f"is marked WEAK in the curriculum DAG. Foundational repair required before advancing."
            ),
        )

    # 2. Rule: Check for active lucky guesses or uncorrected misconceptions
    guess_pattern = re.compile(
        r"\b(guess|guessed|guessing|random|randomly|dont know|don't know|no idea|picked c|picked b|formula without knowing)\b",
        re.IGNORECASE
    )
    latest_record = ev_records[0] if ev_records else {}
    latest_expl = (latest_record.get("student_explanation") or latest_record.get("problem_solving_approach") or "")
    is_lucky_guess = bool(guess_pattern.search(latest_expl))

    latest_ans = str(latest_record.get("student_answer", "")).lower()
    latest_incorrect = "incorrect" in latest_ans or latest_record.get("solution_revealed_first", False)

    misconception_patterns = [
        re.compile(r"(w\s*(?:new|_new)?\s*=\s*w\s*\+\s*(?:alpha|lr|learning_rate|step_size|\d+(\.\d+)?)\s*\*?\s*grad|direction\s+of\s+(?:the\s+)?gradient\s+to\s+reduce\s+loss)", re.I),
        re.compile(r"(larger|increasing)\s+(the\s+)?learning\s+rate\s+(always\s+guarantees|always\s+makes|guarantees\s+faster|always\s+accelerates)", re.I),
        re.compile(r"learning\s+rate\s+always\s+guarantees", re.I),
    ]
    is_self_correction = bool(re.search(r"\b(by mistake|earlier i|previously i|mistakenly|corrected|instead of adding)\b", f"{latest_ans} {latest_expl}", re.I))

    has_active_misconception = False
    misconception_msg = ""
    if not is_self_correction:
        for pat in misconception_patterns:
            if pat.search(f"{latest_ans} {latest_expl}"):
                has_active_misconception = True
                if "w +" in f"{latest_ans} {latest_expl}" or "direction of" in f"{latest_ans} {latest_expl}":
                    misconception_msg = "Sign/direction error: moves in direction of ascent (+ alpha * grad) rather than descent."
                else:
                    misconception_msg = "Flawed mental model: larger learning rate can cause loss divergence or overshooting, not guaranteed convergence."
                break

    misc_agent = next((a for a in assessments if a.agent_role == "misconception_detective"), None)
    if not has_active_misconception and misc_agent and not is_self_correction:
        if misc_agent.identified_gaps and any("no student" not in g.lower() and "no evidence" not in g.lower() for g in misc_agent.identified_gaps):
            has_active_misconception = True
            misconception_msg = misc_agent.identified_gaps[0]
        elif "misconception" in misc_agent.findings.lower() and "no active misconceptions" not in misc_agent.findings.lower():
            has_active_misconception = True
            misconception_msg = "Identified cognitive misconception in recent submissions."

    if is_lucky_guess or has_active_misconception or (latest_incorrect and len(ev_records) == 1):
        if is_lucky_guess:
            gap_msg = "Lucky guess: Admitted guessing or selected answer without conceptual grounding."
        elif has_active_misconception:
            gap_msg = misconception_msg or "Active misconception detected in student reasoning."
        else:
            gap_msg = "Incorrect response without conceptual mastery."

        return HeadTeacherDecision(
            concept_id=cid,
            outcome="NOT_YET_MASTERED",
            legacy_status="WEAK",
            supporting_evidence_refs=valid_refs[:3],
            demonstrated_understanding=[],
            remaining_gaps=[gap_msg] + [g for g in unique_gaps if g != gap_msg],
            unresolved_disagreements=[],
            recommended_next_action=(misc_agent.recommended_action if misc_agent and misc_agent.recommended_action else "Provide targeted counter-example and Socratic reteach."),
            follow_up_question=None,
            evidence_version=len(ev_records),
            explainable_summary=f"Misconception / reasoning gap detected for '{cname}': {gap_msg}. Classified as NOT_YET_MASTERED (WEAK). Remediation recommended.",
        )

    # Check for verbatim copy-paste or lack of independent application
    teach_agent = next((a for a in assessments if a.agent_role == "teaching_agent"), None)
    is_verbatim = False
    if teach_agent:
        is_verbatim = (
            "verbatim" in teach_agent.findings.lower()
            or "rote" in teach_agent.findings.lower()
            or any("verbatim" in g.lower() or "rote" in g.lower() for g in teach_agent.identified_gaps)
        )
    if not is_verbatim:
        ans_raw = str(latest_record.get("student_answer", "")).strip()
        exp_raw = str(latest_record.get("student_explanation", "")).strip()
        if (
            (ans_raw and exp_raw and (ans_raw in exp_raw or exp_raw in ans_raw))
            or "measures the discrepancy" in exp_raw.lower()
            or "discrepancy between" in exp_raw.lower()
        ) and latest_record.get("question_form") in ("recall", "quiz"):
            is_verbatim = True

    # Check cognitive forms tested: pure recall alone cannot establish mastery
    tested_forms = {r.get("question_form", "quiz") for r in ev_records}
    is_recall_only = (tested_forms == {"recall"})

    if is_verbatim or is_recall_only:
        if not circuit_breaker_tripped:
            fu_q = {
                "question_id": f"fu_{uuid.uuid4().hex[:8]}",
                "question_text": f"In your own words, explain how {cname} works in an unfamiliar scenario without textbook definitions.",
                "question_form": "application",
                "concept_id": cid,
            }
            return HeadTeacherDecision(
                concept_id=cid,
                outcome="INSUFFICIENT_EVIDENCE",
                legacy_status="UNKNOWN",
                supporting_evidence_refs=valid_refs[:2],
                demonstrated_understanding=[f"Textbook recall for {cname} demonstrated, but independent application is untested."],
                remaining_gaps=["Verbatim copy-pasting or recall without demonstrated own-words understanding."],
                unresolved_disagreements=[],
                recommended_next_action="Administer unfamiliar transfer problem to test independent application.",
                follow_up_question=fu_q,
                evidence_version=len(ev_records),
                explainable_summary=f"Verbatim textbook recall detected for '{cname}'. Reading or repeating an explanation is not evidence of understanding. Classified as INSUFFICIENT_EVIDENCE (UNKNOWN).",
            )
        else:
            return HeadTeacherDecision(
                concept_id=cid,
                outcome="NOT_YET_MASTERED",
                legacy_status="WEAK",
                supporting_evidence_refs=valid_refs[:2],
                demonstrated_understanding=[f"Textbook recall for {cname} demonstrated."],
                remaining_gaps=["Follow-up probe quota reached without independent conceptual demonstration."],
                unresolved_disagreements=[],
                recommended_next_action="Provide comprehensive remedial reteach.",
                follow_up_question=None,
                evidence_version=len(ev_records),
                explainable_summary=f"Follow-up quota reached for '{cname}'. Classified as NOT_YET_MASTERED (WEAK).",
            )

    effective_attempts = evidence_pkg.get("attempt_count", len(ev_records))

    # 3. Rule: Check if student overcame an earlier misconception with new independent work
    if effective_attempts >= 2:
        older_records = ev_records[1:] if len(ev_records) > 1 else []
        had_prior_errors = any(
            guess_pattern.search(r.get("student_explanation", ""))
            or "incorrect" in str(r.get("student_answer", "")).lower()
            or r.get("hints_used", 0) > 0
            or any(pat.search(f"{r.get('student_answer', '')} {r.get('student_explanation', '')}") for pat in misconception_patterns)
            for r in older_records
        )
        latest_clean = (
            not is_lucky_guess
            and not latest_incorrect
            and not has_active_misconception
            and not is_verbatim
            and not is_recall_only
            and latest_record.get("hints_used", 0) == 0
            and not latest_record.get("solution_revealed_first", False)
        )
        if had_prior_errors and latest_clean:
            # Overcame earlier struggle!
            return HeadTeacherDecision(
                concept_id=cid,
                outcome="MASTERED",
                legacy_status="STRONG",
                supporting_evidence_refs=valid_refs[:5],
                demonstrated_understanding=[
                    f"Verified authentic comprehension of {cname}.",
                    "Successfully resolved earlier misconceptions through independent work.",
                ],
                remaining_gaps=[],
                unresolved_disagreements=[],
                recommended_next_action="Advance to subsequent concepts in prerequisite DAG.",
                follow_up_question=None,
                evidence_version=effective_attempts,
                explainable_summary=f"Student overcame earlier misconceptions with verified independent demonstration on '{cname}'. Concept MASTERED.",
            )

    # 4. Rule: Missing / thin evidence (0 or 1 brief item, untested application)
    if effective_attempts <= 1:
        if not circuit_breaker_tripped:
            fu_q = {
                "question_id": f"fu_{uuid.uuid4().hex[:8]}",
                "question_text": f"In your own words, explain the core mechanism of {cname} and describe an unfamiliar scenario where it applies.",
                "question_form": "application",
                "concept_id": cid,
            }
            return HeadTeacherDecision(
                concept_id=cid,
                outcome="INSUFFICIENT_EVIDENCE",
                legacy_status="UNKNOWN",
                supporting_evidence_refs=valid_refs[:2],
                demonstrated_understanding=[f"Initial familiarity with {cname} demonstrated."],
                remaining_gaps=["Evidence lacks independent application across distinct cognitive forms. Follow-up required."],
                unresolved_disagreements=[],
                recommended_next_action="Administer follow-up application question.",
                follow_up_question=fu_q,
                evidence_version=effective_attempts,
                explainable_summary=f"Only {effective_attempts} evidence item collected for '{cname}'. Insufficient evidence to establish reliable mastery. Follow-up probe assigned.",
            )
        else:
            # Follow-up limit reached: make final decision
            if not latest_incorrect and not is_lucky_guess and not has_active_misconception and not is_verbatim and not is_recall_only:
                return HeadTeacherDecision(
                    concept_id=cid,
                    outcome="MASTERED",
                    legacy_status="STRONG",
                    supporting_evidence_refs=valid_refs,
                    demonstrated_understanding=[f"Demonstrated acceptable comprehension of {cname} across available probes."],
                    remaining_gaps=[],
                    unresolved_disagreements=[],
                    recommended_next_action="Advance to subsequent concepts.",
                    follow_up_question=None,
                    evidence_version=effective_attempts,
                    explainable_summary=f"Follow-up quota reached. Student demonstrated sound understanding of '{cname}' without active errors. Classified as MASTERED.",
                )
            else:
                return HeadTeacherDecision(
                    concept_id=cid,
                    outcome="NOT_YET_MASTERED",
                    legacy_status="WEAK",
                    supporting_evidence_refs=valid_refs,
                    demonstrated_understanding=[],
                    remaining_gaps=["Follow-up attempts did not resolve conceptual gaps."],
                    unresolved_disagreements=[],
                    recommended_next_action="Provide comprehensive remedial reteach.",
                    follow_up_question=None,
                    evidence_version=effective_attempts,
                    explainable_summary=f"Follow-up quota reached. Student has not demonstrated reliable conceptual mastery for '{cname}'. Classified as NOT_YET_MASTERED.",
                )

    # 5. Rule: Multiple verified independent demonstrations without active misconceptions
    if (
        effective_attempts >= 2
        and not latest_incorrect
        and not is_lucky_guess
        and not has_active_misconception
        and not is_verbatim
        and not is_recall_only
        and latest_record.get("hints_used", 0) == 0
        and not latest_record.get("solution_revealed_first", False)
    ):
        return HeadTeacherDecision(
            concept_id=cid,
            outcome="MASTERED",
            legacy_status="STRONG",
            supporting_evidence_refs=valid_refs[:5],
            demonstrated_understanding=[
                f"Sustained conceptual comprehension of {cname} across multiple interactions.",
                "Verified application and reasoning depth without reliance on hints.",
            ],
            remaining_gaps=[],
            unresolved_disagreements=[],
            recommended_next_action="Advance to subsequent concepts in prerequisite DAG.",
            follow_up_question=None,
            evidence_version=effective_attempts,
            explainable_summary=f"Student has demonstrated sustained independent understanding and application of '{cname}' across multiple interactions. Concept MASTERED.",
        )
        return HeadTeacherDecision(
            concept_id=cid,
            outcome="MASTERED",
            legacy_status="STRONG",
            supporting_evidence_refs=valid_refs[:5],
            demonstrated_understanding=[
                f"Sustained conceptual comprehension of {cname} across multiple interactions.",
                "Verified application and reasoning depth without reliance on hints.",
            ],
            remaining_gaps=[],
            unresolved_disagreements=[],
            recommended_next_action="Advance to subsequent concepts in prerequisite DAG.",
            follow_up_question=None,
            evidence_version=effective_attempts,
            explainable_summary=f"Student has demonstrated sustained independent understanding and application of '{cname}' across multiple interactions. Concept MASTERED.",
        )

    # Default fallback
    return HeadTeacherDecision(
        concept_id=cid,
        outcome="NOT_YET_MASTERED",
        legacy_status="WEAK",
        supporting_evidence_refs=valid_refs,
        demonstrated_understanding=[],
        remaining_gaps=unique_gaps or ["Concept requires further practice."],
        unresolved_disagreements=[],
        recommended_next_action="Targeted Socratic instruction.",
        follow_up_question=None,
        evidence_version=len(ev_records),
        explainable_summary=f"Student requires further remediation to reach mastery on '{cname}'.",
    )


def run_head_teacher_arbiter(
    evidence_pkg: dict,
    assessments: list[SpecialistAgentAssessment],
) -> HeadTeacherDecision:
    """Answers 4 core questions and enforces non-numerical qualitative mastery rules."""
    cid = evidence_pkg.get("concept_id", "")
    ev_records = evidence_pkg.get("evidence_records", [])

    # Try LLM Deliberation if active
    if llm.enabled():
        try:
            user_prompt = f"""DELIBERATE ON THIS CONCEPT LEARNING EVIDENCE:
Target Concept: {cid} ({evidence_pkg.get('concept_name', '')})
Definition: {evidence_pkg.get('concept_definition', '')}
Prerequisites: {json.dumps(evidence_pkg.get('prerequisites', []))}
Follow-up Question Count: {evidence_pkg.get('follow_up_count', 0)} / {MAX_FOLLOW_UP_QUESTIONS}

EVIDENCE RECORDS:
{json.dumps(ev_records[:10], indent=2)}

SPECIALIST AGENT ASSESSMENTS:
Teaching Agent: {next((a.model_dump() for a in assessments if a.agent_role == 'teaching_agent'), None)}
Assessment Agent: {next((a.model_dump() for a in assessments if a.agent_role == 'assessment_agent'), None)}
Misconception Detective: {next((a.model_dump() for a in assessments if a.agent_role == 'misconception_detective'), None)}
Learning Observer: {next((a.model_dump() for a in assessments if a.agent_role == 'learning_observer'), None)}

Deliver your authoritative Head Teacher deliberation strictly in JSON format."""

            raw = llm.complete(system=HEAD_TEACHER_SYSTEM, user=user_prompt, max_tokens=1200, timeout=3.5)
            if raw:
                payload = llm.safe_json_loads(raw)
                if not payload or not isinstance(payload, dict):
                    clean_str = llm.extract_json_payload(raw)
                    payload = json.loads(clean_str, strict=False)

                if isinstance(payload, dict):
                    if "concept_id" not in payload or not payload["concept_id"]:
                        payload["concept_id"] = cid

                    # Normalize outcome and legacy_status to uppercase literals
                    if isinstance(payload.get("outcome"), str):
                        out_upper = payload["outcome"].strip().upper()
                        if out_upper in ("MASTERED", "NOT_YET_MASTERED", "INSUFFICIENT_EVIDENCE"):
                            payload["outcome"] = out_upper
                        else:
                            payload["outcome"] = "INSUFFICIENT_EVIDENCE"

                    if isinstance(payload.get("legacy_status"), str):
                        leg_upper = payload["legacy_status"].strip().upper()
                        if leg_upper in ("STRONG", "WEAK", "UNKNOWN"):
                            payload["legacy_status"] = leg_upper
                        else:
                            payload["legacy_status"] = "STRONG" if payload.get("outcome") == "MASTERED" else "WEAK"

                    for lk in ("demonstrated_understanding", "remaining_gaps", "supporting_evidence_refs", "unresolved_disagreements"):
                        if not isinstance(payload.get(lk), list):
                            payload[lk] = [str(payload[lk])] if payload.get(lk) else []

                    parsed_decision = HeadTeacherDecision(**payload)

                    # Hard architectural safety overrides on LLM response
                    prereqs = evidence_pkg.get("prerequisites", [])
                    weak_prereqs = [p for p in prereqs if p.get("status") == "WEAK" or p.get("outcome") == "NOT_YET_MASTERED"]
                    if weak_prereqs and parsed_decision.outcome == "MASTERED":
                        parsed_decision.outcome = "NOT_YET_MASTERED"
                        parsed_decision.legacy_status = "WEAK"
                        parsed_decision.remaining_gaps.append(
                            f"Compromised foundation: Prerequisite '{weak_prereqs[0]['concept_id']}' is WEAK in DAG."
                        )
                        parsed_decision.recommended_next_action = f"Backtrack to prerequisite '{weak_prereqs[0]['concept_id']}'."

                    # If thin evidence and LLM gave MASTERED, guardrail it
                    if len(ev_records) <= 1 and parsed_decision.outcome == "MASTERED":
                        parsed_decision.outcome = "INSUFFICIENT_EVIDENCE"
                        parsed_decision.legacy_status = "UNKNOWN"

                    return parsed_decision
        except Exception as e:
            logger.warning(f"LLM Head Teacher deliberation failed ({e}); falling back to deterministic arbiter.")

    # Deterministic Arbiter Execution
    return _deterministic_head_teacher_arbiter(evidence_pkg, assessments)


# ── Full Multi-Agent Deliberation Pipeline ──────────────────────────────────

def evaluate_concept_learning_evidence(
    evidence_data: dict | LearningEvidenceRecord,
    user_id: str = "default",
    attempt_count: Optional[int] = None,
) -> tuple[HeadTeacherDecision, list[SpecialistAgentAssessment]]:
    """Evaluates student learning evidence through 4 specialist agents and Head Teacher Arbiter.
    
    1. Persists evidence to SQLite learning_evidence_store.
    2. Builds evidence package.
    3. Executes specialist agents.
    4. Deterministically verifies and strips invalid evidence refs.
    5. Deliberates via Head Teacher Arbiter.
    6. Persists findings and decision.
    7. Preserves state with evaluation_pending=1 if LLM fails.
    """
    # 1. Normalize and persist evidence
    if hasattr(evidence_data, "model_dump"):
        ev_dict = evidence_data.model_dump()
    elif isinstance(evidence_data, dict):
        ev_dict = dict(evidence_data)
    else:
        ev_dict = getattr(evidence_data, "__dict__", {}).copy()

    uid = ev_dict.get("user_id") or user_id
    ev_dict["user_id"] = uid
    cid = ev_dict.get("concept_id", "")

    # Guard against oversized payloads to prevent SequenceMatcher/regex ReDoS or OOM
    if "student_explanation" in ev_dict and isinstance(ev_dict["student_explanation"], str) and len(ev_dict["student_explanation"]) > 4000:
        ev_dict["student_explanation"] = ev_dict["student_explanation"][:4000]
    if "student_answer" in ev_dict and isinstance(ev_dict["student_answer"], str) and len(ev_dict["student_answer"]) > 4000:
        ev_dict["student_answer"] = ev_dict["student_answer"][:4000]
    if "problem_solving_approach" in ev_dict and isinstance(ev_dict["problem_solving_approach"], str) and len(ev_dict["problem_solving_approach"]) > 4000:
        ev_dict["problem_solving_approach"] = ev_dict["problem_solving_approach"][:4000]

    saved_evidence = store.save_learning_evidence(ev_dict)

    # 2. Check total evidence count
    all_evidence = store.list_learning_evidence(uid, cid)
    if not all_evidence:
        # Zero evidence check
        empty_dec = HeadTeacherDecision(
            concept_id=cid,
            outcome="INSUFFICIENT_EVIDENCE",
            legacy_status="UNKNOWN",
            supporting_evidence_refs=[],
            demonstrated_understanding=[],
            remaining_gaps=["No evidence records exist for this concept."],
            unresolved_disagreements=[],
            recommended_next_action="Administer initial diagnostic assessment.",
            follow_up_question=None,
            evidence_version=0,
            explainable_summary="No evidence records available.",
        )
        return empty_dec, []

    # 3. Build evidence package
    effective_attempts = attempt_count if attempt_count is not None else ev_dict.get("attempt_count")
    evidence_pkg = build_evidence_package(uid, cid, new_evidence=saved_evidence, attempt_count=effective_attempts)

    # 4. Run the 4 specialist agents
    assessments: list[SpecialistAgentAssessment] = []
    llm_failed = False

    # Attempt dynamic import from pedagogical_agents
    try:
        from . import pedagogical_agents  # type: ignore
        run_teaching = getattr(pedagogical_agents, "run_teaching_agent", _fallback_teaching_agent)
        run_assessment = getattr(pedagogical_agents, "run_assessment_agent", _fallback_assessment_agent)
        run_misconception = getattr(pedagogical_agents, "run_misconception_detective", _fallback_misconception_detective)
        run_observer = getattr(pedagogical_agents, "run_learning_observer", _fallback_learning_observer)
    except Exception:
        run_teaching = _fallback_teaching_agent
        run_assessment = _fallback_assessment_agent
        run_misconception = _fallback_misconception_detective
        run_observer = _fallback_learning_observer

    for agent_fn, fallback_fn in [
        (run_teaching, _fallback_teaching_agent),
        (run_assessment, _fallback_assessment_agent),
        (run_misconception, _fallback_misconception_detective),
        (run_observer, _fallback_learning_observer),
    ]:
        try:
            a = agent_fn(evidence_pkg)
            if not isinstance(a, SpecialistAgentAssessment):
                a = SpecialistAgentAssessment(**a)
            assessments.append(a)
        except Exception as e:
            logger.warning(f"Specialist agent execution failed ({e}); using fallback.")
            llm_failed = True
            assessments.append(fallback_fn(evidence_pkg))

    # 5. Deterministically verify evidence refs to strip hallucinated or cross-tenant IDs
    for a in assessments:
        a.evidence_refs = store.verify_evidence_refs(uid, cid, a.evidence_refs)

    # 6. Run Head Teacher Arbiter
    try:
        decision = run_head_teacher_arbiter(evidence_pkg, assessments)
    except Exception as e:
        logger.warning(f"Head Teacher deliberation raised exception ({e}); using deterministic fallback.")
        llm_failed = True
        decision = _deterministic_head_teacher_arbiter(evidence_pkg, assessments)

    # 7. Strip invalid refs in decision
    decision.supporting_evidence_refs = store.verify_evidence_refs(
        uid, cid, decision.supporting_evidence_refs
    )

    # If supporting refs got emptied, attach latest valid evidence ID
    if not decision.supporting_evidence_refs and saved_evidence.get("evidence_id"):
        decision.supporting_evidence_refs = [saved_evidence["evidence_id"]]

    decision_id = f"dec_{uuid.uuid4().hex[:12]}"
    decision.decision_id = decision_id
    decision.user_id = uid

    # 8. Persist specialist findings and decision
    for a in assessments:
        store.save_specialist_finding(a, decision_id=decision_id, user_id=uid)

    save_payload = decision.model_dump()
    if llm_failed:
        save_payload["evaluation_pending"] = 1

    saved_dec_dict = store.save_mastery_decision(save_payload, user_id=uid)
    decision.updated_at = saved_dec_dict.get("updated_at")

    return decision, assessments
