"""Closed-Loop LangGraph Orchestrator.

Implements an agentic cycle of:
  Diagnosis -> Assessment -> Evaluation -> Socratic Remediation / Advancement -> Re-assessment.

Uses LangGraph StateGraph with conditional edges and loop-termination guards.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Dict, Any, Literal
from langgraph.graph import StateGraph, END

from ..core import store
from ..models.schemas import LoopState, MasteryStatus, QuizSubmission, TutorChatResponse
from . import concepts, mastery, quiz, tutor, tools

logger = logging.getLogger("concept_mastery.loop")


class TutorResponse(TutorChatResponse):
    """Container for tutor responses within the closed loop, containing artifacts and external_links."""
    pass

MAX_REMEDIAL_ITERATIONS = 3

# ── Lifecycle Workflow States ──────────────────────────────────────────────
LIFECYCLE_AWAITING_UPLOAD = "AWAITING_UPLOAD"
LIFECYCLE_BASELINE_DIAGNOSTIC = "BASELINE_DIAGNOSTIC"
LIFECYCLE_ACTIVE_LEARNING = "ACTIVE_LEARNING"


def _init_lifecycle_table():
    with mastery.get_db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS user_lifecycle (
                user_id                     TEXT PRIMARY KEY,
                lifecycle_state             TEXT NOT NULL DEFAULT 'AWAITING_UPLOAD',
                diagnostic_completed_at     TEXT,
                updated_at                  TEXT
            )
        """)


def get_lifecycle_state(user_id: str = "default") -> str:
    """Return learner loop lifecycle state: AWAITING_UPLOAD, BASELINE_DIAGNOSTIC, or ACTIVE_LEARNING."""
    mats = store.list_study_materials(user_id)
    if not mats:
        return LIFECYCLE_AWAITING_UPLOAD

    _init_lifecycle_table()
    with mastery.get_db() as conn:
        row = conn.execute("SELECT lifecycle_state FROM user_lifecycle WHERE user_id = ?", (user_id,)).fetchone()
        if row and row["lifecycle_state"] == LIFECYCLE_ACTIVE_LEARNING:
            return LIFECYCLE_ACTIVE_LEARNING

    return LIFECYCLE_BASELINE_DIAGNOSTIC


def set_lifecycle_state(user_id: str, state: str):
    """Set lifecycle state for user_id in user_lifecycle table."""
    _init_lifecycle_table()
    now_iso = datetime.now(timezone.utc).isoformat()
    diag_at = now_iso if state == LIFECYCLE_ACTIVE_LEARNING else None
    with mastery.get_db() as conn:
        conn.execute("""
            INSERT INTO user_lifecycle (user_id, lifecycle_state, diagnostic_completed_at, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET
                lifecycle_state = excluded.lifecycle_state,
                diagnostic_completed_at = COALESCE(user_lifecycle.diagnostic_completed_at, excluded.diagnostic_completed_at),
                updated_at = excluded.updated_at
        """, (user_id, state, diag_at, now_iso))


# ── Node Implementations ───────────────────────────────────────────────────

def diagnose_node(state: LoopState) -> LoopState:
    """Step 1 (Diagnose): Select the earliest unmastered or WEAK node from prerequisite DAG."""
    user_id = state.get("user_id", "default")
    kb = store.get_kb(user_id)
    if not kb.get("ready") and store.KB.get("ready"):
        kb = store.KB
    if not kb.get("ready"):
        state["next_step"] = "COMPLETE"
        state["error"] = "No active knowledge base loaded."
        return state

    # Pull topological ordering of concepts
    path = concepts.get_learning_path(user_id=user_id)
    records = {s["id"]: s for s in mastery.snapshot(user_id)}

    target_cid = None
    # 1. First priority: any concept in WEAK state
    for cid in path:
        if records.get(cid, {}).get("state") == "WEAK":
            target_cid = cid
            break

    # 2. Second priority: earliest unmastered (UNKNOWN) concept
    if not target_cid:
        for cid in path:
            if records.get(cid, {}).get("state") != "STRONG":
                target_cid = cid
                break

    if not target_cid:
        # All nodes are STRONG
        state["next_step"] = "COMPLETE"
        state["state"] = MasteryStatus.STRONG
        return state

    current_record = records.get(target_cid, {})
    current_status = MasteryStatus(current_record.get("state", "UNKNOWN"))

    state["target_concept"] = target_cid
    state["state"] = current_status
    state["next_step"] = "ASSESS"
    return state


def generate_quiz_node(state: LoopState) -> LoopState:
    """Step 2 (Assess): Generate diagnostic multiple-choice items strictly from vectorless chunks."""
    target_cid = state.get("target_concept")
    if not target_cid:
        state["next_step"] = "DIAGNOSE"
        return state

    try:
        questions = quiz.generate_diagnostic_quiz(target_cid, count=3)
        state["quiz_questions"] = [q.model_dump() for q in questions]
        state["next_step"] = "WAITING_FOR_SUBMISSION"
    except Exception as e:
        logger.error(f"Failed generating diagnostic quiz for {target_cid}: {e}")
        state["error"] = str(e)
        state["next_step"] = "REMEDIAL_RETEACH"

    return state


def evaluate_submission_node(state: LoopState) -> LoopState:
    """Step 3 (Evaluate): Score student submission, update epistemic mastery, and check status."""
    user_id = state.get("user_id", "default")
    target_cid = state.get("target_concept")
    submissions = state.get("quiz_results", [])

    if not target_cid:
        state["next_step"] = "DIAGNOSE"
        return state

    # Check updated mastery status from SQLite
    record = mastery.get_record(user_id, target_cid)
    state["state"] = record.status

    # Add evidence trail
    trail = state.get("evidence_trail", [])
    trail.append(f"Evaluated {target_cid}: Score {record.score}, Status {record.status.value}")
    state["evidence_trail"] = trail

    # Route conditionally
    if record.status == MasteryStatus.STRONG:
        state["next_step"] = "UNLOCK_NEXT"
    else:
        state["next_step"] = "REMEDIAL_RETEACH"

    return state


def remedial_reteach_node(state: LoopState) -> LoopState:
    """Step 4a (Intervene): Targeted Socratic remediation on weak concept and prerequisite gaps."""
    user_id = state.get("user_id", "default")
    target_cid = state.get("target_concept")

    # Termination guard: prevent infinite remediation loops
    iterations = state.get("iteration_count", 0) + 1
    state["iteration_count"] = iterations

    if iterations > MAX_REMEDIAL_ITERATIONS:
        logger.warning(f"Max remedial loops reached ({MAX_REMEDIAL_ITERATIONS}) for {target_cid}. Unlocking next with support.")
        state["next_step"] = "UNLOCK_NEXT"
        state["remedial_lesson"] = "Max support attempts reached for this session. Review the textbook pages before advancing."
        return state

    # Pull prerequisite concepts that might be the root cause
    prereqs = concepts.get_prerequisites(target_cid)
    root_cause_cid = target_cid
    for pid in prereqs:
        p_rec = mastery.get_record(user_id, pid)
        if p_rec.status == MasteryStatus.WEAK:
            root_cause_cid = pid
            break

    lesson = tutor.reteach(root_cause_cid, user_id=user_id)
    state["remedial_lesson"] = lesson

    # Attach artifacts and external_links to state for rich remediation
    artifacts = list(state.get("artifacts", []))
    external_links = list(state.get("external_links", []))
    try:
        vis_art = tools.generate_visual_diagram_tool(root_cause_cid, prompt="Remedial diagram", user_id=user_id)
        if vis_art:
            artifacts.append(vis_art)
        res_data = tools.suggest_curated_resources_tool(root_cause_cid, user_id=user_id)
        if res_data and "resources" in res_data:
            external_links.extend(res_data["resources"])
    except Exception as e:
        logger.warning(f"Failed generating remedial artifacts/links: {e}")

    state["artifacts"] = artifacts
    state["external_links"] = external_links
    state["next_step"] = "ASSESS"
    return state


def unlock_next_node(state: LoopState) -> LoopState:
    """Step 4b (Advance): Unlock dependent topological concepts in the DAG."""
    target_cid = state.get("target_concept")
    successors = concepts.get_next_concepts(target_cid) if target_cid else []

    state["iteration_count"] = 0
    state["remedial_lesson"] = None
    state["artifacts"] = []
    state["external_links"] = []

    if successors:
        state["target_concept"] = successors[0]
        state["next_step"] = "ASSESS"
    else:
        # Continue topological diagnosis
        state["next_step"] = "DIAGNOSE"

    return state


# ── Conditional Routing Edge ───────────────────────────────────────────────

def route_next_step(state: LoopState) -> Literal["remedial_reteach", "unlock_next", "diagnose", "generate_quiz", "end"]:
    next_action = state.get("next_step", "DIAGNOSE")

    if next_action == "COMPLETE":
        return "end"
    if next_action == "REMEDIAL_RETEACH":
        return "remedial_reteach"
    if next_action == "UNLOCK_NEXT":
        return "unlock_next"
    if next_action == "ASSESS":
        return "generate_quiz"
    if next_action == "DIAGNOSE":
        return "diagnose"

    return "end"


# ── Build LangGraph StateGraph ─────────────────────────────────────────────

def create_closed_loop_graph():
    builder = StateGraph(LoopState)

    builder.add_node("diagnose", diagnose_node)
    builder.add_node("generate_quiz", generate_quiz_node)
    builder.add_node("evaluate_submission", evaluate_submission_node)
    builder.add_node("remedial_reteach", remedial_reteach_node)
    builder.add_node("unlock_next", unlock_next_node)

    builder.set_entry_point("diagnose")

    builder.add_conditional_edges(
        "diagnose",
        route_next_step,
        {
            "generate_quiz": "generate_quiz",
            "end": END,
        }
    )

    builder.add_edge("generate_quiz", END)

    builder.add_conditional_edges(
        "evaluate_submission",
        route_next_step,
        {
            "remedial_reteach": "remedial_reteach",
            "unlock_next": "unlock_next",
            "end": END,
        }
    )

    builder.add_edge("remedial_reteach", "generate_quiz")
    builder.add_edge("unlock_next", "diagnose")

    return builder.compile()


CLOSED_LOOP_GRAPH = create_closed_loop_graph()


# ── Orchestrator Interface ─────────────────────────────────────────────────

def start_session(user_id: str = "default") -> dict:
    """Initialize or advance the closed-loop learning session."""
    init_state: LoopState = {
        "user_id": user_id,
        "iteration_count": 0,
        "evidence_trail": [],
        "quiz_results": [],
    }

    res = CLOSED_LOOP_GRAPH.invoke(init_state)
    target = res.get("target_concept")
    kb = store.get_kb(user_id)
    by_id = {c["id"]: c for c in kb.get("concepts", [])}
    c_name = by_id.get(target, {}).get("name", target)

    status = res.get("state", MasteryStatus.UNKNOWN)
    questions = res.get("quiz_questions", [])

    action = "quiz" if questions else ("remedial_teach" if res.get("remedial_lesson") else "advance")
    if res.get("next_step") == "COMPLETE":
        action = "complete"

    return {
        "user_id": user_id,
        "lifecycle": get_lifecycle_state(user_id),
        "action": action,
        "concept_id": target,
        "concept_name": c_name,
        "status": status.value if hasattr(status, "value") else str(status),
        "quiz": questions[0] if questions else None,
        "quiz_count": len(questions),
        "remedial_lesson": res.get("remedial_lesson"),
        "artifacts": res.get("artifacts", []),
        "external_links": res.get("external_links", []),
        "reason": f"Active closed-loop step: {res.get('next_step')}",
    }


def submit_quiz_answer(sub: QuizSubmission) -> dict:
    """Submit a student quiz answer and invoke the LangGraph evaluation step."""
    # 1. Authoritative grading
    eval_result = quiz.evaluate_submission(sub)

    # 2. Invoke LangGraph step
    eval_state: LoopState = {
        "user_id": sub.user_id,
        "target_concept": sub.concept_id,
        "quiz_results": [eval_result.model_dump()],
        "iteration_count": 0,
        "evidence_trail": [],
    }

    # Execute evaluate_submission node through LangGraph
    evaluated_state = evaluate_submission_node(eval_state)
    routed = route_next_step(evaluated_state)

    remedial_text = None
    next_concept_id = sub.concept_id

    if routed == "remedial_reteach":
        remedial_state = remedial_reteach_node(evaluated_state)
        remedial_text = remedial_state.get("remedial_lesson")
        action = "remedial_teach"
    elif routed == "unlock_next":
        unlocked_state = unlock_next_node(evaluated_state)
        next_concept_id = unlocked_state.get("target_concept", sub.concept_id)
        action = "advance"
    else:
        action = "complete"

    record = mastery.get_record(sub.user_id, sub.concept_id)

    return {
        "user_id": sub.user_id,
        "question_id": sub.question_id,
        "concept_id": sub.concept_id,
        "is_correct": eval_result.is_correct,
        "correct_option_idx": eval_result.correct_option_idx,
        "explanation": eval_result.explanation,
        "misconceptions": eval_result.misconceptions,
        "question_type": eval_result.question_type,
        "correct_answer_display": eval_result.correct_answer_display,
        "user_answer": eval_result.user_answer,
        "reasoning_depth_level": eval_result.reasoning_depth_level,
        "updated_mastery": {
            "score": record.score,
            "status": record.status.value,
            "attempts": record.attempt_count,
        },
        "next_action": action,
        "next_concept_id": next_concept_id,
        "remedial_lesson": remedial_text,
        "artifacts": remedial_state.get("artifacts", []) if routed == "remedial_reteach" else evaluated_state.get("artifacts", []),
        "external_links": remedial_state.get("external_links", []) if routed == "remedial_reteach" else evaluated_state.get("external_links", []),
        "step_by_step_solution": eval_result.step_by_step_solution,
    }


# Compatibility adapter for previous /api/quiz/answer and /api/loop/next-action
def run(concept_id: str, correct: bool) -> dict:
    record_dict = mastery.record_quiz("default", concept_id, correct)
    return {
        "concept_id": concept_id,
        "correct": correct,
        "mastery": mastery.snapshot(),
        "next_id": concepts.get_learning_path()[0] if concepts.get_learning_path() else concept_id,
    }

def diagnose():
    return start_session("default")


# ── Baseline Diagnostic Workflow ───────────────────────────────────────────

def generate_baseline_diagnostic(user_id: str = "default") -> dict:
    """Generate a calibrated 6-question baseline test sampling root and foundational nodes from DAG."""
    mats = store.list_study_materials(user_id)
    if not mats:
        raise store.UserError("No study materials found. Upload your study material to begin.")

    kb = store.require(user_id=user_id)
    all_concepts = kb.get("concepts", [])
    if not all_concepts:
        raise store.UserError("No concepts extracted from study materials.")

    # 1. Identify root and foundational nodes from DAG
    root_nodes = [c for c in all_concepts if not concepts.get_prerequisites(c["id"], user_id=user_id)]
    root_ids = [c["id"] for c in root_nodes]

    topo_order = concepts.get_learning_path(user_id=user_id)

    foundational_ids: list[str] = []
    for rid in root_ids:
        if rid not in foundational_ids:
            foundational_ids.append(rid)
    for tid in topo_order:
        if tid not in foundational_ids:
            foundational_ids.append(tid)

    if not foundational_ids:
        foundational_ids = [c["id"] for c in all_concepts]

    # 2. Sample concepts to produce 6 calibrated questions
    total_target = 6
    questions: list[dict] = []
    num_foundational = len(foundational_ids)

    if num_foundational >= total_target:
        selected_cids = foundational_ids[:total_target]
        for cid in selected_cids:
            qs = quiz.generate_diagnostic_quiz(cid, count=1, user_id=user_id)
            if qs:
                questions.append(qs[0].model_dump())
    else:
        for i in range(total_target):
            cid = foundational_ids[i % num_foundational]
            qs = quiz.generate_diagnostic_quiz(cid, count=1, user_id=user_id)
            if qs:
                questions.append(qs[0].model_dump())

    by_id = {c["id"]: c for c in all_concepts}
    for q in questions:
        cid = q.get("concept_id")
        q["concept_name"] = by_id.get(cid, {}).get("name", cid)

    return {
        "user_id": user_id,
        "lifecycle": LIFECYCLE_BASELINE_DIAGNOSTIC,
        "total_questions": len(questions),
        "questions": questions,
        "target_concepts": list(dict.fromkeys(q.get("concept_id") for q in questions)),
    }


def evaluate_baseline_diagnostic(user_id: str = "default", submissions: list = None) -> dict:
    """Evaluate baseline diagnostic submissions:
      - Concepts scored >= 0.80 with sound reasoning -> STRONG.
      - Failed or guessed concepts (< 0.80) -> WEAK.
      - Unassessed downstream concepts -> UNKNOWN.
      - Transitions lifecycle to ACTIVE_LEARNING.
    """
    kb = store.require(user_id=user_id)
    all_concepts = kb.get("concepts", [])
    all_cids = [c["id"] for c in all_concepts]

    if not submissions:
        raise store.UserError("No submissions provided for baseline diagnostic evaluation.")

    evals_by_concept: dict[str, list[dict]] = {}

    for item in submissions:
        if isinstance(item, dict):
            sub = QuizSubmission(**item)
        elif isinstance(item, QuizSubmission):
            sub = item
        else:
            continue
        if sub.user_id == "default" and user_id != "default":
            sub.user_id = user_id

        eval_res = quiz.evaluate_submission(sub)
        evals_by_concept.setdefault(sub.concept_id, []).append({
            "sub": sub,
            "eval": eval_res,
        })

    strong_concepts = []
    weak_concepts = []
    concept_results = {}

    with mastery.get_db() as conn:
        for cid, items in evals_by_concept.items():
            avg_score = sum(it["eval"].score for it in items) / len(items)

            # Detect guessing or surface logic
            is_guess = False
            for it in items:
                reasoning = (it["sub"].student_reasoning or it["sub"].typed_answer or "").lower()
                if any(w in reasoning for w in ("guess", "guessed", "random", "picked randomly", "no idea", "dunno", "not sure")):
                    is_guess = True
                    break
                if any("guess" in m.lower() for m in it["eval"].misconceptions):
                    is_guess = True
                    break

            depths = [it["eval"].reasoning_depth_level for it in items]
            has_sound_reasoning = (
                not is_guess
                and any(d in ("DEVELOPING", "DEEP_CONCEPTUAL") for d in depths)
            )

            # Apply baseline criteria: score >= 0.80 with sound reasoning -> STRONG, else WEAK
            if avg_score >= 0.80 and has_sound_reasoning:
                status = MasteryStatus.STRONG
                final_score = round(avg_score, 2)
                audit = f"Baseline diagnostic passed: Score {final_score:.2f} with sound reasoning ({', '.join(depths)})."
                strong_concepts.append(cid)
            else:
                status = MasteryStatus.WEAK
                final_score = round(min(avg_score, 0.45), 2)
                why = "lucky guess or surface logic detected" if is_guess else f"score {avg_score:.2f} below 0.80 threshold"
                audit = f"Baseline diagnostic: WEAK ({final_score:.2f}) - {why}."
                weak_concepts.append(cid)

            row = mastery._get_row(conn, cid, user_id)
            row["state"] = status.value
            row["mastery_score"] = final_score
            row["attempt_count"] = len(items)
            row["quiz_correct"] = sum(1 for it in items if it["eval"].is_correct)
            row["quiz_total"] = len(items)
            row["audit_trail"] = audit
            row["reasoning_depth"] = depths[-1] if depths else "DEVELOPING"
            row["evidence_notes"] = json.dumps([audit])
            miscons = []
            for it in items:
                miscons.extend(it["eval"].misconceptions)
            row["misconceptions"] = json.dumps(miscons)
            mastery._save_row(conn, row)

            concept_results[cid] = {
                "concept_id": cid,
                "status": status.value,
                "score": final_score,
                "audit": audit,
                "is_guess": is_guess,
            }

        # Unassessed downstream concepts -> UNKNOWN (score 0.0)
        unknown_concepts = [cid for cid in all_cids if cid not in evals_by_concept]
        for cid in unknown_concepts:
            row = mastery._get_row(conn, cid, user_id)
            row["state"] = MasteryStatus.UNKNOWN.value
            row["mastery_score"] = 0.0
            row["attempt_count"] = 0
            row["quiz_correct"] = 0
            row["quiz_total"] = 0
            row["audit_trail"] = "Unassessed downstream concept from baseline diagnostic."
            row["evidence_notes"] = json.dumps(["Unassessed in baseline diagnostic."])
            mastery._save_row(conn, row)

    # Transition lifecycle to ACTIVE_LEARNING
    set_lifecycle_state(user_id, LIFECYCLE_ACTIVE_LEARNING)

    all_eval_dicts = [
        it["eval"].model_dump()
        for items in evals_by_concept.values()
        for it in items
    ]

    return {
        "user_id": user_id,
        "lifecycle": LIFECYCLE_ACTIVE_LEARNING,
        "evaluated_count": len(submissions),
        "strong_concepts": strong_concepts,
        "weak_concepts": weak_concepts,
        "unknown_concepts": unknown_concepts,
        "concept_results": concept_results,
        "evaluations": all_eval_dicts,
        "message": "Baseline diagnostic completed. Learning session transitioned to ACTIVE_LEARNING."
    }


# ── Remedial Quiz Generation ───────────────────────────────────────────────

def generate_remedial_quiz(user_id: str = "default", target_concept_id: str | None = None) -> dict:
    """Specifically generates a targeted remedial quiz for the student's top WEAK prerequisite root cause."""
    kb = store.require(user_id=user_id)
    concepts_list = kb.get("concepts", [])
    if not concepts_list:
        raise store.UserError("No concepts available to generate remedial quiz.")

    by_id = {c["id"]: c for c in concepts_list}
    snap = mastery.snapshot(user_id=user_id)
    mastery_by_id = {s["id"]: s for s in snap}

    root_cid = None
    if target_concept_id:
        # Check if any prerequisite of target_concept_id is WEAK
        prereqs = concepts.get_prerequisites(target_concept_id, user_id=user_id)
        for pid in prereqs:
            p_rec = mastery_by_id.get(pid, {})
            if p_rec.get("state") == "WEAK" or p_rec.get("status") == "weak":
                deeper = concepts.get_prerequisites(pid, user_id=user_id)
                deep_weak = [dp for dp in deeper if mastery_by_id.get(dp, {}).get("state") == "WEAK"]
                root_cid = deep_weak[0] if deep_weak else pid
                break
        if not root_cid:
            root_cid = target_concept_id
    else:
        path = concepts.get_learning_path(user_id=user_id)
        weak_cids = [cid for cid in path if mastery_by_id.get(cid, {}).get("state") == "WEAK" or mastery_by_id.get(cid, {}).get("status") == "weak"]
        if weak_cids:
            root_cid = weak_cids[0]
        else:
            unmastered = [cid for cid in path if mastery_by_id.get(cid, {}).get("state") != "STRONG"]
            root_cid = unmastered[0] if unmastered else (path[0] if path else concepts_list[0]["id"])

    qs = quiz.generate_diagnostic_quiz(root_cid, count=1, user_id=user_id)
    if not qs:
        raise store.UserError(f"Unable to generate remedial quiz question for '{root_cid}'.")
    q = qs[0]

    lesson = tutor.reteach(root_cid, user_id=user_id)
    c_name = by_id.get(root_cid, {}).get("name", root_cid)

    quiz.RECENT_QUESTIONS[q.question_id] = {
        "correct": q.correct_option_idx,
        "conceptId": q.concept_id,
        "concept_id": q.concept_id,
        "questionType": q.question_type,
        "question_type": q.question_type,
        "targetBlankAnswer": q.target_blank_answer,
        "rubric": q.rubric,
        "explanation": q.explanation,
        "source_page": q.source_page,
    }

    return {
        "id": q.question_id,
        "question_id": q.question_id,
        "conceptId": q.concept_id,
        "concept_id": q.concept_id,
        "conceptName": c_name,
        "concept_name": c_name,
        "questionType": q.question_type,
        "question_type": q.question_type,
        "question": q.question_text,
        "question_text": q.question_text,
        "options": q.options,
        "correct": q.correct_option_idx,
        "correct_option_idx": q.correct_option_idx,
        "targetBlankAnswer": q.target_blank_answer,
        "rubric": q.rubric,
        "explanation": q.explanation,
        "sourcePage": q.source_page,
        "source_page": q.source_page,
        "is_remedial": True,
        "target_concept": target_concept_id or root_cid,
        "root_cause_concept": root_cid,
        "remedial_lesson": lesson,
        "artifacts": [],
        "external_links": [],
    }
