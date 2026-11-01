"""Closed-Loop LangGraph Orchestrator.

Implements an agentic cycle of:
  Diagnosis -> Assessment -> Evaluation -> Socratic Remediation / Advancement -> Re-assessment.

Uses LangGraph StateGraph with conditional edges and loop-termination guards.
"""

from __future__ import annotations

import logging
from typing import Dict, Any, Literal
from langgraph.graph import StateGraph, END

from ..core import store
from ..models.schemas import LoopState, MasteryStatus, QuizSubmission
from . import concepts, mastery, quiz, tutor

logger = logging.getLogger("concept_mastery.loop")

MAX_REMEDIAL_ITERATIONS = 3


# ── Node Implementations ───────────────────────────────────────────────────

def diagnose_node(state: LoopState) -> LoopState:
    """Step 1 (Diagnose): Select the earliest unmastered or WEAK node from prerequisite DAG."""
    user_id = state.get("user_id", "default")
    kb = store.KB
    if not kb.get("ready"):
        state["next_step"] = "COMPLETE"
        state["error"] = "No active knowledge base loaded."
        return state

    # Pull topological ordering of concepts
    path = concepts.get_learning_path()
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
    state["next_step"] = "ASSESS"
    return state


def unlock_next_node(state: LoopState) -> LoopState:
    """Step 4b (Advance): Unlock dependent topological concepts in the DAG."""
    target_cid = state.get("target_concept")
    successors = concepts.get_next_concepts(target_cid) if target_cid else []

    state["iteration_count"] = 0
    state["remedial_lesson"] = None

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
    kb = store.KB
    by_id = {c["id"]: c for c in kb.get("concepts", [])}
    c_name = by_id.get(target, {}).get("name", target)

    status = res.get("state", MasteryStatus.UNKNOWN)
    questions = res.get("quiz_questions", [])

    action = "quiz" if questions else ("remedial_teach" if res.get("remedial_lesson") else "advance")
    if res.get("next_step") == "COMPLETE":
        action = "complete"

    return {
        "user_id": user_id,
        "action": action,
        "concept_id": target,
        "concept_name": c_name,
        "status": status.value if hasattr(status, "value") else str(status),
        "quiz": questions[0] if questions else None,
        "quiz_count": len(questions),
        "remedial_lesson": res.get("remedial_lesson"),
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
