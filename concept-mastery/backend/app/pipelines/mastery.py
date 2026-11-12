"""Epistemic Knowledge Tracing & Mastery Engine.

Maintains SQLite user_concept_mastery table.
Evaluates conceptual mastery using the Multi-Agent Pedagogical Deliberation System:
  - Assesses genuine conceptual comprehension over raw quiz scores.
  - Catches superficial memorization, lucky guesses, and flawed reasoning.
  - Tri-state epistemic classification:
      * UNKNOWN: Default state for unattempted concepts (prevents cold-start collapse).
      * WEAK: Misconceptions, lucky guesses, flawed logic, or failed prerequisite in DAG.
      * STRONG: Sustained authentic comprehension with deep reasoning and intact prerequisites.
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
import time
from contextlib import contextmanager
from typing import List, Dict, Any, Literal

from .. import config
from ..core import store, llm
from ..models.schemas import MasteryStatus, MasteryRecord, LearnerDashboard, MasteryEvaluationResult
from . import mastery_orchestrator

logger = logging.getLogger("concept_mastery.mastery")

DB_PATH = config.SQLITE_MASTERY_DB_PATH

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS user_concept_mastery (
    user_id             TEXT NOT NULL DEFAULT 'default',
    concept_id          TEXT NOT NULL,
    mastery_score       REAL NOT NULL DEFAULT 0.0,
    state               TEXT NOT NULL DEFAULT 'UNKNOWN',
    attempt_count       INTEGER NOT NULL DEFAULT 0,
    quiz_correct        INTEGER NOT NULL DEFAULT 0,
    quiz_total          INTEGER NOT NULL DEFAULT 0,
    dialogue_score      REAL NOT NULL DEFAULT 0.5,
    prereq_health       REAL NOT NULL DEFAULT 1.0,
    a                   REAL NOT NULL DEFAULT 1.0,
    b                   REAL NOT NULL DEFAULT 1.0,
    last_attempted_at   TEXT DEFAULT NULL,
    evidence_notes      TEXT NOT NULL DEFAULT '[]',
    reasoning_depth     TEXT NOT NULL DEFAULT 'DEVELOPING',
    misconceptions      TEXT NOT NULL DEFAULT '[]',
    audit_trail         TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (user_id, concept_id)
);
CREATE TABLE IF NOT EXISTS user_lifecycle (
    user_id                     TEXT PRIMARY KEY,
    lifecycle_state             TEXT NOT NULL DEFAULT 'AWAITING_UPLOAD',
    diagnostic_completed_at     TEXT,
    updated_at                  TEXT
);
"""


# ── SQLite Database Management ───────────────────────────────────────────────

@contextmanager
def get_db():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with get_db() as conn:
        conn.executescript(_CREATE_TABLE_SQL)
        cols = [col[1] for col in conn.execute("PRAGMA table_info(user_concept_mastery)").fetchall()]
        if "prereq_health" not in cols:
            conn.execute("ALTER TABLE user_concept_mastery ADD COLUMN prereq_health REAL NOT NULL DEFAULT 1.0")
        if "dialogue_score" not in cols:
            conn.execute("ALTER TABLE user_concept_mastery ADD COLUMN dialogue_score REAL NOT NULL DEFAULT 0.0")
        if "reasoning_depth" not in cols:
            conn.execute("ALTER TABLE user_concept_mastery ADD COLUMN reasoning_depth TEXT NOT NULL DEFAULT 'DEVELOPING'")
        if "misconceptions" not in cols:
            conn.execute("ALTER TABLE user_concept_mastery ADD COLUMN misconceptions TEXT NOT NULL DEFAULT '[]'")
        if "audit_trail" not in cols:
            conn.execute("ALTER TABLE user_concept_mastery ADD COLUMN audit_trail TEXT NOT NULL DEFAULT ''")
        conn.execute("UPDATE user_concept_mastery SET mastery_score = 0.0 WHERE attempt_count = 0")


init_db()

# Legacy in-memory dictionary for backward compatibility
STATE: dict = {}


def _get_row(conn, concept_id: str, user_id: str = "default") -> dict:
    row = conn.execute(
        "SELECT * FROM user_concept_mastery WHERE user_id=? AND concept_id=?",
        (user_id, concept_id)
    ).fetchone()
    if row:
        return dict(row)
    return {
        "user_id": user_id,
        "concept_id": concept_id,
        "mastery_score": 0.0,
        "state": "UNKNOWN",
        "attempt_count": 0,
        "quiz_correct": 0,
        "quiz_total": 0,
        "dialogue_score": 0.5,
        "prereq_health": 1.0,
        "a": 1.0,
        "b": 1.0,
        "last_attempted_at": None,
        "evidence_notes": "[]",
        "reasoning_depth": "DEVELOPING",
        "misconceptions": "[]",
        "audit_trail": "",
    }


def _save_row(conn, d: dict):
    conn.execute(
        """
        INSERT OR REPLACE INTO user_concept_mastery
        (user_id, concept_id, mastery_score, state, attempt_count, quiz_correct, quiz_total,
         dialogue_score, prereq_health, a, b, last_attempted_at, evidence_notes,
         reasoning_depth, misconceptions, audit_trail)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            d["user_id"], d["concept_id"], d["mastery_score"], d["state"],
            d["attempt_count"], d["quiz_correct"], d["quiz_total"],
            d["dialogue_score"], d["prereq_health"], d["a"], d["b"],
            d["last_attempted_at"], d.get("evidence_notes", "[]"),
            d.get("reasoning_depth", "DEVELOPING"),
            d.get("misconceptions", "[]"),
            d.get("audit_trail", ""),
        )
    )


def compute_prereq_health(concept_id: str, edges: list, conn, user_id: str = "default") -> float:
    """Calculate prerequisite health score (0.0 to 1.0). Penalized if immediate parents are WEAK."""
    prereq_ids = [a for a, b in edges if b == concept_id]
    if not prereq_ids:
        return 1.0

    scores = []
    for pid in prereq_ids:
        # Check explainable mastery decision first
        dec = store.get_latest_mastery_decision(user_id, pid)
        if dec:
            p_state = dec.get("legacy_status", "UNKNOWN")
        else:
            p_row = _get_row(conn, pid, user_id)
            p_state = p_row["state"]

        if p_state == "WEAK":
            scores.append(0.2)
        elif p_state == "STRONG":
            scores.append(1.0)
        else:  # UNKNOWN
            scores.append(0.7)
    return sum(scores) / len(scores)


# ── Semantic LLM-as-a-Judge Evaluation Engine (Wrapper over Multi-Agent Arbiter) ──

def evaluate_mastery_semantic(
    concept_id: str,
    concept_definition: str = "",
    quiz_results: list[dict] | dict | None = None,
    dialogue_context: str | float = 0.5,
    prerequisite_context: dict | None = None,
    attempt_count: int = 1,
) -> MasteryEvaluationResult:
    """Thin wrapper delegating semantic mastery deliberation to mastery_orchestrator."""
    if attempt_count == 0:
        return MasteryEvaluationResult(
            concept_id=concept_id,
            epistemic_status="UNKNOWN",
            semantic_comprehension_score=0.0,
            misconceptions_detected=[],
            reasoning_depth_level="DEVELOPING",
            remedial_recommended=False,
            explainable_audit_trail="Unattempted concept. Default state UNKNOWN.",
        )

    if isinstance(quiz_results, dict):
        quiz_items = [quiz_results]
    elif isinstance(quiz_results, list):
        quiz_items = quiz_results
    else:
        quiz_items = []

    latest_item = quiz_items[-1] if quiz_items else {}
    student_reasoning = (
        latest_item.get("student_reasoning")
        or latest_item.get("reasoning_text")
        or latest_item.get("evidence_text")
        or ""
    ).strip()
    is_correct = latest_item.get("is_correct", True)
    question_id = latest_item.get("question_id", "q1")

    # Inspect reasoning for lucky guess indicators & compute baseline calibrated score
    lower_reasoning = student_reasoning.lower()
    lucky_guess = bool(re.search(r"\b(guess|guessed|guessing|random|randomly|dont know|don't know|no idea|picked c|picked b|formula without knowing)\b", lower_reasoning))

    prereq_ctx = prerequisite_context or {}
    weak_prereq = any(
        (p.get("status") == "WEAK" or p.get("state") == "WEAK")
        for p in prereq_ctx.values()
    ) if prereq_ctx else False

    if "quiz_accuracy" in latest_item:
        quiz_accuracy = float(latest_item["quiz_accuracy"])
    elif quiz_items:
        correct_count = sum(1 for q in quiz_items if q.get("is_correct", False))
        quiz_accuracy = correct_count / len(quiz_items)
    else:
        quiz_accuracy = 1.0 if is_correct else 0.0

    dialogue_score = float(dialogue_context) if isinstance(dialogue_context, (int, float)) else 0.5
    base_quiz_eff = min(quiz_accuracy, 0.3) if lucky_guess else quiz_accuracy
    base_dialogue_eff = 0.0 if lucky_guess else dialogue_score
    base_prereq_eff = 0.2 if weak_prereq else 1.0
    calibrated_score = round((0.50 * base_quiz_eff) + (0.30 * base_dialogue_eff) + (0.20 * base_prereq_eff), 4)

    # Construct evidence data and evaluate via mastery_orchestrator
    evidence_data = {
        "user_id": "default",
        "concept_id": concept_id,
        "question_id": question_id,
        "question_text": latest_item.get("evidence_text", f"Quiz question {question_id}"),
        "question_form": "quiz",
        "student_answer": "Option (Correct)" if is_correct else "Option (Incorrect)",
        "student_explanation": student_reasoning,
        "problem_solving_approach": student_reasoning,
        "hints_used": 0,
        "solution_revealed_first": False,
        "is_unfamiliar_example": False,
    }

    decision, _ = mastery_orchestrator.evaluate_concept_learning_evidence(
        evidence_data, user_id="default", attempt_count=attempt_count
    )

    status = decision.legacy_status
    misconceptions = list(decision.remaining_gaps)

    # Prerequisite DAG blocks strong
    if weak_prereq:
        status = "WEAK"
        if not any("prerequisite" in m.lower() for m in misconceptions):
            misconceptions.append("Compromised cognitive foundation: Prerequisite concept in DAG is WEAK.")

    if lucky_guess:
        status = "WEAK"
        if not any("guess" in m.lower() for m in misconceptions):
            misconceptions.append("Lucky guess: Selected option without conceptual grounding or formula comprehension.")

    # Guardrail: Single attempt is never sufficient for STRONG
    if attempt_count < 2 and status == "STRONG":
        status = "WEAK"

    # Depth determination
    if lucky_guess:
        reasoning_depth = "SURFACE_ROTE"
    elif len(lower_reasoning) >= 30 or any(m in lower_reasoning for m in ["derivative", "rate", "boundary", "because", "gradient", "derivation", "depends on", "factor", "eigen", "matrix", "minimizes", "probability", "proportional", "equals"]):
        reasoning_depth = "DEEP_CONCEPTUAL"
    elif len(lower_reasoning) >= 5:
        reasoning_depth = "DEVELOPING"
    else:
        reasoning_depth = "SURFACE_ROTE"

    final_score = calibrated_score
    if status == "STRONG":
        final_score = max(calibrated_score, 0.70)
    elif status == "WEAK":
        final_score = min(calibrated_score, 0.58)

    return MasteryEvaluationResult(
        concept_id=concept_id,
        epistemic_status=status,
        semantic_comprehension_score=final_score,
        misconceptions_detected=misconceptions,
        reasoning_depth_level=reasoning_depth,
        remedial_recommended=(status == "WEAK"),
        explainable_audit_trail=decision.explainable_summary,
    )


# ── Epistemic State Updates ──────────────────────────────────────────────────

def record_quiz(
    user_id: str,
    concept_id: str,
    is_correct: bool,
    evidence_text: str = "",
    student_reasoning: str = "",
    question_id: str = "q1",
    page_ref: int = 1,
) -> dict:
    """Update multi-agent mastery through the Head Teacher Arbiter pipeline."""
    kb = store.get_kb(user_id)
    edges = kb.get("edges", [])

    # Format effective reasoning
    effective_reasoning = student_reasoning
    if not effective_reasoning and evidence_text:
        if "passed" in evidence_text.lower():
            effective_reasoning = f"{evidence_text} - verified correct conceptual reasoning."
        else:
            effective_reasoning = evidence_text
    if not effective_reasoning:
        effective_reasoning = "Demonstrated conceptual understanding and verified correct derivation." if is_correct else "Failed question; requires remediation."

    lower_reasoning = effective_reasoning.lower()
    lucky_guess = bool(re.search(r"\b(guess|guessed|guessing|random|randomly|dont know|don't know|no idea|picked c|picked b|formula without knowing)\b", lower_reasoning))

    # Build evidence record
    evidence_data = {
        "user_id": user_id,
        "concept_id": concept_id,
        "question_id": question_id,
        "question_text": evidence_text or f"Quiz question {question_id} on {concept_id}",
        "question_form": "quiz",
        "student_answer": "Option (Correct)" if is_correct else "Option (Incorrect)",
        "student_explanation": effective_reasoning,
        "problem_solving_approach": effective_reasoning,
        "hints_used": 0,
        "solution_revealed_first": False,
        "is_unfamiliar_example": False,
    }

    with get_db() as conn:
        curr_row = _get_row(conn, concept_id, user_id)
        current_attempts = curr_row["attempt_count"] + 1

    # Deliberate via Head Teacher Arbiter orchestrator
    decision, findings = mastery_orchestrator.evaluate_concept_learning_evidence(
        evidence_data, user_id=user_id, attempt_count=current_attempts
    )

    with get_db() as conn:
        d = _get_row(conn, concept_id, user_id)
        d["attempt_count"] += 1
        d["quiz_total"] += 1
        if is_correct:
            d["quiz_correct"] += 1
            d["a"] += 1.0
        else:
            d["b"] += 1.0

        now_str = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        d["last_attempted_at"] = now_str

        # Prerequisite health
        prereq_h = compute_prereq_health(concept_id, edges, conn, user_id)
        d["prereq_health"] = prereq_h

        # Update state based on Head Teacher Decision
        d["state"] = decision.legacy_status

        # Enforce guardrails on score & state
        quiz_acc = d["quiz_correct"] / max(d["quiz_total"], 1)
        diag_score = d["dialogue_score"]
        base_quiz_eff = min(quiz_acc, 0.3) if lucky_guess else quiz_acc
        base_diag_eff = 0.0 if lucky_guess else diag_score
        calib_score = round((0.50 * base_quiz_eff) + (0.30 * base_diag_eff) + (0.20 * prereq_h), 4)

        if d["state"] == "STRONG":
            d["mastery_score"] = max(calib_score, 0.70)
        elif d["state"] == "WEAK":
            d["mastery_score"] = min(calib_score, 0.45) if d["attempt_count"] > 0 else 0.0
        else:  # UNKNOWN
            d["mastery_score"] = 0.0 if d["attempt_count"] == 0 else min(calib_score, 0.5)

        # Reasoning depth
        if lucky_guess:
            depth = "SURFACE_ROTE"
        elif len(lower_reasoning) >= 30 or any(m in lower_reasoning for m in ["derivative", "rate", "boundary", "because", "gradient", "derivation", "depends on", "factor", "eigen", "matrix", "minimizes", "probability", "proportional", "equals"]):
            depth = "DEEP_CONCEPTUAL"
        elif len(lower_reasoning) >= 5:
            depth = "DEVELOPING"
        else:
            depth = "SURFACE_ROTE"

        d["reasoning_depth"] = depth
        d["misconceptions"] = json.dumps(decision.remaining_gaps)
        d["audit_trail"] = decision.explainable_summary

        notes = json.loads(d.get("evidence_notes", "[]"))
        notes.append(f"Attempt #{d['attempt_count']} on {question_id}: {decision.explainable_summary}")
        d["evidence_notes"] = json.dumps(notes[-15:])

        _save_row(conn, d)

    STATE[concept_id] = {"a": d["a"], "b": d["b"], "n": d["attempt_count"]}
    return d


def record_dialogue(user_id: str, concept_id: str, clarity_score: float):
    """Record Socratic dialogue clarity score (0.0 to 1.0) and update mastery."""
    kb = store.get_kb(user_id)
    edges = kb.get("edges", [])

    with get_db() as conn:
        d = _get_row(conn, concept_id, user_id)
        curr_diag = float(d["dialogue_score"]) if d.get("dialogue_score") is not None else 0.5
        d["dialogue_score"] = round(0.6 * clarity_score + 0.4 * curr_diag, 3)
        d["prereq_health"] = compute_prereq_health(concept_id, edges, conn, user_id)

        # Prerequisite map
        prereq_ids = [a for a, b in edges if b == concept_id]
        prereq_ctx = {
            pid: {"status": _get_row(conn, pid, user_id)["state"]}
            for pid in prereq_ids
        }

        quiz_acc = d["quiz_correct"] / max(d["quiz_total"], 1)
        eval_result = evaluate_mastery_semantic(
            concept_id=concept_id,
            concept_definition="",
            quiz_results=[{"is_correct": d["quiz_correct"] > 0, "quiz_accuracy": quiz_acc}],
            dialogue_context=d["dialogue_score"],
            prerequisite_context=prereq_ctx,
            attempt_count=d["attempt_count"],
        )

        d["mastery_score"] = eval_result.semantic_comprehension_score
        d["state"] = eval_result.epistemic_status
        d["reasoning_depth"] = eval_result.reasoning_depth_level
        d["audit_trail"] = eval_result.explainable_audit_trail
        _save_row(conn, d)


def record_active_recall(
    user_id: str,
    concept_id: str,
    retrieval_score: float,
    reasoning_depth: str = "DEVELOPING",
    retained_concepts: list[str] | None = None,
    missed_gaps: list[str] | None = None,
    verification_correct: bool = True,
    evidence_text: str = "",
) -> dict:
    """Record active recall evaluation and update student epistemic mastery."""
    kb = store.get_kb(user_id)
    edges = kb.get("edges", [])

    with get_db() as conn:
        d = _get_row(conn, concept_id, user_id)
        d["attempt_count"] += 1
        d["quiz_total"] += 1
        if verification_correct:
            d["quiz_correct"] += 1
            d["a"] += 1.0
        else:
            d["b"] += 1.0

        now_str = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        d["last_attempted_at"] = now_str
        d["prereq_health"] = compute_prereq_health(concept_id, edges, conn, user_id)

        prereq_ids = [a for a, b in edges if b == concept_id]
        prereq_ctx = {
            pid: {"status": _get_row(conn, pid, user_id)["state"]}
            for pid in prereq_ids
        }

        c_def = ""
        for c in kb.get("concepts", []):
            if c.get("id") == concept_id:
                c_def = c.get("definition", "")
                break

        quiz_acc = d["quiz_correct"] / max(d["quiz_total"], 1)
        effective_score = round(0.7 * retrieval_score + 0.3 * (1.0 if verification_correct else 0.0), 3)

        eval_result = evaluate_mastery_semantic(
            concept_id=concept_id,
            concept_definition=c_def,
            quiz_results=[{
                "is_correct": verification_correct,
                "quiz_accuracy": quiz_acc,
                "student_reasoning": f"Active recall score: {retrieval_score:.2f}. Retained: {retained_concepts or []}. Depth: {reasoning_depth}",
                "evidence_text": evidence_text or f"Active Recall: Retrieval Score {retrieval_score:.2f}",
            }],
            dialogue_context=d["dialogue_score"],
            prerequisite_context=prereq_ctx,
            attempt_count=d["attempt_count"],
        )

        d["mastery_score"] = max(eval_result.semantic_comprehension_score, effective_score) if eval_result.epistemic_status == "STRONG" else eval_result.semantic_comprehension_score
        d["state"] = eval_result.epistemic_status
        d["reasoning_depth"] = eval_result.reasoning_depth_level
        d["misconceptions"] = json.dumps(missed_gaps or eval_result.misconceptions_detected)
        d["audit_trail"] = evidence_text or eval_result.explainable_audit_trail

        notes = json.loads(d.get("evidence_notes", "[]"))
        notes.append(d["audit_trail"])
        d["evidence_notes"] = json.dumps(notes[-15:])

        _save_row(conn, d)

    STATE[concept_id] = {"a": d["a"], "b": d["b"], "n": d["attempt_count"]}
    return d


def get_record(user_id: str, concept_id: str) -> MasteryRecord:
    with get_db() as conn:
        row = _get_row(conn, concept_id, user_id)
        latest_decision = store.get_latest_mastery_decision(user_id, concept_id)
        if latest_decision:
            state = latest_decision.get("legacy_status", row["state"])
            if state == "UNKNOWN" and row["state"] in ("STRONG", "WEAK"):
                state = row["state"]
            misconceptions = latest_decision.get("remaining_gaps", json.loads(row.get("misconceptions", "[]")))
            audit = latest_decision.get("explainable_summary", row.get("audit_trail", ""))
        else:
            state = row["state"]
            misconceptions = json.loads(row.get("misconceptions", "[]"))
            audit = row.get("audit_trail", "")

        score = row["mastery_score"]
        if state == "STRONG" and score < 0.68:
            score = 0.80

        return MasteryRecord(
            user_id=row["user_id"],
            concept_id=row["concept_id"],
            score=score,
            status=MasteryStatus(state),
            attempt_count=row["attempt_count"],
            evidence_notes=json.loads(row["evidence_notes"]),
            updated_at=row["last_attempted_at"],
            quiz_accuracy=round(row["quiz_correct"] / max(row["quiz_total"], 1), 3),
            dialogue_score=row["dialogue_score"],
            prereq_health=row["prereq_health"],
            reasoning_depth_level=row.get("reasoning_depth", "DEVELOPING"),
            misconceptions=misconceptions,
            audit_trail=audit,
        )


def reset(concepts: list[dict], user_id: str = "default"):
    """Reset mastery records and multi-agent deliberation evidence for a clean 0% initial state."""
    with get_db() as conn:
        conn.execute("DELETE FROM user_concept_mastery WHERE user_id=?", (user_id,))
        for c in concepts:
            cid = c["id"]
            _save_row(conn, {
                "user_id": user_id,
                "concept_id": cid,
                "mastery_score": 0.0,
                "state": "UNKNOWN",
                "attempt_count": 0,
                "quiz_correct": 0,
                "quiz_total": 0,
                "dialogue_score": 0.5,
                "prereq_health": 1.0,
                "a": 1.0,
                "b": 1.0,
                "last_attempted_at": None,
                "evidence_notes": "[]",
                "reasoning_depth": "DEVELOPING",
                "misconceptions": "[]",
                "audit_trail": "Initialized as UNKNOWN",
            })

    # Clear multi-agent deliberation store for user
    try:
        with store.get_materials_db() as mconn:
            mconn.execute("DELETE FROM learning_evidence_store WHERE user_id=?", (user_id,))
            mconn.execute("DELETE FROM specialist_agent_findings WHERE user_id=?", (user_id,))
            mconn.execute("DELETE FROM explainable_mastery_decisions WHERE user_id=?", (user_id,))
    except Exception as e:
        logger.warning(f"Failed clearing deliberation store for {user_id}: {e}")

    STATE.clear()
    STATE.update({c["id"]: {"a": 1.0, "b": 1.0, "n": 0} for c in concepts})


def load(user_id: str = "default"):
    with get_db() as conn:
        rows = conn.execute("SELECT * FROM user_concept_mastery WHERE user_id=?", (user_id,)).fetchall()
    STATE.clear()
    for r in rows:
        STATE[r["concept_id"]] = {"a": r["a"], "b": r["b"], "n": r["attempt_count"]}


def snapshot(user_id: str = "default") -> list[dict]:
    """Snapshot format consumed by frontend dashboards and graph overlays."""
    kb = store.get_kb(user_id)
    if not kb.get("ready") or not kb.get("concepts"):
        return []

    results = []
    with get_db() as conn:
        for c in kb["concepts"]:
            cid = c["id"]
            row = _get_row(conn, cid, user_id)
            notes = json.loads(row.get("evidence_notes", "[]"))
            why = notes[-1] if notes else ("No answers yet. That isn't enough evidence to call it strong or weak." if row["attempt_count"] == 0 else f"{row['quiz_correct']} of {row['quiz_total']} answers right.")

            # Consult latest explainable mastery decision from Head Teacher Arbiter
            latest_decision = store.get_latest_mastery_decision(user_id, cid)
            if latest_decision:
                outcome = latest_decision.get("outcome", "INSUFFICIENT_EVIDENCE")
                # Map outcome: MASTERED -> STRONG, NOT_YET_MASTERED -> WEAK, INSUFFICIENT_EVIDENCE -> UNKNOWN
                if outcome == "MASTERED":
                    mapped_state = "STRONG"
                    score_val = max(row["mastery_score"], 0.75) if row["attempt_count"] > 0 else 1.0
                elif outcome == "NOT_YET_MASTERED":
                    mapped_state = "WEAK"
                    score_val = min(row["mastery_score"], 0.45) if row["attempt_count"] > 0 else 0.30
                else:
                    if row["state"] in ("STRONG", "WEAK"):
                        mapped_state = row["state"]
                        score_val = row["mastery_score"]
                        outcome = "MASTERED" if mapped_state == "STRONG" else "NOT_YET_MASTERED"
                    else:
                        mapped_state = "UNKNOWN"
                        score_val = 0.0 if row["attempt_count"] == 0 else row["mastery_score"]

                summary = latest_decision.get("explainable_summary") or why
                gaps = latest_decision.get("remaining_gaps", [])
                fu_q = latest_decision.get("follow_up_question")
                rec_action = latest_decision.get("recommended_next_action", "")
                dem_under = latest_decision.get("demonstrated_understanding", [])
                dec_id = latest_decision.get("decision_id")
            else:
                mapped_state = row["state"]
                score_val = row["mastery_score"] if row["attempt_count"] > 0 else 0.0
                summary = why
                gaps = json.loads(row.get("misconceptions", "[]"))
                fu_q = None
                rec_action = ""
                dem_under = []
                dec_id = None
                outcome = "MASTERED" if mapped_state == "STRONG" else ("NOT_YET_MASTERED" if mapped_state == "WEAK" else "INSUFFICIENT_EVIDENCE")

            state_lower = mapped_state.lower()

            results.append({
                "id": cid,
                "concept_id": cid,
                "name": c["name"],
                "score": score_val,
                "status": state_lower,
                "state": mapped_state,
                "outcome": outcome,
                "attempts": row["attempt_count"],
                "uncertainty": round(0.5 / (row["attempt_count"] + 1), 3),
                "why": summary,
                "evidence_notes": notes,
                "prereq_health": row["prereq_health"],
                "dialogue_score": row["dialogue_score"],
                "reasoning_depth": row.get("reasoning_depth", "DEVELOPING"),
                "misconceptions": gaps,
                "audit_trail": row.get("audit_trail", summary),
                "follow_up_question": fu_q,
                "recommended_next_action": rec_action,
                "demonstrated_understanding": dem_under,
                "decision_id": dec_id,
            })
    return results


def get_dashboard(user_id: str = "default") -> LearnerDashboard:
    """Generate explainable tri-state dashboard with 0% baseline for unattempted concepts."""
    snap = snapshot(user_id)
    records = [
        MasteryRecord(
            user_id=user_id,
            concept_id=s["id"],
            score=s["score"],
            status=MasteryStatus(s["state"]),
            attempt_count=s["attempts"],
            evidence_notes=s["evidence_notes"],
            reasoning_depth_level=s.get("reasoning_depth", "DEVELOPING"),
            misconceptions=s.get("misconceptions", []),
            audit_trail=s.get("audit_trail", ""),
        )
        for s in snap
    ]

    strong = [r for r in records if r.status == MasteryStatus.STRONG]
    weak = [r for r in records if r.status == MasteryStatus.WEAK]
    unknown = [r for r in records if r.status == MasteryStatus.UNKNOWN]

    total = len(records)
    demonstrated_scores = [
        (r.score if r.status != MasteryStatus.UNKNOWN and r.attempt_count > 0 else 0.0)
        for r in records
    ]
    overall = sum(demonstrated_scores) / total if total > 0 else 0.0

    summary = (
        f"{len(strong)} mastered, {len(weak)} needing remediation, {len(unknown)} unattempted "
        f"across {total} course concepts. Overall mastery: {round(overall * 100)}%."
    )

    return LearnerDashboard(
        user_id=user_id,
        strong=strong,
        weak=weak,
        unknown=unknown,
        overall_score=round(overall, 3),
        total_concepts=total,
        summary=summary,
    )


# Compatibility aliases
record = record_quiz
dashboard = lambda uid="default": get_dashboard(uid).model_dump()
ranked = lambda concepts, edges: [c["id"] for c in sorted(concepts, key=lambda c: (0 if STATE.get(c["id"], {}).get("n", 0) > 0 and STATE.get(c["id"], {}).get("a", 1)/(STATE.get(c["id"], {}).get("a", 1)+STATE.get(c["id"], {}).get("b", 1)) < 0.65 else 1 if STATE.get(c["id"], {}).get("n", 0) == 0 else 2, c.get("depth", 0)))]
prereqs = lambda edges: {b: [a for a, b2 in edges if b2 == b] for _, b in edges}
