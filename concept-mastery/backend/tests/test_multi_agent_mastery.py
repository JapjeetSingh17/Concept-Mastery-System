"""Comprehensive Automated Test Suite for Multi-Agent Pedagogical Assessment & Head Teacher Arbiter.

Validates:
  1. test_zero_beta_formula_in_mastery_decisions
  2. test_insufficient_evidence_not_penalized_as_weak
  3. test_correct_mcq_with_misconception_explanation_flagged
  4. test_one_good_definition_triggers_independent_application_followup
  5. test_overcoming_earlier_misconception_reconsiders_mastery
  6. test_fabricated_evidence_reference_rejected
  7. test_workflow_limits_prevent_infinite_followup_loops
  8. test_llm_provider_failure_preserves_evidence_and_marks_pending
  9. test_cross_user_evidence_isolation
  10. test_api_evaluate_evidence_endpoint
  11. test_api_concept_deliberation_endpoint
  12. test_backward_compatible_mastery_snapshot
  13. test_hints_used_solution_revealed_blocks_mastery
  14. test_weak_dag_prerequisite_recommends_backtrack
  15. test_dashboard_analytics_integration
  16. test_multiple_verified_applications_grants_mastery
  17. test_specialist_agent_roles_and_confidence
"""

import os
import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient

import uuid

from app.main import app
from app.core import store
from app.models.schemas import (
    EvaluateEvidenceRequest,
    HeadTeacherDecision,
    SpecialistAgentAssessment,
)
from app.pipelines import mastery, mastery_orchestrator
from app.pipelines.mastery_orchestrator import (
    MAX_FOLLOW_UP_QUESTIONS,
    MAX_AGENT_CALLS,
    _deterministic_head_teacher_arbiter,
    _fallback_teaching_agent,
    _fallback_assessment_agent,
    _fallback_misconception_detective,
    _fallback_learning_observer,
)
from app.pipelines.pedagogical_agents import (
    run_teaching_agent,
    run_assessment_agent,
    run_misconception_detective,
    run_learning_observer,
)

client = TestClient(app)


@pytest.fixture(autouse=True)
def init_test_db():
    """Ensure database schema is initialized for every test."""
    store.init_materials_db()
    mastery.init_db()


# ── Test 1: Zero Beta Formula in Mastery Decisions ───────────────────────────
def test_zero_beta_formula_in_mastery_decisions():
    """Confirms outcomes are determined by the Multi-Agent Arbiter, not mathematical 70% threshold."""
    user_id = "test_arbiter_rule_student"
    concept_id = "overfitting"

    # Single attempt with superficial answer
    ev = {
        "evidence_id": "ev_superficial_1",
        "user_id": user_id,
        "concept_id": concept_id,
        "student_answer": "Option C",
        "student_explanation": "I picked C randomly",
        "hints_used": 0,
    }
    pkg = {
        "concept_id": concept_id,
        "concept_name": "Overfitting",
        "attempt_count": 1,
        "follow_up_count": 0,
        "evidence_records": [ev],
    }

    # Traditional formula might compute 1/1 = 100% or 0.70, but Arbiter catches superficial guess
    finding = _fallback_misconception_detective(pkg)
    dec = _deterministic_head_teacher_arbiter(pkg, [finding])

    # Arbiter must withhold MASTERED
    assert dec.outcome != "MASTERED"
    assert dec.outcome in ("NOT_YET_MASTERED", "INSUFFICIENT_EVIDENCE")
    assert dec.legacy_status in ("WEAK", "UNKNOWN")
    assert "guess" in dec.remaining_gaps[0].lower() or "misconception" in dec.explainable_summary.lower()


# ── Test 2: Insufficient Evidence Not Penalized as Weak ──────────────────────
def test_insufficient_evidence_not_penalized_as_weak():
    """0 or 1 brief definitional answer returns INSUFFICIENT_EVIDENCE (UNKNOWN), not NOT_YET_MASTERED."""
    user_id = "test_thin_evidence_student"
    concept_id = "regularization_l2"

    ev = {
        "evidence_id": "ev_thin_1",
        "user_id": user_id,
        "concept_id": concept_id,
        "question_form": "recall",
        "student_answer": "L2 regularization adds a penalty term to loss",
        "student_explanation": "Basic textbook definition",
        "hints_used": 0,
    }
    pkg = {
        "concept_id": concept_id,
        "concept_name": "L2 Regularization",
        "attempt_count": 1,
        "follow_up_count": 0,
        "evidence_records": [ev],
    }

    dec = _deterministic_head_teacher_arbiter(pkg, [])

    # Must be INSUFFICIENT_EVIDENCE (UNKNOWN), NOT NOT_YET_MASTERED (WEAK)
    assert dec.outcome == "INSUFFICIENT_EVIDENCE"
    assert dec.legacy_status == "UNKNOWN"
    assert dec.outcome != "NOT_YET_MASTERED"
    assert dec.follow_up_question is not None


# ── Test 3: Correct MCQ with Misconception Explanation Flagged ───────────────
def test_correct_mcq_with_misconception_explanation_flagged():
    """Correct answer + contradictory explanation flags misconception and withholds MASTERED."""
    concept_id = "gradient_descent"

    ev = {
        "evidence_id": "ev_misconception_1",
        "question_text": "What happens when learning rate is increased?",
        "student_answer": "Option A (converges faster)",
        "student_explanation": "increasing learning rate always guarantees faster convergence",
        "hints_used": 0,
    }
    pkg = {
        "concept_id": concept_id,
        "evidence_history": [ev],
        "evidence_records": [ev],
        "attempt_count": 1,
        "follow_up_count": 0,
    }

    # Specialist Misconception Detective detects the contradictory mental model
    detective_finding = run_misconception_detective(pkg)
    assert any(
        "learning rate" in g.lower() or "convergence" in g.lower()
        for g in detective_finding.identified_gaps
    )

    # Head Teacher Arbiter must withhold MASTERED
    dec = _deterministic_head_teacher_arbiter(pkg, [detective_finding])
    assert dec.outcome != "MASTERED"
    assert dec.outcome in ("INSUFFICIENT_EVIDENCE", "NOT_YET_MASTERED")


# ── Test 4: One Good Definition Triggers Application Follow-up ──────────────
def test_one_good_definition_triggers_independent_application_followup():
    """Good definition alone triggers Assessment Agent application/prediction follow-up."""
    concept_id = "svm_margin"
    cname = "Support Vector Machine Margins"

    ev = {
        "evidence_id": "ev_def_1",
        "question_form": "explanation",
        "student_answer": "SVM finds the maximum-margin hyperplane separating classes.",
        "student_explanation": "The margin is defined by distance to support vectors.",
        "hints_used": 0,
    }
    pkg = {
        "concept_id": concept_id,
        "concept_name": cname,
        "attempt_count": 1,
        "follow_up_count": 0,
        "evidence_records": [ev],
    }

    dec = _deterministic_head_teacher_arbiter(pkg, [])
    assert dec.outcome == "INSUFFICIENT_EVIDENCE"
    assert dec.follow_up_question is not None
    assert dec.follow_up_question.get("question_form") in ("application", "prediction")
    assert "application" in dec.recommended_next_action.lower()


# ── Test 5: Overcoming Earlier Misconception Reconsiders Mastery ─────────────
def test_overcoming_earlier_misconception_reconsiders_mastery():
    """Student fails earlier, receives remediation, later demonstrates independent application -> transitions to MASTERED without permanent penalty."""
    concept_id = "cross_validation"
    cname = "k-Fold Cross Validation"

    # Attempt 1 (earlier): Student admitted lucky guess
    ev1 = {
        "evidence_id": "ev_cv_1",
        "student_answer": "Option B",
        "student_explanation": "I just guessed option B randomly",
        "hints_used": 1,
    }
    # Attempt 2 (later): Independent, unassisted verified application
    ev2 = {
        "evidence_id": "ev_cv_2",
        "question_form": "application",
        "student_answer": "In 5-fold CV, the dataset is split into 5 equal folds. Each fold acts as validation once while 4 folds train the model.",
        "student_explanation": "This guarantees that every data point is tested exactly once without data leakage.",
        "hints_used": 0,
        "solution_revealed_first": False,
    }

    pkg = {
        "concept_id": concept_id,
        "concept_name": cname,
        "attempt_count": 2,
        "follow_up_count": 0,
        "evidence_records": [ev2, ev1],  # Latest first
    }

    dec = _deterministic_head_teacher_arbiter(pkg, [])
    assert dec.outcome == "MASTERED"
    assert dec.legacy_status == "STRONG"
    assert "overcame" in dec.explainable_summary.lower() or "resolved" in dec.demonstrated_understanding[1].lower()


# ── Test 6: Fabricated Evidence Reference Rejected ───────────────────────────
def test_fabricated_evidence_reference_rejected():
    """Non-existent evidence_id (e.g. 'ans_fake_9999') is stripped by verify_evidence_refs."""
    user_id = "test_evidence_verify_user"
    concept_id = "pca_projection"

    # Genuine evidence
    saved = store.save_learning_evidence({
        "user_id": user_id,
        "concept_id": concept_id,
        "student_answer": "PCA projects onto principal components",
    })
    genuine_id = saved["evidence_id"]
    fake_id = "ans_fake_9999"

    verified = store.verify_evidence_refs(user_id, concept_id, [genuine_id, fake_id])
    assert genuine_id in verified
    assert fake_id not in verified
    assert len(verified) == 1


# ── Test 7: Workflow Limits Prevent Infinite Follow-up Loops ────────────────
def test_workflow_limits_prevent_infinite_followup_loops():
    """Enforces MAX_FOLLOW_UP_QUESTIONS (2) and MAX_AGENT_CALLS (6) caps."""
    assert MAX_FOLLOW_UP_QUESTIONS == 2
    assert MAX_AGENT_CALLS == 6

    concept_id = "softmax_regression"
    ev = {
        "evidence_id": "ev_soft_1",
        "student_answer": "Softmax normalizes logits into class probabilities",
        "hints_used": 0,
    }
    # When follow-up count reaches MAX_FOLLOW_UP_QUESTIONS (2)
    pkg = {
        "concept_id": concept_id,
        "attempt_count": 1,
        "follow_up_count": 2,
        "evidence_records": [ev],
    }

    dec = _deterministic_head_teacher_arbiter(pkg, [])
    # No further follow-up probes allowed
    assert dec.follow_up_question is None
    assert dec.outcome in ("MASTERED", "NOT_YET_MASTERED")


# ── Test 8: LLM Provider Failure Preserves Evidence and Marks Pending ────────
def test_llm_provider_failure_preserves_evidence_and_marks_pending():
    """Evidence is persisted in SQLite and evaluation_pending is handled gracefully if LLM fails."""
    user_id = "user_llm_fail_test"
    concept_id = "decision_trees"

    req = EvaluateEvidenceRequest(
        concept_id=concept_id,
        user_id=user_id,
        student_answer="Decision trees split on information gain or Gini impurity",
    )

    with patch.object(
        mastery_orchestrator,
        "run_head_teacher_arbiter",
        side_effect=RuntimeError("Simulated LLM Provider Timeout"),
    ):
        dec, findings = mastery_orchestrator.evaluate_concept_learning_evidence(req, user_id=user_id)

    # 1. Evidence was preserved in SQLite
    ev_records = store.list_learning_evidence(user_id, concept_id)
    assert len(ev_records) >= 1
    assert "information gain" in ev_records[0]["student_answer"]

    # 2. Decision record marked evaluation_pending
    latest_dec = store.get_latest_mastery_decision(user_id, concept_id)
    assert latest_dec is not None
    assert latest_dec.get("evaluation_pending") is True


# ── Test 9: Cross-User Evidence Isolation ────────────────────────────────────
def test_cross_user_evidence_isolation():
    """User A's evidence records and teacher deliberations cannot be seen or cited by User B."""
    user_a = "student_alpha_001"
    user_b = "student_beta_002"
    concept_id = "isolated_concept"

    ev_a = store.save_learning_evidence({
        "user_id": user_a,
        "concept_id": concept_id,
        "student_answer": "Confidential submission by User A",
    })
    store.save_mastery_decision({
        "concept_id": concept_id,
        "outcome": "MASTERED",
        "legacy_status": "STRONG",
        "explainable_summary": "User A demonstrated full mastery",
    }, user_id=user_a)

    # User B checks evidence and decision
    evs_b = store.list_learning_evidence(user_b, concept_id)
    assert len(evs_b) == 0

    dec_b = store.get_latest_mastery_decision(user_b, concept_id)
    assert dec_b is None

    # User B cannot cite User A's evidence ID
    verified_refs_b = store.verify_evidence_refs(user_b, concept_id, [ev_a["evidence_id"]])
    assert len(verified_refs_b) == 0


# ── Test 10: API Evaluate Evidence Endpoint ──────────────────────────────────
def test_api_evaluate_evidence_endpoint():
    """POST /api/mastery/evaluate-evidence returns valid HeadTeacherDecision and 4 specialist findings."""
    payload = {
        "concept_id": "gradient_descent",
        "user_id": "api_test_user_001",
        "question_id": "q_test_101",
        "question_text": "Describe the gradient descent update rule.",
        "question_form": "application",
        "student_answer": "W = W - alpha * dL/dW",
        "student_explanation": "Weights step opposite to the loss gradient multiplied by learning rate.",
        "hints_used": 0,
        "solution_revealed_first": False,
    }

    res = client.post("/api/mastery/evaluate-evidence", json=payload)
    assert res.status_code == 200
    data = res.json()

    assert "decision" in data
    assert "specialist_findings" in data
    dec = data["decision"]
    assert dec["concept_id"] == "gradient_descent"
    assert dec["outcome"] in ("MASTERED", "NOT_YET_MASTERED", "INSUFFICIENT_EVIDENCE")

    findings = data["specialist_findings"]
    assert len(findings) == 4
    roles = {f["agent_role"] for f in findings}
    assert roles == {"teaching_agent", "assessment_agent", "misconception_detective", "learning_observer"}


# ── Test 11: API Concept Deliberation Endpoint ───────────────────────────────
def test_api_concept_deliberation_endpoint():
    """GET /api/mastery/{concept_id}/deliberation returns full explainable decision, specialist findings, and evidence items."""
    user_id = f"delib_user_{uuid.uuid4().hex[:8]}"
    concept_id = f"backprop_{uuid.uuid4().hex[:4]}"

    # Initially, with no decisions, returns clean empty structure
    res_empty = client.get(f"/api/mastery/{concept_id}/deliberation?user_id={user_id}")
    assert res_empty.status_code == 200
    d_empty = res_empty.json()
    assert d_empty["concept_id"] == concept_id
    assert d_empty["decision"] is None
    assert d_empty["specialist_findings"] == []
    assert d_empty["evidence_records"] == []

    # Submit evidence to populate deliberation
    client.post("/api/mastery/evaluate-evidence", json={
        "concept_id": concept_id,
        "user_id": user_id,
        "student_answer": "Backpropagation applies the chain rule backward from the output layer to calculate gradients.",
        "hints_used": 0,
    })

    # Now deliberation returns populated decision and findings
    res_pop = client.get(f"/api/mastery/{concept_id}/deliberation?user_id={user_id}")
    assert res_pop.status_code == 200
    d_pop = res_pop.json()
    assert d_pop["concept_id"] == concept_id
    assert d_pop["decision"] is not None
    assert len(d_pop["specialist_findings"]) == 4
    assert len(d_pop["evidence_records"]) >= 1


# ── Test 12: Backward Compatible Mastery Snapshot ────────────────────────────
def test_backward_compatible_mastery_snapshot():
    """snapshot() returns MASTERED -> STRONG, NOT_YET_MASTERED -> WEAK, INSUFFICIENT_EVIDENCE -> UNKNOWN."""
    user_id = "snapshot_compat_user"

    store.save_kb(
        file="compat.pdf",
        pages=1,
        index={},
        concepts=[
            {"id": "c_strong", "name": "Strong Concept"},
            {"id": "c_weak", "name": "Weak Concept"},
            {"id": "c_unknown", "name": "Unknown Concept"},
        ],
        edges=[],
        user_id=user_id,
    )

    store.save_mastery_decision({
        "concept_id": "c_strong",
        "outcome": "MASTERED",
        "legacy_status": "STRONG",
    }, user_id=user_id)

    store.save_mastery_decision({
        "concept_id": "c_weak",
        "outcome": "NOT_YET_MASTERED",
        "legacy_status": "WEAK",
    }, user_id=user_id)

    store.save_mastery_decision({
        "concept_id": "c_unknown",
        "outcome": "INSUFFICIENT_EVIDENCE",
        "legacy_status": "UNKNOWN",
    }, user_id=user_id)

    snap = mastery.snapshot(user_id)
    s_map = {item["id"]: item for item in snap}

    assert s_map["c_strong"]["state"] == "STRONG"
    assert s_map["c_strong"]["outcome"] == "MASTERED"

    assert s_map["c_weak"]["state"] == "WEAK"
    assert s_map["c_weak"]["outcome"] == "NOT_YET_MASTERED"

    assert s_map["c_unknown"]["state"] == "UNKNOWN"
    assert s_map["c_unknown"]["outcome"] == "INSUFFICIENT_EVIDENCE"


# ── Test 13: Hints Used & Solution Revealed Blocks Mastery ───────────────────
def test_hints_used_solution_revealed_blocks_mastery():
    """hints_used > 0 prevents instant MASTERED and requires unassisted follow-up."""
    concept_id = "logistic_regression"

    ev = {
        "evidence_id": "ev_hint_1",
        "student_answer": "Sigmoid output function",
        "hints_used": 2,
        "solution_revealed_first": False,
    }
    pkg = {
        "concept_id": concept_id,
        "concept_name": "Logistic Regression",
        "attempt_count": 1,
        "follow_up_count": 0,
        "evidence_records": [ev],
    }

    dec = _deterministic_head_teacher_arbiter(pkg, [])
    # Hints prevent instant MASTERED
    assert dec.outcome != "MASTERED"
    assert dec.follow_up_question is not None

    # Test solution revealed first
    ev_rev = {
        "evidence_id": "ev_rev_1",
        "student_answer": "Sigmoid formula",
        "hints_used": 0,
        "solution_revealed_first": True,
    }
    pkg_rev = {
        "concept_id": concept_id,
        "concept_name": "Logistic Regression",
        "attempt_count": 1,
        "follow_up_count": 0,
        "evidence_records": [ev_rev],
    }
    dec_rev = _deterministic_head_teacher_arbiter(pkg_rev, [])
    assert dec_rev.outcome != "MASTERED"


# ── Test 14: Weak DAG Prerequisite Recommends Backtrack ──────────────────────
def test_weak_dag_prerequisite_recommends_backtrack():
    """When a parent prerequisite is WEAK, Head Teacher recommends remediation on parent node."""
    concept_id = "convolutional_networks"
    parent_prereq = "matrix_multiplication"

    pkg = {
        "concept_id": concept_id,
        "concept_name": "Convolutional Neural Networks",
        "prerequisites": [{"concept_id": parent_prereq, "status": "WEAK", "outcome": "NOT_YET_MASTERED"}],
        "evidence_records": [{"evidence_id": "ev_cnn_1", "student_answer": "Convolutions slide kernels across inputs"}],
    }

    dec = _deterministic_head_teacher_arbiter(pkg, [])
    assert dec.outcome == "NOT_YET_MASTERED"
    assert dec.legacy_status == "WEAK"
    assert parent_prereq in dec.recommended_next_action
    assert "Backtrack" in dec.recommended_next_action


# ── Test 15: Dashboard Analytics Integration ─────────────────────────────────
def test_dashboard_analytics_integration():
    """/api/mastery/dashboard reports correct status counts and qualitative states."""
    user_id = "dashboard_integration_user"

    store.save_kb(
        file="dashboard_test.pdf",
        pages=1,
        index={},
        concepts=[
            {"id": "m1", "name": "Concept 1"},
            {"id": "m2", "name": "Concept 2"},
            {"id": "m3", "name": "Concept 3"},
        ],
        edges=[],
        user_id=user_id,
    )

    store.save_mastery_decision({
        "concept_id": "m1",
        "outcome": "MASTERED",
        "legacy_status": "STRONG",
    }, user_id=user_id)

    store.save_mastery_decision({
        "concept_id": "m2",
        "outcome": "NOT_YET_MASTERED",
        "legacy_status": "WEAK",
    }, user_id=user_id)

    res = client.get(f"/api/mastery/dashboard?user_id={user_id}")
    assert res.status_code == 200
    data = res.json()

    assert "status_counts" in data
    counts = data["status_counts"]
    assert counts["strong"] == 1
    assert counts["weak"] == 1
    assert counts["unknown"] == 1
    assert len(data["concepts"]) == 3


# ── Test 16: Multiple Verified Applications Grants Mastery ───────────────────
def test_multiple_verified_applications_grants_mastery():
    """2+ verified independent applications with 0 hints and correct reasoning grant MASTERED (STRONG)."""
    concept_id = "random_forests"

    ev1 = {
        "evidence_id": "ev_rf_1",
        "question_form": "explanation",
        "student_answer": "Random Forests construct an ensemble of decision trees using bagging and random feature selection.",
        "hints_used": 0,
        "solution_revealed_first": False,
    }
    ev2 = {
        "evidence_id": "ev_rf_2",
        "question_form": "application",
        "student_answer": "By decorrelating trees through random subspace sampling, variance is reduced without increasing bias.",
        "hints_used": 0,
        "solution_revealed_first": False,
    }

    pkg = {
        "concept_id": concept_id,
        "concept_name": "Random Forests",
        "attempt_count": 2,
        "follow_up_count": 0,
        "evidence_records": [ev2, ev1],
    }

    dec = _deterministic_head_teacher_arbiter(pkg, [])
    assert dec.outcome == "MASTERED"
    assert dec.legacy_status == "STRONG"
    assert dec.follow_up_question is None


# ── Test 17: Specialist Agent Roles & Confidence ─────────────────────────────
def test_specialist_agent_roles_and_confidence():
    """Verifies all 4 pedagogical specialist roles produce valid SpecialistAgentAssessment objects."""
    concept_id = "bayes_theorem"
    ev = {
        "evidence_id": "ev_bayes_1",
        "student_answer": "P(A|B) = P(B|A) * P(A) / P(B)",
        "student_explanation": "Posterior equals likelihood times prior divided by marginal evidence.",
        "hints_used": 0,
    }
    pkg = {
        "concept_id": concept_id,
        "concept_name": "Bayes' Theorem",
        "evidence_history": [ev],
        "evidence_records": [ev],
        "attempt_count": 1,
    }

    t = run_teaching_agent(pkg)
    a = run_assessment_agent(pkg)
    m = run_misconception_detective(pkg)
    l = run_learning_observer(pkg)

    assert isinstance(t, SpecialistAgentAssessment) and t.agent_role == "teaching_agent"
    assert isinstance(a, SpecialistAgentAssessment) and a.agent_role == "assessment_agent"
    assert isinstance(m, SpecialistAgentAssessment) and m.agent_role == "misconception_detective"
    assert isinstance(l, SpecialistAgentAssessment) and l.agent_role == "learning_observer"

    for specialist in (t, a, m, l):
        assert specialist.concept_id == concept_id
        assert specialist.assessment_confidence in ("low", "moderate", "high")
        assert len(specialist.findings) > 0
