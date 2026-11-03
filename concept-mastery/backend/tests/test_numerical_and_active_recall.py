"""Unit and integration test suite for Numerical Question Engine & Active Recall Pipeline."""

import pytest
from fastapi.testclient import TestClient

from app import config
from app.core import store
from app.core.concept_index import GLOBAL_INDEX
from app.main import app
from app.models.schemas import (
    QuizQuestion,
    QuizSubmission,
    QuizResult,
    ActiveRecallStartRequest,
    ActiveRecallStudyCard,
    ActiveRecallEvaluateRequest,
    ActiveRecallResult,
    ActiveRecallHistoryItem,
    ChunkMetadata,
)
from app.pipelines import quiz, active_recall, mastery


@pytest.fixture(autouse=True)
def setup_test_kb():
    """Seed test knowledge base and BM25 index with sample concepts and chunks."""
    user_id = "test_architect_user"
    store.init_materials_db()
    store.clear_kb(user_id)

    sample_concepts = [
        {
            "id": "gradient_descent",
            "name": "Gradient Descent",
            "definition": "An iterative first-order optimization algorithm for finding a local minimum of a differentiable function using negative gradient steps.",
            "chunks": [0],
            "prerequisites": [],
        },
        {
            "id": "knn_distance",
            "name": "KNN Euclidean Distance",
            "definition": "Non-parametric classification method calculating Euclidean metric distance d = sqrt(sum((xi - yi)^2)) between feature points.",
            "chunks": [1],
            "prerequisites": [],
        },
        {
            "id": "mse_loss",
            "name": "Mean Squared Error Loss",
            "definition": "Loss metric quantifying average squared differences between predicted values y_hat and ground truth targets y.",
            "chunks": [2],
            "prerequisites": [],
        },
        {
            "id": "cross_entropy",
            "name": "Cross-Entropy Loss",
            "definition": "Information theoretic loss quantifying performance of classification model outputting probability distribution p.",
            "chunks": [3],
            "prerequisites": [],
        },
        {
            "id": "precision_recall",
            "name": "Precision and Recall",
            "definition": "Diagnostic classification performance metrics measuring exactness TP/(TP+FP) and completeness TP/(TP+FN).",
            "chunks": [4],
            "prerequisites": [],
        },
        {
            "id": "bayes_theorem",
            "name": "Bayes' Theorem",
            "definition": "Mathematical formulation determining conditional posterior probability P(A|B) from prior P(A) and likelihood P(B|A).",
            "chunks": [5],
            "prerequisites": [],
        },
        {
            "id": "learning_rate_schedule",
            "name": "Learning Rate Schedule",
            "definition": "Optimization schedule dynamically decaying learning rate alpha_t = alpha_0 * gamma^t over successive training epochs.",
            "chunks": [6],
            "prerequisites": [],
        },
        {
            "id": "matrix_dot_product",
            "name": "Matrix Dot Products",
            "definition": "Algebraic inner product operation summing component-wise multiplications across dimensional vector arrays.",
            "chunks": [7],
            "prerequisites": [],
        },
    ]

    sample_chunks = [
        ChunkMetadata(
            chunk_id=f"chk_{i}",
            page_number=i + 1,
            section_title=f"Section {c['name']}",
            text=f"{c['name']}: {c['definition']} Applied in algorithmic models with exact formula and parameters.",
            canonical_concepts=[c["id"]],
            source_filename="optimization_handbook.pdf",
            locator_label=f"Page {i + 1}, Section: {c['name']}",
            source_type="pdf",
        )
        for i, c in enumerate(sample_concepts)
    ]

    GLOBAL_INDEX.index_chunks(sample_chunks, user_id=user_id)
    store.save_kb(
        file="optimization_handbook.pdf",
        pages=len(sample_concepts),
        index=None,
        concepts=sample_concepts,
        edges=[],
        user_id=user_id,
    )
    yield user_id


# ── 1. Schema Validation Tests ─────────────────────────────────────────────

def test_quiz_question_schema_numerical_extensions():
    """Verify QuizQuestion supports numerical question_type and mathematical fields."""
    q = QuizQuestion(
        question_id="num_test_001",
        concept_id="gradient_descent",
        question_type="numerical",
        question_text="Calculate weight update given w0=2.0, alpha=0.1, grad=1.5.",
        explanation="w_new = 2.0 - (0.1 * 1.5) = 1.85",
        source_page=1,
        given_parameters={"w0": 2.0, "alpha": 0.1, "gradient": 1.5},
        formula_hint="w_new = w0 - alpha * grad",
        correct_numeric_answer=1.85,
        tolerance=0.05,
        unit_or_format="Round to 2 decimal places",
        step_by_step_solution="Step 1: w0=2.0\nStep 2: alpha*grad=0.15\nStep 3: 2.0 - 0.15 = 1.85",
    )
    assert q.question_type == "numerical"
    assert q.correct_numeric_answer == 1.85
    assert q.tolerance == 0.05
    assert q.given_parameters["w0"] == 2.0
    assert "Step 1" in q.step_by_step_solution


def test_quiz_submission_schema_numerical_field():
    """Verify QuizSubmission supports numeric_answer field."""
    sub = QuizSubmission(
        user_id="u123",
        question_id="q1",
        concept_id="gradient_descent",
        numeric_answer=1.85,
        student_reasoning="Used standard gradient update formula.",
    )
    assert sub.numeric_answer == 1.85


def test_active_recall_schemas():
    """Verify Active Recall Pydantic models validate correctly."""
    start_req = ActiveRecallStartRequest(concept_id="gradient_descent", timer_seconds=180, user_id="u1")
    assert start_req.timer_seconds == 180

    card = ActiveRecallStudyCard(
        session_id="ar_test1",
        concept_id="gradient_descent",
        concept_name="Gradient Descent",
        timer_seconds=180,
        core_intuition="Optimization method finding local minima.",
        key_formula_or_mechanism="w = w - alpha * grad",
        worked_mini_example="w=1.5, alpha=0.1, g=2 -> 1.3",
        essential_keywords=["gradient", "weight", "learning rate", "loss"],
        exact_source_citation="[Page 1, Section: Optimization]",
    )
    assert len(card.essential_keywords) == 4
    assert card.session_id == "ar_test1"

    eval_req = ActiveRecallEvaluateRequest(
        session_id="ar_test1",
        concept_id="gradient_descent",
        recall_text="Gradient descent updates weights by subtracting learning rate times loss gradient.",
        verification_answer=1.3,
        user_id="u1",
    )
    assert eval_req.verification_answer == 1.3


# ── 2. Numerical Question Generator Tests ──────────────────────────────────

def test_concept_aware_numerical_question_generation_all_eight_concepts():
    """Verify numerical questions are generated correctly for all 8 required concept families."""
    concepts_to_test = [
        "gradient_descent",
        "knn_distance",
        "mse_loss",
        "cross_entropy",
        "precision_recall",
        "bayes_theorem",
        "learning_rate_schedule",
        "matrix_dot_product",
    ]

    for cid in concepts_to_test:
        q = quiz.generate_numerical_question(cid, concept_name=cid.replace("_", " ").title())
        assert q.question_type == "numerical"
        assert q.correct_numeric_answer is not None
        assert isinstance(q.correct_numeric_answer, float)
        assert q.tolerance == 0.05
        assert q.formula_hint is not None and len(q.formula_hint) > 0
        assert q.step_by_step_solution is not None and "Step 1" in q.step_by_step_solution
        assert q.given_parameters is not None and len(q.given_parameters) > 0


def test_is_mathematical_concept_detection():
    """Verify mathematical and algorithmic concept detector."""
    assert quiz.is_mathematical_concept("gradient_descent") is True
    assert quiz.is_mathematical_concept("knn_distance") is True
    assert quiz.is_mathematical_concept("mse_loss") is True
    assert quiz.is_mathematical_concept("cross_entropy") is True
    assert quiz.is_mathematical_concept("linear_algebra_matrix") is True
    assert quiz.is_mathematical_concept("history_of_computing", "General History") is False


# ── 3. Numerical Evaluation Tests ──────────────────────────────────────────

def test_evaluate_submission_numerical_correct_within_tolerance():
    """Verify evaluation returns is_correct=True and step_by_step_solution within tolerance."""
    q = quiz.generate_numerical_question("gradient_descent")
    target = q.correct_numeric_answer

    # Slightly offset within tolerance of 0.05
    student_answer = round(target + 0.02, 3)

    sub = QuizSubmission(
        user_id="test_student",
        question_id=q.question_id,
        concept_id=q.concept_id,
        numeric_answer=student_answer,
        student_reasoning="Applied w_new = w0 - alpha * grad",
    )
    res = quiz.evaluate_submission(sub)

    assert res.is_correct is True
    assert res.score == 1.0
    assert res.step_by_step_solution == q.step_by_step_solution
    assert res.question_type == "numerical"


def test_evaluate_submission_numerical_incorrect_beyond_tolerance():
    """Verify evaluation returns is_correct=False and step_by_step_solution beyond tolerance."""
    q = quiz.generate_numerical_question("mse_loss")
    target = q.correct_numeric_answer

    sub = QuizSubmission(
        user_id="test_student",
        question_id=q.question_id,
        concept_id=q.concept_id,
        numeric_answer=target + 5.0,  # Far outside tolerance
        student_reasoning="Estimated rough average",
    )
    res = quiz.evaluate_submission(sub)

    assert res.is_correct is False
    assert res.score == 0.0
    assert res.step_by_step_solution == q.step_by_step_solution
    assert len(res.misconceptions) > 0


def test_evaluate_submission_numerical_string_parsing():
    """Verify evaluation parses string numeric inputs from typed_answer."""
    q = quiz.generate_numerical_question("cross_entropy")
    target = q.correct_numeric_answer

    sub = QuizSubmission(
        user_id="test_student",
        question_id=q.question_id,
        concept_id=q.concept_id,
        typed_answer=f"The calculated loss is {target:.2f}",
    )
    res = quiz.evaluate_submission(sub)

    assert res.is_correct is True
    assert res.score == 1.0


# ── 4. SQLite Persistence in store.py Tests ────────────────────────────────

def test_store_active_recall_session_crud():
    """Verify SQLite table active_recall_sessions CRUD operations."""
    sid = "ar_crud_test_001"
    uid = "test_user_persistence"
    cid = "gradient_descent"

    # 1. Save session
    saved = store.save_active_recall_session({
        "session_id": sid,
        "user_id": uid,
        "concept_id": cid,
        "timer_seconds": 90,
        "lesson_json": {"core_intuition": "Gradient test intuition"},
        "verification_question_json": {"qid": "vq1"},
    })
    assert saved["session_id"] == sid
    assert saved["timer_seconds"] == 90

    # 2. Get session
    retrieved = store.get_active_recall_session(sid)
    assert retrieved is not None
    assert retrieved["user_id"] == uid
    assert retrieved["concept_id"] == cid

    # 3. Update session
    updated = store.update_active_recall_session(sid, {
        "recall_submission_text": "Student remembered w - alpha * grad.",
        "retrieval_score": 0.88,
        "retained_concepts_json": ["gradient", "weight"],
        "missed_gaps_json": ["learning rate"],
    })
    assert updated["retrieval_score"] == 0.88
    assert "Student remembered" in updated["recall_submission_text"]

    # 4. List sessions
    sessions = store.list_active_recall_sessions(uid)
    assert len(sessions) >= 1
    assert any(s["session_id"] == sid for s in sessions)


# ── 5. Active Recall Pipeline Engine Tests ─────────────────────────────────

def test_active_recall_start_session(setup_test_kb):
    """Verify active recall start_session generates complete Study Card and verification question."""
    uid = setup_test_kb
    card = active_recall.start_session(user_id=uid, concept_id="gradient_descent", timer_seconds=150)

    assert card.concept_id == "gradient_descent"
    assert card.timer_seconds == 150
    assert len(card.core_intuition) > 0
    assert len(card.key_formula_or_mechanism) > 0
    assert len(card.worked_mini_example) > 0
    assert 3 <= len(card.essential_keywords) <= 5
    assert "[" in card.exact_source_citation
    assert card.verification_question is not None

    # Check persistence in SQLite
    sess = store.get_active_recall_session(card.session_id)
    assert sess is not None


def test_active_recall_evaluate_session_high_score(setup_test_kb):
    """Verify active recall evaluation with comprehensive retrieval and correct verification."""
    uid = setup_test_kb
    card = active_recall.start_session(user_id=uid, concept_id="gradient_descent", timer_seconds=120)

    vq = card.verification_question
    ans = vq.get("correct_numeric_answer") if vq else 1.0

    recall_text = (
        f"Gradient descent is an iterative optimization algorithm that updates weights by taking steps proportional "
        f"to the negative gradient of the loss function. The update formula is w_new = w - alpha * grad. "
        f"Keywords: {', '.join(card.essential_keywords)}."
    )

    result = active_recall.evaluate_session(
        user_id=uid,
        session_id=card.session_id,
        concept_id="gradient_descent",
        recall_text=recall_text,
        verification_answer=ans,
    )

    assert result.retrieval_score >= 0.70
    assert result.verification_correct is True
    assert len(result.retained_concepts) > 0
    assert result.reasoning_depth_level in ("DEVELOPING", "DEEP_CONCEPTUAL")

    # Epistemic mastery record should be updated in SQLite
    rec = mastery.get_record(uid, "gradient_descent")
    assert rec.attempt_count >= 1
    assert rec.score >= 0.50


def test_active_recall_evaluate_session_low_score(setup_test_kb):
    """Verify active recall evaluation with empty or poor recall."""
    uid = setup_test_kb
    card = active_recall.start_session(user_id=uid, concept_id="knn_distance", timer_seconds=120)

    result = active_recall.evaluate_session(
        user_id=uid,
        session_id=card.session_id,
        concept_id="knn_distance",
        recall_text="I don't remember the formula.",
        verification_answer=-999.0,
    )

    assert result.retrieval_score < 0.50
    assert result.verification_correct is False
    assert result.reasoning_depth_level == "SURFACE_ROTE"
    assert len(result.missed_gaps) > 0


def test_active_recall_get_history(setup_test_kb):
    """Verify learner history retrieval."""
    uid = setup_test_kb
    card = active_recall.start_session(user_id=uid, concept_id="mse_loss", timer_seconds=90)
    active_recall.evaluate_session(
        user_id=uid,
        session_id=card.session_id,
        concept_id="mse_loss",
        recall_text="MSE calculates mean squared error across targets and predictions.",
        verification_answer=None,
    )

    history = active_recall.get_history(user_id=uid)
    assert len(history) >= 1
    item = next(h for h in history if h.session_id == card.session_id)
    assert item.concept_id == "mse_loss"
    assert item.timer_seconds == 90
    assert item.retrieval_score is not None


# ── 6. REST API Endpoint Tests ─────────────────────────────────────────────

def test_api_active_recall_endpoints(setup_test_kb):
    """Verify REST endpoints POST /start, POST /evaluate, and GET /history."""
    uid = setup_test_kb
    client = TestClient(app)

    # 1. Start Session
    start_payload = {
        "concept_id": "gradient_descent",
        "timer_seconds": 120,
        "user_id": uid,
    }
    resp_start = client.post("/api/active-recall/start", json=start_payload)
    assert resp_start.status_code == 200, resp_start.text
    card_data = resp_start.json()
    assert "session_id" in card_data
    assert card_data["concept_id"] == "gradient_descent"
    assert len(card_data["essential_keywords"]) >= 3

    session_id = card_data["session_id"]
    vq = card_data.get("verification_question", {})
    ans = vq.get("correct_numeric_answer", 1.0)

    # 2. Evaluate Session
    eval_payload = {
        "session_id": session_id,
        "concept_id": "gradient_descent",
        "recall_text": f"Gradient descent minimizes loss using {', '.join(card_data['essential_keywords'])}.",
        "verification_answer": ans,
        "user_id": uid,
    }
    resp_eval = client.post("/api/active-recall/evaluate", json=eval_payload)
    assert resp_eval.status_code == 200, resp_eval.text
    eval_data = resp_eval.json()
    assert eval_data["session_id"] == session_id
    assert "retrieval_score" in eval_data
    assert "reasoning_depth_level" in eval_data

    # 3. History
    resp_hist = client.get(f"/api/active-recall/history?user_id={uid}")
    assert resp_hist.status_code == 200, resp_hist.text
    hist_data = resp_hist.json()
    assert len(hist_data) >= 1
    assert any(h["session_id"] == session_id for h in hist_data)
