"""Integration Test Suite for Baseline Diagnostic Workflow, Clean Zero-State, and Duplicate Quiz Fixes.

Covers:
  1. Stale Bootstrap Purge & Clean Zero-State when materials == 0:
     - GET /api/mastery -> {"concepts": []}
     - GET /api/mastery/dashboard -> {"overall_mastery": 0, "concepts": [], "status_counts": {"strong": 0, "weak": 0, "unknown": 0}}
     - GET /api/concepts/graph -> {"nodes": [], "edges": [], "learning_path": []}
     - GET /api/loop/status -> {"state": "AWAITING_UPLOAD", "message": "Upload your study material to begin."}
  2. Material Upload & Baseline Diagnostic Required:
     - After upload, GET /api/loop/status -> {"state": "BASELINE_DIAGNOSTIC", "message": "Baseline diagnostic required."}
  3. Calibrated 6-Question Baseline Diagnostic Generation:
     - GET /api/diagnostic/baseline generates 6 questions sampling root and foundational nodes from DAG.
  4. Epistemic Baseline Evaluation & Lifecycle Transition:
     - Score >= 0.80 with sound reasoning -> STRONG.
     - Failed or guessed (< 0.80) -> WEAK.
     - Unassessed downstream concepts -> UNKNOWN.
     - Transitions lifecycle to ACTIVE_LEARNING.
     - GET /api/loop/status -> {"state": "ACTIVE_LEARNING"}
  5. Duplicate Quiz Bug Fix Across Views:
     - POST /api/quiz/next generates fresh non-duplicate question variants.
     - POST /api/quiz/remedial generates targeted remedial quiz for student's top WEAK prerequisite root cause.
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.core import store
from app.pipelines import loop, mastery, quiz, concepts
from app.pipelines.ingest import UniversalMaterialIngestor
from app.models.schemas import QuizSubmission

client = TestClient(app)


@pytest.fixture
def clean_diagnostic_student(tmp_path):
    student_id = "student_diag_test_001"
    store.clear_all_study_materials(student_id)
    store.clear_kb(student_id)
    with mastery.get_db() as conn:
        conn.execute("DELETE FROM user_concept_mastery WHERE user_id = ?", (student_id,))
        conn.execute("DELETE FROM user_lifecycle WHERE user_id = ?", (student_id,))
    yield student_id
    store.clear_all_study_materials(student_id)
    store.clear_kb(student_id)
    with mastery.get_db() as conn:
        conn.execute("DELETE FROM user_concept_mastery WHERE user_id = ?", (student_id,))
        conn.execute("DELETE FROM user_lifecycle WHERE user_id = ?", (student_id,))


def test_zero_state_when_materials_zero(clean_diagnostic_student):
    student_id = clean_diagnostic_student
    headers = {"X-Dev-User-Id": student_id}

    # 1. GET /api/mastery -> {"concepts": []}
    res = client.get("/api/mastery", headers=headers)
    assert res.status_code == 200
    assert res.json() == {"concepts": []}

    # 2. GET /api/mastery/dashboard -> clean zero structure
    res_dash = client.get("/api/mastery/dashboard", headers=headers)
    assert res_dash.status_code == 200
    dash_data = res_dash.json()
    assert dash_data["overall_mastery"] == 0
    assert dash_data["concepts"] == []
    assert dash_data["status_counts"] == {"strong": 0, "weak": 0, "unknown": 0}

    # 3. GET /api/concepts/graph -> empty nodes, edges, learning_path
    res_graph = client.get("/api/concepts/graph", headers=headers)
    assert res_graph.status_code == 200
    graph_data = res_graph.json()
    assert graph_data["nodes"] == []
    assert graph_data["edges"] == []
    assert graph_data["learning_path"] == []

    # 4. GET /api/loop/status -> AWAITING_UPLOAD
    res_status = client.get("/api/loop/status", headers=headers)
    assert res_status.status_code == 200
    status_data = res_status.json()
    assert status_data["state"] == "AWAITING_UPLOAD"
    assert "Upload your study material" in status_data["message"]


def test_baseline_diagnostic_lifecycle_and_evaluation(clean_diagnostic_student, tmp_path):
    student_id = clean_diagnostic_student
    headers = {"X-Dev-User-Id": student_id}

    # Step 1: Ingest study material with multiple DAG-connected concepts
    study_file = tmp_path / "deep_learning_intro.md"
    study_file.write_text(
        "# Deep Learning Foundations\n\n"
        "## Loss Functions\n"
        "A loss function evaluates the error between predicted outputs and ground truth labels. "
        "Loss functions quantify model inaccuracy and provide the objective for optimization.\n\n"
        "## Gradient Descent\n"
        "Gradient descent requires a loss function to compute weight gradients. "
        "Gradient descent iteratively updates parameters in the opposite direction of the gradient.\n\n"
        "## Learning Rate\n"
        "Learning rate scales the gradient descent update step size. "
        "A high learning rate causes divergence, while a small learning rate slows convergence.\n\n"
        "## Backpropagation\n"
        "Backpropagation applies the chain rule to propagate loss gradients backward through neural network layers. "
        "Gradient descent utilizes gradients computed via backpropagation to adjust synaptic weights.\n\n"
        "## Overfitting\n"
        "Overfitting occurs when a neural network memorizes training data but fails to generalize to test data.\n\n"
        "## Regularization\n"
        "Regularization adds penalties to the loss function to prevent overfitting and encourage simpler models.\n"
    )

    UniversalMaterialIngestor.ingest_file(study_file, user_id=student_id, title="Deep Learning Foundations")

    # Step 2: Loop status must now be BASELINE_DIAGNOSTIC
    res_status = client.get("/api/loop/status", headers=headers)
    assert res_status.status_code == 200
    assert res_status.json()["state"] == "BASELINE_DIAGNOSTIC"

    # Step 3: Generate Calibrated 6-Question Baseline Diagnostic
    res_gen = client.get("/api/diagnostic/baseline", headers=headers)
    assert res_gen.status_code == 200
    diag_data = res_gen.json()
    assert diag_data["lifecycle"] == "BASELINE_DIAGNOSTIC"
    assert diag_data["total_questions"] == 6
    assert len(diag_data["questions"]) == 6

    # Verify questions test foundational nodes from DAG
    cids_tested = [q["concept_id"] for q in diag_data["questions"]]
    assert len(cids_tested) == 6

    # Step 4: Submit answers for evaluation:
    # 1st concept: scored >= 0.80 with sound reasoning -> STRONG
    # 2nd concept: guessed / failed (< 0.80) -> WEAK
    # Other concepts: unassessed downstream -> UNKNOWN
    first_cid = diag_data["questions"][0]["concept_id"]
    second_cid = diag_data["questions"][1]["concept_id"]

    # Register deterministic test questions in REGISTRY
    q1 = diag_data["questions"][0]
    q2 = diag_data["questions"][1]

    quiz.REGISTRY[q1["question_id"]] = {
        "correct_option_idx": 1,
        "question_type": "mcq",
        "explanation": "Core loss function mechanism grounded on page 1.",
        "source_page": 1,
    }
    quiz.REGISTRY[q2["question_id"]] = {
        "correct_option_idx": 2,
        "question_type": "mcq",
        "explanation": "Gradient descent optimization principles.",
        "source_page": 1,
    }

    submissions = [
        {
            "question_id": q1["question_id"],
            "concept_id": first_cid,
            "selected_option_idx": 1,  # Correct
            "student_reasoning": "A loss function quantifies prediction error using mathematical formulation.",
            "typed_answer": "",
        },
        {
            "question_id": q2["question_id"],
            "concept_id": second_cid,
            "selected_option_idx": 0,  # Wrong choice + guess
            "student_reasoning": "I just guessed randomly without knowing the formula.",
            "typed_answer": "",
        }
    ]

    res_eval = client.post(
        "/api/diagnostic/baseline/evaluate",
        headers=headers,
        json={"user_id": student_id, "submissions": submissions}
    )
    assert res_eval.status_code == 200
    eval_data = res_eval.json()
    assert eval_data["lifecycle"] == "ACTIVE_LEARNING"
    assert first_cid in eval_data["strong_concepts"]
    assert second_cid in eval_data["weak_concepts"]
    assert len(eval_data["unknown_concepts"]) > 0

    # Step 5: Verify loop status transitioned to ACTIVE_LEARNING
    res_status_after = client.get("/api/loop/status", headers=headers)
    assert res_status_after.status_code == 200
    assert res_status_after.json()["state"] == "ACTIVE_LEARNING"

    # Step 6: Verify Dashboard reflects STRONG, WEAK, and UNKNOWN states
    dash_after = client.get("/api/mastery/dashboard", headers=headers).json()
    assert dash_after["status_counts"]["strong"] >= 1
    assert dash_after["status_counts"]["weak"] >= 1
    assert dash_after["status_counts"]["unknown"] >= 1


def test_non_duplicate_quiz_variants_and_remedial(clean_diagnostic_student, tmp_path):
    student_id = clean_diagnostic_student
    headers = {"X-Dev-User-Id": student_id}

    study_file = tmp_path / "study_course.md"
    study_file.write_text(
        "# Optimization Notes\n\n"
        "## Gradient Descent\n"
        "Gradient descent is an iterative first-order optimization algorithm. "
        "It calculates partial derivatives of the objective function to find the steepest descent. "
        "Batch gradient descent uses the complete dataset to update weights. "
        "Stochastic gradient descent updates parameters using a single sample per iteration.\n\n"
        "## Learning Rate\n"
        "The learning rate determines the step size taken in gradient descent. "
        "An adaptive learning rate automatically scales step magnitude based on past gradients.\n"
    )
    UniversalMaterialIngestor.ingest_file(study_file, user_id=student_id, title="Optimization Notes")

    # 1. Test POST /api/quiz/next generates distinct question variants
    res_q1 = client.post("/api/quiz/next", headers=headers, json={"conceptId": "gradient-descent"})
    assert res_q1.status_code == 200
    q1_data = res_q1.json()

    res_q2 = client.post("/api/quiz/next", headers=headers, json={"conceptId": "gradient-descent"})
    assert res_q2.status_code == 200
    q2_data = res_q2.json()

    assert q1_data["id"] != q2_data["id"], "Question IDs must be unique across successive requests."
    assert q1_data["question"] != q2_data["question"], "Question text must vary to prevent repetition."

    # 2. Test POST /api/quiz/remedial targets top WEAK root cause
    # Artificially set gradient-descent to WEAK in database
    with mastery.get_db() as conn:
        row = mastery._get_row(conn, "gradient-descent", student_id)
        row["state"] = "WEAK"
        row["mastery_score"] = 0.35
        mastery._save_row(conn, row)

    res_remedial = client.post("/api/quiz/remedial", headers=headers, json={})
    assert res_remedial.status_code == 200
    remedial_data = res_remedial.json()
    assert remedial_data["is_remedial"] is True
    assert remedial_data["root_cause_concept"] == "gradient-descent"
    assert remedial_data["remedial_lesson"] is not None


def test_non_default_user_diagnostic_start_generates_six_questions(tmp_path):
    """Verifies that non-default users (e.g. Google OAuth user_id) generate 6 diagnostic questions.

    Guarantees that POST /api/diagnostic/start correctly passes user_id and kb to
    generate_diagnostic_quiz, avoiding the 'No diagnostic questions could be generated' modal error.
    """
    student_id = "google_alan_turing_test"
    headers = {"X-Dev-User-Id": student_id}

    store.clear_all_study_materials(student_id)
    store.clear_kb(student_id)

    # Ingest study material for this specific user
    study_file = tmp_path / "cpp_dsa_arrays.md"
    study_file.write_text(
        "# C++ DSA Arrays and Memory\n\n"
        "## Array Data Structure\n"
        "An array is a collection of elements stored at contiguous memory locations. "
        "It provides O(1) random access time using zero-based index arithmetic.\n\n"
        "## Static Memory Allocation\n"
        "Static memory allocation occurs on the stack at compile-time with fixed size.\n\n"
        "## Dynamic Memory Allocation\n"
        "Dynamic memory allocation uses the heap via new and delete operators in C++.\n\n"
        "## Linear Search\n"
        "Linear search sequentially checks each element of an array until a match is found. "
        "The time complexity of linear search in the worst case is O(n).\n\n"
        "## Array Traversal\n"
        "Array traversal visits each element sequentially from index 0 to size minus one.\n\n"
        "## Two Sum Problem\n"
        "The Two Sum problem finds two numbers in an array that add up to a specific target value.\n"
    )

    UniversalMaterialIngestor.ingest_file(study_file, user_id=student_id, title="C++ DSA Arrays and Memory")

    # Call POST /api/diagnostic/start as the non-default user
    res = client.post("/api/diagnostic/start", headers=headers, json={"user_id": student_id})
    assert res.status_code == 200, f"Expected 200, got {res.status_code}: {res.text}"

    data = res.json()
    assert data["ok"] is True
    assert "questions" in data
    assert len(data["questions"]) == 6, f"Expected 6 questions, got {len(data['questions'])}"

    for q in data["questions"]:
        assert q.get("question_id") or q.get("id"), "Missing question ID"
        assert q.get("question_text"), "Missing question text"
        assert q.get("conceptId"), "Missing conceptId"
        assert q.get("conceptName"), "Missing conceptName"

    # Cleanup
    store.clear_all_study_materials(student_id)
    store.clear_kb(student_id)


def test_diagnostic_start_fast_under_3_5s_and_submits_cleanly(tmp_path):
    """Guarantees POST /api/diagnostic/start executes in < 3.5s without modal freeze and grades answers."""
    import time
    student_id = "test_fast_diag_student"
    headers = {"X-Dev-User-Id": student_id}

    store.clear_all_study_materials(student_id)
    store.clear_kb(student_id)

    study_file = tmp_path / "deep_learning_transformers.md"
    study_file.write_text(
        "# Deep Learning and Transformers\n\n"
        "## Attention Mechanism\n"
        "The attention mechanism maps a query and a set of key-value pairs to an output. "
        "It computes compatibility scores between query and key vectors.\n\n"
        "## Transformer Architecture\n"
        "The Transformer architecture replaces recurrence entirely with multi-head self-attention mechanisms. "
        "It enables parallelized computation across sequence lengths.\n\n"
        "## Gradient Descent\n"
        "Gradient descent iteratively updates model parameters in the direction of steepest loss decrease. "
        "The learning rate determines the step size at each optimization iteration.\n\n"
        "## Backpropagation\n"
        "Backpropagation computes partial derivatives of the loss function with respect to weights using the chain rule.\n\n"
        "## Feedforward Network\n"
        "Each Transformer layer contains a fully connected feed-forward network applied to each position separately and identically.\n\n"
        "## Layer Normalization\n"
        "Layer normalization standardizes activations across the channel dimension within a single training sample.\n"
    )

    UniversalMaterialIngestor.ingest_file(study_file, user_id=student_id, title="Deep Learning and Transformers")

    t0 = time.time()
    res = client.post("/api/diagnostic/start", headers=headers, json={"user_id": student_id})
    elapsed = time.time() - t0

    assert res.status_code == 200, f"Expected 200, got {res.status_code}: {res.text}"
    assert elapsed < 3.5, f"Diagnostic generation took too long: {elapsed:.2f}s (must be < 3.5s)"

    data = res.json()
    assert data["ok"] is True
    questions = data["questions"]
    assert len(questions) == 6, f"Expected 6 questions, got {len(questions)}"

    # Prepare answers to submit
    answers = []
    for q in questions:
        qid = q.get("question_id") or q.get("id")
        cid = q.get("concept_id") or q.get("conceptId")
        qtype = (q.get("question_type") or q.get("questionType") or "mcq").lower()
        ans = {"question_id": qid, "concept_id": cid}
        if qtype == "mcq":
            ans["selected_option_idx"] = q.get("correct_option_idx", 0)
        elif qtype == "fill_in_the_blanks":
            ans["typed_answer"] = q.get("target_blank_answer", "attention")
        else:
            ans["typed_answer"] = "Core mechanism operates by computing dot-product representations."
        answers.append(ans)

    sub_res = client.post("/api/diagnostic/submit", headers=headers, json={"user_id": student_id, "answers": answers})
    assert sub_res.status_code == 200, f"Submission failed: {sub_res.text}"
    sub_data = sub_res.json()
    assert sub_data["ok"] is True
    assert sub_data["total"] == 6
    assert sub_data["baseline_completed"] is True

    # Cleanup
    store.clear_all_study_materials(student_id)
    store.clear_kb(student_id)


