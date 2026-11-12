"""Tests for Real-Time Ingestion Progress Bar, Compulsory Baseline Diagnostic,
and Grounded Post-Baseline Study Guide Reports.
"""

import pytest
from fastapi.testclient import TestClient
from pathlib import Path

from app.main import app
from app.core import store
from app.pipelines import ingest, quiz
from app.pipelines.ingest import UniversalMaterialIngestor

client = TestClient(app)


@pytest.fixture
def clean_user(tmp_path):
    user_id = "test_user_progress_and_report"
    store.clear_all_study_materials(user_id)
    store.clear_kb(user_id)
    yield user_id
    store.clear_all_study_materials(user_id)
    store.clear_kb(user_id)


def test_real_time_ingest_progress_milestones():
    mat_id = "test_mat_progress_001"

    # Test setting and reading progress milestones
    ingest.set_ingest_progress(mat_id, "Extracting transcript / document text...", 15, "Parsing document...", status="processing")
    p1 = ingest.get_ingest_progress(mat_id)
    assert p1["progress_pct"] == 15
    assert p1["stage"] == "Extracting transcript / document text..."
    assert p1["status"] == "processing"

    ingest.set_ingest_progress(mat_id, "Performing boundary-aware semantic chunking...", 40, "Chunking document...", status="processing")
    p2 = ingest.get_ingest_progress(mat_id)
    assert p2["progress_pct"] == 40

    ingest.set_ingest_progress(mat_id, "Building vector & BM25/TF-IDF concept indices...", 65, "Indexing concepts...", status="processing")
    p3 = ingest.get_ingest_progress(mat_id)
    assert p3["progress_pct"] == 65

    ingest.set_ingest_progress(mat_id, "Extracting Concept Prerequisite DAG via LLM...", 85, "Constructing DAG...", status="processing")
    p4 = ingest.get_ingest_progress(mat_id)
    assert p4["progress_pct"] == 85

    ingest.set_ingest_progress(mat_id, "Material indexed. Launching Baseline Diagnostic...", 100, "Done.", status="ready")
    p5 = ingest.get_ingest_progress(mat_id)
    assert p5["progress_pct"] == 100
    assert p5["status"] == "ready"

    # Test endpoint GET /api/materials/{material_id}/ingest-status
    res = client.get(f"/api/materials/{mat_id}/ingest-status")
    assert res.status_code == 200
    data = res.json()
    assert data["ok"] is True
    assert data["progress_pct"] == 100
    assert data["status"] == "ready"


def test_baseline_diagnostic_report_generation_and_persistence(clean_user, tmp_path):
    user_id = clean_user
    headers = {"X-Dev-User-Id": user_id}

    # Ingest document
    study_file = tmp_path / "deep_learning_course.md"
    study_file.write_text(
        "# Deep Learning Foundations\n\n"
        "## Attention Mechanism\n"
        "Attention dynamically weights sequence positions based on relevance. "
        "The formula computes scaled dot-product attention softmax(QK^T / sqrt(d_k))V.\n\n"
        "## Transformer Layer\n"
        "A transformer layer combines multi-head attention mechanisms with feed-forward sublayers. "
        "Attention mechanisms pass representations into the feed-forward networks.\n"
    )

    mat_id = "mat_dl_foundations_01"
    UniversalMaterialIngestor.ingest_file(
        study_file,
        user_id=user_id,
        title="Deep Learning Foundations",
        material_id=mat_id,
    )

    # Verify material has baseline_completed = False initially
    mat = store.get_study_material(mat_id, user_id=user_id)
    assert mat is not None
    assert mat.get("baseline_completed") is False

    # Simulate diagnostic submission with 1 strong and 1 weak concept
    q1_id = "diag_q_attention_1"
    q2_id = "diag_q_transformer_2"

    quiz.REGISTRY[q1_id] = {
        "correct_option_idx": 0,
        "question_type": "mcq",
        "explanation": "Attention mechanisms dynamically compute similarity weights.",
        "source_page": 1,
    }
    quiz.REGISTRY[q2_id] = {
        "correct_option_idx": 1,
        "question_type": "mcq",
        "explanation": "Transformer layers stack multi-head self-attention with feedforward layers.",
        "source_page": 1,
    }

    submissions = [
        {
            "question_id": q1_id,
            "concept_id": "attention-mechanism",
            "selected_option_idx": 0,  # Correct
            "student_reasoning": "Attention computes dot products between queries and keys.",
            "typed_answer": "A",
        },
        {
            "question_id": q2_id,
            "concept_id": "transformer-layer",
            "selected_option_idx": 3,  # Incorrect
            "student_reasoning": "I wasn't sure about the architecture layers.",
            "typed_answer": "D",
        },
    ]

    # Submit via POST /api/diagnostic/submit with material_id
    submit_res = client.post(
        "/api/diagnostic/submit",
        headers=headers,
        json={
            "user_id": user_id,
            "material_id": mat_id,
            "answers": submissions,
        },
    )
    assert submit_res.status_code == 200
    res_data = submit_res.json()
    assert res_data["ok"] is True
    assert "report" in res_data

    report = res_data["report"]
    assert "overall_score" in report
    assert "topic_breakdown" in report
    assert "weak_topics_guide" in report

    # Verify breakdown
    breakdown = report["topic_breakdown"]
    assert "weak" in breakdown
    assert "fragile" in breakdown
    assert "strong" in breakdown

    # Verify weak topics guide sections
    guides = report["weak_topics_guide"]
    assert len(guides) >= 1
    guide = guides[0]
    assert "concept_id" in guide
    assert "why_missed" in guide
    assert "detailed_explanation" in guide
    assert "mental_model" in guide
    assert "action_plan" in guide
    assert len(guide["action_plan"]) >= 1

    # Verify persistence in store.py
    mat_after = store.get_study_material(mat_id, user_id=user_id)
    assert mat_after["baseline_completed"] is True
    assert isinstance(mat_after["baseline_report"], dict)
    assert mat_after["baseline_report"]["overall_score"] == report["overall_score"]

    # Verify GET /api/materials/{material_id}/baseline-report
    get_report_res = client.get(f"/api/materials/{mat_id}/baseline-report", headers=headers)
    assert get_report_res.status_code == 200
    fetched_data = get_report_res.json()
    assert fetched_data["ok"] is True
    assert fetched_data["baseline_completed"] is True
    assert fetched_data["report"]["overall_score"] == report["overall_score"]

    # Verify GET /api/materials/latest/baseline-report
    get_latest_res = client.get("/api/materials/latest/baseline-report", headers=headers)
    assert get_latest_res.status_code == 200
    latest_data = get_latest_res.json()
    assert latest_data["ok"] is True
    assert latest_data["baseline_completed"] is True
    assert latest_data["material_id"] == mat_id
