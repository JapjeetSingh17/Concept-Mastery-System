"""Unit and integration tests for batched /api/dashboard-summary,
store.get_dashboard_summary, deterministic recommend heuristics, and SQLite indexes.
"""

import time
import pytest
import sqlite3
from pathlib import Path
from fastapi.testclient import TestClient

from app.main import app
from app import config
from app.core import store, llm
from app.pipelines import recommend, mastery


@pytest.fixture(autouse=True)
def clean_state():
    """Ensure clean test state for store and KB."""
    store.clear_kb("test_dash_user")
    store.clear_all_study_materials("test_dash_user")
    with store._materials_transaction(immediate=True) as conn:
        conn.execute("DELETE FROM smart_notes WHERE user_id = ?", ("test_dash_user",))
        conn.execute("DELETE FROM learning_evidence_store WHERE user_id = ?", ("test_dash_user",))
        conn.execute("DELETE FROM active_recall_sessions WHERE user_id = ?", ("test_dash_user",))
        conn.execute("DELETE FROM explainable_mastery_decisions WHERE user_id = ?", ("test_dash_user",))
        conn.execute("DELETE FROM user_concept_mastery WHERE user_id = ?", ("test_dash_user",))
    yield
    store.clear_kb("test_dash_user")
    store.clear_all_study_materials("test_dash_user")


def test_sqlite_indexes_exist():
    """Verify composite indexes exist on (user_id, material_id) and (user_id, concept_id)."""
    with store._materials_transaction(immediate=False) as conn:
        indexes = conn.execute("SELECT name, tbl_name, sql FROM sqlite_master WHERE type = 'index'").fetchall()
        index_sqls = {r["name"]: (r["sql"] or "") for r in indexes}

        # 1. study_materials(user_id, material_id)
        assert "idx_materials_user_material" in index_sqls
        assert "(user_id, material_id)" in index_sqls["idx_materials_user_material"]

        # 2. active_recall_sessions(user_id, concept_id)
        assert "idx_active_recall_user_concept" in index_sqls
        assert "(user_id, concept_id)" in index_sqls["idx_active_recall_user_concept"]

        # 3. smart_notes(user_id, concept_id) and (user_id, material_id)
        assert "idx_smart_notes_user_concept" in index_sqls
        assert "(user_id, concept_id)" in index_sqls["idx_smart_notes_user_concept"]
        assert "idx_smart_notes_user_material" in index_sqls
        assert "(user_id, material_id)" in index_sqls["idx_smart_notes_user_material"]

        # 4. learning_evidence_store(user_id, concept_id)
        assert "idx_learning_evidence_user_concept" in index_sqls
        assert "(user_id, concept_id)" in index_sqls["idx_learning_evidence_user_concept"]

        # 5. explainable_mastery_decisions(user_id, concept_id)
        assert "idx_mastery_decisions_user_concept" in index_sqls
        assert "(user_id, concept_id)" in index_sqls["idx_mastery_decisions_user_concept"]


def test_store_get_dashboard_summary_empty():
    """Verify get_dashboard_summary handles empty user state gracefully."""
    summary = store.get_dashboard_summary(user_id="test_dash_user")
    assert summary["ok"] is True
    assert summary["user_id"] == "test_dash_user"
    assert summary["material"] is None
    assert summary["materials"] == []
    assert summary["concepts"] == []
    assert summary["edges"] == []
    assert summary["notes"] == []
    assert summary["stats"]["total_concepts"] == 0
    assert summary["stats"]["average_score"] == 0.0


def test_store_get_dashboard_summary_batched_join():
    """Verify get_dashboard_summary fetches materials, concepts, mastery, and notes in single batched query."""
    uid = "test_dash_user"

    # Ingest study material
    mat = store.save_study_material({
        "material_id": "mat_test_01",
        "user_id": uid,
        "title": "Machine Learning Foundations",
        "source_type": "pdf",
        "original_filename_or_url": "ml.pdf",
        "chunk_count": 12,
        "concepts": [
            {"id": "c1", "name": "Linear Algebra", "summary": "Vectors and matrices"},
            {"id": "c2", "name": "Gradient Descent", "summary": "First-order optimization"}
        ]
    })

    # Save KB with prerequisite edge c1 -> c2
    concepts = [
        {"id": "c1", "name": "Linear Algebra", "summary": "Vectors and matrices", "chunks": [0], "primary": [0]},
        {"id": "c2", "name": "Gradient Descent", "summary": "First-order optimization", "chunks": [1], "primary": [1]}
    ]
    edges = [["c1", "c2"]]
    chunks = [{"page": 1, "text": "Linear algebra is the study of vectors and matrices."},
              {"page": 2, "text": "Gradient descent minimizes loss via parameter updates."}]
    class FakeIndex:
        def __init__(self, ch):
            self.chunks = ch
    store.save_kb("ml.pdf", 2, FakeIndex(chunks), concepts, edges, user_id=uid)

    # Insert explainable mastery decision, quiz evidence, and active recall records
    with store._materials_transaction(immediate=True) as conn:
        conn.execute(
            """
            INSERT INTO explainable_mastery_decisions
            (decision_id, user_id, concept_id, outcome, legacy_status, recommended_next_action, updated_at, evidence_version)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            ("dec_1", uid, "c1", "MASTERED", "STRONG", "Foundations verified", "2026-11-12T12:00:00Z", 1)
        )
        conn.execute(
            """
            INSERT INTO explainable_mastery_decisions
            (decision_id, user_id, concept_id, outcome, legacy_status, recommended_next_action, updated_at, evidence_version)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            ("dec_2", uid, "c2", "NOT_YET_MASTERED", "WEAK", "Review step size updates", "2026-11-12T12:30:00Z", 1)
        )
        conn.execute(
            """
            INSERT INTO learning_evidence_store
            (evidence_id, user_id, concept_id, question_id, question_form, student_answer, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            ("ev_1", uid, "c2", "q1", "quiz", "step size", "2026-11-12T12:35:00Z")
        )
        conn.execute(
            """
            INSERT INTO active_recall_sessions
            (session_id, user_id, concept_id, retrieval_score, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            ("ar_1", uid, "c1", 0.95, "2026-11-12T12:40:00Z")
        )
        conn.execute(
            """
            INSERT INTO smart_notes
            (note_id, user_id, concept_id, material_id, task_type, title, markdown_content, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            ("note_1", uid, "c1", "mat_test_01", "study_guide", "LinAlg Guide", "# Linear Algebra", "2026-11-12T12:45:00Z")
        )

    # Call get_dashboard_summary
    start_t = time.perf_counter()
    summary = store.get_dashboard_summary(user_id=uid, material_id="mat_test_01")
    elapsed_ms = (time.perf_counter() - start_t) * 1000.0

    assert summary["ok"] is True
    assert summary["material"] is not None
    assert summary["material"]["material_id"] == "mat_test_01"
    assert summary["material"]["title"] == "Machine Learning Foundations"
    assert len(summary["materials"]) == 1
    assert len(summary["concepts"]) == 2
    assert summary["edges"] == [["c1", "c2"]]
    assert len(summary["notes"]) == 1
    assert summary["notes"][0]["title"] == "LinAlg Guide"

    # Check hydrated concept details
    by_id = {c["id"]: c for c in summary["concepts"]}
    assert by_id["c1"]["status"] == "strong"
    assert by_id["c1"]["score"] >= 0.75
    assert by_id["c1"]["recall_attempts"] == 1
    assert by_id["c1"]["best_recall_score"] == 0.95

    assert by_id["c2"]["status"] == "weak"
    assert by_id["c2"]["score"] <= 0.45
    assert by_id["c2"]["quiz_attempts"] == 1
    assert by_id["c2"]["why"] == "Review step size updates"

    assert summary["stats"]["strong_count"] == 1
    assert summary["stats"]["weak_count"] == 1
    assert summary["stats"]["total_notes"] == 1


def test_deterministic_recommend_heuristics_under_10ms():
    """Verify recommend() runs in <10ms without making any LLM calls on page load."""
    uid = "test_dash_user"

    concepts = [
        {"id": "c1", "name": "Linear Algebra", "summary": "Vectors and matrices", "chunks": [0], "primary": [0]},
        {"id": "c2", "name": "Gradient Descent", "summary": "First-order optimization", "chunks": [1], "primary": [1]},
        {"id": "c3", "name": "Backpropagation", "summary": "Chain rule gradients", "chunks": [0], "primary": [0]}
    ]
    edges = [["c1", "c2"], ["c2", "c3"]]
    chunks = [
        {"page": 1, "text": "Linear algebra is the study of vectors and matrices. It forms the basis of all ML."},
        {"page": 2, "text": "Gradient descent computes parameter updates iteratively. Learning rates control step size."}
    ]
    class FakeIndex:
        def __init__(self, ch):
            self.chunks = ch
    store.save_kb("ml.pdf", 2, FakeIndex(chunks), concepts, edges, user_id=uid)

    # Set up decisions: c1 is WEAK, c2 is UNKNOWN, c3 is UNKNOWN
    with store._materials_transaction(immediate=True) as conn:
        conn.execute(
            """
            INSERT INTO explainable_mastery_decisions
            (decision_id, user_id, concept_id, outcome, legacy_status, recommended_next_action, updated_at, evidence_version)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            ("dec_c1", uid, "c1", "NOT_YET_MASTERED", "WEAK", "Address vector matrix multiplication gaps", "2026-11-12T10:00:00Z", 1)
        )

    # Measure execution time of deterministic recommend
    start_t = time.perf_counter()
    rec = recommend.recommend(user_id=uid, refresh_ai=False)
    elapsed_ms = (time.perf_counter() - start_t) * 1000.0

    assert elapsed_ms < 10.0, f"Deterministic recommend took {elapsed_ms:.2f}ms, expected < 10ms"
    assert rec["ready"] is True
    assert len(rec["actions"]) > 0

    # Top recommendation should target prerequisite blocker or lowest score
    top_action = rec["actions"][0]
    assert top_action["conceptId"] in ("c1", "c2")
    assert "Linear Algebra" in top_action["name"] or "Gradient Descent" in top_action["name"]
    assert top_action["lesson"] != ""
    assert "decayed_score" in top_action


def test_api_dashboard_summary_route():
    """Verify GET /api/dashboard-summary route returns full payload in single round trip."""
    uid = "test_dash_user"
    client = TestClient(app)

    # Setup basic KB
    concepts = [{"id": "c1", "name": "Probability", "summary": "Random variables", "chunks": [0], "primary": [0]}]
    edges = []
    chunks = [{"page": 1, "text": "Probability theory provides formal tools for uncertain reasoning."}]
    class FakeIndex:
        def __init__(self, ch):
            self.chunks = ch
    store.save_kb("prob.pdf", 1, FakeIndex(chunks), concepts, edges, user_id=uid)

    res = client.get(f"/api/dashboard-summary?user_id={uid}")
    assert res.status_code == 200
    data = res.json()
    assert data["ok"] is True
    assert data["user_id"] == uid
    assert "concepts" in data
    assert "materials" in data
    assert "notes" in data
    assert "loop_status" in data
    assert "recommendations" in data
    assert "stats" in data
    assert data["loop_status"]["ok"] is True


def test_api_recommend_backward_compatibility():
    """Verify both POST and GET /recommend work and accept refresh_ai query param."""
    uid = "test_dash_user"
    client = TestClient(app)

    # Setup basic KB
    chunks = [{"page": 1, "text": "Calculus derivatives."}]
    concepts = [{"id": "c1", "name": "Calculus", "summary": "Derivatives", "chunks": [0], "primary": [0]}]
    class FakeIndex:
        def __init__(self, ch):
            self.chunks = ch
    store.save_kb("calc.pdf", 1, FakeIndex(chunks), concepts, [], user_id=uid)

    res_post = client.post(f"/api/recommend?user_id={uid}&refresh_ai=false")
    assert res_post.status_code == 200
    assert res_post.json()["ready"] is True

    res_get = client.get(f"/api/recommend?user_id={uid}&refresh_ai=false")
    assert res_get.status_code == 200
    assert res_get.json()["ready"] is True
