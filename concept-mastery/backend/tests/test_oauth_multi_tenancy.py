"""Unit & Integration Tests for Google OAuth 2.0 Multi-Tenancy, Session Audit Logging, and Strict Multi-Tenant Isolation."""
import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.core import auth, store
from app.pipelines.ingest import UniversalMaterialIngestor
from app.pipelines import mastery, quiz, tutor

client = TestClient(app)

def test_tc201_clean_boot_unpopulated_state():
    """Verify clean unpopulated boot with no pre-loaded sample materials."""
    # Ensure pristine boot state
    store.clear_kb("default")
    store.KB.clear()
    store.KB["ready"] = False
    
    # Using clean test user
    headers = {"X-Dev-User-Id": "clean_student_001"}
    
    # 1. Materials should be empty
    res_mats = client.get("/api/materials", headers=headers)
    assert res_mats.status_code == 200
    data = res_mats.json()
    assert data["total_materials"] == 0
    assert data["materials"] == []

    # 2. Concept Graph should be empty
    res_graph = client.get("/api/concepts/graph", headers=headers)
    assert res_graph.status_code == 200
    g_data = res_graph.json()
    assert g_data["nodes"] == []
    assert g_data["edges"] == []
    assert g_data["learning_path"] == []

    # 3. Dashboard should be empty
    res_dash = client.get("/api/dashboard/clean_student_001", headers=headers)
    assert res_dash.status_code == 200
    d_data = res_dash.json()
    assert d_data["overall_score"] == 0.0
    assert d_data["total_concepts"] == 0


def test_tc202_google_oauth_login_and_jwt_generation():
    """Verify Google OAuth token endpoint upserts student profile and generates valid JWT."""
    # Using dev simulated token
    dev_token = "dev:google_sub_12345:ada.lovelace@university.edu:Ada Lovelace"
    res = client.post(
        "/api/auth/google",
        json={"credential": dev_token},
        headers={"User-Agent": "PyTestClient/1.0", "X-Forwarded-For": "192.168.1.100"}
    )
    assert res.status_code == 200
    data = res.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"
    assert data["user"]["user_id"] == "google_sub_12345"
    assert data["user"]["email"] == "ada.lovelace@university.edu"
    assert data["user"]["full_name"] == "Ada Lovelace"

    # Verify user in database
    db_user = store.get_user("google_sub_12345")
    assert db_user is not None
    assert db_user["email"] == "ada.lovelace@university.edu"

    # Verify session audit log recorded
    logs = store.get_user_login_logs("google_sub_12345")
    assert len(logs) >= 1
    assert logs[0]["user_id"] == "google_sub_12345"


def test_tc203_auth_me_endpoint_with_bearer_token():
    """Verify /api/auth/me returns authenticated student profile and login history."""
    token = auth.create_access_token({"sub": "student_alan", "email": "alan@turing.ac.uk"})
    res = client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert res.status_code == 200
    data = res.json()
    assert data["user"]["user_id"] == "student_alan"
    assert data["user"]["email"] == "alan@turing.ac.uk"
    assert "recent_logins" in data


def test_tc204_strict_multi_tenant_materials_isolation():
    """Verify that materials uploaded by Student A are invisible to Student B."""
    alice_id = "tenant_student_alice"
    bob_id = "tenant_student_bob"
    
    # Clean previous state if any
    store.clear_all_study_materials(alice_id)
    store.clear_all_study_materials(bob_id)
    store.clear_kb(alice_id)
    store.clear_kb(bob_id)

    # Alice uploads a Markdown study guide
    alice_content = (
        "# Linear Regression\n"
        "Linear regression models the relationship between dependent variable y and features x.\n"
        "## Cost Function\n"
        "Mean Squared Error measures the average squared difference between predictions and actuals."
    )
    mat_alice = UniversalMaterialIngestor.ingest_content(
        content=alice_content,
        source_type="markdown",
        title="Alice Linear Regression Guide",
        origin="alice_notes.md",
        user_id=alice_id
    )
    assert mat_alice["user_id"] == alice_id

    # Verify Alice sees her material
    alice_mats = store.list_study_materials(alice_id)
    assert len(alice_mats) == 1
    assert alice_mats[0]["title"] == "Alice Linear Regression Guide"

    # Verify Bob sees NOTHING
    bob_mats = store.list_study_materials(bob_id)
    assert len(bob_mats) == 0

    # Bob calls /api/materials API endpoint
    res_bob_api = client.get("/api/materials", headers={"X-Dev-User-Id": bob_id})
    assert res_bob_api.status_code == 200
    assert res_bob_api.json()["total_materials"] == 0
    assert res_bob_api.json()["materials"] == []


def test_tc205_strict_multi_tenant_concept_dag_isolation():
    """Verify that Student A's concept graph is completely isolated from Student B's concept graph."""
    alice_id = "tenant_student_alice"
    bob_id = "tenant_student_bob"

    # Alice has Linear Regression material ingested from previous test
    alice_graph = client.get("/api/concepts/graph", headers={"X-Dev-User-Id": alice_id}).json()
    assert len(alice_graph["nodes"]) > 0

    # Bob's concept graph should be completely empty
    bob_graph = client.get("/api/concepts/graph", headers={"X-Dev-User-Id": bob_id}).json()
    assert bob_graph["nodes"] == []
    assert bob_graph["edges"] == []

    # Bob uploads his own distinct material on Deep Learning
    bob_content = (
        "# Convolutional Neural Networks\n"
        "CNNs use convolutional layers to extract spatial feature maps from images.\n"
        "## Max Pooling\n"
        "Max pooling downsamples feature maps to achieve translation invariance."
    )
    UniversalMaterialIngestor.ingest_content(
        content=bob_content,
        source_type="markdown",
        title="Bob CNN Guide",
        origin="bob_cnn.md",
        user_id=bob_id
    )

    # Bob now sees his CNN graph
    bob_updated_graph = client.get("/api/concepts/graph", headers={"X-Dev-User-Id": bob_id}).json()
    assert len(bob_updated_graph["nodes"]) > 0
    bob_concept_names = [n["name"].lower() for n in bob_updated_graph["nodes"]]
    assert any("convolutional" in name or "cnn" in name or "pooling" in name for name in bob_concept_names)

    # Alice's graph should NOT contain Bob's CNN concepts
    alice_current_graph = client.get("/api/concepts/graph", headers={"X-Dev-User-Id": alice_id}).json()
    alice_concept_names = [n["name"].lower() for n in alice_current_graph["nodes"]]
    assert not any("convolutional" in name or "pooling" in name for name in alice_concept_names)


def test_tc206_strict_multi_tenant_mastery_isolation():
    """Verify that quiz responses and mastery scores are strictly isolated between students."""
    alice_id = "tenant_student_alice"
    bob_id = "tenant_student_bob"

    # Alice takes a quiz on her concept
    alice_kb = store.get_kb(alice_id)
    assert len(alice_kb["concepts"]) > 0
    alice_concept_id = alice_kb["concepts"][0]["id"]

    # Record mastery for Alice
    mastery.record_quiz(alice_id, alice_concept_id, is_correct=True)
    mastery.record_quiz(alice_id, alice_concept_id, is_correct=True)

    alice_snap = mastery.snapshot(alice_id)
    alice_map = {s["id"]: s for s in alice_snap}
    assert alice_concept_id in alice_map
    assert alice_map[alice_concept_id]["score"] >= 0.7

    # Bob's snapshot should have 0 known concepts for Alice's concept
    bob_snap = mastery.snapshot(bob_id)
    bob_map = {s["id"]: s for s in bob_snap}
    assert alice_concept_id not in bob_map


def test_tc207_deletion_isolation_between_tenants():
    """Verify that deleting a material for Student A does not affect Student B."""
    alice_id = "tenant_student_alice"
    bob_id = "tenant_student_bob"

    alice_mats = store.list_study_materials(alice_id)
    bob_mats = store.list_study_materials(bob_id)
    assert len(alice_mats) >= 1
    assert len(bob_mats) >= 1

    alice_mat_id = alice_mats[0]["material_id"]

    # Alice deletes her material
    del_res = client.delete(f"/api/materials/{alice_mat_id}", headers={"X-Dev-User-Id": alice_id})
    assert del_res.status_code == 200

    # Alice should have 0 materials
    assert len(store.list_study_materials(alice_id)) == 0

    # Bob should STILL have his material!
    assert len(store.list_study_materials(bob_id)) == len(bob_mats)
