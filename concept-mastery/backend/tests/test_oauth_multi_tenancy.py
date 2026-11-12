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


def test_auth_token_and_dev_header_hardening(monkeypatch):
    """Verify JWT subject validation, malformed header rejection, and production dev-header blocking."""
    import uuid
    from app import config

    uid = f"jwt_user_{uuid.uuid4().hex[:8]}"
    valid_token = auth.create_access_token({"sub": uid, "email": f"{uid}@student.local"})

    # 1. Case-insensitive Bearer scheme works
    res_ok = client.get("/api/auth/me", headers={"Authorization": f"bearer {valid_token}"})
    assert res_ok.status_code == 200
    assert res_ok.json()["user"]["user_id"] == uid

    # 2. JWT missing 'sub' or blank 'sub' must fail with 401
    no_sub_tok = auth.create_access_token({"email": "nosub@student.local"})
    blank_sub_tok = auth.create_access_token({"sub": "   ", "email": "blank@student.local"})
    for bad_tok in (no_sub_tok, blank_sub_tok):
        r1 = client.get("/api/auth/me", headers={"Authorization": f"Bearer {bad_tok}"})
        r2 = client.get("/api/materials", headers={"Authorization": f"Bearer {bad_tok}"})
        assert r1.status_code == 401
        assert r2.status_code == 401

    # 3. Malformed / unsupported Authorization header scheme rejected with 401
    for bad_hdr in ("Basic dXNlcjpwYXNz", "Bearer   ", "Token xyz"):
        r = client.get("/api/auth/me", headers={"Authorization": bad_hdr})
        assert r.status_code == 401

    # 4. X-Dev-User-Id must be rejected with 401 in production when PYTEST_CURRENT_TEST is unset
    monkeypatch.setattr(config, "ENVIRONMENT", "production")
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    res_prod_dev = client.get("/api/auth/me", headers={"X-Dev-User-Id": "attacker_uid"})
    assert res_prod_dev.status_code == 401
    res_prod_anon = client.get("/api/materials")
    assert res_prod_anon.status_code == 401


def test_cross_tenant_user_id_spoofing_rejected_with_403(tmp_path):
    """Authenticated tenant Bob cannot read or mutate Alice's tenant by passing user_id=alice in path/query/body."""
    import uuid
    from app.core.concept_index import GLOBAL_INDEX

    alice_id = f"alice_neg_{uuid.uuid4().hex[:8]}"
    bob_id = f"bob_neg_{uuid.uuid4().hex[:8]}"
    alice_headers = {"Authorization": f"Bearer {auth.create_access_token({'sub': alice_id})}"}
    bob_headers = {"Authorization": f"Bearer {auth.create_access_token({'sub': bob_id})}"}

    md_file = tmp_path / "alice_secret.md"
    md_file.write_text(
        "# Zero Knowledge Snark Protocol\n\n"
        "Zero Knowledge Snark Protocol proves computation integrity without revealing private witness inputs.\n",
        encoding="utf-8",
    )

    try:
        with open(md_file, "rb") as f:
            up_res = client.post(
                "/api/materials/ingest-file",
                headers=alice_headers,
                files={"file": ("alice_secret.md", f, "text/markdown")},
                data={"user_id": alice_id, "title": "Alice Secret ZK Guide"},
            )
        assert up_res.status_code == 200
        alice_mat_id = up_res.json()["materials"][0]["material_id"]

        spoof_requests = [
            ("GET", f"/api/concepts/graph?user_id={alice_id}", {}),
            ("GET", f"/api/graph?user_id={alice_id}", {}),
            ("GET", f"/api/ingest/status?user_id={alice_id}", {}),
            ("GET", f"/api/dashboard/{alice_id}", {}),
            ("GET", f"/api/mastery?user_id={alice_id}", {}),
            ("GET", f"/api/mastery/dashboard?user_id={alice_id}", {}),
            ("GET", f"/api/mastery/zero-knowledge-snark-protocol/deliberation?user_id={alice_id}", {}),
            ("POST", "/api/mastery/evaluate-evidence", {"json": {"user_id": alice_id, "concept_id": "c1", "student_answer": "test"}}),
            ("POST", f"/api/mastery/reset?user_id={alice_id}", {}),
            ("GET", f"/api/loop/status?user_id={alice_id}", {}),
            ("POST", "/api/loop/start", {"json": {"user_id": alice_id}}),
            ("POST", f"/api/loop/next-action?user_id={alice_id}", {}),
            ("POST", f"/api/recommend?user_id={alice_id}", {}),
            ("GET", f"/api/diagnostic/baseline?user_id={alice_id}", {}),
            ("POST", "/api/diagnostic/start", {"json": {"user_id": alice_id}}),
            ("POST", "/api/diagnostic/submit", {"json": {"user_id": alice_id, "answers": []}}),
            ("GET", f"/api/materials?user_id={alice_id}", {}),
            ("POST", "/api/materials/ingest-url", {"json": {"user_id": alice_id, "url": "https://example.com"}}),
            ("GET", f"/api/notes?user_id={alice_id}", {}),
            ("POST", "/api/notes/generate", {"json": {"user_id": alice_id, "concept_id": "zero-knowledge-snark-protocol", "task_type": "exam"}}),
            ("GET", f"/api/active-recall/history?user_id={alice_id}", {}),
            ("POST", "/api/active-recall/start", {"json": {"user_id": alice_id, "concept_id": "zero-knowledge-snark-protocol"}}),
            ("POST", "/api/tutor/chat", {"json": {"user_id": alice_id, "message": "hello"}}),
            ("POST", "/api/quiz/next", {"json": {"user_id": alice_id}}),
            ("POST", "/api/quiz/remedial", {"json": {"user_id": alice_id}}),
            ("POST", "/api/quiz/submit", {"json": {"user_id": alice_id, "question_id": "q1", "concept_id": "c1"}}),
            ("POST", "/api/tools/generate", {"json": {"user_id": alice_id, "concept_id": "c1", "tool_type": "visual"}}),
        ]

        for method, path, kwargs in spoof_requests:
            res = client.request(method, path, headers=bob_headers, **kwargs)
            assert res.status_code == 403, f"Expected 403 for spoofed {method} {path}, got {res.status_code}: {res.text}"

        # Also test Form field spoofing on /api/materials/ingest-file
        with open(md_file, "rb") as f:
            spoof_up = client.post(
                "/api/materials/ingest-file",
                headers=bob_headers,
                files={"file": ("alice_secret.md", f, "text/markdown")},
                data={"user_id": alice_id},
            )
        assert spoof_up.status_code == 403

        # Cleanup verifies delete only works as Alice
        assert store.get_study_material(alice_mat_id, user_id=alice_id) is not None
    finally:
        store.clear_all_study_materials(alice_id)
        store.clear_kb(alice_id)
        GLOBAL_INDEX.clear(user_id=alice_id)


def test_cross_tenant_resource_id_isolation_materials_notes_sessions_quiz(tmp_path):
    """Verify HTTP 403/404 and store isolation when Tenant B probes Tenant A's resource IDs."""
    import uuid
    from app.core.concept_index import GLOBAL_INDEX

    alice_id = f"alice_res_{uuid.uuid4().hex[:8]}"
    bob_id = f"bob_res_{uuid.uuid4().hex[:8]}"
    alice_headers = {"Authorization": f"Bearer {auth.create_access_token({'sub': alice_id})}"}
    bob_headers = {"Authorization": f"Bearer {auth.create_access_token({'sub': bob_id})}"}

    pdf_file = tmp_path / "alice_deck.md"
    pdf_file.write_text(
        "# Merkle Commitments\n\n"
        "Merkle Commitments bind a vector of values into a single succinct cryptographic root hash. "
        "Each internal node hashes its left and right children recursively to enable logarithmic membership proofs.\n",
        encoding="utf-8",
    )

    alice_note_id = f"sn_alice_{uuid.uuid4().hex[:8]}"
    alice_sess_id = f"ar_alice_{uuid.uuid4().hex[:8]}"

    try:
        # 1. Alice ingests a material and generates a quiz question
        with open(pdf_file, "rb") as f:
            up_res = client.post(
                "/api/materials/ingest-file",
                headers=alice_headers,
                files={"file": ("alice_deck.md", f, "text/markdown")},
                data={"title": "Merkle Commitments"},
            )
        assert up_res.status_code == 200
        alice_mat_id = up_res.json()["materials"][0]["material_id"]

        q_res = client.post("/api/quiz/next", headers=alice_headers, json={})
        assert q_res.status_code == 200
        alice_qid = q_res.json()["id"]

        # 2. Seed Alice's active recall session and smart note
        store.save_active_recall_session({
            "session_id": alice_sess_id,
            "user_id": alice_id,
            "concept_id": "merkle-commitments",
            "lesson_json": {"core_intuition": "Hash trees bind values.", "essential_keywords": ["merkle", "hash"]},
            "verification_question_json": {"question_text": "What does the root commit to?", "options": ["All leaves"], "correct_option_idx": 0},
            "recall_submission_text": "",
            "retrieval_score": 0.0,
            "retained_concepts_json": [],
            "missed_gaps_json": [],
            "created_at": "2026-11-11T00:00:00Z",
        })
        store.save_smart_note({
            "note_id": alice_note_id,
            "user_id": alice_id,
            "concept_id": "merkle-commitments",
            "material_id": alice_mat_id,
            "task_type": "exam",
            "custom_instructions": "",
            "title": "Alice Private Merkle Notes",
            "markdown_content": "# Private Content\nSecret theorem notes.",
            "key_takeaways": ["Root hash commits leaves"],
            "citations": ["Page 1"],
            "created_at": "2026-11-11T00:00:00Z",
        })

        # 3. Direct store-layer isolation assertions (no unscoped fallback)
        assert store.get_smart_note(alice_note_id, user_id=bob_id) is None
        assert store.delete_smart_note(alice_note_id, user_id=bob_id) is False
        assert store.update_smart_note_urls(alice_note_id, user_id=bob_id, pdf_url="/evil.pdf") is False
        assert store.get_active_recall_session_for_user(alice_sess_id, user_id=bob_id) is None
        assert store.update_active_recall_session_for_user(alice_sess_id, user_id=bob_id, updates={"recall_submission_text": "tampered"}) is None

        # 4. Cross-tenant material ID attacks -> 403/404, never mutating or leaking Alice's data
        assert client.get(f"/api/materials/{alice_mat_id}/preview", headers=bob_headers).status_code == 403
        assert client.delete(f"/api/materials/{alice_mat_id}", headers=bob_headers).status_code == 403
        assert client.post(f"/api/materials/{alice_mat_id}/deep-audio", headers=bob_headers, json={}).status_code == 403
        assert store.get_study_material(alice_mat_id, user_id=alice_id) is not None
        assert client.get("/api/materials/mat_nonexistent_404/preview", headers=bob_headers).status_code == 404

        # 5. Cross-tenant active recall session ID attack -> 403/404
        ar_eval = client.post(
            "/api/active-recall/evaluate",
            headers=bob_headers,
            json={"session_id": alice_sess_id, "concept_id": "merkle-commitments", "recall_text": "tampered"},
        )
        assert ar_eval.status_code == 403
        alice_sess = store.get_active_recall_session_for_user(alice_sess_id, user_id=alice_id)
        assert alice_sess is not None and alice_sess["recall_submission_text"] == ""
        assert client.post(
            "/api/active-recall/evaluate",
            headers=bob_headers,
            json={"session_id": "ar_missing_404", "concept_id": "merkle-commitments", "recall_text": "test"},
        ).status_code == 404

        # 6. Cross-tenant smart note ID attacks -> 403/404
        assert client.post(f"/api/notes/{alice_note_id}/audio", headers=bob_headers).status_code == 403
        assert client.post(f"/api/notes/{alice_note_id}/export", headers=bob_headers, json={"export_format": "pdf"}).status_code == 403
        assert client.delete(f"/api/notes/{alice_note_id}", headers=bob_headers).status_code == 403
        assert store.get_smart_note(alice_note_id, user_id=alice_id) is not None
        assert client.delete("/api/notes/sn_missing_404", headers=bob_headers).status_code == 404

        # 7. Cross-tenant quiz question ID submission -> 403
        quiz_res = client.post(
            "/api/quiz/submit",
            headers=bob_headers,
            json={"question_id": alice_qid, "concept_id": "merkle-commitments", "selected_option_idx": 0},
        )
        assert quiz_res.status_code == 403

        # 8. Even if global 'default' KB were populated, Bob (authenticated with 0 uploads) sees empty graph/mastery
        bob_graph = client.get("/api/concepts/graph", headers=bob_headers).json()
        assert bob_graph["nodes"] == []
        assert client.get("/api/mastery", headers=bob_headers).json()["concepts"] == []
        assert client.get("/api/mastery/dashboard", headers=bob_headers).json()["total_concepts"] == 0
    finally:
        store.delete_smart_note(alice_note_id, user_id=alice_id)
        store.clear_all_study_materials(alice_id)
        store.clear_kb(alice_id)
        store.clear_kb(bob_id)
        GLOBAL_INDEX.clear(user_id=alice_id)
        GLOBAL_INDEX.clear(user_id=bob_id)

