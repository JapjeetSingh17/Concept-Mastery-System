"""Regression and concurrency tests for SQLite connection factory and transactions in store.py."""

import inspect
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from app.core import store


def test_public_store_signatures_preserved():
    """Verify all expected public store helpers exist with unchanged parameter signatures."""
    expected_params = {
        "get_materials_db": [],
        "init_materials_db": [],
        "upsert_user": ["user_id", "email", "full_name", "picture_url"],
        "get_user": ["user_id"],
        "get_user_by_email": ["email"],
        "record_login_log": ["user_id", "ip_address", "user_agent"],
        "get_user_login_logs": ["user_id", "limit"],
        "save_study_material": ["mat"],
        "list_study_materials": ["user_id"],
        "get_study_material": ["material_id", "user_id"],
        "delete_study_material": ["material_id", "user_id"],
        "clear_all_study_materials": ["user_id"],
        "save_active_recall_session": ["session"],
        "update_active_recall_session": ["session_id", "updates"],
        "get_active_recall_session": ["session_id"],
        "list_active_recall_sessions": ["user_id", "limit"],
        "save_smart_note": ["note_data"],
        "get_smart_note": ["note_id", "user_id"],
        "list_smart_notes": ["user_id", "concept_id"],
        "delete_smart_note": ["note_id", "user_id"],
        "update_smart_note_urls": ["note_id", "user_id", "audio_url", "pdf_url", "pptx_url"],
        "save_learning_evidence": ["rec"],
        "get_learning_evidence": ["evidence_id", "user_id"],
        "list_learning_evidence": ["user_id", "concept_id", "limit"],
        "save_specialist_finding": ["finding", "decision_id", "user_id"],
        "list_specialist_findings": ["decision_id", "user_id"],
        "save_mastery_decision": ["decision", "user_id"],
        "get_latest_mastery_decision": ["user_id", "concept_id"],
        "list_latest_mastery_decisions": ["user_id"],
        "verify_evidence_refs": ["user_id", "concept_id", "claimed_refs"],
        "save_kb": ["file", "pages", "index", "concepts", "edges", "user_id"],
        "get_kb": ["user_id"],
        "clear_kb": ["user_id"],
        "load_kb": [],
        "require": ["user_id"],
        "load_json": ["path", "default"],
        "save_json": ["path", "data"],
    }
    for fn_name, params in expected_params.items():
        fn = getattr(store, fn_name)
        sig = list(inspect.signature(fn).parameters.keys())
        assert sig == params, f"Signature mismatch for store.{fn_name}: {sig} != {params}"


def test_sqlite_pragmas_configured_on_every_connection(monkeypatch, tmp_path):
    """Verify centralized factory configures WAL, synchronous=NORMAL, foreign_keys=ON, busy_timeout=10000."""
    test_db = tmp_path / "isolated" / "materials.db"
    monkeypatch.setattr(store, "MATERIALS_DB_PATH", test_db)

    with store.get_materials_db() as conn:
        journal_mode = conn.execute("PRAGMA journal_mode;").fetchone()[0]
        synchronous = conn.execute("PRAGMA synchronous;").fetchone()[0]
        foreign_keys = conn.execute("PRAGMA foreign_keys;").fetchone()[0]
        busy_timeout = conn.execute("PRAGMA busy_timeout;").fetchone()[0]

    assert str(journal_mode).lower() == "wal"
    assert synchronous == 1  # NORMAL
    assert foreign_keys == 1  # ON
    assert busy_timeout == 10000


def test_transaction_rollback_and_fk_enforcement(monkeypatch, tmp_path):
    """Verify exceptions trigger clean rollback and PRAGMA foreign_keys=ON is enforced."""
    test_db = tmp_path / "rollback.db"
    monkeypatch.setattr(store, "MATERIALS_DB_PATH", test_db)
    store.init_materials_db()

    with pytest.raises(RuntimeError, match="mid-tx failure"):
        with store.get_materials_db() as conn:
            conn.execute(
                """
                INSERT INTO users (user_id, email, full_name, picture_url, created_at, last_login_at)
                VALUES ('u_rollback', 'rb@example.com', 'RB', '', '2026-01-01', '2026-01-01')
                """
            )
            raise RuntimeError("mid-tx failure")

    assert store.get_user("u_rollback") is None

    # PRAGMA foreign_keys=ON rejects login log for unknown user_id
    with pytest.raises(sqlite3.IntegrityError):
        store.record_login_log(user_id="nonexistent_user", ip_address="127.0.0.1")

    # Repeated upsert_user must preserve existing user_login_logs (no CASCADE delete)
    store.upsert_user("u_keep", "keep@example.com", full_name="First")
    store.record_login_log("u_keep", ip_address="10.0.0.1")
    store.upsert_user("u_keep", "keep@example.com", full_name="Second")
    user = store.get_user("u_keep")
    logs = store.get_user_login_logs("u_keep")
    assert user is not None and user["full_name"] == "Second"
    assert len(logs) == 1


def test_concurrent_multi_agent_writes_and_busy_timeout(monkeypatch, tmp_path):
    """Stress-test concurrent reads/writes and external write-lock contention without lock errors."""
    test_db = tmp_path / "concurrent.db"
    monkeypatch.setattr(store, "MATERIALS_DB_PATH", test_db)
    store.init_materials_db()

    # Create base active recall session for concurrent updates
    store.save_active_recall_session(
        {"session_id": "ar_shared", "user_id": "u_conc", "concept_id": "heaps"}
    )

    # Hold a brief external write lock (~250ms) to prove busy_timeout=10000 waits instead of failing
    lock_acquired = threading.Event()
    ext_errors = []

    def external_writer():
        try:
            ext_conn = sqlite3.connect(str(test_db), timeout=10.0)
            ext_conn.execute("PRAGMA busy_timeout=10000;")
            ext_conn.execute("BEGIN IMMEDIATE")
            lock_acquired.set()
            time.sleep(0.25)
            ext_conn.execute(
                """
                INSERT INTO learning_evidence_store
                (evidence_id, user_id, concept_id, student_answer, created_at)
                VALUES ('evi_external', 'u_conc', 'heaps', 'ext', '2026-01-01T00:00:00Z')
                """
            )
            ext_conn.commit()
            ext_conn.close()
        except Exception as exc:  # pragma: no cover
            ext_errors.append(exc)

    ext_thread = threading.Thread(target=external_writer)
    ext_thread.start()
    assert lock_acquired.wait(timeout=2.0)

    store.save_learning_evidence(
        {
            "evidence_id": "evi_waited",
            "user_id": "u_conc",
            "concept_id": "heaps",
            "student_answer": "waited cleanly",
        }
    )
    ext_thread.join(timeout=5.0)
    assert ext_errors == []
    assert store.get_learning_evidence("evi_external", user_id="u_conc") is not None
    assert store.get_learning_evidence("evi_waited", user_id="u_conc") is not None

    # Concurrent multi-agent writes & reads across 12 worker threads
    def agent_worker(worker_id: int) -> None:
        uid = f"user_{worker_id % 3}"
        cid = "binary_search"
        for step in range(4):
            ev_id = f"evi_{worker_id}_{step}"
            dec_id = f"dec_{worker_id}_{step}"
            note_id = f"sn_{worker_id}_{step}"
            store.upsert_user(uid, f"{uid}@example.com", full_name=uid)
            store.save_learning_evidence(
                {
                    "evidence_id": ev_id,
                    "user_id": uid,
                    "concept_id": cid,
                    "student_answer": f"answer {worker_id}-{step}",
                }
            )
            verified = store.verify_evidence_refs(uid, cid, [ev_id, "fake_id"])
            assert verified == [ev_id]
            store.save_specialist_finding(
                {
                    "finding_id": f"find_{worker_id}_{step}",
                    "concept_id": cid,
                    "agent_role": "assessment",
                    "evidence_refs": [ev_id],
                    "findings": "solid",
                },
                decision_id=dec_id,
                user_id=uid,
            )
            store.save_mastery_decision(
                {
                    "decision_id": dec_id,
                    "concept_id": cid,
                    "outcome": "MASTERED",
                    "legacy_status": "STRONG",
                    "supporting_evidence_refs": [ev_id],
                },
                user_id=uid,
            )
            store.save_smart_note(
                {
                    "note_id": note_id,
                    "user_id": uid,
                    "concept_id": cid,
                    "title": f"Note {worker_id}-{step}",
                    "markdown_content": "Body",
                }
            )
            assert store.update_smart_note_urls(note_id, user_id=uid, pdf_url=f"/tmp/{note_id}.pdf")
            updated_ar = store.update_active_recall_session(
                "ar_shared",
                {"recall_submission_text": f"worker {worker_id} step {step}", "retrieval_score": 0.9},
            )
            assert updated_ar is not None

    with ThreadPoolExecutor(max_workers=12) as pool:
        futures = [pool.submit(agent_worker, idx) for idx in range(12)]
        for fut in futures:
            fut.result()

    total_ev = sum(len(store.list_learning_evidence(f"user_{i}", "binary_search", limit=100)) for i in range(3))
    assert total_ev == 48

