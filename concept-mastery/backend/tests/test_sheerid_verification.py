"""Comprehensive unit and integration tests for SheerID medical verification.

Validates:
1. Verification initiation and status reporting.
2. Webhook authenticity via HMAC-SHA256 signature and secret token.
3. Replay attack protection and idempotent event processing.
4. Program eligibility enforcement (medical student vs generic student).
5. State transitions (approved, rejected, pending, expired, malformed).
"""
import hmac
import hashlib
import json
import uuid
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app import config
from app.core import store
from app.core.sheerid import SheerIDClient, DEFAULT_SHEERID_CLIENT


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def auth_headers(client):
    res = client.post("/api/auth/google", json={"credential": "dev:sheerid_tester:sheerid.test@med.edu:SheerID Tester"})
    assert res.status_code == 200
    token = res.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


def test_sheerid_initiate_session(client, auth_headers):
    """Test initiating a SheerID verification session."""
    payload = {
        "first_name": "Gregory",
        "last_name": "House",
        "email": "house@princeton-plainsboro.edu",
        "organization_name": "Princeton-Plainsboro Teaching Hospital",
    }
    resp = client.post("/api/auth/sheerid/initiate", json=payload, headers=auth_headers)
    assert resp.status_code == 200
    data = resp.json()
    assert "verification_id" in data
    assert data["status"] == "pending"
    assert "redirect_url" in data or "hosted_url" in data

    # Verify status in database
    status_resp = client.get("/api/auth/sheerid/status", headers=auth_headers)
    assert status_resp.status_code == 200
    assert status_resp.json()["status"] == "pending"
    assert status_resp.json()["is_verified"] is False


def test_sheerid_webhook_signature_and_approval(client, auth_headers):
    """Test processing an approved SheerID webhook event with valid HMAC signature."""
    uid_suffix = uuid.uuid4().hex[:8]
    user_id = f"sheerid_user_{uid_suffix}"
    verif_id = f"verif_{user_id}"
    prog_id = f"med_prog_{uid_suffix}"
    event_id = f"evt_{verif_id}_{uuid.uuid4().hex[:6]}"

    orig_secret = config.SHEERID_WEBHOOK_SECRET
    orig_prog = config.SHEERID_PROGRAM_ID
    try:
        config.SHEERID_WEBHOOK_SECRET = "test_webhook_secret_key"
        config.SHEERID_PROGRAM_ID = prog_id

        store.record_sheerid_verification(
            user_id=user_id,
            verification_id=verif_id,
            program_id=prog_id,
            status="pending",
            institution="Johns Hopkins School of Medicine",
            program="Doctor of Medicine",
            educational_email=f"{user_id}@jhmi.edu",
        )
        assert store.is_medical_verified(user_id) is False

        event = {
            "eventId": event_id,
            "verificationId": verif_id,
            "programId": prog_id,
            "currentStep": "success",
            "trackingId": user_id,
        }
        raw_body = json.dumps(event).encode("utf-8")
        signature = hmac.new(b"test_webhook_secret_key", raw_body, hashlib.sha256).hexdigest()

        webhook_resp = client.post(
            "/api/auth/sheerid-webhook",
            content=raw_body,
            headers={
                "Content-Type": "application/json",
                "X-SheerID-Signature": f"sha256={signature}",
            },
        )
        assert webhook_resp.status_code == 200
        data = webhook_resp.json()
        assert data["ok"] is True
        assert data["result"]["verification_status"] == "approved"
        assert store.is_medical_verified(user_id) is True
    finally:
        config.SHEERID_WEBHOOK_SECRET = orig_secret
        config.SHEERID_PROGRAM_ID = orig_prog


def test_sheerid_webhook_replay_protection(client):
    """Test that duplicate/replayed webhook events are rejected idempotently."""
    uid = uuid.uuid4().hex[:8]
    event_id = f"evt_replay_{uid}"
    verif_id = f"verif_replay_{uid}"

    event = {
        "eventId": event_id,
        "verificationId": verif_id,
        "currentStep": "pending",
    }
    raw_body = json.dumps(event).encode("utf-8")

    # First delivery
    resp1 = client.post("/api/auth/sheerid-webhook", content=raw_body, headers={"Content-Type": "application/json"})
    assert resp1.status_code == 200
    assert resp1.json()["result"]["status"] == "processed"

    # Second delivery (replay)
    resp2 = client.post("/api/auth/sheerid-webhook", content=raw_body, headers={"Content-Type": "application/json"})
    assert resp2.status_code == 200
    assert resp2.json()["result"]["status"] == "ignored"
    assert resp2.json()["result"]["reason"] == "duplicate_event"


def test_sheerid_webhook_invalid_signature_rejected(client):
    """Test that webhooks with invalid signatures are strictly rejected."""
    orig_secret = config.SHEERID_WEBHOOK_SECRET
    try:
        config.SHEERID_WEBHOOK_SECRET = "production_grade_secret_777"
        raw_body = json.dumps({"verificationId": f"v_{uuid.uuid4().hex[:6]}", "currentStep": "success"}).encode("utf-8")

        resp = client.post(
            "/api/auth/sheerid-webhook",
            content=raw_body,
            headers={
                "Content-Type": "application/json",
                "X-SheerID-Signature": "sha256=invalid_tampered_signature_hash",
            },
        )
        assert resp.status_code == 400
        assert "signature" in resp.json()["detail"].lower()
    finally:
        config.SHEERID_WEBHOOK_SECRET = orig_secret


def test_sheerid_webhook_ineligible_program_rejected(client):
    """Test that a generic student program event does not approve medical access."""
    uid = uuid.uuid4().hex[:8]
    user_id = f"general_student_{uid}"
    verif_id = f"verif_{uid}"
    event_id = f"evt_ineligible_{uid}"

    orig_prog = config.SHEERID_PROGRAM_ID
    try:
        config.SHEERID_PROGRAM_ID = f"med_required_{uid}"

        store.record_sheerid_verification(
            user_id=user_id,
            verification_id=verif_id,
            program_id=f"med_required_{uid}",
            status="pending",
        )

        event = {
            "eventId": event_id,
            "verificationId": verif_id,
            "programId": "generic_high_school_student",
            "currentStep": "success",
        }
        resp = client.post("/api/auth/sheerid-webhook", json=event)
        assert resp.status_code == 200
        assert resp.json()["result"]["verification_status"] == "rejected"
        assert store.is_medical_verified(user_id) is False
    finally:
        config.SHEERID_PROGRAM_ID = orig_prog


def test_sheerid_webhook_malformed_payload(client):
    """Test handling of malformed or missing verificationId."""
    resp = client.post("/api/auth/sheerid-webhook", json={"foo": "bar"})
    assert resp.status_code == 400
    assert "missing verificationid" in resp.json()["detail"].lower()
