"""SheerID Verification Gateway & Webhook Authenticity Engine.

Integrates genuine SheerID student verification for clinical/medical content access.
Enforces:
1. Medical student eligibility program validation (not generic student).
2. Authoritative SheerID API polling & session initiation.
3. Webhook authenticity verification with HMAC/token validation.
4. Idempotent event processing and anti-replay protection.
5. Mock transport support for CI and offline test execution.
"""
from __future__ import annotations

import hmac
import hashlib
import json
import logging
from datetime import datetime, timezone
from typing import Any, Optional
import httpx

from .. import config
from . import store

logger = logging.getLogger("concept_mastery.sheerid")

# Supported SheerID terminal and intermediate verification steps
SHEERID_STATUS_MAPPING = {
    "success": "approved",
    "approved": "approved",
    "verified": "approved",
    "pending": "pending",
    "docupload": "pending",
    "collectstudentpersonalinfo": "pending",
    "rejected": "rejected",
    "fraud": "rejected",
    "expired": "expired",
}


class SheerIDError(Exception):
    """Base exception for SheerID verification failures."""
    pass


class SheerIDClient:
    """Official SheerID REST API v2 Client with Mock Provider fallback."""

    def __init__(
        self,
        api_key: str | None = None,
        program_id: str | None = None,
        base_url: str | None = None,
        webhook_secret: str | None = None,
    ):
        self.api_key = api_key or config.SHEERID_API_KEY
        self.program_id = program_id or config.SHEERID_PROGRAM_ID
        self.base_url = (base_url or config.SHEERID_BASE_URL).rstrip("/")
        self.webhook_secret = webhook_secret or config.SHEERID_WEBHOOK_SECRET

    @property
    def is_configured(self) -> bool:
        """Return True if live SheerID credentials are provided."""
        return bool(self.api_key and self.program_id)

    def get_hosted_url(self, user_id: str, return_url: str | None = None) -> str:
        """Generate hosted SheerID verification URL for redirect or iframe modal."""
        program = self.program_id or "medical-student-default"
        url = f"https://services.sheerid.com/verify/{program}/?externalUserId={user_id}"
        if return_url:
            url += f"&returnUrl={return_url}"
        return url

    async def initiate_verification(
        self,
        user_id: str,
        first_name: str,
        last_name: str,
        email: str,
        birth_date: str | None = None,
        organization_name: str | None = None,
    ) -> dict[str, Any]:
        """Initiate SheerID verification for medical student eligibility."""
        program_id = self.program_id or "medical-student-program"
        now_iso = datetime.now(timezone.utc).isoformat()

        # Offline/Mock behavior if credentials are not configured or in test mode
        if not self.is_configured:
            mock_verification_id = f"mock_verif_{user_id}_{int(datetime.now().timestamp())}"
            record = store.record_sheerid_verification(
                user_id=user_id,
                verification_id=mock_verification_id,
                program_id=program_id,
                status="pending",
                institution=organization_name or "Medical University",
                program="Doctor of Medicine (MD)",
                educational_email=email,
                metadata={
                    "mode": "mock",
                    "tracking_id": user_id,
                    "initiated_at": now_iso,
                },
            )
            return {
                "verification_id": mock_verification_id,
                "status": "pending",
                "redirect_url": self.get_hosted_url(user_id),
                "message": "SheerID verification session initiated (Mock provider). Complete verification via hosted flow.",
                "record": record,
            }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "programId": program_id,
            "trackingId": user_id,
        }

        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.post(
                    f"{self.base_url}/verification",
                    json=payload,
                    headers=headers,
                )
                if resp.status_code not in (200, 201):
                    logger.error("SheerID initiation failed: %s %s", resp.status_code, resp.text)
                    raise SheerIDError(f"SheerID verification initiation failed: {resp.text}")

                data = resp.json()
                verification_id = data.get("verificationId")
                current_step = data.get("currentStep", "pending").lower()
                status = SHEERID_STATUS_MAPPING.get(current_step, "pending")

                store.record_sheerid_verification(
                    user_id=user_id,
                    verification_id=verification_id,
                    program_id=program_id,
                    status=status,
                    institution=organization_name or "",
                    program="Medical Student",
                    educational_email=email,
                    metadata=data,
                )

                return {
                    "verification_id": verification_id,
                    "status": status,
                    "redirect_url": data.get("redirectUrl") or self.get_hosted_url(user_id),
                    "current_step": current_step,
                }
        except httpx.HTTPError as exc:
            logger.exception("HTTP error connecting to SheerID: %s", exc)
            raise SheerIDError(f"Failed to communicate with SheerID: {exc}") from exc

    async def get_verification_details(self, verification_id: str) -> dict[str, Any]:
        """Fetch authoritative verification status directly from SheerID API."""
        if not self.is_configured:
            # Query stored record
            rec = store.get_verification_by_sheerid_id(verification_id)
            if rec:
                return {
                    "verificationId": verification_id,
                    "currentStep": rec.get("status", "pending"),
                    "programId": rec.get("program_id", self.program_id),
                    "trackingId": rec.get("user_id"),
                }
            return {
                "verificationId": verification_id,
                "currentStep": "pending",
                "programId": self.program_id,
            }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
        }
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(
                f"{self.base_url}/verification/{verification_id}",
                headers=headers,
            )
            if resp.status_code != 200:
                raise SheerIDError(f"SheerID lookup failed ({resp.status_code}): {resp.text}")
            return resp.json()

    def verify_webhook_authenticity(
        self,
        raw_body: bytes,
        signature_header: str | None = None,
        token_header: str | None = None,
    ) -> bool:
        """Validate webhook authenticity using SheerID secret token or HMAC-SHA256 signature."""
        secret = self.webhook_secret or config.SHEERID_WEBHOOK_SECRET
        # If no secret configured in development, check token if provided
        if not secret:
            # If secret is unset, reject in production; accept in development if offline
            if config.ENVIRONMENT == "production":
                return False
            return True

        # Check static token comparison
        if token_header and hmac.compare_digest(token_header.strip(), secret.strip()):
            return True

        # Check HMAC-SHA256 signature header if provided
        if signature_header:
            clean_sig = signature_header.strip()
            if clean_sig.startswith("sha256="):
                clean_sig = clean_sig.split("sha256=", 1)[1]
            computed = hmac.new(secret.encode("utf-8"), raw_body, hashlib.sha256).hexdigest()
            return hmac.compare_digest(computed.lower(), clean_sig.lower())

        return False

    async def process_webhook_event(
        self,
        raw_payload: bytes,
        signature_header: str | None = None,
        token_header: str | None = None,
    ) -> dict[str, Any]:
        """Validate, check replay, and safely update user medical verification state."""
        # 1. Authenticity check
        if not self.verify_webhook_authenticity(raw_payload, signature_header, token_header):
            logger.warning("SheerID webhook authenticity check failed.")
            raise SheerIDError("Invalid webhook signature or token.")

        # 2. Parse payload
        try:
            event = json.loads(raw_payload.decode("utf-8"))
        except Exception as exc:
            raise SheerIDError(f"Malformed webhook JSON: {exc}") from exc

        verification_id = event.get("verificationId")
        if not verification_id:
            raise SheerIDError("Missing verificationId in webhook payload.")

        event_id = event.get("eventId") or f"evt_{verification_id}_{event.get('currentStep')}"
        event_type = event.get("eventType") or event.get("currentStep") or "status_update"

        # 3. Replay Protection / Idempotency Check
        is_new = store.record_sheerid_webhook_event(
            event_id=event_id,
            verification_id=verification_id,
            event_type=str(event_type),
            payload_json=json.dumps(event),
        )
        if not is_new:
            logger.info("Ignoring duplicate or replayed SheerID webhook event %s", event_id)
            return {"status": "ignored", "reason": "duplicate_event", "event_id": event_id}

        # 4. Fetch authoritative state from provider if configured, or use payload step
        current_step = str(event.get("currentStep", "")).lower()
        if self.is_configured:
            try:
                details = await self.get_verification_details(verification_id)
                current_step = str(details.get("currentStep", current_step)).lower()
            except Exception as e:
                logger.warning("Could not query authoritative SheerID details for %s: %s", verification_id, e)

        # 5. Program Eligibility Validation
        # Verify the program matches medical student eligibility
        event_program_id = event.get("programId") or ""
        expected_prog = self.program_id or config.SHEERID_PROGRAM_ID
        if expected_prog and event_program_id and event_program_id != expected_prog:
            logger.warning("SheerID webhook program mismatch: %s != %s", event_program_id, expected_prog)
            status = "rejected"
            review_notes = f"Rejected: Ineligible program ID '{event_program_id}'. Medical student program required."
        else:
            status = SHEERID_STATUS_MAPPING.get(current_step, "pending")
            review_notes = f"SheerID verification step: {current_step}"

        # 6. Authoritative state transition in DB
        updated = store.update_sheerid_verification_status(
            verification_id=verification_id,
            status=status,
            metadata=event,
            review_notes=review_notes,
        )

        return {
            "status": "processed",
            "verification_id": verification_id,
            "verification_status": status,
            "user_id": updated.get("user_id") if updated else None,
        }


# Global Client Singleton
DEFAULT_SHEERID_CLIENT = SheerIDClient()
