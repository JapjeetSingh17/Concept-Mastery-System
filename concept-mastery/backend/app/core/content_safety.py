"""Conservative content screening and medical-access policy helpers.

This module intentionally separates explicit-content detection from access
authorization. A text-only heuristic is a screening signal, not a definitive
classifier; production deployments should add image/video moderation and
human review for borderline material.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Iterable


class ContentDecision(str, Enum):
    ALLOW = "allow"
    RESTRICTED = "restricted"
    REVIEW = "review"


@dataclass(frozen=True)
class ScreeningResult:
    decision: ContentDecision
    reason: str
    matched_signals: tuple[str, ...] = ()


# Strong phrase signals only. Do not flag isolated clinical terms such as
# "breast", "penis", "vagina", "reproduction", "anatomy", or "sexual health".
_EXPLICIT_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("pornographic phrase", re.compile(r"\b(?:hardcore|amateur|homemade)\s+porn(?:ography)?\b", re.I)),
    ("explicit-content phrase", re.compile(r"\b(?:explicit sexual content|graphic sexual content|sexual intercourse video)\b", re.I)),
    ("sexual-act phrase", re.compile(r"\b(?:oral sex|anal sex|sexual acts? for arousal)\b", re.I)),
    ("adult-content label", re.compile(r"\b(?:xxx|18\+\s*adult)\b", re.I)),
)
_EDUCATIONAL_CONTEXT = re.compile(
    r"\b(?:medical|medicine|anatomy|physiology|histology|clinical|"
    r"reproductive health|obstetrics|gynecology|gynaecology|urology|"
    r"nursing|pathology|medical education|textbook|lecture|patient care)\b",
    re.I,
)


def screen_text(text: str) -> ScreeningResult:
    """Screen extracted text conservatively, exempting clear educational context."""
    if not text or not text.strip():
        return ScreeningResult(ContentDecision.REVIEW, "No extractable text; manual review is required.")
    signals = tuple(label for label, pattern in _EXPLICIT_PATTERNS if pattern.search(text))
    if not signals:
        return ScreeningResult(ContentDecision.ALLOW, "No strong explicit-content signal detected.")
    if _EDUCATIONAL_CONTEXT.search(text):
        return ScreeningResult(
            ContentDecision.REVIEW,
            "Explicit terms occur in an educational/clinical context; request a human review.",
            signals,
        )
    return ScreeningResult(
        ContentDecision.RESTRICTED,
        "Strong explicit-content signals detected in study material.",
        signals,
    )


def may_access_restricted_content(*, medical_verified: bool) -> bool:
    """Only verified medical students may access restricted explicit materials."""
    return bool(medical_verified)


def allowed_educational_domains(domains: Iterable[str]) -> tuple[str, ...]:
    """Normalize allowlisted institution domains for verification workflows."""
    return tuple(sorted({d.strip().lower().lstrip("@") for d in domains if d and d.strip()}))
