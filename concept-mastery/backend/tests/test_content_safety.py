from app.core.content_safety import (
    ContentDecision,
    may_access_restricted_content,
    screen_text,
)


def test_normal_study_content_is_allowed():
    result = screen_text("Introduction to machine learning and neural networks.")
    assert result.decision is ContentDecision.ALLOW


def test_clinical_anatomy_terms_are_not_explicit_by_themselves():
    result = screen_text("Human anatomy: breast, penis, vagina, reproductive system.")
    assert result.decision is ContentDecision.ALLOW


def test_explicit_material_is_restricted_for_non_medical_users():
    result = screen_text("This is hardcore porn and graphic sexual content.")
    assert result.decision is ContentDecision.RESTRICTED
    assert not may_access_restricted_content(medical_verified=False)


def test_medical_context_is_sent_for_review_not_auto_blocked():
    result = screen_text(
        "Medical education lecture on reproductive health and clinical anatomy with explicit sexual content."
    )
    assert result.decision is ContentDecision.REVIEW


def test_verified_medical_user_can_access_restricted_material():
    assert may_access_restricted_content(medical_verified=True)
