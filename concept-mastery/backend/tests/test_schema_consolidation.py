"""Regression tests verifying single-source-of-truth schema consolidation."""

import pytest
from pydantic import BaseModel, ValidationError

from app import routes, schemas as legacy_schemas
from app.models import schemas as canonical_schemas
from app.pipelines import concepts, ingest, loop, tutor


def test_legacy_schemas_reexports_all_canonical_models_and_aliases():
    """Verify app.schemas thinly re-exports all canonical symbols with identity equality."""
    expected_symbols = [
        "ChunkMetadata",
        "ChunkSchema",
        "ConceptMastery",
        "ConceptDeliberationResponse",
        "ConceptEdge",
        "ConceptNode",
        "DiagnosticSubmitReq",
        "EvaluateEvidenceRequest",
        "EvaluateEvidenceResponse",
        "ExtractedConceptItem",
        "ExtractedConceptsResponse",
        "GraphExport",
        "IngestResponse",
        "IngestStatusResponse",
        "IngestionState",
        "LearnerDashboard",
        "LearningEvidenceRecord",
        "LegacyAnswer",
        "LegacyAsk",
        "LegacyEvaluateReq",
        "LegacyQuizReq",
        "LoopState",
        "MasteryDashboard",
        "MasteryRecord",
        "MasteryStatus",
        "QuizQuestion",
        "QuizResult",
        "QuizSubmission",
        "RemedialQuizReq",
        "StartSessionReq",
        "TutorChatRequest",
        "TutorChatResponse",
        "TutorResponse",
        "UploadResponse",
        "WebSearchRequest",
    ]
    for name in expected_symbols:
        assert hasattr(canonical_schemas, name), f"Missing canonical schema symbol: {name}"
        assert hasattr(legacy_schemas, name), f"Missing re-exported schema symbol: {name}"
        assert getattr(legacy_schemas, name) is getattr(canonical_schemas, name)

    assert canonical_schemas.ChunkSchema is canonical_schemas.ChunkMetadata
    assert canonical_schemas.ConceptMastery is canonical_schemas.MasteryRecord
    assert canonical_schemas.MasteryDashboard is canonical_schemas.LearnerDashboard
    assert canonical_schemas.TutorResponse is canonical_schemas.TutorChatResponse


def test_routes_and_pipelines_bind_canonical_schemas():
    """Verify routes and pipeline modules reference canonical schema objects without local duplicates."""
    assert concepts.ExtractedConceptItem is canonical_schemas.ExtractedConceptItem
    assert concepts.ExtractedConceptsResponse is canonical_schemas.ExtractedConceptsResponse
    assert ingest.IngestionState is canonical_schemas.IngestionState
    assert loop.TutorResponse is canonical_schemas.TutorResponse
    assert tutor.TutorResponse is canonical_schemas.TutorResponse

    for route_model in (
        "LegacyAsk",
        "WebSearchRequest",
        "LegacyQuizReq",
        "RemedialQuizReq",
        "LegacyAnswer",
        "LegacyEvaluateReq",
        "StartSessionReq",
        "DiagnosticSubmitReq",
    ):
        assert getattr(routes, route_model) is getattr(canonical_schemas, route_model)


def test_merged_model_signatures_and_defaults():
    """Verify field defaults, constraints, and non-shared mutable defaults on merged models."""
    status = canonical_schemas.IngestStatusResponse()
    assert status.ready is False
    assert status.file is None
    assert status.pages is None
    assert status.concepts == 0

    ingest_resp = canonical_schemas.IngestResponse(
        job_id="job-1", file="notes.pdf", pages=10, chunk_count=25
    )
    assert ingest_resp.concepts == []
    with pytest.raises(ValidationError):
        canonical_schemas.IngestResponse(
            job_id="job-1", file="notes.pdf", pages=-1, chunk_count=0
        )

    extracted = canonical_schemas.ExtractedConceptsResponse(
        concepts=[canonical_schemas.ExtractedConceptItem(concept="Binary Search")]
    )
    assert extracted.concepts[0].definition == ""
    assert extracted.concepts[0].prerequisites == []

    diag_a = canonical_schemas.DiagnosticSubmitReq()
    diag_b = canonical_schemas.DiagnosticSubmitReq()
    diag_a.answers.append({"concept_id": "arrays"})
    assert diag_b.answers == []
    assert diag_a.user_id == "default"

    ask = canonical_schemas.LegacyAsk(question="What is a heap?")
    assert ask.docId is None and ask.conceptId is None
    assert canonical_schemas.WebSearchRequest(query="heaps").max_results == 5
    assert canonical_schemas.RemedialQuizReq().user_id == "default"
    assert canonical_schemas.StartSessionReq().user_id == "default"
    assert canonical_schemas.LegacyAnswer(conceptId="heaps").typedAnswer is None
    assert canonical_schemas.LegacyEvaluateReq(conceptId="heaps", studentAnswer="tree").groundTruth == ""

    tutor_msg = canonical_schemas.TutorResponse(answer="Grounded answer", citations=[])
    assert isinstance(tutor_msg, canonical_schemas.TutorChatResponse)
    assert isinstance(tutor_msg, canonical_schemas.TutorResponse)
    assert issubclass(canonical_schemas.TutorResponse, BaseModel)


def test_oversized_payload_validator_preserved():
    """Verify reject_oversized_payload rejects strings > 50,000 chars across both models."""
    oversized = "x" * 50_001
    boundary = "x" * 50_000

    valid_record = canonical_schemas.LearningEvidenceRecord(
        evidence_id="ev-1",
        concept_id="arrays",
        student_answer=boundary,
    )
    assert len(valid_record.student_answer) == 50_000

    with pytest.raises(ValidationError, match="50,000"):
        canonical_schemas.LearningEvidenceRecord(
            evidence_id="ev-2",
            concept_id="arrays",
            student_answer=oversized,
        )

    valid_eval = canonical_schemas.EvaluateEvidenceRequest(
        concept_id="arrays",
        student_answer="ok",
        student_explanation=boundary,
    )
    assert len(valid_eval.student_explanation) == 50_000

    with pytest.raises(ValidationError, match="50,000"):
        canonical_schemas.EvaluateEvidenceRequest(
            concept_id="arrays",
            student_answer="ok",
            problem_solving_approach=oversized,
        )
