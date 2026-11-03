"""Schema exports maintaining backward compatibility and new canonical models."""

from .models.schemas import (
    MasteryStatus,
    ChunkMetadata,
    ConceptNode,
    ConceptEdge,
    GraphExport,
    MasteryRecord,
    LearnerDashboard,
    QuizQuestion,
    QuizSubmission,
    QuizResult,
    Citation,
    TutorChatRequest,
    TutorChatResponse,
    LoopState,
    UploadResponse,
    LoopActionResponse,
    ActiveRecallStartRequest,
    ActiveRecallStudyCard,
    ActiveRecallEvaluateRequest,
    ActiveRecallResult,
    ActiveRecallHistoryItem,
)

# Aliases for existing API compatibility
ChunkSchema = ChunkMetadata
ConceptMastery = MasteryRecord
MasteryDashboard = LearnerDashboard

class IngestResponse(UploadResponse):
    pass

class IngestStatusResponse(BaseModel if "BaseModel" in globals() else object):
    pass

from pydantic import BaseModel, Field

class IngestStatusResponse(BaseModel):
    ready: bool = False
    file: str | None = None
    pages: int | None = None
    concepts: int = 0

class IngestResponse(BaseModel):
    job_id: str = Field(..., description="Identifier for this ingest job (filename)")
    file: str = Field(..., description="Uploaded PDF filename")
    pages: int = Field(..., ge=0, description="Total pages extracted")
    chunk_count: int = Field(..., ge=0, description="Number of chunks produced")
    concepts: list[str] = Field(default_factory=list, description="Canonical concept names discovered")
