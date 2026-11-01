"""Pydantic v2 schemas for the Closed-Loop Concept Mastery Platform."""

from __future__ import annotations

from enum import Enum
from typing import TypedDict, Literal
from pydantic import BaseModel, Field


# ── 1. Mastery Status Enum ─────────────────────────────────────────────────

class MasteryStatus(str, Enum):
    UNKNOWN = "UNKNOWN"
    WEAK = "WEAK"
    STRONG = "STRONG"


# ── 2. Vectorless Document Chunk Schema ────────────────────────────────────

class ChunkMetadata(BaseModel):
    """Chunk with exact 1-indexed page metadata, structural heading, and canonical concept tags."""
    chunk_id: str = Field(..., description="Unique chunk UUID")
    page_number: int = Field(..., ge=1, description="1-indexed source document page")
    section_title: str = Field(..., description="Section / subsection title boundary")
    text: str = Field(..., description="Text content within this semantic section chunk")
    canonical_concepts: list[str] = Field(default_factory=list, description="Canonical concept IDs detected in chunk")
    source_filename: str = Field(default="", description="Original file name")


# ── 3. Prerequisite Concept Graph Schemas ──────────────────────────────────

class ConceptNode(BaseModel):
    """Node in the Directed Acyclic Graph (DAG) of concepts."""
    concept_id: str = Field(..., description="Normalized unique concept identifier (slug)")
    name: str = Field(..., description="Human-readable concept display name")
    definition: str = Field(default="", description="Grounded definition extracted from textbook")
    prerequisites: list[str] = Field(default_factory=list, description="Concept IDs that must be mastered first")
    source_pages: list[int] = Field(default_factory=list, description="Referenced 1-indexed page numbers")
    depth: int = Field(default=0, ge=0, description="Topological distance from root")
    x: float = Field(default=0.0, description="Visualization layout x-coordinate")
    y: float = Field(default=0.0, description="Visualization layout y-coordinate")
    mastery_status: MasteryStatus = Field(default=MasteryStatus.UNKNOWN, description="Current user mastery state")


class ConceptEdge(BaseModel):
    """Directed edge representing prerequisite dependencies."""
    source: str = Field(..., description="Prerequisite concept ID")
    target: str = Field(..., description="Dependent concept ID")
    relationship: str = Field(default="prerequisite", description="Relationship descriptor")


class GraphExport(BaseModel):
    """Exportable DAG structure for frontend visualization."""
    nodes: list[ConceptNode] = Field(default_factory=list)
    edges: list[ConceptEdge] = Field(default_factory=list)
    learning_path: list[str] = Field(default_factory=list, description="Topological sort order of concepts")


# ── 4. Knowledge Tracing & Epistemic Mastery Record ────────────────────────

class MasteryEvaluationResult(BaseModel):
    """Pydantic v2 schema for Semantic LLM-as-a-Judge Conceptual Evaluation."""
    concept_id: str = Field(..., description="Target canonical concept ID")
    epistemic_status: Literal["UNKNOWN", "WEAK", "STRONG"] = Field(
        ..., description="3-state epistemic classification determined by semantic comprehension"
    )
    semantic_comprehension_score: float = Field(
        ..., ge=0.0, le=1.0, description="Calibrated comprehension score (0.0 to 1.0) by AI"
    )
    misconceptions_detected: list[str] = Field(
        default_factory=list, description="Specific conceptual misconceptions or reasoning flaws detected"
    )
    reasoning_depth_level: Literal["SURFACE_ROTE", "DEVELOPING", "DEEP_CONCEPTUAL"] = Field(
        default="DEVELOPING", description="Cognitive depth level of student explanation"
    )
    remedial_recommended: bool = Field(
        default=False, description="Flag indicating if targeted remedial intervention is recommended"
    )
    explainable_audit_trail: str = Field(
        ..., description="Transparent pedagogical evidence and audit trail for student dashboard"
    )


class MasteryRecord(BaseModel):
    """Explainable mastery record for a user-concept pair."""
    user_id: str = Field(default="default")
    concept_id: str
    score: float = Field(default=0.5, ge=0.0, le=1.0, description="Multi-signal mastery score")
    status: MasteryStatus = Field(default=MasteryStatus.UNKNOWN)
    attempt_count: int = Field(default=0, ge=0)
    evidence_notes: list[str] = Field(default_factory=list, description="Explainable chronological audit notes")
    updated_at: str | None = None
    quiz_accuracy: float = Field(default=0.0, ge=0.0, le=1.0)
    dialogue_score: float = Field(default=0.5, ge=0.0, le=1.0)
    prereq_health: float = Field(default=1.0, ge=0.0, le=1.0)
    reasoning_depth_level: str = Field(default="DEVELOPING", description="Cognitive depth level of student explanation")
    misconceptions: list[str] = Field(default_factory=list, description="Detected conceptual misconceptions")
    audit_trail: str = Field(default="", description="Transparent explainable audit trail from evaluator")


class LearnerDashboard(BaseModel):
    """Explainable learner state partitioned into epistemic buckets."""
    user_id: str
    strong: list[MasteryRecord] = Field(default_factory=list)
    weak: list[MasteryRecord] = Field(default_factory=list)
    unknown: list[MasteryRecord] = Field(default_factory=list)
    overall_score: float = Field(default=0.0, ge=0.0, le=1.0)
    total_concepts: int = 0
    summary: str = ""


# ── 5. Assessment & Quiz Schemas ───────────────────────────────────────────

class QuizQuestion(BaseModel):
    """A strictly grounded diagnostic assessment question (MCQ, FITB, or Typed Q&A)."""
    question_id: str
    concept_id: str
    question_type: Literal["mcq", "fill_in_the_blanks", "typed_qa"] = "mcq"
    question_text: str
    options: list[str] = Field(default_factory=list)
    correct_option_idx: int = Field(default=0, ge=0, le=5)
    target_blank_answer: str = Field(default="", description="Target keyword for fill-in-the-blanks")
    rubric: str = Field(default="", description="Pedagogical rubric for evaluating open-ended typed answers")
    key_terms: list[str] = Field(default_factory=list, description="Essential concept keywords expected")
    explanation: str
    source_page: int = Field(..., ge=1)


class QuizSubmission(BaseModel):
    """Student submission with option choice, typed text, and conceptual reasoning."""
    user_id: str = "default"
    question_id: str
    concept_id: str
    selected_option_idx: int | None = None
    typed_answer: str = Field(default="", description="Typed response for fill-in-the-blanks or open-ended Q&A")
    student_reasoning: str = Field(default="", description="Student self-explanation for multi-signal verification")


class QuizResult(BaseModel):
    """Graded result of a quiz submission."""
    question_id: str
    concept_id: str
    question_type: str = "mcq"
    is_correct: bool
    correct_option_idx: int = 0
    correct_answer_display: str = ""
    user_answer: str = ""
    explanation: str
    score: float
    misconceptions: list[str] = Field(default_factory=list)
    reasoning_clarity: float = Field(default=0.5)
    reasoning_depth_level: str = "DEVELOPING"


# ── 6. Socratic Tutor Schemas ──────────────────────────────────────────────

class Citation(BaseModel):
    page: int | None = None
    snippet: str
    text: str
    source_type: str = "document"  # "document" | "web"
    url: str | None = None
    title: str | None = None


class TutorChatRequest(BaseModel):
    user_id: str = "default"
    concept_id: str | None = None
    message: str


class TutorChatResponse(BaseModel):
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    detected_concepts: list[str] = Field(default_factory=list)
    dialogue_score: float = 0.5


# ── 7. LangGraph Closed-Loop State ─────────────────────────────────────────

class LoopState(TypedDict, total=False):
    """State schema for the Closed-Loop LangGraph orchestrator."""
    user_id: str
    target_concept: str
    state: MasteryStatus
    quiz_questions: list[dict]
    quiz_results: list[dict]
    misconception_notes: list[str]
    next_step: str  # "DIAGNOSE" | "ASSESS" | "EVALUATE" | "REMEDIAL_RETEACH" | "UNLOCK_NEXT" | "COMPLETE"
    remedial_lesson: str | None
    iteration_count: int
    evidence_trail: list[str]
    error: str | None


# ── 8. API Responses ───────────────────────────────────────────────────────

class UploadResponse(BaseModel):
    job_id: str
    filename: str
    filenames: list[str] = Field(default_factory=list)
    files_count: int = 1
    pages: int
    chunks_indexed: int
    concepts_found: list[str]
    graph_nodes: int
    graph_edges: int


class IngestStatusResponse(BaseModel):
    ready: bool = False
    file: str | None = None
    pages: int | None = None
    concepts: int = 0


class LoopActionResponse(BaseModel):
    user_id: str
    action: str  # "quiz" | "remedial_teach" | "advance" | "complete"
    concept_id: str | None = None
    concept_name: str | None = None
    reason: str = ""
    quiz: dict | None = None
    remedial_lesson: str | None = None
    status: MasteryStatus = MasteryStatus.UNKNOWN
    mastery_score: float = 0.0
