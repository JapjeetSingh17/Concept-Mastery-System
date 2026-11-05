"""Pydantic v2 schemas for the Closed-Loop Concept Mastery Platform."""

from __future__ import annotations

from enum import Enum
from typing import TypedDict, Literal, Optional, Any
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
    page_number: int = Field(default=1, ge=1, description="1-indexed source document page")
    section_title: str = Field(..., description="Section / subsection title boundary")
    text: str = Field(..., description="Text content within this semantic section chunk")
    canonical_concepts: list[str] = Field(default_factory=list, description="Canonical concept IDs detected in chunk")
    source_filename: str = Field(default="", description="Original file name")
    locator_label: str = Field(default="", description="Citation anchor (e.g. Page 4, Slide 7, 04:15, Section: Dynamic Programming)")
    source_id: str = Field(default="", description="Identifier of origin study material")
    source_type: str = Field(default="pdf", description="Material format: pdf, pptx, markdown, docx, youtube, web, audio")
    source_title: str = Field(default="", description="Display title of the source")
    origin_url_or_filename: str = Field(default="", description="Original URL or uploaded filename")


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
    """A strictly grounded diagnostic assessment question (MCQ, FITB, Typed Q&A, or Numerical)."""
    question_id: str
    concept_id: str
    question_type: Literal["mcq", "fill_in_the_blanks", "typed_qa", "numerical"] = "mcq"
    question_text: str
    options: list[str] = Field(default_factory=list)
    correct_option_idx: int = Field(default=0, ge=0, le=5)
    target_blank_answer: str = Field(default="", description="Target keyword for fill-in-the-blanks")
    rubric: str = Field(default="", description="Pedagogical rubric for evaluating open-ended typed answers")
    key_terms: list[str] = Field(default_factory=list, description="Essential concept keywords expected")
    explanation: str
    source_page: int = Field(..., ge=1)
    given_parameters: Optional[dict[str, float]] = None
    formula_hint: Optional[str] = None
    correct_numeric_answer: Optional[float] = None
    tolerance: float = 0.05
    unit_or_format: str = "Round to 2 decimal places"
    step_by_step_solution: Optional[str] = None


class QuizSubmission(BaseModel):
    """Student submission with option choice, typed text, numerical answer, and conceptual reasoning."""
    user_id: str = "default"
    question_id: str
    concept_id: str
    selected_option_idx: int | None = None
    typed_answer: str = Field(default="", description="Typed response for fill-in-the-blanks or open-ended Q&A")
    student_reasoning: str = Field(default="", description="Student self-explanation for multi-signal verification")
    numeric_answer: Optional[float] = None


class BaselineSubmissionRequest(BaseModel):
    user_id: str = "default"
    submissions: list[QuizSubmission]


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
    step_by_step_solution: Optional[str] = None


# ── 6. Socratic Tutor Schemas ──────────────────────────────────────────────

class Citation(BaseModel):
    page: int | None = None
    snippet: str
    text: str
    source_type: str = "document"  # "document" | "web" | "youtube" | "audio" | "pptx" | "markdown" | "docx"
    url: str | None = None
    title: str | None = None
    locator_label: str | None = None
    source_id: str | None = None


class ArtifactItem(BaseModel):
    artifact_type: Literal["visual", "pptx", "pdf", "audio"]
    title: str
    url: str
    source_locator: str = ""
    metadata: dict = Field(default_factory=dict)


class ExternalResourceItem(BaseModel):
    title: str
    url: str
    resource_type: Literal["youtube", "article"]
    label: str = "[External Resource]"


class TutorChatRequest(BaseModel):
    user_id: str = "default"
    concept_id: str | None = None
    message: str


class TutorChatResponse(BaseModel):
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    detected_concepts: list[str] = Field(default_factory=list)
    dialogue_score: float = 0.5
    artifacts: list[ArtifactItem] = Field(default_factory=list)
    external_links: list[ExternalResourceItem] = Field(default_factory=list)


# ── 7. LangGraph Closed-Loop State ─────────────────────────────────────────

class LoopState(TypedDict, total=False):
    """State schema for the Closed-Loop LangGraph orchestrator."""
    user_id: str
    target_concept: str
    state: MasteryStatus
    lifecycle: str  # "AWAITING_UPLOAD" | "BASELINE_DIAGNOSTIC" | "ACTIVE_LEARNING"
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


# ── 9. Universal Study Material & Ingestion Schemas ────────────────────────

class IngestUrlRequest(BaseModel):
    """Payload to ingest a YouTube video or educational website."""
    url: str = Field(..., description="Target web or YouTube URL")
    source_type: Literal["youtube", "web", "auto"] = Field(default="auto", description="Source format")
    title: str | None = Field(default=None, description="Optional custom title")
    user_id: str = Field(default="default", description="Active student user ID")


class StudyMaterialMetadata(BaseModel):
    """Compact summary of an ingested study material."""
    material_id: str
    user_id: str = "default"
    title: str
    source_type: str
    original_filename_or_url: str
    chunk_count: int = 0
    concepts: list[str] = Field(default_factory=list)
    concept_count: int = 0
    created_at: str


class StudyMaterialPreview(BaseModel):
    """Rich interactive preview payload for source inspection."""
    material_id: str
    user_id: str = "default"
    title: str
    source_type: str
    original_filename_or_url: str
    chunk_count: int = 0
    concepts: list[str] = Field(default_factory=list)
    concept_count: int = 0
    structured_preview: list[dict] | dict = Field(default_factory=list)
    raw_text_snippet: str = ""
    created_at: str


class MaterialsListResponse(BaseModel):
    materials: list[StudyMaterialMetadata] = Field(default_factory=list)
    total_materials: int = 0
    total_chunks: int = 0
    total_concepts: int = 0


# ── 10. User Authentication & Login Audit Schemas ──────────────────────────

class User(BaseModel):
    user_id: str = Field(..., description="Unique user identifier, derived from Google sub")
    email: str = Field(..., description="User email address")
    full_name: str = Field(default="", description="User full name")
    picture_url: str = Field(default="", description="User profile picture URL")
    created_at: str = Field(default="", description="Account creation timestamp (ISO)")
    last_login_at: str = Field(default="", description="Most recent login timestamp (ISO)")


class UserLoginLog(BaseModel):
    log_id: str = Field(..., description="Unique log UUID")
    user_id: str = Field(..., description="Associated user ID")
    logged_in_at: str = Field(..., description="Login event timestamp (ISO)")
    ip_address: str | None = Field(default=None, description="Client IP address")
    user_agent: str | None = Field(default=None, description="Client User-Agent header")


class GoogleAuthRequest(BaseModel):
    credential: str = Field(..., description="Google ID Token credential from GIS SDK")


class AuthTokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: User


# ── 11. Active Recall Pipeline Schemas ─────────────────────────────────────

class ActiveRecallStartRequest(BaseModel):
    concept_id: str = Field(..., description="Target canonical concept ID")
    timer_seconds: int = Field(default=120, ge=10, le=900, description="Active recall study card display timer in seconds")
    user_id: str = Field(default="default", description="Active student user ID")


class ActiveRecallStudyCard(BaseModel):
    session_id: str
    concept_id: str
    concept_name: str
    timer_seconds: int = 120
    core_intuition: str
    key_formula_or_mechanism: str
    worked_mini_example: str
    essential_keywords: list[str] = Field(default_factory=list)
    exact_source_citation: str
    verification_question: Optional[dict] = None
    created_at: str = ""


class ActiveRecallEvaluateRequest(BaseModel):
    session_id: Optional[str] = Field(default="", description="Active recall session ID")
    concept_id: str
    recall_text: str = Field(default="", description="Student active recall retrieval text")
    verification_answer: Optional[Any] = Field(default=None, description="Answer to the verification question")
    study_duration_seconds: Optional[int] = Field(default=0, description="Elapsed study timer in seconds")
    user_id: str = Field(default="default", description="Active student user ID")


class ActiveRecallResult(BaseModel):
    session_id: str
    concept_id: str
    retrieval_score: float = Field(..., ge=0.0, le=1.0, description="Calibrated recall retrieval score (0.0 to 1.0)")
    reasoning_depth_level: Literal["SURFACE_ROTE", "DEVELOPING", "DEEP_CONCEPTUAL"] = "DEVELOPING"
    retained_concepts: list[str] = Field(default_factory=list)
    missed_gaps: list[str] = Field(default_factory=list)
    verification_correct: bool = False
    feedback: str = ""
    mastery_status: Literal["UNKNOWN", "WEAK", "STRONG"] = "UNKNOWN"
    audit_trail: str = ""


class ActiveRecallHistoryItem(BaseModel):
    session_id: str
    user_id: str = "default"
    concept_id: str
    concept_name: str = ""
    timer_seconds: int = 120
    retrieval_score: Optional[float] = None
    retained_concepts: list[str] = Field(default_factory=list)
    missed_gaps: list[str] = Field(default_factory=list)
    created_at: str = ""


# ── 10. Smart Notes & Tool Request Schemas ─────────────────────────────────

class SmartNoteRequest(BaseModel):
    user_id: str = "default"
    concept_id: str
    material_id: Optional[str] = None
    task_type: str  # "exam" | "viva" | "quiz" | "essay" | "custom"
    custom_instructions: Optional[str] = None


class SmartNoteResponse(BaseModel):
    note_id: str
    user_id: str
    concept_id: str
    task_type: str
    title: str
    markdown_content: str
    key_takeaways: list[str] = Field(default_factory=list)
    citations: list[str] = Field(default_factory=list)
    audio_url: Optional[str] = None
    pdf_url: Optional[str] = None
    pptx_url: Optional[str] = None
    created_at: str = ""


class ToolGenerateRequest(BaseModel):
    tool_type: Literal["visual", "pptx", "pdf", "resources"]
    concept_id: str
    prompt: Optional[str] = ""
    user_id: str = "default"
    slide_count: Optional[int] = 5


class DeepAudioRequest(BaseModel):
    section_or_slide_index: int = 0
    user_id: str = "default"


# ── 12. Multi-Agent Pedagogical Assessment Schemas ─────────────────────────

class LearningEvidenceRecord(BaseModel):
    evidence_id: str
    user_id: str = "default"
    concept_id: str
    question_id: str = ""
    question_text: str = ""
    question_form: Literal["recall", "explanation", "application", "transfer", "prediction", "quiz", "reassessment"] = "quiz"
    student_answer: str
    student_explanation: str = ""
    problem_solving_approach: str = ""
    hints_used: int = 0
    solution_revealed_first: bool = False
    corrected_after_feedback: bool = False
    is_unfamiliar_example: bool = False
    session_id: Optional[str] = None
    created_at: str = ""


class SpecialistAgentAssessment(BaseModel):
    agent_role: Literal["teaching_agent", "assessment_agent", "misconception_detective", "learning_observer"]
    concept_id: str
    evidence_refs: list[str] = Field(default_factory=list)
    findings: str
    identified_gaps: list[str] = Field(default_factory=list)
    recommended_action: str = ""
    assessment_confidence: str = "moderate"
    limitations: str = ""


class HeadTeacherDecision(BaseModel):
    concept_id: str
    outcome: Literal["MASTERED", "NOT_YET_MASTERED", "INSUFFICIENT_EVIDENCE"] = "INSUFFICIENT_EVIDENCE"
    legacy_status: Literal["STRONG", "WEAK", "UNKNOWN"] = "UNKNOWN"
    supporting_evidence_refs: list[str] = Field(default_factory=list)
    demonstrated_understanding: list[str] = Field(default_factory=list)
    remaining_gaps: list[str] = Field(default_factory=list)
    unresolved_disagreements: list[str] = Field(default_factory=list)
    recommended_next_action: str = ""
    follow_up_question: Optional[dict] = None
    evidence_version: int = 1
    explainable_summary: str = ""
    decision_id: Optional[str] = None
    user_id: Optional[str] = None
    updated_at: Optional[str] = None


class EvaluateEvidenceRequest(BaseModel):
    concept_id: str
    user_id: str = "default"
    question_id: Optional[str] = ""
    question_text: Optional[str] = ""
    question_form: Optional[str] = "quiz"
    student_answer: str
    student_explanation: Optional[str] = ""
    problem_solving_approach: Optional[str] = ""
    hints_used: Optional[int] = 0
    solution_revealed_first: Optional[bool] = False
    is_unfamiliar_example: Optional[bool] = False


class EvaluateEvidenceResponse(BaseModel):
    decision: HeadTeacherDecision
    specialist_findings: list[SpecialistAgentAssessment]


class ConceptDeliberationResponse(BaseModel):
    concept_id: str
    user_id: str
    decision: Optional[HeadTeacherDecision] = None
    specialist_findings: list[SpecialistAgentAssessment] = Field(default_factory=list)
    evidence_records: list[LearningEvidenceRecord] = Field(default_factory=list)


