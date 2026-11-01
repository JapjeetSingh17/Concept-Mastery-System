"""FastAPI Routes for the Closed-Loop Concept Mastery Platform.

Endpoints:
  - POST /api/upload: Upload PDF, run vectorless indexing, construct prerequisite DAG.
  - GET  /api/concepts/graph: Return full DAG nodes, edges, and user mastery states.
  - POST /api/loop/start: Initialize or resume the closed-loop learning session.
  - POST /api/quiz/submit: Submit answers, trigger LangGraph evaluation, and return next adaptive action.
  - POST /api/tutor/chat: Grounded Socratic conversational endpoint.
  - GET  /api/dashboard/{user_id}: Return explainable learner dashboard.
  - All existing routes preserved for seamless SPA frontend compatibility.
"""

from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel

from . import config
from .core import store
from .pipelines import ingest, tutor, quiz, mastery, recommend, loop
from .pipelines.concepts import export_graph
from .models.schemas import (
    UploadResponse,
    GraphExport,
    QuizSubmission,
    TutorChatRequest,
    TutorChatResponse,
    LearnerDashboard,
    LoopActionResponse,
    IngestStatusResponse,
)

router = APIRouter(prefix="/api")


# ── Health ─────────────────────────────────────────────────────────────────

@router.get("/health")
def health():
    return {"ok": True, "service": "concept-mastery-platform", "version": "2.0.0"}


# ── 1. Document Ingestion & Vectorless Index ───────────────────────────────

@router.post("/upload", response_model=UploadResponse)
@router.post("/upload/", response_model=UploadResponse)
async def post_upload(file: UploadFile = File(...)):
    """Upload PDF, run section chunking & vectorless indexing, build prerequisite DAG."""
    filename = Path(file.filename or "uploaded_material.pdf").name
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    dest = config.UPLOAD_DIR / filename
    content = await file.read()
    dest.write_bytes(content)

    try:
        res = await ingest.run_ingest_async(dest)
        return UploadResponse(**res)
    except store.UserError as ue:
        raise HTTPException(status_code=400, detail=str(ue))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {e}")


@router.post("/ingest")
@router.post("/ingest/")
async def post_ingest(file: UploadFile = File(...)):
    """Legacy ingest endpoint maintaining backward compatibility with existing SPA."""
    name = Path(file.filename or "upload.pdf").name
    if not name.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Please upload a PDF file.")
    dest = config.UPLOAD_DIR / name
    content = await file.read()
    dest.write_bytes(content)
    try:
        res = await ingest.run_ingest_async(dest)
        return {"jobId": name, **res}
    except store.UserError as ue:
        raise HTTPException(status_code=400, detail=str(ue))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Couldn't read this PDF: {e}")


@router.post("/ingest/upload")
@router.post("/ingest/upload/")
async def post_ingest_upload(file: UploadFile = File(...)):
    return await post_upload(file)


@router.get("/ingest/status", response_model=IngestStatusResponse)
@router.get("/ingest/status/", response_model=IngestStatusResponse)
@router.post("/ingest/status", response_model=IngestStatusResponse)
@router.post("/ingest/status/", response_model=IngestStatusResponse)
def ingest_status():
    kb = store.KB
    return IngestStatusResponse(
        ready=kb.get("ready", False),
        file=kb.get("file"),
        pages=kb.get("pages"),
        concepts=len(kb.get("concepts", []))
    )


@router.post("/ingest/sample")
@router.post("/ingest/sample/")
@router.get("/ingest/sample")
@router.get("/ingest/sample/")
def post_ingest_sample():
    candidates = [
        config.UPLOAD_DIR / "ml-basics.pdf",
        config.BASE.parent / "sample" / "ml-basics.pdf",
        config.BASE / "sample" / "ml-basics.pdf",
        Path.cwd() / "sample" / "ml-basics.pdf",
        Path.cwd() / "concept-mastery" / "sample" / "ml-basics.pdf"
    ]
    sample_path = next((p for p in candidates if p.exists()), None)
    if not sample_path:
        raise HTTPException(
            status_code=404,
            detail="Sample PDF not found. The sample directory was removed. Please upload your course or study PDF using the file uploader above."
        )
    dest = config.UPLOAD_DIR / sample_path.name
    if sample_path != dest:
        dest.write_bytes(sample_path.read_bytes())
    res = ingest.run_ingest(dest)
    return {"jobId": sample_path.name, **res}


# ── 2. Prerequisite Concept Graph ──────────────────────────────────────────

@router.get("/concepts/graph")
@router.get("/concepts/graph/")
def get_concepts_graph(user_id: str = "default"):
    """Return full DAG nodes, directed edges, and user mastery states."""
    snap = mastery.snapshot(user_id)
    mastery_map = {s["id"]: s["state"] for s in snap}
    return export_graph(mastery_map)


@router.get("/graph")
def get_graph():
    """Legacy graph endpoint for existing frontend view."""
    kb = store.KB
    if not kb.get("ready"):
        return {"nodes": [], "edges": []}
    return {
        "nodes": [{k: c[k] for k in ("id", "name", "x", "y")} for c in kb["concepts"]],
        "edges": kb["edges"]
    }


# ── 3. Grounded Socratic Tutor ─────────────────────────────────────────────

@router.post("/tutor/chat", response_model=TutorChatResponse)
@router.post("/tutor/chat/", response_model=TutorChatResponse)
def post_tutor_chat(req: TutorChatRequest):
    """Grounded Socratic conversational endpoint with two-layer anti-hallucination guardrails."""
    return tutor.chat(
        message=req.message,
        concept_id=req.concept_id,
        user_id=req.user_id,
    )


class LegacyAsk(BaseModel):
    question: str
    docId: str | None = None
    conceptId: str | None = None

@router.post("/tutor/ask")
def legacy_ask(body: LegacyAsk):
    return tutor.chat(message=body.question, concept_id=body.conceptId).model_dump()


# ── 4. Assessment & Quiz Engine ────────────────────────────────────────────

@router.post("/quiz/submit")
@router.post("/quiz/submit/")
def post_quiz_submit(sub: QuizSubmission):
    """Submit quiz option + reasoning, trigger LangGraph evaluation, return next adaptive action."""
    return loop.submit_quiz_answer(sub)


class LegacyQuizReq(BaseModel):
    conceptId: str | None = None

@router.post("/quiz/next")
def legacy_quiz_next(body: LegacyQuizReq):
    return quiz.next_question(body.conceptId)


class LegacyAnswer(BaseModel):
    questionId: str | None = None
    conceptId: str
    correct: bool | None = None
    selectedIndex: int | None = None
    studentReasoning: str | None = None

@router.post("/quiz/answer")
def legacy_quiz_answer(body: LegacyAnswer):
    sub = QuizSubmission(
        user_id="default",
        question_id=body.questionId or "legacy_q",
        concept_id=body.conceptId,
        selected_option_idx=body.selectedIndex if body.selectedIndex is not None else (0 if body.correct else 1),
        student_reasoning=body.studentReasoning or "",
    )
    res = loop.submit_quiz_answer(sub)
    q_data = quiz.REGISTRY.get(sub.question_id, quiz.RECENT_QUESTIONS.get(sub.question_id, {}))
    return {
        "mastery": mastery.snapshot(),
        "next": res.get("next_concept_id", body.conceptId),
        "correct": res["is_correct"],
        "correctIndex": res["correct_option_idx"],
        "explanation": res["explanation"],
        "next_action": res["next_action"],
        "remedial_lesson": res.get("remedial_lesson"),
    }


class LegacyEvaluateReq(BaseModel):
    conceptId: str
    questionText: str = ""
    studentAnswer: str
    groundTruth: str = ""

@router.post("/quiz/evaluate")
def legacy_quiz_evaluate(body: LegacyEvaluateReq):
    return quiz.evaluate_answer(
        student_answer=body.studentAnswer,
        concept_id=body.conceptId,
        question_text=body.questionText,
        ground_truth=body.groundTruth
    )


# ── 5. Epistemic Mastery Engine & Learner Dashboard ────────────────────────

@router.get("/dashboard/{user_id}", response_model=LearnerDashboard)
@router.get("/dashboard/{user_id}/", response_model=LearnerDashboard)
def get_user_dashboard(user_id: str):
    """Return explainable learner dashboard with UNKNOWN, WEAK, and STRONG concept buckets."""
    return mastery.get_dashboard(user_id)


@router.get("/mastery")
def get_mastery():
    return {"concepts": mastery.snapshot()}


@router.get("/mastery/dashboard")
def get_mastery_dashboard():
    return mastery.get_dashboard("default").model_dump()


@router.post("/mastery/reset")
def post_mastery_reset():
    kb = store.KB
    if not kb.get("ready"):
        raise HTTPException(status_code=400, detail="No active study material loaded.")
    mastery.reset(kb["concepts"])
    return {"ok": True, "concepts": mastery.snapshot()}


# ── 6. Closed-Loop LangGraph Orchestrator ──────────────────────────────────

class StartSessionReq(BaseModel):
    user_id: str = "default"

@router.post("/loop/start")
@router.post("/loop/start/")
def post_loop_start(body: StartSessionReq | None = None):
    """Initialize or advance the closed-loop learning session for a student."""
    uid = body.user_id if body else "default"
    return loop.start_session(uid)


@router.post("/loop/next-action")
def post_loop_next_action():
    return loop.start_session("default")


@router.post("/recommend")
def post_recommend():
    return recommend.recommend()
