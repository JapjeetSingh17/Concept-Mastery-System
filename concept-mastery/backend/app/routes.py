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

import logging
import time
from pathlib import Path
from typing import Optional, Literal
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Request, Depends, Query, Body, Response
from fastapi.concurrency import run_in_threadpool

from . import config
from .core import store, tts
from .core.auth import get_current_user, create_access_token, verify_google_credential
from .pipelines import ingest, tutor, quiz, mastery, recommend, loop, concepts, active_recall, notes, tools, mastery_orchestrator
from .pipelines.concepts import export_graph
from .models.schemas import (
    UploadResponse,
    GraphExport,
    QuizSubmission,
    BaselineSubmissionRequest,
    TutorChatRequest,
    TutorChatResponse,
    LearnerDashboard,
    LoopActionResponse,
    IngestStatusResponse,
    IngestUrlRequest,
    StudyMaterialMetadata,
    StudyMaterialPreview,
    MaterialsListResponse,
    User,
    UserLoginLog,
    GoogleAuthRequest,
    AuthTokenResponse,
    ActiveRecallStartRequest,
    ActiveRecallStudyCard,
    ActiveRecallEvaluateRequest,
    ActiveRecallResult,
    ActiveRecallHistoryItem,
    SmartNoteRequest,
    SmartNoteResponse,
    ToolGenerateRequest,
    DeepAudioRequest,
    EvaluateEvidenceRequest,
    EvaluateEvidenceResponse,
    ConceptDeliberationResponse,
    HeadTeacherDecision,
    SpecialistAgentAssessment,
    LearningEvidenceRecord,
    LegacyAsk,
    WebSearchRequest,
    LegacyQuizReq,
    RemedialQuizReq,
    LegacyAnswer,
    LegacyEvaluateReq,
    StartSessionReq,
    DiagnosticSubmitReq,
)
from .pipelines.ingest import UniversalMaterialIngestor

# Ensure store.get_material alias exists for study materials
if not hasattr(store, "get_material"):
    store.get_material = store.get_study_material

logger = logging.getLogger("concept_mastery.routes")

router = APIRouter(prefix="/api")


# ── Health ─────────────────────────────────────────────────────────────────

@router.get("/health")
def health():
    return {
        "ok": True,
        "service": "concept-mastery-platform",
        "version": "2.0.0",
        "google_client_id": config.GOOGLE_CLIENT_ID,
        "google_auth_configured": bool(config.GOOGLE_CLIENT_ID)
    }


# ── Authentication & Student Profiles ─────────────────────────────────────

@router.get("/auth/config")
def get_auth_config():
    """Return public OAuth configuration for frontend initialization."""
    return {
        "google_client_id": config.GOOGLE_CLIENT_ID,
        "google_auth_configured": bool(config.GOOGLE_CLIENT_ID)
    }

@router.post("/auth/google", response_model=AuthTokenResponse)
@router.post("/auth/google/", response_model=AuthTokenResponse)
def post_auth_google(req: GoogleAuthRequest, request: Request):
    """Verify Google ID Token credential, upsert student account, log session, and issue session JWT."""
    idinfo = verify_google_credential(req.credential)
    user_id = idinfo["sub"]
    email = idinfo["email"]
    full_name = idinfo.get("name", "")
    picture_url = idinfo.get("picture", "")

    user_dict = store.upsert_user(
        user_id=user_id,
        email=email,
        full_name=full_name,
        picture_url=picture_url
    )

    client_ip = request.client.host if request.client else None
    user_agent = request.headers.get("User-Agent")
    store.record_login_log(user_id=user_id, ip_address=client_ip, user_agent=user_agent)

    token = create_access_token({"sub": user_id, "email": email, "name": full_name, "picture": picture_url})
    return AuthTokenResponse(
        access_token=token,
        token_type="bearer",
        user=User(**user_dict)
    )


@router.get("/auth/me")
@router.get("/auth/me/")
def get_auth_me(current_user: User = Depends(get_current_user)):
    """Return profile and active session for the authenticated student."""
    logs = store.get_user_login_logs(current_user.user_id, limit=5)
    return {
        "authenticated": True,
        "user": current_user,
        "recent_logins": logs
    }


@router.get("/auth/logs")
@router.get("/auth/logs/")
def get_auth_logs(current_user: User = Depends(get_current_user)):
    """Return full login activity trail for the active student."""
    return {
        "user_id": current_user.user_id,
        "logs": store.get_user_login_logs(current_user.user_id)
    }


# ── 1. Document Ingestion & Vectorless Index ───────────────────────────────

def _write_uploaded_bytes(dest: Path, content: bytes) -> None:
    """Create parent directory if needed and write uploaded bytes synchronously (called via threadpool)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(content)


async def _save_uploaded_pdfs(files: list[UploadFile], file: UploadFile | None) -> list[Path]:
    all_uploads: list[UploadFile] = list(files or [])
    if file:
        all_uploads.append(file)

    valid_pdfs = [f for f in all_uploads if f.filename and f.filename.lower().endswith(".pdf")]
    if not valid_pdfs:
        raise HTTPException(status_code=400, detail="Please upload at least one valid PDF file.")

    saved_paths: list[Path] = []
    for f in valid_pdfs:
        fname = Path(f.filename).name
        dest = config.UPLOAD_DIR / fname
        content = await f.read()
        await run_in_threadpool(_write_uploaded_bytes, dest, content)
        saved_paths.append(dest)
    return saved_paths


def _empty_kb(user_id: str) -> dict:
    return {
        "ready": False,
        "file": None,
        "pages": 0,
        "index": None,
        "concepts": [],
        "edges": [],
        "user_id": user_id,
    }


def _is_explicit_auth(request: Request) -> bool:
    return getattr(request.state, "auth_source", "jwt") in ("jwt", "dev_header")


def _resolve_tenant_user_id(
    current_user: User,
    request: Request,
    *requested_user_ids: Optional[str],
) -> str:
    """Resolve active tenant user_id and block cross-tenant user_id spoofing."""
    auth_uid = (current_user.user_id if current_user else "").strip() or "default"
    explicit = _is_explicit_auth(request)
    resolved_fallback: Optional[str] = None

    for raw_uid in requested_user_ids:
        if raw_uid is None:
            continue
        candidate = str(raw_uid).strip()
        if not candidate or candidate == "default":
            continue
        if explicit:
            if candidate != auth_uid:
                raise HTTPException(
                    status_code=403,
                    detail=f"Access denied: authenticated user '{auth_uid}' cannot act as user '{candidate}'.",
                )
        elif resolved_fallback is None:
            resolved_fallback = candidate

    if explicit:
        return auth_uid
    return resolved_fallback or auth_uid


def _get_tenant_kb(uid: str, request: Request) -> dict:
    """Fetch tenant KB without falling back to the global 'default' KB for authenticated users."""
    if _is_explicit_auth(request) and uid != "default" and uid not in store.USER_KBS:
        if not store.list_study_materials(uid):
            return _empty_kb(uid)
    return store.get_kb(uid)


def _require_owned_material(material_id: str, uid: str, request: Request) -> dict:
    """Ensure a study material belongs to the resolved tenant or return 403/404."""
    mat = store.get_study_material(material_id, user_id=uid)
    if mat:
        return mat
    existing = store.get_study_material(material_id, user_id=None)
    if existing:
        if not _is_explicit_auth(request) and uid == "default" and not request.query_params.get("user_id"):
            return existing
        raise HTTPException(
            status_code=403,
            detail=f"Access denied: study material '{material_id}' belongs to another user.",
        )
    raise HTTPException(status_code=404, detail=f"Study material '{material_id}' not found.")


def _require_owned_note(note_id: str, uid: str) -> dict:
    """Ensure a smart note belongs to the resolved tenant or return 403/404."""
    note = store.get_smart_note(note_id, user_id=uid)
    if note:
        return note
    existing = store.get_smart_note(note_id, user_id=None)
    if existing and existing.get("user_id") != uid:
        raise HTTPException(
            status_code=403,
            detail=f"Access denied: smart note '{note_id}' belongs to another user.",
        )
    raise HTTPException(status_code=404, detail=f"Smart note '{note_id}' not found.")


def _require_owned_active_recall_session(session_id: Optional[str], uid: str) -> None:
    """Ensure a referenced active recall session belongs to the resolved tenant."""
    if not session_id:
        return
    if store.get_active_recall_session_for_user(session_id, user_id=uid):
        return
    existing = store.get_active_recall_session(session_id)
    if existing and existing.get("user_id") != uid:
        raise HTTPException(
            status_code=403,
            detail=f"Access denied: active recall session '{session_id}' belongs to another user.",
        )
    raise HTTPException(status_code=404, detail=f"Active recall session '{session_id}' not found.")


def _require_owned_question(question_id: Optional[str], uid: str, request: Request) -> None:
    """Ensure a submitted quiz question ID belongs to the resolved tenant."""
    if not question_id:
        return
    q_data = quiz.REGISTRY.get(question_id) or quiz.RECENT_QUESTIONS.get(question_id)
    if not q_data:
        return
    owner = q_data.get("user_id")
    if not owner:
        return
    if _is_explicit_auth(request):
        if owner != uid:
            raise HTTPException(
                status_code=403,
                detail=f"Access denied: quiz question '{question_id}' belongs to another user.",
            )
    elif owner != "default" and owner != uid:
        raise HTTPException(
            status_code=403,
            detail=f"Access denied: quiz question '{question_id}' belongs to another user.",
        )


@router.post("/upload", response_model=UploadResponse)
@router.post("/upload/", response_model=UploadResponse)
async def post_upload(
    request: Request,
    files: list[UploadFile] = File(default=[]),
    file: UploadFile | None = File(default=None),
    user_id: Optional[str] = Query(None),
    current_user: User = Depends(get_current_user),
):
    """Upload one or multiple PDFs, run section chunking & vectorless indexing, build prerequisite DAG."""
    uid = _resolve_tenant_user_id(current_user, request, user_id)
    saved_paths = await _save_uploaded_pdfs(files, file)
    try:
        res = await ingest.run_ingest_async(saved_paths, user_id=uid)
        return UploadResponse(**res)
    except store.UserError as ue:
        raise HTTPException(status_code=400, detail=str(ue))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {e}")


@router.post("/ingest")
@router.post("/ingest/")
async def post_ingest(
    request: Request,
    files: list[UploadFile] = File(default=[]),
    file: UploadFile | None = File(default=None),
    user_id: Optional[str] = Query(None),
    current_user: User = Depends(get_current_user),
):
    """Ingest one or multiple PDFs maintaining backward compatibility with existing SPA."""
    uid = _resolve_tenant_user_id(current_user, request, user_id)
    saved_paths = await _save_uploaded_pdfs(files, file)
    try:
        res = await ingest.run_ingest_async(saved_paths, user_id=uid)
        return {"jobId": res["job_id"], **res}
    except store.UserError as ue:
        raise HTTPException(status_code=400, detail=str(ue))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Couldn't read PDF(s): {e}")


@router.post("/ingest/upload")
@router.post("/ingest/upload/")
async def post_ingest_upload(
    request: Request,
    files: list[UploadFile] = File(default=[]),
    file: UploadFile | None = File(default=None),
    user_id: Optional[str] = Query(None),
    current_user: User = Depends(get_current_user),
):
    return await post_upload(request=request, files=files, file=file, user_id=user_id, current_user=current_user)


@router.get("/ingest/status", response_model=IngestStatusResponse)
@router.get("/ingest/status/", response_model=IngestStatusResponse)
@router.post("/ingest/status", response_model=IngestStatusResponse)
@router.post("/ingest/status/", response_model=IngestStatusResponse)
def ingest_status(
    request: Request,
    user_id: str = "default",
    current_user: User = Depends(get_current_user),
):
    uid = _resolve_tenant_user_id(current_user, request, user_id)
    kb = _get_tenant_kb(uid, request)
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
def post_ingest_sample(
    request: Request,
    user_id: str = "default",
    current_user: User = Depends(get_current_user),
):
    uid = _resolve_tenant_user_id(current_user, request, user_id)
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
    res = ingest.run_ingest(dest, user_id=uid)
    return {"jobId": sample_path.name, **res}


# ── 2. Prerequisite Concept Graph ──────────────────────────────────────────

@router.get("/concepts/graph")
@router.get("/concepts/graph/")
def get_concepts_graph(
    request: Request,
    user_id: str = "default",
    current_user: User = Depends(get_current_user),
):
    """Return full DAG nodes, directed edges, and user mastery states."""
    uid = _resolve_tenant_user_id(current_user, request, user_id)
    kb = _get_tenant_kb(uid, request)
    mats = store.list_study_materials(uid)
    if not mats and not kb.get("ready"):
        return {"nodes": [], "edges": [], "learning_path": []}
    if not kb.get("ready") or not kb.get("concepts"):
        return {"nodes": [], "edges": [], "learning_path": []}
    snap = mastery.snapshot(uid)
    mastery_map = {s["id"]: s["state"] for s in snap}
    return export_graph(mastery_map, user_id=uid)


@router.get("/graph")
def get_graph(
    request: Request,
    user_id: str = "default",
    current_user: User = Depends(get_current_user),
):
    """Legacy graph endpoint for existing frontend view."""
    uid = _resolve_tenant_user_id(current_user, request, user_id)
    mats = store.list_study_materials(uid)
    if not mats:
        return {"nodes": [], "edges": []}
    kb = _get_tenant_kb(uid, request)
    if not kb.get("ready"):
        return {"nodes": [], "edges": []}
    return {
        "nodes": [{k: c[k] for k in ("id", "name", "x", "y")} for c in kb.get("concepts", [])],
        "edges": kb.get("edges", [])
    }


# ── 3. Grounded Socratic Tutor ─────────────────────────────────────────────

@router.post("/tutor/chat", response_model=TutorChatResponse)
@router.post("/tutor/chat/", response_model=TutorChatResponse)
def post_tutor_chat(
    req: TutorChatRequest,
    request: Request,
    current_user: User = Depends(get_current_user),
):
    """Grounded Socratic conversational endpoint with two-layer anti-hallucination guardrails."""
    uid = _resolve_tenant_user_id(current_user, request, req.user_id)
    kb = _get_tenant_kb(uid, request)
    return tutor.chat(
        message=req.message,
        concept_id=req.concept_id,
        user_id=uid,
        kb=kb,
    )


@router.post("/tutor/ask")
def legacy_ask(
    body: LegacyAsk,
    request: Request,
    current_user: User = Depends(get_current_user),
):
    uid = _resolve_tenant_user_id(current_user, request)
    kb = _get_tenant_kb(uid, request)
    return tutor.chat(message=body.question, concept_id=body.conceptId, user_id=uid, kb=kb).model_dump()



@router.post("/tutor/search")
@router.post("/web/search")
async def post_web_search(req: WebSearchRequest):
    """Execute live web search via TinyFish Search API."""
    from .core import web_search
    if getattr(web_search.search, "_is_default_sync_search", False):
        results = await web_search.search_async(req.query, max_results=req.max_results)
    else:
        results = await run_in_threadpool(web_search.search, req.query, max_results=req.max_results)
    return {"query": req.query, "results": results, "count": len(results)}


# ── 4. Assessment & Quiz Engine ────────────────────────────────────────────

@router.post("/quiz/submit")
@router.post("/quiz/submit/")
def post_quiz_submit(
    sub: QuizSubmission,
    request: Request,
    current_user: User = Depends(get_current_user),
):
    """Submit quiz option + reasoning, trigger LangGraph evaluation, return next adaptive action."""
    uid = _resolve_tenant_user_id(current_user, request, sub.user_id)
    _require_owned_question(sub.question_id, uid, request)
    sub.user_id = uid
    return loop.submit_quiz_answer(sub)


@router.post("/quiz/next")
@router.post("/quiz/next/")
def legacy_quiz_next(
    request: Request,
    body: LegacyQuizReq = LegacyQuizReq(),
    current_user: User = Depends(get_current_user),
):
    """Generate fresh non-duplicate question variants for Concept Check."""
    uid = _resolve_tenant_user_id(current_user, request, body.user_id if body else None)
    kb = _get_tenant_kb(uid, request)
    if not kb.get("ready") or not kb.get("concepts"):
        raise HTTPException(status_code=400, detail="No study materials indexed yet.")
    cid = body.conceptId or body.concept_id
    try:
        return quiz.next_question(cid, user_id=uid)
    except store.UserError as ue:
        raise HTTPException(status_code=400, detail=str(ue))


@router.post("/quiz/remedial")
@router.post("/quiz/remedial/", include_in_schema=False)
def post_quiz_remedial(
    request: Request,
    body: RemedialQuizReq = RemedialQuizReq(),
    current_user: User = Depends(get_current_user),
):
    """Generate a targeted remedial quiz for the student's top WEAK prerequisite root cause."""
    uid = _resolve_tenant_user_id(current_user, request, body.user_id if body else None)
    kb = _get_tenant_kb(uid, request)
    if not kb.get("ready") or not kb.get("concepts"):
        raise HTTPException(status_code=400, detail="No study materials indexed yet.")
    cid = body.conceptId or body.concept_id
    try:
        res = loop.generate_remedial_quiz(user_id=uid, target_concept_id=cid)
        if "question" not in res or not isinstance(res["question"], dict):
            res["question"] = dict(res)
        res["ok"] = True
        return res
    except store.UserError as ue:
        raise HTTPException(status_code=400, detail=str(ue))


@router.post("/quiz/answer")
def legacy_quiz_answer(
    body: LegacyAnswer,
    request: Request,
    current_user: User = Depends(get_current_user),
):
    uid = _resolve_tenant_user_id(current_user, request)
    _require_owned_question(body.questionId, uid, request)
    num_val = body.numericAnswer if body.numericAnswer is not None else body.numeric_answer
    sub = QuizSubmission(
        user_id=uid,
        question_id=body.questionId or "legacy_q",
        concept_id=body.conceptId,
        selected_option_idx=body.selectedIndex,
        typed_answer=body.typedAnswer or "",
        student_reasoning=body.studentReasoning or (body.typedAnswer or ""),
        numeric_answer=num_val,
    )
    res = loop.submit_quiz_answer(sub)
    q_data = quiz.REGISTRY.get(sub.question_id, quiz.RECENT_QUESTIONS.get(sub.question_id, {}))
    return {
        "mastery": mastery.snapshot(uid),
        "next": res.get("next_concept_id", body.conceptId),
        "correct": res["is_correct"],
        "correctIndex": res["correct_option_idx"],
        "explanation": res["explanation"],
        "next_action": res["next_action"],
        "remedial_lesson": res.get("remedial_lesson"),
        "question_type": res.get("question_type", "mcq"),
        "correct_answer_display": res.get("correct_answer_display", ""),
        "user_answer": res.get("user_answer", ""),
        "reasoning_depth_level": res.get("reasoning_depth_level", "DEVELOPING"),
        "step_by_step_solution": res.get("step_by_step_solution") or q_data.get("step_by_step_solution"),
    }


@router.post("/quiz/evaluate")
def legacy_quiz_evaluate(
    body: LegacyEvaluateReq,
    request: Request,
    current_user: User = Depends(get_current_user),
):
    uid = _resolve_tenant_user_id(current_user, request)
    return quiz.evaluate_answer(
        student_answer=body.studentAnswer,
        concept_id=body.conceptId,
        question_text=body.questionText,
        ground_truth=body.groundTruth,
        user_id=uid,
    )


# ── 4.5 Active Recall Pipeline ─────────────────────────────────────────────

@router.post("/active-recall/start", response_model=ActiveRecallStudyCard)
@router.post("/active-recall/start/", response_model=ActiveRecallStudyCard)
def post_active_recall_start(
    req: ActiveRecallStartRequest,
    request: Request,
    current_user: User = Depends(get_current_user),
):
    """Initialize an Active Recall study session with grounded Study Card & verification question."""
    uid = _resolve_tenant_user_id(current_user, request, req.user_id)
    try:
        return active_recall.start_session(
            user_id=uid,
            concept_id=req.concept_id,
            timer_seconds=req.timer_seconds,
        )
    except store.UserError as ue:
        raise HTTPException(status_code=400, detail=str(ue))
    except Exception as e:
        logger.error(f"Failed to start active recall session: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start active recall session: {e}")


@router.post("/active-recall/evaluate", response_model=ActiveRecallResult)
@router.post("/active-recall/evaluate/", response_model=ActiveRecallResult)
def post_active_recall_evaluate(
    req: ActiveRecallEvaluateRequest,
    request: Request,
    current_user: User = Depends(get_current_user),
):
    """Evaluate student active recall retrieval text & verification answer using AI Mastery Judge."""
    uid = _resolve_tenant_user_id(current_user, request, req.user_id)
    _require_owned_active_recall_session(req.session_id, uid)
    try:
        return active_recall.evaluate_session(
            user_id=uid,
            session_id=req.session_id,
            concept_id=req.concept_id,
            recall_text=req.recall_text,
            verification_answer=req.verification_answer,
        )
    except PermissionError as pe:
        raise HTTPException(status_code=403, detail=str(pe))
    except HTTPException:
        raise
    except store.UserError as ue:
        raise HTTPException(status_code=400, detail=str(ue))
    except Exception as e:
        logger.error(f"Failed to evaluate active recall session: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to evaluate active recall session: {e}")


@router.get("/active-recall/history", response_model=list[ActiveRecallHistoryItem])
@router.get("/active-recall/history/", response_model=list[ActiveRecallHistoryItem])
def get_active_recall_history(
    request: Request,
    user_id: str = "default",
    current_user: User = Depends(get_current_user),
):
    """Retrieve active recall session history for active learner."""
    uid = _resolve_tenant_user_id(current_user, request, user_id)
    return active_recall.get_history(user_id=uid)


# ── 5. Epistemic Mastery Engine & Learner Dashboard ────────────────────────

@router.get("/dashboard/{user_id}", response_model=LearnerDashboard)
@router.get("/dashboard/{user_id}/", response_model=LearnerDashboard)
def get_user_dashboard(
    user_id: str,
    request: Request,
    current_user: User = Depends(get_current_user),
):
    """Return explainable learner dashboard with UNKNOWN, WEAK, and STRONG concept buckets."""
    uid = _resolve_tenant_user_id(current_user, request, user_id)
    kb = _get_tenant_kb(uid, request)
    if not kb.get("ready") or not kb.get("concepts"):
        return LearnerDashboard(
            user_id=uid,
            strong=[],
            weak=[],
            unknown=[],
            overall_score=0.0,
            total_concepts=0,
            summary="No study materials uploaded yet. Upload a document or URL to begin."
        )
    return mastery.get_dashboard(uid)


@router.get("/mastery")
@router.get("/mastery/")
def get_mastery(
    request: Request,
    user_id: str = "default",
    current_user: User = Depends(get_current_user),
):
    """Return concepts mastery snapshot scoped to the authenticated student."""
    uid = _resolve_tenant_user_id(current_user, request, user_id)
    mats = store.list_study_materials(uid)
    if uid != "default" and uid not in store.USER_KBS and not mats:
        return {"concepts": []}
    kb = _get_tenant_kb(uid, request)
    if not mats and not (kb and kb.get("concepts")):
        return {"concepts": []}
    return {"concepts": mastery.snapshot(uid)}


@router.get("/mastery/dashboard")
@router.get("/mastery/dashboard/")
def get_mastery_dashboard(
    request: Request,
    user_id: str = "default",
    current_user: User = Depends(get_current_user),
):
    """Return dashboard scoped to authenticated student with status_counts and overall_mastery."""
    uid = _resolve_tenant_user_id(current_user, request, user_id)
    mats = store.list_study_materials(uid)
    kb = _empty_kb(uid) if (uid != "default" and uid not in store.USER_KBS and not mats) else _get_tenant_kb(uid, request)
    if not mats and not (kb and kb.get("concepts")):
        return {
            "overall_mastery": 0,
            "concepts": [],
            "status_counts": {"strong": 0, "weak": 0, "unknown": 0},
            "overall_score": 0.0,
            "total_concepts": 0,
            "strong": [],
            "weak": [],
            "unknown": [],
            "summary": "No study materials uploaded yet. Upload a document or URL to begin."
        }
    dash = mastery.get_dashboard(uid)
    snap = mastery.snapshot(uid)
    res = dash.model_dump()
    res["overall_mastery"] = dash.overall_score
    res["concepts"] = snap
    res["status_counts"] = {
        "strong": len(dash.strong),
        "weak": len(dash.weak),
        "unknown": len(dash.unknown)
    }
    return res


@router.post("/mastery/evaluate-evidence", response_model=EvaluateEvidenceResponse)
@router.post("/mastery/evaluate-evidence/", response_model=EvaluateEvidenceResponse)
def post_mastery_evaluate_evidence(
    req: EvaluateEvidenceRequest,
    request: Request,
    current_user: User = Depends(get_current_user),
):
    """Evaluate student learning evidence via 4 specialist agents & Head Teacher Arbiter."""
    uid = _resolve_tenant_user_id(current_user, request, req.user_id)
    req.user_id = uid
    if req.student_explanation and len(req.student_explanation) > 4000:
        req.student_explanation = req.student_explanation[:4000]
    if req.student_answer and len(req.student_answer) > 4000:
        req.student_answer = req.student_answer[:4000]
    decision, findings = mastery_orchestrator.evaluate_concept_learning_evidence(req, user_id=uid)
    return EvaluateEvidenceResponse(decision=decision, specialist_findings=findings)


@router.get("/mastery/{concept_id}/deliberation", response_model=ConceptDeliberationResponse)
@router.get("/mastery/{concept_id}/deliberation/", response_model=ConceptDeliberationResponse)
def get_concept_deliberation(
    concept_id: str,
    request: Request,
    user_id: str = "default",
    current_user: User = Depends(get_current_user),
):
    """Return full explainable deliberation, 4 specialist findings, and evidence items for a concept."""
    uid = _resolve_tenant_user_id(current_user, request, user_id)
    decision = store.get_latest_mastery_decision(uid, concept_id)
    if not decision:
        return ConceptDeliberationResponse(
            concept_id=concept_id,
            user_id=uid,
            decision=None,
            specialist_findings=[],
            evidence_records=[],
        )

    findings_dicts = store.list_specialist_findings(decision["decision_id"], user_id=uid)
    evidence_dicts = store.list_learning_evidence(uid, concept_id)

    dec_obj = HeadTeacherDecision(**decision)
    findings_objs = [SpecialistAgentAssessment(**f) for f in findings_dicts]
    evidence_objs = [LearningEvidenceRecord(**e) for e in evidence_dicts]

    return ConceptDeliberationResponse(
        concept_id=concept_id,
        user_id=uid,
        decision=dec_obj,
        specialist_findings=findings_objs,
        evidence_records=evidence_objs,
    )





@router.get("/diagnostic/baseline")
@router.get("/diagnostic/baseline/")
@router.post("/diagnostic/baseline")
@router.post("/diagnostic/baseline/")
def get_or_post_diagnostic_baseline(
    request: Request,
    user_id: str = "default",
    current_user: User = Depends(get_current_user),
):
    """Generate calibrated 6-question baseline test sampling root and foundational nodes from DAG."""
    uid = _resolve_tenant_user_id(current_user, request, user_id)
    kb = _get_tenant_kb(uid, request)
    if not kb.get("ready") or not kb.get("concepts"):
        raise HTTPException(status_code=400, detail="No study materials indexed yet.")
    try:
        return loop.generate_baseline_diagnostic(user_id=uid)
    except store.UserError as ue:
        raise HTTPException(status_code=400, detail=str(ue))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate baseline diagnostic: {e}")


@router.post("/diagnostic/baseline/evaluate")
@router.post("/diagnostic/baseline/evaluate/")
@router.post("/diagnostic/baseline/submit")
@router.post("/diagnostic/baseline/submit/")
def post_diagnostic_baseline_evaluate(
    body: BaselineSubmissionRequest,
    request: Request,
    current_user: User = Depends(get_current_user),
):
    """Evaluate baseline diagnostic answers and transition learner lifecycle to ACTIVE_LEARNING."""
    uid = _resolve_tenant_user_id(current_user, request, body.user_id)
    try:
        return loop.evaluate_baseline_diagnostic(user_id=uid, submissions=body.submissions)
    except store.UserError as ue:
        raise HTTPException(status_code=400, detail=str(ue))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to evaluate baseline diagnostic: {e}")


@router.post("/mastery/reset")
def post_mastery_reset(
    request: Request,
    user_id: str = "default",
    current_user: User = Depends(get_current_user),
):
    uid = _resolve_tenant_user_id(current_user, request, user_id)
    kb = _get_tenant_kb(uid, request)
    if not kb.get("ready"):
        raise HTTPException(status_code=400, detail="No active study material loaded.")
    mastery.reset(kb["concepts"], user_id=uid)
    return {"ok": True, "concepts": mastery.snapshot(uid)}


# ── 6. Closed-Loop LangGraph Orchestrator ──────────────────────────────────

@router.post("/loop/start")
@router.post("/loop/start/")
def post_loop_start(
    request: Request,
    body: StartSessionReq | None = None,
    current_user: User = Depends(get_current_user),
):
    """Initialize or advance the closed-loop learning session for a student."""
    uid = _resolve_tenant_user_id(current_user, request, body.user_id if body else None)
    if _is_explicit_auth(request) and not _get_tenant_kb(uid, request).get("ready"):
        return {
            "user_id": uid,
            "lifecycle": loop.get_lifecycle_state(uid),
            "action": "upload",
            "concept_id": None,
            "concept_name": None,
            "status": "UNKNOWN",
            "quiz": None,
            "quiz_count": 0,
            "follow_up_question": None,
            "remedial_lesson": None,
            "artifacts": [],
            "external_links": [],
            "reason": "No study materials indexed yet for this account.",
        }
    return loop.start_session(uid)


@router.post("/loop/next-action")
def post_loop_next_action(
    request: Request,
    user_id: str = "default",
    current_user: User = Depends(get_current_user),
):
    uid = _resolve_tenant_user_id(current_user, request, user_id)
    if _is_explicit_auth(request) and not _get_tenant_kb(uid, request).get("ready"):
        return {"user_id": uid, "action": "upload", "quiz": None, "quiz_count": 0}
    return loop.start_session(uid)


@router.post("/recommend")
@router.post("/recommend/")
@router.get("/recommend")
@router.get("/recommend/")
def post_recommend(
    request: Request,
    user_id: str = "default",
    refresh_ai: bool = Query(False),
    current_user: User = Depends(get_current_user),
):
    uid = _resolve_tenant_user_id(current_user, request, user_id)
    if not _get_tenant_kb(uid, request).get("ready"):
        return {"actions": [], "ready": False}
    return recommend.recommend(user_id=uid, refresh_ai=refresh_ai)


@router.get("/dashboard-summary")
@router.get("/dashboard-summary/", include_in_schema=False)
@router.get("/api/dashboard-summary", include_in_schema=False)
@router.get("/api/dashboard-summary/", include_in_schema=False)
def get_dashboard_summary(
    request: Request,
    user_id: str = "default",
    material_id: Optional[str] = None,
    refresh_ai: bool = Query(False),
    current_user: User = Depends(get_current_user),
):
    """Fast-path one-round-trip Dashboard loading: fetches active material metadata, concepts with
    current mastery scores, prerequisite edges, quiz/recall attempt counts, saved notes metadata,
    and fast deterministic next-step recommendations (<10ms) without synchronous LLM calls.
    """
    uid = _resolve_tenant_user_id(current_user, request, user_id)
    summary = store.get_dashboard_summary(user_id=uid, material_id=material_id)

    # Fast deterministic graph + mastery recommendations (<10ms, no OpenRouter LLM call unless refresh_ai=True)
    rec_res = recommend.recommend(user_id=uid, refresh_ai=refresh_ai)
    actions = rec_res.get("actions", [])
    summary["recommendations"] = actions
    summary["top_recommendation"] = actions[0] if actions else None

    # Loop lifecycle state
    kb = _get_tenant_kb(uid, request)
    concepts_list = summary.get("concepts", [])
    if not concepts_list and not (kb and kb.get("ready")):
        summary["loop_status"] = {
            "ok": True,
            "loop_state": "UNINITIALIZED",
            "state": "AWAITING_UPLOAD",
            "message": "Upload your study material to begin.",
            "material_ready": False,
            "concepts_count": 0,
            "baseline_completed": False,
            "total_attempts": 0,
        }
    else:
        total_attempts = sum(c.get("attempts", 0) for c in concepts_list)
        lifecycle = loop.get_lifecycle_state(uid)
        baseline_done = (lifecycle == "ACTIVE_LEARNING" or total_attempts >= 3)
        summary["loop_status"] = {
            "ok": True,
            "loop_state": "ACTIVE_LOOP" if baseline_done else "BASELINE_DIAGNOSTIC",
            "state": "ACTIVE_LEARNING" if baseline_done else lifecycle,
            "message": "Active learning session in progress." if baseline_done else "Baseline diagnostic required.",
            "material_ready": True,
            "concepts_count": len(concepts_list),
            "baseline_completed": baseline_done,
            "total_attempts": total_attempts,
        }

    return summary


# ── Closed-Loop Status, Diagnostic & Active Recall ─────────────────────────


@router.get("/loop/status")
@router.get("/loop/status/", include_in_schema=False)
def get_loop_status(
    request: Request,
    user_id: str = "default",
    current_user: User = Depends(get_current_user),
):
    """Return closed-loop learning state, baseline diagnostic status, and concept count."""
    uid = _resolve_tenant_user_id(current_user, request, user_id)
    kb = _get_tenant_kb(uid, request)
    concepts_list = kb.get("concepts", []) if (kb and kb.get("ready")) else []
    if not concepts_list:
        return {
            "ok": True,
            "loop_state": "UNINITIALIZED",
            "state": "AWAITING_UPLOAD",
            "message": "Upload your study material to begin.",
            "material_ready": False,
            "concepts_count": 0,
            "baseline_completed": False,
            "total_attempts": 0,
        }

    snap = mastery.snapshot(uid)
    total_attempts = sum(c.get("attempts", 0) for c in snap)
    lifecycle = loop.get_lifecycle_state(uid)
    baseline_done = (lifecycle == "ACTIVE_LEARNING" or total_attempts >= 3)
    loop_state = "ACTIVE_LOOP" if baseline_done else "BASELINE_DIAGNOSTIC"
    lifecycle_state = "ACTIVE_LEARNING" if baseline_done else lifecycle

    return {
        "ok": True,
        "loop_state": loop_state,
        "state": lifecycle_state,
        "message": "Active learning session in progress." if baseline_done else "Baseline diagnostic required.",
        "material_ready": True,
        "concepts_count": len(concepts_list),
        "baseline_completed": baseline_done,
        "total_attempts": total_attempts,
    }


@router.post("/diagnostic/start")
@router.post("/diagnostic/start/", include_in_schema=False)
def post_diagnostic_start(
    request: Request,
    body: dict | None = None,
    current_user: User = Depends(get_current_user),
):
    """Generate 6-question baseline diagnostic quiz covering the curriculum to baseline knowledge."""
    req_uid = body.get("user_id") if isinstance(body, dict) else None
    uid = _resolve_tenant_user_id(current_user, request, req_uid)

    kb = _get_tenant_kb(uid, request)
    concepts_list = kb.get("concepts", []) if (kb and kb.get("ready")) else []
    if not concepts_list:
        raise HTTPException(status_code=400, detail="No study materials indexed yet.")

    path = concepts.get_learning_path(unmastered_only=False, user_id=uid) if hasattr(concepts, "get_learning_path") else [c["id"] for c in concepts_list]
    target_cids = list(path[:6])
    while len(target_cids) < 6 and concepts_list:
        target_cids.append(concepts_list[len(target_cids) % len(concepts_list)]["id"])

    questions = quiz.generate_full_diagnostic_suite(
        target_cids=target_cids,
        user_id=uid,
        count=6,
        kb=kb
    )

    if not questions:
        raise HTTPException(
            status_code=400,
            detail="Unable to generate diagnostic questions from the active study material."
        )

    return {
        "ok": True,
        "questions": questions,
        "count": len(questions),
    }


@router.post("/diagnostic/submit")
@router.post("/diagnostic/submit/", include_in_schema=False)
def post_diagnostic_submit(
    body: DiagnosticSubmitReq,
    request: Request,
    current_user: User = Depends(get_current_user),
):
    """Grade baseline diagnostic submissions, update epistemic beliefs, and unlock active loop."""
    uid = _resolve_tenant_user_id(current_user, request, body.user_id)
    results = []
    correct_count = 0

    for a in body.answers:
        qid = a.get("question_id", a.get("id", "diag_sub"))
        _require_owned_question(qid, uid, request)
        sub = QuizSubmission(
            user_id=uid,
            question_id=qid,
            concept_id=a.get("concept_id", a.get("conceptId", "")),
            selected_option_idx=a.get("selected_option_idx", a.get("selectedIndex")),
            typed_answer=a.get("typed_answer", a.get("typedAnswer", "")),
            student_reasoning=a.get("student_reasoning", a.get("studentReasoning", "")),
            numeric_answer=a.get("numeric_answer", a.get("numericAnswer")),
        )
        try:
            res = loop.submit_quiz_answer(sub)
            results.append(res)
            if res.get("is_correct"):
                correct_count += 1
        except Exception:
            pass

    total = max(len(body.answers), 1)
    score_pct = round((correct_count / total) * 100)
    return {
        "ok": True,
        "score": score_pct,
        "correct_count": correct_count,
        "total": len(body.answers),
        "baseline_completed": True,
        "results": results,
    }


# ── 7. Universal Multi-Format Study Materials & Interactive Preview ──────────

@router.post("/materials/ingest-file")
@router.post("/materials/ingest-file/")
async def post_materials_ingest_file(
    request: Request,
    files: list[UploadFile] = File(default=[]),
    file: UploadFile | None = File(default=None),
    user_id: str = Form("default"),
    title: str | None = Form(None),
    current_user: User = Depends(get_current_user),
):
    """Ingest multi-format study material (.pdf, .pptx, .md, .txt, .docx, .mp3, .wav, .m4a, .webm)."""
    uid = _resolve_tenant_user_id(current_user, request, user_id)
    all_uploads: list[UploadFile] = list(files or [])
    if file:
        all_uploads.append(file)

    if not all_uploads:
        raise HTTPException(status_code=400, detail="No study material file provided.")

    allowed_exts = {".pdf", ".pptx", ".ppt", ".md", ".txt", ".docx", ".mp3", ".wav", ".m4a", ".webm"}
    ingested_materials = []
    safe_uid_dir = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in uid).strip("_") or "default"
    tenant_upload_dir = config.UPLOAD_DIR / safe_uid_dir

    for f in all_uploads:
        if not f.filename:
            continue
        ext = Path(f.filename).suffix.lower()
        if ext not in allowed_exts:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported format '{ext}'. Supported: PDF, PPTX, MD, TXT, DOCX, MP3, WAV, M4A, WEBM."
            )
        dest = tenant_upload_dir / Path(f.filename).name
        content = await f.read()
        await run_in_threadpool(_write_uploaded_bytes, dest, content)

        try:
            mat_res = await run_in_threadpool(
                UniversalMaterialIngestor.ingest_file,
                path=dest,
                user_id=uid,
                title=title or Path(f.filename).stem.replace("_", " ").title(),
            )
            ingested_materials.append(mat_res)
        except store.UserError as ue:
            raise HTTPException(status_code=400, detail=str(ue))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to ingest file '{f.filename}': {e}")

    active_kb = await run_in_threadpool(_get_tenant_kb, uid, request)
    return {
        "ok": True,
        "count": len(ingested_materials),
        "materials": ingested_materials,
        "active_kb": {
            "ready": active_kb.get("ready", False),
            "title": active_kb.get("file", ""),
            "pages": active_kb.get("pages", 0),
            "concepts": len(active_kb.get("concepts", [])),
        }
    }


@router.post("/materials/ingest-url")
@router.post("/materials/ingest-url/")
async def post_materials_ingest_url(
    req: IngestUrlRequest,
    request: Request,
    current_user: User = Depends(get_current_user),
):
    """Ingest educational websites (GeeksforGeeks, docs) or YouTube video lectures with timestamped transcripts."""
    uid = _resolve_tenant_user_id(current_user, request, req.user_id)
    url = req.url.strip()
    if not (url.startswith("http://") or url.startswith("https://")):
        raise HTTPException(status_code=400, detail="URL must begin with http:// or https://")

    try:
        mat_res = await run_in_threadpool(
            UniversalMaterialIngestor.ingest_url,
            url=url,
            source_type=req.source_type,
            user_id=uid,
            title=req.title,
        )
    except store.UserError as ue:
        raise HTTPException(status_code=400, detail=str(ue))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to ingest URL '{url}': {e}")

    active_kb = await run_in_threadpool(_get_tenant_kb, uid, request)
    return {
        "ok": True,
        "material": mat_res,
        "active_kb": {
            "ready": active_kb.get("ready", False),
            "title": active_kb.get("file", ""),
            "pages": active_kb.get("pages", 0),
            "concepts": len(active_kb.get("concepts", [])),
        }
    }


@router.get("/materials", response_model=MaterialsListResponse)
@router.get("/materials/", response_model=MaterialsListResponse)
def get_study_materials(
    request: Request,
    user_id: str = "default",
    current_user: User = Depends(get_current_user),
):
    """List all ingested study materials for the learner with metadata badges."""
    uid = _resolve_tenant_user_id(current_user, request, user_id)
    materials = store.list_study_materials(user_id=uid)
    models = [StudyMaterialMetadata(**m) for m in materials]
    return MaterialsListResponse(
        materials=models,
        total_materials=len(models),
        total_chunks=sum(m.chunk_count for m in models),
        total_concepts=sum(m.concept_count for m in models),
    )


@router.get("/materials/{material_id}/preview", response_model=StudyMaterialPreview)
@router.get("/materials/{material_id}/preview/", response_model=StudyMaterialPreview)
def get_study_material_preview(
    material_id: str,
    request: Request,
    user_id: Optional[str] = Query(None),
    current_user: User = Depends(get_current_user),
):
    """Return structured preview payload (pages, slides, timestamps, or reader view) for interactive inspection."""
    uid = _resolve_tenant_user_id(current_user, request, user_id)
    mat = _require_owned_material(material_id, uid, request)
    return StudyMaterialPreview(**mat)


@router.delete("/materials/{material_id}")
@router.delete("/materials/{material_id}/")
def delete_study_material(
    material_id: str,
    request: Request,
    user_id: str = "default",
    current_user: User = Depends(get_current_user),
):
    """Delete a study material and automatically rebuild the unified index and Prerequisite DAG."""
    uid = _resolve_tenant_user_id(current_user, request, user_id)
    _require_owned_material(material_id, uid, request)
    try:
        res = UniversalMaterialIngestor.delete_material(material_id=material_id, user_id=uid)
        return {
            "ok": True,
            "deleted_material_id": material_id,
            "rebuilt_kb": res
        }
    except store.UserError as ue:
        raise HTTPException(status_code=404, detail=str(ue))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete material: {e}")


# ── Multimodal Artifacts & Smart Study Notes ─────────────────────────────────

@router.get("/artifacts/{filename}")
def get_generated_artifact(filename: str, download: Optional[bool] = None):
    """Serve generated .pdf, .pptx, .svg, and .mp3 files from config.GENERATED_ARTIFACTS_DIR."""
    safe_name = Path(filename).name
    file_path = (config.GENERATED_ARTIFACTS_DIR / safe_name).resolve()
    base_dir = config.GENERATED_ARTIFACTS_DIR.resolve()

    if not file_path.is_file() or not str(file_path).startswith(str(base_dir)):
        raise HTTPException(status_code=404, detail="Artifact not found")

    suffix = file_path.suffix.lower()
    media_types = {
        ".svg": "image/svg+xml",
        ".pdf": "application/pdf",
        ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        ".mp3": "audio/mpeg",
        ".wav": "audio/wav",
    }
    media_type = media_types.get(suffix, "application/octet-stream")
    is_attachment = download is True or suffix in [".pdf", ".pptx"]
    raw_bytes = file_path.read_bytes()

    if len(raw_bytes) == 0:
        raise HTTPException(status_code=503, detail="Artifact file is empty")
    if suffix == ".mp3" and len(raw_bytes) <= 100:
        raise HTTPException(status_code=503, detail="Audio artifact contains invalid or truncated stream")
    if suffix == ".wav" and len(raw_bytes) <= 44:
        raise HTTPException(status_code=503, detail="Audio artifact contains invalid or truncated stream")

    headers = {
        "Content-Type": media_type,
        "Content-Length": str(len(raw_bytes)),
        "Accept-Ranges": "bytes",
        "Content-Disposition": (
            f'attachment; filename="{safe_name}"'
            if is_attachment
            else f'inline; filename="{safe_name}"'
        ),
    }
    return Response(
        content=raw_bytes,
        media_type=media_type,
        headers=headers,
    )


@router.post("/materials/{material_id}/deep-audio")
@router.post("/materials/{material_id}/deep-audio/")
async def post_material_deep_audio(
    material_id: str,
    request: Request,
    req: Optional[DeepAudioRequest] = None,
    current_user: User = Depends(get_current_user),
):
    """Generate professor-style deep spoken explanation for a PDF or PPTX slide."""
    uid = _resolve_tenant_user_id(current_user, request, req.user_id if req else None)
    section_index = req.section_or_slide_index if req else 0
    mat = await run_in_threadpool(_require_owned_material, material_id, uid, request)

    try:
        if getattr(tts.generate_deep_material_audio, "_is_default_sync_deep_audio", False):
            res = await tts.generate_deep_material_audio_async(
                material=mat,
                user_id=uid,
                section_or_slide_index=section_index,
            )
        else:
            res = await run_in_threadpool(
                tts.generate_deep_material_audio,
                material=mat,
                user_id=uid,
                section_or_slide_index=section_index,
            )
        if res and isinstance(res, dict) and res.get("audio_url"):
            filename = Path(res["audio_url"]).name
            audio_file = config.GENERATED_ARTIFACTS_DIR / filename
            if audio_file.exists() and audio_file.stat().st_size == 0:
                raise HTTPException(
                    status_code=503,
                    detail="Deep audio synthesis produced empty audio stream"
                )
        return res
    except store.UserError as ue:
        raise HTTPException(status_code=400, detail=str(ue))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Failed to generate deep audio: {e}")


@router.post("/notes/generate", response_model=SmartNoteResponse)
@router.post("/notes/generate/", response_model=SmartNoteResponse)
def post_notes_generate(
    req: SmartNoteRequest,
    request: Request,
    current_user: User = Depends(get_current_user),
):
    """Generate smart study notes adhering strictly to task-type specifications."""
    uid = _resolve_tenant_user_id(current_user, request, req.user_id)
    req.user_id = uid

    try:
        res = notes.generate_smart_note(req)
        return res
    except store.UserError as ue:
        raise HTTPException(status_code=400, detail=str(ue))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to generate smart note: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate smart note: {e}")


@router.post("/notes/{note_id}/audio")
@router.post("/notes/{note_id}/audio/")
async def post_note_audio(
    note_id: str,
    request: Request,
    user_id: Optional[str] = Query(None),
    current_user: User = Depends(get_current_user),
):
    """Synthesize high-yield spoken audio executive summary using ElevenLabs Flash."""
    uid = _resolve_tenant_user_id(current_user, request, user_id)
    await run_in_threadpool(_require_owned_note, note_id, uid)
    try:
        res = await run_in_threadpool(notes.generate_note_audio, note_id=note_id, user_id=uid)
        if res and isinstance(res, dict) and res.get("audio_url"):
            filename = Path(res["audio_url"]).name
            audio_file = config.GENERATED_ARTIFACTS_DIR / filename
            if audio_file.exists() and audio_file.stat().st_size == 0:
                raise HTTPException(
                    status_code=503,
                    detail="Audio synthesis service unavailable: generated audio stream is empty"
                )
        return res
    except store.UserError as ue:
        err_msg = str(ue)
        status = 404 if "not found" in err_msg.lower() else 400
        raise HTTPException(status_code=status, detail=err_msg)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Failed to generate note audio: {e}")


@router.post("/notes/{note_id}/export")
@router.post("/notes/{note_id}/export/")
@router.get("/notes/{note_id}/export")
@router.get("/notes/{note_id}/export/")
def post_note_export(
    note_id: str,
    request: Request,
    export_format: Optional[str] = Query(None),
    payload: Optional[dict] = Body(None),
    download: Optional[bool] = Query(None),
    user_id: Optional[str] = Query(None),
    current_user: User = Depends(get_current_user),
):
    """Export a smart note as a PDF study handout or PPTX slide deck."""
    fmt = (export_format or (payload.get("export_format") if isinstance(payload, dict) else None) or "pdf").lower().strip()
    body_uid = payload.get("user_id") if isinstance(payload, dict) else None
    uid = _resolve_tenant_user_id(current_user, request, body_uid, user_id)
    _require_owned_note(note_id, uid)
    try:
        # If client passed full markdown content in payload, update and ensure expansion
        stored_note = store.get_smart_note(note_id, user_id=uid)
        if isinstance(payload, dict) and payload.get("markdown_content"):
            incoming_md = str(payload["markdown_content"]).strip()
            if incoming_md and stored_note:
                stored_note["markdown_content"] = incoming_md
                if payload.get("title"):
                    stored_note["title"] = str(payload["title"]).strip()
                if len(incoming_md.split()) < 300:
                    stored_note["markdown_content"] = notes.expand_note_content(stored_note, user_id=uid)
                store.save_smart_note(stored_note)
        elif stored_note:
            raw_md = stored_note.get("markdown_content") or ""
            if len(raw_md.split()) < 300:
                stored_note["markdown_content"] = notes.expand_note_content(stored_note, user_id=uid)
                store.save_smart_note(stored_note)

        res = notes.export_note_artifact(note_id=note_id, user_id=uid, export_format=fmt)
        wants_download = download is True or (isinstance(payload, dict) and payload.get("download") is True)
        accept_header = (request.headers.get("accept") or "").lower()
        if wants_download or (fmt == "pdf" and "application/pdf" in accept_header) or (fmt == "pptx" and "presentationml" in accept_header):
            target_url = res.get("download_url") or res.get("url") or res.get(f"{fmt}_url") or ""
            safe_filename = Path(target_url).name
            file_path = (config.GENERATED_ARTIFACTS_DIR / safe_filename).resolve()
            base_dir = config.GENERATED_ARTIFACTS_DIR.resolve()
            if safe_filename and file_path.is_file() and str(file_path).startswith(str(base_dir)):
                raw_bytes = file_path.read_bytes()
                media_type = (
                    "application/pdf"
                    if fmt == "pdf"
                    else "application/vnd.openxmlformats-officedocument.presentationml.presentation"
                )
                return Response(
                    content=raw_bytes,
                    media_type=media_type,
                    headers={
                        "Content-Disposition": f'attachment; filename="{safe_filename}"',
                        "Content-Length": str(len(raw_bytes)),
                    },
                )
        return res
    except store.UserError as ue:
        err_msg = str(ue)
        status = 404 if "not found" in err_msg.lower() else 400
        raise HTTPException(status_code=status, detail=err_msg)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to export note artifact: {e}")


@router.get("/notes")
@router.get("/notes/")
def get_notes(
    request: Request,
    concept_id: Optional[str] = Query(None),
    user_id: Optional[str] = Query(None),
    current_user: User = Depends(get_current_user),
):
    """Retrieve saved smart study notes for the active learner."""
    uid = _resolve_tenant_user_id(current_user, request, user_id)
    return store.list_smart_notes(user_id=uid, concept_id=concept_id)


@router.delete("/notes/{note_id}")
@router.delete("/notes/{note_id}/")
def delete_smart_note(
    note_id: str,
    request: Request,
    user_id: Optional[str] = Query(None),
    current_user: User = Depends(get_current_user),
):
    """Delete a saved smart note from the archive."""
    uid = _resolve_tenant_user_id(current_user, request, user_id)
    _require_owned_note(note_id, uid)
    deleted = store.delete_smart_note(note_id=note_id, user_id=uid)
    return {"ok": True, "deleted": deleted, "note_id": note_id}


@router.post("/tools/generate")
@router.post("/tools/generate/")
def post_tools_generate(
    req: ToolGenerateRequest,
    request: Request,
    current_user: User = Depends(get_current_user),
):
    """Invoke generative study tools (visual SVG diagram, PPTX deck, PDF handout, curated resources)."""
    uid = _resolve_tenant_user_id(current_user, request, req.user_id)
    tool_type = req.tool_type.lower().strip()
    try:
        if tool_type == "visual":
            return tools.generate_visual_diagram_tool(
                concept_id=req.concept_id,
                prompt=req.prompt or "",
                user_id=uid
            )
        elif tool_type == "pptx":
            return tools.generate_pptx_deck_tool(
                concept_id=req.concept_id,
                user_id=uid,
                slide_count=req.slide_count or 8
            )
        elif tool_type == "pdf":
            return tools.generate_pdf_handout_tool(
                concept_id=req.concept_id,
                user_id=uid
            )
        elif tool_type == "resources":
            return tools.suggest_curated_resources_tool(
                concept_id=req.concept_id,
                query=req.prompt or "",
                user_id=uid
            )
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported tool type '{tool_type}'. Must be 'visual', 'pptx', 'pdf', or 'resources'.")
    except store.UserError as ue:
        raise HTTPException(status_code=400, detail=str(ue))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Tool generation failed: {e}")


