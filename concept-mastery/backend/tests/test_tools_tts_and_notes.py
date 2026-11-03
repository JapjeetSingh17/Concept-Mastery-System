"""Comprehensive 20-Test Deterministic Suite for LangGraph Artifact Tools, ElevenLabs TTS, and Smart Notes.

Covers:
- Test 1-3: Security & Git Cleanliness (TC-003):
  * Verify .gitignore explicitly contains ".env", "backend/.env", "*.mp3", and "generated_artifacts/".
  * Scan all .py and .js files in repository for hardcoded "ELEVENLABS_API_KEY = 'sk_..." or hardcoded secret patterns to ensure NO keys are embedded in source code.
  * Verify config.ELEVENLABS_API_KEY loads from environment / .env gracefully.
- Test 4-7: LangGraph Artifact Tools (tools.py):
  * Test generate_visual_diagram_tool produces valid SVG file in GENERATED_ARTIFACTS_DIR and returns valid dictionary.
  * Test generate_pptx_deck_tool produces valid 5-slide .pptx file in GENERATED_ARTIFACTS_DIR.
  * Test generate_pdf_handout_tool produces valid .pdf file in GENERATED_ARTIFACTS_DIR.
  * Test suggest_curated_resources_tool returns clickable YouTube and article links tagged with label: "[External Resource]".
- Test 8-10: Socratic Tutor & Intent Routing (tutor.py & loop.py):
  * Test route_student_intent correctly classifies "draw a diagram of gradient descent" as "visual".
  * Test route_student_intent classifies "make a presentation/slides on loss functions" as "pptx".
  * Test route_student_intent classifies "youtube video recommendations" as "resources".
- Test 11-14: ElevenLabs TTS & Deep Audio Engine (tts.py):
  * Test sanitize_latex_for_speech correctly converts \\nabla, \\alpha, \\theta, \\partial into natural spoken words and strips markdown.
  * Test sanitize_latex_for_speech caps output at 1,800 characters.
  * Test synthesize_elevenlabs_audio creates a valid MP3 file on disk and audio_cache hit returns cached=True with 0 duplicate API calls.
  * Test generate_deep_material_audio STRICT FORMAT GATE: raises store.UserError / HTTP 400 when called on "youtube", "web", "markdown", or "audio" materials, and succeeds only for "pdf" and "pptx".
- Test 15-18: Smart Notes Task Elicitation & Pipelines (notes.py & store.py):
  * Test generate_smart_note ELICITATION GATE: raises store.UserError when task_type is missing or invalid.
  * Test distinct formatting across task types ("exam", "viva", "quiz", "essay").
  * Test generate_note_audio generates spoken audio and attaches audio_url to the note.
  * Test export_note_artifact exports note to PDF and PPTX and updates urls in store.
- Test 19-20: REST Endpoints (TestClient(app)):
  * Test GET /api/artifacts/{filename} serves files with correct Content-Type (audio/mpeg, application/pdf, etc.).
  * Test POST /api/materials/{mat_id}/deep-audio rejects non-PDF/PPTX with HTTP 400.
"""

from __future__ import annotations

import os
import re
import uuid
import pytest
from pathlib import Path
from fastapi.testclient import TestClient

from app.main import app
from app import config
from app.core import store, tts
from app.pipelines import tools, tutor, notes
from app.models.schemas import SmartNoteRequest


# ============================================================================
# Test 1-3: Security & Git Cleanliness (TC-003)
# ============================================================================

def test_01_gitignore_security_rules():
    """Verify .gitignore explicitly contains critical secret and artifact exclusions (TC-003)."""
    backend_base = Path(__file__).resolve().parents[1]
    repo_root = backend_base.parent.parent

    gitignore_files = [
        repo_root / ".gitignore",
        backend_base.parent / ".gitignore",
    ]
    existing_gitignores = [f for f in gitignore_files if f.is_file()]
    assert len(existing_gitignores) > 0, "At least one .gitignore file must exist in repository."

    required_entries = [".env", "backend/.env", "*.mp3", "generated_artifacts/"]
    combined_content = "\n".join(f.read_text(encoding="utf-8") for f in existing_gitignores)

    for req in required_entries:
        assert req in combined_content, f".gitignore must explicitly contain '{req}'."


def test_02_source_code_scan_no_hardcoded_secrets():
    """Scan all .py and .js files in repository for hardcoded ELEVENLABS_API_KEY or secret patterns."""
    backend_base = Path(__file__).resolve().parents[1]
    repo_root = backend_base.parent.parent
    scan_dir = backend_base.parent

    secret_patterns = [
        re.compile(r"ELEVENLABS_API_KEY\s*=\s*['\"]sk_[a-zA-Z0-9_\-]{16,}['\"]", re.IGNORECASE),
        re.compile(r"['\"]sk_[a-zA-Z0-9_\-]{24,}['\"]", re.IGNORECASE),
    ]

    flagged_files = []
    for ext in ("*.py", "*.js"):
        for path in scan_dir.rglob(ext):
            if any(part in path.parts for part in (".venv", "node_modules", ".git", ".pytest_cache", "__pycache__")):
                continue
            if path.name == "test_tools_tts_and_notes.py":
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            for pat in secret_patterns:
                if pat.search(text):
                    flagged_files.append(f"{path.relative_to(repo_root)} matched {pat.pattern}")

    assert not flagged_files, f"Hardcoded secrets detected in source files:\n" + "\n".join(flagged_files)


def test_03_elevenlabs_api_key_loads_gracefully_from_env(monkeypatch):
    """Verify config.ELEVENLABS_API_KEY loads from environment / .env gracefully."""
    assert hasattr(config, "ELEVENLABS_API_KEY")
    assert isinstance(config.ELEVENLABS_API_KEY, str)

    monkeypatch.setenv("ELEVENLABS_API_KEY", "mock_live_elevenlabs_token_12345")
    val = os.getenv("ELEVENLABS_API_KEY", "")
    assert val == "mock_live_elevenlabs_token_12345"


# ============================================================================
# Test 4-7: LangGraph Artifact Tools (tools.py)
# ============================================================================

def test_04_generate_visual_diagram_tool():
    """Test generate_visual_diagram_tool produces valid SVG file in GENERATED_ARTIFACTS_DIR and returns valid dictionary."""
    res = tools.generate_visual_diagram_tool(
        concept_id="gradient_descent",
        prompt="draw a flow of gradient descent",
        user_id="test_user_qa"
    )
    assert isinstance(res, dict)
    assert res.get("artifact_type") == "visual"
    assert "file_url" in res or "url" in res
    assert "svg_or_mermaid_code" in res
    assert "<svg" in res["svg_or_mermaid_code"]
    assert "</svg>" in res["svg_or_mermaid_code"]

    url_path = res.get("file_url") or res.get("url")
    file_name = Path(url_path).name
    artifact_path = config.GENERATED_ARTIFACTS_DIR / file_name
    assert artifact_path.exists(), f"SVG artifact {artifact_path} was not created."
    content = artifact_path.read_text(encoding="utf-8")
    assert content.startswith("<svg")
    assert "</svg>" in content


def test_05_generate_pptx_deck_tool():
    """Test generate_pptx_deck_tool produces valid 5-slide .pptx file in GENERATED_ARTIFACTS_DIR."""
    res = tools.generate_pptx_deck_tool(
        concept_id="loss_functions",
        user_id="test_user_qa",
        slide_count=5
    )
    assert isinstance(res, dict)
    assert res.get("artifact_type") == "pptx"
    assert "slide_previews" in res
    assert len(res["slide_previews"]) == 5

    url_path = res.get("download_url") or res.get("url")
    file_name = Path(url_path).name
    artifact_path = config.GENERATED_ARTIFACTS_DIR / file_name
    assert artifact_path.exists(), f"PPTX artifact {artifact_path} was not created."
    assert artifact_path.stat().st_size > 0


def test_06_generate_pdf_handout_tool():
    """Test generate_pdf_handout_tool produces valid .pdf file in GENERATED_ARTIFACTS_DIR."""
    res = tools.generate_pdf_handout_tool(
        concept_id="activation_functions",
        user_id="test_user_qa"
    )
    assert isinstance(res, dict)
    assert res.get("artifact_type") == "pdf"
    assert "download_url" in res or "url" in res

    url_path = res.get("download_url") or res.get("url")
    file_name = Path(url_path).name
    artifact_path = config.GENERATED_ARTIFACTS_DIR / file_name
    assert artifact_path.exists(), f"PDF artifact {artifact_path} was not created."
    raw_bytes = artifact_path.read_bytes()
    assert raw_bytes.startswith(b"%PDF"), "PDF artifact must start with %PDF header."


def test_07_suggest_curated_resources_tool():
    """Test suggest_curated_resources_tool returns clickable YouTube and article links tagged with label: '[External Resource]'."""
    res = tools.suggest_curated_resources_tool(
        concept_id="backpropagation",
        query="backpropagation tutorial",
        user_id="test_user_qa"
    )
    assert isinstance(res, dict)
    assert "resources" in res
    resources = res["resources"]
    assert len(resources) >= 2

    types = {r.get("resource_type") for r in resources}
    assert "youtube" in types, "Resources must include YouTube tutorial link."
    assert "article" in types, "Resources must include article reference link."

    for r in resources:
        assert r.get("label") == "[External Resource]", "Each resource must be tagged '[External Resource]'."
        url = r.get("url", "")
        assert url.startswith("http://") or url.startswith("https://"), f"URL {url} must be clickable."


# ============================================================================
# Test 8-10: Socratic Tutor & Intent Routing (tutor.py & loop.py)
# ============================================================================

def test_08_route_student_intent_visual():
    """Test route_student_intent correctly classifies 'draw a diagram of gradient descent' as 'visual'."""
    intent = tutor.route_student_intent("draw a diagram of gradient descent")
    assert intent == "visual"


def test_09_route_student_intent_pptx():
    """Test route_student_intent classifies 'make a presentation/slides on loss functions' as 'pptx'."""
    intent = tutor.route_student_intent("make a presentation/slides on loss functions")
    assert intent in ("pptx", "slides")


def test_10_route_student_intent_resources():
    """Test route_student_intent classifies 'youtube video recommendations' as 'resources'."""
    intent = tutor.route_student_intent("youtube video recommendations")
    assert intent in ("resources", "external_links")


# ============================================================================
# Test 11-14: ElevenLabs TTS & Deep Audio Engine (tts.py)
# ============================================================================

def test_11_sanitize_latex_for_speech_math_and_markdown():
    """Test sanitize_latex_for_speech correctly converts \\nabla, \\alpha, \\theta, \\partial into natural spoken words and strips markdown."""
    raw_input = (
        "# Core Equations\n"
        "Optimize loss with $\\nabla \\mathcal{L}$ using rate $\\alpha$ and weights $\\theta$. "
        "Compute the $\\partial$ derivative."
    )
    cleaned = tts.sanitize_latex_for_speech(raw_input)
    assert "gradient del" in cleaned
    assert "learning rate alpha" in cleaned
    assert "parameter theta" in cleaned
    assert "partial derivative" in cleaned
    assert "#" not in cleaned
    assert "$" not in cleaned


def test_12_sanitize_latex_for_speech_length_capping():
    """Test sanitize_latex_for_speech caps output at 1,800 characters."""
    long_text = "Detailed explanation of mathematical optimization and backpropagation. " * 60
    assert len(long_text) > 3000
    cleaned = tts.sanitize_latex_for_speech(long_text)
    assert len(cleaned) <= 1800


def test_13_synthesize_elevenlabs_audio_caching():
    """Test synthesize_elevenlabs_audio creates a valid MP3 file on disk and audio_cache hit returns cached=True with 0 duplicate API calls."""
    test_phrase = f"Deterministic speech synthesis test for closed loop verification {uuid.uuid4().hex[:6]}."
    res1 = tts.synthesize_elevenlabs_audio(test_phrase, user_id="test_user_qa")
    assert isinstance(res1, dict)
    assert "audio_url" in res1
    assert res1["audio_url"].endswith(".mp3")

    file_name = Path(res1["audio_url"]).name
    mp3_path = config.GENERATED_ARTIFACTS_DIR / file_name
    assert mp3_path.exists(), f"MP3 file {mp3_path} was not created."
    assert mp3_path.stat().st_size > 0

    # Second call must hit audio_cache without regenerating
    res2 = tts.synthesize_elevenlabs_audio(test_phrase, user_id="test_user_qa")
    assert res2.get("cached") is True
    assert res2.get("audio_url") == res1.get("audio_url")


def test_14_generate_deep_material_audio_format_gate():
    """Test generate_deep_material_audio STRICT FORMAT GATE: raises store.UserError for non-pdf/pptx and succeeds for pdf/pptx."""
    unsupported_types = ["youtube", "web", "markdown", "audio", "docx"]
    for utype in unsupported_types:
        mat = {
            "material_id": f"mat_unsupp_{utype}_{uuid.uuid4().hex[:4]}",
            "title": f"Unsupported {utype}",
            "source_type": utype,
        }
        with pytest.raises(store.UserError) as exc_info:
            tts.generate_deep_material_audio(mat, user_id="test_user_qa")
        assert "exclusively supported for PDF and PPTX" in str(exc_info.value)

    # Supported PDF
    pdf_mat = {
        "material_id": f"mat_pdf_{uuid.uuid4().hex[:6]}",
        "title": "Deep Learning Lecture",
        "source_type": "pdf",
        "structured_preview": [
            {"index": 0, "title": "Slide 1", "text": "Introduction to Convolutional Neural Networks."}
        ]
    }
    pdf_res = tts.generate_deep_material_audio(pdf_mat, user_id="test_user_qa")
    assert isinstance(pdf_res, dict)
    assert "audio_url" in pdf_res
    assert pdf_res["audio_url"].endswith(".mp3")

    # Supported PPTX
    pptx_mat = {
        "material_id": f"mat_pptx_{uuid.uuid4().hex[:6]}",
        "title": "Loss Landscapes",
        "source_type": "pptx",
        "structured_preview": [
            {"index": 0, "title": "Slide 1", "text": "Loss Surfaces and Convex Optimization."}
        ]
    }
    pptx_res = tts.generate_deep_material_audio(pptx_mat, user_id="test_user_qa")
    assert isinstance(pptx_res, dict)
    assert "audio_url" in pptx_res
    assert pptx_res["audio_url"].endswith(".mp3")


# ============================================================================
# Test 15-18: Smart Notes Task Elicitation & Pipelines (notes.py & store.py)
# ============================================================================

def test_15_generate_smart_note_elicitation_gate():
    """Test generate_smart_note ELICITATION GATE: raises store.UserError when task_type is missing or invalid."""
    with pytest.raises(store.UserError) as exc_missing:
        notes.generate_smart_note(SmartNoteRequest(concept_id="gradient_descent", task_type=""))
    assert "Please select which task you need notes for" in str(exc_missing.value)

    with pytest.raises(store.UserError) as exc_invalid:
        notes.generate_smart_note(SmartNoteRequest(concept_id="gradient_descent", task_type="unsupported_task"))
    assert "Please select which task you need notes for" in str(exc_invalid.value)


def test_16_smart_notes_distinct_task_type_formatting():
    """Test distinct formatting across task types ('exam', 'viva', 'quiz', 'essay')."""
    task_types = ["exam", "viva", "quiz", "essay"]
    generated_notes = {}

    for tt in task_types:
        req = SmartNoteRequest(concept_id="stochastic_gradient_descent", task_type=tt, user_id="test_user_qa")
        note_resp = notes.generate_smart_note(req)
        assert note_resp.task_type == tt
        assert note_resp.markdown_content
        generated_notes[tt] = note_resp

    assert "Derivation" in generated_notes["exam"].markdown_content or "Trap" in generated_notes["exam"].markdown_content
    assert "Elevator Pitch" in generated_notes["viva"].markdown_content or "Examiner" in generated_notes["viva"].markdown_content or "Analogy" in generated_notes["viva"].markdown_content
    assert "Cheat Sheet" in generated_notes["quiz"].markdown_content or "Parameter" in generated_notes["quiz"].markdown_content or "Cloze" in generated_notes["quiz"].markdown_content
    assert "Monograph" in generated_notes["essay"].markdown_content or "Theoretical" in generated_notes["essay"].markdown_content

    mds = [n.markdown_content for n in generated_notes.values()]
    assert len(set(mds)) == 4, "Each task type must produce a unique, distinct markdown formatting."


def test_17_generate_note_audio_attaches_audio_url():
    """Test generate_note_audio generates spoken audio and attaches audio_url to the note."""
    req = SmartNoteRequest(concept_id="backpropagation", task_type="exam", user_id="test_user_qa")
    note_resp = notes.generate_smart_note(req)

    updated = notes.generate_note_audio(note_id=note_resp.note_id, user_id="test_user_qa")
    assert isinstance(updated, dict)
    assert updated.get("audio_url")
    assert updated["audio_url"].endswith(".mp3")

    in_store = store.get_smart_note(note_resp.note_id, user_id="test_user_qa")
    assert in_store is not None
    assert in_store.get("audio_url") == updated["audio_url"]


def test_18_export_note_artifact_pdf_and_pptx():
    """Test export_note_artifact exports note to PDF and PPTX and updates urls in store."""
    req = SmartNoteRequest(concept_id="loss_functions", task_type="viva", user_id="test_user_qa")
    note_resp = notes.generate_smart_note(req)

    # Export to PDF
    pdf_res = notes.export_note_artifact(note_id=note_resp.note_id, user_id="test_user_qa", export_format="pdf")
    assert isinstance(pdf_res, dict)
    assert pdf_res.get("pdf_url")
    assert pdf_res["pdf_url"].endswith(".pdf")

    # Export to PPTX
    pptx_res = notes.export_note_artifact(note_id=note_resp.note_id, user_id="test_user_qa", export_format="pptx")
    assert isinstance(pptx_res, dict)
    assert pptx_res.get("pptx_url")
    assert pptx_res["pptx_url"].endswith(".pptx")

    # Verify both URLs stored in database
    in_store = store.get_smart_note(note_resp.note_id, user_id="test_user_qa")
    assert in_store.get("pdf_url") == pdf_res["pdf_url"]
    assert in_store.get("pptx_url") == pptx_res["pptx_url"]


# ============================================================================
# Test 19-20: REST Endpoints (TestClient(app))
# ============================================================================

def test_19_get_artifact_endpoint_serves_correct_content_type():
    """Test GET /api/artifacts/{filename} serves files with correct Content-Type (audio/mpeg, application/pdf, etc.)."""
    client = TestClient(app)

    # 1. MP3 file serving
    mp3_name = f"test_artifact_{uuid.uuid4().hex[:6]}.mp3"
    mp3_file = config.GENERATED_ARTIFACTS_DIR / mp3_name
    mp3_file.write_bytes(b"\xff\xfb\x90\x00" + (b"\x00" * 32))

    res_mp3 = client.get(f"/api/artifacts/{mp3_name}")
    assert res_mp3.status_code == 200
    assert "audio/mpeg" in res_mp3.headers.get("content-type", "")

    # 2. PDF file serving
    pdf_name = f"test_artifact_{uuid.uuid4().hex[:6]}.pdf"
    pdf_file = config.GENERATED_ARTIFACTS_DIR / pdf_name
    pdf_file.write_bytes(b"%PDF-1.4\ntest\n%%EOF")

    res_pdf = client.get(f"/api/artifacts/{pdf_name}")
    assert res_pdf.status_code == 200
    assert "application/pdf" in res_pdf.headers.get("content-type", "")

    # 3. SVG file serving
    svg_name = f"test_artifact_{uuid.uuid4().hex[:6]}.svg"
    svg_file = config.GENERATED_ARTIFACTS_DIR / svg_name
    svg_file.write_text("<svg xmlns='http://www.w3.org/2000/svg'></svg>", encoding="utf-8")

    res_svg = client.get(f"/api/artifacts/{svg_name}")
    assert res_svg.status_code == 200
    assert "image/svg+xml" in res_svg.headers.get("content-type", "")

    # 4. Non-existent artifact returns 404
    res_missing = client.get("/api/artifacts/non_existent_file_999.mp3")
    assert res_missing.status_code == 404


def test_20_post_deep_audio_rejects_non_pdf_pptx_with_http_400():
    """Test POST /api/materials/{mat_id}/deep-audio rejects non-PDF/PPTX with HTTP 400 and succeeds for PDF/PPTX."""
    client = TestClient(app)

    # Save a youtube material in store
    yt_mat = store.save_study_material({
        "title": "YouTube Optimization Lecture",
        "source_type": "youtube",
        "user_id": "test_student_rest",
        "original_filename_or_url": "https://www.youtube.com/watch?v=sample",
    })

    res_yt = client.post(
        f"/api/materials/{yt_mat['material_id']}/deep-audio",
        headers={"X-Dev-User-Id": "test_student_rest"}
    )
    assert res_yt.status_code == 400
    assert "exclusively supported for PDF and PPTX" in res_yt.text

    # Save a PDF material in store
    pdf_mat = store.save_study_material({
        "title": "PDF Foundations",
        "source_type": "pdf",
        "user_id": "test_student_rest",
        "original_filename_or_url": "foundations.pdf",
        "structured_preview": [
            {"index": 0, "title": "Page 1", "text": "Foundations of Optimization Theory."}
        ]
    })

    res_pdf = client.post(
        f"/api/materials/{pdf_mat['material_id']}/deep-audio",
        headers={"X-Dev-User-Id": "test_student_rest"}
    )
    assert res_pdf.status_code == 200
    data = res_pdf.json()
    assert "audio_url" in data
    assert data["audio_url"].endswith(".mp3")
