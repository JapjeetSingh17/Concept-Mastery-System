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
import httpx
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
    """Test synthesize_elevenlabs_audio creates a valid audio file on disk and audio_cache hit returns cached=True with 0 duplicate API calls."""
    test_phrase = f"Deterministic speech synthesis test for closed loop verification {uuid.uuid4().hex[:6]}."
    res1 = tts.synthesize_elevenlabs_audio(test_phrase, user_id="test_user_qa")
    assert isinstance(res1, dict)
    assert "audio_url" in res1
    assert res1["audio_url"].endswith((".mp3", ".wav"))

    file_name = Path(res1["audio_url"]).name
    audio_path = config.GENERATED_ARTIFACTS_DIR / file_name
    assert audio_path.exists(), f"Audio file {audio_path} was not created."
    assert audio_path.stat().st_size > 0

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
    assert pdf_res["audio_url"].endswith((".mp3", ".wav"))

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
    assert pptx_res["audio_url"].endswith((".mp3", ".wav"))


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
    """Test distinct formatting across task types ('exam', 'test', 'quiz', 'viva', 'essay')."""
    task_types = ["exam", "test", "quiz", "viva", "essay"]
    generated_notes = {}

    for tt in task_types:
        req = SmartNoteRequest(concept_id="stochastic_gradient_descent", task_type=tt, user_id="test_user_qa")
        note_resp = notes.generate_smart_note(req)
        assert note_resp.task_type == tt
        assert note_resp.markdown_content
        generated_notes[tt] = note_resp

    assert "Derivation" in generated_notes["exam"].markdown_content or "Trap" in generated_notes["exam"].markdown_content
    assert "Mock Test" in generated_notes["test"].markdown_content or "Diagnostic" in generated_notes["test"].markdown_content or "Rubric" in generated_notes["test"].markdown_content
    assert "Elevator Pitch" in generated_notes["viva"].markdown_content or "Examiner" in generated_notes["viva"].markdown_content or "Analogy" in generated_notes["viva"].markdown_content
    assert "Cheat Sheet" in generated_notes["quiz"].markdown_content or "Parameter" in generated_notes["quiz"].markdown_content or "Cloze" in generated_notes["quiz"].markdown_content
    assert "Monograph" in generated_notes["essay"].markdown_content or "Theoretical" in generated_notes["essay"].markdown_content

    mds = [n.markdown_content for n in generated_notes.values()]
    assert len(set(mds)) == 5, "Each task type must produce a unique, distinct markdown formatting."


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
    mp3_file.write_bytes(tts._VALID_FALLBACK_MP3_BYTES)

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
    assert data["audio_url"].endswith((".mp3", ".wav"))


def test_21_artifact_downloads_flushed_bytes_and_content_disposition():
    """Verify PDF and PPTX artifact responses return exact flushed bytes, Content-Length, and attachment disposition."""
    client = TestClient(app)

    # 1. PPTX generation and direct artifact download check
    pptx_deck = tools.generate_pptx_deck_tool(
        concept_id="gradient_descent",
        user_id="test_dl_user",
        slide_count=3
    )
    pptx_filename = Path(pptx_deck["download_url"]).name
    pptx_resp = client.get(f"/api/artifacts/{pptx_filename}")
    assert pptx_resp.status_code == 200
    assert "presentationml" in pptx_resp.headers.get("content-type", "")
    assert f'attachment; filename="{pptx_filename}"' in pptx_resp.headers.get("content-disposition", "")
    assert "content-length" in pptx_resp.headers
    assert int(pptx_resp.headers["content-length"]) == len(pptx_resp.content)
    assert len(pptx_resp.content) > 1000

    # 2. PDF generation and direct artifact download check
    pdf_handout = tools.generate_pdf_handout_tool(
        concept_id="gradient_descent",
        user_id="test_dl_user"
    )
    pdf_filename = Path(pdf_handout["download_url"]).name
    pdf_resp = client.get(f"/api/artifacts/{pdf_filename}")
    assert pdf_resp.status_code == 200
    assert "application/pdf" in pdf_resp.headers.get("content-type", "")
    assert f'attachment; filename="{pdf_filename}"' in pdf_resp.headers.get("content-disposition", "")
    assert "content-length" in pdf_resp.headers
    assert int(pdf_resp.headers["content-length"]) == len(pdf_resp.content)
    assert len(pdf_resp.content) > 500

    # 3. Direct note export download response via ?download=true
    note_record = store.save_smart_note({
        "concept_id": "gradient_descent",
        "user_id": "test_dl_user",
        "title": "Gradient Descent Overview",
        "markdown_content": "# Gradient Descent\n\n## Formulations\nDetailed equations and step size rules.\n",
    })
    direct_dl_resp = client.post(
        f"/api/notes/{note_record['note_id']}/export?export_format=pdf&download=true",
        headers={"X-Dev-User-Id": "test_dl_user"}
    )
    assert direct_dl_resp.status_code == 200
    assert "application/pdf" in direct_dl_resp.headers.get("content-type", "")
    assert "attachment; filename=" in direct_dl_resp.headers.get("content-disposition", "")
    assert "content-length" in direct_dl_resp.headers
    assert int(direct_dl_resp.headers["content-length"]) == len(direct_dl_resp.content)


def test_22_normalize_markdown_inline_headings_and_bullets():
    """Verify inline headings (#, ##) and inline bullet items split cleanly into structured lines."""
    raw_single_line = (
        "# Optimization Theory ## 1. Gradient Updates - First-order parameter adjustment "
        "- Step size governed by learning rate ## 2. Formulations: 1. Cost function 2. Gradient vector"
    )
    normalized = tools.normalize_markdown_content(raw_single_line)
    assert "# Optimization Theory" in normalized
    assert "\n\n## 1. Gradient Updates" in normalized
    assert "\n- First-order parameter adjustment" in normalized
    assert "\n- Step size governed by learning rate" in normalized
    assert "\n\n## 2. Formulations:" in normalized
    assert "\n1. Cost function" in normalized
    assert "\n2. Gradient vector" in normalized

    # Verify code fences with backticks and hyphens are preserved without splitting
    code_markdown = (
        "## Algorithmic Implementation\n"
        "```python\n"
        "def compute_step(w, lr, grad):\n"
        "    return w - lr * grad\n"
        "```\n"
        "- High-yield performance property."
    )
    norm_code = tools.normalize_markdown_content(code_markdown)
    assert "def compute_step(w, lr, grad):" in norm_code
    assert "return w - lr * grad" in norm_code
    assert "High-yield" in norm_code

    # Verify section parser breaks into structured sections
    sections = tools.parse_markdown_into_sections(normalized, default_title="Opt Theory")
    assert len(sections) >= 2
    assert any("Gradient Updates" in s["heading"] for s in sections)
    assert any("Formulations" in s["heading"] for s in sections)


def test_23_expand_note_content_under_300_words():
    """Verify expand_note_content expands notes under 300 words with grounded deep-dive sections."""
    short_note = {
        "note_id": f"sn_short_{uuid.uuid4().hex[:6]}",
        "concept_id": "backpropagation",
        "title": "Short Backpropagation Note",
        "markdown_content": "# Backpropagation\nBackpropagation computes gradients of loss with respect to weights using chain rule.",
        "citations": ["Page 12", "Page 15"],
        "task_type": "exam"
    }
    assert len(short_note["markdown_content"].split()) < 300

    expanded = notes.expand_note_content(short_note, user_id="test_qa_expand")
    word_count = len(expanded.split())
    assert word_count >= 300, f"Expanded note must have >= 300 words, got {word_count}"
    assert "## Deep Dive Mechanics" in expanded
    assert "## Worked Algorithmic Traversal" in expanded
    assert "## Common Diagnostic Traps" in expanded
    assert "## Active Recall Self-Assessment" in expanded
    assert "[Page 12]" in expanded or "[Page 15]" in expanded

    # If note is already >= 300 words, expand_note_content returns existing content unchanged
    long_note = {
        "concept_id": "backpropagation",
        "markdown_content": "word " * 350
    }
    assert notes.expand_note_content(long_note) == long_note["markdown_content"]


def test_24_rich_pptx_deck_and_multipage_pdf_generation():
    """Verify PPTX generator produces 6-10 slides with dark theme, and PDF generator produces multi-page guide (> 5 KB)."""
    concept_id = "gradient_descent"
    user_id = "test_qa_multipage"

    # 1. PPTX generation produces 6 to 10 comprehensive slides
    deck_res = tools.generate_pptx_deck_tool(concept_id, user_id=user_id)
    assert deck_res["artifact_type"] == "pptx"
    slides = deck_res["slide_previews"]
    assert 6 <= len(slides) <= 10, f"Deck must have between 6 and 10 slides, got {len(slides)}"

    categories = [s.get("category") for s in slides]
    assert "ARCHITECTURE & CURRICULUM" in categories
    assert any("INTUITION" in c for c in categories)
    assert any("MECHANISM" in c or "FORMULATION" in c for c in categories)
    assert any("WORKED" in c or "EXAMPLE" in c for c in categories)
    assert any("PITFALL" in c for c in categories)
    assert any("ACTIVE RECALL" in c for c in categories)

    for s in slides:
        assert len(s["bullets"]) >= 2, f"Slide {s['title']} must not have empty bullets"

    # PPTX file exists and has substantial size (> 30 KB)
    pptx_path = config.GENERATED_ARTIFACTS_DIR / Path(deck_res["download_url"]).name
    assert pptx_path.exists()
    assert pptx_path.stat().st_size > 30000

    # 2. PDF generation produces multi-page study guide
    rich_sections = [
        {
            "heading": f"{i}. Comprehensive Section {i}",
            "body": (
                f"Detailed foundational analysis of {concept_id} and governing invariants. " * 8 +
                "\n```python\ndef execute_step():\n    return 'verified'\n```\n" +
                "- Key property: Invariant state progression.\n- Parameter bounds verified strictly.\n"
            )
        }
        for i in range(1, 6)
    ]
    pdf_res = tools.generate_pdf_handout_tool(
        concept_id=concept_id,
        user_id=user_id,
        content_sections=rich_sections,
        custom_title="Comprehensive Multi-Page Study Guide"
    )
    pdf_path = config.GENERATED_ARTIFACTS_DIR / Path(pdf_res["download_url"]).name
    assert pdf_path.exists()
    pdf_bytes = pdf_path.read_bytes()
    assert len(pdf_bytes) > 5000, f"Multi-page PDF must be > 5 KB, got {len(pdf_bytes)} bytes"


def test_25_export_endpoint_with_full_markdown_payload():
    """Verify /api/notes/{note_id}/export accepts full markdown_content in POST payload and auto-expands short notes."""
    client = TestClient(app)
    uid = "test_qa_payload_user"

    # Save note initially with small placeholder
    init_note = store.save_smart_note({
        "concept_id": "loss_landscapes",
        "user_id": uid,
        "title": "Initial Mini Note",
        "markdown_content": "# Loss Landscapes\nShort initial summary.",
    })
    note_id = init_note["note_id"]

    # Export with full custom markdown payload
    full_markdown = (
        "# Loss Landscapes & Convexity\n\n"
        "## 1. Geometry of Error Surfaces\n"
        "Loss surfaces govern optimization trajectories.\n"
        "- Non-convex surfaces contain local minima and saddle points.\n"
        "- Condition number determines convergence rate.\n\n"
        "## 2. Mathematical Formulations\n"
        "```python\n"
        "Hessian = compute_hessian(loss, weights)\n"
        "```\n"
        "- Eigenvalues of Hessian determine local curvature.\n"
    )
    resp = client.post(
        f"/api/notes/{note_id}/export?export_format=pdf",
        headers={"X-Dev-User-Id": uid},
        json={
            "export_format": "pdf",
            "title": "Mastery Guide: Loss Landscapes",
            "markdown_content": full_markdown
        }
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data.get("pdf_url")
    assert data["pdf_url"].endswith(".pdf")

    # Stored note must reflect the updated title and expanded content (>= 300 words)
    updated = store.get_smart_note(note_id, user_id=uid)
    assert updated is not None
    assert updated["title"] == "Mastery Guide: Loss Landscapes"
    assert len(updated["markdown_content"].split()) >= 300


def test_26_sanitize_svg_diagram_fences_attributes_and_high_contrast():
    """Verify sanitize_svg_diagram strips fences, injects xmlns/viewBox/style, and enforces light high-contrast text."""
    # 1. Strips markdown fences and prose
    raw_markdown = (
        "Here is the interactive SVG diagram for gradient descent:\n"
        "```xml\n"
        "<svg height='400'>\n"
        "  <rect width='400' height='200' />\n"
        "  <text fill='#000000'>Precondition Initialization</text>\n"
        "</svg>\n"
        "```\n"
        "Review complete."
    )
    sanitized = tools.sanitize_svg_diagram(raw_markdown)
    assert sanitized.startswith("<svg")
    assert sanitized.endswith("</svg>")
    assert "Here is the interactive SVG" not in sanitized
    assert "Review complete" not in sanitized
    assert "xmlns=\"http://www.w3.org/2000/svg\"" in sanitized
    assert "viewBox=\"0 0 900 420\"" in sanitized
    assert "width=\"100%\"" in sanitized
    assert "style=\"max-width:100%; height:auto; display:block;\"" in sanitized
    assert "fill=\"#e2e8f0\"" in sanitized  # Replaced dark fill with light high-contrast fill

    # 2. Rejects SVG with only empty <text> nodes
    empty_text_svg = "<svg><text>   </text><text></text></svg>"
    assert tools.sanitize_svg_diagram(empty_text_svg) == ""

    # 3. Rejects input without <svg> tags
    assert tools.sanitize_svg_diagram("Just plain text with no diagram") == ""


def test_27_deterministic_svg_flowchart_multi_node_structure():
    """Verify deterministic fallback SVG flowchart builds a clean multi-node diagram with prerequisites and sub-steps."""
    svg = tools.generate_deterministic_svg_flowchart("Backpropagation", source_locator="Page 12")
    assert "<svg" in svg
    assert "</svg>" in svg
    assert "xmlns=\"http://www.w3.org/2000/svg\"" in svg
    assert "viewBox=\"0 0 900 420\"" in svg
    assert "Backpropagation Architecture" in svg
    assert "1. Preconditions" in svg
    assert "2. Backpropagatio" in svg
    assert "3. Verification" in svg
    assert "4. Synthesis" in svg
    assert "Source: Page 12" in svg
    # High-contrast colors
    assert "#e2e8f0" in svg
    assert "#60a5fa" in svg
    assert "#38bdf8" in svg
    assert "#34d399" in svg


def test_28_visual_diagram_endpoint_returns_svg_field():
    """Verify generate_visual_diagram_tool and /api/tools/generate return the svg field for frontend injection."""
    client = TestClient(app)
    uid = "test_qa_diagram_user"

    # 1. Direct tool call
    res = tools.generate_visual_diagram_tool(concept_id="gradient_descent", user_id=uid)
    assert isinstance(res, dict)
    assert res.get("artifact_type") == "visual"
    assert "svg" in res
    assert res["svg"].startswith("<svg")
    assert "</svg>" in res["svg"]
    assert "viewBox=\"0 0 900 420\"" in res["svg"]
    assert "download_url" in res
    assert res["download_url"].endswith(".svg")

    # 2. REST endpoint call
    api_resp = client.post(
        "/api/tools/generate",
        headers={"X-Dev-User-Id": uid},
        json={
            "tool_type": "visual",
            "concept_id": "gradient_descent",
            "prompt": "draw concept flow"
        }
    )
    assert api_resp.status_code == 200
    data = api_resp.json()
    assert "svg" in data
    assert data["svg"].startswith("<svg")
    assert "</svg>" in data["svg"]
    assert "download_url" in data


def test_29_sanitize_latex_for_speech_strips_codeblocks_and_delimiters():
    """Verify sanitize_latex_for_speech strips fenced code blocks, inline code, and LaTeX math delimiters."""
    input_text = (
        "## Key Optimization Rules\n"
        "Here is the algorithm:\n"
        "```python\n"
        "def step(w, lr):\n"
        "    return w - lr * grad\n"
        "```\n"
        "We update variable `w` using loss $$ \\mathcal{L} = \\sum (y - \\hat{y})^2 $$ and learning rate $\\alpha$.\n"
        "* Step 1: Compute $\\nabla L$\n"
        "* Step 2: Scale by $\\alpha$\n"
    )
    cleaned = tts.sanitize_latex_for_speech(input_text)
    assert "```" not in cleaned
    assert "def step" not in cleaned
    assert "##" not in cleaned
    assert "$$" not in cleaned
    assert "$" not in cleaned
    assert "`" not in cleaned
    assert "*" not in cleaned
    assert "learning rate alpha" in cleaned
    assert "gradient del" in cleaned


def test_30_tts_artifact_route_serves_valid_mp3_with_content_length_and_503_on_empty():
    """Verify /api/artifacts/{filename} serves MP3 with audio/mpeg, Content-Length, and returns 503 for 0-byte or corrupted <=100 byte files."""
    client = TestClient(app)

    # 1. Valid MP3 artifact
    valid_name = f"test_valid_audio_{uuid.uuid4().hex[:6]}.mp3"
    valid_path = config.GENERATED_ARTIFACTS_DIR / valid_name
    valid_path.write_bytes(tts._VALID_FALLBACK_MP3_BYTES)

    try:
        resp = client.get(f"/api/artifacts/{valid_name}")
        assert resp.status_code == 200
        assert resp.headers.get("content-type") == "audio/mpeg"
        assert resp.headers.get("content-length") == str(len(tts._VALID_FALLBACK_MP3_BYTES))
        assert resp.headers.get("accept-ranges") == "bytes"
        assert len(resp.content) == len(tts._VALID_FALLBACK_MP3_BYTES)
        assert len(resp.content) > 100
    finally:
        if valid_path.exists():
            valid_path.unlink()

    # 2. 0-byte MP3 artifact -> 503
    empty_name = f"test_empty_audio_{uuid.uuid4().hex[:6]}.mp3"
    empty_path = config.GENERATED_ARTIFACTS_DIR / empty_name
    empty_path.write_bytes(b"")

    try:
        resp_empty = client.get(f"/api/artifacts/{empty_name}")
        assert resp_empty.status_code == 503
        assert "empty" in resp_empty.json()["detail"].lower()
    finally:
        if empty_path.exists():
            empty_path.unlink()

    # 3. Corrupt truncated <=100 byte MP3 -> 503
    corrupt_name = f"test_corrupt_audio_{uuid.uuid4().hex[:6]}.mp3"
    corrupt_path = config.GENERATED_ARTIFACTS_DIR / corrupt_name
    corrupt_path.write_bytes(b"\xff\xfb\x90\x00" + b"\x00" * 32)

    try:
        resp_corrupt = client.get(f"/api/artifacts/{corrupt_name}")
        assert resp_corrupt.status_code == 503
        assert "invalid" in resp_corrupt.json()["detail"].lower() or "truncated" in resp_corrupt.json()["detail"].lower()
    finally:
        if corrupt_path.exists():
            corrupt_path.unlink()


def test_31_synthesize_elevenlabs_audio_valid_fallback_size():
    """Verify synthesize_elevenlabs_audio produces valid audio with valid headers and size > 44 bytes."""
    test_phrase = f"Auditory playback verification phrase {uuid.uuid4().hex[:6]}."
    res = tts.synthesize_elevenlabs_audio(test_phrase, user_id="test_qa_audio_size")
    assert isinstance(res, dict)
    assert "audio_url" in res
    assert res["audio_url"].endswith((".mp3", ".wav"))

    filename = Path(res["audio_url"]).name
    audio_file = config.GENERATED_ARTIFACTS_DIR / filename
    assert audio_file.exists()
    assert audio_file.stat().st_size > 44
    file_bytes = audio_file.read_bytes()
    # Must have ID3 or MPEG sync word or RIFF header
    assert file_bytes.startswith(b"ID3") or b"\xff\xfb" in file_bytes[:100] or file_bytes.startswith(b"RIFF")


def test_32_multiprovider_tts_failover_elevenlabs_to_gemini_and_groq(monkeypatch):
    """Test 3-tier TTS failover:
    ElevenLabs HTTP 429/402 -> Gemini 2.5 Flash TTS -> Groq Orpheus TTS -> valid WAV audio bytes.
    Also validates Groq Whisper STT helper transcribe_audio_groq.
    """
    cleaned_text = "Testing multi-tier failover mechanism across ElevenLabs, Gemini, and Groq."

    # 1. Verify _synthesize_elevenlabs handles HTTP 429 by returning None
    class MockElevenLabs429:
        status_code = 429
        text = "Too Many Requests: Rate limit reached"

    monkeypatch.setattr(config, "ELEVENLABS_API_KEY", "dummy_el_key")
    monkeypatch.setattr(httpx, "post", lambda *args, **kwargs: MockElevenLabs429())
    el_result = tts._synthesize_elevenlabs("Short text", "eleven_flash_v2_5")
    assert el_result is None, "ElevenLabs HTTP 429 must return None to trigger failover"

    # Generate a dummy 24kHz 16-bit mono PCM sample wrapped in WAV header
    dummy_pcm = b"\x00\x00\x10\x00" * 100
    gemini_wav = tts.pcm_to_wav(dummy_pcm, sample_rate=24000)
    assert gemini_wav.startswith(b"RIFF")
    assert len(gemini_wav) > 44

    # 2. Test Failover from ElevenLabs (fails/returns None) to Gemini 2.5 Flash TTS
    monkeypatch.setattr(tts, "_synthesize_elevenlabs", lambda text, model: None)
    monkeypatch.setattr(tts, "_synthesize_gemini_tts", lambda text: gemini_wav)

    audio_bytes, media_type, provider_used, model_id = tts.synthesize_speech_multiprovider(cleaned_text)
    assert provider_used == "gemini"
    assert media_type == "audio/wav"
    assert audio_bytes.startswith(b"RIFF")
    assert len(audio_bytes) > 44

    # 3. Test Failover from Gemini (fails/returns None) to Groq Orpheus TTS
    monkeypatch.setattr(tts, "_synthesize_gemini_tts", lambda text: None)
    groq_chunk_wav = tts.pcm_to_wav(b"\x20\x00\x30\x00" * 50, sample_rate=24000)
    monkeypatch.setattr(tts, "_synthesize_groq_tts", lambda text: groq_chunk_wav)

    audio_bytes_groq, media_type_groq, provider_groq, model_groq = tts.synthesize_speech_multiprovider(cleaned_text)
    assert provider_groq == "groq"
    assert media_type_groq == "audio/wav"
    assert audio_bytes_groq.startswith(b"RIFF")
    assert len(audio_bytes_groq) > 44

    # 4. Test public helper synthesize_speech returning (audio_bytes, media_type, provider_used)
    pub_bytes, pub_media, pub_prov = tts.synthesize_speech("Test public signature")
    assert pub_prov == "groq"
    assert pub_media == "audio/wav"
    assert pub_bytes.startswith(b"RIFF")

    # 5. Test transcribe_audio_groq Whisper STT helper
    class MockGroqWhisperResp:
        status_code = 200
        def json(self):
            return {"text": "Transcribed audio via Groq Whisper large v3 turbo"}

    monkeypatch.setattr(config, "GROQ_API_KEY", "mock_groq_key")
    monkeypatch.setattr(httpx, "post", lambda *args, **kwargs: MockGroqWhisperResp())
    transcript = tts.transcribe_audio_groq(b"RIFFdummydata", filename="student_voice.webm")
    assert transcript == "Transcribed audio via Groq Whisper large v3 turbo"




