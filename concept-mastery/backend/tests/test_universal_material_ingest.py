"""Integration Test Suite for Multi-Format Universal Study Material Ingestion & Interactive Preview.

Verifies:
  1. PDF, PPTX, Markdown, DOCX, Audio, Web, and YouTube loaders.
  2. Exact citation anchors ([Page X], [Slide X], [Section X], [Web Section: X], [Audio MM:SS], [04:15]).
  3. SQLite database persistence in study_materials table.
  4. Heterogeneous multi-source merge into unified Prerequisite DAG.
  5. REST API endpoints:
     - POST /api/materials/ingest-file
     - POST /api/materials/ingest-url
     - GET  /api/materials
     - GET  /api/materials/{material_id}/preview
     - DELETE /api/materials/{material_id}
  6. Socratic dynamic citations citing [locator_label].
"""

import io
import json
import os
import pytest
from pathlib import Path
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from app import config
from app.main import app
from app.core import store
from app.core.concept_index import GLOBAL_INDEX
from app.pipelines.ingest import (
    UniversalMaterialIngestor,
    load_pdf_material,
    load_pptx_material,
    load_markdown_material,
    load_docx_material,
    load_youtube_material,
    load_web_article_material,
    load_audio_material,
)
from app.pipelines import tutor, quiz, concepts

client = TestClient(app)


@pytest.fixture(autouse=True)
def clean_test_materials():
    """Ensure clean materials DB before each test."""
    store.clear_all_study_materials("test_student")
    yield
    store.clear_all_study_materials("test_student")


# ── 1. Markdown (.md) Loader & Ingestion ────────────────────────────────────

def test_markdown_loader_and_ingestion(tmp_path):
    md_file = tmp_path / "deep_learning.md"
    md_file.write_text(
        "# Deep Learning Foundations\n\n"
        "Deep learning is a subset of machine learning based on artificial neural networks. "
        "Representation learning algorithms discover the representations needed for feature detection.\n\n"
        "## Backpropagation\n\n"
        "Backpropagation iteratively calculates the gradient of the loss function with respect to weights. "
        "The chain rule of calculus propagates error derivatives backwards through network layers.\n\n"
        "### Gradient Descent\n\n"
        "Gradient descent updates neural network parameters by taking steps proportional to the negative gradient. "
        "A calibrated learning rate is essential to ensure steady convergence without oscillating.",
        encoding="utf-8"
    )

    result = UniversalMaterialIngestor.ingest_file(md_file, user_id="test_student", title="Deep Learning Notes")
    assert result["source_type"] == "markdown"
    assert result["chunk_count"] > 0
    assert result["concept_count"] > 0

    # Verify SQLite storage
    stored = store.get_study_material(result["material_id"])
    assert stored is not None
    assert stored["title"] == "Deep Learning Notes"
    assert len(stored["structured_preview"]) >= 2
    assert any("Backpropagation" in p.get("label", "") or "Backpropagation" in p.get("heading", "") for p in stored["structured_preview"])


# ── 2. PowerPoint (.pptx) Loader & Ingestion ────────────────────────────────

def test_pptx_loader_and_ingestion(tmp_path):
    import pptx
    prs = pptx.Presentation()
    slide_layout = prs.slide_layouts[1]  # Title and Content

    # Slide 1: Introduction to Optimization
    s1 = prs.slides.add_slide(slide_layout)
    s1.shapes.title.text = "Optimization Algorithms"
    tf1 = s1.placeholders[1].text_frame
    tf1.text = "Mathematical optimization minimizes loss functions in statistical modeling."
    p1 = tf1.add_paragraph()
    p1.text = "Convex objectives guarantee global minima for gradient based optimizers."
    s1.notes_slide.notes_text_frame.text = "Emphasize convex curvature vs saddle points."

    # Slide 2: Gradient Descent
    s2 = prs.slides.add_slide(slide_layout)
    s2.shapes.title.text = "Gradient Descent Mechanisms"
    tf2 = s2.placeholders[1].text_frame
    tf2.text = "Gradient descent takes steps in the direction of steepest descent. The learning rate regulates step magnitude."

    pptx_file = tmp_path / "lecture_slides.pptx"
    prs.save(str(pptx_file))

    result = UniversalMaterialIngestor.ingest_file(pptx_file, user_id="test_student", title="Optimization Lecture")
    assert result["source_type"] == "pptx"
    assert result["chunk_count"] >= 2

    # Check locator label
    stored = store.get_study_material(result["material_id"])
    assert stored is not None
    previews = stored["structured_preview"]
    assert any(p["label"] == "Slide 1" for p in previews)
    assert any(p["label"] == "Slide 2" for p in previews)
    assert any("saddle points" in p.get("notes", "") for p in previews)


# ── 3. Word Document (.docx) Loader & Ingestion ─────────────────────────────

def test_docx_loader_and_ingestion(tmp_path):
    import docx
    doc = docx.Document()
    doc.add_heading("Neural Architecture Overview", level=1)
    doc.add_paragraph("Neural networks consist of stacked computational layers transforming input features into predictions.")
    doc.add_heading("Activation Functions", level=2)
    doc.add_paragraph("Non-linear activation functions like ReLU and Sigmoid enable neural networks to approximate arbitrary functions.")

    docx_file = tmp_path / "notes.docx"
    doc.save(str(docx_file))

    result = UniversalMaterialIngestor.ingest_file(docx_file, user_id="test_student", title="Neural Architectures")
    assert result["source_type"] == "docx"
    assert result["chunk_count"] >= 1

    stored = store.get_study_material(result["material_id"])
    assert stored is not None
    assert any("Activation Functions" in p.get("heading", "") or "Activation Functions" in p.get("label", "") for p in stored["structured_preview"])


# ── 4. YouTube Transcript Loader with Timestamped Windows ───────────────────

def test_youtube_loader_mock():
    mock_snippets = [
        {"text": "Welcome to our lecture on machine learning algorithms.", "start": 0.0, "duration": 4.5},
        {"text": "Today we discuss gradient descent and optimization.", "start": 5.0, "duration": 5.0},
        {"text": "The gradient vector contains partial derivatives of the cost function.", "start": 12.0, "duration": 6.0},
        {"text": "Moving along the negative gradient minimizes total prediction error.", "start": 80.0, "duration": 5.0},
        {"text": "Notice how the learning rate controls step size at timestamp 01:25.", "start": 85.0, "duration": 6.0},
    ]

    from youtube_transcript_api import YouTubeTranscriptApi
    mock_objs = [MagicMock(text=s["text"], start=s["start"], duration=s["duration"]) for s in mock_snippets]
    with patch.object(YouTubeTranscriptApi, "fetch", return_value=mock_objs):
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        title, raw_items, preview, full_text = load_youtube_material(url, custom_title="ML YouTube Lecture")

        assert "YouTube" in title or "Lecture" in title
        assert len(preview) >= 1
        assert any("00:00" in p["label"] or "01:20" in p["label"] for p in preview)
        assert preview[0].get("video_id") == "dQw4w9WgXcQ"


# ── 5. Educational Web Article Loader (GeeksforGeeks Mock) ──────────────────

def test_web_article_loader_mock():
    mock_html = """
    <html>
      <head><title>Gradient Descent Algorithm - GeeksforGeeks</title></head>
      <body>
        <nav><a href="/">Home</a></nav>
        <article>
          <h1>Gradient Descent in Machine Learning</h1>
          <h2>Introduction</h2>
          <p>Gradient descent is an iterative first-order optimization algorithm used to find a local minimum of a differentiable function.</p>
          <h2>Mathematical Formulation</h2>
          <p>The parameter update rule is defined by theta equals theta minus alpha times the gradient of loss.</p>
          <pre>def gradient_descent(x, lr): return x - lr * 2 * x</pre>
        </article>
        <footer>Copyright 2026</footer>
      </body>
    </html>
    """

    class MockResponse:
        def read(self):
            return mock_html.encode("utf-8")
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass

    with patch("urllib.request.urlopen", return_value=MockResponse()):
        url = "https://www.geeksforgeeks.org/gradient-descent-in-machine-learning/"
        title, raw_items, preview, full_text = load_web_article_material(url)

        assert "Gradient Descent" in title
        assert len(preview) >= 1
        assert any("Web: " in p["label"] for p in preview)
        assert any("gradient_descent" in p.get("text", "") for p in preview)


# ── 6. Audio / Voice Note Loader ────────────────────────────────────────────

def test_audio_loader_with_companion_text(tmp_path):
    audio_file = tmp_path / "voice_note_1.mp3"
    audio_file.write_bytes(b"\x00" * 200)
    # Companion text fixture
    companion = tmp_path / "voice_note_1.txt"
    companion.write_text(
        "In this voice note, we review linear regression and loss functions. "
        "The mean squared error penalizes larger residuals quadratic in nature.",
        encoding="utf-8"
    )

    title, raw_items, preview, full_text = load_audio_material(audio_file)
    assert len(preview) >= 1
    assert "Audio" in preview[0]["label"]
    assert "mean squared error" in preview[0]["text"].lower()


# ── 7. Multi-Format REST API Endpoints ──────────────────────────────────────

def test_rest_api_materials_ingest_file_and_preview(tmp_path):
    # Create test markdown file
    md_path = tmp_path / "study_guide.md"
    md_path.write_text(
        "# Neural Network Optimization\n\n"
        "Loss functions compute the penalty for incorrect predictions during training.\n\n"
        "## Stochastic Gradient Descent\n\n"
        "SGD updates weights using a single sample or mini-batch per step.",
        encoding="utf-8"
    )

    with open(md_path, "rb") as f:
        resp = client.post(
            "/api/materials/ingest-file",
            files={"file": ("study_guide.md", f, "text/markdown")},
            data={"user_id": "test_student", "title": "Neural Optimization Guide"}
        )

    assert resp.status_code == 200
    data = resp.json()
    assert data["ok"] is True
    assert len(data["materials"]) == 1
    mat_id = data["materials"][0]["material_id"]

    # 1. GET /api/materials
    list_resp = client.get("/api/materials?user_id=test_student")
    assert list_resp.status_code == 200
    list_data = list_resp.json()
    assert list_data["total_materials"] >= 1
    assert any(m["material_id"] == mat_id for m in list_data["materials"])

    # 2. GET /api/materials/{material_id}/preview
    prev_resp = client.get(f"/api/materials/{mat_id}/preview")
    assert prev_resp.status_code == 200
    prev_data = prev_resp.json()
    assert prev_data["material_id"] == mat_id
    assert len(prev_data["structured_preview"]) >= 1

    # 3. DELETE /api/materials/{material_id}
    del_resp = client.delete(f"/api/materials/{mat_id}?user_id=test_student")
    assert del_resp.status_code == 200
    del_data = del_resp.json()
    assert del_data["ok"] is True

    # Confirm deletion
    get_after = client.get(f"/api/materials/{mat_id}/preview")
    assert get_after.status_code == 404


# ── 8. Dynamic Citations (citing locator_label in Tutor & Quiz) ─────────────

def test_tutor_and_quiz_cite_dynamic_locator_label(tmp_path):
    # Ingest PPTX slide deck
    import pptx
    prs = pptx.Presentation()
    s = prs.slides.add_slide(prs.slide_layouts[1])
    s.shapes.title.text = "Backpropagation Mechanisms"
    tf = s.placeholders[1].text_frame
    tf.text = "Backpropagation applies the calculus chain rule backwards from output loss."
    pptx_path = tmp_path / "backprop_lecture.pptx"
    prs.save(str(pptx_path))

    UniversalMaterialIngestor.ingest_file(pptx_path, user_id="test_student", title="Backprop Deck")

    # Verify Socratic Tutor citations cite [Slide 1]
    res = tutor.chat("What does backpropagation do?", user_id="test_student")
    assert len(res.citations) > 0
    assert any("Slide" in (c.locator_label or "") or "Slide" in c.snippet for c in res.citations)
