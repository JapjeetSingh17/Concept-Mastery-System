"""Unit and integration tests for boundary-aware text chunking in text.py, notes.py, and tutor.py."""

from __future__ import annotations

import inspect
import re

import pytest

from app.core import llm, store, web_search
from app.core.concept_index import ChunkMetadata, GLOBAL_INDEX
from app.core.text import (
    build_context_window,
    chunk_text_boundary_aware,
    sentences,
    truncate_boundary_aware,
)
from app.models.schemas import SmartNoteRequest
from app.pipelines import notes, tutor


def _has_balanced_delimiters(chunk: str) -> bool:
    """Verify a chunk has balanced ``` fences, $$ display blocks, and $ inline math outside fences."""
    fence_open = False
    i = 0
    outside_code_parts: list[str] = []
    part_start = 0

    while i < len(chunk):
        if chunk.startswith("```", i) and (i == 0 or chunk[i - 1] != "\\"):
            if not fence_open:
                outside_code_parts.append(chunk[part_start:i])
                fence_open = True
            else:
                fence_open = False
                part_start = i + 3
            i += 3
            continue
        i += 1

    if fence_open:
        return False
    outside_code_parts.append(chunk[part_start:])
    prose = "\n".join(outside_code_parts)

    # Check balanced unescaped $$ display blocks
    display_delims = len(re.findall(r"(?<!\\)\$\$", prose))
    if display_delims % 2 != 0:
        return False

    # Remove $$...$$ blocks and escaped \$ before checking inline single-$ balance
    no_display = re.sub(r"(?<!\\)\$\$[\s\S]*?(?<!\\)\$\$", "", prose)
    inline_dollars = len(re.findall(r"(?<!\\)\$", no_display))
    return inline_dollars % 2 == 0


def test_public_signatures_preserved():
    """Verify refactoring did not alter public function signatures in text.py, notes.py, or tutor.py."""
    assert list(inspect.signature(sentences).parameters.keys()) == ["text"]
    assert list(inspect.signature(notes.generate_smart_note).parameters.keys()) == ["req"]
    assert list(inspect.signature(notes.generate_note_audio).parameters.keys()) == ["note_id", "user_id"]
    assert list(inspect.signature(notes.export_note_artifact).parameters.keys()) == [
        "note_id",
        "user_id",
        "export_format",
    ]
    assert list(inspect.signature(tutor.chat).parameters.keys()) == [
        "message",
        "concept_id",
        "user_id",
        "mode",
        "chunks",
        "kb",
    ]
    assert list(inspect.signature(tutor.reteach).parameters.keys()) == ["concept_id", "user_id"]


def test_chunk_text_respects_fenced_code_and_dollar_inside_code():
    """Fenced code blocks (including stray $VAR inside shell/python code) must never split mid-block."""
    text = (
        "First we inspect gradient updates in Python before moving to optimization theory.\n\n"
        "```python\n"
        "def sgd_step(theta, grad, lr=0.01):\n"
        "    # Shell note: echo $HOME shouldn't toggle math state!\n"
        "    for idx in range(len(theta)):\n"
        "        theta[idx] = theta[idx] - lr * grad[idx]\n"
        "    return theta\n"
        "```\n\n"
        "After applying the update rule repeatedly, parameter variance decreases steadily."
    )

    chunks = chunk_text_boundary_aware(text, max_chars=95)
    assert len(chunks) >= 2
    assert all(_has_balanced_delimiters(c) for c in chunks), f"Unbalanced chunk found: {chunks}"
    assert any("def sgd_step" in c and "return theta" in c and "```python" in c for c in chunks)


def test_chunk_text_respects_display_and_inline_latex():
    """Neither $$...$$ nor $...$ math expressions should ever be split across chunks."""
    text = (
        "Let us examine the multi-line objective minimization problem across training epochs.\n\n"
        "$$\n"
        "\\mathcal{L}(\\theta) = \\frac{1}{N} \\sum_{i=1}^{N} \\left(y_i - f_\\theta(x_i)\\right)^2 + \\lambda \\|\\theta\\|_2^2.\n"
        "$$\n\n"
        "Each iteration updates $\\theta_{t+1} = \\theta_t - \\eta \\nabla_\\theta \\mathcal{L}(\\theta_t)$ safely, "
        "whereas cost like \\$50 per GPU stays literal."
    )

    chunks = chunk_text_boundary_aware(text, max_chars=80)
    assert len(chunks) >= 2
    assert all(_has_balanced_delimiters(c) for c in chunks), f"Broken math chunk: {chunks}"
    assert any("\\mathcal{L}(\\theta)" in c and c.count("$$") == 2 for c in chunks)
    assert any("$\\theta_{t+1} = \\theta_t - \\eta \\nabla_\\theta \\mathcal{L}(\\theta_t)$" in c for c in chunks)


def test_sentences_does_not_split_inside_math_or_code():
    """sentences() must keep periods inside $$...$$, $...$, and ```...``` blocks together."""
    text = (
        "First explanation sentence introduces notation. "
        "Next is $$a = 1.0.\\quad b = 2.0$$ followed by prose. "
        "Then compute $f(x) = 3.14. x$ and run ```python\nx = 1.0 # Wait! Step.\n``` afterwards."
    )
    extracted = sentences(text)
    assert all(_has_balanced_delimiters(s) for s in extracted), f"Sentence split broke delimiters: {extracted}"


def test_truncate_boundary_aware_keeps_delimiters_intact():
    """Short truncations landing inside math/code delimiters must not leave half-open blocks."""
    expr = "By the chain rule $$\\frac{\\partial L}{\\partial w} = \\frac{\\partial L}{\\partial y} \\cdot \\frac{\\partial y}{\\partial w}$$ we obtain gradients."
    short = truncate_boundary_aware(expr, max_chars=50)
    assert _has_balanced_delimiters(short), f"Unbalanced truncation: {short}"

    code_text = "Overview text spanning a few words before ```python\nprint('hello world from block')\n``` and extra prose afterwards."
    short_code = truncate_boundary_aware(code_text, max_chars=55)
    assert _has_balanced_delimiters(short_code), f"Unbalanced code truncation: {short_code}"


def test_build_context_window_preserves_blocks_under_budget():
    """Context window builder splits overflowing chunks along protected boundaries instead of slicing raw strings."""
    long_chunk = type("ChunkMeta", (), {
        "locator_label": "Page 3",
        "page_number": 3,
        "section_title": "Loss Functions",
        "text": (
            "Section intro explaining regularized empirical risk.\n\n"
            + ("Background prose explaining convergence guarantees under smooth objectives. " * 6)
            + "\n\n$$\nJ(\\theta) = \\frac{1}{m}\\sum_{i=1}^m L(h_\\theta(x^{(i)}), y^{(i)}) + \\frac{\\lambda}{2}\\|\\theta\\|^2\n$$\n\n"
            + ("Trailing discussion extending beyond the max context window budget. " * 8)
        ),
    })()

    ctx = build_context_window([long_chunk], max_chars=380, chunk_max_chars=260)
    assert ctx
    assert len(ctx) <= 380
    assert _has_balanced_delimiters(ctx), f"Unbalanced context window:\n{ctx}"


def test_notes_and_tutor_llm_context_and_fallbacks_preserve_blocks(monkeypatch):
    """notes.generate_smart_note and tutor.chat prepare LLM context windows and deterministic fallbacks without breaking blocks."""
    user_id = "test_boundary_chunk_user"
    store.clear_all_study_materials(user_id)
    store.clear_kb(user_id)
    GLOBAL_INDEX.clear(user_id)

    try:
        math_chunk_text = (
            "Optimization minimizes empirical error across gradient steps smoothly over time. "
            "Update rule uses $$\\Delta \\theta = -\\eta \\nabla_\\theta J(\\theta) + \\mu v_{t-1}$$ during momentum descent.\n\n"
            "```python\n"
            "for step in range(steps):\n"
            "    v = mu * v - lr * grad(theta)\n"
            "    theta = theta + v\n"
            "```\n\n"
            + ("Additional long derivation detail explaining Lipschitz continuity and step-size selection bounds. " * 60)
        )
        chunk = ChunkMetadata(
            chunk_id="chk_math_1",
            page_number=4,
            section_title="Momentum Gradient Descent",
            text=math_chunk_text,
            canonical_concepts=["momentum gradient descent"],
            source_filename="opt.pdf",
            locator_label="Page 4",
            source_id="mat_opt",
            source_type="pdf",
            source_title="Optimization Notes",
        )
        GLOBAL_INDEX.index_chunks([chunk], user_id=user_id)

        captured_prompts: list[str] = []
        monkeypatch.setattr(llm, "enabled", lambda: True)

        def fake_complete(system: str, user: str, max_tokens: int = 800, **kwargs):
            captured_prompts.append(user)
            return ""

        monkeypatch.setattr(llm, "complete", fake_complete)

        # 1. Notes LLM prompt preparation + deterministic quiz table fallback
        note_res = notes.generate_smart_note(
            SmartNoteRequest(
                concept_id="momentum_gradient_descent",
                task_type="quiz",
                user_id=user_id,
            )
        )
        assert captured_prompts, "notes.generate_smart_note should invoke llm.complete when enabled"
        assert _has_balanced_delimiters(captured_prompts[-1]), "LLM prompt from notes.py had broken code/LaTeX block"
        assert _has_balanced_delimiters(note_res.markdown_content), "Deterministic quiz note had broken code/LaTeX block"

        # 2. Tutor LLM prompt preparation (local chunks) + deterministic fallback
        captured_prompts.clear()
        tutor_res = tutor.chat(
            message="How does momentum gradient descent update theta?",
            concept_id="momentum_gradient_descent",
            user_id=user_id,
            chunks=[chunk],
            kb={"ready": True, "concepts": [{"id": "momentum_gradient_descent", "name": "Momentum Gradient Descent"}]},
        )
        assert captured_prompts, "tutor.chat should invoke llm.complete when chunks are provided"
        assert _has_balanced_delimiters(captured_prompts[-1]), "LLM prompt from tutor.py had broken code/LaTeX block"
        assert _has_balanced_delimiters(tutor_res.answer), "Deterministic tutor fallback had broken code/LaTeX block"

        # 3. Tutor web fallback excerpt truncation
        monkeypatch.setattr(
            web_search,
            "search",
            lambda q, max_results=3: [
                {
                    "title": "Momentum Formula",
                    "site_name": "ML Docs",
                    "url": "https://example.org/momentum",
                    "snippet": (
                        "Momentum accelerates stochastic gradient descent in relevant directions using $$v_t = \\gamma v_{t-1} + \\eta \\nabla_\\theta J(\\theta)$$ "
                        + ("and dampens oscillations across ravines with high curvature. " * 6)
                    ),
                }
            ],
        )
        web_res = tutor.chat(
            message="search the web for momentum gradient descent",
            concept_id="momentum_gradient_descent",
            user_id=user_id,
            kb={"ready": False, "concepts": []},
        )
        assert _has_balanced_delimiters(web_res.answer), f"Tutor web fallback broke delimiters: {web_res.answer}"
    finally:
        store.clear_all_study_materials(user_id)
        store.clear_kb(user_id)
        GLOBAL_INDEX.clear(user_id)

