"""ElevenLabs Flash v2.5 TTS Integration & Audio Caching Engine.

Provides speech synthesis with LaTeX-to-spoken-English sanitization,
zero-credit SQLite audio caching, mock audio fallbacks, and deep lecture explanations
for PDF and PPTX study materials.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import httpx

from .. import config
from . import store
from . import llm

logger = logging.getLogger("concept_mastery.tts")

_AUDIO_CACHE_SCHEMA = """
CREATE TABLE IF NOT EXISTS audio_cache (
    cache_key TEXT PRIMARY KEY,
    user_id TEXT NOT NULL DEFAULT 'default',
    file_path TEXT NOT NULL,
    transcript TEXT NOT NULL,
    model_id TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_audio_cache_user ON audio_cache(user_id);
"""


@contextmanager
def get_audio_cache_db():
    """Context manager for SQLite audio cache database connections."""
    config.VECTOR_STORE_PATH.mkdir(parents=True, exist_ok=True)
    db_path = config.VECTOR_STORE_PATH / "study_materials.db"
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_audio_cache_db():
    """Initialize audio_cache table in study_materials.db."""
    config.VECTOR_STORE_PATH.mkdir(parents=True, exist_ok=True)
    with get_audio_cache_db() as conn:
        conn.executescript(_AUDIO_CACHE_SCHEMA)


try:
    init_audio_cache_db()
except Exception as _e:
    logger.warning("Could not initialize audio_cache table on startup: %s", _e)


def _record_audio_cache(
    cache_key: str,
    user_id: str,
    file_path: str,
    transcript: str,
    model_id: str,
) -> None:
    """Persist an audio generation record in the audio_cache table."""
    try:
        init_audio_cache_db()
        now_iso = datetime.now(timezone.utc).isoformat()
        with get_audio_cache_db() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO audio_cache
                    (cache_key, user_id, file_path, transcript, model_id, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (cache_key, user_id, file_path, transcript, model_id, now_iso),
            )
    except Exception as e:
        logger.warning("Failed to write to audio_cache: %s", e)


def sanitize_latex_for_speech(text: str) -> str:
    """Convert math notation into natural spoken English before TTS.

    Transforms mathematical symbols, strips markdown formatting symbols,
    and caps output length to 1,800 characters to protect TTS credits.
    """
    if not text:
        return ""

    s = text

    # Handle common LaTeX fractions and roots
    s = re.sub(r"\\frac\{([^}]+)\}\{([^}]+)\}", r"\1 divided by \2", s)
    s = re.sub(r"\\sqrt\{([^}]+)\}", r"square root of \1", s)

    # Core required LaTeX symbol mappings
    core_mappings = [
        (r"\nabla", "gradient del"),
        ("\nnabla", "gradient del"),
        ("\nabla", "gradient del"),
        (r"\alpha", "learning rate alpha"),
        ("\a" + "lpha", "learning rate alpha"),
        (r"\theta", "parameter theta"),
        ("\t" + "heta", "parameter theta"),
        (r"\partial", "partial derivative"),
        (r"\Delta", "delta change"),
        (r"\sum", "sum of"),
        (r"\lambda", "lambda regularization parameter"),
        (r"\sigma", "sigmoid activation function"),
    ]
    for pat, repl in core_mappings:
        s = s.replace(pat, repl)

    # Additional standard LaTeX mathematical notation
    additional_latex = [
        (r"\\beta(?![a-zA-Z])", "beta"),
        (r"\\gamma(?![a-zA-Z])", "gamma"),
        (r"\\delta(?![a-zA-Z])", "delta"),
        (r"\\mu(?![a-zA-Z])", "mu"),
        (r"\\pi(?![a-zA-Z])", "pi"),
        (r"\\omega(?![a-zA-Z])", "omega"),
        (r"\\epsilon(?![a-zA-Z])", "epsilon"),
        (r"\\int(?![a-zA-Z])", "integral of"),
        (r"\\infty(?![a-zA-Z])", "infinity"),
        (r"\\approx(?![a-zA-Z])", "approximately equal to"),
        (r"\\neq(?![a-zA-Z])", "is not equal to"),
        (r"\\leq(?![a-zA-Z])", "less than or equal to"),
        (r"\\geq(?![a-zA-Z])", "greater than or equal to"),
        (r"\\times(?![a-zA-Z])", "times"),
        (r"\\cdot(?![a-zA-Z])", "times"),
        (r"\\pm(?![a-zA-Z])", "plus or minus"),
    ]
    for pat, repl in additional_latex:
        s = re.sub(pat, repl, s)

    # Strip remaining generic LaTeX command formatting
    s = re.sub(r"\\[a-zA-Z]+\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\[a-zA-Z]+", " ", s)

    # Markdown links and images: extract text or alt text
    s = re.sub(r"!\[([^\]]*)\]\([^)]+\)", r"\1", s)
    s = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", s)

    # Strip raw markdown formatting symbols (#, *, `, $$, $, etc.)
    s = s.replace("$$", " ")
    s = s.replace("$", " ")
    s = re.sub(r"[#*`~_>\^|\\{}]", " ", s)

    # Normalize whitespace
    s = re.sub(r"\s+", " ", s).strip()

    # Cap output length to 1,800 characters (~2 minutes of speech)
    if len(s) > 1800:
        truncated = s[:1800]
        boundary = max(truncated.rfind(". "), truncated.rfind("! "), truncated.rfind("? "))
        if boundary > 1200:
            s = truncated[: boundary + 1].strip()
        elif " " in truncated:
            s = truncated.rsplit(" ", 1)[0].strip()
        else:
            s = truncated

    return s


def synthesize_elevenlabs_audio(
    text: str,
    user_id: str = "default",
    cache_key: Optional[str] = None,
) -> dict:
    """Synthesize speech audio using ElevenLabs or local cache / mock fallback.

    - Cleans input using sanitize_latex_for_speech.
    - If cache_key is None, computes MD5 hash of cleaned text.
    - Checks SQLite audio_cache table and disk for existing MP3 file.
    - If cached: returns cached URL immediately (0 API credits spent).
    - If live API key is present: calls ElevenLabs Flash v2.5 endpoint.
    - If key is empty or network fails / offline in pytest: writes valid mock MP3.
    """
    cleaned_text = sanitize_latex_for_speech(text)
    if not cache_key:
        cache_key = hashlib.md5(cleaned_text.encode("utf-8")).hexdigest()

    config.GENERATED_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    mp3_file = config.GENERATED_ARTIFACTS_DIR / f"{cache_key}.mp3"
    audio_url = f"/api/artifacts/{cache_key}.mp3"
    model_id = config.ELEVENLABS_MODEL_ID or "eleven_flash_v2_5"

    # 1. Check cache
    init_audio_cache_db()
    is_cached = False
    try:
        with get_audio_cache_db() as conn:
            row = conn.execute(
                "SELECT * FROM audio_cache WHERE cache_key = ?", (cache_key,)
            ).fetchone()
            if row and mp3_file.exists():
                is_cached = True
    except Exception as e:
        logger.warning("Error checking audio_cache: %s", e)
        if mp3_file.exists():
            is_cached = True

    if is_cached:
        return {
            "audio_url": audio_url,
            "cached": True,
            "model_id": model_id,
            "transcript": cleaned_text,
        }

    # 2. Live ElevenLabs call if API key is present
    if config.ELEVENLABS_API_KEY:
        try:
            voice_id = config.ELEVENLABS_VOICE_ID or "JBFqnCBsd6RMkjVDRZzb"
            url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
            headers = {
                "xi-api-key": config.ELEVENLABS_API_KEY,
                "Content-Type": "application/json",
            }
            body = {
                "text": cleaned_text,
                "model_id": model_id,
                "voice_settings": {
                    "stability": 0.5,
                    "similarity_boost": 0.75,
                },
            }
            resp = httpx.post(url, headers=headers, json=body, timeout=30.0)
            if resp.status_code == 200 and resp.content:
                mp3_file.write_bytes(resp.content)
                _record_audio_cache(cache_key, user_id, str(mp3_file), cleaned_text, model_id)
                return {
                    "audio_url": audio_url,
                    "cached": False,
                    "model_id": model_id,
                    "transcript": cleaned_text,
                }
            else:
                logger.warning(
                    "ElevenLabs API returned %s: %s", resp.status_code, resp.text[:200]
                )
        except Exception as e:
            logger.warning("ElevenLabs TTS request failed: %s", e)

    # 3. Fallback: valid mock MP3 file
    mock_audio = b"\xff\xfb\x90\x00" + (b"\x00" * 128)
    mp3_file.write_bytes(mock_audio)
    _record_audio_cache(cache_key, user_id, str(mp3_file), cleaned_text, "mock_tts")
    return {
        "audio_url": audio_url,
        "cached": False,
        "fallback_mode": "browser_speech_synthesis",
        "model_id": "mock_tts",
        "transcript": cleaned_text,
    }


def generate_deep_material_audio(
    material: dict,
    user_id: str = "default",
    section_or_slide_index: int = 0,
) -> dict:
    """Generate a professor-style deep spoken explanation for a PDF or PPTX slide.

    - STRICT GATE: Inspects source_type. Only 'pdf' and 'pptx' are permitted.
      Raises store.UserError if material is youtube, web, audio, markdown, or docx.
    - Extracts page or slide content from structured_preview_json or raw_extracted_text.
    - Generates a deep conversational lecture explanation and synthesizes it.
    """
    # Strict format gate
    source_type = (material.get("source_type") or "").strip().lower()
    if source_type not in ("pdf", "pptx"):
        raise store.UserError(
            "Deep audio explanation is exclusively supported for PDF and PPTX materials."
        )

    # Extract preview items
    preview = material.get("structured_preview")
    if not preview:
        preview_raw = material.get("structured_preview_json")
        if isinstance(preview_raw, str) and preview_raw.strip():
            try:
                preview = json.loads(preview_raw)
            except Exception:
                preview = []
        elif isinstance(preview_raw, list):
            preview = preview_raw
        else:
            preview = []

    target_item = None
    if isinstance(preview, list) and preview:
        # Match by explicit index field (both 0-based and 1-based)
        for item in preview:
            if isinstance(item, dict) and item.get("index") == section_or_slide_index:
                target_item = item
                break
        if not target_item:
            for item in preview:
                if isinstance(item, dict) and item.get("index") == (section_or_slide_index + 1):
                    target_item = item
                    break
        # Match by array offset
        if not target_item:
            if 0 <= section_or_slide_index < len(preview):
                target_item = preview[section_or_slide_index]
            elif 1 <= section_or_slide_index <= len(preview):
                target_item = preview[section_or_slide_index - 1]
            else:
                target_item = preview[0]

    target_label = f"Slide {section_or_slide_index}"
    slide_text = ""
    if isinstance(target_item, dict):
        target_label = target_item.get("title") or target_item.get("label") or target_label
        parts = []
        if target_item.get("title"):
            parts.append(str(target_item["title"]))
        if target_item.get("text"):
            parts.append(str(target_item["text"]))
        if target_item.get("notes"):
            parts.append(f"Speaker notes: {target_item['notes']}")
        slide_text = "\n".join(parts).strip()

    if not slide_text:
        slide_text = str(material.get("raw_extracted_text", "")).strip()

    if not slide_text:
        slide_text = str(material.get("title", "Study Material")).strip()

    material_title = str(material.get("title", "Study Material")).strip()

    # Generate professor-style spoken explanation
    spoken_explanation = ""
    prompt_system = (
        "You are an inspiring university professor providing an engaging, crystal-clear spoken audio lecture "
        "explaining a specific slide or page from the study material. Break down the core intuition, mathematical formulas, "
        "and practical significance. Speak conversationally and directly to the student as if giving an expert one-on-one tutorial. "
        "Do not include markdown headers, bullet points, or special formatting. Provide natural, spoken English prose."
    )
    prompt_user = (
        f"Lecture Title: {material_title}\n"
        f"Section/Slide: {target_label}\n"
        f"Content:\n{slide_text}\n\n"
        "Deliver a deep, spoken tutorial explaining this concept clearly."
    )

    try:
        if hasattr(llm, "complete"):
            res = llm.complete(prompt_system, prompt_user, max_tokens=500)
            if res and len(res.strip()) > 30:
                spoken_explanation = res.strip()
    except Exception as e:
        logger.warning("LLM call for deep material audio explanation failed: %s", e)

    if not spoken_explanation:
        lines = [line.strip() for line in slide_text.splitlines() if line.strip() and not line.startswith("#")]
        summary_snippet = " ".join(lines[:4]) if lines else "the fundamental concepts presented on this slide"
        spoken_explanation = (
            f"Welcome back. In this lecture section on {target_label} from {material_title}, "
            f"let us examine the key mechanics. {summary_snippet}. "
            f"Observe how these mathematical and theoretical foundations operate together in practice. "
            f"Keep this intuition in mind as we build toward the next set of concepts."
        )

    mat_id = material.get("material_id")
    cache_key = f"deep_{mat_id}_{section_or_slide_index}" if mat_id else None
    return synthesize_elevenlabs_audio(spoken_explanation, user_id=user_id, cache_key=cache_key)
