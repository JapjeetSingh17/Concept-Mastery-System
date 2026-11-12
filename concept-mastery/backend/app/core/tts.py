"""ElevenLabs Flash v2.5 TTS Integration & Audio Caching Engine.

Provides speech synthesis with LaTeX-to-spoken-English sanitization,
zero-credit SQLite audio caching, mock audio fallbacks, and deep lecture explanations
for PDF and PPTX study materials.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import logging
import os
import re
import sqlite3
import wave
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Tuple

from fastapi.concurrency import run_in_threadpool
import httpx

from .. import config
from . import store
from . import llm
from .fallback_sound import FALLBACK_MP3_BYTES as _VALID_FALLBACK_MP3_BYTES

logger = logging.getLogger("concept_mastery.tts")
_TTS_TIMEOUT = httpx.Timeout(30.0, connect=5.0)
_MOCK_MP3_BYTES = _VALID_FALLBACK_MP3_BYTES

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
    conn = sqlite3.connect(str(db_path), timeout=10.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=10000")
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

    Transforms mathematical symbols, strips markdown formatting symbols (#, *, `, $$, $),
    and caps output length to 1,800 characters to protect TTS credits.
    """
    if not text:
        return ""

    s = text

    # Strip fenced code blocks
    s = re.sub(r"```[\w]*\n?[\s\S]*?```", " ", s)

    # Convert inline code blocks: `code` -> code
    s = re.sub(r"`([^`]+)`", r"\1", s)

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

    # Strip LaTeX delimiters ($$...$$ and $...$)
    s = re.sub(r"\$\$([^\$]+)\$\$", r"\1", s)
    s = s.replace("$$", " ")
    s = re.sub(r"\$([^\$]+)\$", r"\1", s)
    s = s.replace("$", " ")

    # Strip raw markdown formatting symbols: headers (#), bullets (*, -, +), bold/italics (*, _, ~), blockquotes (>), backticks (`), pipes (|)
    s = re.sub(r"(?m)^\s*#{1,6}\s+", " ", s)
    s = re.sub(r"(?m)^\s*[-*+]\s+", " ", s)
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


def pcm_to_wav(
    pcm_bytes: bytes,
    sample_rate: int = 24000,
    channels: int = 1,
    sample_width: int = 2
) -> bytes:
    """Wrap raw 16-bit PCM bytes into standard WAV format with a 44-byte header."""
    if pcm_bytes.startswith(b"RIFF"):
        return pcm_bytes
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
    return buf.getvalue()


def split_text_for_groq_tts(text: str, max_chars: int = 190) -> list[str]:
    """Split text into sentence/phrase chunks of <= max_chars characters for Groq Orpheus."""
    if not text:
        return []
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    chunks = []
    current = ""
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(sentence) <= max_chars:
            if current and (len(current) + len(sentence) + 1 > max_chars):
                chunks.append(current.strip())
                current = sentence
            else:
                current = f"{current} {sentence}".strip() if current else sentence
        else:
            if current:
                chunks.append(current.strip())
                current = ""
            words = sentence.split(" ")
            word_chunk = ""
            for w in words:
                if not w:
                    continue
                if len(word_chunk) + len(w) + 1 <= max_chars:
                    word_chunk = f"{word_chunk} {w}".strip() if word_chunk else w
                else:
                    if word_chunk:
                        chunks.append(word_chunk.strip())
                    word_chunk = w[:max_chars]
            if word_chunk:
                current = word_chunk
    if current:
        chunks.append(current.strip())
    return [c for c in chunks if c.strip()]


def concatenate_wav_streams(wav_bytes_list: list[bytes]) -> bytes:
    """Concatenate multiple WAV streams into a single valid WAV file using Python wave module."""
    if not wav_bytes_list:
        return b""
    if len(wav_bytes_list) == 1:
        return wav_bytes_list[0]

    params = None
    frames_data = []

    for wb in wav_bytes_list:
        if not wb or len(wb) < 44 or not wb.startswith(b"RIFF"):
            continue
        try:
            with wave.open(io.BytesIO(wb), "rb") as wf:
                p = wf.getparams()
                if params is None:
                    params = p
                frames = wf.readframes(wf.getnframes())
                frames_data.append(frames)
        except Exception as e:
            logger.warning("Error reading WAV chunk: %s", e)

    if not frames_data or params is None:
        return wav_bytes_list[0] if wav_bytes_list else b""

    out_buf = io.BytesIO()
    with wave.open(out_buf, "wb") as out_wf:
        out_wf.setparams(params)
        for f in frames_data:
            out_wf.writeframes(f)
    return out_buf.getvalue()


def transcribe_audio_groq(audio_bytes: bytes, filename: str = "audio.webm") -> str:
    """Transcribe speech audio bytes to text using Groq's whisper-large-v3-turbo."""
    if not (config.GROQ_API_KEY or "").strip():
        logger.warning("GROQ_API_KEY is unset; audio transcription skipped.")
        return ""
    try:
        url = f"{config.GROQ_BASE_URL.rstrip('/')}/audio/transcriptions"
        headers = {"Authorization": f"Bearer {config.GROQ_API_KEY}"}
        model = config.GROQ_STT_MODEL or "whisper-large-v3-turbo"
        mime = "audio/wav" if filename.endswith(".wav") else "audio/mpeg" if filename.endswith(".mp3") else "audio/webm"
        files = {"file": (filename, audio_bytes, mime)}
        data = {"model": model}
        resp = httpx.post(url, headers=headers, data=data, files=files, timeout=_TTS_TIMEOUT)
        if resp.status_code == 200:
            return resp.json().get("text", "").strip()
        logger.warning("Groq Whisper STT returned HTTP %s: %s", resp.status_code, resp.text[:200])
    except Exception as e:
        logger.warning("Groq Whisper STT error: %s", e)
    return ""


def _extract_gemini_audio(data: dict) -> Optional[bytes]:
    """Extract audio bytes from Gemini generateContent response and wrap PCM in WAV header."""
    try:
        candidates = data.get("candidates") or []
        if not candidates:
            return None
        parts = candidates[0].get("content", {}).get("parts", [])
        for part in parts:
            inline_data = part.get("inlineData")
            if inline_data and "data" in inline_data:
                raw_audio = base64.b64decode(inline_data["data"])
                mime_type = inline_data.get("mimeType", "").lower()
                if raw_audio.startswith(b"RIFF"):
                    return raw_audio
                rate_match = re.search(r"rate=(\d+)", mime_type)
                rate = int(rate_match.group(1)) if rate_match else 24000
                return pcm_to_wav(raw_audio, sample_rate=rate)
    except Exception as e:
        logger.warning("Error parsing Gemini TTS response: %s", e)
    return None


def _synthesize_elevenlabs(cleaned_text: str, model_id: str) -> Optional[bytes]:
    """Tier 1: ElevenLabs Flash v2.5 TTS."""
    if not (config.ELEVENLABS_API_KEY or "").strip():
        return None
    try:
        url, headers, body = _build_elevenlabs_request(cleaned_text, model_id)
        resp = httpx.post(url, headers=headers, json=body, timeout=_TTS_TIMEOUT)
        if resp.status_code == 200 and resp.content and len(resp.content) > 32:
            return resp.content
        logger.warning(
            "Tier 1 (ElevenLabs) failed with HTTP %s: %s; falling over to Tier 2 (Gemini)...",
            resp.status_code,
            resp.text[:200],
        )
    except Exception as e:
        logger.warning("Tier 1 (ElevenLabs) request error: %s; falling over to Tier 2 (Gemini)...", e)
    return None


async def _synthesize_elevenlabs_async(cleaned_text: str, model_id: str) -> Optional[bytes]:
    """Tier 1 Async: ElevenLabs Flash v2.5 TTS."""
    if not (config.ELEVENLABS_API_KEY or "").strip():
        return None
    try:
        url, headers, body = _build_elevenlabs_request(cleaned_text, model_id)
        async with httpx.AsyncClient(timeout=_TTS_TIMEOUT) as client:
            resp = await client.post(url, headers=headers, json=body)
        if resp.status_code == 200 and resp.content and len(resp.content) > 32:
            return resp.content
        logger.warning(
            "Tier 1 (ElevenLabs) async failed with HTTP %s: %s; falling over to Tier 2 (Gemini)...",
            resp.status_code,
            resp.text[:200],
        )
    except Exception as e:
        logger.warning("Tier 1 (ElevenLabs) async error: %s; falling over to Tier 2 (Gemini)...", e)
    return None


def _synthesize_gemini_tts(cleaned_text: str) -> Optional[bytes]:
    """Tier 2: Gemini 2.5 Flash TTS."""
    if not (config.GEMINI_API_KEY or "").strip():
        return None
    try:
        model = config.GEMINI_TTS_MODEL or "gemini-2.5-flash-preview-tts"
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={config.GEMINI_API_KEY}"
        payload = {
            "contents": [{"parts": [{"text": cleaned_text}]}],
            "generationConfig": {
                "responseModalities": ["AUDIO"],
                "speechConfig": {
                    "voiceConfig": {
                        "prebuiltVoiceConfig": {
                            "voiceName": config.GEMINI_TTS_VOICE or "Kore"
                        }
                    }
                }
            }
        }
        resp = httpx.post(url, headers={"Content-Type": "application/json"}, json=payload, timeout=_TTS_TIMEOUT)
        if resp.status_code == 200:
            audio_bytes = _extract_gemini_audio(resp.json())
            if audio_bytes and len(audio_bytes) > 44:
                return audio_bytes
        logger.warning(
            "Tier 2 (Gemini TTS) failed with HTTP %s: %s; falling over to Tier 3 (Groq)...",
            resp.status_code,
            resp.text[:200],
        )
    except Exception as e:
        logger.warning("Tier 2 (Gemini TTS) error: %s; falling over to Tier 3 (Groq)...", e)
    return None


async def _synthesize_gemini_tts_async(cleaned_text: str) -> Optional[bytes]:
    """Tier 2 Async: Gemini 2.5 Flash TTS."""
    if not (config.GEMINI_API_KEY or "").strip():
        return None
    try:
        model = config.GEMINI_TTS_MODEL or "gemini-2.5-flash-preview-tts"
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={config.GEMINI_API_KEY}"
        payload = {
            "contents": [{"parts": [{"text": cleaned_text}]}],
            "generationConfig": {
                "responseModalities": ["AUDIO"],
                "speechConfig": {
                    "voiceConfig": {
                        "prebuiltVoiceConfig": {
                            "voiceName": config.GEMINI_TTS_VOICE or "Kore"
                        }
                    }
                }
            }
        }
        async with httpx.AsyncClient(timeout=_TTS_TIMEOUT) as client:
            resp = await client.post(url, headers={"Content-Type": "application/json"}, json=payload)
        if resp.status_code == 200:
            audio_bytes = _extract_gemini_audio(resp.json())
            if audio_bytes and len(audio_bytes) > 44:
                return audio_bytes
        logger.warning(
            "Tier 2 (Gemini TTS) async failed with HTTP %s: %s; falling over to Tier 3 (Groq)...",
            resp.status_code,
            resp.text[:200],
        )
    except Exception as e:
        logger.warning("Tier 2 (Gemini TTS) async error: %s; falling over to Tier 3 (Groq)...", e)
    return None


def _synthesize_groq_tts(cleaned_text: str) -> Optional[bytes]:
    """Tier 3: Groq Orpheus TTS with chunking <= 190 characters."""
    if not (config.GROQ_API_KEY or "").strip():
        return None
    try:
        chunks = split_text_for_groq_tts(cleaned_text, max_chars=190)
        if not chunks:
            return None
        headers = {
            "Authorization": f"Bearer {config.GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        url = f"{config.GROQ_BASE_URL.rstrip('/')}/audio/speech"
        model = config.GROQ_TTS_MODEL or "canopylabs/orpheus-v1-english"
        voice = config.GROQ_TTS_VOICE or "troy"
        wav_list = []
        for ch in chunks:
            body = {
                "model": model,
                "voice": voice,
                "response_format": "wav",
                "input": ch
            }
            resp = httpx.post(url, headers=headers, json=body, timeout=_TTS_TIMEOUT)
            if resp.status_code == 200 and resp.content and len(resp.content) >= 44:
                wav_list.append(resp.content)
            else:
                logger.warning("Groq TTS chunk failed with HTTP %s: %s", resp.status_code, resp.text[:200])
        if wav_list:
            return concatenate_wav_streams(wav_list)
    except Exception as e:
        logger.warning("Tier 3 (Groq TTS) error: %s; falling back to local fallback...", e)
    return None


async def _synthesize_groq_tts_async(cleaned_text: str) -> Optional[bytes]:
    """Tier 3 Async: Groq Orpheus TTS with chunking <= 190 characters."""
    if not (config.GROQ_API_KEY or "").strip():
        return None
    try:
        chunks = split_text_for_groq_tts(cleaned_text, max_chars=190)
        if not chunks:
            return None
        headers = {
            "Authorization": f"Bearer {config.GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        url = f"{config.GROQ_BASE_URL.rstrip('/')}/audio/speech"
        model = config.GROQ_TTS_MODEL or "canopylabs/orpheus-v1-english"
        voice = config.GROQ_TTS_VOICE or "troy"
        wav_list = []
        async with httpx.AsyncClient(timeout=_TTS_TIMEOUT) as client:
            for ch in chunks:
                body = {
                    "model": model,
                    "voice": voice,
                    "response_format": "wav",
                    "input": ch
                }
                resp = await client.post(url, headers=headers, json=body)
                if resp.status_code == 200 and resp.content and len(resp.content) >= 44:
                    wav_list.append(resp.content)
                else:
                    logger.warning("Groq TTS async chunk failed with HTTP %s: %s", resp.status_code, resp.text[:200])
        if wav_list:
            return concatenate_wav_streams(wav_list)
    except Exception as e:
        logger.warning("Tier 3 (Groq TTS) async error: %s; falling back to local fallback...", e)
    return None


def synthesize_speech_multiprovider(cleaned_text: str) -> tuple[bytes, str, str, str]:
    """Execute 3-tier TTS failover:
    Tier 1: ElevenLabs -> audio/mpeg
    Tier 2: Gemini 2.5 Flash TTS -> audio/wav
    Tier 3: Groq Orpheus TTS -> audio/wav
    Fallback: Local valid MP3 frames -> audio/mpeg

    Returns (audio_bytes, media_type, provider_used, model_id).
    """
    # Tier 1: ElevenLabs
    el_model = config.ELEVENLABS_MODEL_ID or "eleven_flash_v2_5"
    audio_bytes = _synthesize_elevenlabs(cleaned_text, el_model)
    if audio_bytes and len(audio_bytes) > 32:
        return audio_bytes, "audio/mpeg", "elevenlabs", el_model

    # Tier 2: Gemini 2.5 Flash TTS
    gem_model = config.GEMINI_TTS_MODEL or "gemini-2.5-flash-preview-tts"
    audio_bytes = _synthesize_gemini_tts(cleaned_text)
    if audio_bytes and len(audio_bytes) > 44:
        return audio_bytes, "audio/wav", "gemini", gem_model

    # Tier 3: Groq Orpheus TTS
    groq_model = config.GROQ_TTS_MODEL or "canopylabs/orpheus-v1-english"
    audio_bytes = _synthesize_groq_tts(cleaned_text)
    if audio_bytes and len(audio_bytes) > 44:
        return audio_bytes, "audio/wav", "groq", groq_model

    # Local fallback
    return _VALID_FALLBACK_MP3_BYTES, "audio/mpeg", "local_fallback", "mock_tts"


async def synthesize_speech_multiprovider_async(cleaned_text: str) -> tuple[bytes, str, str, str]:
    """Execute async 3-tier TTS failover:
    Returns (audio_bytes, media_type, provider_used, model_id).
    """
    # Tier 1: ElevenLabs
    el_model = config.ELEVENLABS_MODEL_ID or "eleven_flash_v2_5"
    audio_bytes = await _synthesize_elevenlabs_async(cleaned_text, el_model)
    if audio_bytes and len(audio_bytes) > 32:
        return audio_bytes, "audio/mpeg", "elevenlabs", el_model

    # Tier 2: Gemini 2.5 Flash TTS
    gem_model = config.GEMINI_TTS_MODEL or "gemini-2.5-flash-preview-tts"
    audio_bytes = await _synthesize_gemini_tts_async(cleaned_text)
    if audio_bytes and len(audio_bytes) > 44:
        return audio_bytes, "audio/wav", "gemini", gem_model

    # Tier 3: Groq Orpheus TTS
    groq_model = config.GROQ_TTS_MODEL or "canopylabs/orpheus-v1-english"
    audio_bytes = await _synthesize_groq_tts_async(cleaned_text)
    if audio_bytes and len(audio_bytes) > 44:
        return audio_bytes, "audio/wav", "groq", groq_model

    # Local fallback
    return _VALID_FALLBACK_MP3_BYTES, "audio/mpeg", "local_fallback", "mock_tts"


def synthesize_speech(
    text: str,
    user_id: str = "default",
    cache_key: Optional[str] = None
) -> tuple[bytes, str, str]:
    """Public helper returning (audio_bytes, media_type, provider_used)."""
    cleaned_text = sanitize_latex_for_speech(text)
    audio_bytes, media_type, provider_used, _ = synthesize_speech_multiprovider(cleaned_text)
    return audio_bytes, media_type, provider_used


def _get_cached_audio(cache_key: str, default_path: Optional[Path] = None) -> Optional[dict]:
    """Check SQLite audio_cache and disk for existing synthesized audio with valid bytes."""
    init_audio_cache_db()
    try:
        with get_audio_cache_db() as conn:
            row = conn.execute(
                "SELECT * FROM audio_cache WHERE cache_key = ?", (cache_key,)
            ).fetchone()
            if row:
                f_path = Path(row["file_path"])
                if f_path.exists() and f_path.stat().st_size > 100:
                    media_type = "audio/wav" if f_path.suffix.lower() == ".wav" else "audio/mpeg"
                    data_bytes = f_path.read_bytes()
                    data_uri = f"data:{media_type};base64,{base64.b64encode(data_bytes).decode('ascii')}"
                    return {
                        "audio_url": f"/api/artifacts/{f_path.name}",
                        "cached": True,
                        "file_path": str(f_path),
                        "model_id": row["model_id"],
                        "transcript": row["transcript"],
                        "media_type": media_type,
                        "provider_used": "cache",
                        "data_uri": data_uri,
                    }
    except Exception as e:
        logger.warning("Error checking audio_cache: %s", e)
    if default_path and default_path.exists() and default_path.stat().st_size > 100:
        media_type = "audio/wav" if default_path.suffix.lower() == ".wav" else "audio/mpeg"
        data_bytes = default_path.read_bytes()
        data_uri = f"data:{media_type};base64,{base64.b64encode(data_bytes).decode('ascii')}"
        return {
            "audio_url": f"/api/artifacts/{default_path.name}",
            "cached": True,
            "file_path": str(default_path),
            "model_id": "cached",
            "transcript": "",
            "media_type": media_type,
            "provider_used": "cache",
            "data_uri": data_uri,
        }
    return None


def _is_audio_cached(cache_key: str, mp3_file: Optional[Path] = None) -> bool:
    """Boolean check for whether audio is cached on disk."""
    return bool(_get_cached_audio(cache_key, mp3_file))


def _prepare_synthesis_context(text: str, cache_key: Optional[str]) -> tuple[str, str, Path, str, str]:
    """Sanitize input text and resolve cache paths and model identifier."""
    cleaned_text = sanitize_latex_for_speech(text)
    key = cache_key or hashlib.md5(cleaned_text.encode("utf-8")).hexdigest()
    config.GENERATED_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    mp3_file = config.GENERATED_ARTIFACTS_DIR / f"{key}.mp3"
    audio_url = f"/api/artifacts/{key}.mp3"
    model_id = config.ELEVENLABS_MODEL_ID or "eleven_flash_v2_5"
    return cleaned_text, key, mp3_file, audio_url, model_id


def _build_elevenlabs_request(cleaned_text: str, model_id: str) -> tuple[str, dict[str, str], dict]:
    """Build ElevenLabs Flash v2.5 HTTP request URL, headers, and payload."""
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
    return url, headers, body


def _persist_tts_result(
    cache_key: str,
    user_id: str,
    target_file: Path,
    audio_url: str,
    cleaned_text: str,
    model_id: str,
    audio_bytes: bytes,
    media_type: str = "audio/mpeg",
    provider_used: str = "elevenlabs",
    fallback_mode: Optional[str] = None,
) -> dict:
    """Write audio bytes to disk, record cache metadata, and build response dict."""
    target_file.parent.mkdir(parents=True, exist_ok=True)
    target_file.write_bytes(audio_bytes)
    _record_audio_cache(cache_key, user_id, str(target_file), cleaned_text, model_id)
    data_uri = f"data:{media_type};base64,{base64.b64encode(audio_bytes).decode('ascii')}"
    res = {
        "audio_url": audio_url,
        "media_type": media_type,
        "provider_used": provider_used,
        "data_uri": data_uri,
        "cached": False,
        "model_id": model_id,
        "transcript": cleaned_text,
    }
    if fallback_mode:
        res["fallback_mode"] = fallback_mode
    return res


def synthesize_elevenlabs_audio(
    text: str,
    user_id: str = "default",
    cache_key: Optional[str] = None,
) -> dict:
    """Synthesize speech audio using 3-tier failover (ElevenLabs -> Gemini 2.5 Flash -> Groq Orpheus -> Local)."""
    cleaned_text = sanitize_latex_for_speech(text)
    key = cache_key or hashlib.md5(cleaned_text.encode("utf-8")).hexdigest()
    config.GENERATED_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    cached = _get_cached_audio(key, config.GENERATED_ARTIFACTS_DIR / f"{key}.mp3")
    if not cached:
        cached = _get_cached_audio(key, config.GENERATED_ARTIFACTS_DIR / f"{key}.wav")
    if cached:
        return cached

    audio_bytes, media_type, provider_used, model_id = synthesize_speech_multiprovider(cleaned_text)
    ext = "wav" if media_type == "audio/wav" else "mp3"
    target_file = config.GENERATED_ARTIFACTS_DIR / f"{key}.{ext}"
    audio_url = f"/api/artifacts/{key}.{ext}"

    return _persist_tts_result(
        cache_key=key,
        user_id=user_id,
        target_file=target_file,
        audio_url=audio_url,
        cleaned_text=cleaned_text,
        model_id=model_id,
        audio_bytes=audio_bytes,
        media_type=media_type,
        provider_used=provider_used,
    )


async def synthesize_elevenlabs_audio_async(
    text: str,
    user_id: str = "default",
    cache_key: Optional[str] = None,
) -> dict:
    """Non-blocking async variant using httpx.AsyncClient and threadpool disk/SQLite I/O."""
    cleaned_text = sanitize_latex_for_speech(text)
    key = cache_key or hashlib.md5(cleaned_text.encode("utf-8")).hexdigest()
    config.GENERATED_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    cached = await run_in_threadpool(_get_cached_audio, key, config.GENERATED_ARTIFACTS_DIR / f"{key}.mp3")
    if not cached:
        cached = await run_in_threadpool(_get_cached_audio, key, config.GENERATED_ARTIFACTS_DIR / f"{key}.wav")
    if cached:
        return cached

    audio_bytes, media_type, provider_used, model_id = await synthesize_speech_multiprovider_async(cleaned_text)
    ext = "wav" if media_type == "audio/wav" else "mp3"
    target_file = config.GENERATED_ARTIFACTS_DIR / f"{key}.{ext}"
    audio_url = f"/api/artifacts/{key}.{ext}"

    return await run_in_threadpool(
        _persist_tts_result,
        key,
        user_id,
        target_file,
        audio_url,
        cleaned_text,
        model_id,
        audio_bytes,
        media_type,
        provider_used,
    )


def _prepare_deep_audio_spec(
    material: dict,
    section_or_slide_index: int = 0,
) -> tuple[str, str, str, str, Optional[str]]:
    """Validate material format and prepare LLM prompt, fallback script, and cache key."""
    source_type = (material.get("source_type") or "").strip().lower()
    if source_type not in ("pdf", "pptx"):
        raise store.UserError(
            "Deep audio explanation is exclusively supported for PDF and PPTX materials."
        )

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
        for item in preview:
            if isinstance(item, dict) and item.get("index") == section_or_slide_index:
                target_item = item
                break
        if not target_item:
            for item in preview:
                if isinstance(item, dict) and item.get("index") == (section_or_slide_index + 1):
                    target_item = item
                    break
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

    lines = [line.strip() for line in slide_text.splitlines() if line.strip() and not line.startswith("#")]
    summary_snippet = " ".join(lines[:4]) if lines else "the fundamental concepts presented on this slide"
    fallback_script = (
        f"Welcome back. In this lecture section on {target_label} from {material_title}, "
        f"let us examine the key mechanics. {summary_snippet}. "
        f"Observe how these mathematical and theoretical foundations operate together in practice. "
        f"Keep this intuition in mind as we build toward the next set of concepts."
    )

    mat_id = material.get("material_id")
    cache_key = f"deep_{mat_id}_{section_or_slide_index}" if mat_id else None
    return prompt_system, prompt_user, fallback_script, slide_text, cache_key


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
    prompt_system, prompt_user, fallback_script, _, cache_key = _prepare_deep_audio_spec(
        material, section_or_slide_index
    )

    spoken_explanation = ""
    try:
        if hasattr(llm, "complete"):
            res = llm.complete(prompt_system, prompt_user, max_tokens=500)
            if res and len(res.strip()) > 30:
                spoken_explanation = res.strip()
    except Exception as e:
        logger.warning("LLM call for deep material audio explanation failed: %s", e)

    return synthesize_elevenlabs_audio(
        spoken_explanation or fallback_script, user_id=user_id, cache_key=cache_key
    )


generate_deep_material_audio._is_default_sync_deep_audio = True


async def generate_deep_material_audio_async(
    material: dict,
    user_id: str = "default",
    section_or_slide_index: int = 0,
) -> dict:
    """Async non-blocking version of generate_deep_material_audio."""
    prompt_system, prompt_user, fallback_script, _, cache_key = _prepare_deep_audio_spec(
        material, section_or_slide_index
    )

    spoken_explanation = ""
    try:
        if hasattr(llm, "complete"):
            res = await run_in_threadpool(llm.complete, prompt_system, prompt_user, 500)
            if res and len(res.strip()) > 30:
                spoken_explanation = res.strip()
    except Exception as e:
        logger.warning("LLM call for deep material audio explanation failed: %s", e)

    return await synthesize_elevenlabs_audio_async(
        spoken_explanation or fallback_script, user_id=user_id, cache_key=cache_key
    )
