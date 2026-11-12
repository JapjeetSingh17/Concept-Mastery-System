"""Application configuration, environment loading, and startup validation."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger("concept_mastery.config")

BASE = Path(__file__).resolve().parents[1]          # backend/
try:
    from dotenv import load_dotenv
    if os.getenv("SKIP_DOTENV", "").strip().lower() not in ("1", "true", "yes"):
        load_dotenv(BASE / ".env")
except ImportError:
    pass


def _e(key: str, default: str = "") -> str:
    return os.getenv(key, default).split("#")[0].strip()


def _p(key: str, default: str) -> Path:
    val = _e(key, default) or default
    return (BASE / val).resolve()


def _int(key: str, default: int) -> int:
    raw = _e(key, "")
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning("Invalid integer for %s=%r; falling back to %d", key, raw, default)
        return default


def _float(key: str, default: float) -> float:
    raw = _e(key, "")
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning("Invalid float for %s=%r; falling back to %s", key, raw, default)
        return default


def _bool(key: str, default: bool = False) -> bool:
    raw = _e(key, "")
    if not raw:
        return default
    return raw.lower() in ("1", "true", "yes", "on")


def _opt_path(key: str, default_path: Path) -> Path:
    raw = _e(key, "")
    if not raw:
        return default_path
    p = Path(raw)
    return p.resolve() if p.is_absolute() else (BASE / p).resolve()


# ── Runtime Environment & Offline/Test Controls ────────────────────────────────
ENVIRONMENT = _e("ENVIRONMENT", "development")
OFFLINE_TEST = _bool("OFFLINE_TEST", False)
OPENROUTER_LIVE = _bool("OPENROUTER_LIVE", False)

# ── Storage & SQLite Database Paths ────────────────────────────────────────────
UPLOAD_DIR = _p("UPLOAD_DIR", "../data/uploads")
VECTOR_STORE_PATH = _p("VECTOR_STORE_PATH", "../data/vectorstore")
FRONTEND_DIR = (BASE.parent / "frontend").resolve()
SYNONYMS_FILE = BASE / "synonyms.json"
GENERATED_ARTIFACTS_DIR = _p("GENERATED_ARTIFACTS_DIR", "generated_artifacts")

SQLITE_MATERIALS_DB_PATH = _opt_path("SQLITE_MATERIALS_DB_PATH", VECTOR_STORE_PATH / "study_materials.db")
SQLITE_INDEX_DB_PATH = _opt_path("SQLITE_INDEX_DB_PATH", VECTOR_STORE_PATH / "vectorless_index.db")
SQLITE_MASTERY_DB_PATH = _opt_path("SQLITE_MASTERY_DB_PATH", VECTOR_STORE_PATH / "mastery.db")
SQLITE_BUSY_TIMEOUT_MS = _int("SQLITE_BUSY_TIMEOUT_MS", 10000)

# ── PageIndex API ──────────────────────────────────────────────────────────────
PAGEINDEX_API_KEY = _e("PAGEINDEX_API_KEY", "")

# ── Google OAuth 2.0 & Session Security ────────────────────────────────────────
DEFAULT_INSECURE_JWT_SECRET = "concept-mastery-jwt-secret-key-32bytes"
_PLACEHOLDER_SECRET_MARKERS = (
    "replace-with",
    "change-me",
    "changeme",
    "your_",
    "your-",
    "<set-",
)

GOOGLE_CLIENT_ID = _e("GOOGLE_CLIENT_ID", "")
GOOGLE_CLIENT_SECRET = _e("GOOGLE_CLIENT_SECRET", "")
JWT_SECRET = _e("JWT_SECRET", DEFAULT_INSECURE_JWT_SECRET) or DEFAULT_INSECURE_JWT_SECRET
JWT_ALGORITHM = _e("JWT_ALGORITHM", "HS256")
JWT_EXPIRES_MINUTES = _int("JWT_EXPIRES_MINUTES", 10080)  # 7 days

# ── Unified OpenRouter Gateway & Fallback Models ───────────────────────────────
DEFAULT_OPENROUTER_MODELS = [
    "google/gemini-2.5-flash",
    "nvidia/nemotron-3-ultra-550b-a55b:free",
    "deepseek/deepseek-chat-v3-0324:free",
    "nvidia/nemotron-3.5-super-vl:free",
    "meta-llama/llama-3.3-70b-instruct:free",
    "openrouter/auto",
]
OPENROUTER_API_KEY = _e("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = _e("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
OPENROUTER_PRIMARY_MODEL = _e("OPENROUTER_MODEL", "google/gemini-2.5-flash")
_raw_openrouter_models = _e("OPENROUTER_MODELS", "")
OPENROUTER_MODELS = (
    [m.strip() for m in _raw_openrouter_models.split(",") if m.strip()]
    if _raw_openrouter_models
    else list(DEFAULT_OPENROUTER_MODELS)
)

# ── Secondary LLM Providers ────────────────────────────────────────────────────
# Primary Provider: Google Gemini (Legacy/Direct)
GEMINI_API_KEY = _e("GEMINI_API_KEY", "")
GEMINI_MODEL = _e("GEMINI_MODEL", "gemini-3.8-flash")

# Fallback Provider: Groq Cloud
GROQ_API_KEY = _e("GROQ_API_KEY", "")
GROQ_MODEL = _e("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_BASE_URL = _e("GROQ_BASE_URL", "https://api.groq.com/openai/v1")

# TinyFish Web Search API
TINYFISH_API_KEY = _e("TINYFISH_API_KEY", "")
TINYFISH_SEARCH_URL = _e("TINYFISH_SEARCH_URL", "https://api.search.tinyfish.ai")

# ElevenLabs Flash v2.5 TTS
ELEVENLABS_API_KEY = _e("ELEVENLABS_API_KEY", "")
ELEVENLABS_MODEL_ID = _e("ELEVENLABS_MODEL_ID", "eleven_flash_v2_5")
ELEVENLABS_VOICE_ID = _e("ELEVENLABS_VOICE_ID", "JBFqnCBsd6RMkjVDRZzb")

# Legacy / Generic Provider
LLM_PROVIDER, LLM_API_KEY, LLM_MODEL = _e("LLM_PROVIDER", "openai").lower(), _e("LLM_API_KEY"), _e("LLM_MODEL")
LLM_BASE_URL = _e("LLM_BASE_URL", "https://api.openai.com/v1")

# ── Timeouts & Rate Limits ─────────────────────────────────────────────────────
LLM_TIMEOUT_SECONDS = _float("LLM_TIMEOUT_SECONDS", 8.0)
TTS_TIMEOUT_SECONDS = _float("TTS_TIMEOUT_SECONDS", 30.0)
WEB_SEARCH_TIMEOUT_SECONDS = _float("WEB_SEARCH_TIMEOUT_SECONDS", 6.0)
AGENT_TIMEOUT_SECONDS = _int("AGENT_TIMEOUT_SECONDS", 20)
LLM_RATE_LIMIT_RPM = _int("LLM_RATE_LIMIT_RPM", 60)
API_RATE_LIMIT_RPM = _int("API_RATE_LIMIT_RPM", 120)

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
VECTOR_STORE_PATH.mkdir(parents=True, exist_ok=True)
GENERATED_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


class ConfigValidationError(RuntimeError):
    """Raised when required configuration secrets are unset or unsafe at startup."""


def _is_unsafe_secret(secret: str) -> bool:
    s = (secret or "").strip()
    if not s or s == DEFAULT_INSECURE_JWT_SECRET or len(s) < 16:
        return True
    lowered = s.lower()
    return any(marker in lowered for marker in _PLACEHOLDER_SECRET_MARKERS)


def _has_any_llm_key() -> bool:
    return bool(
        (OPENROUTER_API_KEY or "").strip()
        or (GEMINI_API_KEY or "").strip()
        or (GROQ_API_KEY or "").strip()
        or ((LLM_API_KEY or "").strip() and (LLM_MODEL or "").strip())
    )


def validate_startup_config(strict: bool | None = None) -> dict[str, Any]:
    """Validate configuration at startup.

    Logs clear warnings for missing optional integrations (such as ElevenLabs TTS,
    TinyFish search, PageIndex, or Google OAuth) and fails fast when required
    secrets (JWT_SECRET or an LLM API key) are unset or placeholder values in
    non-test production/staging environments (or when ``strict=True``).
    """
    env_name = (ENVIRONMENT or "development").strip().lower()
    offline_flag = bool(OFFLINE_TEST) or os.getenv("OFFLINE_TEST", "").strip().lower() in ("1", "true", "yes")
    is_test_env = env_name in ("test", "testing") or bool(os.getenv("PYTEST_CURRENT_TEST")) or offline_flag
    is_dev_env = env_name in ("development", "dev", "local")
    enforce_required = strict if strict is not None else (not is_test_env and not is_dev_env)

    warnings: list[str] = []
    errors: list[str] = []

    if _is_unsafe_secret(JWT_SECRET):
        msg = (
            "JWT_SECRET is unset or using an insecure default/placeholder value. "
            "Set a strong random secret in .env before deploying."
        )
        if enforce_required:
            errors.append(msg)
        else:
            warnings.append(msg)

    if not _has_any_llm_key():
        msg = (
            "No LLM API key configured (OPENROUTER_API_KEY, GEMINI_API_KEY, GROQ_API_KEY, or LLM_API_KEY). "
            "Platform will use offline heuristic fallbacks."
        )
        if enforce_required:
            errors.append(
                "No LLM API key configured (OPENROUTER_API_KEY, GEMINI_API_KEY, GROQ_API_KEY, or LLM_API_KEY)."
            )
        else:
            warnings.append(msg)
    elif not (OPENROUTER_API_KEY or "").strip():
        warnings.append(
            "OPENROUTER_API_KEY is unset; primary OpenRouter failover is disabled, using secondary LLM provider."
        )

    # Optional integrations: always emit informative warnings when unset.
    if not (ELEVENLABS_API_KEY or "").strip():
        warnings.append("ELEVENLABS_API_KEY is unset; ElevenLabs TTS audio generation is disabled.")
    if not (TINYFISH_API_KEY or "").strip():
        warnings.append("TINYFISH_API_KEY is unset; external web search enrichment is disabled.")
    if not (PAGEINDEX_API_KEY or "").strip():
        warnings.append("PAGEINDEX_API_KEY is unset; local PDF hierarchy extraction will be used.")
    if not (GOOGLE_CLIENT_ID or "").strip():
        warnings.append("GOOGLE_CLIENT_ID is unset; live Google OAuth token verification is disabled.")
    elif not (GOOGLE_CLIENT_SECRET or "").strip():
        warnings.append("GOOGLE_CLIENT_SECRET is unset alongside GOOGLE_CLIENT_ID.")

    for w in warnings:
        logger.warning(w)

    if errors:
        for err in errors:
            logger.error(err)
        raise ConfigValidationError(
            "Startup configuration validation failed: " + "; ".join(errors)
        )

    return {
        "environment": env_name,
        "is_test": is_test_env,
        "is_dev": is_dev_env,
        "strict": enforce_required,
        "warnings": warnings,
        "errors": errors,
    }


validate_startup_config()
