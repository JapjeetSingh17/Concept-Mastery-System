"""Multi-tier LLM client with OpenRouter gateway, native auto-failover, and offline fallback.

Execution Order:
  1. Primary Gateway: OpenRouter (with native fallback chain:
     Gemini 2.5 Flash -> NVIDIA Nemotron 3 Ultra -> DeepSeek Chat ->
     NVIDIA Nemotron 3.5 Super VL -> Llama 3.3 70B -> OpenRouter Auto)
  2. Legacy Direct: Google Gemini (gemini-3.8-flash, auto-falling back to gemini-3.5-flash)
  3. Legacy Failover: Groq Cloud (llama-3.3-70b-versatile, auto-falling back to available Groq models)
  4. Generic / Custom OpenAI-compatible endpoints
  5. Offline: Extractive / heuristic deterministic pipelines
"""

import logging
import os
import re
from typing import Optional
from .. import config

logger = logging.getLogger("concept_mastery.llm")

# Track provider health state in memory
_STATE = {
    "openrouter_exhausted": False,
    "gemini_exhausted": False,
    "groq_exhausted": False,
}


def is_offline() -> bool:
    """Return True if offline test mode is active."""
    if os.getenv("OPENROUTER_LIVE"):
        return False
    if os.getenv("OFFLINE_TEST", "").lower() in ("1", "true", "yes"):
        return True
    if os.getenv("PYTEST_CURRENT_TEST"):
        return True
    return False


def enabled() -> bool:
    """Return True if at least one LLM provider is configured and offline mode is not active."""
    if is_offline():
        return False
    return bool(
        config.OPENROUTER_API_KEY
        or config.GEMINI_API_KEY
        or config.GROQ_API_KEY
        or (config.LLM_API_KEY and config.LLM_MODEL)
    )


def sanitize_llm_response(text: str) -> str:
    """Sanitize LLM output by removing reasoning/thinking tags and extra whitespace."""
    if not text:
        return ""
    # Strip <think>...</think>, <thought>...</thought>, <reasoning>...</reasoning>
    cleaned = re.sub(r"<(think|thought|reasoning)>.*?</\1>", "", text, flags=re.DOTALL | re.IGNORECASE)
    # Strip unclosed thinking tags (in case max_tokens cut off generation during thinking)
    cleaned = re.sub(r"<(think|thought|reasoning)>.*$", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
    return cleaned.strip()


def extract_json_payload(text: str) -> str:
    """Sanitize output, remove markdown code fences, and extract raw JSON object or array."""
    if not text:
        return ""
    cleaned = sanitize_llm_response(text)

    # Strip markdown code fences if present (```json ... ``` or ``` ... ```)
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned, re.IGNORECASE)
    if fence_match:
        cleaned = fence_match.group(1).strip()

    # Locate outermost JSON structure ({...} or [...])
    first_obj = cleaned.find("{")
    first_arr = cleaned.find("[")

    if first_obj != -1 and (first_arr == -1 or first_obj < first_arr):
        last_obj = cleaned.rfind("}")
        if last_obj != -1 and last_obj > first_obj:
            return cleaned[first_obj:last_obj + 1].strip()
    elif first_arr != -1:
        last_arr = cleaned.rfind("]")
        if last_arr != -1 and last_arr > first_arr:
            return cleaned[first_arr:last_arr + 1].strip()

    return cleaned.strip()


def _call_openrouter(system: str, user: str, max_tokens: int = 700) -> Optional[str]:
    """Call OpenRouter Gateway with native auto-failover routing and model batching."""
    if not config.OPENROUTER_API_KEY or _STATE["openrouter_exhausted"]:
        return None

    import httpx

    headers = {
        "Authorization": f"Bearer {config.OPENROUTER_API_KEY}",
        "HTTP-Referer": "http://localhost:8000",
        "X-Title": "Closed-Loop Concept Mastery Platform",
        "Content-Type": "application/json",
    }

    models_pool = config.OPENROUTER_MODELS if config.OPENROUTER_MODELS else [config.OPENROUTER_PRIMARY_MODEL]
    # OpenRouter limits 'models' array in fallback route to <= 3 items.
    # We partition OPENROUTER_MODELS into sequential fallback batches of <= 3.
    batches = [models_pool[i:i + 3] for i in range(0, len(models_pool), 3)]

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})

    for batch_idx, batch in enumerate(batches, 1):
        payload = {
            "model": batch[0],
            "models": batch,
            "route": "fallback",
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.4,
        }

        try:
            r = httpx.post(
                f"{config.OPENROUTER_BASE_URL.rstrip('/')}/chat/completions",
                headers=headers,
                json=payload,
                timeout=25.0,
            )

            # Check for quota exhaustion / rate limiting
            if r.status_code in (429, 403) or "rate_limit_exceeded" in r.text:
                logger.warning(
                    f"OpenRouter rate limit / quota exceeded on batch {batch_idx} ({r.status_code})."
                )
                continue

            if r.status_code == 200:
                data = r.json()
                choices = data.get("choices", [])
                if choices and "message" in choices[0]:
                    content = choices[0]["message"].get("content", "")
                    if content:
                        logger.info(
                            f"OpenRouter answered successfully via model: {data.get('model', batch[0])}"
                        )
                        return content.strip()
            else:
                logger.warning(
                    f"OpenRouter batch {batch_idx} returned {r.status_code}: {r.text[:120]}"
                )
        except Exception as e:
            logger.warning(f"OpenRouter batch {batch_idx} request failed ({e})")

    logger.warning("All OpenRouter model fallback batches failed. Marking OpenRouter as exhausted.")
    _STATE["openrouter_exhausted"] = True
    return None


def _call_gemini(system: str, user: str, max_tokens: int = 700) -> Optional[str]:
    """Call Google Gemini REST API."""
    if not config.GEMINI_API_KEY or _STATE["gemini_exhausted"]:
        return None

    import httpx

    # Models to try: primary first, followed by stable 3.5-flash
    models = [config.GEMINI_MODEL]
    if config.GEMINI_MODEL != "gemini-3.5-flash":
        models.append("gemini-3.5-flash")

    for model in models:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={config.GEMINI_API_KEY}"
        body = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"parts": [{"text": user}]}],
            "generationConfig": {"maxOutputTokens": max_tokens, "temperature": 0.4},
        }

        try:
            timeout = 8.0 if model == config.GEMINI_MODEL and model != "gemini-3.5-flash" else 15.0
            r = httpx.post(url, json=body, timeout=timeout)

            if r.status_code in (429, 403) or "RESOURCE_EXHAUSTED" in r.text:
                logger.warning(f"Gemini quota/limit reached ({r.status_code}). Marking Gemini as exhausted.")
                _STATE["gemini_exhausted"] = True
                return None

            if r.status_code == 200:
                data = r.json()
                candidates = data.get("candidates", [])
                if candidates:
                    parts = candidates[0].get("content", {}).get("parts", [])
                    if parts and "text" in parts[0]:
                        return parts[0]["text"].strip()
            else:
                logger.info(f"Gemini {model} returned {r.status_code}, trying next option...")
        except Exception as e:
            logger.info(f"Gemini {model} call timed out or failed ({e}), trying next option...")

    return None


def _call_groq(system: str, user: str, max_tokens: int = 700) -> Optional[str]:
    """Call Groq Cloud with requested model and automatic fallback to available models."""
    if not config.GROQ_API_KEY or _STATE["groq_exhausted"]:
        return None

    import httpx

    models = [config.GROQ_MODEL]
    if config.GROQ_MODEL != "openai/gpt-oss-120b":
        models.append("openai/gpt-oss-120b")
    if "qwen/qwen3.8-27b" not in models:
        models.append("qwen/qwen3.8-27b")

    for model in models:
        try:
            r = httpx.post(
                f"{config.GROQ_BASE_URL.rstrip('/')}/chat/completions",
                headers={"Authorization": f"Bearer {config.GROQ_API_KEY}"},
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    "max_tokens": max_tokens,
                    "temperature": 0.4,
                },
                timeout=20.0,
            )

            if r.status_code in (429, 403) or "rate_limit_exceeded" in r.text:
                logger.warning(f"Groq quota/rate limit reached ({r.status_code}). Marking Groq as exhausted.")
                _STATE["groq_exhausted"] = True
                return None

            if r.status_code == 200:
                data = r.json()
                choices = data.get("choices", [])
                if choices and "message" in choices[0]:
                    return choices[0]["message"]["content"].strip()
            elif r.status_code == 404:
                logger.info(f"Groq model {model} not found, trying fallback...")
            else:
                logger.warning(f"Groq {model} returned {r.status_code}: {r.text[:100]}")
        except Exception as e:
            logger.warning(f"Groq {model} request failed: {e}")

    return None


def _call_generic(system: str, user: str, max_tokens: int = 700) -> Optional[str]:
    """Generic fallback for any custom OpenAI/Anthropic endpoint."""
    if not (config.LLM_API_KEY and config.LLM_MODEL):
        return None

    import httpx

    try:
        if config.LLM_PROVIDER == "anthropic":
            r = httpx.post(
                "https://api.anthropic.com/v1/messages",
                timeout=30,
                headers={"x-api-key": config.LLM_API_KEY, "anthropic-version": "2023-06-01"},
                json={
                    "model": config.LLM_MODEL,
                    "max_tokens": max_tokens,
                    "system": system,
                    "messages": [{"role": "user", "content": user}],
                },
            )
            r.raise_for_status()
            return r.json()["content"][0]["text"].strip()

        r = httpx.post(
            config.LLM_BASE_URL.rstrip("/") + "/chat/completions",
            timeout=30,
            headers={"Authorization": f"Bearer {config.LLM_API_KEY}"},
            json={
                "model": config.LLM_MODEL,
                "max_tokens": max_tokens,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            },
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        logger.warning(f"Generic LLM call failed: {e}")
        return None


def complete(system: str, user: str, max_tokens: int = 700) -> Optional[str]:
    """Generate completion using tiered provider failover.

    Execution Order:
      1. OpenRouter Gateway (Gemini 2.5 Flash -> NVIDIA Nemotron 3 -> DeepSeek -> NVIDIA 3.5 -> Llama 3.3 -> Auto)
      2. Google Gemini direct
      3. Groq Cloud
      4. Generic LLM provider
      5. Offline deterministic fallback (returns None)
    """
    if not enabled():
        return None

    # Tier 1: Try OpenRouter
    if config.OPENROUTER_API_KEY and not _STATE["openrouter_exhausted"]:
        res = _call_openrouter(system, user, max_tokens)
        if res:
            return sanitize_llm_response(res)
        logger.info("[LLM Failover] OpenRouter call did not return content. Switching to Gemini direct...")

    # Tier 2: Try Gemini direct
    if config.GEMINI_API_KEY and not _STATE["gemini_exhausted"]:
        res = _call_gemini(system, user, max_tokens)
        if res:
            return sanitize_llm_response(res)
        logger.info("[LLM Failover] Gemini call did not return content. Switching to Groq Cloud...")

    # Tier 3: Try Groq Cloud
    if config.GROQ_API_KEY and not _STATE["groq_exhausted"]:
        res = _call_groq(system, user, max_tokens)
        if res:
            return sanitize_llm_response(res)
        logger.info("[LLM Failover] Groq Cloud call did not return content.")

    # Tier 4: Try Generic Provider
    if config.LLM_API_KEY and config.LLM_MODEL:
        res = _call_generic(system, user, max_tokens)
        if res:
            return sanitize_llm_response(res)

    logger.info("[LLM Failover] All active LLMs exhausted or unavailable. Falling back to offline pipeline.")
    return None


def reset_exhaustion_state():
    """Reset failover flags to allow retrying providers."""
    _STATE["openrouter_exhausted"] = False
    _STATE["gemini_exhausted"] = False
    _STATE["groq_exhausted"] = False
