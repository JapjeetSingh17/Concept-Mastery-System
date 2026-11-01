"""Multi-tier LLM client with automatic failover.

Execution Order:
  1. Primary: Google Gemini (gemini-3.8-flash, auto-falling back to gemini-3.5-flash)
  2. Failover: Groq Cloud (llama-3.3-70b-versatile, auto-falling back to available Groq models)
  3. Offline: Extractive / heuristic pipelines
"""

import logging
from .. import config

logger = logging.getLogger("concept_mastery.llm")

# Track provider health state in memory
_STATE = {
    "gemini_exhausted": False,
    "groq_exhausted": False,
}


def enabled() -> bool:
    """Return True if at least one LLM provider is configured."""
    return bool(config.GEMINI_API_KEY or config.GROQ_API_KEY or (config.LLM_API_KEY and config.LLM_MODEL))


def _call_gemini(system: str, user: str, max_tokens: int = 700) -> str | None:
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
            # 8s timeout for 3.8-flash, fast failover
            timeout = 8.0 if model == config.GEMINI_MODEL and model != "gemini-3.5-flash" else 15.0
            r = httpx.post(url, json=body, timeout=timeout)

            # Check for quota exhaustion / rate limiting
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


def _call_groq(system: str, user: str, max_tokens: int = 700) -> str | None:
    """Call Groq Cloud with requested model and automatic fallback to available models."""
    if not config.GROQ_API_KEY or _STATE["groq_exhausted"]:
        return None

    import httpx

    # Models to try: configured model first, then standard text model
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

            # Check for quota exhaustion / rate limiting
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
                # Model not found on this Groq account, try next in list
                logger.info(f"Groq model {model} not found, trying fallback...")
            else:
                logger.warning(f"Groq {model} returned {r.status_code}: {r.text[:100]}")
        except Exception as e:
            logger.warning(f"Groq {model} request failed: {e}")

    return None


def _call_generic(system: str, user: str, max_tokens: int = 700) -> str | None:
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


def complete(system: str, user: str, max_tokens: int = 700) -> str | None:
    """Generate completion using tiered provider failover.

    Order:
      1. Gemini (primary)
      2. Groq Cloud (automatic failover when Gemini quota ends or fails)
      3. Generic LLM provider
      4. Offline fallback (returns None)
    """
    if not enabled():
        return None

    # Tier 1: Try Gemini
    if config.GEMINI_API_KEY and not _STATE["gemini_exhausted"]:
        res = _call_gemini(system, user, max_tokens)
        if res:
            return res
        print("[LLM Failover] Gemini call did not return content. Switching to Groq Cloud...")

    # Tier 2: Try Groq Cloud
    if config.GROQ_API_KEY and not _STATE["groq_exhausted"]:
        res = _call_groq(system, user, max_tokens)
        if res:
            return res
        print("[LLM Failover] Groq Cloud call did not return content.")

    # Tier 3: Try Generic Provider
    if config.LLM_API_KEY and config.LLM_MODEL:
        res = _call_generic(system, user, max_tokens)
        if res:
            return res

    print("[LLM Failover] All active LLMs exhausted or unavailable. Falling back to offline pipeline.")
    return None


def reset_exhaustion_state():
    """Reset failover flags to allow retrying providers."""
    _STATE["gemini_exhausted"] = False
    _STATE["groq_exhausted"] = False
