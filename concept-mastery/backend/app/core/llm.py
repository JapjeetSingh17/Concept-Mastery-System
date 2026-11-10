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

import json
import logging
import os
import re
import time
from typing import Any, Optional
from .. import config

logger = logging.getLogger("concept_mastery.llm")

# Track provider health state in memory
_STATE = {
    "openrouter_exhausted": False,
    "gemini_exhausted": False,
    "groq_exhausted": False,
}

# In-memory token/credit health tracking per model
# Format: {model_name: {"has_tokens": bool, "cooldown_until": float, "reason": str}}
_MODEL_TOKEN_STATUS: dict[str, dict] = {}
_LAST_CREDITS_CHECK: float = 0.0
_ACCOUNT_HAS_PAID_CREDITS: Optional[bool] = None


def reset_exhaustion_state():
    """Reset provider exhaustion flags and model token health caches."""
    global _LAST_CREDITS_CHECK, _ACCOUNT_HAS_PAID_CREDITS
    _STATE["openrouter_exhausted"] = False
    _STATE["gemini_exhausted"] = False
    _STATE["groq_exhausted"] = False
    _MODEL_TOKEN_STATUS.clear()
    _LAST_CREDITS_CHECK = 0.0
    _ACCOUNT_HAS_PAID_CREDITS = None


def mark_model_token_exhausted(model: str, reason: str = "402 Payment Required / Out of Credits", cooldown_seconds: int = 3600):
    """Mark a model as having 0 tokens/credits left so it is immediately skipped at the start."""
    now = time.time()
    _MODEL_TOKEN_STATUS[model] = {
        "has_tokens": False,
        "cooldown_until": now + cooldown_seconds,
        "reason": reason,
    }
    logger.info(
        f"[OpenRouter Model Verifier] Marked '{model}' as exhausted ({reason}). "
        f"Permanently switching starting model to next in priority chain."
    )


def verify_account_has_credits(timeout: float = 4.0) -> bool:
    """Query OpenRouter credits API to verify if the account has paid credits remaining.
    Caches result for 60 seconds to avoid spamming the credits endpoint.
    """
    global _LAST_CREDITS_CHECK, _ACCOUNT_HAS_PAID_CREDITS
    now = time.time()
    if _ACCOUNT_HAS_PAID_CREDITS is not None and (now - _LAST_CREDITS_CHECK) < 60.0:
        return _ACCOUNT_HAS_PAID_CREDITS

    if is_offline() or os.getenv("PYTEST_CURRENT_TEST"):
        return True

    if not config.OPENROUTER_API_KEY:
        _ACCOUNT_HAS_PAID_CREDITS = False
        return False

    import httpx
    try:
        r = httpx.get(
            f"{config.OPENROUTER_BASE_URL.rstrip('/')}/credits",
            headers={"Authorization": f"Bearer {config.OPENROUTER_API_KEY}"},
            timeout=timeout,
        )
        if r.status_code == 200:
            data = r.json().get("data", {})
            total_credits = float(data.get("total_credits", 0))
            _ACCOUNT_HAS_PAID_CREDITS = total_credits > 0.0001
            _LAST_CREDITS_CHECK = now
            return _ACCOUNT_HAS_PAID_CREDITS
    except Exception as e:
        logger.debug(f"[OpenRouter Verifier] Credits preflight check skipped: {e}")

    return _ACCOUNT_HAS_PAID_CREDITS if _ACCOUNT_HAS_PAID_CREDITS is not None else True


def is_free_model(model_name: str) -> bool:
    """Return True if model slug denotes a known free tier model on OpenRouter."""
    return ":free" in model_name.lower() or "free" in model_name.lower()


def get_verified_candidate_models(models_pool: list[str]) -> list[str]:
    """Verify and order models so that models with active tokens come FIRST.

    If a model (like Model 1) is verified to have 0 tokens/credits, it is moved past
    so the pipeline starts directly with Model 2. If Model 2 also runs out, it repeats
    the same and switches the start to Model 3.
    """
    now = time.time()
    candidates = []
    for m in models_pool:
        if m not in candidates:
            candidates.append(m)
        alias = OPENROUTER_SLUG_ALIASES.get(m)
        if alias and alias not in candidates:
            candidates.append(alias)

    # Preflight check: If OpenRouter account has 0 paid credits, mark paid models as out of tokens
    if not verify_account_has_credits():
        for m in candidates:
            if not is_free_model(m) and m not in _MODEL_TOKEN_STATUS:
                _MODEL_TOKEN_STATUS[m] = {
                    "has_tokens": False,
                    "cooldown_until": now + 3600,
                    "reason": "OpenRouter account balance is $0.00",
                }

    available_models = []
    exhausted_models = []

    for m in candidates:
        st = _MODEL_TOKEN_STATUS.get(m)
        if st and st.get("cooldown_until", 0) > now and not st.get("has_tokens", True):
            exhausted_models.append(m)
        else:
            available_models.append(m)

    # Models with tokens available come first at the start!
    # Any exhausted models are placed at the end as a last-resort fallback.
    return available_models + exhausted_models


OPENROUTER_SLUG_ALIASES = {
    "deepseek/deepseek-chat-v3-0324:free": "deepseek/deepseek-chat",
    "meta-llama/llama-3.3-70b-instruct:free": "meta-llama/llama-3.3-70b-instruct",
    "google/gemini-flash-1.5": "google/gemini-2.5-flash",
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


def repair_json_string(s: str) -> str:
    """Repair common malformations in LLM-generated JSON strings.

    Fixes:
      1. Strips markdown fences (```json ... ```) and reasoning tags (<think>...</think>).
      2. Locates opening object '{' or array '['.
      3. Normalizes Python literals (None -> null, True -> true, False -> false).
      4. Fixes escaped quotes immediately preceding delimiters or closures (e.g. \\"} -> "}).
      5. Fixes unescaped internal double quotes inside string values.
      6. Converts literal newlines and control characters inside strings into valid escape sequences.
      7. Balances unclosed quotes and unclosed curly/square brackets caused by token truncation.
      8. Strips invalid trailing commas before '}' and ']'.
    """
    if not s or not s.strip():
        return ""
    cleaned = sanitize_llm_response(s)

    # 1. Strip markdown fences if present
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned, re.IGNORECASE)
    if fence:
        cleaned = fence.group(1).strip()

    # 2. Slice from first opening { or [
    first_obj = cleaned.find("{")
    first_arr = cleaned.find("[")
    if first_obj != -1 and (first_arr == -1 or first_obj < first_arr):
        cleaned = cleaned[first_obj:]
    elif first_arr != -1:
        cleaned = cleaned[first_arr:]

    # 3. Replace common Python constants if present outside quotes
    cleaned = re.sub(r':\s*None\b', ': null', cleaned)
    cleaned = re.sub(r':\s*True\b', ': true', cleaned)
    cleaned = re.sub(r':\s*False\b', ': false', cleaned)

    # 4. Pre-pass: replace \" before delimiters or closures
    cleaned = re.sub(r'\\\"(\s*[,\}\]])', r'"\1', cleaned)
    # Strip trailing commas before closures
    cleaned = re.sub(r",\s*([\]}])", r"\1", cleaned)

    # 5. Token scanner to balance quotes, escape inner quotes, and close brackets
    in_string = False
    escape = False
    stack = []
    repaired_chars = []

    i = 0
    n = len(cleaned)
    while i < n:
        c = cleaned[i]
        if in_string:
            if escape:
                escape = False
                repaired_chars.append(c)
            elif c == "\\":
                escape = True
                repaired_chars.append(c)
            elif c == '"':
                # Check if this quote closes the string
                rest = cleaned[i + 1:].lstrip()
                if not rest or rest[0] in ":,}]":
                    in_string = False
                    repaired_chars.append(c)
                else:
                    # Inner unescaped quote -> escape it
                    repaired_chars.append('\\"')
            elif c == "\n":
                repaired_chars.append("\\n")
            elif c == "\r":
                pass
            elif c == "\t":
                repaired_chars.append("\\t")
            else:
                repaired_chars.append(c)
        else:
            if c == '"':
                in_string = True
                repaired_chars.append(c)
            elif c in "{[":
                stack.append(c)
                repaired_chars.append(c)
            elif c == "}":
                if stack and stack[-1] == "{":
                    stack.pop()
                    repaired_chars.append(c)
            elif c == "]":
                if stack and stack[-1] == "[":
                    stack.pop()
                    repaired_chars.append(c)
            else:
                repaired_chars.append(c)
        i += 1

    result = "".join(repaired_chars)

    # If still in string at end of text (e.g. truncated generation), terminate string
    if in_string:
        while result.endswith("\\"):
            result = result[:-1]
        result += '"'

    result = re.sub(r",\s*$", "", result.strip())

    # Close remaining open brackets
    while stack:
        b = stack.pop()
        if b == "{":
            result += "}"
        elif b == "[":
            result += "]"

    result = re.sub(r",\s*([\]}])", r"\1", result)
    return result


def _regex_extract_json_dict(text: str) -> Optional[dict]:
    """Fallback extraction of top-level key-values from partially mangled JSON strings."""
    res = {}
    for m in re.finditer(r'["\']([a-zA-Z0-9_]+)["\']\s*:\s*["\']([^"\']*)["\']', text):
        res[m.group(1)] = m.group(2)
    for m in re.finditer(r'["\']([a-zA-Z0-9_]+)["\']\s*:\s*\[(.*?)\]', text, flags=re.DOTALL):
        k = m.group(1)
        raw_list = m.group(2)
        items = re.findall(r'["\']([^"\']+)["\']', raw_list)
        res[k] = items
    return res if res else None


def safe_json_loads(text: str) -> Any:
    """Safely parse JSON from LLM output, applying tiered repair passes if standard parsing fails."""
    if not text or not text.strip():
        return None

    cleaned = sanitize_llm_response(text)
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned, re.IGNORECASE)
    candidate = fence.group(1).strip() if fence else cleaned.strip()

    # Pass 1: Direct strict=False json.loads (fast path for well-formed JSON)
    try:
        return json.loads(candidate, strict=False)
    except Exception:
        pass

    # Pass 2: Syntactic repair (escaped quotes, unclosed strings, truncated braces, trailing commas)
    try:
        repaired = repair_json_string(cleaned)
        if repaired:
            return json.loads(repaired, strict=False)
    except Exception:
        pass

    # Pass 3: Python AST literal evaluation (if LLM returned Python-formatted dict or list)
    try:
        import ast
        data = ast.literal_eval(candidate)
        if isinstance(data, (dict, list)):
            return data
    except Exception:
        pass

    # Pass 4: Regex-based recovery for dictionary fields
    try:
        extracted = _regex_extract_json_dict(cleaned)
        if extracted:
            return extracted
    except Exception:
        pass

    return None


def extract_json_payload(text: str) -> str:
    """Sanitize output, remove markdown code fences, and extract repaired raw JSON object or array."""
    if not text:
        return ""
    repaired = repair_json_string(text)
    if repaired:
        return repaired
    return sanitize_llm_response(text).strip()


def _call_openrouter(system: str, user: str, max_tokens: int = 700) -> Optional[str]:
    """Call OpenRouter Gateway with automated per-model client-side failover."""
    if not config.OPENROUTER_API_KEY or _STATE["openrouter_exhausted"]:
        return None

    import httpx

    headers = {
        "Authorization": f"Bearer {config.OPENROUTER_API_KEY}",
        "HTTP-Referer": "http://localhost:8000",
        "X-Title": "Closed-Loop Concept Mastery Platform",
        "Content-Type": "application/json",
    }

    models_pool = list(config.OPENROUTER_MODELS) if config.OPENROUTER_MODELS else [config.OPENROUTER_PRIMARY_MODEL]

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})
    is_json = "json" in (system + " " + user).lower()

    # Use verified candidate models so models with active tokens start FIRST.
    # If Model 1 has 0 tokens, Model 2 starts first; if Model 2 runs out, Model 3 starts, repeating down the chain.
    candidate_models = get_verified_candidate_models(models_pool)

    for idx, model in enumerate(candidate_models, 1):
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.4,
        }
        if is_json:
            payload["response_format"] = {"type": "json_object"}

        for attempt in (1, 2):
            try:
                # Per-model timeout: 8s so a hanging model doesn't stall the system
                r = httpx.post(
                    f"{config.OPENROUTER_BASE_URL.rstrip('/')}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=8.0,
                )

                if r.status_code == 200:
                    data = r.json()
                    choices = data.get("choices", [])
                    if choices and "message" in choices[0]:
                        msg = choices[0]["message"]
                        content = msg.get("content") or msg.get("reasoning") or ""
                        if content:
                            resolved = data.get("model", model)
                            logger.info(f"OpenRouter answered successfully via model: {resolved}")
                            return content.strip()

                # If 400 Bad Request, check if it's due to response_format
                if r.status_code == 400 and attempt == 1 and "response_format" in payload:
                    logger.info(f"OpenRouter model {model} rejected response_format; retrying without it...")
                    payload.pop("response_format", None)
                    continue

                # Handle 402 Payment Required / Insufficient credits
                if r.status_code == 402:
                    mark_model_token_exhausted(model, reason="402 Payment Required / Credit Exhausted", cooldown_seconds=3600)
                    logger.warning(
                        f"[OpenRouter Failover] Model '{model}' returned 402 (Payment Required / Credit Exhausted). "
                        f"Permanently switching starting model to next in priority chain..."
                    )
                    break

                # Handle 404 Model Not Found / Retired
                if r.status_code == 404:
                    mark_model_token_exhausted(model, reason="404 Model Retired / Unavailable", cooldown_seconds=86400)
                    logger.warning(
                        f"[OpenRouter Failover] Model '{model}' returned 404 (Retired/Unavailable). "
                        f"Permanently switching starting model to next in priority chain..."
                    )
                    break

                # Handle 429/403 Rate Limit / Quota Exceeded
                if r.status_code in (429, 403) or "rate_limit" in r.text.lower():
                    mark_model_token_exhausted(model, reason=f"HTTP {r.status_code} Rate Limit / Quota Exceeded", cooldown_seconds=600)
                    logger.warning(
                        f"[OpenRouter Failover] Model '{model}' returned {r.status_code} (Rate Limit). "
                        f"Automatically switching starting model to next in priority chain..."
                    )
                    break

                # Other HTTP errors (5xx, etc.)
                logger.warning(
                    f"[OpenRouter Failover] Model '{model}' returned HTTP {r.status_code}: {r.text[:120]}. "
                    f"Switching to next model..."
                )
                break

            except httpx.TimeoutException:
                mark_model_token_exhausted(model, reason="Timeout after 8.0s", cooldown_seconds=300)
                logger.warning(
                    f"[OpenRouter Failover] Model '{model}' timed out after 8.0s. "
                    f"Automatically switching starting model to next in priority chain..."
                )
                break
            except Exception as e:
                logger.warning(
                    f"[OpenRouter Failover] Model '{model}' request failed ({e}). "
                    f"Switching to next model..."
                )
                break

    logger.warning("All OpenRouter models failed or exhausted. Cascading to downstream providers...")
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


# Public aliases for testing & compatibility
call_openrouter = _call_openrouter
generate = complete
