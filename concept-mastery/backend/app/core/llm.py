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
import math
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

    if is_offline() or (os.getenv("PYTEST_CURRENT_TEST") and not os.getenv("OPENROUTER_LIVE")):
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


_NONFINITE_JSON_STRINGS = frozenset({"nan", "+nan", "-nan", "inf", "+inf", "-inf", "infinity", "+infinity", "-infinity"})

# Schema-defined numeric bounds: (min_val, max_val, non_finite_default)
_SCHEMA_FLOAT_BOUNDS: dict[str, tuple[Optional[float], Optional[float], float]] = {
    "score": (0.0, 1.0, 0.0),
    "retrieval_score": (0.0, 1.0, 0.0),
    "semantic_comprehension_score": (0.0, 1.0, 0.0),
    "quiz_accuracy": (0.0, 1.0, 0.0),
    "dialogue_score": (0.0, 1.0, 0.0),
    "prereq_health": (0.0, 1.0, 0.0),
    "overall_score": (0.0, 1.0, 0.0),
    "reasoning_clarity": (0.0, 1.0, 0.0),
    "mastery_score": (0.0, 1.0, 0.0),
    "confidence_score": (0.0, 1.0, 0.0),
    "confidence": (0.0, 1.0, 0.0),
    "tolerance": (0.0, None, 0.0),
    "correct_numeric_answer": (None, None, 0.0),
    "numeric_answer": (None, None, 0.0),
    "numericAnswer": (None, None, 0.0),
    "x": (None, None, 0.0),
    "y": (None, None, 0.0),
}

_UNQUOTED_LITERAL_RE = re.compile(
    r"(?:None|True|False|null|[+-]?(?:infinity|inf|nan))(?![A-Za-z0-9_])",
    re.IGNORECASE,
)


def clamp_float(
    value: Any,
    min_val: Optional[float] = 0.0,
    max_val: Optional[float] = 1.0,
    default: float = 0.0,
) -> float:
    """Convert ``value`` to a finite float clamped within ``[min_val, max_val]``.

    Non-finite values (``NaN``, ``+inf``, ``-inf``) are mapped into ``[0.0, 1.0]``
    (or the supplied bounds when tighter) so downstream Pydantic models and
    FastAPI JSON encoders never reject out-of-range floats.
    """
    fallback = default
    if min_val is not None:
        fallback = max(min_val, fallback)
    if max_val is not None:
        fallback = min(max_val, fallback)

    if isinstance(value, bool) or value is None:
        return fallback

    try:
        num = float(value)
    except (TypeError, ValueError):
        return fallback

    if math.isnan(num):
        return fallback
    if math.isinf(num):
        if num > 0:
            upper = max_val if max_val is not None else 1.0
            return max(min_val, upper) if min_val is not None else upper
        lower = min_val if min_val is not None else 0.0
        return min(max_val, lower) if max_val is not None else lower

    if min_val is not None and num < min_val:
        return min_val
    if max_val is not None and num > max_val:
        return max_val
    return num


def _parse_nonfinite_json_constant(const: str) -> float:
    """Intercept non-standard JSON NaN/Infinity/-Infinity tokens at parse time."""
    lower = const.strip().lower()
    if "nan" in lower or lower.startswith("-"):
        return 0.0
    return 1.0


def _try_sanitize_schema_value(
    value: Any,
    bounds: tuple[Optional[float], Optional[float], float],
) -> Any:
    """Sanitize numeric or stringified-float values for schema-bounded fields."""
    lo, hi, default = bounds
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, float):
        return clamp_float(value, lo, hi, default)
    if isinstance(value, int):
        if lo is not None and value < lo:
            return lo
        if hi is not None and value > hi:
            return hi
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.lower() in _NONFINITE_JSON_STRINGS:
            return clamp_float(stripped, lo, hi, default)
        try:
            parsed_num = float(stripped)
            return clamp_float(parsed_num, lo, hi, default)
        except ValueError:
            return value
    return value


def sanitize_json_floats(
    data: Any,
    *,
    field_bounds: Optional[dict[str, tuple[Optional[float], Optional[float], float]]] = None,
    default_bounds: tuple[float, float] = (0.0, 1.0),
    clamp_all_floats: bool = False,
) -> Any:
    """Recursively sanitize floats in parsed JSON structures.

    - Converts any ``NaN``, ``inf``, or ``-inf`` float anywhere in dicts/lists to
      finite values within ``default_bounds`` (``[0.0, 1.0]`` by default).
    - Clamps known schema-bounded fields (e.g. ``retrieval_score``, ``score``,
      ``tolerance``) using ``field_bounds`` or ``_SCHEMA_FLOAT_BOUNDS``.
    """
    bounds_map = field_bounds if field_bounds is not None else _SCHEMA_FLOAT_BOUNDS
    lo_def, hi_def = default_bounds

    if isinstance(data, bool) or data is None:
        return data
    if isinstance(data, float):
        if math.isnan(data) or math.isinf(data) or clamp_all_floats:
            return clamp_float(data, lo_def, hi_def, lo_def)
        return data
    if isinstance(data, dict):
        sanitized: dict[Any, Any] = {}
        for k, v in data.items():
            if isinstance(k, str) and k in bounds_map and not isinstance(v, (dict, list, tuple)):
                sanitized[k] = _try_sanitize_schema_value(v, bounds_map[k])
            else:
                sanitized[k] = sanitize_json_floats(
                    v,
                    field_bounds=bounds_map,
                    default_bounds=default_bounds,
                    clamp_all_floats=clamp_all_floats,
                )
        return sanitized
    if isinstance(data, (list, tuple)):
        return [
            sanitize_json_floats(
                item,
                field_bounds=bounds_map,
                default_bounds=default_bounds,
                clamp_all_floats=clamp_all_floats,
            )
            for item in data
        ]
    return data


def _strip_markdown_fences(text: str) -> str:
    """Remove surrounding ```json / ``` markdown code fences if present."""
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
    if fence:
        return fence.group(1).strip()
    stripped = re.sub(r"^\s*```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    return re.sub(r"\s*```\s*$", "", stripped).strip()


def _replace_unquoted_json_literals(text: str, *, python_literals: bool = False) -> str:
    """Replace Python/non-standard literals (None/True/False/NaN/Infinity) outside quotes."""
    out: list[str] = []
    quote_char: Optional[str] = None
    escape = False
    last_non_ws = ""
    i = 0
    n = len(text)

    while i < n:
        ch = text[i]
        if quote_char is not None:
            out.append(ch)
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == quote_char:
                quote_char = None
                last_non_ws = ch
            i += 1
            continue

        if ch in ('"', "'"):
            quote_char = ch
            out.append(ch)
            last_non_ws = ch
            i += 1
            continue

        if last_non_ws in ("", ":", ",", "["):
            m = _UNQUOTED_LITERAL_RE.match(text, i)
            if m:
                tok = m.group(0)
                lower = tok.lower()
                if python_literals:
                    if lower in ("none", "null"):
                        repl = "None"
                    elif lower == "true":
                        repl = "True"
                    elif lower == "false":
                        repl = "False"
                    elif lower in _NONFINITE_JSON_STRINGS:
                        repl = "0.0" if ("nan" in lower or lower.startswith("-")) else "1.0"
                    else:
                        repl = None
                else:
                    if tok == "None" or lower == "null":
                        repl = "null"
                    elif tok == "True" or lower == "true":
                        repl = "true"
                    elif tok == "False" or lower == "false":
                        repl = "false"
                    elif lower in _NONFINITE_JSON_STRINGS:
                        repl = "0.0" if ("nan" in lower or lower.startswith("-")) else "1.0"
                    else:
                        repl = None
                if repl is not None:
                    out.append(repl)
                    last_non_ws = repl[-1]
                    i += len(tok)
                    continue

        out.append(ch)
        if not ch.isspace():
            last_non_ws = ch
        i += 1

    return "".join(out)


def _strip_trailing_commas(text: str) -> str:
    """Remove trailing commas before '}' or ']' outside string literals."""
    out: list[str] = []
    quote_char: Optional[str] = None
    escape = False
    i = 0
    n = len(text)

    while i < n:
        ch = text[i]
        if quote_char is not None:
            out.append(ch)
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == quote_char:
                quote_char = None
            i += 1
            continue

        if ch in ('"', "'"):
            quote_char = ch
            out.append(ch)
            i += 1
            continue

        if ch == ",":
            j = i + 1
            while j < n and text[j].isspace():
                j += 1
            if j >= n or text[j] in ("}", "]", ","):
                i += 1
                continue

        out.append(ch)
        i += 1

    return "".join(out)


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
      3. Normalizes Python literals (None -> null, True -> true, False -> false)
         and non-finite float tokens (NaN/nan/-Infinity/-inf -> 0.0, Infinity/inf -> 1.0)
         strictly outside string literals.
      4. Fixes escaped quotes immediately preceding delimiters or closures (e.g. \\"} -> "}).
      5. Fixes unescaped internal double quotes inside string values.
      6. Converts literal newlines and control characters inside strings into valid escape sequences.
      7. Balances unclosed quotes and unclosed curly/square brackets caused by token truncation.
      8. Strips invalid trailing commas before '}' and ']' outside strings.
    """
    if not s or not s.strip():
        return ""
    cleaned = sanitize_llm_response(s)

    # 1. Strip markdown fences if present
    cleaned = _strip_markdown_fences(cleaned)

    # 2. Slice from first opening { or [
    first_obj = cleaned.find("{")
    first_arr = cleaned.find("[")
    if first_obj != -1 and (first_arr == -1 or first_obj < first_arr):
        cleaned = cleaned[first_obj:]
    elif first_arr != -1:
        cleaned = cleaned[first_arr:]

    # 3. Replace Python constants and non-finite float tokens outside quoted strings
    cleaned = _replace_unquoted_json_literals(cleaned)

    # 4. Pre-pass: replace \" before delimiters or closures and strip trailing commas
    cleaned = re.sub(r'\\\"(\s*[,\}\]])', r'"\1', cleaned)
    cleaned = _strip_trailing_commas(cleaned)

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

    return _strip_trailing_commas(result)


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


def parse_and_sanitize_json(
    text: str,
    *,
    field_bounds: Optional[dict[str, tuple[Optional[float], Optional[float], float]]] = None,
    default_bounds: tuple[float, float] = (0.0, 1.0),
    clamp_all_floats: bool = False,
) -> Any:
    """Parse LLM JSON safely with fence stripping, syntax repair, and float sanitization."""
    if not text or not text.strip():
        return None

    cleaned = sanitize_llm_response(text)
    candidate = _strip_markdown_fences(cleaned)

    def _finish(parsed_obj: Any) -> Any:
        return sanitize_json_floats(
            parsed_obj,
            field_bounds=field_bounds,
            default_bounds=default_bounds,
            clamp_all_floats=clamp_all_floats,
        )

    # Pass 1: Direct strict=False json.loads with non-finite constant clamping
    try:
        return _finish(
            json.loads(candidate, strict=False, parse_constant=_parse_nonfinite_json_constant)
        )
    except Exception:
        pass

    # Pass 2: Syntactic repair (escaped quotes, unclosed strings, truncated braces, trailing commas, non-finite tokens)
    try:
        repaired = repair_json_string(cleaned)
        if repaired:
            return _finish(
                json.loads(repaired, strict=False, parse_constant=_parse_nonfinite_json_constant)
            )
    except Exception:
        pass

    # Pass 3: Python AST literal evaluation (if LLM returned Python-formatted dict or list)
    try:
        import ast
        literal_candidate = _strip_trailing_commas(
            _replace_unquoted_json_literals(candidate, python_literals=True)
        )
        data = ast.literal_eval(literal_candidate)
        if isinstance(data, (dict, list)):
            return _finish(data)
    except Exception:
        pass

    # Pass 4: Regex-based recovery for dictionary fields
    try:
        extracted = _regex_extract_json_dict(cleaned)
        if extracted:
            return _finish(extracted)
    except Exception:
        pass

    return None


def safe_json_loads(text: str) -> Any:
    """Safely parse JSON from LLM output, applying tiered repair passes and float sanitization."""
    return parse_and_sanitize_json(text)


def extract_json_payload(text: str) -> str:
    """Sanitize output, remove markdown code fences, and extract repaired raw JSON object or array."""
    if not text:
        return ""
    repaired = repair_json_string(text)
    if repaired:
        return repaired
    return sanitize_llm_response(text).strip()


def _call_openrouter(system: str, user: str, max_tokens: int = 700, timeout: Optional[float] = None) -> Optional[str]:
    """Call OpenRouter Gateway with automated per-model client-side failover."""
    if not config.OPENROUTER_API_KEY or _STATE["openrouter_exhausted"]:
        return None

    import httpx

    start_t = time.time()
    per_model_timeout = min(timeout, 8.0) if timeout is not None else 8.0

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
        if timeout is not None and (timeout - (time.time() - start_t)) < 1.0:
            logger.info(f"[OpenRouter] Insufficient time remaining (<1s); aborting model candidates.")
            break

        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.4,
        }
        if is_json:
            payload["response_format"] = {"type": "json_object"}

        for attempt in (1, 2):
            if timeout is not None and (timeout - (time.time() - start_t)) < 1.0:
                break
            try:
                cur_rem = timeout - (time.time() - start_t) if timeout is not None else per_model_timeout
                cur_timeout = min(per_model_timeout, cur_rem) if timeout is not None else per_model_timeout

                r = httpx.post(
                    f"{config.OPENROUTER_BASE_URL.rstrip('/')}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=cur_timeout,
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
                mark_model_token_exhausted(model, reason="Timeout", cooldown_seconds=300)
                logger.warning(
                    f"[OpenRouter Failover] Model '{model}' timed out. "
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


def _call_gemini(system: str, user: str, max_tokens: int = 700, timeout: Optional[float] = None) -> Optional[str]:
    """Call Google Gemini REST API."""
    if not config.GEMINI_API_KEY or _STATE["gemini_exhausted"]:
        return None

    import httpx

    start_t = time.time()
    # Models to try: primary first, followed by stable 3.5-flash
    models = [config.GEMINI_MODEL]
    if config.GEMINI_MODEL != "gemini-3.5-flash":
        models.append("gemini-3.5-flash")

    for model in models:
        if timeout is not None and (timeout - (time.time() - start_t)) < 1.0:
            break

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={config.GEMINI_API_KEY}"
        body = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"parts": [{"text": user}]}],
            "generationConfig": {"maxOutputTokens": max_tokens, "temperature": 0.4},
        }

        try:
            base_timeout = 8.0 if model == config.GEMINI_MODEL and model != "gemini-3.5-flash" else 15.0
            cur_rem = timeout - (time.time() - start_t) if timeout is not None else base_timeout
            call_timeout = min(base_timeout, cur_rem) if timeout is not None else base_timeout
            r = httpx.post(url, json=body, timeout=call_timeout)

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


def _call_groq(system: str, user: str, max_tokens: int = 700, timeout: Optional[float] = None) -> Optional[str]:
    """Call Groq Cloud with requested model and automatic fallback to available models."""
    if not config.GROQ_API_KEY or _STATE["groq_exhausted"]:
        return None

    import httpx

    start_t = time.time()
    models = [config.GROQ_MODEL]
    if config.GROQ_MODEL != "openai/gpt-oss-120b":
        models.append("openai/gpt-oss-120b")
    if "qwen/qwen3.8-27b" not in models:
        models.append("qwen/qwen3.8-27b")

    for model in models:
        if timeout is not None and (timeout - (time.time() - start_t)) < 1.0:
            break

        try:
            cur_rem = timeout - (time.time() - start_t) if timeout is not None else 20.0
            call_timeout = min(20.0, cur_rem) if timeout is not None else 20.0
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
                timeout=call_timeout,
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


def _call_generic(system: str, user: str, max_tokens: int = 700, timeout: Optional[float] = None) -> Optional[str]:
    """Generic fallback for any custom OpenAI/Anthropic endpoint."""
    if not (config.LLM_API_KEY and config.LLM_MODEL):
        return None

    import httpx

    start_t = time.time()
    cur_rem = timeout - (time.time() - start_t) if timeout is not None else 30.0
    if timeout is not None and cur_rem < 1.0:
        return None
    call_timeout = min(30.0, cur_rem) if timeout is not None else 30.0

    try:
        if config.LLM_PROVIDER == "anthropic":
            r = httpx.post(
                "https://api.anthropic.com/v1/messages",
                timeout=call_timeout,
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
            timeout=call_timeout,
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


def complete(system: str, user: str, max_tokens: int = 700, timeout: Optional[float] = None) -> Optional[str]:
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

    start_t = time.time()
    deadline = start_t + timeout if timeout is not None else None

    # Tier 1: Try OpenRouter
    if config.OPENROUTER_API_KEY and not _STATE["openrouter_exhausted"]:
        rem = deadline - time.time() if deadline is not None else None
        res = _call_openrouter(system, user, max_tokens, timeout=rem)
        if res:
            return sanitize_llm_response(res)
        logger.info("[LLM Failover] OpenRouter call did not return content. Switching to Gemini direct...")

    if deadline is not None and (deadline - time.time()) < 1.0:
        return None

    # Tier 2: Try Gemini direct
    if config.GEMINI_API_KEY and not _STATE["gemini_exhausted"]:
        rem = deadline - time.time() if deadline is not None else None
        res = _call_gemini(system, user, max_tokens, timeout=rem)
        if res:
            return sanitize_llm_response(res)
        logger.info("[LLM Failover] Gemini call did not return content. Switching to Groq Cloud...")

    if deadline is not None and (deadline - time.time()) < 1.0:
        return None

    # Tier 3: Try Groq Cloud
    if config.GROQ_API_KEY and not _STATE["groq_exhausted"]:
        rem = deadline - time.time() if deadline is not None else None
        res = _call_groq(system, user, max_tokens, timeout=rem)
        if res:
            return sanitize_llm_response(res)
        logger.info("[LLM Failover] Groq Cloud call did not return content.")

    if deadline is not None and (deadline - time.time()) < 1.0:
        return None

    # Tier 4: Try Generic Provider
    if config.LLM_API_KEY and config.LLM_MODEL:
        rem = deadline - time.time() if deadline is not None else None
        res = _call_generic(system, user, max_tokens, timeout=rem)
        if res:
            return sanitize_llm_response(res)

    logger.info("[LLM Failover] All active LLMs exhausted or unavailable. Falling back to offline pipeline.")
    return None


def reset_exhaustion_state():
    """Reset failover flags to allow retrying providers."""
    global _ACCOUNT_HAS_PAID_CREDITS, _LAST_CREDITS_CHECK
    _STATE["openrouter_exhausted"] = False
    _STATE["gemini_exhausted"] = False
    _STATE["groq_exhausted"] = False
    _MODEL_TOKEN_STATUS.clear()
    _ACCOUNT_HAS_PAID_CREDITS = None
    _LAST_CREDITS_CHECK = 0.0


# Public aliases for testing & compatibility
call_openrouter = _call_openrouter
generate = complete
