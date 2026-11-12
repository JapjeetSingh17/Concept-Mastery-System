"""TinyFish Web Search Client.

Provides live web search capabilities for the Socratic AI Tutor when concepts
require broader external context, updated real-world examples, or when local PDF
material does not contain the answer.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from .. import config

logger = logging.getLogger("concept_mastery.web_search")

_WEB_SEARCH_TIMEOUT = httpx.Timeout(6.0, connect=3.0)


def _prepare_search_request(query: str) -> tuple[str, dict[str, str], dict[str, str]] | None:
    """Validate query/API key and build TinyFish Search request parameters."""
    clean_query = query.strip()
    if not clean_query:
        return None

    api_key = config.TINYFISH_API_KEY
    if not api_key:
        logger.warning("TinyFish API key not configured; skipping web search.")
        return None

    headers = {
        "X-API-Key": api_key,
        "User-Agent": "ConceptMasteryPlatform/2.0",
        "Accept": "application/json",
    }
    return config.TINYFISH_SEARCH_URL, {"query": clean_query}, headers


def _parse_search_response(resp: httpx.Response, max_results: int) -> list[dict[str, Any]]:
    """Parse HTTP response from TinyFish Search API into normalized result dicts."""
    if resp.status_code != 200:
        logger.warning("TinyFish Search responded with status %s", resp.status_code)
        return []
    data = resp.json()
    results = data.get("results", [])
    return [
        {
            "title": item.get("title", ""),
            "snippet": item.get("snippet", ""),
            "url": item.get("url", ""),
            "site_name": item.get("site_name", ""),
            "date": item.get("date", ""),
        }
        for item in results[:max_results]
    ]


def search(query: str, max_results: int = 5) -> list[dict[str, Any]]:
    """Execute live web search via TinyFish Search API (synchronous).

    Returns:
        List of dicts with 'title', 'snippet', 'url', 'site_name'.
    """
    req_spec = _prepare_search_request(query)
    if req_spec is None:
        return []
    url, params, headers = req_spec

    try:
        with httpx.Client(timeout=_WEB_SEARCH_TIMEOUT) as client:
            resp = client.get(url, params=params, headers=headers)
        return _parse_search_response(resp, max_results)
    except Exception as exc:
        logger.warning("TinyFish Search failed: %s", exc)
        return []


search._is_default_sync_search = True


async def search_async(query: str, max_results: int = 5) -> list[dict[str, Any]]:
    """Execute live web search via TinyFish Search API without blocking the event loop."""
    req_spec = _prepare_search_request(query)
    if req_spec is None:
        return []
    url, params, headers = req_spec

    try:
        async with httpx.AsyncClient(timeout=_WEB_SEARCH_TIMEOUT) as client:
            resp = await client.get(url, params=params, headers=headers)
        return _parse_search_response(resp, max_results)
    except Exception as exc:
        logger.warning("TinyFish Search async request failed: %s", exc)
        return []

