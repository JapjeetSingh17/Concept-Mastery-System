"""TinyFish Web Search Client.

Provides live web search capabilities for the Socratic AI Tutor when concepts
require broader external context, updated real-world examples, or when local PDF
material does not contain the answer.
"""

from __future__ import annotations

import json
import logging
import urllib.parse
import urllib.request
from typing import Any

from .. import config

logger = logging.getLogger("concept_mastery.web_search")


def search(query: str, max_results: int = 5) -> list[dict[str, Any]]:
    """Execute live web search via TinyFish Search API.
    
    Returns:
        List of dicts with 'title', 'snippet', 'url', 'site_name'.
    """
    clean_query = query.strip()
    if not clean_query:
        return []

    api_key = config.TINYFISH_API_KEY
    if not api_key:
        logger.warning("TinyFish API key not configured; skipping web search.")
        return []

    url = f"{config.TINYFISH_SEARCH_URL}?{urllib.parse.urlencode({'query': clean_query})}"
    req = urllib.request.Request(
        url,
        headers={
            "X-API-Key": api_key,
            "User-Agent": "ConceptMasteryPlatform/2.0",
            "Accept": "application/json",
        },
        method="GET",
    )

    try:
        with urllib.request.urlopen(req, timeout=6.0) as resp:
            if resp.status != 200:
                logger.warning("TinyFish Search responded with status %s", resp.status)
                return []
            raw_body = resp.read().decode("utf-8")
            data = json.loads(raw_body)
            results = data.get("results", [])
            output = []
            for item in results[:max_results]:
                output.append({
                    "title": item.get("title", ""),
                    "snippet": item.get("snippet", ""),
                    "url": item.get("url", ""),
                    "site_name": item.get("site_name", ""),
                    "date": item.get("date", ""),
                })
            return output
    except Exception as exc:
        logger.warning("TinyFish Search failed: %s", exc)
        return []
