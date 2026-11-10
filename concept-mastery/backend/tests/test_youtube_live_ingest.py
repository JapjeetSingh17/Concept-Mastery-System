"""Regression and Unit Tests for YouTube Live Stream and Subtitle-Disabled URL Ingestion.

Verifies:
  1. extract_youtube_id extracts 11-char IDs from /live/, /watch, youtu.be, /shorts, /embed/, /v/, with tracking params.
  2. clean_youtube_title removes emojis, live markers, and scheduled times.
  3. extract_description_chapters parses timestamp lines [MM:SS] or [HH:MM:SS].
  4. load_youtube_material gracefully falls back to structured lecture curriculum when transcripts are unavailable.
  5. UniversalMaterialIngestor.ingest_url successfully parses and indexes live stream materials without 400 Bad Request.
"""

from __future__ import annotations

import pytest
from unittest.mock import patch, MagicMock

from app.core import store
from app.core.concept_index import GLOBAL_INDEX
from app.pipelines.ingest import (
    extract_youtube_id,
    clean_youtube_title,
    extract_description_chapters,
    fetch_youtube_metadata,
    build_live_lecture_segments,
    load_youtube_material,
    UniversalMaterialIngestor,
)


def test_extract_youtube_id_comprehensive():
    expected_id = "mEbKrzFhWdM"
    test_urls = [
        "https://www.youtube.com/live/mEbKrzFhWdM",
        "https://www.youtube.com/live/mEbKrzFhWdM?si=e2f8c5b6a7",
        "https://youtube.com/live/mEbKrzFhWdM/",
        "https://m.youtube.com/live/mEbKrzFhWdM",
        "https://www.youtube.com/watch?v=mEbKrzFhWdM",
        "https://www.youtube.com/watch?feature=shared&v=mEbKrzFhWdM",
        "https://youtu.be/mEbKrzFhWdM",
        "https://youtu.be/mEbKrzFhWdM?t=60",
        "https://www.youtube.com/embed/mEbKrzFhWdM",
        "https://www.youtube.com/shorts/mEbKrzFhWdM",
        "https://www.youtube.com/v/mEbKrzFhWdM",
        "mEbKrzFhWdM",
    ]

    for url in test_urls:
        vid = extract_youtube_id(url)
        assert vid == expected_id, f"Failed to extract ID from '{url}', got '{vid}'"

    assert extract_youtube_id("https://example.com/not-youtube") is None
    assert extract_youtube_id("") is None
    assert extract_youtube_id(None) is None


def test_clean_youtube_title():
    raw = "✨ DSSSB -Database Management System Revision Class - 1 | DSSSB 2026 | LIVE | ⏰ 03:00 PM - YouTube"
    clean = clean_youtube_title(raw)
    assert "DSSSB" in clean
    assert "Database Management System" in clean
    assert "Revision Class" in clean
    assert "⏰" not in clean
    assert "✨" not in clean
    assert "- YouTube" not in clean


def test_extract_description_chapters():
    desc = """
    Lecture overview and notes:
    00:00 - Introduction to Relational Databases
    06:45 - 3-Tier Schema Architecture
    18:30 - Relational Algebra & Keys
    01:05:00 - Normalization Forms (1NF, 2NF, 3NF)
    """
    chapters = extract_description_chapters(desc)
    assert len(chapters) == 4
    assert chapters[0]["seconds"] == 0
    assert chapters[0]["title"] == "Introduction to Relational Databases"
    assert chapters[1]["seconds"] == 405
    assert chapters[3]["seconds"] == 3900


def test_live_stream_fallback_builds_structured_lecture_segments():
    video_id = "mEbKrzFhWdM"
    mock_meta = {
        "title": "✨ DSSSB -Database Management System Revision Class - 1 | LIVE | ⏰ 03:00 PM",
        "author": "KnowledgeGATE by Sanchit Sir",
        "description": "DBMS Revision Class for exam candidates.",
        "is_live": True,
    }

    title, raw_items, preview, full_text = build_live_lecture_segments(video_id, mock_meta)

    assert "[Live Stream]" in title
    assert len(preview) == 4
    assert len(raw_items) == 4
    assert preview[0]["video_id"] == video_id
    assert preview[0]["start_seconds"] == 0
    assert preview[0]["end_seconds"] == 720
    assert "00:00 - 12:00" in preview[0]["label"]

    for p in preview:
        assert p["video_id"] == video_id
        # Ensure minimum length guardrail is satisfied
        assert len(p["text"].split()) >= 100


def test_load_youtube_material_handles_transcripts_disabled():
    from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled

    url = "https://www.youtube.com/live/mEbKrzFhWdM"

    with patch.object(YouTubeTranscriptApi, "fetch", side_effect=TranscriptsDisabled("mEbKrzFhWdM")):
        title, raw_items, preview, full_text = load_youtube_material(url)
        assert len(preview) >= 1
        assert preview[0]["video_id"] == "mEbKrzFhWdM"
        assert any("00:00" in p["label"] for p in preview)
        assert len(full_text.split()) >= 150


def test_universal_material_ingest_live_url():
    user_id = "test_live_ingest_user"
    store.clear_all_study_materials(user_id)
    store.clear_kb(user_id)
    GLOBAL_INDEX.clear(user_id)

    url = "https://www.youtube.com/live/mEbKrzFhWdM"

    # Ingest the live stream URL
    res = UniversalMaterialIngestor.ingest_url(url=url, user_id=user_id)
    assert res["material_id"].startswith("mat_")
    assert "Database Management System" in res["title"]
    assert len(res["concepts"]) >= 1

    # Verify storage record
    stored = store.get_study_material(res["material_id"])
    assert stored is not None
    assert stored["source_type"] == "youtube"
    assert len(stored["structured_preview"]) >= 1
    assert stored["structured_preview"][0]["video_id"] == "mEbKrzFhWdM"

    # Clean up
    store.clear_all_study_materials(user_id)
    store.clear_kb(user_id)
    GLOBAL_INDEX.clear(user_id)

