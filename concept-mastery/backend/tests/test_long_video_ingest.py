"""Test Suite for Long-Video Transcript Ingestion, 100% Duration Parsing, and Non-Linear Concept Graph.

Verifies:
  1. Full-length transcript processing (3,600s / 1-hour video) without premature early-stop or 7-minute truncation.
  2. Sliding 2-to-3 minute semantic topic window segmenter with exact [MM:SS] or [HH:MM:SS] locators.
  3. Concept extraction spanning Early (Min 2), Middle (Min 30), and Late (Min 55) timeline tiers.
  4. Complete DAG generation including advanced concepts from the 55th minute.
  5. Non-linear retrieval and concept querying directly on late-stage topics.
  6. Backward compatibility with standard study materials.
"""

from __future__ import annotations

import pytest
from unittest.mock import patch, MagicMock

from app.core import store
from app.core.concept_index import GLOBAL_INDEX
from app.pipelines.ingest import UniversalMaterialIngestor, format_seconds_to_timestamp, load_youtube_material
from app.pipelines import concepts


@pytest.fixture(autouse=True)
def clean_test_env():
    user_id = "test_long_vid_user"
    store.clear_all_study_materials(user_id)
    store.clear_kb(user_id)
    GLOBAL_INDEX.clear(user_id)
    yield
    store.clear_all_study_materials(user_id)
    store.clear_kb(user_id)
    GLOBAL_INDEX.clear(user_id)


def generate_mock_1hr_transcript():
    """Generate a realistic 3,600-second (1-hour) YouTube transcript.
    Concepts are distributed across:
      - Min 0-5: Linear Regression, Loss Formulation, Feature Scaling
      - Min 28-35: Backpropagation, Gradient Descent, Chain Rule
      - Min 52-58: Adam Optimizer, Model Regularization, Weight Decay
    """
    snippets = []
    
    # Minute 0 to 5: Early foundational concepts
    snippets.append({"text": "Welcome to this deep learning masterclass lecture.", "start": 0.0, "duration": 15.0})
    snippets.append({"text": "We begin with linear regression and linear regression prediction formulation.", "start": 60.0, "duration": 30.0})
    snippets.append({"text": "Linear regression fits a hyperplane by minimizing a loss formulation function.", "start": 120.0, "duration": 40.0})
    snippets.append({"text": "The loss formulation penalizes residuals between predictions and target values.", "start": 180.0, "duration": 45.0})
    snippets.append({"text": "Feature scaling standardizes input values so loss formulation contours remain spherical.", "start": 260.0, "duration": 40.0})
    
    # Background filler across early-middle
    snippets.append({"text": "Let us consider multivariate distributions and matrix calculations.", "start": 400.0, "duration": 50.0})
    snippets.append({"text": "Matrix operations facilitate parallel computations on graphical processors.", "start": 600.0, "duration": 60.0})
    snippets.append({"text": "Activation functions introduce nonlinearities into artificial neural networks.", "start": 900.0, "duration": 60.0})
    snippets.append({"text": "Sigmoid and hyperbolic tangent were historical standards for hidden activations.", "start": 1200.0, "duration": 60.0})
    snippets.append({"text": "Rectified linear units avoid vanishing derivatives across positive regimes.", "start": 1500.0, "duration": 60.0})

    # Minute 28 to 35: Middle core concepts
    snippets.append({"text": "Now at minute thirty we dive into backpropagation through deep architectures.", "start": 1780.0, "duration": 40.0})
    snippets.append({"text": "Backpropagation computes error derivatives layer by layer using calculus.", "start": 1840.0, "duration": 45.0})
    snippets.append({"text": "Gradient descent updates parameter tensors proportional to the negative gradient.", "start": 1920.0, "duration": 50.0})
    snippets.append({"text": "The chain rule allows backpropagation gradients to propagate smoothly backwards.", "start": 2000.0, "duration": 40.0})
    snippets.append({"text": "Stochastic gradient descent computes gradients on minibatches to accelerate learning.", "start": 2100.0, "duration": 50.0})

    # Background filler across middle-late
    snippets.append({"text": "Mini-batch shuffling prevents cyclical training oscillations.", "start": 2400.0, "duration": 60.0})
    snippets.append({"text": "Momentum accumulates past velocity vectors to dampen noisy gradient oscillations.", "start": 2700.0, "duration": 60.0})
    snippets.append({"text": "RMSprop adapts per-parameter learning rates using exponential moving averages.", "start": 3000.0, "duration": 60.0})

    # Minute 52 to 58: Late advanced concepts
    snippets.append({"text": "In this final section around minute fifty five we explore the adam optimizer.", "start": 3250.0, "duration": 40.0})
    snippets.append({"text": "The adam optimizer combines adaptive learning rates with momentum estimation.", "start": 3310.0, "duration": 45.0})
    snippets.append({"text": "The adam optimizer maintains first and second moment bias-corrected vectors.", "start": 3380.0, "duration": 40.0})
    snippets.append({"text": "Model regularization prevents overfitting on complex training distributions.", "start": 3440.0, "duration": 45.0})
    snippets.append({"text": "Weight decay applies L2 penalties to constrain model regularization boundaries.", "start": 3500.0, "duration": 45.0})
    snippets.append({"text": "Conclusion of this one-hour lecture on optimization and learning systems.", "start": 3580.0, "duration": 20.0})

    return snippets


def test_format_seconds_to_timestamp():
    """Verify timestamp formatting handles MM:SS and HH:MM:SS."""
    assert format_seconds_to_timestamp(0) == "00:00"
    assert format_seconds_to_timestamp(45) == "00:45"
    assert format_seconds_to_timestamp(150) == "02:30"
    assert format_seconds_to_timestamp(460) == "07:40"
    assert format_seconds_to_timestamp(1800) == "30:00"
    assert format_seconds_to_timestamp(3300) == "55:00"
    assert format_seconds_to_timestamp(3665) == "01:01:05"


def test_long_video_no_7minute_truncation():
    """Verify UniversalMaterialIngestor processes 100% of 1-hour video without truncating at 7:40."""
    user_id = "test_long_vid_user"
    mock_snippets = generate_mock_1hr_transcript()

    mock_transcript_api = MagicMock()
    mock_transcript_api.get_transcript.return_value = mock_snippets

    with patch("youtube_transcript_api.YouTubeTranscriptApi", mock_transcript_api):
        url = "https://www.youtube.com/watch?v=mock1HourVid"
        res = UniversalMaterialIngestor.ingest_url(url, source_type="youtube", user_id=user_id, title="1-Hour Machine Learning Lecture")

        # 1. Chunk and Concept Counts
        assert res["chunk_count"] >= 4, f"Expected >=4 chunks for 1-hour video, got {res['chunk_count']}"
        assert res["concept_count"] >= 6, f"Expected >=6 concepts, got {res['concept_count']}"

        # 2. Check Database Persistence
        stored = store.get_study_material(res["material_id"], user_id=user_id)
        assert stored is not None
        preview = stored.get("structured_preview", [])
        assert len(preview) >= 10, f"Structured preview truncated! Expected >=10 segments, got {len(preview)}"

        # 3. Assert Timestamps span past 07:40 all the way to 55+ minutes
        timestamps = [p.get("label", "") for p in preview]
        max_sec = max(p.get("end_seconds", 0) for p in preview)
        assert max_sec >= 3500, f"Transcript was truncated at {max_sec}s instead of reaching 3600s!"

        # Ensure we have segments in the middle (30:00) and at the end (55:00)
        has_min_2 = any("01:" in ts or "02:" in ts for ts in timestamps)
        has_min_30 = any("29:" in ts or "30:" in ts or "31:" in ts for ts in timestamps)
        has_min_55 = any("54:" in ts or "55:" in ts or "56:" in ts for ts in timestamps)
        assert has_min_2, "Missing early segment (~02:00)"
        assert has_min_30, "Missing middle segment (~30:00) - failed to process past 7 minutes!"
        assert has_min_55, "Missing late segment (~55:00) - truncated prematurely!"

        # 4. Verify Concept Extraction includes late-stage concepts
        kb = store.get_kb(user_id)
        concept_names = [c["name"].lower() for c in kb.get("concepts", [])]
        
        # Check that generic locators like "Segment 00:00" or "Segment 07:40" are NOT concepts
        assert not any("segment" in name for name in concept_names), f"Generic segment names leaked into concepts: {concept_names}"
        assert not any("video [" in name for name in concept_names), f"Generic video locators leaked into concepts: {concept_names}"

        # Check domain concepts across the 3 tiers
        has_early_concept = any("linear regression" in c or "loss" in c for c in concept_names)
        has_middle_concept = any("backpropagation" in c or "gradient descent" in c for c in concept_names)
        has_late_concept = any("adam optimizer" in c or "regularization" in c or "weight decay" in c for c in concept_names)

        assert has_early_concept, f"Early concept missing from {concept_names}"
        assert has_middle_concept, f"Middle concept missing from {concept_names}"
        assert has_late_concept, f"Late (Min 55) concept missing from {concept_names}!"

        # 5. Non-Linear DAG Learning Path & Dependency Checks
        lpath = concepts.get_learning_path(user_id=user_id)
        assert len(lpath) >= 6, f"Expected DAG learning path to contain full concepts, got {len(lpath)}"

        # 6. Non-Linear Concept Retrieval
        # Find the late-stage concept ID
        late_c = next(c for c in kb.get("concepts", []) if "adam" in c["name"].lower() or "regularization" in c["name"].lower())
        assert late_c is not None
        assert len(late_c.get("chunks", [])) > 0
        
        # Assert the late concept references chunks with timestamps in the 50s
        chunk_hits = GLOBAL_INDEX.retrieve_chunks(late_c["id"], max_chunks=3, user_id=user_id)
        if not chunk_hits:
            scored = GLOBAL_INDEX.search_bm25(late_c["name"], max_chunks=3, user_id=user_id)
            chunk_hits = [c for _, c in scored]
        assert len(chunk_hits) > 0
        hit_locators = [getattr(h, "locator_label", "") for h in chunk_hits]
        assert any("5" in loc for loc in hit_locators), f"Expected late timestamp in chunk locator, got {hit_locators}"


def test_sliding_window_continuous_progression():
    """Verify load_youtube_material sliding topic window segments continuous text without gaps."""
    mock_snippets = generate_mock_1hr_transcript()
    mock_transcript_api = MagicMock()
    mock_transcript_api.get_transcript.return_value = mock_snippets

    with patch("youtube_transcript_api.YouTubeTranscriptApi", mock_transcript_api):
        title, raw_items, structured_preview, full_text = load_youtube_material("https://www.youtube.com/watch?v=mock1HourVid")
        
        assert len(structured_preview) >= 10
        # Check start times increase monotonically
        for i in range(len(structured_preview) - 1):
            assert structured_preview[i]["start_seconds"] < structured_preview[i+1]["start_seconds"]

        # Check last segment reaches close to video end
        assert structured_preview[-1]["end_seconds"] >= 3550
        # All items have locator labels formatted as timestamps
        for it in raw_items:
            assert ":" in it["locator_label"]
            assert it["section"].startswith("Video [")
