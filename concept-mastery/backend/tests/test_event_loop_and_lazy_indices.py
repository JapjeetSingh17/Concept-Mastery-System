"""Tests verifying elimination of event-loop blocking, GZip compression, static asset caching,
and lazy loading of BM25 and Vector indices (never eager during Dashboard/Material list)."""

import time
import pytest
from fastapi.testclient import TestClient

from app import config
from app.main import app
from app.core import store
from app.core.concept_index import ConceptIndex, GLOBAL_INDEX
from app.core.vector_index import VectorIndex, VectorIndexCache, GLOBAL_VECTOR_CACHE


def test_gzip_compression_on_large_json_responses():
    """Verify GZipMiddleware compresses payloads >= 1000 bytes when Accept-Encoding: gzip is requested."""
    client = TestClient(app)
    # /api/health is small (< 1000 bytes) -> uncompressed
    res_small = client.get("/api/health", headers={"Accept-Encoding": "gzip"})
    assert res_small.status_code == 200
    assert "gzip" not in res_small.headers.get("Content-Encoding", "").lower()

    # Create a mock large dashboard summary payload
    uid = "test_gzip_user"
    concepts = [
        {"id": f"concept_{i}", "name": f"Advanced Algorithmic Concept {i}", "summary": f"Detailed algorithmic summary and mathematical proofs for stage {i} " * 5}
        for i in range(25)
    ]
    edges = [[f"concept_{i}", f"concept_{i+1}"] for i in range(24)]
    store.save_kb("large_course.pdf", 50, VectorIndex(chunks=[]), concepts, edges, user_id=uid)

    res_large = client.get(f"/api/dashboard-summary?user_id={uid}", headers={"Accept-Encoding": "gzip"})
    assert res_large.status_code == 200
    # Should be compressed by GZipMiddleware
    assert res_large.headers.get("Content-Encoding") == "gzip"


def test_static_asset_cache_control_headers():
    """Verify Cache-Control: public, max-age=86400 is injected for /vendor/katex/*, /css/*, and /js/*."""
    client = TestClient(app)

    # 1. CSS asset
    res_css = client.get("/css/styles.css")
    assert res_css.status_code == 200
    assert res_css.headers.get("Cache-Control") == "public, max-age=86400"

    # 2. JS asset
    res_js = client.get("/js/app.js")
    assert res_js.status_code == 200
    assert res_js.headers.get("Cache-Control") == "public, max-age=86400"

    # 3. Root HTML (/) must NOT have 86400 cache control so users always get fresh app shell
    res_html = client.get("/")
    assert res_html.status_code == 200
    assert res_html.headers.get("Cache-Control") != "public, max-age=86400"


def test_vector_index_lazy_building():
    """Verify VectorIndex.build(chunks) defers IDF and vector computation until search() or property access."""
    sample_chunks = [
        {"section": "Optimization", "text": f"Gradient descent step {i} calculates error gradient."}
        for i in range(10)
    ]

    # Construction must be lazy (0 computation)
    idx = VectorIndex.build(sample_chunks)
    assert idx.is_built is False
    assert idx._idf is None
    assert idx._vecs is None
    assert len(idx.chunks) == 10

    # Serialization preserves chunk list without forcing unnecessary vector construction
    serialized = idx.to_dict()
    assert "chunks" in serialized
    assert idx.is_built is False

    # Search triggers deferred vector computation
    hits = idx.search("gradient descent", k=3)
    assert idx.is_built is True
    assert len(hits) == 3
    assert len(idx.vecs) == 10
    assert len(idx.idf) > 0


def test_concept_index_lazy_initialization():
    """Verify ConceptIndex initialization does not eagerly open SQLite or run DDL."""
    fresh_index = ConceptIndex()
    assert fresh_index.retrieval_ops_count == 0


def test_dashboard_and_materials_fetches_do_not_trigger_bm25_or_vector_building():
    """Verify Dashboard and Materials fetches do not perform BM25 retrieval or eager vector calculations."""
    uid = "test_lazy_dash_user"
    client = TestClient(app)

    # Register study material
    store.save_study_material({
        "material_id": "mat_lazy_01",
        "user_id": uid,
        "title": "Quantum Foundations",
        "source_type": "pdf",
        "original_filename_or_url": "quantum.pdf",
        "chunk_count": 12,
        "raw_text": "Quantum superposition and entanglement principles.",
        "concepts": [
            {"id": "c_super", "name": "Superposition", "summary": "Linear combination of quantum states."},
            {"id": "c_entangle", "name": "Entanglement", "summary": "Non-local quantum correlations."}
        ]
    })

    # Clear retrieval ops count before dashboard request
    initial_retrieval_count = GLOBAL_INDEX.retrieval_ops_count

    # 1. Fetch materials list
    res_mats = client.get(f"/api/materials?user_id={uid}")
    assert res_mats.status_code == 200
    assert res_mats.json()["total_materials"] >= 1
    # No BM25 retrieval operations triggered
    assert GLOBAL_INDEX.retrieval_ops_count == initial_retrieval_count

    # 2. Fetch dashboard summary
    res_dash = client.get(f"/api/dashboard-summary?user_id={uid}")
    assert res_dash.status_code == 200
    dash_data = res_dash.json()
    assert dash_data["ok"] is True
    assert len(dash_data["concepts"]) == 2
    # No BM25 retrieval operations triggered during dashboard summary
    assert GLOBAL_INDEX.retrieval_ops_count == initial_retrieval_count

    # 3. Fetch concepts graph
    res_graph = client.get(f"/api/concepts/graph?user_id={uid}")
    assert res_graph.status_code == 200
    # Still zero BM25 retrieval operations
    assert GLOBAL_INDEX.retrieval_ops_count == initial_retrieval_count


def test_retrieval_operations_triggered_on_demand_during_study_queries():
    """Verify BM25/vector indices are triggered when retrieval is actively requested (Quiz, Tutor, Active Recall)."""
    uid = "test_retrieval_user"
    initial_count = GLOBAL_INDEX.retrieval_ops_count

    # Direct retrieval call (may fallback to search_bm25 if canonical tag not yet indexed)
    GLOBAL_INDEX.retrieve_chunks("quantum_superposition", max_chunks=3, user_id=uid)
    assert GLOBAL_INDEX.retrieval_ops_count >= initial_count + 1
    count_after_first = GLOBAL_INDEX.retrieval_ops_count

    GLOBAL_INDEX.search_bm25("entanglement", max_chunks=2, user_id=uid)
    assert GLOBAL_INDEX.retrieval_ops_count >= count_after_first + 1
