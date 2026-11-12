from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys

import pytest

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app import config
from app.core import concept_index, store
from app.core.concept_index import ConceptIndex, MAX_ACTIVE_CONCEPT_CACHES
from app.core.vector_index import GLOBAL_VECTOR_CACHE, MAX_ACTIVE_VECTOR_INDICES, VectorIndex, VectorIndexCache
from app.models.schemas import ChunkMetadata


def _sample_chunk_dicts(tag: str) -> list[dict]:
    return [
        {
            "section": f"{tag} Overview",
            "text": f"Detailed explanation of {tag} algorithm and optimization equations.",
            "page": 1,
        },
        {
            "section": f"{tag} Practice",
            "text": f"Worked example for {tag} convergence rate.",
            "page": 2,
        },
    ]


def test_vector_index_cache_lru_eviction_and_disk_reload(tmp_path):
    cache_dir = tmp_path / "vector_cache"
    cache = VectorIndexCache(maxsize=3, cache_dir=cache_dir)

    assert MAX_ACTIVE_VECTOR_INDICES == 64
    assert GLOBAL_VECTOR_CACHE.maxsize == 64

    for i in range(6):
        key = cache.make_key(f"user_{i}", f"mat_{i}")
        idx = VectorIndex.build(_sample_chunk_dicts(f"topic{i}"))
        cache.put(key, idx)

    # Active RAM count must remain bounded at maxsize=3, while evicted 3 spill to disk.
    assert cache.in_memory_count() == 3
    assert cache.disk_count() == 3
    assert len(cache) == 6

    # Oldest entry (user_0) should still be present via disk reload and searchable.
    oldest_key = cache.make_key("user_0", "mat_0")
    assert oldest_key in cache
    reloaded = cache.get(oldest_key)
    assert isinstance(reloaded, VectorIndex)
    hits = reloaded.search("topic0 algorithm", k=2)
    assert len(hits) >= 1
    assert "topic0" in hits[0][1]["text"]

    # After reloading user_0 into RAM, active count remains bounded at 3 and oldest RAM item evicted.
    assert cache.in_memory_count() == 3
    assert cache.disk_count() == 3

    # Tenant invalidation removes only that tenant's entries without prefix collision.
    cache.put(cache.make_key("user_10", "mat_a"), VectorIndex.build(_sample_chunk_dicts("ten")))
    cache.put(cache.make_key("user_1", "mat_b"), VectorIndex.build(_sample_chunk_dicts("one")))
    assert cache.invalidate_user("user_1") >= 1
    assert cache.make_key("user_1", "mat_b") not in cache
    assert cache.make_key("user_10", "mat_a") in cache


def test_store_user_kbs_bounds_active_indices_and_reloads_vector_index(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "VECTOR_STORE_PATH", tmp_path)
    original_kb = dict(store.KB)
    store.USER_KBS.clear()
    try:
        total_users = 70  # > default 64 active limit
        for i in range(total_users):
            idx = VectorIndex.build(_sample_chunk_dicts(f"tenant{i}"))
            store.save_kb(
                f"doc_{i}.pdf",
                2,
                idx,
                [{"id": f"c-{i}", "name": f"Topic {i}"}],
                [],
                user_id=f"tenant_{i}",
            )

        assert store.USER_KBS.maxsize == 64
        assert store.USER_KBS.in_memory_count() == 64
        assert store.USER_KBS.disk_count() == total_users - 64

        # Evicted early tenant is still tracked and reloads with a functional VectorIndex.
        assert "tenant_0" in store.USER_KBS
        kb_zero = store.get_kb("tenant_0")
        assert kb_zero["ready"] is True
        assert isinstance(kb_zero["index"], VectorIndex)
        hits = kb_zero["index"].search("tenant0 optimization", k=1)
        assert len(hits) == 1
        assert store.USER_KBS.in_memory_count() == 64
    finally:
        store.USER_KBS.clear()
        store.KB.clear()
        store.KB.update(original_kb)


def test_concept_index_lru_cache_evicts_reloads_and_invalidates(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "VECTOR_STORE_PATH", tmp_path)
    monkeypatch.setattr(concept_index, "INDEX_DB_PATH", tmp_path / "concept_idx.db")
    concept_index.init_db()

    idx = ConceptIndex()
    assert idx._index_cache.maxsize == MAX_ACTIVE_CONCEPT_CACHES == 64
    idx.configure_cache(maxsize=3, cache_dir=lambda: config.VECTOR_STORE_PATH / "concept_cache")

    chunks_u1 = [
        ChunkMetadata(
            chunk_id=f"u1_c_{i}",
            page_number=i + 1,
            section_title=f"Section {i}",
            text=f"Gradient concept{i} convergence proof details.",
            canonical_concepts=[f"concept-{i}"],
            source_filename="notes.pdf",
        )
        for i in range(6)
    ]
    chunks_u10 = [
        ChunkMetadata(
            chunk_id="u10_c_0",
            page_number=1,
            section_title="Other Tenant",
            text="Orthogonal projections for user ten only.",
            canonical_concepts=["orthogonal-projections"],
            source_filename="u10.pdf",
        )
    ]
    idx.index_chunks(chunks_u1, user_id="user1")
    idx.index_chunks(chunks_u10, user_id="user10")

    # Populate 6 distinct cached concept lookups -> 3 stay in RAM, 3 spill to disk.
    for i in range(6):
        res = idx.retrieve_chunks(f"concept-{i}", max_chunks=2, user_id="user1")
        assert len(res) == 1
        assert res[0].chunk_id == f"u1_c_{i}"

    assert idx._index_cache.in_memory_count() == 3
    assert idx._index_cache.disk_count() == 3

    # Clear user1 SQLite rows WITHOUT going through idx.clear would break if reloaded from SQLite,
    # Wait: verify disk cache actually serves concept-0 without SQLite fallback!
    key0 = idx._cache_key("user1", "concept", "concept-0", 2)
    assert key0 in idx._index_cache
    with concept_index.get_db() as conn:
        conn.execute("DELETE FROM concept_chunks WHERE user_id = 'user1' AND concept_id = 'concept-0'")
    reloaded_zero = idx.retrieve_chunks("concept-0", max_chunks=2, user_id="user1")
    assert len(reloaded_zero) == 1
    assert reloaded_zero[0].chunk_id == "u1_c_0"
    assert idx._index_cache.in_memory_count() == 3

    # Tenant cache for user10 must stay unaffected by user1 clear (no prefix collision).
    u10_res = idx.retrieve_chunks("orthogonal-projections", max_chunks=2, user_id="user10")
    assert len(u10_res) == 1

    idx.clear(user_id="user1")
    assert idx.count(user_id="user1") == 0
    assert idx.retrieve_chunks("concept-1", max_chunks=2, user_id="user1") == []
    assert idx.count(user_id="user10") == 1
    assert len(idx.retrieve_chunks("orthogonal-projections", max_chunks=2, user_id="user10")) == 1

    # Re-indexing for user1 must invalidate stale cached empty results.
    idx.index_chunks([chunks_u1[0]], user_id="user1")
    fresh = idx.retrieve_chunks("concept-0", max_chunks=2, user_id="user1")
    assert len(fresh) == 1
    assert fresh[0].chunk_id == "u1_c_0"


def test_concurrent_vector_and_concept_index_cache_access(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "VECTOR_STORE_PATH", tmp_path)
    monkeypatch.setattr(concept_index, "INDEX_DB_PATH", tmp_path / "concurrent_idx.db")
    concept_index.init_db()

    vec_cache = VectorIndexCache(maxsize=5, cache_dir=tmp_path / "vec_concurrent")
    c_index = ConceptIndex()
    c_index.configure_cache(maxsize=5, cache_dir=tmp_path / "concept_concurrent")

    for i in range(12):
        uid = f"worker_{i}"
        vec_cache.put(vec_cache.make_key(uid, "doc"), VectorIndex.build(_sample_chunk_dicts(f"item{i}")))
        c_index.index_chunks(
            [
                ChunkMetadata(
                    chunk_id=f"chk_{i}",
                    page_number=1,
                    section_title=f"Sec {i}",
                    text=f"Concurrent item{i} theorem proof.",
                    canonical_concepts=[f"conc-{i}"],
                    source_filename="c.pdf",
                )
            ],
            user_id=uid,
        )

    def _exercise(worker_num: int) -> None:
        i = worker_num % 12
        uid = f"worker_{i}"
        v = vec_cache.get(vec_cache.make_key(uid, "doc"))
        assert v is not None
        assert len(v.search(f"item{i}", k=1)) >= 1
        hits = c_index.retrieve_chunks(f"conc-{i}", max_chunks=2, user_id=uid)
        assert len(hits) == 1 and hits[0].chunk_id == f"chk_{i}"
        bm25 = c_index.search_bm25(f"item{i} theorem", max_chunks=2, user_id=uid)
        assert len(bm25) >= 1

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(_exercise, range(60)))

    assert vec_cache.in_memory_count() <= 5
    assert c_index._index_cache.in_memory_count() <= 5

