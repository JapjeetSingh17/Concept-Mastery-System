"""Regression tests proving blocking ingest, HTTP, and TTS work is offloaded from the asyncio event loop."""

from __future__ import annotations

import asyncio
import inspect
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from app import config, routes
from app.core import store, tts, web_search
from app.main import app
from app.pipelines import ingest, notes
from app.pipelines.ingest import UniversalMaterialIngestor


def test_sync_public_signatures_preserved():
    """Verify all public synchronous functions remain synchronous with intact signatures."""
    assert not inspect.iscoroutinefunction(web_search.search)
    assert not inspect.iscoroutinefunction(tts.synthesize_elevenlabs_audio)
    assert not inspect.iscoroutinefunction(tts.generate_deep_material_audio)
    assert not inspect.iscoroutinefunction(UniversalMaterialIngestor.ingest_file)
    assert not inspect.iscoroutinefunction(UniversalMaterialIngestor.ingest_url)
    assert not inspect.iscoroutinefunction(ingest.run_ingest)

    assert list(inspect.signature(web_search.search).parameters.keys()) == ["query", "max_results"]
    assert list(inspect.signature(tts.synthesize_elevenlabs_audio).parameters.keys()) == [
        "text",
        "user_id",
        "cache_key",
    ]
    assert list(inspect.signature(tts.generate_deep_material_audio).parameters.keys()) == [
        "material",
        "user_id",
        "section_or_slide_index",
    ]
    assert list(inspect.signature(UniversalMaterialIngestor.ingest_file).parameters.keys()) == [
        "path",
        "user_id",
        "title",
    ]
    assert list(inspect.signature(UniversalMaterialIngestor.ingest_url).parameters.keys()) == [
        "url",
        "source_type",
        "user_id",
        "title",
    ]

    assert inspect.iscoroutinefunction(web_search.search_async)
    assert inspect.iscoroutinefunction(tts.synthesize_elevenlabs_audio_async)
    assert inspect.iscoroutinefunction(tts.generate_deep_material_audio_async)
    assert inspect.iscoroutinefunction(UniversalMaterialIngestor.ingest_file_async)
    assert inspect.iscoroutinefunction(UniversalMaterialIngestor.ingest_url_async)
    assert inspect.iscoroutinefunction(ingest.run_ingest_async)


@pytest.mark.asyncio
async def test_ingest_file_route_does_not_block_event_loop(monkeypatch, tmp_path):
    """Slow synchronous file ingest in POST /api/materials/ingest-file must run in a threadpool."""
    monkeypatch.setattr(config, "UPLOAD_DIR", tmp_path / "uploads")
    loop_thread_id = threading.get_ident()
    observed = {}

    def slow_ingest_file(path: Path, user_id: str = "default", title=None):
        observed["thread_id"] = threading.get_ident()
        observed["saved_exists"] = Path(path).exists()
        time.sleep(0.25)
        return {
            "material_id": "mat_async_file",
            "title": title or "Async File",
            "source_type": "markdown",
            "original_filename_or_url": Path(path).name,
            "chunk_count": 1,
            "concepts": ["Concurrency"],
            "concept_count": 1,
            "graph_nodes": 1,
            "graph_edges": 0,
        }

    monkeypatch.setattr(UniversalMaterialIngestor, "ingest_file", staticmethod(slow_ingest_file))
    monkeypatch.setattr(
        routes,
        "_get_tenant_kb",
        lambda uid, req: {"ready": True, "file": "notes.md", "pages": 1, "concepts": ["Concurrency"]},
    )

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        ingest_task = asyncio.create_task(
            client.post(
                "/api/materials/ingest-file",
                headers={"X-Dev-User-Id": "async_file_user"},
                files=[("files", ("notes.md", b"# Concurrency\nThreadpool offload test.", "text/markdown"))],
            )
        )
        await asyncio.sleep(0.02)

        t0 = time.monotonic()
        health_responses = await asyncio.gather(*(client.get("/api/health") for _ in range(5)))
        health_elapsed = time.monotonic() - t0

        ingest_resp = await ingest_task

    assert ingest_resp.status_code == 200
    body = ingest_resp.json()
    assert body["ok"] is True
    assert body["count"] == 1
    assert observed["saved_exists"] is True
    assert observed["thread_id"] != loop_thread_id
    assert all(r.status_code == 200 for r in health_responses)
    assert health_elapsed < 0.20


@pytest.mark.asyncio
async def test_ingest_url_route_does_not_block_event_loop(monkeypatch):
    """Slow blocking urllib/HTML fetch in POST /api/materials/ingest-url must run off event loop."""
    loop_thread_id = threading.get_ident()
    observed = {}

    def slow_ingest_url(url: str, source_type: str = "auto", user_id: str = "default", title=None):
        observed["thread_id"] = threading.get_ident()
        time.sleep(0.25)
        return {
            "material_id": "mat_async_url",
            "title": title or "Example Article",
            "source_type": "web",
            "original_filename_or_url": url,
            "chunk_count": 2,
            "concepts": ["Async I/O"],
            "concept_count": 1,
            "graph_nodes": 1,
            "graph_edges": 0,
        }

    monkeypatch.setattr(UniversalMaterialIngestor, "ingest_url", staticmethod(slow_ingest_url))
    monkeypatch.setattr(
        routes,
        "_get_tenant_kb",
        lambda uid, req: {"ready": True, "file": "https://example.com/async", "pages": 1, "concepts": ["Async I/O"]},
    )

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        ingest_task = asyncio.create_task(
            client.post(
                "/api/materials/ingest-url",
                headers={"X-Dev-User-Id": "async_url_user"},
                json={"url": "https://example.com/async", "source_type": "web", "title": "Async Article"},
            )
        )
        await asyncio.sleep(0.02)

        t0 = time.monotonic()
        health_responses = await asyncio.gather(*(client.get("/api/health") for _ in range(5)))
        health_elapsed = time.monotonic() - t0

        ingest_resp = await ingest_task

    assert ingest_resp.status_code == 200
    assert ingest_resp.json()["material"]["material_id"] == "mat_async_url"
    assert observed["thread_id"] != loop_thread_id
    assert all(r.status_code == 200 for r in health_responses)
    assert health_elapsed < 0.20


@pytest.mark.asyncio
async def test_web_search_async_httpx_timeout_and_route_fallback(monkeypatch):
    """Verify web_search.search_async uses httpx.AsyncClient with explicit timeout and fallback offloads."""
    real_async_client = httpx.AsyncClient
    captured = {}

    class FakeSearchAsyncClient:
        def __new__(cls, *args, **kwargs):
            if "transport" in kwargs:
                return real_async_client(*args, **kwargs)
            return super().__new__(cls)

        def __init__(self, timeout=None, **kwargs):
            captured["timeout"] = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def get(self, url, params=None, headers=None):
            captured["url"] = url
            captured["params"] = params
            captured["headers"] = headers
            return SimpleNamespace(
                status_code=200,
                raise_for_status=lambda: None,
                json=lambda: {
                    "results": [
                        {
                            "title": "FastAPI Concurrency",
                            "url": "https://fastapi.tiangolo.com/async/",
                            "snippet": "Non-blocking async I/O.",
                        }
                    ]
                },
            )

    monkeypatch.setattr(config, "TINYFISH_API_KEY", "tf_test_key")
    monkeypatch.setattr(httpx, "AsyncClient", FakeSearchAsyncClient)

    results = await web_search.search_async("fastapi async", max_results=3)
    assert len(results) == 1
    assert results[0]["title"] == "FastAPI Concurrency"
    assert isinstance(captured["timeout"], httpx.Timeout)
    assert captured["timeout"].read == 6.0
    assert captured["timeout"].connect == 3.0

    # When sync web_search.search is monkeypatched by a test, /api/web/search runs it via run_in_threadpool.
    loop_thread_id = threading.get_ident()
    fallback_threads = []

    def patched_sync_search(query: str, max_results: int = 5):
        fallback_threads.append(threading.get_ident())
        return [{"title": f"Patched {query}", "url": "https://example.org", "snippet": "ok"}]

    monkeypatch.setattr(web_search, "search", patched_sync_search)
    transport = httpx.ASGITransport(app=app)
    async with real_async_client(transport=transport, base_url="http://test") as client:
        resp = await client.post("/api/web/search", json={"query": "event loop", "max_results": 2})

    assert resp.status_code == 200
    assert resp.json()["count"] == 1
    assert fallback_threads and fallback_threads[0] != loop_thread_id


@pytest.mark.asyncio
async def test_tts_async_httpx_timeout_and_audio_routes_offloaded(monkeypatch, tmp_path):
    """Verify ElevenLabs async TTS client timeout, non-blocking deep-audio, and note-audio offload."""
    monkeypatch.setattr(config, "VECTOR_STORE_PATH", tmp_path / "vec")
    monkeypatch.setattr(config, "GENERATED_ARTIFACTS_DIR", tmp_path / "artifacts")
    monkeypatch.setattr(config, "ELEVENLABS_API_KEY", "el_test_key")

    real_async_client = httpx.AsyncClient
    tts_captured = {}

    class FakeTTSAsyncClient:
        def __new__(cls, *args, **kwargs):
            if "transport" in kwargs:
                return real_async_client(*args, **kwargs)
            return super().__new__(cls)

        def __init__(self, timeout=None, **kwargs):
            tts_captured["timeout"] = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, url, headers=None, json=None):
            tts_captured["url"] = url
            await asyncio.sleep(0.05)
            return SimpleNamespace(status_code=200, content=b"ID3" + b"\x00" * 64, text="ok")

    monkeypatch.setattr(httpx, "AsyncClient", FakeTTSAsyncClient)

    res = await tts.synthesize_elevenlabs_audio_async(
        "Gradient $\\nabla L$ minimizes error.",
        user_id="tts_async_user",
        cache_key="async_tts_cache_test",
    )
    assert res["cached"] is False
    assert isinstance(tts_captured["timeout"], httpx.Timeout)
    assert tts_captured["timeout"].read == 30.0
    assert tts_captured["timeout"].connect == 5.0
    assert (tmp_path / "artifacts" / "async_tts_cache_test.mp3").exists()

    # Deep audio endpoint uses generate_deep_material_audio_async (which offloads llm.complete + httpx.AsyncClient).
    loop_thread_id = threading.get_ident()
    deep_threads = []

    monkeypatch.setattr(
        routes,
        "_require_owned_material",
        lambda material_id, uid, req: {
            "material_id": material_id,
            "user_id": uid,
            "title": "Slides",
            "source_type": "pptx",
            "raw_text": "Backpropagation computes gradients via the chain rule.",
            "sections_json": [{"title": "Slide 1", "content": "Chain rule for neural networks."}],
        },
    )

    def slow_llm_complete(prompt: str, max_tokens: int = 800, system: str = ""):
        deep_threads.append(threading.get_ident())
        time.sleep(0.20)
        return "Imagine sliding down a foggy hill using local slope information."

    monkeypatch.setattr(tts.llm, "complete", slow_llm_complete)

    # Note audio endpoint also offloads blocking notes.generate_note_audio off the event loop.
    note_threads = []
    monkeypatch.setattr(routes, "_require_owned_note", lambda note_id, uid: {"note_id": note_id, "user_id": uid})

    def slow_note_audio(note_id: str, user_id: str = "default"):
        note_threads.append(threading.get_ident())
        time.sleep(0.15)
        return {"note_id": note_id, "audio_url": "/api/artifacts/note.mp3", "transcript": "Summary"}

    monkeypatch.setattr(notes, "generate_note_audio", slow_note_audio)

    transport = httpx.ASGITransport(app=app)
    async with real_async_client(transport=transport, base_url="http://test") as client:
        deep_task = asyncio.create_task(
            client.post(
                "/api/materials/mat_slide_1/deep-audio",
                headers={"X-Dev-User-Id": "tts_async_user"},
                json={"section_or_slide_index": 0},
            )
        )
        await asyncio.sleep(0.02)
        t0 = time.monotonic()
        health_res = await client.get("/api/health")
        health_elapsed = time.monotonic() - t0
        deep_resp = await deep_task

        note_resp = await client.post(
            "/api/notes/note_1/audio",
            headers={"X-Dev-User-Id": "tts_async_user"},
        )

    assert health_res.status_code == 200
    assert health_elapsed < 0.15
    assert deep_resp.status_code == 200
    assert deep_threads and deep_threads[0] != loop_thread_id
    assert note_resp.status_code == 200
    assert note_threads and note_threads[0] != loop_thread_id

