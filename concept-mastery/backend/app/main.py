"""Main FastAPI Application Entrypoint."""

from contextlib import asynccontextmanager
import re
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from . import config
from .core import store
from .pipelines import mastery
from .routes import router


@asynccontextmanager
async def lifespan(_app: FastAPI):
    config.validate_startup_config()
    yield


app = FastAPI(
    title="Closed-Loop Concept Mastery Platform",
    description="Vectorless RAG, Prerequisite Concept Graph (DAG), Socratic Guardrails, and LangGraph Engine.",
    version="2.0.0",
    lifespan=lifespan,
)

# CORS middleware for development and deployment
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from starlette.datastructures import Headers
from starlette.middleware.gzip import GZipMiddleware, GZipResponder, IdentityResponder
from starlette.types import ASGIApp, Message, Receive, Scope, Send


class SmartGZipResponder(GZipResponder):
    EXCLUDED_PREFIXES = (
        "audio/",
        "video/",
        "image/",
        "application/pdf",
        "application/zip",
        "application/vnd.",
        "application/octet-stream",
        "text/event-stream",
    )

    async def send_with_compression(self, message: Message) -> None:
        await super().send_with_compression(message)
        if message["type"] == "http.response.start":
            headers = Headers(raw=message["headers"])
            ct = headers.get("content-type", "").lower()
            if any(ct.startswith(p) for p in self.EXCLUDED_PREFIXES):
                self.content_type_is_excluded = True


class SmartGZipMiddleware(GZipMiddleware):
    """GZipMiddleware that compresses large JSON payloads and text/html/css/js
    while skipping already-compressed binary media (audio/mpeg, video/*, application/pdf, pptx, zip)."""

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = Headers(scope=scope)
        if "gzip" in headers.get("Accept-Encoding", ""):
            responder: ASGIApp = SmartGZipResponder(
                self.app, self.minimum_size, compresslevel=self.compresslevel
            )
        else:
            responder = IdentityResponder(self.app, self.minimum_size)

        await responder(scope, receive, send)


# Compress large JSON responses (transcripts, graphs, summaries) >= 1000 bytes
app.add_middleware(SmartGZipMiddleware, minimum_size=1000)


@app.middleware("http")
async def normalize_path_middleware(request: Request, call_next):
    """Normalize duplicate slashes (e.g., //api/ingest -> /api/ingest) preventing 405 Method Not Allowed."""
    path = request.scope.get("path", "")
    if "//" in path:
        request.scope["path"] = re.sub(r"/+", "/", path)
    return await call_next(request)


@app.middleware("http")
async def static_cache_control_middleware(request: Request, call_next):
    """Add Cache-Control: public, max-age=86400 headers for static vendor, css, and js assets."""
    response = await call_next(request)
    path = request.scope.get("path", "")
    if (
        path.startswith("/vendor/katex/")
        or path.startswith("/vendor/")
        or path.startswith("/css/")
        or path.startswith("/js/")
    ):
        response.headers["Cache-Control"] = "public, max-age=86400"
    return response


class CachedStaticFiles(StaticFiles):
    """StaticFiles subclass that injects Cache-Control headers for vendor, css, and js assets."""
    async def get_response(self, path: str, scope):
        response = await super().get_response(path, scope)
        norm_path = path.lstrip("/")
        if (
            norm_path.startswith("vendor/katex/")
            or norm_path.startswith("vendor/")
            or norm_path.startswith("css/")
            or norm_path.startswith("js/")
        ):
            response.headers["Cache-Control"] = "public, max-age=86400"
        return response


@app.exception_handler(store.UserError)
async def user_error_handler(_: Request, exc: store.UserError):
    return JSONResponse(status_code=400, content={"detail": str(exc)})


# Mount API Router
app.include_router(router)

# Mount Static UI Files with Cache-Control headers (must come last)
app.mount("/", CachedStaticFiles(directory=config.FRONTEND_DIR, html=True), name="ui")
