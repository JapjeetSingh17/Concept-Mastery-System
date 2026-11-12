"""Main FastAPI Application Entrypoint."""

from contextlib import asynccontextmanager
import re
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
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


@app.middleware("http")
async def normalize_path_middleware(request: Request, call_next):
    """Normalize duplicate slashes (e.g., //api/ingest -> /api/ingest) preventing 405 Method Not Allowed."""
    path = request.scope.get("path", "")
    if "//" in path:
        request.scope["path"] = re.sub(r"/+", "/", path)
    return await call_next(request)


@app.exception_handler(store.UserError)
async def user_error_handler(_: Request, exc: store.UserError):
    return JSONResponse(status_code=400, content={"detail": str(exc)})


# Mount API Router
app.include_router(router)

# Mount Static UI Files (must come last)
app.mount("/", StaticFiles(directory=config.FRONTEND_DIR, html=True), name="ui")
