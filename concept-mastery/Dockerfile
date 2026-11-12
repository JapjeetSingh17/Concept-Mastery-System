# syntax=docker/dockerfile:1

# ── Stage 1: Python Dependency Builder ─────────────────────────────────────────
FROM python:3.12-slim AS backend-builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /build

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY backend/requirements.txt /build/requirements.txt
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r /build/requirements.txt


# ── Stage 2: Static Frontend Assets ────────────────────────────────────────────
FROM alpine:3.20 AS frontend-assets

WORKDIR /frontend
COPY frontend/index.html /frontend/index.html
COPY frontend/css /frontend/css
COPY frontend/js /frontend/js


# ── Stage 3: Production Runtime (FastAPI Backend + Static Frontend) ────────────
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    UPLOAD_DIR="/app/data/uploads" \
    VECTOR_STORE_PATH="/app/data/vectorstore" \
    GENERATED_ARTIFACTS_DIR="/app/backend/generated_artifacts" \
    PORT="8000"

RUN groupadd --system appuser && useradd --system --gid appuser --create-home appuser

WORKDIR /app/backend

COPY --from=backend-builder /opt/venv /opt/venv
COPY backend/app /app/backend/app
COPY backend/synonyms.json /app/backend/synonyms.json
COPY backend/.env.example /app/backend/.env.example
COPY --from=frontend-assets /frontend /app/frontend

RUN mkdir -p /app/data/uploads /app/data/vectorstore /app/backend/generated_artifacts \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=3)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

