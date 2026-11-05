# Concept Mastery Closed-Loop Learning Platform - Antigravity Agent Rules

You are working on the **Concept Mastery Platform**, a high-precision, closed-loop educational engine that tracks epistemic student belief distributions, provides vectorless BM25 SQLite indexing with exact citation anchors, generates multimodal study artifacts (audio, PDF, PPTX), and enforces Active Recall.

## Key Project Locations & Boundaries
- Backend Root: `concept-mastery/backend`
- Frontend Root: `concept-mastery/frontend`
- Storage: `concept-mastery/data/uploads` and `concept-mastery/data/vectorstore`
- Generated Artifacts: `concept-mastery/backend/generated_artifacts`
- Test Suite: `concept-mastery/backend/tests`

## Architecture & Principles
1. **Vectorless SQLite BM25 Indexing**:
   - Chunks are stored in `chunks` and indexed with `chunk_fts`.
   - Never replace this with external vector databases or dense embedding models unless explicitly instructed.
2. **Multi-Agent Pedagogical Assessment & Head Teacher Arbiter**:
   - Replaced the old Bayesian Beta formula with a 6-layer Multi-Agent Pedagogical Assessment panel (`pipelines/mastery_orchestrator.py` & `pipelines/pedagogical_agents.py`).
   - 4 Specialist Teachers (Teaching Agent, Assessment Agent, Misconception Detective, Learning Observer) deliberate on student work.
   - Head Teacher Arbiter issues qualitative outcomes:
     * `MASTERED` (maps to legacy `STRONG`)
     * `NOT_YET_MASTERED` (maps to legacy `WEAK`)
     * `INSUFFICIENT_EVIDENCE` (maps to legacy `UNKNOWN`)
   - Software guardrails: hallucination-stripping via `store.verify_evidence_refs`, `MAX_FOLLOW_UP_QUESTIONS = 2`, `MAX_AGENT_CALLS = 6`.
3. **Empty-State Gating & Multi-Tenancy**:
   - When a user has 0 materials ingested, never show hardcoded dummy concepts.
   - Lifecycle state must be `AWAITING_UPLOAD` until material is uploaded, then `BASELINE_DIAGNOSTIC` until the 6-question baseline test is completed.
   - Strict per-user isolation: User A's uploaded materials and mastery states must never bleed into User B's state.
4. **Multimodal Artifacts & Downloads**:
   - `/api/artifacts/{filename}` must serve files with `Content-Disposition: attachment; filename="..."` for `.pdf` and `.pptx`.
   - Never use `target="_blank"` on browser download links to avoid stalled `.crdownload` files.
5. **Security & Secrets**:
   - Never hardcode API keys or secrets in source code.
   - `.env`, `backend/.env`, and `generated_artifacts/` are strictly ignored by `.gitignore`.

## Standard Commands
- Run backend server:
  ```bash
  cd concept-mastery/backend && source .venv/bin/activate && uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
  ```
- Run automated tests:
  ```bash
  cd concept-mastery/backend && PYTHONPATH=. pytest
  ```
- Validate frontend syntax:
  ```bash
  node -c concept-mastery/frontend/js/app.js
  ```

