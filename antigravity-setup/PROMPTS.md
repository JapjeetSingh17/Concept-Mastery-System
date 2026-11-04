# 💬 Recommended Antigravity Prompts

This guide provides exact copy-paste prompts to give to the **Google Antigravity AI Agent** after downloading and opening this repository on your desktop.

---

## 🌟 The Primary "One-Shot Starter Prompt" (Recommended First Prompt)

Paste this into your Antigravity chat immediately after opening the project:

```text
Hi Antigravity! I just downloaded the Concept Mastery System onto my desktop. 

Please perform a complete pre-flight check and help me get started:
1. Verify that Python 3, the virtual environment in concept-mastery/backend/.venv, and all dependencies from requirements.txt are installed.
2. Verify that .env exists (copy from .env.example if missing) and check directory permissions for data/uploads and generated_artifacts/.
3. Start the FastAPI backend server on http://127.0.0.1:8000 using Uvicorn in the background.
4. Verify the server health by curling http://127.0.0.1:8000/api/health and report the active status to me.
5. Once the server is ready, give me a quick walkthrough on how to upload my first study material and take my Baseline Diagnostic Assessment!
```

---

## 🎯 Workflow-Specific Prompts

### 1. Ingesting Study Materials & Baseline Diagnostic

```text
I want to ingest new study materials into my learning platform. 
Please help me:
1. Inspect the UniversalMaterialIngestor pipeline to ensure my file format is supported (PDF, PPTX, Markdown, DOCX, YouTube URL, or Audio).
2. Upload and process a sample document from the platform or one of my own study PDFs.
3. Verify that the SQLite BM25 indexer generated grounded chunk citations [Page X] / [Slide X] and built the unified Prerequisite Concept DAG.
4. Check the user lifecycle state at /api/loop/status to confirm that the platform is in "BASELINE_DIAGNOSTIC" mode.
5. Guide me through generating and submitting my 6-question calibrated Baseline Diagnostic Quiz so my root concepts transition out of UNKNOWN state.
```

---

### 2. Active Recall Focus Sprints (The 4-Stage Engine)

```text
I want to practice Active Recall on my weakest concept.
Please:
1. Query GET /api/mastery/dashboard to find my lowest-mastery or highest-uncertainty concept.
2. Initiate a new Active Recall session via POST /api/active-recall/start for that concept.
3. Present the micro-lesson key takeaways to me in chat.
4. Start a 120-second focus countdown timer.
5. When the timer expires, simulate hiding the lesson notes and prompt me for blind retrieval.
6. Submit my written retrieval to POST /api/active-recall/verify, and explain the AI Audit breakdown: what I retained, what gaps I missed, and how my Bayesian Beta mastery score updated.
```

---

### 3. Smart Study Notes Studio & Multimodal Generation

```text
I want to generate a complete Smart Study Notes package for the concept "Gradient Descent" (or my currently selected concept).
Please:
1. Trigger the Smart Notes generation endpoint with target task format "exam" (or "viva", "quiz", "custom").
2. Ensure the notes contain high-yield derivations, formula reference sheets, common exam traps, and exact textbook citations.
3. Synthesize the accompanying ElevenLabs Flash v2.5 voice explanation and verify the audio URL.
4. Trigger the PDF handout export and PPTX slide deck export, verifying that both files download cleanly with valid Content-Disposition headers.
5. Summarize the key takeaways and provide clickable links to the generated artifacts.
```

---

### 4. Numerical & Quantitative Step-by-Step Questions

```text
I want to practice calculation and numerical problem-solving.
Please:
1. Inspect backend/app/pipelines/quiz.py to see how numerical step-by-step questions are generated with given_parameters and tolerance thresholds.
2. Generate a numerical question for "Gradient Descent" (e.g. calculating updated weight given w=2.0, learning_rate=0.1, grad=4.0) or "Euclidean Distance".
3. Display the formula hint and parameter box.
4. Let me enter an answer, and submit it to POST /api/quiz/submit with tolerance grading (+/- 2%).
5. Show the step-by-step worked solution derivation.
```

---

### 5. Running Full Automated Verification & Quality Audit

```text
Please run a complete verification audit across the entire Concept Mastery codebase:
1. Run the full pytest test suite in concept-mastery/backend with:
   PYTHONPATH=. pytest -v
2. Check that all 65+ unit and integration tests pass (including baseline diagnostic, closed-loop engine, numerical active recall, OAuth multi-tenancy, OpenRouter gateway, and multimodal artifact generation).
3. Validate frontend JavaScript syntax with:
   node -c concept-mastery/frontend/js/app.js
4. Report any test failures, performance bottlenecks, or warnings, and suggest optimizations if needed.
```

---

### 6. Adding a New Study Material Extractor or Custom Feature

```text
I want to pair-program with you to extend the platform. 
Here is what I want to add:
[Describe your feature, e.g., "Add EPUB or Notion workspace ingestion" or "Add a spaced-repetition Leitner box review schedule"].

Please:
1. Analyze the existing pipeline architecture in concept-mastery/backend/app/pipelines/.
2. Identify which files need to be modified (schemas.py, store.py, routes.py, and app.js).
3. Propose a minimal, clean implementation plan before making edits.
4. Implement the changes with strict error handling and backward compatibility.
5. Add automated pytest test cases covering the new functionality.
```

---

### 7. Google OAuth & Domain Production Deployment

```text
I want to configure Google OAuth 2.0 and prepare this platform for deployment on a custom domain.
Please:
1. Review concept-mastery/backend/app/core/auth.py and backend/app/routes.py for OAuth redirect flows.
2. Guide me through adding the Authorized JavaScript Origins and Authorized Redirect URIs in Google Cloud Console for my domain.
3. Explain how user sessions, JWT tokens, and login audit logs (user_login_logs) are persisted in SQLite.
4. Provide the production deployment configuration (e.g., systemd, Docker, or reverse proxy with Caddy/Nginx).
```

---

## 💡 Antigravity Slash Commands to Use

When interacting with Antigravity, use these built-in slash commands:
- `/plan`: When you want Antigravity to architect a complex multi-file feature before coding.
- `/browser`: To launch Antigravity's embedded browser directly to `http://127.0.0.1:8000`.
- `/teamwork`: To launch multiple parallel sub-agents across isolated file boundaries.
- `/boost`: To enable deep thinking and multi-perspective verification for math/algorithmic tasks.
