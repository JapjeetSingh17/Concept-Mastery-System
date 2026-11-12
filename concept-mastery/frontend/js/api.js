/**
 * Concept Mastery Platform - API Client
 * Centralized communication layer connecting the frontend to FastAPI backend endpoints.
 */

function isMockModeEnabled() {
  try {
    if (typeof window !== "undefined" && window.location && typeof window.location.search === "string") {
      const params = new URLSearchParams(window.location.search);
      if (params.get("mock") === "true") return true;
    }
  } catch {}
  try {
    if (typeof localStorage !== "undefined" && localStorage.getItem("USE_MOCK") === "true") {
      return true;
    }
  } catch {}
  return false;
}

const API_CONFIG = {
  base: "",
  storageKey: "concept_mastery_config_v1"
};
Object.defineProperty(API_CONFIG, "mock", {
  enumerable: true,
  configurable: true,
  get() {
    return isMockModeEnabled();
  }
});

/**
 * Automatically determine the backend base URL.
 * When the frontend is viewed on a different port (e.g. port 5500 in IDE preview)
 * or via file://, auto-direct traffic to the FastAPI backend on port 8000.
 */
function getAutoBackendBase() {
  if (typeof window !== "undefined" && window.location) {
    if (window.location.protocol === "file:" || (window.location.port && window.location.port !== "8000")) {
      const host = window.location.hostname || "localhost";
      return `http://${host}:8000`;
    }
  }
  return "";
}

function getEffectiveBase() {
  if (API_CONFIG.base && typeof API_CONFIG.base === "string" && API_CONFIG.base.trim()) {
    return API_CONFIG.base.trim().replace(/\/+$/, "");
  }
  return getAutoBackendBase();
}

// Load persisted base URL if available (mock mode is never auto-enabled from saved config)
try {
  const saved = localStorage.getItem(API_CONFIG.storageKey);
  if (saved) {
    const parsed = JSON.parse(saved);
    if (typeof parsed.base === "string" && parsed.base.trim()) API_CONFIG.base = parsed.base.trim();
  }
} catch (e) {
  console.warn("Could not read API config from localStorage", e);
}

const ENDPOINTS = {
  health:           { method: "GET",  path: "/api/health" },
  ingest:           { method: "POST", path: "/api/ingest" },
  ingestUpload:     { method: "POST", path: "/api/ingest/upload" },
  ingestSample:     { method: "POST", path: "/api/ingest/sample" },
  ingestStatus:     { method: "GET",  path: "/api/ingest/status" },
  ask:              { method: "POST", path: "/api/tutor/ask" },
  webSearch:        { method: "POST", path: "/api/web/search" },
  graph:            { method: "GET",  path: "/api/graph" },
  conceptsGraph:    { method: "GET",  path: "/api/concepts/graph" },
  quizNext:         { method: "POST", path: "/api/quiz/next" },
  quizAnswer:       { method: "POST", path: "/api/quiz/answer" },
  quizEvaluate:     { method: "POST", path: "/api/quiz/evaluate" },
  mastery:          { method: "GET",  path: "/api/mastery" },
  masteryDashboard: { method: "GET",  path: "/api/mastery/dashboard" },
  masteryReset:     { method: "POST", path: "/api/mastery/reset" },
  loopNextAction:   { method: "POST", path: "/api/loop/next-action" },
  recommend:        { method: "POST", path: "/api/recommend" },
  dashboardSummary: { method: "GET",  path: "/api/dashboard-summary" },
  materialsList:       { method: "GET",    path: "/api/materials" },
  materialsIngestFile: { method: "POST",   path: "/api/materials/ingest-file" },
  materialsIngestUrl:  { method: "POST",   path: "/api/materials/ingest-url" },
  materialsIngestStatus:{ method: "GET",    path: "/api/materials/{material_id}/ingest-status" },
  materialsBaselineReport:{ method: "GET",  path: "/api/materials/{material_id}/baseline-report" },
  materialsPreview:    { method: "GET",    path: "/api/materials/{material_id}/preview" },
  materialsDelete:     { method: "DELETE", path: "/api/materials/{material_id}" },
  loopStatus:          { method: "GET",    path: "/api/loop/status" },
  diagnosticStart:     { method: "POST",   path: "/api/diagnostic/start" },
  diagnosticSubmit:    { method: "POST",   path: "/api/diagnostic/submit" },
  quizRemedial:        { method: "POST",   path: "/api/quiz/remedial" },
  activeRecallStart:   { method: "POST",   path: "/api/active-recall/start" },
  activeRecallEvaluate:{ method: "POST",   path: "/api/active-recall/evaluate" },
  activeRecallHistory: { method: "GET",    path: "/api/active-recall/history" },
  authGoogle:          { method: "POST",   path: "/api/auth/google" },
  authMe:              { method: "GET",    path: "/api/auth/me" },
  authLogs:            { method: "GET",    path: "/api/auth/logs" },
  authConfig:          { method: "GET",    path: "/api/auth/config" },
  notesGenerate:       { method: "POST",   path: "/api/notes/generate" },
  notesList:           { method: "GET",    path: "/api/notes" },
  notesDelete:         { method: "DELETE", path: "/api/notes/{note_id}" },
  notesAudio:          { method: "POST",   path: "/api/notes/{note_id}/audio" },
  notesExport:         { method: "POST",   path: "/api/notes/{note_id}/export" },
  materialDeepAudio:   { method: "POST",   path: "/api/materials/{material_id}/deep-audio" },
  toolsGenerate:       { method: "POST",   path: "/api/tools/generate" },
  artifactFile:        { method: "GET",    path: "/api/artifacts/{filename}" }
};

const IN_FLIGHT_REQUESTS = new Map();
const ABORT_SCOPES = new Map();
const SWR_MEMORY_CACHE = new Map();

function getSwrStorage() {
  try {
    if (typeof window !== "undefined" && window.sessionStorage) return window.sessionStorage;
    if (typeof sessionStorage !== "undefined") return sessionStorage;
  } catch {}
  return null;
}

function buildSwrKey(endpointKey, userId = "", materialId = "", options = {}) {
  let uid = userId || (options && (options.userId || options.user_id)) || (options && options.params && (options.params.userId || options.params.user_id));
  if (!uid && typeof ApiService !== "undefined" && typeof ApiService.getUser === "function") {
    const u = ApiService.getUser();
    uid = u ? (u.user_id || u.id) : "";
  }
  uid = uid || "default";
  const mid = materialId || (options && (options.materialId || options.material_id)) || (options && options.params && (options.params.materialId || options.params.material_id)) || (options && options.pathParams && options.pathParams.material_id) || "all";
  let query = "";
  if (options && options.params) {
    try {
      const filteredParams = { ...options.params };
      delete filteredParams.user_id;
      delete filteredParams.userId;
      delete filteredParams.material_id;
      delete filteredParams.materialId;
      const keys = Object.keys(filteredParams).sort();
      if (keys.length > 0) {
        const sortedObj = {};
        keys.forEach(k => { sortedObj[k] = filteredParams[k]; });
        query = JSON.stringify(sortedObj);
      }
    } catch {}
  }
  return `cm_swr:${uid}:${mid}:${endpointKey}${query ? `:${query}` : ""}`;
}

function createAbortError(message = "Request aborted") {
  if (typeof DOMException !== "undefined") {
    return new DOMException(message, "AbortError");
  }
  const err = new Error(message);
  err.name = "AbortError";
  return err;
}

function buildFormDataSignature(formData) {
  const parts = [];
  if (typeof formData.entries === "function") {
    for (const [k, v] of formData.entries()) {
      if (typeof File !== "undefined" && v instanceof File) {
        parts.push(`${k}:file:${v.name}:${v.size}:${v.lastModified}`);
      } else {
        parts.push(`${k}:${String(v)}`);
      }
    }
  }
  return parts.join("|");
}

function buildDedupeKey(endpointKey, method, url, payload, options) {
  if (options && options.dedupe === false) return null;
  if (options && options.dedupeKey) return `${endpointKey}:${options.dedupeKey}`;
  let bodySig = "";
  if (typeof FormData !== "undefined" && payload instanceof FormData) {
    bodySig = buildFormDataSignature(payload);
  } else if (payload && method !== "GET") {
    try {
      bodySig = JSON.stringify(payload);
    } catch {
      return null;
    }
  }
  return `${endpointKey}:${method}:${url}:${bodySig}`;
}

/**
 * Core API caller with error parsing, in-flight deduplication, AbortController support, and mock support
 */
async function callApi(endpointKey, payload = null, options = {}) {
  const ep = ENDPOINTS[endpointKey];
  if (!ep) throw new Error(`Unknown endpoint: ${endpointKey}`);

  const externalSignal = options && options.signal ? options.signal : null;
  if (externalSignal && externalSignal.aborted) {
    throw createAbortError();
  }

  const base = getEffectiveBase();
  let cleanPath = ep.path.startsWith("/") ? ep.path : `/${ep.path}`;
  if (options && options.pathParams) {
    for (const [k, v] of Object.entries(options.pathParams)) {
      cleanPath = cleanPath.replace(`{${k}}`, encodeURIComponent(v));
    }
  }
  let url = `${base}${cleanPath}`;
  if (options && options.params) {
    const qs = new URLSearchParams(options.params).toString();
    if (qs) url += (url.includes("?") ? "&" : "?") + qs;
  }

  const dedupeKey = buildDedupeKey(endpointKey, ep.method, url, payload, options);
  let entry = dedupeKey ? IN_FLIGHT_REQUESTS.get(dedupeKey) : null;

  if (!entry || entry.controller.signal.aborted) {
    const controller = typeof AbortController !== "undefined" ? new AbortController() : { signal: { aborted: false }, abort() {} };
    const execPromise = (async () => {
      if (isMockModeEnabled()) {
        if (typeof MOCK === "undefined" && typeof window !== "undefined" && window.__MOCK_READY__) {
          await window.__MOCK_READY__.catch(() => {});
        }
        if (typeof MOCK !== "undefined" && typeof MOCK[endpointKey] === "function") {
          return await MOCK[endpointKey](payload, { signal: controller.signal });
        }
        throw new Error(`Mock implementation missing for ${endpointKey}`);
      }

      const fetchOpts = {
        method: ep.method,
        headers: {},
        signal: controller.signal
      };

      const authToken = ApiService.getToken();
      if (authToken) {
        fetchOpts.headers["Authorization"] = `Bearer ${authToken}`;
      }

      if (typeof FormData !== "undefined" && payload instanceof FormData) {
        fetchOpts.body = payload;
      } else if (payload && ep.method !== "GET") {
        fetchOpts.headers["Content-Type"] = "application/json";
        fetchOpts.body = JSON.stringify(payload);
      }

      let response;
      try {
        response = await fetch(url, fetchOpts);
      } catch (netErr) {
        if ((netErr && netErr.name === "AbortError") || controller.signal.aborted) {
          throw createAbortError(netErr && netErr.message ? netErr.message : "Request aborted");
        }
        const err = new Error(`Network error connecting to backend: ${netErr.message}`);
        err.status = 0;
        err.endpoint = endpointKey;
        if (typeof window !== "undefined" && !(options && options.silentError)) {
          window.dispatchEvent(new CustomEvent("cm:api-error", {
            detail: { endpointKey, status: 0, message: err.message, method: ep.method, path: cleanPath, error: err }
          }));
        }
        throw err;
      }

      if (response.status === 401) {
        ApiService.clearAuth();
        if (typeof window !== "undefined") {
          window.dispatchEvent(new CustomEvent("cm:unauthorized"));
        }
      }

      let data;
      const contentType = response.headers.get("content-type") || "";
      if (contentType.includes("application/json")) {
        try {
          data = await response.json();
        } catch (e) {
          data = null;
        }
      } else {
        data = await response.text();
      }

      if (!response.ok) {
        let message = "";
        if (data && typeof data === "object") {
          message = data.detail || data.message || data.error;
        } else if (typeof data === "string") {
          message = data;
        }
        const finalMsg = message || `${endpointKey} failed with HTTP ${response.status}`;
        const err = new Error(finalMsg);
        err.status = response.status;
        err.endpoint = endpointKey;
        if (typeof window !== "undefined" && !(options && options.silentError)) {
          window.dispatchEvent(new CustomEvent("cm:api-error", {
            detail: { endpointKey, status: response.status, message: finalMsg, method: ep.method, path: cleanPath, error: err }
          }));
        }
        throw err;
      }

      if (ep.method === "GET" && (!options || options.swr !== false)) {
        try {
          const swrKey = buildSwrKey(endpointKey, options && (options.userId || options.user_id), options && options.materialId, options);
          const entry = { data, timestamp: Date.now() };
          SWR_MEMORY_CACHE.set(swrKey, entry);
          const storage = getSwrStorage();
          if (storage) {
            storage.setItem(swrKey, JSON.stringify(entry));
          }
        } catch {}
      }

      return data;
    })().finally(() => {
      if (dedupeKey && IN_FLIGHT_REQUESTS.get(dedupeKey) === entry) {
        IN_FLIGHT_REQUESTS.delete(dedupeKey);
      }
    });

    entry = { controller, promise: execPromise, subscribers: 0 };
    if (dedupeKey) {
      IN_FLIGHT_REQUESTS.set(dedupeKey, entry);
    }
  }

  if (!externalSignal) {
    entry.subscribers += 1;
    try {
      return await entry.promise;
    } finally {
      entry.subscribers = Math.max(0, entry.subscribers - 1);
    }
  }

  entry.subscribers += 1;
  return await new Promise((resolve, reject) => {
    let settled = false;
    const onAbort = () => {
      if (settled) return;
      settled = true;
      externalSignal.removeEventListener("abort", onAbort);
      entry.subscribers = Math.max(0, entry.subscribers - 1);
      if (entry.subscribers === 0) {
        entry.controller.abort();
        if (dedupeKey && IN_FLIGHT_REQUESTS.get(dedupeKey) === entry) {
          IN_FLIGHT_REQUESTS.delete(dedupeKey);
        }
      }
      reject(createAbortError());
    };

    if (externalSignal.aborted) {
      onAbort();
      return;
    }

    externalSignal.addEventListener("abort", onAbort, { once: true });
    entry.promise.then(
      val => {
        if (settled) return;
        settled = true;
        externalSignal.removeEventListener("abort", onAbort);
        entry.subscribers = Math.max(0, entry.subscribers - 1);
        resolve(val);
      },
      err => {
        if (settled) return;
        settled = true;
        externalSignal.removeEventListener("abort", onAbort);
        entry.subscribers = Math.max(0, entry.subscribers - 1);
        reject(err);
      }
    );
  });
}

// Convenience API services
const ApiService = {
  beginAbortScope(scopeKey = "navigation") {
    const prev = ABORT_SCOPES.get(scopeKey);
    if (prev && typeof prev.abort === "function") {
      prev.abort();
    }
    const next = typeof AbortController !== "undefined" ? new AbortController() : { signal: { aborted: false }, abort() {} };
    ABORT_SCOPES.set(scopeKey, next);
    return next;
  },

  abortScope(scopeKey = "navigation") {
    const active = ABORT_SCOPES.get(scopeKey);
    if (active && typeof active.abort === "function") {
      active.abort();
    }
    ABORT_SCOPES.delete(scopeKey);
  },

  getSwr(endpointKey, options = {}) {
    const swrKey = buildSwrKey(endpointKey, options.userId || options.user_id, options.materialId, options);
    if (SWR_MEMORY_CACHE.has(swrKey)) {
      const entry = SWR_MEMORY_CACHE.get(swrKey);
      return entry ? entry.data : null;
    }
    const storage = getSwrStorage();
    if (storage) {
      try {
        const raw = storage.getItem(swrKey);
        if (raw) {
          const parsed = JSON.parse(raw);
          if (parsed && typeof parsed === "object" && "data" in parsed) {
            SWR_MEMORY_CACHE.set(swrKey, parsed);
            return parsed.data;
          }
        }
      } catch {}
    }
    return null;
  },

  setSwr(endpointKey, data, options = {}) {
    if (data === undefined || data === null) return;
    const swrKey = buildSwrKey(endpointKey, options.userId || options.user_id, options.materialId, options);
    const entry = { data, timestamp: Date.now() };
    SWR_MEMORY_CACHE.set(swrKey, entry);
    const storage = getSwrStorage();
    if (storage) {
      try {
        storage.setItem(swrKey, JSON.stringify(entry));
      } catch {}
    }
  },

  clearSwr(filter = "") {
    if (!filter) {
      SWR_MEMORY_CACHE.clear();
      const storage = getSwrStorage();
      if (storage) {
        try {
          const toRemove = [];
          for (let i = 0; i < storage.length; i++) {
            const k = storage.key(i);
            if (k && k.startsWith("cm_swr:")) toRemove.push(k);
          }
          toRemove.forEach(k => storage.removeItem(k));
        } catch {}
      }
    } else {
      for (const k of SWR_MEMORY_CACHE.keys()) {
        if (k.includes(filter)) SWR_MEMORY_CACHE.delete(k);
      }
      const storage = getSwrStorage();
      if (storage) {
        try {
          const toRemove = [];
          for (let i = 0; i < storage.length; i++) {
            const k = storage.key(i);
            if (k && k.includes(filter)) toRemove.push(k);
          }
          toRemove.forEach(k => storage.removeItem(k));
        } catch {}
      }
    }
  },

  async fetchWithSwr(endpointKey, payload = null, options = {}, onRevalidate = null) {
    const cached = this.getSwr(endpointKey, options);
    const backgroundPromise = callApi(endpointKey, payload, options).then(fresh => {
      this.setSwr(endpointKey, fresh, options);
      if (typeof onRevalidate === "function") {
        try { onRevalidate(fresh); } catch (e) { console.warn("SWR onRevalidate error", e); }
      }
      return fresh;
    });

    if (cached !== null && cached !== undefined) {
      backgroundPromise.catch(() => {});
      return cached;
    }
    return await backgroundPromise;
  },

  async listNotes(userId = null, options = {}) {
    return this.getNotes(null, userId, options);
  },

  getInFlightCount() {
    return IN_FLIGHT_REQUESTS.size;
  },

  async getHealth(options = {}) {
    return callApi("health", null, options);
  },

  async getIngestStatus(options = {}) {
    return callApi("ingestStatus", null, options);
  },

  async uploadPdf(files, options = {}) {
    const formData = new FormData();
    if (files instanceof FileList || Array.isArray(files)) {
      for (let i = 0; i < files.length; i++) {
        formData.append("files", files[i]);
      }
    } else if (files instanceof File) {
      formData.append("files", files);
    }
    return callApi("ingest", formData, options);
  },

  async searchWeb(query, maxResults = 5, options = {}) {
    return callApi("webSearch", { query, max_results: maxResults }, options);
  },

  async loadSamplePdf(options = {}) {
    return callApi("ingestSample", null, options);
  },

  async askTutor(question, docId = null, options = {}) {
    return callApi("ask", { question, docId }, options);
  },

  async getGraph(options = {}) {
    return callApi("graph", null, options);
  },

  async getConceptsGraph(options = {}) {
    return callApi("conceptsGraph", null, options);
  },

  async getMasteryDashboard(options = {}) {
    return callApi("masteryDashboard", null, options);
  },

  async getLoopNextAction(options = {}) {
    return callApi("loopNextAction", null, options);
  },

  async evaluateAnswer(conceptId, questionText, studentAnswer, groundTruth = "", options = {}) {
    return callApi("quizEvaluate", {
      conceptId,
      questionText,
      studentAnswer,
      groundTruth
    }, options);
  },

  async getNextQuestion(conceptId = null, options = {}) {
    return callApi("quizNext", { conceptId }, options);
  },

  /**
   * Submit quiz answer for authoritative backend evaluation (supports MCQ, FITB, Typed Q&A, and Numerical)
   */
  async submitAnswer(questionId, conceptId, selectedIndex = null, correct = null, studentReasoning = "", typedAnswer = "", numericAnswer = null, options = {}) {
    const payload = {
      questionId,
      conceptId,
      selectedIndex: selectedIndex !== null ? selectedIndex : undefined,
      correct: correct !== null ? correct : undefined,
      studentReasoning: studentReasoning || undefined,
      typedAnswer: typedAnswer || undefined,
      numericAnswer: numericAnswer !== null && numericAnswer !== undefined ? numericAnswer : undefined,
      numeric_answer: numericAnswer !== null && numericAnswer !== undefined ? numericAnswer : undefined
    };
    return callApi("quizAnswer", payload, options);
  },

  async getLoopStatus(userId = "default", options = {}) {
    return callApi("loopStatus", null, { ...options, params: { user_id: userId } });
  },

  async startDiagnostic(userId = "default", options = {}) {
    return callApi("diagnosticStart", { user_id: userId }, options);
  },

  async submitDiagnostic(answers, userId = "default", materialId = null, options = {}) {
    const payload = { user_id: userId, answers };
    if (materialId) payload.material_id = materialId;
    return callApi("diagnosticSubmit", payload, options);
  },

  async getRemedialQuiz(conceptId = null, userId = "default", options = {}) {
    return callApi("quizRemedial", { concept_id: conceptId, conceptId, user_id: userId }, options);
  },

  async startActiveRecall(conceptId, userId = "default", options = {}) {
    return callApi("activeRecallStart", { concept_id: conceptId, user_id: userId }, options);
  },

  async evaluateActiveRecall(conceptId, recallText, verificationAnswer = "", studyDurationSeconds = 0, userId = "default", sessionId = "", options = {}) {
    return callApi("activeRecallEvaluate", {
      concept_id: conceptId,
      recall_text: recallText,
      verification_answer: verificationAnswer,
      study_duration_seconds: studyDurationSeconds,
      user_id: userId,
      session_id: sessionId
    }, options);
  },

  async getActiveRecallHistory(conceptId = null, userId = "default", options = {}) {
    const params = { user_id: userId };
    if (conceptId) params.concept_id = conceptId;
    return callApi("activeRecallHistory", null, { ...options, params });
  },

  async getMastery(options = {}) {
    return callApi("mastery", null, options);
  },

  async resetMastery(options = {}) {
    return callApi("masteryReset", null, options);
  },

  async getDashboardSummary(userId = "default", materialId = null, options = {}) {
    const params = { user_id: userId };
    if (materialId) params.material_id = materialId;
    if (options && (options.refresh_ai || options.refreshAi)) params.refresh_ai = true;
    return callApi("dashboardSummary", null, { ...options, params: { ...(options.params || {}), ...params } });
  },

  async getRecommendations(options = {}) {
    const params = {};
    if (options && (options.refresh_ai || options.refreshAi)) params.refresh_ai = true;
    return callApi("recommend", null, { ...options, params: { ...(options.params || {}), ...params } });
  },

  async listMaterials(userId = "default", options = {}) {
    return callApi("materialsList", null, { ...options, params: { user_id: userId } });
  },

  async ingestMaterialFile(files, userId = "default", title = "", materialId = null, options = {}) {
    const formData = new FormData();
    const fileList = Array.isArray(files) ? files : [files];
    fileList.forEach(f => formData.append("files", f));
    if (userId) formData.append("user_id", userId);
    if (title) formData.append("title", title);
    if (materialId) formData.append("material_id", materialId);
    return callApi("materialsIngestFile", formData, options);
  },

  async ingestMaterialUrl(url, sourceType = "auto", userId = "default", title = "", materialId = null, options = {}) {
    const payload = {
      url: url.trim(),
      source_type: sourceType,
      user_id: userId,
      title: title || undefined
    };
    if (materialId) payload.material_id = materialId;
    return callApi("materialsIngestUrl", payload, options);
  },

  async getIngestStatus(materialId, options = {}) {
    return callApi("materialsIngestStatus", null, {
      ...options,
      pathParams: { material_id: materialId || "latest" },
      skipCache: true
    });
  },

  async getBaselineReport(materialId, options = {}) {
    return callApi("materialsBaselineReport", null, {
      ...options,
      pathParams: { material_id: materialId || "latest" },
      skipCache: true
    });
  },

  async getMaterialPreview(materialId, options = {}) {
    return callApi("materialsPreview", null, { ...options, pathParams: { material_id: materialId } });
  },

  async deleteMaterial(materialId, userId = "default", options = {}) {
    return callApi("materialsDelete", null, {
      ...options,
      pathParams: { material_id: materialId },
      params: { user_id: userId }
    });
  },

  // ── Smart Study Notes & Multimodal Artifacts ───────────────────────────────

  async generateNote(req, options = {}) {
    return callApi("notesGenerate", req, options);
  },

  async getNotes(conceptId = null, userId = null, options = {}) {
    const params = {};
    if (conceptId) params.concept_id = conceptId;
    if (userId) params.user_id = userId;
    const res = await callApi("notesList", null, { ...options, params });
    return Array.isArray(res) ? res : (res && Array.isArray(res.notes) ? res.notes : []);
  },

  async deleteNote(noteId, userId = null, options = {}) {
    const params = {};
    if (userId) params.user_id = userId;
    return callApi("notesDelete", null, {
      ...options,
      pathParams: { note_id: noteId },
      params
    });
  },

  async generateNoteAudio(noteId, userId = null, options = {}) {
    const params = {};
    if (userId) params.user_id = userId;
    return callApi("notesAudio", null, {
      ...options,
      pathParams: { note_id: noteId },
      params
    });
  },

  async exportNoteArtifact(noteId, exportFormat = "pdf", userId = null, options = {}) {
    const fmt = exportFormat.toLowerCase().trim();
    const params = { export_format: fmt };
    if (userId) params.user_id = userId;
    const body = {
      export_format: fmt,
      ...(options.markdown_content ? { markdown_content: options.markdown_content } : {}),
      ...(options.title ? { title: options.title } : {}),
      ...(options.concept_id ? { concept_id: options.concept_id } : {}),
      ...(options.body || {})
    };
    return callApi("notesExport", body, {
      ...options,
      pathParams: { note_id: noteId },
      params
    });
  },

  async generateDeepAudio(materialId, sectionOrSlideIndex = 0, userId = null, options = {}) {
    return callApi("materialDeepAudio", {
      section_or_slide_index: sectionOrSlideIndex,
      user_id: userId || "default"
    }, {
      ...options,
      pathParams: { material_id: materialId }
    });
  },

  async generateToolArtifact(toolType, conceptId, prompt = "", slideCount = 5, userId = null, options = {}) {
    return callApi("toolsGenerate", {
      tool_type: toolType,
      concept_id: conceptId,
      prompt: prompt || "",
      slide_count: slideCount,
      user_id: userId || "default"
    }, options);
  },

  getArtifactUrl(pathOrFilename) {
    if (!pathOrFilename) return "";
    if (pathOrFilename.startsWith("http://") || pathOrFilename.startsWith("https://")) {
      return pathOrFilename;
    }
    const base = getEffectiveBase();
    let clean = pathOrFilename;
    if (!clean.startsWith("/")) {
      clean = `/api/artifacts/${clean}`;
    }
    return base ? `${base}${clean}` : clean;
  },

  async fetchAudioBlob(audioUrl, options = {}) {
    if (!audioUrl) throw new Error("Audio URL is required");
    const fullUrl = this.getArtifactUrl(audioUrl);
    const response = await fetch(fullUrl, options);
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: Failed to load audio stream`);
    }
    const rawBlob = await response.blob();
    const contentTypeHeader = (response.headers && typeof response.headers.get === "function")
      ? (response.headers.get("content-type") || "").toLowerCase()
      : "";
    const isWav = fullUrl.toLowerCase().endsWith(".wav") || contentTypeHeader.includes("wav");
    const minBytes = isWav ? 44 : 100;
    if (!rawBlob || rawBlob.size <= minBytes) {
      throw new Error(`Invalid or empty audio stream (${rawBlob ? rawBlob.size : 0} bytes)`);
    }
    const targetType = isWav ? "audio/wav" : "audio/mpeg";
    const BlobCtor = (typeof Blob !== "undefined")
      ? Blob
      : (typeof window !== "undefined" && window.Blob ? window.Blob : null);
    if (BlobCtor) {
      return new BlobCtor([rawBlob], { type: targetType });
    }
    return rawBlob;
  },

  async downloadArtifact(url, filename = "", options = {}) {
    if (!url) return null;
    const safeFilename = filename || url.split("/").pop().split("?")[0] || "download";
    try {
      const response = await fetch(url, options);
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      let contentType = (response.headers && typeof response.headers.get === "function")
        ? (response.headers.get("content-type") || "")
        : "";
      if (!contentType) {
        if (safeFilename.endsWith(".pdf")) contentType = "application/pdf";
        else if (safeFilename.endsWith(".pptx")) contentType = "application/vnd.openxmlformats-officedocument.presentationml.presentation";
        else if (safeFilename.endsWith(".mp3")) contentType = "audio/mpeg";
        else if (safeFilename.endsWith(".wav")) contentType = "audio/wav";
        else if (safeFilename.endsWith(".svg")) contentType = "image/svg+xml";
        else contentType = "application/octet-stream";
      }
      const rawBlob = await response.blob();
      if ((safeFilename.endsWith(".mp3") && (!rawBlob || rawBlob.size <= 100)) ||
          (safeFilename.endsWith(".wav") && (!rawBlob || rawBlob.size <= 44))) {
        throw new Error(`Audio download failed: invalid audio stream (${rawBlob ? rawBlob.size : 0} bytes)`);
      }
      const BlobCtor = (typeof Blob !== "undefined")
        ? Blob
        : (typeof window !== "undefined" && window.Blob ? window.Blob : null);
      const blob = BlobCtor ? new BlobCtor([rawBlob], { type: contentType }) : rawBlob;
      if (typeof window !== "undefined" && typeof document !== "undefined" && document.createElement) {
        const objectUrl = (window.URL && typeof window.URL.createObjectURL === "function")
          ? window.URL.createObjectURL(blob)
          : url;
        const a = document.createElement("a");
        a.style.display = "none";
        a.href = objectUrl;
        a.download = safeFilename;
        if (document.body && typeof document.body.appendChild === "function") {
          document.body.appendChild(a);
        }
        if (typeof a.click === "function") a.click();
        setTimeout(() => {
          try {
            if (typeof a.remove === "function") a.remove();
            else if (a.parentNode && typeof a.parentNode.removeChild === "function") a.parentNode.removeChild(a);
          } catch {}
          try {
            if (window.URL && typeof window.URL.revokeObjectURL === "function") {
              window.URL.revokeObjectURL(objectUrl);
            }
          } catch {}
        }, 60000);
      }
      return blob;
    } catch (err) {
      console.warn("Direct blob download failed, falling back to anchor:", err);
      if (typeof window !== "undefined" && typeof document !== "undefined" && document.createElement) {
        const a = document.createElement("a");
        a.style.display = "none";
        a.href = url;
        a.download = safeFilename;
        if (document.body && typeof document.body.appendChild === "function") {
          document.body.appendChild(a);
        }
        if (typeof a.click === "function") a.click();
        setTimeout(() => {
          try {
            if (typeof a.remove === "function") a.remove();
            else if (a.parentNode && typeof a.parentNode.removeChild === "function") a.parentNode.removeChild(a);
          } catch {}
        }, 60000);
      }
      throw err;
    }
  },

  async downloadFile(url, filename = "", options = {}) {
    return this.downloadArtifact(url, filename, options);
  },

  // ── Authentication & Session Management ─────────────────────────────────────

  getToken() {
    try {
      return localStorage.getItem("cm_access_token") || "";
    } catch {
      return "";
    }
  },

  setToken(token) {
    try {
      localStorage.setItem("cm_access_token", token);
    } catch (e) {
      console.warn("Could not set access token in localStorage", e);
    }
  },

  getUser() {
    try {
      const u = localStorage.getItem("cm_user");
      return u ? JSON.parse(u) : null;
    } catch {
      return null;
    }
  },

  setUser(user) {
    try {
      localStorage.setItem("cm_user", JSON.stringify(user));
    } catch (e) {
      console.warn("Could not set user profile in localStorage", e);
    }
  },

  clearAuth() {
    try {
      localStorage.removeItem("cm_access_token");
      localStorage.removeItem("cm_user");
    } catch {}
  },

  isAuthenticated() {
    return Boolean(this.getToken());
  },

  async loginGoogle(credential) {
    const res = await callApi("authGoogle", { credential });
    if (res && res.access_token) {
      this.setToken(res.access_token);
      if (res.user) {
        this.setUser(res.user);
      }
    }
    return res;
  },

  async getCurrentUser(options = {}) {
    return callApi("authMe", null, options);
  },

  async getLoginLogs(limit = 10, options = {}) {
    return callApi("authLogs", null, { ...options, params: { limit } });
  },

  async getAuthConfig(options = {}) {
    return callApi("authConfig", null, options);
  },

  logout() {
    this.clearAuth();
    return true;
  },

  isMockMode() {
    return isMockModeEnabled();
  },

  saveConfig(newConfig) {
    if (newConfig && typeof newConfig.base === "string") {
      API_CONFIG.base = newConfig.base;
    }
    try {
      if (newConfig && typeof newConfig.mock === "boolean") {
        if (newConfig.mock) {
          localStorage.setItem("USE_MOCK", "true");
        } else {
          localStorage.removeItem("USE_MOCK");
        }
      }
      localStorage.setItem(API_CONFIG.storageKey, JSON.stringify({
        base: API_CONFIG.base
      }));
    } catch (e) {
      console.warn("Failed saving API config", e);
    }
  },

  getConfig() {
    return {
      ...API_CONFIG,
      mock: isMockModeEnabled(),
      effectiveBase: getEffectiveBase()
    };
  }
};

// Global helper for determining concept status
function getConceptStatus(concept) {
  if (!concept) return "unknown";
  if (concept.status) return concept.status;
  if (!concept.attempts || concept.attempts < 2) return "unknown";
  return concept.score >= 0.7 ? "strong" : "weak";
}

if (typeof window !== "undefined") {
  window.ApiService = ApiService;
  window.getConceptStatus = getConceptStatus;
}
if (typeof module !== "undefined" && module.exports) {
  module.exports = { ApiService, getConceptStatus, callApi, ENDPOINTS };
}
