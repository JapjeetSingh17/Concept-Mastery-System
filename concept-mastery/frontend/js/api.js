/**
 * Concept Mastery Platform - API Client
 * Centralized communication layer connecting the frontend to FastAPI backend endpoints.
 */

const API_CONFIG = {
  mock: false,
  base: "",
  storageKey: "concept_mastery_config_v1"
};

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

// Load persisted configuration if available
try {
  const saved = localStorage.getItem(API_CONFIG.storageKey);
  if (saved) {
    const parsed = JSON.parse(saved);
    if (typeof parsed.mock === "boolean") API_CONFIG.mock = parsed.mock;
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
  graph:            { method: "GET",  path: "/api/graph" },
  conceptsGraph:    { method: "GET",  path: "/api/concepts/graph" },
  quizNext:         { method: "POST", path: "/api/quiz/next" },
  quizAnswer:       { method: "POST", path: "/api/quiz/answer" },
  quizEvaluate:     { method: "POST", path: "/api/quiz/evaluate" },
  mastery:          { method: "GET",  path: "/api/mastery" },
  masteryDashboard: { method: "GET",  path: "/api/mastery/dashboard" },
  masteryReset:     { method: "POST", path: "/api/mastery/reset" },
  loopNextAction:   { method: "POST", path: "/api/loop/next-action" },
  recommend:        { method: "POST", path: "/api/recommend" }
};

/**
 * Core API caller with error parsing and mock support
 */
async function callApi(endpointKey, payload = null, options = {}) {
  const ep = ENDPOINTS[endpointKey];
  if (!ep) throw new Error(`Unknown endpoint: ${endpointKey}`);

  if (API_CONFIG.mock) {
    if (typeof MOCK !== "undefined" && typeof MOCK[endpointKey] === "function") {
      return await MOCK[endpointKey](payload);
    }
    throw new Error(`Mock implementation missing for ${endpointKey}`);
  }

  const base = getEffectiveBase();
  const cleanPath = ep.path.startsWith("/") ? ep.path : `/${ep.path}`;
  const url = `${base}${cleanPath}`;
  const fetchOpts = {
    method: ep.method,
    headers: {}
  };

  if (payload instanceof FormData) {
    fetchOpts.body = payload;
  } else if (payload && ep.method !== "GET") {
    fetchOpts.headers["Content-Type"] = "application/json";
    fetchOpts.body = JSON.stringify(payload);
  }

  let response;
  try {
    response = await fetch(url, fetchOpts);
  } catch (netErr) {
    throw new Error(`Network error connecting to backend: ${netErr.message}`);
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
    throw new Error(message || `${endpointKey} failed with HTTP ${response.status}`);
  }

  return data;
}

// Convenience API services
const ApiService = {
  async getHealth() {
    return callApi("health");
  },

  async getIngestStatus() {
    return callApi("ingestStatus");
  },

  async uploadPdf(file) {
    const formData = new FormData();
    formData.append("file", file);
    return callApi("ingest", formData);
  },

  async loadSamplePdf() {
    return callApi("ingestSample");
  },

  async askTutor(question, docId = null) {
    return callApi("ask", { question, docId });
  },

  async getGraph() {
    return callApi("graph");
  },

  async getConceptsGraph() {
    return callApi("conceptsGraph");
  },

  async getMasteryDashboard() {
    return callApi("masteryDashboard");
  },

  async getLoopNextAction() {
    return callApi("loopNextAction");
  },

  async evaluateAnswer(conceptId, questionText, studentAnswer, groundTruth = "") {
    return callApi("quizEvaluate", {
      conceptId,
      questionText,
      studentAnswer,
      groundTruth
    });
  },

  async getNextQuestion(conceptId = null) {
    return callApi("quizNext", { conceptId });
  },

  /**
   * Submit quiz answer for authoritative backend evaluation
   */
  async submitAnswer(questionId, conceptId, selectedIndex, correct = null) {
    const payload = {
      questionId,
      conceptId,
      selectedIndex,
      correct: correct !== null ? correct : undefined
    };
    return callApi("quizAnswer", payload);
  },

  async getMastery() {
    return callApi("mastery");
  },

  async resetMastery() {
    return callApi("masteryReset");
  },

  async getRecommendations() {
    return callApi("recommend");
  },

  saveConfig(newConfig) {
    Object.assign(API_CONFIG, newConfig);
    try {
      localStorage.setItem(API_CONFIG.storageKey, JSON.stringify({
        mock: API_CONFIG.mock,
        base: API_CONFIG.base
      }));
    } catch (e) {
      console.warn("Failed saving API config", e);
    }
  },

  getConfig() {
    return {
      ...API_CONFIG,
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

window.ApiService = ApiService;
window.getConceptStatus = getConceptStatus;
