/**
 * Automated concurrency, deduplication, AbortController, and button-locking
 * regression tests for frontend/js/api.js and frontend/js/app.js.
 */

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

const API_JS_PATH = path.resolve(__dirname, "../js/api.js");
const APP_JS_PATH = path.resolve(__dirname, "../js/app.js");
const MOCK_JS_PATH = path.resolve(__dirname, "../js/mock.js");
const INDEX_HTML_PATH = path.resolve(__dirname, "../index.html");

function createDeferred() {
  let resolve, reject;
  const promise = new Promise((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
}

function createAbortError(message = "Aborted") {
  const err = new Error(message);
  err.name = "AbortError";
  return err;
}

function makeMockElement(id = "", tagName = "DIV") {
  const attrs = new Map();
  if (id) attrs.set("id", id);
  const listeners = new Map();
  const classSet = new Set();
  const childrenMap = new Map();

  const el = {
    id,
    tagName: tagName.toUpperCase(),
    disabled: false,
    dataset: {},
    style: {},
    value: "",
    children: [],
    _innerHTML: "",
    _textContent: "",
    classList: {
      add: (...cls) => cls.forEach(c => classSet.add(c)),
      remove: (...cls) => cls.forEach(c => classSet.delete(c)),
      toggle: (c, force) => {
        if (force === undefined) {
          if (classSet.has(c)) classSet.delete(c);
          else classSet.add(c);
        } else if (force) classSet.add(c);
        else classSet.delete(c);
      },
      contains: c => classSet.has(c)
    },
    setAttribute(k, v) {
      attrs.set(k, String(v));
      if (k === "id") el.id = String(v);
    },
    getAttribute(k) {
      return attrs.has(k) ? attrs.get(k) : null;
    },
    removeAttribute(k) {
      attrs.delete(k);
    },
    hasAttribute(k) {
      return attrs.has(k);
    },
    addEventListener(type, fn) {
      if (!listeners.has(type)) listeners.set(type, new Set());
      listeners.get(type).add(fn);
    },
    removeEventListener(type, fn) {
      if (listeners.has(type)) listeners.get(type).delete(fn);
    },
    dispatchEvent(evt) {
      const set = listeners.get(evt.type);
      if (set) {
        for (const fn of [...set]) fn(evt);
      }
    },
    click() {
      if (el.disabled) return;
      if (typeof el.onclick === "function") {
        return el.onclick({ preventDefault() {}, stopPropagation() {}, target: el });
      }
      el.dispatchEvent({ type: "click", preventDefault() {}, stopPropagation() {}, target: el });
    },
    focus() {},
    scrollIntoView() {},
    appendChild(child) {
      el.children.push(child);
      return child;
    },
    removeChild(child) {
      el.children = el.children.filter(c => c !== child);
      return child;
    },
    remove() {},
    insertAdjacentHTML(_pos, html) {
      el._innerHTML += html;
    },
    closest() {
      return null;
    },
    play() {
      return Promise.resolve();
    },
    load() {},
    pause() {},
    get innerHTML() {
      return el._innerHTML;
    },
    set innerHTML(val) {
      el._innerHTML = String(val);
    },
    get textContent() {
      return el._textContent || el._innerHTML.replace(/<[^>]+>/g, " ");
    },
    set textContent(val) {
      el._textContent = String(val);
      el._innerHTML = String(val);
    },
    querySelector(sel) {
      if (sel.startsWith("#")) return childrenMap.get(sel.slice(1)) || null;
      return null;
    },
    querySelectorAll() {
      return [];
    },
    _registerChild(child) {
      if (child && child.id) childrenMap.set(child.id, child);
      return child;
    }
  };
  return el;
}

function createApiSandbox(fetchImpl) {
  const storage = new Map();
  const sessionStorageMap = new Map();
  const sessionStorageObj = {
    getItem: k => (sessionStorageMap.has(k) ? sessionStorageMap.get(k) : null),
    setItem: (k, v) => sessionStorageMap.set(k, String(v)),
    removeItem: k => sessionStorageMap.delete(k),
    clear: () => sessionStorageMap.clear(),
    get length() { return sessionStorageMap.size; },
    key: i => Array.from(sessionStorageMap.keys())[i] || null
  };
  const windowObj = {
    location: { protocol: "http:", hostname: "localhost", port: "8000", hash: "#dashboard" },
    sessionStorage: sessionStorageObj,
    dispatchEvent() {},
    addEventListener() {},
    removeEventListener() {}
  };
  const context = {
    console,
    URLSearchParams,
    AbortController,
    CustomEvent: class CustomEvent {
      constructor(type, init = {}) {
        this.type = type;
        this.detail = init.detail;
      }
    },
    localStorage: {
      getItem: k => (storage.has(k) ? storage.get(k) : null),
      setItem: (k, v) => storage.set(k, String(v)),
      removeItem: k => storage.delete(k)
    },
    sessionStorage: sessionStorageObj,
    window: windowObj,
    fetch: fetchImpl,
    setTimeout,
    clearTimeout,
    FormData: class FormData {
      constructor() {
        this._entries = [];
      }
      append(k, v) {
        this._entries.push([k, v]);
      }
      entries() {
        return this._entries[Symbol.iterator]();
      }
    },
    Blob
  };
  context.globalThis = context;
  vm.createContext(context);
  vm.runInContext(fs.readFileSync(API_JS_PATH, "utf8"), context, { filename: "api.js" });
  return { context, ApiService: context.window.ApiService };
}

async function testApiServiceExportsAndDedupe() {
  let fetchCalls = [];
  const pendingByPath = new Map();

  const fetchImpl = (url, opts = {}) => {
    fetchCalls.push({ url, method: opts.method || "GET", signal: opts.signal });
    const d = createDeferred();
    if (opts.signal) {
      if (opts.signal.aborted) {
        return Promise.reject(createAbortError("Already aborted"));
      }
      opts.signal.addEventListener("abort", () => d.reject(createAbortError("Signal aborted")), { once: true });
    }
    const key = `${opts.method || "GET"} ${url}`;
    if (!pendingByPath.has(key)) pendingByPath.set(key, []);
    pendingByPath.get(key).push(d);
    return d.promise;
  };

  const { ApiService, context } = createApiSandbox(fetchImpl);

  // 1. Verify expected global exports & original method names on ApiService
  const expectedMethods = [
    "uploadPdf", "searchWeb", "loadSamplePdf", "askTutor", "getGraph", "getConceptsGraph",
    "getMasteryDashboard", "getLoopNextAction", "evaluateAnswer", "getNextQuestion",
    "submitAnswer", "getLoopStatus", "startDiagnostic", "submitDiagnostic",
    "getRemedialQuiz", "startActiveRecall", "evaluateActiveRecall", "getActiveRecallHistory",
    "getMastery", "resetMastery", "getRecommendations", "getDashboardSummary", "listMaterials",
    "ingestMaterialFile", "ingestMaterialUrl", "getMaterialPreview", "deleteMaterial",
    "generateNote", "getNotes", "deleteNote", "generateNoteAudio", "exportNoteArtifact",
    "generateDeepAudio", "generateToolArtifact", "getArtifactUrl",
    "getToken", "setToken", "getUser", "setUser", "clearAuth", "isAuthenticated",
    "loginGoogle", "getCurrentUser", "getLoginLogs", "getAuthConfig",
    "beginAbortScope", "abortScope", "getInFlightCount"
  ];
  for (const name of expectedMethods) {
    assert.equal(typeof ApiService[name], "function", `Missing ApiService.${name}`);
  }
  assert.equal(typeof context.window.getConceptStatus, "function");

  // 2. Concurrent identical GET calls dedupe into 1 fetch
  const p1 = ApiService.getMastery();
  const p2 = ApiService.getMastery();
  assert.equal(fetchCalls.length, 1, "Concurrent getMastery calls must dedupe to 1 fetch");
  assert.equal(ApiService.getInFlightCount(), 1);

  const masteryKey = "GET /api/mastery";
  pendingByPath.get(masteryKey)[0].resolve({
    ok: true,
    status: 200,
    headers: { get: () => "application/json" },
    json: async () => ({ concepts: [{ id: "c1", name: "Gradient Descent", score: 0.8 }] })
  });

  const [r1, r2] = await Promise.all([p1, p2]);
  assert.deepEqual(r1, r2);
  assert.equal(ApiService.getInFlightCount(), 0);

  // 3. Subsequent call after settling triggers a fresh fetch
  const p3 = ApiService.getMastery();
  assert.equal(fetchCalls.length, 2, "Subsequent getMastery call after settle should hit network again");
  pendingByPath.get(masteryKey)[1].resolve({
    ok: true,
    status: 200,
    headers: { get: () => "application/json" },
    json: async () => ({ concepts: [] })
  });
  await p3;

  // 4. Per-caller abort isolation on shared in-flight request
  const ac1 = new AbortController();
  const ac2 = new AbortController();
  const sp1 = ApiService.getGraph({ signal: ac1.signal });
  const sp2 = ApiService.getGraph({ signal: ac2.signal });
  assert.equal(fetchCalls.length, 3);

  ac1.abort();
  await assert.rejects(sp1, err => err && err.name === "AbortError");
  assert.equal(fetchCalls[2].signal.aborted, false, "Upstream fetch must stay alive while another caller is subscribed");

  const graphKey = "GET /api/graph";
  pendingByPath.get(graphKey)[0].resolve({
    ok: true,
    status: 200,
    headers: { get: () => "application/json" },
    json: async () => ({ nodes: [{ id: "n1" }], edges: [] })
  });
  const graphRes = await sp2;
  assert.equal(graphRes.nodes.length, 1);

  // 5. Aborting last/sole caller aborts upstream signal
  const navCtrl1 = ApiService.beginAbortScope("navigation");
  const np1 = ApiService.getRecommendations({ signal: navCtrl1.signal });
  assert.equal(fetchCalls.length, 4);
  const navCtrl2 = ApiService.beginAbortScope("navigation");
  assert.equal(navCtrl1.signal.aborted, true, "beginAbortScope must abort previous navigation controller");
  assert.equal(fetchCalls[3].signal.aborted, true, "Upstream signal must abort when sole caller aborts");
  await assert.rejects(np1, err => err && err.name === "AbortError");
  assert.equal(navCtrl2.signal.aborted, false);
}

async function createAppHarness(routeHandlers) {
  const storage = new Map([
    ["cm_access_token", "test-token"],
    ["cm_user", JSON.stringify({ user_id: "user-1", full_name: "Test User" })]
  ]);

  const sessionStorageMap = new Map();
  const sessionStorageObj = {
    getItem: k => (sessionStorageMap.has(k) ? sessionStorageMap.get(k) : null),
    setItem: (k, v) => sessionStorageMap.set(k, String(v)),
    removeItem: k => sessionStorageMap.delete(k),
    clear: () => sessionStorageMap.clear(),
    get length() { return sessionStorageMap.size; },
    key: i => Array.from(sessionStorageMap.keys())[i] || null
  };

  const byId = new Map();
  const classElements = new Map();
  const docListeners = new Map();
  const winListeners = new Map();

  function registerIdsFromHtml(html) {
    if (!html) return;
    const matches = [...String(html).matchAll(/id="([^"]+)"/g)];
    for (const m of matches) {
      const eid = m[1];
      if (!byId.has(eid)) {
        makeStubContainer(eid);
      }
    }
  }

  function makeStubContainer(id, tag = "DIV") {
    const el = makeMockElement(id, tag);
    byId.set(id, el);
    let elemHtml = "";
    Object.defineProperty(el, "innerHTML", {
      get() {
        return elemHtml || el._innerHTML;
      },
      set(val) {
        elemHtml = String(val);
        el._innerHTML = elemHtml;
        registerIdsFromHtml(elemHtml);
      }
    });
    return el;
  }

  // Known core layout elements
  const viewEl = makeStubContainer("view");
  for (const id of [
    "api-error-banner",
    "sidebar-doc-title", "sidebar-doc-meta", "sidebar-doc-pill", "topbar-doc-name",
    "diagnostic-modal", "diagnostic-modal-body", "toast-container",
    "quiz-card-box", "quiz-perf-number", "quiz-perf-donut", "quiz-encouragement",
    "btn-quiz-submit", "btn-quiz-skip", "quiz-feedback-container",
    "btn-start-recall-timer", "btn-ready-test-now", "btn-submit-active-recall",
    "active-recall-text", "verification-answer-input", "recall-stats-display",
    "btn-generate-smart-notes", "btn-note-audio", "btn-note-pdf", "btn-note-pptx", "btn-note-diagram",
    "btn-save-note-mp3",
    "note-audio-player-box", "note-audio-element", "task-selector-error", "task-selector-grid",
    "smart-notes-concept-select", "custom-instructions-input", "active-note-view",
    "note-diagram-box", "note-diagram-inner"
  ]) {
    if (!byId.has(id)) makeStubContainer(id);
  }

  const quizBox = byId.get("quiz-card-box");
  let quizOptButtons = [];
  let quizRawHtml = "";
  Object.defineProperty(quizBox, "innerHTML", {
    get() {
      return quizRawHtml;
    },
    set(val) {
      quizRawHtml = String(val);
      if (quizRawHtml.includes('id="btn-quiz-submit"')) {
        const submitBtn = makeStubContainer("btn-quiz-submit", "BUTTON");
        submitBtn.textContent = "Submit Answer";
        submitBtn.disabled = quizRawHtml.includes('id="btn-quiz-submit" disabled');
        const skipBtn = makeStubContainer("btn-quiz-skip", "BUTTON");
        skipBtn.textContent = "Skip Question";
        makeStubContainer("quiz-feedback-container");

        quizOptButtons = [];
        const optMatches = [...quizRawHtml.matchAll(/class="quiz-opt-btn"[^>]*data-index="(\d+)"/g)];
        for (const m of optMatches) {
          const b = makeMockElement("", "BUTTON");
          b.setAttribute("data-index", m[1]);
          quizOptButtons.push(b);
        }
      }
    }
  });
  quizBox.querySelectorAll = sel => (sel === ".quiz-opt-btn" ? quizOptButtons : []);

  let viewRawHtml = "";
  Object.defineProperty(viewEl, "innerHTML", {
    get() {
      return viewRawHtml;
    },
    set(val) {
      viewRawHtml = String(val);
      registerIdsFromHtml(viewRawHtml);
      if (viewRawHtml.includes('id="quiz-card-box"')) {
        byId.set("quiz-card-box", quizBox);
      }
      for (const btnId of [
        "btn-start-recall-timer",
        "btn-ready-test-now",
        "btn-submit-active-recall",
        "btn-generate-smart-notes",
        "btn-note-audio",
        "btn-note-pdf",
        "btn-note-pptx",
        "btn-note-diagram",
        "btn-save-note-mp3"
      ]) {
        if (viewRawHtml.includes(`id="${btnId}"`)) {
          makeStubContainer(btnId, "BUTTON");
        }
      }
      if (viewRawHtml.includes('id="note-diagram-box"')) {
        const diagramBox = makeStubContainer("note-diagram-box", "DIV");
        const match = viewRawHtml.match(/id="note-diagram-box"[^>]*style="([^"]*)"/) ||
                      viewRawHtml.match(/style="([^"]*)"[^>]*id="note-diagram-box"/);
        if (match) {
          diagramBox.setAttribute("style", match[1]);
          if (match[1].includes("display:none")) {
            diagramBox.style.display = "none";
          }
        }
      }
      if (viewRawHtml.includes('id="note-diagram-inner"')) {
        const diagramInner = makeStubContainer("note-diagram-inner", "DIV");
        const match = viewRawHtml.match(/id="note-diagram-inner"[^>]*style="([^"]*)"/) ||
                      viewRawHtml.match(/style="([^"]*)"[^>]*id="note-diagram-inner"/);
        if (match) {
          diagramInner.setAttribute("style", match[1]);
        }
      }
      if (viewRawHtml.includes('class="task-card')) {
        const cards = [];
        const matches = [...viewRawHtml.matchAll(/class="task-card[^"]*"[^>]*data-task="([^"]+)"/g)];
        for (const m of matches) {
          const c = makeMockElement("", "DIV");
          c.setAttribute("data-task", m[1]);
          cards.push(c);
        }
        classElements.set(".task-card", cards);
      } else {
        classElements.set(".task-card", []);
      }
    }
  });

  const documentObj = {
    readyState: "complete",
    activeElement: null,
    documentElement: makeMockElement("html"),
    body: makeMockElement("body"),
    querySelector(sel) {
      if (sel.startsWith("#")) return byId.get(sel.slice(1)) || null;
      return null;
    },
    getElementById(id) {
      return byId.get(id) || null;
    },
    querySelectorAll(sel) {
      return classElements.get(sel) || [];
    },
    createElement(tag) {
      return makeMockElement("", tag);
    },
    addEventListener(type, fn) {
      if (!docListeners.has(type)) docListeners.set(type, new Set());
      docListeners.get(type).add(fn);
    },
    removeEventListener(type, fn) {
      if (docListeners.has(type)) docListeners.get(type).delete(fn);
    }
  };

  const windowObj = {
    location: { protocol: "http:", hostname: "localhost", port: "8000", hash: "#settings" },
    sessionStorage: sessionStorageObj,
    addEventListener(type, fn) {
      if (!winListeners.has(type)) winListeners.set(type, new Set());
      winListeners.get(type).add(fn);
    },
    removeEventListener(type, fn) {
      if (winListeners.has(type)) winListeners.get(type).delete(fn);
    },
    dispatchEvent() {},
    __winListeners: winListeners
  };

  const fetchImpl = async (url, opts = {}) => {
    const method = (opts.method || "GET").toUpperCase();
    const pathname = url.replace(/^https?:\/\/[^/]+/, "").split("?")[0];
    const handled = await routeHandlers({ method, url, pathname, opts });
    if (handled) return handled;
    if (pathname.startsWith("/api/artifacts/") && pathname.endsWith(".mp3")) {
      return audioBlobResponse(5000, 200);
    }
    return jsonResponse({});
  };

  const context = {
    console,
    URLSearchParams,
    AbortController,
    CustomEvent: class CustomEvent {
      constructor(type, init = {}) {
        this.type = type;
        this.detail = init.detail;
      }
    },
    localStorage: {
      getItem: k => (storage.has(k) ? storage.get(k) : null),
      setItem: (k, v) => storage.set(k, String(v)),
      removeItem: k => storage.delete(k)
    },
    sessionStorage: sessionStorageObj,
    window: windowObj,
    document: documentObj,
    location: windowObj.location,
    fetch: fetchImpl,
    setTimeout,
    clearTimeout,
    setInterval: () => 1,
    clearInterval: () => {},
    confirm: () => true,
    renderMathInElement: null,
    Blob,
    URL
  };
  context.globalThis = context;
  vm.createContext(context);
  vm.runInContext(fs.readFileSync(API_JS_PATH, "utf8"), context, { filename: "api.js" });
  vm.runInContext(fs.readFileSync(APP_JS_PATH, "utf8"), context, { filename: "app.js" });

  return { context, documentObj, windowObj, byId, classElements, viewEl, quizBox, getQuizOptButtons: () => quizOptButtons };
}

function jsonResponse(obj, status = 200) {
  const str = JSON.stringify(obj);
  return {
    ok: status >= 200 && status < 300,
    status,
    headers: { get: (k) => (k && k.toLowerCase() === "content-type" ? "application/json" : null) },
    json: async () => obj,
    text: async () => str,
    blob: async () => new Blob([str], { type: "application/json" })
  };
}

function audioBlobResponse(size = 5000, status = 200) {
  const buffer = new Uint8Array(size);
  const blob = new Blob([buffer], { type: "audio/mpeg" });
  return {
    ok: status >= 200 && status < 300,
    status,
    headers: {
      get: (k) => {
        if (!k) return null;
        const lk = k.toLowerCase();
        if (lk === "content-type") return "audio/mpeg";
        if (lk === "content-length") return String(size);
        return null;
      }
    },
    blob: async () => blob,
    arrayBuffer: async () => buffer.buffer
  };
}

async function testQuizDoubleClickPreventsCounterOverflow() {
  let nextQuestionCalls = 0;
  let submitAnswerCalls = 0;
  const nextQuestionDeferreds = [];
  const submitDeferreds = [];

  const harness = await createAppHarness(async ({ pathname, opts }) => {
    if (pathname === "/api/materials") return jsonResponse({ materials: [{ id: "m1", title: "ML Notes", chunk_count: 3, concept_count: 2 }] });
    if (pathname === "/api/mastery") return jsonResponse({ concepts: [{ id: "c1", name: "Gradient Descent", score: 0.5, attempts: 1 }] });
    if (pathname === "/api/quiz/next") {
      nextQuestionCalls += 1;
      const d = createDeferred();
      nextQuestionDeferreds.push(d);
      if (opts.signal) {
        opts.signal.addEventListener("abort", () => d.reject(createAbortError()), { once: true });
      }
      return await d.promise;
    }
    if (pathname === "/api/quiz/answer") {
      submitAnswerCalls += 1;
      const d = createDeferred();
      submitDeferreds.push(d);
      return await d.promise;
    }
    return jsonResponse({});
  });

  harness.windowObj.location.hash = "#practice";
  const hashListeners = [...(harness.windowObj.__winListeners.get("hashchange") || [])];
  assert.equal(hashListeners.length, 1);
  hashListeners[0]();

  await new Promise(r => setTimeout(r, 20));
  assert.equal(nextQuestionCalls, 1);
  nextQuestionDeferreds[0].resolve(jsonResponse({
    id: "q1",
    conceptId: "c1",
    conceptName: "Gradient Descent",
    questionType: "mcq",
    question: "Question 1 text",
    options: ["Opt A", "Opt B", "Opt C", "Opt D"],
    correct: 0
  }));
  await new Promise(r => setTimeout(r, 15));

  assert.ok(harness.quizBox.innerHTML.includes("Question 1 of 6"), "Should render Question 1 of 6");

  // Rapid double-click Submit Answer
  harness.getQuizOptButtons()[0].click();
  const submitBtn = harness.byId.get("btn-quiz-submit");
  assert.equal(submitBtn.disabled, false);

  const submitClick1 = submitBtn.click();
  const submitClick2 = submitBtn.click();
  assert.equal(submitBtn.disabled, true, "Quiz Submit button must be disabled immediately on click");
  assert.equal(submitAnswerCalls, 1, "Rapid double-click on Quiz Submit must only fire 1 request");

  submitDeferreds[0].resolve(jsonResponse({
    correct: true,
    correctIndex: 0,
    explanation: "Correct reasoning.",
    next: "c1"
  }));
  await Promise.all([submitClick1, submitClick2]);

  assert.equal(submitBtn.textContent, "Next Question →");
  assert.equal(submitBtn.disabled, false);

  const keyListenersBeforeNext = [...(harness.windowObj.__winListeners.get("keydown") || [])];
  assert.equal(keyListenersBeforeNext.length, 1, "Must have 1 active quiz keydown listener for Q1");

  // Rapid triple-click + Enter keydown on Next Question while loadQuizQuestion is pending
  const nextClick1 = submitBtn.click();
  assert.equal(submitBtn.disabled, true, "Next Question button must be disabled immediately until next question settles");
  const nextClick2 = submitBtn.click();
  assert.equal((harness.windowObj.__winListeners.get("keydown") || new Set()).size, 0, "Quiz keydown listener should detach while next question loads");
  keyListenersBeforeNext[0]({ key: "Enter" });

  await new Promise(r => setTimeout(r, 10));
  assert.equal(nextQuestionCalls, 2, "Rapid Next Question clicks/Enters must only request Q2 once");

  nextQuestionDeferreds[1].resolve(jsonResponse({
    id: "q2",
    conceptId: "c1",
    conceptName: "Gradient Descent",
    questionType: "mcq",
    question: "Question 2 text",
    options: ["Opt A", "Opt B", "Opt C", "Opt D"],
    correct: 1
  }));
  await Promise.all([nextClick1, nextClick2]);

  assert.ok(harness.quizBox.innerHTML.includes("Question 2 of 6"), "Expected Question 2 of 6");
  assert.equal((harness.windowObj.__winListeners.get("keydown") || new Set()).size, 1, "Must keep only 1 active quiz keydown listener on Q2");

  // Rapid double-click on Skip Question at Q2 -> must only advance once to Q3
  const skipBtn = harness.byId.get("btn-quiz-skip");
  const skip1 = skipBtn.click();
  assert.equal(skipBtn.disabled, true, "Skip button must lock while advancing");
  const skip2 = skipBtn.click();
  await new Promise(r => setTimeout(r, 10));
  assert.equal(nextQuestionCalls, 3, "Rapid double-click on Skip must only request 1 next question");

  nextQuestionDeferreds[2].resolve(jsonResponse({
    id: "q3",
    conceptId: "c1",
    conceptName: "Gradient Descent",
    questionType: "mcq",
    question: "Question 3 text",
    options: ["Opt A", "Opt B", "Opt C", "Opt D"],
    correct: 2
  }));
  await Promise.all([skip1, skip2]);
  assert.ok(harness.quizBox.innerHTML.includes("Question 3 of 6"), "Expected Question 3 of 6 after single skip");
}

async function testActiveRecallNotesAudioLocksAndTabAbort() {
  let evalCalls = 0;
  let noteGenCalls = 0;
  let audioGenCalls = 0;
  let dashboardMasteryDeferred = null;
  let dashboardAborted = false;
  const evalDeferreds = [];
  const noteDeferreds = [];
  const audioDeferreds = [];

  const harness = await createAppHarness(async ({ pathname, opts }) => {
    if (pathname === "/api/materials") return jsonResponse({ materials: [{ id: "m1", title: "ML Notes", chunk_count: 3, concept_count: 2 }] });
    if (pathname === "/api/mastery") {
      if (opts && opts.signal && !dashboardMasteryDeferred) {
        dashboardMasteryDeferred = createDeferred();
        opts.signal.addEventListener("abort", () => {
          dashboardAborted = true;
          dashboardMasteryDeferred.reject(createAbortError("Tab switched"));
        }, { once: true });
        return await dashboardMasteryDeferred.promise;
      }
      return jsonResponse({ concepts: [{ id: "c1", name: "Backpropagation", score: 0.4, attempts: 1 }] });
    }
    if (pathname === "/api/recommendations") return jsonResponse({ actions: [] });
    if (pathname === "/api/loop/status") return jsonResponse({ loop_state: "READY", material_ready: true, baseline_completed: true });
    if (pathname === "/api/concepts/graph") return jsonResponse({ nodes: [{ id: "c1", label: "Backpropagation" }], edges: [] });
    if (pathname === "/api/active-recall/start") {
      return jsonResponse({
        session_id: "ar-1",
        concept_id: "c1",
        concept_name: "Backpropagation",
        core_intuition: "Chain rule",
        formula_or_mechanism: "$\\nabla L$",
        worked_example: "Ex",
        keywords: ["chain rule"],
        verification_question: "Explain backprop."
      });
    }
    if (pathname === "/api/active-recall/evaluate") {
      evalCalls += 1;
      const d = createDeferred();
      evalDeferreds.push(d);
      return await d.promise;
    }
    if (pathname === "/api/notes") return jsonResponse([]);
    if (pathname === "/api/notes/generate") {
      noteGenCalls += 1;
      const d = createDeferred();
      noteDeferreds.push(d);
      return await d.promise;
    }
    if (pathname === "/api/notes/note-1/audio") {
      audioGenCalls += 1;
      const d = createDeferred();
      audioDeferreds.push(d);
      return await d.promise;
    }
    if (pathname.startsWith("/api/artifacts/") && pathname.endsWith(".mp3")) {
      return audioBlobResponse(5000, 200);
    }
    return jsonResponse({});
  });

  // 1. Tab switch AbortController test: trigger #dashboard, then immediately switch to #smart-notes
  const hashListeners = [...(harness.windowObj.__winListeners.get("hashchange") || [])];
  harness.windowObj.location.hash = "#dashboard";
  hashListeners[0]();
  await new Promise(r => setTimeout(r, 10));

  harness.windowObj.location.hash = "#smart-notes";
  hashListeners[0]();
  await new Promise(r => setTimeout(r, 30));

  assert.equal(dashboardAborted, true, "Switching tabs must abort in-flight previous tab fetches");
  assert.ok(harness.viewEl.innerHTML.includes("Smart Study Notes"), "Active view should render Smart Notes cleanly");

  // 2. Smart Notes: select a task card, click #btn-generate-smart-notes twice concurrently, verify 1 call + lock
  const taskCards = harness.classElements.get(".task-card") || [];
  assert.ok(taskCards.length > 0, "Expected .task-card elements in Smart Notes view");
  taskCards[0].click();

  const genBtn = harness.byId.get("btn-generate-smart-notes");
  const genClick1 = genBtn.click();
  assert.equal(genBtn.disabled, true, "Generate Notes button must disable immediately");
  const genClick2 = genBtn.click();
  await new Promise(r => setTimeout(r, 10));
  assert.equal(noteGenCalls, 1, "Double-click on Generate Notes must only trigger 1 request");

  noteDeferreds[0].resolve(jsonResponse({
    note_id: "note-1",
    concept_id: "c1",
    task_type: "exam",
    title: "Backprop Notes",
    content_markdown: "Core chain rule explanation.",
    citations: []
  }));
  await Promise.all([genClick1, genClick2]);
  await new Promise(r => setTimeout(r, 25));

  // Now activeNote is rendered -> #btn-note-audio is active! Test failure + finally recovery first
  const audioBtn = harness.byId.get("btn-note-audio");
  assert.ok(audioBtn, "Expected #btn-note-audio after generating a note");
  const audClick1 = audioBtn.click();
  assert.equal(audioBtn.disabled, true, "Generate Audio button must disable immediately");
  const audClick2 = audioBtn.click();
  await new Promise(r => setTimeout(r, 10));
  assert.equal(audioGenCalls, 1, "Double-click on Generate Audio must only trigger 1 request");

  audioDeferreds[0].resolve(jsonResponse({ detail: "TTS busy" }, 503));
  await Promise.all([audClick1, audClick2]);
  assert.equal(audioBtn.disabled, false, "Generate Audio button must re-enable cleanly in finally on error");

  // Retry succeeds and sets Audio Ready
  const audRetry = audioBtn.click();
  assert.equal(audioBtn.disabled, true);
  audioDeferreds[1].resolve(jsonResponse({ audio_url: "/api/artifacts/audio_note-1.mp3" }));
  await audRetry;
  assert.equal(audioBtn.disabled, false);

  // 3. Active Recall Stage 1 -> Stage 2 -> Stage 3 Submit double-click lock & error recovery
  harness.windowObj.location.hash = "#active-recall";
  hashListeners[0]();
  await new Promise(r => setTimeout(r, 25));

  harness.byId.get("btn-start-recall-timer").click();
  await new Promise(r => setTimeout(r, 25));
  harness.byId.get("btn-ready-test-now").click();
  await new Promise(r => setTimeout(r, 25));

  const arSubmitBtn = harness.byId.get("btn-submit-active-recall");
  assert.ok(arSubmitBtn, "Expected #btn-submit-active-recall in Stage 3");

  const arClick1 = arSubmitBtn.click();
  assert.equal(arSubmitBtn.disabled, true, "Active Recall Submit must disable immediately on click");
  const arClick2 = arSubmitBtn.click();
  await new Promise(r => setTimeout(r, 10));
  assert.equal(evalCalls, 1, "Double-click on Active Recall Submit must only fire 1 request");

  evalDeferreds[0].resolve(jsonResponse({ detail: "Temporary error" }, 500));
  await Promise.all([arClick1, arClick2]);
  assert.equal(arSubmitBtn.disabled, false, "Active Recall Submit must re-enable cleanly in try/finally after error");
}

async function testMockGatingAndErrorBanner() {
  // 1. index.html must not unconditionally include js/mock.js and must check ?mock=true / USE_MOCK
  const html = fs.readFileSync(INDEX_HTML_PATH, "utf8");
  assert.ok(!html.includes('<script src="js/mock.js"></script>'), "index.html must not unconditionally load js/mock.js");
  assert.ok(html.includes('USE_MOCK') && html.includes('"mock") === "true"'), "index.html loader must gate js/mock.js behind ?mock=true or USE_MOCK");
  assert.ok(html.includes('id="api-error-banner"'), "index.html must include #api-error-banner container");

  // 2. Live mode ignores stale concept_mastery_config_v1.mock=true and never registers window.MOCK
  let fetchCount = 0;
  function buildSandboxWithSearch(searchStr, storageSeed = {}, fetchFn = null) {
    const storage = new Map(Object.entries(storageSeed));
    const context = {
      console,
      URLSearchParams,
      AbortController,
      CustomEvent: class CustomEvent {
        constructor(type, init = {}) {
          this.type = type;
          this.detail = init.detail;
        }
      },
      localStorage: {
        getItem: k => (storage.has(k) ? storage.get(k) : null),
        setItem: (k, v) => storage.set(k, String(v)),
        removeItem: k => storage.delete(k)
      },
      location: { protocol: "http:", hostname: "localhost", port: "8000", hash: "#dashboard", search: searchStr },
      dispatchEvent() {},
      addEventListener() {},
      removeEventListener() {},
      fetch: fetchFn || (async () => {
        fetchCount += 1;
        return jsonResponse({ detail: "Backend exploded" }, 500);
      }),
      setTimeout,
      clearTimeout
    };
    context.window = context;
    context.globalThis = context;
    vm.createContext(context);
    vm.runInContext(fs.readFileSync(API_JS_PATH, "utf8"), context, { filename: "api.js" });
    vm.runInContext(fs.readFileSync(MOCK_JS_PATH, "utf8"), context, { filename: "mock.js" });
    return context;
  }

  const liveCtx = buildSandboxWithSearch("", {
    concept_mastery_config_v1: JSON.stringify({ mock: true, base: "" })
  });
  assert.equal(liveCtx.window.MOCK, undefined, "mock.js must not expose window.MOCK in live mode even when loaded");
  assert.equal(liveCtx.window.ApiService.isMockMode(), false, "Stale config mock:true must NOT enable mock mode");
  assert.equal(liveCtx.window.ApiService.getConfig().mock, false);
  await assert.rejects(
    liveCtx.window.ApiService.getMastery(),
    err => err && err.status === 500 && err.message.includes("Backend exploded")
  );
  assert.equal(fetchCount, 1, "Live mode must call real fetch and reject on 500 instead of falling back to mock");

  // 3. Explicit ?mock=true URL query param enables mock mode
  let queryFetchCalls = 0;
  const queryMockCtx = buildSandboxWithSearch("?mock=true", {}, async () => {
    queryFetchCalls += 1;
    return jsonResponse({});
  });
  assert.ok(queryMockCtx.window.MOCK, "mock.js must attach window.MOCK when ?mock=true");
  assert.equal(queryMockCtx.window.ApiService.isMockMode(), true);
  const queryMastery = await queryMockCtx.window.ApiService.getMastery();
  assert.ok(Array.isArray(queryMastery.concepts) && queryMastery.concepts.length > 0);
  assert.equal(queryFetchCalls, 0);

  // 4. Explicit localStorage.USE_MOCK === "true" enables mock mode
  let lsFetchCalls = 0;
  const lsMockCtx = buildSandboxWithSearch("", { USE_MOCK: "true" }, async () => {
    lsFetchCalls += 1;
    return jsonResponse({});
  });
  assert.ok(lsMockCtx.window.MOCK, "mock.js must attach window.MOCK when USE_MOCK === 'true'");
  assert.equal(lsMockCtx.window.ApiService.isMockMode(), true);
  const lsMastery = await lsMockCtx.window.ApiService.getMastery();
  assert.ok(Array.isArray(lsMastery.concepts) && lsMastery.concepts.length > 0);
  assert.equal(lsFetchCalls, 0);

  // 5. Real backend errors in UI surface toast AND #api-error-banner instead of silent fallback or fake empty state
  const harness = await createAppHarness(async ({ pathname }) => {
    if (pathname === "/api/materials") return jsonResponse({ materials: [] });
    if (pathname === "/api/mastery") return jsonResponse({ detail: "Database offline" }, 503);
    if (pathname === "/api/recommend") return jsonResponse({ actions: [] });
    if (pathname === "/api/loop/status") return jsonResponse({ loop_state: "UNINITIALIZED" });
    return jsonResponse({});
  });

  const hashListeners = [...(harness.windowObj.__winListeners.get("hashchange") || [])];
  harness.windowObj.location.hash = "#dashboard";
  hashListeners[0]();
  await new Promise(r => setTimeout(r, 30));

  const errorBanner = harness.byId.get("api-error-banner");
  const toastContainer = harness.byId.get("toast-container");
  assert.equal(errorBanner.hidden, false, "#api-error-banner must be visible on backend error");
  assert.ok(errorBanner.innerHTML.includes("Database offline"), "Error banner must display backend error detail");
  assert.ok(toastContainer.children.length >= 1, "Error toast must be shown on backend failure");
  assert.ok(harness.viewEl.innerHTML.includes("Backend Unavailable"), "Dashboard must show Backend Unavailable instead of silent empty state");
}

async function testBrowserDownloadHelperBlobAndRevokeDelay() {
  const timeouts = [];
  const revokedUrls = [];
  const createdUrls = [];
  const clickedAnchors = [];

  const fakeBlob = new Blob([new Uint8Array([37, 80, 68, 70])], { type: "application/pdf" });

  const customURL = {
    createObjectURL(blob) {
      const url = `blob:test-${Math.random()}`;
      createdUrls.push({ url, blob });
      return url;
    },
    revokeObjectURL(url) {
      revokedUrls.push(url);
    }
  };

  const harness = await createAppHarness(async ({ url }) => {
    if (url.includes("/api/artifacts/")) {
      return {
        ok: true,
        status: 200,
        headers: {
          get: (h) => (h.toLowerCase() === "content-type" ? "application/pdf" : null)
        },
        blob: async () => fakeBlob
      };
    }
    return jsonResponse({});
  });

  // Inject tracking spys into harness context
  harness.context.URL = customURL;
  harness.context.window.URL = customURL;
  const originalSetTimeout = harness.context.setTimeout;
  harness.context.setTimeout = (fn, delay) => {
    timeouts.push({ fn, delay });
    return originalSetTimeout(fn, 10); // Run faster in test while recording real delay
  };

  const origCreateElement = harness.documentObj.createElement;
  harness.documentObj.createElement = (tag) => {
    const el = origCreateElement(tag);
    if (tag.toLowerCase() === "a") {
      el.click = () => {
        clickedAnchors.push(el);
      };
    }
    return el;
  };

  // Test ApiService.downloadArtifact directly
  const apiSvc = harness.windowObj.ApiService || harness.context.window.ApiService;
  await apiSvc.downloadArtifact("/api/artifacts/test.pdf", "CustomNote.pdf");

  assert.equal(createdUrls.length, 1, "Must create ObjectURL from Blob");
  assert.equal(createdUrls[0].blob.type, "application/pdf", "Blob must preserve application/pdf content type");
  assert.equal(clickedAnchors.length, 1, "Anchor click must be triggered");
  assert.equal(clickedAnchors[0].download, "CustomNote.pdf", "Anchor download attribute must match filename");
  assert.equal(clickedAnchors[0].href, createdUrls[0].url, "Anchor href must point to created Blob URL");

  const timeoutEntry = timeouts.find(t => t.delay === 60000);
  assert.ok(timeoutEntry, "URL.revokeObjectURL and anchor cleanup must be scheduled with a 60-second delay (60000ms)");
  assert.equal(revokedUrls.length, 0, "ObjectURL must NOT be revoked immediately after click");

  // Fire timeout callback to verify revocation
  timeoutEntry.fn();
  assert.equal(revokedUrls.length, 1, "ObjectURL must be revoked after timeout callback fires");
  assert.equal(revokedUrls[0], createdUrls[0].url, "Revoked URL must match created Blob URL");
}

async function testConceptFlowDiagramSvgInjectionAndSpinner() {
  let resolveDiagram = null;
  let noteDeferred = null;

  const mockSvg = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 900 420"><text fill="#e2e8f0">Flow Node</text></svg>';

  const harness = await createAppHarness(async ({ pathname, opts }) => {
    if (pathname === "/api/tools/generate") {
      return new Promise((resolve) => {
        resolveDiagram = () => resolve(jsonResponse({
          artifact_type: "visual",
          svg: mockSvg,
          download_url: "/api/artifacts/diagram_123.svg"
        }));
      });
    }
    if (pathname === "/api/notes/generate") {
      return new Promise((resolve) => {
        noteDeferred = () => resolve(jsonResponse({
          note_id: "note-diag-1",
          concept_id: "gradient_descent",
          task_type: "exam",
          title: "Gradient Descent Notes",
          markdown_content: "# Gradient Descent\nOptimization mechanics."
        }));
      });
    }
    if (pathname === "/api/notes") {
      return jsonResponse([]);
    }
    if (pathname === "/api/mastery") {
      return jsonResponse({
        overall_score: 85,
        concepts: [{ concept_id: "gradient_descent", name: "Gradient Descent", mastery_score: 85 }]
      });
    }
    return jsonResponse({});
  });

  // Switch to smart-notes view
  const hashListeners = [...(harness.windowObj.__winListeners.get("hashchange") || [])];
  harness.windowObj.location.hash = "#smart-notes";
  hashListeners[0]();
  await new Promise(r => setTimeout(r, 30));

  // Select task card and generate note
  const taskCards = harness.classElements.get(".task-card") || [];
  assert.ok(taskCards.length > 0, "Expected .task-card elements in Smart Notes view");
  taskCards[0].click();

  const genBtn = harness.byId.get("btn-generate-smart-notes");
  assert.ok(genBtn, "Expected #btn-generate-smart-notes");
  const genPromise = genBtn.click();
  await new Promise(r => setTimeout(r, 10));
  assert.ok(noteDeferred, "Expected note generation request in flight");
  noteDeferred();
  await genPromise;
  await new Promise(r => setTimeout(r, 30));

  const diagramBtn = harness.byId.get("btn-note-diagram");
  const diagramBox = harness.byId.get("note-diagram-box");
  const diagramInner = harness.byId.get("note-diagram-inner");

  assert.ok(diagramBtn, "#btn-note-diagram must exist after note is generated");
  assert.ok(diagramBox, "#note-diagram-box must exist");
  assert.ok(diagramInner, "#note-diagram-inner must exist");

  // Initial style check on diagram inner container
  const innerStyle = diagramInner.getAttribute("style") || "";
  assert.ok(
    innerStyle.includes("min-height: 280px") || innerStyle.includes("min-height:280px"),
    "Diagram inner container must specify min-height: 280px"
  );
  assert.ok(
    innerStyle.includes("overflow-x: auto") || innerStyle.includes("overflow-x:auto"),
    "Diagram inner container must specify overflow-x: auto"
  );

  // Trigger diagram generation click
  const clickPromise = diagramBtn.click();
  await new Promise(r => setTimeout(r, 10));

  // Box must be visible and inner container must show loading spinner
  assert.equal(diagramBox.style.display, "block", "Diagram box must be displayed while generating");
  assert.ok(
    diagramInner.innerHTML.includes("Rendering Concept Flow Diagram") ||
    diagramInner.innerHTML.includes("spinner") ||
    diagramInner.innerHTML.includes("Synthesizing"),
    "Inner container must show loading spinner while generating"
  );
  assert.equal(diagramBtn.disabled, true, "Generate Diagram button must be locked while generating");

  // Resolve API promise
  assert.ok(resolveDiagram, "Diagram API call must be in-flight");
  resolveDiagram();
  await clickPromise;
  await new Promise(r => setTimeout(r, 10));

  // After completion: SVG must be injected
  assert.ok(diagramInner.innerHTML.includes('<svg xmlns="http://www.w3.org/2000/svg"'), "Inner container must contain the injected SVG element");
  assert.ok(diagramInner.innerHTML.includes("Flow Node"), "Inner container must contain the SVG text node");
  assert.equal(diagramBtn.disabled, false, "Generate Diagram button must be unlocked after resolution");
}

async function testAudioPlayerBlobSizeAndDelayedRevoke() {
  // 1. ApiService.fetchAudioBlob unit validation
  let fetchStatus = 200;
  let fetchBlobSize = 5000;
  const { ApiService } = createApiSandbox(async () => {
    return {
      ok: fetchStatus >= 200 && fetchStatus < 300,
      status: fetchStatus,
      blob: async () => new Blob([new Uint8Array(fetchBlobSize)], { type: "audio/mpeg" })
    };
  });

  // Rejects on HTTP error (e.g. 503)
  fetchStatus = 503;
  await assert.rejects(
    () => ApiService.fetchAudioBlob("/api/artifacts/audio_test.mp3"),
    err => err.message.includes("503")
  );

  // Rejects on empty or truncated blob (size <= 100)
  fetchStatus = 200;
  fetchBlobSize = 40;
  await assert.rejects(
    () => ApiService.fetchAudioBlob("/api/artifacts/audio_test.mp3"),
    err => err.message.includes("100") || err.message.includes("empty") || err.message.includes("invalid")
  );

  // Succeeds on valid blob size > 100 and returns Blob with type audio/mpeg
  fetchStatus = 200;
  fetchBlobSize = 5000;
  const validBlob = await ApiService.fetchAudioBlob("/api/artifacts/audio_test.mp3");
  assert.ok(validBlob, "Expected valid blob");
  assert.equal(validBlob.type, "audio/mpeg");
  assert.equal(validBlob.size, 5000);

  // 2. Integration test with UI harness
  let audioGenDeferred = null;
  const createdBlobUrls = [];
  const revokedBlobUrls = [];
  const timeoutCallbacks = [];

  const harness = await createAppHarness(async ({ pathname }) => {
    if (pathname === "/api/notes/generate") {
      return jsonResponse({
        note_id: "note-audio-test",
        concept_id: "neural_nets",
        task_type: "exam",
        title: "Neural Nets Note",
        markdown_content: "# Neural Nets\nBackprop and SGD mechanics."
      });
    }
    if (pathname === "/api/notes/note-audio-test/audio") {
      return new Promise(resolve => {
        audioGenDeferred = () => resolve(jsonResponse({
          audio_url: "/api/artifacts/audio_nn.mp3"
        }));
      });
    }
    if (pathname === "/api/artifacts/audio_nn.mp3") {
      return audioBlobResponse(6311, 200);
    }
    if (pathname === "/api/notes") return jsonResponse([]);
    if (pathname === "/api/mastery") {
      return jsonResponse({
        overall_score: 90,
        concepts: [{ concept_id: "neural_nets", name: "Neural Networks", mastery_score: 90 }]
      });
    }
    return jsonResponse({});
  });

  harness.context.URL.createObjectURL = (blob) => {
    const url = `blob:http://localhost:8000/${Math.random().toString(36).slice(2)}`;
    createdBlobUrls.push({ url, blob });
    return url;
  };
  harness.context.URL.revokeObjectURL = (url) => {
    revokedBlobUrls.push(url);
  };
  harness.context.setTimeout = (fn, ms) => {
    timeoutCallbacks.push({ fn, ms });
    return 1;
  };

  // Switch to smart-notes view
  const hashListeners = [...(harness.windowObj.__winListeners.get("hashchange") || [])];
  harness.windowObj.location.hash = "#smart-notes";
  hashListeners[0]();
  await new Promise(r => setTimeout(r, 30));

  // Select task card and generate note
  const taskCards = harness.classElements.get(".task-card") || [];
  assert.ok(taskCards.length > 0, "Expected .task-card elements");
  taskCards[0].click();

  const genBtn = harness.byId.get("btn-generate-smart-notes");
  assert.ok(genBtn);
  await genBtn.click();
  await new Promise(r => setTimeout(r, 30));

  const audioBtn = harness.byId.get("btn-note-audio");
  const audioElement = harness.byId.get("note-audio-element");
  const playerBox = harness.byId.get("note-audio-player-box");
  const saveMp3Btn = harness.byId.get("btn-save-note-mp3");

  assert.ok(audioBtn, "Audio button must exist");
  assert.ok(audioElement, "Audio element must exist");
  assert.ok(playerBox, "Player box must exist");
  assert.ok(saveMp3Btn, "Save MP3 button must exist");

  // Track load() calls on audio element
  let loadCalled = false;
  audioElement.load = () => { loadCalled = true; };

  // Click audio button
  const audClickPromise = audioBtn.click();
  await new Promise(r => setTimeout(r, 10));
  assert.equal(audioBtn.disabled, true, "Audio button must lock while synthesizing");

  assert.ok(audioGenDeferred, "Audio generation must be in flight");
  audioGenDeferred();
  await audClickPromise;
  await new Promise(r => setTimeout(r, 20));

  assert.equal(audioBtn.disabled, false, "Audio button must unlock after completion");
  assert.ok(audioBtn.innerHTML.includes("Audio Ready"), "Audio button must show Audio Ready");
  assert.ok(audioElement.src.startsWith("blob:"), "Audio element src must be a Blob URL");
  assert.equal(loadCalled, true, "audioElement.load() must have been called");
  assert.ok(playerBox.classList.contains("visible"), "Audio player box must be visible");

  // Click Save MP3 button
  await saveMp3Btn.click();
  const revokeEntry = timeoutCallbacks.find(e => e.ms === 60000);
  assert.ok(revokeEntry, "Delayed revokeObjectURL timeout of 60000ms (60s) must be registered");
}

async function testDashboardSwrCacheParallelSettledAndSkeletonDecoupling() {
  // --- Part 1: SWR Cache Unit Tests on ApiService ---
  const { ApiService, context } = createApiSandbox(async () => jsonResponse({}));

  assert.equal(typeof ApiService.getSwr, "function", "ApiService.getSwr must be defined");
  assert.equal(typeof ApiService.setSwr, "function", "ApiService.setSwr must be defined");
  assert.equal(typeof ApiService.clearSwr, "function", "ApiService.clearSwr must be defined");
  assert.equal(typeof ApiService.fetchWithSwr, "function", "ApiService.fetchWithSwr must be defined");
  assert.equal(typeof ApiService.listNotes, "function", "ApiService.listNotes alias must be defined");

  // In-memory + sessionStorage caching
  ApiService.setSwr("mastery", { concepts: [{ id: "c1", name: "Backpropagation" }] }, { userId: "user-1" });
  const cachedVal = ApiService.getSwr("mastery", { userId: "user-1" });
  assert.deepEqual(cachedVal, { concepts: [{ id: "c1", name: "Backpropagation" }] });
  assert.ok(context.sessionStorage.getItem("cm_swr:user-1:all:mastery"), "SWR cache should persist to sessionStorage");

  // SWR clearing
  ApiService.clearSwr("mastery");
  assert.equal(ApiService.getSwr("mastery", { userId: "user-1" }), null);

  // fetchWithSwr with revalidation
  let networkCalls = 0;
  const swrFetchImpl = async () => {
    networkCalls++;
    return jsonResponse({ concepts: [{ id: "c2", name: "Convolutional Networks" }] });
  };
  const { ApiService: swrApi } = createApiSandbox(swrFetchImpl);
  swrApi.setSwr("mastery", { concepts: [{ id: "c1", name: "Cached ML" }] });

  let revalidatedData = null;
  const instantRes = await swrApi.fetchWithSwr("mastery", null, {}, (fresh) => {
    revalidatedData = fresh;
  });
  assert.deepEqual(instantRes, { concepts: [{ id: "c1", name: "Cached ML" }] });
  await new Promise(r => setTimeout(r, 20));
  assert.deepEqual(revalidatedData, { concepts: [{ id: "c2", name: "Convolutional Networks" }] });
  assert.equal(networkCalls, 1);

  // --- Part 2: Integration Test with App Harness ---
  const recDeferred = createDeferred();
  let fetchCounts = {
    materials: 0,
    mastery: 0,
    loopStatus: 0,
    notes: 0,
    recommend: 0
  };

  const harness = await createAppHarness(async ({ pathname }) => {
    if (pathname === "/api/materials") {
      fetchCounts.materials++;
      return jsonResponse({
        materials: [{ material_id: "mat_dl", title: "Deep Learning Foundations", ready: true, source_type: "pdf" }]
      });
    }
    if (pathname === "/api/mastery") {
      fetchCounts.mastery++;
      return jsonResponse({
        concepts: [
          { id: "c1", name: "Gradient Descent", score: 0.9, attempts: 4 },
          { id: "c2", name: "Backpropagation", score: 0.45, attempts: 2 }
        ]
      });
    }
    if (pathname === "/api/loop/status") {
      fetchCounts.loopStatus++;
      return jsonResponse({
        loop_state: "ACTIVE_LOOP",
        material_ready: true,
        baseline_completed: true
      });
    }
    if (pathname === "/api/notes") {
      fetchCounts.notes++;
      return jsonResponse({ notes: [] });
    }
    if (pathname === "/api/recommend") {
      fetchCounts.recommend++;
      return recDeferred.promise;
    }
    return jsonResponse({});
  });

  // Start with clean SWR cache
  harness.context.window.ApiService.clearSwr();

  // Navigate to Dashboard
  const hashListeners = [...(harness.windowObj.__winListeners.get("hashchange") || [])];
  assert.ok(hashListeners.length > 0, "hashchange listener must be registered");
  harness.windowObj.location.hash = "#dashboard";
  hashListeners[0]();

  // Allow fast local stats (Promise.allSettled) to resolve
  await new Promise(r => setTimeout(r, 35));

  // 1. Fast SQLite metrics and concept lists render immediately
  assert.ok(harness.viewEl.innerHTML.includes("Gradient Descent"), "Dashboard must render concept list from fast metrics");
  assert.ok(harness.viewEl.innerHTML.includes("Backpropagation"), "Dashboard must render concept list from fast metrics");
  assert.ok(harness.viewEl.innerHTML.includes("Your Learning Progress"), "Dashboard must render learning progress ring");

  // 2. Slow AI recommendations decouple into asynchronous skeleton shimmer
  const recContainer = harness.byId.get("dashboard-rec-container");
  assert.ok(recContainer, "#dashboard-rec-container must exist");
  assert.ok(harness.viewEl.innerHTML.includes("skeleton-shimmer"), "Recommendation card must display skeleton-shimmer while pending");
  assert.ok(harness.viewEl.innerHTML.includes("recommendation-card-skeleton"), "Skeleton container class must be present");

  // 3. Resolve decoupled AI recommendation
  recDeferred.resolve(jsonResponse({
    actions: [
      {
        type: "Remedial Review",
        name: "Backpropagation",
        conceptId: "c2",
        reason: "Target gradient vanishing and chain rule calculations"
      }
    ]
  }));
  await new Promise(r => setTimeout(r, 35));

  // 4. Recommendation sub-container updates cleanly without tearing down the dashboard
  assert.ok(!recContainer.innerHTML.includes("skeleton-shimmer"), "Skeleton shimmer must be replaced once recommendation resolves");
  assert.ok(recContainer.innerHTML.includes("Backpropagation"), "Recommendation card must display recommended concept name");
  assert.ok(recContainer.innerHTML.includes("Target gradient vanishing"), "Recommendation card must display recommendation reason");

  // 5. Verify SWR caching on tab switch: 0ms perceived latency, no "Loading dashboard..." spinner
  harness.windowObj.location.hash = "#upload";
  hashListeners[0]();
  await new Promise(r => setTimeout(r, 20));

  // Track if "Loading dashboard..." was ever set in viewEl
  let sawBlockingSpinner = false;
  const originalDesc = Object.getOwnPropertyDescriptor(harness.viewEl, "innerHTML");
  Object.defineProperty(harness.viewEl, "innerHTML", {
    get() {
      return originalDesc ? originalDesc.get.call(harness.viewEl) : harness.viewEl._innerHTML;
    },
    set(val) {
      if (String(val).includes("Loading dashboard...")) {
        sawBlockingSpinner = true;
      }
      if (originalDesc && originalDesc.set) {
        originalDesc.set.call(harness.viewEl, val);
      } else {
        harness.viewEl._innerHTML = String(val);
      }
    }
  });

  // Switch back to #dashboard with populated SWR cache
  harness.windowObj.location.hash = "#dashboard";
  hashListeners[0]();

  // Instant render check: blocking spinner was skipped entirely
  assert.equal(sawBlockingSpinner, false, "Dashboard tab switch must not show blocking 'Loading dashboard...' spinner when cache is warm");

  // Allow microtask to mount cached view
  await new Promise(r => setTimeout(r, 20));
  assert.ok(harness.viewEl.innerHTML.includes("Gradient Descent"), "Cached dashboard must be rendered immediately");
  assert.ok(harness.viewEl.innerHTML.includes("Backpropagation"), "Cached recommendation must be preserved");
}

async function testDashboardSummaryBatchedEndpointFastPath() {
  const requestLog = [];

  const harness = await createAppHarness(async ({ pathname, url }) => {
    requestLog.push({ pathname, url });
    if (pathname === "/api/dashboard-summary") {
      return jsonResponse({
        ok: true,
        user_id: "user-1",
        materials: [
          { material_id: "mat_batched", title: "Transformer Architecture", chunk_count: 8, concept_count: 3 }
        ],
        concepts: [
          { id: "c_attn", name: "Self-Attention Mechanism", score: 0.92, attempts: 4, status: "strong", state: "STRONG" }
        ],
        edges: [],
        notes: [
          { note_id: "note_attn", title: "Self-Attention Summary", task_type: "study_guide" }
        ],
        loop_status: {
          loop_state: "ACTIVE_LOOP",
          material_ready: true,
          baseline_completed: true,
          concepts_count: 1
        },
        recommendations: [
          {
            type: "Reinforce",
            name: "Self-Attention Mechanism",
            conceptId: "c_attn",
            reason: "High retention score; quick practice to lock in conceptual mastery."
          }
        ],
        top_recommendation: {
          type: "Reinforce",
          name: "Self-Attention Mechanism",
          conceptId: "c_attn",
          reason: "High retention score; quick practice to lock in conceptual mastery."
        }
      });
    }
    return jsonResponse({});
  });

  // Clear boot requests and SWR cache before test
  harness.context.window.ApiService.clearSwr();
  requestLog.length = 0;

  // Navigate to Dashboard
  const hashListeners = [...(harness.windowObj.__winListeners.get("hashchange") || [])];
  harness.windowObj.location.hash = "#dashboard";
  hashListeners[0]();

  await new Promise(r => setTimeout(r, 35));

  // 1. One single round-trip was executed to load the dashboard
  const summaryCalls = requestLog.filter(r => r.pathname === "/api/dashboard-summary");
  assert.equal(summaryCalls.length, 1, "Expected exactly 1 request to /api/dashboard-summary");
  assert.equal(requestLog.length, 1, "Batched fast path should only issue 1 network request for cold load");

  // 2. Verified rendered view contains batched data
  assert.ok(harness.viewEl.innerHTML.includes("Self-Attention Mechanism"), "Batched concept name should render in view");
  assert.ok(harness.viewEl.innerHTML.includes("Your Learning Progress"), "Progress donut should render");

  // 3. Recommendation container rendered immediately from batched payload
  const recContainer = harness.byId.get("dashboard-rec-container");
  assert.ok(recContainer, "#dashboard-rec-container must exist");
  assert.ok(!harness.viewEl.innerHTML.includes("skeleton-shimmer"), "Skeleton shimmer should be cleared immediately with batched recommendation");
  assert.ok(harness.viewEl.innerHTML.includes("Self-Attention Mechanism"), "Top recommendation must be rendered");

  // 4. Individual SWR caches are automatically populated for downstream views
  const ApiService = harness.context.window.ApiService;
  assert.ok(ApiService.getSwr("dashboardSummary", { userId: "user-1" }), "dashboardSummary must be cached in SWR");
  assert.ok(ApiService.getSwr("materialsList", { userId: "user-1" }), "materialsList must be seeded into SWR");
  assert.ok(ApiService.getSwr("mastery", { userId: "user-1" }), "mastery must be seeded into SWR");
  assert.ok(ApiService.getSwr("loopStatus", { userId: "user-1" }), "loopStatus must be seeded into SWR");
  assert.ok(ApiService.getSwr("notesList", { userId: "user-1" }), "notesList must be seeded into SWR");
  assert.ok(ApiService.getSwr("recommend", { userId: "user-1" }), "recommend must be seeded into SWR");
}

function testInputHighContrastAndThemeTokensAcrossProject() {
  const cssPath = path.resolve(__dirname, "../css/styles.css");
  const cssContent = fs.readFileSync(cssPath, "utf-8");
  const appJsContent = fs.readFileSync(APP_JS_PATH, "utf-8");

  // 1. Verify CSS variables exist in :root and [data-theme="dark"]
  assert.ok(cssContent.includes("--surface-input:"), "styles.css must define --surface-input");
  assert.ok(cssContent.includes("--surface-input-focus:"), "styles.css must define --surface-input-focus");
  assert.ok(cssContent.includes("--ink-input:"), "styles.css must define --ink-input");
  assert.ok(cssContent.includes("--border-input:"), "styles.css must define --border-input");

  // 2. Dark theme must have color-scheme: dark and dark-adapted surface inputs
  const darkThemeMatch = cssContent.match(/\[data-theme=["']dark["']\]\s*\{([\s\S]*?)\}/);
  assert.ok(darkThemeMatch, "styles.css must have [data-theme='dark'] block");
  const darkCss = darkThemeMatch[1];
  assert.ok(darkCss.includes("color-scheme: dark"), "[data-theme='dark'] must declare color-scheme: dark;");
  assert.ok(darkCss.includes("--surface-input: #131f37"), "Dark theme --surface-input must be deep navy (#131f37)");
  assert.ok(darkCss.includes("--ink-input: #f8fafc"), "Dark theme --ink-input must be high-contrast white (#f8fafc)");

  // 3. Global inputs, textareas, and selects must have baseline styling
  assert.ok(cssContent.includes("input, textarea, select {"), "Global input, textarea, select styling rule must exist");
  assert.ok(cssContent.includes("input::placeholder, textarea::placeholder {"), "Placeholder styling must exist");
  assert.ok(cssContent.includes(".form-control,"), ".form-control, .form-input, .form-select classes must exist");

  // 4. Baseline diagnostic classes must use CSS tokens and NOT hardcode #fff
  const fitbBlockMatch = cssContent.match(/\.fitb-blank-input\s*\{([\s\S]*?)\}/);
  assert.ok(fitbBlockMatch, ".fitb-blank-input must be defined");
  assert.ok(fitbBlockMatch[1].includes("var(--surface-input)"), ".fitb-blank-input must use var(--surface-input)");
  assert.ok(fitbBlockMatch[1].includes("var(--ink-input)"), ".fitb-blank-input must use var(--ink-input)");

  const fitbFocusMatch = cssContent.match(/\.fitb-blank-input:focus\s*\{([\s\S]*?)\}/);
  assert.ok(fitbFocusMatch, ".fitb-blank-input:focus must be defined");
  assert.ok(!fitbFocusMatch[1].includes("#fff"), ".fitb-blank-input:focus must NOT hardcode #fff");
  assert.ok(fitbFocusMatch[1].includes("var(--surface-input-focus)"), ".fitb-blank-input:focus must use var(--surface-input-focus)");

  const quizTextareaMatch = cssContent.match(/\.quiz-textarea\s*\{([\s\S]*?)\}/);
  assert.ok(quizTextareaMatch, ".quiz-textarea must be defined");
  assert.ok(quizTextareaMatch[1].includes("var(--surface-input)"), ".quiz-textarea must use var(--surface-input)");
  assert.ok(quizTextareaMatch[1].includes("var(--ink-input)"), ".quiz-textarea must use var(--ink-input)");

  const quizTextareaFocusMatch = cssContent.match(/\.quiz-textarea:focus\s*\{([\s\S]*?)\}/);
  assert.ok(quizTextareaFocusMatch, ".quiz-textarea:focus must be defined");
  assert.ok(!quizTextareaFocusMatch[1].includes("#fff"), ".quiz-textarea:focus must NOT hardcode #fff");
  assert.ok(quizTextareaFocusMatch[1].includes("var(--surface-input-focus)"), ".quiz-textarea:focus must use var(--surface-input-focus)");

  // 5. Numerical, Active Recall, and Verification inputs
  assert.ok(cssContent.includes(".numerical-input:focus"), ".numerical-input:focus must be styled");
  assert.ok(cssContent.includes(".recall-textarea:focus"), ".recall-textarea:focus must be styled");
  assert.ok(cssContent.includes(".verification-input:focus"), ".verification-input:focus must be styled");

  // 6. Ensure no lingering undefined var(--surface) in app.js
  assert.ok(!appJsContent.includes("background:var(--surface);"), "app.js must not contain raw background:var(--surface)");

  // 7. Calculate WCAG 2.1 relative luminance and contrast ratio for dark mode input
  // sRGB to linear luminance helper
  function getLuminance(r, g, b) {
    const a = [r, g, b].map(v => {
      v /= 255;
      return v <= 0.03928 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4);
    });
    return a[0] * 0.2126 + a[1] * 0.7152 + a[2] * 0.0722;
  }
  // #f8fafc: rgb(248, 250, 252)
  const lTextDark = getLuminance(248, 250, 252);
  // #131f37: rgb(19, 31, 55)
  const lBgDark = getLuminance(19, 31, 55);
  const darkContrast = (lTextDark + 0.05) / (lBgDark + 0.05);
  assert.ok(darkContrast >= 7.0, `Dark theme input contrast must exceed 7:1 (WCAG AAA); actual is ${darkContrast.toFixed(2)}:1`);

  // Light theme: #0f172a (15, 23, 42) on #ffffff (255, 255, 255)
  const lTextLight = getLuminance(15, 23, 42);
  const lBgLight = getLuminance(255, 255, 255);
  const lightContrast = (lBgLight + 0.05) / (lTextLight + 0.05);
  assert.ok(lightContrast >= 7.0, `Light theme input contrast must exceed 7:1 (WCAG AAA); actual is ${lightContrast.toFixed(2)}:1`);
}

async function testTutorMarkdownFormattingAndBlockElements() {
  const harness = await createAppHarness(async () => jsonResponse({ ok: true }));

  // Test Assistant Message Rendering with Markdown
  const sampleAssistantMsg = {
    sender: "assistant",
    text: `### 1. Core Concept Explanation\nAttention mechanisms allow models to focus dynamically.\n\n### 2. Grounded Evidence from Your Material\nAccording to [Page 4]:\n* Key projection matrix\n* Query-Key dot product scaling\n\n### 3. Intuitive Analogy / Worked Example\n1. Calculate attention scores\n2. Apply softmax normalization`,
    citations: [
      { page: 4, snippet: "Self-Attention", source_type: "document" }
    ]
  };

  const renderedHtml = harness.windowObj.renderChatMessage(sampleAssistantMsg);

  // Assert headings are formatted as block h4 elements with proper spacing
  assert.ok(renderedHtml.includes('<h4 class="tutor-heading"'), "Headings (###) must render as h4.tutor-heading block elements");
  assert.ok(renderedHtml.includes("1. Core Concept Explanation"), "Heading text must be preserved");

  // Assert bullet points render as unordered lists
  assert.ok(renderedHtml.includes('<ul class="tutor-bullet-list"'), "Bullet points (* or -) must render as ul.tutor-bullet-list");
  assert.ok(renderedHtml.includes("Key projection matrix</li>"), "Bullet items must render as li items");

  // Assert numbered lists render as ordered lists
  assert.ok(renderedHtml.includes('<ol class="tutor-numbered-list"'), "Numbered lists (1.) must render as ol.tutor-numbered-list");
  assert.ok(renderedHtml.includes("Calculate attention scores</li>"), "Numbered items must render as li items");

  // Assert paragraph breaks exist and are not collapsed into inline text
  assert.ok(renderedHtml.includes("<p"), "Paragraphs must render as block elements rather than collapsed inline text");

  // Test User Message preserves pre-wrap
  const sampleUserMsg = {
    sender: "user",
    text: "Line 1\nLine 2"
  };
  const renderedUserHtml = harness.windowObj.renderChatMessage(sampleUserMsg);
  assert.ok(renderedUserHtml.includes("white-space: pre-wrap"), "User messages must preserve white-space: pre-wrap");
}

(async function run() {
  await testApiServiceExportsAndDedupe();
  console.log("PASS: testApiServiceExportsAndDedupe");
  await testQuizDoubleClickPreventsCounterOverflow();
  console.log("PASS: testQuizDoubleClickPreventsCounterOverflow");
  await testActiveRecallNotesAudioLocksAndTabAbort();
  console.log("PASS: testActiveRecallNotesAudioLocksAndTabAbort");
  await testMockGatingAndErrorBanner();
  console.log("PASS: testMockGatingAndErrorBanner");
  await testBrowserDownloadHelperBlobAndRevokeDelay();
  console.log("PASS: testBrowserDownloadHelperBlobAndRevokeDelay");
  await testConceptFlowDiagramSvgInjectionAndSpinner();
  console.log("PASS: testConceptFlowDiagramSvgInjectionAndSpinner");
  await testAudioPlayerBlobSizeAndDelayedRevoke();
  console.log("PASS: testAudioPlayerBlobSizeAndDelayedRevoke");
  await testDashboardSwrCacheParallelSettledAndSkeletonDecoupling();
  console.log("PASS: testDashboardSwrCacheParallelSettledAndSkeletonDecoupling");
  await testDashboardSummaryBatchedEndpointFastPath();
  console.log("PASS: testDashboardSummaryBatchedEndpointFastPath");
  testInputHighContrastAndThemeTokensAcrossProject();
  console.log("PASS: testInputHighContrastAndThemeTokensAcrossProject");
  await testTutorMarkdownFormattingAndBlockElements();
  console.log("PASS: testTutorMarkdownFormattingAndBlockElements");
})();
