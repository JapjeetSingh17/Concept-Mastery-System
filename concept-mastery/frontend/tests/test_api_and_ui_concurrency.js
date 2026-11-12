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
  const windowObj = {
    location: { protocol: "http:", hostname: "localhost", port: "8000", hash: "#dashboard" },
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
    }
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
    "getMastery", "resetMastery", "getRecommendations", "listMaterials",
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

  const byId = new Map();
  const classElements = new Map();
  const docListeners = new Map();
  const winListeners = new Map();

  function makeStubContainer(id, tag = "DIV") {
    const el = makeMockElement(id, tag);
    byId.set(id, el);
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
    "btn-generate-smart-notes", "btn-note-audio", "note-audio-player-box",
    "note-audio-element", "task-selector-error", "task-selector-grid",
    "smart-notes-concept-select", "custom-instructions-input", "active-note-view"
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
      if (viewRawHtml.includes('id="quiz-card-box"')) {
        byId.set("quiz-card-box", quizBox);
      }
      for (const btnId of [
        "btn-start-recall-timer",
        "btn-ready-test-now",
        "btn-submit-active-recall",
        "btn-generate-smart-notes",
        "btn-note-audio"
      ]) {
        if (viewRawHtml.includes(`id="${btnId}"`)) {
          makeStubContainer(btnId, "BUTTON");
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

  const fetchImpl = (url, opts = {}) => {
    const method = (opts.method || "GET").toUpperCase();
    const pathname = url.replace(/^https?:\/\/[^/]+/, "").split("?")[0];
    return routeHandlers({ method, url, pathname, opts });
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
    window: windowObj,
    document: documentObj,
    location: windowObj.location,
    fetch: fetchImpl,
    setTimeout,
    clearTimeout,
    setInterval: () => 1,
    clearInterval: () => {},
    confirm: () => true,
    renderMathInElement: null
  };
  context.globalThis = context;
  vm.createContext(context);
  vm.runInContext(fs.readFileSync(API_JS_PATH, "utf8"), context, { filename: "api.js" });
  vm.runInContext(fs.readFileSync(APP_JS_PATH, "utf8"), context, { filename: "app.js" });

  return { context, documentObj, windowObj, byId, classElements, viewEl, quizBox, getQuizOptButtons: () => quizOptButtons };
}

function jsonResponse(obj, status = 200) {
  return {
    ok: status >= 200 && status < 300,
    status,
    headers: { get: () => "application/json" },
    json: async () => obj
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

(async function run() {
  await testApiServiceExportsAndDedupe();
  console.log("PASS: testApiServiceExportsAndDedupe");
  await testQuizDoubleClickPreventsCounterOverflow();
  console.log("PASS: testQuizDoubleClickPreventsCounterOverflow");
  await testActiveRecallNotesAudioLocksAndTabAbort();
  console.log("PASS: testActiveRecallNotesAudioLocksAndTabAbort");
  await testMockGatingAndErrorBanner();
  console.log("PASS: testMockGatingAndErrorBanner");
})();
