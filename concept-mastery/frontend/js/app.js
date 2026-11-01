/**
 * Concept Mastery Platform - Application Engine
 * Orchestrates views, state, KaTeX math formatting, and backend integrations.
 */

(function() {
  // Application State
  const AppState = {
    currentRoute: "dashboard",
    activeDoc: null,
    masteryList: [],
    recommendations: [],
    graphData: null,
    studentName: localStorage.getItem("cm_student_name") || "Alex",
    chatMessages: [
      {
        sender: "assistant",
        text: "Hello! I am your AI Study Tutor. Ask me any question about your study material, or select a topic to get started.",
        citations: []
      }
    ],
    quizSession: {
      active: false,
      currentQuestion: null,
      currentIndex: 1,
      totalQuestions: 6,
      score: 0,
      history: [],
      targetConceptId: null
    }
  };

  // Helper DOM Selectors
  const $ = (selector, root = document) => root.querySelector(selector);
  const $$ = (selector, root = document) => Array.from(root.querySelectorAll(selector));

  // HTML Entity Escaping
  function esc(str) {
    if (str === null || str === undefined) return "";
    return String(str).replace(/[&<>"']/g, c => ({
      "&": "&amp;",
      "<": "&lt;",
      ">": "&gt;",
      '"': "&quot;",
      "'": "&#39;"
    }[c]));
  }

  // Toast Notification
  function showToast(message, type = "info") {
    const container = $("#toast-container");
    if (!container) return;
    const toast = document.createElement("div");
    toast.className = `toast ${type}`;
    toast.innerHTML = `<span>${esc(message)}</span>`;
    container.appendChild(toast);
    setTimeout(() => {
      toast.style.opacity = "0";
      toast.style.transform = "translateY(10px)";
      setTimeout(() => toast.remove(), 250);
    }, 3200);
  }

  // KaTeX Math Rendering Utility
  function renderMath(element) {
    if (!element) return;
    if (window.renderMathInElement) {
      try {
        window.renderMathInElement(element, {
          delimiters: [
            { left: "$$", right: "$$", display: true },
            { left: "$", right: "$", display: false },
            { left: "\\(", right: "\\)", display: false },
            { left: "\\[", right: "\\]", display: true }
          ],
          throwOnError: false
        });
      } catch (err) {
        console.warn("KaTeX render error:", err);
      }
    }
  }

  // Theme Management
  function initTheme() {
    const savedTheme = localStorage.getItem("cm_theme") || "light";
    applyTheme(savedTheme);

    const toggleBtn = $("#theme-toggle-btn");
    if (toggleBtn) {
      toggleBtn.onclick = () => {
        const current = document.documentElement.getAttribute("data-theme") || "light";
        const next = current === "light" ? "dark" : "light";
        applyTheme(next);
      };
    }
  }

  function applyTheme(theme) {
    document.documentElement.setAttribute("data-theme", theme);
    localStorage.setItem("cm_theme", theme);
    const sunIcon = $(".sun-icon");
    const moonIcon = $(".moon-icon");
    if (sunIcon && moonIcon) {
      if (theme === "dark") {
        sunIcon.style.display = "block";
        moonIcon.style.display = "none";
      } else {
        sunIcon.style.display = "none";
        moonIcon.style.display = "block";
      }
    }
  }

  // Mobile Menu Navigation
  function initMobileNav() {
    const toggle = $("#mobile-toggle");
    const sidebar = $("#sidebar");
    if (toggle && sidebar) {
      toggle.onclick = () => sidebar.classList.toggle("open");
      document.addEventListener("click", e => {
        if (!sidebar.contains(e.target) && !toggle.contains(e.target) && sidebar.classList.contains("open")) {
          sidebar.classList.remove("open");
        }
      });
    }
  }

  // Global Document / Ingest Status Poller & Synchronizer
  async function syncActiveDocument() {
    try {
      const status = await ApiService.getIngestStatus();
      AppState.activeDoc = status;
      updateSidebarWidget(status);
    } catch (e) {
      console.warn("Could not sync active document status:", e);
      updateSidebarWidget({ ready: false, file: null });
    }
  }

  function updateSidebarWidget(doc) {
    const titleEl = $("#sidebar-doc-title");
    const metaEl = $("#sidebar-doc-meta");
    const pillEl = $("#sidebar-doc-pill");
    const topbarBadge = $("#topbar-doc-name");

    if (!titleEl || !metaEl || !pillEl || !topbarBadge) return;

    if (doc && doc.ready && doc.file) {
      titleEl.textContent = doc.file;
      metaEl.textContent = `${doc.pages || 1} page(s) · ${doc.concepts || 0} concepts`;
      pillEl.className = "status-pill strong";
      pillEl.textContent = "Ready";
      topbarBadge.textContent = doc.file;
    } else {
      titleEl.textContent = "No document active";
      metaEl.textContent = "Upload a study PDF";
      pillEl.className = "status-pill unknown";
      pillEl.textContent = "Offline";
      topbarBadge.textContent = "Upload PDF";
    }
  }

  // Source Citation Modal Logic
  function openSourceModal(citation) {
    const modal = $("#source-modal");
    if (!modal) return;
    const titleEl = $("#source-modal-title");
    const metaEl = $("#source-modal-meta");
    const textEl = $("#source-modal-text");

    titleEl.textContent = `Source Reference: ${citation.snippet || "Study Material"}`;
    metaEl.textContent = `Page ${citation.page || "1"} · Verified Grounded Chunk`;
    textEl.textContent = citation.text || `Citation reference on page ${citation.page}.`;

    if (typeof modal.showModal === "function") {
      modal.showModal();
    } else {
      modal.setAttribute("open", "");
    }
    renderMath(textEl);
  }

  function initModals() {
    const sourceClose = $("#source-modal-close");
    const sourceModal = $("#source-modal");
    if (sourceClose && sourceModal) {
      sourceClose.onclick = () => sourceModal.close();
      sourceModal.addEventListener("click", e => {
        if (e.target === sourceModal) sourceModal.close();
      });
    }

    const conceptClose = $("#concept-modal-close");
    const conceptModal = $("#concept-modal");
    if (conceptClose && conceptModal) {
      conceptClose.onclick = () => conceptModal.close();
      conceptModal.addEventListener("click", e => {
        if (e.target === conceptModal) conceptModal.close();
      });
    }
  }

  // SVG Chart Builders
  function createDonutSvg(percent, size = 130, strokeWidth = 10, color = "var(--primary)") {
    const radius = (size - strokeWidth) / 2;
    const circumference = 2 * Math.PI * radius;
    const clamped = Math.max(0, Math.min(100, Math.round(percent)));
    const offset = circumference - (clamped / 100) * circumference;

    return `
      <svg class="donut-chart-svg" width="${size}" height="${size}" viewBox="0 0 ${size} ${size}">
        <circle class="donut-chart-bg" cx="${size/2}" cy="${size/2}" r="${radius}" stroke-width="${strokeWidth}"/>
        <circle class="donut-chart-segment" cx="${size/2}" cy="${size/2}" r="${radius}" stroke="${color}"
          stroke-width="${strokeWidth}" stroke-dasharray="${circumference}" stroke-dashoffset="${offset}"/>
      </svg>
    `;
  }

  function createRadarSvg(concepts, size = 360) {
    if (!concepts || concepts.length < 3) {
      // Fallback if fewer than 3 concepts
      return `<div style="padding:1.5rem; text-align:center; color:var(--ink-muted);">Upload material with 3+ concepts to render radar chart.</div>`;
    }

    const items = concepts.slice(0, 6);
    const n = items.length;
    const center = size / 2;
    const radius = 78;
    const angleStep = (2 * Math.PI) / n;

    // Build polygon points for target (1.0 = 100%) and student score
    const targetPoints = [];
    const scorePoints = [];
    const axes = [];
    const labels = [];

    const shortName = name => {
      if (name.includes("Loss")) return "Loss";
      if (name.includes("Gradient")) return "Gradient";
      if (name.includes("Learning")) return "Learn Rate";
      if (name.includes("Backprop")) return "Backprop";
      if (name.includes("Overfitting")) return "Overfitting";
      if (name.includes("Regular")) return "Regularization";
      return name.split(" ")[0];
    };

    items.forEach((c, i) => {
      const angle = i * angleStep - Math.PI / 2;
      const xMax = center + radius * Math.cos(angle);
      const yMax = center + radius * Math.sin(angle);
      targetPoints.push(`${xMax},${yMax}`);
      axes.push(`<line x1="${center}" y1="${center}" x2="${xMax}" y2="${yMax}" stroke="var(--border-subtle)" stroke-width="1.5" stroke-dasharray="3 3"/>`);

      const s = (c.attempts && c.attempts > 0) ? Math.max(0.1, Math.min(1.0, c.score || 0)) : 0.05;
      const xScore = center + (radius * s) * Math.cos(angle);
      const yScore = center + (radius * s) * Math.sin(angle);
      scorePoints.push(`${xScore},${yScore}`);

      // Label coordinate with generous padding
      const xLabel = center + (radius + 24) * Math.cos(angle);
      const yLabel = center + (radius + 18) * Math.sin(angle);
      const anchor = Math.abs(xLabel - center) < 20 ? "middle" : (xLabel > center ? "start" : "end");
      labels.push(`<text x="${xLabel}" y="${yLabel}" font-size="11" fill="var(--ink-secondary)" font-weight="600" text-anchor="${anchor}">${esc(shortName(c.name))}</text>`);
    });

    // Background concentric rings
    const rings = [0.25, 0.5, 0.75, 1.0].map(r => {
      const pts = [];
      for (let i = 0; i < n; i++) {
        const a = i * angleStep - Math.PI / 2;
        pts.push(`${center + (radius * r) * Math.cos(a)},${center + (radius * r) * Math.sin(a)}`);
      }
      return `<polygon points="${pts.join(" ")}" fill="none" stroke="var(--border-subtle)" stroke-width="1"/>`;
    }).join("");

    return `
      <svg class="radar-chart-svg" viewBox="0 0 ${size} ${size}">
        ${rings}
        ${axes.join("")}
        <polygon points="${targetPoints.join(" ")}" fill="rgba(37, 99, 235, 0.05)" stroke="var(--primary-border)" stroke-width="1.5" stroke-dasharray="4 4"/>
        <polygon points="${scorePoints.join(" ")}" fill="rgba(37, 99, 235, 0.25)" stroke="var(--primary)" stroke-width="2.5"/>
        ${scorePoints.map(pt => {
          const [px, py] = pt.split(",");
          return `<circle cx="${px}" cy="${py}" r="4" fill="var(--primary)" stroke="#fff" stroke-width="2"/>`;
        }).join("")}
        ${labels.join("")}
      </svg>
    `;
  }

  // =========================================================================
  // VIEW RENDERERS
  // =========================================================================

  const Views = {
    // -----------------------------------------------------------------------
    // A. DASHBOARD VIEW
    // -----------------------------------------------------------------------
    async dashboard() {
      await syncActiveDocument();
      let masteryData = [];
      let recData = { actions: [] };

      try {
        const [mRes, rRes] = await Promise.all([
          ApiService.getMastery(),
          ApiService.getRecommendations()
        ]);
        masteryData = mRes.concepts || [];
        recData = rRes || { actions: [] };
        AppState.masteryList = masteryData;
        AppState.recommendations = recData.actions || [];
      } catch (err) {
        console.warn("Failed fetching dashboard data", err);
      }

      const totalConcepts = masteryData.length;
      const strongCount = masteryData.filter(c => getConceptStatus(c) === "strong").length;
      const weakCount = masteryData.filter(c => getConceptStatus(c) === "weak").length;
      const unknownCount = masteryData.filter(c => getConceptStatus(c) === "unknown").length;

      const avgScore = totalConcepts > 0
        ? Math.round((masteryData.reduce((acc, c) => acc + (c.attempts > 0 ? (c.score || 0) : 0), 0) / totalConcepts) * 100)
        : 0;

      const topRec = recData.actions && recData.actions.length > 0 ? recData.actions[0] : null;

      return `
        <div class="page-header">
          <div class="page-header-text">
            <h1>Welcome back, ${esc(AppState.studentName)}! 👋</h1>
            <p>Let's continue your closed-loop learning journey.</p>
          </div>
          <div class="page-header-actions">
            <a href="#practice" class="btn btn-primary">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M13 2 3 14h9l-1 8 10-12h-9l1-8z"/></svg>
              Start Quick Quiz
            </a>
          </div>
        </div>

        <!-- Hero Section Grid -->
        <div class="grid grid-dashboard-hero">
          <!-- Card 1: Visual Learning Progress -->
          <div class="card progress-hero-card">
            <h3 style="margin-bottom: 1.25rem; font-size: 1.15rem;">Your Learning Progress</h3>
            <div class="hero-content-split">
              <div class="donut-chart-container">
                ${createDonutSvg(avgScore, 130, 11, "#3b82f6")}
                <div class="donut-chart-value">
                  <span class="val-number" style="color:#ffffff;">${avgScore}%</span>
                  <span class="val-label" style="color:#cbd5e1;">Mastery</span>
                </div>
              </div>

              <div class="hero-stats-breakdown">
                <div class="hero-metric-legend">
                  <div class="metric-legend-item">
                    <div class="metric-legend-tag">
                      <span class="legend-dot strong"></span>
                      <span>Strong Concepts</span>
                    </div>
                    <span class="metric-legend-num">${strongCount} (${totalConcepts ? Math.round((strongCount/totalConcepts)*100) : 0}%)</span>
                  </div>
                  <div class="metric-legend-item">
                    <div class="metric-legend-tag">
                      <span class="legend-dot weak"></span>
                      <span>Weak Concepts</span>
                    </div>
                    <span class="metric-legend-num">${weakCount} (${totalConcepts ? Math.round((weakCount/totalConcepts)*100) : 0}%)</span>
                  </div>
                  <div class="metric-legend-item">
                    <div class="metric-legend-tag">
                      <span class="legend-dot unknown"></span>
                      <span>Unknown Concepts</span>
                    </div>
                    <span class="metric-legend-num">${unknownCount} (${totalConcepts ? Math.round((unknownCount/totalConcepts)*100) : 0}%)</span>
                  </div>
                </div>

                <div style="margin-top:0.5rem;">
                  <a href="#learning-path" class="btn btn-secondary btn-sm" style="background:rgba(255,255,255,0.15); color:#fff; border-color:rgba(255,255,255,0.25);">
                    Continue Learning &rarr;
                  </a>
                </div>
              </div>
            </div>
          </div>

          <!-- Card 2: Proactive Recommendation -->
          <div class="card recommendation-card">
            <div>
              <div class="rec-header-badge">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M15 14c.2-1 .7-1.7 1.5-2.5 1-.9 1.5-2.2 1.5-3.5A6 6 0 0 0 6 8c0 1 .2 2.2 1.5 3.5.7.7 1.3 1.5 1.5 2.5"/><path d="M9 18h6"/><path d="M10 22h4"/></svg>
                <span>Next Recommendation</span>
              </div>
              <h3 class="rec-concept-name">
                ${topRec ? esc(topRec.type) + ": " + esc(topRec.name) : "Ready to Learn"}
              </h3>
              <div class="rec-reason-box">
                ${topRec ? esc(topRec.reason) : "Upload study material to generate personalized concept recommendations."}
              </div>
            </div>

            <div style="display:flex; gap:0.65rem;">
              ${topRec ? `
                <a href="#practice" data-target-concept="${esc(topRec.conceptId)}" class="btn btn-primary start-target-quiz">
                  Start Review &rarr;
                </a>
                <a href="#tutor" class="btn btn-secondary">
                  Ask Tutor
                </a>
              ` : `
                <a href="#upload" class="btn btn-primary">
                  Upload PDF &rarr;
                </a>
              `}
            </div>
          </div>
        </div>

        <!-- Learning Journey Workflow Stepper -->
        <div class="card journey-card">
          <div class="card-header">
            <h3 class="card-title">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>
              Your Learning Journey
            </h3>
            <a href="#learning-path" class="btn btn-outline btn-sm">View Full Path &rarr;</a>
          </div>

          <div class="journey-pipeline">
            <div class="journey-step ${AppState.activeDoc && AppState.activeDoc.ready ? 'completed' : 'active'}" data-nav="upload">
              <div class="step-circle">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>
              </div>
              <span class="step-label">1. Upload<br>Material</span>
            </div>

            <div class="journey-step ${totalConcepts > 0 ? 'completed' : ''}" data-nav="learning-path">
              <div class="step-circle">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z"/><path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z"/></svg>
              </div>
              <span class="step-label">2. Learn<br>Concepts</span>
            </div>

            <div class="journey-step active" data-nav="practice">
              <div class="step-circle">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><path d="m9 12 2 2 4-4"/></svg>
              </div>
              <span class="step-label">3. Take<br>Quiz</span>
            </div>

            <div class="journey-step ${weakCount > 0 ? 'active' : ''}" data-nav="progress">
              <div class="step-circle">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 20h9"/><path d="M16.5 3.5a2.121 2.121 0 0 1 3 3L7 19l-4 1 1-4L16.5 3.5z"/></svg>
              </div>
              <span class="step-label">4. Review<br>Weak Topics</span>
            </div>

            <div class="journey-step ${strongCount >= totalConcepts && totalConcepts > 0 ? 'completed' : ''}" data-nav="progress">
              <div class="step-circle">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"/></svg>
              </div>
              <span class="step-label">5. Master<br>Concept</span>
            </div>
          </div>
        </div>

        <!-- Concept Progress Quick Glance -->
        <div class="card" style="margin-top: 1.5rem;">
          <div class="card-header">
            <h3 class="card-title">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M22 12h-4l-3 9L9 3l-3 9H2"/></svg>
              Concept Mastery Status
            </h3>
            <a href="#progress" class="btn btn-outline btn-sm">Full Analytics &rarr;</a>
          </div>

          <div class="concept-preview-list">
            ${totalConcepts === 0 ? `
              <div style="padding:2rem; text-align:center; color:var(--ink-muted);">
                No concepts indexed yet. <a href="#upload" style="font-weight:600;">Upload your study PDF</a> to build the concept graph.
              </div>
            ` : masteryData.slice(0, 5).map(c => {
              const st = getConceptStatus(c);
              const scorePercent = Math.round((c.score || 0.5) * 100);
              return `
                <div class="concept-preview-item">
                  <div class="concept-preview-left">
                    <span class="concept-preview-name">${esc(c.name)}</span>
                    <span class="concept-preview-stats">${c.attempts || 0} attempts · ${scorePercent}% belief score</span>
                  </div>
                  <div style="display:flex; align-items:center; gap:0.85rem;">
                    <div style="width:100px;" class="progress-bar-track">
                      <div class="progress-bar-fill ${st}" style="width:${scorePercent}%;"></div>
                    </div>
                    <span class="status-pill ${st}">${st}</span>
                    <a href="#practice" data-target-concept="${esc(c.id)}" class="btn btn-secondary btn-sm start-target-quiz">Quiz</a>
                  </div>
                </div>
              `;
            }).join("")}
          </div>
        </div>
      `;
    },

    // -----------------------------------------------------------------------
    // B. STUDY MATERIAL / UPLOAD VIEW
    // -----------------------------------------------------------------------
    async upload() {
      await syncActiveDocument();
      const doc = AppState.activeDoc;

      return `
        <div class="page-header">
          <div class="page-header-text">
            <h1>Study Material Library</h1>
            <p>Upload a PDF to parse semantic sections, build the vector index with page metadata, and construct the prerequisite graph.</p>
          </div>
          <div class="page-header-actions">
            <button id="btn-load-sample" class="btn btn-secondary">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>
              Try Sample: ML Basics
            </button>
          </div>
        </div>

        <div class="grid grid-cols-2">
          <!-- Upload Dropzone Card -->
          <div class="card">
            <h3 style="margin-bottom: 1.25rem;">Upload Document(s)</h3>
            <div class="upload-dropzone" id="upload-dropzone">
              <input type="file" id="pdf-file-input" accept=".pdf" multiple style="display:none;">
              <div class="upload-icon-pulse">
                <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2">
                  <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
                  <polyline points="17 8 12 3 7 8"/>
                  <line x1="12" y1="3" x2="12" y2="15"/>
                </svg>
              </div>
              <h4 style="font-size:1.1rem; margin-bottom:0.35rem;">Drag & drop your study PDF(s) here</h4>
              <p style="font-size:0.85rem; color:var(--ink-muted);">or click to select one or multiple PDFs</p>

              <div id="selected-file-display" style="display:none;" class="upload-file-details">
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>
                <span id="selected-file-name">filename.pdf</span>
              </div>
            </div>

            <div style="margin-top:1.25rem; display:flex; justify-content:flex-end;">
              <button class="btn btn-primary" id="btn-process-pdf" disabled>
                Process PDF Material(s)
              </button>
            </div>
          </div>

          <!-- Processing Stages & Status Card -->
          <div class="card">
            <h3 style="margin-bottom: 1.25rem;">Ingestion Pipeline Stages</h3>
            <div class="stages-list" id="ingest-stages">
              <div class="stage-item ${doc && doc.ready ? 'done' : ''}" id="stage-1">
                <div class="stage-status-icon">
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="20 6 9 17 4 12"/></svg>
                </div>
                <div class="stage-item-text">
                  <h4>1. Semantic Section Chunker</h4>
                  <p>Detects chapter headings and splits body text into ~110-word passages with page tags.</p>
                </div>
              </div>

              <div class="stage-item ${doc && doc.ready ? 'done' : ''}" id="stage-2">
                <div class="stage-status-icon">
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="20 6 9 17 4 12"/></svg>
                </div>
                <div class="stage-item-text">
                  <h4>2. Synonym & Term Normalizer</h4>
                  <p>Standardizes technical terminology and variants against domain synonyms.</p>
                </div>
              </div>

              <div class="stage-item ${doc && doc.ready ? 'done' : ''}" id="stage-3">
                <div class="stage-status-icon">
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="20 6 9 17 4 12"/></svg>
                </div>
                <div class="stage-item-text">
                  <h4>3. Vector Index & Metadata Store</h4>
                  <p>Computes TF-IDF vector embeddings with cosine similarity for sub-second retrieval.</p>
                </div>
              </div>

              <div class="stage-item ${doc && doc.ready ? 'done' : ''}" id="stage-4">
                <div class="stage-status-icon">
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="20 6 9 17 4 12"/></svg>
                </div>
                <div class="stage-item-text">
                  <h4>4. Prerequisite Concept Graph</h4>
                  <p>Maps causal dependencies and reading orders between discovered concepts.</p>
                </div>
              </div>
            </div>

            <!-- Active Material Summary -->
            <div style="margin-top:1.5rem; padding-top:1.25rem; border-top:1px solid var(--border-subtle);">
              <h4 style="margin-bottom:0.5rem; font-size:0.95rem;">Current Active Knowledge Base</h4>
              ${doc && doc.ready ? `
                <div style="display:flex; justify-content:space-between; align-items:center;">
                  <div>
                    <span style="font-weight:700; color:var(--ink-primary);">${esc(doc.file)}</span>
                    <p style="font-size:0.8rem; margin:0;">${doc.pages} page(s) · ${doc.concepts} concepts indexed</p>
                  </div>
                  <span class="status-pill strong">Online</span>
                </div>
              ` : `
                <p style="color:var(--ink-muted); font-size:0.85rem;">No document currently loaded. Upload a PDF or click the sample button above.</p>
              `}
            </div>
          </div>
        </div>
      `;
    },

    // -----------------------------------------------------------------------
    // C. AI TUTOR VIEW
    // -----------------------------------------------------------------------
    async tutor() {
      await syncActiveDocument();
      const doc = AppState.activeDoc;
      const docTitle = doc && doc.ready ? doc.file : "Calculus / ML Notes";

      return `
        <div class="page-header" style="margin-bottom:1.25rem;">
          <div class="page-header-text">
            <h1>AI Tutor</h1>
            <p>Ask anything about your study material. Answers are strictly grounded in your document with verifiable citations.</p>
          </div>
          <div class="page-header-actions">
            <span class="active-doc-badge">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/></svg>
              <span>Using: <strong>${esc(docTitle)}</strong></span>
            </span>
          </div>
        </div>

        <div class="tutor-container">
          <div class="tutor-header-bar">
            <div style="display:flex; align-items:center; gap:0.65rem;">
              <div class="chat-avatar bot-avatar">
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M22 10v6M2 10l10-5 10 5-10 5z"/><path d="M6 12v5c3 3 9 3 12 0v-5"/></svg>
              </div>
              <div>
                <span style="font-weight:700; font-size:0.92rem; color:var(--ink-primary);">Concept Mastery Tutor</span>
                <span style="display:block; font-size:0.75rem; color:var(--color-strong);">● Grounded RAG + TinyFish Web Search active</span>
              </div>
            </div>
            <button id="btn-clear-chat" class="btn btn-secondary btn-sm" title="Clear conversation history">
              Clear History
            </button>
          </div>

          <!-- Message Stream -->
          <div class="tutor-messages-stream" id="tutor-messages-stream">
            ${AppState.chatMessages.map(msg => renderChatMessage(msg)).join("")}
          </div>

          <!-- Suggested Action Chips -->
          <div class="tutor-suggestions" id="tutor-suggestions">
            <button class="suggestion-chip" data-prompt="Can you give me an example with math formulas?">Can you give me an example?</button>
            <button class="suggestion-chip" data-prompt="What are the essential prerequisites for this concept?">What are the prerequisites?</button>
            <button class="suggestion-chip" data-prompt="Search the web for recent industry applications of this concept.">Search web for applications</button>
            <button class="suggestion-chip" data-prompt="Explain this simply as if I were a beginner.">Explain simply</button>
          </div>

          <!-- Tutor Input Form -->
          <form class="tutor-input-form" id="tutor-input-form">
            <input type="text" id="tutor-question-input" placeholder="Ask about your study material, concepts, or search the web..." autocomplete="off" required>
            <button type="submit" class="btn btn-primary" id="btn-tutor-send">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2"><line x1="22" y1="2" x2="11" y2="13"/><polygon points="22 2 15 22 11 13 2 9 22 2"/></svg>
              Send
            </button>
          </form>
        </div>
      `;
    },

    // -----------------------------------------------------------------------
    // D. CONCEPT CHECK / QUIZ VIEW
    // -----------------------------------------------------------------------
    async practice() {
      await syncActiveDocument();
      return `
        <div class="page-header">
          <div class="page-header-text">
            <h1>Concept Check</h1>
            <p>Test your understanding of key concepts with adaptive questions generated from verified material.</p>
          </div>
        </div>

        <div class="quiz-container" id="quiz-wrapper">
          <div class="card quiz-main-card" id="quiz-card-box">
            <div style="padding:2.5rem; text-align:center;">
              <div class="upload-icon-pulse" style="width:48px; height:48px;">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>
              </div>
              <h3 style="margin-bottom:0.5rem;">Loading Adaptive Assessment...</h3>
              <p>Selecting targeted concept based on epistemic belief engine...</p>
            </div>
          </div>

          <!-- Quiz Side Panel: Live Gauge & Concept Tip -->
          <div style="display:flex; flex-direction:column; gap:1.25rem;">
            <div class="card" style="text-align:center;">
              <h4 style="margin-bottom:1rem; color:var(--ink-secondary);">Session Performance</h4>
              <div class="donut-chart-container" style="margin: 0 auto;">
                <div id="quiz-perf-donut">
                  ${createDonutSvg(0, 110, 8, "var(--primary)")}
                </div>
                <div class="donut-chart-value">
                  <span class="val-number" id="quiz-perf-number">0%</span>
                  <span class="val-label">Accuracy</span>
                </div>
              </div>
              <p style="margin-top:0.85rem; font-size:0.85rem;" id="quiz-encouragement">
                Keep going! Every answer sharpens your mastery profile.
              </p>
            </div>

            <div class="card" style="background:var(--surface-subtle); border-color:var(--border-subtle);">
              <div style="display:flex; align-items:center; gap:0.5rem; margin-bottom:0.5rem; color:var(--color-weak-text);">
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M15 14c.2-1 .7-1.7 1.5-2.5 1-.9 1.5-2.2 1.5-3.5A6 6 0 0 0 6 8c0 1 .2 2.2 1.5 3.5.7.7 1.3 1.5 1.5 2.5"/><path d="M9 18h6"/><path d="M10 22h4"/></svg>
                <span style="font-weight:700; font-size:0.85rem;">Study Tip</span>
              </div>
              <p style="font-size:0.82rem; margin:0;" id="quiz-sidebar-tip">
                Answer independently. The backend Epistemic Mastery engine tracks uncertainty using Beta beliefs.
              </p>
            </div>
          </div>
        </div>
      `;
    },

    // -----------------------------------------------------------------------
    // E. LEARNING PATH VIEW
    // -----------------------------------------------------------------------
    async learningPath() {
      await syncActiveDocument();
      let masteryData = [];
      let graphData = { nodes: [], edges: [] };

      try {
        const [mRes, gRes] = await Promise.all([
          ApiService.getMastery(),
          ApiService.getGraph()
        ]);
        masteryData = mRes.concepts || [];
        graphData = gRes || { nodes: [], edges: [] };
        AppState.masteryList = masteryData;
        AppState.graphData = graphData;
      } catch (e) {
        console.warn("Could not fetch learning path", e);
      }

      if (!masteryData.length) {
        return `
          <div class="page-header">
            <div class="page-header-text">
              <h1>Your Learning Path</h1>
              <p>A personalized roadmap based on prerequisite concepts and current mastery.</p>
            </div>
          </div>
          <div class="card" style="padding:3.5rem; text-align:center;">
            <h3>No Concept Graph Available</h3>
            <p style="margin:0.5rem 0 1.5rem 0;">Upload study material first to generate the prerequisite graph.</p>
            <a href="#upload" class="btn btn-primary">Go to Study Material &rarr;</a>
          </div>
        `;
      }

      return `
        <div class="page-header">
          <div class="page-header-text">
            <h1>Your Learning Path</h1>
            <p>A personalized roadmap based on your performance and prerequisite dependencies.</p>
          </div>
          <div class="page-header-actions">
            <button id="toggle-graph-view" class="btn btn-secondary">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="18" cy="5" r="3"/><circle cx="6" cy="12" r="3"/><circle cx="18" cy="19" r="3"/><line x1="8.59" y1="13.51" x2="15.42" y2="17.49"/><line x1="15.41" y1="6.51" x2="8.59" y2="10.49"/></svg>
              Toggle Topological Graph
            </button>
          </div>
        </div>

        <!-- Optional Full Graph Canvas (Hidden by default) -->
        <div id="full-graph-container" style="display:none; margin-bottom:1.75rem;" class="graph-canvas-wrap">
          <div class="card-header">
            <h3 class="card-title">Directed Prerequisite Dependency Graph</h3>
            <span style="font-size:0.8rem; color:var(--ink-muted);">Arrows point from prerequisite &rarr; downstream concept</span>
          </div>
          <div id="graph-svg-box">
            ${renderGraphSvg(graphData, masteryData)}
          </div>
        </div>

        <div class="grid grid-split-path">
          <!-- Left Column: Ordered Concept Roadmap Timeline -->
          <div class="card">
            <div class="card-header">
              <h3 class="card-title">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="12 2 2 7 12 12 22 7 12 2"/><polyline points="2 17 12 22 22 17"/><polyline points="2 12 12 17 22 12"/></svg>
                Concept Milestones
              </h3>
              <span class="active-doc-badge">Prerequisite Sequence</span>
            </div>

            <div class="roadmap-timeline">
              ${masteryData.map((c, i) => {
                const st = getConceptStatus(c);
                const scorePercent = (c.attempts && c.attempts > 0) ? Math.round((c.score || 0) * 100) : 0;
                let actionBtn = `<a href="#practice" data-target-concept="${esc(c.id)}" class="btn btn-primary btn-sm start-target-quiz">Practice</a>`;
                let statusLabel = "Not Assessed";

                if (st === "strong") {
                  statusLabel = "Mastered";
                  actionBtn = `<a href="#practice" data-target-concept="${esc(c.id)}" class="btn btn-secondary btn-sm start-target-quiz">Review</a>`;
                } else if (st === "weak") {
                  statusLabel = "Needs Review";
                  actionBtn = `<a href="#practice" data-target-concept="${esc(c.id)}" class="btn btn-primary btn-sm start-target-quiz" style="background:var(--color-weak);">Strengthen</a>`;
                }

                return `
                  <div class="roadmap-node-card ${st}">
                    <span class="roadmap-status-dot"></span>
                    <div>
                      <div style="display:flex; align-items:center; gap:0.5rem; margin-bottom:0.25rem;">
                        <span style="font-weight:700; font-size:1.05rem;">${esc(c.name)}</span>
                        <span class="status-pill ${st}">${statusLabel}</span>
                      </div>
                      <p style="font-size:0.82rem; margin:0; color:var(--ink-muted);">
                        ${c.attempts || 0} answer(s) · Belief: <strong>${scorePercent}%</strong>
                        ${c.why ? `· ${esc(c.why)}` : ""}
                      </p>
                    </div>

                    <div style="display:flex; align-items:center; gap:0.5rem;">
                      ${actionBtn}
                      <button class="btn btn-secondary btn-sm inspect-concept-btn" data-cid="${esc(c.id)}">Details</button>
                    </div>
                  </div>
                `;
              }).join("")}
            </div>
          </div>

          <!-- Right Column: Concept Mastery Radar / Spider Visualization -->
          <div class="card radar-chart-card">
            <h3 style="margin-bottom:0.4rem; font-size:1.15rem;">Concept Mastery Radar</h3>
            <p style="font-size:0.8rem; margin-bottom:1.5rem; color:var(--ink-muted);">Comparing current belief score vs target 100% mastery</p>

            ${createRadarSvg(masteryData)}

            <div style="display:flex; gap:1.5rem; margin-top:1.5rem; font-size:0.82rem;">
              <div style="display:flex; align-items:center; gap:0.45rem;">
                <span style="width:12px; height:12px; background:var(--primary); border-radius:2px;"></span>
                <span style="font-weight:600;">You (Estimated)</span>
              </div>
              <div style="display:flex; align-items:center; gap:0.45rem;">
                <span style="width:12px; height:12px; border:1px dashed var(--primary-border); background:rgba(37,99,235,0.1); border-radius:2px;"></span>
                <span style="font-weight:600; color:var(--ink-muted);">Target Mastery</span>
              </div>
            </div>

            <div style="margin-top:1.75rem; width:100%; text-align:left; border-top:1px solid var(--border-subtle); padding-top:1.25rem;">
              <h4 style="font-size:0.9rem; margin-bottom:0.5rem;">Why this order matters</h4>
              <p style="font-size:0.82rem; line-height:1.5; color:var(--ink-secondary);">
                Concepts are sorted by prerequisite depth. If a foundational concept is weak, practicing downstream concepts first often fails because the prerequisite knowledge isn't solid yet.
              </p>
            </div>
          </div>
        </div>
      `;
    },

    // -----------------------------------------------------------------------
    // F. PROGRESS & MASTERY ANALYTICS VIEW
    // -----------------------------------------------------------------------
    async progress() {
      await syncActiveDocument();
      let masteryData = [];
      try {
        const res = await ApiService.getMastery();
        masteryData = res.concepts || [];
        AppState.masteryList = masteryData;
      } catch (e) {
        console.warn("Could not fetch mastery snapshot", e);
      }

      const total = masteryData.length;
      const strongCount = masteryData.filter(c => getConceptStatus(c) === "strong").length;
      const weakCount = masteryData.filter(c => getConceptStatus(c) === "weak").length;
      const unknownCount = masteryData.filter(c => getConceptStatus(c) === "unknown").length;
      const avg = total > 0 ? Math.round((masteryData.reduce((acc, c) => acc + ((c.attempts && c.attempts > 0) ? (c.score || 0) : 0), 0) / total) * 100) : 0;

      return `
        <div class="page-header">
          <div class="page-header-text">
            <h1>Progress &amp; Mastery Analytics</h1>
            <p>Epistemic belief tracking using Beta(a,b) distributions across all verified concepts.</p>
          </div>
          <div class="page-header-actions">
            <a href="#practice" class="btn btn-primary">Take Practice Quiz</a>
          </div>
        </div>

        <!-- 4 KPI Cards -->
        <div class="grid grid-cols-4" style="margin-bottom: 1.75rem;">
          <div class="card" style="text-align:center;">
            <span style="font-size:0.75rem; text-transform:uppercase; color:var(--ink-muted); font-weight:700;">Overall Mastery</span>
            <h2 style="font-size:2rem; margin:0.35rem 0; color:var(--primary);">${avg}%</h2>
            <span style="font-size:0.75rem; color:var(--ink-secondary);">Average belief mean</span>
          </div>

          <div class="card" style="text-align:center;">
            <span style="font-size:0.75rem; text-transform:uppercase; color:var(--color-strong); font-weight:700;">Strong Concepts</span>
            <h2 style="font-size:2rem; margin:0.35rem 0; color:var(--color-strong);">${strongCount}</h2>
            <span style="font-size:0.75rem; color:var(--ink-secondary);">&ge; 70% estimated mastery</span>
          </div>

          <div class="card" style="text-align:center;">
            <span style="font-size:0.75rem; text-transform:uppercase; color:var(--color-weak); font-weight:700;">Weak Concepts</span>
            <h2 style="font-size:2rem; margin:0.35rem 0; color:var(--color-weak);">${weakCount}</h2>
            <span style="font-size:0.75rem; color:var(--ink-secondary);">&ge; 2 answers &amp; &lt; 70%</span>
          </div>

          <div class="card" style="text-align:center;">
            <span style="font-size:0.75rem; text-transform:uppercase; color:var(--color-unknown); font-weight:700;">Unknown Concepts</span>
            <h2 style="font-size:2rem; margin:0.35rem 0; color:var(--color-unknown);">${unknownCount}</h2>
            <span style="font-size:0.75rem; color:var(--ink-secondary);">&lt; 2 answers recorded</span>
          </div>
        </div>

        <!-- Epistemic Explanation Banner -->
        <div class="card" style="margin-bottom:1.75rem; background:var(--surface-subtle); border-color:var(--border-subtle);">
          <div style="display:flex; align-items:flex-start; gap:0.85rem;">
            <div style="width:32px; height:32px; border-radius:50%; background:var(--primary-light); color:var(--primary); display:flex; align-items:center; justify-content:center; flex-shrink:0;">
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/></svg>
            </div>
            <div>
              <h4 style="font-size:0.95rem; margin-bottom:0.25rem;">How the Epistemic Engine Works</h4>
              <p style="font-size:0.85rem; margin:0; line-height:1.5;">
                Unlike crude score-percentages, each concept maintains a Bayesian $\\text{Beta}(\\alpha, \\beta)$ belief state. <strong>Unknown</strong> means fewer than 2 answers exist—we do not penalize unseen concepts as weak. As you answer questions, the uncertainty spread shrinks and estimates stabilize.
              </p>
            </div>
          </div>
        </div>

        <!-- Detailed Concept Mastery Table -->
        <div class="mastery-table-wrap">
          <table class="mastery-table">
            <thead>
              <tr>
                <th>Concept</th>
                <th>Status</th>
                <th>Mastery Score</th>
                <th>Total Attempts</th>
                <th>Uncertainty Spread</th>
                <th>Diagnostic Explanation</th>
                <th>Action</th>
              </tr>
            </thead>
            <tbody>
              ${masteryData.length === 0 ? `
                <tr><td colspan="7" style="text-align:center; padding:2rem; color:var(--ink-muted);">No concepts assessed yet. Upload material and take a quiz.</td></tr>
              ` : masteryData.map(c => {
                const st = getConceptStatus(c);
                const scorePercent = (c.attempts && c.attempts > 0) ? Math.round((c.score || 0) * 100) : 0;
                const uncertainty = c.uncertainty !== undefined ? `&plusmn;${Math.round(c.uncertainty * 100)}%` : "&plusmn;25%";
                return `
                  <tr>
                    <td><strong>${esc(c.name)}</strong></td>
                    <td><span class="status-pill ${st}">${st}</span></td>
                    <td>
                      <div style="display:flex; align-items:center; gap:0.65rem;">
                        <div style="width:70px;" class="progress-bar-track">
                          <div class="progress-bar-fill ${st}" style="width:${scorePercent}%;"></div>
                        </div>
                        <span style="font-weight:700; font-size:0.85rem;">${scorePercent}%</span>
                      </div>
                    </td>
                    <td>${c.attempts || 0}</td>
                    <td><span style="color:var(--ink-muted); font-size:0.85rem;">${uncertainty}</span></td>
                    <td style="max-width:340px; font-size:0.82rem; color:var(--ink-secondary);">${esc(c.why || "Tracking belief")}</td>
                    <td>
                      <div style="display:flex; gap:0.4rem;">
                        <a href="#tutor" class="btn btn-secondary btn-sm" title="Ask Tutor">Ask</a>
                        <a href="#practice" data-target-concept="${esc(c.id)}" class="btn btn-primary btn-sm start-target-quiz" title="Quiz this concept">Quiz</a>
                      </div>
                    </td>
                  </tr>
                `;
              }).join("")}
            </tbody>
          </table>
        </div>
      `;
    },

    // -----------------------------------------------------------------------
    // G. SETTINGS VIEW
    // -----------------------------------------------------------------------
    async settings() {
      const conf = ApiService.getConfig();
      let healthOk = false;
      try {
        const h = await ApiService.getHealth();
        healthOk = Boolean(h && h.ok);
      } catch (e) {
        healthOk = false;
      }

      return `
        <div class="page-header">
          <div class="page-header-text">
            <h1>Platform Settings</h1>
            <p>Manage local preferences, student profile, and verified backend connections.</p>
          </div>
        </div>

        <div class="grid grid-cols-2">
          <!-- Profile & Appearance -->
          <div class="card">
            <h3 style="margin-bottom:1.25rem;">Profile &amp; Appearance</h3>
            <div style="display:flex; flex-direction:column; gap:1.25rem;">
              <div>
                <label style="display:block; font-size:0.85rem; font-weight:600; margin-bottom:0.4rem;">Student Name</label>
                <input type="text" id="setting-student-name" value="${esc(AppState.studentName)}" style="padding:0.65rem 0.85rem; border:1px solid var(--border-subtle); border-radius:var(--radius-md); width:100%; background:var(--surface-subtle); color:var(--ink-primary);">
              </div>

              <div>
                <label style="display:block; font-size:0.85rem; font-weight:600; margin-bottom:0.4rem;">Color Theme</label>
                <div style="display:flex; gap:0.75rem;">
                  <button class="btn btn-secondary btn-sm theme-select-btn" data-set-theme="light">Light Theme</button>
                  <button class="btn btn-secondary btn-sm theme-select-btn" data-set-theme="dark">Dark Theme</button>
                </div>
              </div>

              <div style="padding-top:0.75rem; border-top:1px solid var(--border-subtle);">
                <button class="btn btn-primary" id="btn-save-settings">Save Preferences</button>
              </div>
            </div>
          </div>

          <!-- Backend Diagnostics -->
          <div class="card">
            <h3 style="margin-bottom:1.25rem;">Backend Architecture</h3>
            <div style="display:flex; flex-direction:column; gap:1rem;">
              <div style="display:flex; justify-content:space-between; align-items:center; padding:0.75rem; background:var(--surface-subtle); border-radius:var(--radius-md);">
                <span style="font-weight:600; font-size:0.88rem;">Server Status</span>
                <span class="status-pill ${healthOk ? 'strong' : 'weak'}">${healthOk ? 'Connected (FastAPI)' : 'Unreachable'}</span>
              </div>

              <div>
                <label style="display:block; font-size:0.85rem; font-weight:600; margin-bottom:0.4rem;">Backend Base URL</label>
                <input type="text" id="setting-backend-base" value="${esc(conf.base)}" placeholder="e.g. http://localhost:8000 (empty for same host)" style="padding:0.65rem 0.85rem; border:1px solid var(--border-subtle); border-radius:var(--radius-md); width:100%; background:var(--surface-subtle); color:var(--ink-primary);">
              </div>

              <div style="display:flex; align-items:center; gap:0.6rem;">
                <input type="checkbox" id="setting-mock-mode" ${conf.mock ? 'checked' : ''} style="width:18px; height:18px;">
                <label for="setting-mock-mode" style="font-size:0.88rem; font-weight:600;">Enable offline demo dataset (Mock mode)</label>
              </div>

              <div style="padding-top:0.75rem; border-top:1px solid var(--border-subtle); display:flex; justify-content:space-between; align-items:center;">
                <button class="btn btn-secondary" id="btn-ping-backend">Ping Health Check</button>
                <button class="btn btn-outline" id="btn-reset-mastery" style="color:var(--color-danger); border-color:var(--color-danger-border);">Reset Mastery</button>
              </div>
              <div id="settings-diag-output" style="font-size:0.8rem; color:var(--ink-muted);"></div>
            </div>
          </div>
        </div>
      `;
    }
  };

  // Helper to render chat message
  function renderChatMessage(msg) {
    const isUser = msg.sender === "user";
    const citationsHtml = msg.citations && msg.citations.length > 0 ? `
      <div class="citation-block">
        ${msg.citations.map((c, i) => {
          if (c.source_type === "web" || c.url) {
            return `
              <a href="${esc(c.url || '#')}" target="_blank" rel="noopener noreferrer" class="citation-chip web-citation-chip" style="text-decoration:none; display:inline-flex; align-items:center; gap:0.35rem; color:var(--primary);">
                <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="2" y1="12" x2="22" y2="12"/><path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"/></svg>
                <span>${esc(c.snippet || c.title || "Web Source")} &nearr;</span>
              </a>
            `;
          }
          return `
            <button class="citation-chip view-citation-btn" data-page="${c.page || 1}" data-snippet="${esc(c.snippet)}" data-text="${esc(c.text || '')}">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>
              <span>Page ${c.page || 1}: <strong>${esc(c.snippet || "Section")}</strong> &rarr;</span>
            </button>
          `;
        }).join("")}
      </div>
    ` : "";

    return `
      <div class="chat-row ${isUser ? 'user' : 'assistant'}">
        <div class="chat-avatar ${isUser ? 'user-avatar' : 'bot-avatar'}">
          ${isUser ? esc(AppState.studentName[0] || 'A') : `
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M22 10v6M2 10l10-5 10 5-10 5z"/><path d="M6 12v5c3 3 9 3 12 0v-5"/></svg>
          `}
        </div>
        <div class="message-bubble">
          <div>${msg.text}</div>
          ${citationsHtml}
        </div>
      </div>
    `;
  }

  // Helper to render graph SVG
  function renderGraphSvg(graph, mastery) {
    if (!graph.nodes || !graph.nodes.length) {
      return `<p style="color:var(--ink-muted); text-align:center; padding:2rem;">No nodes in prerequisite graph.</p>`;
    }

    const byId = Object.fromEntries(graph.nodes.map(n => [n.id, n]));
    const statusMap = Object.fromEntries((mastery || []).map(m => [m.id, getConceptStatus(m)]));
    const colors = {
      strong: "var(--color-strong)",
      weak: "var(--color-weak)",
      unknown: "var(--color-unknown)"
    };

    const lines = (graph.edges || []).map(([from, to]) => {
      const a = byId[from];
      const b = byId[to];
      if (!a || !b) return "";
      return `<line x1="${a.x}" y1="${a.y}" x2="${b.x}" y2="${b.y}" stroke="var(--ink-muted)" stroke-width="2" marker-end="url(#arrowhead)" opacity="0.5"/>`;
    }).join("");

    const nodes = graph.nodes.map(n => {
      const st = statusMap[n.id] || "unknown";
      const col = colors[st];
      return `
        <g class="graph-node-group" style="cursor:pointer;" data-cid="${esc(n.id)}">
          <circle cx="${n.x}" cy="${n.y}" r="24" fill="${col}" stroke="#ffffff" stroke-width="3" filter="drop-shadow(0 2px 4px rgba(0,0,0,0.15))"/>
          <text x="${n.x}" y="${n.y + 42}" text-anchor="middle" font-size="12" font-weight="600" fill="var(--ink-primary)">${esc(n.name)}</text>
        </g>
      `;
    }).join("");

    return `
      <svg viewBox="0 0 740 280" style="width:100%; min-width:600px; height:auto;">
        <defs>
          <marker id="arrowhead" viewBox="0 0 10 10" refX="28" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
            <path d="M 0 0 L 10 5 L 0 10 z" fill="var(--ink-muted)" opacity="0.7"/>
          </marker>
        </defs>
        ${lines}
        ${nodes}
      </svg>
    `;
  }

  // =========================================================================
  // VIEW INTERACTION HANDLERS (After Render)
  // =========================================================================

  const AfterRender = {
    dashboard() {
      $$(".start-target-quiz").forEach(btn => {
        btn.onclick = () => {
          AppState.quizSession.targetConceptId = btn.getAttribute("data-target-concept");
        };
      });

      $$(".journey-step").forEach(step => {
        step.onclick = () => {
          const navTarget = step.getAttribute("data-nav");
          if (navTarget) location.hash = `#${navTarget}`;
        };
      });
    },

    upload() {
      const dropzone = $("#upload-dropzone");
      const fileInput = $("#pdf-file-input");
      const processBtn = $("#btn-process-pdf");
      const sampleBtn = $("#btn-load-sample");
      const fileDisplay = $("#selected-file-display");
      const fileNameEl = $("#selected-file-name");
      let selectedFiles = [];

      if (!dropzone || !fileInput || !processBtn) return;

      dropzone.onclick = () => fileInput.click();

      function updateFileSelection(fileList) {
        const pdfs = Array.from(fileList || []).filter(f => f.name.toLowerCase().endsWith(".pdf"));
        if (!pdfs.length) {
          showToast("Please select at least one valid PDF file.", "warning");
          return;
        }
        selectedFiles = pdfs;
        if (pdfs.length === 1) {
          fileNameEl.textContent = pdfs[0].name;
        } else {
          fileNameEl.textContent = `${pdfs.length} PDFs selected (${pdfs.map(f => f.name).join(", ")})`;
        }
        fileDisplay.style.display = "inline-flex";
        processBtn.disabled = false;
        processBtn.textContent = `Process ${pdfs.length > 1 ? pdfs.length + " PDFs" : "PDF Material"}`;
      }

      fileInput.onchange = () => {
        if (fileInput.files && fileInput.files.length) {
          updateFileSelection(fileInput.files);
        }
      };

      ["dragover", "dragenter"].forEach(ev => {
        dropzone.addEventListener(ev, e => {
          e.preventDefault();
          dropzone.classList.add("over");
        });
      });

      ["dragleave", "drop"].forEach(ev => {
        dropzone.addEventListener(ev, e => {
          e.preventDefault();
          dropzone.classList.remove("over");
          if (ev === "drop" && e.dataTransfer.files && e.dataTransfer.files.length) {
            updateFileSelection(e.dataTransfer.files);
          }
        });
      });

      // Process Ingestion
      processBtn.onclick = async () => {
        if (!selectedFiles.length) return;
        processBtn.disabled = true;
        processBtn.textContent = `Processing ${selectedFiles.length > 1 ? selectedFiles.length + " PDFs" : "PDF"}...`;

        const stages = [$("#stage-1"), $("#stage-2"), $("#stage-3"), $("#stage-4")];
        stages.forEach(s => s && (s.className = "stage-item running"));

        try {
          const result = await ApiService.uploadPdf(selectedFiles);
          stages.forEach(s => s && (s.className = "stage-item done"));
          const discoveredCount = result.concepts ? (Array.isArray(result.concepts) ? result.concepts.length : result.concepts) : 6;
          const filesCount = result.files_count || selectedFiles.length;
          showToast(`Ingestion complete! ${discoveredCount} concepts discovered across ${filesCount} document(s).`, "success");
          await syncActiveDocument();
          setTimeout(() => {
            location.hash = "#dashboard";
          }, 1200);
        } catch (err) {
          stages.forEach(s => s && (s.className = "stage-item"));
          showToast(`Ingestion error: ${err.message}`, "danger");
          processBtn.disabled = false;
          processBtn.textContent = "Retry Processing";
        }
      };

      // Sample 1-Click Loader
      if (sampleBtn) {
        sampleBtn.onclick = async () => {
          sampleBtn.disabled = true;
          sampleBtn.textContent = "Loading sample...";
          const stages = [$("#stage-1"), $("#stage-2"), $("#stage-3"), $("#stage-4")];
          stages.forEach(s => s && (s.className = "stage-item running"));

          try {
            const res = await ApiService.loadSamplePdf();
            stages.forEach(s => s && (s.className = "stage-item done"));
            showToast("Sample 'Machine Learning Basics' loaded successfully!", "success");
            await syncActiveDocument();
            setTimeout(() => {
              location.hash = "#dashboard";
            }, 1000);
          } catch (e) {
            stages.forEach(s => s && (s.className = "stage-item"));
            showToast(`Could not load sample: ${e.message}`, "danger");
            sampleBtn.disabled = false;
            sampleBtn.textContent = "Try Sample: ML Basics";
          }
        };
      }
    },

    tutor() {
      const stream = $("#tutor-messages-stream");
      const form = $("#tutor-input-form");
      const input = $("#tutor-question-input");
      const clearBtn = $("#btn-clear-chat");

      // Math formatting on load
      renderMath(stream);

      // Delegate citation click
      if (stream) {
        stream.onclick = e => {
          const chip = e.target.closest(".view-citation-btn");
          if (chip) {
            openSourceModal({
              page: chip.getAttribute("data-page"),
              snippet: chip.getAttribute("data-snippet"),
              text: chip.getAttribute("data-text")
            });
          }
        };
      }

      // Suggestion chips click
      $$(".suggestion-chip").forEach(chip => {
        chip.onclick = () => {
          const prompt = chip.getAttribute("data-prompt");
          if (input && prompt) {
            input.value = prompt;
            form.dispatchEvent(new Event("submit"));
          }
        };
      });

      // Clear chat
      if (clearBtn) {
        clearBtn.onclick = () => {
          AppState.chatMessages = [
            {
              sender: "assistant",
              text: "Conversation cleared. Ask me anything about your study notes!",
              citations: []
            }
          ];
          stream.innerHTML = AppState.chatMessages.map(msg => renderChatMessage(msg)).join("");
        };
      }

      // Submit question
      if (form) {
        form.onsubmit = async e => {
          e.preventDefault();
          const question = input.value.trim();
          if (!question) return;
          input.value = "";

          // Append student message
          const userMsg = { sender: "user", text: esc(question) };
          AppState.chatMessages.push(userMsg);
          stream.insertAdjacentHTML("beforeend", renderChatMessage(userMsg));
          renderMath(stream);
          stream.scrollTop = stream.scrollHeight;

          // Placeholder thinking bubble
          const tempId = `thinking-${Date.now()}`;
          stream.insertAdjacentHTML("beforeend", `
            <div class="chat-row assistant" id="${tempId}">
              <div class="chat-avatar bot-avatar">
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M22 10v6M2 10l10-5 10 5-10 5z"/></svg>
              </div>
              <div class="message-bubble" style="color:var(--ink-muted);">
                Analyzing material &amp; verifying citations...
              </div>
            </div>
          `);
          stream.scrollTop = stream.scrollHeight;

          try {
            const resp = await ApiService.askTutor(question);
            const thinkingEl = $(`#${tempId}`);
            if (thinkingEl) thinkingEl.remove();

            const tutorMsg = {
              sender: "assistant",
              text: esc(resp.answer),
              citations: resp.citations || []
            };
            AppState.chatMessages.push(tutorMsg);
            stream.insertAdjacentHTML("beforeend", renderChatMessage(tutorMsg));
            renderMath(stream);
            stream.scrollTop = stream.scrollHeight;
          } catch (err) {
            const thinkingEl = $(`#${tempId}`);
            if (thinkingEl) {
              thinkingEl.querySelector(".message-bubble").textContent = `Error: ${err.message}`;
            }
          }
        };
      }
    },

    practice() {
      loadQuizQuestion(AppState.quizSession.targetConceptId);
      AppState.quizSession.targetConceptId = null;
    },

    learningPath() {
      const toggleBtn = $("#toggle-graph-view");
      const graphContainer = $("#full-graph-container");
      if (toggleBtn && graphContainer) {
        toggleBtn.onclick = () => {
          const isHidden = graphContainer.style.display === "none";
          graphContainer.style.display = isHidden ? "block" : "none";
          toggleBtn.textContent = isHidden ? "Hide Topological Graph" : "Toggle Topological Graph";
        };
      }

      $$(".start-target-quiz").forEach(btn => {
        btn.onclick = () => {
          AppState.quizSession.targetConceptId = btn.getAttribute("data-target-concept");
        };
      });

      // Concept Details Modal
      $$(".inspect-concept-btn").forEach(btn => {
        btn.onclick = () => {
          const cid = btn.getAttribute("data-cid");
          const c = (AppState.masteryList || []).find(x => x.id === cid);
          if (!c) return;

          const modal = $("#concept-modal");
          const title = $("#concept-modal-title");
          const body = $("#concept-modal-body");
          if (!modal || !title || !body) return;

          title.textContent = c.name;
          const st = getConceptStatus(c);
          body.innerHTML = `
            <div style="margin-bottom:1rem;">
              <span class="status-pill ${st}">${st}</span>
              <span style="font-size:0.85rem; color:var(--ink-muted); margin-left:0.5rem;">Estimated Belief: ${Math.round((c.score||0.5)*100)}%</span>
            </div>
            <p style="margin-bottom:1rem; font-size:0.92rem; color:var(--ink-primary);">${esc(c.why || "No diagnostic notes available.")}</p>
            <div style="display:flex; gap:0.65rem; margin-top:1.5rem;">
              <a href="#tutor" class="btn btn-secondary btn-sm" onclick="document.getElementById('concept-modal').close()">Ask Tutor</a>
              <a href="#practice" data-target-concept="${esc(c.id)}" class="btn btn-primary btn-sm start-target-quiz" onclick="document.getElementById('concept-modal').close()">Practice Now</a>
            </div>
          `;

          if (typeof modal.showModal === "function") modal.showModal();
          else modal.setAttribute("open", "");
        };
      });
    },

    progress() {
      renderMath(document);
      $$(".start-target-quiz").forEach(btn => {
        btn.onclick = () => {
          AppState.quizSession.targetConceptId = btn.getAttribute("data-target-concept");
        };
      });
    },

    settings() {
      const nameInput = $("#setting-student-name");
      const baseInput = $("#setting-backend-base");
      const mockCheck = $("#setting-mock-mode");
      const saveBtn = $("#btn-save-settings");
      const pingBtn = $("#btn-ping-backend");
      const resetBtn = $("#btn-reset-mastery");
      const diagOut = $("#settings-diag-output");

      // Theme selector buttons
      $$(".theme-select-btn").forEach(btn => {
        btn.onclick = () => {
          const t = btn.getAttribute("data-set-theme");
          applyTheme(t);
        };
      });

      if (saveBtn) {
        saveBtn.onclick = () => {
          if (nameInput) {
            const newName = nameInput.value.trim() || "Alex";
            AppState.studentName = newName;
            localStorage.setItem("cm_student_name", newName);
            const topName = $("#topbar-user-name");
            const topAvatar = $("#topbar-user-avatar");
            if (topName) topName.textContent = newName;
            if (topAvatar) topAvatar.textContent = newName[0] || "A";
          }
          if (baseInput && mockCheck) {
            ApiService.saveConfig({
              base: baseInput.value.trim().replace(/\/$/, ""),
              mock: mockCheck.checked
            });
          }
          showToast("Settings successfully saved!", "success");
        };
      }

      if (pingBtn && diagOut) {
        pingBtn.onclick = async () => {
          diagOut.textContent = "Pinging health endpoint...";
          try {
            const start = performance.now();
            const res = await ApiService.getHealth();
            const latency = Math.round(performance.now() - start);
            diagOut.textContent = `Server responded in ${latency}ms: ${JSON.stringify(res)}`;
          } catch (e) {
            diagOut.textContent = `Ping failed: ${e.message}`;
          }
        };
      }

      if (resetBtn) {
        resetBtn.onclick = async () => {
          if (!confirm("Are you sure you want to reset mastery belief history? This cannot be undone.")) return;
          try {
            await ApiService.resetMastery();
            showToast("Mastery history reset.", "info");
            setTimeout(() => { location.hash = "#dashboard"; }, 600);
          } catch (e) {
            showToast(`Reset failed: ${e.message}`, "danger");
          }
        };
      }
    }
  };

  // =========================================================================
  // QUIZ ENGINE CONTROLLER (Authoritative Server Evaluation)
  // =========================================================================

  async function loadQuizQuestion(conceptId = null) {
    const box = $("#quiz-card-box");
    if (!box) return;

    box.innerHTML = `
      <div style="padding:2.5rem; text-align:center;">
        <div class="upload-icon-pulse" style="width:40px; height:40px;">
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>
        </div>
        <p style="margin-top:0.75rem;">Generating adaptive question...</p>
      </div>
    `;

    try {
      const q = await ApiService.getNextQuestion(conceptId);
      AppState.quizSession.currentQuestion = q;
      renderActiveQuestion(q);
    } catch (err) {
      box.innerHTML = `
        <div style="padding:2rem; text-align:center;">
          <h3 style="margin-bottom:0.5rem;">Could not load question</h3>
          <p style="color:var(--ink-secondary); margin-bottom:1.5rem;">${esc(err.message)}</p>
          <a href="#upload" class="btn btn-primary">Go to Study Material &rarr;</a>
        </div>
      `;
    }
  }

  function renderActiveQuestion(q) {
    const box = $("#quiz-card-box");
    if (!box) return;

    const letters = ["A", "B", "C", "D"];
    const progressPercent = Math.round((AppState.quizSession.currentIndex / AppState.quizSession.totalQuestions) * 100);
    const qType = (q.questionType || q.question_type || (q.options && q.options.length ? "mcq" : "typed_qa")).toLowerCase();

    let typeBadgeLabel = "Multiple Choice";
    let typeBadgeClass = "mcq";
    if (qType === "fill_in_the_blanks") {
      typeBadgeLabel = "Fill in the Blank";
      typeBadgeClass = "fitb";
    } else if (qType === "typed_qa") {
      typeBadgeLabel = "Open-Ended Q&A";
      typeBadgeClass = "typed_qa";
    }

    // Dynamic input area based on question type
    let inputHtml = "";
    if (qType === "fill_in_the_blanks") {
      const sentenceText = q.question || "";
      const hasBlank = sentenceText.includes("_____") || sentenceText.includes("___");
      inputHtml = `
        <div class="fitb-input-container">
          <div class="fitb-sentence-box">
            ${hasBlank 
              ? esc(sentenceText).replace(/_{3,}/g, `<input type="text" id="fitb-user-input" class="fitb-blank-input" placeholder="type term..." autocomplete="off" />`)
              : `<p style="margin-bottom:0.75rem;">${esc(sentenceText)}</p><input type="text" id="fitb-user-input" class="fitb-blank-input" placeholder="Type missing keyword..." autocomplete="off" style="width:100%; max-width:320px;" />`
            }
          </div>
          <div class="qa-helper-bar">
            <span>Type the exact term or concept keyword that completes the statement.</span>
          </div>
        </div>
      `;
    } else if (qType === "typed_qa") {
      inputHtml = `
        <div class="typed-qa-container">
          <textarea id="typed-qa-user-input" class="quiz-textarea" rows="4" placeholder="Explain the principle, mechanism, or relationship in your own words..."></textarea>
          <div class="qa-helper-bar">
            <span>AI Mastery Judge evaluates your conceptual explanation and depth.</span>
            <span id="qa-char-count">0 characters</span>
          </div>
        </div>
      `;
    } else {
      // Standard MCQ
      const options = q.options || [];
      inputHtml = `
        <div class="quiz-options-group" id="quiz-options-group">
          ${options.map((opt, i) => `
            <button class="quiz-opt-btn" data-index="${i}">
              <span class="opt-letter-badge">${letters[i] || (i + 1)}</span>
              <span class="opt-text">${esc(opt)}</span>
            </button>
          `).join("")}
        </div>
      `;
    }

    box.innerHTML = `
      <div class="quiz-progress-header">
        <div style="display:flex; align-items:center; gap:0.6rem;">
          <span class="status-pill unknown">${esc(q.conceptName || q.conceptId)}</span>
          <span class="badge-question-type ${typeBadgeClass}">${typeBadgeLabel}</span>
        </div>
        <div style="display:flex; align-items:center; gap:0.75rem;">
          <span style="font-size:0.82rem; font-weight:600; color:var(--ink-muted);">
            Question ${AppState.quizSession.currentIndex} of ${AppState.quizSession.totalQuestions}
          </span>
          <div style="width:110px;" class="progress-bar-track">
            <div class="progress-bar-fill" style="width:${progressPercent}%;"></div>
          </div>
        </div>
      </div>

      <h3 class="question-prompt" id="question-prompt-text">${esc(q.question)}</h3>

      ${inputHtml}

      <div id="quiz-feedback-container"></div>

      <div class="quiz-actions-bar">
        <button class="btn btn-secondary btn-sm" id="btn-quiz-skip">Skip Question</button>
        <button class="btn btn-primary" id="btn-quiz-submit" disabled>Submit Answer</button>
      </div>
    `;

    renderMath(box);

    let selectedIdx = null;
    const submitBtn = $("#btn-quiz-submit");
    const skipBtn = $("#btn-quiz-skip");
    const fitbInput = $("#fitb-user-input");
    const typedQaInput = $("#typed-qa-user-input");
    const optionBtns = box.querySelectorAll(".quiz-opt-btn");

    // Input listeners per type
    if (qType === "fill_in_the_blanks" && fitbInput) {
      fitbInput.focus();
      fitbInput.addEventListener("input", () => {
        submitBtn.disabled = !fitbInput.value.trim();
      });
      fitbInput.addEventListener("keydown", e => {
        if (e.key === "Enter" && !submitBtn.disabled) {
          submitBtn.click();
        }
      });
    } else if (qType === "typed_qa" && typedQaInput) {
      typedQaInput.focus();
      const countEl = $("#qa-char-count");
      typedQaInput.addEventListener("input", () => {
        const len = typedQaInput.value.trim().length;
        if (countEl) countEl.textContent = `${len} character${len === 1 ? '' : 's'}`;
        submitBtn.disabled = (len < 4);
      });
    } else {
      // MCQ
      optionBtns.forEach(btn => {
        btn.onclick = () => {
          optionBtns.forEach(b => b.classList.remove("selected"));
          btn.classList.add("selected");
          selectedIdx = parseInt(btn.getAttribute("data-index"), 10);
          submitBtn.disabled = false;
        };
      });

      const handleKeyNav = e => {
        if (AppState.currentRoute !== "practice") return;
        if (document.activeElement && ["INPUT", "TEXTAREA"].includes(document.activeElement.tagName)) return;
        const k = e.key.toUpperCase();
        const keyMap = { "1": 0, "2": 1, "3": 2, "4": 3, "A": 0, "B": 1, "C": 2, "D": 3 };
        if (k in keyMap && optionBtns[keyMap[k]] && !optionBtns[keyMap[k]].disabled) {
          optionBtns[keyMap[k]].click();
        } else if (e.key === "Enter" && submitBtn && !submitBtn.disabled) {
          submitBtn.click();
        }
      };
      window.addEventListener("keydown", handleKeyNav);
    }

    if (skipBtn) {
      skipBtn.onclick = () => {
        AppState.quizSession.currentIndex++;
        loadQuizQuestion();
      };
    }

    if (submitBtn) {
      submitBtn.onclick = async () => {
        let typedVal = "";
        if (qType === "fill_in_the_blanks") {
          typedVal = fitbInput ? fitbInput.value.trim() : "";
          if (!typedVal) return;
        } else if (qType === "typed_qa") {
          typedVal = typedQaInput ? typedQaInput.value.trim() : "";
          if (!typedVal) return;
        } else {
          if (selectedIdx === null) return;
        }

        submitBtn.disabled = true;
        skipBtn.disabled = true;
        if (fitbInput) fitbInput.disabled = true;
        if (typedQaInput) typedQaInput.disabled = true;
        optionBtns.forEach(b => (b.disabled = true));

        try {
          // Authoritative submission to backend supporting mixed question formats
          const result = await ApiService.submitAnswer(
            q.id,
            q.conceptId,
            selectedIdx,
            null,
            typedVal,
            typedVal
          );
          const isCorrect = Boolean(result.correct);
          const correctIdx = result.correctIndex !== undefined ? result.correctIndex : q.correct;
          const explanation = result.explanation || q.explanation || "Evaluation recorded in learner belief model.";
          const reasoningDepth = result.reasoning_depth_level || result.reasoningDepth || "EVALUATED";
          const correctDisplay = result.correct_answer_display || result.correctAnswerDisplay || q.targetBlankAnswer || q.target_blank_answer || "";

          if (isCorrect) {
            AppState.quizSession.score++;
          }
          AppState.quizSession.history.push({
            question: q.question,
            concept: q.conceptName,
            questionType: qType,
            userAnswer: typedVal || (optionBtns[selectedIdx]?.innerText || `Option ${selectedIdx}`),
            isCorrect
          });

          // Style option buttons for MCQ
          if (qType === "mcq") {
            optionBtns.forEach((b, idx) => {
              if (idx === correctIdx) {
                b.classList.add("correct");
              } else if (idx === selectedIdx && !isCorrect) {
                b.classList.add("incorrect");
              }
            });
          }

          // Update session donut accuracy
          updateQuizDonut();

          // Render comprehensive feedback box
          const feedbackBox = $("#quiz-feedback-container");
          let reviewSnippet = "";

          if (qType === "fill_in_the_blanks") {
            reviewSnippet = `
              <div class="quiz-answer-review">
                <div>Your answer: <strong>${esc(typedVal)}</strong></div>
                ${!isCorrect ? `<div>Target keyword: <strong style="color:var(--color-strong);">${esc(correctDisplay)}</strong></div>` : ''}
              </div>
            `;
          } else if (qType === "typed_qa") {
            reviewSnippet = `
              <div class="quiz-answer-review">
                <div>AI Judge Assessment: <strong>${isCorrect ? 'Conceptual Comprehension Verified' : 'Needs Conceptual Clarification'}</strong> (Depth: <em>${esc(reasoningDepth)}</em>)</div>
                ${correctDisplay ? `<div style="margin-top:0.25rem; font-size:0.83rem;">Expected Concepts: ${esc(correctDisplay)}</div>` : ''}
              </div>
            `;
          }

          feedbackBox.innerHTML = `
            <div class="quiz-explanation-box ${isCorrect ? 'correct' : 'incorrect'}">
              <div style="display:flex; align-items:center; gap:0.5rem; margin-bottom:0.35rem; font-weight:700;">
                ${isCorrect ? `
                  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="20 6 9 17 4 12"/></svg>
                  <span>${qType === 'typed_qa' ? 'Concept Well Explained!' : 'Correct!'}</span>
                ` : `
                  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
                  <span>${qType === 'typed_qa' ? 'Review Recommended.' : 'Not quite.'}</span>
                `}
              </div>
              ${reviewSnippet}
              <p style="margin:0; font-size:0.9rem; margin-top:0.4rem;">${esc(explanation)}</p>
            </div>
          `;
          renderMath(feedbackBox);

          // Change submit button to next question
          submitBtn.disabled = false;
          submitBtn.textContent = AppState.quizSession.currentIndex >= AppState.quizSession.totalQuestions
            ? "View Quiz Results →"
            : "Next Question →";

          submitBtn.onclick = () => {
            if (AppState.quizSession.currentIndex >= AppState.quizSession.totalQuestions) {
              renderQuizResults();
            } else {
              AppState.quizSession.currentIndex++;
              loadQuizQuestion(result.next);
            }
          };
        } catch (e) {
          showToast(`Submission error: ${e.message}`, "danger");
          submitBtn.disabled = false;
        }
      };
    }
  }

  function updateQuizDonut() {
    const numEl = $("#quiz-perf-number");
    const donutEl = $("#quiz-perf-donut");
    const encourEl = $("#quiz-encouragement");
    if (!numEl || !donutEl) return;

    const answered = AppState.quizSession.history.length;
    if (answered === 0) return;
    const pct = Math.round((AppState.quizSession.score / answered) * 100);

    numEl.textContent = `${pct}%`;
    donutEl.innerHTML = createDonutSvg(pct, 110, 8, pct >= 70 ? "var(--color-strong)" : "var(--color-weak)");

    if (encourEl) {
      if (pct >= 80) encourEl.textContent = "Superb! Your concept mastery is soaring.";
      else if (pct >= 50) encourEl.textContent = "Good progress! Targeted review will solidify weak areas.";
      else encourEl.textContent = "Keep practicing! Reviewing prerequisites will help unlock these concepts.";
    }
  }

  function renderQuizResults() {
    const box = $("#quiz-card-box");
    if (!box) return;

    const total = AppState.quizSession.history.length;
    const correct = AppState.quizSession.score;
    const pct = total > 0 ? Math.round((correct / total) * 100) : 0;

    box.innerHTML = `
      <div style="padding:2.5rem; text-align:center;">
        <div class="donut-chart-container" style="margin:0 auto 1.5rem auto;">
          ${createDonutSvg(pct, 130, 10, pct >= 70 ? "var(--color-strong)" : "var(--color-weak)")}
          <div class="donut-chart-value">
            <span class="val-number">${pct}%</span>
            <span class="val-label">Final Score</span>
          </div>
        </div>

        <h2 style="margin-bottom:0.5rem;">Assessment Completed!</h2>
        <p style="margin-bottom:1.75rem;">
          You answered <strong>${correct} of ${total}</strong> questions correctly. Your epistemic belief model has been updated.
        </p>

        <div style="display:flex; justify-content:center; gap:1rem;">
          <button class="btn btn-primary" id="btn-restart-quiz">Start Another Quiz</button>
          <a href="#learning-path" class="btn btn-secondary">Review Learning Path &rarr;</a>
        </div>
      </div>
    `;

    const restartBtn = $("#btn-restart-quiz");
    if (restartBtn) {
      restartBtn.onclick = () => {
        AppState.quizSession.currentIndex = 1;
        AppState.quizSession.score = 0;
        AppState.quizSession.history = [];
        loadQuizQuestion();
      };
    }
  }

  // =========================================================================
  // ROUTING & APP INITIALIZATION
  // =========================================================================

  function updateNavigation(routeId) {
    $$(".nav-link").forEach(link => {
      const target = link.getAttribute("data-nav");
      link.classList.toggle("active", target === routeId);
    });

    const pageTitle = $("#topbar-page-title");
    if (pageTitle) {
      const titles = {
        dashboard: "Dashboard",
        upload: "Study Material",
        tutor: "AI Tutor",
        practice: "Concept Check",
        "learning-path": "Learning Path",
        progress: "Mastery Analytics",
        settings: "Settings"
      };
      pageTitle.textContent = titles[routeId] || "Platform";
    }

    const sidebar = $("#sidebar");
    if (sidebar && sidebar.classList.contains("open")) {
      sidebar.classList.remove("open");
    }
  }

  async function navigateTo(routeId) {
    const validRoutes = ["dashboard", "upload", "tutor", "practice", "learning-path", "progress", "settings"];
    const activeRoute = validRoutes.includes(routeId) ? routeId : "dashboard";
    AppState.currentRoute = activeRoute;
    updateNavigation(activeRoute);

    const view = $("#view");
    if (!view) return;

    view.innerHTML = `
      <div style="padding:4rem 2rem; text-align:center; color:var(--ink-muted);">
        <div class="upload-icon-pulse" style="width:40px; height:40px; margin-bottom:1rem;">
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>
        </div>
        Loading ${activeRoute}...
      </div>
    `;

    try {
      const html = await Views[toCamelCase(activeRoute)]();
      view.innerHTML = html;
      renderMath(view);
      if (AfterRender[toCamelCase(activeRoute)]) {
        AfterRender[toCamelCase(activeRoute)]();
      }
    } catch (err) {
      view.innerHTML = `
        <div class="card" style="padding:3rem; text-align:center;">
          <h2 style="color:var(--color-danger); margin-bottom:0.75rem;">Something went wrong</h2>
          <p style="margin-bottom:1.5rem;">${esc(err.message)}</p>
          <a href="#dashboard" class="btn btn-primary">Return to Dashboard</a>
        </div>
      `;
    }
  }

  function toCamelCase(str) {
    return str.replace(/-([a-z])/g, g => g[1].toUpperCase());
  }

  function handleHashChange() {
    const rawHash = location.hash.slice(1).trim();
    navigateTo(rawHash);
  }

  // Application Entry Point
  function initApp() {
    initTheme();
    initMobileNav();
    initModals();

    const topName = $("#topbar-user-name");
    const topAvatar = $("#topbar-user-avatar");
    if (topName) topName.textContent = AppState.studentName;
    if (topAvatar) topAvatar.textContent = AppState.studentName[0] || "A";

    window.addEventListener("hashchange", handleHashChange);
    handleHashChange();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initApp);
  } else {
    initApp();
  }
})();
