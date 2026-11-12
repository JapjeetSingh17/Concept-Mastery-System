/**
 * Concept Mastery Platform - Mock Demo Service
 * Isolated demo dataset for frontend-only previews and offline testing.
 * Strictly gated behind explicit ?mock=true or localStorage.getItem("USE_MOCK") === "true".
 */

(function() {
  function isMockModeAllowed() {
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

  if (!isMockModeAllowed()) {
    return;
  }

  const MOCK_CONCEPTS = [
    { id: "loss-functions", name: "Loss Functions", x: 90, y: 70, deps: [] },
    { id: "gradient-descent", name: "Gradient Descent", x: 260, y: 70, deps: ["loss-functions"] },
    { id: "learning-rate", name: "Learning Rate", x: 260, y: 190, deps: ["gradient-descent"] },
    { id: "backpropagation", name: "Backpropagation", x: 430, y: 70, deps: ["gradient-descent"] },
    { id: "overfitting", name: "Overfitting", x: 430, y: 190, deps: ["backpropagation"] },
    { id: "regularization", name: "Regularization", x: 600, y: 130, deps: ["overfitting", "learning-rate"] }
  ];

  const QUESTION_BANK = {
    "loss-functions": {
      question: "What is the primary role of a loss function in machine learning?",
      options: [
        "To measure the gap between predicted outputs and ground-truth targets",
        "To compress the dataset into lower-dimensional embeddings",
        "To set the learning rate for each gradient descent step",
        "To calculate the total number of epochs needed for training"
      ],
      correct: 0,
      explanation: "A loss function measures the gap between a model's predictions and the true targets. Training tries to make the loss function as small as possible. (p. 1)",
      text: "A loss function measures the gap between a model's predictions and the true targets. Training tries to make the loss function as small as possible. Mean squared error is a common loss function for regression tasks."
    },
    "gradient-descent": {
      question: "How does gradient descent update parameters during training?",
      options: [
        "By moving weights in the direction that maximizes the loss function",
        "By moving weights against the gradient of the loss to reach a minimum",
        "By randomly assigning new values to weights until error drops",
        "By computing the matrix inverse of the feature dataset"
      ],
      correct: 1,
      explanation: "Gradient descent is an optimization method that updates the weights by moving them against the gradient of the loss. (p. 1)",
      text: "Gradient descent is an optimization method that updates the weights by moving them against the gradient of the loss. Each step reduces the loss a little, and repeating many steps leads toward a minimum. Gradient descent needs a loss function to tell it which direction is downhill."
    },
    "learning-rate": {
      question: "What is the consequence of setting an excessively high learning rate?",
      options: [
        "The model converges to an exact minimum too quickly",
        "Training overshoots the minimum and diverges",
        "Memory consumption doubles with each epoch",
        "Gradients become exactly zero at step one"
      ],
      correct: 1,
      explanation: "A learning rate that is too high makes training overshoot the minimum and diverge, while a learning rate that is too low makes training very slow. (p. 1)",
      text: "The learning rate controls how large each gradient descent step is. A learning rate that is too high makes training overshoot the minimum and diverge, while a learning rate that is too low makes training very slow. Many practitioners decay the learning rate as training progresses."
    },
    "backpropagation": {
      question: "Which mathematical rule powers backpropagation across multiple layers?",
      options: [
        "L'Hôpital's Rule for limits",
        "The Chain Rule of calculus",
        "The Law of Total Probability",
        "Bayes' Theorem"
      ],
      correct: 1,
      explanation: "Backpropagation computes the gradient of the loss with respect to every weight by applying the chain rule layer by layer. (p. 1)",
      text: "Backpropagation computes the gradient of the loss with respect to every weight by applying the chain rule layer by layer. Backpropagation reuses intermediate results from the forward pass, which makes it efficient. The gradients from backpropagation are then used by gradient descent to update the weights."
    },
    "overfitting": {
      question: "What is a telltale signature of overfitting in model evaluation curves?",
      options: [
        "Training loss decreases while validation loss begins to increase",
        "Both training loss and validation loss decrease simultaneously",
        "Training loss stays high and refuses to drop",
        "The learning rate decay rate spikes to infinity"
      ],
      correct: 0,
      explanation: "A widening gap between training loss and validation loss is the classic sign of overfitting. (p. 1)",
      text: "Overfitting happens when a model memorizes the training data and fails to generalize to new examples. A widening gap between training loss and validation loss is the classic sign of overfitting. Overfitting is more likely with small datasets and large models."
    },
    "regularization": {
      question: "How does L2 regularization help prevent overfitting?",
      options: [
        "By discarding all training samples with high variance",
        "By penalizing large weights to keep the model smoother",
        "By reducing the number of input features to zero",
        "By artificially increasing the learning rate step size"
      ],
      correct: 1,
      explanation: "Regularization discourages overfitting by adding a penalty to the loss function. L2 regularization penalizes large weights. (p. 1)",
      text: "Regularization discourages overfitting by adding a penalty to the loss function. L2 regularization penalizes large weights, which keeps the model smoother. Dropout and early stopping are other popular regularization techniques."
    }
  };

  const mockState = {
    docReady: true,
    file: "ml-basics.pdf",
    pages: 1,
    chunks: 6,
    mastery: {
      "loss-functions": { a: 3, b: 1, n: 3, score: 0.75, attempts: 3, status: "strong", uncertainty: 0.19, why: "2 of 3 answers right, so estimated mastery is 75%." },
      "gradient-descent": { a: 2, b: 2, n: 2, score: 0.50, attempts: 2, status: "weak", uncertainty: 0.22, why: "1 of 2 answers right, so estimated mastery is 50%." },
      "learning-rate": { a: 1, b: 1, n: 0, score: 0.50, attempts: 0, status: "unknown", uncertainty: 0.28, why: "No answers yet. That isn’t enough evidence to call it strong or weak." },
      "backpropagation": { a: 1, b: 1, n: 0, score: 0.50, attempts: 0, status: "unknown", uncertainty: 0.28, why: "No answers yet. That isn’t enough evidence to call it strong or weak." },
      "overfitting": { a: 1, b: 1, n: 0, score: 0.50, attempts: 0, status: "unknown", uncertainty: 0.28, why: "No answers yet. That isn’t enough evidence to call it strong or weak." },
      "regularization": { a: 1, b: 1, n: 0, score: 0.50, attempts: 0, status: "unknown", uncertainty: 0.28, why: "No answers yet. That isn’t enough evidence to call it strong or weak." }
    },
    questionCounter: 1
  };

  const delay = ms => new Promise(res => setTimeout(res, ms));

  window.MOCK = {
    async health() {
      await delay(50);
      return { ok: true, mock: true };
    },

    async ingestStatus() {
      await delay(60);
      return {
        ready: mockState.docReady,
        file: mockState.file,
        pages: mockState.pages,
        concepts: MOCK_CONCEPTS.length
      };
    },

    async ingest(formData) {
      await delay(800);
      mockState.docReady = true;
      mockState.file = "custom-material.pdf";
      return {
        jobId: "mock_job_1",
        file: mockState.file,
        pages: 3,
        chunks: 14,
        concepts: MOCK_CONCEPTS.length
      };
    },

    async ingestSample() {
      await delay(400);
      mockState.docReady = true;
      mockState.file = "ml-basics.pdf";
      return {
        jobId: "ml-basics.pdf",
        file: "ml-basics.pdf",
        pages: 1,
        chunks: 6,
        concepts: 6
      };
    },

    async ask({ question }) {
      await delay(600);
      const lower = (question || "").toLowerCase();
      let match = MOCK_CONCEPTS.find(c => lower.includes(c.id) || lower.includes(c.name.toLowerCase()));
      if (!match) match = MOCK_CONCEPTS[0];

      const item = QUESTION_BANK[match.id];
      return {
        answer: `From your notes: ${item.explanation} ${item.text.slice(0, 160)}...`,
        citations: [
          {
            page: 1,
            snippet: match.name,
            text: item.text
          }
        ],
        concepts: [match.id]
      };
    },

    async graph() {
      await delay(100);
      return {
        nodes: MOCK_CONCEPTS.map(c => ({ id: c.id, name: c.name, x: c.x, y: c.y })),
        edges: MOCK_CONCEPTS.flatMap(c => c.deps.map(d => [d, c.id]))
      };
    },

    async quizNext({ conceptId } = {}) {
      await delay(250);
      let targetId = conceptId;
      if (!targetId || !QUESTION_BANK[targetId]) {
        // Find a weak or unknown concept
        const keys = Object.keys(QUESTION_BANK);
        targetId = keys.find(k => mockState.mastery[k]?.status === "weak") ||
                   keys.find(k => mockState.mastery[k]?.status === "unknown") ||
                   keys[0];
      }
      const item = QUESTION_BANK[targetId];
      const concept = MOCK_CONCEPTS.find(c => c.id === targetId);

      return {
        id: `mock_q_${mockState.questionCounter++}`,
        conceptId: targetId,
        conceptName: concept ? concept.name : targetId,
        question: item.question,
        options: item.options,
        correct: item.correct,
        explanation: item.explanation
      };
    },

    async quizAnswer({ questionId, conceptId, selectedIndex, correct }) {
      await delay(250);
      const item = QUESTION_BANK[conceptId] || QUESTION_BANK["loss-functions"];
      const isCorrect = (selectedIndex !== undefined && selectedIndex !== null)
        ? selectedIndex === item.correct
        : Boolean(correct);

      const m = mockState.mastery[conceptId] || { a: 1, b: 1, n: 0 };
      m.n += 1;
      if (isCorrect) m.a += 1;
      else m.b += 1;

      const mean = m.a / (m.a + m.b);
      m.score = Math.round(mean * 1000) / 1000;
      m.attempts = m.n;
      m.status = m.n < 2 ? "unknown" : (m.score >= 0.7 ? "strong" : "weak");
      m.why = m.status === "unknown"
        ? `Only ${m.n} answer so far. That isn’t enough evidence to call it strong or weak.`
        : `${Math.round(m.a - 1)} of ${m.n} answers right, so estimated mastery is ${Math.round(m.score * 100)}%.`;

      // Next recommended concept
      const nextId = Object.keys(QUESTION_BANK).find(k => k !== conceptId) || conceptId;

      return {
        correct: isCorrect,
        correctIndex: item.correct,
        explanation: item.explanation,
        mastery: Object.keys(mockState.mastery).map(k => ({
          id: k,
          name: MOCK_CONCEPTS.find(c => c.id === k)?.name || k,
          ...mockState.mastery[k]
        })),
        next: nextId
      };
    },

    async mastery() {
      await delay(120);
      return {
        concepts: MOCK_CONCEPTS.map(c => ({
          id: c.id,
          name: c.name,
          ...mockState.mastery[c.id]
        }))
      };
    },

    async masteryReset() {
      await delay(100);
      Object.keys(mockState.mastery).forEach(k => {
        mockState.mastery[k] = {
          a: 1, b: 1, n: 0, score: 0.0, attempts: 0,
          status: "unknown", uncertainty: 0.28,
          why: "No answers yet. That isn’t enough evidence to call it strong or weak."
        };
      });
      return {
        ok: true,
        concepts: MOCK_CONCEPTS.map(c => ({ id: c.id, name: c.name, ...mockState.mastery[c.id] }))
      };
    },

    async recommend() {
      await delay(150);
      const actions = [];
      MOCK_CONCEPTS.forEach(c => {
        const m = mockState.mastery[c.id];
        if (m && m.status !== "strong") {
          actions.push({
            conceptId: c.id,
            name: c.name,
            type: m.status === "unknown" ? "Teach first" : "Re-teach with a new angle",
            reason: m.why,
            pages: [1],
            lesson: QUESTION_BANK[c.id]?.explanation || "Study this concept thoroughly."
          });
        }
      });
      return { actions, ready: true };
    }
  };
})();
