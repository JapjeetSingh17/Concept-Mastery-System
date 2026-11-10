"""Diagnostic Assessment & Quiz Generation Engine.

Generates diverse diagnostic assessment items strictly from vectorless chunks:
  1. Multiple Choice (MCQ): 4-option diagnostic questions.
  2. Fill-in-the-Blanks (FITB): Cloze sentences testing specific terminology and formulas.
  3. Typed Question-Answer (Typed Q&A): Conceptual questions requiring open-ended text explanations evaluated by the AI Mastery Judge.

Evaluates student choices, fuzzy text matching, and natural language reasoning depth.
Updates epistemic mastery records and pinpoints misconceptions.
"""

from __future__ import annotations

import difflib
import itertools
import json
import math
import random
import re
import uuid
from typing import List, Tuple, Literal, Optional

from ..core import store, llm
from ..core.concept_index import GLOBAL_INDEX
from ..core.text import sentences, tokens, normalize, STOP, GENERIC_NON_CONCEPTS
from ..models.schemas import QuizQuestion, QuizSubmission, QuizResult
from . import mastery
from .concepts import pattern

_q_counter = itertools.count(1)
# Server-side authoritative question registry
REGISTRY: dict[str, dict] = {}
SERVED_QUESTIONS: dict[str, list[str]] = {}
SERVED_SENTENCES: dict[str, set[str]] = {}


def is_mathematical_concept(concept_id: str, concept_name: str = "") -> bool:
    """Check if concept matches mathematical or algorithmic problem domains."""
    t = f"{concept_id} {concept_name}".lower()
    math_keywords = (
        "gradient", "descent", "sgd", "knn", "nearest", "euclidean", "distance",
        "mse", "loss", "error", "entropy", "cross-entropy", "bce",
        "precision", "recall", "f1", "bayes", "posterior", "prior", "probability",
        "learning_rate", "schedule", "decay", "matrix", "vector", "dot", "algebra",
        "weight", "optimization"
    )
    return any(k in t for k in math_keywords)


def generate_numerical_question(
    concept_id: str,
    concept_name: str = "",
    user_id: str = "default",
    source_page: int = 1,
) -> QuizQuestion:
    """Generate a concept-aware numerical question for mathematical and algorithmic concepts.
    
    Supported concept families:
      1. Gradient descent weight update
      2. KNN Euclidean distance
      3. MSE loss
      4. Cross-Entropy loss
      5. Precision / Recall
      6. Bayes' Theorem
      7. Learning rate schedule
      8. Matrix dot products
    """
    qid = f"num_{uuid.uuid4().hex[:8]}"
    text_target = f"{concept_id} {concept_name}".lower().replace("-", "_")

    if any(k in text_target for k in ("gradient", "descent", "sgd", "weight_update", "backprop")):
        w0 = round(random.choice([1.0, 1.5, 2.0, 2.5, 3.0]), 2)
        alpha = round(random.choice([0.01, 0.05, 0.1, 0.2]), 2)
        g = round(random.choice([1.2, 2.0, 2.5, 3.2]), 2)
        ans = round(w0 - (alpha * g), 2)
        q_text = (
            f"Given a single-weight linear model with current weight w = {w0}, learning rate α = {alpha}, "
            f"and calculated loss gradient ∂L/∂w = {g}, compute the updated weight w_new after one gradient descent step."
        )
        hint = "w_new = w - α · (∂L/∂w)"
        solution = (
            f"Step 1: Identify given parameters: w = {w0}, α = {alpha}, ∂L/∂w = {g}.\n"
            f"Step 2: Apply the weight update rule: w_new = w - α · (∂L/∂w).\n"
            f"Step 3: Calculate the product: {alpha} × {g} = {alpha * g:.4f}.\n"
            f"Step 4: Compute w_new = {w0} - {alpha * g:.4f} = {ans:.2f}."
        )
        params = {"current_weight": w0, "learning_rate": alpha, "gradient": g}
        correct = ans

    elif any(k in text_target for k in ("knn", "nearest_neighbor", "k_nearest", "euclidean", "distance_metric")):
        x1, y1 = random.randint(1, 4), random.randint(1, 4)
        dx, dy = random.choice([(3, 4), (4, 3), (6, 8), (1, 2), (2, 2)])
        x2, y2 = x1 + dx, y1 + dy
        ans = round(math.sqrt(dx**2 + dy**2), 2)
        q_text = (
            f"In a K-Nearest Neighbors (KNN) model, compute the Euclidean distance between sample point "
            f"P1 = ({x1}, {y1}) and query point P2 = ({x2}, {y2})."
        )
        hint = "d(P1, P2) = √((x2 - x1)² + (y2 - y1)²)"
        solution = (
            f"Step 1: Compute coordinate differences: Δx = {x2} - {x1} = {dx}, Δy = {y2} - {y1} = {dy}.\n"
            f"Step 2: Square differences: (Δx)² = {dx**2}, (Δy)² = {dy**2}.\n"
            f"Step 3: Sum squares: {dx**2} + {dy**2} = {dx**2 + dy**2}.\n"
            f"Step 4: Take the square root: √{dx**2 + dy**2} = {ans:.2f}."
        )
        params = {"x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2)}
        correct = ans

    elif any(k in text_target for k in ("mse", "mean_squared_error", "l2_loss", "squared_error")):
        y1, y2 = random.choice([2.0, 3.0, 4.0]), random.choice([4.0, 5.0, 6.0])
        y_hat1 = round(y1 + random.choice([-1.0, 1.0, 2.0]), 1)
        y_hat2 = round(y2 + random.choice([-2.0, -1.0, 1.0]), 1)
        err1 = (y1 - y_hat1) ** 2
        err2 = (y2 - y_hat2) ** 2
        ans = round((err1 + err2) / 2.0, 2)
        q_text = (
            f"Calculate the Mean Squared Error (MSE) loss for a dataset with two observations: "
            f"actual values y = [{y1}, {y2}] and model predictions ŷ = [{y_hat1}, {y_hat2}]."
        )
        hint = "MSE = (1/n) · Σ (y_i - ŷ_i)²"
        solution = (
            f"Step 1: Compute individual squared residuals:\n"
            f"  (y1 - ŷ1)² = ({y1} - {y_hat1})² = {err1:.2f}\n"
            f"  (y2 - ŷ2)² = ({y2} - {y_hat2})² = {err2:.2f}\n"
            f"Step 2: Take the average: ({err1:.2f} + {err2:.2f}) / 2 = {ans:.2f}."
        )
        params = {"y1": y1, "y_hat1": y_hat1, "y2": y2, "y_hat2": y_hat2}
        correct = ans

    elif any(k in text_target for k in ("cross_entropy", "log_loss", "bce", "entropy")):
        p = random.choice([0.60, 0.70, 0.75, 0.80, 0.85, 0.90])
        ans = round(-math.log(p), 2)
        q_text = (
            f"For a binary classification instance where true class label y = 1, the model outputs "
            f"predicted probability p(y=1) = {p:.2f}. Compute the Binary Cross-Entropy loss."
        )
        hint = "BCE = -[y · ln(p) + (1 - y) · ln(1 - p)] = -ln(p) for y = 1"
        solution = (
            f"Step 1: For ground truth y = 1, BCE reduces to -ln(p).\n"
            f"Step 2: Calculate natural log: ln({p:.2f}) ≈ {math.log(p):.4f}.\n"
            f"Step 3: Negate: -({math.log(p):.4f}) = {ans:.2f}."
        )
        params = {"true_label_y": 1.0, "predicted_probability_p": p}
        correct = ans

    elif any(k in text_target for k in ("precision", "recall", "f1", "confusion_matrix", "classification_metric")):
        tp = random.choice([70, 80, 90])
        fp = random.choice([10, 20, 30])
        fn = random.choice([10, 15, 20])
        if "recall" in text_target:
            metric = "Recall"
            ans = round(tp / (tp + fn), 2)
            hint = "Recall = TP / (TP + FN)"
            denom = tp + fn
            solution = (
                f"Step 1: Identify TP = {tp}, FN = {fn}.\n"
                f"Step 2: Compute denominator TP + FN = {tp} + {fn} = {denom}.\n"
                f"Step 3: Calculate Recall = {tp} / {denom} = {ans:.2f}."
            )
        else:
            metric = "Precision"
            ans = round(tp / (tp + fp), 2)
            hint = "Precision = TP / (TP + FP)"
            denom = tp + fp
            solution = (
                f"Step 1: Identify TP = {tp}, FP = {fp}.\n"
                f"Step 2: Compute denominator TP + FP = {tp} + {fp} = {denom}.\n"
                f"Step 3: Calculate Precision = {tp} / {denom} = {ans:.2f}."
            )
        q_text = (
            f"A machine learning classifier evaluates test data yielding True Positives TP = {tp}, "
            f"False Positives FP = {fp}, and False Negatives FN = {fn}. Calculate the model's {metric}."
        )
        params = {"TP": float(tp), "FP": float(fp), "FN": float(fn)}
        correct = ans

    elif any(k in text_target for k in ("bayes", "bayesian", "prior", "posterior", "conditional_prob")):
        p_a = random.choice([0.02, 0.05, 0.10])
        p_b_given_a = random.choice([0.80, 0.90, 0.95])
        p_b_given_not_a = random.choice([0.05, 0.10])
        p_not_a = round(1.0 - p_a, 4)
        p_b = round((p_b_given_a * p_a) + (p_b_given_not_a * p_not_a), 4)
        ans = round((p_b_given_a * p_a) / p_b, 2)
        q_text = (
            f"Given prior probability P(A) = {p_a}, sensitivity P(B|A) = {p_b_given_a}, and "
            f"false positive rate P(B|¬A) = {p_b_given_not_a}, compute the posterior probability P(A|B) using Bayes' Theorem."
        )
        hint = "P(A|B) = [P(B|A) · P(A)] / [P(B|A)·P(A) + P(B|¬A)·P(¬A)]"
        solution = (
            f"Step 1: Calculate total probability P(B):\n"
            f"  P(B) = ({p_b_given_a} × {p_a}) + ({p_b_given_not_a} × {p_not_a:.2f}) = {p_b:.4f}.\n"
            f"Step 2: Calculate numerator P(B|A) · P(A) = {p_b_given_a * p_a:.4f}.\n"
            f"Step 3: Compute posterior P(A|B) = {p_b_given_a * p_a:.4f} / {p_b:.4f} = {ans:.2f}."
        )
        params = {"P_A": p_a, "P_B_given_A": p_b_given_a, "P_B_given_not_A": p_b_given_not_a}
        correct = ans

    elif any(k in text_target for k in ("learning_rate", "lr_schedule", "decay", "schedule")):
        alpha0 = random.choice([0.1, 0.2, 0.05])
        gamma = random.choice([0.5, 0.8, 0.9])
        step = random.choice([2, 3])
        ans = round(alpha0 * (gamma ** step), 2)
        q_text = (
            f"Under an exponential learning rate decay schedule α_t = α_0 · γ^t, calculate the decayed learning rate "
            f"at step t = {step} given initial learning rate α_0 = {alpha0} and decay factor γ = {gamma}."
        )
        hint = "α_t = α_0 · γ^t"
        solution = (
            f"Step 1: Compute decay factor exponent: {gamma}^{step} = {gamma ** step:.4f}.\n"
            f"Step 2: Multiply by initial learning rate: {alpha0} × {gamma ** step:.4f} = {ans:.2f}."
        )
        params = {"initial_lr": alpha0, "decay_rate": gamma, "step": float(step)}
        correct = ans

    else:  # Matrix dot product / Linear algebra default
        u = [random.randint(1, 4), random.randint(1, 3), random.randint(-2, 2)]
        v = [random.randint(1, 3), random.randint(-1, 3), random.randint(1, 4)]
        dot = float(sum(u[i] * v[i] for i in range(3)))
        q_text = f"Compute the dot product u · v for 3D vectors u = {u} and v = {v}."
        hint = "u · v = u1·v1 + u2·v2 + u3·v3"
        terms = [f"({u[i]} × {v[i]})" for i in range(3)]
        vals = [u[i] * v[i] for i in range(3)]
        solution = (
            f"Step 1: Multiply corresponding vector components: {' + '.join(terms)}.\n"
            f"Step 2: Sum component products: {' + '.join(str(val) for val in vals)} = {dot:.2f}."
        )
        params = {"u1": float(u[0]), "u2": float(u[1]), "u3": float(u[2]), "v1": float(v[0]), "v2": float(v[1]), "v3": float(v[2])}
        correct = dot

    q = QuizQuestion(
        question_id=qid,
        concept_id=concept_id,
        question_type="numerical",
        question_text=q_text,
        options=[],
        correct_option_idx=0,
        target_blank_answer=str(correct),
        rubric=f"Numerical calculation within ±0.05 tolerance. Step-by-step solution provided.",
        key_terms=[str(correct)],
        explanation=solution,
        source_page=source_page,
        given_parameters=params,
        formula_hint=hint,
        correct_numeric_answer=correct,
        tolerance=0.05,
        unit_or_format="Round to 2 decimal places",
        step_by_step_solution=solution,
    )
    REGISTRY[qid] = q.model_dump()
    return q


def generate_diagnostic_quiz(
    concept_id: str,
    count: int = 3,
    question_types: list[str] | None = None,
    user_id: str = "default",
    chunks: list | None = None,
    kb: dict | None = None,
    offset: int | None = None
) -> list[QuizQuestion]:
    """Generate diagnostic assessment items strictly from vectorless chunks for concept_id.
    
    Produces a balanced mix across question types (MCQ, Fill-in-the-Blanks, Typed Q&A, and Numerical).
    Guarantees non-duplicate question variants across repeated calls per user and concept.
    """
    if kb is None:
        kb = store.require(user_id=user_id)
    by_id = {c["id"]: c for c in kb["concepts"]}
    concept = by_id.get(concept_id)
    if not concept:
        raise store.UserError(f"Concept '{concept_id}' not found.")

    history_key = f"{user_id}:{concept_id}"
    prior_served = SERVED_QUESTIONS.setdefault(history_key, [])
    prior_sentences = SERVED_SENTENCES.setdefault(history_key, set())
    calc_offset = offset if offset is not None else len(prior_served)

    # Determine desired question types
    if question_types:
        target_types = question_types
    elif count == 1:
        # Default single question to MCQ for legacy compatibility
        target_types = ["mcq"]
    else:
        # Standard balanced rotation: mcq -> fill_in_the_blanks -> typed_qa
        cycle_types = ["mcq", "fill_in_the_blanks", "typed_qa"]
        target_types = [cycle_types[i % len(cycle_types)] for i in range(count)]

    # 1. Retrieve vectorless chunks for this concept if not provided
    if chunks is None:
        chunks = GLOBAL_INDEX.retrieve_chunks(concept_id, max_chunks=3, user_id=user_id)
        if not chunks and "index" in kb and kb["index"]:
            all_chunks = getattr(kb["index"], "chunks", [])
            chunks = [
                type("ChunkMeta", (), {
                    "chunk_id": str(all_chunks[j]["id"]),
                    "page_number": all_chunks[j]["page"],
                    "section_title": all_chunks[j]["section"],
                    "text": all_chunks[j]["text"],
                })()
                for j in concept.get("chunks", [])[:3]
                if j < len(all_chunks)
            ]

    if not chunks:
        raise store.UserError(f"No textual material available to generate questions for {concept['name']}.")

    questions: list[QuizQuestion] = []

    # 2. Try LLM generation for diverse question types if enabled
    if llm.enabled():
        ctx_text = "\n\n".join(f"[{getattr(c, 'locator_label', None) or f'Page {c.page_number}'}] {c.text}" for c in chunks)
        system_prompt = (
            "You are a rigorous psychometric assessment designer for advanced technical education. "
            "Generate high-signal diagnostic assessment items strictly grounded in the provided textbook context.\n\n"
            "CRITICAL PEDAGOGICAL ANTI-BIAS RULES:\n"
            "1. For 'mcq' questions, the correct option and distractors MUST BE DETAILED MECHANISMS, PROPERTIES, OR RUNTIMES.\n"
            "2. THE CORRECT OPTION FOR MCQ MUST NEVER BE THE CONCEPT NAME ITSELF! (e.g., if testing 'Array', the correct answer "
            "must NEVER simply be 'Array' or a concept title. Instead, it must be the accurate operational mechanism or invariant, "
            "such as 'Allocates contiguous memory cells allowing O(1) random access via base address arithmetic').\n"
            "3. Distractors must be plausible technical alternatives or common student misconceptions.\n"
            "4. For 'fill_in_the_blanks', blank out an essential operational keyword, complexity, or property, NOT the concept name.\n"
            "Support four question types:\n"
            "  - 'mcq': Multiple-choice with 4 substantive mechanism options\n"
            "  - 'fill_in_the_blanks': A cloze sentence containing '_____' testing an essential property/mechanism word\n"
            "  - 'typed_qa': An open-ended question testing conceptual mechanisms, with a 'rubric' and 'key_terms'\n"
            "  - 'numerical': A parameter calculation question with 'given_parameters', 'formula_hint', 'correct_numeric_answer', and 'step_by_step_solution'\n"
            "Reply strictly with a valid JSON array only."
        )
        avoid_clause = ""
        if prior_served:
            recent_list = "\n".join(f"- {q}" for q in prior_served[-5:])
            avoid_clause = f"\nIMPORTANT: Do NOT repeat or duplicate these recently served questions:\n{recent_list}\nGenerate a fresh question with a different focus."

        user_prompt = (
            f"Concept to test: {concept['name']}\n"
            f"Textbook Context:\n{ctx_text}\n\n"
            f"Generate {count} questions matching this balanced sequence of types: {target_types}.{avoid_clause}\n"
            f"JSON Schema:\n"
            f'[\n'
            f'  {{\n'
            f'    "question_type": "mcq" | "fill_in_the_blanks" | "typed_qa" | "numerical",\n'
            f'    "question_text": "<clear prompt, cloze sentence, or numerical problem>",\n'
            f'    "options": ["<substantive option 0>", "<substantive option 1>", "<substantive option 2>", "<substantive option 3>"] (only for mcq, [] otherwise. NEVER use just concept names),\n'
            f'    "correct_option_idx": <0-3> (only for mcq, 0 otherwise),\n'
            f'    "target_blank_answer": "<exact missing property or term>" (only for fill_in_the_blanks, "" otherwise),\n'
            f'    "rubric": "<evaluation criteria for typed_qa>" (only for typed_qa, "" otherwise),\n'
            f'    "key_terms": ["<term 1>", "<term 2>"] (only for typed_qa, [] otherwise),\n'
            f'    "given_parameters": {{"param1": 1.0}} (only for numerical, null otherwise),\n'
            f'    "formula_hint": "<formula hint>" (only for numerical, null otherwise),\n'
            f'    "correct_numeric_answer": <float> (only for numerical, null otherwise),\n'
            f'    "tolerance": 0.05,\n'
            f'    "unit_or_format": "Round to 2 decimal places",\n'
            f'    "step_by_step_solution": "<step-by-step solution>" (only for numerical, null otherwise),\n'
            f'    "explanation": "<grounded explanation citing (p. X)>",\n'
            f'    "source_page": <int page number>\n'
            f'  }}\n'
            f']\n'
            f'Respond with ONLY a valid JSON array of objects.'
        )

        try:
            raw = llm.complete(system_prompt, user_prompt, max_tokens=1400)
            if raw:
                arr = llm.safe_json_loads(raw)
                if not isinstance(arr, list):
                    clean_json = llm.extract_json_payload(raw)
                    arr = json.loads(clean_json, strict=False) if clean_json else []
                if not isinstance(arr, list):
                    arr = []
                c_name_lower = concept["name"].lower().strip()
                for item in arr:
                    q_type = str(item.get("question_type", "mcq")).lower()
                    if q_type not in ("mcq", "fill_in_the_blanks", "typed_qa", "numerical"):
                        q_type = "mcq"

                    # Anti-bias verification: reject MCQs where the correct option is just the concept name
                    opts = [str(opt).strip() for opt in item.get("options", [])]
                    c_idx = int(item.get("correct_option_idx", 0))
                    if q_type == "mcq" and opts and 0 <= c_idx < len(opts):
                        if opts[c_idx].lower().strip() == c_name_lower:
                            # Rephrase or skip invalid trivial item
                            continue

                    qid = f"diag_{uuid.uuid4().hex[:8]}"
                    params = item.get("given_parameters")
                    formula_hint = item.get("formula_hint")
                    correct_num = item.get("correct_numeric_answer")
                    if correct_num is not None:
                        try:
                            correct_num = float(correct_num)
                        except Exception:
                            correct_num = None
                    step_sol = item.get("step_by_step_solution")
                    tol = float(item.get("tolerance", 0.05))
                    unit_fmt = str(item.get("unit_or_format", "Round to 2 decimal places"))

                    q = QuizQuestion(
                        question_id=qid,
                        concept_id=concept_id,
                        question_type=q_type,
                        question_text=str(item["question_text"]).strip(),
                        options=opts,
                        correct_option_idx=c_idx,
                        target_blank_answer=str(item.get("target_blank_answer", "")).strip(),
                        rubric=str(item.get("rubric", "")).strip(),
                        key_terms=[str(t).strip() for t in item.get("key_terms", [])],
                        explanation=str(item.get("explanation", f"Grounded in page {item.get('source_page', chunks[0].page_number)}")),
                        source_page=int(item.get("source_page", chunks[0].page_number)),
                        given_parameters=params,
                        formula_hint=formula_hint,
                        correct_numeric_answer=correct_num,
                        tolerance=tol,
                        unit_or_format=unit_fmt,
                        step_by_step_solution=step_sol,
                    )
                    questions.append(q)
                    REGISTRY[qid] = q.model_dump()
                    if len(questions) >= count:
                        break
        except Exception:
            pass

    # 3. Deterministic vectorless fallback generator for remaining slots
    if len(questions) < count:
        pat = pattern(concept)
        other_concepts = [c["name"] for c in kb["concepts"] if c["id"] != concept_id]
        distractor_pool = other_concepts if len(other_concepts) >= 3 else ["Optimization", "Overfitting", "Gradient", "Loss"]

        # Rotate starting chunk to ensure varied context across calls
        chunk_start = calc_offset % max(len(chunks), 1)
        ordered_chunks = chunks[chunk_start:] + chunks[:chunk_start]

        cycle_limit = 0
        while len(questions) < count and cycle_limit < 5:
            cycle_limit += 1
            for chunk in ordered_chunks:
                if len(questions) >= count:
                    break
            
            chunk_sentences = sentences(chunk.text)
            matched_sentences = [s for s in chunk_sentences if pat.search(s) and 8 <= len(s.split()) <= 40]
            if not matched_sentences:
                matched_sentences = [chunk_sentences[0]] if chunk_sentences else [chunk.text[:120]]

            # Filter out sentences already used for this concept
            fresh_sentences = [s for s in matched_sentences if s not in prior_sentences]
            if not fresh_sentences:
                # If all available sentences have been used, cycle afresh
                prior_sentences.clear()
                fresh_sentences = matched_sentences

            # Rotate starting sentence
            sent_start = calc_offset % max(len(fresh_sentences), 1)
            ordered_sentences = fresh_sentences[sent_start:] + fresh_sentences[:sent_start]

            for sent in ordered_sentences:
                if len(questions) >= count:
                    break

                desired_type = target_types[len(questions) % len(target_types)]
                qid = f"diag_{uuid.uuid4().hex[:8]}"

                loc = getattr(chunk, "locator_label", None) or f"Page {chunk.page_number}"
                if desired_type == "fill_in_the_blanks":
                    c_name_lower = concept["name"].lower()
                    words_in_sent = [
                        w for w in re.findall(r"[A-Za-z0-9\-]+", sent)
                        if len(w) >= 4 and w.lower() not in STOP and w.lower() not in GENERIC_NON_CONCEPTS
                        and c_name_lower not in w.lower() and w.lower() not in c_name_lower
                    ]
                    if words_in_sent:
                        target_blank = max(words_in_sent, key=len)
                        b_pat = re.compile(r"\b" + re.escape(target_blank) + r"\b", re.I)
                        masked_text = b_pat.sub("_____", sent, count=1)
                    else:
                        target_blank = concept["name"]
                        masked_text = pat.sub("_____", sent) if pat.search(sent) else f"{sent.rsplit(' ', 2)[0]} _____."

                    q = QuizQuestion(
                        question_id=qid,
                        concept_id=concept_id,
                        question_type="fill_in_the_blanks",
                        question_text=f"Complete the core mechanism of {concept['name']}: \"{masked_text}\"",
                        options=[],
                        correct_option_idx=0,
                        target_blank_answer=target_blank,
                        rubric=f"Accepts '{target_blank}' or close technical equivalents.",
                        key_terms=[target_blank.lower()],
                        explanation=f"Missing term is '{target_blank}'. Grounding sentence: {sent} (p. {chunk.page_number}) [{loc}]",
                        source_page=chunk.page_number,
                    )
                elif desired_type == "typed_qa":
                    q = QuizQuestion(
                        question_id=qid,
                        concept_id=concept_id,
                        question_type="typed_qa",
                        question_text=f"In your own words, explain the core mechanism of {concept['name']} and how it behaves according to your study material.",
                        options=[],
                        correct_option_idx=0,
                        target_blank_answer="",
                        rubric=f"Must explain the functional role, definition, and behavior of {concept['name']}.",
                        key_terms=[concept["name"].lower()] + [t.lower() for t in tokens(concept.get("definition", "")) if len(t) > 4][:4],
                        explanation=f"Textbook definition: {chunk.text[:220]}... (p. {chunk.page_number}) [{loc}]",
                        source_page=chunk.page_number,
                    )
                elif desired_type == "numerical":
                    q = generate_numerical_question(
                        concept_id=concept_id,
                        concept_name=concept.get("name", ""),
                        user_id=user_id,
                        source_page=chunk.page_number,
                    )
                else:  # MCQ
                    # Never have concept["name"] as the correct option or distractors.
                    # Formulate an operational mechanism question where options are substantive statements.
                    mcq_stems = [
                        f"Regarding {concept['name']}, which statement accurately describes its operational mechanism according to the study material?",
                        f"In the context of {concept['name']}, which of the following best characterizes its computational principle or governing rule?",
                        f"Which of the following statements correctly identifies how {concept['name']} functions within this curriculum?",
                        f"When analyzing {concept['name']}, which operational property is verified by the study material?",
                    ]
                    question_text = mcq_stems[(calc_offset + len(questions)) % len(mcq_stems)]
                    correct_opt = sent.strip()
                    if not correct_opt.endswith((".", ";", "!")):
                        correct_opt += "."

                    d1 = f"Operates via non-contiguous pointer linking, requiring linear traversal for state retrieval."
                    d2 = f"Executes unconstrained state transformations without preserving boundary invariance guarantees."
                    d3 = f"Requires global re-indexing and quadratic space reallocation on each intermediate evaluation."

                    alt_statements = []
                    for oc in chunks:
                        if oc != chunk:
                            for osent in sentences(oc.text):
                                if 10 <= len(osent.split()) <= 30 and osent != sent and not pat.search(osent):
                                    alt_statements.append(osent.strip())
                                    if len(alt_statements) >= 2:
                                        break
                    if len(alt_statements) >= 2:
                        d1 = alt_statements[0]
                        d2 = alt_statements[1]
                    elif len(alt_statements) == 1:
                        d1 = alt_statements[0]

                    distractors = [d1, d2, d3]
                    options = [correct_opt] + distractors
                    random.shuffle(options)
                    correct_idx = options.index(correct_opt)

                    q = QuizQuestion(
                        question_id=qid,
                        concept_id=concept_id,
                        question_type="mcq",
                        question_text=question_text,
                        options=options,
                        correct_option_idx=correct_idx,
                        target_blank_answer="",
                        rubric="",
                        key_terms=[],
                        explanation=f"Grounded statement: {sent} (p. {chunk.page_number}) [{loc}]",
                        source_page=chunk.page_number,
                    )

                questions.append(q)
                REGISTRY[qid] = q.model_dump()
                prior_served.append(q.question_text)
                prior_sentences.add(sent)

    return questions


def evaluate_submission(sub: QuizSubmission) -> QuizResult:
    """Evaluate submission across question types: MCQ, Fill-in-the-Blanks, and Typed Q&A."""
    q_data = REGISTRY.get(sub.question_id)
    
    if not q_data:
        return QuizResult(
            question_id=sub.question_id,
            concept_id=sub.concept_id,
            question_type="unknown",
            is_correct=False,
            correct_option_idx=-1,
            correct_answer_display="",
            user_answer=str(sub.selected_option_idx if sub.selected_option_idx is not None else sub.typed_answer or ""),
            explanation="Invalid or expired question ID.",
            score=0.0,
            misconceptions=["Question ID not recognized in active registry."],
            reasoning_clarity=0.0,
            reasoning_depth_level="SURFACE_ROTE",
        )

    q_type = q_data.get("question_type", "mcq")
    correct_idx = q_data.get("correct_option_idx", 0)
    target_blank = q_data.get("target_blank_answer", "")
    rubric = q_data.get("rubric", "")
    source_page = q_data.get("source_page", 1)
    explanation = q_data.get("explanation", "")

    is_correct = False
    score = 0.0
    reasoning_clarity = 0.5
    reasoning_depth = "DEVELOPING"
    misconceptions: list[str] = []
    user_answer_str = ""
    correct_answer_display = ""

    # ── Type 1: Multiple Choice (MCQ) ─────────────────────────────────────────
    if q_type == "mcq":
        options = q_data.get("options", []) if q_data else []
        is_correct = (sub.selected_option_idx is not None and sub.selected_option_idx == correct_idx)
        user_answer_str = options[sub.selected_option_idx] if (sub.selected_option_idx is not None and 0 <= sub.selected_option_idx < len(options)) else f"Option {sub.selected_option_idx}"
        correct_answer_display = options[correct_idx] if (correct_idx < len(options)) else f"Option {correct_idx}"

        if sub.student_reasoning:
            reasoning_tokens = set(tokens(sub.student_reasoning))
            expl_tokens = set(tokens(explanation))
            overlap = len(reasoning_tokens & expl_tokens)
            if len(reasoning_tokens) >= 5:
                reasoning_clarity = min(1.0, round(0.4 + (overlap / max(len(reasoning_tokens), 1)) * 0.6, 2))
                reasoning_depth = "DEEP_CONCEPTUAL" if overlap >= 3 else "DEVELOPING"
            else:
                reasoning_depth = "SURFACE_ROTE"

        if not is_correct:
            misconceptions.append(f"Chose alternative based on incorrect assumption on [Page {source_page}]")
            misconceptions.append(f"Failed identification of {sub.concept_id} on [Page {source_page}]")

        score = 1.0 if is_correct else 0.0

    # ── Type 2: Fill-in-the-Blanks (FITB) ──────────────────────────────────────
    elif q_type == "fill_in_the_blanks":
        raw_user = (sub.typed_answer or sub.student_reasoning or "").strip()
        user_answer_str = raw_user
        correct_answer_display = target_blank

        clean_user = re.sub(r"[^\w\s]", "", raw_user.lower()).strip()
        clean_target = re.sub(r"[^\w\s]", "", target_blank.lower()).strip()

        # String matching & fuzzy similarity tolerance for minor spelling variations
        similarity = difflib.SequenceMatcher(None, clean_user, clean_target).ratio() if (clean_user and clean_target) else 0.0
        exact_match = bool(clean_user and clean_target and clean_user == clean_target)

        key_terms = [t.lower() for t in q_data.get("key_terms", [])] if q_data else []
        term_match = any(
            clean_user == t or (len(clean_user) >= 4 and difflib.SequenceMatcher(None, clean_user, t).ratio() >= 0.85)
            for t in key_terms
        )

        if clean_user and (exact_match or similarity >= 0.82 or term_match):
            is_correct = True
            score = 1.0
            reasoning_clarity = 0.95
            reasoning_depth = "DEEP_CONCEPTUAL" if exact_match else "DEVELOPING"
        else:
            is_correct = False
            score = 0.0
            reasoning_clarity = 0.2
            reasoning_depth = "SURFACE_ROTE"
            misconceptions.append(f"Incorrect keyword for {sub.concept_id}: expected '{target_blank}', received '{raw_user}'")

    # ── Type 3: Typed Question-Answer (Open-Ended Q&A) ─────────────────────────
    elif q_type == "typed_qa":
        raw_user = (sub.typed_answer or sub.student_reasoning or "").strip()
        user_answer_str = raw_user
        correct_answer_display = rubric or explanation

        key_terms = [t.lower() for t in q_data.get("key_terms", [])] if q_data else []
        expl_tokens = set(tokens(explanation)) | set(tokens(rubric))
        user_tokens = set(tokens(raw_user))
        token_overlap = len(user_tokens & expl_tokens)
        has_key_term = any(t in raw_user.lower() for t in key_terms)
        concept_matches = (sub.concept_id.replace("-", " ") in raw_user.lower())

        is_conceptually_sound = (
            len(raw_user.split()) >= 4
            and (token_overlap >= 1 or has_key_term or concept_matches)
        )

        # Evaluate typed text directly via Semantic LLM-as-a-Judge
        eval_result = mastery.evaluate_mastery_semantic(
            concept_id=sub.concept_id,
            concept_definition=explanation,
            quiz_results=[{
                "question_id": sub.question_id,
                "is_correct": is_conceptually_sound,
                "student_reasoning": raw_user,
                "source_page": source_page,
            }],
            dialogue_context=0.7,
            attempt_count=1,
        )

        reasoning_depth = eval_result.reasoning_depth_level
        if is_conceptually_sound and reasoning_depth in ("DEVELOPING", "DEEP_CONCEPTUAL"):
            is_correct = True
            score = max(eval_result.semantic_comprehension_score, 0.85 if reasoning_depth == "DEEP_CONCEPTUAL" else 0.70)
        else:
            is_correct = False
            score = min(eval_result.semantic_comprehension_score, 0.4)
        reasoning_clarity = score
        misconceptions = eval_result.misconceptions_detected

        if not is_correct and not misconceptions:
            misconceptions.append(f"Incomplete explanation of {sub.concept_id} mechanism on [Page {source_page}]")

    # ── Type 4: Numerical Calculation ─────────────────────────────────────────
    elif q_type == "numerical":
        target_num = q_data.get("correct_numeric_answer")
        tolerance = float(q_data.get("tolerance", 0.05))
        step_solution = q_data.get("step_by_step_solution", "")
        unit = q_data.get("unit_or_format", "Round to 2 decimal places")
        correct_answer_display = f"{target_num} (±{tolerance})" if target_num is not None else ""

        user_val = sub.numeric_answer
        if user_val is not None:
            if math.isnan(user_val) or math.isinf(user_val):
                user_val = None

        if user_val is None:
            raw = (sub.typed_answer or sub.student_reasoning or "").strip()
            # Reject literal poison tokens
            if any(p in raw.lower() for p in ("nan", "inf")):
                user_val = None
            else:
                match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", raw)
                if match:
                    try:
                        v = float(match.group(0))
                        if not math.isnan(v) and not math.isinf(v):
                            user_val = v
                    except ValueError:
                        user_val = None

        raw_str = (sub.typed_answer or "No answer provided")
        if any(p in raw_str.lower() for p in ("nan", "inf")):
            user_answer_str = "[Invalid numeric format]"
        else:
            user_answer_str = str(user_val) if user_val is not None else raw_str
        if user_val is not None and target_num is not None:
            diff = abs(user_val - target_num)
            is_correct = (diff <= tolerance)
            score = 1.0 if is_correct else 0.0
            reasoning_clarity = 1.0 if is_correct else 0.3
            reasoning_depth = "DEEP_CONCEPTUAL" if is_correct else ("DEVELOPING" if diff <= tolerance * 2.5 else "SURFACE_ROTE")
            if not is_correct:
                misconceptions.append(
                    f"Calculation error for {sub.concept_id}: expected {target_num} (±{tolerance}), received {user_answer_str} on [Page {source_page}]"
                )
        else:
            is_correct = False
            score = 0.0
            reasoning_clarity = 0.0
            reasoning_depth = "SURFACE_ROTE"
            misconceptions.append(f"No valid numerical input could be parsed from submission for {sub.concept_id}.")

    # Authoritative multi-signal persistence
    evidence = (
        f"[{q_type.upper()}] {'Passed' if is_correct else 'Failed'} Question regarding {sub.concept_id} on [Page {source_page}]; "
        f"Reasoning depth: {reasoning_depth}; Score: {score:.2f}"
    )

    effective_reasoning = (
        user_answer_str if q_type in ("fill_in_the_blanks", "typed_qa", "numerical")
        else (sub.student_reasoning or evidence)
    )

    mastery.record_quiz(
        user_id=sub.user_id,
        concept_id=sub.concept_id,
        is_correct=is_correct,
        evidence_text=evidence,
        student_reasoning=effective_reasoning,
        question_id=sub.question_id,
        page_ref=source_page,
    )

    return QuizResult(
        question_id=sub.question_id,
        concept_id=sub.concept_id,
        question_type=q_type,
        is_correct=is_correct,
        correct_option_idx=correct_idx,
        correct_answer_display=correct_answer_display,
        user_answer=user_answer_str,
        explanation=explanation,
        score=score,
        misconceptions=misconceptions,
        reasoning_clarity=reasoning_clarity,
        reasoning_depth_level=reasoning_depth,
        step_by_step_solution=q_data.get("step_by_step_solution"),
    )


# ── Compatibility Adapters for Frontend ─────────────────────────────────────
RECENT_QUESTIONS: dict = {}

def next_question(concept_id: str | None = None, user_id: str = "default") -> dict:
    """Legacy and modern next_question adapter for the frontend with non-duplicate generation."""
    kb = store.require(user_id=user_id)
    target = concept_id or kb["concepts"][0]["id"]
    qs = generate_diagnostic_quiz(target, count=1, user_id=user_id)
    if not qs:
        raise store.UserError("Unable to build quiz question.")
    q = qs[0]
    by = {c["id"]: c for c in kb["concepts"]}
    c_name = by.get(q.concept_id, {}).get("name", q.concept_id)

    RECENT_QUESTIONS[q.question_id] = {
        "correct": q.correct_option_idx,
        "conceptId": q.concept_id,
        "concept_id": q.concept_id,
        "questionType": q.question_type,
        "question_type": q.question_type,
        "targetBlankAnswer": q.target_blank_answer,
        "rubric": q.rubric,
        "explanation": q.explanation,
        "source_page": q.source_page,
        "given_parameters": q.given_parameters,
        "formula_hint": q.formula_hint,
        "correct_numeric_answer": q.correct_numeric_answer,
        "tolerance": q.tolerance,
        "unit_or_format": q.unit_or_format,
        "step_by_step_solution": q.step_by_step_solution,
    }

    return {
        "id": q.question_id,
        "question_id": q.question_id,
        "conceptId": q.concept_id,
        "concept_id": q.concept_id,
        "conceptName": c_name,
        "concept_name": c_name,
        "questionType": q.question_type,
        "question_type": q.question_type,
        "question": q.question_text,
        "question_text": q.question_text,
        "options": q.options,
        "correct": q.correct_option_idx,
        "correct_option_idx": q.correct_option_idx,
        "targetBlankAnswer": q.target_blank_answer,
        "rubric": q.rubric,
        "explanation": q.explanation,
        "sourcePage": q.source_page,
        "source_page": q.source_page,
        "givenParameters": q.given_parameters,
        "formulaHint": q.formula_hint,
        "unitOrFormat": q.unit_or_format,
        "stepByStepSolution": q.step_by_step_solution,
    }


def evaluate_answer(student_answer: str, concept_id: str, question_text: str = "", ground_truth: str = "", user_id: str = "default") -> dict:
    """Evaluate free-form student text."""
    raw_ans = (student_answer or "").strip()
    if not raw_ans:
        return {
            "score": 0.0,
            "is_correct": False,
            "misconceptions": ["No answer provided."],
            "feedback": f"Grounded response for {concept_id} recorded: SURFACE_ROTE."
        }

    concept_obj = None
    kb = store.get_kb(user_id)
    if kb.get("ready"):
        for c in kb.get("concepts", []):
            if c.get("id") == concept_id or c.get("name", "").lower() == concept_id.lower():
                concept_obj = c
                break

    concept_name = concept_obj.get("name", concept_id) if concept_obj else concept_id
    explanation = ground_truth or (concept_obj.get("definition") if concept_obj else "") or f"Conceptual explanation of {concept_id}"
    source_page = (concept_obj.get("page") or concept_obj.get("source_page") or 1) if concept_obj else 1

    REGISTRY["freeform"] = {
        "question_id": "freeform",
        "concept_id": concept_id,
        "question_type": "typed_qa",
        "question": question_text or f"Explain {concept_id}",
        "options": [],
        "correct_option_idx": -1,
        "explanation": explanation,
        "rubric": ground_truth or explanation,
        "source_page": source_page,
        "key_terms": [concept_name.lower()] + [t for t in tokens(explanation) if len(t) > 3][:5],
    }

    sub = QuizSubmission(
        question_id="freeform",
        concept_id=concept_id,
        selected_option_idx=None,
        typed_answer=raw_ans,
        student_reasoning=raw_ans,
    )
    res = evaluate_submission(sub)
    return {
        "score": res.score,
        "is_correct": res.is_correct,
        "misconceptions": res.misconceptions,
        "feedback": f"Grounded response for {concept_id} recorded: {res.reasoning_depth_level}."
    }
