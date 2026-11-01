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
import random
import re
import uuid
from typing import List, Tuple, Literal

from ..core import store, llm
from ..core.concept_index import GLOBAL_INDEX
from ..core.text import sentences, tokens, normalize
from ..models.schemas import QuizQuestion, QuizSubmission, QuizResult
from . import mastery
from .concepts import pattern

_q_counter = itertools.count(1)
# Server-side authoritative question registry
REGISTRY: dict[str, dict] = {}


def generate_diagnostic_quiz(
    concept_id: str,
    count: int = 3,
    question_types: list[str] | None = None,
) -> list[QuizQuestion]:
    """Generate diagnostic assessment items strictly from vectorless chunks for concept_id.
    
    Produces a balanced mix across question types (MCQ, Fill-in-the-Blanks, and Typed Q&A).
    """
    kb = store.require()
    by_id = {c["id"]: c for c in kb["concepts"]}
    concept = by_id.get(concept_id)
    if not concept:
        raise store.UserError(f"Concept '{concept_id}' not found.")

    # Determine desired question types
    if question_types:
        target_types = question_types
    elif count == 1:
        # Default single question to MCQ for legacy compatibility
        target_types = ["mcq"]
    else:
        # Balanced rotation: mcq -> fill_in_the_blanks -> typed_qa
        cycle_types = ["mcq", "fill_in_the_blanks", "typed_qa"]
        target_types = [cycle_types[i % len(cycle_types)] for i in range(count)]

    # 1. Retrieve vectorless chunks for this concept
    chunks = GLOBAL_INDEX.retrieve_chunks(concept_id, max_chunks=3)
    if not chunks:
        all_chunks = kb["index"].chunks
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
        ctx_text = "\n\n".join(f"[Page {c.page_number}] {c.text}" for c in chunks)
        system_prompt = (
            "You are a psychometric assessment designer. Generate diverse diagnostic assessment items "
            "strictly grounded in the provided textbook context. Support three question types:\n"
            "  - 'mcq': Multiple-choice with 4 plausible options\n"
            "  - 'fill_in_the_blanks': A cloze sentence containing '_____' with 'target_blank_answer'\n"
            "  - 'typed_qa': An open-ended question testing conceptual mechanisms, with a 'rubric' and 'key_terms'\n"
            "Reply strictly with a valid JSON array only."
        )
        user_prompt = (
            f"Concept to test: {concept['name']}\n"
            f"Textbook Context:\n{ctx_text}\n\n"
            f"Generate {count} questions matching this balanced sequence of types: {target_types}.\n"
            f"JSON Schema:\n"
            f'[\n'
            f'  {{\n'
            f'    "question_type": "mcq" | "fill_in_the_blanks" | "typed_qa",\n'
            f'    "question_text": "<clear prompt or cloze sentence with _____>",\n'
            f'    "options": ["<option 0>", "<option 1>", "<option 2>", "<option 3>"] (only for mcq, [] otherwise),\n'
            f'    "correct_option_idx": <0-3> (only for mcq, 0 otherwise),\n'
            f'    "target_blank_answer": "<exact missing term>" (only for fill_in_the_blanks, "" otherwise),\n'
            f'    "rubric": "<evaluation criteria for typed_qa>" (only for typed_qa, "" otherwise),\n'
            f'    "key_terms": ["<term 1>", "<term 2>"] (only for typed_qa, [] otherwise),\n'
            f'    "explanation": "<grounded explanation citing (p. X)>",\n'
            f'    "source_page": <int page number>\n'
            f'  }}\n'
            f']'
        )

        try:
            raw = llm.complete(system_prompt, user_prompt, max_tokens=1400)
            if raw:
                clean_json = raw.strip()
                if "```json" in clean_json:
                    clean_json = clean_json.split("```json", 1)[1].split("```", 1)[0].strip()
                elif "```" in clean_json:
                    clean_json = clean_json.split("```", 1)[1].split("```", 1)[0].strip()
                
                arr = json.loads(re.search(r"\[.*\]", clean_json, re.S).group(0))
                for item in arr:
                    q_type = str(item.get("question_type", "mcq")).lower()
                    if q_type not in ("mcq", "fill_in_the_blanks", "typed_qa"):
                        q_type = "mcq"

                    qid = f"diag_{uuid.uuid4().hex[:8]}"
                    q = QuizQuestion(
                        question_id=qid,
                        concept_id=concept_id,
                        question_type=q_type,
                        question_text=str(item["question_text"]).strip(),
                        options=[str(opt).strip() for opt in item.get("options", [])],
                        correct_option_idx=int(item.get("correct_option_idx", 0)),
                        target_blank_answer=str(item.get("target_blank_answer", "")).strip(),
                        rubric=str(item.get("rubric", "")).strip(),
                        key_terms=[str(t).strip() for t in item.get("key_terms", [])],
                        explanation=str(item.get("explanation", f"Grounded in page {item.get('source_page', chunks[0].page_number)}")),
                        source_page=int(item.get("source_page", chunks[0].page_number)),
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

        for idx, chunk in enumerate(chunks):
            if len(questions) >= count:
                break
            
            chunk_sentences = sentences(chunk.text)
            matched_sentences = [s for s in chunk_sentences if pat.search(s) and 8 <= len(s.split()) <= 40]
            if not matched_sentences:
                matched_sentences = [chunk_sentences[0]] if chunk_sentences else [chunk.text[:120]]

            for sent in matched_sentences:
                if len(questions) >= count:
                    break

                desired_type = target_types[len(questions) % len(target_types)]
                qid = f"diag_{uuid.uuid4().hex[:8]}"

                if desired_type == "fill_in_the_blanks":
                    masked_text = pat.sub("_____", sent) if pat.search(sent) else f"{sent.rsplit(' ', 3)[0]} _____."
                    q = QuizQuestion(
                        question_id=qid,
                        concept_id=concept_id,
                        question_type="fill_in_the_blanks",
                        question_text=f"Fill in the blank: \"{masked_text}\"",
                        options=[],
                        correct_option_idx=0,
                        target_blank_answer=concept["name"],
                        rubric=f"Accepts '{concept['name']}' or exact canonical concept variations.",
                        key_terms=[concept["name"].lower()],
                        explanation=f"Correct keyword is '{concept['name']}'. Source sentence: {sent} (p. {chunk.page_number})",
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
                        explanation=f"Textbook definition: {chunk.text[:220]}... (p. {chunk.page_number})",
                        source_page=chunk.page_number,
                    )
                else:  # MCQ
                    masked_text = pat.sub("_____", sent) if pat.search(sent) else f"{sent.rsplit(' ', 2)[0]} _____."
                    options = random.sample(distractor_pool, 3) + [concept["name"]]
                    random.shuffle(options)
                    correct_idx = options.index(concept["name"])

                    q = QuizQuestion(
                        question_id=qid,
                        concept_id=concept_id,
                        question_type="mcq",
                        question_text=f"Which core concept completes this statement? \u201c{masked_text}\u201d",
                        options=options,
                        correct_option_idx=correct_idx,
                        target_blank_answer="",
                        rubric="",
                        key_terms=[],
                        explanation=f"{sent} (p. {chunk.page_number})",
                        source_page=chunk.page_number,
                    )

                questions.append(q)
                REGISTRY[qid] = q.model_dump()

    return questions


def evaluate_submission(sub: QuizSubmission) -> QuizResult:
    """Evaluate submission across question types: MCQ, Fill-in-the-Blanks, and Typed Q&A."""
    q_data = REGISTRY.get(sub.question_id)
    
    if not q_data:
        # Fallback check
        q_type = "mcq"
        correct_idx = 0
        target_blank = ""
        rubric = ""
        source_page = 1
        explanation = "Assessment evaluation complete."
    else:
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
        similarity = difflib.SequenceMatcher(None, clean_user, clean_target).ratio()
        exact_match = (clean_user == clean_target)
        contains_match = (clean_target in clean_user or clean_user in clean_target) and len(clean_user) >= 3 and len(clean_target) >= 3

        key_terms = [t.lower() for t in q_data.get("key_terms", [])] if q_data else []
        term_match = any(t in clean_user for t in key_terms)

        if exact_match or similarity >= 0.82 or contains_match or term_match:
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

        # Evaluate typed text directly via Semantic LLM-as-a-Judge
        eval_result = mastery.evaluate_mastery_semantic(
            concept_id=sub.concept_id,
            concept_definition=explanation,
            quiz_results=[{
                "question_id": sub.question_id,
                "is_correct": len(raw_user.split()) >= 4,
                "student_reasoning": raw_user,
                "source_page": source_page,
            }],
            dialogue_context=0.7,
            attempt_count=1,
        )

        reasoning_depth = eval_result.reasoning_depth_level
        is_correct = (reasoning_depth in ("DEVELOPING", "DEEP_CONCEPTUAL") and not eval_result.remedial_recommended)
        score = eval_result.semantic_comprehension_score
        reasoning_clarity = score
        misconceptions = eval_result.misconceptions_detected

        if not is_correct and not misconceptions:
            misconceptions.append(f"Incomplete explanation of {sub.concept_id} mechanism on [Page {source_page}]")

    # Authoritative multi-signal persistence
    evidence = (
        f"[{q_type.upper()}] {'Passed' if is_correct else 'Failed'} Question regarding {sub.concept_id} on [Page {source_page}]; "
        f"Reasoning depth: {reasoning_depth}; Score: {score:.2f}"
    )

    effective_reasoning = (
        user_answer_str if q_type in ("fill_in_the_blanks", "typed_qa")
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
    )


# ── Compatibility Adapters for Frontend ─────────────────────────────────────
RECENT_QUESTIONS: dict = {}

def next_question(concept_id: str | None = None) -> dict:
    """Legacy and modern next_question adapter for the frontend."""
    kb = store.require()
    target = concept_id or kb["concepts"][0]["id"]
    qs = generate_diagnostic_quiz(target, count=1)
    if not qs:
        raise store.UserError("Unable to build quiz question.")
    q = qs[0]
    by = {c["id"]: c for c in kb["concepts"]}
    c_name = by.get(q.concept_id, {}).get("name", q.concept_id)

    RECENT_QUESTIONS[q.question_id] = {
        "correct": q.correct_option_idx,
        "conceptId": q.concept_id,
        "questionType": q.question_type,
        "targetBlankAnswer": q.target_blank_answer,
        "rubric": q.rubric,
        "explanation": q.explanation,
        "source_page": q.source_page,
    }

    return {
        "id": q.question_id,
        "conceptId": q.concept_id,
        "conceptName": c_name,
        "questionType": q.question_type,
        "question": q.question_text,
        "options": q.options,
        "correct": q.correct_option_idx,
        "targetBlankAnswer": q.target_blank_answer,
        "rubric": q.rubric,
        "explanation": q.explanation,
        "sourcePage": q.source_page,
    }


def evaluate_answer(student_answer: str, concept_id: str, question_text: str = "", ground_truth: str = "") -> dict:
    """Evaluate free-form student text."""
    sub = QuizSubmission(
        question_id="freeform",
        concept_id=concept_id,
        selected_option_idx=0,
        typed_answer=student_answer,
        student_reasoning=student_answer,
    )
    res = evaluate_submission(sub)
    return {
        "score": res.score,
        "is_correct": res.is_correct,
        "misconceptions": res.misconceptions,
        "feedback": f"Grounded response for {concept_id} recorded: {res.reasoning_depth_level}."
    }
