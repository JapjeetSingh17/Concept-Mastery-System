"""Diagnostic Assessment & Quiz Generation Engine.

Generates diagnostic multiple-choice questions strictly from vectorless chunks.
Evaluates both selected option choices and open-ended student reasoning.
Updates epistemic mastery records and pinpoints misconceptions.
"""

from __future__ import annotations

import itertools
import json
import random
import re
import uuid
from typing import List, Tuple

from ..core import store, llm
from ..core.concept_index import GLOBAL_INDEX
from ..core.text import sentences, tokens, normalize
from ..models.schemas import QuizQuestion, QuizSubmission, QuizResult
from . import mastery
from .concepts import pattern

_q_counter = itertools.count(1)
# Server-side authoritative question registry
REGISTRY: dict[str, dict] = {}


def generate_diagnostic_quiz(concept_id: str, count: int = 3) -> list[QuizQuestion]:
    """Generate diagnostic multiple-choice questions strictly from vectorless chunks for concept_id."""
    kb = store.require()
    by_id = {c["id"]: c for c in kb["concepts"]}
    concept = by_id.get(concept_id)
    if not concept:
        raise store.UserError(f"Concept '{concept_id}' not found.")

    # 1. Retrieve vectorless chunks for this concept
    chunks = GLOBAL_INDEX.retrieve_chunks(concept_id, max_chunks=3)
    if not chunks:
        # Fallback to KB index chunks if needed
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

    # 2. Try LLM generation if enabled
    if llm.enabled():
        ctx_text = "\n\n".join(f"[Page {c.page_number}] {c.text}" for c in chunks)
        system_prompt = (
            "You are a rigorous psychometric assessment designer. Generate diagnostic multiple-choice questions "
            "strictly grounded in the provided textbook context. Reply with a valid JSON array only."
        )
        user_prompt = (
            f"Concept to test: {concept['name']}\n"
            f"Textbook Context:\n{ctx_text}\n\n"
            f"Generate {count} diagnostic multiple-choice questions testing core principles and diagnosing common misconceptions.\n"
            f"JSON Schema:\n"
            f'[\n'
            f'  {{\n'
            f'    "question_text": "<clear question>",\n'
            f'    "options": ["<option 0>", "<option 1>", "<option 2>", "<option 3>"],\n'
            f'    "correct_option_idx": <0-3>,\n'
            f'    "explanation": "<why correct citing (p. X)>",\n'
            f'    "source_page": <int page number>\n'
            f'  }}\n'
            f']'
        )

        try:
            raw = llm.complete(system_prompt, user_prompt, max_tokens=1200)
            if raw:
                arr = json.loads(re.search(r"\[.*\]", raw, re.S).group(0))
                for item in arr:
                    if len(item.get("options", [])) >= 2 and 0 <= int(item.get("correct_option_idx", 0)) < len(item["options"]):
                        qid = f"diag_{uuid.uuid4().hex[:8]}"
                        q = QuizQuestion(
                            question_id=qid,
                            concept_id=concept_id,
                            question_text=str(item["question_text"]).strip(),
                            options=[str(opt).strip() for opt in item["options"]],
                            correct_option_idx=int(item["correct_option_idx"]),
                            explanation=str(item.get("explanation", f"Grounded in page {item.get('source_page', chunks[0].page_number)}")),
                            source_page=int(item.get("source_page", chunks[0].page_number)),
                        )
                        questions.append(q)
                        REGISTRY[qid] = q.model_dump()
                        if len(questions) >= count:
                            break
        except Exception:
            pass

    # 3. Deterministic vectorless cloze fallback if LLM produces fewer than requested
    if len(questions) < count:
        pat = pattern(concept)
        other_concepts = [c["name"] for c in kb["concepts"] if c["id"] != concept_id]
        distractor_pool = other_concepts if len(other_concepts) >= 3 else ["Optimization", "Overfitting", "Gradient", "Loss"]

        for chunk in chunks:
            if len(questions) >= count:
                break
            for sent in sentences(chunk.text):
                if pat.search(sent) and 10 <= len(sent.split()) <= 35:
                    qid = f"diag_{uuid.uuid4().hex[:8]}"
                    masked_text = pat.sub("_____", sent)
                    options = random.sample(distractor_pool, 3) + [concept["name"]]
                    random.shuffle(options)
                    correct_idx = options.index(concept["name"])

                    q = QuizQuestion(
                        question_id=qid,
                        concept_id=concept_id,
                        question_text=f"Which core concept completes this statement? \u201c{masked_text}\u201d",
                        options=options,
                        correct_option_idx=correct_idx,
                        explanation=f"{sent} (p. {chunk.page_number})",
                        source_page=chunk.page_number,
                    )
                    questions.append(q)
                    REGISTRY[qid] = q.model_dump()
                    if len(questions) >= count:
                        break

    return questions


def evaluate_submission(sub: QuizSubmission) -> QuizResult:
    """Evaluate both selected option and student reasoning for a quiz question."""
    q_data = REGISTRY.get(sub.question_id)
    if not q_data:
        # Fallback check
        is_correct = (sub.selected_option_idx == 0)
        correct_idx = 0
        explanation = "Assessment evaluation complete."
        source_page = 1
    else:
        correct_idx = q_data["correct_option_idx"]
        is_correct = (sub.selected_option_idx == correct_idx)
        explanation = q_data["explanation"]
        source_page = q_data.get("source_page", 1)

    # Evaluate student reasoning
    reasoning_clarity = 0.5
    misconceptions: list[str] = []

    if sub.student_reasoning:
        reasoning_tokens = set(tokens(sub.student_reasoning))
        expl_tokens = set(tokens(explanation))
        overlap = len(reasoning_tokens & expl_tokens)
        if len(reasoning_tokens) >= 5:
            reasoning_clarity = min(1.0, round(0.4 + (overlap / max(len(reasoning_tokens), 1)) * 0.6, 2))
        if not is_correct:
            misconceptions.append(f"Chose alternative based on incorrect assumption on Page {source_page}")

    if not is_correct:
        misconceptions.append(f"Failed identification of {sub.concept_id} on [Page {source_page}]")

    score = 1.0 if is_correct else 0.0
    evidence = (
        f"{'Passed' if is_correct else 'Failed'} Question regarding {sub.concept_id} on [Page {source_page}]; "
        f"Reasoning clarity: {int(reasoning_clarity * 100)}%"
    )

    # Authoritative multi-signal update: record both quiz correctness and student reasoning clarity
    if sub.student_reasoning and reasoning_clarity > 0.5:
        mastery.record_dialogue(sub.user_id, sub.concept_id, reasoning_clarity)

    mastery.record_quiz(
        user_id=sub.user_id,
        concept_id=sub.concept_id,
        is_correct=is_correct,
        evidence_text=evidence,
        student_reasoning=sub.student_reasoning,
        question_id=sub.question_id,
        page_ref=source_page,
    )

    return QuizResult(
        question_id=sub.question_id,
        concept_id=sub.concept_id,
        is_correct=is_correct,
        correct_option_idx=correct_idx,
        explanation=explanation,
        score=score,
        misconceptions=misconceptions,
        reasoning_clarity=reasoning_clarity,
    )


# Compatibility functions for existing frontend
RECENT_QUESTIONS = {}

def next_question(concept_id: str | None = None) -> dict:
    """Legacy next_question adapter for the frontend."""
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
        "explanation": q.explanation,
        "source_page": q.source_page,
    }

    return {
        "id": q.question_id,
        "conceptId": q.concept_id,
        "conceptName": c_name,
        "question": q.question_text,
        "options": q.options,
        "correct": q.correct_option_idx,
        "explanation": q.explanation,
        "sourcePage": q.source_page,
    }


def evaluate_answer(student_answer: str, concept_id: str, question_text: str = "", ground_truth: str = "") -> dict:
    """Evaluate free-form student text."""
    sub = QuizSubmission(
        question_id="freeform",
        concept_id=concept_id,
        selected_option_idx=0,
        student_reasoning=student_answer,
    )
    # Check concept presence
    c_pat = re.compile(re.escape(concept_id.replace("-", " ")), re.I)
    is_ok = bool(c_pat.search(student_answer)) or len(student_answer.split()) >= 8
    score = 0.85 if is_ok else 0.3
    return {
        "score": score,
        "is_correct": is_ok,
        "misconceptions": [] if is_ok else ["Answer lacked core concept terminology"],
        "feedback": f"Grounded response for {concept_id} recorded."
    }
