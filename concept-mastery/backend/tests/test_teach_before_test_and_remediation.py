"""Tests for Teach-Before-Test Micro-Lesson and Misconception Remediation."""

import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.pipelines import pedagogical_agents, quiz
from app.models.schemas import (
    QuizSubmission,
    TeachConceptResponse,
    QuestionRemediation,
)


@pytest.fixture
def client():
    return TestClient(app)


def test_generate_interactive_micro_lesson_pillars():
    """Verify that generate_interactive_micro_lesson outputs all 4 pedagogical pillars."""
    lesson = pedagogical_agents.generate_interactive_micro_lesson(
        concept_id="attention-mechanism",
        user_id="test_user",
    )
    assert lesson is not None
    assert lesson["concept_id"] == "attention-mechanism"
    assert lesson["concept_name"] != ""

    # Pillar 1: Big Picture Intuition
    assert len(lesson["big_picture_intuition"].strip()) > 10

    # Pillar 2: Deep Dive from Material (with citations)
    assert len(lesson["material_deep_dive"].strip()) > 10
    assert isinstance(lesson["citations"], list)

    # Pillar 3: Worked Example / Walkthrough
    assert len(lesson["worked_example"].strip()) > 10

    # Pillar 4: Common Pitfalls
    assert isinstance(lesson["common_pitfalls"], list)
    assert len(lesson["common_pitfalls"]) >= 1


def test_post_mastery_teach_concept_endpoint(client):
    """Verify POST /api/mastery/teach-concept endpoint returns 200 and formatted micro-lesson."""
    res = client.post(
        "/api/mastery/teach-concept",
        json={"concept_id": "gradient-descent", "user_id": "test_student"},
        headers={"X-User-Id": "test_student"},
    )
    assert res.status_code == 200
    data = res.json()
    assert data.get("ok") is True
    assert "lesson" in data
    lesson = data["lesson"]
    assert lesson["concept_id"] == "gradient-descent"
    assert "big_picture_intuition" in lesson
    assert "material_deep_dive" in lesson
    assert "worked_example" in lesson
    assert "common_pitfalls" in lesson


def test_generate_question_remediation_on_failure():
    """Verify that misconception diagnosis, grounded rule, missing step, and follow-up question are generated."""
    rem = pedagogical_agents.generate_question_remediation(
        question_id="test_q_101",
        concept_id="backpropagation",
        student_answer="It computes forward activation weights instead of gradients.",
        user_id="test_student",
    )
    assert rem is not None
    # Validate against Pydantic schema
    rem_obj = QuestionRemediation(**rem)
    assert len(rem_obj.misconception_diagnosis.strip()) > 5
    assert len(rem_obj.grounded_rule.strip()) > 5
    assert len(rem_obj.missing_step.strip()) > 5

    # Follow-up question verification
    if rem_obj.follow_up_question:
        fq = rem_obj.follow_up_question
        assert "question" in fq or "question_text" in fq
        assert "options" in fq
        assert len(fq["options"]) >= 2
        assert "correct_index" in fq or "correct_option_idx" in fq


def test_quiz_evaluation_includes_remediation_when_incorrect():
    """Verify that when a student's answer is evaluated as incorrect, remediation is attached."""
    # Using evaluate_answer directly
    eval_res = quiz.evaluate_answer(
        student_answer="Completely incorrect unrelated random text",
        concept_id="transformers",
        question_text="Explain how self-attention weights relate query and key vectors.",
        ground_truth="Self-attention computes dot products between query and key vectors scaled by sqrt(d_k).",
        user_id="test_student",
    )
    assert eval_res["is_correct"] is False
    assert "remediation" in eval_res
    rem = eval_res["remediation"]
    assert rem is not None
    assert "misconception_diagnosis" in rem
    assert "grounded_rule" in rem
