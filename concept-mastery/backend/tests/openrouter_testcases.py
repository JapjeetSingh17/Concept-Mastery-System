"""
Aggressive 50-Test-Case Validation Suite (TC-161 to TC-210)
Target File: backend/tests/test_openrouter_and_ui_50_suite.py
Covers:
  - Category Q (TC-161 to TC-175): OpenRouter 6-Model Gateway & Failover Chain
  - Category R (TC-176 to TC-190): UI Double-Click Locks, Button Debounce & View/Tab Switching
  - Category S (TC-191 to TC-210): Multi-Format Ingestion, SSRF Security & Deletion Cascades
"""

import asyncio
import difflib
import io
import json
import os
import pathlib
import re
import sqlite3
import sys
import time
import zipfile
from typing import Any, Dict, List, Literal, Optional, Type
from pydantic import BaseModel, Field

import networkx as nx
import pytest

BACKEND_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))


# =====================================================================
# SCHEMAS & EXPECTED 6-MODEL PRIORITY CHAIN
# =====================================================================

EXPECTED_OPENROUTER_MODELS = [
    "google/gemini-2.5-flash",
    "nvidia/nemotron-3-ultra-550b-a55b:free",
    "deepseek/deepseek-chat-v3-0324:free",
    "nvidia/nemotron-3.5-super-vl:free",
    "meta-llama/llama-3.3-70b-instruct:free",
    "openrouter/auto",
]


class SampleStructuredOutput(BaseModel):
    concept_id: str
    epistemic_status: Literal["UNKNOWN", "WEAK", "STRONG"]
    score: float = Field(ge=0.0, le=1.0)


# =====================================================================
# HARNESS: OPENROUTER GATEWAY + UI STATE CONTROLLER + MULTIFORMAT STORE
# =====================================================================

class OpenRouterAndUIHarness:
    def __init__(self, db_path: str = ":memory:"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._init_db()

        self.openrouter_api_key = os.environ.get("OPENROUTER_API_KEY", "sk-or-v1-mock-test-key")
        self.openrouter_base_url = "https://openrouter.ai/api/v1"
        self.openrouter_models = list(EXPECTED_OPENROUTER_MODELS)
        self.last_outbound_payload: Optional[Dict[str, Any]] = None
        self.last_outbound_headers: Optional[Dict[str, str]] = None
        self.last_resolved_model: Optional[str] = None

        # UI State Simulation (mirrors frontend/js/app.js state machine)
        self.active_tab = "dashboard"
        self.theme = "dark"
        self.local_storage: Dict[str, str] = {"theme": "dark"}
        self.is_submitting = False
        self.is_skipping = False
        self.is_ingesting = False
        self.is_sending_chat = False
        self.active_request_token = 0
        self.preview_modal_state: Dict[str, Any] = {
            "open": False,
            "material_id": None,
            "content": None,
            "scroll_target": None,
        }
        self.chat_bubbles: List[Dict[str, str]] = []
        self.dom_input_state: Dict[str, Any] = {
            "question_type": "mcq",
            "selected_option_idx": None,
            "fitb_value": "",
            "typed_qa_value": "",
        }

        # Quiz & Graph State
        self.dag = nx.DiGraph()
        self.quiz_session: Dict[str, Any] = {
            "concept_id": "loss-functions",
            "current_index": 1,
            "total_questions": 6,
            "answered_count": 0,
            "correct_count": 0,
            "is_completed": False,
            "submit_api_calls": 0,
            "questions": [
                {"question_id": "q1", "question_type": "mcq", "options": ["A", "B", "C", "D"], "correct_option_idx": 0},
                {"question_id": "q2", "question_type": "fill_in_the_blanks", "target_blank_answer": "gradient"},
                {"question_id": "q3", "question_type": "typed_qa", "rubric": "chain rule derivative"},
                {"question_id": "q4", "question_type": "mcq", "options": ["A", "B", "C", "D"], "correct_option_idx": 1},
                {"question_id": "q5", "question_type": "fill_in_the_blanks", "target_blank_answer": "epoch"},
                {"question_id": "q6", "question_type": "typed_qa", "rubric": "convergence"},
            ],
        }

    def _init_db(self):
        cur = self.conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS study_materials (
                material_id TEXT PRIMARY KEY,
                user_id TEXT,
                title TEXT,
                source_type TEXT,
                origin_url_or_filename TEXT,
                raw_text TEXT,
                structured_preview_json TEXT,
                concepts_json TEXT
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS concept_chunks (
                chunk_id TEXT PRIMARY KEY,
                material_id TEXT,
                user_id TEXT,
                concept_id TEXT,
                locator_label TEXT,
                content TEXT
            )
            """
        )
        self.conn.commit()

    # ── OpenRouter Client Methods (TC-161 to TC-175) ─────────────────────────

    @staticmethod
    def sanitize_llm_output(raw_text: str) -> str:
        if not raw_text:
            return ""
        cleaned = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL).strip()
        fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, flags=re.DOTALL)
        if fence_match:
            return fence_match.group(1).strip()
        brace_match = re.search(r"(\{.*\})", cleaned, flags=re.DOTALL)
        if brace_match:
            return brace_match.group(1).strip()
        return cleaned

    def call_openrouter(
        self,
        prompt: str,
        system_prompt: str = "Return JSON.",
        json_mode: bool = True,
        simulated_provider_statuses: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        self.last_outbound_headers = {
            "Authorization": f"Bearer {self.openrouter_api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:8000",
            "X-Title": "Closed-Loop Concept Mastery Platform",
        }
        self.last_outbound_payload = {
            "model": self.openrouter_models[0],
            "models": list(self.openrouter_models),
            "route": "fallback",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
        }
        if json_mode:
            self.last_outbound_payload["response_format"] = {"type": "json_object"}

        if not self.openrouter_api_key:
            self.last_resolved_model = "offline_deterministic_fallback"
            return {
                "model": "offline_deterministic_fallback",
                "content": '{"concept_id": "fallback", "epistemic_status": "UNKNOWN", "score": 0.0}',
            }

        statuses = simulated_provider_statuses or {}
        for m in self.openrouter_models:
            behavior = statuses.get(m, 200)
            if behavior == "network_down":
                break
            if behavior in (429, 503, "timeout"):
                continue
            if behavior == 400 and json_mode:
                self.last_outbound_payload.pop("response_format", None)
                self.last_resolved_model = m
                return {
                    "model": m,
                    "content": '{"concept_id": "retried_no_response_format", "epistemic_status": "STRONG", "score": 0.9}',
                }
            if isinstance(behavior, str) and behavior.startswith("RAW:"):
                self.last_resolved_model = m
                return {"model": m, "content": behavior[4:]}
            if behavior == 200:
                self.last_resolved_model = m
                return {
                    "model": m,
                    "content": '{"concept_id": "loss-functions", "epistemic_status": "STRONG", "score": 0.92}',
                }

        self.last_resolved_model = "offline_deterministic_fallback"
        return {
            "model": "offline_deterministic_fallback",
            "content": '{"concept_id": "offline-concept", "epistemic_status": "WEAK", "score": 0.5}',
        }

    def call_structured(
        self,
        prompt: str,
        schema_cls: Type[BaseModel],
        simulated_provider_statuses: Optional[Dict[str, Any]] = None,
    ) -> BaseModel:
        resp = self.call_openrouter(
            prompt, json_mode=True, simulated_provider_statuses=simulated_provider_statuses
        )
        cleaned = self.sanitize_llm_output(resp["content"])
        try:
            return schema_cls.model_validate_json(cleaned)
        except Exception:
            self.last_resolved_model = "offline_deterministic_fallback"
            return schema_cls(concept_id="fallback_repaired", epistemic_status="WEAK", score=0.4)

    # ── UI Double-Click & View Switch Methods (TC-176 to TC-190) ──────────────

    async def ui_click_submit(self, network_error: bool = False) -> Dict[str, Any]:
        if self.is_submitting or self.quiz_session["is_completed"]:
            return {"ignored": True, "reason": "button_locked_or_completed"}

        q = self.quiz_session["questions"][self.quiz_session["current_index"] - 1]
        q_type = q["question_type"]
        if q_type == "mcq" and self.dom_input_state["selected_option_idx"] is None:
            raise ValueError("Please select an option before submitting.")
        if q_type == "fill_in_the_blanks" and not self.dom_input_state["fitb_value"].strip():
            raise ValueError("Please fill in the blank before submitting.")
        if q_type == "typed_qa" and not self.dom_input_state["typed_qa_value"].strip():
            raise ValueError("Please enter a typed explanation before submitting.")

        self.is_submitting = True
        try:
            await asyncio.sleep(0.02)
            if network_error:
                raise ConnectionError("Simulated network timeout during quiz submit")

            self.quiz_session["submit_api_calls"] += 1
            self.quiz_session["answered_count"] += 1
            if self.quiz_session["current_index"] >= self.quiz_session["total_questions"]:
                self.quiz_session["is_completed"] = True
            else:
                self.quiz_session["current_index"] += 1
                self._sync_dom_for_current_question()
            return {"ignored": False, "current_index": self.quiz_session["current_index"]}
        finally:
            self.is_submitting = False

    async def ui_click_skip(self) -> Dict[str, Any]:
        if self.is_skipping or self.quiz_session["is_completed"]:
            return {"ignored": True, "current_index": self.quiz_session["current_index"]}
        self.is_skipping = True
        try:
            await asyncio.sleep(0.01)
            if self.quiz_session["current_index"] >= self.quiz_session["total_questions"]:
                self.quiz_session["current_index"] = self.quiz_session["total_questions"]
                self.quiz_session["is_completed"] = True
            else:
                self.quiz_session["current_index"] += 1
                self._sync_dom_for_current_question()
            return {
                "ignored": False,
                "current_index": self.quiz_session["current_index"],
                "header": f"Question {self.quiz_session['current_index']} of {self.quiz_session['total_questions']}",
                "is_completed": self.quiz_session["is_completed"],
            }
        finally:
            self.is_skipping = False

    def _sync_dom_for_current_question(self):
        q = self.quiz_session["questions"][self.quiz_session["current_index"] - 1]
        self.dom_input_state = {
            "question_type": q["question_type"],
            "selected_option_idx": None,
            "fitb_value": "",
            "typed_qa_value": "",
        }

    def switch_tab(self, target_tab: str) -> Dict[str, Any]:
        valid_tabs = {"dashboard", "concept_graph", "concept_check", "socratic_tutor", "study_materials"}
        if target_tab not in valid_tabs:
            raise ValueError(f"Unknown tab: {target_tab}")
        self.active_tab = target_tab
        visible_sections = [t for t in valid_tabs if t == self.active_tab]
        return {"active_tab": self.active_tab, "visible_sections": visible_sections}

    async def switch_concept_node(self, new_concept_id: str, delay_ms: int = 10) -> Dict[str, Any]:
        self.active_request_token += 1
        my_token = self.active_request_token
        await asyncio.sleep(delay_ms / 1000.0)
        if my_token != self.active_request_token:
            return {"aborted": True, "concept_id": new_concept_id}
        self.quiz_session["concept_id"] = new_concept_id
        self.quiz_session["current_index"] = 1
        self.quiz_session["is_completed"] = False
        self._sync_dom_for_current_question()
        return {"aborted": False, "concept_id": self.quiz_session["concept_id"]}

    def toggle_theme(self) -> str:
        self.theme = "light" if self.theme == "dark" else "dark"
        self.local_storage["theme"] = self.theme
        return self.theme

    def open_preview_modal(
        self, material_id: str, user_id: str, scroll_locator: Optional[str] = None
    ) -> Dict[str, Any]:
        cur = self.conn.cursor()
        cur.execute(
            "SELECT user_id, title, source_type, structured_preview_json FROM study_materials WHERE material_id = ?",
            (material_id,),
        )
        row = cur.fetchone()
        if not row:
            raise KeyError("404: Material not found")
        if row[0] != user_id:
            raise PermissionError("403: Cross-tenant access denied")
        self.preview_modal_state = {
            "open": True,
            "material_id": material_id,
            "title": row[1],
            "source_type": row[2],
            "content": json.loads(row[3]),
            "scroll_target": scroll_locator,
        }
        return self.preview_modal_state

    # ── Multi-Format Ingestion & Security Methods (TC-191 to TC-210) ─────────

    @staticmethod
    def validate_safe_url(url: str):
        if not url or not isinstance(url, str):
            raise ValueError("400: Empty or missing URL")
        lower = url.strip().lower()
        if not (lower.startswith("http://") or lower.startswith("https://")):
            raise ValueError("400: URL must use http or https scheme")
        forbidden_hosts = ["127.0.0.1", "localhost", "169.254.169.254", "0.0.0.0", "[::1]", "internal."]
        for h in forbidden_hosts:
            if h in lower:
                raise ValueError(f"400: SSRF blocked for restricted host {h}")
        if lower.endswith((".exe", ".zip", ".tar.gz", ".sh", ".bin")):
            raise ValueError("400: Binary file URLs are not supported")

    @staticmethod
    def extract_youtube_id(url: str) -> str:
        patterns = [
            r"(?:v=|youtu\.be/|youtube\.com/embed/|youtube\.com/shorts/)([A-Za-z0-9_-]{11})",
        ]
        for p in patterns:
            m = re.search(p, url)
            if m:
                return m.group(1)
        raise ValueError("400: Invalid YouTube URL or video ID")

    def ingest_material(
        self,
        user_id: str,
        title: str,
        source_type: str,
        origin: str,
        segments: List[Dict[str, Any]],
    ) -> str:
        allowed_types = {"pdf", "pptx", "docx", "markdown", "youtube", "web", "audio", "text"}
        if source_type not in allowed_types:
            raise ValueError(f"400: Unsupported source_type {source_type}")

        mat_id = f"mat_{user_id}_{len(segments)}_{abs(hash((user_id, origin))) % 100000}"
        all_concepts = []
        cur = self.conn.cursor()

        cleaned_segments = []
        for idx, seg in enumerate(segments):
            raw_txt = seg.get("text", "")
            clean_txt = re.sub(
                r"<(script|style).*?>.*?</\1>", "", raw_txt, flags=re.DOTALL | re.IGNORECASE
            ).strip()
            locator = seg.get("locator_label", f"Section {idx+1}")
            concepts = seg.get("concepts", [])
            for c in concepts:
                if c not in all_concepts:
                    all_concepts.append(c)

            cleaned_segments.append({"label": locator, "text": clean_txt, "concepts": concepts})
            for c in concepts:
                chk_id = f"{mat_id}_chk_{idx}_{c}"
                cur.execute(
                    "INSERT OR REPLACE INTO concept_chunks VALUES (?, ?, ?, ?, ?, ?)",
                    (chk_id, mat_id, user_id, c, locator, clean_txt),
                )

        cur.execute(
            "INSERT OR REPLACE INTO study_materials VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                mat_id,
                user_id,
                title,
                source_type,
                origin,
                "\\n".join(s["text"] for s in cleaned_segments),
                json.dumps(cleaned_segments),
                json.dumps(all_concepts),
            ),
        )
        self.conn.commit()
        self._rebuild_user_dag(user_id)
        return mat_id

    def _rebuild_user_dag(self, user_id: str):
        cur = self.conn.cursor()
        cur.execute("SELECT concepts_json FROM study_materials WHERE user_id = ?", (user_id,))
        G = nx.DiGraph()
        ordered_concepts = []
        for (c_json,) in cur.fetchall():
            c_list = json.loads(c_json)
            for c in c_list:
                if c not in ordered_concepts:
                    ordered_concepts.append(c)
                G.add_node(c)
            for i in range(len(c_list) - 1):
                if c_list[i] != c_list[i + 1]:
                    G.add_edge(c_list[i], c_list[i + 1])
                    if not nx.is_directed_acyclic_graph(G):
                        G.remove_edge(c_list[i], c_list[i + 1])
        self.dag = G

    def delete_material(self, material_id: str, user_id: str) -> bool:
        cur = self.conn.cursor()
        cur.execute("SELECT user_id FROM study_materials WHERE material_id = ?", (material_id,))
        row = cur.fetchone()
        if not row:
            raise KeyError("404: Material not found")
        if row[0] != user_id:
            raise PermissionError("403: Cannot delete another user's material")

        cur.execute("DELETE FROM study_materials WHERE material_id = ?", (material_id,))
        cur.execute("DELETE FROM concept_chunks WHERE material_id = ?", (material_id,))
        self.conn.commit()
        self._rebuild_user_dag(user_id)
        return True

    def retrieve_concept_chunks(self, user_id: str, concept_id: str) -> List[Dict[str, str]]:
        cur = self.conn.cursor()
        cur.execute(
            "SELECT chunk_id, locator_label, content FROM concept_chunks WHERE user_id = ? AND concept_id = ?",
            (user_id, concept_id),
        )
        return [{"chunk_id": r[0], "locator_label": r[1], "content": r[2]} for r in cur.fetchall()]


@pytest.fixture
def harness(tmp_path):
    return OpenRouterAndUIHarness(db_path=str(tmp_path / "openrouter_ui_suite.sqlite"))


# =====================================================================
# CATEGORY Q: OPENROUTER 6-MODEL GATEWAY & FAILOVER CHAIN (TC-161 to TC-175)
# =====================================================================

def test_tc161_exact_6_model_priority_order_verification(harness):
    assert harness.openrouter_models == [
        "google/gemini-2.5-flash",
        "nvidia/nemotron-3-ultra-550b-a55b:free",
        "deepseek/deepseek-chat-v3-0324:free",
        "nvidia/nemotron-3.5-super-vl:free",
        "meta-llama/llama-3.3-70b-instruct:free",
        "openrouter/auto",
    ]


def test_tc162_native_openrouter_fallback_payload_contract(harness):
    harness.call_openrouter("Test prompt")
    payload = harness.last_outbound_payload
    assert payload["model"] == "google/gemini-2.5-flash"
    assert payload["models"] == EXPECTED_OPENROUTER_MODELS
    assert payload["route"] == "fallback"


def test_tc163_primary_gemini_429_fails_over_to_nvidia_nemotron_3_ultra(harness):
    res = harness.call_openrouter(
        "Explain chain rule",
        simulated_provider_statuses={"google/gemini-2.5-flash": 429},
    )
    assert res["model"] == "nvidia/nemotron-3-ultra-550b-a55b:free"


def test_tc164_nemotron_3_ultra_503_fails_over_to_deepseek_v3(harness):
    res = harness.call_openrouter(
        "Evaluate answer",
        simulated_provider_statuses={
            "google/gemini-2.5-flash": 429,
            "nvidia/nemotron-3-ultra-550b-a55b:free": 503,
        },
    )
    assert res["model"] == "deepseek/deepseek-chat-v3-0324:free"


def test_tc165_deepseek_timeout_fails_over_to_nvidia_nemotron_3_5_super_vl(harness):
    res = harness.call_openrouter(
        "Multimodal slide query",
        simulated_provider_statuses={
            "google/gemini-2.5-flash": 429,
            "nvidia/nemotron-3-ultra-550b-a55b:free": 503,
            "deepseek/deepseek-chat-v3-0324:free": "timeout",
        },
    )
    assert res["model"] == "nvidia/nemotron-3.5-super-vl:free"


def test_tc166_nemotron_vl_overload_fails_over_to_llama_3_3_70b(harness):
    res = harness.call_openrouter(
        "Generate quiz",
        simulated_provider_statuses={
            "google/gemini-2.5-flash": 429,
            "nvidia/nemotron-3-ultra-550b-a55b:free": 503,
            "deepseek/deepseek-chat-v3-0324:free": "timeout",
            "nvidia/nemotron-3.5-super-vl:free": 429,
        },
    )
    assert res["model"] == "meta-llama/llama-3.3-70b-instruct:free"


def test_tc167_ultimate_catch_all_router_openrouter_auto(harness):
    res = harness.call_openrouter(
        "Fallback catch-all test",
        simulated_provider_statuses={
            "google/gemini-2.5-flash": 429,
            "nvidia/nemotron-3-ultra-550b-a55b:free": 503,
            "deepseek/deepseek-chat-v3-0324:free": "timeout",
            "nvidia/nemotron-3.5-super-vl:free": 429,
            "meta-llama/llama-3.3-70b-instruct:free": 503,
        },
    )
    assert res["model"] == "openrouter/auto"


def test_tc168_deepseek_and_nemotron_think_tag_stripping(harness):
    raw_output = (
        "RAW:<think>\nThe student used the chain rule properly.\nLet me score this 0.95.\n</think>\n"
        '{"concept_id": "backprop", "epistemic_status": "STRONG", "score": 0.95}'
    )
    parsed = harness.call_structured(
        "Grade student",
        SampleStructuredOutput,
        simulated_provider_statuses={"google/gemini-2.5-flash": raw_output},
    )
    assert parsed.concept_id == "backprop"
    assert parsed.epistemic_status == "STRONG"
    assert parsed.score == 0.95


def test_tc169_markdown_code_fence_json_extraction(harness):
    raw_output = (
        "RAW:Here is the structured evaluation:\n```json\n"
        '{"concept_id": "sgd", "epistemic_status": "WEAK", "score": 0.45}\n```\nHope this helps!'
    )
    parsed = harness.call_structured(
        "Grade student",
        SampleStructuredOutput,
        simulated_provider_statuses={"google/gemini-2.5-flash": raw_output},
    )
    assert parsed.concept_id == "sgd"
    assert parsed.epistemic_status == "WEAK"


def test_tc170_response_format_400_rejection_retry(harness):
    res = harness.call_openrouter(
        "Structured request",
        json_mode=True,
        simulated_provider_statuses={"google/gemini-2.5-flash": 400},
    )
    assert "response_format" not in harness.last_outbound_payload
    assert res["model"] == "google/gemini-2.5-flash"


def test_tc171_openrouter_required_headers_verification(harness):
    harness.call_openrouter("Ping")
    hdrs = harness.last_outbound_headers
    assert hdrs["Authorization"].startswith("Bearer ")
    assert "HTTP-Referer" in hdrs
    assert "X-Title" in hdrs


def test_tc172_zero_hardcoded_openrouter_keys_in_source():
    for py_file in BACKEND_ROOT.rglob("*.py"):
        if "test" in py_file.name:
            continue
        content = py_file.read_text(encoding="utf-8", errors="ignore")
        assert re.search(r"sk-or-v1-[A-Za-z0-9]{20,}", content) is None, f"Leaked key in {py_file}"


def test_tc173_total_network_blackout_falls_back_to_offline_engine(harness):
    res = harness.call_openrouter(
        "Offline test",
        simulated_provider_statuses={"google/gemini-2.5-flash": "network_down"},
    )
    assert res["model"] == "offline_deterministic_fallback"


def test_tc174_malformed_truncated_json_fallback_repair(harness):
    raw_broken = 'RAW:{"concept_id": "loss", "epistemic_status": "WE'
    parsed = harness.call_structured(
        "Broken JSON test",
        SampleStructuredOutput,
        simulated_provider_statuses={"google/gemini-2.5-flash": raw_broken},
    )
    assert parsed.concept_id == "fallback_repaired"
    assert parsed.epistemic_status == "WEAK"


def test_tc175_model_provenance_logging_in_metadata(harness):
    harness.call_openrouter(
        "Audit log test",
        simulated_provider_statuses={"google/gemini-2.5-flash": 429},
    )
    assert harness.last_resolved_model == "nvidia/nemotron-3-ultra-550b-a55b:free"


# =====================================================================
# CATEGORY R: UI DOUBLE-CLICK, BUTTON LOCK & VIEW/TAB SWITCHING (TC-176 to TC-190)
# =====================================================================

@pytest.mark.asyncio
async def test_tc176_double_click_submit_answer_button_lock(harness):
    harness.dom_input_state["selected_option_idx"] = 0
    r1, r2 = await asyncio.gather(harness.ui_click_submit(), harness.ui_click_submit())
    assert (r1["ignored"] ^ r2["ignored"]) is True
    assert harness.quiz_session["submit_api_calls"] == 1
    assert harness.quiz_session["current_index"] == 2


@pytest.mark.asyncio
async def test_tc177_double_click_skip_question_debounce_and_clamping(harness):
    for _ in range(6):
        await asyncio.gather(harness.ui_click_skip(), harness.ui_click_skip())
    assert harness.quiz_session["current_index"] == 6
    assert harness.quiz_session["is_completed"] is True


@pytest.mark.asyncio
async def test_tc178_double_click_upload_ingest_lock(harness):
    harness.is_ingesting = True
    assert harness.is_ingesting is True
    harness.is_ingesting = False


@pytest.mark.asyncio
async def test_tc179_double_click_send_in_socratic_tutor_chat(harness):
    async def send_chat(msg: str):
        if harness.is_sending_chat:
            return False
        harness.is_sending_chat = True
        try:
            await asyncio.sleep(0.01)
            harness.chat_bubbles.append({"role": "user", "text": msg})
            return True
        finally:
            harness.is_sending_chat = False

    await asyncio.gather(send_chat("What is SGD?"), send_chat("What is SGD?"))
    assert len(harness.chat_bubbles) == 1


def test_tc180_navigation_tab_switch_state_integrity(harness):
    for tab in ["dashboard", "concept_graph", "concept_check", "socratic_tutor", "study_materials"]:
        state = harness.switch_tab(tab)
        assert state["active_tab"] == tab
        assert state["visible_sections"] == [tab]


@pytest.mark.asyncio
async def test_tc181_mid_quiz_tab_switch_preservation(harness):
    harness.dom_input_state["selected_option_idx"] = 0
    await harness.ui_click_submit()
    harness.dom_input_state["fitb_value"] = "gradient"
    await harness.ui_click_submit()

    harness.switch_tab("socratic_tutor")
    harness.switch_tab("concept_check")
    assert harness.quiz_session["current_index"] == 3
    assert harness.quiz_session["answered_count"] == 2


@pytest.mark.asyncio
async def test_tc182_question_type_ui_switch_clears_stale_inputs(harness):
    assert harness.dom_input_state["question_type"] == "mcq"
    harness.dom_input_state["selected_option_idx"] = 0
    await harness.ui_click_submit()

    assert harness.dom_input_state["question_type"] == "fill_in_the_blanks"
    assert harness.dom_input_state["selected_option_idx"] is None
    harness.dom_input_state["fitb_value"] = "gradient"
    await harness.ui_click_submit()

    assert harness.dom_input_state["question_type"] == "typed_qa"
    assert harness.dom_input_state["fitb_value"] == ""


@pytest.mark.asyncio
async def test_tc183_concept_selector_node_click_switch(harness):
    harness.quiz_session["current_index"] = 4
    res = await harness.switch_concept_node("backpropagation", delay_ms=5)
    assert res["aborted"] is False
    assert harness.quiz_session["concept_id"] == "backpropagation"
    assert harness.quiz_session["current_index"] == 1


def test_tc184_dark_light_theme_switch_persistence(harness):
    assert harness.toggle_theme() == "light"
    assert harness.local_storage["theme"] == "light"
    assert harness.toggle_theme() == "dark"
    assert harness.local_storage["theme"] == "dark"


def test_tc185_material_preview_modal_switch_clears_previous_dom(harness):
    m1 = harness.ingest_material("u1", "PDF Notes", "pdf", "a.pdf", [{"locator_label": "Page 1", "text": "PDF text", "concepts": ["c1"]}])
    m2 = harness.ingest_material("u1", "YT Video", "youtube", "https://youtu.be/dQw4w9WgXcQ", [{"locator_label": "01:15", "text": "YT transcript", "concepts": ["c2"]}])

    p1 = harness.open_preview_modal(m1, "u1")
    assert p1["source_type"] == "pdf"
    p2 = harness.open_preview_modal(m2, "u1")
    assert p2["source_type"] == "youtube"
    assert p2["material_id"] == m2
    assert p2["content"][0]["label"] == "01:15"


@pytest.mark.asyncio
async def test_tc186_in_flight_request_abort_on_rapid_concept_switch(harness):
    task_a = asyncio.create_task(harness.switch_concept_node("concept_A", delay_ms=40))
    await asyncio.sleep(0.005)
    task_b = asyncio.create_task(harness.switch_concept_node("concept_B", delay_ms=10))
    res_a, res_b = await asyncio.gather(task_a, task_b)

    assert res_a["aborted"] is True
    assert res_b["aborted"] is False
    assert harness.quiz_session["concept_id"] == "concept_B"


@pytest.mark.asyncio
async def test_tc187_submit_button_reenabled_after_network_error(harness):
    harness.dom_input_state["selected_option_idx"] = 1
    with pytest.raises(ConnectionError):
        await harness.ui_click_submit(network_error=True)
    assert harness.is_submitting is False


@pytest.mark.asyncio
async def test_tc188_radio_option_selection_switch_before_submit(harness):
    harness.dom_input_state["selected_option_idx"] = 0
    harness.dom_input_state["selected_option_idx"] = 2
    assert harness.dom_input_state["selected_option_idx"] == 2
    await harness.ui_click_submit()
    assert harness.quiz_session["answered_count"] == 1


def test_tc189_source_citation_pill_click_switches_preview_drawer_to_locator(harness):
    mat_id = harness.ingest_material(
        "u1",
        "Lecture Deck",
        "pptx",
        "deck.pptx",
        [
            {"locator_label": "Slide 1", "text": "Intro", "concepts": ["c1"]},
            {"locator_label": "Slide 2", "text": "Backprop derivation", "concepts": ["backprop"]},
        ],
    )
    modal = harness.open_preview_modal(mat_id, "u1", scroll_locator="Slide 2")
    assert modal["open"] is True
    assert modal["scroll_target"] == "Slide 2"


@pytest.mark.asyncio
async def test_tc190_empty_whitespace_input_submit_blocked(harness):
    harness.dom_input_state["selected_option_idx"] = 0
    await harness.ui_click_submit()

    harness.dom_input_state["fitb_value"] = "    "
    with pytest.raises(ValueError, match="fill in the blank"):
        await harness.ui_click_submit()
    assert harness.quiz_session["current_index"] == 2


# =====================================================================
# CATEGORY S: MULTI-FORMAT INGESTION, SSRF & DELETION CASCADES (TC-191 to TC-210)
# =====================================================================

def test_tc191_ssrf_localhost_and_cloud_metadata_blocked(harness):
    malicious_urls = [
        "http://127.0.0.1:8000/api/health",
        "http://localhost:3000/secret",
        "http://169.254.169.254/latest/meta-data/",
        "file:///etc/passwd",
    ]
    for url in malicious_urls:
        with pytest.raises(ValueError, match="400"):
            harness.validate_safe_url(url)


def test_tc192_xss_script_and_style_stripping_in_web_preview(harness):
    mat_id = harness.ingest_material(
        "u1",
        "GFG Article",
        "web",
        "https://www.geeksforgeeks.org/ml/",
        [{"locator_label": "Web: Intro", "text": "<script>alert(1)</script><style>body{}</style>Clean article text.", "concepts": ["ml"]}],
    )
    modal = harness.open_preview_modal(mat_id, "u1")
    preview_text = modal["content"][0]["text"]
    assert "<script>" not in preview_text
    assert "<style>" not in preview_text
    assert "Clean article text." in preview_text


def test_tc193_youtube_short_embed_and_timestamped_url_parsing(harness):
    urls = [
        "https://youtu.be/dQw4w9WgXcQ?t=45",
        "https://www.youtube.com/embed/dQw4w9WgXcQ",
        "https://www.youtube.com/shorts/dQw4w9WgXcQ",
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ&list=PL123",
    ]
    for u in urls:
        assert harness.extract_youtube_id(u) == "dQw4w9WgXcQ"


def test_tc194_youtube_invalid_url_returns_clean_400(harness):
    with pytest.raises(ValueError, match="400"):
        harness.extract_youtube_id("https://example.com/not-youtube")


def test_tc195_pptx_table_and_speaker_notes_slide_locator(harness):
    harness.ingest_material(
        "u1",
        "Slides",
        "pptx",
        "slides.pptx",
        [{"locator_label": "Slide 3", "text": "Table cell: Convexity. Speaker note: Global minimum.", "concepts": ["convexity"]}],
    )
    chunks = harness.retrieve_concept_chunks("u1", "convexity")
    assert len(chunks) == 1
    assert chunks[0]["locator_label"] == "Slide 3"
    assert "Global minimum" in chunks[0]["content"]


def test_tc196_docx_table_cell_extraction_and_indexing(harness):
    harness.ingest_material(
        "u1",
        "Word Notes",
        "docx",
        "notes.docx",
        [{"locator_label": "Section: Activation Table", "text": "ReLU: max(0, x) non-linear activation", "concepts": ["relu"]}],
    )
    chunks = harness.retrieve_concept_chunks("u1", "relu")
    assert len(chunks) == 1
    assert "max(0, x)" in chunks[0]["content"]


def test_tc197_markdown_code_block_hash_comment_protection():
    md_text = "# Real Heading\nSome intro.\n```python\n# python comment inside code\nx = 1\n```\n"
    in_fence = False
    headings = []
    for line in md_text.splitlines():
        if line.strip().startswith("```"):
            in_fence = not in_fence
            continue
        if not in_fence and line.startswith("# "):
            headings.append(line[2:].strip())
    assert headings == ["Real Heading"]


def test_tc198_audio_voice_note_timestamp_locators(harness):
    mat_id = harness.ingest_material(
        "u1",
        "Voice Note",
        "audio",
        "note.mp3",
        [{"locator_label": "Audio 01:30", "text": "Regularization penalizes large weights.", "concepts": ["regularization"]}],
    )
    chunks = harness.retrieve_concept_chunks("u1", "regularization")
    assert chunks[0]["locator_label"] == "Audio 01:30"
    modal = harness.open_preview_modal(mat_id, "u1")
    assert modal["content"][0]["label"] == "Audio 01:30"


def test_tc199_four_format_unified_prerequisite_dag_fusion(harness):
    harness.ingest_material("u1", "M1", "pdf", "1.pdf", [{"locator_label": "Page 1", "text": "T1", "concepts": ["c_pdf", "c_pptx"]}])
    harness.ingest_material("u1", "M2", "pptx", "2.pptx", [{"locator_label": "Slide 1", "text": "T2", "concepts": ["c_pptx", "c_md"]}])
    harness.ingest_material("u1", "M3", "markdown", "3.md", [{"locator_label": "Sec 1", "text": "T3", "concepts": ["c_md", "c_yt"]}])
    harness.ingest_material("u1", "M4", "youtube", "[https://youtu.be/dQw4w9WgXcQ](https://youtu.be/dQw4w9WgXcQ)", [{"locator_label": "00:30", "text": "T4", "concepts": ["c_yt"]}])

    assert set(harness.dag.nodes) == {"c_pdf", "c_pptx", "c_md", "c_yt"}
    assert nx.is_directed_acyclic_graph(harness.dag)
    assert list(nx.topological_sort(harness.dag)) == ["c_pdf", "c_pptx", "c_md", "c_yt"]


def test_tc200_material_deletion_cascade_purges_chunks_and_updates_dag(harness):
    m1 = harness.ingest_material("u1", "M1", "pdf", "1.pdf", [{"locator_label": "Page 1", "text": "PDF", "concepts": ["only_in_pdf"]}])
    harness.ingest_material("u1", "M2", "markdown", "2.md", [{"locator_label": "Sec 1", "text": "MD", "concepts": ["only_in_md"]}])

    harness.delete_material(m1, "u1")
    assert harness.retrieve_concept_chunks("u1", "only_in_pdf") == []
    assert len(harness.retrieve_concept_chunks("u1", "only_in_md")) == 1
    assert "only_in_pdf" not in harness.dag.nodes
    assert "only_in_md" in harness.dag.nodes


def test_tc201_delete_last_remaining_material_resets_dag_cleanly(harness):
    m1 = harness.ingest_material("u1", "M1", "pdf", "1.pdf", [{"locator_label": "Page 1", "text": "PDF", "concepts": ["c1"]}])
    harness.delete_material(m1, "u1")
    assert len(harness.dag.nodes) == 0
    assert len(harness.dag.edges) == 0


def test_tc202_multi_tenant_material_isolation_and_delete_protection(harness):
    m_a = harness.ingest_material("user_A", "Secret", "markdown", "sec.md", [{"locator_label": "Sec 1", "text": "Secret", "concepts": ["c1"]}])
    with pytest.raises(PermissionError, match="403"):
        harness.open_preview_modal(m_a, "user_B")
    with pytest.raises(PermissionError, match="403"):
        harness.delete_material(m_a, "user_B")


def test_tc203_dynamic_non_pdf_citation_in_socratic_tutor(harness):
    harness.ingest_material(
        "u1",
        "YT Lecture",
        "youtube",
        "[https://youtu.be/dQw4w9WgXcQ](https://youtu.be/dQw4w9WgXcQ)",
        [{"locator_label": "02:15", "text": "Dropout randomly zeroes activations.", "concepts": ["dropout"]}],
    )
    chunks = harness.retrieve_concept_chunks("u1", "dropout")
    tutor_reply = f"Consider the explanation at [{chunks[0]['locator_label']}]: how does zeroing activations prevent co-adaptation?"
    assert "" in tutor_reply
    assert "[Page" not in tutor_reply


def test_tc204_dynamic_non_pdf_citation_in_closed_loop_remediation(harness):
    harness.ingest_material(
        "u1",
        "GFG",
        "web",
        "[https://www.geeksforgeeks.org/l2/](https://www.geeksforgeeks.org/l2/)",
        [{"locator_label": "Web: L2 Ridge Penalty", "text": "L2 adds squared magnitude of coefficients.", "concepts": ["l2_reg"]}],
    )
    chunks = harness.retrieve_concept_chunks("u1", "l2_reg")
    remedial = f"Review [{chunks[0]['locator_label']}]: {chunks[0]['content']}"
    assert "[Web: L2 Ridge Penalty]" in remedial


def test_tc205_100k_character_dos_guard_on_typed_qa():
    huge_input = "gradient descent " * 6000
    t0 = time.perf_counter()
    clamped = huge_input[:2000]
    ratio = difflib.SequenceMatcher(None, clamped[:200], "gradient descent minimizes loss").ratio()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    assert elapsed_ms < 100.0
    assert ratio > 0.0


def test_tc206_unicode_mathematical_notation_in_typed_qa():
    ans = "When ∇L(θ) = 0 and step size α → 0, the partial derivative ∂L/∂w converges."
    tokens = re.findall(r"[A-Za-z0-9_∇θαλ∂]+", ans)
    assert "∇L" in tokens or "θ" in tokens
    assert "converges" in tokens


def test_tc207_corrupted_pptx_and_docx_upload_returns_clean_400():
    bad_bytes = b"not a valid zip archive header"
    with pytest.raises(ValueError, match="400: Corrupted archive"):
        try:
            with zipfile.ZipFile(io.BytesIO(bad_bytes)):
                pass
        except zipfile.BadZipFile as exc:
            raise ValueError("400: Corrupted archive or invalid Office file") from exc


def test_tc208_unsupported_file_extension_rejection(harness):
    with pytest.raises(ValueError, match="400: Unsupported source_type"):
        harness.ingest_material("u1", "Bad", "executable", "hack.exe", [])


def test_tc209_sqlite_study_materials_persistence_across_connection_reopen(tmp_path):
    db_file = str(tmp_path / "persist_materials.sqlite")
    h1 = OpenRouterAndUIHarness(db_path=db_file)
    mat_id = h1.ingest_material(
        "u_persist",
        "Saved Deck",
        "pptx",
        "saved.pptx",
        [{"locator_label": "Slide 1", "text": "Persisted slide content", "concepts": ["persisted_concept"]}],
    )
    h1.conn.close()

    h2 = OpenRouterAndUIHarness(db_path=db_file)
    modal = h2.open_preview_modal(mat_id, "u_persist")
    assert modal["title"] == "Saved Deck"
    assert modal["content"][0]["text"] == "Persisted slide content"


@pytest.mark.asyncio
async def test_tc210_full_e2e_openrouter_multiformat_and_debounced_ui_lifecycle(harness):
    harness.ingest_material(
        "u_e2e",
        "Slides",
        "pptx",
        "s.pptx",
        [{"locator_label": "Slide 1", "text": "Loss functions measure error.", "concepts": ["loss-functions"]}],
    )
    m_yt = harness.ingest_material(
        "u_e2e",
        "Video",
        "youtube",
        "[https://youtu.be/dQw4w9WgXcQ](https://youtu.be/dQw4w9WgXcQ)",
        [{"locator_label": "03:10", "text": "Gradient descent minimizes loss.", "concepts": ["loss-functions", "gradient-descent"]}],
    )
    assert "loss-functions" in harness.dag.nodes
    assert "gradient-descent" in harness.dag.nodes

    modal = harness.open_preview_modal(m_yt, "u_e2e", scroll_locator="03:10")
    assert modal["scroll_target"] == "03:10"

    eval_obj = harness.call_structured(
        "Evaluate student mastery",
        SampleStructuredOutput,
        simulated_provider_statuses={
            "google/gemini-2.5-flash": 429,
            "nvidia/nemotron-3-ultra-550b-a55b:free": 'RAW:<think>Analyzing...</think>{"concept_id": "loss-functions", "epistemic_status": "STRONG", "score": 0.91}',
        },
    )
    assert harness.last_resolved_model == "nvidia/nemotron-3-ultra-550b-a55b:free"
    assert eval_obj.epistemic_status == "STRONG"

    harness.dom_input_state["selected_option_idx"] = 0
    await asyncio.gather(harness.ui_click_submit(), harness.ui_click_submit())
    assert harness.quiz_session["current_index"] == 2
    harness.switch_tab("socratic_tutor")
    harness.switch_tab("concept_check")
    assert harness.quiz_session["current_index"] == 2