"""
final_100_test_cases.py
=======================
100 test cases for the Concept Mastery System (FastAPI + LangGraph + SQLite BM25 + Beta/epistemic mastery).

HOW TO RUN
----------
    pip install fastapi httpx rapidfuzz networkx langgraph pypdf pytest python-multipart pydantic
    cd concept-mastery/backend
    pytest tests/final_100_test_cases.py -v        # put this file in backend/tests/ (or anywhere under the repo)
    pytest final_100_test_cases.py -v -rx          # -rx also lists the known-bug xfails

IMPORTANT: run this file ON ITS OWN (not together with the older aggressive suites). It redirects
UPLOAD_DIR / VECTOR_STORE_PATH to a throw-away temp dir and disables every LLM / web-search key, so it
is fully offline, deterministic and never touches your real data/vectorstore.

LAYOUT (TC-001 ... TC-100)
--------------------------
  A. Config / security / app wiring ........ TC-001 - TC-006
  B. Text utilities & synonym normalizer ... TC-007 - TC-018
  C. PDF ingestion & semantic chunking ..... TC-019 - TC-032
  D. TF-IDF vector index ................... TC-033 - TC-038
  E. SQLite vectorless BM25 index .......... TC-039 - TC-046
  F. Prerequisite concept DAG .............. TC-047 - TC-056
  G. Epistemic mastery engine .............. TC-057 - TC-072
  H. Quiz engine (MCQ / FITB / typed) ...... TC-073 - TC-086
  I. Socratic tutor & web search ........... TC-087 - TC-094
  J. LangGraph loop, HTTP end-to-end ....... TC-095 - TC-100

Tests marked  @KNOWN_BUG  are real defects found while reading/probing the code. They are `xfail`
(non-strict), so the suite stays green today, but they will flip to XPASS once you fix the bug.
"""
from __future__ import annotations

import json
import os
import re
import sys
import tempfile
import uuid
from pathlib import Path

import pytest

# ───────────────────────────── 0. Environment isolation (must run BEFORE importing `app`) ─────────────────────────────


def _find_backend() -> Path:
    env = os.getenv("CMS_BACKEND_DIR")
    if env and (Path(env) / "app" / "main.py").exists():
        return Path(env).resolve()
    here = Path(__file__).resolve()
    for base in [here.parent, *here.parents]:
        for cand in (base, base / "backend", base / "concept-mastery" / "backend"):
            if (cand / "app" / "main.py").exists():
                return cand.resolve()
    raise RuntimeError("Could not locate backend/app/main.py. Set CMS_BACKEND_DIR=/path/to/concept-mastery/backend")


BACKEND = _find_backend()
_TMP = Path(tempfile.mkdtemp(prefix="cms_tests_")).resolve()   # .resolve(): macOS /var -> /private/var symlink

if "app.config" in sys.modules:  # imported earlier in the same pytest session -> cannot isolate safely
    from app import config as _cfg0
    if not str(_cfg0.VECTOR_STORE_PATH).startswith(str(_TMP.parent.resolve())):
        pytest.skip("Run final_100_test_cases.py in its own pytest session (app was already imported with real paths).",
                    allow_module_level=True)

os.environ.update(
    UPLOAD_DIR=str(_TMP / "uploads"),
    VECTOR_STORE_PATH=str(_TMP / "vectorstore"),
    GEMINI_API_KEY="", GROQ_API_KEY="", LLM_API_KEY="", LLM_MODEL="",   # force the deterministic offline pipeline
    TINYFISH_API_KEY="", PAGEINDEX_API_KEY="",                         # no live web search
)
sys.path.insert(0, str(BACKEND))

import networkx as nx  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from pydantic import ValidationError  # noqa: E402

from app import config  # noqa: E402
from app.core import llm, store, web_search  # noqa: E402
from app.core import text as T  # noqa: E402
from app.core.concept_index import GLOBAL_INDEX, ConceptIndex  # noqa: E402
from app.core.vector_index import VectorIndex  # noqa: E402
from app.main import app  # noqa: E402
from app.models.schemas import (  # noqa: E402
    ChunkMetadata, ConceptNode, LearnerDashboard, MasteryEvaluationResult, MasteryRecord, MasteryStatus,
    QuizQuestion, QuizResult, QuizSubmission, TutorChatRequest,
)
from app.pipelines import concepts, ingest, loop, mastery, quiz, recommend, tutor  # noqa: E402

KNOWN_BUG = lambda reason: pytest.mark.xfail(reason=reason, strict=False)  # noqa: E731

# ───────────────────────────── 1. Fixtures & helpers ─────────────────────────────


def make_pdf(pages: list[list[str]]) -> bytes:
    """Tiny dependency-free PDF writer: one list of text lines per page."""
    objs: list[bytes] = []

    def add(b: bytes) -> int:
        objs.append(b)
        return len(objs)

    n = len(pages)
    add(b"<< /Type /Catalog /Pages 2 0 R >>")
    add(f"<< /Type /Pages /Kids [{' '.join(f'{3 + 2 * i} 0 R' for i in range(n))}] /Count {n} >>".encode())
    font_id = 3 + 2 * n
    for i, lines in enumerate(pages):
        add(f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents {4 + 2 * i} 0 R "
            f"/Resources << /Font << /F1 {font_id} 0 R >> >> >>".encode())
        y, ops = 750, ["BT /F1 10 Tf"]
        for line in lines:
            safe = line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
            ops.append(f"1 0 0 1 30 {y} Tm ({safe}) Tj")
            y -= 15
        ops.append("ET")
        stream = "\n".join(ops).encode()
        add(b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream")
    add(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    out, offs = bytearray(b"%PDF-1.4\n"), []
    for i, o in enumerate(objs, 1):
        offs.append(len(out))
        out += f"{i} 0 obj\n".encode() + o + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
    for o in offs:
        out += f"{o:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF".encode()
    return bytes(out)


MAIN_PAGES = [
    ["Machine Learning Basics", "Page 1 of 3",
     "1. Loss Functions",
     "A loss function measures the gap between a model prediction and the true target value.",
     "Training tries to make the loss function as small as possible for every example.",
     "Mean squared error is a common loss function for regression problems."],
    ["Page 2 of 3",
     "2. Gradient Descent",
     "Gradient descent is an optimization method that moves the weights against the gradient of the loss function.",
     "Each step reduces the loss function a little and repeating many steps reaches a minimum.",
     "3. Learning Rate",
     "The learning rate controls how large each gradient descent step is during training.",
     "A learning rate that is too high makes gradient descent overshoot the minimum and diverge."],
    ["Page 3 of 3",
     "4. Overfitting",
     "Overfitting happens when a model memorizes noise in the training data and fails on new data.",
     "A model with too many parameters tends toward overfitting on small datasets.",
     "5. Regularization",
     "Regularization adds a penalty to the loss function to reduce overfitting in a flexible model.",
     "Weight decay is a common regularization technique that shrinks large weights during training."],
]
MAIN_NAME = "ml-test-main.pdf"


def _upload_dir() -> Path:
    return Path(config.UPLOAD_DIR)


def _write_pdf(name: str, pages: list[list[str]]) -> Path:
    p = _upload_dir() / name
    p.write_bytes(make_pdf(pages))
    return p


def _ingest_main() -> dict:
    return ingest.run_ingest(_write_pdf(MAIN_NAME, MAIN_PAGES))


@pytest.fixture(scope="session", autouse=True)
def _session_setup():
    _upload_dir().mkdir(parents=True, exist_ok=True)
    yield


@pytest.fixture(autouse=True)
def kb():
    """Every test starts with the deterministic 3-page KB loaded (re-ingested only if a previous test replaced it)."""
    if not store.KB.get("ready") or store.KB.get("file") != MAIN_NAME:
        _ingest_main()
    llm.reset_exhaustion_state()
    return store.KB


@pytest.fixture()
def uid() -> str:
    return f"user_{uuid.uuid4().hex[:10]}"


@pytest.fixture()
def client() -> TestClient:
    return TestClient(app)


def cid(i: int = 0) -> str:
    return store.KB["concepts"][i]["id"]


def _by_id(concept_id: str) -> dict:
    return next(c for c in store.KB["concepts"] if c["id"] == concept_id)


def _qa(concept_id: str, qtype: str, user: str = "default", **kw) -> tuple[QuizQuestion, QuizSubmission]:
    q = quiz.generate_diagnostic_quiz(concept_id, count=1, question_types=[qtype])[0]
    return q, QuizSubmission(user_id=user, question_id=q.question_id, concept_id=concept_id, **kw)


class _EmptyKB:
    """Temporarily simulate 'no PDF uploaded yet'."""
    def __enter__(self):
        self.saved = dict(store.KB)
        store.KB.clear()
        store.KB["ready"] = False
        return self

    def __exit__(self, *a):
        store.KB.clear()
        store.KB.update(self.saved)


# ═════════════════════════════ A. CONFIG / SECURITY / APP WIRING (TC-001 – TC-006) ═════════════════════════════


def test_tc001_health_endpoint_reports_service_and_version(client):
    """Health probe returns ok + service name + semver for uptime monitors."""
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True and body["service"] == "concept-mastery-platform"
    assert re.fullmatch(r"\d+\.\d+\.\d+", body["version"])


def test_tc002_config_paths_are_isolated_and_created():
    """Upload & vector store dirs resolve to the env-provided paths and exist (prevents writing into real data)."""
    assert str(config.UPLOAD_DIR).startswith(str(_TMP)) and config.UPLOAD_DIR.is_dir()
    assert str(config.VECTOR_STORE_PATH).startswith(str(_TMP)) and config.VECTOR_STORE_PATH.is_dir()


@KNOWN_BUG("SECURITY: config.py ships a real-looking TinyFish API key as the default value. Remove it, rotate the key, "
           "and default to ''.")
def test_tc003_no_hardcoded_api_keys_in_source():
    """Secrets must never be committed as defaults in source code."""
    src = (BACKEND / "app" / "config.py").read_text(encoding="utf-8")
    assert not re.search(r"sk-[A-Za-z0-9_\-]{12,}", src), "hard-coded secret found in app/config.py"


def test_tc004_cors_headers_present_for_browser_clients(client):
    """CORS middleware allows the SPA to call the API from another origin."""
    r = client.options("/api/health", headers={"Origin": "http://example.com", "Access-Control-Request-Method": "GET"})
    assert r.status_code in (200, 204)
    assert r.headers.get("access-control-allow-origin") in ("*", "http://example.com")


def test_tc005_openapi_docs_and_all_core_routes_registered(client):
    """/docs and /openapi.json load and every documented endpoint exists."""
    assert client.get("/docs").status_code == 200
    paths = client.get("/openapi.json").json()["paths"]
    for p in ("/api/upload", "/api/concepts/graph", "/api/loop/start", "/api/quiz/submit", "/api/tutor/chat",
              "/api/dashboard/{user_id}", "/api/health"):
        assert p in paths, f"missing route {p}"


def test_tc006_frontend_index_is_served_and_unknown_static_is_404(client):
    """StaticFiles mount serves the SPA root but does not invent files."""
    assert client.get("/").status_code == 200
    assert client.get("/definitely-not-a-file.xyz").status_code == 404


# ═════════════════════════════ B. TEXT UTILITIES & SYNONYM NORMALIZER (TC-007 – TC-018) ═════════════════════════════


def test_tc007_stem_strips_common_suffixes_but_protects_short_words():
    assert T.stem("training") == "train"
    assert T.stem("weights") == "weight"
    assert T.stem("is") == "is" and T.stem("bus") == "bus"      # too short to stem
    assert T.stem("loss") == "loss"


def test_tc008_tokens_drop_stopwords_and_short_tokens():
    toks = T.tokens("The learning of an optimizer is to be used")
    assert "the" not in toks and "of" not in toks and "an" not in toks
    assert any(t.startswith("learn") for t in toks)


def test_tc009_tokens_normalize_synonyms_before_tokenizing():
    """'backprop' must be seen as 'backpropagation' by the retrieval layer."""
    assert any(t.startswith("backpropagation") for t in T.tokens("I do not get backprop"))


def test_tc010_tokens_on_empty_and_symbol_only_text():
    assert T.tokens("") == []
    assert T.tokens("!!! ### 123") == []     # tokens must start with a letter


def test_tc011_sentences_splits_on_terminal_punctuation_only_before_capitals():
    s = T.sentences("Loss is low. Gradient descent runs! Does it converge? yes it does.")
    assert s[0] == "Loss is low." and s[1] == "Gradient descent runs!"
    assert len(s) == 3                        # 'yes it does.' stays glued to the previous sentence


def test_tc012_sentences_empty_input_returns_empty_list():
    assert T.sentences("") == [] and T.sentences("   ") == []


def test_tc013_normalizer_exact_synonym_lookup_is_case_insensitive():
    n = T.NORMALIZER
    assert n.canonicalize("LR") == "learning rate"
    assert n.canonicalize("  Cost Function ") == "loss function"
    assert n.canonicalize("SGD") == "stochastic gradient descent"


def test_tc014_normalizer_fuzzy_typo_resolves_but_garbage_does_not():
    n = T.NORMALIZER
    assert n.canonicalize("backpropogation") == "backpropagation"          # one-letter typo still resolves
    assert n.canonicalize("zzzqqq") is None
    assert n.canonicalize("") is None and n.canonicalize("   ") is None


def test_tc015_normalize_text_uses_word_boundaries_not_substrings():
    """'lr' inside 'colorless' must NOT be rewritten to 'learning rate'."""
    assert "learning rate" in T.normalize("Tune the LR carefully")
    assert "learning rate" not in T.normalize("colorless green ideas")


def test_tc016_detect_concepts_handles_plurals_and_aliases():
    names = ["Loss Functions", "Learning Rate", "Overfitting"]
    found = T.NORMALIZER.detect_canonical_concepts("A cost function and the lr both matter; models overfit.", names)
    assert "Learning Rate" in found and "Overfitting" in found


def test_tc017_detect_concepts_empty_inputs_return_empty():
    assert T.NORMALIZER.detect_canonical_concepts("", ["Loss"]) == []
    assert T.NORMALIZER.detect_canonical_concepts("some text", []) == []


def test_tc018_normalizer_survives_missing_or_corrupt_synonym_file(tmp_path):
    bad = tmp_path / "syn.json"
    bad.write_text("{not json", encoding="utf-8")
    n = T.SynonymNormalizer(bad)
    assert n.synonyms == {} and n.canonicalize("lr") is None
    missing = T.SynonymNormalizer(tmp_path / "nope.json")
    assert missing.normalize_text("Hello World") == "hello world"


# ═════════════════════════════ C. PDF INGESTION & SEMANTIC CHUNKING (TC-019 – TC-032) ═════════════════════════════


def test_tc019_extract_pages_is_one_indexed_and_complete():
    pages = ingest.extract_pages(_upload_dir() / MAIN_NAME)
    assert [p for p, _ in pages] == [1, 2, 3]
    assert "Loss Functions" in pages[0][1] and "Regularization" in pages[2][1]


def test_tc020_is_heading_accepts_numbered_titlecase_and_upper():
    assert ingest.is_heading("1. Loss Functions")
    assert ingest.is_heading("GRADIENT DESCENT")
    assert ingest.is_heading("Bias and Variance")


def test_tc021_is_heading_rejects_sentences_and_long_lines():
    assert not ingest.is_heading("This is just a normal sentence about training.")
    assert not ingest.is_heading("A heading that is much too long to be a real heading in any textbook ever written")
    assert not ingest.is_heading("Why is this a question?")


def test_tc022_clean_heading_removes_numbering_and_fixes_all_caps():
    assert ingest.clean_heading("2.1 Gradient Descent") == "Gradient Descent"
    assert ingest.clean_heading("OVERFITTING") == "Overfitting"
    assert ingest.clean_heading("Chapter 4 Regularization") == "Regularization"


def test_tc023_chunks_carry_exact_page_numbers_and_section_titles():
    ch = ingest.chunk_pages(ingest.extract_pages(_upload_dir() / MAIN_NAME), MAIN_NAME)
    by_sec = {c["section"]: c for c in ch}
    assert by_sec["Loss Functions"]["page"] == 1
    assert by_sec["Gradient Descent"]["page"] == 2
    assert by_sec["Regularization"]["page"] == 3


def test_tc024_page_number_footer_lines_never_leak_into_chunk_text():
    ch = ingest.chunk_pages(ingest.extract_pages(_upload_dir() / MAIN_NAME), MAIN_NAME)
    assert all(not re.search(r"Page \d+ of \d+", c["text"]) for c in ch)


def test_tc025_every_chunk_has_minimum_words_and_unique_ids():
    ch = ingest.chunk_pages(ingest.extract_pages(_upload_dir() / MAIN_NAME), MAIN_NAME)
    assert ch and all(len(c["text"].split()) >= 8 for c in ch)
    assert len({c["chunk_id"] for c in ch}) == len(ch)


def test_tc026_long_sections_are_split_on_sentence_boundaries():
    long_body = [f"Sentence number {i} explains an important idea about neural network training in depth." for i in range(40)]
    ch = ingest.chunk_pages([(1, "\n".join(["1. Big Topic"] + long_body))], "x.pdf")
    assert len(ch) >= 3
    assert all(c["text"].rstrip().endswith(".") for c in ch)      # never cut mid-sentence
    assert max(len(c["text"].split()) for c in ch) <= ingest.CHUNK_WORDS + 25


def test_tc027_ingest_result_contract_has_all_frontend_keys():
    res = _ingest_main()
    for k in ("job_id", "file", "pages", "chunks_indexed", "concepts_found", "graph_nodes", "graph_edges", "filenames"):
        assert k in res
    assert res["pages"] == 3 and res["graph_nodes"] == len(res["concepts_found"])
    assert res["chunks_indexed"] == GLOBAL_INDEX.count()


def test_tc028_ingest_discovers_the_five_textbook_sections_as_concepts():
    names = {c["name"] for c in store.KB["concepts"]}
    assert {"Loss Functions", "Gradient Descent", "Learning Rate", "Overfitting", "Regularization"} <= names


def test_tc029_every_chunk_is_tagged_with_at_least_its_own_concept():
    tagged = [c for c in GLOBAL_INDEX.get_all_chunks() if c.section_title == "Gradient Descent"]
    assert tagged and any("Gradient Descent" in c.canonical_concepts for c in tagged)


def test_tc030_ingest_with_no_paths_raises_user_error():
    with pytest.raises(store.UserError):
        ingest.run_ingest_multiple([])


def test_tc031_scanned_or_empty_pdf_raises_helpful_user_error():
    p = _write_pdf("blank.pdf", [[""]])
    with pytest.raises(store.UserError) as e:
        ingest.run_ingest(p)
    assert "OCR" in str(e.value) or "readable" in str(e.value)


def test_tc032_multi_pdf_ingest_aggregates_pages_and_filenames():
    a = _write_pdf("multi-a.pdf", MAIN_PAGES)
    b = _write_pdf("multi-b.pdf", MAIN_PAGES[:1])
    res = ingest.run_ingest_multiple([a, b, _upload_dir() / "ghost-missing.pdf"])   # missing file is skipped, not fatal
    assert res["files_count"] == 2 and res["pages"] == 4
    assert res["filenames"] == ["multi-a.pdf", "multi-b.pdf"]
    _ingest_main()


# ═════════════════════════════ D. TF-IDF VECTOR INDEX (TC-033 – TC-038) ═════════════════════════════


def _vi() -> VectorIndex:
    return VectorIndex.build([
        {"section": "Loss", "text": "loss function measures prediction error between target and output"},
        {"section": "Gradient", "text": "gradient descent follows the negative gradient downhill to a minimum"},
        {"section": "Overfit", "text": "overfitting memorizes noise in the training data"},
    ])


def test_tc033_search_ranks_the_relevant_chunk_first():
    hits = _vi().search("how does gradient descent reach a minimum", k=3)
    assert hits and hits[0][1]["section"] == "Gradient"


def test_tc034_search_respects_k_and_returns_descending_scores():
    hits = _vi().search("loss gradient overfitting data", k=2)
    assert len(hits) <= 2
    assert [s for s, _ in hits] == sorted([s for s, _ in hits], reverse=True)


def test_tc035_search_with_unknown_or_empty_query_returns_nothing():
    v = _vi()
    assert v.search("xyzzy plugh", k=3) == []
    assert v.search("", k=3) == []


def test_tc036_index_serialization_roundtrip_preserves_results():
    v = _vi()
    v2 = VectorIndex.from_dict(json.loads(json.dumps(v.to_dict())))
    assert [c["section"] for _, c in v.search("noise training", 2)] == [c["section"] for _, c in v2.search("noise training", 2)]


def test_tc037_vectors_are_l2_normalized():
    for vec in _vi().vecs:
        assert abs(sum(x * x for x in vec.values()) - 1.0) < 1e-6


def test_tc038_building_an_empty_index_is_safe():
    v = VectorIndex.build([])
    assert v.chunks == [] and v.search("anything") == []


# ═════════════════════════════ E. SQLITE VECTORLESS BM25 INDEX (TC-039 – TC-046) ═════════════════════════════


def test_tc039_retrieve_by_concept_returns_pages_in_ascending_order():
    got = GLOBAL_INDEX.retrieve_chunks("loss functions", max_chunks=5)
    assert got and [c.page_number for c in got] == sorted(c.page_number for c in got)


def test_tc040_retrieve_is_case_and_whitespace_insensitive():
    a = GLOBAL_INDEX.retrieve_chunks("  GRADIENT DESCENT ")
    b = GLOBAL_INDEX.retrieve_chunks("gradient descent")
    assert [c.chunk_id for c in a] == [c.chunk_id for c in b] and a


def test_tc041_retrieve_respects_max_chunks_limit():
    assert len(GLOBAL_INDEX.retrieve_chunks("loss functions", max_chunks=1)) == 1


def test_tc042_unknown_concept_falls_back_to_bm25_text_search():
    hits = GLOBAL_INDEX.retrieve_chunks("momentum penalty weights")       # not a concept tag, but words appear in text
    assert hits and any("penalty" in h.text.lower() or "weights" in h.text.lower() for h in hits)


def test_tc043_bm25_returns_normalized_scores_and_best_hit_first():
    hits = GLOBAL_INDEX.search_bm25("overfitting memorizes noise", max_chunks=3)
    assert hits and hits[0][1].section_title == "Overfitting"
    assert all(0.0 <= s <= 1.0 for s, _ in hits)


def test_tc044_bm25_survives_fts_special_characters_and_sql_injection_strings():
    for q in ['"; DROP TABLE chunks; --', "AND OR NOT ( ) *", "loss* NEAR/2 gradient", "' OR 1=1 --"]:
        GLOBAL_INDEX.search_bm25(q)                       # must not raise
    assert GLOBAL_INDEX.count() > 0                       # table still intact


def test_tc045_bm25_empty_or_too_short_query_returns_empty_list():
    assert GLOBAL_INDEX.search_bm25("") == []
    assert GLOBAL_INDEX.search_bm25("a an of") == []


def test_tc046_clear_then_reindex_replaces_not_duplicates_data():
    n = GLOBAL_INDEX.count()
    _ingest_main()
    _ingest_main()
    assert GLOBAL_INDEX.count() == n                      # re-upload must not double the index
    idx = ConceptIndex()
    idx.clear()
    assert idx.count() == 0
    _ingest_main()


# ═════════════════════════════ F. PREREQUISITE CONCEPT DAG (TC-047 – TC-056) ═════════════════════════════


def _graph() -> nx.DiGraph:
    G = nx.DiGraph()
    G.add_nodes_from(c["id"] for c in store.KB["concepts"])
    G.add_edges_from(store.KB["edges"])
    return G


def test_tc047_prerequisite_graph_is_a_strict_dag():
    assert nx.is_directed_acyclic_graph(_graph())


def test_tc048_all_edge_endpoints_are_real_concept_ids_and_no_self_loops():
    ids = {c["id"] for c in store.KB["concepts"]}
    for a, b in store.KB["edges"]:
        assert a in ids and b in ids and a != b


def test_tc049_learning_path_is_a_valid_topological_order_covering_every_concept():
    path = concepts.get_learning_path()
    assert sorted(path) == sorted(c["id"] for c in store.KB["concepts"])
    pos = {c: i for i, c in enumerate(path)}
    assert all(pos[a] < pos[b] for a, b in store.KB["edges"])


def test_tc050_depth_equals_longest_prerequisite_chain():
    G = _graph()
    for c in store.KB["concepts"]:
        preds = list(G.predecessors(c["id"]))
        expected = 0 if not preds else 1 + max(_by_id(p)["depth"] for p in preds)
        assert c["depth"] == expected


def test_tc051_expected_prerequisites_from_textbook_cooccurrence():
    """Gradient descent text mentions 'loss function'; learning-rate text mentions 'gradient descent'."""
    assert "loss-functions" in concepts.get_prerequisites("gradient-descent")
    assert "gradient-descent" in concepts.get_prerequisites("learning-rate")
    assert "overfitting" in concepts.get_prerequisites("regularization")


def test_tc052_next_concepts_is_the_inverse_of_prerequisites():
    for a, b in store.KB["edges"]:
        assert b in concepts.get_next_concepts(a)
        assert a in concepts.get_prerequisites(b)


def test_tc053_cycle_breaking_removes_back_edge_from_the_later_page(monkeypatch):
    # (a) unit level: in a 3-cycle the edge pointing from the LATEST page back to the earliest is the one dropped
    G = nx.DiGraph([("a", "b"), ("b", "c"), ("c", "a")])
    out = concepts._break_cycles(G, {"a": 1, "b": 2, "c": 3})
    assert nx.is_directed_acyclic_graph(out) and out.number_of_edges() == 2
    assert not out.has_edge("c", "a")                       # 'c' (page 3) must not be a prerequisite of 'a' (page 1)

    # (b) integration level: an (LLM-style) prerequisite that closes a loop must NOT survive concepts.build()
    chunks = ingest.chunk_pages(ingest.extract_pages(_upload_dir() / MAIN_NAME), MAIN_NAME)
    names = [c["name"].lower() for c in store.KB["concepts"]]
    cyclic = {n: {"definition": "d", "prerequisites": []} for n in names}
    cyclic["loss functions"]["prerequisites"] = ["regularization"]       # regularization already depends on loss functions
    monkeypatch.setattr(concepts, "_llm_extract_definitions_and_prereqs", lambda *_a, **_k: cyclic)
    built, edges = concepts.build(chunks)
    H = nx.DiGraph()
    H.add_nodes_from(c["id"] for c in built)
    H.add_edges_from(edges)
    assert nx.is_directed_acyclic_graph(H)
    monkeypatch.undo()
    _ingest_main()                                           # restore the module-level graph state


def test_tc054_slug_is_url_safe_bounded_and_never_empty():
    assert concepts._slug("Loss Functions!") == "loss-functions"
    assert concepts._slug("???") == "concept"
    assert len(concepts._slug("x" * 200)) <= 40


def test_tc055_export_graph_overlays_mastery_state_and_lists_edges():
    g = concepts.export_graph({cid(0): "strong"})
    node = next(n for n in g["nodes"] if n["id"] == cid(0))
    assert node["mastery_status"] == "STRONG" and node["status"] == "strong"
    other = next(n for n in g["nodes"] if n["id"] != cid(0))
    assert other["mastery_status"] == "UNKNOWN"
    assert len(g["edges"]) == len(store.KB["edges"]) and g["learning_path"] == concepts.get_learning_path()


def test_tc056_export_graph_without_kb_returns_empty_structure():
    with _EmptyKB():
        assert concepts.export_graph() == {"nodes": [], "edges": [], "learning_path": []}


# ═════════════════════════════ G. EPISTEMIC MASTERY ENGINE (TC-057 – TC-072) ═════════════════════════════

DEEP = "The gradient depends on the derivative because the loss is minimized when the rate of change equals zero."


def test_tc057_new_learner_starts_unknown_with_zero_score(uid):
    mastery.reset(store.KB["concepts"], uid)
    r = mastery.get_record(uid, cid())
    assert r.status == MasteryStatus.UNKNOWN and r.score == 0.0 and r.attempt_count == 0


def test_tc058_single_correct_answer_is_never_enough_for_strong(uid):
    mastery.reset(store.KB["concepts"], uid)
    mastery.record_quiz(uid, cid(), True, student_reasoning=DEEP)
    assert mastery.get_record(uid, cid()).status != MasteryStatus.STRONG


def test_tc059_two_deep_correct_answers_reach_strong_without_misconceptions(uid):
    mastery.reset(store.KB["concepts"], uid)
    c = cid(0)                                              # root concept: no prerequisites
    mastery.record_quiz(uid, c, True, student_reasoning=DEEP)
    mastery.record_quiz(uid, c, True, student_reasoning=DEEP)
    r = mastery.get_record(uid, c)
    assert r.status == MasteryStatus.STRONG and r.score >= 0.68 and not r.misconceptions
    assert r.reasoning_depth_level == "DEEP_CONCEPTUAL"


def test_tc060_wrong_answer_marks_concept_weak_not_unknown(uid):
    mastery.reset(store.KB["concepts"], uid)
    mastery.record_quiz(uid, cid(), False)
    assert mastery.get_record(uid, cid()).status == MasteryStatus.WEAK


def test_tc061_lucky_guess_on_correct_answer_is_flagged_weak_surface_rote(uid):
    mastery.reset(store.KB["concepts"], uid)
    for _ in range(3):
        mastery.record_quiz(uid, cid(), True, student_reasoning="I just guessed C randomly")
    r = mastery.get_record(uid, cid())
    assert r.status == MasteryStatus.WEAK and r.reasoning_depth_level == "SURFACE_ROTE"
    assert any("guess" in m.lower() for m in r.misconceptions)


def test_tc062_guess_detector_uses_word_boundaries():
    """'randomness' / 'guesswork' style words must not accuse a good explanation of guessing."""
    e = mastery.evaluate_mastery_semantic(
        "x", quiz_results=[{"is_correct": True, "quiz_accuracy": 1.0,
                            "student_reasoning": "SGD adds randomness because each batch estimates the gradient"}],
        dialogue_context=0.8, attempt_count=3)
    assert e.epistemic_status == "STRONG" and not e.misconceptions_detected


def test_tc063_weak_prerequisite_blocks_strong_even_with_perfect_answers(uid):
    mastery.reset(store.KB["concepts"], uid)
    mastery.record_quiz(uid, "loss-functions", False)                       # parent becomes WEAK
    for _ in range(3):
        mastery.record_quiz(uid, "gradient-descent", True, student_reasoning=DEEP)
    r = mastery.get_record(uid, "gradient-descent")
    assert r.status == MasteryStatus.WEAK and r.prereq_health < 0.5


def test_tc064_semantic_eval_zero_attempts_is_unknown_and_no_remediation():
    e = mastery.evaluate_mastery_semantic("x", attempt_count=0)
    assert e.epistemic_status == "UNKNOWN" and e.remedial_recommended is False


def test_tc065_semantic_eval_calibration_matches_documented_0_81():
    """0.5*0.8 + 0.3*0.7 + 0.2*1.0 = 0.81  (documented calibration example)."""
    e = mastery.evaluate_mastery_semantic(
        "x", quiz_results=[{"is_correct": True, "quiz_accuracy": 0.8, "student_reasoning": "because the derivative"}],
        dialogue_context=0.7, attempt_count=2)
    assert e.semantic_comprehension_score == pytest.approx(0.81, abs=1e-3) and e.epistemic_status == "STRONG"


def test_tc066_scores_always_stay_within_zero_and_one(uid):
    mastery.reset(store.KB["concepts"], uid)
    for ok in (True, False, True, True, False, False, True):
        mastery.record_quiz(uid, cid(1), ok, student_reasoning=DEEP if ok else "")
        s = mastery.get_record(uid, cid(1)).score
        assert 0.0 <= s <= 1.0


def test_tc067_attempt_counter_and_quiz_accuracy_are_tracked(uid):
    mastery.reset(store.KB["concepts"], uid)
    for ok in (True, False, True, True):
        mastery.record_quiz(uid, cid(), ok, student_reasoning="because loss decreases")
    r = mastery.get_record(uid, cid())
    assert r.attempt_count == 4 and r.quiz_accuracy == pytest.approx(0.75)


def test_tc068_evidence_notes_are_capped_at_fifteen_most_recent(uid):
    mastery.reset(store.KB["concepts"], uid)
    for i in range(20):
        mastery.record_quiz(uid, cid(), i % 2 == 0, question_id=f"q{i}", student_reasoning="because loss")
    notes = mastery.get_record(uid, cid()).evidence_notes
    assert len(notes) == 15 and "q19" in notes[-1]


def test_tc069_learners_are_fully_isolated_from_each_other(uid):
    other = uid + "_b"
    mastery.reset(store.KB["concepts"], uid)
    mastery.reset(store.KB["concepts"], other)
    mastery.record_quiz(uid, cid(), False)
    assert mastery.get_record(other, cid()).attempt_count == 0
    assert mastery.get_record(other, cid()).status == MasteryStatus.UNKNOWN


def test_tc070_dashboard_buckets_partition_all_concepts_and_overall_is_average(uid):
    mastery.reset(store.KB["concepts"], uid)
    mastery.record_quiz(uid, cid(0), False)
    d = mastery.get_dashboard(uid)
    assert len(d.strong) + len(d.weak) + len(d.unknown) == d.total_concepts == len(store.KB["concepts"])
    expect = sum(r.score for r in d.strong + d.weak) / d.total_concepts
    assert d.overall_score == pytest.approx(expect, abs=1e-3)
    assert "unattempted" in d.summary and f"{d.total_concepts} course concepts" in d.summary


def test_tc071_unattempted_concepts_count_as_zero_not_fifty_percent(uid):
    mastery.reset(store.KB["concepts"], uid)
    d = mastery.get_dashboard(uid)
    assert d.overall_score == 0.0 and len(d.unknown) == d.total_concepts


def test_tc072_reset_wipes_progress_and_unknown_user_dashboard_is_safe(uid):
    mastery.reset(store.KB["concepts"], uid)
    mastery.record_quiz(uid, cid(), True, student_reasoning=DEEP)
    mastery.reset(store.KB["concepts"], uid)
    assert mastery.get_record(uid, cid()).attempt_count == 0
    d = mastery.get_dashboard("someone_who_never_logged_in")
    assert d.total_concepts == len(store.KB["concepts"]) and d.overall_score == 0.0


# ═════════════════════════════ H. QUIZ ENGINE (TC-073 – TC-086) ═════════════════════════════


def test_tc073_generates_requested_number_of_questions_with_balanced_types():
    qs = quiz.generate_diagnostic_quiz("loss-functions", count=3)
    assert [q.question_type for q in qs] == ["mcq", "fill_in_the_blanks", "typed_qa"]


def test_tc074_single_question_defaults_to_mcq_and_explicit_types_are_honored():
    assert quiz.generate_diagnostic_quiz("loss-functions", count=1)[0].question_type == "mcq"
    assert quiz.generate_diagnostic_quiz("loss-functions", count=1, question_types=["typed_qa"])[0].question_type == "typed_qa"


def test_tc075_mcq_has_four_unique_options_with_a_valid_correct_index():
    q = quiz.generate_diagnostic_quiz("gradient-descent", count=1)[0]
    assert len(q.options) == 4 and len(set(q.options)) == 4
    assert 0 <= q.correct_option_idx < 4 and q.options[q.correct_option_idx] == "Gradient Descent"


def test_tc076_question_does_not_leak_the_answer_in_its_stem():
    qs = quiz.generate_diagnostic_quiz("loss-functions", count=2, question_types=["mcq", "fill_in_the_blanks"])
    for item in qs:
        stem = item.question_text.lower()
        assert "_____" in stem                                              # the key term was blanked out
        answer = item.options[item.correct_option_idx] if item.question_type == "mcq" else item.target_blank_answer
        assert answer.lower() not in stem, f"{item.question_type} stem leaks its own answer"


def test_tc077_every_question_is_grounded_in_a_real_page_and_registered_server_side():
    for q in quiz.generate_diagnostic_quiz("learning-rate", count=3):
        assert 1 <= q.source_page <= store.KB["pages"]
        assert q.question_id in quiz.REGISTRY and quiz.REGISTRY[q.question_id]["concept_id"] == "learning-rate"
        assert q.explanation.strip()


def test_tc078_unknown_concept_raises_user_error_and_no_kb_raises_upload_hint():
    with pytest.raises(store.UserError):
        quiz.generate_diagnostic_quiz("no-such-concept")
    with _EmptyKB(), pytest.raises(store.UserError) as e:
        quiz.generate_diagnostic_quiz("loss-functions")
    assert "Upload" in str(e.value)


def test_tc079_mcq_grading_correct_vs_incorrect_with_misconception_trail(uid):
    q, sub = _qa("loss-functions", "mcq", uid, selected_option_idx=None)
    sub.selected_option_idx = q.correct_option_idx
    good = quiz.evaluate_submission(sub)
    assert good.is_correct and good.score == 1.0 and good.correct_answer_display == "Loss Functions"
    sub.selected_option_idx = (q.correct_option_idx + 1) % 4
    bad = quiz.evaluate_submission(sub)
    assert not bad.is_correct and bad.score == 0.0 and bad.misconceptions


def test_tc080_mcq_none_or_out_of_range_selection_is_wrong_but_does_not_crash(uid):
    q, sub = _qa("loss-functions", "mcq", uid)
    assert quiz.evaluate_submission(sub).is_correct is False                # selected_option_idx=None
    sub.selected_option_idx = 99
    assert quiz.evaluate_submission(sub).is_correct is False
    sub.selected_option_idx = -1
    assert quiz.evaluate_submission(sub).is_correct is False


def test_tc081_fill_in_blank_accepts_exact_and_minor_typos_rejects_wrong_and_empty(uid):
    q, sub = _qa("loss-functions", "fill_in_the_blanks", uid)
    for ok in ("Loss Functions", "loss function", "LOSS FUNCTIONS!", "los functions"):
        sub.typed_answer = ok
        assert quiz.evaluate_submission(sub).is_correct, ok
    for bad in ("gradient", "", "   "):
        sub.typed_answer = bad
        assert not quiz.evaluate_submission(sub).is_correct, bad


@KNOWN_BUG("FITB grading is too lenient: any answer that merely CONTAINS the keyword passes, so 'I have no idea what loss "
           "functions are' is marked correct. Require near-equality (or LLM judging) instead of substring containment.")
def test_tc082_fill_in_blank_rejects_a_sentence_that_only_mentions_the_keyword(uid):
    q, sub = _qa("loss-functions", "fill_in_the_blanks", uid, typed_answer="I have no idea what loss functions are")
    assert quiz.evaluate_submission(sub).is_correct is False


def test_tc083_typed_answer_depth_separates_empty_lucky_and_conceptual(uid):
    q, sub = _qa("loss-functions", "typed_qa", uid)
    sub.typed_answer = ""
    assert not quiz.evaluate_submission(sub).is_correct
    sub.typed_answer = "I guessed randomly, no idea"
    lucky = quiz.evaluate_submission(sub)
    assert not lucky.is_correct and lucky.reasoning_depth_level == "SURFACE_ROTE"
    sub.typed_answer = "A loss function measures the gap between predictions and targets because training minimizes it."
    deep = quiz.evaluate_submission(sub)
    assert deep.is_correct and deep.reasoning_depth_level == "DEEP_CONCEPTUAL" and deep.score > lucky.score


def test_tc084_grading_a_question_updates_the_authoritative_mastery_record(uid):
    mastery.reset(store.KB["concepts"], uid)
    q, sub = _qa("loss-functions", "mcq", uid)
    sub.selected_option_idx = (q.correct_option_idx + 1) % 4
    quiz.evaluate_submission(sub)
    r = mastery.get_record(uid, "loss-functions")
    assert r.attempt_count == 1 and r.status == MasteryStatus.WEAK


@KNOWN_BUG("Server-side grading trusts unknown question_ids: with no registry entry it silently grades as MCQ with answer "
           "index 0, so a forged/stale id + selected_option_idx=0 is 'correct' AND writes mastery for any concept string.")
def test_tc085_forged_question_id_cannot_earn_credit(uid):
    sub = QuizSubmission(user_id=uid, question_id="forged_id_123", concept_id="loss-functions", selected_option_idx=0)
    assert quiz.evaluate_submission(sub).is_correct is False


def test_tc086_next_question_adapter_returns_the_frontend_contract():
    q = quiz.next_question("gradient-descent")
    for k in ("id", "conceptId", "conceptName", "questionType", "question", "options", "correct", "sourcePage", "explanation"):
        assert k in q
    assert q["conceptName"] == "Gradient Descent" and q["id"] in quiz.RECENT_QUESTIONS


# ═════════════════════════════ I. SOCRATIC TUTOR & WEB SEARCH (TC-087 – TC-094) ═════════════════════════════


def test_tc087_greeting_detector_matches_openers_but_not_real_questions():
    for g in ("hi", "Hello!", "hey  ", "Good Morning", "who are you", "help"):
        assert tutor.is_greeting(g), g
    for q in ("hi can you explain overfitting", "what is a loss function", "helpful tips?"):
        assert not tutor.is_greeting(q), q


def test_tc088_greeting_with_kb_mentions_document_and_creates_no_citations():
    r = tutor.chat("hello")
    assert MAIN_NAME in r.answer and r.citations == [] and r.dialogue_score == 0.5


def test_tc089_greeting_without_kb_asks_user_to_upload_a_pdf():
    with _EmptyKB():
        assert "upload" in tutor.chat("hi").answer.lower()


def test_tc090_grounded_answer_cites_the_correct_page_and_detects_concepts(uid):
    r = tutor.chat("What is gradient descent and how does the learning rate affect it?", concept_id="gradient-descent", user_id=uid)
    assert r.citations and all(c.source_type == "document" and c.page and c.page >= 1 for c in r.citations)
    assert r.citations[0].page == 2
    assert "Gradient Descent" in r.detected_concepts
    assert re.search(r"\[Page \d+\]|\(p\. \d+\)", r.answer)


def test_tc091_tutor_dialogue_score_is_bounded_and_updates_mastery_dialogue_signal(uid):
    mastery.reset(store.KB["concepts"], uid)
    r = tutor.chat("Explain why a learning rate that is too high makes gradient descent diverge", concept_id="learning-rate", user_id=uid)
    assert 0.0 <= r.dialogue_score <= 1.0
    assert mastery.get_record(uid, "learning-rate").dialogue_score != 0.5


def test_tc092_empty_whitespace_and_unknown_topic_messages_are_handled_gracefully():
    for m in ("", "   ", "qwertyuiop zxcvbnm asdfgh"):
        r = tutor.chat(m)
        assert isinstance(r.answer, str) and r.answer.strip() and r.citations == []


def test_tc093_web_search_trigger_uses_web_results_and_marks_citations_as_web(monkeypatch):
    fake = [{"title": "Gradient Descent - Wiki", "snippet": "An iterative optimisation algorithm.", "url": "https://example.org/gd",
             "site_name": "Example", "date": ""}]
    monkeypatch.setattr(web_search, "search", lambda q, max_results=5: fake)
    r = tutor.chat("search the web for gradient descent")
    assert r.citations and r.citations[0].source_type == "web" and r.citations[0].url == "https://example.org/gd"
    assert r.citations[0].page is None


def test_tc094_web_search_client_is_a_noop_without_api_key_or_query(monkeypatch):
    assert web_search.search("   ") == []
    monkeypatch.setattr(config, "TINYFISH_API_KEY", "")
    assert web_search.search("gradient descent") == []


# ═════════════════════════════ J. LANGGRAPH LOOP / API / SCHEMAS (TC-095 – TC-100) ═════════════════════════════


def test_tc095_loop_router_maps_every_next_step_to_the_right_node():
    r = loop.route_next_step
    assert r({"next_step": "COMPLETE"}) == "end" and r({"next_step": "REMEDIAL_RETEACH"}) == "remedial_reteach"
    assert r({"next_step": "UNLOCK_NEXT"}) == "unlock_next" and r({"next_step": "ASSESS"}) == "generate_quiz"
    assert r({"next_step": "DIAGNOSE"}) == "diagnose" and r({}) == "diagnose" and r({"next_step": "???"}) == "end"


def test_tc096_diagnose_prefers_weak_concepts_then_earliest_unmastered_then_completes(uid):
    mastery.reset(store.KB["concepts"], uid)
    first = loop.diagnose_node({"user_id": uid})
    assert first["next_step"] == "ASSESS" and first["target_concept"] == concepts.get_learning_path()[0]

    mastery.record_quiz(uid, "regularization", False)                          # a later concept is WEAK -> takes priority
    assert loop.diagnose_node({"user_id": uid})["target_concept"] == "regularization"

    for c in store.KB["concepts"]:                                              # everything STRONG -> COMPLETE
        for _ in range(2):
            mastery.record_quiz(uid, c["id"], True, student_reasoning=DEEP)
    done = loop.diagnose_node({"user_id": uid})
    assert done["next_step"] == "COMPLETE" and done["state"] == MasteryStatus.STRONG


def test_tc097_remedial_loop_is_bounded_and_then_unlocks_next_with_support():
    st = {"user_id": "loop_guard", "target_concept": "gradient-descent", "iteration_count": loop.MAX_REMEDIAL_ITERATIONS}
    out = loop.remedial_reteach_node(st)
    assert out["next_step"] == "UNLOCK_NEXT" and "textbook" in out["remedial_lesson"].lower()
    nxt = loop.unlock_next_node({"target_concept": "loss-functions", "iteration_count": 3})
    assert nxt["iteration_count"] == 0 and nxt["target_concept"] in concepts.get_next_concepts("loss-functions")


def test_tc098_start_session_returns_first_quiz_for_new_learner_and_complete_without_kb(uid):
    res = loop.start_session(uid)
    assert res["action"] == "quiz" and res["quiz_count"] == 3 and res["concept_id"] == concepts.get_learning_path()[0]
    assert res["quiz"]["question_type"] == "mcq"
    with _EmptyKB():
        empty = loop.start_session(uid)
    assert empty["action"] == "complete" and empty["quiz"] is None


def test_tc099_http_end_to_end_upload_quiz_submit_dashboard_and_edge_errors(client, uid):
    # --- validation / error paths ---
    assert client.post("/api/upload", files={"file": ("notes.txt", b"hello", "text/plain")}).status_code == 400
    assert client.post("/api/upload").status_code == 400
    assert client.post("/api/tutor/chat", json={}).status_code == 422
    assert client.post("/api/quiz/submit", json={"question_id": "x"}).status_code == 422
    assert client.post("/api/quiz/next", json={"conceptId": "zzz"}).status_code == 400
    # --- path traversal in filename is neutralised ---
    r = client.post("/api/upload", files={"file": ("../../evil.pdf", make_pdf(MAIN_PAGES), "application/pdf")})
    assert r.status_code == 200 and r.json()["filename"] == "evil.pdf"
    assert not (_TMP / "evil.pdf").exists() and (_upload_dir() / "evil.pdf").exists()
    # --- multi-file upload ---
    pdf = make_pdf(MAIN_PAGES)
    r = client.post("/api/upload", files=[("files", ("m1.pdf", pdf, "application/pdf")), ("files", ("m2.pdf", pdf, "application/pdf"))])
    assert r.status_code == 200 and r.json()["files_count"] == 2
    # --- happy path through the closed loop ---
    client.post("/api/upload", files={"file": (MAIN_NAME, make_pdf(MAIN_PAGES), "application/pdf")})
    assert client.get("/api/ingest/status").json()["ready"] is True
    q = client.post("/api/quiz/next", json={"conceptId": "loss-functions"}).json()
    sub = client.post("/api/quiz/submit", json={"user_id": uid, "question_id": q["id"], "concept_id": "loss-functions",
                                                "selected_option_idx": q["correct"], "student_reasoning": DEEP}).json()
    assert sub["is_correct"] is True and sub["updated_mastery"]["attempts"] == 1 and sub["next_action"] in ("advance", "remedial_teach", "complete")
    dash = client.get(f"/api/dashboard/{uid}").json()
    assert dash["total_concepts"] == len(store.KB["concepts"]) and len(dash["strong"]) + len(dash["weak"]) + len(dash["unknown"]) == dash["total_concepts"]
    graph = client.get("/api/concepts/graph", params={"user_id": uid}).json()
    assert graph["nodes"] and graph["edges"] and graph["learning_path"]
    assert client.post("/api/mastery/reset").json()["ok"] is True
    assert client.post("/api/recommend").json()["actions"][0]["type"] == "Teach first"
    _ingest_main()


@KNOWN_BUG("API robustness: (a) /api/quiz/evaluate grades ANY free-text (even 'banana pizza' or '') as correct with score 1.0 "
           "because the synthetic 'freeform' question falls into the MCQ fallback (selected idx 0 == correct idx 0); "
           "(b) a corrupt PDF to /api/upload returns 500 instead of a 400 user error.")
def test_tc100_free_text_evaluation_and_corrupt_pdf_are_handled_correctly(client):
    nonsense = client.post("/api/quiz/evaluate", json={"conceptId": "loss-functions", "studentAnswer": "banana pizza"}).json()
    empty = client.post("/api/quiz/evaluate", json={"conceptId": "loss-functions", "studentAnswer": ""}).json()
    assert nonsense["is_correct"] is False and empty["is_correct"] is False
    assert client.post("/api/upload", files={"file": ("bad.pdf", b"this is not a pdf", "application/pdf")}).status_code == 400