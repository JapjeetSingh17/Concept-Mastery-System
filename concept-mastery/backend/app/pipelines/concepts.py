"""Prerequisite Concept Graph Pipeline (DAG).

Constructs a Directed Acyclic Graph (DAG) of prerequisite concepts from uploaded course chunks.
Eliminates cycles based on earliest page number appearance.
Provides topological learning paths and dependency queries.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path
import networkx as nx

from ..core import store, llm
from ..core.text import normalize, STOP, GENERIC_NON_CONCEPTS, SYN, NORMALIZER, is_generic_term
from ..models.schemas import (
    ConceptNode,
    ConceptEdge,
    GraphExport,
    ExtractedConceptItem,
    ExtractedConceptsResponse,
)

MAX_CONCEPTS = 24

ACADEMIC_CS_NOUNS = {
    "array", "arrays", "queue", "queues", "stack", "stacks", "matrix", "matrices",
    "tensor", "tensors", "vector", "vectors", "gradient", "gradients", "entropy",
    "regression", "convolution", "perceptron", "perceptrons", "heuristic", "heuristics",
    "recursion", "tree", "trees", "graph", "graphs", "node", "nodes", "edge", "edges",
    "hash", "hashing", "pointer", "pointers", "memory", "cache", "caching", "mutex",
    "semaphore", "semaphores", "thread", "threads", "process", "processes", "socket",
    "schema", "schemas", "relation", "relations", "index", "indices", "transaction",
    "transactions", "normalization", "database", "databases", "embedding", "embeddings",
    "classifier", "clustering", "loss", "weights", "bias", "activation", "transformer",
    "attention", "backpropagation", "regularization", "optimizer", "token", "tokens",
    "tokenizer", "retrieval", "encoder", "decoder", "feedforward", "pipeline", "latency",
    "throughput", "serializability", "concurrency", "deadlock", "algorithm", "algorithms"
}


GENERIC_SECTION_PATTERNS = [
    re.compile(r"^(?:lecture\s+)?segment\b", re.I),
    re.compile(r"^video\s*\[", re.I),
    re.compile(r"^voice\s+note\b", re.I),
    re.compile(r"^page\s+\d+", re.I),
    re.compile(r"^slide\s+\d+", re.I),
    re.compile(r"^section\s+\d+", re.I),
    re.compile(r"^section\s*:\s*overview", re.I),
    re.compile(r"^introduction$", re.I),
    re.compile(r"^overview$", re.I),
    re.compile(r"^summary$", re.I),
    re.compile(r"^conclusion$", re.I),
    re.compile(r"^\d{1,2}:\d{2}(?::\d{2})?$", re.I),
    re.compile(r"^core\s+intuition\b", re.I),
    re.compile(r"^governing\s+mechanism\b", re.I),
    re.compile(r"^worked\s+example\b", re.I),
    re.compile(r"^essential\s+concept\b", re.I),
    re.compile(r"^key\s+takeaways?\b", re.I),
    re.compile(r"^formula\s*/?\s*mechanism\b", re.I),
    re.compile(r"^diagnostic\s+check\b", re.I),
    re.compile(r"^(?:video|audio|segment|slide|page)\s*\d*", re.I),
]


def is_generic_section_heading(heading: str) -> bool:
    """Detect if a heading is an arbitrary locator/segment label rather than a real domain concept."""
    if not heading or not heading.strip():
        return True
    h = heading.strip()
    if is_generic_term(h):
        return True
    return any(p.search(h) for p in GENERIC_SECTION_PATTERNS)

# In-memory graph references
_GRAPH: nx.DiGraph = nx.DiGraph()
_CONCEPTS_MAP: dict[str, dict] = {}


@lru_cache(maxsize=None)
def _pat(name: str, key: str):
    base = {name.lower(), key} | {k for k, v in SYN.items() if v.rstrip("s") == key.rstrip("s")}
    parts = [re.escape(n.rstrip("s") if len(n) > 4 else n) + ("s?" if len(n) > 4 else "") for n in base]
    return re.compile(r"\b(?:" + "|".join(sorted(parts, key=len, reverse=True)) + r")\b", re.I)


def pattern(c):
    return _pat(c["name"], c.get("key", normalize(c["name"])))


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")[:40] or "concept"


def _candidate_terms(chunks: list[dict], limit: int) -> list[str]:
    """Extract candidate domain terms from chunks, partitioning long materials across full timeline.
    
    Strictly filters out conversational sentence-starters, verbs, adverbs, and single generic words.
    """
    if not chunks:
        return []

    # Partition chunks into Early (0-33%), Middle (33-66%), Late (66-100%)
    # to guarantee representation of concepts appearing deep into the material/video
    n = len(chunks)
    partitions: list[list[dict]] = []
    if n >= 6:
        p1 = chunks[: n // 3]
        p2 = chunks[n // 3 : 2 * n // 3]
        p3 = chunks[2 * n // 3 :]
        partitions = [p1, p2, p3]
    else:
        partitions = [chunks]

    def _extract_from_subset(subset: list[dict], sub_limit: int) -> list[str]:
        tf, df = Counter(), Counter()
        for c in subset:
            words = re.findall(r"[a-z][a-z\-]*", normalize(c["text"]))
            seen = set()
            for n_gram in (2, 3, 1):
                for i in range(len(words) - n_gram + 1):
                    g = words[i : i + n_gram]
                    if any(w in STOP or w in GENERIC_NON_CONCEPTS for w in g):
                        continue
                    # 1-grams: MUST be in the strict academic computer science / engineering vocabulary
                    if n_gram == 1:
                        w1 = g[0]
                        if w1 not in ACADEMIC_CS_NOUNS or len(w1) < 4:
                            continue
                    if any(len(w) < 3 for w in g):
                        continue
                    t = " ".join(g)
                    if is_generic_term(t) or is_generic_section_heading(t):
                        continue
                    tf[t] += 1
                    seen.add(t)
            df.update(seen)

        sub_len = len(subset)
        ok = [
            (tf[t] * (2.5 if " " in t else 1.0), t)
            for t in tf
            if (1 <= df[t] <= max(0.85 * sub_len, 2)) and tf[t] >= (1 if " " in t else (2 if sub_len <= 5 else 3))
        ]
        res: list[str] = []
        for _, t in sorted(ok, reverse=True):
            if is_generic_section_heading(t) or is_generic_term(t):
                continue
            if not any(t in o or o in t for o in res):
                res.append(t)
            if len(res) >= sub_limit:
                break
        return res

    global_terms = _extract_from_subset(chunks, limit)
    tier_terms: list[str] = []
    if len(partitions) > 1:
        per_tier = max(4, limit // 3)
        for part in partitions:
            tier_terms.extend(_extract_from_subset(part, per_tier))

    out: list[str] = []
    for t in global_terms + tier_terms:
        if not any(t in o or o in t for o in out):
            out.append(t)
        if len(out) >= limit:
            break
    return out


def extract_concepts_with_llm(chunks: list[dict]) -> tuple[list[dict], list[tuple[str, str]]]:
    """Extract authoritative domain concepts, definitions, and pedagogical prerequisites using OpenRouter LLM.
    
    Enforces strict Pydantic validation and rejects conversational fillers or timeline locators.
    """
    if not llm.enabled() or not chunks:
        return [], []

    # Sample chunks distributed evenly across beginning (0-33%), middle (33-66%), and end (66-100%)
    if len(chunks) <= 12:
        sampled = chunks
    else:
        step = len(chunks) / 12.0
        sampled = [chunks[int(i * step)] for i in range(12)]

    ctx_parts = []
    for c in sampled:
        loc = c.get("locator_label") or f"Page {c.get('page', 1)}"
        sec = c.get("section", "")
        txt = str(c.get("text", "")).strip()
        ctx_parts.append(f"[{loc} · {sec}]\n{txt[:600]}")
    ctx = "\n\n".join(ctx_parts)

    system_prompt = (
        "You are an expert university curriculum architect and computer science professor. "
        "Analyze the provided course material chunks (from beginning to end) and extract 6 to 16 genuine "
        "academic, technical, or scientific concepts taught in this lecture/material.\n\n"
        "STRICT EXTRACTION RULES:\n"
        "1. Extract ONLY authentic domain concepts (e.g. data structures, algorithms, mathematical models, system components, engineering frameworks).\n"
        "2. NEVER extract conversational filler words, verbs, adverbs, or sentence starters (STRICTLY FORBIDDEN: 'Going', 'Right', 'Because', 'Later', 'Already', 'Creating', 'Asking', 'Once Again', 'Small', 'Understand', 'Number', 'Insert', 'Create', 'Model', 'Project', 'Paragraphs', 'Document', 'Extract Text').\n"
        "3. NEVER extract timeline segments or arbitrary locators (STRICTLY FORBIDDEN: 'Segment 00:00', 'Video [01:15]', 'Page 1', 'Slide 2').\n"
        "4. For each concept, provide:\n"
        "   - 'concept': canonical technical concept name in Title Case (e.g. 'Retrieval-Augmented Generation', 'Vector Embeddings', 'Cosine Similarity', 'Binary Search', 'Linked List').\n"
        "   - 'definition': concise, accurate 1-2 sentence textbook definition.\n"
        "   - 'prerequisites': list of other extracted concepts that are direct pedagogical prerequisites (must be studied first).\n\n"
        "Return a valid JSON object matching this schema:\n"
        "{\n"
        '  "concepts": [\n'
        '    {"concept": "<Concept Name>", "definition": "<1-2 sentence definition>", "prerequisites": ["<Prerequisite Concept>", ...]}\n'
        "  ]\n"
        "}"
    )

    user_prompt = f"Study material excerpts across lecture timeline:\n\n{ctx}"

    try:
        raw = llm.complete(system_prompt, user_prompt, max_tokens=1200)
        if not raw:
            return [], []
        data = llm.safe_json_loads(raw)
        if not data:
            clean_json = llm.extract_json_payload(raw)
            data = json.loads(clean_json, strict=False) if clean_json else {}
        items_raw = data.get("concepts", []) if isinstance(data, dict) else (data if isinstance(data, list) else [])

        validated_response = ExtractedConceptsResponse(
            concepts=[
                ExtractedConceptItem(
                    concept=str(item.get("concept", "")).strip().title(),
                    definition=str(item.get("definition", "")).strip(),
                    prerequisites=[str(p).strip().title() for p in item.get("prerequisites", []) if str(p).strip()]
                )
                for item in items_raw
                if str(item.get("concept", "")).strip()
            ]
        )

        valid_items: list[dict] = []
        for item in validated_response.concepts:
            c_name = item.concept
            if not c_name or is_generic_term(c_name) or is_generic_section_heading(c_name):
                continue
            valid_items.append({
                "concept": c_name,
                "definition": item.definition,
                "prerequisites": item.prerequisites,
            })

        if len(valid_items) < 3:
            return [], []

        return valid_items, []
    except Exception as e:
        return [], []


def _llm_extract_definitions_and_prereqs(chunks: list[dict], concept_names: list[str]) -> dict[str, dict]:
    """Prompt the LLM to extract key concepts, concise textbook definitions, and explicit prerequisites."""
    if not llm.enabled() or not concept_names:
        return {}

    # Sample chunks distributed evenly across the full duration/page timeline
    if len(chunks) <= 16:
        sampled_chunks = chunks
    else:
        step = len(chunks) / 16.0
        sampled_chunks = [chunks[int(i * step)] for i in range(16)]

    ctx_parts = []
    for c in sampled_chunks:
        loc = c.get("locator_label") or f"Page {c.get('page', 1)}"
        sec = c.get("section", "")
        txt = c.get("text", "")
        ctx_parts.append(f"[{loc} · {sec}] {txt}")
    ctx = "\n\n".join(ctx_parts)

    system_prompt = (
        "You are an expert curriculum designer. Return a JSON array of concepts with concise textbook definitions "
        "grounded in the text and explicit prerequisite dependencies. Respond with valid JSON only."
    )

    user_prompt = (
        f"Concepts to analyze: {json.dumps(concept_names)}\n\n"
        f"Context from study material:\n{ctx}\n\n"
        f"Output JSON array with exact keys: concept, definition, prerequisites:\n"
        f'[\n  {{"concept": "<name>", "definition": "<concise definition under 25 words>", "prerequisites": ["<prereq concept>", ...]}}\n]'
    )

    try:
        raw = llm.complete(system_prompt, user_prompt, max_tokens=800)
        if not raw:
            return {}
        arr = llm.safe_json_loads(raw)
        if not isinstance(arr, list):
            clean_json = llm.extract_json_payload(raw)
            arr = json.loads(clean_json, strict=False) if clean_json else []
        if not isinstance(arr, list):
            arr = []
        results = {}
        valid = {n.lower() for n in concept_names}
        for item in arr:
            name = str(item.get("concept", "")).strip()
            if name.lower() in valid:
                prereqs = [p for p in item.get("prerequisites", []) if p.lower() in valid and p.lower() != name.lower()]
                results[name.lower()] = {
                    "definition": str(item.get("definition", "")).strip(),
                    "prerequisites": prereqs,
                }
        return results
    except Exception:
        return {}


def _break_cycles(G: nx.DiGraph, page_map: dict[str, int]) -> nx.DiGraph:
    """Check for cycles and break back-edges using earliest page number appearance."""
    while True:
        try:
            cycles = list(nx.simple_cycles(G))
            if not cycles:
                break
            # For each cycle, remove the back-edge whose source appears on a later page
            cycle = cycles[0]
            cycle_edges = list(zip(cycle, cycle[1:] + [cycle[0]]))
            worst_edge = max(cycle_edges, key=lambda e: (page_map.get(e[0], 999), page_map.get(e[1], 0)))
            G.remove_edge(worst_edge[0], worst_edge[1])
        except Exception:
            break
    return G


def _layout(concepts: list[dict]):
    """Calculate x, y coordinates for clean DAG visualization."""
    if not concepts:
        return
    maxd = max((c.get("depth", 0) for c in concepts), default=0)
    cols: dict[int, list] = {}
    for c in concepts:
        cols.setdefault(c.get("depth", 0), []).append(c)

    if maxd <= 5:
        y_step = 85.0
        for d, col in cols.items():
            for k, c in enumerate(col):
                c["x"] = 80.0 + d * (560.0 / max(maxd, 1)) if maxd else 360.0
                c["y"] = 55.0 + k * y_step
    else:
        for i, c in enumerate(concepts):
            c["x"] = float(80 + (i % 4) * 180)
            c["y"] = float(55 + (i // 4) * 85)


def build(chunks: list[dict]) -> tuple[list[dict], list[list[str]]]:
    """Construct DAG: extract concepts, mine prerequisites, break cycles, and compute topological depth."""
    global _GRAPH, _CONCEPTS_MAP

    if not chunks:
        return [], []

    chunk_map = {k.get("id", i): k for i, k in enumerate(chunks)}

    # 1. Primary Concept Extraction via OpenRouter LLM (when enabled)
    llm_concepts, _ = extract_concepts_with_llm(chunks)

    names: list[str] = []
    definitions: dict[str, str] = {}
    llm_prereqs: dict[str, list[str]] = {}

    if llm_concepts:
        for item in llm_concepts:
            name = item["concept"]
            if not is_generic_section_heading(name) and not is_generic_term(name) and name not in names:
                names.append(name)
                definitions[name.lower()] = item.get("definition", "")
                llm_prereqs[name.lower()] = item.get("prerequisites", [])

    # If LLM extracted fewer than 4 concepts (or LLM is disabled/offline):
    # Supplement with strictly filtered section names and candidate domain terms
    if len(names) < 4:
        for c in chunks:
            sec = str(c.get("section", "")).strip()
            if not is_generic_section_heading(sec) and not is_generic_term(sec) and sec not in names:
                names.append(sec)

        names = names[:MAX_CONCEPTS]
        if len(names) < MAX_CONCEPTS:
            for t in _candidate_terms(chunks, MAX_CONCEPTS):
                if len(names) >= MAX_CONCEPTS:
                    break
                t_title = t.title()
                if not is_generic_section_heading(t_title) and not is_generic_term(t_title) and t_title not in names:
                    names.append(t_title)

    # Final post-processing filter to guarantee NO generic terms or locators enter the DAG
    names = [n for n in names if not is_generic_term(n) and not is_generic_section_heading(n)][:MAX_CONCEPTS]

    concepts: list[dict] = []
    used_ids: set[str] = set()

    for name in names:
        c: dict = {"name": name, "key": normalize(name), "definition": definitions.get(name.lower(), "")}
        pat = pattern(c)
        primary = [k["id"] for k in chunks if k.get("section") == name]
        mentions = [k["id"] for k in chunks if pat.search(normalize(k.get("text", "")))]
        ids = list(dict.fromkeys(primary + mentions))
        if not ids:
            words = [w for w in normalize(name).split() if w not in STOP and w not in GENERIC_NON_CONCEPTS and len(w) >= 3]
            for w in words:
                w_pat = re.compile(r"\b" + re.escape(w) + r"\b", re.I)
                ids.extend([k["id"] for k in chunks if w_pat.search(normalize(k.get("text", "")))])
            ids = list(dict.fromkeys(ids))
        if not ids:
            ids = [chunks[0]["id"]] if chunks else [0]

        cid = _slug(name)
        while cid in used_ids:
            cid += "-2"
        used_ids.add(cid)

        source_pages = sorted(list({chunk_map[j]["page"] for j in (primary or ids) if j in chunk_map and "page" in chunk_map[j]}))
        c.update({
            "id": cid,
            "concept_id": cid,
            "primary": primary,
            "chunks": ids,
            "order": len(concepts),
            "page": source_pages[0] if source_pages else 1,
            "source_pages": source_pages,
            "prerequisites": [],
        })
        concepts.append(c)

    if not concepts:
        return [], []

    # Build validated prerequisite edges:
    edges: set[tuple[str, str]] = set()
    by_name = {c["name"].lower(): c for c in concepts}

    # 1. Edges from LLM extraction (explicit pedagogical dependencies)
    for c_item in concepts:
        p_list = llm_prereqs.get(c_item["name"].lower(), [])
        for p_name in p_list:
            p_node = by_name.get(p_name.lower())
            if p_node and p_node["id"] != c_item["id"]:
                edges.add((p_node["id"], c_item["id"]))

    # 2. Structural co-occurrence prerequisite candidate edges (only between genuine domain concepts)
    for b in concepts:
        text = normalize(" ".join(chunk_map[j]["text"] for j in (b["primary"] or b["chunks"][:2]) if j in chunk_map and "text" in chunk_map[j]))
        pre = [a for a in concepts if a["order"] < b["order"] and pattern(a).search(text)]
        for a in pre[-2:]:
            if a["id"] != b["id"]:
                edges.add((a["id"], b["id"]))

    # 3. LLM grounded definitions and verified prerequisites enrichment if sparse
    if not edges and llm.enabled():
        llm_enrichment = _llm_extract_definitions_and_prereqs(chunks, [c["name"] for c in concepts])
        for name_lower, info in llm_enrichment.items():
            c = by_name.get(name_lower)
            if not c:
                continue
            if info.get("definition") and not c["definition"]:
                c["definition"] = info["definition"]
            for p_name in info.get("prerequisites", []):
                p_node = by_name.get(p_name.lower())
                if p_node and p_node["id"] != c["id"]:
                    edges.add((p_node["id"], c["id"]))

    # 4. Construct NetworkX DiGraph & Break Cycles
    G = nx.DiGraph()
    page_map = {c["id"]: c["page"] for c in concepts}
    for c in concepts:
        G.add_node(c["id"], **c)
    for src, tgt in edges:
        G.add_edge(src, tgt)

    G = _break_cycles(G, page_map)

    # 5. Compute depths from validated DAG
    depths: dict[str, int] = {}
    for node in nx.topological_sort(G):
        preds = list(G.predecessors(node))
        depths[node] = max((depths.get(p, 0) + 1 for p in preds), default=0)

    for c in concepts:
        c["depth"] = depths.get(c["id"], 0)
        c["prerequisites"] = list(G.predecessors(c["id"]))

    _layout(concepts)

    _GRAPH = G
    _CONCEPTS_MAP = {c["id"]: c for c in concepts}

    final_edges = sorted([list(e) for e in G.edges()])
    return concepts, final_edges


def get_prerequisites(concept_id: str, user_id: str = "default") -> list[str]:
    """Return immediate prerequisite concept IDs for concept_id."""
    kb = store.get_kb(user_id)
    if kb.get("ready") and "edges" in kb:
        return [a for a, b in kb.get("edges", []) if b == concept_id]
    if _GRAPH and concept_id in _GRAPH:
        return list(_GRAPH.predecessors(concept_id))
    return []


def get_next_concepts(concept_id: str, user_id: str = "default") -> list[str]:
    """Return concepts that directly depend on concept_id."""
    kb = store.get_kb(user_id)
    if kb.get("ready") and "edges" in kb:
        return [b for a, b in kb.get("edges", []) if a == concept_id]
    if _GRAPH and concept_id in _GRAPH:
        return list(_GRAPH.successors(concept_id))
    return []


def get_learning_path(unmastered_only: bool = True, mastered_ids: set[str] | None = None, user_id: str = "default") -> list[str]:
    """Topological sort of concepts in the prerequisite DAG for user_id."""
    kb = store.get_kb(user_id)
    if kb.get("ready") and kb.get("concepts"):
        # Build DAG for this user's edges
        G = nx.DiGraph()
        for a, b in kb.get("edges", []):
            G.add_edge(a, b)
        for c in kb.get("concepts", []):
            if c["id"] not in G:
                G.add_node(c["id"])
        try:
            order = list(nx.topological_sort(G))
        except Exception:
            order = [c["id"] for c in kb.get("concepts", [])]
    elif _GRAPH and len(_GRAPH) > 0:
        order = list(nx.topological_sort(_GRAPH))
    else:
        order = []

    if unmastered_only and mastered_ids:
        order = [cid for cid in order if cid not in mastered_ids]
    return order


def export_graph(user_mastery_map: dict[str, str] | None = None, user_id: str = "default") -> dict:
    """Export nodes and directed edges formatted for frontend rendering scoped to user_id."""
    kb = store.get_kb(user_id)
    if not kb.get("ready") or not kb.get("concepts"):
        return {"nodes": [], "edges": [], "learning_path": []}

    mastery = user_mastery_map or {}
    nodes: list[dict] = []
    for c in kb["concepts"]:
        cid = c["id"]
        status = mastery.get(cid, "UNKNOWN").upper()
        nodes.append({
            "concept_id": cid,
            "id": cid,
            "name": c["name"],
            "definition": c.get("definition", f"Key topic covered on page {c.get('page', 1)}"),
            "prerequisites": get_prerequisites(cid, user_id=user_id),
            "source_pages": c.get("source_pages", [c.get("page", 1)]),
            "depth": c.get("depth", 0),
            "x": c.get("x", 0.0),
            "y": c.get("y", 0.0),
            "mastery_status": status,
            "status": status.lower(),
        })

    edges = [
        {"source": a, "target": b, "relationship": "prerequisite"}
        for a, b in kb.get("edges", [])
    ]

    return {
        "nodes": nodes,
        "edges": edges,
        "learning_path": get_learning_path(user_id=user_id),
    }
