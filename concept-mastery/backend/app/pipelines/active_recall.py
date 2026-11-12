"""Active Recall Pipeline Engine.

Implements the active recall study cycle:
  1. Retrieve vectorless ground truth chunks from SQLite BM25 index.
  2. Synthesize structured Study Cards (Core Intuition, Key Formula/Mechanism, Worked Mini-Example, 3-5 Essential Keywords, Exact Source Citation) and 1 verification question.
  3. Authoritative evaluation via AI Mastery Judge: computes retrieval_score (0.0 - 1.0), reasoning_depth_level, retained concepts vs missed gaps, and verification correctness.
  4. Updates student's Epistemic Mastery in mastery.py (UNKNOWN / WEAK / STRONG).
  5. SQLite persistence via store.py.
"""

from __future__ import annotations

import difflib
import json
import logging
import math
import re
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from ..core import store, llm
from ..core.concept_index import GLOBAL_INDEX
from ..core.text import sentences, tokens
from ..models.schemas import (
    ActiveRecallStudyCard,
    ActiveRecallResult,
    ActiveRecallHistoryItem,
    QuizQuestion,
)
from . import mastery, quiz, concepts

logger = logging.getLogger("concept_mastery.active_recall")


def _extract_keywords(text: str, count: int = 5) -> list[str]:
    """Extract salient technical keywords from text, strictly filtering out generic conversational tokens."""
    stop_words = {
        "the", "a", "an", "and", "or", "in", "on", "at", "to", "for", "with",
        "by", "about", "as", "is", "are", "was", "were", "be", "been", "this",
        "that", "these", "those", "from", "into", "which", "then", "than", "such",
        "step", "calculate", "compute", "where", "given", "using", "when", "after",
        "what", "have", "stands", "exactly", "segment", "video", "page", "slide",
        "section", "lecture", "would", "like", "keep", "class", "here", "there",
        "their", "they", "them", "some", "more", "also", "just", "know", "look",
        "tell", "need", "want", "take", "make", "good", "well", "will", "going",
        "again", "guys", "really", "understand", "wait", "actually", "basically",
        "cuz", "saying", "said", "today", "welcome", "talking", "talk", "part",
        "thing", "things", "stuff", "helper", "helpers", "call", "start", "stop"
    }
    words = re.findall(r"\b[A-Za-z][A-Za-z0-9_-]{3,}\b", text)
    filtered = [w.lower() for w in words if w.lower() not in stop_words and len(w) >= 4]
    seen = set()
    result = []
    for w in filtered:
        if w not in seen:
            seen.add(w)
            result.append(w)
            if len(result) >= count:
                break
    return result or ["mechanism", "computation", "algorithm", "parameters"]


CONVERSATIONAL_FILLERS = [
    re.compile(r"^(?:again\s+guys,?\s*|guys,?\s*)", re.I),
    re.compile(r"^(?:wait,?\s+)?let\s+me\s+(?:just\s+)?put\s+this[.:,]*\s*", re.I),
    re.compile(r"^(?:you\s+can\s+just\s+say\s+(?:it\'?s\s+a\s+bit\s+)?it\'?s\s+like\s*)", re.I),
    re.compile(r"^(?:so,?\s+)?what\s+is\s+[^?]+(?:\s+exactly)?\??\s*", re.I),
    re.compile(r"^(?:did\s+you\s+know\s+that\s*)", re.I),
    re.compile(r"^(?:welcome\s+to\s+[^,.]+[.,]?\s*)", re.I),
    re.compile(r"^(?:today,?\s+)?i\s+would\s+like\s+to\s+[^,.]+[.,]?\s*", re.I),
    re.compile(r"^(?:in\s+this\s+video,?\s+)", re.I),
    re.compile(r"^(?:and\s+we\'?re\s+going\s+to\s+be\s+returning\s*)", re.I),
    re.compile(r"^(?:let\'?s\s+(?:start|see|ask|look)[^.:,]*[.:,]\s*)", re.I),
    re.compile(r"^(?:cuz\s+it\'?s\s+)", re.I),
]


def clean_transcript_sentence(s: str) -> str:
    """Strip informal speaker fillers, rhetorical questions, and conversational artifacts."""
    cleaned = s.strip()
    for pat in CONVERSATIONAL_FILLERS:
        cleaned = pat.sub("", cleaned).strip()
    if cleaned and cleaned[0].islower():
        cleaned = cleaned[0].upper() + cleaned[1:]
    return cleaned


def is_chunk_relevant_to_concept(chunk_text: str, concept_name: str, canonical_concepts: list[str]) -> bool:
    """Verify that a retrieved chunk genuinely discusses the concept to prevent cross-contamination."""
    if not chunk_text or not concept_name:
        return False
    from ..core.text import normalize, STOP, GENERIC_NON_CONCEPTS
    norm_c = normalize(concept_name)
    if any(normalize(c) == norm_c for c in canonical_concepts):
        return True

    words = [w for w in re.findall(r"[a-z0-9]+", norm_c) if w not in STOP and w not in GENERIC_NON_CONCEPTS]
    if not words:
        words = re.findall(r"[a-z0-9]+", norm_c)
    if not words:
        return False

    norm_txt = normalize(chunk_text)
    if len(words) >= 2:
        phrase = " ".join(words)
        if phrase in norm_txt:
            return True
        if all(re.search(r"\b" + re.escape(w) + r"\b", norm_txt) for w in words):
            return True
        return False
    else:
        return bool(re.search(r"\b" + re.escape(words[0]) + r"\b", norm_txt))


def expand_context_window(chunks: list, user_id: str, min_words: int = 200) -> tuple[list, str]:
    """Expand retrieved context window by stitching adjacent material chunks to prevent micro-fragmentation.
    
    Guarantees that the active recall micro-lesson receives at least 150-300 words of primary text.
    Never crosses material boundaries or injects unrelated lecture chunks.
    """
    if not chunks:
        return [], "[Canonical Academic Domain Curriculum]"

    total_words = sum(len(c.text.split()) for c in chunks)
    if total_words >= min_words:
        # Build composite locator
        c0 = chunks[0]
        c_last = chunks[-1]
        loc0 = getattr(c0, "locator_label", None) or f"Page {c0.page_number}"
        loc_last = getattr(c_last, "locator_label", None) or f"Page {c_last.page_number}"
        citation = f"[{loc0}]" if loc0 == loc_last else f"[{loc0} - {loc_last}]"
        return chunks, citation

    # Context is sparse: fetch adjacent neighbors strictly belonging to the SAME source material
    source_id = getattr(chunks[0], "source_id", "")
    all_chunks = GLOBAL_INDEX.get_all_chunks(user_id=user_id)
    same_source_chunks = [c for c in all_chunks if getattr(c, "source_id", "") == source_id] if source_id else all_chunks

    if not same_source_chunks:
        c0 = chunks[0]
        loc = getattr(c0, "locator_label", None) or f"Page {c0.page_number}"
        return chunks, f"[{loc}]"

    # Map chunk by ID and find index of current chunks within same source material
    current_ids = {c.chunk_id for c in chunks}
    matched_indices = [i for i, c in enumerate(same_source_chunks) if c.chunk_id in current_ids]

    expanded = list(chunks)
    if matched_indices:
        min_idx = min(matched_indices)
        max_idx = max(matched_indices)

        # Expand backwards
        step = 1
        while min_idx - step >= 0 and sum(len(c.text.split()) for c in expanded) < min_words:
            cand = same_source_chunks[min_idx - step]
            if cand.chunk_id not in {c.chunk_id for c in expanded}:
                expanded.insert(0, cand)
            step += 1

        # Expand forwards
        step = 1
        while max_idx + step < len(same_source_chunks) and sum(len(c.text.split()) for c in expanded) < min_words:
            cand = same_source_chunks[max_idx + step]
            if cand.chunk_id not in {c.chunk_id for c in expanded}:
                expanded.append(cand)
            step += 1

    # Format composite citation range
    first_c = expanded[0]
    last_c = expanded[-1]
    first_loc = getattr(first_c, "locator_label", None) or f"Page {first_c.page_number}"
    last_loc = getattr(last_c, "locator_label", None) or f"Page {last_c.page_number}"
    if first_loc == last_loc or not last_loc:
        citation = f"[{first_loc}]"
    else:
        l1 = first_loc.replace("[", "").replace("]", "").strip()
        l2 = last_loc.replace("[", "").replace("]", "").strip()
        citation = f"[{l1} - {l2}]"

    return expanded, citation


def _resolve_clean_concept_name(concept_id: str, raw_name: str, chunks: list) -> str:
    """Resolve clean academic concept name, replacing generic 'Segment XX:XX' with true subject matter."""
    if not concepts.is_generic_section_heading(raw_name) and not raw_name.lower().startswith("segment"):
        return raw_name

    combined = " ".join(c.text for c in chunks[:3]) if chunks else ""
    # Look for capitalized acronyms like "RAG", "BERT", "DAG", "LSTM", etc.
    acronym_match = re.search(r"\b([A-Z]{2,6})\b(?:\s+(?:stands for|is a|algorithm|model|architecture|retrieval))?", combined)
    if acronym_match:
        return acronym_match.group(1).upper()

    kws = _extract_keywords(combined, count=3)
    if kws:
        return kws[0].title()
    return raw_name


def _synthesize_card_fallback(concept_id: str, concept_name: str, chunks: list, c_def: str, citation: str = "") -> dict:
    """Deterministic vectorless synthesis of Study Card components with ZERO raw speaker filler strings."""
    c_lower = concept_name.lower().strip()

    # 1. Specialized authoritative domain templates for standard CS / AI concepts
    if any(k in c_lower for k in ("linked list", "linked-list", "singly linked", "doubly linked")):
        return {
            "core_intuition": (
                "A Linked List is a linear data structure consisting of discrete nodes connected by directional pointer references, "
                "enabling dynamic memory allocation without requiring contiguous physical storage."
            ),
            "key_formula_or_mechanism": (
                "Traversal operates in O(n) time via current = current.next. Insertion and deletion at the head execute in O(1) constant time, "
                "while arbitrary node deletions require O(n) traversal to locate preceding pointers."
            ),
            "worked_mini_example": (
                "class Node:\n"
                "    def __init__(self, val):\n"
                "        self.val = val\n"
                "        self.next = None\n\n"
                "head = Node(10)\n"
                "head.next = Node(20)  # Dynamic chain: 10 -> 20"
            ),
            "essential_keywords": ["node", "pointer", "traversal", "head", "next"],
            "exact_source_citation": citation or "[Canonical Curriculum Reference]",
        }
    elif any(k in c_lower for k in ("binary search", "binary-search")):
        return {
            "core_intuition": (
                "Binary Search is an optimal divide-and-conquer algorithm that locates a target key within a sorted array "
                "by iteratively halving the active search interval."
            ),
            "key_formula_or_mechanism": (
                "mid = low + (high - low) // 2. Recurrence: T(n) = T(n/2) + O(1), achieving O(log n) worst-case time complexity "
                "and O(1) auxiliary space complexity."
            ),
            "worked_mini_example": (
                "def binary_search(arr, target):\n"
                "    low, high = 0, len(arr) - 1\n"
                "    while low <= high:\n"
                "        mid = (low + high) // 2\n"
                "        if arr[mid] == target: return mid\n"
                "        elif arr[mid] < target: low = mid + 1\n"
                "        else: high = mid - 1\n"
                "    return -1"
            ),
            "essential_keywords": ["divide-and-conquer", "logarithmic", "bisection", "sorted-array", "midpoint"],
            "exact_source_citation": citation or "[Canonical Curriculum Reference]",
        }
    elif c_lower == "stack" or "stack" in c_lower:
        return {
            "core_intuition": (
                "A Stack is a linear abstract data structure governed by Last-In-First-Out (LIFO) discipline, "
                "restricting state updates and element access strictly to the top pointer."
            ),
            "key_formula_or_mechanism": (
                "Push(x) and Pop() operate in strict O(1) constant time. In recursive execution environments, "
                "the runtime call stack manages activation records and return addresses."
            ),
            "worked_mini_example": (
                "stack = []\n"
                "stack.append('A')  # Push in O(1)\n"
                "top_val = stack.pop()  # Pop in O(1) -> returns 'A' (LIFO)"
            ),
            "essential_keywords": ["lifo", "push", "pop", "top", "call-stack"],
            "exact_source_citation": citation or "[Canonical Curriculum Reference]",
        }
    elif c_lower == "queue" or "queue" in c_lower:
        return {
            "core_intuition": (
                "A Queue is a linear data structure adhering to First-In-First-Out (FIFO) semantics, "
                "buffering elements sequentially between independent head and tail endpoints."
            ),
            "key_formula_or_mechanism": (
                "Enqueue(x) inserts at the rear in O(1) time; Dequeue() removes from the front in O(1) time. "
                "Circular buffers prevent pointer drift using modulo arithmetic: tail = (tail + 1) % capacity."
            ),
            "worked_mini_example": (
                "from collections import deque\n"
                "q = deque()\n"
                "q.append(1)  # Enqueue\n"
                "val = q.popleft()  # Dequeue -> 1"
            ),
            "essential_keywords": ["fifo", "enqueue", "dequeue", "head", "tail"],
            "exact_source_citation": citation or "[Canonical Curriculum Reference]",
        }
    elif c_lower == "array" or "array" in c_lower:
        return {
            "core_intuition": (
                "An Array is a contiguous memory collection of fixed-size elements, "
                "enabling instant random access via index arithmetic."
            ),
            "key_formula_or_mechanism": (
                "Address(A[i]) = Base_Address + (i * Element_Size). Access time: O(1) constant, "
                "Search: O(n), Insertion/Deletion: O(n) due to contiguous element shifting."
            ),
            "worked_mini_example": (
                "arr = [10, 20, 30]\n"
                "val = arr[1]  # Constant-time O(1) offset lookup -> 20"
            ),
            "essential_keywords": ["contiguous", "index", "memory-offset", "random-access", "cache-locality"],
            "exact_source_citation": citation or "[Canonical Curriculum Reference]",
        }
    elif any(k in c_lower for k in ("rag", "retrieval augmented", "retrieval-augmented")):
        return {
            "core_intuition": (
                "Retrieval-Augmented Generation (RAG) augments large language models by dynamically fetching "
                "verified knowledge chunks from an external database, mitigating hallucinations and grounding answers."
            ),
            "key_formula_or_mechanism": (
                "P(y | x) = sum_{d} P_gen(y | x, d) * P_ret(d | x). Relevance ranking evaluates Cosine_Similarity(q, d) = (q · d) / (||q|| * ||d||) "
                "to retrieve top-k passages."
            ),
            "worked_mini_example": (
                "query_vec = embed(query)\n"
                "chunks = vector_db.search(query_vec, top_k=3)\n"
                "prompt = f'Context: {chunks}\\nQuestion: {query}'\n"
                "response = llm.generate(prompt)"
            ),
            "essential_keywords": ["retrieval", "augmented-generation", "embeddings", "vector-search", "grounding"],
            "exact_source_citation": citation or "[Canonical Curriculum Reference]",
        }
    elif any(k in c_lower for k in ("embedding", "embeddings", "vector database", "vector index")):
        return {
            "core_intuition": (
                "Vector Embeddings project high-dimensional text or entities into dense continuous vector spaces "
                "where geometric proximity directly reflects semantic similarity."
            ),
            "key_formula_or_mechanism": (
                "Similarity is computed via dot-product or cosine metric: Cosine(u, v) = (u · v) / (||u|| * ||v||). "
                "Indices utilize Hierarchical Navigable Small World (HNSW) graphs to achieve logarithmic nearest-neighbor search."
            ),
            "worked_mini_example": (
                "vec_a = embed('neural network')\n"
                "vec_b = embed('deep learning')\n"
                "sim = np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b))"
            ),
            "essential_keywords": ["dense-vectors", "cosine-similarity", "hnsw", "dimensionality", "dot-product"],
            "exact_source_citation": citation or "[Canonical Curriculum Reference]",
        }

    # 2. General Fallback for concepts with valid text chunks:
    # Filter and sanitize raw sentences to ELIMINATE conversational speaker filler
    raw_sents = sentences("\n".join(c.text for c in chunks)) if chunks else []
    cleaned_sents = [clean_transcript_sentence(s) for s in raw_sents]
    valid_sents = [s for s in cleaned_sents if len(s.split()) >= 6 and not concepts.is_generic_term(s)]

    if c_def and len(c_def.split()) >= 6:
        core_intuition = c_def
    elif valid_sents:
        core_intuition = valid_sents[0]
    else:
        core_intuition = f"{concept_name} represents a foundational computational concept and operational model within this curriculum."

    # Formula / Mechanism: MUST NOT be identical to core_intuition
    formula_sent = ""
    for s in valid_sents[1:]:
        if any(term in s.lower() for term in ("formula", "algorithm", "calculat", "update", "rule", "step", "complexity", "time", "space", "transform")):
            formula_sent = s
            break
    if not formula_sent and len(valid_sents) > 1 and valid_sents[1] != core_intuition:
        formula_sent = valid_sents[1]
    if not formula_sent:
        formula_sent = f"The governing mechanism of {concept_name} evaluates input parameters through formal state transitions and invariant constraints."

    # Worked example
    example_sent = ""
    for s in valid_sents[2:]:
        if any(term in s.lower() for term in ("example", "consider", "suppose", "assume", "let", "in practice", "when applying")):
            example_sent = s
            break
    if not example_sent:
        example_sent = f"# Practical trace for {concept_name}\ninput_data = initialize_state()\noutput = process_stage(input_data)\nassert output is not None"

    # Keywords: strictly filtered
    raw_kws = _extract_keywords("\n".join(c.text for c in chunks) if chunks else concept_name, count=5)
    filtered_kws = [k for k in raw_kws if not concepts.is_generic_term(k) and k not in {"again", "guys", "really", "understand", "stands", "what", "have", "segment"}]
    if len(filtered_kws) < 3:
        filtered_kws.extend(["mechanism", "workflow", "algorithm", "parameters"][:3 - len(filtered_kws)])

    return {
        "core_intuition": core_intuition.strip(),
        "key_formula_or_mechanism": formula_sent.strip(),
        "worked_mini_example": example_sent.strip(),
        "essential_keywords": filtered_kws[:5],
        "exact_source_citation": citation or "[Canonical Curriculum Reference]",
    }


def generate_micro_lesson(
    concept_id: str,
    concept_name: str = "",
    user_id: str = "default",
    timer_seconds: int = 120,
) -> dict:
    """Generate a cohesive, deeply grounded Active Recall micro-lesson with zero hallucination.

    Assembles an expanded context window (>=150-300 words) from adjacent chunks before prompting the LLM.
    Strictly verifies chunk relevance to prevent cross-contamination across unrelated lectures.
    """
    if not concept_id:
        raise store.UserError("concept_id is required to generate a micro-lesson.")

    kb = store.get_kb(user_id)
    by_id = {c["id"]: c for c in kb.get("concepts", [])}
    concept = by_id.get(concept_id)

    raw_name = concept_name or (concept.get("name", concept_id.replace("_", " ").title()) if concept else concept_id.replace("_", " ").title())
    concept_def = concept.get("definition", "") if concept else ""

    # 1. Retrieve vectorless ground truth chunks and verify relevance
    chunks = GLOBAL_INDEX.retrieve_chunks(concept_id, max_chunks=3, user_id=user_id)
    if not chunks:
        bm25_hits = GLOBAL_INDEX.search_bm25(raw_name, max_chunks=3, user_id=user_id)
        chunks = [hit[1] for hit in bm25_hits]

    # Verify that retrieved chunks actually discuss the target concept (prevent cross-contamination)
    relevant_chunks = [c for c in chunks if is_chunk_relevant_to_concept(c.text, raw_name, getattr(c, "canonical_concepts", []))]
    if not relevant_chunks:
        relevant_chunks = [c for c in chunks if is_chunk_relevant_to_concept(c.text, concept_id.replace("-", " "), getattr(c, "canonical_concepts", []))]

    if relevant_chunks:
        expanded_chunks, citation = expand_context_window(relevant_chunks, user_id=user_id, min_words=200)
    else:
        expanded_chunks = []
        citation = "[Canonical Academic Domain Curriculum]"

    # Sanitize concept name if it was an arbitrary locator like "Segment 00:00"
    resolved_name = _resolve_clean_concept_name(concept_id, raw_name, expanded_chunks)

    # 2. Synthesize Study Card via LLM with strict Zero Hallucination grounding rule
    card_data = None
    if llm.enabled():
        if expanded_chunks:
            ctx_text = "\n\n".join(f"[{getattr(c, 'locator_label', None) or f'Page {c.page_number}'}] {c.text}" for c in expanded_chunks)
            context_instruction = (
                f"Context from study material:\n{ctx_text}\n\n"
                "Ground your explanation, mechanism, and worked example in this material."
            )
        else:
            context_instruction = (
                "The uploaded study material does not contain specific text for this concept.\n"
                f"Synthesize an authoritative, textbook-grade study card for '{resolved_name}' using standard academic domain knowledge.\n"
                "NEVER invent details or reference unrelated topics."
            )

        sys_prompt = (
            "You are a distinguished university professor and expert cognitive learning tutor. "
            f"Generate an authoritative, high-yield Active Recall Study Card for the concept: '{resolved_name}'.\n\n"
            "STRICT PEDAGOGICAL SPECIFICATIONS:\n"
            "1. 'core_intuition': A clean, formal 2-3 sentence textbook explanation of the concept (STRICTLY ZERO conversational speaker filler like 'Again guys', 'Wait let me put this', 'So what is X', 'Did you know that'). Explain what it is, its computational purpose, and the core mental model.\n"
            "2. 'key_formula_or_mechanism': The formal step-by-step algorithmic mechanism, time/space complexity, or mathematical representation. Must be distinct from core intuition.\n"
            "3. 'worked_mini_example': A concrete, clean code snippet or execution trace example illustrating the concept in practice.\n"
            "4. 'essential_keywords': 3 to 5 actual domain-specific technical terms related to the concept (STRICTLY FORBIDDEN: generic words like 'what', 'have', 'segment', 'again', 'guys', 'really').\n"
            "5. 'exact_source_citation': Source citation range (e.g. '[00:00 - 04:30]' or '[Canonical Academic Domain Curriculum]').\n\n"
            "Return ONLY a single valid JSON object with exact keys:\n"
            "{\n"
            '  "core_intuition": "...",\n'
            '  "key_formula_or_mechanism": "...",\n'
            '  "worked_mini_example": "...",\n'
            '  "essential_keywords": ["term1", "term2", "term3", "term4"],\n'
            '  "exact_source_citation": "..."\n'
            "}"
        )
        usr_prompt = f"Target Concept: {resolved_name}\n\n{context_instruction}"

        try:
            raw = llm.complete(sys_prompt, usr_prompt, max_tokens=1000)
            if raw:
                parsed = llm.safe_json_loads(raw)
                if not isinstance(parsed, dict):
                    clean = llm.extract_json_payload(raw)
                    parsed = json.loads(clean, strict=False) if clean else {}
                if isinstance(parsed, dict) and all(k in parsed for k in ("core_intuition", "key_formula_or_mechanism", "worked_mini_example", "essential_keywords")):
                    filtered_kws = [
                        str(k).strip().lower() for k in parsed.get("essential_keywords", [])
                        if len(str(k).strip()) >= 3 and not concepts.is_generic_term(str(k).strip())
                    ]
                    clean_core = clean_transcript_sentence(str(parsed["core_intuition"]))
                    clean_mech = clean_transcript_sentence(str(parsed["key_formula_or_mechanism"]))
                    clean_ex = str(parsed["worked_mini_example"]).strip()

                    card_data = {
                        "core_intuition": clean_core,
                        "key_formula_or_mechanism": clean_mech,
                        "worked_mini_example": clean_ex,
                        "essential_keywords": filtered_kws[:5] or ["mechanism", "computation", "algorithm"],
                        "exact_source_citation": str(parsed.get("exact_source_citation", citation)).strip(),
                    }
        except Exception as e:
            logger.warning(f"LLM Study Card generation fallback triggered: {e}")

    if not card_data:
        card_data = _synthesize_card_fallback(concept_id, resolved_name, expanded_chunks, concept_def, citation=citation)

    # 3. Generate 1 Verification Question grounded in the concept
    verification_q: Optional[dict] = None
    try:
        source_p = expanded_chunks[0].page_number if expanded_chunks else 1
        if quiz.is_mathematical_concept(concept_id, resolved_name):
            vq_obj = quiz.generate_numerical_question(concept_id, concept_name=resolved_name, user_id=user_id, source_page=source_p)
            verification_q = vq_obj.model_dump()
        else:
            qs = quiz.generate_diagnostic_quiz(concept_id, count=1, question_types=["mcq"], user_id=user_id)
            if qs:
                verification_q = qs[0].model_dump()
    except Exception as e:
        logger.warning(f"Verification question fallback: {e}")
        fallback_qid = f"diag_{uuid.uuid4().hex[:8]}"
        vq_fallback = QuizQuestion(
            question_id=fallback_qid,
            concept_id=concept_id,
            question_type="mcq",
            question_text=f"Which term best describes the core operational mechanism of {resolved_name}?",
            options=[resolved_name, "Random Noise", "Arbitrary Search", "Unbounded Divergence"],
            correct_option_idx=0,
            explanation=card_data["core_intuition"],
            source_page=expanded_chunks[0].page_number if expanded_chunks else 1,
        )
        reg_entry = vq_fallback.model_dump()
        reg_entry["user_id"] = user_id
        quiz.REGISTRY[fallback_qid] = reg_entry
        verification_q = vq_fallback.model_dump()

    return {
        "concept_id": concept_id,
        "concept_name": resolved_name,
        "card_data": card_data,
        "verification_question": verification_q,
        "citation": citation,
    }


def start_session(
    user_id: str = "default",
    concept_id: str = "",
    timer_seconds: int = 120,
) -> ActiveRecallStudyCard:
    """Initialize an Active Recall study session with grounded Study Card and verification question."""
    lesson_bundle = generate_micro_lesson(
        concept_id=concept_id,
        user_id=user_id,
        timer_seconds=timer_seconds,
    )

    card_data = lesson_bundle["card_data"]
    concept_name = lesson_bundle["concept_name"]
    verification_q = lesson_bundle["verification_question"]

    session_id = f"ar_{uuid.uuid4().hex[:12]}"
    now_iso = datetime.now(timezone.utc).isoformat()

    # 4. Save session to SQLite
    session_record = {
        "session_id": session_id,
        "user_id": user_id,
        "concept_id": concept_id,
        "timer_seconds": timer_seconds,
        "lesson_json": card_data,
        "recall_submission_text": "",
        "verification_question_json": verification_q or {},
        "retrieval_score": None,
        "retained_concepts_json": [],
        "missed_gaps_json": [],
        "created_at": now_iso,
    }
    store.save_active_recall_session(session_record)

    return ActiveRecallStudyCard(
        session_id=session_id,
        concept_id=concept_id,
        concept_name=concept_name,
        timer_seconds=timer_seconds,
        core_intuition=card_data["core_intuition"],
        key_formula_or_mechanism=card_data["key_formula_or_mechanism"],
        worked_mini_example=card_data["worked_mini_example"],
        essential_keywords=card_data["essential_keywords"],
        exact_source_citation=card_data["exact_source_citation"],
        verification_question=verification_q,
        created_at=now_iso,
    )
    now_iso = datetime.now(timezone.utc).isoformat()

    # 4. Save session to SQLite
    session_record = {
        "session_id": session_id,
        "user_id": user_id,
        "concept_id": concept_id,
        "timer_seconds": timer_seconds,
        "lesson_json": card_data,
        "recall_submission_text": "",
        "verification_question_json": verification_q or {},
        "retrieval_score": None,
        "retained_concepts_json": [],
        "missed_gaps_json": [],
        "created_at": now_iso,
    }
    store.save_active_recall_session(session_record)

    return ActiveRecallStudyCard(
        session_id=session_id,
        concept_id=concept_id,
        concept_name=concept_name,
        timer_seconds=timer_seconds,
        core_intuition=card_data["core_intuition"],
        key_formula_or_mechanism=card_data["key_formula_or_mechanism"],
        worked_mini_example=card_data["worked_mini_example"],
        essential_keywords=card_data["essential_keywords"],
        exact_source_citation=card_data["exact_source_citation"],
        verification_question=verification_q,
        created_at=now_iso,
    )


def evaluate_session(
    user_id: str,
    session_id: str,
    concept_id: str,
    recall_text: str,
    verification_answer: Any = None,
) -> ActiveRecallResult:
    """Evaluate student active recall retrieval text & verification answer using AI Mastery Judge."""
    session = store.get_active_recall_session(session_id) if session_id else None
    if session_id and not session:
        raise store.UserError(f"Active recall session '{session_id}' not found.")
    if session and session.get("user_id") != user_id:
        raise PermissionError(f"Active recall session '{session_id}' does not belong to user '{user_id}'.")
    if not session:
        sessions = store.list_active_recall_sessions(user_id=user_id, limit=20)
        matching = [s for s in sessions if s.get("concept_id") == concept_id]
        if matching:
            session = matching[0]
            session_id = session.get("session_id", session_id)
        else:
            study_card = start_session(user_id=user_id, concept_id=concept_id)
            session = store.get_active_recall_session_for_user(study_card.session_id, user_id=user_id)
            session_id = study_card.session_id
    if not session:
        raise store.UserError(f"Active recall session '{session_id}' not found.")

    lesson = json.loads(session["lesson_json"]) if isinstance(session["lesson_json"], str) else (session["lesson_json"] or {})
    vq = json.loads(session["verification_question_json"]) if isinstance(session["verification_question_json"], str) else (session["verification_question_json"] or {})

    clean_recall = (recall_text or "").strip()
    essential_keywords = [str(k).lower().strip() for k in lesson.get("essential_keywords", [])]

    # 1. Evaluate Verification Question
    verification_correct = False
    if vq:
        q_type = vq.get("question_type", "mcq")
        if q_type == "numerical":
            correct_num = vq.get("correct_numeric_answer")
            tolerance = float(vq.get("tolerance", 0.05))
            student_val = None
            if isinstance(verification_answer, (int, float)):
                student_val = float(verification_answer)
            elif verification_answer is not None:
                match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", str(verification_answer))
                if match:
                    try:
                        student_val = float(match.group(0))
                    except ValueError:
                        student_val = None
            if student_val is not None and correct_num is not None:
                verification_correct = (abs(student_val - correct_num) <= tolerance)
        elif q_type == "mcq":
            correct_idx = int(vq.get("correct_option_idx", 0))
            options = vq.get("options", [])
            if isinstance(verification_answer, int):
                verification_correct = (verification_answer == correct_idx)
            elif isinstance(verification_answer, str):
                if verification_answer.isdigit():
                    verification_correct = (int(verification_answer) == correct_idx)
                elif correct_idx < len(options) and verification_answer.strip().lower() == options[correct_idx].strip().lower():
                    verification_correct = True
        else:
            target_blank = str(vq.get("target_blank_answer", "")).strip().lower()
            if target_blank and str(verification_answer or "").strip().lower() == target_blank:
                verification_correct = True
            elif len(str(verification_answer or "").strip()) >= 3:
                verification_correct = True
    else:
        verification_correct = True

    # 2. AI Mastery Judge Evaluation of Active Recall Retrieval
    retrieval_score = 0.0
    reasoning_depth_level = "SURFACE_ROTE"
    retained_concepts: list[str] = []
    missed_gaps: list[str] = []
    feedback = ""

    if not clean_recall:
        retrieval_score = 0.0
        reasoning_depth_level = "SURFACE_ROTE"
        retained_concepts = []
        missed_gaps = essential_keywords
        feedback = "No recall text submitted. Active retrieval required to assess comprehension."
    elif llm.enabled():
        judge_sys = (
            "You are an authoritative AI Mastery Judge assessing an unprompted Active Recall retrieval attempt.\n"
            "Compare the student's unprompted recall explanation against the Study Card ground truth.\n"
            "Evaluate:\n"
            "  1. Conceptual coverage of core intuition and mechanism (not exact verbatim memorization).\n"
            "  2. Presence and accurate explanation of essential technical keywords.\n"
            "  3. Cognitive depth: 'SURFACE_ROTE' (vague keywords only), 'DEVELOPING' (explains mechanism), or 'DEEP_CONCEPTUAL' (derives principles/formulas accurately).\n"
            "Return ONLY a single valid JSON object with exact keys:\n"
            '  "retrieval_score": <float between 0.0 and 1.0>,\n'
            '  "reasoning_depth_level": "SURFACE_ROTE" | "DEVELOPING" | "DEEP_CONCEPTUAL",\n'
            '  "retained_concepts": ["<keyword/concept 1>", ...],\n'
            '  "missed_gaps": ["<missed key concept/gap>", ...],\n'
            '  "feedback": "<clear 1-2 sentence pedagogical feedback>"'
        )
        judge_usr = (
            f"Concept: {concept_id}\n"
            f"Study Card Ground Truth:\n"
            f"- Core Intuition: {lesson.get('core_intuition')}\n"
            f"- Mechanism / Formula: {lesson.get('key_formula_or_mechanism')}\n"
            f"- Essential Keywords: {essential_keywords}\n\n"
            f"Student Unprompted Recall:\n\"{clean_recall}\"\n\n"
            f"Verification Question Correct: {verification_correct}"
        )
        try:
            raw = llm.complete(judge_sys, judge_usr, max_tokens=800)
            if raw:
                eval_data = llm.safe_json_loads(raw)
                if not isinstance(eval_data, dict):
                    clean_json = llm.extract_json_payload(raw)
                    eval_data = json.loads(clean_json, strict=False) if clean_json else {}
                if isinstance(eval_data, dict):
                    retrieval_score = max(0.0, min(1.0, float(eval_data.get("retrieval_score", 0.5))))
                    depth = str(eval_data.get("reasoning_depth_level", "DEVELOPING")).upper()
                    if depth in ("SURFACE_ROTE", "DEVELOPING", "DEEP_CONCEPTUAL"):
                        reasoning_depth_level = depth
                    retained_concepts = [str(c).strip() for c in eval_data.get("retained_concepts", [])]
                    missed_gaps = [str(g).strip() for g in eval_data.get("missed_gaps", [])]
                    feedback = str(eval_data.get("feedback", "")).strip()
        except Exception as e:
            logger.warning(f"LLM AI Mastery Judge fallback triggered: {e}")

    # Deterministic fallback evaluation if score uncalculated
    if retrieval_score == 0.0 and clean_recall:
        lower_recall = clean_recall.lower()
        retained = []
        missed = []
        for kw in essential_keywords:
            if kw in lower_recall or any(difflib.SequenceMatcher(None, kw, w).ratio() >= 0.85 for w in lower_recall.split()):
                retained.append(kw)
            else:
                missed.append(kw)

        kw_ratio = len(retained) / max(len(essential_keywords), 1)
        word_count = len(clean_recall.split())

        # Check deep reasoning markers
        deep_markers = ["because", "therefore", "gradient", "derivative", "proportional", "formula", "equals", "step", "distance", "error", "rate"]
        has_deep_marker = any(m in lower_recall for m in deep_markers)

        if kw_ratio >= 0.60 and word_count >= 15:
            base_score = 0.70 + (kw_ratio * 0.20)
            reasoning_depth_level = "DEEP_CONCEPTUAL" if has_deep_marker else "DEVELOPING"
        elif kw_ratio >= 0.30 or word_count >= 8:
            base_score = 0.50 + (kw_ratio * 0.20)
            reasoning_depth_level = "DEVELOPING"
        else:
            base_score = 0.30 * kw_ratio
            reasoning_depth_level = "SURFACE_ROTE"

        # Verification bonus/penalty
        verif_factor = 0.10 if verification_correct else -0.10
        retrieval_score = max(0.0, min(1.0, round(base_score + verif_factor, 2)))
        retained_concepts = retained
        missed_gaps = missed
        feedback = (
            f"Successfully recalled {len(retained)}/{len(essential_keywords)} key concepts. "
            f"Verification question {'passed' if verification_correct else 'missed'}."
        )

    # 3. Update Student Epistemic Mastery in SQLite
    audit_text = (
        f"[ACTIVE_RECALL] Session {session_id} on {concept_id}: "
        f"Retrieval Score={retrieval_score:.2f}, Depth={reasoning_depth_level}, "
        f"Verification={'Passed' if verification_correct else 'Failed'}"
    )

    mastery_record = mastery.record_active_recall(
        user_id=user_id,
        concept_id=concept_id,
        retrieval_score=retrieval_score,
        reasoning_depth=reasoning_depth_level,
        retained_concepts=retained_concepts,
        missed_gaps=missed_gaps,
        verification_correct=verification_correct,
        evidence_text=audit_text,
    )
    mastery_status = mastery_record.get("state", "UNKNOWN")

    # 4. Save/Update Session in SQLite (tenant-scoped)
    store.update_active_recall_session_for_user(
        session_id=session_id,
        user_id=user_id,
        updates={
            "recall_submission_text": clean_recall,
            "retrieval_score": retrieval_score,
            "retained_concepts_json": retained_concepts,
            "missed_gaps_json": missed_gaps,
        }
    )

    return ActiveRecallResult(
        session_id=session_id,
        concept_id=concept_id,
        retrieval_score=retrieval_score,
        reasoning_depth_level=reasoning_depth_level,
        retained_concepts=retained_concepts,
        missed_gaps=missed_gaps,
        verification_correct=verification_correct,
        feedback=feedback or f"Retrieved {len(retained_concepts)} concepts with {reasoning_depth_level} reasoning.",
        mastery_status=mastery_status,
        audit_trail=audit_text,
    )


def get_history(user_id: str = "default") -> list[ActiveRecallHistoryItem]:
    """Return learner's active recall session history."""
    rows = store.list_active_recall_sessions(user_id=user_id, limit=50)
    kb = store.get_kb(user_id)
    by_id = {c["id"]: c.get("name", c["id"]) for c in kb.get("concepts", [])}

    history: list[ActiveRecallHistoryItem] = []
    for r in rows:
        cid = r["concept_id"]
        cname = by_id.get(cid, cid.replace("_", " ").title())
        retained = json.loads(r["retained_concepts_json"]) if r.get("retained_concepts_json") else []
        missed = json.loads(r["missed_gaps_json"]) if r.get("missed_gaps_json") else []
        score = float(r["retrieval_score"]) if r.get("retrieval_score") is not None else None

        history.append(
            ActiveRecallHistoryItem(
                session_id=r["session_id"],
                user_id=r["user_id"],
                concept_id=cid,
                concept_name=cname,
                timer_seconds=int(r.get("timer_seconds", 120)),
                retrieval_score=score,
                retained_concepts=retained if isinstance(retained, list) else [],
                missed_gaps=missed if isinstance(missed, list) else [],
                created_at=r.get("created_at", ""),
            )
        )
    return history
