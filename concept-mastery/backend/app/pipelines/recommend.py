"""Step 6. Targeted re-teaching: what to review, in prerequisite order, with a short lesson from the PDF."""
from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional

from ..core import store, llm
from ..core.text import sentences
from . import mastery


def _lesson(c: dict, chunks: list[dict], use_llm: bool = False) -> str:
    """Return a lesson for the concept. If use_llm is False, extracts text deterministically in <1ms."""
    chunk_ids = (c.get("primary") or c.get("chunks") or [])
    if not chunk_ids or not chunks:
        return c.get("summary") or f"Review foundational materials for {c.get('name', 'this concept')}."

    first_chunk_idx = chunk_ids[0]
    first_chunk = chunks[first_chunk_idx] if 0 <= first_chunk_idx < len(chunks) else None
    fallback_text = (
        " ".join(sentences(first_chunk["text"])[:2])
        if first_chunk and "text" in first_chunk
        else (c.get("summary") or f"Key concepts and definitions for {c.get('name', 'this concept')}.")
    )

    if use_llm and llm.enabled():
        valid_ids = [j for j in chunk_ids[:2] if 0 <= j < len(chunks)]
        ctx = "\n".join(f"[p. {chunks[j].get('page', 1)}] {chunks[j].get('text', '')}" for j in valid_ids)
        t = llm.complete(
            "You are a patient tutor.",
            f"Context:\n{ctx}\n\nRe-teach “{c.get('name')}” in under 90 words with one simple analogy, using only this context.",
            300,
        )
        if t:
            return t
    return fallback_text


def _compute_topological_sort(concepts: list[dict], edges: list[list[str]]) -> list[str]:
    """Compute deterministic topological ordering of concepts using prerequisite DAG edges."""
    concept_ids = [c["id"] for c in concepts]
    id_set = set(concept_ids)

    in_degree: dict[str, int] = {cid: 0 for cid in concept_ids}
    adj: dict[str, list[str]] = {cid: [] for cid in concept_ids}

    for edge in edges:
        if len(edge) >= 2:
            src, tgt = edge[0], edge[1]
            if src in id_set and tgt in id_set and src != tgt:
                adj[src].append(tgt)
                in_degree[tgt] += 1

    queue = sorted([cid for cid in concept_ids if in_degree[cid] == 0])
    ordered = []

    while queue:
        curr = queue.pop(0)
        ordered.append(curr)
        for nxt in adj.get(curr, []):
            in_degree[nxt] -= 1
            if in_degree[nxt] == 0:
                queue.append(nxt)
                queue.sort()

    remaining = [cid for cid in concept_ids if cid not in set(ordered)]
    return ordered + remaining


def recommend(user_id: str = "default", refresh_ai: bool = False) -> dict:
    """Generate next-step recommendations in <10ms using deterministic graph + mastery heuristics.

    Heuristic Priority:
      1. Prerequisite blockers (upstream unmastered concepts in the DAG).
      2. Lowest mastery score concepts (remedial intervention).
      3. Unattempted concepts whose prerequisites are satisfied (next learning step).
      4. Spaced repetition decay (stale mastered concepts needing retention refresh).

    LLM calls in core/llm.py are skipped unless refresh_ai=True.
    """
    kb = store.get_kb(user_id)
    if not kb.get("ready") or not kb.get("concepts"):
        return {"actions": [], "ready": False}

    concepts_list = kb.get("concepts", [])
    edges_list = kb.get("edges", [])
    chunks = kb.get("index").chunks if (kb.get("index") and hasattr(kb.get("index"), "chunks")) else []

    snap_list = mastery.snapshot(user_id)
    snap_by_id = {s["id"]: s for s in snap_list}
    concepts_by_id = {c["id"]: c for c in concepts_list}

    prereqs_map: dict[str, list[str]] = {}
    for edge in edges_list:
        if len(edge) >= 2:
            src, tgt = edge[0], edge[1]
            prereqs_map.setdefault(tgt, []).append(src)

    topo_order = _compute_topological_sort(concepts_list, edges_list)
    topo_depth = {cid: idx for idx, cid in enumerate(topo_order)}

    now = datetime.now(timezone.utc)
    llm_budget = 3 if refresh_ai else 0

    scored_actions = []

    for cid in topo_order:
        c = concepts_by_id.get(cid)
        if not c:
            continue
        s = snap_by_id.get(cid, {
            "id": cid,
            "status": "unknown",
            "score": 0.0,
            "attempts": 0,
            "why": "Unattempted concept."
        })

        status = str(s.get("status", "unknown")).lower()
        score = float(s.get("score", 0.0))

        # Check prerequisite health
        prereq_ids = prereqs_map.get(cid, [])
        unmastered_prereqs = [
            concepts_by_id[p]["name"]
            for p in prereq_ids
            if p in concepts_by_id and str(snap_by_id.get(p, {}).get("status", "unknown")).lower() != "strong"
        ]

        # Spaced repetition decay calculation
        last_attempt_at = s.get("last_attempted_at")
        hours_elapsed = 48.0
        if last_attempt_at:
            try:
                last_dt = datetime.fromisoformat(str(last_attempt_at).replace("Z", "+00:00"))
                hours_elapsed = max(0.0, (now - last_dt).total_seconds() / 3600.0)
            except Exception:
                hours_elapsed = 48.0

        stability_hours = 168.0 if status == "strong" else (48.0 if status == "weak" else 24.0)
        retention = math.exp(-hours_elapsed / stability_hours)
        decayed_score = round(score * retention, 3)

        if unmastered_prereqs:
            action_type = "Fix prerequisites first"
            priority_weight = 10.0 + (score * 2.0)
            reason = f"Prerequisite blocker: Master '{unmastered_prereqs[0]}' first to build foundation for {c['name']}."
        elif status == "weak":
            action_type = "Remedial Review"
            priority_weight = 20.0 + (score * 10.0)
            reason = s.get("why") or f"Score is {int(score * 100)}%. Target conceptual gaps and re-evaluate."
        elif status == "unknown":
            action_type = "Teach first"
            priority_weight = 40.0 + (topo_depth.get(cid, 0) * 0.5)
            reason = "Next logical concept in the learning path. All prerequisites are satisfied."
        else:
            if hours_elapsed >= 72.0 or retention < 0.70:
                action_type = "Spaced Repetition Review"
                priority_weight = 50.0 + (retention * 10.0)
                reason = f"Retention decay: Last practiced {int(hours_elapsed / 24)} days ago. Quick refresher recommended."
            else:
                continue

        pages = sorted({
            chunks[j]["page"]
            for j in (c.get("primary") or c.get("chunks") or [])
            if 0 <= j < len(chunks) and "page" in chunks[j]
        })[:4]

        scored_actions.append({
            "priority_weight": priority_weight,
            "conceptId": cid,
            "concept_id": cid,
            "name": c["name"],
            "concept_name": c["name"],
            "type": action_type,
            "reason": reason,
            "pages": pages,
            "mastery_score": score,
            "decayed_score": decayed_score,
            "c_obj": c
        })

    scored_actions.sort(key=lambda a: a["priority_weight"])

    actions = []
    for sa in scored_actions:
        c = sa.pop("c_obj")
        sa.pop("priority_weight", None)
        sa["lesson"] = _lesson(c, chunks, use_llm=(llm_budget > 0))
        if llm_budget > 0:
            llm_budget -= 1
        actions.append(sa)

    return {"actions": actions, "ready": True}
