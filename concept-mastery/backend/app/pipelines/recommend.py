"""Step 6. Targeted re-teaching: what to review, in prerequisite order, with a short lesson from the PDF."""
from ..core import store, llm
from ..core.text import sentences
from . import mastery

def _lesson(c, chunks, use_llm):
    ids = (c["primary"] or c["chunks"])[:2]
    if use_llm:
        ctx = "\n".join(f"[p. {chunks[j]['page']}] {chunks[j]['text']}" for j in ids)
        t = llm.complete("You are a patient tutor.", f"Context:\n{ctx}\n\nRe-teach “{c['name']}” in under 90 words with one simple analogy, using only this context.", 300)
        if t:
            return t
    return " ".join(sentences(chunks[ids[0]]["text"])[:2])

def recommend(user_id: str = "default") -> dict:
    kb = store.get_kb(user_id)
    if not kb.get("ready"):
        return {"actions": [], "ready": False}
    snap, by = {s["id"]: s for s in mastery.snapshot(user_id)}, {c["id"]: c for c in kb["concepts"]}
    pre, chunks = mastery.prereqs(kb["edges"]), kb["index"].chunks
    actions, llm_budget = [], 3
    for cid in mastery.ranked(kb["concepts"], kb["edges"]):
        s, c = snap[cid], by[cid]
        if s["status"] == "strong":
            continue
        shaky = [by[p]["name"] for p in pre.get(cid, []) if snap[p]["status"] != "strong"]
        if s["status"] == "unknown":
            kind = "Teach first"
        elif shaky:
            kind = "Fix prerequisites first"
        else:
            kind = "Re-teach with a new angle"
        actions.append({"conceptId": cid, "name": c["name"], "type": kind, "reason": s["why"],
                        "pages": sorted({chunks[j]["page"] for j in (c["primary"] or c["chunks"])})[:4],
                        "lesson": _lesson(c, chunks, llm_budget > 0 and llm.enabled())})
        llm_budget -= 1
    return {"actions": actions, "ready": True}
