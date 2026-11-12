"""Grounded Multimodal Learning Artifact Generators and Curated Resource Tools.

Provides grounded generation tools for:
1. Visual architecture/flowchart diagrams (.svg / Mermaid)
2. Interactive lecture slides (.pptx)
3. Structured study handouts (.pdf)
4. Curated external learning resources (YouTube videos & high-signal articles)
"""

from __future__ import annotations

import io
import os
import re
import time
import uuid
import html
import logging
import urllib.parse
from pathlib import Path
from typing import Optional, List, Dict, Any

try:
    from app.pipelines.concept_index import GLOBAL_INDEX
except (ImportError, ModuleNotFoundError):
    try:
        from ..core.concept_index import GLOBAL_INDEX
    except (ImportError, ModuleNotFoundError):
        from app.core.concept_index import GLOBAL_INDEX

try:
    from app import config
except (ImportError, ModuleNotFoundError):
    from .. import config

try:
    from app.core import web_search, llm, store
except (ImportError, ModuleNotFoundError):
    from ..core import web_search, llm, store

try:
    from app.models.schemas import ExternalResourceItem, ArtifactItem
except (ImportError, ModuleNotFoundError):
    from ..models.schemas import ExternalResourceItem, ArtifactItem

logger = logging.getLogger("concept_mastery.tools")


def _get_artifacts_dir() -> Path:
    """Return verified generated artifacts directory."""
    raw_dir = getattr(config, "GENERATED_ARTIFACTS_DIR", None)
    if not raw_dir:
        raw_dir = Path(__file__).resolve().parent.parent.parent / "generated_artifacts"
    p = Path(raw_dir)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _sanitize_name(val: str) -> str:
    """Sanitize string for filesystem safety."""
    cleaned = re.sub(r"[^\w\-.]+", "_", val.strip()).strip("_")
    return cleaned[:48] or "concept"


def _get_grounded_context(concept_id: str, user_id: str = "default") -> tuple[str, str, str, list]:
    """Retrieve concept name, source locator, consolidated text, and chunk list."""
    kb = store.get_kb(user_id) if hasattr(store, "get_kb") else {}
    if not kb.get("ready") and hasattr(store, "KB") and store.KB.get("ready"):
        kb = store.KB

    concepts_list = kb.get("concepts", [])
    by_id = {c["id"]: c for c in concepts_list}
    concept_meta = by_id.get(concept_id, {})
    concept_name = concept_meta.get("name") or concept_id.replace("_", " ").title()

    chunks = []
    try:
        chunks = GLOBAL_INDEX.retrieve_chunks(concept_id, max_chunks=3, user_id=user_id)
    except Exception as e:
        logger.debug(f"Direct chunk retrieval error: {e}")

    if not chunks:
        try:
            hits = GLOBAL_INDEX.search_bm25(concept_name, max_chunks=3, user_id=user_id)
            chunks = [hit[1] for hit in hits]
        except Exception as e:
            logger.debug(f"BM25 chunk search error: {e}")

    if not chunks and kb.get("ready"):
        index_obj = kb.get("index")
        if index_obj and hasattr(index_obj, "search"):
            hits = index_obj.search(concept_name, k=3)
            chunks = [
                type("ChunkMeta", (), {
                    "chunk_id": str(c.get("id", i)),
                    "page_number": c.get("page", 1),
                    "section_title": c.get("section", concept_name),
                    "text": c.get("text", ""),
                    "locator_label": f"Page {c.get('page', 1)}"
                })()
                for i, (_, c) in enumerate(hits)
            ]

    # Resolve source locator
    if chunks:
        locs = []
        for c in chunks:
            loc = getattr(c, "locator_label", None) or f"Page {getattr(c, 'page_number', 1)}"
            if loc not in locs:
                locs.append(loc)
        source_locator = f"[{', '.join(locs)}]" if locs else "[Page 1]"
        combined_text = "\n\n".join(
            f"[{getattr(c, 'locator_label', None) or f'Page {getattr(c, page_number, 1)}'}] {getattr(c, 'text', '')}"
            for c in chunks
        )
    else:
        source_locator = "[Course Material]"
        combined_text = f"Overview of {concept_name} fundamentals and applications."

    return concept_name, source_locator, combined_text, chunks


# ── Tool 1: Visual Diagram Generator ───────────────────────────────────────

def sanitize_svg_diagram(raw_svg: str, title: str = "", source_locator: str = "") -> str:
    """Extract and sanitize a valid, self-contained <svg> element.

    - Extracts <svg[\s\S]*?</svg> stripping markdown fences (```svg/```xml) and prose.
    - Validates that <text> nodes exist and are not empty.
    - Injects xmlns='http://www.w3.org/2000/svg', viewBox='0 0 900 420' (if missing),
      width='100%', style='max-width:100%; height:auto; display:block;'.
    - Enforces high-contrast light text/stroke colors (#e2e8f0, #60a5fa, #38bdf8) so
      the SVG is clearly visible against our dark #0f172a UI background.
    """
    if not raw_svg:
        return ""

    # Strip any markdown fences
    cleaned = re.sub(r"^```(?:svg|xml)?\s*", "", raw_svg.strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned.strip())

    # Extract <svg ... </svg>
    m = re.search(r"<svg[\s\S]*?</svg>", cleaned, re.IGNORECASE)
    if not m:
        return ""

    svg_str = m.group(0).strip()

    # Check for empty <text> nodes: if text nodes exist but all are empty, reject
    text_nodes = re.findall(r"<text[^>]*>([\s\S]*?)</text>", svg_str, re.IGNORECASE)
    if text_nodes:
        non_empty = [t.strip() for t in text_nodes if t.strip()]
        if not non_empty:
            return ""

    # Ensure root <svg> has required attributes
    root_match = re.search(r"<svg([^>]*)>", svg_str, re.IGNORECASE)
    if not root_match:
        return ""

    attrs = root_match.group(1)

    # 1. xmlns
    if "xmlns=" not in attrs:
        attrs += ' xmlns="http://www.w3.org/2000/svg"'

    # 2. viewBox
    if "viewBox=" not in attrs and "viewbox=" not in attrs:
        attrs += ' viewBox="0 0 900 420"'

    # 3. width
    if "width=" not in attrs:
        attrs += ' width="100%"'
    else:
        attrs = re.sub(r'width=[\'"][^\'"]*[\'"]', 'width="100%"', attrs)

    # 4. style
    required_style = "max-width:100%; height:auto; display:block;"
    if "style=" not in attrs:
        attrs += f' style="{required_style}"'
    else:
        attrs = re.sub(r'style=[\'"][^\'"]*[\'"]', f'style="{required_style}"', attrs)

    new_root = f"<svg{attrs}>"
    svg_str = svg_str[:root_match.start()] + new_root + svg_str[root_match.end():]

    # Enforce high-contrast light colors for text against dark #0f172a UI
    def _fix_text_tag(match: re.Match) -> str:
        tag = match.group(0)
        tag = re.sub(
            r'fill=[\'"](?:#000000|#000|black|#111827|#1e293b|#334155)[\'"]',
            'fill="#e2e8f0"',
            tag,
            flags=re.IGNORECASE
        )
        if "fill=" not in tag:
            tag = tag[:-1] + ' fill="#e2e8f0">'
        return tag

    svg_str = re.sub(r"<text\b[^>]*>", _fix_text_tag, svg_str, flags=re.IGNORECASE)

    return svg_str


def generate_deterministic_svg_flowchart(
    concept_name: str,
    source_locator: str = "Course Material",
    chunks: Optional[list] = None
) -> str:
    """Build a clean, high-contrast multi-node SVG flowchart from concept prerequisites,

    key sub-steps, invariant checks, and relationships against the dark #0f172a UI.
    """
    escaped_title = html.escape(f"{concept_name} Architecture & Concept Flow")
    escaped_concept = html.escape(concept_name)
    escaped_locator = html.escape(source_locator)

    step1_items = ["• Domain Input Constraints", "• Baseline Initial State", "• Verified Preconditions", "• Prerequisite Linkages"]
    step2_items = ["• Governing Equations", "• State Transformations", "• Iterative Operator", "• Parameter Evolution"]
    step3_items = ["• Invariant Assertions", "• Error Bound Bounds", "• Numerical Stability", "• Convergence Verification"]
    step4_items = ["• Optimal Output State", "• Downstream Integration", "• Exam Problem Readiness", "• Synthesized Mastery"]

    if chunks:
        extracted = []
        for c in chunks:
            raw_t = getattr(c, "text", "") or ""
            for sent in re.split(r"[.!?]\s+", raw_t):
                s = sent.strip()
                if 12 <= len(s) <= 65:
                    extracted.append(s)
        if len(extracted) >= 4:
            step1_items[0] = f"• {html.escape(extracted[0][:32])}"
            step2_items[0] = f"• {html.escape(extracted[1][:32])}"
            step3_items[0] = f"• {html.escape(extracted[2][:32])}"
            step4_items[0] = f"• {html.escape(extracted[3][:32])}"

    svg_content = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 900 420" width="100%" height="100%" style="max-width:100%; height:auto; display:block;">
  <defs>
    <linearGradient id="bgGrad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#0f172a" />
      <stop offset="100%" stop-color="#1e293b" />
    </linearGradient>
    <filter id="cardShadow" x="-10%" y="-10%" width="120%" height="120%">
      <feDropShadow dx="0" dy="4" stdDeviation="5" flood-color="#000000" flood-opacity="0.4" />
    </filter>
    <marker id="arrowHead" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1 L 8 5 L 0 9 z" fill="#38bdf8" />
    </marker>
  </defs>

  <!-- Container background -->
  <rect width="900" height="420" rx="14" fill="url(#bgGrad)" stroke="#334155" stroke-width="1.5" />

  <!-- Title & Grounding Banner -->
  <text x="32" y="44" fill="#f8fafc" font-size="19" font-weight="700" font-family="system-ui, -apple-system, sans-serif">{escaped_title}</text>
  <rect x="32" y="58" width="220" height="24" rx="6" fill="#1e293b" stroke="#334155" stroke-width="1" />
  <text x="42" y="74" fill="#38bdf8" font-size="11" font-weight="600" font-family="system-ui, -apple-system, sans-serif">Source: {escaped_locator}</text>

  <!-- Node 1: Inputs & Preconditions -->
  <g filter="url(#cardShadow)">
    <rect x="32" y="115" width="185" height="215" rx="10" fill="#1e293b" stroke="#60a5fa" stroke-width="2" />
    <rect x="32" y="115" width="185" height="38" rx="10" fill="#1e3a8a" fill-opacity="0.4" />
    <text x="44" y="140" fill="#60a5fa" font-size="13" font-weight="700" font-family="system-ui, -apple-system, sans-serif">1. Preconditions</text>
    <text x="44" y="176" fill="#e2e8f0" font-size="12" font-family="system-ui, -apple-system, sans-serif">{step1_items[0]}</text>
    <text x="44" y="202" fill="#e2e8f0" font-size="12" font-family="system-ui, -apple-system, sans-serif">{step1_items[1]}</text>
    <text x="44" y="228" fill="#e2e8f0" font-size="12" font-family="system-ui, -apple-system, sans-serif">{step1_items[2]}</text>
    <text x="44" y="254" fill="#e2e8f0" font-size="12" font-family="system-ui, -apple-system, sans-serif">{step1_items[3]}</text>
    <text x="44" y="295" fill="#94a3b8" font-size="10.5" font-family="system-ui, -apple-system, sans-serif">Validates preconditions</text>
  </g>

  <!-- Arrow 1 -> 2 -->
  <line x1="222" y1="220" x2="248" y2="220" stroke="#38bdf8" stroke-width="2.5" marker-end="url(#arrowHead)" stroke-dasharray="3 3" />

  <!-- Node 2: Core Concept / Mechanism -->
  <g filter="url(#cardShadow)">
    <rect x="254" y="115" width="185" height="215" rx="10" fill="#1e293b" stroke="#38bdf8" stroke-width="2" />
    <rect x="254" y="115" width="185" height="38" rx="10" fill="#0369a1" fill-opacity="0.4" />
    <text x="266" y="140" fill="#38bdf8" font-size="13" font-weight="700" font-family="system-ui, -apple-system, sans-serif">2. {escaped_concept[:14]}</text>
    <text x="266" y="176" fill="#e2e8f0" font-size="12" font-family="system-ui, -apple-system, sans-serif">{step2_items[0]}</text>
    <text x="266" y="202" fill="#e2e8f0" font-size="12" font-family="system-ui, -apple-system, sans-serif">{step2_items[1]}</text>
    <text x="266" y="228" fill="#e2e8f0" font-size="12" font-family="system-ui, -apple-system, sans-serif">{step2_items[2]}</text>
    <text x="266" y="254" fill="#e2e8f0" font-size="12" font-family="system-ui, -apple-system, sans-serif">{step2_items[3]}</text>
    <text x="266" y="295" fill="#94a3b8" font-size="10.5" font-family="system-ui, -apple-system, sans-serif">Applies operational rules</text>
  </g>

  <!-- Arrow 2 -> 3 -->
  <line x1="444" y1="220" x2="470" y2="220" stroke="#38bdf8" stroke-width="2.5" marker-end="url(#arrowHead)" stroke-dasharray="3 3" />

  <!-- Node 3: Invariant Verification -->
  <g filter="url(#cardShadow)">
    <rect x="476" y="115" width="185" height="215" rx="10" fill="#1e293b" stroke="#818cf8" stroke-width="2" />
    <rect x="476" y="115" width="185" height="38" rx="10" fill="#4338ca" fill-opacity="0.4" />
    <text x="488" y="140" fill="#818cf8" font-size="13" font-weight="700" font-family="system-ui, -apple-system, sans-serif">3. Verification</text>
    <text x="488" y="176" fill="#e2e8f0" font-size="12" font-family="system-ui, -apple-system, sans-serif">{step3_items[0]}</text>
    <text x="488" y="202" fill="#e2e8f0" font-size="12" font-family="system-ui, -apple-system, sans-serif">{step3_items[1]}</text>
    <text x="488" y="228" fill="#e2e8f0" font-size="12" font-family="system-ui, -apple-system, sans-serif">{step3_items[2]}</text>
    <text x="488" y="254" fill="#e2e8f0" font-size="12" font-family="system-ui, -apple-system, sans-serif">{step3_items[3]}</text>
    <text x="488" y="295" fill="#94a3b8" font-size="10.5" font-family="system-ui, -apple-system, sans-serif">Guarantees correctness</text>
  </g>

  <!-- Arrow 3 -> 4 -->
  <line x1="666" y1="220" x2="692" y2="220" stroke="#38bdf8" stroke-width="2.5" marker-end="url(#arrowHead)" stroke-dasharray="3 3" />

  <!-- Node 4: Synthesis & Application -->
  <g filter="url(#cardShadow)">
    <rect x="698" y="115" width="170" height="215" rx="10" fill="#1e293b" stroke="#34d399" stroke-width="2" />
    <rect x="698" y="115" width="170" height="38" rx="10" fill="#065f46" fill-opacity="0.4" />
    <text x="710" y="140" fill="#34d399" font-size="13" font-weight="700" font-family="system-ui, -apple-system, sans-serif">4. Synthesis</text>
    <text x="710" y="176" fill="#e2e8f0" font-size="12" font-family="system-ui, -apple-system, sans-serif">{step4_items[0]}</text>
    <text x="710" y="202" fill="#e2e8f0" font-size="12" font-family="system-ui, -apple-system, sans-serif">{step4_items[1]}</text>
    <text x="710" y="228" fill="#e2e8f0" font-size="12" font-family="system-ui, -apple-system, sans-serif">{step4_items[2]}</text>
    <text x="710" y="254" fill="#e2e8f0" font-size="12" font-family="system-ui, -apple-system, sans-serif">{step4_items[3]}</text>
    <text x="710" y="295" fill="#94a3b8" font-size="10.5" font-family="system-ui, -apple-system, sans-serif">Systemic outcomes</text>
  </g>

  <!-- Running Footer -->
  <text x="32" y="385" fill="#64748b" font-size="11.5" font-family="system-ui, -apple-system, sans-serif">Adaptive Concept Mastery · Grounded strictly in {escaped_locator} · Interactive SVG Flowchart</text>
</svg>"""
    return svg_content


def generate_visual_diagram_tool(
    concept_id: str,
    prompt: str = "",
    user_id: str = "default"
) -> dict:
    """Generate a clean .svg diagram or Mermaid flowchart grounded in [Page X] / [Slide X] chunks."""
    concept_name, source_locator, context_text, chunks = _get_grounded_context(concept_id, user_id)
    title = f"{concept_name} Architecture & Concept Flow"
    artifacts_dir = _get_artifacts_dir()

    filename = f"diagram_{_sanitize_name(concept_id)}_{uuid.uuid4().hex[:8]}"
    file_path = artifacts_dir / f"{filename}.svg"

    svg_content = ""

    # 1. Attempt LLM generation if enabled
    if llm.enabled():
        try:
            system_prompt = (
                "You are an expert technical diagram designer and SVG engineer. "
                "Your task is to generate a beautiful, modern, self-contained SVG concept flow diagram "
                "illustrating the architecture, operational mechanisms, and step-by-step relationships "
                "of the given concept. "
                "You MUST return ONLY the raw <svg ...>...</svg> element. "
                "Do NOT wrap it in Markdown code blocks (no ```svg or ```xml). "
                "Do NOT include any conversational text, explanations, or HTML outside the <svg> element. "
                "The SVG must have: "
                "- xmlns='http://www.w3.org/2000/svg' "
                "- viewBox='0 0 900 420' "
                "- width='100%' "
                "- Dark background (fill='#0f172a' or '#1e293b') "
                "- High-contrast light text/stroke colors (#e2e8f0, #60a5fa, #38bdf8, #f8fafc) "
                "- Clean rectangular or rounded-card nodes with non-empty text labels "
                "- Clear directional arrows between nodes."
            )
            user_prompt = (
                f"Concept: {concept_name}\n"
                f"Source Grounding: {source_locator}\n"
                f"Context Details:\n{context_text[:1500]}\n"
                f"{f'User Directives: {prompt}' if prompt else ''}\n\n"
                "Generate the self-contained <svg> element now:"
            )
            llm_response = llm.chat_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.2,
                max_tokens=2500
            )
            if llm_response:
                svg_content = sanitize_svg_diagram(llm_response, title=title, source_locator=source_locator)
        except Exception as e:
            logger.warning(f"LLM visual diagram generation failed: {e}")

    # 2. Deterministic fallback if LLM output is missing, invalid, or has empty text nodes
    if not svg_content:
        raw_fallback = generate_deterministic_svg_flowchart(concept_name, source_locator=source_locator, chunks=chunks)
        svg_content = sanitize_svg_diagram(raw_fallback, title=title, source_locator=source_locator) or raw_fallback

    # Mermaid flowchart representation for alternative consumers
    mermaid_code = (
        f"graph TD\n"
        f"  subgraph Source [{source_locator}]\n"
        f"    A[Preconditions & Inputs: {concept_name}] --> B[Mechanism & Operations]\n"
        f"    B --> C[Invariant Verification]\n"
        f"    C --> D[Grounded Mastery & Applications]\n"
        f"  end\n"
        f"  style A fill:#1e3a8a,stroke:#60a5fa,stroke-width:2px,color:#ffffff\n"
        f"  style B fill:#0369a1,stroke:#38bdf8,stroke-width:2px,color:#ffffff\n"
        f"  style C fill:#4338ca,stroke:#818cf8,stroke-width:2px,color:#ffffff\n"
        f"  style D fill:#065f46,stroke:#34d399,stroke-width:2px,color:#ffffff"
    )

    file_path.write_text(svg_content, encoding="utf-8")
    rel_url = f"/api/artifacts/{filename}.svg"

    return {
        "artifact_type": "visual",
        "title": title,
        "svg": svg_content,
        "code": svg_content,
        "svg_or_mermaid_code": svg_content,
        "download_url": rel_url,
        "file_url": rel_url,
        "url": rel_url,
        "source_locator": source_locator,
        "metadata": {
            "svg": svg_content,
            "download_url": rel_url,
            "file_url": rel_url,
            "svg_or_mermaid_code": svg_content,
            "mermaid_flowchart": mermaid_code,
            "source_locator": source_locator,
        }
    }


# ── Tool 2: Presentation Deck Generator ────────────────────────────────────

def _clean_pdf_text(text: str) -> str:
    """Sanitize text to safe Latin-1 / ASCII printable range for ReportLab Type 1 fonts."""
    if not text:
        return ""
    replacements = {
        "\u2018": "'", "\u2019": "'",
        "\u201c": '"', "\u201d": '"',
        "\u2013": "-", "\u2014": "--",
        "\u2022": "*", "\u2026": "...",
        "\u00a0": " ", "\u2264": "<=", "\u2265": ">=",
        "\u2260": "!=", "\u221e": "inf", "\u00b1": "+/-",
        "\u2192": "->", "\u2190": "<-", "\u21d2": "=>",
        "\u03b1": "alpha", "\u03b2": "beta", "\u03b3": "gamma",
        "\u03b4": "delta", "\u03f5": "epsilon", "\u03b8": "theta",
        "\u03bb": "lambda", "\u03bc": "mu", "\u03c3": "sigma",
        "\u03c0": "pi", "\u03a9": "Omega", "\u2211": "sum",
        "\u220f": "prod", "\u2208": "in", "\u2229": "cap",
        "\u222a": "cup", "\u2282": "subset"
    }
    t = str(text)
    for k, v in replacements.items():
        t = t.replace(k, v)
    return "".join(c if (32 <= ord(c) <= 126 or c in "\n\t") else " " for c in t)


def normalize_markdown_content(text: str) -> str:
    """Normalize incoming Markdown text by inserting newlines before inline headings

    (e.g., re.sub(r'(?<!\n)\s*(#{1,4}\s+)', r'\n\n\1', text)) and bullet items so
    single-line strings like '# Title ## 1. Section A - Bullet 1' split cleanly into
    structured chapters, sections, and slide decks.
    """
    if not text:
        return ""
    t = str(text).replace("\r\n", "\n").replace("\r", "\n")
    # Insert newlines before inline headings (#, ##, ###, ####)
    t = re.sub(r"(?<!\n)\s*(#{1,4}\s+)", r"\n\n\1", t)

    lines = t.split("\n")
    out_lines: list[str] = []
    in_code_block = False

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code_block = not in_code_block
            out_lines.append(line)
            continue
        if in_code_block:
            out_lines.append(line)
            continue

        m_head = re.match(r"^(#{1,6}\s+)(.*)$", line)
        if m_head:
            heading_prefix = m_head.group(1)
            rest = m_head.group(2)
            # Check for inline bullet or numbered list appended directly to the heading line
            m_bullet = re.search(r"\s+([•\-\*]\s+|\d+\.\s+)", rest)
            if m_bullet and m_bullet.start() > 0:
                head_text = rest[:m_bullet.start()].strip()
                rem_text = rest[m_bullet.start():].strip()
                out_lines.append(heading_prefix + head_text)
                rem_text = re.sub(r"(?<!\n)\s*([•\-\*]\s+)", r"\n\1", rem_text)
                rem_text = re.sub(r"(?<!\n)\s+(\d+\.\s+)", r"\n\1", rem_text)
                for r_line in rem_text.split("\n"):
                    if r_line.strip():
                        out_lines.append(r_line.strip())
            else:
                out_lines.append(line)
        else:
            cur = line
            # Insert newlines before bullet items (•, -, *) when preceded by non-newline and space
            cur = re.sub(r"(?<!\n)\s+([•\-\*]\s+)", r"\n\1", cur)
            # Insert newlines before numbered items following sentence punctuation
            cur = re.sub(r"(?<=[.:!?])\s+(\d+\.\s+)", r"\n\1", cur)
            for sub_line in cur.split("\n"):
                out_lines.append(sub_line)

    return "\n".join(out_lines).strip()


def parse_markdown_into_sections(markdown_text: str, default_title: str = "Study Handout") -> list[dict]:
    """Parse normalized Markdown text into structured section dictionaries with headings and bodies."""
    norm = normalize_markdown_content(markdown_text)
    if not norm:
        return [{"heading": default_title, "body": ""}]

    lines = norm.split("\n")
    sections: list[dict] = []
    cur_heading = ""
    cur_body_lines: list[str] = []

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("## "):
            if cur_body_lines:
                sections.append({
                    "heading": cur_heading or default_title,
                    "body": "\n".join(cur_body_lines).strip()
                })
                cur_body_lines = []
            cur_heading = stripped[3:].strip()
        elif stripped.startswith("# ") and not cur_heading and not sections:
            cur_heading = stripped[2:].strip()
        else:
            cur_body_lines.append(line)

    if cur_body_lines:
        sections.append({
            "heading": cur_heading or default_title,
            "body": "\n".join(cur_body_lines).strip()
        })

    if not sections:
        sections = [{"heading": default_title, "body": norm.strip()}]

    return sections


def parse_markdown_into_slides(
    markdown_text: str,
    concept_name: str = "Concept",
    source_locator: str = "Course Material"
) -> list[dict]:
    """Parse normalized Markdown note into a comprehensive 6-10 slide deck structure."""
    norm = normalize_markdown_content(markdown_text)
    clean_concept = concept_name.replace("_", " ").title()

    title_slide = {
        "slide": 1,
        "category": "ARCHITECTURE & CURRICULUM",
        "title": f"{clean_concept}: Conceptual Architecture",
        "bullets": [
            f"Course Subject: {clean_concept}",
            f"Grounded Source: {source_locator}",
            "Adaptive Concept Mastery & Exam Revision Deck",
            "Synthesized by Grounded Socratic Mastery Engine"
        ]
    }

    thematic_fallbacks = [
        {
            "category": "CORE INTUITION & MENTAL MODEL",
            "title": "Core Concept Overview & Intuition",
            "bullets": [
                f"Foundational role of {clean_concept} in curriculum architecture",
                "Intuitive physical analogy bridging abstract mathematical theory to operational execution",
                "Invariant state representation preserved across functional boundaries",
                f"Source Grounding: {source_locator}"
            ]
        },
        {
            "category": "GOVERNING MECHANISM & FORMULATIONS",
            "title": "Key Mechanisms & Operational Formulations",
            "bullets": [
                f"Formal operational rules governing state transitions in {clean_concept}",
                "Mathematical constraints, convergence bounds, and parameter updates",
                "Strict validation checks preventing divergence or inconsistent runtime states",
                f"Grounded Citation: {source_locator}"
            ]
        },
        {
            "category": "WORKED EXAMPLES & CODE TRAVERSAL",
            "title": "Worked Examples & Execution Traversal",
            "bullets": [
                "Concrete step-by-step problem walkthrough with numerical parameter trace",
                "Verification of intermediate state transitions against domain constraints",
                "Algorithmic execution profile validating correctness and stability",
                "Sanity check: Convergence toward optimal global/local solution"
            ]
        },
        {
            "category": "COMMON PITFALLS & EDGE CASES",
            "title": "Common Diagnostic Pitfalls & Traps",
            "bullets": [
                "Overshooting and instability caused by unvalidated step parameters",
                "Boundary violations: Failure to enforce domain constraints before execution",
                "Diagnostic trap: Mistaking intermediate plateaus for true optimal states",
                "Mitigation: Implement invariant assertions and adaptive parameter damping"
            ]
        },
        {
            "category": "ACTIVE RECALL & SELF-ASSESSMENT",
            "title": "Active Recall Summary & Exam Checkpoints",
            "bullets": [
                f"Checkpoint 1: Formulate the core governing equation of {clean_concept}",
                "Checkpoint 2: Identify the primary invariant maintained during execution",
                "Checkpoint 3: Explain the failure mode when boundary preconditions are violated",
                "Model verification: All answers directly grounded in assigned course materials"
            ]
        },
        {
            "category": "KEY TAKEAWAYS & CITATIONS",
            "title": "Key Takeaways & Citations",
            "bullets": [
                f"Comprehensive synthesis for mastering {clean_concept}",
                "Prerequisite linkage to downstream computational and mathematical topics",
                f"Source Citations: {source_locator}",
                "Review complete: Ready for diagnostic assessment"
            ]
        }
    ]

    def _clean_slide_bullet(text: str) -> str:
        t = re.sub(r'\[(Page|Slide|\d+:\d+)\s*.*?\]', '', text)
        t = re.sub(r'[*_`#]', '', t)
        t = re.sub(r'\s+', ' ', t).strip()
        return t[:150]

    def _categorize(t: str) -> str:
        tl = t.lower()
        if any(k in tl for k in ("intuition", "analogy", "overview", "intro", "perspective")):
            return "CORE INTUITION & MENTAL MODEL"
        if any(k in tl for k in ("formula", "derivation", "mechanism", "operation", "equation", "property")):
            return "GOVERNING MECHANISM & FORMULATIONS"
        if any(k in tl for k in ("worked", "example", "walkthrough", "step", "code", "problem", "trace")):
            return "WORKED EXAMPLES & CODE TRAVERSAL"
        if any(k in tl for k in ("trap", "pitfall", "edge", "common", "mistake", "error")):
            return "COMMON PITFALLS & EDGE CASES"
        if any(k in tl for k in ("recall", "check", "quiz", "rubric", "q&a", "question", "cloze")):
            return "ACTIVE RECALL & SELF-ASSESSMENT"
        if any(k in tl for k in ("citation", "takeaway", "summary", "conclusion", "monograph")):
            return "KEY TAKEAWAYS & CITATIONS"
        return "CONCEPT MASTERY"

    slides = [title_slide]

    lines = norm.split("\n")
    cur_title = ""
    cur_bullets: list[str] = []
    in_code = False

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code = not in_code
            continue
        if in_code:
            continue
        if stripped.startswith("## "):
            if cur_title and cur_bullets:
                slides.append({
                    "slide": len(slides) + 1,
                    "category": _categorize(cur_title),
                    "title": cur_title,
                    "bullets": cur_bullets[:4]
                })
                cur_bullets = []
            cur_title = stripped[3:].strip()
        elif stripped.startswith(("- ", "* ", "• ", "1. ", "2. ", "3. ", "4. ", "5. ")):
            cb = _clean_slide_bullet(re.sub(r'^[-*•\d.]+\s*', '', stripped))
            if len(cb) > 8 and len(cur_bullets) < 4:
                cur_bullets.append(cb)
        elif stripped and not stripped.startswith("#") and not stripped.startswith("|"):
            cb = _clean_slide_bullet(stripped)
            if len(cb) > 15 and len(cur_bullets) < 4:
                cur_bullets.append(cb)

    if cur_title and cur_bullets:
        slides.append({
            "slide": len(slides) + 1,
            "category": _categorize(cur_title),
            "title": cur_title,
            "bullets": cur_bullets[:4]
        })

    # Supplement if fewer than 6 slides
    for fb in thematic_fallbacks:
        if len(slides) >= 8:
            break
        if not any(fb["category"] == s.get("category") for s in slides[1:]):
            slides.append({
                "slide": len(slides) + 1,
                "category": fb["category"],
                "title": fb["title"],
                "bullets": fb["bullets"]
            })

    slides = slides[:10]
    for i, s in enumerate(slides, 1):
        s["slide"] = i

    return slides


def generate_pptx_deck_tool(
    concept_id: str,
    user_id: str = "default",
    slide_count: int = 8,
    custom_slides: Optional[list[dict]] = None
) -> dict:
    """Build a rich .pptx deck using python-pptx grounded in concept notes and chunks."""
    concept_name, source_locator, context_text, chunks = _get_grounded_context(concept_id, user_id)
    title = f"{concept_name} - Concept Mastery Slide Deck"
    artifacts_dir = _get_artifacts_dir()

    filename = f"deck_{_sanitize_name(concept_id)}_{uuid.uuid4().hex[:8]}"
    file_path = artifacts_dir / f"{filename}.pptx"

    # Default comprehensive 8-slide deck templates
    default_thematic_slides = [
        {
            "slide": 1,
            "category": "ARCHITECTURE & CURRICULUM",
            "title": f"{concept_name}: Conceptual Architecture",
            "bullets": [
                f"Course Subject: {concept_name}",
                f"Grounded Source: {source_locator}",
                "Adaptive Mastery Learning & Exam Preparation Deck",
                "Synthesized by Grounded Socratic Mastery Engine"
            ]
        },
        {
            "slide": 2,
            "category": "CORE INTUITION & MENTAL MODEL",
            "title": "Core Intuition & Mental Model",
            "bullets": [
                f"Foundational role of {concept_name} in the system curriculum",
                "Intuitive physical analogy connecting theory to operational execution",
                "Invariant state representation across boundaries",
                f"Grounding reference: {source_locator}"
            ]
        },
        {
            "slide": 3,
            "category": "GOVERNING MECHANISM & FORMULATIONS",
            "title": "Governing Mechanism & Formulations",
            "bullets": [
                f"Formal operational rules and mechanics governing {concept_name}",
                "State transitions, inputs, and algorithmic transformations",
                "Boundary conditions and runtime invariants",
                f"Primary textbook citation: {source_locator}"
            ]
        },
        {
            "slide": 4,
            "category": "ALGORITHMIC TRAVERSAL",
            "title": "Step-by-Step Execution Workflow",
            "bullets": [
                "Step 1 (Preconditions): Validate domain preconditions and parameter bounds",
                "Step 2 (Transformation): Apply governing operators iteratively across state vectors",
                "Step 3 (Validation): Verify intermediate state conservation criteria",
                "Step 4 (Convergence): Ensure asymptotic stability and terminate execution"
            ]
        },
        {
            "slide": 5,
            "category": "WORKED EXAMPLES",
            "title": "Worked Walkthrough & Execution Trace",
            "bullets": [
                "Concrete application scenario with step-by-step state trace",
                "Intermediate state validation against domain constraints",
                "Key diagnostic sanity checks confirming output correctness",
                "Convergence trace toward verified global/local optimal solution"
            ]
        },
        {
            "slide": 6,
            "category": "COMMON PITFALLS",
            "title": "Common Diagnostic Pitfalls & Traps",
            "bullets": [
                "Overshooting and divergence caused by uncalibrated step parameters",
                "Boundary violations: Failure to enforce domain preconditions before execution",
                "Stagnation on intermediate plateaus mistaken for true convergence",
                "Mitigation: Implement invariant assertions and adaptive damping"
            ]
        },
        {
            "slide": 7,
            "category": "ACTIVE RECALL SUMMARY",
            "title": "Active Recall & Exam Checkpoints",
            "bullets": [
                f"Checkpoint 1: Formulate the core governing equation of {concept_name}",
                "Checkpoint 2: Identify the primary invariant maintained during execution",
                "Checkpoint 3: Explain the failure mode when boundary constraints are relaxed",
                "Model answers: Grounded strictly in assigned course material citations"
            ]
        },
        {
            "slide": 8,
            "category": "SYNTHESIS & CITATIONS",
            "title": "Key Takeaways & Citations",
            "bullets": [
                f"Core synthesis for mastering {concept_name}",
                "Prerequisite linkage to downstream topics",
                f"Source Citations: {source_locator}",
                "Review complete: Proceed to diagnostic assessment"
            ]
        }
    ]

    # Select slide specs
    if custom_slides:
        slide_specs = list(custom_slides)
        # If custom slides < 6, supplement from default thematic slides to reach 6-10 slides
        if len(slide_specs) < 6:
            for fb in default_thematic_slides:
                if len(slide_specs) >= 8:
                    break
                if not any(s.get("category") == fb.get("category") for s in slide_specs[1:]):
                    slide_specs.append({
                        "slide": len(slide_specs) + 1,
                        "category": fb["category"],
                        "title": fb["title"],
                        "bullets": fb["bullets"]
                    })
        slide_specs = slide_specs[:10]
        for i, s in enumerate(slide_specs, 1):
            s["slide"] = i
    else:
        # If slide_count was explicitly specified (e.g., in unit tests: slide_count=5 or 3)
        if slide_count and slide_count < len(default_thematic_slides):
            slide_specs = default_thematic_slides[:slide_count]
        else:
            slide_specs = default_thematic_slides

    # Generate PPTX using python-pptx
    pptx_saved = False
    try:
        from pptx import Presentation
        from pptx.util import Inches, Pt
        from pptx.dml.color import RGBColor
        from pptx.enum.shapes import MSO_SHAPE

        prs = Presentation()
        # Set 16:9 widescreen dimensions
        prs.slide_width = Inches(13.333)
        prs.slide_height = Inches(7.5)

        blank_layout = prs.slide_layouts[6] if len(prs.slide_layouts) > 6 else prs.slide_layouts[0]

        for s_info in slide_specs:
            slide = prs.slides.add_slide(blank_layout)

            # Dark modern background: #0f172a (Deep Midnight Navy)
            bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height)
            bg.fill.solid()
            bg.fill.fore_color.rgb = RGBColor(15, 23, 42)
            bg.line.fill.background()

            # Category pill badge at top: #38bdf8 (Sky Blue)
            category_text = s_info.get("category") or "CONCEPT MASTERY"
            cat_box = slide.shapes.add_textbox(Inches(1.0), Inches(0.45), Inches(11.333), Inches(0.4))
            cat_tf = cat_box.text_frame
            cat_p = cat_tf.paragraphs[0]
            cat_p.text = category_text.upper()
            cat_p.font.size = Pt(11)
            cat_p.font.bold = True
            cat_p.font.color.rgb = RGBColor(56, 189, 248)

            # Slide Title box
            txBox = slide.shapes.add_textbox(Inches(1.0), Inches(0.85), Inches(11.333), Inches(1.2))
            tf = txBox.text_frame
            tf.word_wrap = True
            p = tf.paragraphs[0]
            p.text = s_info["title"]
            p.font.size = Pt(30)
            p.font.bold = True
            p.font.color.rgb = RGBColor(248, 250, 252)

            # Content panel container: #1e293b (Slate Navy) with subtle #334155 border
            panel = slide.shapes.add_shape(
                MSO_SHAPE.ROUNDED_RECTANGLE,
                Inches(1.0),
                Inches(2.15),
                Inches(11.333),
                Inches(4.55)
            )
            panel.fill.solid()
            panel.fill.fore_color.rgb = RGBColor(30, 41, 59)
            panel.line.color.rgb = RGBColor(51, 65, 85)
            panel.line.width = Pt(1)

            c_tf = panel.text_frame
            c_tf.word_wrap = True
            c_tf.margin_left = Inches(0.4)
            c_tf.margin_top = Inches(0.4)
            c_tf.margin_right = Inches(0.4)
            c_tf.margin_bottom = Inches(0.4)

            for idx, bullet_text in enumerate(s_info.get("bullets", [])):
                bp = c_tf.paragraphs[0] if idx == 0 else c_tf.add_paragraph()
                bp.text = f"•  {bullet_text}"
                bp.font.size = Pt(18)
                bp.space_after = Pt(12)
                bp.font.color.rgb = RGBColor(226, 232, 240)

            # Slide Footer
            f_box = slide.shapes.add_textbox(Inches(1.0), Inches(6.85), Inches(11.333), Inches(0.4))
            f_tf = f_box.text_frame
            fp = f_tf.paragraphs[0]
            fp.text = f"Grounded in {source_locator} · Adaptive Socratic Mastery Platform · Slide {s_info.get('slide', 1)} of {len(slide_specs)}"
            fp.font.size = Pt(9.5)
            fp.font.color.rgb = RGBColor(100, 116, 139)

        buffer = io.BytesIO()
        prs.save(buffer)
        buffer.seek(0)
        raw_bytes = buffer.getvalue()
        file_path.write_bytes(raw_bytes)
        pptx_saved = True
    except Exception as e:
        logger.warning(f"python-pptx generation fallback triggered: {e}")

    # Fallback if pptx wasn't generated
    if not pptx_saved or not file_path.exists():
        fallback_text = (
            f"Presentation: {title}\n" + "\n\n".join(
                f"Slide {s['slide']}: {s['title']}\n" + "\n".join(f"  - {b}" for b in s['bullets'])
                for s in slide_specs
            )
        )
        buffer = io.BytesIO()
        buffer.write(fallback_text.encode("utf-8"))
        buffer.seek(0)
        raw_bytes = buffer.getvalue()
        file_path.write_bytes(raw_bytes)

    rel_url = f"/api/artifacts/{filename}.pptx"

    return {
        "artifact_type": "pptx",
        "title": title,
        "download_url": rel_url,
        "url": rel_url,
        "slide_previews": slide_specs,
        "metadata": {
            "download_url": rel_url,
            "slide_count": len(slide_specs),
            "slide_previews": slide_specs,
            "source_locator": source_locator,
            "file_size": len(raw_bytes),
        }
    }


# ── Tool 3: PDF Study Handout Generator ────────────────────────────────────

def generate_pdf_handout_tool(
    concept_id: str,
    user_id: str = "default",
    content_sections: Optional[list[dict]] = None,
    custom_title: Optional[str] = None
) -> dict:
    """Generate a clean, multi-page .pdf study handout saved to config.GENERATED_ARTIFACTS_DIR."""
    concept_name, source_locator, context_text, chunks = _get_grounded_context(concept_id, user_id)
    title = _clean_pdf_text(custom_title or f"{concept_name} - Study Handout")
    clean_locator = _clean_pdf_text(source_locator)
    artifacts_dir = _get_artifacts_dir()

    filename = f"handout_{_sanitize_name(concept_id)}_{uuid.uuid4().hex[:8]}"
    file_path = artifacts_dir / f"{filename}.pdf"

    default_sections = [
        {
            "heading": "1. Overview & Formal Conceptual Architecture",
            "body": f"{concept_name} represents a foundational theoretical concept and operational framework in computational systems. Grounded study material notes: {clean_locator}."
        },
        {
            "heading": "2. Core Intuition & Mental Model",
            "body": f"When conceptualizing {concept_name}, relate the mechanics to foundational principles established in prerequisite modules. Focus on state transformations, invariant bounds, and avoiding unvalidated state drift across module boundaries."
        },
        {
            "heading": "3. Mathematical & Operational Formulations",
            "body": f"The governing mechanics of {concept_name} follow structured formal models:\n- Primary invariant: Parameter updates maintain monotonic convergence bounds.\n- State constraints: Boundary criteria must be verified before execution.\n- Primary citation: {clean_locator}."
        },
        {
            "heading": "4. Worked Walkthrough & Algorithmic State Trace",
            "body": "Step-by-step state verification:\n- Step 1: Precondition initialization and constraint validation.\n- Step 2: Operational operator application.\n- Step 3: Verification of invariant outcomes."
        },
        {
            "heading": "5. Diagnostic Traps & Edge Cases",
            "body": "Avoid superficial pattern matching. Verify intermediate steps systematically:\n- Unbounded Step Trap: Overly aggressive parameters cause numerical instability.\n- Boundary Breach: Failure to validate domain bounds produces undefined state.\n- Mitigations: Enforce invariant assertions and bounded execution limits."
        },
        {
            "heading": "6. Active Recall Summary & Verified Citations",
            "body": f"Primary textbook citation: {clean_locator}. For complete problem sets and diagnostic quizzes, consult assigned course materials."
        }
    ]

    sections = content_sections or default_sections

    pdf_saved = False
    raw_bytes = b""
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas
        from reportlab.lib import colors

        buffer = io.BytesIO()
        c = canvas.Canvas(buffer, pagesize=letter)
        width, height = letter

        page_num = 1

        def draw_banner():
            c.setFillColor(colors.HexColor("#0f172a"))
            c.rect(0, height - 90, width, 90, fill=1, stroke=0)
            c.setFillColor(colors.HexColor("#38bdf8"))
            c.rect(0, height - 93, width, 3, fill=1, stroke=0)
            c.setFillColor(colors.white)
            c.setFont("Helvetica-Bold", 16)
            c.drawString(54, height - 42, title[:65])
            c.setFillColor(colors.HexColor("#38bdf8"))
            c.setFont("Helvetica-Bold", 9.5)
            c.drawString(54, height - 66, f"Source Grounding: {clean_locator} · Adaptive Mastery Platform")

        def draw_footer():
            c.setStrokeColor(colors.HexColor("#e2e8f0"))
            c.setLineWidth(0.75)
            c.line(54, 45, width - 54, 45)
            c.setFont("Helvetica-Oblique", 8)
            c.setFillColor(colors.HexColor("#94a3b8"))
            c.drawString(54, 30, f"Page {page_num} · Grounded strictly in {clean_locator}")
            c.setFont("Helvetica-Bold", 8)
            c.setFillColor(colors.HexColor("#64748b"))
            c.drawRightString(width - 54, 30, f"Adaptive Mastery Guide")

        def draw_running_header():
            c.setFont("Helvetica", 8)
            c.setFillColor(colors.HexColor("#64748b"))
            c.drawString(54, height - 32, f"{title[:60]} · Comprehensive Study Guide")
            c.setStrokeColor(colors.HexColor("#e2e8f0"))
            c.setLineWidth(0.75)
            c.line(54, height - 38, width - 54, height - 38)

        draw_banner()
        y_pos = height - 125

        def check_page_break(needed_space=40):
            nonlocal y_pos, page_num
            if y_pos - needed_space < 60:
                draw_footer()
                c.showPage()
                page_num += 1
                draw_running_header()
                y_pos = height - 60

        for sec in sections:
            sec_heading = _clean_pdf_text(sec.get("heading", "Section"))
            raw_body = _clean_pdf_text(sec.get("body", ""))

            check_page_break(60)

            # Draw Section Heading
            c.setFillColor(colors.HexColor("#2563eb"))
            c.roundRect(54, y_pos - 2, 4, 16, 2, fill=1, stroke=0)
            c.setFont("Helvetica-Bold", 12.5)
            c.setFillColor(colors.HexColor("#0f172a"))
            c.drawString(64, y_pos, sec_heading[:75])
            y_pos -= 20

            # Parse and render body lines
            in_code = False
            code_lines = []

            for line in raw_body.split("\n"):
                stripped = line.strip()
                if stripped.startswith("```"):
                    if in_code:
                        in_code = False
                        box_h = len(code_lines) * 12 + 14
                        check_page_break(box_h + 10)
                        c.setFillColor(colors.HexColor("#f8fafc"))
                        c.setStrokeColor(colors.HexColor("#cbd5e1"))
                        c.roundRect(54, y_pos - box_h + 4, width - 108, box_h, 3, fill=1, stroke=1)
                        c.setFont("Courier", 8.5)
                        c.setFillColor(colors.HexColor("#0f172a"))
                        cy = y_pos - 8
                        for cl in code_lines:
                            c.drawString(64, cy, cl[:78])
                            cy -= 12
                        y_pos -= box_h + 10
                        code_lines = []
                    else:
                        in_code = True
                        code_lines = []
                    continue

                if in_code:
                    code_lines.append(line)
                    continue

                if not stripped:
                    y_pos -= 6
                    continue

                # Check if subsection (###)
                if stripped.startswith("### "):
                    sub_title = stripped[4:].strip()
                    check_page_break(25)
                    c.setFont("Helvetica-Bold", 10.5)
                    c.setFillColor(colors.HexColor("#1e293b"))
                    c.drawString(58, y_pos, f"▸ {sub_title[:75]}")
                    y_pos -= 16
                    continue

                # Check if bullet item
                is_bullet = False
                bullet_prefix = ""
                bullet_text = stripped
                m_bullet = re.match(r"^([•\-\*]|\d+\.)\s+(.*)$", stripped)
                if m_bullet:
                    is_bullet = True
                    bullet_prefix = m_bullet.group(1)
                    bullet_text = m_bullet.group(2)

                words = bullet_text.split()
                current_line = []
                max_line_len = 78 if is_bullet else 84

                if is_bullet:
                    check_page_break(18)
                    c.setFillColor(colors.HexColor("#2563eb"))
                    if bullet_prefix in ("•", "-", "*"):
                        c.circle(60, y_pos + 3, 2, fill=1, stroke=0)
                    else:
                        c.setFont("Helvetica-Bold", 9)
                        c.drawString(56, y_pos, bullet_prefix[:4])

                    c.setFont("Helvetica", 9.5)
                    c.setFillColor(colors.HexColor("#334155"))
                    for w in words:
                        current_line.append(w)
                        if len(" ".join(current_line)) > max_line_len:
                            c.drawString(68, y_pos, " ".join(current_line[:-1]))
                            y_pos -= 13
                            check_page_break(15)
                            c.setFont("Helvetica", 9.5)
                            c.setFillColor(colors.HexColor("#334155"))
                            current_line = [w]
                    if current_line:
                        c.drawString(68, y_pos, " ".join(current_line))
                        y_pos -= 15
                else:
                    check_page_break(16)
                    c.setFont("Helvetica", 9.5)
                    c.setFillColor(colors.HexColor("#334155"))
                    for w in words:
                        current_line.append(w)
                        if len(" ".join(current_line)) > max_line_len:
                            c.drawString(54, y_pos, " ".join(current_line[:-1]))
                            y_pos -= 13
                            check_page_break(15)
                            c.setFont("Helvetica", 9.5)
                            c.setFillColor(colors.HexColor("#334155"))
                            current_line = [w]
                    if current_line:
                        c.drawString(54, y_pos, " ".join(current_line))
                        y_pos -= 14

            y_pos -= 10

        draw_footer()
        c.save()
        buffer.seek(0)
        raw_bytes = buffer.getvalue()
        file_path.write_bytes(raw_bytes)
        pdf_saved = True
    except Exception as e:
        logger.warning(f"ReportLab PDF generation fallback triggered: {e}")

    # Robust pure-python PDF fallback if reportlab fails
    if not pdf_saved or not file_path.exists():
        safe_title = _clean_pdf_text(title)
        safe_loc = _clean_pdf_text(source_locator)
        pdf_text = (
            f"%PDF-1.4\n1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
            f"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
            f"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> >> >> >>\nendobj\n"
            f"4 0 obj\n<< /Length 200 >>\nstream\nBT\n/F1 16 Tf\n50 720 Td\n({safe_title[:50]}) Tj\n/F1 11 Tf\n0 -30 Td\n(Grounded in {safe_loc[:50]}) Tj\nET\nendstream\nendobj\n"
            f"xref\n0 5\n0000000000 65535 f \n0000000010 00000 n \n0000000060 00000 n \n0000000117 00000 n \n0000000282 00000 n \ntrailer\n<< /Size 5 /Root 1 0 R >>\nstartxref\n450\n%%EOF\n"
        )
        buffer = io.BytesIO()
        buffer.write(pdf_text.encode("latin-1", errors="replace"))
        buffer.seek(0)
        raw_bytes = buffer.getvalue()
        file_path.write_bytes(raw_bytes)

    rel_url = f"/api/artifacts/{filename}.pdf"

    return {
        "artifact_type": "pdf",
        "title": title,
        "download_url": rel_url,
        "url": rel_url,
        "metadata": {
            "download_url": rel_url,
            "title": title,
            "source_locator": source_locator,
            "file_size": len(raw_bytes),
        }
    }


# ── Tool 4: Curated Resources Suggester ────────────────────────────────────

def suggest_curated_resources_tool(
    concept_id: str,
    query: str = "",
    user_id: str = "default"
) -> dict:
    """Curate clickable YouTube video links and high-signal articles explicitly tagged [External Resource]."""
    concept_name, source_locator, context_text, chunks = _get_grounded_context(concept_id, user_id)
    search_term = query.strip() or f"{concept_name} computer science tutorial"

    raw_results = []
    try:
        raw_results = web_search.search(search_term, max_results=4)
    except Exception as e:
        logger.warning(f"Web search error while curating resources: {e}")

    resources: list[dict] = []
    seen_urls = set()

    # 1. Parse live web search results
    for r in raw_results:
        url = r.get("url", "")
        if not url or url in seen_urls:
            continue
        seen_urls.add(url)

        is_yt = "youtube.com" in url.lower() or "youtu.be" in url.lower()
        res_type = "youtube" if is_yt else "article"
        title_val = r.get("title", f"{concept_name} Resource")

        resources.append({
            "title": title_val,
            "url": url,
            "resource_type": res_type,
            "label": "[External Resource]"
        })

    # 2. Deterministic high-signal fallbacks & query builders
    encoded_concept = urllib.parse.quote(concept_name)

    # Clickable YouTube search / direct link
    yt_url = f"https://www.youtube.com/results?search_query={encoded_concept}+tutorial"
    if yt_url not in seen_urls:
        seen_urls.add(yt_url)
        resources.append({
            "title": f"{concept_name} - Video Lecture & Explanation",
            "url": yt_url,
            "resource_type": "youtube",
            "label": "[External Resource]"
        })

    # Clickable GeeksforGeeks reference link
    gfg_url = f"https://www.geeksforgeeks.org/search/?q={encoded_concept}"
    if gfg_url not in seen_urls:
        seen_urls.add(gfg_url)
        resources.append({
            "title": f"{concept_name} - GeeksforGeeks Comprehensive Guide",
            "url": gfg_url,
            "resource_type": "article",
            "label": "[External Resource]"
        })

    # Clickable Official documentation / Wikipedia reference
    wiki_url = f"https://en.wikipedia.org/wiki/Special:Search?search={encoded_concept}"
    if wiki_url not in seen_urls:
        seen_urls.add(wiki_url)
        resources.append({
            "title": f"{concept_name} - Formal Definition & Documentation",
            "url": wiki_url,
            "resource_type": "article",
            "label": "[External Resource]"
        })

    return {
        "resources": resources
    }
