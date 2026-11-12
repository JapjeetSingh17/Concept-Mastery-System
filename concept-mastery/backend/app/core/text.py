"""Synonym Normalizer + Tokenizer + Fuzzy Matcher for Vectorless RAG."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable, Optional
from rapidfuzz import fuzz, process as rf_process
from .. import config

STOP = set("""a an the of to in on for and or is are was were be been being it its this that these those with as by at from
which can has have had not but if then so such than into also when where how what why who whom we you they their our your
more most one two do does did will would should could may might about over under between each other any all some very just
use used using called often usually example get gets got getting make makes made making know knows known like see seen
take takes took taken come comes came coming go goes went gone want wants say says said look looks looked tell tells told
well really actually basically simply literally thing things stuff way ways stands stand standing
because since although though unless until while where whereas whether neither nor either
going creating asking understand understanding small number insert create created creates ask asked asks
talking talked talk put puts putting calling call called work working worked show showing showed shown
find finding found keep keeping kept try trying tried need needed needs think thinking thought
guys guy helper helpers document documents paragraphs paragraph reader read reading extract extracting text pdf pptx docx txt mp3 mp4 wav m4a viewer viewers
model models project projects test tests input inputs output outputs
here there now then today tomorrow yesterday maybe perhaps okay ok yeah wait let lets
once again""".split())

GENERIC_NON_CONCEPTS = set("""
segment video lecture voice note section chapter page slide unit overview introduction summary conclusion
timestamp locator question answer problem solution exercise test quiz practice exam rubric score grade
review study learn recall intuition mechanism formula worked essential keyword keywords update logical
physical primary secondary system management revision guarantee storage instructor student teacher class
method function object value result output input process code file data information detail aspect topic
subject case point type kind part step stage phase level state time timer start end begin finish first
second third next previous last core key main major minor simple complex easy hard good bad new old
different similar general specific important useful various several many much less least
going right because later already creating asking understand understanding small number insert
create created creates ask asked asks go goes went gone get gets got getting
want wants wanted saying say says said tell tells told look looks looked see seeing seen
here there now then today tomorrow yesterday maybe perhaps okay ok yeah wait let lets
talk talking talked come comes came coming put puts putting take takes taking took taken
call called calling make makes making made work working worked show showing showed shown
find finding found keep keeping kept try trying tried need needed needs think thinking thought
actually basically simply literally really once once again guys guy helper helpers
document documents paragraphs paragraph reader read reading extract extracting text pdf pptx docx txt mp3 mp4 wav m4a viewer viewers
model models project projects test tests example examples input inputs output outputs
bit lot lots kind parts ways things stuff line lines word words sentence sentences
""".split())


class SynonymNormalizer:
    """Normalizes terms, synonyms, and textbook aliases to canonical concept IDs using rapidfuzz."""

    def __init__(self, synonyms_path: Path | None = None):
        path = synonyms_path or config.SYNONYMS_FILE
        self.synonyms: dict[str, str] = {}
        if path.exists():
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
                self.synonyms = {k.lower().strip(): v.lower().strip() for k, v in raw.items()}
            except Exception:
                self.synonyms = {}

        # Canonical forms map to themselves
        self.canonical_set: set[str] = set(self.synonyms.values())
        for c in self.canonical_set:
            self.synonyms.setdefault(c, c)

        # Precompiled regex replacements for fast exact boundary match
        self._regex_rules = [
            (re.compile(r"\b" + re.escape(k) + r"\b", re.I), v)
            for k, v in sorted(self.synonyms.items(), key=lambda kv: -len(kv[0]))
        ]

    def normalize_text(self, text: str) -> str:
        """Apply exact dictionary substitutions across a block of text."""
        t = text.lower()
        for rx, v in self._regex_rules:
            t = rx.sub(v, t)
        return t

    def canonicalize(self, raw_term: str, score_cutoff: float = 85.0) -> str | None:
        """Map raw student term or textbook alias to canonical concept name."""
        low = raw_term.strip().lower()
        if not low:
            return None

        # 1. Exact lookup
        if low in self.synonyms:
            return self.synonyms[low]

        # 2. Fuzzy token-sort ratio match
        choices = list(self.synonyms.keys())
        if choices:
            match = rf_process.extractOne(low, choices, scorer=fuzz.token_sort_ratio, score_cutoff=score_cutoff)
            if match:
                matched_key = match[0]
                return self.synonyms[matched_key]

        return None

    def detect_canonical_concepts(self, text: str, known_concepts: list[str]) -> list[str]:
        """Detect which canonical concepts are present in text via normalization and fuzzy matching."""
        if not text or not known_concepts:
            return []

        norm_text = self.normalize_text(text)
        detected: set[str] = set()

        for c in known_concepts:
            base = c.lower().strip()
            # Boundary search for singular and plural
            pat = re.compile(r"\b" + re.escape(base.rstrip("s")) + r"s?\b", re.I)
            if pat.search(norm_text):
                detected.add(c)

        # Multi-word n-gram fuzzy extraction
        words = re.findall(r"[a-z0-9\-]+", norm_text)
        remaining = [c for c in known_concepts if c not in detected]
        if remaining and len(words) >= 2:
            ngrams = set()
            for n in (2, 3):
                for i in range(len(words) - n + 1):
                    g = " ".join(words[i:i + n])
                    if len(g) >= 5:
                        ngrams.add(g)

            for g in ngrams:
                match = rf_process.extractOne(g, remaining, scorer=fuzz.token_sort_ratio, score_cutoff=88.0)
                if match:
                    detected.add(match[0])
                    remaining = [c for c in known_concepts if c not in detected]
                    if not remaining:
                        break

        return sorted(detected)


# Global singleton normalizer
NORMALIZER = SynonymNormalizer()
SYN = NORMALIZER.synonyms


# Utility functions for backward compatibility
def normalize(text: str) -> str:
    return NORMALIZER.normalize_text(text)

def normalize_term(raw_term: str) -> str | None:
    return NORMALIZER.canonicalize(raw_term)

def detect_concepts_in_text(text: str, concept_names: list[str]) -> list[str]:
    return NORMALIZER.detect_canonical_concepts(text, concept_names)

def stem(w: str) -> str:
    for suf in ("ing", "ed", "es", "s"):
        if w.endswith(suf) and len(w) - len(suf) >= 4:
            return w[: -len(suf)]
    return w

def is_generic_term(term: str) -> bool:
    """Check if a term or multi-word phrase is composed of generic words, conversational fillers, or locators."""
    if not term or not term.strip():
        return True
    t_clean = term.strip()
    if re.search(r"^(?:lecture\s+)?segment\b", t_clean, re.I):
        return True
    if re.search(r"^video\s*\[", t_clean, re.I):
        return True
    if re.search(r"^voice\s+note\b", t_clean, re.I):
        return True
    if re.search(r"^\d{1,2}:\d{2}(?::\d{2})?$", t_clean):
        return True
    if re.search(r"^(?:page|slide|section|unit|chapter)\s*\d*$", t_clean, re.I):
        return True

    parts = [p.strip() for p in re.findall(r"[a-z0-9]+", t_clean.lower())]
    if not parts:
        return True
    # If all parts are digits or in STOP or GENERIC_NON_CONCEPTS, it's generic
    if all(p.isdigit() or p in STOP or p in GENERIC_NON_CONCEPTS for p in parts):
        return True

    # If single word: strictly block generic verbs, adverbs, pronouns, adjectives, conversational fillers
    if len(parts) == 1:
        w = parts[0]
        if w.isdigit() or len(w) < 4 or w in STOP or w in GENERIC_NON_CONCEPTS:
            return True
        # Reject common verb forms ending in -ing unless in technical allowlist
        if w.endswith("ing") and w not in {
            "embedding", "clustering", "pipelining", "caching", "routing",
            "encoding", "indexing", "ranking", "training", "sampling",
            "backtracking", "threading", "parsing", "branching", "pruning"
        }:
            return True
        # Reject common past-tense/participle forms unless technical
        if w.endswith("ed") and w not in {"feedforward", "embedded", "distributed", "normalized", "quantized"}:
            return True

    # Multi-word phrases: if it starts with conversational starters or consists mostly of fillers
    if parts[0] in {"once", "let", "lets", "how", "why", "what", "when", "where", "who", "which", "tell", "show", "talk", "start", "stop"}:
        return True
    if any(p in {"guys", "again", "helper", "helpers", "wait"} for p in parts):
        return True

    return False


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"“(])")
_CTX_SENTENCE_BREAK_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"“(\\`$])")


def _is_escaped(text: str, idx: int) -> bool:
    """Return True if text[idx] is preceded by an odd number of backslashes."""
    backslashes = 0
    j = idx - 1
    while j >= 0 and text[j] == "\\":
        backslashes += 1
        j -= 1
    return (backslashes % 2) == 1


def _find_fence_close(text: str, search_from: int, fence_char: str, fence_len: int) -> Optional[int]:
    """Locate the end index of a closing code fence with at least fence_len repeated chars."""
    n = len(text)
    j = search_from
    while j < n:
        if text[j] == fence_char:
            run_end = j + 1
            while run_end < n and text[run_end] == fence_char:
                run_end += 1
            if (run_end - j) >= fence_len:
                return run_end
            j = run_end
        else:
            j += 1
    return None


def _find_delimited_close(
    text: str,
    search_from: int,
    closing: str,
    stop_idx: int,
) -> Optional[int]:
    """Locate the end index of an unescaped closing delimiter before stop_idx."""
    clen = len(closing)
    j = search_from
    while j + clen <= stop_idx:
        if text[j:j + clen] == closing and not _is_escaped(text, j):
            return j + clen
        j += 1
    return None


def _find_protected_spans(text: str) -> list[tuple[int, int]]:
    """Return sorted non-overlapping [start, end) spans for code fences and LaTeX math blocks."""
    spans: list[tuple[int, int]] = []
    n = len(text)
    i = 0

    while i < n:
        ch = text[i]

        # 1. Backtick code blocks: ```...``` fenced blocks and inline `...` code spans
        if ch == "`" and not _is_escaped(text, i):
            run_end = i + 1
            while run_end < n and text[run_end] == "`":
                run_end += 1
            tick_len = run_end - i

            if tick_len >= 3:
                close_end = _find_fence_close(text, run_end, "`", tick_len)
                end_idx = close_end if close_end is not None else n
                spans.append((i, end_idx))
                i = end_idx
                continue

            para_end = text.find("\n\n", run_end)
            if para_end == -1:
                para_end = n
            j = run_end
            matched_end: Optional[int] = None
            while j < para_end:
                if text[j] == "`":
                    cand_end = j + 1
                    while cand_end < para_end and text[cand_end] == "`":
                        cand_end += 1
                    if (cand_end - j) == tick_len:
                        matched_end = cand_end
                        break
                    j = cand_end
                else:
                    j += 1
            if matched_end is not None:
                spans.append((i, matched_end))
                i = matched_end
                continue
            i = run_end
            continue

        # 2. Dollar LaTeX math delimiters: $$...$$ display blocks and $...$ inline math
        if ch == "$" and not _is_escaped(text, i):
            if i + 1 < n and text[i + 1] == "$":
                close_end = _find_delimited_close(text, i + 2, "$$", n)
                end_idx = close_end if close_end is not None else n
                spans.append((i, end_idx))
                i = end_idx
                continue

            para_end = text.find("\n\n", i + 1)
            if para_end == -1:
                para_end = n
            j = i + 1
            close_end = None
            while j < para_end:
                if text[j] == "$" and not _is_escaped(text, j):
                    # Do not allow an unmatched single $ to pair across an unescaped $$ display block
                    if j + 1 < para_end and text[j + 1] == "$":
                        break
                    close_end = j + 1
                    break
                j += 1
            if close_end is not None:
                spans.append((i, close_end))
                i = close_end
                continue

        # 3. Backslash KaTeX delimiters: \[...\] and \(...\)
        if ch == "\\" and i + 1 < n and not _is_escaped(text, i):
            pair = text[i:i + 2]
            if pair in ("\\[", "\\("):
                closer = "\\]" if pair == "\\[" else "\\)"
                stop_idx = n if pair == "\\[" else (text.find("\n\n", i + 2) if text.find("\n\n", i + 2) != -1 else n)
                close_end = _find_delimited_close(text, i + 2, closer, stop_idx)
                end_idx = close_end if close_end is not None else (n if pair == "\\[" else None)
                if end_idx is not None:
                    spans.append((i, end_idx))
                    i = end_idx
                    continue

        i += 1

    return spans


def _span_containing(spans: list[tuple[int, int]], idx: int) -> Optional[tuple[int, int]]:
    """Return the protected (start, end) span strictly containing idx, or None if safe to split at idx."""
    for s, e in spans:
        if s < idx < e:
            return (s, e)
        if s >= idx:
            break
    return None


def _best_safe_break(text: str, spans: list[tuple[int, int]], start: int, target_end: int) -> int:
    """Pick the cleanest natural split point in (start, target_end] outside all protected spans."""
    n = len(text)
    preferred_min = min(target_end, start + max(1, (target_end - start) // 3))

    for min_bound in (preferred_min, start + 1):
        if min_bound > target_end:
            continue

        # 1. Paragraph break (\n\n)
        pos = text.rfind("\n\n", min_bound, min(n, target_end + 2))
        while pos >= min_bound:
            if pos <= target_end and _span_containing(spans, pos) is None:
                return pos
            pos = text.rfind("\n\n", min_bound, pos)

        # 2. Line break (\n)
        pos = text.rfind("\n", min_bound, target_end + 1)
        while pos >= min_bound:
            if _span_containing(spans, pos) is None:
                return pos
            pos = text.rfind("\n", min_bound, pos)

        # 3. Sentence boundary
        sentence_cuts = [
            m.start()
            for m in _CTX_SENTENCE_BREAK_RE.finditer(text, max(start, min_bound - 2), min(n, target_end + 1))
            if min_bound <= m.start() <= target_end and _span_containing(spans, m.start()) is None
        ]
        if sentence_cuts:
            return sentence_cuts[-1]

        # 4. Whitespace boundary
        for idx in range(target_end, min_bound - 1, -1):
            if ((idx < n and text[idx].isspace()) or text[idx - 1].isspace()) and _span_containing(spans, idx) is None:
                return idx

    return target_end


def tokens(text: str) -> list[str]:
    return [stem(w) for w in re.findall(r"[a-z][a-z0-9]+", normalize(text)) if w not in STOP and w not in GENERIC_NON_CONCEPTS and len(w) > 2]


def sentences(text: str) -> list[str]:
    """Split text into sentences without breaking inside code fences or LaTeX math delimiters."""
    if not text:
        return []
    spans = _find_protected_spans(text)
    if not spans:
        return [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if len(s.strip()) > 2]

    parts: list[str] = []
    last = 0
    for m in _SENTENCE_SPLIT_RE.finditer(text):
        split_at = m.start()
        if _span_containing(spans, split_at) is not None:
            continue
        seg = text[last:split_at].strip()
        if len(seg) > 2:
            parts.append(seg)
        last = m.end()

    tail = text[last:].strip()
    if len(tail) > 2:
        parts.append(tail)
    return parts


def chunk_text_boundary_aware(
    text: str,
    max_chars: int = 1200,
    overlap_chars: int = 0,
) -> list[str]:
    """Split text into chunks up to max_chars without splitting inside code fences (```) or LaTeX math ($$...$$, $...$)."""
    if not text or not text.strip():
        return []

    stripped = text.strip()
    if max_chars <= 0 or len(stripped) <= max_chars:
        return [stripped]

    spans = _find_protected_spans(text)
    n = len(text)
    chunks: list[str] = []
    start = 0

    while start < n:
        while start < n and text[start].isspace():
            start += 1
        if start >= n:
            break

        target_end = start + max_chars
        if target_end >= n:
            piece = text[start:].strip()
            if piece:
                chunks.append(piece)
            break

        blocking = _span_containing(spans, target_end)
        if blocking is not None:
            s, e = blocking
            # Split cleanly right before the block if meaningful content precedes it;
            # otherwise keep the oversized block intact rather than splitting mid-stream.
            if text[start:s].strip():
                cut = s
            else:
                cut = e
        else:
            cut = _best_safe_break(text, spans, start, target_end)
            if cut <= start:
                cut = target_end

        piece = text[start:cut].strip()
        if piece:
            chunks.append(piece)

        if cut >= n:
            break

        if overlap_chars > 0:
            next_start = max(start + 1, cut - overlap_chars)
            overlap_block = _span_containing(spans, next_start)
            if overlap_block is not None:
                next_start = overlap_block[1]
            # Snap forward to word start if landed inside a word
            while next_start < cut and next_start > 0 and not text[next_start - 1].isspace() and not text[next_start].isspace():
                next_start += 1
            start = next_start if next_start > start else cut
        else:
            start = cut

    return chunks


def truncate_boundary_aware(text: str, max_chars: int = 220, suffix: str = "...") -> str:
    """Truncate text cleanly near max_chars without splitting inside code fences or LaTeX math blocks."""
    cleaned = (text or "").strip()
    if not cleaned or max_chars <= 0 or len(cleaned) <= max_chars:
        return cleaned

    spans = _find_protected_spans(cleaned)
    blocking = _span_containing(spans, max_chars)
    if blocking is not None:
        s, e = blocking
        if e <= max_chars * 2 or not cleaned[:s].strip():
            return cleaned[:e].rstrip() + (suffix if e < len(cleaned) else "")
        cut = s
    else:
        cut = _best_safe_break(cleaned, spans, 0, max_chars)

    prefix = cleaned[:cut].rstrip()
    if not prefix:
        if spans and spans[0][0] == 0:
            e = spans[0][1]
            return cleaned[:e].rstrip() + (suffix if e < len(cleaned) else "")
        return cleaned[:max_chars].rstrip() + suffix
    return prefix + suffix


def _extract_chunk_body(item: Any, body_getter: Optional[Callable[[Any], str]] = None) -> str:
    if body_getter is not None:
        return body_getter(item) or ""
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        return item.get("text") or item.get("content") or item.get("snippet") or ""
    return getattr(item, "text", "") or ""


def build_context_window(
    items: list[Any],
    max_chars: int = 6000,
    chunk_max_chars: int = 1800,
    formatter: Optional[Callable[[Any, str], str]] = None,
    body_getter: Optional[Callable[[Any], str]] = None,
    separator: str = "\n\n",
) -> str:
    """Assemble an LLM context window using boundary-aware chunking so math/code delimiters remain intact."""
    if not items or max_chars <= 0:
        return ""

    format_fn = formatter or (lambda _item, body: body)
    parts: list[str] = []
    used_chars = 0

    for item in items:
        raw_body = str(_extract_chunk_body(item, body_getter)).strip()
        if not raw_body:
            continue

        sep_len = len(separator) if parts else 0
        remaining = max_chars - used_chars - sep_len
        if remaining <= 0:
            break

        overhead = len(format_fn(item, ""))
        body_budget = remaining - overhead
        if body_budget <= 0 or (parts and body_budget < min(40, len(raw_body))):
            if parts:
                break
            body_budget = max_chars

        effective_limit = min(chunk_max_chars, body_budget) if chunk_max_chars > 0 else body_budget
        sub_chunks = chunk_text_boundary_aware(raw_body, max_chars=effective_limit)
        if not sub_chunks:
            continue

        chosen_subs: list[str] = []
        sub_used = 0
        for sub in sub_chunks:
            candidate_body = "\n\n".join(chosen_subs + [sub])
            candidate_block = format_fn(item, candidate_body)
            if len(candidate_block) <= remaining or (not parts and not chosen_subs):
                chosen_subs.append(sub)
                sub_used = len(candidate_block)
            else:
                break

        if not chosen_subs:
            break

        parts.append(format_fn(item, "\n\n".join(chosen_subs)))
        used_chars += sep_len + sub_used

    return separator.join(parts)
