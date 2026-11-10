"""Synonym Normalizer + Tokenizer + Fuzzy Matcher for Vectorless RAG."""

from __future__ import annotations

import json
import re
from pathlib import Path
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


def tokens(text: str) -> list[str]:
    return [stem(w) for w in re.findall(r"[a-z][a-z0-9]+", normalize(text)) if w not in STOP and w not in GENERIC_NON_CONCEPTS and len(w) > 2]

def sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"“(])", text) if len(s.strip()) > 2]
