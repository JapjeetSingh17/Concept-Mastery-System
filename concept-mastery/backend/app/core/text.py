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
use used using called often usually example""".split())


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

def tokens(text: str) -> list[str]:
    return [stem(w) for w in re.findall(r"[a-z][a-z0-9]+", normalize(text)) if w not in STOP and len(w) > 2]

def sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"“(])", text) if len(s.strip()) > 2]
