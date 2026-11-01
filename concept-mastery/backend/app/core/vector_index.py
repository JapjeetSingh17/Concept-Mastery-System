"""Vector index with page metadata. Sparse TF-IDF vectors + cosine similarity, no external model needed.
To use dense embeddings later, replace _vec() and search() (keep the chunk metadata)."""
import math
from collections import Counter
from .text import tokens

class VectorIndex:
    def __init__(self, chunks=None, idf=None, vecs=None):
        self.chunks, self.idf, self.vecs = chunks or [], idf or {}, vecs or []

    @classmethod
    def build(cls, chunks):
        docs = [Counter(tokens(c["section"] + " " + c["text"])) for c in chunks]
        df = Counter(t for d in docs for t in d)
        n = len(docs)
        idf = {t: math.log((1 + n) / (1 + f)) + 1 for t, f in df.items()}
        return cls(chunks, idf, [cls._vec(d, idf) for d in docs])

    @staticmethod
    def _vec(counts, idf):
        v = {t: (1 + math.log(c)) * idf[t] for t, c in counts.items() if t in idf}
        norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        return {t: x / norm for t, x in v.items()}

    def search(self, query, k=4):
        q = self._vec(Counter(tokens(query)), self.idf)
        scored = sorted(((sum(w * v.get(t, 0.0) for t, w in q.items()), i) for i, v in enumerate(self.vecs)), reverse=True)
        return [(s, self.chunks[i]) for s, i in scored[:k] if s > 0]

    def to_dict(self):
        return {"chunks": self.chunks, "idf": self.idf, "vecs": self.vecs}

    @classmethod
    def from_dict(cls, d):
        return cls(d["chunks"], d["idf"], d["vecs"])
