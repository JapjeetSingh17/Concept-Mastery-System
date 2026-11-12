"""Vector index with page metadata. Sparse TF-IDF vectors + cosine similarity, no external model needed.
To use dense embeddings later, replace _vec() and search() (keep the chunk metadata)."""
from collections import Counter
from collections.abc import Callable
import math
from pathlib import Path
import threading

from .. import config
from .lru_cache import DEFAULT_LRU_MAXSIZE, DiskBackedLRUCache
from .text import tokens

MAX_ACTIVE_VECTOR_INDICES = DEFAULT_LRU_MAXSIZE


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


class VectorIndexCache:
    """Thread-safe bounded LRU cache for VectorIndex instances with disk eviction."""

    def __init__(
        self,
        maxsize: int = MAX_ACTIVE_VECTOR_INDICES,
        cache_dir: Path | Callable[[], Path] | None = None,
    ) -> None:
        resolved_dir = cache_dir if cache_dir is not None else (lambda: config.VECTOR_STORE_PATH / "vector_cache")
        self._lock = threading.RLock()
        self._store = DiskBackedLRUCache(
            cache_dir=resolved_dir,
            maxsize=maxsize,
            serializer=lambda idx: idx.to_dict() if isinstance(idx, VectorIndex) else idx,
            deserializer=lambda data: VectorIndex.from_dict(data) if isinstance(data, dict) else data,
        )

    @property
    def maxsize(self) -> int:
        return self._store.maxsize

    @staticmethod
    def make_key(user_id: str = "default", material_id: str = "unified") -> str:
        return f"u:{user_id}|m:{material_id}"

    def get(self, key: str) -> VectorIndex | None:
        return self._store.get(key)

    def put(self, key: str, index: VectorIndex) -> VectorIndex:
        self._store.put(key, index)
        return index

    def get_or_build(self, key: str, chunks: list[dict]) -> VectorIndex:
        with self._lock:
            cached = self._store.get(key)
            if cached is not None:
                return cached
            index = VectorIndex.build(chunks)
            self._store.put(key, index)
            return index

    def delete(self, key: str) -> bool:
        return self._store.delete(key)

    def invalidate_user(self, user_id: str) -> int:
        return self._store.invalidate_prefix(f"u:{user_id}|")

    def clear(self) -> None:
        self._store.clear()

    def in_memory_count(self) -> int:
        return self._store.in_memory_count()

    def disk_count(self) -> int:
        return self._store.disk_count()

    def __contains__(self, key: object) -> bool:
        return key in self._store

    def __len__(self) -> int:
        return len(self._store)


GLOBAL_VECTOR_CACHE = VectorIndexCache()
