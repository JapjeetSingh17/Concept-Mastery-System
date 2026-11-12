"""Thread-safe disk-backed bounded LRU cache for tenant/material indices.

Keeps at most ``maxsize`` active entries in RAM (default 64). When capacity is
exceeded, the least-recently-used entry is atomically serialized to disk under
``data/vectorstore/`` and transparently reloaded into RAM on subsequent access.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
from collections import OrderedDict
from collections.abc import Callable, Iterator, MutableMapping
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_LRU_MAXSIZE = 64


class DiskBackedLRUCache(MutableMapping[str, Any]):
    """Thread-safe bounded LRU mapping that spills evicted entries to JSON files."""

    def __init__(
        self,
        cache_dir: Path | Callable[[], Path],
        maxsize: int = DEFAULT_LRU_MAXSIZE,
        serializer: Callable[[Any], Any] | None = None,
        deserializer: Callable[[Any], Any] | None = None,
    ) -> None:
        if int(maxsize) < 1:
            raise ValueError("maxsize must be >= 1")
        self.maxsize = int(maxsize)
        self._cache_dir = cache_dir
        self._serializer = serializer or (lambda v: v)
        self._deserializer = deserializer or (lambda v: v)
        self._lock = threading.RLock()
        self._cache: OrderedDict[str, Any] = OrderedDict()

    def _resolve_dir(self) -> Path:
        path = self._cache_dir() if callable(self._cache_dir) else Path(self._cache_dir)
        return Path(path)

    @staticmethod
    def _key_digest(key: str) -> str:
        return hashlib.sha256(str(key).encode("utf-8")).hexdigest()

    def _spill_path(self, key: str) -> Path:
        return self._resolve_dir() / f"{self._key_digest(key)}.json"

    def _spill_to_disk(self, key: str, value: Any) -> None:
        cache_dir = self._resolve_dir()
        cache_dir.mkdir(parents=True, exist_ok=True)
        digest = self._key_digest(key)
        target = cache_dir / f"{digest}.json"
        tmp = cache_dir / f".{digest}.{os.getpid()}.{threading.get_ident()}.tmp"
        payload = {"key": str(key), "value": self._serializer(value)}
        try:
            tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            os.replace(tmp, target)
        except Exception:
            if tmp.exists():
                try:
                    tmp.unlink()
                except OSError:
                    pass
            raise

    def _load_from_disk(self, key: str, remove_file: bool = True) -> tuple[bool, Any]:
        target = self._spill_path(key)
        if not target.is_file():
            return False, None
        try:
            raw = json.loads(target.read_text(encoding="utf-8"))
            value = self._deserializer(raw.get("value"))
        except Exception as exc:
            logger.warning("Failed to reload spilled cache entry for %r: %s", key, exc)
            return False, None
        if remove_file:
            try:
                target.unlink()
            except OSError:
                pass
        return True, value

    def _evict_if_needed(self) -> None:
        while len(self._cache) > self.maxsize:
            evict_key, evict_val = self._cache.popitem(last=False)
            self._spill_to_disk(evict_key, evict_val)

    def get(self, key: str, default: Any = None) -> Any:
        key_str = str(key)
        with self._lock:
            if key_str in self._cache:
                self._cache.move_to_end(key_str)
                return self._cache[key_str]
            found, value = self._load_from_disk(key_str, remove_file=True)
            if not found:
                return default
            self._cache[key_str] = value
            self._evict_if_needed()
            return value

    def peek(self, key: str, default: Any = None) -> Any:
        """Read without promoting LRU order or spilling an active item."""
        key_str = str(key)
        with self._lock:
            if key_str in self._cache:
                return self._cache[key_str]
            found, value = self._load_from_disk(key_str, remove_file=False)
            return value if found else default

    def put(self, key: str, value: Any) -> None:
        key_str = str(key)
        with self._lock:
            # Remove any stale spilled copy since RAM now holds the latest value.
            spill = self._spill_path(key_str)
            if spill.is_file():
                try:
                    spill.unlink()
                except OSError:
                    pass
            self._cache[key_str] = value
            self._cache.move_to_end(key_str)
            self._evict_if_needed()

    def __getitem__(self, key: str) -> Any:
        sentinel = object()
        val = self.get(key, sentinel)
        if val is sentinel:
            raise KeyError(key)
        return val

    def __setitem__(self, key: str, value: Any) -> None:
        self.put(key, value)

    def __delitem__(self, key: str) -> None:
        key_str = str(key)
        with self._lock:
            removed = key_str in self._cache
            if removed:
                del self._cache[key_str]
            spill = self._spill_path(key_str)
            if spill.is_file():
                spill.unlink()
                removed = True
            if not removed:
                raise KeyError(key)

    def delete(self, key: str) -> bool:
        key_str = str(key)
        with self._lock:
            removed = key_str in self._cache
            self._cache.pop(key_str, None)
            spill = self._spill_path(key_str)
            if spill.is_file():
                spill.unlink()
                removed = True
            return removed

    def __contains__(self, key: object) -> bool:
        if not isinstance(key, str):
            key_str = str(key)
        else:
            key_str = key
        with self._lock:
            if key_str in self._cache:
                return True
            return self._spill_path(key_str).is_file()

    def _all_disk_keys(self) -> list[tuple[str, Path]]:
        cache_dir = self._resolve_dir()
        if not cache_dir.is_dir():
            return []
        entries: list[tuple[str, Path]] = []
        for path in cache_dir.glob("*.json"):
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
                k = raw.get("key")
                if isinstance(k, str):
                    entries.append((k, path))
            except Exception:
                continue
        return entries

    def invalidate_prefix(self, prefix: str) -> int:
        """Delete all in-memory and disk-spilled entries whose key starts with ``prefix``."""
        with self._lock:
            removed = 0
            mem_keys = [k for k in self._cache.keys() if k.startswith(prefix)]
            for k in mem_keys:
                del self._cache[k]
                removed += 1
            for k, path in self._all_disk_keys():
                if k.startswith(prefix):
                    try:
                        path.unlink()
                        removed += 1
                    except OSError:
                        pass
            return removed

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()
            cache_dir = self._resolve_dir()
            if cache_dir.is_dir():
                for path in cache_dir.glob("*.json"):
                    try:
                        path.unlink()
                    except OSError:
                        pass

    def __iter__(self) -> Iterator[str]:
        with self._lock:
            seen = list(self._cache.keys())
            seen_set = set(seen)
            for k, _ in self._all_disk_keys():
                if k not in seen_set:
                    seen.append(k)
                    seen_set.add(k)
            return iter(seen)

    def __len__(self) -> int:
        with self._lock:
            keys = set(self._cache.keys())
            for k, _ in self._all_disk_keys():
                keys.add(k)
            return len(keys)

    def in_memory_count(self) -> int:
        with self._lock:
            return len(self._cache)

    def disk_count(self) -> int:
        with self._lock:
            cache_dir = self._resolve_dir()
            if not cache_dir.is_dir():
                return 0
            return sum(1 for _ in cache_dir.glob("*.json"))

    def in_memory_keys(self) -> list[str]:
        with self._lock:
            return list(self._cache.keys())

