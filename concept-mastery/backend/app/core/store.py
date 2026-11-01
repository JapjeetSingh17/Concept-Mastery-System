"""Shared knowledge base (index + concepts) and JSON persistence in VECTOR_STORE_PATH."""
import json
from .. import config
from .vector_index import VectorIndex

class UserError(Exception):
    """A problem the user can fix. Becomes a 400 response with this message."""

KB: dict = {"ready": False}
KB_FILE = config.VECTOR_STORE_PATH / "kb.json"
MASTERY_FILE = config.VECTOR_STORE_PATH / "mastery.json"

def save_kb(file, pages, index, concepts, edges):
    KB.update(ready=True, file=file, pages=pages, index=index, concepts=concepts, edges=edges)
    KB_FILE.write_text(json.dumps({"file": file, "pages": pages, "index": index.to_dict(),
                                   "concepts": concepts, "edges": edges}), encoding="utf-8")

def load_kb():
    if KB_FILE.exists():
        try:
            d = json.loads(KB_FILE.read_text(encoding="utf-8"))
            KB.update(ready=True, file=d["file"], pages=d["pages"], index=VectorIndex.from_dict(d["index"]),
                      concepts=d["concepts"], edges=d["edges"])
        except Exception as e:
            print("Could not load saved index:", e)

def require():
    if not KB.get("ready"):
        raise UserError("Upload a study PDF first (Study material page).")
    return KB

def load_json(path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default

def save_json(path, data):
    path.write_text(json.dumps(data), encoding="utf-8")
