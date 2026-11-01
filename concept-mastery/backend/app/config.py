import os
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]          # backend/
try:
    from dotenv import load_dotenv
    load_dotenv(BASE / ".env")
except ImportError:
    pass

def _p(key, default):
    return (BASE / os.getenv(key, default).split("#")[0].strip()).resolve()

UPLOAD_DIR = _p("UPLOAD_DIR", "../data/uploads")
VECTOR_STORE_PATH = _p("VECTOR_STORE_PATH", "../data/vectorstore")
FRONTEND_DIR = (BASE.parent / "frontend").resolve()
SYNONYMS_FILE = BASE / "synonyms.json"
def _e(k, d=""): return os.getenv(k, d).split("#")[0].strip()

# PageIndex API
PAGEINDEX_API_KEY = _e("PAGEINDEX_API_KEY", "")

# Primary Provider: Google Gemini
GEMINI_API_KEY = _e("GEMINI_API_KEY", "")
GEMINI_MODEL = _e("GEMINI_MODEL", "gemini-3.8-flash")

# Fallback Provider: Groq Cloud
GROQ_API_KEY = _e("GROQ_API_KEY", "")
GROQ_MODEL = _e("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_BASE_URL = _e("GROQ_BASE_URL", "https://api.groq.com/openai/v1")

# TinyFish Web Search API
TINYFISH_API_KEY = _e("TINYFISH_API_KEY", "")
TINYFISH_SEARCH_URL = _e("TINYFISH_SEARCH_URL", "https://api.search.tinyfish.ai")

# Legacy / Generic Provider
LLM_PROVIDER, LLM_API_KEY, LLM_MODEL = _e("LLM_PROVIDER", "openai").lower(), _e("LLM_API_KEY"), _e("LLM_MODEL")
LLM_BASE_URL = _e("LLM_BASE_URL", "https://api.openai.com/v1")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
VECTOR_STORE_PATH.mkdir(parents=True, exist_ok=True)
