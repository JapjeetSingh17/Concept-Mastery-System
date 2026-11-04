#!/usr/bin/env bash
set -e

# ==============================================================================
# Closed-Loop Concept Mastery Platform - Antigravity Automated Setup Script
# ==============================================================================

echo "=================================================================="
echo "🚀 Concept Mastery Platform - Google Antigravity Setup"
echo "=================================================================="

# 1. Determine script directory and repo root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BACKEND_DIR="$REPO_ROOT/concept-mastery/backend"

echo ""
echo "[1/5] Checking Python 3 environment..."
if command -v python3 &>/dev/null; then
    PYTHON_CMD="python3"
elif command -v python &>/dev/null; then
    PYTHON_CMD="python"
else
    echo "❌ Error: Python 3 is not installed or not available in PATH."
    echo "Please install Python 3.10+ from https://www.python.org/"
    exit 1
fi

PY_VER=$($PYTHON_CMD -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "  Detected Python $PY_VER ($(which $PYTHON_CMD))"

# 2. Virtual Environment Setup
VENV_PATH="$BACKEND_DIR/.venv"
if [ ! -d "$VENV_PATH" ]; then
    echo ""
    echo "[2/5] Creating isolated virtual environment at:"
    echo "      $VENV_PATH"
    $PYTHON_CMD -m venv "$VENV_PATH"
else
    echo ""
    echo "[2/5] Virtual environment already present at:"
    echo "      $VENV_PATH"
fi

# Activate virtual environment
source "$VENV_PATH/bin/activate"

# 3. Install Dependencies
echo ""
echo "[3/5] Installing dependencies from requirements.txt..."
pip install --upgrade pip --quiet
pip install -r "$BACKEND_DIR/requirements.txt" --quiet

# 4. Configure Environment Variables
echo ""
echo "[4/5] Checking configuration (.env)..."
if [ ! -f "$BACKEND_DIR/.env" ]; then
    echo "  Creating $BACKEND_DIR/.env from .env.example..."
    cp "$BACKEND_DIR/.env.example" "$BACKEND_DIR/.env"
else
    echo "  Using existing $BACKEND_DIR/.env"
fi

if [ ! -f "$REPO_ROOT/.env" ]; then
    cp "$BACKEND_DIR/.env.example" "$REPO_ROOT/.env"
fi

# Ensure storage directories exist
mkdir -p "$REPO_ROOT/concept-mastery/data/uploads"
mkdir -p "$REPO_ROOT/concept-mastery/data/vectorstore"
mkdir -p "$BACKEND_DIR/generated_artifacts"

# 5. Sanity Verification
echo ""
echo "[5/5] Running verification sanity check..."
cd "$BACKEND_DIR"
PYTHONPATH=. python -c "
import sys
from app.main import app
print('  ✅ FastAPI app initialized successfully without import errors.')
"

echo ""
echo "=================================================================="
echo "🎉 Setup Completed Successfully!"
echo "=================================================================="
echo ""
echo "How to start the platform:"
echo "  1. Start the server:"
echo "     cd concept-mastery/backend"
echo "     source .venv/bin/activate"
echo "     uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload"
echo ""
echo "  2. Access in browser:"
echo "     http://127.0.0.1:8000"
echo ""
echo "  3. In Antigravity:"
echo "     Open 'antigravity-setup/PROMPTS.md' to copy-paste prompts into"
echo "     the Antigravity agent chat to automate your learning and dev workflow!"
echo "=================================================================="
