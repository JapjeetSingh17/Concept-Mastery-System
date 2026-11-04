# 🚀 Setting Up Concept Mastery System in Google Antigravity

Welcome to the **Closed-Loop Concept Mastery Platform**! This guide walks you through setting up and running the entire platform inside **Google Antigravity** (IDE, Desktop App 2.0, or CLI) immediately after downloading or cloning this repository to your desktop.

---

## 📋 System Requirements & Prerequisites

Before starting, ensure you have:
1. **Google Antigravity** installed ([Antigravity Download / Docs](https://antigravity.google/docs)).
2. **Python 3.10+** (Python 3.11, 3.12, or 3.13 recommended). Check with:
   ```bash
   python3 --version
   ```
3. **Git** installed on your system.
4. *(Optional but Recommended)* API Keys:
   - **OpenRouter API Key** (`OPENROUTER_API_KEY`): Powers frontier multi-model reasoning and auto-failovers.
   - **Google Gemini API Key** (`GEMINI_API_KEY`): Direct Gemini Flash tutor inference.
   - **ElevenLabs API Key** (`ELEVENLABS_API_KEY`): Fast text-to-speech for Smart Study Notes using `eleven_flash_v2_5`.
   - **Google OAuth Credentials** (`GOOGLE_CLIENT_ID`, `GOOGLE_CLIENT_SECRET`): For multi-tenant student sign-in.
   > **Note:** The platform includes full offline fallbacks! If API keys are omitted, the system still runs using local SQLite BM25 indexing, deterministic Bayesian belief updates, rule-based questions, and local synthesis.

---

## 🛠️ Step-by-Step Setup Guide

### Step 1: Download or Clone to Your Desktop

Open your terminal and clone the repository directly onto your Desktop:

```bash
cd ~/Desktop
git clone https://github.com/JapjeetSingh17/Concept-Mastery-System.git
cd Concept-Mastery-System
```

Alternatively, if you downloaded a `.zip` from GitHub:
1. Unzip the archive onto your Desktop.
2. Rename the extracted folder to `Concept-Mastery-System` (or keep the unzipped name).
3. Open your terminal and `cd` into that directory.

---

### Step 2: Open the Project in Google Antigravity

- **Via Antigravity GUI:**
  1. Launch **Antigravity**.
  2. Click **File > Open Folder...** (or `Cmd+O` / `Ctrl+O`).
  3. Navigate to `~/Desktop/Concept-Mastery-System` and click **Open**.
- **Via Antigravity CLI (`agy`):**
  ```bash
  cd ~/Desktop/Concept-Mastery-System
  agy .
  ```

Once opened, you will see the sidebar navigation containing the `concept-mastery/` directory, `antigravity-setup/`, and Antigravity chat panel.

---

### Step 3: Run the Automated Setup Script

Inside the Antigravity Terminal panel (`Ctrl+`` or `Cmd+``):

```bash
./antigravity-setup/setup.sh
```

#### What this script does automatically:
1. Detects your Python 3 environment.
2. Creates an isolated virtual environment at `concept-mastery/backend/.venv`.
3. Installs all required packages (FastAPI, Uvicorn, LangGraph, ReportLab, python-pptx, PyJWT, Google Auth, etc.).
4. Copies `.env.example` to `.env` if not already present.
5. Prepares all storage directories (`data/uploads/`, `data/vectorstore/`, `generated_artifacts/`).
6. Executes a sanity verification check on the FastAPI application.

*(If you prefer manual setup, see [Manual Setup](#manual-setup) below).*

---

### Step 4: Configure Your Environment Keys

Open the created configuration file in the Antigravity editor:
[`concept-mastery/backend/.env`](file:///Users/japjeetsingh/Documents/Capstone%20project/AI-Learning-Platform/concept-mastery/backend/.env)

Add your keys as desired:
```ini
# OpenRouter Gateway (Recommended: provides Gemini Flash, NVIDIA Nemotron, DeepSeek)
OPENROUTER_API_KEY=your_openrouter_api_key_here
OPENROUTER_MODEL=google/gemini-2.5-flash

# Google Gemini API Key (Direct)
GEMINI_API_KEY=your_gemini_api_key_here

# ElevenLabs High-Speed Audio (Flash v2.5 voice notes)
ELEVENLABS_API_KEY=your_elevenlabs_api_key_here
ELEVENLABS_MODEL_ID=eleven_flash_v2_5
ELEVENLABS_VOICE_ID=JBFqnCBsd6RMkjVDRZzb

# Google OAuth 2.0 (Optional - for multi-tenant sign-in)
GOOGLE_CLIENT_ID=your_client_id.apps.googleusercontent.com
GOOGLE_CLIENT_SECRET=your_client_secret
JWT_SECRET=concept-mastery-jwt-secret-key-32bytes
```

---

### Step 5: Start the Local Development Server

Run the following command in your Antigravity terminal:

```bash
cd concept-mastery/backend
source .venv/bin/activate
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

You will see:
```text
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
INFO:     Started reloader process
INFO:     Application startup complete.
```

---

### Step 6: Open the Platform in Your Browser

Open your browser and navigate to:
👉 **[http://127.0.0.1:8000](http://127.0.0.1:8000)**

You can also use Antigravity's integrated browser preview panel (`/browser` slash command).

---

### Step 7: Verify Everything with the Automated Test Suite

To verify that all 65+ automated test cases pass on your machine:

```bash
cd concept-mastery/backend
source .venv/bin/activate
PYTHONPATH=. pytest
```

Expected result:
```text
============================= 65 passed in ~50s =============================
```

---

## 🤖 Next: Talk to the Antigravity Agent!

Now that the system is running, what should you ask Antigravity to do?
Check out **[`antigravity-setup/PROMPTS.md`](PROMPTS.md)** for a curated collection of copy-paste prompts designed for Antigravity pair programming, including:
- 🚀 **System Health & Inspection Prompt**
- 📚 **Material Ingestion & Concept DAG Construction Prompt**
- 🧠 **Active Recall Interactive Session Prompt**
- 🎙️ **Multimodal Smart Study Notes & Audio Generation Prompt**
- 🧩 **Custom Math/Numerical Quiz Problem Prompt**

---

## 🛠️ Manual Setup (Optional)

If you prefer not to use `setup.sh`:

```bash
# 1. Navigate to backend
cd concept-mastery/backend

# 2. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Upgrade pip and install requirements
pip install --upgrade pip
pip install -r requirements.txt

# 4. Copy environment template
cp .env.example .env

# 5. Create storage directories
mkdir -p ../data/uploads ../data/vectorstore generated_artifacts

# 6. Start the server
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
