import logging
import os
from pathlib import Path
import re
import subprocess
import sys

import pytest

BACKEND_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND_DIR.parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app import config


def _parse_env_keys(path: Path) -> set[str]:
    keys: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "=" in stripped:
            keys.add(stripped.split("=", 1)[0].strip())
    return keys


def test_env_example_symlink_and_parity():
    root_example = REPO_ROOT / ".env.example"
    backend_example = BACKEND_DIR / ".env.example"

    assert root_example.exists()
    assert backend_example.exists()
    assert root_example.is_symlink() or root_example.read_text(encoding="utf-8") == backend_example.read_text(encoding="utf-8")
    assert root_example.resolve() == backend_example.resolve()


def test_all_config_env_vars_documented_in_env_example():
    config_src = (BACKEND_DIR / "app" / "config.py").read_text(encoding="utf-8")
    backend_example = BACKEND_DIR / ".env.example"
    documented_keys = _parse_env_keys(backend_example)

    # Extract all variable names read via _e("VAR"), _p("VAR"), _int("VAR"), _float("VAR"), _bool("VAR"), _opt_path("VAR")
    helper_vars = set(
        re.findall(r"_(?:e|p|int|float|bool|opt_path)\(\s*[\"']([A-Z0-9_]+)[\"']", config_src)
    )
    os_vars = set(
        re.findall(r"os\.getenv\(\s*[\"']([A-Z0-9_]+)[\"']", config_src)
    )
    all_vars = (helper_vars | os_vars) - {"SKIP_DOTENV", "PYTEST_CURRENT_TEST"}

    missing = sorted(v for v in all_vars if v not in documented_keys)
    assert not missing, f"Environment variables read in app/config.py missing from .env.example: {missing}"


def test_optional_keys_missing_logs_clear_warnings(monkeypatch, caplog):
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.delenv("OFFLINE_TEST", raising=False)
    monkeypatch.setattr(config, "ENVIRONMENT", "production")
    monkeypatch.setattr(config, "OFFLINE_TEST", False)
    monkeypatch.setattr(config, "JWT_SECRET", "a_very_strong_random_production_secret_987654")
    monkeypatch.setattr(config, "OPENROUTER_API_KEY", "or_test_configured_key")
    monkeypatch.setattr(config, "ELEVENLABS_API_KEY", "")
    monkeypatch.setattr(config, "TINYFISH_API_KEY", "")
    monkeypatch.setattr(config, "PAGEINDEX_API_KEY", "")
    monkeypatch.setattr(config, "GOOGLE_CLIENT_ID", "")

    with caplog.at_level(logging.WARNING, logger="concept_mastery.config"):
        report = config.validate_startup_config()

    assert report["strict"] is True
    assert report["errors"] == []
    joined = "\n".join(report["warnings"])
    assert "ELEVENLABS_API_KEY" in joined
    assert "TINYFISH_API_KEY" in joined
    assert "PAGEINDEX_API_KEY" in joined
    assert any("ELEVENLABS_API_KEY" in rec.message for rec in caplog.records)


def test_required_secrets_fail_fast_in_production(monkeypatch):
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.delenv("OFFLINE_TEST", raising=False)
    monkeypatch.setattr(config, "ENVIRONMENT", "production")
    monkeypatch.setattr(config, "OFFLINE_TEST", False)

    # 1. Default / placeholder JWT secret fails fast in production
    monkeypatch.setattr(config, "JWT_SECRET", config.DEFAULT_INSECURE_JWT_SECRET)
    monkeypatch.setattr(config, "OPENROUTER_API_KEY", "or_test_configured_key")
    with pytest.raises(config.ConfigValidationError, match="JWT_SECRET"):
        config.validate_startup_config()

    monkeypatch.setattr(config, "JWT_SECRET", "replace-with-random-32byte-hex-secret")
    with pytest.raises(config.ConfigValidationError, match="JWT_SECRET"):
        config.validate_startup_config()

    # 2. Missing all LLM API keys fails fast in production
    monkeypatch.setattr(config, "JWT_SECRET", "strong_prod_jwt_secret_value_0123456789")
    monkeypatch.setattr(config, "OPENROUTER_API_KEY", "")
    monkeypatch.setattr(config, "GEMINI_API_KEY", "")
    monkeypatch.setattr(config, "GROQ_API_KEY", "")
    monkeypatch.setattr(config, "LLM_API_KEY", "")
    with pytest.raises(config.ConfigValidationError, match="LLM API key"):
        config.validate_startup_config()


def test_test_environment_allows_missing_secrets(monkeypatch):
    monkeypatch.setattr(config, "ENVIRONMENT", "test")
    monkeypatch.setattr(config, "JWT_SECRET", config.DEFAULT_INSECURE_JWT_SECRET)
    monkeypatch.setattr(config, "OPENROUTER_API_KEY", "")
    monkeypatch.setattr(config, "GEMINI_API_KEY", "")
    monkeypatch.setattr(config, "GROQ_API_KEY", "")
    monkeypatch.setattr(config, "LLM_API_KEY", "")

    report = config.validate_startup_config()
    assert report["is_test"] is True
    assert report["strict"] is False
    assert report["errors"] == []
    assert len(report["warnings"]) >= 2


def test_import_time_startup_validation_subprocess():
    base_env = {k: v for k, v in os.environ.items() if k != "PYTEST_CURRENT_TEST"}
    base_env["SKIP_DOTENV"] = "1"
    base_env["ENVIRONMENT"] = "production"
    base_env["OFFLINE_TEST"] = "0"
    base_env.pop("JWT_SECRET", None)
    base_env.pop("OPENROUTER_API_KEY", None)

    fail_proc = subprocess.run(
        [sys.executable, "-c", "from app import config"],
        cwd=str(BACKEND_DIR),
        env=base_env,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert fail_proc.returncode != 0
    assert "ConfigValidationError" in fail_proc.stderr

    ok_env = dict(base_env)
    ok_env["JWT_SECRET"] = "prod_secret_at_least_32_characters_long_xyz"
    ok_env["OPENROUTER_API_KEY"] = "or_prod_placeholder_for_subprocess_test"
    pass_proc = subprocess.run(
        [sys.executable, "-c", "from app import config; print(config.SQLITE_BUSY_TIMEOUT_MS)"],
        cwd=str(BACKEND_DIR),
        env=ok_env,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert pass_proc.returncode == 0, pass_proc.stderr
    assert pass_proc.stdout.strip() == "10000"

