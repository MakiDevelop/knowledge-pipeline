"""config.py — Pluggable configuration for knowledge-pipeline.

Reads from environment variables or .env file.
No external dependencies.
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

# ── Load .env if present ──

_ENV_PATH: Path = Path(__file__).parent / ".env"
if _ENV_PATH.exists():
    with open(_ENV_PATH) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip("'\""))

# ── Database ──

DB_PATH: Path = Path(__file__).parent / "knowledge.db"

# ── LLM Backend (OpenAI-compatible) ──

LLM_BASE_URL: str = os.environ.get(
    "LLM_BASE_URL", "http://localhost:11434/v1/chat/completions"
)
LLM_MODEL: str = os.environ.get("LLM_MODEL", "qwen2.5:7b")
LLM_API_KEY: str = os.environ.get("LLM_API_KEY", "")
LLM_TIMEOUT: int = int(os.environ.get("LLM_TIMEOUT", "120"))

# ── Embedding ──

EMBED_MODEL: str = "BAAI/bge-m3"
EMBED_DIM: int = 1024
EMBED_REMOTE_URL: str = os.environ.get("EMBED_REMOTE_URL", "")

# ── Server ──

SERVE_PORT: int = int(os.environ.get("SERVE_PORT", "8780"))

# ── Scoring ──

SCORING_PROMPT_VERSION: str = "v1.0"


def get_db_connection() -> sqlite3.Connection:
    """Return a SQLite connection with row_factory and WAL mode."""
    conn = sqlite3.connect(str(DB_PATH), timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    return conn


def init_db() -> None:
    """Create tables if they don't exist."""
    schema_path = Path(__file__).parent / "schema.sql"
    conn = get_db_connection()
    conn.executescript(schema_path.read_text())
    conn.close()
