"""config.py — Pluggable configuration for knowledge-pipeline.

Reads from environment variables or .env file.
No external dependencies.
"""

import os
from pathlib import Path

# ── Load .env if present ──

_ENV_PATH = Path(__file__).parent / ".env"
if _ENV_PATH.exists():
    with open(_ENV_PATH) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip("'\""))

# ── Database ──

DB_PATH = Path(__file__).parent / "knowledge.db"

# ── LLM Backend (OpenAI-compatible) ──

LLM_BASE_URL = os.environ.get(
    "LLM_BASE_URL", "http://localhost:11434/v1/chat/completions"
)
LLM_MODEL = os.environ.get("LLM_MODEL", "qwen2.5:7b")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "")
LLM_TIMEOUT = int(os.environ.get("LLM_TIMEOUT", "120"))

# ── Embedding ──

EMBED_MODEL = "BAAI/bge-m3"
EMBED_DIM = 1024
EMBED_REMOTE_URL = os.environ.get("EMBED_REMOTE_URL", "")

# ── Server ──

SERVE_PORT = int(os.environ.get("SERVE_PORT", "8780"))

# ── Scoring ──

SCORING_PROMPT_VERSION = "v1.0"


def get_db_connection():
    """Return a SQLite connection with row_factory and WAL mode."""
    import sqlite3
    conn = sqlite3.connect(str(DB_PATH), timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    return conn


def init_db():
    """Create tables if they don't exist."""
    schema_path = Path(__file__).parent / "schema.sql"
    conn = get_db_connection()
    conn.executescript(schema_path.read_text())
    conn.close()
