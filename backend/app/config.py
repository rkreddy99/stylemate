"""Runtime configuration. Everything is overridable by environment variable so the
same image runs on a laptop with Ollama and on a server with a hosted model."""
from __future__ import annotations
import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv(path: Path) -> int:
    """Read backend/.env into os.environ.

    .env.example tells you to copy it to .env, but nothing read the file: every
    setting came from the shell only, so a .env sat there being ignored. This
    is a deliberately tiny parser rather than a python-dotenv dependency —
    KEY=VALUE, # comments, blank lines, optional surrounding quotes, and an
    optional leading "export".

    A value already in the environment WINS. The shell and the launcher are
    more specific than a file checked into the project, and silently
    overriding them would be the wrong way round.
    """
    if not path.is_file():
        return 0
    loaded = 0
    try:
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[7:].lstrip()
            key, sep, val = line.partition("=")
            if not sep:
                continue
            key = key.strip()
            if not key or key in os.environ:
                continue
            val = val.strip()
            if len(val) >= 2 and val[0] == val[-1] and val[0] in "\"'":
                val = val[1:-1]
            os.environ[key] = val
            loaded += 1
    except OSError:
        return 0
    return loaded


DOTENV_PATH = ROOT / ".env"
DOTENV_LOADED = _load_dotenv(DOTENV_PATH)

DATA_DIR = Path(os.getenv("SM_DATA_DIR", ROOT / "var"))
DATA_DIR.mkdir(parents=True, exist_ok=True)


def _bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in ("1", "true", "yes", "on")


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


@dataclass
class Settings:
    # --- LLM -------------------------------------------------------------
    # Ollama's NATIVE endpoint. The OpenAI-compatible /v1 route silently clamps
    # the context window to 4096 and ignores options.num_ctx, which quietly
    # truncates the catalogue we send. Use /api/chat.
    ollama_host: str = os.getenv("SM_OLLAMA_HOST", "http://localhost:11434")
    text_model: str = os.getenv("SM_TEXT_MODEL", "qwen2.5:7b")
    vision_model: str = os.getenv("SM_VISION_MODEL", "qwen2.5vl:7b")
    embed_model: str = os.getenv("SM_EMBED_MODEL", "nomic-embed-text")
    num_ctx: int = _int("SM_NUM_CTX", 16384)
    llm_timeout: float = float(os.getenv("SM_LLM_TIMEOUT", "180"))
    llm_enabled: bool = _bool("SM_LLM_ENABLED", True)

    # --- Gemini ----------------------------------------------------------
    # The key may also be supplied at runtime via POST /api/settings/llm. It is
    # held in memory only - never written to disk, never logged, never returned.
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    gemini_text_model: str = os.getenv("SM_GEMINI_TEXT_MODEL", "gemini-3.8-flash")
    gemini_vision_model: str = os.getenv("SM_GEMINI_VISION_MODEL", "gemini-3.8-flash")
    gemini_embed_model: str = os.getenv("SM_GEMINI_EMBED_MODEL", "gemini-embedding-2-preview")

    # "ollama" | "gemini" | "auto".  "auto" uses Gemini when a key is present
    # and falls back to a local Ollama otherwise.
    provider: str = os.getenv("SM_PROVIDER", "auto")

    # --- retrieval -------------------------------------------------------
    shortlist_size: int = _int("SM_SHORTLIST", 60)   # sent to the LLM reranker
    results_size: int = _int("SM_RESULTS", 24)       # returned to the browser
    use_embeddings: bool = _bool("SM_USE_EMBEDDINGS", True)

    # --- storage ---------------------------------------------------------
    db_path: Path = field(default_factory=lambda: DATA_DIR / "stylemate.db")
    cache_dir: Path = field(default_factory=lambda: DATA_DIR / "cache")
    upload_dir: Path = field(default_factory=lambda: DATA_DIR / "uploads")

    # Where the page posts its audit records. Published to the browser by
    # /api/config so the sink URL lives in one place.
    log_sink: str = os.getenv("SM_LOG_SINK", "http://localhost:8765/log")

    cors_origins: tuple = tuple(
        o.strip() for o in os.getenv("SM_CORS", "*").split(",") if o.strip()
    )

    def __post_init__(self):
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.upload_dir.mkdir(parents=True, exist_ok=True)


settings = Settings()
