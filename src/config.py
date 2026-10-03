"""Central configuration. No secrets in code — everything from env."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env(key: str, default: str = "") -> str:
    return os.getenv(key, default)


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, str(default)))
    except ValueError:
        return default


def _env_float(key: str, default: float) -> float:
    try:
        return float(os.getenv(key, str(default)))
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    """Loaded once at startup. `frozen` prevents accidental mutation."""

    # --- general / tradecraft ---
    log_level: str = field(default_factory=lambda: _env("LOG_LEVEL", "INFO"))
    max_file_mb: int = field(default_factory=lambda: _env_int("MAX_FILE_MB", 500))
    work_dir: str = field(default_factory=lambda: _env("WORK_DIR", "/tmp/opencode/provenance"))
    tlp: str = field(default_factory=lambda: _env("TLP_MARKING", "TLP:CLEAR"))
    audit_log: str = field(default_factory=lambda: _env("AUDIT_LOG", "reports/audit.jsonl"))

    # --- module 1: C2PA trust ---
    # JSON list of trusted signer fingerprints / CAs for air-gapped trust.
    trust_anchors_path: str = field(
        default_factory=lambda: _env("TRUST_ANCHORS_PATH", "config/trust_anchors.json")
    )

    # --- module 2: forensics ---
    max_frames: int = field(default_factory=lambda: _env_int("MAX_FRAMES", 5))
    onnx_model_path: str = field(
        default_factory=lambda: _env("ONNX_MODEL_PATH", "models/deepfake.onnx")
    )
    synthetic_model_path: str = field(
        default_factory=lambda: _env("SYNTHETIC_MODEL_PATH", "models/synthetic_clip.onnx")
    )
    clip_model_path: str = field(
        default_factory=lambda: _env("CLIP_MODEL_PATH", "models/clip_image.onnx")
    )
    model_registry_path: str = field(
        default_factory=lambda: _env("MODEL_REGISTRY_PATH", "models/registry.json")
    )
    enable_heatmap: bool = field(default_factory=lambda: _env("ENABLE_HEATMAP", "0") == "1")
    force_cpu: bool = field(default_factory=lambda: _env("FORCE_CPU", "0") == "1")

    # --- module 3: context ---
    # memory | postgres | milvus | weaviate. Unknown values fall back to memory.
    vector_backend: str = field(default_factory=lambda: _env("VECTOR_BACKEND", "memory"))
    phash_threshold: int = field(default_factory=lambda: _env_int("PHASH_THRESHOLD", 8))
    pdq_threshold: int = field(default_factory=lambda: _env_int("PDQ_THRESHOLD", 48))  # /256 bits
    clip_threshold: float = field(default_factory=lambda: _env_float("CLIP_THRESHOLD", 0.90))
    postgres_dsn: str = field(default_factory=lambda: _env("POSTGRES_DSN", ""))
    milvus_uri: str = field(default_factory=lambda: _env("MILVUS_URI", "http://localhost:19530"))
    weaviate_url: str = field(default_factory=lambda: _env("WEAVIATE_URL", "http://localhost:8080"))
    qdrant_url: str = field(default_factory=lambda: _env("QDRANT_URL", "http://localhost:6333"))
    reference_db_path: str = field(
        default_factory=lambda: _env("REFERENCE_DB_PATH", "data/reference_events.json")
    )

    # --- module 4: LLM ---
    # ollama | hf | llamaindex | rule-based (auto falls back to rule-based offline)
    llm_backend: str = field(default_factory=lambda: _env("LLM_BACKEND", "auto"))
    ollama_host: str = field(default_factory=lambda: _env("OLLAMA_HOST", "http://localhost:11434"))
    ollama_model: str = field(default_factory=lambda: _env("OLLAMA_MODEL", "llama3.1:8b"))
    ollama_timeout_s: int = field(default_factory=lambda: _env_int("OLLAMA_TIMEOUT_S", 60))
    hf_model: str = field(
        default_factory=lambda: _env("HF_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    )
    rag_top_k: int = field(default_factory=lambda: _env_int("RAG_TOP_K", 3))

    # --- speech (Whisper, offline after provisioning) ---
    whisper_model: str = field(default_factory=lambda: _env("WHISPER_MODEL", "tiny"))
    whisper_root: str = field(default_factory=lambda: _env("WHISPER_ROOT", "models/whisper"))

    # --- OSINT ingestion ---
    osint_rss_feeds: str = field(default_factory=lambda: _env("OSINT_RSS_FEEDS", ""))
    telegram_bot_token: str = field(default_factory=lambda: _env("TELEGRAM_BOT_TOKEN", ""))
    telegram_chat_ids: str = field(default_factory=lambda: _env("TELEGRAM_CHAT_IDS", ""))

    # --- active learning ---
    al_retrain_threshold: int = field(default_factory=lambda: _env_int("AL_RETRAIN_THRESHOLD", 50))

    # --- output / serving ---
    report_out_dir: str = field(default_factory=lambda: _env("REPORT_OUT_DIR", "reports"))
    api_host: str = field(default_factory=lambda: _env("API_HOST", "127.0.0.1"))
    api_port: int = field(default_factory=lambda: _env_int("API_PORT", 8088))


SETTINGS = Settings()
