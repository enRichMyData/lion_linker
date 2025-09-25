from __future__ import annotations

from typing import Any, Dict, Optional

DEFAULT_MODEL_NAME = "gemma2:2b"
DEFAULT_MODEL_PROVIDER = "ollama"
DEFAULT_CHUNK_SIZE = 64
DEFAULT_TABLE_CTX_SIZE = 1
DEFAULT_OLLAMA_HOST: Optional[str] = "http://ollama:11434"
DEFAULT_FORMAT_CANDIDATES = True
DEFAULT_COMPACT_CANDIDATES = True
DEFAULT_RETRIEVER_NUM_CANDIDATES = 10
DEFAULT_RETRIEVER_CACHE = False
DEFAULT_RETRIEVER_MAX_RETRIES = 3
DEFAULT_RETRIEVER_BACKOFF_FACTOR = 0.5
DEFAULT_KG_NAME = "wikidata"


def default_lion_config() -> Dict[str, Any]:
    config: Dict[str, Any] = {
        "model_name": DEFAULT_MODEL_NAME,
        "model_api_provider": DEFAULT_MODEL_PROVIDER,
        "chunk_size": DEFAULT_CHUNK_SIZE,
        "table_ctx_size": DEFAULT_TABLE_CTX_SIZE,
        "format_candidates": DEFAULT_FORMAT_CANDIDATES,
        "compact_candidates": DEFAULT_COMPACT_CANDIDATES,
    }
    if DEFAULT_OLLAMA_HOST is not None:
        config["ollama_host"] = DEFAULT_OLLAMA_HOST
    return config


def default_retriever_config(kg_value: Optional[str] = None) -> Dict[str, Any]:
    return {
        "kg": kg_value or DEFAULT_KG_NAME,
        "num_candidates": DEFAULT_RETRIEVER_NUM_CANDIDATES,
        "cache": DEFAULT_RETRIEVER_CACHE,
        "max_retries": DEFAULT_RETRIEVER_MAX_RETRIES,
        "backoff_factor": DEFAULT_RETRIEVER_BACKOFF_FACTOR,
    }
