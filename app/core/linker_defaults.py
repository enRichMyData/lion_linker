from __future__ import annotations

import os
from typing import Any, Dict, Optional

DEFAULT_MODEL_NAME = "gpt-oss:20b"
DEFAULT_MODEL_PROVIDER = "ollama"
DEFAULT_CHUNK_SIZE = 64
DEFAULT_OLLAMA_HOST: Optional[str] = os.getenv("OLLAMA_HOST", "http://ollama:11434")
DEFAULT_OLLAMA_API_KEY: Optional[str] = os.getenv("OLLAMA_API_KEY")
DEFAULT_FORMAT_CANDIDATES = False
DEFAULT_COMPACT_CANDIDATES = False
DEFAULT_MAX_PARALLEL_PROMPTS = 1
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
        "format_candidates": DEFAULT_FORMAT_CANDIDATES,
        "compact_candidates": DEFAULT_COMPACT_CANDIDATES,
        "max_parallel_prompts": DEFAULT_MAX_PARALLEL_PROMPTS,
    }
    if DEFAULT_OLLAMA_HOST is not None:
        config["ollama_host"] = DEFAULT_OLLAMA_HOST
    if DEFAULT_OLLAMA_API_KEY:
        config["model_api_key"] = DEFAULT_OLLAMA_API_KEY
    return config


def default_retriever_config(kg_value: Optional[str] = None) -> Dict[str, Any]:
    return {
        "class_path": "lion_linker.retrievers.LamapiClient",
        "kg": kg_value or DEFAULT_KG_NAME,
        "num_candidates": DEFAULT_RETRIEVER_NUM_CANDIDATES,
        "cache": DEFAULT_RETRIEVER_CACHE,
        "max_retries": DEFAULT_RETRIEVER_MAX_RETRIES,
        "backoff_factor": DEFAULT_RETRIEVER_BACKOFF_FACTOR,
    }
