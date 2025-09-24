from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration loaded from environment variables or defaults."""

    app_name: str = "LionLinker API"
    version: str = "0.1.0"
    workspace_path: Path = Field(default=Path("data/api_runs"), validation_alias="WORKSPACE_PATH")
    mongo_uri: Optional[str] = Field(
        default="mongodb://localhost:27017", validation_alias="MONGO_URI"
    )
    mongo_db: str = Field(default="lion_linker", validation_alias="MONGO_DB")
    mongo_collection_prefix: str = Field(
        default="lion", validation_alias="MONGO_COLLECTION_PREFIX"
    )

    model_name: str = Field(default="gemma2:2b", validation_alias="LION_MODEL_NAME")
    model_api_provider: str = Field(default="ollama", validation_alias="LION_MODEL_PROVIDER")
    model_api_key: Optional[str] = Field(default=None, validation_alias="OPENROUTER_API_KEY")
    ollama_host: Optional[str] = Field(default=None, validation_alias="OLLAMA_HOST")

    retriever_endpoint: Optional[str] = Field(default=None, validation_alias="RETRIEVER_ENDPOINT")
    retriever_token: Optional[str] = Field(default=None, validation_alias="RETRIEVER_TOKEN")
    retriever_num_candidates: int = Field(default=10, validation_alias="RETRIEVER_NUM_CANDIDATES")
    retriever_cache: bool = Field(default=False, validation_alias="RETRIEVER_CACHE")

    default_mention_columns: Optional[List[str]] = Field(
        default=None, validation_alias="LION_MENTION_COLUMNS"
    )
    chunk_size: int = Field(default=32, validation_alias="LION_CHUNK_SIZE")
    table_ctx_size: int = Field(default=1, validation_alias="LION_TABLE_CTX_SIZE")

    queue_workers: int = Field(default=1, ge=1, validation_alias="LION_QUEUE_WORKERS")
    queue_poll_interval_seconds: float = Field(
        default=0.25, gt=0, validation_alias="LION_QUEUE_POLL_INTERVAL"
    )

    dry_run: bool = Field(default=False, validation_alias="LION_DRY_RUN")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="allow",
    )


settings = Settings()

# Ensure paths exist immediately so later code can rely on them.
settings.workspace_path.mkdir(parents=True, exist_ok=True)
