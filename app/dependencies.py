from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, Optional

from fastapi import Depends, Header, HTTPException, Security, status
from fastapi.security import APIKeyHeader

from app.core.config import Settings, settings
from app.storage.state import StateStore

if TYPE_CHECKING:
    from app.services.task_queue import TaskQueue


_queue: Optional["TaskQueue"] = None
_api_key_header = APIKeyHeader(
    name="X-API-Key",
    auto_error=False,
    scheme_name="ApiKeyAuth",
)
_llm_api_key_header = APIKeyHeader(
    name="X-LLM-API-Key",
    auto_error=False,
    scheme_name="LlmKeyAuth",
)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return settings


@lru_cache(maxsize=1)
def get_store() -> StateStore:
    return StateStore(settings)


def set_task_queue(queue: "TaskQueue") -> None:
    global _queue
    _queue = queue


def get_task_queue() -> "TaskQueue":
    if _queue is None:
        raise RuntimeError("Task queue not initialised")
    return _queue


def _extract_bearer_token(authorization: Optional[str]) -> Optional[str]:
    if not authorization:
        return None
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer":
        return None
    token = token.strip()
    return token or None


async def require_api_key(
    x_api_key: Optional[str] = Security(_api_key_header),
    authorization: Optional[str] = Header(
        default=None, alias="Authorization", include_in_schema=False
    ),
    current_settings: Settings = Depends(get_settings),
) -> None:
    expected = (current_settings.api_key or "").strip()
    if not expected:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="API key not configured",
        )

    supplied = x_api_key or _extract_bearer_token(authorization)
    if not supplied or supplied != expected:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")


async def get_llm_api_key(
    llm_api_key: Optional[str] = Security(_llm_api_key_header),
    model_api_key: Optional[str] = Header(
        default=None, alias="X-Model-API-Key", include_in_schema=False
    ),
) -> Optional[str]:
    for value in (llm_api_key, model_api_key):
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None
