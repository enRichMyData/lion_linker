from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, Optional

from app.core.config import Settings, settings
from app.storage.state import StateStore

if TYPE_CHECKING:
    from app.services.task_queue import TaskQueue


_queue: Optional["TaskQueue"] = None


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
