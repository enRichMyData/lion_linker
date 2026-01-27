from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Any

from motor.motor_asyncio import AsyncIOMotorClient

logger = logging.getLogger(__name__)


class CandidateCache:
    def __init__(
        self,
        uri: str,
        db_name: str,
        collection_prefix: str,
        enabled: bool = True,
        server_selection_timeout_ms: int = 1000,
    ) -> None:
        self._enabled = enabled
        self._uri = uri
        self._db_name = db_name
        self._collection_prefix = collection_prefix
        self._timeout_ms = server_selection_timeout_ms
        self._client: AsyncIOMotorClient | None = None
        self._collection = None
        self._init_attempted = False

    @classmethod
    def from_env(cls, enabled: bool | None = None) -> "CandidateCache | None":
        if enabled is False:
            return None
        uri = os.getenv("MONGO_URI", "mongodb://localhost:27017")
        db_name = os.getenv("MONGO_DB", "lion_linker")
        prefix = os.getenv("MONGO_COLLECTION_PREFIX", "lion")
        return cls(uri=uri, db_name=db_name, collection_prefix=prefix, enabled=True)

    async def _ensure_collection(self) -> bool:
        if not self._enabled:
            return False
        if self._collection is not None:
            return True
        if self._init_attempted:
            return False
        self._init_attempted = True

        try:
            self._client = AsyncIOMotorClient(
                self._uri, serverSelectionTimeoutMS=self._timeout_ms
            )
            await self._client.admin.command("ping")
        except Exception as exc:
            logger.info("Candidate cache disabled (Mongo unavailable): %s", exc)
            self._client = None
            return False

        db = self._client[self._db_name]
        self._collection = db[f"{self._collection_prefix}_candidate_cache"]
        await self._collection.create_index("cache_key", unique=True)
        return True

    async def get(self, cache_key: str) -> dict[str, Any] | None:
        if not await self._ensure_collection():
            return None
        return await self._collection.find_one({"cache_key": cache_key})

    async def set(self, cache_key: str, payload: dict[str, Any], candidates: list[dict]) -> None:
        if not await self._ensure_collection():
            return
        now = datetime.now(tz=timezone.utc)
        doc = {
            "cache_key": cache_key,
            "payload": payload,
            "candidates": candidates,
            "updated_at": now,
        }
        await self._collection.update_one(
            {"cache_key": cache_key},
            {"$set": doc},
            upsert=True,
        )
