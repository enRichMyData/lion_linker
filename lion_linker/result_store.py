from __future__ import annotations

import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import ASCENDING

logger = logging.getLogger(__name__)


class ResultStore:
    def __init__(
        self,
        uri: str,
        db_name: str,
        collection_prefix: str,
        *,
        enabled: bool = True,
        server_selection_timeout_ms: int = 1000,
        run_id: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        self._enabled = enabled
        self._uri = uri
        self._db_name = db_name
        self._collection_prefix = collection_prefix
        self._timeout_ms = server_selection_timeout_ms
        self._client: AsyncIOMotorClient | None = None
        self._collection = None
        self._init_attempted = False
        self.run_id = run_id or uuid.uuid4().hex
        self._metadata = dict(metadata or {})

    @classmethod
    def from_env(
        cls,
        *,
        enabled: bool | None = None,
        run_id: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> "ResultStore | None":
        if enabled is False:
            return None
        uri = os.getenv("MONGO_URI", "mongodb://localhost:27017")
        db_name = os.getenv("MONGO_DB", "lion_linker")
        prefix = os.getenv("MONGO_COLLECTION_PREFIX", "lion")
        return cls(
            uri=uri,
            db_name=db_name,
            collection_prefix=prefix,
            enabled=True,
            run_id=run_id,
            metadata=metadata,
        )

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
            logger.info("Result store disabled (Mongo unavailable): %s", exc)
            self._client = None
            return False

        db = self._client[self._db_name]
        self._collection = db[f"{self._collection_prefix}_results"]
        await self._collection.create_index([("run_id", ASCENDING), ("seq", ASCENDING)])
        await self._collection.create_index("created_at")
        return True

    async def record_batch(self, rows: list[dict[str, Any]], *, start_seq: int | None = None) -> None:
        if not await self._ensure_collection():
            return
        if not rows:
            return

        docs: list[dict[str, Any]] = []
        base_seq = start_seq or 0
        now = datetime.now(tz=timezone.utc)
        for offset, row in enumerate(rows):
            doc = dict(row)
            doc["run_id"] = self.run_id
            if "seq" not in doc or doc["seq"] is None:
                doc["seq"] = base_seq + offset
            if self._metadata:
                doc.setdefault("metadata", self._metadata)
            if "created_at" not in doc or not doc["created_at"]:
                doc["created_at"] = now
            docs.append(doc)
        if docs:
            await self._collection.insert_many(docs)
