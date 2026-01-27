from __future__ import annotations

import asyncio
import contextlib
from datetime import datetime, timezone
from typing import Awaitable, Callable, List

from app.models.queue import JobStatus
from app.storage.state import StateStore

JobHandler = Callable[[str], Awaitable[None]]


class TaskQueue:
    """MongoDB-backed queue that claims and processes jobs in the background."""

    def __init__(
        self,
        store: StateStore,
        handler: JobHandler,
        *,
        poll_interval: float = 0.25,
        workers: int = 1,
    ):
        self._store = store
        self._handler = handler
        self._poll_interval = poll_interval
        self._worker_tasks: List[asyncio.Task[None]] = []
        self._stop_event = asyncio.Event()
        self._wake_event = asyncio.Event()
        self._worker_count = max(1, workers)

    async def start(self) -> None:
        if any(task for task in self._worker_tasks if not task.done()):
            return
        self._stop_event.clear()
        await self._store.mark_running_jobs_failed("Worker restarted")
        self._worker_tasks = [
            asyncio.create_task(self._run(), name=f"task-queue-worker-{idx}")
            for idx in range(self._worker_count)
        ]

    async def stop(self) -> None:
        self._stop_event.set()
        self._wake_event.set()
        for task in self._worker_tasks:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        self._worker_tasks.clear()

    async def enqueue(self, job_id: str) -> None:
        _ = job_id
        self._wake_event.set()

    async def _run(self) -> None:
        while not self._stop_event.is_set():
            job = await self._store.claim_next_job()
            if not job:
                try:
                    await asyncio.wait_for(self._wake_event.wait(), timeout=self._poll_interval)
                except asyncio.TimeoutError:
                    pass
                self._wake_event.clear()
                continue

            try:
                await self._handler(job.job_id)
            except Exception as exc:  # pylint: disable=broad-except
                await self._store.update_job(
                    job.job_id,
                    status=JobStatus.failed,
                    finished_at=datetime.now(tz=timezone.utc),
                    error=str(exc),
                )
