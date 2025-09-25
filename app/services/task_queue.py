from __future__ import annotations

import asyncio
import contextlib
from collections import deque
from typing import Awaitable, Callable, Deque, List

from app.models.jobs import JobStatus
from app.storage.state import StateStore

JobHandler = Callable[[str], Awaitable[None]]


class TaskQueue:
    """Simple FIFO queue that processes jobs sequentially in the background."""

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
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self._worker_tasks: List[asyncio.Task[None]] = []
        self._stop_event = asyncio.Event()
        self._recent_jobs: Deque[str] = deque(maxlen=256)
        self._worker_count = max(1, workers)

    async def start(self) -> None:
        if any(task for task in self._worker_tasks if not task.done()):
            return
        self._stop_event.clear()
        self._worker_tasks = [
            asyncio.create_task(self._run(), name=f"task-queue-worker-{idx}")
            for idx in range(self._worker_count)
        ]

    async def stop(self) -> None:
        self._stop_event.set()
        for task in self._worker_tasks:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        self._worker_tasks.clear()

    async def enqueue(self, job_id: str) -> None:
        if job_id in self._recent_jobs:
            return
        await self._queue.put(job_id)
        self._recent_jobs.append(job_id)

    async def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                job_id = await asyncio.wait_for(self._queue.get(), timeout=self._poll_interval)
            except asyncio.TimeoutError:
                continue

            try:
                await self._store.update_job(job_id, status=JobStatus.in_progress)
                await self._handler(job_id)
            except Exception as exc:  # pylint: disable=broad-except
                await self._store.update_job(
                    job_id,
                    status=JobStatus.failed,
                    message=str(exc),
                )
            finally:
                self._queue.task_done()
