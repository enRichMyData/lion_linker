from __future__ import annotations

from fastapi import FastAPI

from app.api.routes import router
from app.core.config import settings
from app.dependencies import get_store, set_task_queue
from app.services.linker import LinkerRunner
from app.services.task_queue import TaskQueue

app = FastAPI(title=settings.app_name, version=settings.version)
app.include_router(router)

_runner: LinkerRunner | None = None
_queue: TaskQueue | None = None


@app.on_event("startup")
async def startup_event() -> None:
    global _runner, _queue
    store = get_store()
    await store.ensure_indexes()
    _runner = LinkerRunner(store=store, app_settings=settings)
    _queue = TaskQueue(
        store=store,
        handler=_runner.run_job,
        poll_interval=settings.queue_poll_interval_seconds,
    )
    set_task_queue(_queue)
    await _queue.start()


@app.on_event("shutdown")
async def shutdown_event() -> None:
    global _queue
    if _queue:
        await _queue.stop()
        _queue = None
    store = get_store()
    await store.close()
