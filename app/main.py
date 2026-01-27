from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from app.api.routes import router
from app.core.config import settings
from app.dependencies import get_store, require_api_key, set_task_queue
from app.services.linker import LinkerRunner
from app.services.task_queue import TaskQueue


@asynccontextmanager
async def lifespan(_: FastAPI):
    store = get_store()
    await store.ensure_indexes()

    runner = LinkerRunner(store=store, app_settings=settings)
    queue = TaskQueue(
        store=store,
        handler=runner.run_job,
        poll_interval=settings.queue_poll_interval_seconds,
        workers=settings.queue_workers,
    )
    set_task_queue(queue)
    await queue.start()
    try:
        yield
    finally:
        await queue.stop()
        await store.close()


app = FastAPI(title=settings.app_name, version=settings.version, lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(router)


@app.get("/docs/reference", response_class=HTMLResponse, dependencies=[Depends(require_api_key)])
async def custom_docs() -> Response:
    reference_path = Path(__file__).resolve().parent.parent / "docs" / "static" / "reference.html"
    if reference_path.exists():
        return HTMLResponse(reference_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>Documentation not available</h1>", status_code=503)
