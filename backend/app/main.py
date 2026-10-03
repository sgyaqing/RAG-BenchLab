import asyncio
import contextlib
import faulthandler
import logging
import os
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from app.api.corpora import router as corpora_router
from app.api.evaluations import router as evaluations_router
from app.api.model_configs import router as model_configs_router
from app.api.rag_systems import router as rag_systems_router
from app.api.routes import router as api_router
from app.api.testsets import router as testsets_router
from app.core.config import get_settings
from app.core.logging import setup_logging
from app.db.session import init_db
from app.services import adapter_assist, corpus_convert, eval_run, testset_gen

logger = logging.getLogger(__name__)

# tiktoken caches its vocabulary under the system temp directory by default,
# which is cleared on reboot — and fetching it again is the synchronous download
# that once held the event loop for 30 s while ragas counted document tokens.
# data/ is the directory that survives. The image sets this itself, to a baked
# copy the runtime never has to fetch, and setdefault leaves that alone.
os.environ.setdefault("TIKTOKEN_CACHE_DIR", str(get_settings().data_dir / ".tiktoken"))


# The loop's heartbeat, written once per cycle of the lag watchdog below and
# read by the stall probe's thread. A float in a dict rather than a module
# global so the thread sees updates without any import-order games.
_heartbeat = {"at": time.monotonic()}


async def _watch_event_loop_lag(interval: float = 1.0, threshold: float = 2.0) -> None:
    """Log whenever the event loop stops running for longer than `threshold`.

    One testset run left 84 s in app.log with no line of any kind — no LLM
    call, no HTTP request, no SDK retry, no error — and froze every page in
    the UI for the same stretch. That is the loop not running, not a slow
    call, and the phase logs alone cannot say when it started or how long it
    lasted. Measuring the lag directly is what localises it.
    """
    loop = asyncio.get_running_loop()
    while True:
        before = loop.time()
        await asyncio.sleep(interval)
        _heartbeat["at"] = time.monotonic()
        lag = loop.time() - before - interval
        if lag >= threshold:
            logger.warning("Event loop blocked for %.1fs", lag)


def _start_stall_probe(silence: float = 10.0, repeat: float = 5.0,
                       stop: threading.Event | None = None) -> None:
    """Dump every thread's stack while the loop is still stuck.

    The lag watchdog above can only report a block once it has ended, and by
    then the frame that caused it is gone — which is why a 20 s stall that hit
    eight times in one run was never attributed to a line of code. A thread
    watching the heartbeat can fire mid-block instead, and faulthandler reads
    the stacks from wherever the threads actually are, so the frame holding
    the loop is in the dump.

    Runs on every stall: several dumps of a long block show whether it is one
    frame or a series of them.
    """
    def watch() -> None:
        last_dump = 0.0
        while stop is None or not stop.is_set():
            time.sleep(1.0)
            if stop is not None and stop.is_set():
                break
            silent = time.monotonic() - _heartbeat["at"]
            if silent < silence or time.monotonic() - last_dump < repeat:
                continue
            last_dump = time.monotonic()
            logger.warning("Event loop silent for %.1fs; dumping all thread stacks", silent)
            try:
                with open(get_settings().log_dir / "stall.log", "a", encoding="utf-8") as fh:
                    fh.write(f"\n===== loop silent {silent:.1f}s at {datetime.now()} =====\n")
                    faulthandler.dump_traceback(file=fh, all_threads=True)
            except Exception as exc:  # a probe must never take the app down
                logger.warning("Stall dump failed: %s", exc)

    threading.Thread(target=watch, name="stall-probe", daemon=True).start()


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    init_db()
    await corpus_convert.resume_interrupted()
    await testset_gen.resume_interrupted()
    await eval_run.resume_interrupted()
    await adapter_assist.resume_interrupted()
    lag_watch = asyncio.create_task(_watch_event_loop_lag())
    if not any(t.name == "stall-probe" for t in threading.enumerate()):
        _start_stall_probe()
    logger.info("RAG-BenchLab backend started")
    try:
        yield
    finally:
        lag_watch.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await lag_watch
        logger.info("RAG-BenchLab backend stopped")


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title=settings.app_name, lifespan=lifespan)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(api_router)
    app.include_router(model_configs_router)
    app.include_router(corpora_router)
    app.include_router(testsets_router)
    app.include_router(rag_systems_router)
    app.include_router(evaluations_router)

    dist = settings.frontend_dist
    if dist.is_dir():
        dist_root = dist.resolve()

        @app.get("/{full_path:path}", include_in_schema=False)
        async def serve_spa(full_path: str):
            # `dist / full_path` resolves ".." straight out of the build
            # directory, so containment has to be checked explicitly. Measured
            # before this guard, against a running server: GET
            # /../../data/rag_benchlab.db returned the whole database (every
            # stored API key in it) and /../../logs/app.log returned the logs.
            # Resolve first — a symlink inside dist must not escape either.
            if full_path:
                candidate = (dist / full_path).resolve()
                if candidate.is_file() and candidate.is_relative_to(dist_root):
                    return FileResponse(candidate)
            # index.html must never be cached: hashed asset names change per build
            return FileResponse(
                dist / "index.html", headers={"Cache-Control": "no-cache"}
            )
    else:
        logger.warning("Frontend build not found at %s; only API routes are served", dist)

    return app


app = create_app()
