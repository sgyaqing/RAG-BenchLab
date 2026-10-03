"""The loop-lag watchdog.

A testset run left 84 s in app.log with no line of any kind while every page
in the UI was frozen — a blocked event loop, which no phase log can show. The
watchdog is what makes that visible, so it needs to be known to fire on a
blocked loop and stay quiet on a healthy one.
"""

import asyncio
import contextlib
import logging
import time

from app.main import _watch_event_loop_lag


def _run_watchdog(block_seconds: float) -> None:
    async def scenario() -> None:
        task = asyncio.create_task(_watch_event_loop_lag(interval=0.05, threshold=0.2))
        await asyncio.sleep(0.1)
        if block_seconds:
            time.sleep(block_seconds)  # blocks the loop, like the stall did
        await asyncio.sleep(0.15)
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    asyncio.run(scenario())


def test_reports_a_blocked_loop(caplog):
    with caplog.at_level(logging.WARNING, logger="app.main"):
        _run_watchdog(block_seconds=0.6)
    messages = [r.getMessage() for r in caplog.records]
    assert any("Event loop blocked" in m for m in messages), messages


def test_stays_quiet_on_a_healthy_loop(caplog):
    with caplog.at_level(logging.WARNING, logger="app.main"):
        _run_watchdog(block_seconds=0)
    assert [r.getMessage() for r in caplog.records] == []
