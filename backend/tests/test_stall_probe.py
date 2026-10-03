"""The stall probe has to fire *while* the loop is stuck.

The lag watchdog can only report a block once it has ended, and the frame that
caused it is gone by then — which is why eight 20 s stalls in one run were
never attributed to a line of code. This one runs on its own thread and reads
the stacks while the block is still in progress.
"""

import threading
import time
from types import SimpleNamespace


def test_dumps_the_stack_of_a_blocked_loop(tmp_path, monkeypatch):
    import app.main as main

    monkeypatch.setattr(main, "get_settings",
                        lambda: SimpleNamespace(log_dir=tmp_path))
    main._heartbeat["at"] = time.monotonic()

    stop = threading.Event()
    main._start_stall_probe(silence=0.5, repeat=0.3, stop=stop)
    try:
        # The probe polls once a second, so give it a couple of beats.
        deadline = time.monotonic() + 6
        dump = ""
        while time.monotonic() < deadline:
            if (tmp_path / "stall.log").exists():
                dump = (tmp_path / "stall.log").read_text()
                if "most recent call first" in dump:
                    break
            time.sleep(0.1)
    finally:
        stop.set()

    assert "loop silent" in dump, dump
    assert "most recent call first" in dump, "faulthandler must dump thread stacks"
    # Every thread, not just the caller. One block per thread, so the count is
    # what says all_threads reached past the current one — the thread *name* is
    # no use here: faulthandler only prints it from 3.14 on, and this asserts
    # the same thing on 3.13.
    assert dump.count("most recent call first") >= 2, (
        "all_threads must dump every thread, not only the calling one")
    # The probe's own frame is in there, so the stacks dumped are real ones.
    assert "in watch" in dump, "the probe's own frame must appear in the dump"


def test_a_healthy_loop_produces_no_dump(tmp_path, monkeypatch):
    """A probe that fires on a working system is noise nobody will read."""
    import app.main as main

    monkeypatch.setattr(main, "get_settings",
                        lambda: SimpleNamespace(log_dir=tmp_path))

    stop = threading.Event()
    main._start_stall_probe(silence=1.0, repeat=0.3, stop=stop)
    try:
        for _ in range(6):  # three beats' worth of a live loop
            main._heartbeat["at"] = time.monotonic()
            time.sleep(0.5)
    finally:
        stop.set()

    assert not (tmp_path / "stall.log").exists()
