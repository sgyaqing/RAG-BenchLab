"""setup_logging must keep test output out of the server's log file.

A watchdog line written by a test (0.2 s threshold) is indistinguishable in
logs/app.log from a real stall, and the log is the only evidence a stall
leaves behind.
"""

import logging
from logging.handlers import RotatingFileHandler

from app.core.logging import setup_logging


def test_no_file_handler_under_pytest():
    root = logging.getLogger()
    before = list(root.handlers)
    try:
        setup_logging()
        assert not any(isinstance(h, RotatingFileHandler) for h in root.handlers)
        assert any(isinstance(h, logging.StreamHandler) for h in root.handlers)
    finally:
        root.handlers = before
