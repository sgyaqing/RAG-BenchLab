import logging
import sys
from logging.handlers import RotatingFileHandler

from app.core.config import get_settings

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


def _running_under_pytest() -> bool:
    return "pytest" in sys.modules


def setup_logging() -> None:
    """Send logs to stdout, and to logs/app.log when running as the app.

    The file handler is deliberately skipped under pytest. The test client
    runs the app lifespan, so every test run otherwise appends its records —
    fixture names like FakeTransform included — to the same logs/app.log the
    real server writes, and a watchdog line from a test with a 0.2 s threshold
    reads exactly like a production stall. The log is the only evidence a
    stall leaves behind, so it has to hold the server's output alone.
    """
    settings = get_settings()
    formatter = logging.Formatter(LOG_FORMAT)

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if not _running_under_pytest():
        settings.log_dir.mkdir(parents=True, exist_ok=True)
        handlers.insert(
            0,
            RotatingFileHandler(
                settings.log_dir / "app.log",
                maxBytes=10 * 1024 * 1024,
                backupCount=5,
                encoding="utf-8",
            ),
        )

    for handler in handlers:
        handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers = handlers
