# Imported first, and deliberately: ragas posts a usage event from inside every
# LLM call, synchronously, on the calling thread — see app/core/telemetry.py.
# This has to land before any other `app.*` import pulls ragas in.
from app.core import telemetry  # noqa: F401
