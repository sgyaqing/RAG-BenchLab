"""Opt out of ragas' anonymous usage reporting.

ragas posts a usage event to t.explodinggradients.com from inside
`agenerate_text`, with `requests` — synchronously, on whatever thread the call
is on. For the testset pipeline that thread is the event loop, so every LLM
call blocks the loop for the duration of a network round trip to a third
party.

The call carries a one-second timeout, but that does not cover name
resolution: `getaddrinfo` runs before the connect timeout applies, and a
stalled lookup froze the loop for 10-20 s at a time — eight times in one
measured run. That freeze is the whole story of the "stalls":

* the UI's every page hung, which is what was reported from the browser;
* the lag watchdog logged "Event loop blocked for 23.7s";
* our own LLM calls timed out and were retried — the responses had arrived in
  under 0.7 s, the loop just was not running to collect them, which is why the
  retry succeeded instantly and why lowering the timeout barely helped.

Stack from a stall, which is what identified it:

    socket.py:987            getaddrinfo
    urllib3/connection.py    connect
    requests/api.py:134      post
    ragas/_analytics.py:233  track
    ragas/llms/base.py:319   agenerate_text

A product also should not be reporting usage to a third party on the
customer's behalf, so this is the right default regardless of the stalls.

`do_not_track()` in ragas is `lru_cache`d, so this has to be set before the
first tracked call — importing this module from `app/__init__.py` puts it
ahead of every other `app.*` import, and therefore ahead of ragas.
"""

import os

os.environ["RAGAS_DO_NOT_TRACK"] = "true"
