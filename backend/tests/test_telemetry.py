"""ragas' usage reporting must stay off.

It posts a usage event to t.explodinggradients.com from inside
`agenerate_text`, with `requests`, synchronously on the calling thread — and
for the testset pipeline that thread is the event loop. The one-second timeout
on that call does not cover name resolution, so a stalled DNS lookup froze the
loop for 10-20 s at a time: the UI hung, our own LLM calls timed out with
their answers already sitting in a socket, and the retries succeeded the
moment the loop came back.

ragas caches the answer with lru_cache, so the opt-out has to be set before
its first tracked call. If the import in app/__init__.py is ever dropped, this
is what notices.
"""


def test_ragas_usage_reporting_is_disabled():
    import app.services.testset_gen  # noqa: F401  - this is what pulls ragas in
    from ragas._analytics import do_not_track

    assert do_not_track() is True


def test_the_opt_out_is_set_before_ragas_loads():
    """Setting it afterwards would be too late, which is why it lives in
    app/__init__.py rather than somewhere convenient."""
    import os

    import app  # noqa: F401

    assert os.environ.get("RAGAS_DO_NOT_TRACK") == "true"
