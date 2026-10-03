"""Which fallback lever a round earns.

The goal is to deliver the count the customer asked for. Both levers serve it —
rotating the window onto material the graph has not been asked about, and
merging new documents — but one is free and the other costs a sub-KG build, an
LLM extraction pass and a full graph save. The free one goes first, and the
graph is asked until it stops answering.
"""

from app.services.testset_gen import _widen_pool_after_a_barren_round


def test_a_round_that_gained_keeps_rotating():
    """The material still has something to say — no need to pay for more."""
    assert not _widen_pool_after_a_barren_round(gained=1)
    assert not _widen_pool_after_a_barren_round(gained=7)


def test_a_barren_round_earns_the_expensive_lever():
    """Round 1 of one run: generated 2, the reviewer dropped both as duplicates."""
    assert _widen_pool_after_a_barren_round(gained=0)


def test_losing_ground_counts_as_barren():
    """`kept` is rebuilt from the whole pool each round and the reviewer's
    rejections are sticky, so a round can end below where it started. That is
    not a round the graph answered either."""
    assert _widen_pool_after_a_barren_round(gained=-1)
