"""Tests for learn.performance_tiers — KMeans forced-k tiering,
niche-scoped, real DB fixture (zz_test_* prefix, cleaned up in a
finally block) — same discipline as test_anomaly_detection.py /
test_training_data.py.
"""
import uuid

import pytest

from qualume.db.connection import get_connection
from qualume.db.watchlist import add_watched_account
from qualume.learn.performance_tiers import compute_performance_tiers


def _cleanup_account(handle: str):
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM external_performance_snapshots WHERE external_content_item_id IN "
                "(SELECT id FROM external_content_items WHERE watched_account_id IN "
                "(SELECT id FROM watched_accounts WHERE handle = %s))",
                (handle,),
            )
            cur.execute(
                "DELETE FROM external_content_items WHERE watched_account_id IN "
                "(SELECT id FROM watched_accounts WHERE handle = %s)",
                (handle,),
            )
            cur.execute("DELETE FROM watched_accounts WHERE handle = %s", (handle,))
        conn.commit()
    finally:
        conn.close()


def _insert_item(account_id):
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO external_content_items
                   (watched_account_id, platform, platform_video_id, url,
                    local_video_path, duration_seconds)
                   VALUES (%s, 'tiktok', %s, 'https://example.com/fake', '/fake/path.mp4', 20)
                   RETURNING id""",
                (account_id, uuid.uuid4().hex),
            )
            item_id = cur.fetchone()[0]
        conn.commit()
        return item_id
    finally:
        conn.close()


def _insert_snapshot(item_id, views, likes=0, comments=0, shares=0):
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO external_performance_snapshots
                   (external_content_item_id, views, likes, comments, shares)
                   VALUES (%s, %s, %s, %s, %s)""",
                (item_id, views, likes, comments, shares),
            )
        conn.commit()
    finally:
        conn.close()


@pytest.fixture
def fake_niche_few_items():
    handle = f"zz_test_tiers_few_{uuid.uuid4().hex[:8]}"
    _cleanup_account(handle)
    account_id = add_watched_account("tiktok", handle, niche="zz_test_tiers_few_niche")

    items = []
    for i in range(5):
        item_id = _insert_item(account_id)
        _insert_snapshot(item_id, views=1000, likes=50, comments=5, shares=2)
        items.append(item_id)

    yield account_id, items, handle
    _cleanup_account(handle)


@pytest.fixture
def fake_niche_with_real_tier_separation():
    """20 items on one account - half real high-performers (high
    views/likes/comments/shares), half real low-performers - a real
    injected signal that should produce distinct tiers with the
    expected top/low ordering, same as every other synthetic-signal
    test in this codebase."""
    handle = f"zz_test_tiers_sep_{uuid.uuid4().hex[:8]}"
    _cleanup_account(handle)
    account_id = add_watched_account("tiktok", handle, niche="zz_test_tiers_sep_niche")

    high_items = []
    low_items = []
    for i in range(10):
        item_id = _insert_item(account_id)
        _insert_snapshot(item_id, views=10_000, likes=1000, comments=100, shares=50)
        high_items.append(item_id)
    for i in range(10):
        item_id = _insert_item(account_id)
        _insert_snapshot(item_id, views=200, likes=2, comments=0, shares=0)
        low_items.append(item_id)

    yield account_id, high_items, low_items, handle
    _cleanup_account(handle)


def test_compute_performance_tiers_returns_insufficient_data_below_floor(fake_niche_few_items):
    account_id, items, handle = fake_niche_few_items
    result = compute_performance_tiers(niche="zz_test_tiers_few_niche", min_sample_size=10)
    assert result["status"] == "insufficient_data"
    assert result["n_clusters"] == 0
    assert result["tiers"] == []


def test_compute_performance_tiers_separates_real_injected_high_and_low_performers(
    fake_niche_with_real_tier_separation,
):
    account_id, high_items, low_items, handle = fake_niche_with_real_tier_separation
    result = compute_performance_tiers(niche="zz_test_tiers_sep_niche", min_sample_size=10)
    assert result["status"] == "ok"
    assert result["n_clusters"] >= 2

    tiers = result["tiers"]
    # tiers sorted descending by avg_ratio_to_median - top tier first
    assert tiers[0]["avg_ratio_to_median"] >= tiers[-1]["avg_ratio_to_median"]
    assert tiers[0]["label"] == "[TOP] Top Performers"

    top_item_ids = set(tiers[0]["item_ids"])
    # The real high-performing items should dominate the top tier.
    high_in_top = len(top_item_ids & set(high_items))
    low_in_top = len(top_item_ids & set(low_items))
    assert high_in_top > low_in_top


def test_compute_performance_tiers_scopes_strictly_to_the_given_niche(
    fake_niche_with_real_tier_separation,
):
    account_id, high_items, low_items, handle = fake_niche_with_real_tier_separation
    result = compute_performance_tiers(
        niche="zz_test_tiers_a_niche_that_does_not_exist", min_sample_size=1
    )
    assert result["status"] == "insufficient_data"
    assert result["n_clusters"] == 0
