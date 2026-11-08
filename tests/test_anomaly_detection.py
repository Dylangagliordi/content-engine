"""Tests for learn.anomaly_detection — IsolationForest, niche-scoped,
real DB fixture (zz_test_* prefix, cleaned up in a finally block) — same
discipline as tests/test_training_data.py.
"""
import uuid

import pytest

from qualume.db.connection import get_connection
from qualume.db.watchlist import add_watched_account
from qualume.learn.anomaly_detection import detect_anomalies


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
def fake_niche_normal():
    """10 normal-performing items, no injected outlier."""
    handle = f"zz_test_anomaly_{uuid.uuid4().hex[:8]}"
    _cleanup_account(handle)
    account_id = add_watched_account("tiktok", handle, niche="zz_test_anomaly_niche")

    items = []
    for i in range(10):
        item_id = _insert_item(account_id)
        _insert_snapshot(item_id, views=1000 + i * 10, likes=50 + i, comments=5, shares=2)
        items.append(item_id)

    yield account_id, items, handle
    _cleanup_account(handle)


@pytest.fixture
def fake_niche_with_viral_outlier():
    handle = f"zz_test_anomaly_viral_{uuid.uuid4().hex[:8]}"
    _cleanup_account(handle)
    account_id = add_watched_account("tiktok", handle, niche="zz_test_anomaly_viral_niche")

    items = []
    for i in range(10):
        item_id = _insert_item(account_id)
        _insert_snapshot(item_id, views=1000, likes=50, comments=5, shares=2)
        items.append(item_id)
    # real injected viral outlier: 50x+ views, proportionally high engagement too
    viral_id = _insert_item(account_id)
    _insert_snapshot(viral_id, views=100_000, likes=8000, comments=500, shares=300)
    items.append(viral_id)

    yield account_id, items, viral_id, handle
    _cleanup_account(handle)


@pytest.fixture
def fake_niche_with_underperformer():
    handle = f"zz_test_anomaly_under_{uuid.uuid4().hex[:8]}"
    _cleanup_account(handle)
    account_id = add_watched_account("tiktok", handle, niche="zz_test_anomaly_under_niche")

    items = []
    for i in range(10):
        item_id = _insert_item(account_id)
        _insert_snapshot(item_id, views=1000, likes=50, comments=5, shares=2)
        items.append(item_id)
    # real injected underperformer: real views, near-zero engagement
    flop_id = _insert_item(account_id)
    _insert_snapshot(flop_id, views=5000, likes=0, comments=0, shares=0)
    items.append(flop_id)

    yield account_id, items, flop_id, handle
    _cleanup_account(handle)


def test_detect_anomalies_returns_insufficient_data_below_floor(fake_niche_normal):
    account_id, items, handle = fake_niche_normal
    result = detect_anomalies(niche="zz_test_anomaly_niche", min_sample_size=20)
    assert result["status"] == "insufficient_data"
    assert result["n_items"] == 10
    assert result["viral_outliers"] == []
    assert result["underperformer_outliers"] == []


def test_detect_anomalies_flags_a_real_injected_viral_outlier(fake_niche_with_viral_outlier):
    account_id, items, viral_id, handle = fake_niche_with_viral_outlier
    result = detect_anomalies(niche="zz_test_anomaly_viral_niche", min_sample_size=10)
    assert result["status"] == "ok"
    viral_ids = {e["item_id"] for e in result["viral_outliers"]}
    assert viral_id in viral_ids


def test_detect_anomalies_flags_a_real_injected_underperformer(fake_niche_with_underperformer):
    account_id, items, flop_id, handle = fake_niche_with_underperformer
    result = detect_anomalies(niche="zz_test_anomaly_under_niche", min_sample_size=10)
    assert result["status"] == "ok"
    underperformer_ids = {e["item_id"] for e in result["underperformer_outliers"]}
    assert flop_id in underperformer_ids


def test_detect_anomalies_scopes_strictly_to_the_given_niche(fake_niche_with_viral_outlier):
    """A different, unrelated niche must never see this niche's items."""
    account_id, items, viral_id, handle = fake_niche_with_viral_outlier
    result = detect_anomalies(niche="zz_test_anomaly_a_niche_that_does_not_exist", min_sample_size=1)
    assert result["status"] == "insufficient_data"
    assert result["n_items"] == 0
