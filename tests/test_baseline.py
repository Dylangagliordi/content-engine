"""Tests for the account-level performance baseline — written FIRST, per
this project's TDD discipline, before src/qualume/learn/baseline.py exists.

Per plans/learn-engine-idea-sourcing-architecture.md ("Own-content
performance needs a baseline"): a raw view count can't be judged in
absolute terms, only against what's normal for THAT specific account.
Computed fresh as a windowed query, never persisted (same append-only
principle as performance_snapshots) — so these tests hit the real
external_performance_snapshots/external_content_items tables with real
fixture rows, not a persisted baseline table (there isn't one, by design).
"""
import uuid

import pytest

from qualume.db.connection import get_connection
from qualume.db.watchlist import add_watched_account
from qualume.learn.baseline import account_median_views, performance_vs_baseline


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


def _insert_video_with_views(account_id, views: int) -> str:
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO external_content_items
                   (watched_account_id, platform, platform_video_id, url)
                   VALUES (%s, 'tiktok', %s, 'https://example.com/fake')
                   RETURNING id""",
                (account_id, uuid.uuid4().hex),
            )
            item_id = cur.fetchone()[0]
            cur.execute(
                """INSERT INTO external_performance_snapshots
                   (external_content_item_id, views, likes, comments, shares)
                   VALUES (%s, %s, 0, 0, 0)""",
                (item_id, views),
            )
        conn.commit()
        return item_id
    finally:
        conn.close()


@pytest.fixture
def fake_account():
    handle = f"test_baseline_{uuid.uuid4().hex[:8]}"
    _cleanup_account(handle)
    account_id = add_watched_account("tiktok", handle)
    yield account_id, handle
    _cleanup_account(handle)


def test_account_median_views_computes_the_real_median(fake_account):
    account_id, _ = fake_account
    for views in [100, 200, 300, 400, 500]:
        _insert_video_with_views(account_id, views)
    assert account_median_views(account_id) == 300


def test_account_median_views_handles_even_count(fake_account):
    account_id, _ = fake_account
    for views in [100, 200, 300, 400]:
        _insert_video_with_views(account_id, views)
    assert account_median_views(account_id) == 250  # (200+300)/2


def test_account_median_views_returns_none_for_account_with_no_videos(fake_account):
    account_id, _ = fake_account
    assert account_median_views(account_id) is None


def test_account_median_views_ignores_other_accounts(fake_account):
    """The whole point of per-account baselining: one account's median must
    not be polluted by a totally different account's numbers."""
    account_id, _ = fake_account
    _insert_video_with_views(account_id, 100)

    other_handle = f"test_baseline_other_{uuid.uuid4().hex[:8]}"
    _cleanup_account(other_handle)
    try:
        other_id = add_watched_account("tiktok", other_handle)
        for views in [100_000, 200_000, 300_000]:  # a much bigger account
            _insert_video_with_views(other_id, views)
        assert account_median_views(account_id) == 100  # unaffected by the other account
        assert account_median_views(other_id) == 200_000
    finally:
        _cleanup_account(other_handle)


def test_performance_vs_baseline_above_median(fake_account):
    account_id, _ = fake_account
    for views in [100, 200, 300]:
        _insert_video_with_views(account_id, views)
    item_id = _insert_video_with_views(account_id, 1000)
    result = performance_vs_baseline(item_id)
    assert result["views"] == 1000
    # median of [100, 200, 300, 1000] (this video's own performance is part
    # of the account's overall pool, not excluded) = (200+300)/2 = 250
    assert result["account_median_views"] == 250
    assert result["ratio_to_median"] == 4.0
    assert result["above_median"] is True


def test_performance_vs_baseline_below_median(fake_account):
    account_id, _ = fake_account
    for views in [500, 600, 700]:
        _insert_video_with_views(account_id, views)
    item_id = _insert_video_with_views(account_id, 10)
    result = performance_vs_baseline(item_id)
    assert result["above_median"] is False
    assert result["ratio_to_median"] < 1.0


def test_performance_vs_baseline_raises_for_unknown_item():
    with pytest.raises(ValueError):
        performance_vs_baseline(str(uuid.uuid4()))


def test_performance_vs_baseline_handles_single_video_account(fake_account):
    """One video IS its own median - ratio should be exactly 1.0, not a
    crash or a division-by-zero."""
    account_id, _ = fake_account
    item_id = _insert_video_with_views(account_id, 500)
    result = performance_vs_baseline(item_id)
    assert result["account_median_views"] == 500
    assert result["ratio_to_median"] == 1.0
