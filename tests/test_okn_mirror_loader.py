"""Tests for okn_mirror's data loader — written FIRST (TDD), before the loader
exists. This is the one piece of okn_mirror that's genuinely NEW qualume-specific
logic (not a straight port) — mapping qualume's real schema onto okn's
UNIFIED_SCHEMA — so it gets real DB-fixture tests, same discipline as
test_baseline.py/test_mining.py, before the analysis/ML ports (which mostly
just need to match upstream's already-proven formulas) get lighter smoke tests.
"""
import uuid
from datetime import datetime, timedelta, timezone

import pytest

from qualume.db.connection import get_connection
from qualume.db.watchlist import add_watched_account
from qualume.okn_mirror.loader import load_niche_dataframe


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


def _insert_video(account_id, views, likes=0, comments=0, shares=0,
                   content_type=None, caption="test caption", posted_at=None,
                   duration_seconds=30):
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO external_content_items
                   (watched_account_id, platform, platform_video_id, url,
                    caption, duration_seconds, content_type, posted_at)
                   VALUES (%s, 'tiktok', %s, 'https://example.com/fake',
                           %s, %s, %s, %s)
                   RETURNING id""",
                (account_id, uuid.uuid4().hex, caption, duration_seconds,
                 content_type, posted_at),
            )
            item_id = cur.fetchone()[0]
            cur.execute(
                """INSERT INTO external_performance_snapshots
                   (external_content_item_id, views, likes, comments, shares)
                   VALUES (%s, %s, %s, %s, %s)""",
                (item_id, views, likes, comments, shares),
            )
        conn.commit()
        return item_id
    finally:
        conn.close()


def _add_another_snapshot(item_id, views, likes=0, comments=0, shares=0):
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
def fake_account():
    handle = f"test_oknmirror_{uuid.uuid4().hex[:8]}"
    niche = f"zz_test_niche_{uuid.uuid4().hex[:8]}"
    _cleanup_account(handle)
    account_id = add_watched_account("tiktok", handle)
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("UPDATE watched_accounts SET niche = %s WHERE id = %s", (niche, account_id))
        conn.commit()
    finally:
        conn.close()
    yield account_id, niche
    _cleanup_account(handle)


def test_loader_maps_niche_into_platform_column(fake_account):
    account_id, niche = fake_account
    _insert_video(account_id, views=100, likes=10, comments=2, shares=1)
    df = load_niche_dataframe(niches=[niche], min_niche_size=1)
    assert len(df) == 1
    assert df.iloc[0]["platform"] == niche


def test_loader_computes_engagement_total_and_rate(fake_account):
    account_id, niche = fake_account
    _insert_video(account_id, views=1000, likes=100, comments=20, shares=5)
    df = load_niche_dataframe(niches=[niche], min_niche_size=1)
    row = df.iloc[0]
    assert row["engagement_total"] == 125  # 100+20+5+saves(0)
    assert row["reach"] == 1000  # TikTok: views stands in for reach, no separate IG-style reach
    assert abs(row["engagement_rate"] - 0.125) < 1e-9


def test_loader_uses_latest_snapshot_per_item_not_every_snapshot(fake_account):
    """Same real bug class as learn.mining/learn.baseline: a re-polled video
    must contribute ONE row (its latest snapshot), not one row per poll."""
    account_id, niche = fake_account
    item_id = _insert_video(account_id, views=50, likes=5)
    _add_another_snapshot(item_id, views=500, likes=50)  # later, bigger poll
    df = load_niche_dataframe(niches=[niche], min_niche_size=1)
    assert len(df) == 1  # not 2
    assert df.iloc[0]["reach"] == 500  # the LATEST snapshot, not the first


def test_loader_excludes_items_with_no_snapshot(fake_account):
    """An item with zero performance snapshots has nothing to analyze —
    okn's own pipeline only ever sees posts that HAVE metrics; same here."""
    account_id, niche = fake_account
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO external_content_items
                   (watched_account_id, platform, platform_video_id, url)
                   VALUES (%s, 'tiktok', %s, 'https://example.com/fake')""",
                (account_id, uuid.uuid4().hex),
            )
        conn.commit()
    finally:
        conn.close()
    _insert_video(account_id, views=100, likes=10)  # the only one WITH a snapshot
    df = load_niche_dataframe(niches=[niche], min_niche_size=1)
    assert len(df) == 1


def test_loader_defaults_missing_content_type_to_unclassified(fake_account):
    account_id, niche = fake_account
    _insert_video(account_id, views=100, likes=10, content_type=None)
    df = load_niche_dataframe(niches=[niche], min_niche_size=1)
    assert df.iloc[0]["content_type"] == "unclassified"


def test_loader_keeps_real_content_type_when_present(fake_account):
    account_id, niche = fake_account
    _insert_video(account_id, views=100, likes=10, content_type="organic")
    df = load_niche_dataframe(niches=[niche], min_niche_size=1)
    assert df.iloc[0]["content_type"] == "organic"


def test_loader_filters_niches_below_min_size(fake_account):
    account_id, niche = fake_account
    _insert_video(account_id, views=100, likes=10)
    _insert_video(account_id, views=200, likes=20)
    # min_niche_size=5 but this niche only has 2 real items -> excluded entirely
    df = load_niche_dataframe(niches=[niche], min_niche_size=5)
    assert len(df) == 0


def test_loader_isolates_one_niche_from_another(fake_account):
    account_id, niche = fake_account
    _insert_video(account_id, views=100, likes=10)

    other_handle = f"test_oknmirror_other_{uuid.uuid4().hex[:8]}"
    other_niche = f"zz_test_niche_other_{uuid.uuid4().hex[:8]}"
    _cleanup_account(other_handle)
    try:
        other_id = add_watched_account("tiktok", other_handle)
        conn = get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("UPDATE watched_accounts SET niche = %s WHERE id = %s", (other_niche, other_id))
            conn.commit()
        finally:
            conn.close()
        _insert_video(other_id, views=999, likes=99)

        df = load_niche_dataframe(niches=[niche], min_niche_size=1)
        assert len(df) == 1
        assert df.iloc[0]["platform"] == niche
    finally:
        _cleanup_account(other_handle)


def test_loader_falls_back_posted_at_to_discovered_at_when_null(fake_account):
    account_id, niche = fake_account
    _insert_video(account_id, views=100, likes=10, posted_at=None)
    df = load_niche_dataframe(niches=[niche], min_niche_size=1)
    assert df.iloc[0]["published_at"] is not None


def test_loader_keeps_engagement_rate_float64_even_with_a_zero_reach_row_elsewhere(fake_account):
    """Real bug found via scripts/run_okn_mirror.py's full-dataset run:
    pd.NA via .replace(0, pd.NA) on an int64 column upcasts the WHOLE
    engagement_rate column to object dtype the moment ANY row (in ANY
    niche, before per-niche filtering) has reach=0 — breaking sklearn/
    .nlargest() downstream the moment a niche gets sliced back out of the
    full multi-niche frame. This must stay float64 regardless of how many
    other niches/rows have reach=0."""
    account_id, niche = fake_account
    _insert_video(account_id, views=0, likes=0)  # reach=0 row, same niche
    _insert_video(account_id, views=100, likes=10)
    df = load_niche_dataframe(niches=[niche], min_niche_size=1)
    assert df["engagement_rate"].dtype == "float64"
    assert df[df["reach"] == 0]["engagement_rate"].iloc[0] == 0.0
