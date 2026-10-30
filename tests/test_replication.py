"""Tests for cross-segment replication checking — written FIRST per this
project's TDD discipline.

Per plans/learn-engine-idea-sourcing-architecture.md's insight-validity
critic (5b'): "Hold-out replication: pattern must hold on held-out data
split (time or account)". This operationalizes the account/niche split
specifically — segment_by="account" checks whether a feature's effect
direction is consistent across DIFFERENT watched accounts (not just one
account's lucky videos); segment_by="watchlist_type" checks whether it
holds across the broader categories already in the schema (competitor,
cross_niche, aspirational, etc. — see watched_accounts.watchlist_type).

This is also the concrete mechanism behind "does this hold across
niches, or is it one account's luck" — not meaningful with today's 2
accounts (both the same watchlist_type), but the function needs to be
correct NOW so it's ready the moment more accounts/niches are added.
"""
import uuid

import pytest

from qualume.db.connection import get_connection
from qualume.db.watchlist import add_watched_account
from qualume.learn.replication import check_segment_replication


def _cleanup_account(handle: str):
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM external_creative_dna WHERE external_content_item_id IN "
                "(SELECT id FROM external_content_items WHERE watched_account_id IN "
                "(SELECT id FROM watched_accounts WHERE handle = %s))",
                (handle,),
            )
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


def _insert_video(account_id, views: int, hook_type: str):
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
            cur.execute(
                """INSERT INTO external_creative_dna
                   (external_content_item_id, hook_type, energy_level,
                    analysis_model, analysis_prompt_version)
                   VALUES (%s, %s, 3, 'test-model', 'v5')""",
                (item_id, hook_type),
            )
        conn.commit()
        return item_id
    finally:
        conn.close()


@pytest.fixture
def two_fake_accounts():
    """Two fake accounts, deliberately NOT the same watchlist_type, so
    segment_by='watchlist_type' tests have something real to segment by."""
    h1 = f"test_repl_a_{uuid.uuid4().hex[:8]}"
    h2 = f"test_repl_b_{uuid.uuid4().hex[:8]}"
    _cleanup_account(h1)
    _cleanup_account(h2)
    a1 = add_watched_account("tiktok", h1, watchlist_type="competitor")
    a2 = add_watched_account("tiktok", h2, watchlist_type="cross_niche")
    yield a1, a2
    _cleanup_account(h1)
    _cleanup_account(h2)


def test_replication_agrees_when_effect_direction_matches_across_accounts(two_fake_accounts):
    """A real injected pattern that holds the SAME direction in both
    accounts: zz_test_high outperforms within EACH account separately,
    not just pooled."""
    a1, a2 = two_fake_accounts
    for account_id in (a1, a2):
        for views in [800, 900, 1000]:
            _insert_video(account_id, views, "zz_test_high")
        for views in [50, 60, 70]:
            _insert_video(account_id, views, "zz_test_low")

    result = check_segment_replication(
        "hook_type", "zz_test_high", segment_by="account", min_group_size=2
    )
    assert result["n_segments_tested"] == 2
    assert result["n_segments_agreeing"] == 2
    assert result["replicates"] is True


def test_replication_disagrees_when_effect_direction_flips_across_accounts(two_fake_accounts):
    """A fake pattern that's actually just one account's quirk: positive
    in account 1, NEGATIVE in account 2 — this should be flagged as NOT
    replicating, exactly the false-positive this check exists to catch."""
    a1, a2 = two_fake_accounts
    # account 1: zz_test_mixed outperforms
    for views in [800, 900, 1000]:
        _insert_video(a1, views, "zz_test_mixed")
    for views in [50, 60, 70]:
        _insert_video(a1, views, "zz_test_other")
    # account 2: zz_test_mixed UNDERperforms - opposite direction
    for views in [50, 60, 70]:
        _insert_video(a2, views, "zz_test_mixed")
    for views in [800, 900, 1000]:
        _insert_video(a2, views, "zz_test_other")

    result = check_segment_replication(
        "hook_type", "zz_test_mixed", segment_by="account", min_group_size=2
    )
    assert result["n_segments_tested"] == 2
    assert result["n_segments_agreeing"] == 1
    assert result["replicates"] is False


def test_replication_segments_by_watchlist_type(two_fake_accounts):
    """segment_by='watchlist_type' groups ALL accounts sharing a type
    together as one segment, rather than one segment per account."""
    a1, a2 = two_fake_accounts  # competitor, cross_niche respectively
    for views in [800, 900, 1000]:
        _insert_video(a1, views, "zz_test_typeseg")
    for views in [50, 60, 70]:
        _insert_video(a1, views, "zz_test_other")
    for views in [800, 900, 1000]:
        _insert_video(a2, views, "zz_test_typeseg")
    for views in [50, 60, 70]:
        _insert_video(a2, views, "zz_test_other")

    result = check_segment_replication(
        "hook_type", "zz_test_typeseg", segment_by="watchlist_type", min_group_size=2
    )
    # competitor and cross_niche are each their own segment here (2 accounts,
    # 2 different types) - same effective shape as per-account in this
    # fixture, but exercising the watchlist_type code path specifically.
    assert result["n_segments_tested"] == 2
    assert result["segment_ids"] == {"competitor", "cross_niche"}


def test_replication_skips_segments_below_min_group_size(two_fake_accounts):
    a1, a2 = two_fake_accounts
    for views in [800, 900, 1000]:
        _insert_video(a1, views, "zz_test_enough")
    for views in [50, 60, 70]:
        _insert_video(a1, views, "zz_test_other")
    # account 2 has only ONE video with this hook_type - below min_group_size
    _insert_video(a2, 500, "zz_test_enough")
    _insert_video(a2, 500, "zz_test_other")

    result = check_segment_replication(
        "hook_type", "zz_test_enough", segment_by="account", min_group_size=3
    )
    assert result["n_segments_tested"] == 1  # only account 1 qualifies


def test_replication_rejects_unknown_segment_by():
    with pytest.raises(ValueError):
        check_segment_replication("hook_type", "x", segment_by="not_a_real_thing")


def test_replication_rejects_unknown_column():
    with pytest.raises(ValueError):
        check_segment_replication("not_a_real_column; DROP TABLE insights_external;--", "x")


def test_replication_handles_zero_qualifying_segments():
    """No real data matches at all - must report a clean 'untested'
    result, not crash or claim replication with zero evidence."""
    result = check_segment_replication(
        "hook_type", "zz_test_nonexistent_value", segment_by="account", min_group_size=5
    )
    assert result["n_segments_tested"] == 0
    assert result["replicates"] is None  # not "False" - genuinely untested


def test_replication_counts_each_video_once_even_with_multiple_performance_snapshots(
        two_fake_accounts):
    """Same real bug class, same fix, as mining.py's regression test of
    the same name: the JOIN against external_performance_snapshots
    matched EVERY snapshot row for a video, not just its latest. A
    re-polled video (the real shape scheduled polling now produces) got
    pseudo-replicated once per snapshot, inflating n_with/n_without and
    corrupting the replication check's own per-segment sample sizes —
    concretely, a segment with only 3 REAL videos could wrongly clear a
    min_group_size of 4 just because one of those videos had multiple
    snapshot rows. min_group_size=4 here is deliberately set ABOVE the
    real distinct-video count (3) so the bug (pseudo-replicated rows
    clearing the threshold) and the fix (correctly NOT clearing it)
    produce different, discriminating results."""
    a1, a2 = two_fake_accounts
    item_id = _insert_video(a1, 1000, "zz_test_repolled")
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO external_performance_snapshots
                   (external_content_item_id, views, likes, comments, shares, ts)
                   VALUES (%s, 1100, 0, 0, 0, now() + interval '1 hour')""",
                (item_id,),
            )
            cur.execute(
                """INSERT INTO external_performance_snapshots
                   (external_content_item_id, views, likes, comments, shares, ts)
                   VALUES (%s, 1200, 0, 0, 0, now() + interval '2 hour')""",
                (item_id,),
            )
        conn.commit()
    finally:
        conn.close()
    # 2 more real distinct videos, each with just one snapshot - 3 real
    # videos total in this group, each with (3, 1, 1) snapshot rows
    # respectively = 5 total rows if the bug is present, 3 if fixed.
    for views in [900, 950]:
        _insert_video(a1, views, "zz_test_repolled")
    # "without" group padded to clear min_group_size on its own, so this
    # test isolates the "with" group's pseudo-replication specifically
    # rather than being skipped for an unrelated reason.
    for views in [50, 60, 70, 80]:
        _insert_video(a1, views, "zz_test_other")

    result = check_segment_replication(
        "hook_type", "zz_test_repolled", segment_by="account", min_group_size=4
    )
    # Correct (deduped) behavior: only 3 real videos, below min_group_size=4
    # -> genuinely untested. The bug would have let 5 pseudo-rows clear it.
    assert result["n_segments_tested"] == 0
