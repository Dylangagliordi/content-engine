"""Tests for the pattern-mining pass — written FIRST per this project's TDD
discipline. Per plans/learn-engine-idea-sourcing-architecture.md
("Statistical method for mining"): simple per-feature tests (Welch's
t-test / Mann-Whitney U for a continuous outcome split by a categorical
DNA field), NOT regression or random forest, at this stage.

Uses real fixture rows (external_content_items + external_performance_
snapshots + external_creative_dna) with a KNOWN INJECTED correlation, per
the plan's own false-positive guard: "simple enough to unit-test against
synthetic data with known injected correlations."
"""
import uuid

import pytest

from qualume.db.connection import get_connection
from qualume.db.watchlist import add_watched_account
from qualume.learn.mining import mine_categorical_feature


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


def _insert_video(account_id, views: int, hook_type: str, energy_level: int):
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
                   VALUES (%s, %s, %s, 'test-model', 'v5')""",
                (item_id, hook_type, energy_level),
            )
        conn.commit()
        return item_id
    finally:
        conn.close()


@pytest.fixture
def fake_account():
    handle = f"test_mining_{uuid.uuid4().hex[:8]}"
    _cleanup_account(handle)
    account_id = add_watched_account("tiktok", handle)
    yield account_id, handle
    _cleanup_account(handle)


def test_mine_categorical_feature_finds_an_injected_correlation(fake_account):
    """Known injected correlation: a fake hook_type value consistently
    outperforms its account's median; another consistently underperforms.
    A real statistical test should surface this with a low p-value.

    Uses clearly fake, never-real values (zz_test_* prefix) so this can't
    collide with the real 58 production-annotated videos already in this
    shared dev DB (test_manual_entry.py's leak taught this project that
    lesson once already — same discipline here, applied to VALUES this
    time, not just row cleanup)."""
    account_id, _ = fake_account
    # zz_test_high hooks: well above a real median
    for views in [800, 900, 1000, 1100, 1200]:
        _insert_video(account_id, views, "zz_test_high", energy_level=3)
    # zz_test_low hooks: well below the same median
    for views in [50, 60, 70, 80, 90]:
        _insert_video(account_id, views, "zz_test_low", energy_level=3)

    results = mine_categorical_feature("hook_type", min_group_size=3)
    by_value = {r["value"]: r for r in results}

    assert "zz_test_high" in by_value
    assert by_value["zz_test_high"]["p_value"] < 0.05
    assert by_value["zz_test_high"]["median_ratio_with"] > by_value["zz_test_high"]["median_ratio_without"]

    assert "zz_test_low" in by_value
    assert by_value["zz_test_low"]["p_value"] < 0.05
    assert by_value["zz_test_low"]["median_ratio_with"] < by_value["zz_test_low"]["median_ratio_without"]


def test_mine_categorical_feature_skips_groups_below_min_size(fake_account):
    account_id, _ = fake_account
    for views in [100, 200, 300, 400, 500]:
        _insert_video(account_id, views, "zz_test_enough", energy_level=3)
    # only 2 videos for this hook_type - below min_group_size=3
    for views in [10, 20]:
        _insert_video(account_id, views, "zz_test_toofew", energy_level=3)

    results = mine_categorical_feature("hook_type", min_group_size=3)
    values = {r["value"] for r in results}
    assert "zz_test_toofew" not in values


def test_mine_categorical_feature_reports_sample_size(fake_account):
    account_id, _ = fake_account
    for views in [100, 200, 300, 400, 500]:
        _insert_video(account_id, views, "zz_test_five", energy_level=3)
    for views in [10, 20, 30]:
        _insert_video(account_id, views, "zz_test_three", energy_level=3)

    results = mine_categorical_feature("hook_type", min_group_size=3)
    by_value = {r["value"]: r for r in results}
    assert by_value["zz_test_five"]["n_with"] == 5
    assert by_value["zz_test_three"]["n_with"] == 3


def test_mine_categorical_feature_rejects_unknown_column():
    """Fail loud on a typo'd/unsupported column name rather than silently
    returning nothing or letting raw SQL injection through."""
    with pytest.raises(ValueError):
        mine_categorical_feature("hook_type; DROP TABLE insights_external;--")


def test_mine_categorical_feature_counts_each_video_once_even_with_multiple_performance_snapshots(
        fake_account):
    """Regression test for a real bug found running this against the
    actual 95-video production dataset after scheduled polling began
    refreshing existing videos' snapshots: the JOIN against
    external_performance_snapshots matches EVERY snapshot row for a
    video, not just the latest. A video polled 3 times (3 snapshot rows,
    the real shape poll_tiktok_watchlist now produces on every scheduled
    refresh) got its single hook_type value counted 3 TIMES with 3
    (possibly near-identical) ratio_to_median values — real production
    run showed 180 rows for 95 videos, artificially inflating every
    sample size and making p-values implausibly small via
    pseudo-replication (the same observation isn't 3 independent data
    points just because it was polled 3 times). Must use only the
    LATEST snapshot per video, same "one row per video" discipline this
    function already applies to multi-version annotations."""
    account_id, _ = fake_account
    item_id = _insert_video(account_id, 500, "zz_test_repolled", energy_level=3)
    # simulate 2 MORE snapshots for the SAME video (same real shape as
    # scheduled re-polling) - without the fix, this video now counts 3x.
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO external_performance_snapshots
                   (external_content_item_id, views, likes, comments, shares, ts)
                   VALUES (%s, 600, 0, 0, 0, now() + interval '1 hour')""",
                (item_id,),
            )
            cur.execute(
                """INSERT INTO external_performance_snapshots
                   (external_content_item_id, views, likes, comments, shares, ts)
                   VALUES (%s, 700, 0, 0, 0, now() + interval '2 hour')""",
                (item_id,),
            )
        conn.commit()
    finally:
        conn.close()
    # pad both groups above min_group_size so they'd otherwise be tested
    for views in [100, 200, 300, 400]:
        _insert_video(account_id, views, "zz_test_repolled", energy_level=3)
        _insert_video(account_id, views, "zz_test_control", energy_level=3)

    results = mine_categorical_feature("hook_type", min_group_size=3)
    by_value = {r["value"]: r for r in results}
    # 4 padding + this 1 re-polled video = 5, NOT 5 padding-equivalents
    # worth of weight from the re-polled video alone being triple-counted.
    assert by_value["zz_test_repolled"]["n_with"] == 5


def test_mine_categorical_feature_counts_each_video_once_even_with_multiple_annotation_versions(
        fake_account):
    """Regression test for a real bug found running this against the
    actual 58 v1+v5-annotated videos: a video re-annotated at a newer
    schema version (v1 then later v5, same external_content_item_id, two
    rows) was counted TWICE by the JOIN, with its v1 value in one group
    and its (possibly different) v5 value in another — the real run
    showed 118 rows instead of 58, silently inflating every sample size
    and corrupting every p-value. Must use only the latest annotation per
    video."""
    account_id, _ = fake_account
    item_id = _insert_video(account_id, 500, "zz_test_old_version", energy_level=3)
    # simulate a second, newer annotation pass on the SAME video with a
    # DIFFERENT hook_type — same real-world shape as v1 -> v5 upgrades.
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO external_creative_dna
                   (external_content_item_id, hook_type, energy_level,
                    analysis_model, analysis_prompt_version, analyzed_at)
                   VALUES (%s, 'zz_test_new_version', 3, 'test-model', 'v5', now() + interval '1 hour')""",
                (item_id,),
            )
        conn.commit()
    finally:
        conn.close()
    # pad both groups above min_group_size so they'd otherwise be tested
    for views in [100, 200, 300, 400]:
        _insert_video(account_id, views, "zz_test_old_version", energy_level=3)
        _insert_video(account_id, views, "zz_test_new_version", energy_level=3)

    results = mine_categorical_feature("hook_type", min_group_size=3)
    by_value = {r["value"]: r for r in results}
    # the old value should NOT get credit for this video (it has a newer
    # annotation now); the new value SHOULD.
    assert by_value["zz_test_new_version"]["n_with"] == 5  # 4 padding + this 1
    assert by_value["zz_test_old_version"]["n_with"] == 4  # padding only, not this video
