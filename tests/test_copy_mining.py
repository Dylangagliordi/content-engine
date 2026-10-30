"""Tests for boolean copy-technique mining — written FIRST per this
project's TDD discipline.

Parallel to mine_categorical_feature (mining.py) but NOT a reuse of it:
that function assumes a text categorical column on external_creative_dna
and tests "value X vs everything else"; copy-technique flags are
BOOLEAN columns on a DIFFERENT table (external_copy_analysis) where the
natural test is simply "flag=True vs flag=False" — a real enough
structural difference (different table, different column type, only two
groups not N) to warrant its own function rather than forcing a boolean
through a function built for multi-value categoricals.
"""
import uuid

import pytest

from qualume.db.connection import get_connection
from qualume.db.watchlist import add_watched_account
from qualume.learn.copy_mining import MINEABLE_COPY_FLAGS, mine_boolean_copy_feature


def _cleanup_account(handle: str):
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM external_copy_analysis WHERE external_content_item_id IN "
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


def _insert_item_with_flag_and_views(account_id, flag_column, flag_value, views):
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
                f"""INSERT INTO external_copy_analysis
                   (external_content_item_id, {flag_column}, analysis_model, analysis_prompt_version)
                   VALUES (%s, %s, 'test-model', 'v1')""",
                (item_id, flag_value),
            )
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
    handle = f"test_copy_mining_{uuid.uuid4().hex[:8]}"
    _cleanup_account(handle)
    account_id = add_watched_account("tiktok", handle)
    yield account_id, handle
    _cleanup_account(handle)


def test_rejects_non_allowlisted_column():
    with pytest.raises(ValueError, match="not a mineable copy flag"):
        mine_boolean_copy_feature("external_content_item_id")  # real column, not a flag


def test_detects_a_real_injected_effect(fake_account):
    """6 videos with creates_urgency=True all at 5x account median, 6 with
    False at ~1x median - mann-whitney should find this real (planted)
    difference, same style of test as mine_categorical_feature's own
    smoke test."""
    account_id, _ = fake_account
    for _ in range(6):
        _insert_item_with_flag_and_views(account_id, "creates_urgency", True, 5000)
    for _ in range(6):
        _insert_item_with_flag_and_views(account_id, "creates_urgency", False, 1000)

    result = mine_boolean_copy_feature("creates_urgency", watched_account_id=account_id)

    assert result is not None
    assert result["flag"] == "creates_urgency"
    assert result["n_true"] == 6
    assert result["n_false"] == 6
    assert result["median_ratio_true"] > result["median_ratio_false"]
    assert result["p_value"] < 0.05


def test_returns_none_when_either_group_below_min_size(fake_account):
    account_id, _ = fake_account
    # Only 2 True, far below default min_group_size=5
    for _ in range(2):
        _insert_item_with_flag_and_views(account_id, "uses_social_proof", True, 5000)
    for _ in range(6):
        _insert_item_with_flag_and_views(account_id, "uses_social_proof", False, 1000)

    result = mine_boolean_copy_feature("uses_social_proof", watched_account_id=account_id)

    assert result is None


def test_skips_null_flag_values_not_just_false(fake_account):
    """A NULL flag (classifier couldn't tell) must not be silently
    counted as False - that would corrupt the false-group's real
    composition. Only rows with an explicit True or False count."""
    account_id, _ = fake_account
    for _ in range(6):
        _insert_item_with_flag_and_views(account_id, "explains_mechanism", True, 5000)
    for _ in range(6):
        _insert_item_with_flag_and_views(account_id, "explains_mechanism", False, 1000)
    for _ in range(10):
        _insert_item_with_flag_and_views(account_id, "explains_mechanism", None, 9999999)

    result = mine_boolean_copy_feature("explains_mechanism", watched_account_id=account_id)

    assert result["n_true"] == 6
    assert result["n_false"] == 6  # NOT 16


def test_mineable_copy_flags_covers_all_22_real_taxonomy_flags():
    """Guard against someone adding a 23rd flag to the schema/classifier
    and forgetting to add it here - this allowlist must track the real
    taxonomy or new flags silently become unmineable."""
    assert len(MINEABLE_COPY_FLAGS) == 22
    assert "creates_urgency" in MINEABLE_COPY_FLAGS
    assert "hook_pain_callout" in MINEABLE_COPY_FLAGS
    assert "external_content_item_id" not in MINEABLE_COPY_FLAGS
    assert "raw_response" not in MINEABLE_COPY_FLAGS


def test_mine_boolean_copy_feature_counts_each_video_once_even_with_multiple_performance_snapshots(
        fake_account):
    """Same real bug, same fix, as mining.py's regression test of the
    same name: the JOIN against external_performance_snapshots matched
    EVERY snapshot row for a video (3 rows for a video re-polled 3 times
    on the real production account), triple-counting that single
    observation and corrupting every p-value via pseudo-replication.
    Must use only the LATEST snapshot per video."""
    account_id, _ = fake_account
    item_id = _insert_item_with_flag_and_views(account_id, "creates_urgency", True, 5000)
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO external_performance_snapshots
                   (external_content_item_id, views, likes, comments, shares, ts)
                   VALUES (%s, 5100, 0, 0, 0, now() + interval '1 hour')""",
                (item_id,),
            )
            cur.execute(
                """INSERT INTO external_performance_snapshots
                   (external_content_item_id, views, likes, comments, shares, ts)
                   VALUES (%s, 5200, 0, 0, 0, now() + interval '2 hour')""",
                (item_id,),
            )
        conn.commit()
    finally:
        conn.close()
    # pad with 4 more True + 5 False, same shape as the existing smoke test
    for _ in range(4):
        _insert_item_with_flag_and_views(account_id, "creates_urgency", True, 5000)
    for _ in range(5):
        _insert_item_with_flag_and_views(account_id, "creates_urgency", False, 1000)

    result = mine_boolean_copy_feature("creates_urgency", watched_account_id=account_id)

    assert result is not None
    assert result["n_true"] == 5  # NOT 7 (the re-polled video triple-counted)
