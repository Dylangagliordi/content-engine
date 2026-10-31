"""Tests for array/nested-field mining — written FIRST per this project's
TDD discipline.

Parallel to learn.copy_mining (mine_boolean_copy_feature tests "flag
present vs absent") but for a DIFFERENT real shape: a JSONB ARRAY column
(music_mood_v2 — e.g. ["upbeat", "calm"]) or a JSONB ARRAY-OF-OBJECTS
column with a nested string attribute to extract (presenters[].
relatable_archetype — e.g. [{"relatable_archetype": "mom", ...}]).
Neither of these is a single flat categorical column (learn.mining) or a
fixed 22-item taxonomy with boolean columns (learn.copy_mining) — the
real values aren't known upfront (a new archetype like "herbalist" or a
new music mood can appear any time a new video is annotated), so this
discovers distinct values from the real data rather than requiring a
hardcoded list, same "the data decides the categories, not a human
upfront" discipline as the DB-level allow-lists elsewhere.

Judges each video against its OWN account's median (learn.baseline), not
raw view counts — same mechanism as every other mining module.
"""
import uuid

import pytest

from qualume.db.connection import get_connection
from qualume.db.watchlist import add_watched_account
from qualume.learn.array_mining import (
    MINEABLE_ARRAY_COLUMNS,
    MINEABLE_NESTED_FIELDS,
    mine_array_value,
    mine_nested_field_value,
)


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


def _insert_video(account_id, views: int, music_mood_v2=None, presenters=None):
    import json

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
                   (external_content_item_id, music_mood_v2, presenters,
                    analysis_model, analysis_prompt_version)
                   VALUES (%s, %s, %s, 'test-model', 'v6')""",
                (item_id, music_mood_v2, json.dumps(presenters) if presenters is not None else None),
            )
        conn.commit()
        return item_id
    finally:
        conn.close()


@pytest.fixture
def fake_account():
    handle = f"test_arraymine_{uuid.uuid4().hex[:8]}"
    _cleanup_account(handle)
    account_id = add_watched_account("tiktok", handle)
    yield account_id, handle
    _cleanup_account(handle)


def test_mine_array_value_rejects_unknown_column():
    with pytest.raises(ValueError, match="not a mineable array column"):
        mine_array_value("not_a_real_column; DROP TABLE insights_external;--", "x")


def test_mine_array_value_finds_a_real_injected_effect(fake_account):
    """6 videos with 'upbeat' in music_mood_v2 at 5x account median, 6
    without it at ~1x — Mann-Whitney should find this real (planted)
    difference."""
    account_id, _ = fake_account
    for _ in range(6):
        _insert_video(account_id, 5000, music_mood_v2=["upbeat", "pop"])
    for _ in range(6):
        _insert_video(account_id, 1000, music_mood_v2=["calm", "sincere"])

    result = mine_array_value("music_mood_v2", "upbeat", min_group_size=5, watched_account_id=account_id)

    assert result is not None
    assert result["n_with"] == 6
    assert result["n_without"] == 6
    assert result["median_ratio_with"] > result["median_ratio_without"]
    assert result["p_value"] < 0.05


def test_mine_array_value_returns_none_below_min_group_size(fake_account):
    account_id, _ = fake_account
    for _ in range(2):
        _insert_video(account_id, 5000, music_mood_v2=["upbeat"])
    for _ in range(6):
        _insert_video(account_id, 1000, music_mood_v2=["calm"])

    result = mine_array_value("music_mood_v2", "upbeat", min_group_size=5, watched_account_id=account_id)
    assert result is None


def test_mine_array_value_videos_without_the_array_at_all_count_as_without(fake_account):
    """A video with music_mood_v2=NULL (no music detected at all) should
    count toward the 'without' group for any specific mood value, not be
    silently excluded — the value genuinely isn't present."""
    account_id, _ = fake_account
    for _ in range(5):
        _insert_video(account_id, 5000, music_mood_v2=["upbeat"])
    for _ in range(6):
        _insert_video(account_id, 1000, music_mood_v2=None)

    result = mine_array_value("music_mood_v2", "upbeat", min_group_size=5, watched_account_id=account_id)
    assert result is not None
    assert result["n_without"] == 6


def test_mine_nested_field_value_rejects_unknown_field():
    with pytest.raises(ValueError, match="not a mineable nested field"):
        mine_nested_field_value(
            "presenters", "not_a_real_field; DROP TABLE insights_external;--", "x"
        )


def test_mine_nested_field_value_finds_a_real_injected_effect(fake_account):
    """6 videos presented by a 'mom' archetype at 5x median, 6 by a
    'nurse' archetype at ~1x — should find this real planted difference."""
    account_id, _ = fake_account
    for _ in range(6):
        _insert_video(account_id, 5000, presenters=[{"relatable_archetype": "mom"}])
    for _ in range(6):
        _insert_video(account_id, 1000, presenters=[{"relatable_archetype": "nurse"}])

    result = mine_nested_field_value(
        "presenters", "relatable_archetype", "mom", min_group_size=5, watched_account_id=account_id
    )

    assert result is not None
    assert result["n_with"] == 6
    assert result["n_without"] == 6
    assert result["median_ratio_with"] > result["median_ratio_without"]
    assert result["p_value"] < 0.05


def test_mine_nested_field_value_handles_multiple_presenters_per_video(fake_account):
    """A video with BOTH a mom and a child presenter should count toward
    the 'with' group for 'mom' once, not be double-counted or skipped."""
    account_id, _ = fake_account
    for _ in range(5):
        _insert_video(
            account_id, 5000,
            presenters=[{"relatable_archetype": "mom"}, {"relatable_archetype": "child"}],
        )
    for _ in range(6):
        _insert_video(account_id, 1000, presenters=[{"relatable_archetype": "nurse"}])

    result = mine_nested_field_value(
        "presenters", "relatable_archetype", "mom", min_group_size=5, watched_account_id=account_id
    )
    assert result["n_with"] == 5


def test_mine_nested_field_value_empty_presenters_list_counts_as_without(fake_account):
    """A product-only/voiceover-over-broll video (presenters=[]) has no
    presenter at all — must count toward 'without' for any archetype,
    not be silently excluded."""
    account_id, _ = fake_account
    for _ in range(5):
        _insert_video(account_id, 5000, presenters=[{"relatable_archetype": "mom"}])
    for _ in range(6):
        _insert_video(account_id, 1000, presenters=[])

    result = mine_nested_field_value(
        "presenters", "relatable_archetype", "mom", min_group_size=5, watched_account_id=account_id
    )
    assert result["n_without"] == 6


def test_discover_array_values_returns_real_distinct_values_seen(fake_account):
    """No hardcoded list of music moods/archetypes — real values are
    discovered from whatever's actually in the data, so a brand new
    archetype/mood Gemini has never produced before is automatically
    minable the moment it appears, no code change required."""
    account_id, _ = fake_account
    _insert_video(account_id, 1000, music_mood_v2=["upbeat", "pop"])
    _insert_video(account_id, 1000, music_mood_v2=["calm"])

    from qualume.learn.array_mining import discover_array_values
    values = discover_array_values("music_mood_v2", watched_account_id=account_id)
    assert values == {"upbeat", "pop", "calm"}
