import uuid
from unittest.mock import patch, MagicMock

from qualume.ingest.tiktok_reader import (
    parse_video_entry,
    ingest_videos_for_account,
)
from qualume.db.watchlist import add_watched_account
from qualume.db.connection import get_connection


def _cleanup_account(handle: str):
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """DELETE FROM external_performance_snapshots
                   WHERE external_content_item_id IN (
                       SELECT eci.id FROM external_content_items eci
                       JOIN watched_accounts wa ON wa.id = eci.watched_account_id
                       WHERE wa.handle = %s)""",
                (handle,),
            )
            cur.execute(
                """DELETE FROM external_content_items
                   WHERE watched_account_id IN (
                       SELECT id FROM watched_accounts WHERE handle = %s)""",
                (handle,),
            )
            cur.execute("DELETE FROM watched_accounts WHERE handle = %s", (handle,))
        conn.commit()
    finally:
        conn.close()


def _fake_entry(video_id="123", views=1000, likes=100, comments=10, shares=5):
    return {
        "id": video_id,
        "webpage_url": f"https://www.tiktok.com/@test/video/{video_id}",
        "description": "test caption",
        "duration": 15,
        "view_count": views,
        "like_count": likes,
        "comment_count": comments,
        "repost_count": shares,
        "track": "original sound",
        "artist": "tester",
        "upload_date": "20260101",
    }


def test_parse_video_entry_maps_fields_correctly():
    entry = _fake_entry()
    parsed = parse_video_entry(entry)
    assert parsed["platform_video_id"] == "123"
    assert parsed["url"] == "https://www.tiktok.com/@test/video/123"
    assert parsed["views"] == 1000
    assert parsed["likes"] == 100
    assert parsed["comments"] == 10
    assert parsed["shares"] == 5
    assert parsed["duration_seconds"] == 15


def test_parse_video_entry_handles_missing_fields_gracefully():
    minimal = {"id": "456", "webpage_url": "https://x.com/456"}
    parsed = parse_video_entry(minimal)
    assert parsed["platform_video_id"] == "456"
    assert parsed["views"] is None
    assert parsed["likes"] is None


@patch("qualume.ingest.tiktok_reader.fetch_account_videos")
def test_ingest_videos_for_account_inserts_new_videos(mock_fetch):
    handle = f"test_ingest_{uuid.uuid4().hex[:8]}"
    _cleanup_account(handle)
    try:
        account_id = add_watched_account("tiktok", handle)
        mock_fetch.return_value = [_fake_entry("111"), _fake_entry("222")]

        result = ingest_videos_for_account(account_id, handle)

        assert result["new_videos"] == 2
        assert result["skipped_duplicates"] == 0

        conn = get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT count(*) FROM external_content_items WHERE watched_account_id = %s",
                    (account_id,),
                )
                count = cur.fetchone()[0]
        finally:
            conn.close()
        assert count == 2
    finally:
        _cleanup_account(handle)


@patch("qualume.ingest.tiktok_reader.fetch_account_videos")
def test_ingest_videos_for_account_dedupes_on_second_run(mock_fetch):
    """Regression test for the core requirement: running the poller twice
    must not create duplicate rows or duplicate performance snapshots for
    videos already seen."""
    handle = f"test_dedup_{uuid.uuid4().hex[:8]}"
    _cleanup_account(handle)
    try:
        account_id = add_watched_account("tiktok", handle)
        mock_fetch.return_value = [_fake_entry("333")]

        first_run = ingest_videos_for_account(account_id, handle)
        assert first_run["new_videos"] == 1

        # Second poll sees the same video (TikTok account hasn't posted
        # anything new) — must be recognized as already-known, not re-inserted.
        second_run = ingest_videos_for_account(account_id, handle)
        assert second_run["new_videos"] == 0
        assert second_run["skipped_duplicates"] == 1

        conn = get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT count(*) FROM external_content_items WHERE watched_account_id = %s",
                    (account_id,),
                )
                count = cur.fetchone()[0]
        finally:
            conn.close()
        assert count == 1, "Duplicate video was inserted on re-poll"
    finally:
        _cleanup_account(handle)


@patch("qualume.ingest.tiktok_reader.fetch_account_videos")
def test_ingest_videos_logs_a_performance_snapshot_per_new_video(mock_fetch):
    handle = f"test_snapshot_{uuid.uuid4().hex[:8]}"
    _cleanup_account(handle)
    try:
        account_id = add_watched_account("tiktok", handle)
        mock_fetch.return_value = [_fake_entry("444", views=5000)]

        ingest_videos_for_account(account_id, handle)

        conn = get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """SELECT eps.views FROM external_performance_snapshots eps
                       JOIN external_content_items eci ON eci.id = eps.external_content_item_id
                       WHERE eci.watched_account_id = %s""",
                    (account_id,),
                )
                row = cur.fetchone()
        finally:
            conn.close()
        assert row == (5000,)
    finally:
        _cleanup_account(handle)
