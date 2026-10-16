import os
import uuid
from unittest.mock import patch, MagicMock

from qualume.ingest.tiktok_reader import (
    parse_video_entry,
    ingest_videos_for_account,
    download_video,
    download_pending_videos,
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


# --- Raw video downloading ---------------------------------------------------
# yt_dlp is always mocked here: real downloads would be slow, flaky, and hit
# TikTok from the test suite. The fake YoutubeDL writes a tiny placeholder
# file at the path yt-dlp would have produced, so return values are real paths.


def _fake_ydl_class(captured_opts: list, ext: str = "mp4", fail_for: set = frozenset()):
    """Build a stand-in for yt_dlp.YoutubeDL that records its options and,
    on extract_info(download=True), writes {outtmpl with id/ext filled in}."""

    class FakeYDL:
        def __init__(self, opts):
            self.opts = opts
            captured_opts.append(opts)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def extract_info(self, url, download=False):
            if url in fail_for:
                raise RuntimeError(f"simulated download failure for {url}")
            video_id = url.rstrip("/").split("/")[-1]
            info = {"id": video_id, "ext": ext, "webpage_url": url}
            path = self.prepare_filename(info)
            with open(path, "wb") as f:
                f.write(b"fake video bytes")
            info["requested_downloads"] = [{"filepath": path}]
            return info

        def prepare_filename(self, info):
            return self.opts["outtmpl"] % {"id": info["id"], "ext": info["ext"]}

    return FakeYDL


def test_download_video_uses_download_options_and_returns_written_path(tmp_path):
    dest = tmp_path / "raw_videos"  # deliberately not created yet
    captured = []
    with patch("qualume.ingest.tiktok_reader.yt_dlp.YoutubeDL", _fake_ydl_class(captured)):
        path = download_video("https://www.tiktok.com/@test/video/789", "789", dest_dir=str(dest))

    assert len(captured) == 1
    opts = captured[0]
    assert opts["skip_download"] is False
    assert opts["outtmpl"] == os.path.join(str(dest), "789.%(ext)s")
    assert path == os.path.join(str(dest), "789.mp4")
    assert os.path.isfile(path)


def test_download_video_is_idempotent_when_file_already_exists(tmp_path):
    dest = tmp_path / "raw_videos"
    dest.mkdir()
    existing = dest / "789.mp4"
    existing.write_bytes(b"already downloaded")

    ydl_cls = MagicMock()
    with patch("qualume.ingest.tiktok_reader.yt_dlp.YoutubeDL", ydl_cls):
        path = download_video("https://www.tiktok.com/@test/video/789", "789", dest_dir=str(dest))

    ydl_cls.assert_not_called()  # no network hit
    assert path == str(existing)
    assert existing.read_bytes() == b"already downloaded"


def test_download_video_ignores_partial_download_leftovers(tmp_path):
    """A .part file from an interrupted download is not a finished video."""
    dest = tmp_path / "raw_videos"
    dest.mkdir()
    (dest / "789.mp4.part").write_bytes(b"half")
    captured = []
    with patch("qualume.ingest.tiktok_reader.yt_dlp.YoutubeDL", _fake_ydl_class(captured)):
        path = download_video("https://www.tiktok.com/@test/video/789", "789", dest_dir=str(dest))
    assert len(captured) == 1
    assert path == os.path.join(str(dest), "789.mp4")


def _local_video_paths(account_id: str) -> dict:
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """SELECT platform_video_id, local_video_path FROM external_content_items
                   WHERE watched_account_id = %s""",
                (account_id,),
            )
            return dict(cur.fetchall())
    finally:
        conn.close()


@patch("qualume.ingest.tiktok_reader.fetch_account_videos")
def test_download_pending_videos_downloads_updates_db_and_survives_one_failure(mock_fetch, tmp_path):
    handle = f"test_dl_{uuid.uuid4().hex[:8]}"
    _cleanup_account(handle)
    try:
        account_id = add_watched_account("tiktok", handle)
        mock_fetch.return_value = [_fake_entry("901"), _fake_entry("902"), _fake_entry("903")]
        ingest_videos_for_account(account_id, handle)

        dest = tmp_path / "raw_videos"
        dest.mkdir()
        # 903 is already on disk (e.g. a prior run downloaded it but crashed
        # before the DB update) — should be recorded without re-downloading.
        (dest / "903.mp4").write_bytes(b"from a previous run")

        captured = []
        bad_url = "https://www.tiktok.com/@test/video/902"
        fake = _fake_ydl_class(captured, fail_for={bad_url})
        with patch("qualume.ingest.tiktok_reader.yt_dlp.YoutubeDL", fake):
            # Scoped to this test's account so the real dev rows are never touched.
            result = download_pending_videos(dest_dir=str(dest), watched_account_id=account_id)

        assert result == {"downloaded": 1, "failed": 1, "already_had_path": 1}
        paths = _local_video_paths(account_id)
        assert paths["901"] == os.path.join(str(dest), "901.mp4")
        assert paths["902"] is None  # failure leaves it pending for a retry
        assert paths["903"] == os.path.join(str(dest), "903.mp4")

        # Second run: only the previously failed video is still pending.
        captured.clear()
        with patch("qualume.ingest.tiktok_reader.yt_dlp.YoutubeDL", _fake_ydl_class(captured)):
            second = download_pending_videos(dest_dir=str(dest), watched_account_id=account_id)
        assert second == {"downloaded": 1, "failed": 0, "already_had_path": 0}
        assert _local_video_paths(account_id)["902"] == os.path.join(str(dest), "902.mp4")
    finally:
        _cleanup_account(handle)


@patch("qualume.ingest.tiktok_reader.fetch_account_videos")
def test_download_pending_videos_respects_limit(mock_fetch, tmp_path):
    handle = f"test_dl_limit_{uuid.uuid4().hex[:8]}"
    _cleanup_account(handle)
    try:
        account_id = add_watched_account("tiktok", handle)
        mock_fetch.return_value = [_fake_entry("911"), _fake_entry("912"), _fake_entry("913")]
        ingest_videos_for_account(account_id, handle)

        captured = []
        with patch("qualume.ingest.tiktok_reader.yt_dlp.YoutubeDL", _fake_ydl_class(captured)):
            result = download_pending_videos(
                limit=2, dest_dir=str(tmp_path), watched_account_id=account_id
            )

        assert result == {"downloaded": 2, "failed": 0, "already_had_path": 0}
        assert len(captured) == 2
        assert sum(p is not None for p in _local_video_paths(account_id).values()) == 2
    finally:
        _cleanup_account(handle)
