"""TikTok reading engine — fetch, parse, and dedup-ingest videos from
watched accounts into the external_content_items lane.

Uses yt-dlp for public metadata extraction — no login required, verified
working live against real TikTok profile/video URLs on 2026-10-10 (see
docs/session-context.md). Deliberately kept separate from owned content
(v2 plan §2.1: external lane is lower-trust, used for pattern/angle
discovery only, never merged into the owned-performance learning signal).
"""

import glob
import json
import logging
import os
from datetime import datetime, timezone

import yt_dlp

from qualume.db.connection import get_connection

DEFAULT_MAX_VIDEOS_PER_ACCOUNT = 30
DEFAULT_RAW_VIDEO_DIR = "data/raw_videos"

# yt-dlp's in-progress / bookkeeping suffixes — a file with one of these is
# not a finished video and must not satisfy the idempotency check.
_INCOMPLETE_SUFFIXES = (".part", ".ytdl", ".temp")

logger = logging.getLogger(__name__)


def fetch_account_videos(handle: str, max_videos: int = DEFAULT_MAX_VIDEOS_PER_ACCOUNT) -> list[dict]:
    """Pull recent video entries from a public TikTok account's profile feed.

    Returns a list of yt-dlp entry dicts (flat extraction — fast, metadata
    only, no video download). Network call — not used directly in unit
    tests, which mock this function instead.
    """
    url = f"https://www.tiktok.com/@{handle}"
    ydl_opts = {
        "quiet": True,
        "skip_download": True,
        "no_warnings": True,
        "extract_flat": False,  # need full metadata (view/like counts), not just IDs
        "playlistend": max_videos,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
    return info.get("entries", []) if info.get("_type") == "playlist" else [info]


def parse_video_entry(entry: dict) -> dict:
    """Normalize a yt-dlp entry into our external_content_items schema.
    Missing fields map to None rather than raising — a partially-available
    entry is still worth storing, per the Read engine's graceful-degradation
    testing strategy in the v2 plan (don't crash the whole batch on one
    incomplete record)."""
    upload_date = entry.get("upload_date")
    posted_at = None
    if upload_date:
        try:
            posted_at = datetime.strptime(upload_date, "%Y%m%d").replace(tzinfo=timezone.utc)
        except ValueError:
            posted_at = None

    return {
        "platform_video_id": str(entry.get("id")),
        "url": entry.get("webpage_url"),
        "caption": entry.get("description"),
        "duration_seconds": entry.get("duration"),
        "audio_track": entry.get("track"),
        "audio_artist": entry.get("artist"),
        "posted_at": posted_at,
        "views": entry.get("view_count"),
        "likes": entry.get("like_count"),
        "comments": entry.get("comment_count"),
        "shares": entry.get("repost_count"),
        "raw_metadata": entry,
    }


def ingest_videos_for_account(watched_account_id: str, handle: str,
                               max_videos: int = DEFAULT_MAX_VIDEOS_PER_ACCOUNT) -> dict:
    """Poll one watched account, dedup against existing rows, insert new
    videos + their first performance snapshot. Idempotent: running this
    repeatedly against an unchanged feed inserts nothing new (v2 plan §1.4)."""
    entries = fetch_account_videos(handle, max_videos=max_videos)
    new_videos = 0
    skipped_duplicates = 0

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            for entry in entries:
                parsed = parse_video_entry(entry)
                video_id = parsed["platform_video_id"]

                cur.execute(
                    """SELECT id FROM external_content_items
                       WHERE platform = 'tiktok' AND platform_video_id = %s""",
                    (video_id,),
                )
                existing = cur.fetchone()
                if existing:
                    skipped_duplicates += 1
                    continue

                cur.execute(
                    """INSERT INTO external_content_items
                       (watched_account_id, platform, platform_video_id, url, caption,
                        duration_seconds, audio_track, audio_artist, posted_at, raw_metadata)
                       VALUES (%s, 'tiktok', %s, %s, %s, %s, %s, %s, %s, %s)
                       RETURNING id""",
                    (
                        watched_account_id,
                        video_id,
                        parsed["url"],
                        parsed["caption"],
                        parsed["duration_seconds"],
                        parsed["audio_track"],
                        parsed["audio_artist"],
                        parsed["posted_at"],
                        json.dumps(parsed["raw_metadata"], default=str),
                    ),
                )
                external_content_item_id = cur.fetchone()[0]

                cur.execute(
                    """INSERT INTO external_performance_snapshots
                       (external_content_item_id, views, likes, comments, shares)
                       VALUES (%s, %s, %s, %s, %s)""",
                    (
                        external_content_item_id,
                        parsed["views"],
                        parsed["likes"],
                        parsed["comments"],
                        parsed["shares"],
                    ),
                )
                new_videos += 1
        conn.commit()
    finally:
        conn.close()

    return {"new_videos": new_videos, "skipped_duplicates": skipped_duplicates}


def _existing_video_file(dest_dir: str, platform_video_id: str) -> str | None:
    """Return the path of an already-downloaded video for this ID, if any."""
    pattern = os.path.join(glob.escape(dest_dir), f"{glob.escape(platform_video_id)}.*")
    for path in sorted(glob.glob(pattern)):
        if os.path.isfile(path) and not path.endswith(_INCOMPLETE_SUFFIXES):
            return path
    return None


def download_video(url: str, platform_video_id: str, dest_dir: str = DEFAULT_RAW_VIDEO_DIR) -> str:
    """Download one video's raw file to {dest_dir}/{platform_video_id}.<ext>
    and return the local path. Idempotent: if a finished file for this ID is
    already on disk, return it without touching the network."""
    existing = _existing_video_file(dest_dir, platform_video_id)
    if existing:
        return existing

    os.makedirs(dest_dir, exist_ok=True)
    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": False,
        "noplaylist": True,
        "outtmpl": os.path.join(dest_dir, f"{platform_video_id}.%(ext)s"),
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        downloads = info.get("requested_downloads") or []
        path = downloads[0].get("filepath") if downloads else None
        path = path or ydl.prepare_filename(info)

    # Fail loud rather than recording a path that doesn't exist.
    if not os.path.isfile(path):
        raise FileNotFoundError(f"yt-dlp reported {path} for {url} but no file was written")
    return path


def download_pending_videos(limit: int | None = None, dest_dir: str = DEFAULT_RAW_VIDEO_DIR,
                            watched_account_id: str | None = None) -> dict:
    """Download raw files for external items with no local_video_path yet and
    record the path. Per-video failures are logged and counted, never fatal —
    the row stays NULL so the next run retries it. Each successful row is
    committed immediately so an interrupted run keeps its progress.

    already_had_path counts rows whose file was already on disk (e.g. a prior
    run downloaded it but died before the DB update) — recorded, not
    re-downloaded. watched_account_id optionally scopes the run to one account.
    """
    counts = {"downloaded": 0, "failed": 0, "already_had_path": 0}

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            query = """SELECT id, url, platform_video_id FROM external_content_items
                       WHERE local_video_path IS NULL"""
            params: list = []
            if watched_account_id is not None:
                query += " AND watched_account_id = %s"
                params.append(watched_account_id)
            query += " ORDER BY discovered_at, id"
            if limit is not None:
                query += " LIMIT %s"
                params.append(limit)
            cur.execute(query, params)
            pending = cur.fetchall()

            for item_id, url, platform_video_id in pending:
                try:
                    already_on_disk = _existing_video_file(dest_dir, platform_video_id) is not None
                    path = download_video(url, platform_video_id, dest_dir=dest_dir)
                    cur.execute(
                        "UPDATE external_content_items SET local_video_path = %s WHERE id = %s",
                        (path, item_id),
                    )
                    conn.commit()
                except Exception:
                    conn.rollback()
                    logger.exception("Failed to download video %s (%s)", platform_video_id, url)
                    counts["failed"] += 1
                    continue
                counts["already_had_path" if already_on_disk else "downloaded"] += 1
    finally:
        conn.close()

    return counts
