"""TikTok reading engine — fetch, parse, and dedup-ingest videos from
watched accounts into the external_content_items lane.

Uses yt-dlp for public metadata extraction — no login required, verified
working live against real TikTok profile/video URLs on 2026-10-10 (see
docs/session-context.md). Deliberately kept separate from owned content
(v2 plan §2.1: external lane is lower-trust, used for pattern/angle
discovery only, never merged into the owned-performance learning signal).
"""

import json
from datetime import datetime, timezone

import yt_dlp

from qualume.db.connection import get_connection

DEFAULT_MAX_VIDEOS_PER_ACCOUNT = 30


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
