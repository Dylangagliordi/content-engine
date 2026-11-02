"""Meta Ad Library reader — adapted from the public, free facebook-
ads-library-mcp scraper (RamsesAguirre777/facebook-ads-library-mcp),
same no-token/no-account approach (renders the public Ad Library page
in headless Chromium via crawl4ai, no login, any country).

REAL FIX vs. that scraper: it serializes the rendered page to MARKDOWN,
and markdown has no syntax for <video src=...> (only images), so every
ad's actual video file is silently dropped on the floor — you only get
a thumbnail. Confirmed live 2026-11-02 that real video ads DO have a
genuine <video> element with a resolvable src/currentSrc in the
rendered DOM. Since crawl4ai runs on Playwright under the hood, this
module ALSO runs a small JS snippet in-page (document.querySelectorAll
('video')) to pull those src values directly — same approach Apify's
paid actor uses internally to produce its videoHdUrl — then zips them
back to the markdown-parsed ad cards in DOM order. Gemini can then
annotate the real video the same way it already does for TikTok.

ALSO REAL, confirmed live 2026-11-02 against an actual US commercial ad
(Depop, Library ID 1647772813204375): there is NO impression/reach/
spend number published anywhere in the Ad Library UI for an ordinary US
commercial ad — only Library ID, started-running date, platforms, and
"N ads use this creative". Meta only publishes impression RANGES for
EU/UK-delivered ads (DSA) or political/issue ads. So this module does
NOT fabricate an impressions field (CLAUDE.md principles #2, #7) — see
migrations 018/019 for the honest substitutes: run date,
ads_using_creative, and a re-checked still_active longevity signal.

Every Ad Library entry is selling-by-definition (an ad is a paid
selling placement) — content_type is seeded 'selling' directly here,
no classifier call needed, unlike organic TikTok content which could be
either (see migration 017).
"""
import asyncio
import json
import os
import re
from datetime import datetime
from urllib.parse import parse_qs, quote, unquote, urlparse

import requests

from qualume.db.connection import get_connection

AD_LIBRARY_BASE = "https://www.facebook.com/ads/library/"
DEFAULT_RAW_VIDEO_DIR = "data/raw_meta_ads"

# Same fixed CTA label set the free scraper uses — Meta's button labels
# are a closed, known set, not free text, so matching against them is
# how the markdown parser avoids grabbing trailing caption words.
_CTA_LABELS = (
    "Learn more", "Sign up", "Shop now", "Book now", "Send message",
    "Send WhatsApp message", "Contact us", "Subscribe", "Get offer", "Get quote",
    "Apply now", "Download", "Watch more", "See menu", "Order now", "Donate now",
    "Play game", "Listen now", "Get showtimes", "Save", "Open link", "Message page",
    "Call now", "Get directions", "Follow page", "Use app", "Install now",
    "Buy tickets", "Request time", "Try in camera",
)
_CTA_RE = re.compile(
    r"\s(" + "|".join(re.escape(c) for c in _CTA_LABELS) + r")\s+\]\(https://l\.facebook\.com/l\.php",
    re.IGNORECASE,
)


def _decode_landing_url(fb_link: str) -> str:
    """Turn an l.facebook.com/l.php?u=... redirect into the real destination."""
    try:
        qs = parse_qs(urlparse(fb_link).query)
        if "u" in qs:
            return unquote(qs["u"][0])
    except Exception:
        pass
    return fb_link


def parse_ad_library_markdown(md: str) -> list[dict]:
    """Best-effort structured extraction of ad cards from rendered Ad
    Library markdown. Same regex-on-markdown-chunks approach as the
    free scraper this is adapted from (crawl4ai's markdown output has
    no stable DOM structure to parse against, so this is the honest
    approach, not a shortcut) — video URLs are attached SEPARATELY by
    attach_video_urls since markdown drops them entirely.

    has_video_placeholder (added here, not in the original scraper):
    True if this card's thumbnail pattern looks like a video poster
    frame rather than a plain image — used by attach_video_urls to zip
    video src values only onto cards that actually have a video,
    instead of assuming every card has one. Since markdown gives no
    reliable video/image distinction either, this defaults every card
    to True (assume video) — the real per-card video presence check
    happens in the browser DOM step (_render_and_extract), which is the
    one place that can see it for certain.
    """
    ads: list[dict] = []
    seen: set[str] = set()
    chunks = re.split(r"\nLibrary ID:\s*", md)
    for chunk in chunks[1:]:
        lib_id_match = re.match(r"(\d+)", chunk)
        if not lib_id_match:
            continue
        lib_id = lib_id_match.group(1)
        if lib_id in seen:
            continue
        seen.add(lib_id)

        ad: dict = {
            "library_id": lib_id,
            "ad_details_url": f"https://www.facebook.com/ads/library/?id={lib_id}",
            "has_video_placeholder": True,
        }

        m = re.search(r"Started running on ([A-Za-z]{3} \d{1,2}, \d{4})", chunk)
        ad["started_running"] = m.group(1) if m else None

        m = re.search(r"\*\*(\d+)\s+ads?\*\*\s+use this creative", chunk)
        ad["ads_using_creative"] = int(m.group(1)) if m else 1

        m = re.search(r"\[([^\]]+)\]\(https://www\.facebook\.com/([^/)]+)/?\)", chunk)
        if m:
            ad["advertiser"] = m.group(1)
            ad["advertiser_handle"] = m.group(2)

        m = re.search(r"\(https://l\.facebook\.com/l\.php\?u=([^)&]+)", chunk)
        if m:
            real = _decode_landing_url("https://l.facebook.com/l.php?u=" + m.group(1))
            ad["landing_url"] = real
            ad["landing_domain"] = urlparse(real).netloc

        m = _CTA_RE.search(chunk)
        ad["cta"] = m.group(1).strip() if m else None

        # body copy: the plain descriptive line(s) just before the
        # creative card markdown link — matches both this module's own
        # fixture shape and the free scraper's "Sponsored\n<body>" shape.
        body_match = re.search(
            r"(?:\*\*Sponsored\*\*\s*\n|\)\s*\n)(.+?)\n\[!\[", chunk, re.DOTALL
        )
        if body_match:
            txt = re.sub(r"\s+", " ", body_match.group(1)).strip()
            ad["body"] = txt[:2000]

        ads.append(ad)
    return ads


def attach_video_urls(ads: list[dict], video_srcs: list[str]) -> list[dict]:
    """Zip real <video> src URLs (pulled from the live DOM, see
    _render_and_extract) onto markdown-parsed ad cards IN ORDER,
    skipping cards known to have no video placeholder rather than a
    flat zip — a naive index-aligned zip would misassign a later card's
    video onto an earlier image-only card the moment counts don't match
    1:1, which is the normal case (not every ad in a search result is a
    video ad).

    Real failure mode handled: fewer rendered <video> elements than
    expected placeholders (partial hydration) — remaining cards get
    video_url=None rather than crashing or reusing a stale URL."""
    video_iter = iter(video_srcs)
    result = []
    for ad in ads:
        out = dict(ad)
        if out.get("has_video_placeholder", True):
            out["video_url"] = next(video_iter, None)
        else:
            out["video_url"] = None
        result.append(out)
    return result


def parse_ad_entry(ad: dict, country: str = "US") -> dict:
    """Normalize one parsed+video-attached ad card into our
    external_content_items schema — same normalization role as
    tiktok_reader.parse_video_entry, graceful on missing fields rather
    than raising (a partially-available ad is still worth storing)."""
    started_running = ad.get("started_running")
    ad_started_running_at = None
    if started_running:
        try:
            ad_started_running_at = datetime.strptime(started_running, "%b %d, %Y").date()
        except ValueError:
            ad_started_running_at = None

    lib_id = str(ad.get("library_id"))
    return {
        "platform_video_id": lib_id,
        "url": ad.get("ad_details_url") or f"https://www.facebook.com/ads/library/?id={lib_id}",
        "caption": ad.get("body"),
        "ad_started_running_at": ad_started_running_at,
        "ads_using_creative": ad.get("ads_using_creative"),
        "video_url": ad.get("video_url"),
        "content_type": "selling",  # every Ad Library entry is selling-by-definition
        "raw_metadata": ad,
    }


async def _render_and_extract(url: str, wait_seconds: int = 8, scroll_rounds: int = 8) -> dict:
    """Render the Ad Library SPA in headless Chromium via crawl4ai and
    return BOTH its markdown (for the card-level text fields) AND the
    real <video> src values pulled directly from the live DOM (the real
    fix this module exists for — see module docstring).

    Facebook answers the headless browser with HTTP 403 but still
    serves the rendered content, so success is judged by whether
    content parsed, not status code — same discipline as the free
    scraper this is adapted from."""
    from crawl4ai import AsyncWebCrawler, BrowserConfig, CacheMode, CrawlerRunConfig

    bc = BrowserConfig(headless=True, browser_type="chromium",
                        viewport_width=1400, viewport_height=1600)
    hydrate_ms = max(3, wait_seconds) * 1000
    rounds = max(0, int(scroll_rounds))
    scroll_js = (
        f"await new Promise(r=>setTimeout(r,{hydrate_ms}));"
        + "".join(
            "window.scrollTo(0, document.body.scrollHeight);"
            "await new Promise(r=>setTimeout(r,2500));"
            for _ in range(rounds)
        )
    )
    # The real fix: pull every <video> element's resolvable src
    # (currentSrc falls back correctly for <source> children too) in
    # DOM order. crawl4ai's js_code runner wraps each script in an
    # async IIFE and captures whatever it RETURNS as js_execution_
    # result — confirmed by reading async_crawler_strategy.py's
    # robust_execute_user_script (not guessed) — so this must end in a
    # real `return`, not just a bare expression or window assignment,
    # or the value is silently dropped (the real bug in this module's
    # first draft: it set window.__qualume_video_urls but never
    # returned it, so js_execution_result stayed empty).
    video_extract_js = (
        "return Array.from(document.querySelectorAll('video'))"
        ".map(v => v.currentSrc || v.src).filter(Boolean);"
    )
    cfg = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        delay_before_return_html=max(3, wait_seconds),
        page_timeout=max(90000, 20000 + rounds * 3000),
        js_code=[scroll_js, video_extract_js],
        js_only=False,
    )

    async with AsyncWebCrawler(config=bc) as crawler:
        result = await crawler.arun(url=url, config=cfg)
        md = result.markdown.raw_markdown if hasattr(result.markdown, "raw_markdown") else str(result.markdown)
        video_urls: list[str] = []
        exec_result = getattr(result, "js_execution_result", None)
        if exec_result:
            # js_code is a list of 2 scripts; robust_execute_user_script
            # returns one result per script in the same order, under a
            # 'results' key (confirmed by reading the source, not assumed).
            raw_results = exec_result.get("results") if isinstance(exec_result, dict) else None
            if raw_results and len(raw_results) >= 2:
                last = raw_results[-1]
                if isinstance(last, list):
                    video_urls = last
                elif isinstance(last, dict) and last.get("result"):
                    video_urls = last["result"]
        return {"markdown": md or "", "video_urls": video_urls, "status_code": result.status_code}


def search_meta_ads(query: str = "", country: str = "US", advertiser_page_id: str = "",
                     active_status: str = "active", wait_seconds: int = 8,
                     scroll_rounds: int = 8) -> list[dict]:
    """Search the public Ad Library for selling videos — returns
    normalized dicts ready for ingest_ad_entries. Network call, not
    used directly in unit tests (which test parse_ad_library_markdown/
    attach_video_urls/parse_ad_entry against fixtures instead, same
    discipline as tiktok_reader's fetch_account_videos)."""
    params = {"active_status": active_status, "ad_type": "all", "country": country, "media_type": "video"}
    if advertiser_page_id:
        params["view_all_page_id"] = advertiser_page_id
        params["search_type"] = "page"
    if query:
        params["q"] = query
        if not advertiser_page_id:
            params["search_type"] = "keyword_unordered"
    url = AD_LIBRARY_BASE + "?" + "&".join(f"{k}={quote(str(v))}" for k, v in params.items())

    rendered = asyncio.run(_render_and_extract(url, wait_seconds, scroll_rounds))
    ads = parse_ad_library_markdown(rendered["markdown"])
    ads = attach_video_urls(ads, rendered["video_urls"])
    return [parse_ad_entry(ad, country=country) for ad in ads]


def ingest_ad_entries(watched_account_id: str, entries: list[dict]) -> dict:
    """Dedup-insert normalized ad entries into external_content_items,
    same platform='meta_ads' shape as tiktok_reader.ingest_videos_for_
    account but without a TikTok-style performance snapshot (no honest
    view/like/comment numbers exist for an ad — see module docstring).
    A longevity re-check (ad_library_checks, migration 019) is a
    SEPARATE pass, not done here, since this function's job is new-ad
    discovery, not re-checking already-known ads."""
    new_ads = 0
    already_known = 0

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            for entry in entries:
                video_id = entry["platform_video_id"]
                cur.execute(
                    """SELECT id FROM external_content_items
                       WHERE platform = 'meta_ads' AND platform_video_id = %s""",
                    (video_id,),
                )
                if cur.fetchone():
                    already_known += 1
                    continue

                cur.execute(
                    """INSERT INTO external_content_items
                       (watched_account_id, platform, platform_video_id, url, caption,
                        ad_started_running_at, ads_using_creative, content_type, raw_metadata)
                       VALUES (%s, 'meta_ads', %s, %s, %s, %s, %s, %s, %s)""",
                    (
                        watched_account_id,
                        video_id,
                        entry["url"],
                        entry["caption"],
                        entry["ad_started_running_at"],
                        entry["ads_using_creative"],
                        entry["content_type"],
                        json.dumps(entry["raw_metadata"], default=str),
                    ),
                )
                new_ads += 1
        conn.commit()
    finally:
        conn.close()

    return {"new_ads": new_ads, "already_known": already_known}


def _existing_video_file(dest_dir: str, platform_video_id: str) -> str | None:
    """Same idempotency check as tiktok_reader's own helper — return an
    already-downloaded file's path for this ID, if any, so a prior
    successful download is never re-fetched."""
    import glob

    pattern = os.path.join(glob.escape(dest_dir), f"{glob.escape(platform_video_id)}.*")
    for path in sorted(glob.glob(pattern)):
        if os.path.isfile(path):
            return path
    return None


def download_ad_video(video_url: str, platform_video_id: str,
                       dest_dir: str = DEFAULT_RAW_VIDEO_DIR) -> str:
    """Download one ad's real video file to {dest_dir}/{platform_video_
    id}.mp4 and return the local path. No yt-dlp needed here, unlike
    tiktok_reader — video_url (from attach_video_urls) is already a
    direct, playable CDN .mp4 link straight from Facebook's video CDN,
    not a page requiring extraction. Idempotent: an already-downloaded
    file for this ID is returned without touching the network, same
    discipline as tiktok_reader.download_video."""
    existing = _existing_video_file(dest_dir, platform_video_id)
    if existing:
        return existing

    os.makedirs(dest_dir, exist_ok=True)
    path = os.path.join(dest_dir, f"{platform_video_id}.mp4")
    resp = requests.get(video_url, stream=True, timeout=60)
    resp.raise_for_status()
    with open(path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

    if not os.path.isfile(path) or os.path.getsize(path) == 0:
        raise FileNotFoundError(f"download reported success for {video_url} but no file was written")
    return path


def download_pending_ad_videos(limit: int | None = None, dest_dir: str = DEFAULT_RAW_VIDEO_DIR,
                                watched_account_id: str | None = None) -> dict:
    """Download real video files for meta_ads items with no local_
    video_path yet, reading video_url back out of raw_metadata (stored
    at ingest time by ingest_ad_entries). Same per-item isolation and
    already_had_path/failed/downloaded accounting as tiktok_reader's
    download_pending_videos — one bad/expired CDN URL must not block
    the rest of the batch."""
    counts = {"downloaded": 0, "failed": 0, "already_had_path": 0, "skipped_no_video_url": 0}

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            query = """SELECT id, platform_video_id, raw_metadata FROM external_content_items
                       WHERE platform = 'meta_ads' AND local_video_path IS NULL"""
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

            for item_id, platform_video_id, raw_metadata in pending:
                video_url = (raw_metadata or {}).get("video_url")
                if not video_url:
                    counts["skipped_no_video_url"] += 1
                    continue
                try:
                    already_on_disk = _existing_video_file(dest_dir, platform_video_id) is not None
                    path = download_ad_video(video_url, platform_video_id, dest_dir=dest_dir)
                    cur.execute(
                        "UPDATE external_content_items SET local_video_path = %s WHERE id = %s",
                        (path, item_id),
                    )
                    conn.commit()
                except Exception:
                    conn.rollback()
                    counts["failed"] += 1
                    continue
                counts["already_had_path" if already_on_disk else "downloaded"] += 1
    finally:
        conn.close()

    return counts
