"""Data loader: qualume's real Postgres schema -> okn-analytics' UNIFIED_SCHEMA
shape (a single pandas DataFrame). This is the one genuinely NEW piece of
okn_mirror (everything else is a port of upstream's already-proven logic) — see
okn_mirror/__init__.py for the "niche stands in for platform" rationale.

Field-by-field mapping from okn's UNIFIED_SCHEMA (scripts/config.py) to what
qualume actually has:

  post_id            -> external_content_items.id
  platform            -> watched_accounts.niche (NOT the literal platform column —
                         see module docstring; qualume's cross-group axis is niche)
  published_at        -> posted_at, falling back to discovered_at if NULL (okn
                         assumes every CSV row has a real publish date; ours can
                         be NULL for older items ingested before posted_at parsing
                         existed)
  content_type        -> content_type, defaulting to 'unclassified' when NULL
                         (okn's CONTENT_TYPE_MAP has no equivalent concept — it
                         normalizes PLATFORM-specific labels; qualume's
                         content_type is already a flat organic/selling/NULL
                         value, nothing to re-map)
  title               -> caption (qualume's equivalent of okn's post caption/title)
  permalink           -> url
  duration_sec        -> duration_seconds
  reach                -> views (TikTok has no separate "reach" metric distinct
                         from views the way Instagram/Meta's business API does;
                         okn's own config.py already treats TikTok this way too
                         for its engagement_rate denominator — same choice here)
  views                -> views
  likes/comments/shares -> same-named columns
  saves                -> NOT available in external_performance_snapshots (no
                         saves column on this table) — defaults to 0, never
                         fabricated. Flagged explicitly: any okn model/metric
                         that depends on saves (e.g. quality_weights in
                         analyze_engagement, "saves" in _anomaly_detection
                         features) is running on a real but INCOMPLETE signal
                         here, same honest-gap discipline as everywhere else
                         in this codebase.
  engagement_total       -> likes + comments + shares + saves(=0)
  engagement_rate        -> engagement_total / reach (reach=views here), 0 if
                         reach is 0 — never divides by zero
  watch_time_sec/avg_watch_sec -> NOT available at all (no watch-time tracking
                         in qualume's schema) — 0, flagged, never fabricated
  followers_gained        -> NOT available per-post (qualume tracks account-level
                         follower counts nowhere yet) — 0, flagged
  link_clicks             -> NOT available — 0, flagged

Only items with at least one performance snapshot are included (same as okn:
a post with zero logged metrics has nothing to analyze). Only the LATEST
snapshot per item is used — same latest-snapshot-wins discipline as
learn.baseline.account_median_views and learn.mining's dedup logic, for the
same real reason (a re-polled video must count once, with its current numbers,
not once per poll).
"""
import numpy as np
import pandas as pd

from qualume.db.connection import get_connection

UNAVAILABLE_FIELDS = ("saves", "watch_time_sec", "avg_watch_sec", "followers_gained", "link_clicks")


def load_niche_dataframe(niches: list[str] | None = None, min_niche_size: int = 5) -> pd.DataFrame:
    """Load qualume's external TikTok data into okn's unified DataFrame shape.

    niches: restrict to these niche strings. None = every niche in
    watched_accounts (still subject to min_niche_size).
    min_niche_size: a niche with fewer than this many items (AFTER the
    has-a-snapshot filter) is dropped entirely — mirrors okn's own
    ANALYSIS["min_posts_for_analysis"] gate, applied here at load time so
    every downstream module (analyze_okn.py, ml_engine_okn.py) never has to
    special-case a near-empty niche.

    Returns an empty DataFrame (correct columns, zero rows) if nothing
    qualifies — callers must handle this, same as okn's own main.py checks
    `if df.empty`.
    """
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            query = """
                SELECT eci.id, wa.niche, eci.posted_at, eci.discovered_at,
                       eci.content_type, eci.caption, eci.url, eci.duration_seconds,
                       latest.views, latest.likes, latest.comments, latest.shares
                FROM external_content_items eci
                JOIN watched_accounts wa ON wa.id = eci.watched_account_id
                JOIN LATERAL (
                    SELECT views, likes, comments, shares
                    FROM external_performance_snapshots eps
                    WHERE eps.external_content_item_id = eci.id
                    ORDER BY eps.ts DESC
                    LIMIT 1
                ) latest ON true
                WHERE wa.niche IS NOT NULL
            """
            params: list = []
            if niches is not None:
                query += " AND wa.niche = ANY(%s)"
                params.append(niches)
            cur.execute(query, params)
            rows = cur.fetchall()
            columns = [desc[0] for desc in cur.description]
    finally:
        conn.close()

    if not rows:
        return pd.DataFrame(columns=[
            "post_id", "platform", "published_at", "content_type", "title",
            "permalink", "duration_sec", "reach", "views", "likes", "comments",
            "shares", "saves", "engagement_total", "engagement_rate",
            "watch_time_sec", "avg_watch_sec", "followers_gained", "link_clicks",
        ])

    df = pd.DataFrame(rows, columns=columns)

    out = pd.DataFrame()
    out["post_id"] = df["id"].astype(str)
    out["platform"] = df["niche"]  # niche stands in for okn's "platform" axis
    out["published_at"] = df["posted_at"].fillna(df["discovered_at"])
    out["content_type"] = df["content_type"].fillna("unclassified")
    out["title"] = df["caption"].fillna("")
    out["permalink"] = df["url"].fillna("")
    out["duration_sec"] = pd.to_numeric(df["duration_seconds"], errors="coerce").fillna(0).astype(int)

    views = pd.to_numeric(df["views"], errors="coerce").fillna(0)
    likes = pd.to_numeric(df["likes"], errors="coerce").fillna(0)
    comments = pd.to_numeric(df["comments"], errors="coerce").fillna(0)
    shares = pd.to_numeric(df["shares"], errors="coerce").fillna(0)
    saves = pd.Series(0, index=df.index)  # not tracked, see UNAVAILABLE_FIELDS

    out["reach"] = views.astype(int)
    out["views"] = views.astype(int)
    out["likes"] = likes.astype(int)
    out["comments"] = comments.astype(int)
    out["shares"] = shares.astype(int)
    out["saves"] = saves.astype(int)
    out["engagement_total"] = (likes + comments + shares + saves).astype(int)
    # Real bug found via scripts/run_okn_mirror.py's full-dataset run: using
    # pd.NA via .replace(0, pd.NA) on an int64 column upcasts the WHOLE
    # column to object dtype the moment ANY row (in ANY niche, before
    # per-niche filtering) has reach=0 — and that object dtype then breaks
    # sklearn/.nlargest() calls downstream in ml_engine_okn.py with
    # "Column 'engagement_rate' has dtype object" once a niche is sliced out
    # of the full frame. np.where keeps this a clean float64 column with the
    # exact same zero-guard behavior, no NA/object coercion anywhere.
    out["engagement_rate"] = pd.Series(
        np.where(out["reach"] > 0, out["engagement_total"] / out["reach"].replace(0, 1), 0.0),
        index=out.index, dtype="float64",
    )
    out["watch_time_sec"] = 0
    out["avg_watch_sec"] = 0.0
    out["followers_gained"] = 0
    out["link_clicks"] = 0

    # Drop niches below min_niche_size AFTER the has-a-snapshot filter
    # (the JOIN LATERAL above already excludes items with zero snapshots).
    counts = out["platform"].value_counts()
    keep_niches = counts[counts >= min_niche_size].index
    out = out[out["platform"].isin(keep_niches)].reset_index(drop=True)

    return out
