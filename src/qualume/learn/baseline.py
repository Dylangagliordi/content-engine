"""Account-level performance baseline — per plans/learn-engine-idea-
sourcing-architecture.md ("Own-content performance needs a baseline").

A raw view/like/save count can't be judged in absolute terms; it has to
be judged against what's normal for THAT specific account (a video from a
5M-follower account and a video from a 2K-follower account need different
baselines, or pattern mining ends up measuring "which accounts are big",
not what makes content work).

Deliberately NOT a persisted column/table — computed fresh as a windowed
aggregate query at mining/validation time, same append-only-data
discipline already in place for performance_snapshots (this project never
UPDATEs a snapshot or caches a derived value that could silently go
stale as new snapshots arrive).

Currently uses ALL of an account's snapshots as the window (no time/count
limit) since real accounts only have a handful of videos so far — the
plan explicitly leaves "exact baseline window (trailing N posts? trailing
N days?)" as an open question pending real publishing-cadence data, not
decided here.
"""


def account_median_views(watched_account_id: str) -> int | float | None:
    """Median view count across all of ONE account's videos that have at
    least one performance snapshot. None if the account has no snapshots
    yet (can't baseline against nothing)."""
    from qualume.db.connection import get_connection

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """SELECT percentile_cont(0.5) WITHIN GROUP (ORDER BY eps.views)
                   FROM external_performance_snapshots eps
                   JOIN external_content_items eci ON eci.id = eps.external_content_item_id
                   WHERE eci.watched_account_id = %s AND eps.views IS NOT NULL""",
                (watched_account_id,),
            )
            (median,) = cur.fetchone()
        return median
    finally:
        conn.close()


def performance_vs_baseline(external_content_item_id: str) -> dict:
    """One video's views measured against its OWN account's median —
    the account-relative signal mining actually needs, not a raw count.

    Returns: {views, account_median_views, ratio_to_median, above_median}.
    Raises ValueError if the item has no performance snapshot (nothing to
    compare) or belongs to an account with no median to compare against.
    """
    from qualume.db.connection import get_connection

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """SELECT eci.watched_account_id, eps.views
                   FROM external_content_items eci
                   JOIN external_performance_snapshots eps
                     ON eps.external_content_item_id = eci.id
                   WHERE eci.id = %s
                   ORDER BY eps.ts DESC
                   LIMIT 1""",
                (external_content_item_id,),
            )
            row = cur.fetchone()
    finally:
        conn.close()

    if row is None:
        raise ValueError(
            f"external_content_items row {external_content_item_id} has no "
            "performance snapshot — nothing to compare against a baseline"
        )
    watched_account_id, views = row

    median = account_median_views(watched_account_id)
    if median is None:
        raise ValueError(
            f"account {watched_account_id} has no performance data at all — "
            "cannot compute a baseline"
        )

    views = views or 0
    ratio = (views / median) if median > 0 else float("inf")
    return {
        "views": views,
        "account_median_views": float(median),
        "ratio_to_median": ratio,
        "above_median": views > median,
    }
