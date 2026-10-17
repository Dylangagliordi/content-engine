"""Cross-segment replication check — operationalizes the hold-out
replication item from plans/learn-engine-idea-sourcing-architecture.md's
planned insight-validity critic (5b'): "pattern must hold on held-out
data split (time or account)".

mine_categorical_feature (learn.mining) tests a feature's effect POOLED
across every account — a real risk with only 2 accounts in one niche:
a "finding" could just be one account's lucky videos, not a real
pattern. This checks whether the SAME feature value has the SAME
effect direction (outperforms or underperforms its segment's own
baseline) independently within each segment, not just in the pooled
total — the concrete mechanism for "does this hold across accounts/
niches, or is it one account's luck" from the broader "how should we
think about this at scale" discussion.

Not statistically powerful with today's 2 accounts/1 niche (both
segments of 'account' would need real volume each; 'watchlist_type' only
has 2 distinct values total right now) — built correctly now so it's
ready the moment more accounts/niches exist, same "capture the shape
before the volume to use it" discipline as other not-yet-useful-at-n
pieces of this project.
"""
from qualume.learn.baseline import performance_vs_baseline

SEGMENT_BY_OPTIONS = frozenset({"account", "watchlist_type"})


def check_segment_replication(
    column: str, value: str, segment_by: str = "account", min_group_size: int = 5
) -> dict:
    """Does `column`=`value` have the SAME effect direction (videos WITH
    it outperforming their OWN segment's median ratio vs. videos WITHOUT
    it, within that segment alone) independently in each qualifying
    segment — not just pooled across all of them?

    segment_by: 'account' (each watched_accounts row is its own segment)
    or 'watchlist_type' (accounts sharing a type are pooled into one
    segment — competitor, cross_niche, aspirational, etc.)

    Returns: {n_segments_tested, n_segments_agreeing, segment_ids,
    replicates}. A segment only counts as "tested" if BOTH its with/
    without groups clear min_group_size — thin segments are skipped, not
    guessed. replicates is True if every tested segment agrees on
    direction, False if any disagree, None if zero segments qualified
    (genuinely untested, not a negative finding).

    Raises ValueError for an unknown segment_by or an unmineable column —
    reuses learn.mining's allow-list since this builds the same column
    name into SQL.
    """
    from qualume.learn.mining import MINEABLE_COLUMNS

    if segment_by not in SEGMENT_BY_OPTIONS:
        raise ValueError(f"segment_by must be one of {sorted(SEGMENT_BY_OPTIONS)}, got {segment_by!r}")
    if column not in MINEABLE_COLUMNS:
        raise ValueError(f"{column!r} is not a mineable column. Allowed: {sorted(MINEABLE_COLUMNS)}")

    from qualume.db.connection import get_connection

    segment_column = "wa.id" if segment_by == "account" else "wa.watchlist_type"
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"""SELECT ecd.external_content_item_id, ecd.{column}, {segment_column}
                    FROM external_creative_dna ecd
                    JOIN external_performance_snapshots eps
                      ON eps.external_content_item_id = ecd.external_content_item_id
                    JOIN external_content_items eci
                      ON eci.id = ecd.external_content_item_id
                    JOIN watched_accounts wa
                      ON wa.id = eci.watched_account_id
                    WHERE ecd.{column} IS NOT NULL
                      AND ecd.id = (
                          SELECT id FROM external_creative_dna ecd2
                          WHERE ecd2.external_content_item_id = ecd.external_content_item_id
                          ORDER BY analyzed_at DESC LIMIT 1
                      )"""
            )
            rows = cur.fetchall()
    finally:
        conn.close()

    # Group item_ids by segment, computing each video's ratio once.
    by_segment: dict = {}
    ratios: dict[str, float] = {}
    for item_id, col_value, segment_id in rows:
        if item_id not in ratios:
            ratios[item_id] = performance_vs_baseline(item_id)["ratio_to_median"]
        by_segment.setdefault(segment_id, []).append((item_id, col_value))

    n_tested = 0
    n_agreeing = 0
    tested_segment_ids: set = set()
    first_direction: bool | None = None

    for segment_id, item_values in by_segment.items():
        with_ratios = [ratios[i] for i, v in item_values if v == value]
        without_ratios = [ratios[i] for i, v in item_values if v != value]
        if len(with_ratios) < min_group_size or len(without_ratios) < min_group_size:
            continue

        n_tested += 1
        tested_segment_ids.add(segment_id)
        direction_outperforms = (
            sorted(with_ratios)[len(with_ratios) // 2]
            > sorted(without_ratios)[len(without_ratios) // 2]
        )
        if first_direction is None:
            first_direction = direction_outperforms
            n_agreeing += 1
        elif direction_outperforms == first_direction:
            n_agreeing += 1

    return {
        "n_segments_tested": n_tested,
        "n_segments_agreeing": n_agreeing,
        "segment_ids": tested_segment_ids,
        "replicates": None if n_tested == 0 else (n_agreeing == n_tested),
    }
