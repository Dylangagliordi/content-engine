"""Pattern mining — the first real statistical pass over annotated video
data, per plans/learn-engine-idea-sourcing-architecture.md ("Statistical
method for mining").

Simple per-feature tests FIRST, not regression or random forest:
Mann-Whitney U for a continuous outcome (here: performance-vs-account-
median ratio) split by a categorical DNA field value (e.g. does
hook_type='question' outperform everything else). Regression/random
forest are deliberate LATER upgrades per the plan, not built here.

Judges each video against its OWN account's median (learn.baseline), not
raw view counts — the whole point being that "did well" means "did well
FOR THAT ACCOUNT", not "has a big number".

Only a small, explicit allow-list of real external_creative_dna columns
can be mined this way (categorical/low-cardinality text fields) — this is
also the SQL-injection guard mentioned in the plan's eventual ad hoc
Analyst sub-agent design (not built yet), applied here first since this
function already accepts a caller-supplied column name.
"""
from scipy import stats

from qualume.learn.baseline import performance_vs_baseline

# Real external_creative_dna columns safe to GROUP BY for mining — plain
# scalar categorical fields only. Deliberately NOT exhaustive (e.g.
# shot_breakdown/full_transcript are JSONB, not categorical; energy_level
# is ordinal/numeric, better suited to a different test than this one).
MINEABLE_COLUMNS = frozenset({
    "hook_type", "format_tag", "cta_type", "caption_style",
    "tone_category", "voice_delivery_style", "pacing_qualitative",
})


def mine_categorical_feature(column: str, min_group_size: int = 5) -> list[dict]:
    """For every distinct real value of `column` seen in
    external_creative_dna, test whether videos WITH that value have a
    different performance-vs-account-median ratio than videos WITHOUT it
    (Mann-Whitney U — makes no normality assumption, appropriate for a
    skewed ratio metric with a small real sample).

    Returns one dict per tested value (groups below min_group_size on
    either side are skipped, not padded or guessed):
    {value, n_with, n_without, median_ratio_with, median_ratio_without,
     p_value}

    Raises ValueError for any column not in MINEABLE_COLUMNS — this
    function builds the column name directly into SQL, so an allow-list
    is the real guard, not string-escaping a caller-supplied identifier.
    """
    if column not in MINEABLE_COLUMNS:
        raise ValueError(
            f"{column!r} is not a mineable column. Allowed: {sorted(MINEABLE_COLUMNS)}"
        )

    from qualume.db.connection import get_connection

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"""SELECT ecd.external_content_item_id, ecd.{column}
                    FROM external_creative_dna ecd
                    JOIN external_performance_snapshots eps
                      ON eps.external_content_item_id = ecd.external_content_item_id
                    WHERE ecd.{column} IS NOT NULL
                      -- one row per video: the latest annotation attempt
                      -- (by analyzed_at) for that external_content_item_id.
                      -- Without this, a video annotated at both v1 and v5
                      -- (or any two versions) is counted TWICE with
                      -- potentially DIFFERENT field values from each
                      -- version — real bug found testing this against the
                      -- actual 58 v1+v5-annotated videos (118 rows showed
                      -- up instead of 58, silently inflating sample sizes
                      -- and corrupting every p-value).
                      AND ecd.id = (
                          SELECT id FROM external_creative_dna ecd2
                          WHERE ecd2.external_content_item_id = ecd.external_content_item_id
                          ORDER BY analyzed_at DESC LIMIT 1
                      )"""
            )
            rows = cur.fetchall()
    finally:
        conn.close()

    if not rows:
        return []

    # One performance_vs_baseline call per distinct item, not per row —
    # several rows can share an item_id if there were ever duplicate
    # snapshots, so dedupe the ratio lookups.
    ratios: dict[str, float] = {}
    item_values: list[tuple[str, str]] = []
    for item_id, value in rows:
        item_values.append((item_id, value))
        if item_id not in ratios:
            ratios[item_id] = performance_vs_baseline(item_id)["ratio_to_median"]

    distinct_values = {value for _, value in item_values}
    results = []
    for value in sorted(distinct_values):
        with_ratios = [ratios[i] for i, v in item_values if v == value]
        without_ratios = [ratios[i] for i, v in item_values if v != value]
        if len(with_ratios) < min_group_size or len(without_ratios) < min_group_size:
            continue

        _, p_value = stats.mannwhitneyu(with_ratios, without_ratios, alternative="two-sided")
        results.append({
            "value": value,
            "n_with": len(with_ratios),
            "n_without": len(without_ratios),
            "median_ratio_with": sorted(with_ratios)[len(with_ratios) // 2],
            "median_ratio_without": sorted(without_ratios)[len(without_ratios) // 2],
            "p_value": float(p_value),
        })
    return results
