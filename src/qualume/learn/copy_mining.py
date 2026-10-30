"""Copy-technique mining — tests whether any of the 22 real
external_copy_analysis boolean flags correlate with performance-vs-
account-median (learn.baseline.performance_vs_baseline).

Parallel to learn.mining.mine_categorical_feature, deliberately NOT a
reuse of it: that function assumes a text categorical column on
external_creative_dna and tests "value X vs everything else" across N
distinct values; a copy-technique flag is a BOOLEAN column on a
DIFFERENT table (external_copy_analysis) where the only real test is
"flag=True vs flag=False" - two groups, not N. Same Mann-Whitney U
choice as mine_categorical_feature (no normality assumption, appropriate
for a skewed performance-ratio metric on a small real sample).
"""
from scipy import stats

from qualume.learn.baseline import performance_vs_baseline
from qualume.models.contracts import copy_technique_names

# The real 22-flag taxonomy from external_copy_analysis (see
# db/migrations/011_external_copy_analysis.sql and
# annotate/copy_classifier.py), read from the SAME single source of
# truth the classifier itself uses (models.contracts.copy_technique_names)
# - not a second hardcoded list that could drift out of sync. Still a
# real frozenset (not a bare list) for the same reason mining.
# MINEABLE_COLUMNS is one: this function builds the column name directly
# into SQL, so this allow-list IS the SQL-injection guard, not
# string-escaping a caller-supplied identifier.
MINEABLE_COPY_FLAGS = frozenset(copy_technique_names())


def mine_boolean_copy_feature(
    flag: str, min_group_size: int = 5, watched_account_id: str | None = None
) -> dict | None:
    """Test whether videos with `flag`=True perform differently
    (performance-vs-account-median ratio) than videos with `flag`=False.

    NULL flag values (classifier couldn't tell) are excluded from BOTH
    groups, not silently folded into False - a NULL is "unknown", not
    "no". One row per video: the latest analysis_prompt_version per
    external_content_item_id (same multi-version dedup discipline as
    mine_categorical_feature - this table will accumulate new classifier
    versions over time the same way external_creative_dna did v1->v5).

    watched_account_id optionally scopes to one account - real mining
    runs want this left None (cross-account sample size is the point),
    but tests need it to avoid colliding with whatever real external
    data already exists in the dev DB (same no-separate-test-DB caveat
    as every other test in this project - see CLAUDE.md).

    Returns None if either group falls below min_group_size (not enough
    signal either way), else:
    {flag, n_true, n_false, median_ratio_true, median_ratio_false, p_value}

    Raises ValueError for any flag not in MINEABLE_COPY_FLAGS.
    """
    if flag not in MINEABLE_COPY_FLAGS:
        raise ValueError(
            f"{flag!r} is not a mineable copy flag. Allowed: {sorted(MINEABLE_COPY_FLAGS)}"
        )

    from qualume.db.connection import get_connection

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            query = f"""SELECT eca.external_content_item_id, eca.{flag}
                    FROM external_copy_analysis eca
                    JOIN external_performance_snapshots eps
                      ON eps.external_content_item_id = eca.external_content_item_id
                    WHERE eca.{flag} IS NOT NULL
                      -- latest analysis_prompt_version per video only -
                      -- same real bug class mine_categorical_feature's
                      -- own dedup fixes (a video classified at two
                      -- prompt versions must not be double-counted).
                      AND eca.id = (
                          SELECT id FROM external_copy_analysis eca2
                          WHERE eca2.external_content_item_id = eca.external_content_item_id
                          ORDER BY analyzed_at DESC LIMIT 1
                      )
                      -- latest performance_snapshot per video only - same
                      -- real bug class, found 2026-10-30 against the
                      -- actual 95-video production dataset: without this
                      -- a re-polled video (now the real shape scheduled
                      -- polling produces) is pseudo-replicated once per
                      -- snapshot row instead of counted once.
                      AND eps.id = (
                          SELECT id FROM external_performance_snapshots eps2
                          WHERE eps2.external_content_item_id = eca.external_content_item_id
                          ORDER BY eps2.ts DESC LIMIT 1
                      )"""
            params: list = []
            if watched_account_id is not None:
                query += """ AND eca.external_content_item_id IN (
                    SELECT id FROM external_content_items WHERE watched_account_id = %s
                )"""
                params.append(watched_account_id)
            cur.execute(query, params)
            rows = cur.fetchall()
    finally:
        conn.close()

    if not rows:
        return None

    ratios: dict[str, float] = {}
    item_values: list[tuple[str, bool]] = []
    for item_id, value in rows:
        item_values.append((item_id, value))
        if item_id not in ratios:
            ratios[item_id] = performance_vs_baseline(item_id)["ratio_to_median"]

    true_ratios = [ratios[i] for i, v in item_values if v is True]
    false_ratios = [ratios[i] for i, v in item_values if v is False]

    if len(true_ratios) < min_group_size or len(false_ratios) < min_group_size:
        return None

    _, p_value = stats.mannwhitneyu(true_ratios, false_ratios, alternative="two-sided")
    return {
        "flag": flag,
        "n_true": len(true_ratios),
        "n_false": len(false_ratios),
        "median_ratio_true": sorted(true_ratios)[len(true_ratios) // 2],
        "median_ratio_false": sorted(false_ratios)[len(false_ratios) // 2],
        "p_value": float(p_value),
    }
