"""Array and nested-field mining — the real complement to learn.mining
(flat categorical columns) and learn.copy_mining (a fixed 22-item
taxonomy). Handles two DIFFERENT real shapes neither of those cover:

- mine_array_value: a JSONB ARRAY column (music_mood_v2 — e.g.
  ["upbeat", "calm"]) — tests "value X present anywhere in the array vs
  absent", same group-comparison style as learn.copy_mining's boolean
  flags, but applied to a column whose real distinct values are NOT a
  fixed known-upfront taxonomy.
- mine_nested_field_value: a JSONB ARRAY-OF-OBJECTS column with a nested
  string attribute to extract (presenters[].relatable_archetype — e.g.
  [{"relatable_archetype": "mom", ...}]) — same test, one more level of
  nesting.

Neither requires a hardcoded list of real values (unlike MINEABLE_
COLUMNS/MINEABLE_COPY_FLAGS, which DO enumerate real known values/flags)
— discover_array_values/discover_nested_field_values pull whatever
distinct values actually exist in the real data, so a brand-new
archetype or music mood Gemini has never produced before becomes
automatically minable the moment it appears, no code change required.
The REAL SQL-injection guard here is the column/field ALLOW-LIST
(MINEABLE_ARRAY_COLUMNS/MINEABLE_NESTED_FIELDS) — the specific VALUE
being tested for (e.g. "upbeat") is always passed as a bound parameter,
never built into SQL text, so it never needs its own allow-list.

Judges each video against its OWN account's median (learn.baseline), not
raw view counts — same mechanism as every other mining module. Same
latest-row-per-video dedup discipline as learn.mining/learn.copy_mining
(a video re-annotated at a newer prompt version must not be counted
twice with potentially different array contents from each version).
"""
from scipy import stats

from qualume.learn.baseline import performance_vs_baseline

MINEABLE_ARRAY_COLUMNS = frozenset({"music_mood_v2", "script_structure_tags"})

MINEABLE_NESTED_FIELDS = frozenset({
    ("presenters", "relatable_archetype"),
    ("presenters", "apparent_age_range"),
    ("presenters", "apparent_gender_presentation"),
    ("presenters", "delivery_affect"),
})


def _latest_rows_query(select_expr: str) -> str:
    return f"""SELECT ecd.external_content_item_id, {select_expr}
               FROM external_creative_dna ecd
               JOIN external_performance_snapshots eps
                 ON eps.external_content_item_id = ecd.external_content_item_id
               WHERE ecd.id = (
                   SELECT id FROM external_creative_dna ecd2
                   WHERE ecd2.external_content_item_id = ecd.external_content_item_id
                   ORDER BY analyzed_at DESC LIMIT 1
               )
               AND eps.id = (
                   SELECT id FROM external_performance_snapshots eps2
                   WHERE eps2.external_content_item_id = ecd.external_content_item_id
                   ORDER BY eps2.ts DESC LIMIT 1
               )"""


def discover_array_values(column: str, watched_account_id: str | None = None) -> set[str]:
    """Every real distinct value actually present anywhere in `column`
    across all (latest-version) videos — no hardcoded list, the data
    decides what's worth testing."""
    if column not in MINEABLE_ARRAY_COLUMNS:
        raise ValueError(
            f"{column!r} is not a mineable array column. Allowed: {sorted(MINEABLE_ARRAY_COLUMNS)}"
        )
    from qualume.db.connection import get_connection

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            query = f"""SELECT DISTINCT unnest(ecd.{column})
                        FROM external_creative_dna ecd
                        WHERE ecd.{column} IS NOT NULL
                          AND ecd.id = (
                              SELECT id FROM external_creative_dna ecd2
                              WHERE ecd2.external_content_item_id = ecd.external_content_item_id
                              ORDER BY analyzed_at DESC LIMIT 1
                          )"""
            params: list = []
            if watched_account_id is not None:
                query += """ AND ecd.external_content_item_id IN (
                    SELECT id FROM external_content_items WHERE watched_account_id = %s
                )"""
                params.append(watched_account_id)
            cur.execute(query, params)
            return {row[0] for row in cur.fetchall()}
    finally:
        conn.close()


def discover_nested_field_values(
    column: str, field: str, watched_account_id: str | None = None
) -> set[str]:
    """Every real distinct value of `field` seen across all presenters
    (or similar array-of-objects) entries — same no-hardcoded-list
    discipline as discover_array_values."""
    if (column, field) not in MINEABLE_NESTED_FIELDS:
        raise ValueError(
            f"({column!r}, {field!r}) is not a mineable nested field. "
            f"Allowed: {sorted(MINEABLE_NESTED_FIELDS)}"
        )
    from qualume.db.connection import get_connection

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            query = f"""SELECT DISTINCT entry ->> %s
                        FROM external_creative_dna ecd,
                             jsonb_array_elements(ecd.{column}) AS entry
                        WHERE ecd.{column} IS NOT NULL
                          AND ecd.id = (
                              SELECT id FROM external_creative_dna ecd2
                              WHERE ecd2.external_content_item_id = ecd.external_content_item_id
                              ORDER BY analyzed_at DESC LIMIT 1
                          )"""
            params: list = [field]
            if watched_account_id is not None:
                query += """ AND ecd.external_content_item_id IN (
                    SELECT id FROM external_content_items WHERE watched_account_id = %s
                )"""
                params.append(watched_account_id)
            cur.execute(query, params)
            return {row[0] for row in cur.fetchall() if row[0] is not None}
    finally:
        conn.close()


def mine_array_value(
    column: str, value: str, min_group_size: int = 5, watched_account_id: str | None = None
) -> dict | None:
    """Test whether videos with `value` present ANYWHERE in `column`'s
    array perform differently than videos without it (NULL array or
    array not containing the value both count as "without" — the value
    genuinely isn't present either way).

    Returns None if either group falls below min_group_size, else:
    {value, n_with, n_without, median_ratio_with, median_ratio_without, p_value}.

    Raises ValueError for any column not in MINEABLE_ARRAY_COLUMNS."""
    if column not in MINEABLE_ARRAY_COLUMNS:
        raise ValueError(
            f"{column!r} is not a mineable array column. Allowed: {sorted(MINEABLE_ARRAY_COLUMNS)}"
        )
    from qualume.db.connection import get_connection

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            query = _latest_rows_query(f"ecd.{column}")
            params: list = []
            if watched_account_id is not None:
                query += " AND ecd.external_content_item_id IN (SELECT id FROM external_content_items WHERE watched_account_id = %s)"
                params.append(watched_account_id)
            cur.execute(query, params)
            rows = cur.fetchall()
    finally:
        conn.close()

    if not rows:
        return None

    with_ratios, without_ratios = [], []
    for item_id, array_val in rows:
        try:
            ratio = performance_vs_baseline(item_id)["ratio_to_median"]
        except ValueError:
            continue
        if array_val is not None and value in array_val:
            with_ratios.append(ratio)
        else:
            without_ratios.append(ratio)

    if len(with_ratios) < min_group_size or len(without_ratios) < min_group_size:
        return None

    _, p_value = stats.mannwhitneyu(with_ratios, without_ratios, alternative="two-sided")
    return {
        "value": value,
        "n_with": len(with_ratios), "n_without": len(without_ratios),
        "median_ratio_with": sorted(with_ratios)[len(with_ratios) // 2],
        "median_ratio_without": sorted(without_ratios)[len(without_ratios) // 2],
        "p_value": float(p_value),
    }


def mine_nested_field_value(
    column: str, field: str, value: str, min_group_size: int = 5,
    watched_account_id: str | None = None,
) -> dict | None:
    """Test whether videos with `field`=`value` present on ANY entry in
    `column` (e.g. ANY presenter with relatable_archetype='mom') perform
    differently than videos without it. An empty array (no presenters at
    all — product-only/voiceover-over-broll shots) or an array where no
    entry has this value both count as "without".

    Returns None if either group falls below min_group_size, else:
    {value, n_with, n_without, median_ratio_with, median_ratio_without, p_value}.

    Raises ValueError for any (column, field) not in MINEABLE_NESTED_FIELDS."""
    if (column, field) not in MINEABLE_NESTED_FIELDS:
        raise ValueError(
            f"({column!r}, {field!r}) is not a mineable nested field. "
            f"Allowed: {sorted(MINEABLE_NESTED_FIELDS)}"
        )
    from qualume.db.connection import get_connection

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            query = _latest_rows_query(f"ecd.{column}")
            params: list = []
            if watched_account_id is not None:
                query += " AND ecd.external_content_item_id IN (SELECT id FROM external_content_items WHERE watched_account_id = %s)"
                params.append(watched_account_id)
            cur.execute(query, params)
            rows = cur.fetchall()
    finally:
        conn.close()

    if not rows:
        return None

    with_ratios, without_ratios = [], []
    for item_id, entries in rows:
        try:
            ratio = performance_vs_baseline(item_id)["ratio_to_median"]
        except ValueError:
            continue
        entries = entries or []
        has_value = any(e.get(field) == value for e in entries if isinstance(e, dict))
        if has_value:
            with_ratios.append(ratio)
        else:
            without_ratios.append(ratio)

    if len(with_ratios) < min_group_size or len(without_ratios) < min_group_size:
        return None

    _, p_value = stats.mannwhitneyu(with_ratios, without_ratios, alternative="two-sided")
    return {
        "value": value,
        "n_with": len(with_ratios), "n_without": len(without_ratios),
        "median_ratio_with": sorted(with_ratios)[len(with_ratios) // 2],
        "median_ratio_without": sorted(without_ratios)[len(without_ratios) // 2],
        "p_value": float(p_value),
    }
