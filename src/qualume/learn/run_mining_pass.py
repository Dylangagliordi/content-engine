"""Automated mining-pass orchestrator — the piece that turns Part 5b's
already-built, already-tested statistical primitives into ONE function a
scheduled n8n cycle can call, same shape as orchestrate.run_ingestion_cycle
wrapping poll/download/annotate/classify into one callable.

Runs learn.mining.mine_categorical_feature over every real DNA column,
learn.copy_mining.mine_boolean_copy_feature over every real copy-technique
flag, and learn.array_mining.mine_array_value/mine_nested_field_value over
every real array/nested-field value (music mood, presenter archetype/age/
gender/delivery-affect — added 2026-10-30, closes the real gap where
visual/presenter data was captured but never tested for correlation with
performance).

SIGNIFICANCE DECISION (Benjamini-Hochberg, added 2026-10-30): this pass
now tests ~70+ features per cycle. At a flat p<0.05 per-test threshold,
that volume alone produces several "significant" findings from pure
chance — the same shape that made the pseudo-replication bug earlier this
session look plausible (22/49 "significant" was implausibly high; BH
correction is the permanent fix for the underlying false-discovery-rate
problem, not just that one bug). Every mine_* call's p-value is collected
FIRST, across the WHOLE pass (every column/flag/array-value/nested-field-
value together, not corrected separately per feature type — one shared
batch, since they're all competing for the same "is this real" budget),
THEN learn.multiple_comparison.benjamini_hochberg decides which ones
survive. Only BH-significant results get upserted into insights_external.

Whatever stops clearing BH-corrected significance on a given pass gets
retire_stale_external_insights'd (proposed-tier only — never touches
accepted/testing rows, which have real owned-evidence backing beyond just
this pass).

Per-column/per-flag/per-value isolation (one failing mine_* call doesn't
abort the whole pass) — same discipline as orchestrate.run_ingestion_
cycle's per-item isolation within a stage.
"""
from qualume.learn.array_mining import (
    MINEABLE_ARRAY_COLUMNS,
    MINEABLE_NESTED_FIELDS,
    discover_array_values,
    discover_nested_field_values,
    mine_array_value,
    mine_nested_field_value,
)
from qualume.learn.copy_mining import MINEABLE_COPY_FLAGS, mine_boolean_copy_feature
from qualume.learn.mining import MINEABLE_COLUMNS, mine_categorical_feature
from qualume.learn.multiple_comparison import benjamini_hochberg
from qualume.learn.validation import retire_stale_external_insights, upsert_external_insight

SIGNIFICANCE_THRESHOLD = 0.05  # the FDR passed to benjamini_hochberg — same
# conventional 5% this project always used, now applied as a false
# discovery rate across the whole batch instead of a flat per-test cutoff.


def _dna_identity(column: str, value) -> dict:
    return {"mined_from": f"external_creative_dna.{column}", "value": value}


def _copy_identity(flag: str, value=True) -> dict:
    return {"mined_from": f"external_copy_analysis.{flag}", "value": value}


def _array_identity(column: str, value: str) -> dict:
    return {"mined_from": f"external_creative_dna.{column}", "value": value}


def _nested_identity(column: str, field: str, value: str) -> dict:
    return {"mined_from": f"external_creative_dna.{column}.{field}", "value": value}


def _dna_statement(column: str, result: dict) -> str:
    direction = "outperforms" if result["median_ratio_with"] > result["median_ratio_without"] else "underperforms"
    return (
        f"Videos where {column}={result['value']!r} {direction} videos without it "
        f"(median {result['median_ratio_with']:.2f}x account baseline vs "
        f"{result['median_ratio_without']:.2f}x, n_with={result['n_with']}, "
        f"n_without={result['n_without']})."
    )


def _copy_statement(flag: str, result: dict) -> str:
    direction = "outperforms" if result["median_ratio_true"] > result["median_ratio_false"] else "underperforms"
    return (
        f"Videos where the copy technique {flag!r} is present {direction} videos "
        f"without it (median {result['median_ratio_true']:.2f}x account baseline vs "
        f"{result['median_ratio_false']:.2f}x, n_true={result['n_true']}, "
        f"n_false={result['n_false']})."
    )


def _array_statement(column: str, result: dict) -> str:
    direction = "outperforms" if result["median_ratio_with"] > result["median_ratio_without"] else "underperforms"
    return (
        f"Videos where {column} includes {result['value']!r} {direction} videos "
        f"without it (median {result['median_ratio_with']:.2f}x account baseline vs "
        f"{result['median_ratio_without']:.2f}x, n_with={result['n_with']}, "
        f"n_without={result['n_without']})."
    )


def _nested_statement(column: str, field: str, result: dict) -> str:
    direction = "outperforms" if result["median_ratio_with"] > result["median_ratio_without"] else "underperforms"
    return (
        f"Videos where {column}.{field}={result['value']!r} is present on any entry "
        f"{direction} videos without it (median {result['median_ratio_with']:.2f}x "
        f"account baseline vs {result['median_ratio_without']:.2f}x, "
        f"n_with={result['n_with']}, n_without={result['n_without']})."
    )


def run_mining_pass(applies_to_workflow_type: str, min_group_size: int = 5) -> dict:
    """Run the full Part 5b statistical pass over every mineable field,
    apply Benjamini-Hochberg correction across the WHOLE batch of
    p-values together (not per-feature-type), upsert BH-significant
    findings into insights_external, retire whatever stopped being
    BH-significant since the last pass.

    Two-pass shape, required by BH (it needs every p-value collected
    BEFORE any individual result can be judged significant — unlike a
    flat threshold, which can decide per-result as it goes): pass 1 runs
    every mine_* call and collects (kind, statement_fn_args..., result)
    tuples with no decision made yet; pass 2 runs BH over all collected
    p-values together, then upserts exactly the ones BH marked True.

    Returns: {tested, proposed_or_refreshed, skipped_not_significant,
    retired, errors} — errors is a list of "column_or_flag: ExceptionType:
    message" strings, one per isolated failure, same shape as
    orchestrate._run_stage_over_items's error reporting."""
    errors: list[str] = []
    # Each entry: (kind, identity, statement, effect_size, sample_size,
    # p_value, supporting_data_refs) — collected across ALL feature types
    # before any BH decision is made.
    candidates: list[dict] = []

    for column in sorted(MINEABLE_COLUMNS):
        try:
            results = mine_categorical_feature(column, min_group_size=min_group_size) or []
        except Exception as e:
            errors.append(f"{column}: {type(e).__name__}: {e}")
            continue
        for result in results:
            candidates.append({
                "identity": _dna_identity(column, result["value"]),
                "statement": _dna_statement(column, result),
                "effect_size": min(result["median_ratio_with"], result["median_ratio_without"])
                / max(result["median_ratio_with"], result["median_ratio_without"]),
                "sample_size": result["n_with"] + result["n_without"],
                "p_value": result["p_value"],
                "supporting_data_refs": {
                    "n_with": result["n_with"], "n_without": result["n_without"],
                    "median_ratio_with": result["median_ratio_with"],
                    "median_ratio_without": result["median_ratio_without"],
                },
            })

    for flag in sorted(MINEABLE_COPY_FLAGS):
        try:
            result = mine_boolean_copy_feature(flag, min_group_size=min_group_size)
        except Exception as e:
            errors.append(f"{flag}: {type(e).__name__}: {e}")
            continue
        if result is None:
            continue  # group(s) below min_group_size — genuinely untested, not a failure
        candidates.append({
            "identity": _copy_identity(flag),
            "statement": _copy_statement(flag, result),
            "effect_size": min(result["median_ratio_true"], result["median_ratio_false"])
            / max(result["median_ratio_true"], result["median_ratio_false"]),
            "sample_size": result["n_true"] + result["n_false"],
            "p_value": result["p_value"],
            "supporting_data_refs": {
                "n_true": result["n_true"], "n_false": result["n_false"],
                "median_ratio_true": result["median_ratio_true"],
                "median_ratio_false": result["median_ratio_false"],
            },
        })

    for column in sorted(MINEABLE_ARRAY_COLUMNS):
        try:
            values = discover_array_values(column)
        except Exception as e:
            errors.append(f"{column}: {type(e).__name__}: {e}")
            continue
        for value in sorted(values):
            try:
                result = mine_array_value(column, value, min_group_size=min_group_size)
            except Exception as e:
                errors.append(f"{column}={value}: {type(e).__name__}: {e}")
                continue
            if result is None:
                continue
            candidates.append({
                "identity": _array_identity(column, value),
                "statement": _array_statement(column, result),
                "effect_size": min(result["median_ratio_with"], result["median_ratio_without"])
                / max(result["median_ratio_with"], result["median_ratio_without"]),
                "sample_size": result["n_with"] + result["n_without"],
                "p_value": result["p_value"],
                "supporting_data_refs": {
                    "n_with": result["n_with"], "n_without": result["n_without"],
                    "median_ratio_with": result["median_ratio_with"],
                    "median_ratio_without": result["median_ratio_without"],
                },
            })

    for column, field in sorted(MINEABLE_NESTED_FIELDS):
        try:
            values = discover_nested_field_values(column, field)
        except Exception as e:
            errors.append(f"{column}.{field}: {type(e).__name__}: {e}")
            continue
        for value in sorted(values):
            try:
                result = mine_nested_field_value(column, field, value, min_group_size=min_group_size)
            except Exception as e:
                errors.append(f"{column}.{field}={value}: {type(e).__name__}: {e}")
                continue
            if result is None:
                continue
            candidates.append({
                "identity": _nested_identity(column, field, value),
                "statement": _nested_statement(column, field, result),
                "effect_size": min(result["median_ratio_with"], result["median_ratio_without"])
                / max(result["median_ratio_with"], result["median_ratio_without"]),
                "sample_size": result["n_with"] + result["n_without"],
                "p_value": result["p_value"],
                "supporting_data_refs": {
                    "n_with": result["n_with"], "n_without": result["n_without"],
                    "median_ratio_with": result["median_ratio_with"],
                    "median_ratio_without": result["median_ratio_without"],
                },
            })

    tested = len(candidates)
    p_values = [c["p_value"] for c in candidates]
    bh_significant = benjamini_hochberg(p_values, fdr=SIGNIFICANCE_THRESHOLD)

    proposed_or_refreshed = 0
    skipped_not_significant = 0
    still_significant_identities: list[dict] = []

    for candidate, is_significant in zip(candidates, bh_significant):
        if not is_significant:
            skipped_not_significant += 1
            continue
        still_significant_identities.append(candidate["identity"])
        upsert_external_insight(
            statement=candidate["statement"],
            effect_size=candidate["effect_size"],
            sample_size=candidate["sample_size"],
            p_value=candidate["p_value"],
            applies_to_workflow_type=applies_to_workflow_type,
            identity=candidate["identity"],
            supporting_data_refs=candidate["supporting_data_refs"],
        )
        proposed_or_refreshed += 1

    retired = retire_stale_external_insights(
        applies_to_workflow_type=applies_to_workflow_type,
        still_significant_identities=still_significant_identities,
    )

    return {
        "tested": tested,
        "proposed_or_refreshed": proposed_or_refreshed,
        "skipped_not_significant": skipped_not_significant,
        "retired": retired,
        "errors": errors,
    }
