"""Automated mining-pass orchestrator — the piece that turns Part 5b's
already-built, already-tested statistical primitives into ONE function a
scheduled n8n cycle can call, same shape as orchestrate.run_ingestion_cycle
wrapping poll/download/annotate/classify into one callable.

Runs learn.mining.mine_categorical_feature over every real DNA column and
learn.copy_mining.mine_boolean_copy_feature over every real copy-technique
flag, keeps only results clearing SIGNIFICANCE_THRESHOLD, and upserts those
into insights_external via learn.validation.upsert_external_insight — keyed
on a stable `identity` dict so re-running this on every scheduled cycle
refreshes existing rows instead of piling up duplicate 'proposed' insights
for an unchanged finding. Whatever stops clearing the threshold on a given
pass gets retire_stale_external_insights'd (proposed-tier only — never
touches accepted/testing rows, which have real owned-evidence backing
beyond just this pass).

Per-column/per-flag isolation (one failing mine_* call doesn't abort the
whole pass) — same discipline as orchestrate.run_ingestion_cycle's
per-item isolation within a stage.
"""
from qualume.learn.copy_mining import MINEABLE_COPY_FLAGS, mine_boolean_copy_feature
from qualume.learn.mining import MINEABLE_COLUMNS, mine_categorical_feature
from qualume.learn.validation import retire_stale_external_insights, upsert_external_insight

SIGNIFICANCE_THRESHOLD = 0.05


def _dna_identity(column: str, value) -> dict:
    return {"mined_from": f"external_creative_dna.{column}", "value": value}


def _copy_identity(flag: str, value=True) -> dict:
    return {"mined_from": f"external_copy_analysis.{flag}", "value": value}


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


def run_mining_pass(applies_to_workflow_type: str, min_group_size: int = 5) -> dict:
    """Run the full Part 5b statistical pass over every mineable field,
    upsert significant findings into insights_external, retire whatever
    stopped being significant since the last pass.

    Returns: {tested, proposed_or_refreshed, skipped_not_significant,
    retired, errors} — errors is a list of "column_or_flag: ExceptionType:
    message" strings, one per isolated failure, same shape as
    orchestrate._run_stage_over_items's error reporting."""
    tested = 0
    proposed_or_refreshed = 0
    skipped_not_significant = 0
    errors: list[str] = []
    still_significant_identities: list[dict] = []

    for column in sorted(MINEABLE_COLUMNS):
        try:
            results = mine_categorical_feature(column, min_group_size=min_group_size) or []
        except Exception as e:
            errors.append(f"{column}: {type(e).__name__}: {e}")
            continue

        for result in results:
            tested += 1
            if result["p_value"] >= SIGNIFICANCE_THRESHOLD:
                skipped_not_significant += 1
                continue
            identity = _dna_identity(column, result["value"])
            still_significant_identities.append(identity)
            upsert_external_insight(
                statement=_dna_statement(column, result),
                effect_size=min(result["median_ratio_with"], result["median_ratio_without"])
                / max(result["median_ratio_with"], result["median_ratio_without"]),
                sample_size=result["n_with"] + result["n_without"],
                p_value=result["p_value"],
                applies_to_workflow_type=applies_to_workflow_type,
                identity=identity,
                supporting_data_refs={
                    "n_with": result["n_with"], "n_without": result["n_without"],
                    "median_ratio_with": result["median_ratio_with"],
                    "median_ratio_without": result["median_ratio_without"],
                },
            )
            proposed_or_refreshed += 1

    for flag in sorted(MINEABLE_COPY_FLAGS):
        try:
            result = mine_boolean_copy_feature(flag, min_group_size=min_group_size)
        except Exception as e:
            errors.append(f"{flag}: {type(e).__name__}: {e}")
            continue
        if result is None:
            continue  # group(s) below min_group_size — genuinely untested, not a failure

        tested += 1
        if result["p_value"] >= SIGNIFICANCE_THRESHOLD:
            skipped_not_significant += 1
            continue
        identity = _copy_identity(flag)
        still_significant_identities.append(identity)
        upsert_external_insight(
            statement=_copy_statement(flag, result),
            effect_size=min(result["median_ratio_true"], result["median_ratio_false"])
            / max(result["median_ratio_true"], result["median_ratio_false"]),
            sample_size=result["n_true"] + result["n_false"],
            p_value=result["p_value"],
            applies_to_workflow_type=applies_to_workflow_type,
            identity=identity,
            supporting_data_refs={
                "n_true": result["n_true"], "n_false": result["n_false"],
                "median_ratio_true": result["median_ratio_true"],
                "median_ratio_false": result["median_ratio_false"],
            },
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
