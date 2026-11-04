"""Config port of okn-analytics' scripts/config.py — same ANALYSIS thresholds,
same recency-weight logic, same structure. Only real difference: PLATFORMS
(okn: instagram/youtube/facebook/tiktok) becomes NICHE_PLACEHOLDER since
qualume's real cross-group axis is niche, not platform (see okn_mirror/__init__.py
docstring for the full rationale). TIMEZONE/KST stuff dropped — that was
OKN-specific (Korean org), qualume has no equivalent real requirement, so this
mirror runs in UTC throughout rather than inventing a fake timezone need.
"""
from datetime import timedelta

import pandas as pd

# ──────────────────────────────────────────────
# ANALYSIS PARAMETERS — verbatim from okn's config.py ANALYSIS dict
# ──────────────────────────────────────────────
ANALYSIS = {
    "min_posts_for_analysis": 5,
    "viral_multiplier": 2.5,
    "underperform_threshold": 0.3,
    "top_n_posts": 5,
    "growth_alert_positive": 0.10,
    "growth_alert_negative": -0.05,
    # okn benchmarks engagement vs cross-industry per-PLATFORM averages.
    # Qualume has no equivalent per-niche external benchmark data, so this
    # mirror has no engagement_benchmarks block — vs_benchmark comparisons
    # from okn's analyze_platforms() are dropped rather than faked with a
    # made-up number. Flagged explicitly in analyze_okn.py where skipped.
}

# ──────────────────────────────────────────────
# RECENCY WEIGHTS — verbatim thresholds from okn's TIMELINE["recency_weights"]
# ──────────────────────────────────────────────
RECENCY_WEIGHTS = {
    "recent_days": 90,
    "recent_weight": 1.0,
    "medium_days": 180,
    "medium_weight": 0.3,
    "old_weight": 0.1,
}


def compute_recency_weights(dates: pd.Series, reference_date=None) -> pd.Series:
    """Direct port of okn's config.compute_recency_weights — same logic,
    same thresholds, UTC instead of KST (see module docstring)."""
    dates = pd.to_datetime(dates, errors="coerce", utc=True)
    if reference_date is None:
        reference_date = dates.max()
    if pd.isna(reference_date):
        return pd.Series(1.0, index=dates.index)
    if hasattr(reference_date, "tzinfo") and reference_date.tzinfo is None:
        reference_date = reference_date.tz_localize("UTC")

    rw = RECENCY_WEIGHTS
    days_ago = (reference_date - dates).dt.total_seconds() / 86400

    weights = pd.Series(rw["old_weight"], index=dates.index, dtype=float)
    weights[days_ago <= rw["medium_days"]] = rw["medium_weight"]
    weights[days_ago <= rw["recent_days"]] = rw["recent_weight"]
    return weights
