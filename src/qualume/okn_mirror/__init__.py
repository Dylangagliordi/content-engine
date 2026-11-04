"""okn_mirror — a faithful, near line-for-line port of CyberSystema/okn-analytics
(https://github.com/CyberSystema/okn-analytics) adapted to qualume's own DB schema.

WHY THIS EXISTS: not a production pipeline. Built on explicit user request to
run okn-analytics' exact methodology (same 8 analysis modules, same 14 ML models,
same sklearn model choices/hyperparameters) against qualume's REAL external TikTok
data, so the two pipelines' outputs can be compared side-by-side on identical data.
Qualume's own pipeline (learn/mining.py, learn/transcript_clustering.py,
learn/prediction.py) is untouched by this module — this is a parallel, standalone
mirror, not a replacement.

KEY ADAPTATION (the only structural change from upstream): okn's code loops
"per-platform" (Instagram vs TikTok vs YouTube vs Facebook) because that's OKN's
real axis of variation — one org, several platforms. Qualume only has one platform
(TikTok) but MANY niches (48 and counting) — so every "per-platform" loop in the
original code becomes "per-niche" here: the loaded DataFrame's "platform" column
is literally populated with the niche string, not "tiktok". This preserves 100%
of the original logic/structure while mapping it onto qualume's real axis of
cross-group comparison. Every other field name, threshold, and model call matches
upstream as closely as qualume's schema allows (documented per-field in loader.py
where a field is unavailable and defaults to 0, same fillna(0) discipline okn's
own code already uses for missing CSV columns).

KEY FINDING that motivated building this: okn's models operate on caption text +
engagement metrics + timestamps only — fields present on EVERY external_content_items
row with a performance snapshot, REGARDLESS of Gemini annotation status. This means
okn-mirror can run against qualume's full ~16,279-item dataset today, not just the
~1,583 items that have cleared the annotation backlog that currently gates
learn.mining/learn.prediction.
"""
