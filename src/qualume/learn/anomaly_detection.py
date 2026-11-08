"""IsolationForest anomaly detection, niche-scoped — ported from
okn_mirror/ml_engine_okn.py's _anomaly_detection (lines 369-397), per
explicit user decision: "add in isolation forest and k means and
gradient boost and mlp regressor, all wired into one system."

Genuinely new capability, not a duplicate of anything else in this
codebase: learn.replication checks whether a KNOWN finding holds across
segments (needs a prior hypothesis); this flags real statistical
outliers with NO prior hypothesis needed — unsupervised, multi-
dimensional (reach/engagement_total/likes/comments/shares together, not
one metric at a time).

Niche-scoped (mandatory, not optional) — outlier-ness is inherently
peer-relative: an item is only "anomalous" relative to a comparable
population, same reasoning as learn.mining/learn.transcript_clustering's
own niche requirement. Pooling across niches here would make "this is
an outlier" compare a Quran-recitation video's view count against a
TikTok-Shop-seller's, which has no real meaning.

Reads directly from external_content_items/external_performance_snapshots
(same data source learn.baseline/learn.training_data read from) via the
exact JOIN LATERAL "latest real snapshot per item" pattern already
proven in okn_mirror/loader.py — does NOT import from okn_mirror itself
(that loader is intentionally okn_mirror-only, per that module's own
docstring; this module ports the ALGORITHM, not the DataFrame-shaped I/O).

Feature mapping (same as okn_mirror/loader.py's own documented TikTok
choice): "reach" has no separate metric distinct from views on TikTok,
so reach = views here; engagement_total = likes + comments + shares
(no "saves" column in this schema, never fabricated).
"""
from sklearn.ensemble import IsolationForest

from qualume.db.connection import get_connection

ANOMALY_FEATURES = ("reach", "engagement_total", "likes", "comments", "shares")

_QUERY = """
    SELECT eci.id, eci.url,
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
    WHERE wa.niche = %s
"""


def _fetch_items(niche: str) -> list[dict]:
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(_QUERY, (niche,))
            rows = cur.fetchall()
    finally:
        conn.close()

    items = []
    for item_id, url, views, likes, comments, shares in rows:
        views = views or 0
        likes = likes or 0
        comments = comments or 0
        shares = shares or 0
        items.append({
            "item_id": item_id,
            "url": url,
            "reach": views,
            "likes": likes,
            "comments": comments,
            "shares": shares,
            "engagement_total": likes + comments + shares,
        })
    return items


def detect_anomalies(niche: str, min_sample_size: int = 10) -> dict:
    """IsolationForest anomaly detection for one niche's real external
    content items with a performance snapshot. Flags viral_outlier
    (statistically unusual AND above-median engagement) and
    underperformer_outlier (statistically unusual AND below-median) -
    real, unsupervised, no prior hypothesis needed (unlike
    learn.mining/learn.replication, which test a KNOWN claim).

    Niche-scoped (mandatory, not optional) - outlier-ness is inherently
    peer-relative; an item is only "anomalous" relative to a comparable
    population, same reasoning as mining/clustering's own niche
    requirement.

    Returns {"status": "ok"|"insufficient_data", "n_items": int,
    "viral_outliers": [...], "underperformer_outliers": [...]} - each
    outlier entry: {item_id, platform_video_id or url, reach,
    engagement_total, anomaly_score}. Raises nothing; returns
    insufficient_data status below min_sample_size, same honest-gap
    discipline as learn.prediction.train_predictor's ValueError-on-too-
    few-rows (different signal - this DEGRADES rather than raises,
    matching run_mining_pass/run_cluster_pass's own per-niche degrade-
    not-crash shape, since this runs inside the same orchestrator loop)."""
    items = _fetch_items(niche)

    if len(items) < min_sample_size:
        return {
            "status": "insufficient_data",
            "n_items": len(items),
            "viral_outliers": [],
            "underperformer_outliers": [],
        }

    X = [[it["reach"], it["engagement_total"], it["likes"], it["comments"], it["shares"]] for it in items]

    iso = IsolationForest(contamination=0.15, random_state=42, n_estimators=100)
    anomaly_label = iso.fit_predict(X)
    anomaly_raw_score = iso.decision_function(X)

    engagement_rates = [
        (it["engagement_total"] / it["reach"]) if it["reach"] > 0 else 0.0
        for it in items
    ]
    sorted_rates = sorted(engagement_rates)
    n = len(sorted_rates)
    median_rate = (
        sorted_rates[n // 2] if n % 2 == 1
        else (sorted_rates[n // 2 - 1] + sorted_rates[n // 2]) / 2
    )

    viral_outliers = []
    underperformer_outliers = []
    for it, label, score, rate in zip(items, anomaly_label, anomaly_raw_score, engagement_rates):
        if label != -1:
            continue
        entry = {
            "item_id": it["item_id"],
            "url": it["url"],
            "reach": it["reach"],
            "engagement_total": it["engagement_total"],
            "anomaly_score": round(float(score), 4),
        }
        if rate > median_rate:
            viral_outliers.append(entry)
        else:
            underperformer_outliers.append(entry)

    viral_outliers.sort(key=lambda e: e["anomaly_score"])
    underperformer_outliers.sort(key=lambda e: e["anomaly_score"])

    return {
        "status": "ok",
        "n_items": len(items),
        "viral_outliers": viral_outliers,
        "underperformer_outliers": underperformer_outliers,
    }
