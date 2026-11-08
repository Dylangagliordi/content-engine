"""KMeans performance tiering (forced k=2-3), niche-scoped — ported from
okn_mirror/ml_engine_okn.py's _content_clustering (lines 335-367), per
explicit user decision: "add in isolation forest and k means and
gradient boost and mlp regressor, all wired into one system."

A distinct job from HDBSCAN clustering (learn.transcript_clustering /
learn.visual_clustering): HDBSCAN finds open-ended structural/semantic
groups (variable count, handles noise, no forced size). This forces a
FIXED, ORDINAL ranking (top/avg/low performance tiers) for a
dashboard-style summary — bucketing into ranked bands is a genuinely
different job from open-ended structural discovery, not a duplicate.

Niche-scoped (mandatory) — same peer-relative reasoning as anomaly
detection/mining/HDBSCAN clustering: a "top performer" only means
anything relative to a comparable population.

Reads from the same JOIN LATERAL "latest real snapshot per item" source
as learn.anomaly_detection (NOT okn_mirror/loader.py — that loader is
intentionally okn_mirror-only). Uses learn.baseline.performance_vs_baseline
per item to get the real account-relative ratio_to_median, same target
every other predictive/mining piece in this codebase uses.
"""
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

from qualume.db.connection import get_connection
from qualume.learn.baseline import performance_vs_baseline

_QUERY = """
    SELECT eci.id,
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
    for item_id, views, likes, comments, shares in rows:
        views = views or 0
        likes = likes or 0
        comments = comments or 0
        shares = shares or 0
        engagement_rate = ((likes + comments + shares) / views) if views > 0 else 0.0
        try:
            baseline = performance_vs_baseline(item_id)
            ratio_to_median = baseline["ratio_to_median"]
        except ValueError:
            # No usable baseline for this item (e.g. the only item in its
            # account, so there's no median to compare against) - real,
            # honest gap, skip this item rather than fabricate a ratio.
            continue
        items.append({
            "item_id": item_id,
            "likes": likes,
            "comments": comments,
            "shares": shares,
            "engagement_rate": engagement_rate,
            "ratio_to_median": ratio_to_median,
        })
    return items


def compute_performance_tiers(niche: str, min_sample_size: int = 10) -> dict:
    """KMeans (forced k=2-3) performance tiering for one niche's real
    external content items with a performance snapshot - a distinct job
    from HDBSCAN clustering (learn.transcript_clustering/visual_clustering):
    HDBSCAN finds open-ended structural/semantic groups; this forces a
    fixed, ORDINAL ranking (top/avg/low) for a dashboard-style summary.
    Niche-scoped (mandatory) - same peer-relative reasoning as anomaly
    detection/mining/HDBSCAN clustering.

    Returns {"status": "ok"|"insufficient_data", "n_clusters": int,
    "tiers": [{"label": "[TOP] Top Performers", "size": int,
    "avg_ratio_to_median": float, "item_ids": [...]}, ...]} sorted
    descending by avg_ratio_to_median."""
    items = _fetch_items(niche)

    if len(items) < min_sample_size:
        return {
            "status": "insufficient_data",
            "n_clusters": 0,
            "tiers": [],
        }

    n = len(items)
    k = min(3, max(2, n // 5))

    X = [[it["engagement_rate"], it["likes"], it["comments"], it["shares"]] for it in items]
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)

    clusters = []
    for c in range(k):
        cluster_items = [it for it, lbl in zip(items, labels) if lbl == c]
        if not cluster_items:
            continue
        avg_ratio = sum(it["ratio_to_median"] for it in cluster_items) / len(cluster_items)
        clusters.append({
            "size": len(cluster_items),
            "avg_ratio_to_median": round(avg_ratio, 4),
            "item_ids": [it["item_id"] for it in cluster_items],
        })

    clusters.sort(key=lambda c: c["avg_ratio_to_median"], reverse=True)
    tier_labels = ["[TOP] Top Performers", "[AVG] Average", "[LOW] Needs Improvement"]
    for i, c in enumerate(clusters):
        c["label"] = tier_labels[min(i, len(tier_labels) - 1)]

    return {
        "status": "ok",
        "n_clusters": k,
        "tiers": clusters,
    }
