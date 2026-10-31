"""Visual (frame-embedding) clustering — the visual-structure
counterpart to learn.transcript_clustering, same real design per the
user's explicit decision this session (2026-10-31): kept as a SEPARATE
clustering pass, not fused with transcript clustering, so each side's
real cluster quality can be measured independently before any fusion
decision is made.

Identical statistical discipline to transcript_clustering (HDBSCAN,
same MIN_SAMPLE_SIZE/MIN_CLUSTER_SIZE_DEFAULT floors, same Mann-Whitney U
test against account-baseline performance ratio) — only the embedding
SOURCE differs (visual_embeddings instead of transcript_embeddings).
"""
from scipy import stats
from sklearn.cluster import HDBSCAN

from qualume.learn.baseline import performance_vs_baseline

MIN_SAMPLE_SIZE = 40  # same floor as transcript_clustering, same reasoning

MIN_CLUSTER_SIZE_DEFAULT = 5  # same floor as transcript_clustering/mining.py


def _fetch_embeddings() -> list[tuple[str, list[float]]]:
    """All (external_content_item_id, embedding) pairs currently stored
    in visual_embeddings. Isolated into its own function so tests can
    patch it directly rather than needing real embeddings in a test DB."""
    from qualume.db.connection import get_connection

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT external_content_item_id, embedding FROM visual_embeddings")
            return [(str(item_id), embedding) for item_id, embedding in cur.fetchall()]
    finally:
        conn.close()


def cluster_visual(min_cluster_size: int = MIN_CLUSTER_SIZE_DEFAULT) -> dict:
    """Run HDBSCAN over every stored visual embedding, test each real
    cluster's performance-vs-account-baseline ratio against everything
    OUTSIDE that cluster — identical statistical shape to
    transcript_clustering.cluster_transcripts, see that module's
    docstring for the full reasoning (HDBSCAN choice, Mann-Whitney U,
    account-baseline judgment).

    Returns: {n_total, n_scored, n_noise, clusters: [...]}. Each cluster
    dict: {cluster_id, size, item_ids, median_ratio_in_cluster,
    median_ratio_outside, p_value}.

    Raises ValueError below MIN_SAMPLE_SIZE real embeddings.
    """
    rows = _fetch_embeddings()
    if len(rows) < MIN_SAMPLE_SIZE:
        raise ValueError(
            f"cluster_visual needs at least {MIN_SAMPLE_SIZE} real embeddings, "
            f"got {len(rows)} — clustering on fewer would mostly produce noise "
            "dressed up as discovery, not real structure"
        )

    item_ids: list[str] = []
    vectors: list[list[float]] = []
    ratios: dict[str, float] = {}
    for item_id, embedding in rows:
        try:
            ratios[item_id] = performance_vs_baseline(item_id)["ratio_to_median"]
        except ValueError:
            continue
        item_ids.append(item_id)
        vectors.append(embedding)

    n_scored = len(item_ids)

    labels = HDBSCAN(min_cluster_size=min_cluster_size).fit_predict(vectors)

    n_noise = int((labels == -1).sum())
    real_cluster_ids = sorted(set(labels.tolist()) - {-1})

    clusters = []
    for cluster_id in real_cluster_ids:
        in_cluster_ids = [item_ids[i] for i, label in enumerate(labels) if label == cluster_id]
        outside_ids = [item_ids[i] for i, label in enumerate(labels) if label != cluster_id]

        in_ratios = [ratios[i] for i in in_cluster_ids]
        outside_ratios = [ratios[i] for i in outside_ids]

        p_value = None
        if len(in_ratios) >= min_cluster_size and len(outside_ratios) >= min_cluster_size:
            _, p_value = stats.mannwhitneyu(in_ratios, outside_ratios, alternative="two-sided")
            p_value = float(p_value)

        clusters.append({
            "cluster_id": int(cluster_id),
            "size": len(in_cluster_ids),
            "item_ids": in_cluster_ids,
            "median_ratio_in_cluster": sorted(in_ratios)[len(in_ratios) // 2],
            "median_ratio_outside": sorted(outside_ratios)[len(outside_ratios) // 2] if outside_ratios else None,
            "p_value": p_value,
        })

    return {
        "n_total": len(rows),
        "n_scored": n_scored,
        "n_noise": n_noise,
        "clusters": clusters,
    }
