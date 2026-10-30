"""Transcript clustering — the genuinely unbiased complement to the
declared-taxonomy mining already built (learn.mining.mine_categorical_
feature, learn.copy_mining.mine_boolean_copy_feature), per this session's
explicit design discussion.

No human decides upfront what's worth checking for. HDBSCAN groups
transcript embeddings (learn.transcript_embedding) by structural
similarity in meaning on its own — videos that persuade/structure their
copy similarly cluster together even if they're about completely
different topics (the "transcends topic" signal the original design doc
specifically called out as valuable: a hook structure that works in both
fitness and finance content says something real about hooks, not about
fitness). HDBSCAN specifically (not k-means) because it doesn't require
picking a cluster count upfront AND produces an honest "noise" label
(-1) for points that don't structurally fit anywhere — forcing every
video into SOME cluster would fabricate structure that isn't there.

Only after a cluster shows a real performance difference vs. everything
else (same account-baseline judgment and Mann-Whitney U test as
learn.mining — never raw view counts) does this module report it as a
real finding. A human still interprets WHAT a significant cluster's
members have in common afterward (that step isn't automatable) — but
which groupings exist in the first place is never hand-picked.

MIN_SAMPLE_SIZE floor is the same real discipline as learn.prediction.
train_predictor's 50-row floor — this project's own design doc flagged
clustering as unreliable at today's real scale (95 videos), and refusing
to run below a floor is the honest behavior, not silently producing
clusters that look like discovery but are mostly noise.
"""
from scipy import stats
from sklearn.cluster import HDBSCAN

from qualume.learn.baseline import performance_vs_baseline

MIN_SAMPLE_SIZE = 40  # ~half of prediction.py's 50-row floor — clustering
# needs less volume than training a supervised model to produce a
# non-degenerate result, but still needs real volume; set low enough to
# be usable at today's real 95-video scale while still refusing to run
# on a handful of videos where every "cluster" would just be noise.

MIN_CLUSTER_SIZE_DEFAULT = 5  # mirrors mining.py's/copy_mining.py's own
# min_group_size=5 default for the SAME reason: a "cluster" of 2-3 videos
# isn't a real enough group to judge performance difference against.


def _fetch_embeddings() -> list[tuple[str, list[float]]]:
    """All (external_content_item_id, embedding) pairs currently stored.
    Real DB query, isolated into its own function so tests can patch it
    directly rather than needing real embeddings in a test DB."""
    from qualume.db.connection import get_connection

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT external_content_item_id, embedding FROM transcript_embeddings")
            return [(str(item_id), embedding) for item_id, embedding in cur.fetchall()]
    finally:
        conn.close()


def cluster_transcripts(min_cluster_size: int = MIN_CLUSTER_SIZE_DEFAULT) -> dict:
    """Run HDBSCAN over every stored transcript embedding, test each real
    cluster's performance-vs-account-baseline ratio against everything
    OUTSIDE that cluster (Mann-Whitney U, same test mine_categorical_
    feature uses — no normality assumption, appropriate for a skewed
    ratio metric with a small real sample).

    Returns: {n_total, n_scored, n_noise, clusters: [...]}. Each cluster
    dict: {cluster_id, size, item_ids, median_ratio_in_cluster,
    median_ratio_outside, p_value} — p_value is None if the cluster (or
    its outside group) has fewer than min_cluster_size members.

    Raises ValueError below MIN_SAMPLE_SIZE real embeddings — same honest
    refuse-rather-than-fabricate behavior as learn.prediction.
    train_predictor's own sample-size floor."""
    rows = _fetch_embeddings()
    if len(rows) < MIN_SAMPLE_SIZE:
        raise ValueError(
            f"cluster_transcripts needs at least {MIN_SAMPLE_SIZE} real embeddings, "
            f"got {len(rows)} — clustering on fewer would mostly produce noise "
            "dressed up as discovery, not real structure"
        )

    # Skip items with no real performance data (an embedding can exist
    # for a video that was never polled for performance, or whose
    # account has no median yet) — never crash the whole pass over one
    # item's missing precondition, same isolation discipline as
    # orchestrate.run_ingestion_cycle's per-item stage handling.
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
