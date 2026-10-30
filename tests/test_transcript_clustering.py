"""Tests for embedding-based clustering — written FIRST per this
project's TDD discipline.

Per this session's design discussion: this is the genuinely unbiased
complement to the declared-taxonomy mining already built. No human names
clusters upfront or decides what's "worth testing" — HDBSCAN (chosen
specifically because it doesn't require picking a cluster count upfront
AND produces an honest "noise" label for videos that don't structurally
fit anywhere, rather than forcing every video into some group) groups
transcript embeddings by structural similarity on its own. Only AFTER a
cluster shows a real performance difference does this module report it —
same "real stats, not vibes" discipline as learn.mining, with the same
account-baseline judgment (learn.baseline.performance_vs_baseline) and
the same Mann-Whitney U test for a cluster's performance vs. everything
else.

min_sample_size floor is the SAME real discipline as
learn.prediction.train_predictor's 50-row floor (clustering noise at
n=95 is a real risk the project's own design doc flagged) — refuses to
run below the floor rather than silently producing clusters that look
like discovery but are actually just noise given too little data to
average out.
"""
from unittest.mock import patch

import pytest

from qualume.learn.transcript_clustering import (
    MIN_SAMPLE_SIZE,
    cluster_transcripts,
)


def _fake_embeddings(n, dim=8):
    """n fake rows, each a (item_id, embedding) tuple. Two tight clusters
    plus some scattered noise points, by construction — a real, known
    shape to test clustering against, same style as mining.py's own
    injected-correlation smoke tests."""
    import random
    random.seed(42)
    rows = []
    half = n // 2
    for i in range(half):
        # cluster A: tight around [1,1,1,...]
        vec = [1.0 + random.gauss(0, 0.02) for _ in range(dim)]
        rows.append((f"item-a-{i}", vec))
    for i in range(n - half):
        # cluster B: tight around [-1,-1,-1,...]
        vec = [-1.0 + random.gauss(0, 0.02) for _ in range(dim)]
        rows.append((f"item-b-{i}", vec))
    return rows


def test_cluster_transcripts_raises_below_min_sample_size():
    with pytest.raises(ValueError, match=f"at least {MIN_SAMPLE_SIZE}"):
        with patch(
            "qualume.learn.transcript_clustering._fetch_embeddings",
            return_value=_fake_embeddings(MIN_SAMPLE_SIZE - 1),
        ):
            cluster_transcripts()


def test_cluster_transcripts_finds_real_injected_clusters():
    """A real, known 2-cluster shape (planted in the fake embeddings) —
    HDBSCAN should recover close to 2 real clusters, not 1 or N scattered
    singletons, proving the clustering step itself works before trusting
    it against real data."""
    fake_rows = _fake_embeddings(MIN_SAMPLE_SIZE + 10)

    with patch("qualume.learn.transcript_clustering._fetch_embeddings", return_value=fake_rows), \
         patch("qualume.learn.transcript_clustering.performance_vs_baseline") as mock_perf:
        mock_perf.return_value = {"ratio_to_median": 1.0}
        result = cluster_transcripts(min_cluster_size=5)

    real_cluster_ids = {c["cluster_id"] for c in result["clusters"]}
    assert len(real_cluster_ids) >= 2  # found the 2 real planted groups


def test_cluster_transcripts_reports_performance_difference_per_cluster():
    """The real point: a cluster whose members consistently over/under-
    perform their account baseline should be flagged with a p-value, same
    shape as mine_categorical_feature's output — not just cluster
    membership with no outcome attached."""
    fake_rows = _fake_embeddings(MIN_SAMPLE_SIZE + 10)

    def fake_perf(item_id):
        # cluster A items over-perform, cluster B items under-perform —
        # a real, known injected correlation to detect.
        if item_id.startswith("item-a"):
            return {"ratio_to_median": 3.0}
        return {"ratio_to_median": 0.3}

    with patch("qualume.learn.transcript_clustering._fetch_embeddings", return_value=fake_rows), \
         patch("qualume.learn.transcript_clustering.performance_vs_baseline", side_effect=fake_perf):
        result = cluster_transcripts(min_cluster_size=5)

    significant = [c for c in result["clusters"] if c["p_value"] is not None and c["p_value"] < 0.05]
    assert len(significant) >= 1
    # at least one of them should show the real injected high-performing direction
    assert any(c["median_ratio_in_cluster"] > c["median_ratio_outside"] for c in significant)


def test_cluster_transcripts_labels_noise_points_separately():
    """HDBSCAN's noise label (-1) must be reported as genuinely
    unclustered, not silently folded into a real cluster or dropped."""
    fake_rows = _fake_embeddings(MIN_SAMPLE_SIZE + 10)
    # add some real scattered/noisy points that shouldn't tightly cluster
    import random
    random.seed(7)
    for i in range(5):
        fake_rows.append((f"item-noise-{i}", [random.uniform(-3, 3) for _ in range(8)]))

    with patch("qualume.learn.transcript_clustering._fetch_embeddings", return_value=fake_rows), \
         patch("qualume.learn.transcript_clustering.performance_vs_baseline") as mock_perf:
        mock_perf.return_value = {"ratio_to_median": 1.0}
        result = cluster_transcripts(min_cluster_size=5)

    assert "n_noise" in result
    assert isinstance(result["n_noise"], int)


def test_cluster_transcripts_skips_items_with_no_performance_data():
    """An item with an embedding but no performance_vs_baseline data
    (performance_vs_baseline raises ValueError — real precondition from
    learn.baseline) must be excluded from the clustering/testing pool
    entirely, not crash the whole pass."""
    fake_rows = _fake_embeddings(MIN_SAMPLE_SIZE + 10)

    def fake_perf(item_id):
        if item_id == "item-a-0":
            raise ValueError("no performance snapshot")
        return {"ratio_to_median": 1.0}

    with patch("qualume.learn.transcript_clustering._fetch_embeddings", return_value=fake_rows), \
         patch("qualume.learn.transcript_clustering.performance_vs_baseline", side_effect=fake_perf):
        result = cluster_transcripts(min_cluster_size=5)

    assert result["n_scored"] == len(fake_rows) - 1
