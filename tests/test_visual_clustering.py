"""Tests for visual (frame-embedding) clustering — mirrors test_
transcript_clustering.py's exact shape, pointed at visual_embeddings
instead of transcript_embeddings. Per the user's explicit decision this
session (2026-10-31): visual and transcript clustering run as separate
pipelines so real cluster quality from each can be compared independently
before any fusion decision is made.
"""
from unittest.mock import patch

import pytest

from qualume.learn.visual_clustering import (
    MIN_SAMPLE_SIZE,
    cluster_visual,
)


def _fake_embeddings(n, dim=8):
    import random
    random.seed(42)
    rows = []
    half = n // 2
    for i in range(half):
        vec = [1.0 + random.gauss(0, 0.02) for _ in range(dim)]
        rows.append((f"item-a-{i}", vec))
    for i in range(n - half):
        vec = [-1.0 + random.gauss(0, 0.02) for _ in range(dim)]
        rows.append((f"item-b-{i}", vec))
    return rows


def test_cluster_visual_raises_below_min_sample_size():
    with pytest.raises(ValueError, match=f"at least {MIN_SAMPLE_SIZE}"):
        with patch(
            "qualume.learn.visual_clustering._fetch_embeddings",
            return_value=_fake_embeddings(MIN_SAMPLE_SIZE - 1),
        ):
            cluster_visual()


def test_cluster_visual_finds_real_injected_clusters():
    fake_rows = _fake_embeddings(MIN_SAMPLE_SIZE + 10)

    with patch("qualume.learn.visual_clustering._fetch_embeddings", return_value=fake_rows), \
         patch("qualume.learn.visual_clustering.performance_vs_baseline") as mock_perf:
        mock_perf.return_value = {"ratio_to_median": 1.0}
        result = cluster_visual(min_cluster_size=5)

    real_cluster_ids = {c["cluster_id"] for c in result["clusters"]}
    assert len(real_cluster_ids) >= 2


def test_cluster_visual_reports_performance_difference_per_cluster():
    fake_rows = _fake_embeddings(MIN_SAMPLE_SIZE + 10)

    def fake_perf(item_id):
        if item_id.startswith("item-a"):
            return {"ratio_to_median": 3.0}
        return {"ratio_to_median": 0.3}

    with patch("qualume.learn.visual_clustering._fetch_embeddings", return_value=fake_rows), \
         patch("qualume.learn.visual_clustering.performance_vs_baseline", side_effect=fake_perf):
        result = cluster_visual(min_cluster_size=5)

    significant = [c for c in result["clusters"] if c["p_value"] is not None and c["p_value"] < 0.05]
    assert len(significant) >= 1
    assert any(c["median_ratio_in_cluster"] > c["median_ratio_outside"] for c in significant)


def test_cluster_visual_labels_noise_points_separately():
    fake_rows = _fake_embeddings(MIN_SAMPLE_SIZE + 10)
    import random
    random.seed(7)
    for i in range(5):
        fake_rows.append((f"item-noise-{i}", [random.uniform(-3, 3) for _ in range(8)]))

    with patch("qualume.learn.visual_clustering._fetch_embeddings", return_value=fake_rows), \
         patch("qualume.learn.visual_clustering.performance_vs_baseline") as mock_perf:
        mock_perf.return_value = {"ratio_to_median": 1.0}
        result = cluster_visual(min_cluster_size=5)

    assert "n_noise" in result
    assert isinstance(result["n_noise"], int)


def test_cluster_visual_skips_items_with_no_performance_data():
    fake_rows = _fake_embeddings(MIN_SAMPLE_SIZE + 10)

    def fake_perf(item_id):
        if item_id == "item-a-0":
            raise ValueError("no performance snapshot")
        return {"ratio_to_median": 1.0}

    with patch("qualume.learn.visual_clustering._fetch_embeddings", return_value=fake_rows), \
         patch("qualume.learn.visual_clustering.performance_vs_baseline", side_effect=fake_perf):
        result = cluster_visual(min_cluster_size=5)

    assert result["n_scored"] == len(fake_rows) - 1
