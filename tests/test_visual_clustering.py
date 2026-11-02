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


def test_cluster_visual_passes_niche_through_to_fetch_embeddings():
    """Same real fix (2026-11-02) as transcript_clustering.
    cluster_transcripts — the whole-watchlist niche-mixing problem
    applies identically to the visual clustering pipeline."""
    fake_rows = _fake_embeddings(MIN_SAMPLE_SIZE + 10)

    with patch("qualume.learn.visual_clustering._fetch_embeddings", return_value=fake_rows) as mock_fetch, \
         patch("qualume.learn.visual_clustering.performance_vs_baseline") as mock_perf:
        mock_perf.return_value = {"ratio_to_median": 1.0}
        cluster_visual(min_cluster_size=5, niche="zz_test_niche")

    mock_fetch.assert_called_once_with(niche="zz_test_niche")


def test_fetch_embeddings_scopes_to_one_niche_when_given():
    """Real integration test against the actual dev DB (not mockable the
    way the pure-math clustering logic is) — niche is inherited from
    watched_accounts via external_content_items, not stored directly on
    visual_embeddings, so this needs a real JOIN proven against real
    rows."""
    import json
    import uuid

    from qualume.db.connection import get_connection
    from qualume.db.watchlist import add_watched_account
    from qualume.learn.visual_clustering import _fetch_embeddings

    target_handle = f"test_visual_niche_target_{uuid.uuid4().hex[:8]}"
    other_handle = f"test_visual_niche_other_{uuid.uuid4().hex[:8]}"

    def _cleanup(handle):
        conn = get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM visual_embeddings WHERE external_content_item_id IN "
                    "(SELECT id FROM external_content_items WHERE watched_account_id IN "
                    "(SELECT id FROM watched_accounts WHERE handle = %s))", (handle,))
                cur.execute(
                    "DELETE FROM external_content_items WHERE watched_account_id IN "
                    "(SELECT id FROM watched_accounts WHERE handle = %s)", (handle,))
                cur.execute("DELETE FROM watched_accounts WHERE handle = %s", (handle,))
            conn.commit()
        finally:
            conn.close()

    _cleanup(target_handle)
    _cleanup(other_handle)
    try:
        target_account = add_watched_account("tiktok", target_handle, niche="zz_test_visual_niche")
        other_account = add_watched_account("tiktok", other_handle, niche="zz_test_visual_other")

        target_item_ids, other_item_ids = [], []
        conn = get_connection()
        try:
            with conn.cursor() as cur:
                for account_id, id_list in [(target_account, target_item_ids), (other_account, other_item_ids)]:
                    cur.execute(
                        """INSERT INTO external_content_items
                           (watched_account_id, platform, platform_video_id, url)
                           VALUES (%s, 'tiktok', %s, 'https://example.com/fake')
                           RETURNING id""",
                        (account_id, uuid.uuid4().hex),
                    )
                    item_id = cur.fetchone()[0]
                    id_list.append(str(item_id))
                    cur.execute(
                        """INSERT INTO visual_embeddings
                           (external_content_item_id, embedding, n_frames_used, embedding_model)
                           VALUES (%s, %s, 6, 'test-model')""",
                        (item_id, json.dumps([0.1, 0.2, 0.3])),
                    )
            conn.commit()
        finally:
            conn.close()

        scoped = _fetch_embeddings(niche="zz_test_visual_niche")
        scoped_ids = {item_id for item_id, _ in scoped}
        assert set(target_item_ids) <= scoped_ids
        assert not (set(other_item_ids) & scoped_ids)
    finally:
        _cleanup(target_handle)
        _cleanup(other_handle)
