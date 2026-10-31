"""Tests for the visual clustering orchestrator — mirrors test_run_
cluster_pass.py's exact shape, pointed at the visual pipeline (visual_
embedding/visual_clustering) instead of transcript. Per the user's
explicit decision this session (2026-10-31): visual clustering is a
SEPARATE pass from transcript clustering, proposing insights tagged
source='visual_clustering' so real quality/signal from each side stays
independently comparable.

Real difference from the transcript version's _fetch_transcripts: there
is no spoken-copy text to show Claude for labeling a visual cluster —
instead this module describes each cluster via its members' REAL
structured shot fields (shot_type, setting, lighting, color_temperature
— already computed by Gemini annotation, not re-derived here), which is
the honest visual-equivalent signal to label from.
"""
from unittest.mock import MagicMock, patch

import pytest

from qualume.learn.run_visual_cluster_pass import run_visual_cluster_pass


def _cluster_result(n_clusters=1, p_value=0.01, size=8, item_ids=None):
    return {
        "n_total": 70, "n_scored": 70, "n_noise": 30,
        "clusters": [
            {
                "cluster_id": i, "size": size, "item_ids": item_ids or [f"item-{i}-{j}" for j in range(size)],
                "median_ratio_in_cluster": 5.0, "median_ratio_outside": 1.0, "p_value": p_value,
            }
            for i in range(n_clusters)
        ],
    }


def test_run_visual_cluster_pass_embeds_pending_videos_before_clustering():
    with patch("qualume.learn.run_visual_cluster_pass._find_pending_visual_embedding", return_value=["item-x", "item-y"]), \
         patch("qualume.learn.run_visual_cluster_pass.embed_and_store") as mock_embed, \
         patch("qualume.learn.run_visual_cluster_pass.cluster_visual", return_value=_cluster_result(n_clusters=0)), \
         patch("qualume.learn.run_visual_cluster_pass.retire_stale_external_insights", return_value=0):

        run_visual_cluster_pass(applies_to_workflow_type="tiktok_reels")

    assert mock_embed.call_count == 2


def test_run_visual_cluster_pass_isolates_one_embedding_failure_from_the_rest():
    with patch("qualume.learn.run_visual_cluster_pass._find_pending_visual_embedding",
               return_value=["item-bad", "item-good"]), \
         patch("qualume.learn.run_visual_cluster_pass.embed_and_store") as mock_embed, \
         patch("qualume.learn.run_visual_cluster_pass.cluster_visual", return_value=_cluster_result(n_clusters=0)), \
         patch("qualume.learn.run_visual_cluster_pass.retire_stale_external_insights", return_value=0):

        def side_effect(item_id):
            if item_id == "item-bad":
                raise RuntimeError("simulated Voyage failure")
        mock_embed.side_effect = side_effect

        result = run_visual_cluster_pass(applies_to_workflow_type="tiktok_reels")

    assert result["embedding"]["succeeded"] == 1
    assert result["embedding"]["failed"] == 1


def test_run_visual_cluster_pass_reports_no_video_items_separately_from_real_failures():
    """A video with no local_video_path/shot_breakdown yet (ValueError
    from embed_and_store's own precondition) is NOT a real failure —
    same honesty discipline as the transcript pipeline's skipped_no_
    transcript counter, just for the visual precondition instead."""
    with patch("qualume.learn.run_visual_cluster_pass._find_pending_visual_embedding",
               return_value=["item-no-video", "item-real-failure"]), \
         patch("qualume.learn.run_visual_cluster_pass.embed_and_store") as mock_embed, \
         patch("qualume.learn.run_visual_cluster_pass.cluster_visual", return_value=_cluster_result(n_clusters=0)), \
         patch("qualume.learn.run_visual_cluster_pass.retire_stale_external_insights", return_value=0):

        def side_effect(item_id):
            if item_id == "item-no-video":
                raise ValueError("no local_video_path — nothing downloaded to embed from yet")
            raise RuntimeError("simulated Voyage failure")
        mock_embed.side_effect = side_effect

        result = run_visual_cluster_pass(applies_to_workflow_type="tiktok_reels")

    assert result["embedding"]["succeeded"] == 0
    assert result["embedding"]["failed"] == 1
    assert result["embedding"]["skipped_no_video"] == 1


def test_run_visual_cluster_pass_skips_clustering_below_sample_floor():
    with patch("qualume.learn.run_visual_cluster_pass._find_pending_visual_embedding", return_value=[]), \
         patch("qualume.learn.run_visual_cluster_pass.cluster_visual",
               side_effect=ValueError("cluster_visual needs at least 40 real embeddings, got 5")):

        result = run_visual_cluster_pass(applies_to_workflow_type="tiktok_reels")

    assert "below sample floor" in result["clustering"]["skipped_reason"]


def test_run_visual_cluster_pass_labels_and_proposes_significant_clusters():
    fake_label_response = MagicMock()
    fake_label_response.content = "Fast-cut product demo close-ups"

    fake_llm = MagicMock()
    fake_llm.invoke.return_value = fake_label_response

    with patch("qualume.learn.run_visual_cluster_pass._find_pending_visual_embedding", return_value=[]), \
         patch("qualume.learn.run_visual_cluster_pass.cluster_visual",
               return_value=_cluster_result(n_clusters=1, p_value=0.001)), \
         patch("qualume.learn.run_visual_cluster_pass.pick_llm", return_value=fake_llm), \
         patch("qualume.learn.run_visual_cluster_pass._fetch_shot_descriptions", return_value=["fake shot desc"]), \
         patch("qualume.learn.run_visual_cluster_pass._fetch_niche_summary", return_value={"fitness"}), \
         patch("qualume.learn.run_visual_cluster_pass.propose_external_insight") as mock_propose, \
         patch("qualume.learn.run_visual_cluster_pass.retire_stale_external_insights", return_value=0):

        result = run_visual_cluster_pass(applies_to_workflow_type="tiktok_reels")

    assert mock_propose.call_count == 1
    call_kwargs = mock_propose.call_args.kwargs
    assert "Fast-cut product" in call_kwargs["statement"]
    assert call_kwargs["applies_to_segment"]["source"] == "visual_clustering"
    assert call_kwargs["applies_to_segment"]["niches"] == ["fitness"]
    assert result["clustering"]["proposed"] == 1


def test_run_visual_cluster_pass_skips_non_significant_clusters():
    with patch("qualume.learn.run_visual_cluster_pass._find_pending_visual_embedding", return_value=[]), \
         patch("qualume.learn.run_visual_cluster_pass.cluster_visual",
               return_value=_cluster_result(n_clusters=1, p_value=0.80)), \
         patch("qualume.learn.run_visual_cluster_pass.propose_external_insight") as mock_propose, \
         patch("qualume.learn.run_visual_cluster_pass.retire_stale_external_insights", return_value=0):

        result = run_visual_cluster_pass(applies_to_workflow_type="tiktok_reels")

    mock_propose.assert_not_called()
    assert result["clustering"]["skipped_not_significant"] == 1


def test_run_visual_cluster_pass_retires_all_previous_proposed_insights():
    with patch("qualume.learn.run_visual_cluster_pass._find_pending_visual_embedding", return_value=[]), \
         patch("qualume.learn.run_visual_cluster_pass.cluster_visual", return_value=_cluster_result(n_clusters=0)), \
         patch("qualume.learn.run_visual_cluster_pass.retire_stale_external_insights") as mock_retire:
        mock_retire.return_value = 3

        result = run_visual_cluster_pass(applies_to_workflow_type="tiktok_reels")

    mock_retire.assert_called_once()
    call_kwargs = mock_retire.call_args.kwargs
    assert call_kwargs["applies_to_workflow_type"] == "tiktok_reels"
    assert call_kwargs["source"] == "visual_clustering"
    assert call_kwargs["still_significant_identities"] == []
    assert call_kwargs["retire_all"] is True
    assert result["clustering"]["retired"] == 3


def test_run_visual_cluster_pass_retires_before_proposing_not_after():
    call_order = []

    def fake_retire(**kwargs):
        call_order.append("retire")
        return 0

    def fake_propose(**kwargs):
        call_order.append("propose")
        return "fake-id"

    with patch("qualume.learn.run_visual_cluster_pass._find_pending_visual_embedding", return_value=[]), \
         patch("qualume.learn.run_visual_cluster_pass.cluster_visual",
               return_value=_cluster_result(n_clusters=2, p_value=0.001)), \
         patch("qualume.learn.run_visual_cluster_pass.pick_llm", return_value=MagicMock()), \
         patch("qualume.learn.run_visual_cluster_pass._fetch_shot_descriptions", return_value=["fake shot desc"]), \
         patch("qualume.learn.run_visual_cluster_pass._fetch_niche_summary", return_value=set()), \
         patch("qualume.learn.run_visual_cluster_pass.propose_external_insight", side_effect=fake_propose), \
         patch("qualume.learn.run_visual_cluster_pass.retire_stale_external_insights", side_effect=fake_retire):

        run_visual_cluster_pass(applies_to_workflow_type="tiktok_reels")

    assert call_order[0] == "retire"
    assert all(c == "propose" for c in call_order[1:])


def test_run_visual_cluster_pass_applies_benjamini_hochberg_across_clusters():
    import random
    random.seed(55)
    clusters = [
        {
            "cluster_id": i, "size": 8, "item_ids": [f"item-{i}-{j}" for j in range(8)],
            "median_ratio_in_cluster": 1.1, "median_ratio_outside": 1.0,
            "p_value": random.uniform(0.2, 0.9),
        }
        for i in range(6)
    ]
    clusters.append({
        "cluster_id": 99, "size": 8, "item_ids": [f"item-99-{j}" for j in range(8)],
        "median_ratio_in_cluster": 5.0, "median_ratio_outside": 1.0, "p_value": 0.04,
    })
    cluster_result = {"n_total": 70, "n_scored": 70, "n_noise": 10, "clusters": clusters}

    with patch("qualume.learn.run_visual_cluster_pass._find_pending_visual_embedding", return_value=[]), \
         patch("qualume.learn.run_visual_cluster_pass.cluster_visual", return_value=cluster_result), \
         patch("qualume.learn.run_visual_cluster_pass._fetch_shot_descriptions", return_value=["fake shot desc"]), \
         patch("qualume.learn.run_visual_cluster_pass._fetch_niche_summary", return_value=set()), \
         patch("qualume.learn.run_visual_cluster_pass.pick_llm", return_value=MagicMock()), \
         patch("qualume.learn.run_visual_cluster_pass.propose_external_insight") as mock_propose, \
         patch("qualume.learn.run_visual_cluster_pass.retire_stale_external_insights", return_value=0):

        result = run_visual_cluster_pass(applies_to_workflow_type="tiktok_reels")

    mock_propose.assert_not_called()
    assert result["clustering"]["proposed"] == 0
