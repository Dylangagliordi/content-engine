"""Tests for the automated clustering-pass orchestrator — written FIRST
per this project's TDD discipline.

Same shape as learn.run_mining_pass wrapping the declared-taxonomy
primitives into one n8n-callable function, applied to Phase B
(clustering) instead of Phase A (declared taxonomy): calls
learn.transcript_embedding.embed_and_store for every pending video,
calls learn.transcript_clustering.cluster_transcripts, auto-labels each
statistically significant cluster with a cheap Claude call (since nobody
manually reads 8 transcripts to name a cluster in an automated pipeline),
and persists findings into insights_external — tagged with niche
information (per explicit user decision: niche is a lightweight
informational signal attached to a finding, NOT baked into the embedding/
clustering math itself, keeping this portable to a different business's
accounts later).

Real difference from run_mining_pass's upsert-by-stable-identity
discipline: HDBSCAN cluster IDs are NOT stable across re-runs (the same
underlying group of videos might be "cluster 1" this run and "cluster 3"
next run, or split/merge entirely as new videos arrive) — there is no
stable identity to upsert against. So instead of upsert-in-place, each
pass RETIRES every cluster-sourced 'proposed' insight from source=
'transcript_clustering' and proposes fresh ones — honest, since clusters
really are re-discovered each run, not persistent named things.
"""
from unittest.mock import MagicMock, patch

import pytest

from qualume.learn.run_cluster_pass import run_cluster_pass


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


def test_run_cluster_pass_embeds_pending_videos_before_clustering():
    with patch("qualume.learn.run_cluster_pass._find_pending_embedding", return_value=["item-x", "item-y"]), \
         patch("qualume.learn.run_cluster_pass.embed_and_store") as mock_embed, \
         patch("qualume.learn.run_cluster_pass.cluster_transcripts", return_value=_cluster_result(n_clusters=0)), \
         patch("qualume.learn.run_cluster_pass.retire_stale_external_insights", return_value=0):

        run_cluster_pass(applies_to_workflow_type="tiktok_reels")

    assert mock_embed.call_count == 2


def test_run_cluster_pass_isolates_one_embedding_failure_from_the_rest():
    with patch("qualume.learn.run_cluster_pass._find_pending_embedding",
               return_value=["item-bad", "item-good"]), \
         patch("qualume.learn.run_cluster_pass.embed_and_store") as mock_embed, \
         patch("qualume.learn.run_cluster_pass.cluster_transcripts", return_value=_cluster_result(n_clusters=0)), \
         patch("qualume.learn.run_cluster_pass.retire_stale_external_insights", return_value=0):

        def side_effect(item_id):
            if item_id == "item-bad":
                raise RuntimeError("simulated Voyage failure")
        mock_embed.side_effect = side_effect

        result = run_cluster_pass(applies_to_workflow_type="tiktok_reels")

    assert result["embedding"]["succeeded"] == 1
    assert result["embedding"]["failed"] == 1
    assert result["embedding"]["skipped_no_transcript"] == 0


def test_run_cluster_pass_reports_no_transcript_items_separately_from_real_failures():
    """A video with no real hook_text/transcript at all (ValueError from
    embed_and_store's own build_embeddable_text precondition — a real,
    expected shape, e.g. a product-only/voiceover-over-broll video Gemini
    never produced spoken copy for) is NOT a failure — reporting it as
    one would be misleading (a real production run showed 25 'failed'
    that were actually just videos with nothing to embed, caught by
    reviewing real output, not assumed)."""
    with patch("qualume.learn.run_cluster_pass._find_pending_embedding",
               return_value=["item-no-transcript", "item-real-failure"]), \
         patch("qualume.learn.run_cluster_pass.embed_and_store") as mock_embed, \
         patch("qualume.learn.run_cluster_pass.cluster_transcripts", return_value=_cluster_result(n_clusters=0)), \
         patch("qualume.learn.run_cluster_pass.retire_stale_external_insights", return_value=0):

        def side_effect(item_id):
            if item_id == "item-no-transcript":
                raise ValueError("build_embeddable_text needs at least one of hook_text or full_transcript")
            raise RuntimeError("simulated Voyage failure")
        mock_embed.side_effect = side_effect

        result = run_cluster_pass(applies_to_workflow_type="tiktok_reels")

    assert result["embedding"]["succeeded"] == 0
    assert result["embedding"]["failed"] == 1
    assert result["embedding"]["skipped_no_transcript"] == 1


def test_run_cluster_pass_skips_clustering_below_sample_floor():
    """cluster_transcripts raises ValueError below its own MIN_SAMPLE_SIZE
    floor — the pass must report this honestly, not crash."""
    with patch("qualume.learn.run_cluster_pass._find_pending_embedding", return_value=[]), \
         patch("qualume.learn.run_cluster_pass.cluster_transcripts",
               side_effect=ValueError("cluster_transcripts needs at least 40 real embeddings, got 5")):

        result = run_cluster_pass(applies_to_workflow_type="tiktok_reels")

    assert "below sample floor" in result["clustering"]["skipped_reason"]


def test_run_cluster_pass_labels_and_proposes_significant_clusters():
    fake_label_response = MagicMock()
    fake_label_response.content = "Mega Live sale event announcements"

    fake_llm = MagicMock()
    fake_llm.invoke.return_value = fake_label_response

    with patch("qualume.learn.run_cluster_pass._find_pending_embedding", return_value=[]), \
         patch("qualume.learn.run_cluster_pass.cluster_transcripts",
               return_value=_cluster_result(n_clusters=1, p_value=0.001)), \
         patch("qualume.learn.run_cluster_pass.pick_llm", return_value=fake_llm), \
         patch("qualume.learn.run_cluster_pass._fetch_transcripts", return_value=["fake transcript 1"]), \
         patch("qualume.learn.run_cluster_pass._fetch_niche_summary", return_value={"women's health supplements"}), \
         patch("qualume.learn.run_cluster_pass.propose_external_insight") as mock_propose, \
         patch("qualume.learn.run_cluster_pass.retire_stale_external_insights", return_value=0):

        result = run_cluster_pass(applies_to_workflow_type="tiktok_reels")

    assert mock_propose.call_count == 1
    call_kwargs = mock_propose.call_args.kwargs
    assert "Mega Live" in call_kwargs["statement"]
    assert call_kwargs["applies_to_segment"]["source"] == "transcript_clustering"
    assert call_kwargs["applies_to_segment"]["niches"] == ["women's health supplements"]
    assert result["clustering"]["proposed"] == 1


def test_run_cluster_pass_skips_non_significant_clusters():
    with patch("qualume.learn.run_cluster_pass._find_pending_embedding", return_value=[]), \
         patch("qualume.learn.run_cluster_pass.cluster_transcripts",
               return_value=_cluster_result(n_clusters=1, p_value=0.80)), \
         patch("qualume.learn.run_cluster_pass.propose_external_insight") as mock_propose, \
         patch("qualume.learn.run_cluster_pass.retire_stale_external_insights", return_value=0):

        result = run_cluster_pass(applies_to_workflow_type="tiktok_reels")

    mock_propose.assert_not_called()
    assert result["clustering"]["skipped_not_significant"] == 1


def test_run_cluster_pass_retires_all_previous_cluster_sourced_proposed_insights():
    """Cluster IDs aren't stable across runs — every pass must retire
    EVERY previous source='transcript_clustering' proposed insight before
    proposing fresh ones, not upsert-match against a stale identity."""
    with patch("qualume.learn.run_cluster_pass._find_pending_embedding", return_value=[]), \
         patch("qualume.learn.run_cluster_pass.cluster_transcripts", return_value=_cluster_result(n_clusters=0)), \
         patch("qualume.learn.run_cluster_pass.retire_stale_external_insights") as mock_retire:
        mock_retire.return_value = 3

        result = run_cluster_pass(applies_to_workflow_type="tiktok_reels")

    mock_retire.assert_called_once()
    call_kwargs = mock_retire.call_args.kwargs
    assert call_kwargs["applies_to_workflow_type"] == "tiktok_reels"
    assert call_kwargs["source"] == "transcript_clustering"
    assert call_kwargs["still_significant_identities"] == []  # EVERY pass retires ALL, none carry over
    assert call_kwargs["retire_all"] is True  # the real flag that makes retire-all honest, not just an empty list
    assert result["clustering"]["retired"] == 3


def test_run_cluster_pass_retires_before_proposing_not_after():
    """Real bug found and fixed 2026-10-30: retire_all=True was called
    AFTER the propose loop in the same pass, which retired the insights
    this exact pass had just proposed seconds earlier, not just stale
    rows from a PREVIOUS run. A real production run showed 0 live
    proposed clustering insights after every single pass despite
    'proposed: 3' being reported — retire-then-propose is the only
    ordering where a pass's own fresh findings survive it."""
    call_order = []

    def fake_retire(**kwargs):
        call_order.append("retire")
        return 0

    def fake_propose(**kwargs):
        call_order.append("propose")
        return "fake-id"

    with patch("qualume.learn.run_cluster_pass._find_pending_embedding", return_value=[]), \
         patch("qualume.learn.run_cluster_pass.cluster_transcripts",
               return_value=_cluster_result(n_clusters=2, p_value=0.001)), \
         patch("qualume.learn.run_cluster_pass.pick_llm", return_value=MagicMock()), \
         patch("qualume.learn.run_cluster_pass._fetch_transcripts", return_value=["fake transcript"]), \
         patch("qualume.learn.run_cluster_pass._fetch_niche_summary", return_value=set()), \
         patch("qualume.learn.run_cluster_pass.propose_external_insight", side_effect=fake_propose), \
         patch("qualume.learn.run_cluster_pass.retire_stale_external_insights", side_effect=fake_retire):

        run_cluster_pass(applies_to_workflow_type="tiktok_reels")

    assert call_order[0] == "retire"  # retire happens FIRST
    assert all(c == "propose" for c in call_order[1:])  # every propose after it


def test_run_cluster_pass_applies_benjamini_hochberg_across_clusters():
    """Same real correction as learn.run_mining_pass, applied here too for
    consistency — a borderline p=0.04 cluster batched with several
    near-random clusters must not survive BH correction even though it
    would clear a flat p<0.05 on its own."""
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

    with patch("qualume.learn.run_cluster_pass._find_pending_embedding", return_value=[]), \
         patch("qualume.learn.run_cluster_pass.cluster_transcripts", return_value=cluster_result), \
         patch("qualume.learn.run_cluster_pass._fetch_transcripts", return_value=["fake transcript"]), \
         patch("qualume.learn.run_cluster_pass._fetch_niche_summary", return_value=set()), \
         patch("qualume.learn.run_cluster_pass.pick_llm", return_value=MagicMock()), \
         patch("qualume.learn.run_cluster_pass.propose_external_insight") as mock_propose, \
         patch("qualume.learn.run_cluster_pass.retire_stale_external_insights", return_value=0):

        result = run_cluster_pass(applies_to_workflow_type="tiktok_reels")

    # BH's corrected threshold for rank 1 of 7 tests at FDR=0.05 is
    # (1/7)*0.05 ≈ 0.0071, far below 0.04 — the borderline cluster must
    # be rejected even though a flat p<0.05 would have let it through.
    mock_propose.assert_not_called()
    assert result["clustering"]["proposed"] == 0
