def test_run_ingestion_cycle_skips_poll_and_download_when_ingestion_paused():
    """Real user decision 2026-11-01: 'stop downloading' — pause poll+
    download permanently until explicitly re-enabled, per the user's own
    choice (not auto-resuming on a threshold), while annotation/
    classification/mining/clustering keep running on the ~9755 videos
    already downloaded. Real reason: annotation throughput (the actual
    bottleneck, ~10% of the corpus annotated so far) matters more right
    now than growing an already-huge, already-9755-video download
    backlog further.

    QUALUME_PAUSE_INGESTION env var (not a code comment someone has to
    remember to revert) — checked live each call, same "read live, don't
    hardcode a decision" discipline as DNA_ANALYSIS_PROMPT_VERSION."""
    import os
    from unittest.mock import patch

    from qualume.orchestrate import run_ingestion_cycle

    with patch.dict(os.environ, {"QUALUME_PAUSE_INGESTION": "true"}), \
         patch("qualume.orchestrate.run_poll_cycle") as mock_poll, \
         patch("qualume.orchestrate.download_pending_videos") as mock_download, \
         patch("qualume.orchestrate.find_pending_annotation", return_value=[]), \
         patch("qualume.orchestrate.find_pending_classification", return_value=[]), \
         patch("qualume.orchestrate.find_pending_content_type_classification", return_value=[]), \
         patch("qualume.orchestrate.list_distinct_niches", return_value=[]), \
         patch("qualume.orchestrate.run_mining_pass", return_value={}), \
         patch("qualume.orchestrate.run_cluster_pass", return_value={}), \
         patch("qualume.orchestrate.run_visual_cluster_pass", return_value={}), \
         patch("qualume.orchestrate.detect_anomalies", return_value={}), \
         patch("qualume.orchestrate.compute_performance_tiers", return_value={}):

        result = run_ingestion_cycle()

    mock_poll.assert_not_called()
    mock_download.assert_not_called()
    assert result["poll"] == {"skipped": "QUALUME_PAUSE_INGESTION is set"}
    assert result["download"] == {"skipped": "QUALUME_PAUSE_INGESTION is set"}


def test_run_ingestion_cycle_runs_poll_and_download_when_not_paused():
    """Backward-compat guarantee: with the env var unset (the default,
    every existing deployment/test), poll+download run exactly as
    before — this toggle must be opt-in, never a silent new default."""
    import os
    from unittest.mock import patch

    from qualume.orchestrate import run_ingestion_cycle

    env_without_pause = {k: v for k, v in os.environ.items() if k != "QUALUME_PAUSE_INGESTION"}
    with patch.dict(os.environ, env_without_pause, clear=True), \
         patch("qualume.orchestrate.run_poll_cycle") as mock_poll, \
         patch("qualume.orchestrate.download_pending_videos") as mock_download, \
         patch("qualume.orchestrate.find_pending_annotation", return_value=[]), \
         patch("qualume.orchestrate.find_pending_classification", return_value=[]), \
         patch("qualume.orchestrate.find_pending_content_type_classification", return_value=[]), \
         patch("qualume.orchestrate.list_distinct_niches", return_value=[]), \
         patch("qualume.orchestrate.run_mining_pass", return_value={}), \
         patch("qualume.orchestrate.run_cluster_pass", return_value={}), \
         patch("qualume.orchestrate.run_visual_cluster_pass", return_value={}), \
         patch("qualume.orchestrate.detect_anomalies", return_value={}), \
         patch("qualume.orchestrate.compute_performance_tiers", return_value={}):

        mock_poll.return_value = {"accounts_polled": 1}
        mock_download.return_value = {"downloaded": 0}

        result = run_ingestion_cycle()

    mock_poll.assert_called_once()
    mock_download.assert_called_once()
    assert result["poll"] == {"accounts_polled": 1}
    assert result["download"] == {"downloaded": 0}
