"""Tests for the orchestration module — written FIRST per this project's
TDD discipline.

This is the n8n integration point: ONE function (run_ingestion_cycle) that
chains every already-built, already-proven step — poll watchlist ->
download pending videos -> annotate pending (Gemini v5) -> classify copy
for newly-annotated videos — so n8n's Schedule Trigger only needs to call
one HTTP endpoint, not orchestrate 4 separate script invocations itself.

Idempotent by construction: every underlying step already is (polling
dedupes on platform_video_id, download_pending_videos only touches rows
with local_video_path IS NULL, annotation only touches items with no v5
row yet, classification only touches items with no copy analysis yet) —
this module adds no NEW state, it just finds "pending" work at each stage
and calls the existing function. Each stage's failure is isolated and
reported, same discipline as poll_tiktok_watchlist's per-account
isolation — one video failing Gemini shouldn't block classification of
every other video that succeeded.
"""
import uuid
from unittest.mock import patch

import pytest

from qualume.annotate.copy_classifier import ANALYSIS_PROMPT_VERSION as COPY_VERSION
from qualume.annotate.gemini_watcher import ANALYSIS_PROMPT_VERSION as DNA_VERSION
from qualume.db.connection import get_connection
from qualume.db.watchlist import add_watched_account
from qualume.orchestrate import (
    find_pending_annotation,
    find_pending_classification,
    find_pending_content_type_classification,
    run_ingestion_cycle,
)


def _cleanup_account(handle: str):
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM external_copy_analysis WHERE external_content_item_id IN "
                "(SELECT id FROM external_content_items WHERE watched_account_id IN "
                "(SELECT id FROM watched_accounts WHERE handle = %s))",
                (handle,),
            )
            cur.execute(
                "DELETE FROM external_creative_dna WHERE external_content_item_id IN "
                "(SELECT id FROM external_content_items WHERE watched_account_id IN "
                "(SELECT id FROM watched_accounts WHERE handle = %s))",
                (handle,),
            )
            cur.execute(
                "DELETE FROM external_performance_snapshots WHERE external_content_item_id IN "
                "(SELECT id FROM external_content_items WHERE watched_account_id IN "
                "(SELECT id FROM watched_accounts WHERE handle = %s))",
                (handle,),
            )
            cur.execute(
                "DELETE FROM external_content_items WHERE watched_account_id IN "
                "(SELECT id FROM watched_accounts WHERE handle = %s)",
                (handle,),
            )
            cur.execute("DELETE FROM watched_accounts WHERE handle = %s", (handle,))
        conn.commit()
    finally:
        conn.close()


def _insert_item(account_id, local_video_path="/fake/path.mp4"):
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO external_content_items
                   (watched_account_id, platform, platform_video_id, url, local_video_path)
                   VALUES (%s, 'tiktok', %s, 'https://example.com/fake', %s)
                   RETURNING id""",
                (account_id, uuid.uuid4().hex, local_video_path),
            )
            item_id = cur.fetchone()[0]
        conn.commit()
        return item_id
    finally:
        conn.close()


def _insert_dna(item_id, version=None):
    if version is None:
        version = DNA_VERSION
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO external_creative_dna
                   (external_content_item_id, hook_text, analysis_model, analysis_prompt_version)
                   VALUES (%s, 'a hook', 'test-model', %s)""",
                (item_id, version),
            )
        conn.commit()
    finally:
        conn.close()


@pytest.fixture
def fake_account():
    handle = f"test_orch_{uuid.uuid4().hex[:8]}"
    _cleanup_account(handle)
    account_id = add_watched_account("tiktok", handle)
    yield account_id, handle
    _cleanup_account(handle)


def test_find_pending_annotation_returns_items_with_video_but_no_current_dna(fake_account):
    """'Current' = qualume.annotate.gemini_watcher.ANALYSIS_PROMPT_VERSION, read
    live, not a hardcoded string — this is a regression test for a real bug:
    orchestrate.py hardcoded 'v5' even after gemini_watcher bumped to v6,
    silently excluding every v6-only video from ever being found pending
    (caught running this against real production data, see git log)."""
    account_id, _ = fake_account
    item_with_video = _insert_item(account_id, local_video_path="/real/path.mp4")
    item_no_video = _insert_item(account_id, local_video_path=None)
    item_already_current = _insert_item(account_id, local_video_path="/real/path2.mp4")
    _insert_dna(item_already_current, version=DNA_VERSION)
    item_only_v1 = _insert_item(account_id, local_video_path="/real/path3.mp4")
    _insert_dna(item_only_v1, version="v1")

    pending = find_pending_annotation(watched_account_id=account_id)

    assert item_with_video in pending
    assert item_no_video not in pending  # no video file -> can't annotate
    assert item_already_current not in pending  # already has the current version
    assert item_only_v1 in pending  # only v1 -> still pending for the current version


def test_find_pending_classification_returns_current_dna_items_with_no_copy_analysis(fake_account):
    """Same live-version-constant discipline as find_pending_annotation above
    — classification pending-check must track gemini_watcher's CURRENT
    ANALYSIS_PROMPT_VERSION, not a string that goes stale the next time that
    version bumps."""
    account_id, _ = fake_account
    item_ready = _insert_item(account_id)
    _insert_dna(item_ready, version=DNA_VERSION)
    item_no_dna_yet = _insert_item(account_id)
    item_already_classified = _insert_item(account_id)
    _insert_dna(item_already_classified, version=DNA_VERSION)

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO external_copy_analysis
                   (external_content_item_id, analysis_model, analysis_prompt_version)
                   VALUES (%s, 'claude', %s)""",
                (item_already_classified, COPY_VERSION),
            )
        conn.commit()
    finally:
        conn.close()

    pending = find_pending_classification(watched_account_id=account_id)

    assert item_ready in pending
    assert item_no_dna_yet not in pending  # no current-version annotation -> nothing to classify from
    assert item_already_classified not in pending


def test_find_pending_annotation_treats_a_current_version_item_as_not_pending_even_if_older_versions_also_exist(
    fake_account,
):
    """A video annotated at BOTH v1 and the current version (re-annotation
    history, same real shape as the v1->v6 backfill already run on
    production data) must not be reported pending just because an older row
    exists alongside the current one."""
    account_id, _ = fake_account
    item_id = _insert_item(account_id, local_video_path="/real/path4.mp4")
    _insert_dna(item_id, version="v1")
    _insert_dna(item_id, version=DNA_VERSION)

    pending = find_pending_annotation(watched_account_id=account_id)
    assert item_id not in pending


def test_find_pending_content_type_classification_returns_tiktok_items_with_dna_but_no_content_type(
    fake_account,
):
    """Real gap closed 2026-11-02: content_type (migration 017) can't be
    inferred from platform/account for TikTok (unlike meta_ads, which
    is selling-by-definition and seeded at ingest) — needs its own real
    classification pass, same 'find what's genuinely missing' shape as
    every other pending-finder in orchestrate.py."""
    account_id, _ = fake_account
    item_ready = _insert_item(account_id)
    _insert_dna(item_ready, version=DNA_VERSION)
    item_no_dna_yet = _insert_item(account_id)
    item_already_classified = _insert_item(account_id)
    _insert_dna(item_already_classified, version=DNA_VERSION)

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE external_content_items SET content_type = 'organic' WHERE id = %s",
                (item_already_classified,),
            )
        conn.commit()
    finally:
        conn.close()

    pending = find_pending_content_type_classification(watched_account_id=account_id)

    assert item_ready in pending
    assert item_no_dna_yet not in pending  # no DNA yet -> nothing to classify from
    assert item_already_classified not in pending


def test_run_ingestion_cycle_chains_all_six_stages_and_reports_each(fake_account):
    account_id, handle = fake_account

    with patch("qualume.orchestrate.run_poll_cycle") as mock_poll, \
         patch("qualume.orchestrate.download_pending_videos") as mock_download, \
         patch("qualume.orchestrate.annotate_and_store") as mock_annotate, \
         patch("qualume.orchestrate.classify_and_store_external") as mock_classify, \
         patch("qualume.orchestrate.classify_and_store_content_type") as mock_classify_ctype, \
         patch("qualume.orchestrate.find_pending_annotation") as mock_find_ann, \
         patch("qualume.orchestrate.find_pending_classification") as mock_find_cls, \
         patch("qualume.orchestrate.find_pending_content_type_classification") as mock_find_ctype, \
         patch("qualume.orchestrate.run_mining_pass") as mock_mining, \
         patch("qualume.orchestrate.run_cluster_pass") as mock_cluster, \
         patch("qualume.orchestrate.run_visual_cluster_pass") as mock_visual_cluster:

        mock_poll.return_value = {"accounts_polled": 1, "new_videos": 2, "refreshed_snapshots": 0}
        mock_download.return_value = {"downloaded": 2, "failed": 0, "already_had_path": 0}
        mock_find_ann.return_value = ["item-a", "item-b"]
        mock_annotate.side_effect = ["dna-a", "dna-b"]
        mock_find_cls.return_value = ["item-a", "item-b"]
        mock_classify.side_effect = ["copy-a", "copy-b"]
        mock_find_ctype.return_value = ["item-a", "item-b"]
        mock_classify_ctype.side_effect = ["item-a", "item-b"]
        mock_mining.return_value = {
            "tested": 5, "proposed_or_refreshed": 2,
            "skipped_not_significant": 3, "retired": 0, "errors": [],
        }
        mock_cluster.return_value = {
            "embedding": {"succeeded": 3, "failed": 0},
            "clustering": {"n_total": 70, "proposed": 1, "retired": 0},
        }
        mock_visual_cluster.return_value = {
            "embedding": {"succeeded": 2, "failed": 0, "skipped_no_video": 0},
            "clustering": {"n_total": 50, "proposed": 1, "retired": 0},
        }

        result = run_ingestion_cycle()

    assert result["poll"] == mock_poll.return_value
    assert result["download"] == mock_download.return_value
    assert result["annotation"]["succeeded"] == 2
    assert result["annotation"]["failed"] == 0
    assert result["classification"]["succeeded"] == 2
    assert result["classification"]["failed"] == 0
    assert result["content_type_classification"]["succeeded"] == 2
    assert result["content_type_classification"]["failed"] == 0
    assert result["mining"] == mock_mining.return_value
    mock_mining.assert_called_once_with(applies_to_workflow_type="tiktok_reels")
    assert result["clustering"] == mock_cluster.return_value
    mock_cluster.assert_called_once_with(applies_to_workflow_type="tiktok_reels")
    assert result["visual_clustering"] == mock_visual_cluster.return_value
    mock_visual_cluster.assert_called_once_with(applies_to_workflow_type="tiktok_reels")


def test_run_ingestion_cycle_isolates_a_whole_mining_stage_failure():
    """A mining-pass failure (e.g. a transient DB hiccup across the whole
    stage) must not be fatal to the cycle's result — same whole-stage
    isolation discipline as poll/download/annotation/classification."""
    with patch("qualume.orchestrate.run_poll_cycle") as mock_poll, \
         patch("qualume.orchestrate.download_pending_videos") as mock_download, \
         patch("qualume.orchestrate.find_pending_annotation", return_value=[]), \
         patch("qualume.orchestrate.find_pending_classification", return_value=[]), \
         patch("qualume.orchestrate.run_mining_pass") as mock_mining, \
         patch("qualume.orchestrate.run_cluster_pass", return_value={}), \
         patch("qualume.orchestrate.run_visual_cluster_pass", return_value={}):

        mock_poll.return_value = {}
        mock_download.return_value = {}
        mock_mining.side_effect = RuntimeError("simulated mining failure")

        result = run_ingestion_cycle()

    assert result["mining"]["error"] is not None
    assert "simulated mining failure" in result["mining"]["error"]


def test_run_ingestion_cycle_calls_cluster_pass_as_its_6th_stage():
    with patch("qualume.orchestrate.run_poll_cycle", return_value={}), \
         patch("qualume.orchestrate.download_pending_videos", return_value={}), \
         patch("qualume.orchestrate.find_pending_annotation", return_value=[]), \
         patch("qualume.orchestrate.find_pending_classification", return_value=[]), \
         patch("qualume.orchestrate.run_mining_pass", return_value={}), \
         patch("qualume.orchestrate.run_cluster_pass") as mock_cluster, \
         patch("qualume.orchestrate.run_visual_cluster_pass", return_value={}):

        mock_cluster.return_value = {
            "embedding": {"succeeded": 3, "failed": 0},
            "clustering": {"n_total": 70, "proposed": 1, "retired": 0},
        }

        result = run_ingestion_cycle()

    assert result["clustering"] == mock_cluster.return_value
    mock_cluster.assert_called_once_with(applies_to_workflow_type="tiktok_reels")


def test_run_ingestion_cycle_isolates_a_whole_clustering_stage_failure():
    with patch("qualume.orchestrate.run_poll_cycle", return_value={}), \
         patch("qualume.orchestrate.download_pending_videos", return_value={}), \
         patch("qualume.orchestrate.find_pending_annotation", return_value=[]), \
         patch("qualume.orchestrate.find_pending_classification", return_value=[]), \
         patch("qualume.orchestrate.run_mining_pass", return_value={}), \
         patch("qualume.orchestrate.run_cluster_pass") as mock_cluster, \
         patch("qualume.orchestrate.run_visual_cluster_pass", return_value={}):

        mock_cluster.side_effect = RuntimeError("simulated Voyage outage")

        result = run_ingestion_cycle()

    assert result["clustering"]["error"] is not None
    assert "simulated Voyage outage" in result["clustering"]["error"]


def test_run_ingestion_cycle_calls_visual_cluster_pass_as_its_7th_stage():
    with patch("qualume.orchestrate.run_poll_cycle", return_value={}), \
         patch("qualume.orchestrate.download_pending_videos", return_value={}), \
         patch("qualume.orchestrate.find_pending_annotation", return_value=[]), \
         patch("qualume.orchestrate.find_pending_classification", return_value=[]), \
         patch("qualume.orchestrate.run_mining_pass", return_value={}), \
         patch("qualume.orchestrate.run_cluster_pass", return_value={}), \
         patch("qualume.orchestrate.run_visual_cluster_pass") as mock_visual_cluster:

        mock_visual_cluster.return_value = {
            "embedding": {"succeeded": 2, "failed": 0, "skipped_no_video": 0},
            "clustering": {"n_total": 50, "proposed": 1, "retired": 0},
        }

        result = run_ingestion_cycle()

    assert result["visual_clustering"] == mock_visual_cluster.return_value
    mock_visual_cluster.assert_called_once_with(applies_to_workflow_type="tiktok_reels")


def test_run_ingestion_cycle_isolates_a_whole_visual_clustering_stage_failure():
    with patch("qualume.orchestrate.run_poll_cycle", return_value={}), \
         patch("qualume.orchestrate.download_pending_videos", return_value={}), \
         patch("qualume.orchestrate.find_pending_annotation", return_value=[]), \
         patch("qualume.orchestrate.find_pending_classification", return_value=[]), \
         patch("qualume.orchestrate.run_mining_pass", return_value={}), \
         patch("qualume.orchestrate.run_cluster_pass", return_value={}), \
         patch("qualume.orchestrate.run_visual_cluster_pass") as mock_visual_cluster:

        mock_visual_cluster.side_effect = RuntimeError("simulated Voyage multimodal outage")

        result = run_ingestion_cycle()

    assert result["visual_clustering"]["error"] is not None
    assert "simulated Voyage multimodal outage" in result["visual_clustering"]["error"]


def test_run_ingestion_cycle_isolates_one_annotation_failure_from_others(fake_account):
    with patch("qualume.orchestrate.run_poll_cycle") as mock_poll, \
         patch("qualume.orchestrate.download_pending_videos") as mock_download, \
         patch("qualume.orchestrate.annotate_and_store") as mock_annotate, \
         patch("qualume.orchestrate.classify_and_store_external") as mock_classify, \
         patch("qualume.orchestrate.find_pending_annotation") as mock_find_ann, \
         patch("qualume.orchestrate.find_pending_classification") as mock_find_cls, \
         patch("qualume.orchestrate.run_mining_pass", return_value={}), \
         patch("qualume.orchestrate.run_cluster_pass", return_value={}), \
         patch("qualume.orchestrate.run_visual_cluster_pass", return_value={}):

        mock_poll.return_value = {}
        mock_download.return_value = {}
        mock_find_ann.return_value = ["item-a", "item-bad", "item-c"]

        def annotate_side_effect(item_id):
            if item_id == "item-bad":
                raise RuntimeError("simulated Gemini failure")
            return f"dna-{item_id}"
        mock_annotate.side_effect = annotate_side_effect
        mock_find_cls.return_value = []

        result = run_ingestion_cycle()

    assert result["annotation"]["succeeded"] == 2
    assert result["annotation"]["failed"] == 1
    assert "item-bad" in str(result["annotation"]["errors"])


def test_run_ingestion_cycle_isolates_one_classification_failure_from_others(fake_account):
    with patch("qualume.orchestrate.run_poll_cycle") as mock_poll, \
         patch("qualume.orchestrate.download_pending_videos") as mock_download, \
         patch("qualume.orchestrate.annotate_and_store") as mock_annotate, \
         patch("qualume.orchestrate.classify_and_store_external") as mock_classify, \
         patch("qualume.orchestrate.find_pending_annotation") as mock_find_ann, \
         patch("qualume.orchestrate.find_pending_classification") as mock_find_cls, \
         patch("qualume.orchestrate.run_mining_pass", return_value={}), \
         patch("qualume.orchestrate.run_cluster_pass", return_value={}), \
         patch("qualume.orchestrate.run_visual_cluster_pass", return_value={}):

        mock_poll.return_value = {}
        mock_download.return_value = {}
        mock_find_ann.return_value = []
        mock_find_cls.return_value = ["item-a", "item-bad"]

        def classify_side_effect(item_id):
            if item_id == "item-bad":
                raise RuntimeError("simulated Claude failure")
            return f"copy-{item_id}"
        mock_classify.side_effect = classify_side_effect

        result = run_ingestion_cycle()

    assert result["classification"]["succeeded"] == 1
    assert result["classification"]["failed"] == 1


def test_run_ingestion_cycle_continues_past_a_failed_poll_or_download_stage():
    """A poll or download failure (e.g. TikTok temporarily blocking) must
    not prevent annotation/classification from running on whatever
    videos ARE already downloaded — same per-stage isolation as
    per-account isolation in poll_tiktok_watchlist."""
    with patch("qualume.orchestrate.run_poll_cycle") as mock_poll, \
         patch("qualume.orchestrate.download_pending_videos") as mock_download, \
         patch("qualume.orchestrate.annotate_and_store") as mock_annotate, \
         patch("qualume.orchestrate.classify_and_store_external") as mock_classify, \
         patch("qualume.orchestrate.find_pending_annotation") as mock_find_ann, \
         patch("qualume.orchestrate.find_pending_classification") as mock_find_cls, \
         patch("qualume.orchestrate.run_mining_pass", return_value={}), \
         patch("qualume.orchestrate.run_cluster_pass", return_value={}), \
         patch("qualume.orchestrate.run_visual_cluster_pass", return_value={}):

        mock_poll.side_effect = RuntimeError("simulated poll failure")
        mock_download.return_value = {"downloaded": 0, "failed": 0, "already_had_path": 0}
        mock_find_ann.return_value = []
        mock_find_cls.return_value = []

        result = run_ingestion_cycle()

    assert result["poll"]["error"] is not None
    assert result["download"] == {"downloaded": 0, "failed": 0, "already_had_path": 0}
