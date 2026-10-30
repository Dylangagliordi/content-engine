"""Tests for the automated mining-pass orchestrator — written FIRST per
this project's TDD discipline.

This is the piece that turns Part 5b's already-built, already-tested
primitives (learn.mining.mine_categorical_feature, learn.copy_mining.
mine_boolean_copy_feature, learn.validation.upsert_external_insight /
retire_stale_external_insights) into ONE function a scheduled n8n cycle
can call — same shape as orchestrate.run_ingestion_cycle wrapping poll/
download/annotate/classify into one callable, applied to Part 5b instead
of Part 5a.

Mocks the underlying mine_* functions (already proven correct against
real synthetic-injected-correlation data in their own test files) rather
than re-testing statistics here — this module's own job is purely
plumbing: call every mineable column/flag, keep only results clearing
the significance threshold, upsert those, retire whatever stopped being
significant. That plumbing is what's actually new and needs its own
tests.
"""
from unittest.mock import patch

import pytest

from qualume.learn.run_mining_pass import (
    SIGNIFICANCE_THRESHOLD,
    run_mining_pass,
)


def _dna_result(value, p_value, n_with=10, n_without=10, ratio_with=1.3, ratio_without=0.9):
    return {
        "value": value, "n_with": n_with, "n_without": n_without,
        "median_ratio_with": ratio_with, "median_ratio_without": ratio_without,
        "p_value": p_value,
    }


def _copy_result(flag, p_value, n_true=10, n_false=10, ratio_true=1.3, ratio_false=0.9):
    return {
        "flag": flag, "n_true": n_true, "n_false": n_false,
        "median_ratio_true": ratio_true, "median_ratio_false": ratio_false,
        "p_value": p_value,
    }


def test_run_mining_pass_upserts_only_results_clearing_significance_threshold():
    with patch("qualume.learn.run_mining_pass.MINEABLE_COLUMNS", frozenset({"hook_type"})), \
         patch("qualume.learn.run_mining_pass.MINEABLE_COPY_FLAGS", frozenset()), \
         patch("qualume.learn.run_mining_pass.mine_categorical_feature") as mock_mine_dna, \
         patch("qualume.learn.run_mining_pass.mine_boolean_copy_feature") as mock_mine_copy, \
         patch("qualume.learn.run_mining_pass.upsert_external_insight") as mock_upsert, \
         patch("qualume.learn.run_mining_pass.retire_stale_external_insights") as mock_retire:

        mock_mine_dna.return_value = [
            _dna_result("question", p_value=0.01),       # significant
            _dna_result("bold_claim", p_value=0.80),      # NOT significant
        ]
        mock_mine_copy.return_value = []
        mock_upsert.return_value = "insight-id"
        mock_retire.return_value = 0

        result = run_mining_pass(applies_to_workflow_type="tiktok_video")

    assert mock_upsert.call_count == 1
    call_kwargs = mock_upsert.call_args.kwargs
    assert call_kwargs["identity"]["value"] == "question"
    assert result["proposed_or_refreshed"] == 1
    assert result["tested"] == 2
    assert result["skipped_not_significant"] == 1


def test_run_mining_pass_covers_every_mineable_dna_column_and_copy_flag():
    with patch("qualume.learn.run_mining_pass.MINEABLE_COLUMNS", frozenset({"hook_type", "format_tag"})), \
         patch("qualume.learn.run_mining_pass.MINEABLE_COPY_FLAGS", frozenset({"uses_urgency"})), \
         patch("qualume.learn.run_mining_pass.mine_categorical_feature") as mock_mine_dna, \
         patch("qualume.learn.run_mining_pass.mine_boolean_copy_feature") as mock_mine_copy, \
         patch("qualume.learn.run_mining_pass.upsert_external_insight"), \
         patch("qualume.learn.run_mining_pass.retire_stale_external_insights", return_value=0):

        mock_mine_dna.return_value = []
        mock_mine_copy.return_value = None  # mimics a real None return (group too small)

        run_mining_pass(applies_to_workflow_type="tiktok_video")

    dna_columns_called = {c.args[0] for c in mock_mine_dna.call_args_list}
    assert dna_columns_called == {"hook_type", "format_tag"}
    copy_flags_called = {c.args[0] for c in mock_mine_copy.call_args_list}
    assert copy_flags_called == {"uses_urgency"}


def test_run_mining_pass_handles_none_return_from_copy_mining_without_crashing():
    """mine_boolean_copy_feature returns None (not a list) when a group is
    too small — must not be treated as an iterable list of results."""
    with patch("qualume.learn.run_mining_pass.MINEABLE_COLUMNS", frozenset()), \
         patch("qualume.learn.run_mining_pass.MINEABLE_COPY_FLAGS", frozenset({"uses_urgency"})), \
         patch("qualume.learn.run_mining_pass.mine_boolean_copy_feature", return_value=None), \
         patch("qualume.learn.run_mining_pass.upsert_external_insight") as mock_upsert, \
         patch("qualume.learn.run_mining_pass.retire_stale_external_insights", return_value=0):

        result = run_mining_pass(applies_to_workflow_type="tiktok_video")

    mock_upsert.assert_not_called()
    assert result["tested"] == 0


def test_run_mining_pass_retires_identities_that_no_longer_clear_significance():
    with patch("qualume.learn.run_mining_pass.MINEABLE_COLUMNS", frozenset({"hook_type"})), \
         patch("qualume.learn.run_mining_pass.MINEABLE_COPY_FLAGS", frozenset()), \
         patch("qualume.learn.run_mining_pass.mine_categorical_feature") as mock_mine_dna, \
         patch("qualume.learn.run_mining_pass.upsert_external_insight"), \
         patch("qualume.learn.run_mining_pass.retire_stale_external_insights") as mock_retire:

        mock_mine_dna.return_value = [_dna_result("question", p_value=0.01)]
        mock_retire.return_value = 3

        result = run_mining_pass(applies_to_workflow_type="tiktok_video")

    mock_retire.assert_called_once()
    call_kwargs = mock_retire.call_args.kwargs
    assert call_kwargs["applies_to_workflow_type"] == "tiktok_video"
    assert call_kwargs["still_significant_identities"] == [
        {"mined_from": "external_creative_dna.hook_type", "value": "question"}
    ]
    assert result["retired"] == 3


def test_run_mining_pass_isolates_one_column_failure_from_the_rest():
    """One mineable column raising (e.g. a transient DB hiccup) must not
    abort the whole pass — same per-item isolation discipline as
    orchestrate.run_ingestion_cycle's stages."""
    with patch("qualume.learn.run_mining_pass.MINEABLE_COLUMNS", frozenset({"hook_type", "format_tag"})), \
         patch("qualume.learn.run_mining_pass.MINEABLE_COPY_FLAGS", frozenset()), \
         patch("qualume.learn.run_mining_pass.mine_categorical_feature") as mock_mine_dna, \
         patch("qualume.learn.run_mining_pass.upsert_external_insight"), \
         patch("qualume.learn.run_mining_pass.retire_stale_external_insights", return_value=0):

        def side_effect(column, **kwargs):
            if column == "hook_type":
                raise RuntimeError("simulated DB hiccup")
            return [_dna_result("talking_head", p_value=0.02)]
        mock_mine_dna.side_effect = side_effect

        result = run_mining_pass(applies_to_workflow_type="tiktok_video")

    assert result["errors"] == ["hook_type: RuntimeError: simulated DB hiccup"]
    assert result["proposed_or_refreshed"] == 1  # format_tag still succeeded


def test_significance_threshold_is_the_conventional_0_05():
    assert SIGNIFICANCE_THRESHOLD == 0.05
