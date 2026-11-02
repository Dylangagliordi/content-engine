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
         patch("qualume.learn.run_mining_pass.MINEABLE_ARRAY_COLUMNS", frozenset()), \
         patch("qualume.learn.run_mining_pass.MINEABLE_NESTED_FIELDS", frozenset()), \
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
         patch("qualume.learn.run_mining_pass.MINEABLE_ARRAY_COLUMNS", frozenset()), \
         patch("qualume.learn.run_mining_pass.MINEABLE_NESTED_FIELDS", frozenset()), \
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
         patch("qualume.learn.run_mining_pass.MINEABLE_ARRAY_COLUMNS", frozenset()), \
         patch("qualume.learn.run_mining_pass.MINEABLE_NESTED_FIELDS", frozenset()), \
         patch("qualume.learn.run_mining_pass.mine_boolean_copy_feature", return_value=None), \
         patch("qualume.learn.run_mining_pass.upsert_external_insight") as mock_upsert, \
         patch("qualume.learn.run_mining_pass.retire_stale_external_insights", return_value=0):

        result = run_mining_pass(applies_to_workflow_type="tiktok_video")

    mock_upsert.assert_not_called()
    assert result["tested"] == 0


def test_run_mining_pass_retires_identities_that_no_longer_clear_significance():
    with patch("qualume.learn.run_mining_pass.MINEABLE_COLUMNS", frozenset({"hook_type"})), \
         patch("qualume.learn.run_mining_pass.MINEABLE_COPY_FLAGS", frozenset()), \
         patch("qualume.learn.run_mining_pass.MINEABLE_ARRAY_COLUMNS", frozenset()), \
         patch("qualume.learn.run_mining_pass.MINEABLE_NESTED_FIELDS", frozenset()), \
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
         patch("qualume.learn.run_mining_pass.MINEABLE_ARRAY_COLUMNS", frozenset()), \
         patch("qualume.learn.run_mining_pass.MINEABLE_NESTED_FIELDS", frozenset()), \
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


def _array_result(value, p_value, n_with=10, n_without=10, ratio_with=1.3, ratio_without=0.9):
    return {
        "value": value, "n_with": n_with, "n_without": n_without,
        "median_ratio_with": ratio_with, "median_ratio_without": ratio_without,
        "p_value": p_value,
    }


def test_run_mining_pass_covers_every_mineable_array_column_and_nested_field():
    """Real gap closed this session: music_mood_v2 (array), presenters.
    relatable_archetype (nested field) and friends were captured in the
    DB but never tested for correlation with performance. This pass must
    now cover them too, same discovery-then-test shape as DNA columns."""
    with patch("qualume.learn.run_mining_pass.MINEABLE_COLUMNS", frozenset()), \
         patch("qualume.learn.run_mining_pass.MINEABLE_COPY_FLAGS", frozenset()), \
         patch("qualume.learn.run_mining_pass.MINEABLE_ARRAY_COLUMNS", frozenset({"music_mood_v2"})), \
         patch("qualume.learn.run_mining_pass.MINEABLE_NESTED_FIELDS",
               frozenset({("presenters", "relatable_archetype")})), \
         patch("qualume.learn.run_mining_pass.discover_array_values",
               return_value={"upbeat", "calm"}), \
         patch("qualume.learn.run_mining_pass.discover_nested_field_values",
               return_value={"mom", "nurse"}), \
         patch("qualume.learn.run_mining_pass.mine_array_value") as mock_mine_array, \
         patch("qualume.learn.run_mining_pass.mine_nested_field_value") as mock_mine_nested, \
         patch("qualume.learn.run_mining_pass.upsert_external_insight"), \
         patch("qualume.learn.run_mining_pass.retire_stale_external_insights", return_value=0):

        mock_mine_array.return_value = _array_result("upbeat", p_value=0.01)
        mock_mine_nested.return_value = _array_result("mom", p_value=0.02)

        result = run_mining_pass(applies_to_workflow_type="tiktok_video")

    array_values_called = {c.args[1] for c in mock_mine_array.call_args_list}
    assert array_values_called == {"upbeat", "calm"}
    nested_values_called = {c.args[2] for c in mock_mine_nested.call_args_list}
    assert nested_values_called == {"mom", "nurse"}
    assert result["tested"] == 4  # 2 array values + 2 nested values


def test_run_mining_pass_handles_none_return_from_array_and_nested_mining():
    with patch("qualume.learn.run_mining_pass.MINEABLE_COLUMNS", frozenset()), \
         patch("qualume.learn.run_mining_pass.MINEABLE_COPY_FLAGS", frozenset()), \
         patch("qualume.learn.run_mining_pass.MINEABLE_ARRAY_COLUMNS", frozenset({"music_mood_v2"})), \
         patch("qualume.learn.run_mining_pass.MINEABLE_NESTED_FIELDS", frozenset()), \
         patch("qualume.learn.run_mining_pass.discover_array_values", return_value={"upbeat"}), \
         patch("qualume.learn.run_mining_pass.mine_array_value", return_value=None), \
         patch("qualume.learn.run_mining_pass.upsert_external_insight") as mock_upsert, \
         patch("qualume.learn.run_mining_pass.retire_stale_external_insights", return_value=0):

        result = run_mining_pass(applies_to_workflow_type="tiktok_video")

    mock_upsert.assert_not_called()
    assert result["tested"] == 0


def test_run_mining_pass_upserts_significant_array_and_nested_findings():
    with patch("qualume.learn.run_mining_pass.MINEABLE_COLUMNS", frozenset()), \
         patch("qualume.learn.run_mining_pass.MINEABLE_COPY_FLAGS", frozenset()), \
         patch("qualume.learn.run_mining_pass.MINEABLE_ARRAY_COLUMNS", frozenset({"music_mood_v2"})), \
         patch("qualume.learn.run_mining_pass.MINEABLE_NESTED_FIELDS",
               frozenset({("presenters", "relatable_archetype")})), \
         patch("qualume.learn.run_mining_pass.discover_array_values", return_value={"upbeat"}), \
         patch("qualume.learn.run_mining_pass.discover_nested_field_values", return_value={"mom"}), \
         patch("qualume.learn.run_mining_pass.mine_array_value",
               return_value=_array_result("upbeat", p_value=0.01)), \
         patch("qualume.learn.run_mining_pass.mine_nested_field_value",
               return_value=_array_result("mom", p_value=0.70)), \
         patch("qualume.learn.run_mining_pass.upsert_external_insight") as mock_upsert, \
         patch("qualume.learn.run_mining_pass.retire_stale_external_insights", return_value=0):

        result = run_mining_pass(applies_to_workflow_type="tiktok_video")

    assert mock_upsert.call_count == 1  # only the significant music_mood_v2 finding
    call_kwargs = mock_upsert.call_args.kwargs
    assert "music_mood_v2" in call_kwargs["identity"]["mined_from"]
    assert result["proposed_or_refreshed"] == 1
    assert result["skipped_not_significant"] == 1


def test_run_mining_pass_isolates_one_array_discovery_failure_from_the_rest():
    with patch("qualume.learn.run_mining_pass.MINEABLE_COLUMNS", frozenset()), \
         patch("qualume.learn.run_mining_pass.MINEABLE_COPY_FLAGS", frozenset()), \
         patch("qualume.learn.run_mining_pass.MINEABLE_ARRAY_COLUMNS",
               frozenset({"music_mood_v2", "script_structure_tags"})), \
         patch("qualume.learn.run_mining_pass.MINEABLE_NESTED_FIELDS", frozenset()), \
         patch("qualume.learn.run_mining_pass.discover_array_values") as mock_discover, \
         patch("qualume.learn.run_mining_pass.mine_array_value",
               return_value=_array_result("x", p_value=0.02)), \
         patch("qualume.learn.run_mining_pass.upsert_external_insight"), \
         patch("qualume.learn.run_mining_pass.retire_stale_external_insights", return_value=0):

        def side_effect(column, **kwargs):
            if column == "music_mood_v2":
                raise RuntimeError("simulated DB hiccup")
            return {"tag_a"}
        mock_discover.side_effect = side_effect

        result = run_mining_pass(applies_to_workflow_type="tiktok_video")

    assert any("music_mood_v2" in e for e in result["errors"])
    assert result["proposed_or_refreshed"] == 1  # script_structure_tags still succeeded


def test_run_mining_pass_applies_benjamini_hochberg_across_the_whole_batch():
    """Real proof BH is actually wired in, not coincidentally passing
    because other tests use extreme p-values: a moderately significant
    p=0.04 result that would clear a FLAT p<0.05 threshold on its own
    must be REJECTED once batched with 19 other near-random p-values
    (same real shape as this project's own 70+-test-per-cycle reality) —
    BH's corrected threshold for rank k=1 of 20 tests at FDR=0.05 is
    (1/20)*0.05 = 0.0025, far below 0.04."""
    with patch("qualume.learn.run_mining_pass.MINEABLE_COLUMNS", frozenset({"hook_type"})), \
         patch("qualume.learn.run_mining_pass.MINEABLE_COPY_FLAGS", frozenset()), \
         patch("qualume.learn.run_mining_pass.MINEABLE_ARRAY_COLUMNS", frozenset()), \
         patch("qualume.learn.run_mining_pass.MINEABLE_NESTED_FIELDS", frozenset()), \
         patch("qualume.learn.run_mining_pass.mine_categorical_feature") as mock_mine_dna, \
         patch("qualume.learn.run_mining_pass.upsert_external_insight") as mock_upsert, \
         patch("qualume.learn.run_mining_pass.retire_stale_external_insights", return_value=0):

        # 19 near-random p-values + 1 borderline p=0.04 (the "finding").
        import random
        random.seed(99)
        results = [_dna_result(f"val_{i}", p_value=random.uniform(0.2, 0.95)) for i in range(19)]
        results.append(_dna_result("borderline", p_value=0.04))
        mock_mine_dna.return_value = results

        result = run_mining_pass(applies_to_workflow_type="tiktok_video")

    # Flat p<0.05 would have upserted "borderline" alone; BH correction
    # across this batch of 20 rejects it too — proves real correction,
    # not a no-op that happens to match a flat threshold.
    mock_upsert.assert_not_called()
    assert result["tested"] == 20
    assert result["skipped_not_significant"] == 20


def test_run_mining_pass_threads_niche_through_every_mine_call_and_upsert_retire(prefix=None):
    """Real fix (2026-11-02): niche must reach EVERY underlying mine_*
    call (so the test itself is scoped to one niche's videos) AND the
    upsert/retire calls (so the resulting insight is tagged and future
    retirement stays scoped to this niche) — the whole point of the fix
    is useless if niche stops at run_mining_pass's own signature and
    never actually reaches the real DB-touching calls underneath."""
    with patch("qualume.learn.run_mining_pass.MINEABLE_COLUMNS", frozenset({"hook_type"})), \
         patch("qualume.learn.run_mining_pass.MINEABLE_COPY_FLAGS", frozenset({"creates_urgency"})), \
         patch("qualume.learn.run_mining_pass.MINEABLE_ARRAY_COLUMNS", frozenset({"music_mood_v2"})), \
         patch("qualume.learn.run_mining_pass.MINEABLE_NESTED_FIELDS",
               frozenset({("presenters", "relatable_archetype")})), \
         patch("qualume.learn.run_mining_pass.mine_categorical_feature") as mock_mine_dna, \
         patch("qualume.learn.run_mining_pass.mine_boolean_copy_feature") as mock_mine_copy, \
         patch("qualume.learn.run_mining_pass.discover_array_values", return_value={"upbeat"}) as mock_disc_arr, \
         patch("qualume.learn.run_mining_pass.discover_nested_field_values", return_value={"mom"}) as mock_disc_nested, \
         patch("qualume.learn.run_mining_pass.mine_array_value") as mock_mine_array, \
         patch("qualume.learn.run_mining_pass.mine_nested_field_value") as mock_mine_nested, \
         patch("qualume.learn.run_mining_pass.upsert_external_insight") as mock_upsert, \
         patch("qualume.learn.run_mining_pass.retire_stale_external_insights", return_value=0) as mock_retire:

        mock_mine_dna.return_value = [_dna_result("question", p_value=0.01)]
        mock_mine_copy.return_value = _copy_result("creates_urgency", p_value=0.01)
        mock_mine_array.return_value = _array_result("upbeat", p_value=0.01)
        mock_mine_nested.return_value = _array_result("mom", p_value=0.01)

        run_mining_pass(applies_to_workflow_type="tiktok_video", niche="zz_test_niche_thread")

    mock_mine_dna.assert_called_once_with("hook_type", min_group_size=5, niche="zz_test_niche_thread")
    mock_mine_copy.assert_called_once_with("creates_urgency", min_group_size=5, niche="zz_test_niche_thread")
    mock_disc_arr.assert_called_once_with("music_mood_v2", niche="zz_test_niche_thread")
    mock_disc_nested.assert_called_once_with("presenters", "relatable_archetype", niche="zz_test_niche_thread")
    mock_mine_array.assert_called_once_with("music_mood_v2", "upbeat", min_group_size=5, niche="zz_test_niche_thread")
    mock_mine_nested.assert_called_once_with(
        "presenters", "relatable_archetype", "mom", min_group_size=5, niche="zz_test_niche_thread"
    )
    assert mock_upsert.call_count == 4
    for call in mock_upsert.call_args_list:
        assert call.kwargs["niche"] == "zz_test_niche_thread"
    mock_retire.assert_called_once()
    assert mock_retire.call_args.kwargs["niche"] == "zz_test_niche_thread"
