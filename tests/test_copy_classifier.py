"""Tests for the copy-technique classifier — written FIRST per this
project's TDD discipline.

Claude-only (NOT the Gemini video-watching exception): a text classification
pass over copy that's ALREADY been extracted somewhere else — full_transcript/
on_screen_text/hook_text for a video (organic OR paid/VSL), or just
on-screen/body copy text for a static ad or static organic post. classify_copy()
itself is format-agnostic by design (plain text/list args, no DB, no
knowledge of which table the copy came from) so it works unchanged for
external video data (built/tested here, real data exists today), owned
content, or statics once that ingestion exists later — same
"capture the shape before the volume" discipline as learn/replication.py
and learn/validation.py.

v2: classify_copy returns an ORDERED technique_sequence (see
contracts.CopyAnalysisV2/TechniqueOccurrence) instead of v1's 22
independent booleans. derive_booleans_from_sequence converts that
ordered list back into the 22 real boolean columns for storage/mining —
tested directly here since it's the single source of truth for both.
"""
import json
import uuid
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from qualume.annotate.copy_classifier import (
    ANALYSIS_PROMPT_VERSION,
    ANNOTATION_PROMPT,
    classify_copy,
    classify_and_store_external,
    derive_booleans_from_sequence,
)
from qualume.models.contracts import CopyAnalysisV2, TechniqueOccurrence, copy_technique_names
from qualume.db.connection import get_connection
from qualume.db.watchlist import add_watched_account


def _good_sequence(*techniques):
    """Build a CopyAnalysisV2 with the given techniques, in order, each
    with a throwaway evidence quote."""
    if not techniques:
        techniques = ("hook_curiosity_open_loop", "mentions_customer_problem",
                      "explains_mechanism", "cta_direct_address")
    return CopyAnalysisV2(
        technique_sequence=[
            TechniqueOccurrence(technique=t, evidence_quote="evidence") for t in techniques
        ]
    )


def test_classify_copy_calls_llm_with_all_available_text(monkeypatch):
    """Every text source that's actually present (hook_text, transcript,
    on-screen text, caption) must reach the prompt — this is what makes
    it format-agnostic: a static with only on_screen_text, or a video
    with everything, both just pass what they have."""
    captured = {}

    def fake_pick_llm(level):
        mock_base = MagicMock()

        def fake_with_structured_output(schema):
            def fake_invoke(messages):
                captured["messages"] = messages
                return _good_sequence()
            llm = MagicMock()
            llm.invoke = fake_invoke
            return llm
        mock_base.with_structured_output = fake_with_structured_output
        return mock_base

    monkeypatch.setattr("qualume.annotate.copy_classifier.pick_llm", fake_pick_llm)

    classify_copy(
        hook_text="Still drained by 3pm?",
        full_transcript=[{"start_sec": 0.0, "text": "Still drained by 3pm?"}],
        on_screen_text=[{"text": "DAILY IMMUNE SUPPORT", "timestamp_sec": 0.0}],
        caption="link in bio #wellness",
    )

    prompt_text = str(captured["messages"])
    assert "Still drained by 3pm?" in prompt_text
    assert "DAILY IMMUNE SUPPORT" in prompt_text
    assert "link in bio #wellness" in prompt_text


def test_classify_copy_works_with_only_on_screen_text(monkeypatch):
    """The static-ad/static-organic case: no transcript, no hook_text
    (no speech at all) — must still run on whatever text IS there, not
    require every field."""
    def fake_pick_llm(level):
        mock_base = MagicMock()
        def fake_with_structured_output(schema):
            llm = MagicMock()
            llm.invoke = lambda messages: _good_sequence()
            return llm
        mock_base.with_structured_output = fake_with_structured_output
        return mock_base
    monkeypatch.setattr("qualume.annotate.copy_classifier.pick_llm", fake_pick_llm)

    result = classify_copy(
        hook_text=None, full_transcript=None,
        on_screen_text=[{"text": "3 SUPPLEMENTS EVERY NEW MOM NEEDS", "timestamp_sec": None}],
        caption=None,
    )
    assert result.annotation is not None


def test_classify_copy_raises_when_absolutely_no_text_available():
    """Nothing to classify at all - fail loud rather than silently
    returning an all-False/all-None result that looks like a real
    classification of empty input."""
    with pytest.raises(ValueError):
        classify_copy(hook_text=None, full_transcript=None, on_screen_text=None, caption=None)


def test_classify_copy_returns_validation_error_gracefully(monkeypatch):
    """A malformed/partial LLM response shouldn't crash the whole call —
    same graceful-degradation discipline as gemini_watcher.annotate_video."""
    def fake_pick_llm(level):
        mock_base = MagicMock()
        def fake_with_structured_output(schema):
            llm = MagicMock()
            def fake_invoke(messages):
                raise ValidationError.from_exception_data("CopyAnalysisV2", [])
            llm.invoke = fake_invoke
            return llm
        mock_base.with_structured_output = fake_with_structured_output
        return mock_base
    monkeypatch.setattr("qualume.annotate.copy_classifier.pick_llm", fake_pick_llm)

    result = classify_copy(hook_text="some hook", full_transcript=None, on_screen_text=None, caption=None)
    assert result.annotation is None
    assert result.validation_error is not None


# --- derive_booleans_from_sequence -----------------------------------


def test_derive_booleans_marks_present_techniques_true():
    annotation = _good_sequence("hook_pain_callout", "explains_mechanism")
    flags = derive_booleans_from_sequence(annotation)
    assert flags["hook_pain_callout"] is True
    assert flags["explains_mechanism"] is True


def test_derive_booleans_marks_absent_techniques_false_not_none():
    """v2 behavior change from v1: absence from the sequence is always
    False, never None - there's no 'couldn't tell' state in v2 (see
    CopyAnalysisV2's docstring for why this was accepted)."""
    annotation = _good_sequence("hook_pain_callout")
    flags = derive_booleans_from_sequence(annotation)
    assert flags["uses_social_proof"] is False
    assert flags["creates_urgency"] is False


def test_derive_booleans_covers_all_22_real_techniques():
    annotation = _good_sequence()
    flags = derive_booleans_from_sequence(annotation)
    assert set(flags.keys()) == set(copy_technique_names())
    assert len(flags) == 22


def test_derive_booleans_true_for_a_technique_that_recurs_twice():
    """A technique appearing twice in the sequence (e.g. urgency early
    AND at the CTA) is still just True once as a boolean - the sequence
    itself (not the boolean) is where the recurrence is visible."""
    annotation = _good_sequence("creates_urgency", "mentions_customer_problem", "creates_urgency")
    flags = derive_booleans_from_sequence(annotation)
    assert flags["creates_urgency"] is True


def test_derive_booleans_empty_sequence_is_all_false():
    annotation = _good_sequence()
    annotation.technique_sequence = []
    flags = derive_booleans_from_sequence(annotation)
    assert all(v is False for v in flags.values())


# --- classify_and_store_external (DB-backed, real external video data) -----


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
                "DELETE FROM external_content_items WHERE watched_account_id IN "
                "(SELECT id FROM watched_accounts WHERE handle = %s)",
                (handle,),
            )
            cur.execute("DELETE FROM watched_accounts WHERE handle = %s", (handle,))
        conn.commit()
    finally:
        conn.close()


def _insert_item_with_dna(account_id, caption="a caption", hook_text="a hook",
                           full_transcript=None, on_screen_text=None):
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO external_content_items
                   (watched_account_id, platform, platform_video_id, url, caption)
                   VALUES (%s, 'tiktok', %s, 'https://example.com/fake', %s)
                   RETURNING id""",
                (account_id, uuid.uuid4().hex, caption),
            )
            item_id = cur.fetchone()[0]
            cur.execute(
                """INSERT INTO external_creative_dna
                   (external_content_item_id, hook_text, full_transcript, on_screen_text,
                    analysis_model, analysis_prompt_version)
                   VALUES (%s, %s, %s, %s, 'test-model', 'v5')""",
                (item_id, hook_text, json.dumps(full_transcript), json.dumps(on_screen_text)),
            )
        conn.commit()
        return item_id
    finally:
        conn.close()


@pytest.fixture
def fake_account():
    handle = f"test_copy_{uuid.uuid4().hex[:8]}"
    _cleanup_account(handle)
    account_id = add_watched_account("tiktok", handle)
    yield account_id, handle
    _cleanup_account(handle)


def test_classify_and_store_external_pulls_real_row_and_stores_result(fake_account, monkeypatch):
    account_id, _ = fake_account
    item_id = _insert_item_with_dna(
        account_id, hook_text="Still drained by 3pm?",
        full_transcript=[{"start_sec": 0.0, "text": "Still drained by 3pm?"}],
        on_screen_text=[{"text": "DAILY IMMUNE", "timestamp_sec": 0.0}],
    )

    def fake_classify_copy(**kwargs):
        from qualume.annotate.copy_classifier import ClassificationResult
        ann = _good_sequence("hook_curiosity_open_loop", "cta_direct_address")
        return ClassificationResult(
            annotation=ann, raw_response=ann.model_dump(), validation_error=None,
        )
    monkeypatch.setattr("qualume.annotate.copy_classifier.classify_copy", fake_classify_copy)

    analysis_id = classify_and_store_external(item_id)

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT hook_curiosity_open_loop, cta_direct_address, analysis_prompt_version, "
                "technique_sequence "
                "FROM external_copy_analysis WHERE id = %s",
                (analysis_id,),
            )
            row = cur.fetchone()
    finally:
        conn.close()
    assert row[0] is True
    assert row[1] is True
    assert row[2] == ANALYSIS_PROMPT_VERSION
    assert len(row[3]) == 2  # the real ordered sequence was stored, not just derived booleans


def test_classify_and_store_external_raises_for_unknown_item():
    with pytest.raises(ValueError):
        classify_and_store_external(str(uuid.uuid4()))


def test_classify_and_store_external_raises_when_item_has_no_copy_annotation_yet(fake_account):
    """A video that hasn't been through gemini_watcher yet has no
    external_creative_dna row at all - nothing to classify from."""
    account_id, _ = fake_account
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO external_content_items
                   (watched_account_id, platform, platform_video_id, url, caption)
                   VALUES (%s, 'tiktok', %s, 'https://example.com/fake', 'cap')
                   RETURNING id""",
                (account_id, uuid.uuid4().hex),
            )
            item_id = cur.fetchone()[0]
        conn.commit()
    finally:
        conn.close()

    with pytest.raises(ValueError):
        classify_and_store_external(item_id)
