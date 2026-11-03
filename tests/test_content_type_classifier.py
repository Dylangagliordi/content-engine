"""Tests for the content_type (selling/organic) classifier — written
FIRST per this project's TDD discipline.

Real gap this closes (migration 017): niche (what TOPIC a video is
about) and content_type (whether a video is SELLING or ORGANIC) are two
independent axes. Meta Ad Library videos are selling-by-definition and
get seeded automatically at ingest (see meta_ads_reader.parse_ad_entry)
— but a TikTok organic account can ALSO post hard-sell content (DTC
brands, TikTok Shop sellers), so content_type can't be inferred from
platform/account alone for TikTok. This needs its own real
classification pass, same Claude-only text-classification shape as
copy_classifier (no video, no new Gemini call — works off hook_text/
full_transcript/on_screen_text/caption already stored).

Deliberately a SIMPLE binary judgment, not a new multi-label taxonomy —
unlike the 22 copy techniques (which can co-occur), content_type is a
single either/or classification of the whole video.
"""
import uuid
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from qualume.annotate.content_type_classifier import (
    ANALYSIS_PROMPT_VERSION,
    classify_content_type,
    classify_and_store_content_type,
    ContentTypeResult,
)
from qualume.db.connection import get_connection
from qualume.db.watchlist import add_watched_account


def test_classify_content_type_returns_selling_for_hard_sell_copy():
    fake_response = MagicMock()
    fake_response.content_type = "selling"
    fake_response.confidence_note = "Direct CTA to buy, product name and price mentioned."

    with patch("qualume.annotate.content_type_classifier.pick_llm") as mock_pick:
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value.invoke.return_value = fake_response
        mock_pick.return_value = mock_llm

        result = classify_content_type(
            hook_text="Stop wasting money on supplements that don't work",
            full_transcript=None,
            on_screen_text=None,
            caption="Shop now - link in bio! 20% off today only",
        )

    assert result.content_type == "selling"
    assert result.validation_error is None


def test_classify_content_type_returns_organic_for_storytime_copy():
    fake_response = MagicMock()
    fake_response.content_type = "organic"
    fake_response.confidence_note = "Personal story, no product mention or CTA."

    with patch("qualume.annotate.content_type_classifier.pick_llm") as mock_pick:
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value.invoke.return_value = fake_response
        mock_pick.return_value = mock_llm

        result = classify_content_type(
            hook_text="Day 3 postpartum and I finally slept",
            full_transcript=[{"text": "just sharing my journey with you guys"}],
            on_screen_text=None,
            caption=None,
        )

    assert result.content_type == "organic"


def test_classify_content_type_raises_with_no_copy_at_all():
    with pytest.raises(ValueError):
        classify_content_type(hook_text=None, full_transcript=None, on_screen_text=None, caption=None)


def test_classify_content_type_handles_validation_error_gracefully():
    with patch("qualume.annotate.content_type_classifier.pick_llm") as mock_pick:
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value.invoke.side_effect = ValidationError.from_exception_data(
            "ContentTypeClassification", []
        )
        mock_pick.return_value = mock_llm

        result = classify_content_type(
            hook_text="some text", full_transcript=None, on_screen_text=None, caption=None
        )

    assert result.content_type is None
    assert result.validation_error is not None


# --- DB-writing wrapper, mirrors copy_classifier.classify_and_store_external


def _cleanup_account(handle: str):
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """DELETE FROM external_creative_dna WHERE external_content_item_id IN (
                       SELECT eci.id FROM external_content_items eci
                       JOIN watched_accounts wa ON wa.id = eci.watched_account_id
                       WHERE wa.handle = %s)""",
                (handle,),
            )
            cur.execute(
                """DELETE FROM external_content_items WHERE watched_account_id IN (
                       SELECT id FROM watched_accounts WHERE handle = %s)""",
                (handle,),
            )
            cur.execute("DELETE FROM watched_accounts WHERE handle = %s", (handle,))
        conn.commit()
    finally:
        conn.close()


@pytest.fixture
def fake_account():
    handle = f"test_ctype_{uuid.uuid4().hex[:8]}"
    _cleanup_account(handle)
    account_id = add_watched_account("tiktok", handle)
    yield account_id, handle
    _cleanup_account(handle)


def _insert_item_with_dna(account_id, caption="Shop our new tincture today!", hook_text="Hook text here"):
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
                   (external_content_item_id, hook_text, analysis_model, analysis_prompt_version)
                   VALUES (%s, %s, 'test-model', 'v6')""",
                (item_id, hook_text),
            )
        conn.commit()
        return item_id
    finally:
        conn.close()


def test_classify_and_store_content_type_writes_result_to_db(fake_account):
    account_id, _ = fake_account
    item_id = _insert_item_with_dna(account_id)

    fake_response = MagicMock()
    fake_response.content_type = "selling"
    fake_response.confidence_note = "Direct CTA."

    with patch("qualume.annotate.content_type_classifier.pick_llm") as mock_pick:
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value.invoke.return_value = fake_response
        mock_pick.return_value = mock_llm

        classify_and_store_content_type(item_id)

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT content_type FROM external_content_items WHERE id = %s", (item_id,))
            stored = cur.fetchone()[0]
    finally:
        conn.close()
    assert stored == "selling"


def test_classify_and_store_content_type_raises_for_missing_item():
    with pytest.raises(ValueError):
        classify_and_store_content_type("00000000-0000-0000-0000-000000000000")


def test_classify_and_store_content_type_raises_without_dna(fake_account):
    """Same precondition discipline as copy_classifier.classify_and_
    store_external — nothing to classify from without annotation yet."""
    account_id, _ = fake_account
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO external_content_items
                   (watched_account_id, platform, platform_video_id, url)
                   VALUES (%s, 'tiktok', %s, 'https://example.com/fake')
                   RETURNING id""",
                (account_id, uuid.uuid4().hex),
            )
            item_id = cur.fetchone()[0]
        conn.commit()
    finally:
        conn.close()

    with pytest.raises(ValueError):
        classify_and_store_content_type(item_id)
