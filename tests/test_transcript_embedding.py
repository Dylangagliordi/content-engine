"""Tests for transcript embedding — written FIRST per this project's TDD
discipline.

Per this session's explicit design discussion: embeddings + clustering is
the genuinely unbiased complement to the declared-taxonomy mining already
built (mine_categorical_feature, mine_boolean_copy_feature) — no human
decides upfront what's worth checking for, videos that are structurally
similar in MEANING group together on their own.

embed_video_text() is deliberately format-agnostic (plain text in, vector
out, no DB dependency) — same discipline as copy_classifier.classify_copy.
embed_and_store() is the DB-aware wrapper, mirrors
copy_classifier.classify_and_store_external's shape: pull a video's
already-stored transcript/hook_text, call the embedder, persist a row.

Mocks the real Voyage API call (network, real cost) rather than hitting
it in every test run — one live integration test (not here, run manually
per CLAUDE.md's "verify live before trusting" discipline) already
confirmed voyage-3.5 returns real 1024-dim vectors for real text.
"""
import uuid
from unittest.mock import MagicMock, patch

import pytest

from qualume.db.connection import get_connection
from qualume.db.watchlist import add_watched_account
from qualume.learn.transcript_embedding import (
    EMBEDDING_MODEL,
    build_embeddable_text,
    embed_and_store,
    embed_video_text,
)


def _cleanup_account(handle: str):
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM transcript_embeddings WHERE external_content_item_id IN "
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


def _insert_item_with_transcript(account_id, hook_text=None, full_transcript=None):
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
            if hook_text or full_transcript:
                import json
                cur.execute(
                    """INSERT INTO external_creative_dna
                       (external_content_item_id, hook_text, full_transcript,
                        analysis_model, analysis_prompt_version)
                       VALUES (%s, %s, %s, 'test-model', 'v6')""",
                    (item_id, hook_text, json.dumps(full_transcript) if full_transcript else None),
                )
        conn.commit()
        return item_id
    finally:
        conn.close()


@pytest.fixture
def fake_account():
    handle = f"test_embed_{uuid.uuid4().hex[:8]}"
    _cleanup_account(handle)
    account_id = add_watched_account("tiktok", handle)
    yield account_id, handle
    _cleanup_account(handle)


def test_build_embeddable_text_joins_hook_and_transcript():
    text = build_embeddable_text(
        hook_text="Stop doing this to your skin",
        full_transcript=[{"text": "First sentence."}, {"text": "Second sentence."}],
    )
    assert "Stop doing this to your skin" in text
    assert "First sentence." in text
    assert "Second sentence." in text


def test_build_embeddable_text_handles_missing_hook():
    text = build_embeddable_text(hook_text=None, full_transcript=[{"text": "Only this."}])
    assert "Only this." in text


def test_build_embeddable_text_raises_on_nothing_to_embed():
    with pytest.raises(ValueError, match="nothing to embed"):
        build_embeddable_text(hook_text=None, full_transcript=None)
    with pytest.raises(ValueError, match="nothing to embed"):
        build_embeddable_text(hook_text="", full_transcript=[])


def test_embed_video_text_calls_voyage_with_the_configured_model():
    fake_client = MagicMock()
    fake_client.embed.return_value = MagicMock(embeddings=[[0.1, 0.2, 0.3]])

    with patch("qualume.learn.transcript_embedding._get_voyage_client", return_value=fake_client):
        vector = embed_video_text("a real transcript about skincare")

    assert vector == [0.1, 0.2, 0.3]
    fake_client.embed.assert_called_once()
    call_kwargs = fake_client.embed.call_args.kwargs
    assert call_kwargs["model"] == EMBEDDING_MODEL


def test_embed_and_store_persists_a_row(fake_account):
    account_id, _ = fake_account
    item_id = _insert_item_with_transcript(
        account_id, hook_text="A real hook", full_transcript=[{"text": "A real transcript."}]
    )

    fake_client = MagicMock()
    fake_client.embed.return_value = MagicMock(embeddings=[[0.5] * 1024])

    with patch("qualume.learn.transcript_embedding._get_voyage_client", return_value=fake_client):
        row_id = embed_and_store(item_id)

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT embedding, embedding_model, source_text_hash FROM transcript_embeddings "
                "WHERE id = %s",
                (row_id,),
            )
            embedding, model, text_hash = cur.fetchone()
    finally:
        conn.close()
    assert len(embedding) == 1024
    assert model == EMBEDDING_MODEL
    assert len(text_hash) == 64  # sha256 hex digest


def test_embed_and_store_skips_the_real_api_call_when_text_is_unchanged(fake_account):
    """Real cost/latency discipline, same shape as find_pending_annotation
    only touching rows genuinely missing output: re-embedding a video
    whose underlying transcript text hasn't changed since the last embed
    should NOT make a new Voyage API call."""
    account_id, _ = fake_account
    item_id = _insert_item_with_transcript(
        account_id, hook_text="A real hook", full_transcript=[{"text": "A real transcript."}]
    )

    fake_client = MagicMock()
    fake_client.embed.return_value = MagicMock(embeddings=[[0.5] * 1024])

    with patch("qualume.learn.transcript_embedding._get_voyage_client", return_value=fake_client):
        first_id = embed_and_store(item_id)
        second_id = embed_and_store(item_id)  # same text, should skip the real call

    assert first_id == second_id
    assert fake_client.embed.call_count == 1


def test_embed_and_store_re_embeds_when_text_actually_changed(fake_account):
    account_id, _ = fake_account
    item_id = _insert_item_with_transcript(
        account_id, hook_text="Original hook", full_transcript=[{"text": "Original transcript."}]
    )

    fake_client = MagicMock()
    fake_client.embed.return_value = MagicMock(embeddings=[[0.5] * 1024])

    with patch("qualume.learn.transcript_embedding._get_voyage_client", return_value=fake_client):
        embed_and_store(item_id)

    # Simulate a re-annotation with DIFFERENT transcript text (same real
    # shape as a video getting re-annotated at a newer prompt version).
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            import json
            cur.execute(
                """INSERT INTO external_creative_dna
                   (external_content_item_id, hook_text, full_transcript,
                    analysis_model, analysis_prompt_version, analyzed_at)
                   VALUES (%s, %s, %s, 'test-model', 'v7', now() + interval '1 hour')""",
                (item_id, "Completely different hook", json.dumps([{"text": "Different transcript."}])),
            )
        conn.commit()
    finally:
        conn.close()

    with patch("qualume.learn.transcript_embedding._get_voyage_client", return_value=fake_client):
        embed_and_store(item_id)

    assert fake_client.embed.call_count == 2


def test_embed_and_store_raises_on_item_with_no_creative_dna(fake_account):
    account_id, _ = fake_account
    item_id = _insert_item_with_transcript(account_id)  # no DNA row at all

    with pytest.raises(ValueError, match="no external_creative_dna row"):
        embed_and_store(item_id)
