"""Tests for fetching real (annotation, performance_ratio) training rows
from the DB, scoped per niche — the real missing piece that let
learn.prediction stay an untrained scaffold. Written FIRST per this
project's TDD discipline.

Real gap this closes (2026-11-03): prediction.py's train_predictor has
always needed (dict, float) pairs, but nothing in the codebase ever
assembled those pairs from real DB rows — it was tested only against
synthetic fixtures. Confirmed live that real data now clears the
50-row floor in 7 niches (fitness=344, women's health=335, food=149,
mom tok=132, small business=62, general health=58, menopause=57), so
this is now a real, not synthetic, training job.
"""
import uuid

import pytest

from qualume.annotate.gemini_watcher import ANALYSIS_PROMPT_VERSION as DNA_VERSION
from qualume.db.connection import get_connection
from qualume.db.watchlist import add_watched_account
from qualume.learn.training_data import fetch_training_rows


def _cleanup_account(handle: str):
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM external_performance_snapshots WHERE external_content_item_id IN "
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


def _insert_item(account_id, duration_seconds=20):
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO external_content_items
                   (watched_account_id, platform, platform_video_id, url,
                    local_video_path, duration_seconds)
                   VALUES (%s, 'tiktok', %s, 'https://example.com/fake', '/fake/path.mp4', %s)
                   RETURNING id""",
                (account_id, uuid.uuid4().hex, duration_seconds),
            )
            item_id = cur.fetchone()[0]
        conn.commit()
        return item_id
    finally:
        conn.close()


def _insert_dna(item_id, hook_type="question", format_tag="talking_head", energy_level=3):
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO external_creative_dna
                   (external_content_item_id, hook_text, analysis_model,
                    analysis_prompt_version, hook_type, format_tag, energy_level,
                    shot_breakdown, full_transcript)
                   VALUES (%s, 'a hook', 'test-model', %s, %s, %s, %s, %s, %s)""",
                (
                    item_id, DNA_VERSION, hook_type, format_tag, energy_level,
                    '[{"start_sec": 0, "end_sec": 10, "shot_type": "medium_shot"}]',
                    '[{"start_sec": 0, "text": "hello world"}]',
                ),
            )
        conn.commit()
    finally:
        conn.close()


def _insert_snapshot(item_id, views):
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO external_performance_snapshots
                   (external_content_item_id, views) VALUES (%s, %s)""",
                (item_id, views),
            )
        conn.commit()
    finally:
        conn.close()


@pytest.fixture
def fake_niche():
    handle = f"zz_test_training_{uuid.uuid4().hex[:8]}"
    _cleanup_account(handle)
    account_id = add_watched_account("tiktok", handle, niche="zz_test_training_niche")

    items = []
    for i in range(6):
        item_id = _insert_item(account_id)
        _insert_dna(item_id)
        # Real baseline shape: account median comes from these views;
        # half above, half below so there's real variance to learn from.
        _insert_snapshot(item_id, views=1000 if i % 2 == 0 else 100)
        items.append(item_id)

    yield account_id, items

    _cleanup_account(handle)


def test_fetch_training_rows_returns_annotation_and_ratio_pairs_for_a_niche(fake_niche):
    account_id, items = fake_niche

    rows = fetch_training_rows(niche="zz_test_training_niche")

    assert len(rows) == 6
    for annotation, ratio in rows:
        # Real shape prediction.build_feature_vector expects.
        assert "hook_type" in annotation
        assert "format_tag" in annotation
        assert "energy_level" in annotation
        assert "shot_breakdown" in annotation
        assert "full_transcript" in annotation
        assert "video_duration_seconds" in annotation
        assert isinstance(ratio, float)


def test_fetch_training_rows_scopes_strictly_to_the_given_niche(fake_niche):
    """A real cross-niche leak would silently corrupt every niche's
    trained model with another niche's unrelated performance scale —
    same discipline as niche-scoped mining/clustering."""
    other_handle = f"zz_test_other_niche_{uuid.uuid4().hex[:8]}"
    _cleanup_account(other_handle)
    other_account = add_watched_account("tiktok", other_handle, niche="zz_test_other_niche")
    other_item = _insert_item(other_account)
    _insert_dna(other_item)
    _insert_snapshot(other_item, views=500)

    try:
        rows = fetch_training_rows(niche="zz_test_training_niche")
        assert len(rows) == 6  # only the fixture's own niche, not the other one
    finally:
        _cleanup_account(other_handle)


def test_fetch_training_rows_skips_items_without_a_real_performance_baseline(fake_niche):
    """An item with annotation but no snapshot (or whose account has no
    median yet) can't produce a real ratio — must be skipped, not
    crash the whole fetch or fabricate a ratio."""
    account_id, items = fake_niche
    unscored_item = _insert_item(account_id)
    _insert_dna(unscored_item)
    # Deliberately no snapshot for this one.

    rows = fetch_training_rows(niche="zz_test_training_niche")

    assert len(rows) == 6  # the unscored item is excluded, not crashing the fetch
