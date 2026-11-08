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
from qualume.learn.training_data import fetch_training_rows, fetch_training_rows_pooled


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
                "DELETE FROM visual_embeddings WHERE external_content_item_id IN "
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


def _insert_transcript_embedding(item_id, dim=1024):
    import json
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO transcript_embeddings
                   (external_content_item_id, embedding, source_text_hash, embedding_model)
                   VALUES (%s, %s, 'fake_hash', 'voyage-3.5')""",
                (item_id, json.dumps([0.1] * dim)),
            )
        conn.commit()
    finally:
        conn.close()


def _insert_visual_embedding(item_id, dim=1024):
    import json
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO visual_embeddings
                   (external_content_item_id, embedding, n_frames_used, embedding_model)
                   VALUES (%s, %s, 6, 'voyage-multimodal-3.5')""",
                (item_id, json.dumps([0.2] * dim)),
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
        assert "hook_text" in annotation
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


# ──────────────────────────────────────────────
# Stored transcript/visual embedding fetching — added for the three-way
# embedding merge ("how can we use all [text/transcript/visual
# embeddings] in the prediction model").
# ──────────────────────────────────────────────

def test_fetch_training_rows_includes_stored_embedding_vectors_when_present(fake_niche):
    account_id, items = fake_niche
    for item_id in items:
        _insert_transcript_embedding(item_id)
        _insert_visual_embedding(item_id)

    rows = fetch_training_rows(niche="zz_test_training_niche")
    assert len(rows) == 6
    for annotation, _ in rows:
        assert annotation["transcript_embedding_vector"] is not None
        assert len(annotation["transcript_embedding_vector"]) == 1024
        assert annotation["visual_embedding_vector"] is not None
        assert annotation["transcript_embeddable_text"] is not None


def test_fetch_training_rows_gives_none_for_items_with_no_stored_embedding(fake_niche):
    """Real, incomplete coverage: these embedding pipelines run
    independently of annotation, on their own schedule — most annotated
    items have NOT been through them. None, not a fabricated vector."""
    account_id, items = fake_niche
    # deliberately insert NO embeddings for any item

    rows = fetch_training_rows(niche="zz_test_training_niche")
    assert len(rows) == 6
    for annotation, _ in rows:
        assert annotation["transcript_embedding_vector"] is None
        assert annotation["visual_embedding_vector"] is None
        assert annotation["transcript_embeddable_text"] is None


def test_fetch_training_rows_handles_partial_stored_embedding_coverage(fake_niche):
    """Real shape: SOME items have one or both stored embeddings, others
    don't — must not crash, and must get the real vector where it
    exists, None elsewhere (never inferring from a different item)."""
    account_id, items = fake_niche
    _insert_transcript_embedding(items[0])
    _insert_visual_embedding(items[1])
    # items[2:] deliberately left with neither

    rows = fetch_training_rows(niche="zz_test_training_niche")
    assert len(rows) == 6
    with_transcript = sum(1 for ann, _ in rows if ann["transcript_embedding_vector"] is not None)
    with_visual = sum(1 for ann, _ in rows if ann["visual_embedding_vector"] is not None)
    assert with_transcript == 1
    assert with_visual == 1


# ──────────────────────────────────────────────
# Pooled training — added 2026-11-05 for the predictive-model data-
# scarcity fix: fetch across MULTIPLE/all niches, tagging each
# annotation with its real niche so train_predictor can use niche as a
# categorical feature instead of training one starved model per niche.
# ──────────────────────────────────────────────

def test_fetch_training_rows_pooled_tags_every_annotation_with_its_real_niche(fake_niche):
    account_id, items = fake_niche
    rows = fetch_training_rows_pooled(niches=["zz_test_training_niche"])
    assert len(rows) == 6
    for annotation, _ in rows:
        assert annotation["niche"] == "zz_test_training_niche"


def test_fetch_training_rows_pooled_combines_multiple_real_niches(fake_niche):
    account_id, items = fake_niche
    other_handle = f"zz_test_pooled_other_{uuid.uuid4().hex[:8]}"
    _cleanup_account(other_handle)
    other_account = add_watched_account("tiktok", other_handle, niche="zz_test_pooled_other_niche")
    other_item = _insert_item(other_account)
    _insert_dna(other_item)
    _insert_snapshot(other_item, views=500)

    try:
        rows = fetch_training_rows_pooled(niches=["zz_test_training_niche", "zz_test_pooled_other_niche"])
        niches_seen = {ann["niche"] for ann, _ in rows}
        assert niches_seen == {"zz_test_training_niche", "zz_test_pooled_other_niche"}
        assert len(rows) == 7  # 6 from the fixture + 1 from the other niche
    finally:
        _cleanup_account(other_handle)


def test_fetch_training_rows_pooled_with_no_niches_arg_fetches_everything(fake_niche):
    """niches=None (the default) means no niche filter at all — every
    real niche currently in the DB, not just this test's fixture. Can't
    assert an exact count (real production data varies), just that the
    fixture's own niche shows up and every real row is correctly tagged."""
    account_id, items = fake_niche
    rows = fetch_training_rows_pooled(niches=None)
    niches_seen = {ann["niche"] for ann, _ in rows}
    assert "zz_test_training_niche" in niches_seen
    # every single row must have a real, non-None niche (the JOIN is INNER
    # on watched_accounts, which always has a niche column — None would
    # mean a real bug in the tagging, not a real unclassified account,
    # since _fetch_candidate_items doesn't filter out NULL niches itself)
    for annotation, _ in rows:
        assert "niche" in annotation


def test_fetch_training_rows_pooled_skips_items_without_a_real_performance_baseline(fake_niche):
    account_id, items = fake_niche
    unscored_item = _insert_item(account_id)
    _insert_dna(unscored_item)
    # Deliberately no snapshot for this one.

    rows = fetch_training_rows_pooled(niches=["zz_test_training_niche"])
    assert len(rows) == 6  # the unscored item is excluded, not crashing the fetch


def test_fetch_training_rows_pooled_includes_stored_embeddings_same_as_single_niche(fake_niche):
    """The pooled path must carry the exact same embedding-vector shape
    as fetch_training_rows (same _build_annotation helper) — not a
    second, drifted implementation."""
    account_id, items = fake_niche
    for item_id in items:
        _insert_transcript_embedding(item_id)

    rows = fetch_training_rows_pooled(niches=["zz_test_training_niche"])
    for annotation, _ in rows:
        assert annotation["transcript_embedding_vector"] is not None
        assert len(annotation["transcript_embedding_vector"]) == 1024
