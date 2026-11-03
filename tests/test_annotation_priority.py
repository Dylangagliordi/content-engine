"""Tests for sample-floor-priority annotation ordering — real fix per
2026-11-03 user decision: FIFO annotation ordering was actively working
against the goal of clearing niches below their statistical sample
floors first. A brand-new niche (0 annotated) sat behind whatever got
downloaded earliest, even though it needs annotation volume far more
urgently than an already-344-annotated niche like fitness.
"""
import uuid

import pytest

from qualume.annotate.gemini_watcher import ANALYSIS_PROMPT_VERSION as DNA_VERSION
from qualume.db.connection import get_connection
from qualume.db.watchlist import add_watched_account
from qualume.orchestrate import find_pending_annotation


def _cleanup_account(handle: str):
    conn = get_connection()
    try:
        with conn.cursor() as cur:
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
def two_niche_accounts():
    """A niche far below the sample floor (zz_test_starved, 0 annotated,
    1 pending item) and a niche already well past it (zz_test_saturated,
    50 already-annotated items via other accounts, 1 pending item) —
    real shape of the production starved-vs-saturated split (religion/
    philosophy niches at 0-2 annotated vs. fitness at 344)."""
    starved_handle = f"zz_test_starved_{uuid.uuid4().hex[:8]}"
    saturated_handle = f"zz_test_saturated_{uuid.uuid4().hex[:8]}"
    saturated_other_handle = f"zz_test_sat_other_{uuid.uuid4().hex[:8]}"
    for h in (starved_handle, saturated_handle, saturated_other_handle):
        _cleanup_account(h)

    starved_account = add_watched_account("tiktok", starved_handle, niche="zz_test_starved_niche")
    saturated_account = add_watched_account("tiktok", saturated_handle, niche="zz_test_saturated_niche")
    saturated_other_account = add_watched_account(
        "tiktok", saturated_other_handle, niche="zz_test_saturated_niche"
    )

    # 50 already-annotated items on the OTHER saturated-niche account —
    # pushes this niche's real annotated count well past clustering's
    # 40-sample floor, same as production's fitness/food/women's-health
    # niches today. Inserted BEFORE the starved/saturated pending items
    # below so a naive FIFO/insertion-order read would rank the
    # saturated niche's pending item ahead of the starved one — the test
    # only passes once real sample-floor priority ordering exists, not
    # by accident of insertion order.
    for _ in range(50):
        other_item = _insert_item(saturated_other_account, local_video_path="/fake/other.mp4")
        _insert_dna(other_item)

    saturated_item = _insert_item(saturated_account, local_video_path="/fake/saturated.mp4")
    starved_item = _insert_item(starved_account, local_video_path="/fake/starved.mp4")

    yield starved_item, saturated_item

    for h in (starved_handle, saturated_handle, saturated_other_handle):
        _cleanup_account(h)


def test_find_pending_annotation_orders_starved_niches_before_saturated_ones(two_niche_accounts):
    """The real fix: a niche with 0 real annotated videos (starved,
    nowhere near mining's 5-sample floor or clustering's 40-sample
    floor) must be annotated BEFORE a niche that already has 50+
    annotated videos (saturated, long past both floors) — regardless of
    DB insertion order or which got downloaded first. Default FIFO
    ordering actively fights this goal when a brand-new niche (e.g. a
    newly-added medspa/peptide watchlist) is downloaded alongside an
    established, already-saturated one."""
    starved_item, saturated_item = two_niche_accounts

    pending = find_pending_annotation()

    assert starved_item in pending
    assert saturated_item in pending
    # The starved niche's item must come first — it's furthest below
    # every real sample floor and most urgently needs annotation volume.
    assert pending.index(starved_item) < pending.index(saturated_item)


def test_find_pending_annotation_still_filters_by_watched_account_id(two_niche_accounts):
    """Real backward-compat guarantee: the existing watched_account_id
    scoping (used by tests and any future per-account caller) must keep
    working unchanged — priority ordering is a secondary sort key within
    whatever set watched_account_id already scoped to, not a replacement
    for it."""
    starved_item, saturated_item = two_niche_accounts

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT watched_account_id FROM external_content_items WHERE id = %s",
                (saturated_item,),
            )
            saturated_account_id = str(cur.fetchone()[0])
    finally:
        conn.close()

    pending = find_pending_annotation(watched_account_id=saturated_account_id)

    assert saturated_item in pending
    assert starved_item not in pending
