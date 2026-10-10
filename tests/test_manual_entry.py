import pytest
import uuid
from qualume.ingest.manual_entry import log_snapshot
from qualume.db.connection import get_connection


def _make_test_content_item() -> str:
    """Insert a throwaway content_item row to attach test snapshots to."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO content_items (workflow_type, platform, status)
                   VALUES ('ig_static', 'instagram', 'published')
                   RETURNING id"""
            )
            content_item_id = cur.fetchone()[0]
        conn.commit()
        return str(content_item_id)
    finally:
        conn.close()


def _delete_test_content_item(content_item_id: str) -> None:
    """Clean up a throwaway content_item (and its snapshots via FK cascade).

    No test fixture here is allowed to leave rows behind — this project has
    no separate test DB yet, so every row left behind pollutes the real dev
    counts (content_items.status='published' here is otherwise meaningful:
    it means a human actually posted something). See CLAUDE.md principle 6.
    """
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM performance_snapshots WHERE content_item_id = %s",
                (content_item_id,),
            )
            cur.execute(
                "DELETE FROM content_items WHERE id = %s",
                (content_item_id,),
            )
        conn.commit()
    finally:
        conn.close()


@pytest.fixture
def content_item_id():
    """A throwaway content_item id, auto-deleted (with its snapshots) after the test."""
    cid = _make_test_content_item()
    yield cid
    _delete_test_content_item(cid)


def test_log_snapshot_persists_row_with_manual_entry_method(content_item_id):
    log_snapshot(content_item_id, views=100, likes=10, saves=2, comments=1)

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """SELECT views, likes, saves, comments, entry_method
                   FROM performance_snapshots WHERE content_item_id = %s""",
                (content_item_id,),
            )
            row = cur.fetchone()
    finally:
        conn.close()

    assert row == (100, 10, 2, 1, "manual")


def test_log_snapshot_allows_multiple_snapshots_over_time(content_item_id):
    """Append-only event log, not an upsert — v2 plan §1 principle 4."""
    log_snapshot(content_item_id, views=100, likes=10, saves=2, comments=1)
    log_snapshot(content_item_id, views=250, likes=30, saves=5, comments=3)

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT count(*) FROM performance_snapshots WHERE content_item_id = %s",
                (content_item_id,),
            )
            count = cur.fetchone()[0]
    finally:
        conn.close()

    assert count == 2


def test_log_snapshot_rejects_nonexistent_content_item():
    fake_id = str(uuid.uuid4())
    with pytest.raises(Exception):
        log_snapshot(fake_id, views=1, likes=1, saves=1, comments=1)
