import uuid
from qualume.db.watchlist import add_watched_account, list_watched_accounts, deactivate_watched_account


def _cleanup(handle: str):
    """Test helper: remove a test account by handle so re-runs don't collide
    with the UNIQUE(platform, handle) constraint."""
    from qualume.db.connection import get_connection
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM watched_accounts WHERE handle = %s", (handle,))
        conn.commit()
    finally:
        conn.close()


def test_add_watched_account_persists_and_is_listed():
    handle = f"test_account_{uuid.uuid4().hex[:8]}"
    _cleanup(handle)
    try:
        account_id = add_watched_account("tiktok", handle)
        assert account_id is not None

        accounts = list_watched_accounts(platform="tiktok", active_only=True)
        handles = [a["handle"] for a in accounts]
        assert handle in handles
    finally:
        _cleanup(handle)


def test_add_watched_account_rejects_duplicate():
    handle = f"test_dup_{uuid.uuid4().hex[:8]}"
    _cleanup(handle)
    try:
        add_watched_account("tiktok", handle)
        import pytest
        with pytest.raises(Exception):
            add_watched_account("tiktok", handle)
    finally:
        _cleanup(handle)


def test_deactivate_watched_account_excludes_from_active_list():
    handle = f"test_deactivate_{uuid.uuid4().hex[:8]}"
    _cleanup(handle)
    try:
        account_id = add_watched_account("tiktok", handle)
        deactivate_watched_account(account_id)

        active = list_watched_accounts(platform="tiktok", active_only=True)
        handles = [a["handle"] for a in active]
        assert handle not in handles

        all_accounts = list_watched_accounts(platform="tiktok", active_only=False)
        handles_all = [a["handle"] for a in all_accounts]
        assert handle in handles_all
    finally:
        _cleanup(handle)
