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


def test_add_watched_account_defaults_watchlist_type_to_general():
    handle = f"test_type_default_{uuid.uuid4().hex[:8]}"
    _cleanup(handle)
    try:
        add_watched_account("tiktok", handle)
        accounts = list_watched_accounts(platform="tiktok", active_only=True)
        match = [a for a in accounts if a["handle"] == handle]
        assert len(match) == 1
        assert match[0]["watchlist_type"] == "general"
    finally:
        _cleanup(handle)


def test_add_watched_account_persists_explicit_watchlist_type():
    handle = f"test_type_explicit_{uuid.uuid4().hex[:8]}"
    _cleanup(handle)
    try:
        add_watched_account("tiktok", handle, watchlist_type="competitor")
        accounts = list_watched_accounts(platform="tiktok", active_only=True)
        match = [a for a in accounts if a["handle"] == handle]
        assert len(match) == 1
        assert match[0]["watchlist_type"] == "competitor"
    finally:
        _cleanup(handle)


def test_list_watched_accounts_filters_by_watchlist_type():
    competitor = f"test_type_comp_{uuid.uuid4().hex[:8]}"
    cross = f"test_type_cross_{uuid.uuid4().hex[:8]}"
    _cleanup(competitor)
    _cleanup(cross)
    try:
        add_watched_account("tiktok", competitor, watchlist_type="competitor")
        add_watched_account("tiktok", cross, watchlist_type="cross_niche")

        filtered = list_watched_accounts(watchlist_type="cross_niche", active_only=True)
        handles = [a["handle"] for a in filtered]
        assert cross in handles
        assert competitor not in handles
        assert all(a["watchlist_type"] == "cross_niche" for a in filtered)
    finally:
        _cleanup(competitor)
        _cleanup(cross)
