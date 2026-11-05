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


def test_add_watched_account_niche_defaults_to_none():
    """niche is a real topic/vertical field (e.g. 'women's health
    supplements'), deliberately separate from watchlist_type (which is
    about WHY we track an account relative to us, not WHAT it's about).
    Nullable by design — an account added without a niche shouldn't
    silently get a fabricated default, same discipline as every other
    optional field in this schema."""
    handle = f"test_niche_default_{uuid.uuid4().hex[:8]}"
    _cleanup(handle)
    try:
        add_watched_account("tiktok", handle)
        accounts = list_watched_accounts(platform="tiktok", active_only=True)
        match = [a for a in accounts if a["handle"] == handle]
        assert len(match) == 1
        assert match[0]["niche"] is None
    finally:
        _cleanup(handle)


def test_add_watched_account_persists_explicit_niche():
    handle = f"test_niche_explicit_{uuid.uuid4().hex[:8]}"
    _cleanup(handle)
    try:
        add_watched_account("tiktok", handle, niche="women's health supplements")
        accounts = list_watched_accounts(platform="tiktok", active_only=True)
        match = [a for a in accounts if a["handle"] == handle]
        assert len(match) == 1
        assert match[0]["niche"] == "women's health supplements"
    finally:
        _cleanup(handle)


def test_list_distinct_niches_returns_only_real_niches_no_duplicates(monkeypatch=None):
    """Real gap closed 2026-11-04: orchestrate.run_ingestion_cycle needs
    this to loop mining/clustering per-niche instead of calling each pass
    once unscoped. Two accounts sharing a niche must appear ONCE, not
    twice; a NULL niche must never appear at all."""
    from qualume.db.watchlist import list_distinct_niches

    a = f"test_niches_a_{uuid.uuid4().hex[:8]}"
    b = f"test_niches_b_{uuid.uuid4().hex[:8]}"
    c = f"test_niches_c_{uuid.uuid4().hex[:8]}"
    niche = f"zz_test_shared_niche_{uuid.uuid4().hex[:8]}"
    _cleanup(a)
    _cleanup(b)
    _cleanup(c)
    try:
        add_watched_account("tiktok", a, niche=niche)
        add_watched_account("tiktok", b, niche=niche)  # same niche, different account
        add_watched_account("tiktok", c, niche=None)   # not yet classified

        niches = list_distinct_niches(active_only=True)
        assert niches.count(niche) == 1
        assert None not in niches
    finally:
        _cleanup(a)
        _cleanup(b)
        _cleanup(c)


def test_list_distinct_niches_respects_active_only():
    from qualume.db.watchlist import list_distinct_niches

    handle = f"test_niches_inactive_{uuid.uuid4().hex[:8]}"
    niche = f"zz_test_inactive_niche_{uuid.uuid4().hex[:8]}"
    _cleanup(handle)
    try:
        account_id = add_watched_account("tiktok", handle, niche=niche)
        deactivate_watched_account(account_id)

        assert niche not in list_distinct_niches(active_only=True)
        assert niche in list_distinct_niches(active_only=False)
    finally:
        _cleanup(handle)
