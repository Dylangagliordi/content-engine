from qualume.db.connection import get_connection


def add_watched_account(
    platform: str, handle: str, notes: str | None = None, watchlist_type: str = "general",
    niche: str | None = None,
) -> str:
    """Register an account to poll. Raises on duplicate (platform, handle) —
    fail loud rather than silently ignoring a re-add attempt.

    niche: a real topic/vertical (e.g. "women's health supplements",
    "fitness", "finance"), deliberately SEPARATE from watchlist_type
    (which is about WHY we track an account relative to US —
    competitor/aspirational/cross_niche/etc — not WHAT its content is
    about). Nullable — an account added without a niche shouldn't
    silently get a fabricated default; NULL means "not yet classified"."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO watched_accounts (platform, handle, notes, watchlist_type, niche)
                   VALUES (%s, %s, %s, %s, %s) RETURNING id""",
                (platform, handle, notes, watchlist_type, niche),
            )
            account_id = cur.fetchone()[0]
        conn.commit()
        return str(account_id)
    finally:
        conn.close()


def list_watched_accounts(
    platform: str | None = None, active_only: bool = True, watchlist_type: str | None = None
) -> list[dict]:
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            query = (
                "SELECT id, platform, handle, active, notes, watchlist_type, niche"
                " FROM watched_accounts WHERE 1=1"
            )
            params = []
            if platform:
                query += " AND platform = %s"
                params.append(platform)
            if watchlist_type:
                query += " AND watchlist_type = %s"
                params.append(watchlist_type)
            if active_only:
                query += " AND active = true"
            cur.execute(query, params)
            rows = cur.fetchall()
    finally:
        conn.close()

    return [
        {"id": str(r[0]), "platform": r[1], "handle": r[2], "active": r[3], "notes": r[4],
         "watchlist_type": r[5], "niche": r[6]}
        for r in rows
    ]


def deactivate_watched_account(account_id: str) -> None:
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE watched_accounts SET active = false WHERE id = %s", (account_id,)
            )
        conn.commit()
    finally:
        conn.close()
