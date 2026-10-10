from qualume.db.connection import get_connection


def add_watched_account(platform: str, handle: str, notes: str | None = None) -> str:
    """Register an account to poll. Raises on duplicate (platform, handle) —
    fail loud rather than silently ignoring a re-add attempt."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO watched_accounts (platform, handle, notes)
                   VALUES (%s, %s, %s) RETURNING id""",
                (platform, handle, notes),
            )
            account_id = cur.fetchone()[0]
        conn.commit()
        return str(account_id)
    finally:
        conn.close()


def list_watched_accounts(platform: str | None = None, active_only: bool = True) -> list[dict]:
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            query = "SELECT id, platform, handle, active, notes FROM watched_accounts WHERE 1=1"
            params = []
            if platform:
                query += " AND platform = %s"
                params.append(platform)
            if active_only:
                query += " AND active = true"
            cur.execute(query, params)
            rows = cur.fetchall()
    finally:
        conn.close()

    return [
        {"id": str(r[0]), "platform": r[1], "handle": r[2], "active": r[3], "notes": r[4]}
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
