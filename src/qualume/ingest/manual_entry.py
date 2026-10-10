"""Phase 1a: manual performance-entry CLI.

Greenfield workaround — no IG Graph API access configured yet (see
docs/session-context.md), so performance data is logged by hand after
checking the Instagram app for stats. Appends to performance_snapshots as an
event log (never upserts) per the v2 plan's idempotency/immutability
principle (§1.4): you need the full time-series to see growth/decay, not
just the latest number.
"""

import argparse

from qualume.db.connection import get_connection


def log_snapshot(
    content_item_id: str,
    views: int = 0,
    likes: int = 0,
    saves: int = 0,
    comments: int = 0,
    shares: int = 0,
    follows_gained: int = 0,
) -> None:
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO performance_snapshots
                   (content_item_id, views, likes, saves, comments, shares, follows_gained, entry_method)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, 'manual')""",
                (content_item_id, views, likes, saves, comments, shares, follows_gained),
            )
        conn.commit()
    finally:
        conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Log a manual performance snapshot.")
    parser.add_argument("content_item_id")
    parser.add_argument("--views", type=int, default=0)
    parser.add_argument("--likes", type=int, default=0)
    parser.add_argument("--saves", type=int, default=0)
    parser.add_argument("--comments", type=int, default=0)
    parser.add_argument("--shares", type=int, default=0)
    parser.add_argument("--follows-gained", type=int, default=0)
    args = parser.parse_args()

    log_snapshot(
        args.content_item_id,
        views=args.views,
        likes=args.likes,
        saves=args.saves,
        comments=args.comments,
        shares=args.shares,
        follows_gained=args.follows_gained,
    )
    print("Snapshot logged.")
