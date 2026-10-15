"""Convenience CLI for managing the TikTok watch-list.

Usage:
    uv run python -m qualume.manage_watchlist add tiktok some_handle [--type competitor]
    uv run python -m qualume.manage_watchlist list [--type competitor]
    uv run python -m qualume.manage_watchlist deactivate <account_id>
"""

import argparse

from qualume.db.watchlist import add_watched_account, list_watched_accounts, deactivate_watched_account

WATCHLIST_TYPES = [
    "general", "competitor", "cross_niche", "aspirational", "emerging", "format_specific",
]


def main():
    parser = argparse.ArgumentParser(description="Manage the content-reading watch-list.")
    sub = parser.add_subparsers(dest="command", required=True)

    add_p = sub.add_parser("add", help="Add an account to watch")
    add_p.add_argument("platform", choices=["tiktok"])
    add_p.add_argument("handle", help="Account handle, no @ prefix")
    add_p.add_argument("--notes", default=None)
    add_p.add_argument("--type", dest="watchlist_type", choices=WATCHLIST_TYPES, default="general",
                       help="Why this account is watched")

    list_p = sub.add_parser("list", help="List watched accounts")
    list_p.add_argument("--platform", default=None)
    list_p.add_argument("--all", action="store_true", help="Include deactivated accounts")
    list_p.add_argument("--type", dest="watchlist_type", choices=WATCHLIST_TYPES, default=None,
                        help="Only show accounts of this watchlist type")

    deact_p = sub.add_parser("deactivate", help="Stop polling an account")
    deact_p.add_argument("account_id")

    args = parser.parse_args()

    if args.command == "add":
        account_id = add_watched_account(
            args.platform, args.handle, notes=args.notes, watchlist_type=args.watchlist_type
        )
        print(f"Added @{args.handle} ({args.platform}, {args.watchlist_type}) — id: {account_id}")

    elif args.command == "list":
        accounts = list_watched_accounts(
            platform=args.platform, active_only=not args.all, watchlist_type=args.watchlist_type
        )
        if not accounts:
            print("No watched accounts yet. Add one with: add tiktok <handle>")
        for a in accounts:
            status = "active" if a["active"] else "inactive"
            print(f"  [{status}] {a['id']}  @{a['handle']} ({a['platform']})"
                  f"  watchlist_type={a['watchlist_type']}"
                  + (f"  — {a['notes']}" if a["notes"] else ""))

    elif args.command == "deactivate":
        deactivate_watched_account(args.account_id)
        print(f"Deactivated {args.account_id}")


if __name__ == "__main__":
    main()
