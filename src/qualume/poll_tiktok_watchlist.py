"""Scheduled poller: sweep every active watched TikTok account, ingest new
videos, report a summary. This is the script that a cron job / n8n schedule
/ systemd timer invokes on a cadence (e.g. every few hours) — see
docs/session-context.md for the "polling + dedup, not true push events"
design rationale (matches how commercial tools like Apify's monitoring
pattern actually work).

Designed to run identically on this Mac now and on a Raspberry Pi later —
no machine-specific assumptions beyond Docker + the .env file.
"""

import argparse
import time

from qualume.db.watchlist import list_watched_accounts
from qualume.ingest.tiktok_reader import ingest_videos_for_account, DEFAULT_MAX_VIDEOS_PER_ACCOUNT

# Be a polite, residential-looking client, not a hammer: small delay between
# accounts so a sweep of N accounts doesn't look like a burst of automated
# traffic. Tune this up if accounts start getting rate-limited/blocked.
DELAY_BETWEEN_ACCOUNTS_SECONDS = 3


def run_poll_cycle(max_videos: int = DEFAULT_MAX_VIDEOS_PER_ACCOUNT) -> dict:
    accounts = list_watched_accounts(platform="tiktok", active_only=True)
    summary = {"accounts_polled": 0, "accounts_failed": 0, "new_videos": 0, "skipped_duplicates": 0}

    for i, account in enumerate(accounts):
        try:
            result = ingest_videos_for_account(account["id"], account["handle"], max_videos=max_videos)
            summary["accounts_polled"] += 1
            summary["new_videos"] += result["new_videos"]
            summary["skipped_duplicates"] += result["skipped_duplicates"]
            print(f"  @{account['handle']}: {result['new_videos']} new, "
                  f"{result['skipped_duplicates']} already known")
        except Exception as e:
            # Fail loud per-account, don't let one broken/blocked account
            # kill the whole sweep (v2 plan §1.9 — fail loud, fail cheap,
            # but ALSO: one bad account shouldn't silently stop the others).
            summary["accounts_failed"] += 1
            print(f"  @{account['handle']}: FAILED — {e}")

        if i < len(accounts) - 1:
            time.sleep(DELAY_BETWEEN_ACCOUNTS_SECONDS)

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Poll all active watched TikTok accounts.")
    parser.add_argument("--max-videos", type=int, default=DEFAULT_MAX_VIDEOS_PER_ACCOUNT)
    args = parser.parse_args()

    print("Starting poll cycle...")
    result = run_poll_cycle(max_videos=args.max_videos)
    print(f"\nDone. Polled {result['accounts_polled']} accounts "
          f"({result['accounts_failed']} failed), "
          f"{result['new_videos']} new videos, "
          f"{result['skipped_duplicates']} already known.")
