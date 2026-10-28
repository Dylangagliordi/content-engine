"""One-off backfill: re-annotate all external_content_items at v6 (presenter/scene fields).

Reads pending ids from /tmp/v6_pending_ids.txt, calls annotate_and_store for each,
retries on Gemini 503 (transient high-demand), logs progress to /tmp/v6_backfill_progress.log
line by line so a crash/timeout doesn't lose completed work. Idempotent-ish: re-running
just re-does whatever's still pending per the input list (caller regenerates that list fresh
if resuming after a partial run, by re-querying which ids are NOT YET v6 in the DB).
"""
import time
import traceback
from datetime import datetime, timezone

from qualume.annotate.gemini_watcher import annotate_and_store

PENDING_FILE = "/tmp/v6_pending_ids.txt"
LOG_FILE = "/tmp/v6_backfill_progress.log"

MAX_RETRIES = 4
BASE_WAIT = 20  # seconds, exponential backoff: 20, 40, 80, 160


def log(msg: str) -> None:
    line = f"{datetime.now(timezone.utc).isoformat()} {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def main() -> None:
    with open(PENDING_FILE) as f:
        ids = [line.strip() for line in f if line.strip()]

    log(f"START backfill, {len(ids)} items pending")
    ok, failed = 0, 0

    for i, content_id in enumerate(ids, 1):
        attempt = 0
        while True:
            attempt += 1
            try:
                dna_id = annotate_and_store(content_id)
                log(f"[{i}/{len(ids)}] OK content_id={content_id} dna_id={dna_id}")
                ok += 1
                break
            except Exception as e:
                is_503 = "503" in str(e) or "UNAVAILABLE" in str(e)
                if is_503 and attempt <= MAX_RETRIES:
                    wait = BASE_WAIT * (2 ** (attempt - 1))
                    log(f"[{i}/{len(ids)}] 503, retry {attempt}/{MAX_RETRIES} in {wait}s content_id={content_id}")
                    time.sleep(wait)
                    continue
                log(f"[{i}/{len(ids)}] FAILED content_id={content_id} attempt={attempt} err={e!r}")
                log(traceback.format_exc())
                failed += 1
                break

    log(f"DONE backfill: ok={ok} failed={failed} total={len(ids)}")


if __name__ == "__main__":
    main()
