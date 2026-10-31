"""Lightweight progress monitor — polls DB counts every 60s, appends one
compact line per poll to data/ingestion_progress.log. Avoids re-reading
giant yt-dlp stdout; just the real numbers that matter."""
import time
from datetime import datetime

from qualume.db.connection import get_connection

LOG = "data/ingestion_progress.log"

while True:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT count(*) FROM external_content_items")
    total = cur.fetchone()[0]
    cur.execute("SELECT count(distinct watched_account_id) FROM external_content_items")
    accounts = cur.fetchone()[0]
    cur.execute("SELECT count(*) FROM external_content_items WHERE local_video_path IS NOT NULL")
    downloaded = cur.fetchone()[0]
    cur.execute("SELECT count(*) FROM external_creative_dna")
    annotated = cur.fetchone()[0]
    cur.execute("SELECT count(*) FROM external_copy_analysis")
    classified = cur.fetchone()[0]
    conn.close()

    line = (f"{datetime.now().isoformat()} total={total} accounts={accounts} "
            f"downloaded={downloaded} annotated={annotated} classified={classified}\n")
    with open(LOG, "a") as f:
        f.write(line)
    print(line, end="", flush=True)
    time.sleep(60)
