"""One-off verification script: check each candidate handle in
data/candidate_handles.txt is a REAL, scrapable TikTok account via yt-dlp
flat-playlist metadata-only extraction (playlistend=1, no download) before
committing it to the real watchlist table. Sequential + small delay
between checks (same TikTok-facing-conservative discipline as
poll_tiktok_watchlist.py) — this is a one-time batch, not the scheduled
ingestion path, so it still deserves real pacing to avoid IP rate-limiting.

Writes results to data/candidate_handles_verified.csv: handle,niche,status
where status is 'valid', 'invalid', or 'error:<msg>'.
"""
import csv
import sys
import time

import yt_dlp

INPUT = "data/candidate_handles.txt"
OUTPUT = "data/candidate_handles_verified.csv"
DELAY_SECONDS = 1.5


def check_handle(handle: str) -> str:
    url = f"https://www.tiktok.com/@{handle}"
    ydl_opts = {
        "quiet": True,
        "skip_download": True,
        "no_warnings": True,
        "extract_flat": True,
        "playlistend": 1,
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            entries = info.get("entries") if info else None
            if entries:
                return "valid"
            return "invalid:no_entries"
    except Exception as e:
        return f"error:{type(e).__name__}:{str(e)[:120]}"


def main():
    rows = []
    with open(INPUT) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            handle, niche = line.split("|")
            rows.append((handle, niche))

    results = []
    for i, (handle, niche) in enumerate(rows):
        status = check_handle(handle)
        results.append((handle, niche, status))
        print(f"[{i+1}/{len(rows)}] @{handle} ({niche}): {status}", flush=True)
        if i < len(rows) - 1:
            time.sleep(DELAY_SECONDS)

    with open(OUTPUT, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["handle", "niche", "status"])
        writer.writerows(results)

    valid = sum(1 for _, _, s in results if s == "valid")
    print(f"\nDone: {valid}/{len(results)} valid. Written to {OUTPUT}")


if __name__ == "__main__":
    main()
