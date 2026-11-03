import sys
sys.path.insert(0, "src")
from qualume.db.watchlist import add_watched_account
import csv

angle_to_note = {
    "peptide_education": "angle: education/explainer content",
    "peptide_selling": "angle: direct selling/vendor content",
    "peptide_controversial": "angle: controversial/skeptical/debate content",
    "peptide_day_in_life": "angle: day-in-the-life/routine content",
    "peptide_results_tracking": "angle: results-tracking/before-after content",
    "peptide_doctor_content": "angle: doctor/clinician content",
    "peptide_gym_bro": "angle: gym-bro/male fitness-oriented content",
    "peptide_skincare_female": "angle: female-oriented/skincare content",
    "peptide_antiaging_older": "angle: older demographic/anti-aging content",
}

with open("data/candidate_handles_peptide_batch2_verified.csv") as f:
    reader = csv.DictReader(f)
    valid = [row for row in reader if row["status"] == "valid"]

added, skipped = 0, 0
for row in valid:
    handle = row["handle"]
    angle = row["niche"]
    note = angle_to_note[angle]
    try:
        account_id = add_watched_account(
            "tiktok", handle, notes=note, watchlist_type="emerging", niche="peptide/medspa content"
        )
        print(f"ADDED @{handle} (peptide/medspa content) id={account_id}")
        added += 1
    except Exception as e:
        print(f"SKIP @{handle}: {type(e).__name__}: {e}")
        skipped += 1

print(f"\nDone: {added} added, {skipped} skipped")
