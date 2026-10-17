import sys
import time

from qualume.annotate.gemini_watcher import annotate_and_store
from qualume.db.connection import get_connection

results = {"success": [], "validation_error": [], "coverage_warning": [], "failed": []}

with open("/tmp/v5_todo_ids.txt") as f:
    ids = [line.strip() for line in f if line.strip()]

print(f"=== starting v5 batch, {len(ids)} videos ===", flush=True)

for i, item_id in enumerate(ids, 1):
    try:
        dna_id = annotate_and_store(item_id)
    except Exception as e:
        print(f"[{i}/{len(ids)}] {item_id} FAILED: {type(e).__name__}: {e}", flush=True)
        results["failed"].append((item_id, str(e)))
        continue

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT validation_error, coverage_warning FROM external_creative_dna WHERE id = %s",
                (dna_id,),
            )
            val_err, cov_warn = cur.fetchone()
    finally:
        conn.close()

    if val_err:
        print(f"[{i}/{len(ids)}] {item_id} -> {dna_id} VALIDATION_ERROR: {val_err}", flush=True)
        results["validation_error"].append((item_id, dna_id, val_err))
    elif cov_warn:
        print(f"[{i}/{len(ids)}] {item_id} -> {dna_id} COVERAGE_WARNING: {cov_warn}", flush=True)
        results["coverage_warning"].append((item_id, dna_id, cov_warn))
    else:
        print(f"[{i}/{len(ids)}] {item_id} -> {dna_id} CLEAN", flush=True)
        results["success"].append((item_id, dna_id))

    time.sleep(1)  # small courtesy gap between calls

print("\n=== SUMMARY ===", flush=True)
print(f"Total attempted: {len(ids)}", flush=True)
print(f"Clean success: {len(results['success'])}", flush=True)
print(f"Validation error (schema failed): {len(results['validation_error'])}", flush=True)
print(f"Coverage warning (schema OK, incomplete): {len(results['coverage_warning'])}", flush=True)
print(f"Hard failed (exception): {len(results['failed'])}", flush=True)

import json
with open("/tmp/v5_batch_results.json", "w") as f:
    json.dump(results, f, indent=2)
