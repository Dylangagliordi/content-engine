import sys
sys.path.insert(0, "src")
from qualume.orchestrate import find_pending_annotation, _run_stage_over_items, ANNOTATION_CONCURRENCY
from qualume.annotate.gemini_watcher import annotate_and_store

pending = find_pending_annotation()
print(f"pending annotation (priority-ordered): {len(pending)}")
result = _run_stage_over_items(pending, annotate_and_store, max_workers=ANNOTATION_CONCURRENCY)
print(f"succeeded={result['succeeded']} failed={result['failed']}")
if result["errors"]:
    print("sample errors:", result["errors"][:5])
