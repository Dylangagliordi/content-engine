"""Tests for orchestrate's controlled-concurrency stage runner — written
FIRST per this project's TDD discipline.

Real problem this fixes: _run_stage_over_items processed every item
strictly one at a time, even for stages (annotation, classification)
whose real work is a NETWORK CALL to Gemini/Claude, not anything TikTok-
facing. TikTok-facing work (poll, download) genuinely needs to stay
conservative/sequential to avoid rate-limiting or soft-blocking the real
account — but annotation/classification have no such constraint, and
serializing them was pure wasted wall-clock time once the real watchlist
grew from 2 to 20+ accounts (2026-10-30, user asked to reach 5000
ingested videos "as fast as possible").

max_workers parameter, default 1 (fully backward compatible — existing
callers that don't pass it get identical sequential behavior). Uses
ThreadPoolExecutor (I/O-bound work — real HTTP calls to Gemini/Claude,
not CPU-bound), each worker gets its own DB connection via the existing
get_connection() pattern already used everywhere in this codebase (no
new thread-safety concern introduced — connections were never shared
across calls to begin with).
"""
import time
from unittest.mock import patch

from qualume.orchestrate import _run_stage_over_items


def test_run_stage_over_items_sequential_by_default_preserves_existing_behavior():
    """max_workers unset (or =1) must behave EXACTLY like the original
    for-loop — same success/failure counts, same error format — so every
    existing caller (orchestrate tests, real production usage) keeps
    working unchanged."""
    def fn(item_id):
        if item_id == "bad":
            raise RuntimeError("simulated failure")
        return f"ok-{item_id}"

    result = _run_stage_over_items(["a", "bad", "c"], fn)

    assert result["succeeded"] == 2
    assert result["failed"] == 1
    assert len(result["errors"]) == 1
    assert result["errors"][0][0] == "bad"
    assert "simulated failure" in result["errors"][0][1]


def test_run_stage_over_items_concurrent_runs_items_in_parallel():
    """Real proof concurrency is actually happening, not just accepted
    as a no-op parameter: 5 items that each sleep 0.2s should complete in
    well under 5*0.2=1.0s total with max_workers=5 (would take ~1.0s+
    sequentially), proving they ran overlapping, not one after another."""
    def slow_fn(item_id):
        time.sleep(0.2)
        return item_id

    start = time.monotonic()
    result = _run_stage_over_items(["a", "b", "c", "d", "e"], slow_fn, max_workers=5)
    elapsed = time.monotonic() - start

    assert result["succeeded"] == 5
    assert result["failed"] == 0
    assert elapsed < 0.6  # well under the ~1.0s+ sequential would take


def test_run_stage_over_items_concurrent_isolates_failures_same_as_sequential():
    """One item's failure must not affect any other item's outcome, same
    isolation discipline as the sequential path — just running
    concurrently now."""
    def fn(item_id):
        if item_id in ("bad1", "bad2"):
            raise RuntimeError(f"simulated failure for {item_id}")
        return item_id

    result = _run_stage_over_items(
        ["a", "bad1", "b", "bad2", "c"], fn, max_workers=5
    )

    assert result["succeeded"] == 3
    assert result["failed"] == 2
    failed_ids = {item_id for item_id, _ in result["errors"]}
    assert failed_ids == {"bad1", "bad2"}


def test_run_stage_over_items_concurrent_handles_empty_list():
    result = _run_stage_over_items([], lambda x: x, max_workers=5)
    assert result["succeeded"] == 0
    assert result["failed"] == 0


def test_run_stage_over_items_max_workers_one_is_still_sequential():
    """Explicit max_workers=1 must produce byte-identical results to the
    default (no max_workers passed at all) — same backward-compat
    guarantee, just passed explicitly."""
    call_order = []

    def fn(item_id):
        call_order.append(item_id)
        time.sleep(0.01)
        return item_id

    result = _run_stage_over_items(["a", "b", "c"], fn, max_workers=1)

    assert result["succeeded"] == 3
    assert call_order == ["a", "b", "c"]  # strict order preserved at max_workers=1
