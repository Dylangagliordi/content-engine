"""Ingestion orchestration — the single entry point n8n's Schedule
Trigger calls (via the local HTTP wrapper, see server.py) to run one full
cycle of the pipeline: poll watchlist -> download pending videos ->
annotate pending (Gemini v5) -> classify copy for newly-annotated videos.

Every underlying step is already idempotent on its own (polling dedupes
on platform_video_id, download_pending_videos only touches NULL
local_video_path rows, annotation/classification below only touch rows
genuinely missing their respective output) — this module adds no new
state, it just finds "what's pending" at each stage and calls the
existing function. Stages are isolated from each other (a poll failure
doesn't block annotation of already-downloaded videos) and within a
stage, one item's failure doesn't block the rest (same discipline as
poll_tiktok_watchlist's per-account isolation) — this is the real shape
a scheduled, unattended job on an always-on host needs: one bad video or
one transient API 503 must never silently stop the whole cycle.
"""
from qualume.annotate.copy_classifier import classify_and_store_external
from qualume.annotate.gemini_watcher import ANALYSIS_PROMPT_VERSION as DNA_ANALYSIS_PROMPT_VERSION
from qualume.annotate.gemini_watcher import annotate_and_store
from qualume.ingest.tiktok_reader import download_pending_videos
from qualume.learn.run_mining_pass import run_mining_pass
from qualume.poll_tiktok_watchlist import run_poll_cycle


def find_pending_annotation(watched_account_id: str | None = None) -> list[str]:
    """external_content_items with a local video file but no row at the
    CURRENT annotation version yet (qualume.annotate.gemini_watcher.
    ANALYSIS_PROMPT_VERSION, read live — NOT a hardcoded string. Real bug
    found and fixed 2026-10-28: this used to hardcode 'v5', so after
    gemini_watcher bumped to v6 every genuinely-new video silently never
    got picked up by a scheduled cycle, even though annotate_and_store
    itself was already producing v6 output correctly when called by hand.
    An item with only an older version, e.g. v1, still counts as pending —
    same "always annotate forward to the latest version" discipline the
    rest of this project uses)."""
    from qualume.db.connection import get_connection

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            query = """SELECT eci.id FROM external_content_items eci
                       WHERE eci.local_video_path IS NOT NULL
                         AND eci.id NOT IN (
                             SELECT external_content_item_id FROM external_creative_dna
                             WHERE analysis_prompt_version = %s
                         )"""
            params: list = [DNA_ANALYSIS_PROMPT_VERSION]
            if watched_account_id is not None:
                query += " AND eci.watched_account_id = %s"
                params.append(watched_account_id)
            cur.execute(query, params)
            return [str(row[0]) for row in cur.fetchall()]
    finally:
        conn.close()


def find_pending_classification(watched_account_id: str | None = None) -> list[str]:
    """external_content_items with a CURRENT-version annotation (same live
    DNA_ANALYSIS_PROMPT_VERSION constant as find_pending_annotation above,
    same 2026-10-28 bugfix — this used to hardcode 'v5' too, same stale-
    string failure mode) but no copy-analysis row yet at all (any version
    — this module doesn't re-run classification just because a newer
    copy-classifier ANALYSIS_PROMPT_VERSION exists; that re-running
    decision is the same kind of explicit, deliberate batch job the
    v3->v4->v5 Gemini re-annotations were, not something a scheduled
    cycle does automatically)."""
    from qualume.db.connection import get_connection

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            query = """SELECT DISTINCT ecd.external_content_item_id
                       FROM external_creative_dna ecd
                       JOIN external_content_items eci ON eci.id = ecd.external_content_item_id
                       WHERE ecd.analysis_prompt_version = %s
                         AND ecd.external_content_item_id NOT IN (
                             SELECT external_content_item_id FROM external_copy_analysis
                         )"""
            params: list = [DNA_ANALYSIS_PROMPT_VERSION]
            if watched_account_id is not None:
                query += " AND eci.watched_account_id = %s"
                params.append(watched_account_id)
            cur.execute(query, params)
            return [str(row[0]) for row in cur.fetchall()]
    finally:
        conn.close()


def _run_stage_over_items(items: list[str], fn) -> dict:
    succeeded = 0
    errors: list[tuple[str, str]] = []
    for item_id in items:
        try:
            fn(item_id)
            succeeded += 1
        except Exception as e:
            errors.append((item_id, f"{type(e).__name__}: {e}"))
    return {"succeeded": succeeded, "failed": len(errors), "errors": errors}


def run_ingestion_cycle() -> dict:
    """Run one full cycle. Returns a dict with one key per stage
    (poll/download/annotation/classification/mining); a stage that raised
    entirely (not a per-item failure, a whole-stage failure, e.g.
    run_poll_cycle itself raising) is recorded as {"error": "..."} for
    that stage rather than aborting the remaining stages — whatever
    videos are ALREADY downloaded should still get annotated/classified
    even if this cycle's poll attempt failed outright.

    Mining is the 5th stage (added 2026-10-30): turns the same cycle that
    just ingested new external data into the trigger for re-running Part
    5b's statistical pass over the WHOLE dataset (not just what's new this
    cycle — the pattern-mining primitives always query everything with
    performance data, same as a manual run). learn.run_mining_pass is
    itself idempotent on repeat calls (upsert keyed on stable identity,
    not a blind insert), so running it every 6 hours alongside ingestion
    is safe and is what actually keeps insights_external current as the
    dataset grows, instead of needing someone to remember to run it by
    hand."""
    result: dict = {}

    try:
        result["poll"] = run_poll_cycle()
    except Exception as e:
        result["poll"] = {"error": f"{type(e).__name__}: {e}"}

    try:
        result["download"] = download_pending_videos()
    except Exception as e:
        result["download"] = {"error": f"{type(e).__name__}: {e}"}

    try:
        pending_annotation = find_pending_annotation()
        result["annotation"] = _run_stage_over_items(pending_annotation, annotate_and_store)
    except Exception as e:
        result["annotation"] = {"error": f"{type(e).__name__}: {e}"}

    try:
        pending_classification = find_pending_classification()
        result["classification"] = _run_stage_over_items(
            pending_classification, classify_and_store_external
        )
    except Exception as e:
        result["classification"] = {"error": f"{type(e).__name__}: {e}"}

    try:
        # tiktok_reels is the only real workflow_type with external data
        # today (watched_accounts is TikTok-only so far) — hardcoded here,
        # not read from anywhere dynamic, same as run_batch.py's own
        # workflow_type handling; becomes a real parameter the day a
        # second external-reading workflow (e.g. Instagram) exists.
        result["mining"] = run_mining_pass(applies_to_workflow_type="tiktok_reels")
    except Exception as e:
        result["mining"] = {"error": f"{type(e).__name__}: {e}"}

    return result
