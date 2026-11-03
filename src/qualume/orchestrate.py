"""Ingestion orchestration — the single entry point n8n's Schedule
Trigger calls (via the local HTTP wrapper, see server.py) to run one full
cycle of the pipeline: poll watchlist -> download pending videos ->
annotate pending (Gemini v5) -> classify copy -> classify content_type
for newly-annotated videos.

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
from qualume.annotate.content_type_classifier import classify_and_store_content_type
from qualume.annotate.copy_classifier import classify_and_store_external
from qualume.annotate.gemini_watcher import ANALYSIS_PROMPT_VERSION as DNA_ANALYSIS_PROMPT_VERSION
from qualume.annotate.gemini_watcher import annotate_and_store
from qualume.ingest.tiktok_reader import download_pending_videos
from qualume.learn.run_cluster_pass import run_cluster_pass
from qualume.learn.run_mining_pass import run_mining_pass
from qualume.learn.run_visual_cluster_pass import run_visual_cluster_pass
from qualume.poll_tiktok_watchlist import run_poll_cycle

# Real concurrency (added 2026-10-30, per user's explicit request to reach
# real volume "as fast as possible" after the watchlist grew from 2 to 20+
# accounts): Gemini annotation and Claude classification are each a single
# I/O-bound network call per video with no TikTok-facing rate-limit risk —
# safe to run many at once.
#
# LOWERED from 8 to 4 on 2026-10-31 after a real production incident: with
# ANNOTATION_CONCURRENCY=8, CLASSIFICATION_CONCURRENCY=8, and visual
# embedding's own EMBEDDING_CONCURRENCY=8 (ffmpeg subprocess per worker)
# all running at once for hours, physical memory dropped to ~111MB free /
# load avg 7.4, and OrbStack's Docker VM hung hard enough that Postgres
# became genuinely unreachable (connection timeout, not just a slow
# query) — required a full OrbStack restart to recover. 4 workers is a
# real, deliberately conservative choice to leave headroom for the other
# concurrent pipelines (visual embedding) sharing the same host, not a
# guess — revisit upward only with real memory-pressure monitoring in
# place, not just because throughput feels slow.
ANNOTATION_CONCURRENCY = 4
CLASSIFICATION_CONCURRENCY = 4


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
    rest of this project uses).

    Ordered by sample-floor priority (real fix, 2026-11-03): items whose
    niche is furthest below learn.transcript_clustering.MIN_SAMPLE_SIZE
    (40 annotated videos) come FIRST. Default insertion/FIFO order
    actively fought the project's own stated goal — only 5/27 niches
    ever cleared the clustering floor because whatever got downloaded
    earliest got annotated first, regardless of which niche most needed
    annotation volume. A niche with 0 annotated videos (e.g. a brand-new
    medspa/peptide watchlist added today) now jumps ahead of an
    already-saturated niche like fitness (344 annotated) that gains
    nothing from more annotation volume right now. Items with a NULL
    niche (not yet classified) sort last — same as any niche already
    past the floor, since there's no real floor-urgency signal for them
    either way. Ties within the same priority broken by insertion order
    (id), same as the original behavior, not left to be nondeterministic.
    """
    from qualume.db.connection import get_connection
    from qualume.learn.transcript_clustering import MIN_SAMPLE_SIZE

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            query = """
                WITH niche_annotated_counts AS (
                    SELECT wa.niche, count(DISTINCT ecd.external_content_item_id) AS annotated_count
                    FROM watched_accounts wa
                    LEFT JOIN external_content_items eci2 ON eci2.watched_account_id = wa.id
                    LEFT JOIN external_creative_dna ecd ON ecd.external_content_item_id = eci2.id
                        AND ecd.analysis_prompt_version = %s
                    GROUP BY wa.niche
                )
                SELECT eci.id
                FROM external_content_items eci
                JOIN watched_accounts wa ON wa.id = eci.watched_account_id
                LEFT JOIN niche_annotated_counts nac ON nac.niche = wa.niche
                WHERE eci.local_video_path IS NOT NULL
                  AND eci.id NOT IN (
                      SELECT external_content_item_id FROM external_creative_dna
                      WHERE analysis_prompt_version = %s
                  )
            """
            params: list = [DNA_ANALYSIS_PROMPT_VERSION, DNA_ANALYSIS_PROMPT_VERSION]
            if watched_account_id is not None:
                query += " AND eci.watched_account_id = %s"
                params.append(watched_account_id)
            # GREATEST(..., 0): a niche already past MIN_SAMPLE_SIZE gets
            # priority 0 (same as NULL/unclassified) rather than a
            # negative number — being FURTHER past the floor isn't
            # "more done", it's just equally not-urgent.
            query += """
                ORDER BY
                    CASE WHEN wa.niche IS NULL THEN 0
                         ELSE GREATEST(%s - COALESCE(nac.annotated_count, 0), 0)
                    END DESC,
                    eci.id
            """
            params.append(MIN_SAMPLE_SIZE)
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


def find_pending_content_type_classification(watched_account_id: str | None = None) -> list[str]:
    """TikTok external_content_items with a CURRENT-version annotation
    (same live DNA_ANALYSIS_PROMPT_VERSION discipline as find_pending_
    classification above) but no content_type set yet (migration 017).

    Scoped to platform='tiktok' only — meta_ads items are selling-by-
    definition and already get content_type seeded directly at ingest
    (see meta_ads_reader.parse_ad_entry), so they're never genuinely
    pending here; including them would just waste a classifier call
    re-confirming something already known for certain."""
    from qualume.db.connection import get_connection

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            query = """SELECT DISTINCT ecd.external_content_item_id
                       FROM external_creative_dna ecd
                       JOIN external_content_items eci ON eci.id = ecd.external_content_item_id
                       WHERE ecd.analysis_prompt_version = %s
                         AND eci.platform = 'tiktok'
                         AND eci.content_type IS NULL"""
            params: list = [DNA_ANALYSIS_PROMPT_VERSION]
            if watched_account_id is not None:
                query += " AND eci.watched_account_id = %s"
                params.append(watched_account_id)
            cur.execute(query, params)
            return [str(row[0]) for row in cur.fetchall()]
    finally:
        conn.close()


def _run_stage_over_items(items: list[str], fn, max_workers: int = 1) -> dict:
    """Run fn(item_id) for every item, isolating one item's failure from
    the rest. max_workers=1 (default) is the original strictly-sequential
    behavior — every existing caller keeps working unchanged.

    max_workers>1 (added 2026-10-30, per user's explicit request to reach
    real volume "as fast as possible"): runs items CONCURRENTLY via
    ThreadPoolExecutor — appropriate ONLY for stages whose real work is
    an I/O-bound network call with no TikTok-facing rate-limit concern
    (Gemini annotation, Claude classification). Deliberately NOT used for
    poll/download in run_ingestion_cycle below — those hit TikTok's own
    servers directly via yt-dlp, and parallelizing them risks real IP
    rate-limiting/soft-blocking, a materially different risk than
    parallelizing calls to Gemini/Claude's APIs."""
    succeeded = 0
    errors: list[tuple[str, str]] = []

    if max_workers <= 1:
        for item_id in items:
            try:
                fn(item_id)
                succeeded += 1
            except Exception as e:
                errors.append((item_id, f"{type(e).__name__}: {e}"))
        return {"succeeded": succeeded, "failed": len(errors), "errors": errors}

    from concurrent.futures import ThreadPoolExecutor, as_completed

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_item = {executor.submit(fn, item_id): item_id for item_id in items}
        for future in as_completed(future_to_item):
            item_id = future_to_item[future]
            try:
                future.result()
                succeeded += 1
            except Exception as e:
                errors.append((item_id, f"{type(e).__name__}: {e}"))

    return {"succeeded": succeeded, "failed": len(errors), "errors": errors}


def run_ingestion_cycle() -> dict:
    """Run one full cycle. Returns a dict with one key per stage
    (poll/download/annotation/classification/mining/clustering); a stage that raised
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

    import os

    # Real user decision 2026-11-01 ("stop downloading"): pause poll +
    # download permanently until explicitly re-enabled, while
    # annotation/classification/mining/clustering keep running on
    # whatever's already downloaded. Real reason: at the time this was
    # added, ~9755 videos were already downloaded but only ~10% had been
    # annotated — annotation throughput, not download volume, was the
    # actual bottleneck, so growing the download backlog further was
    # actively unhelpful. Checked live via env var (not a code comment
    # someone has to remember to revert), same "read live, don't
    # hardcode a decision" discipline as DNA_ANALYSIS_PROMPT_VERSION.
    # Opt-in: unset (the default) means poll/download run exactly as
    # before, no silent new default for every existing deployment.
    if os.environ.get("QUALUME_PAUSE_INGESTION"):
        result["poll"] = {"skipped": "QUALUME_PAUSE_INGESTION is set"}
        result["download"] = {"skipped": "QUALUME_PAUSE_INGESTION is set"}
    else:
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
        result["annotation"] = _run_stage_over_items(
            pending_annotation, annotate_and_store, max_workers=ANNOTATION_CONCURRENCY
        )
    except Exception as e:
        result["annotation"] = {"error": f"{type(e).__name__}: {e}"}

    try:
        pending_classification = find_pending_classification()
        result["classification"] = _run_stage_over_items(
            pending_classification, classify_and_store_external, max_workers=CLASSIFICATION_CONCURRENCY
        )
    except Exception as e:
        result["classification"] = {"error": f"{type(e).__name__}: {e}"}

    try:
        # content_type classification — added 2026-11-02. Same stage
        # shape/isolation as classification above, but TikTok-only
        # (find_pending_content_type_classification already excludes
        # meta_ads, which gets content_type seeded for free at ingest —
        # see meta_ads_reader.parse_ad_entry). This closes the real gap
        # that made niche-scoped/content_type-scoped clustering
        # impossible for TikTok content: without this stage running,
        # content_type stays NULL forever for every TikTok video.
        pending_content_type = find_pending_content_type_classification()
        result["content_type_classification"] = _run_stage_over_items(
            pending_content_type, classify_and_store_content_type,
            max_workers=CLASSIFICATION_CONCURRENCY,
        )
    except Exception as e:
        result["content_type_classification"] = {"error": f"{type(e).__name__}: {e}"}

    try:
        # tiktok_reels is the only real workflow_type with external data
        # today (watched_accounts is TikTok-only so far) — hardcoded here,
        # not read from anywhere dynamic, same as run_batch.py's own
        # workflow_type handling; becomes a real parameter the day a
        # second external-reading workflow (e.g. Instagram) exists.
        result["mining"] = run_mining_pass(applies_to_workflow_type="tiktok_reels")
    except Exception as e:
        result["mining"] = {"error": f"{type(e).__name__}: {e}"}

    try:
        # Phase B (clustering) — 6th stage, added 2026-10-30. Same
        # tiktok_reels hardcode as the mining stage above, same reasoning.
        # Runs independently of the mining stage (Phase A/declared-
        # taxonomy) — a Voyage outage or embedding failure here must not
        # block the already-proven mining pass, and vice versa.
        result["clustering"] = run_cluster_pass(applies_to_workflow_type="tiktok_reels")
    except Exception as e:
        result["clustering"] = {"error": f"{type(e).__name__}: {e}"}

    try:
        # Visual clustering — 7th stage, added 2026-10-31 per user's
        # explicit request ("we need to use direct video embedding too
        # along with our transcript embedding"). Kept as a SEPARATE
        # stage/pass from the transcript clustering above, not fused —
        # the user explicitly wants to compare real cluster quality from
        # each side independently before any fusion decision. A Voyage
        # multimodal outage or frame-extraction failure here must not
        # block transcript clustering or mining, and vice versa.
        result["visual_clustering"] = run_visual_cluster_pass(applies_to_workflow_type="tiktok_reels")
    except Exception as e:
        result["visual_clustering"] = {"error": f"{type(e).__name__}: {e}"}

    return result
