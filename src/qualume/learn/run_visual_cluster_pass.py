"""Automated visual-clustering-pass orchestrator — the visual-structure
counterpart to learn.run_cluster_pass, same shape per the user's explicit
decision this session (2026-10-31): visual clustering runs as its own
separate pass, proposing insights tagged source='visual_clustering' so
real signal quality from each side (transcript vs. visual) stays
independently comparable before any fusion decision is made.

Real difference from the transcript version's _fetch_transcripts/
LABEL_PROMPT: there is no spoken-copy text available to show Claude for
labeling a visual cluster — _fetch_shot_descriptions instead pulls each
member video's REAL structured shot fields (shot_type, setting, lighting,
color_temperature — already computed by Gemini annotation, nothing
re-derived here) as the honest visual-equivalent signal to label from.
"""
from qualume.learn.multiple_comparison import benjamini_hochberg
from qualume.learn.validation import propose_external_insight, retire_stale_external_insights
from qualume.learn.visual_clustering import cluster_visual
from qualume.learn.visual_embedding import embed_and_store
from qualume.llm.pick_llm import pick_llm

SIGNIFICANCE_THRESHOLD = 0.05

# Real throughput finding this session (2026-10-31): a single visual
# embed (frame extraction + Voyage multimodal_embed call) took ~22s end
# to end run serially — the real 8857-video backlog would take ~54 hours
# at that rate. Embedding is an I/O-bound network call with no
# TikTok-facing rate-limit risk, same reasoning as orchestrate.py's
# ANNOTATION_CONCURRENCY/CLASSIFICATION_CONCURRENCY — safe to run many
# at once.
#
# LOWERED from 8 to 4 same day, same reasoning as orchestrate.py's own
# 8->4 reduction: running this pipeline's 8 ffmpeg-subprocess workers
# alongside annotation/classification's own 8+8 workers drove physical
# memory down to ~111MB free and caused a real OrbStack/Docker VM hang
# (Postgres connection timeouts, required a full restart to recover).
EMBEDDING_CONCURRENCY = 4

LABEL_PROMPT = """\
You are looking at {n} TikTok video shot descriptions that a clustering \
algorithm grouped together because their VISUAL structure (shot types, \
settings, lighting, pacing) is similar — NOT because they share a topic.

Write ONE short, concrete phrase (5-10 words) describing what these \
videos have in common VISUALLY (e.g. "fast-cut product close-up demos", \
"single static talking-head shot", "warm-lit home kitchen setting") — \
not a vague generality like "promotional videos".

SHOT DESCRIPTIONS:
{descriptions}

Respond with ONLY the short phrase, nothing else.\
"""


def _find_pending_visual_embedding() -> list[str]:
    """external_content_items with a real local_video_path AND a real
    external_creative_dna row (something to size the frame budget from)
    but no visual_embeddings row yet — same "find what's genuinely
    missing" shape as every other pending-finder in this codebase."""
    from qualume.db.connection import get_connection

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """SELECT DISTINCT eci.id
                   FROM external_content_items eci
                   JOIN external_creative_dna ecd ON ecd.external_content_item_id = eci.id
                   WHERE eci.local_video_path IS NOT NULL
                     AND eci.id NOT IN (
                         SELECT external_content_item_id FROM visual_embeddings
                     )"""
            )
            return [str(row[0]) for row in cur.fetchall()]
    finally:
        conn.close()


def _fetch_niche_summary(item_ids: list[str]) -> set[str]:
    """Identical to run_cluster_pass._fetch_niche_summary — same
    informational-only niche tagging, duplicated rather than imported
    since this module has no dependency on the transcript pipeline."""
    from qualume.db.connection import get_connection

    if not item_ids:
        return set()

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """SELECT DISTINCT wa.niche
                   FROM external_content_items eci
                   JOIN watched_accounts wa ON wa.id = eci.watched_account_id
                   WHERE eci.id = ANY(%s::uuid[]) AND wa.niche IS NOT NULL""",
                (item_ids,),
            )
            return {row[0] for row in cur.fetchall()}
    finally:
        conn.close()


def _fetch_shot_descriptions(item_ids: list[str], limit: int = 10) -> list[str]:
    """Real shot_breakdown structured fields for up to `limit` of this
    cluster's videos, turned into a short plain-text description per
    video — the visual-equivalent of transcript text for labeling
    purposes. Capped at `limit` for the same cheap-labeling-call reason
    as the transcript pipeline's own _fetch_transcripts."""
    from qualume.db.connection import get_connection

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """SELECT DISTINCT ON (external_content_item_id)
                       external_content_item_id, shot_breakdown
                   FROM external_creative_dna
                   WHERE external_content_item_id = ANY(%s::uuid[])
                   ORDER BY external_content_item_id, analyzed_at DESC""",
                (item_ids[:limit],),
            )
            rows = cur.fetchall()
    finally:
        conn.close()

    descriptions = []
    for _, shot_breakdown in rows:
        if not shot_breakdown:
            continue
        parts = [
            f"{s.get('shot_type', '?')} shot, {s.get('setting', '?')} setting, "
            f"{s.get('lighting', '?')} lighting"
            for s in shot_breakdown
        ]
        descriptions.append("; ".join(parts))
    return descriptions


def _label_cluster(item_ids: list[str]) -> str:
    """One cheap Claude call ('low' tier) to turn a cluster's real shot
    descriptions into a short, human-readable label — same automation
    reasoning as run_cluster_pass._label_cluster, just fed visual shot
    fields instead of transcript text."""
    descriptions = _fetch_shot_descriptions(item_ids)
    if not descriptions:
        return "unlabeled cluster (no shot data available)"

    prompt = LABEL_PROMPT.format(
        n=len(descriptions),
        descriptions="\n---\n".join(descriptions),
    )
    llm = pick_llm("low")
    response = llm.invoke(prompt)
    return str(response.content).strip()


def run_visual_cluster_pass(applies_to_workflow_type: str) -> dict:
    """Run one full visual clustering pass: embed pending videos'
    frames, cluster, label and propose significant findings, retire
    every previous visual-cluster-sourced proposal first (HDBSCAN
    cluster IDs have no stable identity across runs — identical real
    bug-fix discipline as run_cluster_pass: retire BEFORE propose, never
    after, or a pass retires its own fresh findings).

    Returns: {embedding: {succeeded, failed, skipped_no_video},
    clustering: {...}}."""
    pending = _find_pending_visual_embedding()
    embed_succeeded, embed_failed, embed_skipped_no_video = 0, 0, 0

    from concurrent.futures import ThreadPoolExecutor, as_completed

    with ThreadPoolExecutor(max_workers=EMBEDDING_CONCURRENCY) as executor:
        future_to_item = {executor.submit(embed_and_store, item_id): item_id for item_id in pending}
        for future in as_completed(future_to_item):
            try:
                future.result()
                embed_succeeded += 1
            except ValueError:
                # embed_and_store's own precondition failure (no
                # local_video_path or no external_creative_dna row yet) — a
                # real, expected shape, NOT a failure, same honesty
                # discipline as run_cluster_pass's skipped_no_transcript.
                embed_skipped_no_video += 1
            except Exception:
                embed_failed += 1

    result = {"embedding": {
        "succeeded": embed_succeeded, "failed": embed_failed,
        "skipped_no_video": embed_skipped_no_video,
    }}

    try:
        cluster_result = cluster_visual()
    except ValueError as e:
        result["clustering"] = {"skipped_reason": f"below sample floor: {e}"}
        return result

    proposed = 0
    testable_clusters = [c for c in cluster_result["clusters"] if c["p_value"] is not None]
    bh_significant = benjamini_hochberg(
        [c["p_value"] for c in testable_clusters], fdr=SIGNIFICANCE_THRESHOLD
    )
    significant_clusters = [
        c for c, is_sig in zip(testable_clusters, bh_significant) if is_sig
    ]
    skipped_not_significant = len(testable_clusters) - len(significant_clusters)

    retired = retire_stale_external_insights(
        applies_to_workflow_type=applies_to_workflow_type,
        still_significant_identities=[],
        source="visual_clustering",
        retire_all=True,
    )

    for cluster in significant_clusters:
        label = _label_cluster(cluster["item_ids"])
        niches = sorted(_fetch_niche_summary(cluster["item_ids"]))
        direction = "outperforms" if cluster["median_ratio_in_cluster"] > cluster["median_ratio_outside"] else "underperforms"

        statement = (
            f"Videos clustering visually as {label!r} {direction} videos outside "
            f"that cluster (median {cluster['median_ratio_in_cluster']:.2f}x account "
            f"baseline vs {cluster['median_ratio_outside']:.2f}x, n={cluster['size']})."
        )
        propose_external_insight(
            statement=statement,
            effect_size=min(cluster["median_ratio_in_cluster"], cluster["median_ratio_outside"])
            / max(cluster["median_ratio_in_cluster"], cluster["median_ratio_outside"]),
            sample_size=cluster["size"],
            p_value=cluster["p_value"],
            applies_to_workflow_type=applies_to_workflow_type,
            applies_to_segment={
                "source": "visual_clustering",
                "label": label,
                "niches": niches,
            },
            source="visual_clustering",
        )
        proposed += 1

    result["clustering"] = {
        "n_total": cluster_result["n_total"],
        "n_scored": cluster_result["n_scored"],
        "n_noise": cluster_result["n_noise"],
        "proposed": proposed,
        "skipped_not_significant": skipped_not_significant,
        "retired": retired,
    }
    return result
