"""Automated clustering-pass orchestrator — the Phase B equivalent of
learn.run_mining_pass, per this session's explicit design discussion.

Embeds any video with a transcript but no embedding yet, runs HDBSCAN
clustering (learn.transcript_clustering), auto-labels each statistically
significant cluster with a cheap Claude call (no human manually reads
transcripts to name a cluster in an automated pipeline), and persists
findings into insights_external tagged source='transcript_clustering' —
distinguishable from the declared-taxonomy mining's source=
'internal_mining' rows in the same table.

niche is attached to each finding as a lightweight INFORMATIONAL tag
(applies_to_segment.niches — every distinct niche among the accounts
contributing to that cluster), per explicit user decision this session:
niche stays OUT of the embedding/clustering math itself, so this whole
pipeline stays portable to a different business's accounts later without
assuming "our niche" anywhere in the clustering logic.

Real structural difference from run_mining_pass's upsert-by-stable-
identity discipline: HDBSCAN cluster IDs are NOT stable across re-runs —
the same underlying group of videos might be "cluster 1" this run and
"cluster 3" (or split across two clusters) next run as new videos arrive.
There is no stable identity to upsert against. So every pass RETIRES
ALL previous 'proposed'-tier source='transcript_clustering' insights
before proposing fresh ones from this pass's real clusters — honest,
since clusters really are re-discovered each run, not persistent named
things with a continuous identity.
"""
from qualume.learn.multiple_comparison import benjamini_hochberg
from qualume.learn.transcript_clustering import cluster_transcripts
from qualume.learn.transcript_embedding import embed_and_store
from qualume.learn.validation import propose_external_insight, retire_stale_external_insights
from qualume.llm.pick_llm import pick_llm

SIGNIFICANCE_THRESHOLD = 0.05

LABEL_PROMPT = """\
You are looking at {n} TikTok video transcripts that a clustering algorithm \
grouped together because they share real structural/semantic similarity — \
NOT because of a shared topic keyword, a real pattern in how they're \
constructed or what they're doing.

Write ONE short, concrete phrase (5-10 words) describing what these videos \
have in common. Be specific about the STRUCTURE or PURPOSE of the content \
(e.g. "live-shopping event announcements", "before/after transformation \
reveals", "myth-busting pattern-interrupt openers") — not a vague \
generality like "health content" or "promotional videos".

TRANSCRIPTS:
{transcripts}

Respond with ONLY the short phrase, nothing else.\
"""


def _find_pending_embedding() -> list[str]:
    """external_content_items with a real external_creative_dna row
    (something to embed from) but no transcript_embeddings row yet —
    same "find what's genuinely missing" shape as orchestrate.
    find_pending_annotation/find_pending_classification."""
    from qualume.db.connection import get_connection

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """SELECT DISTINCT ecd.external_content_item_id
                   FROM external_creative_dna ecd
                   WHERE ecd.external_content_item_id NOT IN (
                       SELECT external_content_item_id FROM transcript_embeddings
                   )"""
            )
            return [str(row[0]) for row in cur.fetchall()]
    finally:
        conn.close()


def _fetch_niche_summary(item_ids: list[str]) -> set[str]:
    """Every distinct real niche (watched_accounts.niche, excluding NULL —
    an unset niche contributes nothing, not a fake 'unknown' label) among
    the accounts that own these items. Purely informational — this is the
    ONLY place niche touches this module; it never enters the clustering
    math itself."""
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


def _fetch_transcripts(item_ids: list[str], limit: int = 10) -> list[str]:
    """Real transcript text for up to `limit` of this cluster's videos —
    capped since a 50-video cluster doesn't need every transcript sent to
    Claude to be labeled accurately, and it keeps the labeling call cheap
    (this is a 'low' tier task, not meant to be an expensive one)."""
    from qualume.db.connection import get_connection

    from qualume.learn.transcript_embedding import build_embeddable_text

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """SELECT DISTINCT ON (external_content_item_id)
                       external_content_item_id, hook_text, full_transcript
                   FROM external_creative_dna
                   WHERE external_content_item_id = ANY(%s::uuid[])
                   ORDER BY external_content_item_id, analyzed_at DESC""",
                (item_ids[:limit],),
            )
            rows = cur.fetchall()
    finally:
        conn.close()

    texts = []
    for _, hook_text, full_transcript in rows:
        try:
            texts.append(build_embeddable_text(hook_text, full_transcript))
        except ValueError:
            continue
    return texts


def _label_cluster(item_ids: list[str]) -> str:
    """One cheap Claude call ('low' tier, same as copy_classifier) to
    turn a cluster of transcripts into a short, concrete, human-readable
    label — the step that makes this genuinely automated instead of
    requiring a human to manually read transcripts after every run."""
    transcripts = _fetch_transcripts(item_ids)
    if not transcripts:
        return "unlabeled cluster (no transcript text available)"

    prompt = LABEL_PROMPT.format(
        n=len(transcripts),
        transcripts="\n---\n".join(transcripts),
    )
    llm = pick_llm("low")
    response = llm.invoke(prompt)
    # response.text (NOT str(response.content)) — real bug found 2026-11-03
    # switching 'low' from Claude to Gemini: Claude's AIMessage.content is a
    # plain string, but Gemini's (via langchain-google-genai) is a list of
    # content blocks ([{'type': 'text', 'text': ..., 'extras': {...}}]), so
    # str(response.content) stringified the whole list (including an opaque
    # signature blob) into the stored label instead of the clean text.
    # .text is langchain-core's own provider-agnostic accessor, confirmed
    # live to normalize both shapes correctly.
    return response.text.strip()


def run_cluster_pass(applies_to_workflow_type: str, niche: str | None = None) -> dict:
    """Run one full Phase B pass: embed pending videos, cluster, label
    and propose significant findings, retire every previous cluster-
    sourced proposal (clusters have no stable identity across runs, so
    there's nothing to upsert-match against — a clean retire-then-
    propose each time is the honest behavior).

    niche (added 2026-11-02, real fix for a real problem): when given,
    scopes cluster_transcripts itself to only this niche's embeddings
    (so HDBSCAN clusters within one real audience's vector space, not
    a 10-niche mixed pool where "performed well" isn't comparable) AND
    tags every proposed/retired insight with it, so retirement for a
    niche-scoped pass only touches that niche's own rows (see
    validation.retire_stale_external_insights' own niche scoping).
    Still None by default — unscoped, original cross-niche behavior,
    for any caller not yet looping per-niche.

    Returns: {embedding: {succeeded, failed}, clustering: {...}}. Mirrors
    orchestrate.run_ingestion_cycle's stage-isolation shape — one
    embedding failure doesn't block the rest, a clustering-sample-floor
    miss is reported honestly rather than crashing the whole pass."""
    pending = _find_pending_embedding()
    embed_succeeded, embed_failed, embed_skipped_no_transcript = 0, 0, 0
    for item_id in pending:
        try:
            embed_and_store(item_id)
            embed_succeeded += 1
        except ValueError:
            # build_embeddable_text's own precondition failure — a real,
            # expected shape (e.g. a product-only/voiceover-over-broll
            # video Gemini never produced spoken copy for), NOT a real
            # failure. Reporting this as "failed" is misleading: a real
            # production run showed 25 "failed" items that were actually
            # just videos with nothing real to embed from, caught by
            # reviewing real output rather than assumed correct.
            embed_skipped_no_transcript += 1
        except Exception:
            embed_failed += 1

    result = {"embedding": {
        "succeeded": embed_succeeded, "failed": embed_failed,
        "skipped_no_transcript": embed_skipped_no_transcript,
    }}

    try:
        cluster_result = cluster_transcripts(niche=niche)
    except ValueError as e:
        result["clustering"] = {"skipped_reason": f"below sample floor: {e}"}
        return result

    proposed = 0
    # BH correction (added 2026-10-30, same reasoning as learn.
    # run_mining_pass): testable clusters (p_value not None — clusters
    # below min_cluster_size already report None and are never part of
    # this decision either way) are judged TOGETHER as one batch, not
    # each against a flat threshold independently.
    testable_clusters = [c for c in cluster_result["clusters"] if c["p_value"] is not None]
    bh_significant = benjamini_hochberg(
        [c["p_value"] for c in testable_clusters], fdr=SIGNIFICANCE_THRESHOLD
    )
    significant_clusters = [
        c for c, is_sig in zip(testable_clusters, bh_significant) if is_sig
    ]
    skipped_not_significant = len(testable_clusters) - len(significant_clusters)

    # RETIRE FIRST, then propose — real bug fixed 2026-10-30: retiring
    # AFTER the propose loop wiped out this exact pass's own fresh
    # findings (clusters have no stable identity, so retire_all=True
    # retires every proposed row regardless of when it was created —
    # doing that after proposing meant the pass always ended at 0 live
    # proposed clustering insights, no matter what it just found).
    retired = retire_stale_external_insights(
        applies_to_workflow_type=applies_to_workflow_type,
        still_significant_identities=[],  # clusters have no stable identity — retire_all=True is what actually makes this real
        source="transcript_clustering",
        retire_all=True,  # bypasses identity matching entirely — see validation.py's real-bug-fix docstring (2026-10-30)
        niche=niche,
    )

    for cluster in significant_clusters:
        label = _label_cluster(cluster["item_ids"])
        niches = sorted(_fetch_niche_summary(cluster["item_ids"]))
        direction = "outperforms" if cluster["median_ratio_in_cluster"] > cluster["median_ratio_outside"] else "underperforms"

        statement = (
            f"Videos clustering structurally as {label!r} {direction} videos outside "
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
                "source": "transcript_clustering",
                "label": label,
                "niches": niches,
            },
            source="transcript_clustering",
            niche=niche,
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
