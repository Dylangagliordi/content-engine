"""Fetch real (annotation_dict, performance_ratio) training rows for
learn.prediction.train_predictor from the live DB, scoped per niche OR
pooled across many/all niches (see fetch_training_rows_pooled).

Real gap this closes (2026-11-03, "ensure annotations get sped up" led
to confirming enough real volume now exists): train_predictor has
always needed (dict, float) pairs, but nothing in the codebase ever
assembled real pairs from the DB, so the random forest scaffold had
literally never been trained on real data despite 7 niches now clearing
the 50-row floor (fitness=344, women's health=335, food=149, mom
tok=132, small business=62, general health=58, menopause=57 —
confirmed live via a direct DB query).

Also fetches the already-computed transcript_embeddings/visual_embeddings
rows for each candidate item (added for the full embedding merge: "how
can we use all [text/transcript/visual embeddings] in the prediction
model and have them all be interpretable") — these are real, precomputed
Voyage vectors created independently by learn.transcript_embedding/
learn.visual_embedding for HDBSCAN clustering, not re-embedded here.
Coverage is real and incomplete (not every annotated item has been
through either embedding pipeline yet — those run on their own
schedule); an item with no stored vector gets None for that field, never
a fabricated placeholder.

POOLED TRAINING (added 2026-11-05, real fix for a real data-scarcity
problem): fetch_training_rows(niche) stays niche-REQUIRED for mining/
clustering-style use (a cross-niche pool there would corrupt a
statistical claim's meaning — comparing a prenatal-vitamin video
against a Quran-recitation video has no real meaning). But
learn.baseline.performance_vs_baseline ALREADY account-relativizes the
target (ratio_to_median) before niche enters the picture, so the
justification for ALSO hard-partitioning the PREDICTIVE model by niche
doesn't hold the same way — confirmed live: of 27 niches with any real
annotated+baselined data, only 8 clear even a 50-row floor; 19 sit at
n=2-20, unusable by a per-niche model but real data a pooled model can
still learn from (treating niche as a categorical FEATURE, letting
RandomForest's own splits discover niche-interaction effects natively).
fetch_training_rows_pooled(niches) returns the same annotation shape as
fetch_training_rows, PLUS a "niche" key on every annotation dict.
"""
from qualume.annotate.gemini_watcher import ANALYSIS_PROMPT_VERSION as DNA_VERSION
from qualume.learn.baseline import performance_vs_baseline
from qualume.learn.transcript_embedding import build_embeddable_text


def _fetch_candidate_items(niche: str | None, niches: list[str] | None = None) -> list[dict]:
    """Every (external_content_items.id, duration_seconds, DNA fields,
    stored embedding vectors, real niche) row, at the CURRENT annotation
    version (same live-version discipline as orchestrate.find_pending_
    annotation — never a hardcoded version string). Does NOT filter on
    whether a performance snapshot exists yet; the fetch_training_rows*
    functions below do that per-item via performance_vs_baseline.

    Exactly one of niche (single string, exact match) / niches (list,
    ANY match) / neither (no niche filter at all — every real niche) is
    meaningful at a time; pass only one. Always returns the item's real
    niche in the result dict so callers needing it (the pooled path) can
    tag the annotation without a second query."""
    from qualume.db.connection import get_connection

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            query = """SELECT eci.id, eci.duration_seconds,
                              ecd.hook_type, ecd.format_tag, ecd.energy_level,
                              ecd.shot_breakdown, ecd.full_transcript, ecd.hook_text,
                              te.embedding, ve.embedding, wa.niche
                       FROM external_content_items eci
                       JOIN watched_accounts wa ON wa.id = eci.watched_account_id
                       JOIN external_creative_dna ecd ON ecd.external_content_item_id = eci.id
                       LEFT JOIN transcript_embeddings te ON te.external_content_item_id = eci.id
                       LEFT JOIN visual_embeddings ve ON ve.external_content_item_id = eci.id
                       WHERE ecd.analysis_prompt_version = %s"""
            params: list = [DNA_VERSION]
            if niche is not None:
                query += " AND wa.niche = %s"
                params.append(niche)
            elif niches is not None:
                query += " AND wa.niche = ANY(%s)"
                params.append(niches)
            cur.execute(query, params)
            return [
                {
                    "item_id": str(row[0]),
                    "video_duration_seconds": row[1],
                    "hook_type": row[2],
                    "format_tag": row[3],
                    "energy_level": row[4],
                    "shot_breakdown": row[5] or [],
                    "full_transcript": row[6] or [],
                    "hook_text": row[7],
                    "transcript_embedding_vector": row[8] if row[8] is not None else None,
                    "visual_embedding_vector": row[9] if row[9] is not None else None,
                    "niche": row[10],
                }
                for row in cur.fetchall()
            ]
    finally:
        conn.close()


def _build_annotation(c: dict, include_niche: bool) -> dict:
    """Shared row->annotation-dict mapping used by both fetch_training_rows
    and fetch_training_rows_pooled — kept as one function so the two
    paths can never silently drift in shape."""
    transcript_vector = c["transcript_embedding_vector"]
    if transcript_vector is not None:
        transcript_vector = list(transcript_vector)  # psycopg2 JSONB -> already-parsed list

    transcript_text = None
    if transcript_vector is not None:
        try:
            transcript_text = build_embeddable_text(c["hook_text"], c["full_transcript"])
        except ValueError:
            # Real edge case: the embedding was created from a different
            # hook_text/full_transcript than what's on the CURRENT
            # (latest) annotation row (re-annotation changed the text
            # after the embedding was last computed). Fall back to no
            # text for interpretability rather than show a stale mismatch
            # — the vector itself still trains, just without a keyword
            # contrast for this one item.
            transcript_text = None

    visual_vector = c["visual_embedding_vector"]
    if visual_vector is not None:
        visual_vector = list(visual_vector)

    annotation = {
        "hook_type": c["hook_type"],
        "format_tag": c["format_tag"],
        "energy_level": c["energy_level"],
        "shot_breakdown": c["shot_breakdown"],
        "full_transcript": c["full_transcript"],
        "video_duration_seconds": c["video_duration_seconds"],
        "hook_text": c["hook_text"],
        "transcript_embedding_vector": transcript_vector,
        "transcript_embeddable_text": transcript_text,
        "visual_embedding_vector": visual_vector,
    }
    if include_niche:
        annotation["niche"] = c["niche"]
    return annotation


def fetch_training_rows(niche: str) -> list[tuple[dict, float]]:
    """Real (annotation_dict, performance_ratio) pairs for ONE niche,
    ready to pass straight into learn.prediction.train_predictor. See
    module docstring for when to use this (mining/clustering-style,
    niche-required) vs fetch_training_rows_pooled (the predictive-model
    default going forward).

    An item with no real performance snapshot yet (or whose account has
    no median to compare against) is skipped, not fabricated — same
    "refuse rather than guess" discipline as every other learn.* module."""
    candidates = _fetch_candidate_items(niche=niche)
    rows: list[tuple[dict, float]] = []
    for c in candidates:
        try:
            perf = performance_vs_baseline(c["item_id"])
        except ValueError:
            continue
        rows.append((_build_annotation(c, include_niche=False), float(perf["ratio_to_median"])))
    return rows


def fetch_training_rows_pooled(niches: list[str] | None = None) -> list[tuple[dict, float]]:
    """Real (annotation_dict, performance_ratio) pairs pooled across
    MULTIPLE niches (or every real niche, if niches=None) — the
    predictive-model default going forward (see module docstring for the
    real data-scarcity reasoning). Every annotation dict carries a real
    "niche" key (never fabricated — straight from watched_accounts),
    letting learn.prediction.train_predictor treat niche as a categorical
    feature instead of training 8+ separate starved models.

    Same skip-not-fabricate discipline as fetch_training_rows for items
    with no real performance baseline."""
    candidates = _fetch_candidate_items(niche=None, niches=niches)
    rows: list[tuple[dict, float]] = []
    for c in candidates:
        try:
            perf = performance_vs_baseline(c["item_id"])
        except ValueError:
            continue
        rows.append((_build_annotation(c, include_niche=True), float(perf["ratio_to_median"])))
    return rows
