"""Fetch real (annotation_dict, performance_ratio) training rows for
learn.prediction.train_predictor from the live DB, scoped per niche.

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

niche-scoped (required, not optional, unlike mining/clustering's
niche=None default) — this module's whole reason to exist is per-niche
training; a cross-niche pool would corrupt every niche's model with
another niche's unrelated performance scale, same reasoning as
transcript_clustering's niche isolation.
"""
from qualume.annotate.gemini_watcher import ANALYSIS_PROMPT_VERSION as DNA_VERSION
from qualume.learn.baseline import performance_vs_baseline
from qualume.learn.transcript_embedding import build_embeddable_text


def _fetch_candidate_items(niche: str) -> list[dict]:
    """Every (external_content_items.id, duration_seconds, DNA fields,
    stored embedding vectors) row for one niche, at the CURRENT
    annotation version (same live-version discipline as
    orchestrate.find_pending_annotation — never a hardcoded version
    string). Does NOT filter on whether a performance snapshot exists
    yet; fetch_training_rows below does that per-item via
    performance_vs_baseline, so a missing baseline is skipped there
    rather than silently excluded by a JOIN here (keeps this function's
    own behavior easy to reason about independently). Same discipline
    applied to transcript_embeddings/visual_embeddings below (LEFT JOIN,
    not INNER — an item missing either embedding is still a valid
    candidate, just with None for that vector)."""
    from qualume.db.connection import get_connection

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """SELECT eci.id, eci.duration_seconds,
                          ecd.hook_type, ecd.format_tag, ecd.energy_level,
                          ecd.shot_breakdown, ecd.full_transcript, ecd.hook_text,
                          te.embedding, ve.embedding
                   FROM external_content_items eci
                   JOIN watched_accounts wa ON wa.id = eci.watched_account_id
                   JOIN external_creative_dna ecd ON ecd.external_content_item_id = eci.id
                   LEFT JOIN transcript_embeddings te ON te.external_content_item_id = eci.id
                   LEFT JOIN visual_embeddings ve ON ve.external_content_item_id = eci.id
                   WHERE wa.niche = %s AND ecd.analysis_prompt_version = %s""",
                (niche, DNA_VERSION),
            )
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
                }
                for row in cur.fetchall()
            ]
    finally:
        conn.close()


def fetch_training_rows(niche: str) -> list[tuple[dict, float]]:
    """Real (annotation_dict, performance_ratio) pairs for one niche,
    ready to pass straight into learn.prediction.train_predictor.

    annotation_dict matches build_feature_vector's expected shape
    exactly: hook_type, format_tag, energy_level, shot_breakdown,
    full_transcript, video_duration_seconds, hook_text,
    transcript_embedding_vector (list[float] or None),
    visual_embedding_vector (list[float] or None), and
    transcript_embeddable_text (the exact text that was embedded into
    transcript_embedding_vector, reconstructed via the same
    build_embeddable_text join learn.transcript_embedding itself uses —
    needed by build_feature_vector's transcript interpretability, not a
    new/different text).

    An item with no real performance snapshot yet (or whose account has
    no median to compare against) is skipped, not fabricated — same
    "refuse rather than guess" discipline as every other learn.* module
    (performance_vs_baseline already raises ValueError for exactly
    this case; caught and skipped here, not propagated, since one
    missing baseline among hundreds of candidates isn't a real failure
    of the whole fetch)."""
    candidates = _fetch_candidate_items(niche)

    rows: list[tuple[dict, float]] = []
    for c in candidates:
        try:
            perf = performance_vs_baseline(c["item_id"])
        except ValueError:
            continue

        transcript_vector = c["transcript_embedding_vector"]
        if transcript_vector is not None:
            # psycopg2 returns JSONB as an already-parsed Python list.
            transcript_vector = list(transcript_vector)

        transcript_text = None
        if transcript_vector is not None:
            try:
                transcript_text = build_embeddable_text(c["hook_text"], c["full_transcript"])
            except ValueError:
                # Real edge case: the embedding was created from a
                # different hook_text/full_transcript than what's on the
                # CURRENT (latest) annotation row (re-annotation changed
                # the text after the embedding was last computed). Rather
                # than silently show stale/mismatched "original text" for
                # interpretability, fall back to no text at all for this
                # item's transcript axis — the vector itself is still used
                # for prediction, just without a text contrast for it.
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
        rows.append((annotation, float(perf["ratio_to_median"])))

    return rows
