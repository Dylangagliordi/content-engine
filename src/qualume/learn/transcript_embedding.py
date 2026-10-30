"""Transcript embedding — the real foundation for Phase B (emergent
pattern-finding via clustering), per this session's explicit design
discussion.

build_embeddable_text() is deliberately format-agnostic (plain text in,
plain text out, no DB dependency) — same discipline as
copy_classifier.classify_copy. embed_video_text() wraps the real Voyage
API call (voyage-3.5, 1024-dim — confirmed live against real text before
this module was built). embed_and_store() is the DB-aware wrapper,
mirrors copy_classifier.classify_and_store_external's shape: pull a
video's already-stored transcript/hook_text, embed it, persist a row —
skipping the real API call (cost + latency) when the underlying text
hasn't changed since the last embed (source_text_hash dedup, same
"don't redo real work" discipline as find_pending_annotation).

Voyage, not Gemini embeddings or a local model, per this session's
explicit decision: keeps the project's "Claude-only, one scoped Gemini
exception for video-watching" discipline intact (doesn't quietly expand
Gemini's role), avoids a heavy new local torch dependency for a
low-volume exploratory feature, and Voyage is purpose-built for
embeddings specifically (vs. a secondary feature on a generative model),
which matters for real semantic-structure clustering quality rather than
just keyword-overlap clustering.
"""
import hashlib
import json

EMBEDDING_MODEL = "voyage-3.5"

_voyage_client = None


def _get_voyage_client():
    """Lazy singleton — avoids constructing a real client (and reading
    VOYAGE_API_KEY) at import time, same pattern as pick_llm.py's lazy
    LLM construction. Tests patch this function directly rather than
    hitting the real API."""
    global _voyage_client
    if _voyage_client is None:
        import os

        import voyageai
        from dotenv import load_dotenv

        load_dotenv()
        _voyage_client = voyageai.Client(api_key=os.environ["VOYAGE_API_KEY"])
    return _voyage_client


def build_embeddable_text(hook_text: str | None, full_transcript: list | None) -> str:
    """Join hook_text + full_transcript into one plain string for
    embedding. Deliberately NOT including on_screen_text/caption (unlike
    copy_classifier's prompt) — embeddings are meant to capture the
    SPOKEN persuasion structure specifically, matching what mining already
    treats as the video's core copy; on-screen text overlays and captions
    are a separate, noisier signal that would dilute the semantic
    structure this is trying to cluster on.

    Raises ValueError if there's nothing real to embed — same "fail loud
    on missing input" discipline as classify_copy."""
    parts = []
    if hook_text:
        parts.append(hook_text.strip())
    if full_transcript:
        transcript_text = " ".join(
            seg.get("text", "") if isinstance(seg, dict) else str(seg)
            for seg in full_transcript
        ).strip()
        if transcript_text:
            parts.append(transcript_text)

    if not parts:
        raise ValueError(
            "build_embeddable_text needs at least one of hook_text or "
            "full_transcript with real content — nothing to embed"
        )
    return " ".join(parts)


def embed_video_text(text: str) -> list[float]:
    """Real Voyage API call. input_type='document' (not 'query') — these
    embeddings are being stored for later clustering/comparison among
    themselves, not used to search against a query string."""
    client = _get_voyage_client()
    result = client.embed([text], model=EMBEDDING_MODEL, input_type="document")
    return list(result.embeddings[0])


def embed_and_store(external_content_item_id: str) -> str:
    """Pull a video's already-stored transcript/hook_text (latest
    external_creative_dna row, any version — same "latest wins" discipline
    as classify_and_store_external), embed it, upsert a transcript_
    embeddings row.

    Skips the real API call and returns the EXISTING row id when the
    source text hasn't changed since the last embed (source_text_hash
    match) — real cost/latency discipline, same shape as
    find_pending_annotation only touching genuinely-pending rows.

    Raises ValueError if the item has no external_creative_dna row at all
    (nothing to embed from yet — run gemini_watcher.annotate_and_store
    first, same precondition classify_and_store_external already enforces)."""
    from qualume.db.connection import get_connection

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """SELECT hook_text, full_transcript
                   FROM external_creative_dna
                   WHERE external_content_item_id = %s
                   ORDER BY analyzed_at DESC LIMIT 1""",
                (external_content_item_id,),
            )
            row = cur.fetchone()
            if row is None:
                raise ValueError(
                    f"external_content_items row {external_content_item_id} has "
                    "no external_creative_dna row yet — run "
                    "gemini_watcher.annotate_and_store first"
                )
            hook_text, full_transcript = row
    finally:
        conn.close()

    text = build_embeddable_text(hook_text, full_transcript)
    text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """SELECT id, source_text_hash FROM transcript_embeddings
                   WHERE external_content_item_id = %s""",
                (external_content_item_id,),
            )
            existing = cur.fetchone()

            if existing is not None and existing[1] == text_hash:
                return str(existing[0])  # unchanged text, skip the real API call

            vector = embed_video_text(text)

            if existing is not None:
                cur.execute(
                    """UPDATE transcript_embeddings
                       SET embedding = %s, source_text_hash = %s,
                           embedding_model = %s, embedded_at = now()
                       WHERE id = %s""",
                    (json.dumps(vector), text_hash, EMBEDDING_MODEL, existing[0]),
                )
                row_id = existing[0]
            else:
                cur.execute(
                    """INSERT INTO transcript_embeddings
                       (external_content_item_id, embedding, source_text_hash, embedding_model)
                       VALUES (%s, %s, %s, %s)
                       RETURNING id""",
                    (external_content_item_id, json.dumps(vector), text_hash, EMBEDDING_MODEL),
                )
                row_id = cur.fetchone()[0]
        conn.commit()
        return str(row_id)
    finally:
        conn.close()
