"""Visual embedding — the frame-sampled complement to transcript_embedding,
per the user's explicit decision this session (2026-10-31): transcript-
only clustering is blind to visual structure (editing pace, shot type,
product vs. talking-head framing) that `shot_breakdown`/`learn.features`
already capture but nothing mines as a vector today.

Run as a SEPARATE pipeline/table from transcript_embedding (not fused at
build time) per explicit user decision: fusion is a real open question
whose answer depends on seeing real cluster quality from each side
independently first — concatenating now would make "is visual embedding
even worth it" impossible to measure on its own.

Voyage voyage-multimodal-3.5, not Gemini embeddings or a local model,
same reasoning as transcript_embedding.py's own choice: same provider as
the already-working text embeddings (keeps "Claude for generation, Voyage
for embeddings" discipline intact), purpose-built multimodal model with
real stated video support (not CLIP-era, confirmed via its Jan 2026
model-card page before this module was written) and a genuinely free
usage path for this corpus (confirmed 2026-10-31: 150B free pixels/
account, this project's real corpus of 8,857 videos needs ~130K frames at
a 15-frame average — comfortably inside the free tier at today's volume).
"""
import hashlib
import io
import os

MIN_FRAMES = 3  # real floor: even a video with zero shot_breakdown data
# (cuts_per_min=None) must get SOME visual signal, never zero frames.

MAX_FRAMES = 30  # real cost-control ceiling, same "cap it, don't trust
# input blindly" discipline as everywhere else in this codebase — guards
# against a pathological/bad cuts_per_min value blowing the frame budget.

BASE_FRAMES = 6  # frames for a video with near-zero cuts_per_min (static
# talking-head content) — enough to capture real visual variety (framing,
# background, presenter appearance) without wastefully oversampling a
# shot that never changes.

FRAMES_PER_CUT_PER_MIN = 0.5  # scaling factor: every +1 cuts_per_min adds
# half a frame to the budget, so a 34-cuts/min video (this project's own
# real observed fast-cut rate) lands around BASE_FRAMES + 17 ≈ 23 frames —
# enough to catch most real shots without hitting the MAX_FRAMES ceiling.

EMBEDDING_MODEL = "voyage-multimodal-3.5"

_voyage_client = None


def _get_voyage_client():
    """Lazy singleton, same pattern as transcript_embedding._get_voyage_
    client — avoids constructing a real client (and reading
    VOYAGE_API_KEY) at import time. Tests patch this function directly."""
    global _voyage_client
    if _voyage_client is None:
        import os

        import voyageai
        from dotenv import load_dotenv

        load_dotenv()
        _voyage_client = voyageai.Client(api_key=os.environ["VOYAGE_API_KEY"])
    return _voyage_client


def pick_frame_count(cuts_per_min: float | None, duration_seconds: float) -> int:
    """How many frames to sample from one video, scaled by its REAL
    observed cut rate (learn.features.engineered_features' cuts_per_min,
    computed from exact shot_breakdown timestamps — never re-derived
    here, this function only consumes it).

    Bounded three ways, in order: a real minimum floor (MIN_FRAMES, so a
    video with no shot_breakdown data still gets real visual signal), a
    real cost-control ceiling (MAX_FRAMES), and never more than roughly
    one frame per second of the video's OWN real duration (a 5-second
    video cannot need 20 frames — oversampling a short clip adds cost
    with zero extra signal, the same waste this function exists to avoid
    on the slow-pacing side).
    """
    if cuts_per_min is None:
        budget = MIN_FRAMES
    else:
        budget = BASE_FRAMES + cuts_per_min * FRAMES_PER_CUT_PER_MIN

    budget = max(MIN_FRAMES, min(MAX_FRAMES, budget))
    budget = min(budget, max(MIN_FRAMES, int(duration_seconds)))
    return int(budget)


def extract_frames(video_path: str, n_frames: int) -> list:
    """Evenly-spaced frame extraction from a real local video file via
    ffmpeg, returned as PIL Images (Voyage's SDK accepts PIL.Image.Image
    directly in multimodal_embed inputs — no need to round-trip through
    a temp file per frame).

    Fails loud (FileNotFoundError) on a missing path — same "don't
    silently produce an empty/wrong result" discipline as
    build_embeddable_text's own precondition check.

    Uses ffmpeg's own duration probe + select filter rather than a fixed
    fps (which would need the caller to pre-compute an fps from n_frames
    and duration, and ffmpeg's frame selection by index is both simpler
    and avoids rounding-induced off-by-one frame counts).
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"extract_frames: no such file: {video_path}")

    import subprocess

    from PIL import Image

    duration_str = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", video_path],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    duration = float(duration_str)

    timestamps = [duration * (i + 0.5) / n_frames for i in range(n_frames)]

    frames = []
    for ts in timestamps:
        result = subprocess.run(
            ["ffmpeg", "-ss", str(ts), "-i", video_path, "-frames:v", "1",
             "-f", "image2pipe", "-vcodec", "png", "-"],
            capture_output=True, check=True,
        )
        frames.append(Image.open(io.BytesIO(result.stdout)))
    return frames


def embed_frames(frames: list) -> list:
    """Real Voyage multimodal_embed call. One frame LIST -> one 1024-dim
    vector (confirmed live 2026-10-31 — voyage-multimodal-3.5 fuses the
    whole input list into a single embedding, matching transcript_
    embedding's own one-vector-per-video shape).

    input_type='document', same reasoning as embed_video_text: these
    vectors are stored for later clustering/comparison among themselves,
    not used to search against a query."""
    client = _get_voyage_client()
    result = client.multimodal_embed(
        inputs=[frames], model=EMBEDDING_MODEL, input_type="document",
    )
    return list(result.embeddings[0])


def embed_and_store(external_content_item_id: str) -> str:
    """Pull a video's real local_video_path + shot_breakdown/duration
    (latest external_creative_dna row, same "latest wins" discipline as
    transcript_embedding.embed_and_store), extract an adaptively-sized
    frame set, embed via Voyage, upsert a visual_embeddings row.

    Skips the real API call (returns the EXISTING row id) when a row
    already exists for this item and nothing that would change the
    result has changed (same n_frames_used would be picked again) — real
    cost/latency discipline, same shape as transcript_embedding's
    source_text_hash dedup, just keyed on frame count instead of a text
    hash since there's no single "source text" here to hash.

    Raises ValueError if the item has no local_video_path yet (nothing
    downloaded to embed from) or no external_creative_dna row (no
    duration/shot_breakdown to size the frame budget from) — same
    fail-loud precondition discipline as transcript_embedding.
    """
    from qualume.db.connection import get_connection
    from qualume.learn.features import engineered_features

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT local_video_path, duration_seconds FROM external_content_items WHERE id = %s",
                (external_content_item_id,),
            )
            row = cur.fetchone()
            if row is None or row[0] is None:
                raise ValueError(
                    f"external_content_items row {external_content_item_id} has no "
                    "local_video_path — nothing downloaded to embed from yet"
                )
            video_path, duration_seconds = row

            cur.execute(
                """SELECT shot_breakdown FROM external_creative_dna
                   WHERE external_content_item_id = %s
                   ORDER BY analyzed_at DESC LIMIT 1""",
                (external_content_item_id,),
            )
            dna_row = cur.fetchone()
            if dna_row is None:
                raise ValueError(
                    f"external_content_items row {external_content_item_id} has no "
                    "external_creative_dna row yet — run gemini_watcher.annotate_and_store first"
                )
            shot_breakdown = dna_row[0]
    finally:
        conn.close()

    features = engineered_features(shot_breakdown or [], [], duration_seconds)
    cuts_per_min = features["cuts_per_min"]
    n_frames = pick_frame_count(cuts_per_min, duration_seconds)

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, n_frames_used FROM visual_embeddings WHERE external_content_item_id = %s",
                (external_content_item_id,),
            )
            existing = cur.fetchone()

            if existing is not None and existing[1] == n_frames:
                return str(existing[0])  # same frame budget would be picked again, skip the real API call

            frames = extract_frames(video_path, n_frames)
            vector = embed_frames(frames)

            if existing is not None:
                cur.execute(
                    """UPDATE visual_embeddings
                       SET embedding = %s, n_frames_used = %s,
                           cuts_per_min_at_embed_time = %s,
                           embedding_model = %s, embedded_at = now()
                       WHERE id = %s""",
                    (_json_dumps(vector), n_frames, cuts_per_min, EMBEDDING_MODEL, existing[0]),
                )
                row_id = existing[0]
            else:
                cur.execute(
                    """INSERT INTO visual_embeddings
                       (external_content_item_id, embedding, n_frames_used,
                        cuts_per_min_at_embed_time, embedding_model)
                       VALUES (%s, %s, %s, %s, %s)
                       RETURNING id""",
                    (external_content_item_id, _json_dumps(vector), n_frames,
                     cuts_per_min, EMBEDDING_MODEL),
                )
                row_id = cur.fetchone()[0]
        conn.commit()
        return str(row_id)
    finally:
        conn.close()


def _json_dumps(vector: list) -> str:
    import json
    return json.dumps(vector)
