"""Tests for visual_embedding's adaptive frame-budget function — written
FIRST per this project's TDD discipline.

Real problem this solves: a flat frames-per-video number is wrong in both
directions. Real cut-rate data pulled from this project's own corpus
(2026-10-31) showed shot length ranging from 1.7s (fast-cut TikTok edits,
34 cuts/min) to 25s+ (static talking-head, 0 cuts/min) within the SAME
dataset. A flat low fps (e.g. 1 frame every 2s) would miss whole shots on
fast-cut videos; a flat high fps would waste money/compute oversampling
static content for zero extra signal. The fix: use cuts_per_min (already
computed by learn.features.engineered_features from real shot_breakdown
timestamps, not re-derived here) to scale the frame budget per video.
"""
from qualume.learn.visual_embedding import pick_frame_count
from qualume.learn.visual_embedding import extract_frames
from qualume.learn.visual_embedding import embed_frames
from qualume.learn.visual_embedding import embed_and_store


def test_pick_frame_count_scales_up_for_fast_cut_video():
    """A fast-cut video (34 cuts/min, matches this project's own real
    observed data) should get MORE frames than a static one, so no shot
    is missed between samples."""
    fast_cut = pick_frame_count(cuts_per_min=34.0, duration_seconds=90.0)
    static = pick_frame_count(cuts_per_min=0.0, duration_seconds=90.0)
    assert fast_cut > static


def test_pick_frame_count_never_below_minimum_floor():
    """Even a video with zero real shot data (cuts_per_min=None, e.g.
    Gemini never produced shot_breakdown) must still get a sane minimum
    number of frames — never zero, that would mean no visual signal at
    all gets captured."""
    result = pick_frame_count(cuts_per_min=None, duration_seconds=10.0)
    assert result >= 3


def test_pick_frame_count_never_exceeds_cap():
    """A pathologically high cuts_per_min (bad data, or a genuinely
    extreme video) must not blow the frame budget — real cost control,
    same discipline as every other 'cap it, don't trust input blindly'
    pattern in this codebase."""
    result = pick_frame_count(cuts_per_min=500.0, duration_seconds=60.0)
    assert result <= 30


def test_pick_frame_count_never_exceeds_one_frame_per_second_of_video():
    """A 5-second video genuinely cannot need 20 frames to represent its
    visual content — frame count should also be bounded by the video's
    own real duration, not just the cap constant."""
    result = pick_frame_count(cuts_per_min=100.0, duration_seconds=5.0)
    assert result <= 5


def test_extract_frames_returns_requested_count_as_pil_images():
    """Real ffmpeg call against a real local video file in this
    project's own data/raw_videos — not mocked, this is the actual
    extraction path that will run against the real 8,857-video corpus.
    Picks an evenly-spaced set of frames across the video's duration."""
    import glob

    from PIL import Image

    real_videos = glob.glob("data/raw_videos/*.mp4")
    assert real_videos, "need at least one real downloaded video to test against"
    video_path = real_videos[0]

    frames = extract_frames(video_path, n_frames=5)

    assert len(frames) == 5
    for frame in frames:
        assert isinstance(frame, Image.Image)


def test_extract_frames_raises_on_missing_file():
    """Fail loud on a bad path — same discipline as build_embeddable_
    text's own precondition check, never silently return an empty list
    that looks like 'this video has no frames' instead of 'this video
    doesn't exist'."""
    import pytest

    with pytest.raises(FileNotFoundError):
        extract_frames("data/raw_videos/does_not_exist_12345.mp4", n_frames=3)


def test_embed_frames_returns_one_vector_for_the_whole_frame_list():
    """Real Voyage multimodal_embed call (confirmed live 2026-10-31,
    voyage-multimodal-3.5 returns ONE 1024-dim embedding per input frame
    LIST, not one per frame — matching transcript_embedding's own
    one-vector-per-video shape, so both embedding tables stay directly
    comparable/joinable by external_content_item_id)."""
    import glob

    real_videos = glob.glob("data/raw_videos/*.mp4")
    assert real_videos
    frames = extract_frames(real_videos[0], n_frames=3)

    vector = embed_frames(frames)

    assert isinstance(vector, list)
    assert len(vector) == 1024
    assert all(isinstance(v, float) for v in vector)


class TestEmbedAndStore:
    """Real DB integration tests against the real dev Postgres instance.
    Per CLAUDE.md principle 6 (no separate test DB yet): every row this
    test creates is deleted in a finally block, even on failure."""

    def _make_real_content_item_with_dna(self, video_path, duration, shot_breakdown):
        """Insert a real external_content_items row (minimal required
        fields) + a real external_creative_dna row pointing at it, using
        the project's own get_connection — returns the new content item
        id so the test can call embed_and_store against something real."""
        from qualume.db.connection import get_connection

        conn = get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """INSERT INTO external_content_items
                       (watched_account_id, platform, platform_video_id,
                        url, local_video_path, duration_seconds)
                       SELECT id, 'tiktok', %s, %s, %s, %s
                       FROM watched_accounts LIMIT 1
                       RETURNING id""",
                    (f"test_visual_embed_{video_path}",
                     f"https://tiktok.com/test/{video_path}", video_path, duration),
                )
                content_item_id = cur.fetchone()[0]
                cur.execute(
                    """INSERT INTO external_creative_dna
                       (external_content_item_id, shot_breakdown,
                        analysis_model, analysis_prompt_version)
                       VALUES (%s, %s, 'test', 'test')""",
                    (content_item_id, __import__("json").dumps(shot_breakdown)),
                )
            conn.commit()
            return str(content_item_id)
        finally:
            conn.close()

    def _cleanup(self, content_item_id):
        from qualume.db.connection import get_connection

        conn = get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM visual_embeddings WHERE external_content_item_id = %s", (content_item_id,))
                cur.execute("DELETE FROM external_creative_dna WHERE external_content_item_id = %s", (content_item_id,))
                cur.execute("DELETE FROM external_content_items WHERE id = %s", (content_item_id,))
            conn.commit()
        finally:
            conn.close()

    def test_embed_and_store_persists_a_real_visual_embedding_row(self):
        import glob

        real_videos = glob.glob("data/raw_videos/*.mp4")
        assert real_videos
        video_path = real_videos[0]

        content_item_id = self._make_real_content_item_with_dna(
            video_path, duration=20.0, shot_breakdown=[{"start_sec": 0, "end_sec": 10}, {"start_sec": 10, "end_sec": 20}],
        )
        try:
            row_id = embed_and_store(content_item_id)
            assert row_id

            from qualume.db.connection import get_connection
            conn = get_connection()
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT n_frames_used, embedding_model FROM visual_embeddings WHERE external_content_item_id = %s",
                        (content_item_id,),
                    )
                    n_frames, model = cur.fetchone()
                    assert n_frames > 0
                    assert model == "voyage-multimodal-3.5"
            finally:
                conn.close()
        finally:
            self._cleanup(content_item_id)
