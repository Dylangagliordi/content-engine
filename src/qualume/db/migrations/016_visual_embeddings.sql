-- visual_embeddings: one row per external_content_items video, the
-- frame-sampled visual complement to transcript_embeddings. Per the
-- user's explicit decision this session (2026-10-31): kept as a
-- SEPARATE table, not fused into transcript_embeddings or concatenated
-- at build time — fusion is a real open question whose answer depends
-- on seeing real cluster quality from EACH side independently first.
-- Running them as two genuinely separate pipelines is what makes that
-- comparison possible at all.
--
-- embedding stored as JSONB (plain float array), same reasoning as
-- transcript_embeddings: no pgvector extension installed on this
-- Postgres instance, and this is a batch/offline clustering job, not a
-- live similarity-search feature yet.
--
-- n_frames_used and cuts_per_min_at_embed_time are provenance, not
-- just metadata — the frame BUDGET is adaptive (learn.visual_embedding.
-- pick_frame_count, scaled by real cuts_per_min), so unlike transcript
-- embeddings (always the same text in, same model), two videos with
-- identical duration can get very different frame counts. Recording
-- what was actually used makes a later "why did this cluster look odd"
-- investigation possible without re-deriving it.
CREATE TABLE visual_embeddings (
    id                        UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    external_content_item_id  UUID NOT NULL UNIQUE
                              REFERENCES external_content_items(id),
    embedding                 JSONB NOT NULL,
    n_frames_used             INTEGER NOT NULL,
    cuts_per_min_at_embed_time FLOAT,  -- nullable: a video with no real
    -- shot_breakdown data (Gemini degraded path) still gets embedded at
    -- the MIN_FRAMES floor, just with no cuts_per_min to record.
    embedding_model            TEXT NOT NULL,
    embedded_at                 TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_visual_embeddings_content_item
    ON visual_embeddings(external_content_item_id);
