-- transcript_embeddings: one row per external_content_items video, the
-- real foundation for Phase B ("emergent" pattern-finding via clustering,
-- as opposed to Phase A/existing mining's declared-taxonomy approach).
--
-- Per the explicit design discussion this session: the declared-taxonomy
-- approach (hook_type enum, the 22 copy-technique flags, etc.) requires a
-- human to decide upfront what's worth checking for — real, useful, but
-- bounded; a pattern that doesn't fit any declared category is invisible
-- to it. Embeddings + clustering is the genuinely unbiased complement:
-- turn each video's transcript into a vector (voyage-3.5, 1024-dim) and
-- let videos that are STRUCTURALLY similar in meaning group together on
-- their own, with no human naming the groups upfront. Only after a
-- cluster shows a real performance difference does a human look at what
-- its members have in common.
--
-- embedding stored as JSONB (plain float array), NOT a pgvector column —
-- this Postgres instance doesn't have the pgvector extension installed
-- (confirmed: `SELECT * FROM pg_available_extensions WHERE name='vector'`
-- returns zero rows, plain postgres:17 image, not pgvector/pgvector).
-- Not needed for this stage anyway: clustering is a batch/offline job
-- (pull everything into Python, run scikit-learn) not a live "find
-- similar video" query — would genuinely need pgvector (or at least a
-- vector index) the day that becomes a real feature; noted here as the
-- known future upgrade, not built ahead of need.
--
-- One row per external_content_item_id (UNIQUE), upserted on re-embed —
-- same "latest wins, don't duplicate" discipline as every other table in
-- this schema that tracks a per-video derived artifact.
CREATE TABLE transcript_embeddings (
    id                        UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    external_content_item_id  UUID NOT NULL UNIQUE
                              REFERENCES external_content_items(id),
    embedding                 JSONB NOT NULL,
    source_text_hash          TEXT NOT NULL,
    -- sha256 of the exact text that was embedded — lets embed_and_store
    -- skip a real API call (cost + latency) when the underlying
    -- transcript/hook_text hasn't actually changed since the last
    -- embed, same "don't redo real work that's already done" discipline
    -- as find_pending_annotation/find_pending_classification.
    embedding_model            TEXT NOT NULL,
    embedded_at                 TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_transcript_embeddings_content_item
    ON transcript_embeddings(external_content_item_id);
