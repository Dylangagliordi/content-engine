-- Structured DNA for external (watched-account) videos — kept structurally
-- separate from owned creative_dna per the owned-vs-external trust principle.
-- Schema per plans/learn-engine-idea-sourcing-architecture.md.
--
-- Filled in two parts: model-observed signals (Gemini, qualume.annotate.
-- gemini_watcher) and computed signals (pacing_cuts_per_min, music_mood —
-- future PySceneDetect/beatlyze job; Gemini never guesses these, so they stay
-- NULL until that job exists).

CREATE TABLE external_creative_dna (
    id                        UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    external_content_item_id  UUID NOT NULL REFERENCES external_content_items(id),

    -- mirrors owned creative_dna's shape, for direct comparability
    hook_type                 TEXT,
    hook_text                 TEXT,              -- transcribed/observed, not authored
    visual_hook_description   TEXT,
    format_tag                TEXT,
    script_structure_tags     TEXT[],
    cta_type                  TEXT,
    caption_style             TEXT,
    social_proof_used         BOOLEAN DEFAULT false,

    -- vibe fields, external-only
    energy_level              SMALLINT,          -- 1-5, anchored scale
    tone_category             TEXT,              -- enum, not free text
    pacing_cuts_per_min       FLOAT,             -- COMPUTED, not LLM-guessed
    music_mood                TEXT[],            -- tags, from audio analysis
    voice_delivery_style      TEXT,

    -- unproven/candidate features land here, NOT as new typed columns
    extended_features         JSONB DEFAULT '{}',

    -- provenance
    analysis_model            TEXT NOT NULL,
    analysis_prompt_version   TEXT NOT NULL,
    analyzed_at               TIMESTAMPTZ NOT NULL DEFAULT now()
);
