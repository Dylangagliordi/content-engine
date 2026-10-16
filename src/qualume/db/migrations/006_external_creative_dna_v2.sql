-- external_creative_dna v2 (analysis_prompt_version='v2'): richer,
-- timestamped signals, all asked of Gemini in the same annotation call. The
-- planned separate PySceneDetect/beatlyze job is cancelled.
--
-- ADD only: existing columns and v1 rows are untouched. v2 rows still fill
-- the v1 columns (hook_type etc.). Legacy columns pacing_cuts_per_min and
-- music_mood stay NULL going forward — Gemini samples at 1 FPS, so a precise
-- cut count is unreliable (pacing_qualitative replaces it), and audio-derived
-- music mood lands in music_mood_v2.

ALTER TABLE external_creative_dna
    ADD COLUMN script_structure_timeline  JSONB,   -- [{beat, start_sec, end_sec}], ordered
    ADD COLUMN on_screen_text             JSONB,   -- [{text, timestamp_sec}], ordered
    ADD COLUMN emotion_timeline           JSONB,   -- [{timestamp_sec, emotion}], ordered
    ADD COLUMN pacing_qualitative         TEXT,    -- slow | medium | fast
    ADD COLUMN music_mood_v2              TEXT[],  -- tags, from Gemini's own audio analysis
    -- the complete Gemini response for this call, nothing discarded — fields
    -- not promoted to a typed column are still queryable here
    ADD COLUMN raw_gemini_response        JSONB NOT NULL DEFAULT '{}';
