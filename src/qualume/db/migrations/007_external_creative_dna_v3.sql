-- v3: per-shot visual composition + full spoken transcript. Prompted by the
-- user's direct ask: "enough raw/derived features to theoretically recreate
-- the vid from scratch - visuals, audio, vibe, colors, everything." v1/v2
-- rows are untouched (ADD, not a migration-in-place); new nullable columns.
--
-- script_structure_timeline (v2) covers STORY beats (hook/problem/demo);
-- shot_breakdown is finer-grained — the actual CAMERA shots, since one story
-- beat can span several shots with different framing/color/lighting.
--
-- full_transcript covers the ENTIRE spoken track; v1/v2's hook_text only
-- ever captured the opening ~3 seconds.
--
-- NOT re-added here: an "exact audio source" field. external_content_items
-- already has audio_track/audio_artist from yt-dlp metadata (confirmed
-- populated on all 60 real rows) — that's an exact, free, lossless
-- reference to the real audio, strictly better than asking Gemini to
-- describe it. No duplicate field needed.

ALTER TABLE external_creative_dna ADD COLUMN shot_breakdown JSONB;
ALTER TABLE external_creative_dna ADD COLUMN full_transcript JSONB;
