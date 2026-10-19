-- v6: presenter/avatar description + one-sentence topic summary + the
-- derived scene_description sentence the user specifically asked for
-- ("this vid was a mom in her kitchen passionately talking about night
-- sweats"). A real, total blind spot before this: the project could
-- already say WHAT a video's copy does (external_copy_analysis) and
-- WHERE/HOW it's shot (shot_breakdown's setting/lighting/camera), but
-- had zero fields describing WHO is on screen or WHAT it's about in
-- plain language - confirmed by schema inspection before adding this.
--
-- presenters: JSONB array of PresenterDescription (see contracts.py) -
-- OBSERVABLE PRESENTATION ONLY (apparent age bucket, apparent gender
-- presentation, wardrobe, archetype, delivery affect), never identity
-- inference - same purpose as real "avatar testing" in paid media
-- buying (deciding who to CAST in future content), not profiling the
-- real person in someone else's video. Empty array if no person is
-- ever visibly on screen (product-only shots, voiceover-over-broll).
--
-- scene_description: the DERIVED sentence itself (e.g. "A mom in a
-- kitchen, passionately talking about night sweats and hot flashes
-- during menopause"), computed in Python from presenters + shot_breakdown
-- setting + video_topic_summary (see
-- gemini_watcher.derive_scene_description) - NOT asked of Gemini
-- directly, same single-source-of-truth discipline as v4's hook_text
-- derivation. Stored so it's directly queryable/displayable without
-- recomputing from the component JSONB fields every time.
ALTER TABLE external_creative_dna
    ADD COLUMN presenters JSONB,
    ADD COLUMN delivery_mode TEXT,
    ADD COLUMN video_topic_summary TEXT,
    ADD COLUMN scene_description TEXT;
