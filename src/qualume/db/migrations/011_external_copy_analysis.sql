-- external_copy_analysis: copy-technique classification over the already-
-- stored transcript/on-screen-text data. Deliberately a SEPARATE table
-- from external_creative_dna, not new columns on it, because:
--   1. This is a text-only Claude classification pass (no video, no audio,
--      no Gemini) - a genuinely different analysis stage with its own
--      model/prompt lifecycle, not a variant of the video-watching pass.
--   2. It can be re-run any time full_transcript/on_screen_text/hook_text
--      already exist, without ever re-touching Gemini - keeping it in its
--      own table makes that re-run story (and its own versioning) clean,
--      same reasoning that kept external_creative_dna separate from
--      external_content_items.
--
-- One row per (external_content_item_id, analysis_prompt_version) - same
-- append-only-across-versions discipline as external_creative_dna (v1
-- through v5 all coexist); never UPDATE a prior version's row in place.
--
-- 19 boolean copy-technique flags, grounded in real reference material
-- (Winning Hook Criteria, Features/Benefits/Outcomes, VOC doc, VSL and
-- Advetorial breakdowns in ~/Downloads/Copywriting Resources) - see
-- src/qualume/annotate/copy_classifier.py's module docstring for the
-- full taxonomy rationale. Multi-label by design: a real video can be
-- true on several at once (e.g. hook_pain_callout AND hook_specificity
-- in the same hook).

CREATE TABLE external_copy_analysis (
    id                        UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    external_content_item_id UUID NOT NULL REFERENCES external_content_items(id),

    -- Hook-level (first ~3s)
    hook_curiosity_open_loop       BOOLEAN,
    hook_pain_callout              BOOLEAN,
    hook_promise_without_mechanism BOOLEAN,
    hook_specificity                BOOLEAN,
    hook_credibility_signal        BOOLEAN,
    hook_pattern_interrupt          BOOLEAN,
    hook_status_shift               BOOLEAN,

    -- Body: problem/agitation
    mentions_customer_problem BOOLEAN,
    uses_voc_language          BOOLEAN,
    agitates_pain_emotionally BOOLEAN,
    burns_alternatives          BOOLEAN,

    -- Body: mechanism/solution
    explains_mechanism            BOOLEAN,
    discovery_moment_framing     BOOLEAN,
    uses_sensory_visual_language BOOLEAN,
    features_framed_as_benefits BOOLEAN,

    -- Body: proof/credibility
    uses_social_proof            BOOLEAN,
    uses_authority_or_expert_proof BOOLEAN,
    uses_before_after_contrast   BOOLEAN,

    -- Close: CTA/offer
    creates_urgency                     BOOLEAN,
    uses_risk_reversal                  BOOLEAN,
    cta_direct_address                  BOOLEAN,
    promises_identity_or_status_shift   BOOLEAN,

    -- Full raw response, same never-discard-data discipline as
    -- external_creative_dna.raw_gemini_response.
    raw_response JSONB DEFAULT '{}',
    validation_error TEXT,

    -- Provenance - same discipline as external_creative_dna.
    analysis_model          TEXT NOT NULL,
    analysis_prompt_version TEXT NOT NULL,
    analyzed_at              TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_external_copy_analysis_item_id ON external_copy_analysis(external_content_item_id);
