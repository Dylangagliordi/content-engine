-- v5: coverage_warning. Prompted by a real v4 re-run on the same video
-- whose full_transcript collapsed into one segment. The v5 contract now
-- rejects under-segmentation outright (ValidationError -> validation_error),
-- but a schema-valid annotation can still fail to cover the real video
-- (e.g. shots only spanning 0-10s of a 32s video). That's flagged here,
-- checked against external_content_items.duration_seconds, as "; "-joined
-- human-readable warnings — queryable after the fact, not just a log line.
--
-- NULL means either "covered the whole video" (clean row) or "not checked"
-- (degraded row: validation_error is set instead). Pre-v5 rows stay NULL —
-- they were never checked; filter on analysis_prompt_version to tell apart.
--
-- ADD, not a migration-in-place; existing rows untouched.

ALTER TABLE external_creative_dna ADD COLUMN coverage_warning TEXT;
