-- Retroactive migration: validation_error was added directly via psql during
-- the v2 annotation-pipeline fix (2026-10-15) rather than through a numbered
-- migration file, so a fresh database built from migrations alone would be
-- missing this column even though it's live in the real dev DB. This file
-- documents it so the migration history matches reality. IF NOT EXISTS makes
-- it safe to run against the real DB (where the column already exists) and
-- against a fresh DB alike.

ALTER TABLE external_creative_dna ADD COLUMN IF NOT EXISTS validation_error TEXT;
