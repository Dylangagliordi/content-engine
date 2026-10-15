-- Why an account is watched, per plans/learn-engine-idea-sourcing-architecture.md
-- ("watched_accounts.watchlist_type — multiple watchlist support"). Values used:
-- general, competitor, cross_niche, aspirational, emerging, format_specific.
-- Kept as free TEXT (no CHECK) to match the doc's one-column addition; the CLI
-- enforces the allowed choices.

ALTER TABLE watched_accounts ADD COLUMN watchlist_type TEXT NOT NULL DEFAULT 'general';
