# Qualume — Session Handoff (2026-10-30, Phase B built)

## Start here next session
Read this file first, not chat history. Detail docs referenced below have the full writeups.

## Project state (all verified live against real data, not assumed)
- 282/282 tests passing. Git clean.
- n8n (`qualume-n8n`, published/active) runs a real 6-stage pipeline every 6 hours via the
  launchd-managed orchestration server (`~/Library/LaunchAgents/com.qualume.orchestration-
  server.plist`, RunAtLoad+KeepAlive — survives logout/restart/crash, does NOT survive the
  Mac sleeping/being off): poll -> download -> annotate -> classify -> mine (declared-taxonomy)
  -> cluster (emergent/Phase B, new this session).
- 95 external videos, all fully annotated/classified. 70 have real transcript text and are
  embedded (Voyage voyage-3.5); 25 genuinely have no spoken copy (product-only/voiceover shots)
  and are correctly excluded, not a bug.
- `watched_accounts.niche` (new column) set for both real accounts: "women's health supplements".
  Purely informational — attached to findings as a tag, never baked into clustering math, so this
  stays portable to a different business's accounts later (explicit user decision this session).

## Two mining pipelines now live, both automated, both feeding insights_external
1. **Declared-taxonomy (Phase A, built last session)** — `learn.run_mining_pass`. Tests every
   hardcoded mineable DNA column + 22 copy-technique flags. 7-8 live `proposed` findings,
   source='internal_mining'. Human decided the categories upfront; bounded but interpretable.
2. **Emergent clustering (Phase B, built THIS session)** — `learn.run_cluster_pass`. HDBSCAN
   groups transcript embeddings with NO human-named categories; a cheap Claude call auto-labels
   each statistically significant cluster afterward. 3 live `proposed` findings,
   source='transcript_clustering'. Currently finds (all tier=proposed, correlational):
   - "Live-shopping event announcements with urgency and deal highlights" OUTPERFORMS 5.16x vs
     0.96x baseline (p=0.0007, n=8) — strongest signal in the whole dataset, either pipeline.
   - "Seasonal wellness prep with product integration storytelling" underperforms (p=0.033, n=12)
   - "Relatable humor about perimenopause symptom struggles" underperforms (p=0.004, n=6)
   Real structural insight the declared-taxonomy approach could never have found (no one had
   "live-shopping announcement" as a pre-defined category) — this is Phase B doing its real job.

## Real bugs found and fixed this session (3, all via live verification against real data)
1. **uuid/text type mismatch** in run_cluster_pass's SQL — Python string item_ids vs native uuid
   column, needed explicit `::uuid[]` casts. Mocked tests never caught it (mocks never touch real
   SQL) — only surfaced running against the real DB.
2. **Misleading "25 failed" in embedding stage** — videos with genuinely no transcript (expected,
   not an error) were counted identically to real crashes. Added skipped_no_transcript as a
   distinct counter.
3. **THE SERIOUS ONE: clustering insights never actually persisted correctly.**
   `retire_stale_external_insights`'s identity-matching silently skipped rows with no 'identity'
   key (exactly clustering's shape — clusters have no stable identity across runs). Added
   retire_all=True to bypass that. BUT the retire call was placed AFTER the propose loop in the
   same pass, so every pass retired its own just-created findings seconds later — real DB showed
   0 live proposed clustering insights after every single cycle despite "proposed: 3" being
   reported. Fixed by reordering to retire-then-propose. Verified live twice in a row: stable at
   exactly 3 proposed findings, no accumulation, no self-deletion.

## What's actually NOT done yet (don't assume otherwise)
- Multiple-comparison correction (Benjamini-Hochberg or similar) — NOT built. Both pipelines
  together now run ~50+ tests per cycle; flat p<0.05 with no correction means some "significant"
  findings in insights_external are near-certainly false positives from pure test volume. This
  is the real next statistical-rigor gap, explicitly flagged in conversation but not yet built.
- Cross-account/cross-niche replication auto-check (learn.replication.check_segment_replication,
  already built and tested) is NOT wired into either automated pipeline — findings aren't
  currently required to hold independently across both accounts before being proposed.
- No owned content published — insights_owned pipeline (seed_owned_insight/record_owned_evidence,
  built, tested, idle) has nothing real to validate against yet.
- Instagram account-risk decision (Part 3) — still unresolved, untouched for weeks.
- Separate test DB — this session hit a SECOND real leaked-test-fixture-in-production incident
  (test_orch_* watched_account, active=True, polled on a real scheduled cycle) from this same
  known gap. Flagged again, still not fixed, now with two real incidents as evidence it matters.

## Immediate next action (pick one)
1. Build multiple-comparison correction — the real statistical-rigor gap now that both pipelines
   run automatically and the combined test count is large enough to matter.
2. Wire learn.replication's cross-account check into both pipelines so a proposed finding must
   hold independently in both accounts (not just pooled) before being surfaced — directly
   addresses "is this real or one account's luck" with only 2 accounts currently watched.
3. Let both pipelines run for a while (days/weeks) and watch which findings stay significant vs.
   get retired as more data comes in — the clustering pipeline's own retire mechanism now
   correctly reflects this in real time.
4. Publish one of the 5 draft IG statics to finally start the insights_owned validation loop.
