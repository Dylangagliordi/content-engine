# Spike 001: Fused (concatenated) transcript+visual embedding clustering

## Question
Does concatenating transcript_embedding + visual_embedding vectors into one
fused vector improve HDBSCAN cluster quality over either embedding alone?

## Method
Real DB query joining transcript_embeddings and visual_embeddings on the
same external_content_item_id (70 real overlapping videos — every
transcript-embedded video also has a visual embedding as of 2026-10-31).
Ran HDBSCAN(min_cluster_size=4 and 5) three ways on the IDENTICAL 70 videos:
transcript-only (1024-dim), visual-only (1024-dim), fused (2048-dim
concatenation). Mann-Whitney U per cluster vs. everything else, same test
the real pipelines use.

## Real results (min_cluster_size=5, matches production default)

| Approach     | n_noise | n_clusters | cluster sizes      | significant (p<0.05) |
|--------------|---------|------------|--------------------|-----------------------|
| transcript   | 34      | 4          | [12, 8, 10, 6]     | 3/4                   |
| visual       | 13      | 2          | [50, 7]            | 0/2                   |
| fused        | 3       | 2          | [58, 9]            | 0/2                   |

## Verdict: INVALIDATED (at today's real data volume)

Fusion did NOT improve cluster quality — it made it WORSE than transcript
alone. Visual embedding's structure at n=70 collapses into one giant,
undifferentiated 50/70-video blob (vs. transcript's 4 genuinely tight,
separated clusters). Concatenating drowns transcript's real signal in
visual's noise rather than adding complementary information: fused
produced an even bigger 58/70 blob with zero significant clusters.

### What worked
- Transcript clustering remains strong and reproducible at n=70 (3/4
  clusters significant, same as production's existing proposed insights).

### What didn't
- Visual clustering on the SAME 70 videos found almost no real structure
  (one dominant blob). This contrasts with the full visual_embeddings
  table (n=369, built earlier this session) which DID find 2 significant
  clusters including a real n=248 cross-niche finding — so visual
  clustering genuinely needs more data/volume to separate real structure
  than transcript does, not that it's inherently useless.
- Fusion inherited visual's weakness without gaining transcript's
  strength — concatenation isn't a free lunch when one half of the vector
  has a much lower signal-to-noise ratio than the other at this scale.

### Surprises
- Expected fusion to be a modest improvement or at worst neutral; it was
  measurably worse than the weaker standalone component (visual), not
  just worse than the stronger one (transcript).

### Recommendation for the real build
1. Keep transcript and visual clustering SEPARATE pipelines, as already
   built — this spike confirms that decision was right, not just cautious.
2. Do NOT build fusion yet. Revisit only once visual_embeddings has grown
   enough (same order of magnitude as transcript's eventual real volume)
   that visual clustering reliably finds tight, well-separated clusters
   on its own — fusing two WEAK signals is worse than either alone; fusing
   two independently-STRONG signals might be genuinely additive. That's
   an untested, different question from what this spike answered.
3. If fusion is revisited later, a weighted combination (not flat
   concatenation) is worth trying first — e.g. scale each half by its own
   recent standalone discriminative power, so a currently-noisier visual
   signal doesn't get equal voting weight against a currently-cleaner
   transcript signal.
