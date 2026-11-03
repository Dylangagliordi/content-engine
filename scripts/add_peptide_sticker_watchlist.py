import sys
sys.path.insert(0, "src")
from qualume.db.watchlist import add_watched_account

rows = [
    ('thejonathancoyle', 'peptide/medspa content', 'angle: education/explainer content'),
    ('drmichaelsays', 'peptide/medspa content', 'angle: education/explainer content'),
    ('gavinyates_', 'peptide/medspa content', 'angle: direct selling/vendor content'),
    ('noah.jay_', 'peptide/medspa content', 'angle: direct selling/vendor content'),
    ('kaitlynn__neises', 'peptide/medspa content', 'angle: direct selling/vendor content'),
    ('drkarlnadolsky', 'peptide/medspa content', 'angle: controversial/skeptical/debate content'),
    ('jaclynlondon', 'peptide/medspa content', 'angle: controversial/skeptical/debate content'),
    ('drjoe_md', 'peptide/medspa content', 'angle: controversial/skeptical/debate content'),
    ('meremyersfnp', 'peptide/medspa content', 'angle: day-in-the-life/routine content'),
    ('itsalicianicole', 'peptide/medspa content', 'angle: day-in-the-life/routine content'),
    ('tayylor.elizabethh', 'peptide/medspa content', 'angle: results-tracking/before-after content'),
    ('axtifact', 'peptide/medspa content', 'angle: results-tracking/before-after content'),
    ('prajagopta', 'peptide/medspa content', 'angle: education/explainer content'),
    ('dalmadanielaa', 'peptide/medspa content', 'angle: education/explainer content'),
    ('msnicolelynnshops', 'peptide/medspa content', 'angle: education/explainer content'),
    ('realnicktrigi', 'peptide/medspa content', 'angle: education/explainer content'),
    ('andypbodybuilding', 'peptide/medspa content', 'angle: direct selling/vendor content'),
    ('s2jgy', 'peptide/medspa content', 'angle: controversial/skeptical/debate content'),
    ('bricesmithhh', 'peptide/medspa content', 'angle: controversial/skeptical/debate content'),
    ('supercarblondie', 'peptide/medspa content', 'angle: controversial/skeptical/debate content'),
    ('bot.creates', 'peptide/medspa content', 'angle: controversial/skeptical/debate content'),
    ('marissathepa', 'peptide/medspa content', 'angle: day-in-the-life/routine content'),
    ('socalurologyinstitute', 'peptide/medspa content', 'angle: doctor/clinician content'),
    ('4complexion', 'peptide/medspa content', 'angle: female-oriented/skincare content'),
    ('dr.meganlee', 'peptide/medspa content', 'angle: older demographic/anti-aging content'),
    ('busymomwellness', 'peptide/medspa content', 'angle: older demographic/anti-aging content'),
    ('midihealth', 'peptide/medspa content', 'angle: older demographic/anti-aging content'),
    ('chris_lifts04', 'peptide/medspa content', 'angle: gym-bro/male fitness-oriented content'),
    ('ego.lift', 'peptide/medspa content', 'angle: gym-bro/male fitness-oriented content'),
    ('armztrainz', 'peptide/medspa content', 'angle: gym-bro/male fitness-oriented content'),
    ('stickerclubshop', 'tiktok shop stickers', 'funny/novelty sticker seller or sticker-content creator'),
    ('stickerzoo', 'tiktok shop stickers', 'funny/novelty sticker seller or sticker-content creator'),
    ('stickershopcl', 'tiktok shop stickers', 'funny/novelty sticker seller or sticker-content creator'),
    ('munchkindesigns', 'tiktok shop stickers', 'funny/novelty sticker seller or sticker-content creator'),
    ('onegirloriginal', 'tiktok shop stickers', 'funny/novelty sticker seller or sticker-content creator'),
    ('jayceejustreads', 'tiktok shop stickers', 'funny/novelty sticker seller or sticker-content creator'),
    ('deals.tts1', 'tiktok shop stickers', 'funny/novelty sticker seller or sticker-content creator'),
    ('lindsforthegirls', 'tiktok shop stickers', 'funny/novelty sticker seller or sticker-content creator'),
]

added, skipped = 0, 0
for handle, niche, note in rows:
    try:
        account_id = add_watched_account("tiktok", handle, notes=note, watchlist_type="emerging", niche=niche)
        print(f"ADDED @{handle} ({niche}) id={account_id}")
        added += 1
    except Exception as e:
        print(f"SKIP @{handle}: {type(e).__name__}: {e}")
        skipped += 1

print(f"\nDone: {added} added, {skipped} skipped (likely duplicates)")
