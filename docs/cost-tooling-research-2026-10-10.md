# Cost, Tooling & Analysis Research — 2026-10-10

Answers to: where did $20 go, OpenRouter, scanning the GitHub repo + finding more, how we
actually read TikTok video/audio, and Canva Connect API setup.

---

## 1. Where the $20 went / what you might be missing

**Confirmed, not guessed:**
- Claude Code itself is billed through your **Claude Pro subscription** (`apple_subscription`
  billing type, confirmed via `claude auth status`) — NOT pay-per-token. My test calls today
  (`claude -p ...`) did not touch the $20.
- `ANTHROPIC_API_KEY` is set ONLY inside `~/qualume/.env`, not globally on this machine — confirmed
  via `env | grep anthropic` (empty) and checking shell profiles. So the $20 is coming specifically
  from direct API calls the **Qualume Python code** makes (`write_ig_static.py`'s LangGraph nodes),
  not from Claude Code, not from some leaked/duplicated key.
- I could NOT pull your exact per-call cost breakdown — Anthropic's usage-report endpoint requires
  an **Admin API key** (confirmed: got a 401 "requires an Admin API key or organization-scoped API
  key" when querying with the standard key in `.env`). **Action for you**: check
  https://console.anthropic.com/settings/usage directly — that's the fastest way to see the actual
  per-model/per-day breakdown. I don't have that visibility from here.

**What's realistically burning cost, based on what the code actually does (no guessing needed
here — this is just counting actual API calls in the code):**
- Every `write_ig_static.py` run makes **1 (ideation) + N (generate, one per angle) + N (critique,
  one per angle) = 2N+1 separate Claude calls per batch**. A count=5 batch is 11 calls. We ran this
  several times today (smoke tests + live batches + the water-bottle test) — that adds up fast
  especially since `generate_variants` and `self_critique` both use `pick_llm("medium")` (Sonnet),
  not the cheaper Haiku tier, for every single variant.
- **No prompt caching is implemented anywhere in the codebase right now.** This is a real, concrete
  cost lever we're missing — confirmed via Anthropic's docs: cached input tokens cost ~10% of the
  normal input rate. Every one of our calls resends the full rubric/playbook text and system
  instructions from scratch, uncached, every single time. For a system designed to run repeatedly
  (batches, polling, eventually 24/7), this is the single highest-leverage fix available.
- **The `self_critique` node uses `pick_llm("low")` correctly already** (good), but
  `generate_variants` uses `pick_llm("medium")` for every variant — worth reconsidering whether
  ideation-stage generation genuinely needs Sonnet-tier reasoning per variant, or whether Haiku
  could handle the bulk of it with Sonnet reserved for a final pass.

**Concrete fixes, in priority order:**
1. **Add Anthropic prompt caching** to the Write engine's repeated prompt components (the rubric
   summary, system framing) — real, direct cost reduction, no architecture change needed, Anthropic
   supports it natively via a `cache_control` field on the request.
2. **Check the Console usage dashboard** (link above) to see if a specific stage is disproportionately
   expensive — don't optimize blind.
3. **Set a spend cap** — Anthropic's Console lets you set a monthly spend limit per API key; do this
   now so a runaway loop or accidental repeated test run can't silently burn more before you notice.
4. Consider whether `generate_variants` should default to `pick_llm("low")` and only escalate to
   medium/high when `self_critique` actually flags a problem (ties into the conditional-loop
   discussion from earlier — generate cheap first, only spend more on retries).

## 2. OpenRouter — is it worth using

**What it actually is**: a routing gateway that sits in front of 500+ models (including all of
Anthropic's), giving you one API key and one endpoint format, with automatic fallback across
providers if one is down/overloaded.

**Real, confirmed tradeoffs (not vibes):**
| | OpenRouter | Direct Anthropic |
|---|---|---|
| Cost | ~30-38% cheaper on average (provider price auction) | Standard list price |
| Latency | ~2.3x slower (112ms+ overhead, worse under load) | Fastest |
| Prompt caching | Supported, BUT diluted unless you pin a single provider (cache is per-provider; routing you to a different provider on the next call breaks the cache) | Full native support |
| Reliability | Built-in fallback if Anthropic has an outage | Single point of failure, but Anthropic's own uptime is already 99.9% |
| Feature lag | New Claude features arrive 2-18 days late on OpenRouter | Same-day |

**My honest read for Qualume specifically**: OpenRouter's main value (automatic fallback across
*different model providers*) doesn't really apply to us — we deliberately committed to Claude-only.
Its cost savings come from bidding across providers for the *same* model, which would actually work
against our prompt-caching fix above (pinning to one provider to preserve cache hits negates most of
OpenRouter's price-arbitrage benefit anyway). **Recommendation: skip OpenRouter for now, fix prompt
caching directly with Anthropic first** — it's a bigger, cleaner win for our actual bottleneck
(repeated prompts, not provider price shopping). Revisit OpenRouter only if we ever add a second
model provider as a deliberate fallback (e.g., Gemini for video, discussed below) and want one
unified endpoint for both.

## 3. The GitHub repo — scanned, plus a broader search

### `ScrapeCreators/social-media-research-skills` (the repo flagged earlier)
Actually opened and read the skill files (`outlier-post-finder`, `transcript-intelligence`,
`creator-profile-teardown`, `scrapecreators-api`). Important correction to my earlier
characterization: **this is not free or self-contained** — every skill in it is a prompt-template
wrapper around the **ScrapeCreators paid API** (confirmed: credit-metered, 100 free credits then
paid, same commercial model as Apify). Useful as a reference for *prompt structure* (their output
templates for "outlier post" and "creator teardown" are genuinely well-designed — I pulled the
actual templates above), but not something we can "open and use" for free the way I implied earlier.
Not worth paying for right now given `yt-dlp` already gives us the same raw TikTok data for free.

### New, more directly useful finds from a broader search:

**`vyralcontent/content-skills`** (MIT license, **free**, no paid API dependency) — this one IS
directly usable. I read two of its skill files in full:
- `viral-hooks` — encodes named-creator hook frameworks (Hormozi, MrBeast, Brunson, etc.), the
  "three-layer hook" concept (visual + verbal + on-screen text — same idea Hormozi's video flagged
  back when we first researched this), explicit anti-patterns that kill retention.
- `viral-tiktok-content` — encodes **actual TikTok algorithm mechanics with concrete numbers**:
  completion-rate benchmarks by video length (>100% for sub-15s loops, ~70% for 15-30s, ~50% for
  30-60s, ~40% for 60s+), confirms "saves and shares outweigh likes" (matches what we already had in
  the v0 playbook from the Hormozi video), and a genuinely useful mechanic we didn't have yet: TikTok
  re-tests every video against a small lookalike cohort first regardless of follower count or past
  hits ("test-then-expand" distribution — explains why a small/new account can still go viral and
  why deleting underperforming videos doesn't help, since they can re-enter testing later).

**Action**: this content is genuinely worth folding into the `ig-static-playbook` skill (and a
future `tiktok-video-playbook`) as additional seeded hypotheses — same honesty discipline as before
(`tier: proposed, source: external_field_notes`), since it's still unvalidated against our own data,
but it's free, well-reasoned, and platform-mechanics-grounded rather than vibes.

**Other finds, noted but lower priority:**
- `m13v/social-autoposter` (aka "S4L") — **correction, re-verified 2026-10-12**: this is NOT a
  content-performance/engagement-tracking schema (my earlier characterization was wrong — either
  the repo changed or I mischaracterized it originally). It's actually a Reddit/X **reply-drafting
  and auto-posting** plugin: finds high-traffic threads, drafts contribution-first replies in your
  voice, review queue, optional posting autopilot. **Not relevant to Qualume's content-generation/
  Learn-engine pipeline** — different domain (conversation discovery + replies, not
  TikTok/IG content performance mining) and its autopilot-posting capability runs counter to this
  project's "publishing is always manual" principle anyway. **Keeping it on the list regardless**,
  per explicit instruction (2026-10-12) — user may build a separate, distinct automation for
  Reddit/comment-reply workflows later, unrelated to Qualume's content-creation engine. Worth
  revisiting then, not now.
- `ViralLab/SMDT` — general-purpose social data normalization toolkit (Postgres/TimescaleDB/
  PostGIS-backed, re-verified 2026-10-12: 5 core tables — Communities/Accounts/Posts/Actions/
  Entities — standardizes raw exports from 11+ platforms into one shared schema). Confirms our
  own owned/external table-separation instinct is a recognized real pattern. Note: its
  "normalization" means standardizing field names/schema ACROSS platforms (e.g. Twitter's
  `like_count` vs Reddit's equivalent field mapping to one shared column) — NOT relative/baseline
  performance normalization (e.g. judging a video against its own account's median). Don't
  conflate the two meanings of "normalize" — this repo doesn't address the account-baseline idea
  discussed 2026-10-12 (see `plans/learn-engine-idea-sourcing-architecture.md`), that idea came
  from design discussion, not from this repo.

## 4. How we actually "see" TikTok's visuals/audio — the real gap, now actually solvable

**You're right to push on this — it was a real gap.** Confirmed via research: **Claude has no
native video or audio understanding at the API level** (text + images only). Gemini is the model
with native video/audio in a single prompt; Claude requires a workaround. This matters a lot since
our whole stack is Claude-only.

**The workaround, confirmed by multiple independent real implementations (not just one blog post):**
1. `yt-dlp` downloads the actual video file (not just metadata — we're currently only pulling
   metadata, never the video itself).
2. `ffmpeg` (already installed on this machine, confirmed) extracts frames at an auto-scaled rate —
   more frames for short/dense content, fewer for long content, to control cost. One real example
   I found: 10 seconds of 1080p video at "standard" mode (2fps, 20 frames) costs about **$0.01 in
   Claude Sonnet vision tokens** — genuinely cheap per video.
3. Audio: either (a) pull TikTok's own auto-generated caption/subtitle track via yt-dlp when one
   exists (free, no extra API call — this is what the `GoViral-Tiktok` project I found does), or
   (b) extract the audio and send it to a transcription service — but since Claude doesn't do audio
   either, this means something like Whisper (OpenAI, or Groq's hosted whisper-large-v3 which is
   cheaper/faster) as a separate small dependency, OR Gemini if we want native audio+video without
   the frame-extraction workaround at all.
4. The extracted frames (as images) + the transcript (as text) both go into one Claude prompt —
   Claude then genuinely "sees" the frames and "reads" the transcript together, close to true
   multimodal understanding despite not having it natively.

**This directly maps onto our unbuilt Part 5a (Learn engine annotation stage)** — it's not
additional scope, it's the concrete implementation of what that part needed anyway. I'd suggest this
becomes a Claude Code delegated task: build `annotate_external_content.py` using this
frame-extraction + transcript pattern, store results into a new `external_creative_dna` table
(matching our existing `creative_dna` shape).

**Open design question worth flagging, not deciding here**: since we need ffmpeg/frame-extraction
regardless of model, do we (a) stick with Claude + this frame workaround, keeping the whole stack
single-provider, or (b) bring in Gemini specifically for this one stage since it has native
video+audio and might be cheaper/cleaner for this specific job? This is a real architectural fork —
worth deciding deliberately when we actually build Part 5a, not assumed now.

## 5. Canva Connect API — real, usable, but needs real setup (not instant)

Confirmed Canva has a genuine Connect API with an **Exports endpoint** (`POST /exports`) that can
pull a design out as PNG/JPG/PDF/MP4 programmatically, plus a Designs API for reading design
metadata/content. This is real and matches what you asked for.

**What setup actually requires** (confirmed from Canva's developer docs) — this is NOT a quick
toggle, it's a small one-time registration:
1. Create a Canva Developer app at canva.com/developers (free).
2. Choose "Connect API" / "External Application" as the app type.
3. Generate OAuth client ID + secret.
4. Define scopes — for our use case: `design:meta:read` and `design:content:read` (read-only,
   no need for write access to your actual designs).
5. Set a redirect URI for the OAuth flow (Canva uses full OAuth 2.0 authorization-code flow, meaning
   there's a real login/consent step — you'll need to authorize the app once in a browser, not just
   paste an API key).
6. Once authorized, the app gets an access token, which I'd use to call the Exports endpoint on
   whichever design ID you want pulled.

**This needs you to do the app registration + OAuth consent step** (I can't register a developer app
or click through an OAuth consent screen on your behalf) — once you've created the app and have a
client ID/secret, I can build the token-exchange + export-pull code. Simpler alternative if you want
to unblock this today without the OAuth setup: just export the current static as a PNG from Canva's
UI (File → Download) and drop the file path here — I can review it that way immediately, no API
work needed. Your call on which path given how often you'll want this (one-off review = just export
manually; ongoing automated pulling = worth the Connect API setup).
