# Retry-Loop Verdict — 2026-10-10

## What was built
Conditional LangGraph retry loop on `self_critique` in `write_ig_static.py`: a failed variant
is regenerated (same angle, judge's critique notes injected as explicit feedback) up to
`max_attempts` times, hard-ceilinged at 3 regardless of config. Instrumented via
`data/eval_logs/write_critique_attempts.jsonl` (every judged attempt logged: angle, attempt
number, pass/fail, full critique notes, decision). Comparison script:
`src/qualume/eval/compare_retry_benefit.py`.

## Live measurement (3 briefs x 5 variants, real API, ~$0.23 total spend)
| | max_attempts=1 (no retry) | max_attempts=2 (retry once) |
|---|---|---|
| Critique pass rate | 15/15 | 15/15 (14 on attempt 1, 1 on attempt 2) |
| API calls | 33 | 35 |
| Lift from retry | — | 0 extra passing variants, for +2 calls (~$0.007) |

## Verdict — UPDATED after the anti-fabrication fix (see below)
**Default is now `max_attempts=2` (retry ON), re-measured with the stricter judge in place.**
With the claims-gate fix applied, the SAME retry mechanism now measurably earns its cost:
pass rate went from 11/15 (73%, no retry) to 15/15 (100%, retry once), for ~$0.03 extra total
on a 3-brief/5-variant live sample. All 4 rescued variants had their flagged claims genuinely
removed on retry, not just reworded. This supersedes the original "default 1 / retry off"
verdict below, which was correct at the time but was measured against a judge that barely
failed anything (1/15) — once the judge got strict, retry had real failures to rescue.

Also fixed in this pass: `run()`/`run_state()` (what `run_batch.py` actually calls) had their
own independent `max_attempts=2` default that was never synced with the state field's default —
meaning retry was quietly already running in every real batch regardless of the field default.
The field default now matches, closing that drift.

### Original verdict (superseded, kept for history)
Default set to `max_attempts=1` (retry off). Retry wasn't worth it *yet* because the judge
almost never failed anything (1/15 in this sample) — the bottleneck wasn't "good copy getting
unfairly rejected," it was that the judge was too lenient to begin with.

## Bigger finding: the eval gate is passing fabricated claims
While reviewing the judge's actual notes on all 30 real-API variants generated during this test,
multiple variants passed with invented statistics and product features that have no basis in the
brief:
- *"Your gym bottle has 300,000x more bacteria than a toilet seat... self-cleaning UV bottle kills
  99.9% of germs"* — brief only said "reusable water bottle"; UV/self-cleaning feature invented
- *"I spent $200/session on therapy for burnout. A $20 journal did what 6 months of talking
  couldn't"* — fabricated personal testimonial implying therapy-replacement, for a plain $20
  journal brief
- *"tested to withstand drops from 6 feet"*, *"73% of millennials live paycheck to paycheck"*,
  *"Over 3,400 new grads now save an average of $847/month"* — invented numbers with no source

The LLM judge rated all of these "strong execution, credible data point." The deterministic
keyword gate (`must_avoid: [doctor, cures]`) doesn't catch this category of problem either —
these aren't banned words, they're fabricated facts.

**This is a real risk, not a quality nitpick** — fabricated health/safety/statistical claims
reaching an actual ad account risk platform bans and potential legal exposure, independent of
whether the copy "sounds good."

## Two proposed fixes (not yet built, need sign-off before touching rubric/eval gate)
1. Add an explicit rubric rule: no statistics, test results, or product features not present in
   the brief.
2. Pass the brief's actual facts to the judge as reference material (system block, framed as
   ground truth to check against — not as instructions, to avoid the same prompt-injection risk
   already guarded against for the copy itself), so the judge can verify claims instead of just
   vibe-checking tone/format.

Both are expected to raise the judge's failure rate meaningfully, which would also make the
retry loop's cost/benefit worth re-measuring (the "remove the invented stat" feedback is exactly
the kind of concrete, actionable critique note the retry mechanism is built to use).

## Update (same day): both fixes built, retry re-measured
- Playbook `ig_static/v1` (seeded via `uv run python -m qualume.db.seed_playbooks`) adds a `claims`
  rubric category. `approved_by` is NULL, pending review.
- Judge prompt `critique_v2`: brief facts go in the system block as `<brief_facts>` reference data
  (not instructions), and the judge must list `unsupported_claims`. When the playbook has `claims`,
  a non-empty list rejects the variant even if the judge said `passes=true`.
- Replaying all 5 fabricated variants quoted above through the live judge: **5/5 now rejected**,
  with each invented stat or feature listed verbatim.
- Same comparison as before (3 briefs x 5 variants, `compare_retry_20261010T212030Z.jsonl`):

| | max_attempts=1 | max_attempts=2 |
|---|---|---|
| Critique/final pass rate | 11/15 (73%) | 15/15 (100%) |
| API calls | 33 | 41 |
| Est. cost | $0.117 | $0.149 |
| Lift | — | +4 passing variants (4/4 retries rescued), 2 extra calls per rescued variant |

Retry now pays for itself. Calls per passing variant drop from 3.00 to 2.73, and every rescue
removed the flagged claims without adding new ones. Known judge gaps (Haiku): it passed one implied
insulation feature ("keeps your water ice-cold") with an empty claims list, and it is inconsistent on
generic benefit phrasing ("helps you prioritize" was flagged, "track progress" was not).
