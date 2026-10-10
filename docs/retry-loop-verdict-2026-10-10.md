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

## Verdict
**Default set to `max_attempts=1` (retry off).** The code stays in place and is fully tested
(5 dedicated retry tests + existing suite, 41/41 total) — flipping it back to 2 is a one-line
change. Retry isn't worth it *yet* because the judge almost never fails anything (1/15 in this
sample) — the bottleneck isn't "good copy getting unfairly rejected," it's that the judge is too
lenient to begin with. Revisit this once the anti-fabrication rubric fix (below) makes the judge
actually strict — at that point more variants will genuinely need a second attempt, and retry
should be re-measured with the same script.

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
