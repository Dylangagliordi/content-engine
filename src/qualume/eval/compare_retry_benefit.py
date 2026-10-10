"""Measure whether the self_critique retry loop is worth its extra API calls.

Runs the SAME golden-set briefs through the Write graph twice — max_attempts=1
(no retry, the pre-loop behavior) and max_attempts=2 (one retry) — against the
REAL API, then scores survivors with the deterministic run_constraint_checks.

    uv run python -m qualume.eval.compare_retry_benefit            # prints cost estimate, exits
    uv run python -m qualume.eval.compare_retry_benefit --yes      # actually runs (real $)

"Final pass" = survived self_critique AND passed the golden example's
must_include/must_avoid checks. API calls are counted by a LangChain callback
(every real chat-model call) and cross-checked against the per-attempt log.
"""

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone

from langchain_core.callbacks import BaseCallbackHandler

from qualume.eval.harness import load_golden_set, run_constraint_checks
from qualume.graphs import write_ig_static
from qualume.models.contracts import CampaignBrief

GOLDEN_SET_PATH = "src/qualume/eval/golden_sets/ig_static_write_v1.jsonl"
GOAL = "drive engagement and sales"

# USD per million tokens. NOT live-verified in this session (pricing page fetch
# was unavailable) — confirm against Anthropic's pricing page before trusting
# the dollar figures. Call/token counts below are measured, not estimated.
PRICING = {
    "claude-sonnet-4-5": {"input": 3.00, "output": 15.00},
    "claude-haiku-4-5": {"input": 1.00, "output": 5.00},
}
CACHE_WRITE_MULT, CACHE_READ_MULT = 1.25, 0.10
# Rough per-call sizes for the pre-run estimate only.
_EST_IN, _EST_OUT = 700, 150


class CallCounter(BaseCallbackHandler):
    """Counts every real chat-model call and its token usage, keyed by model."""

    def __init__(self):
        self.calls = Counter()
        self.tokens = defaultdict(Counter)

    def on_llm_end(self, response, **kwargs):
        for gens in response.generations:
            for gen in gens:
                msg = getattr(gen, "message", None)
                model = (msg.response_metadata or {}).get("model_name") or (
                    msg.response_metadata or {}).get("model", "unknown")
                model = _normalize_model(model)
                self.calls[model] += 1
                usage = msg.usage_metadata or {}
                details = usage.get("input_token_details", {}) or {}
                self.tokens[model]["input"] += usage.get("input_tokens", 0)
                self.tokens[model]["output"] += usage.get("output_tokens", 0)
                self.tokens[model]["cache_read"] += details.get("cache_read", 0) or 0
                self.tokens[model]["cache_creation"] += details.get("cache_creation", 0) or 0

    def cost_usd(self) -> float:
        total = 0.0
        for model, t in self.tokens.items():
            price = PRICING.get(model)
            if price is None:
                continue
            uncached = t["input"] - t["cache_read"] - t["cache_creation"]
            total += (uncached * price["input"]
                      + t["cache_creation"] * price["input"] * CACHE_WRITE_MULT
                      + t["cache_read"] * price["input"] * CACHE_READ_MULT
                      + t["output"] * price["output"]) / 1e6
        return total


def _normalize_model(model: str) -> str:
    # API may return a dated snapshot id (e.g. claude-haiku-4-5-2025...); map to the alias.
    for alias in PRICING:
        if model.startswith(alias):
            return alias
    return model


def estimate_cost(n_briefs: int, count: int) -> tuple[int, float]:
    """Upper-bound estimate: no-retry run + retry run where every variant retries once."""
    sonnet_calls = n_briefs * ((1 + count) + (1 + 2 * count))
    haiku_calls = n_briefs * (count + 2 * count)
    s, h = PRICING["claude-sonnet-4-5"], PRICING["claude-haiku-4-5"]
    usd = (sonnet_calls * (_EST_IN * s["input"] + _EST_OUT * s["output"])
           + haiku_calls * (_EST_IN * h["input"] + _EST_OUT * h["output"])) / 1e6
    return sonnet_calls + haiku_calls, usd


def run_condition(examples: list[dict], max_attempts: int, count: int, log_path: str) -> dict:
    write_ig_static.ATTEMPT_LOG_PATH = log_path
    counter = CallCounter()
    slots = critique_pass = final_pass = 0
    run_ids = []
    per_brief = []
    for ex in examples:
        brief = CampaignBrief(product_or_topic=ex["brief"], offer_or_goal=GOAL,
                              playbook_version="pending")
        state = write_ig_static.run_state(brief, count=count, max_attempts=max_attempts,
                                          config={"callbacks": [counter]})
        run_ids.append(state.run_id)
        n_slots = len(state.attempts)  # one slot per angle/variant
        survivors = state.variants
        passing = [v for v in survivors
                   if run_constraint_checks(f"{v.hook_text} {v.body_text} {v.cta_text}", ex)]
        slots += n_slots
        critique_pass += len(survivors)
        final_pass += len(passing)
        per_brief.append((ex["brief"], n_slots, len(survivors), len(passing)))

    rows = [r for r in _read_log(log_path) if r["run_id"] in set(run_ids)]
    # Expected calls: 1 ideation per brief + (1 generate + 1 critique) per judged attempt.
    expected_calls = len(examples) + 2 * len(rows)
    return {
        "max_attempts": max_attempts,
        "slots": slots,
        "critique_pass": critique_pass,
        "final_pass": final_pass,
        "api_calls": sum(counter.calls.values()),
        "calls_by_model": dict(counter.calls),
        "tokens": {m: dict(t) for m, t in counter.tokens.items()},
        "expected_calls_from_log": expected_calls,
        "cost_usd": counter.cost_usd(),
        "per_brief": per_brief,
        "log_rows": rows,
    }


def _read_log(path: str) -> list[dict]:
    try:
        with open(path) as f:
            return [json.loads(line) for line in f if line.strip()]
    except FileNotFoundError:
        return []


def _pct(a: int, b: int) -> str:
    return f"{a}/{b} ({a / b:.0%})" if b else "0/0"


def report(base: dict, retry: dict) -> None:
    print("\n=== Retry benefit: max_attempts=1 vs max_attempts=2 ===")
    for r in (base, retry):
        print(f"\n[max_attempts={r['max_attempts']}]")
        print(f"  critique pass rate : {_pct(r['critique_pass'], r['slots'])}")
        print(f"  FINAL pass rate    : {_pct(r['final_pass'], r['slots'])}  (critique + constraint checks)")
        print(f"  API calls          : {r['api_calls']}  by model {r['calls_by_model']}"
              f"  (log cross-check expects {r['expected_calls_from_log']})")
        print(f"  tokens             : {r['tokens']}")
        print(f"  est. cost          : ${r['cost_usd']:.4f}")
        for brief, n, cp, fp in r["per_brief"]:
            print(f"    - {brief[:55]:<55} slots={n} critique_pass={cp} final_pass={fp}")
        if r["final_pass"]:
            print(f"  calls per final-passing variant: {r['api_calls'] / r['final_pass']:.2f}")

    # Paired counterfactual inside the retry run: what would attempt 1 alone have yielded?
    first = [x for x in retry["log_rows"] if x["attempt_number"] == 1]
    retried = [x for x in retry["log_rows"] if x["attempt_number"] == 2]
    print("\n[within the retry run]")
    print(f"  first-attempt critique pass : {_pct(sum(x['passed'] for x in first), len(first))}")
    print(f"  retries attempted           : {len(retried)}, rescued: {sum(x['passed'] for x in retried)}")

    lift = retry["final_pass"] - base["final_pass"]
    extra_calls = retry["api_calls"] - base["api_calls"]
    extra_cost = retry["cost_usd"] - base["cost_usd"]
    print("\n[delta: retry - no retry]")
    print(f"  final-passing variants : {lift:+d}")
    print(f"  extra API calls        : {extra_calls:+d}  (${extra_cost:+.4f})")
    if lift > 0:
        print(f"  extra calls per extra passing variant: {extra_calls / lift:.2f}")

    print("\n[all critique failures — notes]")
    for r in (base, retry):
        for x in r["log_rows"]:
            if not x["passed"]:
                print(f"  (max={r['max_attempts']} att={x['attempt_number']} {x['decision']}) "
                      f"[{x['variant_angle']}] {x['critique_notes']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--limit", type=int, default=5, help="max golden briefs to use")
    parser.add_argument("--count", type=int, default=5, help="variants per brief")
    parser.add_argument("--yes", action="store_true", help="actually call the real API")
    parser.add_argument("--json-out", help="optional path to dump raw results")
    args = parser.parse_args()

    examples = load_golden_set(GOLDEN_SET_PATH)[: args.limit]
    max_calls, est_usd = estimate_cost(len(examples), args.count)
    print(f"{len(examples)} briefs x {args.count} variants x 2 conditions -> up to "
          f"{max_calls} real API calls, est. <= ${est_usd:.2f} (rough upper bound).")
    if not args.yes:
        print("Re-run with --yes to spend it.")
        return

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_path = f"data/eval_logs/compare_retry_{stamp}.jsonl"
    base = run_condition(examples, max_attempts=1, count=args.count, log_path=log_path)
    retry = run_condition(examples, max_attempts=2, count=args.count, log_path=log_path)
    report(base, retry)
    print(f"\nPer-attempt log: {log_path}")
    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump({"no_retry": base, "retry": retry}, f, indent=2, default=str)


if __name__ == "__main__":
    main()
