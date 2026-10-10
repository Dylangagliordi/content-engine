from types import SimpleNamespace

from qualume.eval.compare_retry_benefit import CallCounter, estimate_cost


def _response(model, input_tokens, output_tokens, cache_read=0, cache_creation=0):
    msg = SimpleNamespace(
        response_metadata={"model_name": model},
        usage_metadata={
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "input_token_details": {"cache_read": cache_read, "cache_creation": cache_creation},
        },
    )
    return SimpleNamespace(generations=[[SimpleNamespace(message=msg)]])


def test_call_counter_counts_calls_and_prices_cache_tokens_separately():
    c = CallCounter()
    # Dated snapshot id must map onto the priced alias.
    c.on_llm_end(_response("claude-sonnet-4-5-20250929", 1_000_000, 0))
    c.on_llm_end(_response("claude-haiku-4-5", 1_000_000, 1_000_000,
                           cache_read=500_000, cache_creation=250_000))
    assert c.calls == {"claude-sonnet-4-5": 1, "claude-haiku-4-5": 1}
    # sonnet: 1M uncached in @ $3 = 3.00
    # haiku: 250k uncached @ $1 = 0.25, 250k write @ 1.25 = 0.3125,
    #        500k read @ 0.1 = 0.05, 1M out @ $5 = 5.00
    assert abs(c.cost_usd() - (3.00 + 0.25 + 0.3125 + 0.05 + 5.00)) < 1e-9


def test_estimate_cost_assumes_every_variant_retries_once():
    calls, usd = estimate_cost(n_briefs=1, count=2)
    # no-retry: 1 ideate + 2 gen + 2 critique; retry worst case: 1 + 4 gen + 4 critique
    assert calls == 5 + 9
    assert usd > 0
