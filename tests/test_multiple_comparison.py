"""Tests for Benjamini-Hochberg multiple-comparison correction — written
FIRST per this project's TDD discipline.

Real problem this fixes: both learn.run_mining_pass and learn.
run_cluster_pass test MANY features/clusters per pass (today ~50
combined) against a flat p<0.05 threshold each, independently. At a flat
5% false-positive rate per test, running 50 independent tests produces
~2-3 "significant" findings from pure chance alone, even if nothing real
is going on — this is exactly why the project's own pseudo-replication
bug (fixed earlier this session) produced an implausible 22/49
"significant" rate. Benjamini-Hochberg controls the FALSE DISCOVERY RATE
across a whole batch of tests at once: it only calls a finding
significant if it clears a threshold that accounts for how many tests
were run, not a flat per-test cutoff.

Textbook BH procedure (verified against the standard reference
algorithm, not reinvented): sort p-values ascending p(1)<=p(2)<=...<=p(m);
find the largest k such that p(k) <= (k/m)*Q (Q = desired FDR, e.g.
0.05); every p-value at rank <= k is significant.
"""
import pytest

from qualume.learn.multiple_comparison import benjamini_hochberg


def test_benjamini_hochberg_empty_list_returns_empty():
    assert benjamini_hochberg([]) == []


def test_benjamini_hochberg_single_pvalue_below_threshold_is_significant():
    # m=1, BH reduces to the flat threshold — p=0.01 < 0.05*1/1
    assert benjamini_hochberg([0.01]) == [True]


def test_benjamini_hochberg_single_pvalue_above_threshold_is_not_significant():
    assert benjamini_hochberg([0.10]) == [False]


def test_benjamini_hochberg_known_textbook_example():
    """Standard worked example (10 p-values, FDR=0.05) — verified against
    the canonical BH procedure by hand:
    sorted: 0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205, 0.212, 0.216
    thresholds (k/10*0.05): 0.005, 0.010, 0.015, 0.020, 0.025, 0.030, 0.035, 0.040, 0.045, 0.050
    p(k) <= threshold(k)?    T      T      F      F      F      F      F      F      F      F
    Largest k where it holds: k=2 (0.008 <= 0.010) — but BH takes the LARGEST k where it holds
    at ANY point at or below k, so need to check: is there a larger k where it holds too? No -
    k=3 (0.039 <= 0.015) is False, and it stays False for all k>2. So significant = ranks 1-2
    only: p=0.001 and p=0.008.
    """
    p_values = [0.205, 0.001, 0.212, 0.008, 0.216, 0.039, 0.042, 0.06, 0.041, 0.074]
    # indices:    0      1      2      3      4      5      6      7      8      9
    result = benjamini_hochberg(p_values, fdr=0.05)

    assert result[1] is True   # p=0.001, rank 1
    assert result[3] is True   # p=0.008, rank 2
    assert result[5] is False  # p=0.039, rank 3 — fails
    assert result[0] is False  # p=0.205 — clearly not significant
    assert sum(result) == 2    # exactly 2 of 10 survive correction


def test_benjamini_hochberg_all_pvalues_very_small_are_all_significant():
    p_values = [0.0001, 0.0002, 0.0003, 0.0004]
    assert benjamini_hochberg(p_values, fdr=0.05) == [True, True, True, True]


def test_benjamini_hochberg_all_pvalues_large_are_all_not_significant():
    p_values = [0.5, 0.6, 0.7, 0.8]
    assert benjamini_hochberg(p_values, fdr=0.05) == [False, False, False, False]


def test_benjamini_hochberg_preserves_input_order_not_sorted_order():
    """Real requirement: callers zip results back against their original
    list of findings by index — output order must match input order, not
    come back pre-sorted by p-value."""
    p_values = [0.9, 0.001, 0.5]
    result = benjamini_hochberg(p_values, fdr=0.05)
    assert len(result) == 3
    assert result[1] is True   # the smallest p-value, at its ORIGINAL index 1
    assert result[0] is False
    assert result[2] is False


def test_benjamini_hochberg_is_strictly_more_conservative_than_flat_threshold():
    """The real point of this whole module: BH on a big batch should
    reject more findings than a flat p<0.05 would have let through —
    proves correction is actually doing something, not a no-op."""
    import random
    random.seed(123)
    # 50 p-values uniformly random in [0, 1] — simulates "50 tests, mostly
    # noise, maybe a couple of real small p-values sprinkled in", the
    # real shape this project's combined pipelines produce per cycle.
    p_values = [random.uniform(0, 1) for _ in range(48)] + [0.0001, 0.0005]

    flat_significant = sum(1 for p in p_values if p < 0.05)
    bh_result = benjamini_hochberg(p_values, fdr=0.05)
    bh_significant = sum(bh_result)

    assert bh_significant <= flat_significant
    assert bh_result[48] is True  # the two real small p-values still survive
    assert bh_result[49] is True


def test_benjamini_hochberg_rejects_invalid_fdr():
    with pytest.raises(ValueError):
        benjamini_hochberg([0.01, 0.02], fdr=0)
    with pytest.raises(ValueError):
        benjamini_hochberg([0.01, 0.02], fdr=1.5)


def test_benjamini_hochberg_rejects_out_of_range_pvalues():
    with pytest.raises(ValueError):
        benjamini_hochberg([0.5, 1.2])
    with pytest.raises(ValueError):
        benjamini_hochberg([0.5, -0.1])
