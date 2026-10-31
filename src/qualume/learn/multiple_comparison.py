"""Benjamini-Hochberg multiple-comparison correction — controls the false
discovery rate (FDR) across a BATCH of statistical tests run in the same
pass, per this session's explicit design discussion.

Real problem this fixes: both learn.run_mining_pass and learn.
run_cluster_pass test MANY features per pass (today ~50 combined)
against a flat p<0.05 threshold, independently per test. At a flat 5%
false-positive rate per test, 50 independent tests produce ~2-3
"significant" findings from pure chance alone — this is exactly the
shape the project's own pseudo-replication bug (fixed earlier this
session) produced before being caught (22/49 "significant", implausibly
high). BH doesn't require every test to independently clear 0.05; it
asks whether a finding's p-value is small enough GIVEN how many tests
were run in the same batch.

Standard textbook BH procedure, not reinvented: sort p-values ascending
p(1)<=p(2)<=...<=p(m); find the largest rank k such that
p(k) <= (k/m)*fdr; every p-value at rank <= k is significant.
"""


def benjamini_hochberg(p_values: list[float], fdr: float = 0.05) -> list[bool]:
    """Returns a list of booleans, SAME LENGTH AND ORDER as p_values —
    True at index i means p_values[i] is significant after BH correction
    across this whole batch. Callers zip this back against their
    original list of findings by index; it is NOT returned pre-sorted.

    fdr: desired false discovery rate (0 < fdr <= 1), 0.05 by default —
    same conventional 5% this project already uses as its flat per-test
    threshold, now applied to the whole batch instead of each test alone.

    Raises ValueError on an empty p_values list treated as a no-op
    (returns [] rather than raising — nothing to correct), an invalid
    fdr, or any p-value outside [0, 1] (a real, fail-loud precondition —
    same discipline as every other stats function in this project)."""
    if not p_values:
        return []
    if not (0 < fdr <= 1):
        raise ValueError(f"fdr must be in (0, 1], got {fdr!r}")
    for p in p_values:
        if not (0 <= p <= 1):
            raise ValueError(f"p-value out of valid range [0, 1]: {p!r}")

    m = len(p_values)
    # Sort (p_value, original_index) pairs ascending by p-value.
    ranked = sorted(enumerate(p_values), key=lambda pair: pair[1])

    # Find the largest rank k (1-indexed) where p(k) <= (k/m)*fdr.
    largest_passing_rank = 0
    for rank, (_, p) in enumerate(ranked, start=1):
        if p <= (rank / m) * fdr:
            largest_passing_rank = rank

    result = [False] * m
    for rank, (original_index, _) in enumerate(ranked, start=1):
        if rank <= largest_passing_rank:
            result[original_index] = True
    return result
