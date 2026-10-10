from qualume.run_batch import (
    build_universal_avoid_list,
    build_eval_example_for_brief,
    _extract_product_keyword,
)


def test_build_universal_avoid_list_unions_all_examples():
    golden_set = [
        {"brief": "a", "must_avoid": ["guaranteed results", "doctor"]},
        {"brief": "b", "must_avoid": ["cures", "doctor"]},
    ]
    result = build_universal_avoid_list(golden_set)
    assert result == sorted({"guaranteed results", "doctor", "cures"})


def test_extract_product_keyword_handles_unrelated_brief():
    # Regression test: a brief with no keyword overlap with any golden
    # example used to silently fall back to an unrelated example's
    # must_include list (e.g. requiring "journal" in a wallet ad).
    keyword = _extract_product_keyword("a premium leather wallet for men")
    assert keyword == "wallet"


def test_build_eval_example_for_brief_uses_brief_specific_keyword_not_golden_set():
    universal_avoid = ["guaranteed results", "cures"]
    example = build_eval_example_for_brief("a premium leather wallet for men", universal_avoid)
    assert example["must_include"] == ["wallet"]
    assert example["must_avoid"] == universal_avoid


def test_build_eval_example_for_brief_rejects_content_missing_its_own_keyword():
    from qualume.eval.harness import run_constraint_checks
    universal_avoid = ["guaranteed results"]
    example = build_eval_example_for_brief("a premium leather wallet for men", universal_avoid)
    # Copy about a completely different product should fail the gate.
    assert run_constraint_checks("This journal will change your life", example) is False
    assert run_constraint_checks("This wallet is handcrafted from full-grain leather", example) is True
