from qualume.eval.harness import run_constraint_checks


def test_run_constraint_checks_fails_on_missing_required_term():
    example = {"must_include": ["journal"], "must_avoid": []}
    assert run_constraint_checks("Buy this amazing planner today", example) is False


def test_run_constraint_checks_fails_on_banned_term():
    example = {"must_include": [], "must_avoid": ["guaranteed results"]}
    assert run_constraint_checks("This journal gives guaranteed results", example) is False


def test_run_constraint_checks_passes_clean_text():
    example = {"must_include": ["journal"], "must_avoid": ["guaranteed results"]}
    assert run_constraint_checks("This journal helps you plan your week", example) is True
