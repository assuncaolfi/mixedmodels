from mixedmodels.formula import parse_formula


def test_parses_intercept_and_slope():
    p = parse_formula("y ~ 1 + x + (1 + x | g)")
    assert p.response == "y"
    assert p.fixed_rhs == "1 + x"
    assert p.random == [("1 + x", "g")]


def test_two_random_terms():
    p = parse_formula("y ~ x + (1 | s) + (1 | i)")
    assert p.fixed_rhs == "x"
    assert p.random == [("1", "s"), ("1", "i")]


def test_intercept_only_fixed():
    p = parse_formula("y ~ (1 | g)")
    assert p.fixed_rhs == "1"
    assert p.random == [("1", "g")]


def test_nested_paren_in_fixed_term_is_not_random():
    p = parse_formula("y ~ poly(x, 2) + (1 | g)")
    assert p.fixed_rhs.startswith("poly(x, 2)")
    assert p.random == [("1", "g")]
