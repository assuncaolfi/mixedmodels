"""Formula parsing.

We accept lme4/MixedModels.jl-style formulas, e.g.::

    y ~ 1 + x + (1 + x | subject) + (1 | item)

Strategy: do a small string preprocessor that extracts each top-level
`(expr | group)` group, then delegate to `formulaic` for both the fixed-effects
formula and each random-effects expression. This is exactly what lme4 does
internally, and keeps the rest of the pipeline ignorant of the `(...|...)`
syntax.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class ParsedFormula:
    response: str
    fixed_rhs: str  # right-hand side suitable for formulaic
    random: list[tuple[str, str]]  # list of (expr, group) strings


def _find_top_level_paren_with_bar(s: str) -> tuple[int, int, str] | None:
    """Find the first top-level (...) group containing a '|' not inside nested parens.

    Returns (start_index, end_index, inner_content) or None.
    """
    depth = 0
    start = -1
    for i, c in enumerate(s):
        if c == "(":
            if depth == 0:
                start = i
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0 and start >= 0:
                inner = s[start + 1 : i]
                # ensure the | is at depth 0 within `inner`
                d = 0
                for ch in inner:
                    if ch == "(":
                        d += 1
                    elif ch == ")":
                        d -= 1
                    elif ch == "|" and d == 0:
                        return start, i, inner
                start = -1
    return None


def parse_formula(formula: str) -> ParsedFormula:
    if "~" not in formula:
        raise ValueError(f"Formula must contain '~': {formula!r}")
    lhs, rhs = formula.split("~", 1)
    response = lhs.strip()

    random: list[tuple[str, str]] = []
    fixed = rhs
    while True:
        match = _find_top_level_paren_with_bar(fixed)
        if match is None:
            break
        s, e, inner = match
        expr, group = inner.split("|", 1)
        random.append((expr.strip(), group.strip()))
        # Replace the whole `(...)` (including any surrounding +/- and whitespace)
        # with empty. We do this carefully so the remaining formula stays valid.
        fixed = fixed[:s] + fixed[e + 1 :]

    # Clean up stray operators left by removal.
    fixed = re.sub(r"\+\s*\+", "+", fixed)
    fixed = re.sub(r"^\s*\+", "", fixed)
    fixed = re.sub(r"\+\s*$", "", fixed)
    fixed = fixed.strip()
    if not fixed:
        fixed = "1"  # intercept-only fixed effects

    return ParsedFormula(response=response, fixed_rhs=fixed, random=random)
