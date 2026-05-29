"""Parametrized test: each :class:`Case` is fit and compared against the
stored `glmmTMB` reference values.

Reference values were produced once by `scripts/fit_canonical.R` and live
verbatim in `tests/cases.py`. To regenerate after a `glmmTMB` upgrade or
when adding a new case, rerun that script and paste the JSON.
"""

from __future__ import annotations

import re

import numpy as np
import pytest
from scipy.special import gammaln

from mixedmodels import MixedModel

from cases import CASES, Case


# ---------------------------------------------------------------------------
# Extractors
# ---------------------------------------------------------------------------


_SD = re.compile(r"^sd\[(.+)\|(.+)\]$")
_CORR = re.compile(r"^corr\[(.+),(.+)\|(.+)\]$")


def _extract(fit, key: str, *, logLik_offset: float) -> float:
    """Pull the named quantity out of a fitted :class:`MixedModel`.

    The `"sigma"` key follows glmmTMB's conventions, which differ across
    families: it is the residual SD for Gaussian, sqrt(φ) for Gamma, and
    the size parameter θ for NB2. Our ``MixedModel.sigma()`` returns
    ``sqrt(dispersion)`` in every dispersion family, so for NB2 we convert
    to glmmTMB's scale on the fly.
    """
    if key == "sigma":
        s = fit.sigma()
        if s is None:
            return 1.0
        if fit.family.name == "negative_binomial":
            return 1.0 / (s * s)  # θ = 1/dispersion (glmmTMB's `sigma` for nbinom2)
        return s
    if key == "logLik":
        return fit.log_likelihood() + logLik_offset

    m = _SD.match(key)
    if m:
        col_name, grp = m.group(1), m.group(2)
        return _sd_for(fit, grp, col_name)
    m = _CORR.match(key)
    if m:
        a, b, grp = m.group(1), m.group(2), m.group(3)
        return _corr_for(fit, grp, a, b)

    # Otherwise it's a fixed-effect name.
    try:
        return float(fit.fixed_effects()[key])
    except KeyError as exc:
        raise KeyError(f"Unknown quantity {key!r} on the fitted model") from exc


def _sd_for(fit, group: str, col: str) -> float:
    for blk in fit.variance_components():
        if blk["group"] == group and col in blk["columns"]:
            return float(blk["sd"][blk["columns"].index(col)])
    raise KeyError(f"No RE block for group={group!r}, col={col!r}")


def _corr_for(fit, group: str, a: str, b: str) -> float:
    for blk in fit.variance_components():
        if blk["group"] == group and a in blk["columns"] and b in blk["columns"]:
            i, j = blk["columns"].index(a), blk["columns"].index(b)
            return float(blk["corr"][i, j])
    raise KeyError(f"No RE block for group={group!r} containing {a!r}, {b!r}")


# ---------------------------------------------------------------------------
# Family-specific logLik constants we drop from the joint nll
# ---------------------------------------------------------------------------


def _loglik_offset(family_name: str, df) -> float:
    """`fit.log_likelihood() + offset` matches the absolute glmmTMB logLik."""
    if family_name == "binomial":
        # We drop log C(n, y) from the per-obs Binomial nll because it's
        # constant in θ; glmmTMB reports the full likelihood.
        n = df["size"].to_numpy().astype(float)
        y = df["incidence"].to_numpy().astype(float)
        return float((gammaln(n + 1) - gammaln(y + 1) - gammaln(n - y + 1)).sum())
    return 0.0


# ---------------------------------------------------------------------------
# The single parametrized test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_case_matches_glmmtmb(case: Case):
    df = case.data_loader()
    w = case.weights(df) if case.weights else None
    fit = MixedModel.from_formula(case.formula, df, family=case.family, weights=w).fit()
    assert fit.converged, f"{case.name}: did not converge ({fit._opt_message})"

    offset = _loglik_offset(case.family.name, df)
    failures = []
    for key, expected in case.expected.items():
        got = _extract(fit, key, logLik_offset=offset)
        tol = case.default_tol(key)
        if not np.isclose(got, expected, atol=tol):
            failures.append((key, got, expected, tol))

    if failures:
        msg = [f"{case.name}: {len(failures)} quantity/quantities off:"]
        for key, got, expected, tol in failures:
            msg.append(f"  {key!r}: got {got!r}, expected {expected!r} (tol {tol!r})")
        pytest.fail("\n".join(msg))
