"""Non-canonical link functions.

Each family is constructed explicitly as `Family(Link())`. For each
non-canonical link we exercise we check against `statsmodels.GLM` in the
no-random-effects limit, and (where it matters) do a GLMM
simulation-recovery test.
"""

import numpy as np
import pandas as pd
import pytest

statsmodels = pytest.importorskip("statsmodels.api")
import statsmodels.api as sm  # noqa: E402

from mixedmodels import (  # noqa: E402
    Bernoulli,
    CloglogLink,
    Gaussian,
    LogLink,
    MixedModel,
    Poisson,
    ProbitLink,
    SqrtLink,
)


def _fit(formula, data, **kw):
    return MixedModel.from_formula(formula, data, **kw).fit()


# ---------------------------------------------------------------------------
# Bernoulli + probit
# ---------------------------------------------------------------------------


def test_bernoulli_probit_matches_statsmodels():
    rng = np.random.default_rng(0)
    n = 500
    x = rng.normal(size=n)
    from scipy.stats import norm

    p = norm.cdf(-0.2 + 0.7 * x)
    y = rng.binomial(1, p).astype(float)
    df = pd.DataFrame({"y": y, "x": x})

    fit = _fit("y ~ x", df, family=Bernoulli(ProbitLink()))
    sm_fit = sm.GLM(
        y, sm.add_constant(x), family=sm.families.Binomial(sm.families.links.Probit())
    ).fit()
    np.testing.assert_allclose(fit.fixed_effects().to_numpy(), sm_fit.params, atol=2e-3)


def test_bernoulli_cloglog_matches_statsmodels():
    rng = np.random.default_rng(1)
    n = 600
    x = rng.normal(size=n)
    eta = -0.1 + 0.5 * x
    p = 1.0 - np.exp(-np.exp(eta))
    y = rng.binomial(1, p).astype(float)
    df = pd.DataFrame({"y": y, "x": x})

    fit = _fit("y ~ x", df, family=Bernoulli(CloglogLink()))
    sm_fit = sm.GLM(
        y, sm.add_constant(x), family=sm.families.Binomial(sm.families.links.CLogLog())
    ).fit()
    np.testing.assert_allclose(fit.fixed_effects().to_numpy(), sm_fit.params, atol=2e-3)


def test_bernoulli_probit_glmm_recovery():
    rng = np.random.default_rng(2)
    G, n_per = 50, 30
    n = G * n_per
    g = np.repeat(np.arange(G), n_per)
    x = rng.normal(size=n)
    b = rng.normal(scale=0.7, size=G)
    from scipy.stats import norm

    eta = -0.2 + 0.6 * x + b[g]
    y = rng.binomial(1, norm.cdf(eta)).astype(float)
    df = pd.DataFrame({"y": y, "x": x, "g": g})

    fit = _fit("y ~ x + (1 | g)", df, family=Bernoulli(ProbitLink()))
    assert fit.converged
    beta = fit.fixed_effects()
    assert abs(beta["Intercept"] - (-0.2)) < 0.25
    assert abs(beta["x"] - 0.6) < 0.15
    assert abs(fit.variance_components()[0]["sd"][0] - 0.7) < 0.2


# ---------------------------------------------------------------------------
# Gaussian + log
# ---------------------------------------------------------------------------


def test_gaussian_log_matches_statsmodels():
    rng = np.random.default_rng(3)
    n = 400
    x = rng.normal(size=n)
    mu = np.exp(0.4 + 0.3 * x)
    y = mu + rng.normal(scale=0.5, size=n)
    df = pd.DataFrame({"y": y, "x": x})

    fit = _fit("y ~ x", df, family=Gaussian(LogLink()))
    sm_fit = sm.GLM(
        y, sm.add_constant(x), family=sm.families.Gaussian(sm.families.links.Log())
    ).fit()
    np.testing.assert_allclose(fit.fixed_effects().to_numpy(), sm_fit.params, atol=5e-3)


# ---------------------------------------------------------------------------
# Poisson + sqrt
# ---------------------------------------------------------------------------


def test_poisson_sqrt_matches_statsmodels():
    rng = np.random.default_rng(4)
    n = 500
    x = rng.normal(size=n)
    eta = 2.0 + 0.4 * x
    mu = eta * eta  # sqrt link inverse
    y = rng.poisson(mu).astype(float)
    df = pd.DataFrame({"y": y, "x": x})

    fit = _fit("y ~ x", df, family=Poisson(SqrtLink()))
    sm_fit = sm.GLM(
        y, sm.add_constant(x), family=sm.families.Poisson(sm.families.links.Sqrt())
    ).fit()
    np.testing.assert_allclose(fit.fixed_effects().to_numpy(), sm_fit.params, atol=5e-3)


# ---------------------------------------------------------------------------
# Round-trips
# ---------------------------------------------------------------------------


def test_link_attribute_on_fitted_model():
    rng = np.random.default_rng(5)
    n = 200
    x = rng.normal(size=n)
    y = rng.binomial(1, 0.5, size=n).astype(float)
    df = pd.DataFrame({"y": y, "x": x})
    fit = _fit("y ~ x", df, family=Bernoulli(CloglogLink()))
    assert isinstance(fit.family.link, CloglogLink)
    assert fit.family.link.name == "cloglog"


def test_default_family_is_gaussian_identity():
    """Calling `from_formula` with no `family=` uses Gaussian + identity."""
    rng = np.random.default_rng(6)
    n = 150
    y = rng.normal(size=n)
    x = rng.normal(size=n)
    df = pd.DataFrame({"y": y, "x": x})
    fit = _fit("y ~ x", df)
    assert isinstance(fit.family, Gaussian)
    assert fit.family.link.name == "identity"
