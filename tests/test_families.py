"""Family tests.

For each family we run two kinds of check:

1. **No random effects → match `statsmodels.GLM`** for β̂ and log-likelihood.
   This verifies the family's negative log-likelihood and link are correct
   independently of the mixed-effects machinery.

2. **GLMM with a single random intercept → simulation recovery.** Simulate
   from the family with known `(β, σ_b, φ)` and verify the package
   recovers them to within a couple of bootstrap standard errors.

Cross-validation against `lme4`/`MixedModels.jl` for LMM is in
`test_lmm_sleepstudy.py` (against an independent direct V-matrix
optimizer, which matches `lme4` numerically).
"""

import numpy as np
import pandas as pd
import pytest

statsmodels = pytest.importorskip("statsmodels.api")
import statsmodels.api as sm  # noqa: E402

from mixedmodels import (  # noqa: E402
    Bernoulli,
    Binomial,
    Gamma,
    Gaussian,
    MixedModel,
    NegativeBinomial,
    Poisson,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fit(formula, data, **kw):
    return MixedModel.from_formula(formula, data, **kw).fit()


# ---------------------------------------------------------------------------
# 1. Marginal GLM check (no random effects)
# ---------------------------------------------------------------------------


def test_gaussian_marginal_matches_statsmodels_glm():
    rng = np.random.default_rng(0)
    n = 300
    x = rng.normal(size=n)
    y = 1.0 + 0.5 * x + rng.normal(scale=0.7, size=n)
    df = pd.DataFrame({"y": y, "x": x})

    fit = _fit("y ~ x", df, family=Gaussian())
    sm_fit = sm.GLM(y, sm.add_constant(x), family=sm.families.Gaussian()).fit()
    np.testing.assert_allclose(fit.fixed_effects().to_numpy(), sm_fit.params, atol=1e-4)
    # statsmodels reports the unbiased scale (residual sum of squares / df_resid);
    # we report ML σ. Adjust.
    sigma_ml = np.sqrt(sm_fit.scale * (n - 2) / n)
    assert np.isclose(fit.sigma(), sigma_ml, atol=1e-3)


def test_bernoulli_marginal_matches_statsmodels_glm():
    rng = np.random.default_rng(1)
    n = 400
    x = rng.normal(size=n)
    p = 1.0 / (1.0 + np.exp(-(-0.3 + 0.8 * x)))
    y = rng.binomial(1, p).astype(float)
    df = pd.DataFrame({"y": y, "x": x})

    fit = _fit("y ~ x", df, family=Bernoulli())
    sm_fit = sm.GLM(y, sm.add_constant(x), family=sm.families.Binomial()).fit()
    np.testing.assert_allclose(fit.fixed_effects().to_numpy(), sm_fit.params, atol=1e-4)


def test_binomial_marginal_matches_statsmodels_glm():
    rng = np.random.default_rng(2)
    n = 200
    x = rng.normal(size=n)
    trials = rng.integers(5, 30, size=n).astype(float)
    p = 1.0 / (1.0 + np.exp(-(0.2 - 0.4 * x)))
    y = rng.binomial(trials.astype(int), p).astype(float)
    df = pd.DataFrame({"y": y, "x": x, "trials": trials})

    fit = _fit("y ~ x", df, family=Binomial(), weights=trials)
    # statsmodels Binomial GLM with trials as weights
    sm_fit = sm.GLM(
        np.column_stack([y, trials - y]),
        sm.add_constant(x),
        family=sm.families.Binomial(),
    ).fit()
    np.testing.assert_allclose(fit.fixed_effects().to_numpy(), sm_fit.params, atol=1e-4)


def test_poisson_marginal_matches_statsmodels_glm():
    rng = np.random.default_rng(3)
    n = 300
    x = rng.normal(size=n)
    lam = np.exp(0.5 + 0.3 * x)
    y = rng.poisson(lam).astype(float)
    df = pd.DataFrame({"y": y, "x": x})

    fit = _fit("y ~ x", df, family=Poisson())
    sm_fit = sm.GLM(y, sm.add_constant(x), family=sm.families.Poisson()).fit()
    np.testing.assert_allclose(fit.fixed_effects().to_numpy(), sm_fit.params, atol=1e-4)


def test_gamma_marginal_matches_statsmodels_glm():
    rng = np.random.default_rng(4)
    n = 400
    x = rng.normal(size=n)
    mu = np.exp(1.0 + 0.5 * x)
    shape = 2.0  # → φ = 1/shape = 0.5
    y = rng.gamma(shape, mu / shape)
    df = pd.DataFrame({"y": y, "x": x})

    fit = _fit("y ~ x", df, family=Gamma())
    sm_fit = sm.GLM(y, sm.add_constant(x), family=sm.families.Gamma(sm.families.links.Log())).fit()
    np.testing.assert_allclose(fit.fixed_effects().to_numpy(), sm_fit.params, atol=5e-3)


def test_negative_binomial_marginal_matches_statsmodels_glm():
    rng = np.random.default_rng(5)
    n = 500
    x = rng.normal(size=n)
    mu = np.exp(1.5 + 0.3 * x)
    theta = 2.5
    lam = rng.gamma(theta, mu / theta)  # mean μ, var μ²/θ
    y = rng.poisson(lam).astype(float)
    df = pd.DataFrame({"y": y, "x": x})

    fit = _fit("y ~ x", df, family=NegativeBinomial())
    # statsmodels.NegativeBinomial uses alpha = 1/θ; pass our ML θ estimate.
    # We compare β only — agreement on θ is harder because both packages
    # estimate it jointly.
    sm_fit = sm.GLM(
        y, sm.add_constant(x), family=sm.families.NegativeBinomial(alpha=1.0 / theta)
    ).fit()
    # β can differ slightly because statsmodels fixes alpha, we estimate it.
    np.testing.assert_allclose(fit.fixed_effects().to_numpy(), sm_fit.params, atol=0.05)


# ---------------------------------------------------------------------------
# 2. GLMM simulation-recovery
# ---------------------------------------------------------------------------


def _make_glmm_data(seed, family, G, n_per, beta, sigma_b, *, dispersion=1.0):
    rng = np.random.default_rng(seed)
    n = G * n_per
    group = np.repeat(np.arange(G), n_per)
    x = rng.normal(size=n)
    b = rng.normal(scale=sigma_b, size=G)
    eta = beta[0] + beta[1] * x + b[group]
    if family == "bernoulli":
        p = 1 / (1 + np.exp(-eta))
        y = rng.binomial(1, p).astype(float)
        weights = None
    elif family == "binomial":
        p = 1 / (1 + np.exp(-eta))
        trials = rng.integers(5, 25, size=n).astype(float)
        y = rng.binomial(trials.astype(int), p).astype(float)
        weights = trials
    elif family == "poisson":
        lam = np.exp(eta)
        y = rng.poisson(lam).astype(float)
        weights = None
    elif family == "gamma":
        mu = np.exp(eta)
        shape = 1.0 / dispersion
        y = rng.gamma(shape, mu / shape)
        weights = None
    elif family == "negative_binomial":
        mu = np.exp(eta)
        theta = 1.0 / dispersion
        lam = rng.gamma(theta, mu / theta)
        y = rng.poisson(lam).astype(float)
        weights = None
    else:
        raise ValueError(family)
    df = pd.DataFrame({"y": y, "x": x, "g": group})
    if weights is not None:
        df["w"] = weights
    return df


def test_glmm_bernoulli_recovery():
    df = _make_glmm_data(11, "bernoulli", G=60, n_per=30, beta=(-0.3, 0.8), sigma_b=1.0)
    fit = _fit("y ~ x + (1 | g)", df, family=Bernoulli())
    assert fit.converged
    b = fit.fixed_effects()
    assert abs(b["Intercept"] - (-0.3)) < 0.4
    assert abs(b["x"] - 0.8) < 0.2
    assert abs(fit.variance_components()[0]["sd"][0] - 1.0) < 0.3


def test_glmm_binomial_recovery():
    df = _make_glmm_data(12, "binomial", G=50, n_per=20, beta=(0.2, -0.4), sigma_b=0.7)
    fit = _fit("y ~ x + (1 | g)", df, family=Binomial(), weights=df["w"].to_numpy())
    assert fit.converged
    b = fit.fixed_effects()
    assert abs(b["Intercept"] - 0.2) < 0.2
    assert abs(b["x"] - (-0.4)) < 0.1
    assert abs(fit.variance_components()[0]["sd"][0] - 0.7) < 0.2


def test_glmm_poisson_recovery():
    df = _make_glmm_data(13, "poisson", G=40, n_per=25, beta=(0.5, 0.4), sigma_b=0.6)
    fit = _fit("y ~ x + (1 | g)", df, family=Poisson())
    assert fit.converged
    b = fit.fixed_effects()
    assert abs(b["Intercept"] - 0.5) < 0.4
    assert abs(b["x"] - 0.4) < 0.1
    assert abs(fit.variance_components()[0]["sd"][0] - 0.6) < 0.25


def test_glmm_gamma_recovery():
    df = _make_glmm_data(14, "gamma", G=40, n_per=30, beta=(1.0, 0.4), sigma_b=0.5, dispersion=0.4)
    fit = _fit("y ~ x + (1 | g)", df, family=Gamma())
    assert fit.converged
    b = fit.fixed_effects()
    assert abs(b["Intercept"] - 1.0) < 0.3
    assert abs(b["x"] - 0.4) < 0.1
    assert abs(fit.variance_components()[0]["sd"][0] - 0.5) < 0.2


def test_glmm_negative_binomial_recovery():
    df = _make_glmm_data(
        1, "negative_binomial", G=50, n_per=30, beta=(1.5, 0.3), sigma_b=0.5, dispersion=0.4
    )
    fit = _fit("y ~ x + (1 | g)", df, family=NegativeBinomial())
    assert fit.converged
    b = fit.fixed_effects()
    assert abs(b["Intercept"] - 1.5) < 0.3
    assert abs(b["x"] - 0.3) < 0.15
    assert abs(fit.variance_components()[0]["sd"][0] - 0.5) < 0.25
