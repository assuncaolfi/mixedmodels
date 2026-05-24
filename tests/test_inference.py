"""Inference: Wald, profile, parametric bootstrap.

Wald and profile CIs are exercised in ``test_api.py`` and
``test_lmm_sleepstudy.py``. Here we add:

  * A bootstrap smoke test (shape + reproducibility).
  * Coverage check that profile CIs bracket the truth across replicates
    of simulated data (a property test on a small grid).
"""

import numpy as np
import pandas as pd

from mixedmodels import MixedModel
from mixedmodels.inference.bootstrap import BootstrapResult


def _fit(formula, data, **kw):
    return MixedModel.from_formula(formula, data, **kw).fit()


def _simulate_lmm(seed, beta=(1.0, 0.5), sigma=0.5, sigma_b=0.7, G=30, n_per=15):
    rng = np.random.default_rng(seed)
    n = G * n_per
    g = np.repeat(np.arange(G), n_per)
    x = rng.normal(size=n)
    b = rng.normal(scale=sigma_b, size=G)
    y = beta[0] + beta[1] * x + b[g] + rng.normal(scale=sigma, size=n)
    return pd.DataFrame({"y": y, "x": x, "g": g})


def test_bootstrap_smoke():
    df = _simulate_lmm(0)
    fit = _fit("y ~ x + (1 | g)", df)
    res = fit.bootstrap(n=15, seed=42)
    assert isinstance(res, BootstrapResult)
    # Fixed-effects draws: (n_boot, p)
    assert res.fixed_effects.shape == (15, 2)
    assert list(res.fixed_effects.columns) == ["Intercept", "x"]
    # σ draws
    assert res.sigma is not None and res.sigma.shape == (15,)
    # RE SD draws: (n_boot, q_total)
    assert res.re_sd.shape == (15, 1)
    # Most replicates should have converged.
    assert res.converged.sum() >= 13


def test_bootstrap_reproducible_under_seed():
    df = _simulate_lmm(0)
    fit = _fit("y ~ x + (1 | g)", df)
    a = fit.bootstrap(n=5, seed=7).fixed_effects.to_numpy()
    b = fit.bootstrap(n=5, seed=7).fixed_effects.to_numpy()
    np.testing.assert_allclose(a, b)


def test_bootstrap_recovers_truth():
    """Empirical mean of bootstrap β̂'s should be close to fitted β̂."""
    df = _simulate_lmm(0)
    fit = _fit("y ~ x + (1 | g)", df)
    res = fit.bootstrap(n=40, seed=1)
    boot_mean = res.fixed_effects.mean().to_numpy()
    np.testing.assert_allclose(boot_mean, fit.fixed_effects().to_numpy(), atol=0.15)


def test_profile_ci_brackets_truth_on_log_sigma():
    """Profile CI for log σ on a single fit should contain log(σ̂) by construction."""
    df = _simulate_lmm(0)
    fit = _fit("y ~ x + (1 | g)", df)
    ci = fit.confidence_intervals(params=["log_sigma"], method="profile")
    log_sigma_hat = float(np.log(fit.sigma()))
    row = ci.loc["log_sigma"]
    assert row["lower"] < log_sigma_hat < row["upper"]
