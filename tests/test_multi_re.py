"""Multiple grouping factors (dense `H_bb` path).

When the formula contains more than one `(... | g)` term, the model uses
`DenseHessian` rather than the block-diagonal fast path. We verify:

  * The dense backend is selected.
  * The fit recovers simulated parameters in a two-grouping-factor LMM.
  * The fit recovers simulated parameters in a two-grouping-factor GLMM.
"""

import numpy as np
import pandas as pd

from mixedmodels import Bernoulli, MixedModel
from mixedmodels.linalg import DenseHessian


def _fit(formula, data, **kw):
    return MixedModel.from_formula(formula, data, **kw).fit()


def test_two_grouping_factors_uses_dense_backend():
    rng = np.random.default_rng(0)
    n = 50
    df = pd.DataFrame(
        {
            "y": rng.normal(size=n),
            "x": rng.normal(size=n),
            "g1": rng.integers(0, 5, size=n),
            "g2": rng.integers(0, 4, size=n),
        }
    )
    fit = _fit("y ~ x + (1 | g1) + (1 | g2)", df)
    assert isinstance(fit._structure, DenseHessian)


def test_lmm_two_grouping_factors_recovery():
    """LMM with two crossed grouping factors recovers β, σ, and both σ_b's."""
    rng = np.random.default_rng(123)
    G1, G2 = 25, 20
    n = 600
    g1 = rng.integers(0, G1, size=n)
    g2 = rng.integers(0, G2, size=n)
    x = rng.normal(size=n)
    sigma_1, sigma_2, sigma = 0.8, 0.5, 0.4
    b1 = rng.normal(scale=sigma_1, size=G1)
    b2 = rng.normal(scale=sigma_2, size=G2)
    beta = np.array([1.0, 0.3])
    mu = beta[0] + beta[1] * x + b1[g1] + b2[g2]
    y = mu + rng.normal(scale=sigma, size=n)
    df = pd.DataFrame({"y": y, "x": x, "g1": g1, "g2": g2})

    fit = _fit("y ~ x + (1 | g1) + (1 | g2)", df)
    assert fit.converged
    b = fit.fixed_effects()
    assert abs(b["Intercept"] - 1.0) < 0.4
    assert abs(b["x"] - 0.3) < 0.1
    vcs = fit.variance_components()
    # Order matches the formula: (1|g1) first, then (1|g2).
    assert abs(vcs[0]["sd"][0] - sigma_1) < 0.15
    assert abs(vcs[1]["sd"][0] - sigma_2) < 0.15
    assert abs(fit.sigma() - sigma) < 0.05


def test_glmm_two_grouping_factors_recovery():
    """Bernoulli GLMM with two crossed grouping factors recovers β."""
    rng = np.random.default_rng(321)
    G1, G2 = 20, 15
    n = 800
    g1 = rng.integers(0, G1, size=n)
    g2 = rng.integers(0, G2, size=n)
    x = rng.normal(size=n)
    b1 = rng.normal(scale=0.6, size=G1)
    b2 = rng.normal(scale=0.4, size=G2)
    eta = -0.2 + 0.5 * x + b1[g1] + b2[g2]
    p = 1.0 / (1.0 + np.exp(-eta))
    y = rng.binomial(1, p).astype(float)
    df = pd.DataFrame({"y": y, "x": x, "g1": g1, "g2": g2})

    fit = _fit("y ~ x + (1 | g1) + (1 | g2)", df, family=Bernoulli())
    assert fit.converged
    b = fit.fixed_effects()
    assert abs(b["Intercept"] - (-0.2)) < 0.3
    assert abs(b["x"] - 0.5) < 0.2
