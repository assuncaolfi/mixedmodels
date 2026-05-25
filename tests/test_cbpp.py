"""Validate the cbpp binomial GLMM against `glmmTMB`.

This model exercises:

  * Binomial family with `weights=` (trial counts).
  * Categorical fixed effects (`period` as a 4-level factor).
  * Two crossed random effects: `(1 | herd) + (1 | obs)`, where the
    observation-level RE is the standard trick for binomial overdispersion.

Reference values from R::

    cbpp$obs <- 1:nrow(cbpp)
    glmmTMB(cbind(incidence, size - incidence) ~ period + (1 | herd) + (1 | obs),
            family = binomial, data = cbpp)

    Fixed effects: (Intercept) = -1.500,  period2 = -1.227,
                   period3     = -1.329,  period4 = -1.866
    Random effects: σ_herd = 0.1839,  σ_obs = 0.8911
    logLik = -87.31916,  AIC = 186.638,  BIC = 198.790
"""

import numpy as np
import pandas as pd
import pytest

statsmodels = pytest.importorskip("statsmodels.api")
from scipy.special import gammaln  # noqa: E402
from statsmodels.datasets import get_rdataset  # noqa: E402

from mixedmodels import Binomial, MixedModel  # noqa: E402


@pytest.fixture(scope="module")
def cbpp() -> pd.DataFrame:
    d = get_rdataset("cbpp", "lme4").data.copy()
    d["obs"] = range(len(d))
    d["period"] = d["period"].astype("category")
    return d


@pytest.fixture(scope="module")
def fit(cbpp):
    return MixedModel.from_formula(
        "incidence ~ period + (1 | herd) + (1 | obs)",
        cbpp,
        family=Binomial(),
        weights=cbpp["size"].to_numpy(),
    ).fit()


def test_cbpp_fixed_effects_match_glmmtmb(fit):
    beta = fit.fixed_effects()
    np.testing.assert_allclose(beta["Intercept"], -1.500, atol=1e-3)
    np.testing.assert_allclose(beta["period[T.2]"], -1.227, atol=1e-3)
    np.testing.assert_allclose(beta["period[T.3]"], -1.329, atol=1e-3)
    np.testing.assert_allclose(beta["period[T.4]"], -1.866, atol=1e-3)


def test_cbpp_random_effects_match_glmmtmb(fit):
    vcs = {b["group"]: b["sd"][0] for b in fit.variance_components()}
    np.testing.assert_allclose(vcs["herd"], 0.1839, atol=1e-3)
    np.testing.assert_allclose(vcs["obs"], 0.8911, atol=1e-3)


def test_cbpp_loglik_matches_glmmtmb(fit, cbpp):
    """glmmTMB's logLik includes the log binomial coefficient `Σ log C(n_i, y_i)`,
    which we drop from the nll (it's constant in θ). Adding it back must match."""
    n = cbpp["size"].to_numpy().astype(float)
    y = cbpp["incidence"].to_numpy().astype(float)
    binom_coef = float((gammaln(n + 1) - gammaln(y + 1) - gammaln(n - y + 1)).sum())
    full_loglik = fit.log_likelihood() + binom_coef
    np.testing.assert_allclose(full_loglik, -87.31916, atol=1e-3)


def test_cbpp_converged(fit):
    assert fit.converged
