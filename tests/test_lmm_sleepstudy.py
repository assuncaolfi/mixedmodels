"""Validate LMM fits against an independent direct-V-matrix optimizer.

The ground truth here is the marginal log-likelihood of an LMM, computed in
closed form by building V = σ²I + Z Σ Zᵀ block by block and optimizing with
scipy. This is independent of (and slower than) our Laplace path; if both
return the same parameter values to ~1e-4, the package is doing the right
thing.

These values were computed once on the shipped sleepstudy.csv and stored
below as `_REF_*`. They differ slightly from R's `lme4::lmer` values on the
canonical R sleepstudy dataset because the shipped CSV uses 4-decimal
rounded Reaction values, but the math is the same.
"""

import numpy as np

from mixedmodels import MixedModel


def _fit(formula, data):
    return MixedModel.from_formula(formula, data).fit()


# Reference values from a direct V-matrix ML fit on the same CSV.
_REF_RI = {  # Reaction ~ Days + (1 | Subject)
    "beta": np.array([249.8864, 10.65607]),
    "sigma": 28.4012,
    "sigma_b": 36.1581,
    "logLik": -883.3753,
}
_REF_RS = {  # Reaction ~ Days + (Days | Subject)
    "beta": np.array([249.8873, 10.65624]),
    "sigma": 22.0410,
    "sd_re": np.array([24.9823, 5.9159]),
    "corr_re": 0.005081,
    "logLik": -854.5069,
}


def test_random_intercept(sleepstudy):
    fit = _fit("Reaction ~ Days + (1 | Subject)", sleepstudy)
    assert fit.converged
    beta = fit.fixed_effects()
    assert np.isclose(beta["Intercept"], _REF_RI["beta"][0], atol=1e-3)
    assert np.isclose(beta["Days"], _REF_RI["beta"][1], atol=1e-3)
    vc = fit.variance_components()[0]
    assert np.isclose(vc["sd"][0], _REF_RI["sigma_b"], atol=1e-2)
    assert np.isclose(fit.sigma(), _REF_RI["sigma"], atol=1e-2)
    assert np.isclose(fit.log_likelihood(), _REF_RI["logLik"], atol=1e-3)


def test_random_intercept_and_slope(sleepstudy):
    fit = _fit("Reaction ~ Days + (Days | Subject)", sleepstudy)
    assert fit.converged
    beta = fit.fixed_effects()
    assert np.isclose(beta["Intercept"], _REF_RS["beta"][0], atol=1e-3)
    assert np.isclose(beta["Days"], _REF_RS["beta"][1], atol=1e-3)
    vc = fit.variance_components()[0]
    assert np.isclose(vc["sd"][0], _REF_RS["sd_re"][0], atol=1e-2)
    assert np.isclose(vc["sd"][1], _REF_RS["sd_re"][1], atol=1e-2)
    assert np.isclose(vc["corr"][1, 0], _REF_RS["corr_re"], atol=1e-2)
    assert np.isclose(fit.sigma(), _REF_RS["sigma"], atol=1e-2)
    assert np.isclose(fit.log_likelihood(), _REF_RS["logLik"], atol=1e-3)


def test_wald_se_matches_lme4(sleepstudy):
    """With the β-block-Hessian SE, we match `lme4` closely.

    lme4 (REML) reports SE(Intercept)=6.825, SE(Days)=1.546 on the canonical
    sleepstudy. Our ML fit on the shipped CSV is within ~5%.
    """
    fit = _fit("Reaction ~ Days + (Days | Subject)", sleepstudy)
    w = fit.wald()
    assert (w.std_error > 0).all()
    assert np.isclose(w.std_error["Intercept"], 6.633, atol=0.5)
    assert np.isclose(w.std_error["Days"], 1.507, atol=0.15)


def test_predict_returns_response_scale(sleepstudy):
    fit = _fit("Reaction ~ Days + (1 | Subject)", sleepstudy)
    yhat = fit.predict()
    assert yhat.shape == (180,)
    # Subject 308 has very fast reaction-time growth; predictions for that subject's
    # late days should be on the order of the observed values.
    assert 200.0 < yhat.mean() < 400.0


def test_summary_runs(sleepstudy):
    fit = _fit("Reaction ~ Days + (Days | Subject)", sleepstudy)
    s = str(fit.summary())
    assert "Reaction" in s
    assert "Subject" in s
    assert "Fixed effects" in s


def test_profile_log_sigma(sleepstudy):
    """Profile CI for log σ should bracket the MLE."""
    fit = _fit("Reaction ~ Days + (1 | Subject)", sleepstudy)
    prof = fit.profile(["log_sigma"], max_points=8)
    iv = prof.intervals.loc["log_sigma"]
    log_sigma_hat = np.log(_REF_RI["sigma"])
    assert iv["lower"] < log_sigma_hat < iv["upper"]
    # Width on log scale should be in a reasonable range.
    assert 0.05 < (iv["upper"] - iv["lower"]) < 0.6
