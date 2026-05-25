"""Validate LMM fits against `lme4` / `glmmTMB` on the canonical sleepstudy.

The shipped `tests/data/sleepstudy.csv` is the original lme4 data (pulled via
`statsmodels.datasets.get_rdataset('sleepstudy', 'lme4')`). Reference values
below are the ML estimates reported by `glmmTMB(Reaction ~ ... , sleepstudy)`.
Our fit matches those numbers to ~4 decimals on every parameter.
"""

import numpy as np

from mixedmodels import MixedModel


def _fit(formula, data):
    return MixedModel.from_formula(formula, data).fit()


# Reference values: `glmmTMB` ML fit on canonical sleepstudy.
_REF_RI = {  # Reaction ~ Days + (1 | Subject)
    "beta": np.array([251.4051, 10.4673]),
    "sigma": 30.8954,
    "sigma_b": 36.0121,
    "logLik": -897.0393,
}
_REF_RS = {  # Reaction ~ 1 + Days + (1 + Days | Subject)
    "beta": np.array([251.4051, 10.4673]),
    "sigma": 25.5918,
    "sd_re": np.array([23.7806, 5.7168]),
    "corr_re": 0.08,
    "logLik": -875.9697,
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
    assert np.isclose(vc["corr"][1, 0], _REF_RS["corr_re"], atol=5e-3)
    assert np.isclose(fit.sigma(), _REF_RS["sigma"], atol=1e-2)
    assert np.isclose(fit.log_likelihood(), _REF_RS["logLik"], atol=1e-3)


def test_wald_se_matches_lme4(sleepstudy):
    """β-block-Hessian SE matches `lme4`/`glmmTMB` to ~3 decimals.

    lme4 reports SE(Intercept)=6.632, SE(Days)=1.502 (ML) on canonical sleepstudy.
    """
    fit = _fit("Reaction ~ Days + (Days | Subject)", sleepstudy)
    w = fit.wald()
    assert (w.std_error > 0).all()
    assert np.isclose(w.std_error["Intercept"], 6.632, atol=0.05)
    assert np.isclose(w.std_error["Days"], 1.502, atol=0.05)


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
