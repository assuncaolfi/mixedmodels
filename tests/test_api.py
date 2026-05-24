"""Surface-level tests for the explicit accessor API."""

import numpy as np
import pandas as pd
import pytest

from mixedmodels import MixedModel


@pytest.fixture(scope="module")
def fit(sleepstudy):
    return MixedModel.from_formula("Reaction ~ Days + (Days | Subject)", sleepstudy).fit()


# ------ Fixed-effects accessors ------


def test_coefficients_is_alias_of_fixed_effects(fit):
    assert isinstance(fit.coefficients(), pd.Series)
    np.testing.assert_array_equal(fit.coefficients().to_numpy(), fit.fixed_effects().to_numpy())


def test_covariance_matrix_is_square_named_dataframe(fit):
    V = fit.covariance_matrix()
    assert isinstance(V, pd.DataFrame)
    assert list(V.index) == list(V.columns) == ["Intercept", "Days"]
    # SPD
    eig = np.linalg.eigvalsh(V.to_numpy())
    assert (eig > 0).all()
    # Diagonal should match standard_errors² to numerical precision.
    np.testing.assert_allclose(
        np.sqrt(np.diag(V.to_numpy())), fit.standard_errors().to_numpy(), rtol=1e-10
    )


def test_standard_errors(fit):
    se = fit.standard_errors()
    assert isinstance(se, pd.Series)
    assert list(se.index) == ["Intercept", "Days"]
    assert (se > 0).all()


def test_coefficients_table(fit):
    tab = fit.coefficients_table()
    assert isinstance(tab, pd.DataFrame)
    assert {"estimate", "std_error", "z", "p", "ci_lower", "ci_upper"} <= set(tab.columns)
    assert list(tab.index) == ["Intercept", "Days"]


# ------ Likelihood / IC ------


def test_deviance_relation(fit):
    assert np.isclose(fit.deviance(), -2.0 * fit.log_likelihood())


def test_n_obs_dof_aic_bic(fit):
    assert fit.n_observations() == 180
    # 2 fixed + 1 dispersion + 3 cholesky entries = 6
    assert fit.degrees_of_freedom() == 6
    aic = fit.aic()
    bic = fit.bic()
    expected_aic = 2 * 6 - 2 * fit.log_likelihood()
    expected_bic = np.log(180) * 6 - 2 * fit.log_likelihood()
    assert np.isclose(aic, expected_aic)
    assert np.isclose(bic, expected_bic)
    assert bic > aic


# ------ Random effects ------


def test_random_effects_shape(fit):
    re = fit.random_effects()
    assert len(re) == 1
    df = re[0]
    assert df.shape == (18, 2)  # 18 subjects, 2 cols
    assert df.index.name == "Subject"


def test_variance_components(fit):
    vc = fit.variance_components()
    assert len(vc) == 1
    blk = vc[0]
    assert {"name", "group", "columns", "cov", "sd", "corr"} <= set(blk)
    assert blk["group"] == "Subject"


# ------ Fitted, residuals, predict ------


def test_fitted_predict_consistency(fit):
    np.testing.assert_array_equal(fit.fitted(), fit.predict())


def test_residuals_types(fit):
    y = fit.matrices.y
    mu = fit.fitted()
    r_resp = fit.residuals("response")
    np.testing.assert_allclose(r_resp, y - mu)
    r_pear = fit.residuals("pearson")
    np.testing.assert_allclose(r_pear, (y - mu) / fit.sigma(), rtol=1e-10)
    np.testing.assert_allclose(fit.residuals("working"), r_resp)


def test_residuals_unknown_type_raises(fit):
    with pytest.raises(ValueError, match="Unknown residual type"):
        fit.residuals("nope")


# ------ Simulation ------


def test_simulate_shape_and_seed_reproducibility(fit):
    y1 = fit.simulate(n=3, seed=42)
    assert y1.shape == (3, 180)
    y2 = fit.simulate(n=3, seed=42)
    np.testing.assert_array_equal(y1, y2)
    y_single = fit.simulate(n=1, seed=0)
    assert y_single.shape == (180,)


def test_simulate_marginal_mean(fit):
    """Simulate draws fresh `b ~ N(0, Σ̂)` (lme4's `use.u=FALSE` default)."""
    sims = fit.simulate(n=200, seed=0)
    mean_sim = sims.mean(axis=0)
    pop_mu = fit.predict(include_re=False)
    assert np.mean(np.abs(mean_sim - pop_mu)) < 5.0


# ------ Unified confidence_intervals ------


def test_confidence_intervals_wald(fit):
    ci = fit.confidence_intervals(method="wald")
    assert list(ci.columns) == ["estimate", "lower", "upper"]
    assert list(ci.index) == ["Intercept", "Days"]
    assert (ci["lower"] < ci["estimate"]).all()
    assert (ci["estimate"] < ci["upper"]).all()
    ci_90 = fit.confidence_intervals(method="wald", level=0.90)
    assert (ci_90["upper"] - ci_90["lower"] < ci["upper"] - ci["lower"]).all()


def test_confidence_intervals_select_params(fit):
    ci = fit.confidence_intervals(params=["Days"], method="wald")
    assert list(ci.index) == ["Days"]


def test_confidence_intervals_profile_logsigma(fit):
    ci = fit.confidence_intervals(params=["log_sigma"], method="profile")
    assert "log_sigma" in ci.index
    row = ci.loc["log_sigma"]
    assert row["lower"] < row["estimate"] < row["upper"]


def test_confidence_intervals_unknown_method(fit):
    with pytest.raises(ValueError, match="Unknown method"):
        fit.confidence_intervals(method="bayes")


# ------ Bookkeeping ------


def test_formula_attr(fit):
    assert "Reaction" in fit.formula
    assert "Subject" in fit.formula


def test_converged_flag(fit):
    assert fit.converged is True
