"""Predict on new data.

Covers:
  * `predict()` on training data agrees with `fitted()`.
  * `predict(newdata)` on data with a subset of the training group levels.
  * `predict(newdata)` with an unseen grouping-factor level raises by
    default and uses `b = 0` when `allow_new_levels=True`.
  * `predict(newdata, include_re=False)` is the population-level prediction.
"""

import numpy as np
import pandas as pd
import pytest

from mixedmodels import MixedModel


@pytest.fixture(scope="module")
def fit(sleepstudy):
    return MixedModel.from_formula("Reaction ~ Days + (Days | Subject)", sleepstudy).fit()


def test_predict_no_newdata_matches_fitted(fit):
    np.testing.assert_array_equal(fit.predict(), fit.fitted())


def test_predict_population_level_drops_random(fit, sleepstudy):
    # Population-level prediction is X β̂; it must not depend on the per-subject draws.
    pop = fit.predict(sleepstudy, include_re=False)
    cond = fit.predict(sleepstudy, include_re=True)
    assert pop.shape == (180,)
    # For subjects whose random effects are nonzero (essentially all 18), the
    # population-level prediction differs from the conditional one.
    assert not np.allclose(pop, cond)


def test_predict_known_subset_matches_fitted_rows(fit, sleepstudy):
    """A subset of the training rows should give the same fitted means."""
    subset = sleepstudy.iloc[10:30].reset_index(drop=True)
    pred = fit.predict(subset)
    np.testing.assert_allclose(pred, fit.fitted()[10:30], rtol=1e-10, atol=1e-10)


def test_predict_unknown_level_raises_by_default(fit, sleepstudy):
    newdata = sleepstudy.iloc[:5].copy()
    newdata["Subject"] = 99999  # not in training
    with pytest.raises(ValueError, match="not seen during fitting"):
        fit.predict(newdata)


def test_predict_unknown_level_allowed_uses_zero_random_effect(fit, sleepstudy):
    """With `allow_new_levels=True`, predictions on an unseen subject equal the
    population-level prediction (b = 0 for that subject)."""
    newdata = sleepstudy.iloc[:10].copy()
    newdata["Subject"] = 99999  # unseen subject; covariates kept
    pred_with_re = fit.predict(newdata, allow_new_levels=True)
    pred_pop = fit.predict(newdata, include_re=False)
    np.testing.assert_allclose(pred_with_re, pred_pop, rtol=1e-10, atol=1e-10)


def test_predict_mixed_known_and_unknown_levels(fit, sleepstudy):
    """A mix of known and unknown subjects: known rows match the fitted
    conditional mean, unknown rows match the population-level prediction."""
    n_train_rows = 10
    base = sleepstudy.iloc[:n_train_rows].copy()
    new = sleepstudy.iloc[10:15].copy()
    new["Subject"] = 99999
    combined = pd.concat([base, new], ignore_index=True)

    pred = fit.predict(combined, allow_new_levels=True)
    pop = fit.predict(combined, include_re=False)

    # First n_train_rows rows: conditional mean (matches training fitted).
    np.testing.assert_allclose(pred[:n_train_rows], fit.fitted()[:n_train_rows], rtol=1e-10)
    # Last 5 rows: population-level (because subject 99999 is new).
    np.testing.assert_allclose(pred[n_train_rows:], pop[n_train_rows:], rtol=1e-10)


def test_predict_missing_grouping_column_raises(fit, sleepstudy):
    newdata = sleepstudy.drop(columns=["Subject"])
    with pytest.raises(ValueError, match="missing from newdata"):
        fit.predict(newdata, allow_new_levels=True)
