"""Parametric bootstrap.

Simulate `n` new responses from the fitted model (drawing fresh b ~ N(0,Σ̂)
and y | b ~ family), refit each replicate, and collect the empirical
distribution of any quantity of interest. Slow but the reference for
inference; profile() is the default for variance components and Wald the
default for fixed effects.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
import pandas as pd
from jax import random

from ..covariance import vec_to_L
from ..nll import linear_predictor


@dataclass
class BootstrapResult:
    fixed_effects: pd.DataFrame  # n × p
    sigma: pd.Series | None
    re_sd: pd.DataFrame  # n × (sum_k q_k) SDs of variance components
    converged: pd.Series

    def quantiles(self, q=(0.025, 0.5, 0.975)) -> pd.DataFrame:
        out = self.fixed_effects.quantile(list(q)).T
        out.columns = [f"q{int(qi * 1000):04d}" for qi in q]
        return out


def _simulate_one(model, key) -> np.ndarray:
    """Simulate a response vector from the fitted model."""
    theta = jnp.asarray(model._theta)
    beta, log_phi, re_params = model.spec.split_theta(theta)
    # Sample b ~ N(0, Σ_k) per RE block
    keys = random.split(key, model.spec.n_re_blocks + 1)
    y_key = keys[0]
    b_blocks = []
    for k, (q, G, v) in enumerate(zip(model.spec.re_q, model.spec.re_G, re_params)):
        L = vec_to_L(v, q)
        z = random.normal(keys[k + 1], shape=(G, q))
        b_blocks.append(z @ L.T)
    re_jax = model._data_jax[3]
    X = model._data_jax[1]
    eta = linear_predictor(beta, b_blocks, X, re_jax)
    weights = model._data_jax[2]
    if model.family.has_dispersion:
        dispersion = jnp.exp(2.0 * log_phi)
    else:
        dispersion = jnp.array(1.0)
    y_new = model.family.simulate(y_key, eta, weights, dispersion)
    return np.asarray(y_new)


def bootstrap(model, *, n: int = 100, seed: int = 0) -> BootstrapResult:
    if model._theta is None:
        raise RuntimeError("Fit the model first.")

    # Lazy import to avoid circular
    from ..model import MixedModel

    key = random.PRNGKey(seed)
    fixef_rows = []
    sigma_vals = []
    re_sd_rows = []
    converged = []
    re_sd_cols: list[str] = []
    for b in model.matrices.re:
        for c in b.col_names:
            re_sd_cols.append(f"sd[{c}|{b.group_name}]")

    response = model.matrices.response_name
    base_df = model.data.copy()

    for i in range(n):
        key, sub = random.split(key)
        y_new = _simulate_one(model, sub)
        df_i = base_df.copy()
        df_i[response] = y_new
        try:
            m_i = MixedModel.from_formula(
                model.formula, df_i, family=model.family, weights=model.matrices.weights
            ).fit(theta0=model._theta)  # warm-start at the true theta
            fixef_rows.append(np.asarray(m_i.fixed_effects()))
            if model.family.has_dispersion:
                sigma_vals.append(m_i.sigma())
            sds_i = []
            for blk in m_i.variance_components():
                sds_i.extend(list(blk["sd"]))
            re_sd_rows.append(sds_i)
            converged.append(m_i.converged)
        except Exception:  # noqa: BLE001
            fixef_rows.append(np.full(model.spec.p, np.nan))
            if model.family.has_dispersion:
                sigma_vals.append(np.nan)
            re_sd_rows.append([np.nan] * len(re_sd_cols))
            converged.append(False)

    return BootstrapResult(
        fixed_effects=pd.DataFrame(fixef_rows, columns=model.matrices.fixed_names),
        sigma=pd.Series(sigma_vals, name="sigma") if model.family.has_dispersion else None,
        re_sd=pd.DataFrame(re_sd_rows, columns=re_sd_cols),
        converged=pd.Series(converged, name="converged"),
    )
