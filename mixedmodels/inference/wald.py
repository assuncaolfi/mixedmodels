"""Wald confidence intervals for fixed effects.

We compute `cov(β̂)` from the **closed-form β-block of the observed
information** at the MLE, using the Schur complement of the joint Hessian
of `g(β, b)`:

    I(β) = XᵀWX − (XᵀWZ) H_bb⁻¹ (ZᵀWX),

where `W = diag(w_i · ∂²nll/∂η_i²)` at `b̂` and `H_bb = ZᵀWZ + Σ⁻¹`. This
is what `lme4` and `glmmTMB` use; it drops the `½ log det H_bb` correction
that depends on β through `b̂`, which is `O(1/n)` and standard practice to
ignore for finite-sample SEs.

This avoids taking `jax.hessian` of the full Laplace marginal (which would
require second-order autodiff through the inner Newton loop and is ~10⁴×
slower).
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from scipy import stats

from ..linalg import BlockDiagHessian, per_obs_d2_eta
from ..nll import linear_predictor


@dataclass
class WaldResult:
    estimate: pd.Series
    std_error: pd.Series
    z_value: pd.Series
    p_value: pd.Series
    ci_lower: pd.Series
    ci_upper: pd.Series

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "estimate": self.estimate,
                "std_error": self.std_error,
                "z": self.z_value,
                "p": self.p_value,
                "ci_lower": self.ci_lower,
                "ci_upper": self.ci_upper,
            }
        )

    def __repr__(self) -> str:
        return f"<WaldResult\n{self.to_frame()!s}\n>"


def _vcov_beta_blockdiag_jax(model):
    """Closed-form vcov(β̂) for single-RE-term models. Pure JAX; returns a JAX array."""
    spec = model.spec
    family = model.family
    theta = jnp.asarray(model._theta)
    b_hat = jnp.asarray(model._b_hat)
    y, X, weights, re_jax = model._data_jax
    Ze, group_idx = re_jax[0]
    G, q = spec.re_G[0], spec.re_q[0]
    p = spec.p

    beta, log_phi, re_params = spec.split_theta(theta)
    dispersion = jnp.exp(2.0 * log_phi) if family.has_dispersion else jnp.array(1.0)
    b_blocks = spec.split_b(b_hat)
    eta = linear_predictor(beta, b_blocks, X, re_jax)
    w = per_obs_d2_eta(family, y, eta, weights, dispersion)  # (n,)

    # XᵀWX, (p, p)
    XWX = X.T @ (w[:, None] * X)

    # ZᵀWX per group: shape (G, q, p), scatter-add of w_i · Ze_i · X_iᵀ
    contribs = (w[:, None, None] * Ze[:, :, None]) * X[:, None, :]  # (n, q, p)
    ZWX = jnp.zeros((G, q, p), dtype=X.dtype).at[group_idx].add(contribs)

    # H_bb per group, (G, q, q)
    outer = (Ze[:, :, None] * Ze[:, None, :]) * w[:, None, None]
    H_data = jnp.zeros((G, q, q), dtype=X.dtype).at[group_idx].add(outer)
    from ..covariance import vec_to_L

    Lk = vec_to_L(re_params[0], q)
    Sigma_inv = jnp.linalg.inv(Lk @ Lk.T)
    H = H_data + Sigma_inv[None]
    H = 0.5 * (H + jnp.swapaxes(H, -1, -2))
    L = jnp.linalg.cholesky(H)  # (G, q, q)

    # Schur correction: ZWX' @ H⁻¹ @ ZWX, summed over groups.
    sol = jax.vmap(lambda Lg, rhs: jax.scipy.linalg.cho_solve((Lg, True), rhs))(L, ZWX)
    correction = jnp.einsum("gqp,gqr->pr", ZWX, sol)

    I_beta = XWX - correction
    I_beta = 0.5 * (I_beta + I_beta.T)
    return jnp.linalg.inv(I_beta)


def _vcov_beta(model) -> np.ndarray:
    """vcov(β̂). Dispatches to the closed-form path for single-RE-term models."""
    if model._theta is None:
        raise RuntimeError("Fit the model first.")
    if isinstance(model._structure, BlockDiagHessian):
        return np.asarray(_vcov_beta_blockdiag_jax(model))
    # Fallback for crossed / multi-RE models: jax.hessian of the marginal.
    return _vcov_beta_dense(model)


def _vcov_beta_dense(model) -> np.ndarray:
    p = model.spec.p
    theta_hat = jnp.asarray(model._theta)
    b_hat = jnp.asarray(model._b_hat)
    data = model._data_jax
    marginal = model._marginal_nll

    def m_beta(beta):
        theta_full = theta_hat.at[:p].set(beta)
        val, _ = marginal(theta_full, b_hat, *data)
        return val

    H_beta = jax.hessian(m_beta)(theta_hat[:p])
    H_beta = 0.5 * (H_beta + H_beta.T)
    return np.asarray(jnp.linalg.inv(H_beta))


def wald(model, *, level: float = 0.95) -> WaldResult:
    """Wald CIs for the fixed-effects block β."""
    cov_beta = _vcov_beta(model)
    se = np.sqrt(np.clip(np.diag(cov_beta), 0.0, None))
    beta = np.asarray(model.fixed_effects())
    z = beta / se
    pval = 2.0 * (1.0 - stats.norm.cdf(np.abs(z)))
    crit = stats.norm.ppf(0.5 + level / 2.0)
    lo = beta - crit * se
    hi = beta + crit * se
    idx = pd.Index(model.matrices.fixed_names, name="term")
    return WaldResult(
        estimate=pd.Series(beta, idx),
        std_error=pd.Series(se, idx),
        z_value=pd.Series(z, idx),
        p_value=pd.Series(pval, idx),
        ci_lower=pd.Series(lo, idx),
        ci_upper=pd.Series(hi, idx),
    )
