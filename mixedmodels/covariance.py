"""Covariance parameterizations for random-effects blocks.

A grouping factor with `q` random effects per group has a `q × q` covariance
`Σ`. We parameterize it in unconstrained ℝ-space via **log-Cholesky**:

  - For q = 1: one parameter, `log σ`.
  - For q > 1: q(q+1)/2 parameters laying out a lower-triangular L
    column-by-column (matching `jnp.tril_indices`), with `log` applied to the
    diagonal entries. Σ = L Lᵀ.

This is the same parameterization used by Stan and by TMB's `UNSTRUCTURED_CORR`.
It is bijective with the cone of positive-definite matrices and gives the outer
optimizer an unconstrained domain.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def n_chol_params(q: int) -> int:
    return q * (q + 1) // 2


def vec_to_L(v: jax.Array, q: int) -> jax.Array:
    """Lower-triangular Cholesky factor from `q(q+1)/2` unconstrained reals.

    Diagonal entries are exponentiated; off-diagonals are taken as-is.
    Ordering matches `jnp.tril_indices(q)` (row-major over the lower triangle).
    """
    rows, cols = jnp.tril_indices(q)
    is_diag = rows == cols
    vals = jnp.where(is_diag, jnp.exp(v), v)
    L = jnp.zeros((q, q), dtype=v.dtype).at[rows, cols].set(vals)
    return L


def L_to_vec(L: jax.Array) -> jax.Array:
    """Inverse of vec_to_L."""
    q = L.shape[0]
    rows, cols = jnp.tril_indices(q)
    vals = L[rows, cols]
    is_diag = rows == cols
    return jnp.where(is_diag, jnp.log(jnp.clip(vals, 1e-300, None)), vals)


def cov_from_vec(v: jax.Array, q: int) -> jax.Array:
    L = vec_to_L(v, q)
    return L @ L.T


def logdet_L(v: jax.Array, q: int) -> jax.Array:
    """log |L| where L = vec_to_L(v, q). Sum of the diagonal log-cholesky params."""
    rows, cols = jnp.tril_indices(q)
    is_diag = rows == cols
    return jnp.sum(jnp.where(is_diag, v, 0.0))


def neg_log_prior_block(b_block: jax.Array, v: jax.Array, q: int) -> jax.Array:
    """−log p(b) for b of shape (G, q) iid N(0, Σ), Σ = L Lᵀ, L = vec_to_L(v).

    Computed by solving L Y = bᵀ once and summing Yᵀ Y, avoiding any explicit
    Σ⁻¹.
    """
    G = b_block.shape[0]
    L = vec_to_L(v, q)
    # Solve L @ Y = b.T  =>  Y of shape (q, G), then sum Y^2 across all entries.
    Y = jax.scipy.linalg.solve_triangular(L, b_block.T, lower=True)
    quad = jnp.sum(Y * Y)
    logdet_Sigma = 2.0 * logdet_L(v, q)
    return 0.5 * (G * (q * jnp.log(2.0 * jnp.pi) + logdet_Sigma) + quad)
