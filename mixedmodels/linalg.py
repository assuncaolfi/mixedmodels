"""Structured factorizations of the random-effects Hessian H_bb = ∂²g/∂b∂bᵀ.

The Hessian has the form

    H_bb = ZᵀW(b)Z + blockdiag(I_{G_k} ⊗ Σ_k⁻¹)_k,

where `W = diag(w_i · ∂²nll/∂η_i²)`. For models with a single random-effects
term `(expr | group)`, `H_bb` is **exactly block-diagonal** with `G` blocks
of size `q × q` (one per group). We exploit this to avoid:

  * forming the full `(Gq) × (Gq)` Hessian via `jax.hessian`,
  * doing a dense Cholesky / log-det / solve on that matrix.

Two backends share an interface:

  * `DenseHessian` — uses `jax.hessian`; works for any RE structure.
  * `BlockDiagHessian` — for one-RE-term models; batched `q × q` Choleskys.

The `MixedModel` constructor picks the cheapest applicable backend.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Per-observation 2nd derivative of family nll wrt η.
# ---------------------------------------------------------------------------


def per_obs_d2_eta(family, y, eta, weights, dispersion):
    """Vector `[∂²nll_i/∂η_i²]_i`, computed by AD over a scalarized family nll."""

    def per_i(eta_i, y_i, w_i):
        return family.nll(y_i.reshape(1), eta_i.reshape(1), w_i.reshape(1), dispersion).sum()

    return jax.vmap(jax.grad(jax.grad(per_i, argnums=0), argnums=0))(eta, y, weights)


# ---------------------------------------------------------------------------
# Dense backend (fallback)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DenseFactor:
    L: jax.Array  # Cholesky factor of H_bb, shape (q_tot, q_tot)


def dense_build(g_fn, theta, b, data) -> DenseFactor:
    H = jax.hessian(g_fn, argnums=1)(theta, b, *data)
    # Symmetrize defensively before factorization (Hessian round-off).
    H = 0.5 * (H + H.T)
    L = jnp.linalg.cholesky(H)
    return DenseFactor(L=L)


def dense_solve(factor: DenseFactor, rhs: jax.Array) -> jax.Array:
    # Solve L Lᵀ x = rhs.
    z = jax.scipy.linalg.solve_triangular(factor.L, rhs, lower=True)
    return jax.scipy.linalg.solve_triangular(factor.L.T, z, lower=False)


def dense_logdet(factor: DenseFactor) -> jax.Array:
    return 2.0 * jnp.sum(jnp.log(jnp.diag(factor.L)))


class DenseHessian:
    """Always-applicable backend using `jax.hessian`."""

    name = "dense"

    def __init__(self, spec, family):
        self.spec = spec
        self.family = family

    def build(self, g_fn, theta, b, data):
        return dense_build(g_fn, theta, b, data)

    def solve(self, factor, rhs):
        return dense_solve(factor, rhs)

    def logdet(self, factor):
        return dense_logdet(factor)


# ---------------------------------------------------------------------------
# Block-diagonal backend (single RE term)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BlockDiagFactor:
    """Cholesky factors of the per-group blocks.

    `L` has shape (G, q, q); each (g, :, :) is the lower-triangular Cholesky
    of the g-th `q × q` block of H_bb.
    """

    L: jax.Array
    q: int  # static


class BlockDiagHessian:
    """Backend for models with exactly one RE term.

    `H_bb` is block-diagonal in the natural ordering (each group's `q` random
    effects form one block). The block for group `g` is

        H_g = Σ_{i: gᵢ=g} w_i · Ze[i] Ze[i]ᵀ  +  Σ⁻¹,

    with `Σ⁻¹ = (L_θ L_θᵀ)⁻¹` from the log-Cholesky parameterization.
    """

    name = "block_diag"

    def __init__(self, spec, family):
        if spec.n_re_blocks != 1:
            raise ValueError("BlockDiagHessian requires exactly one RE term")
        self.spec = spec
        self.family = family
        self.G = spec.re_G[0]
        self.q = spec.re_q[0]

    @staticmethod
    def applies(spec) -> bool:
        return spec.n_re_blocks == 1

    def _compute_W(self, theta, b, data):
        """Per-observation weights wᵢ = ∂²nll_i/∂η_i²."""
        from .nll import linear_predictor

        y, X, weights, re_jax = data
        beta, log_phi, _ = self.spec.split_theta(theta)
        b_blocks = self.spec.split_b(b)
        eta = linear_predictor(beta, b_blocks, X, re_jax)
        if self.family.has_dispersion:
            dispersion = jnp.exp(2.0 * log_phi)
        else:
            dispersion = jnp.array(1.0)
        return per_obs_d2_eta(self.family, y, eta, weights, dispersion)

    def build(self, g_fn, theta, b, data):
        """Build H block-by-block and Cholesky-factor each."""
        from .covariance import vec_to_L

        del g_fn  # not needed; structure is exploited analytically
        _, _, re_params = self.spec.split_theta(theta)
        w = self._compute_W(theta, b, data)
        # Single RE term:
        y, X, weights, re_jax = data
        Ze, group_idx = re_jax[0]
        G, q = self.G, self.q  # static
        # Z'WZ per group via scatter-add of outer products.
        outer = (Ze[:, :, None] * Ze[:, None, :]) * w[:, None, None]  # (n, q, q)
        H_data = jnp.zeros((G, q, q), dtype=Ze.dtype).at[group_idx].add(outer)
        # Add Σ⁻¹ to each block: Σ = L Lᵀ ⇒ Σ⁻¹ = L⁻ᵀ L⁻¹.
        Lk = vec_to_L(re_params[0], q)
        Sigma = Lk @ Lk.T
        Sigma_inv = jnp.linalg.inv(Sigma)
        H = H_data + Sigma_inv[None, :, :]
        # Symmetrize defensively.
        H = 0.5 * (H + jnp.swapaxes(H, -1, -2))
        L = jnp.linalg.cholesky(H)  # (G, q, q)
        return BlockDiagFactor(L=L, q=q)

    def solve(self, factor: BlockDiagFactor, rhs: jax.Array) -> jax.Array:
        """Solve H x = rhs where rhs is flat of length G·q."""
        q = factor.q
        G = factor.L.shape[0]
        rhs_g = rhs.reshape(G, q)
        # Solve L Lᵀ x = b per group via two triangular solves.
        z = jax.vmap(lambda L, r: jax.scipy.linalg.solve_triangular(L, r, lower=True))(
            factor.L, rhs_g
        )
        x = jax.vmap(lambda L, z_: jax.scipy.linalg.solve_triangular(L.T, z_, lower=False))(
            factor.L, z
        )
        return x.reshape(-1)

    def logdet(self, factor: BlockDiagFactor) -> jax.Array:
        diag = jnp.diagonal(factor.L, axis1=1, axis2=2)  # (G, q)
        return 2.0 * jnp.sum(jnp.log(diag))


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def make_structure(spec, family):
    """Pick the cheapest applicable Hessian structure for `spec`."""
    if BlockDiagHessian.applies(spec):
        return BlockDiagHessian(spec, family)
    return DenseHessian(spec, family)
