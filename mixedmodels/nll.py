"""Assemble the joint negative log-likelihood `g(θ, b)`.

The flat parameter vector `θ` (what the outer optimizer sees) is laid out as

    [ β  |  log_phi  |  vech-log-Cholesky for RE block 1  |  ...  ]

where `log_phi = log σ` is present iff the family has a dispersion parameter.
Latent `b` is a single flat vector concatenating the (G_k · q_k) entries of
each random-effects block in order.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp

from .covariance import n_chol_params, neg_log_prior_block
from .families import Family


@dataclass(frozen=True)
class ModelSpec:
    """Static (shape-level) information about a model. Hashable for jit."""

    p: int
    has_dispersion: bool
    re_q: tuple[int, ...]  # q_k for each RE block
    re_G: tuple[int, ...]  # G_k for each RE block
    family_name: str  # used to fetch the family; families are stateless

    @property
    def n_re_blocks(self) -> int:
        return len(self.re_q)

    @property
    def n_b(self) -> int:
        return sum(G * q for G, q in zip(self.re_G, self.re_q))

    @property
    def n_theta(self) -> int:
        m = self.p + (1 if self.has_dispersion else 0)
        for q in self.re_q:
            m += n_chol_params(q)
        return m

    def split_theta(self, theta):
        i = 0
        beta = theta[i : i + self.p]
        i += self.p
        if self.has_dispersion:
            log_phi = theta[i]
            i += 1
        else:
            log_phi = jnp.array(0.0)
        re_params = []
        for q in self.re_q:
            m = n_chol_params(q)
            re_params.append(theta[i : i + m])
            i += m
        return beta, log_phi, re_params

    def split_b(self, b):
        out = []
        i = 0
        for G, q in zip(self.re_G, self.re_q):
            m = G * q
            out.append(b[i : i + m].reshape(G, q))
            i += m
        return out


def linear_predictor(beta, b_blocks, X, re_jax):
    eta = X @ beta
    for (Ze, gidx), b in zip(re_jax, b_blocks):
        # Per-observation: dot product of expression row with the group's b.
        eta = eta + jnp.sum(Ze * b[gidx], axis=1)
    return eta


def make_joint_nll(spec: ModelSpec, family: Family):
    """Return a function `g(theta, b, data) -> scalar`."""

    def g(theta, b, y, X, weights, re_jax):
        beta, log_phi, re_params = spec.split_theta(theta)
        b_blocks = spec.split_b(b)
        eta = linear_predictor(beta, b_blocks, X, re_jax)
        # Gaussian: log_phi = log σ, dispersion = σ².
        if family.has_dispersion:
            dispersion = jnp.exp(2.0 * log_phi)
        else:
            dispersion = jnp.array(1.0)
        nll_data = jnp.sum(family.nll(y, eta, weights, dispersion))
        nll_re = jnp.array(0.0)
        for theta_k, q, b_k in zip(re_params, spec.re_q, b_blocks):
            nll_re = nll_re + neg_log_prior_block(b_k, theta_k, q)
        return nll_data + nll_re

    return g
