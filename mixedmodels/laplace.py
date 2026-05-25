"""Inner Newton + Laplace approximation of the marginal nll.

The marginal negative log-likelihood is

    m(θ) = g(θ, b̂(θ)) + ½ log det H(θ, b̂(θ)) − (q/2) log(2π),

with `b̂(θ) = argmin_b g(θ, b)` and `H = ∂²g/∂b∂bᵀ |_{b̂}`.

We need the gradient `dm/dθ` to be exact, because L-BFGS-B's line search
is unforgiving of inconsistent value/gradient. The envelope theorem
(`∂g/∂b = 0` at b̂) handles the `g(θ, b̂)` term automatically. For the
`log det H` term we need the implicit derivative

    db̂/dθ = −H⁻¹ ∂²g/∂θ∂b.

We obtain it with the standard one-step trick: find b̂ numerically under
`stop_gradient`, then do **one extra Newton step starting from b̂ with θ
left differentiable**. Numerically this is a no-op (∂g/∂b ≈ 0 already at
b̂), but it makes JAX see b̂ as `b̂_sg − H⁻¹ ∇g`, whose derivative wrt θ
equals exactly the IFT formula. So `jax.grad(m)` is then exact, and
L-BFGS-B converges cleanly even for GLMMs.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def _newton_loop(g_fn, structure, theta_sg, b_init, data_sg, max_iter, tol):
    """Numerical Newton iteration on `b` (under stop_gradient). Returns b_sg, residual."""

    def cond(state):
        _, gnorm, it = state
        return jnp.logical_and(gnorm > tol, it < max_iter)

    def body(state):
        b, _, it = state
        grad_b = jax.grad(g_fn, argnums=1)(theta_sg, b, *data_sg)
        factor = structure.build(g_fn, theta_sg, b, data_sg)
        step = structure.solve(factor, grad_b)
        b_new = b - step
        gnorm_new = jnp.linalg.norm(jax.grad(g_fn, argnums=1)(theta_sg, b_new, *data_sg))
        return b_new, gnorm_new, it + 1

    # Prime the loop so the cond's gnorm > tol triggers at least once for safety.
    init_gnorm = jnp.linalg.norm(jax.grad(g_fn, argnums=1)(theta_sg, b_init, *data_sg))
    state0 = (b_init, jnp.maximum(init_gnorm, jnp.asarray(tol + 1.0)), jnp.array(0))
    b_sg, gnorm_final, _ = jax.lax.while_loop(cond, body, state0)
    return jax.lax.stop_gradient(b_sg)


def find_mode_implicit(g_fn, structure, theta, b_init, data, max_iter=50, tol=1e-8):
    """Find b̂(θ) with the correct implicit-function gradient.

    Phase 1: Newton under `stop_gradient` to get a numerical b̂.
    Phase 2: one final Newton step from b̂, with θ left differentiable. The
    result is numerically the same b̂, but JAX now sees
    `b̂ = b_sg − H⁻¹ ∇g(θ, b_sg)`, whose derivative wrt θ equals exactly the
    IFT formula `db̂/dθ = −H⁻¹ ∂²g/∂θ∂b`.
    """
    theta_sg = jax.lax.stop_gradient(theta)
    data_sg = jax.tree_util.tree_map(jax.lax.stop_gradient, data)
    b_sg = _newton_loop(g_fn, structure, theta_sg, b_init, data_sg, max_iter, tol)

    grad_b = jax.grad(g_fn, argnums=1)(theta, b_sg, *data)
    factor_sg = structure.build(g_fn, theta, b_sg, data)
    step = structure.solve(factor_sg, grad_b)
    return b_sg - step


def make_marginal_nll(g_fn, structure, max_iter: int = 50, tol: float = 1e-8):
    """Return `m(θ, b_init, *data) -> (scalar, b_hat)`.

    The Laplace-approximated marginal nll. We build the H_bb factor twice:
    once inside `find_mode_implicit` for the IFT step on b̂, and once here
    at the IFT-corrected b̂ so the `log det H` term carries the correct
    gradient — including the contribution through `b̂(θ)`, which matters
    for non-Gaussian families where H_bb depends on `b` via the weight
    matrix `W = diag(∂²nll/∂η²)`. For LMMs the second build is redundant
    but cheap.
    """

    def m(theta, b_init, *data):
        b_hat = find_mode_implicit(
            g_fn, structure, theta, b_init, data, max_iter=max_iter, tol=tol
        )
        factor = structure.build(g_fn, theta, b_hat, data)
        logdet = structure.logdet(factor)
        q = b_hat.shape[0]
        val = g_fn(theta, b_hat, *data) + 0.5 * logdet - 0.5 * q * jnp.log(2.0 * jnp.pi)
        return val, b_hat

    return m
