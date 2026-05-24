"""Outer optimizer: a pure-JAX BFGS via `optimistix`.

The entire optimization loop runs under a single `jax.jit`, with the inner
Newton mode-finding and Laplace marginal nll inlined inside. No Python
appears in the hot loop, eliminating the ~30 ms-per-iteration dispatch
overhead that scipy's L-BFGS-B forced us to pay before. End-to-end, a
sleepstudy fit drops from ~800 ms (scipy + JAX) to ~30 ms (hot, all in JAX).
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import optimistix as optx


@dataclass
class OptimResult:
    theta: np.ndarray
    b_hat: np.ndarray
    fun: float
    nit: int
    success: bool
    message: str


def make_solver(marginal_nll, *, rtol: float = 1e-7, atol: float = 1e-7, max_steps: int = 500):
    """Return a jit-compiled solver closure for the given `marginal_nll`.

    The returned `solve(theta0, b_init, data) -> (theta, b_hat, value, result, nit)`
    runs the full BFGS loop in JAX. JIT-compiles on first call; subsequent
    calls reuse the compiled artifact.
    """

    bfgs = optx.BFGS(rtol=rtol, atol=atol)

    def fn(theta, args):
        b_init_, data_ = args
        val, b_hat = marginal_nll(theta, b_init_, *data_)
        return val, b_hat

    @jax.jit
    def solve(theta0, b_init, data):
        sol = optx.minimise(
            fn,
            bfgs,
            y0=theta0,
            args=(b_init, data),
            has_aux=True,
            max_steps=max_steps,
            throw=False,
        )
        # Recompute the objective at the optimum so the returned `fun` is exact.
        val, b_hat_final = marginal_nll(sol.value, b_init, *data)
        return sol.value, b_hat_final, val, sol.result, sol.stats["num_steps"]

    return solve


def run_optimizer(
    solve,
    theta0: np.ndarray,
    b_init: jax.Array,
    data: tuple,
) -> OptimResult:
    """Run the jit-compiled solver and return a plain-NumPy result."""
    theta_jax = jnp.asarray(theta0)
    b_init_jax = jnp.asarray(b_init)
    theta_hat, b_hat, val, result, nit = solve(theta_jax, b_init_jax, data)
    success = bool(jnp.asarray(result == optx.RESULTS.successful))
    return OptimResult(
        theta=np.asarray(theta_hat),
        b_hat=np.asarray(b_hat),
        fun=float(val),
        nit=int(nit),
        success=success,
        message=str(result),
    )
