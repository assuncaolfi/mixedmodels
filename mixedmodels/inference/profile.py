"""Profile-likelihood confidence intervals.

For a parameter `ψ_j` (a single coordinate of the unconstrained θ-vector),
we fix `ψ_j = c` and refit the remaining parameters at a sequence of values
fanning out from the MLE. The signed square-root statistic

    ζ_j(c) = sign(c − ψ̂_j) · √( D(c) − D̂ )

is monotone in `c` under regularity and crosses the critical value
`z_{1−α/2}` at the CI endpoints. We interpolate `ζ` to find these crossings.

Because every parameter in our θ is unconstrained (log-Cholesky for
covariances, log σ for dispersion), profiling on the raw θ scale is already
the right thing to do for variance components.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import optimistix as optx
import pandas as pd
from scipy import stats
from scipy.interpolate import PchipInterpolator


@dataclass
class ProfileResult:
    """Result of `model.profile()`. Holds the ζ-traces per parameter."""

    traces: dict[str, pd.DataFrame]  # param_name -> DataFrame[value, zeta, deviance]
    intervals: pd.DataFrame  # one row per param, cols [estimate, lower, upper, level]

    def __repr__(self) -> str:
        return f"<ProfileResult\n{self.intervals!s}\n>"


def _theta_names(model) -> list[str]:
    names: list[str] = list(model.matrices.fixed_names)
    if model.family.has_dispersion:
        names.append("log_sigma")
    for b in model.matrices.re:
        q = b.q
        # log-Cholesky entries; name diagonals as log_sd_<col>|<group>,
        # off-diagonals as chol_<row>,<col>|<group>.
        rows, cols = np.tril_indices(q)
        for r, c in zip(rows, cols):
            if r == c:
                names.append(f"log_sd[{b.col_names[r]}|{b.group_name}]")
            else:
                names.append(f"chol[{b.col_names[r]},{b.col_names[c]}|{b.group_name}]")
    return names


# One jit'd profile-refit solver per (model, fixed-index j). Cached on the model.
_PROFILE_SOLVERS: dict[tuple[int, int], object] = {}


def _make_profile_solver(model, j: int):
    n = model.spec.n_theta
    free_idx = jnp.asarray([i for i in range(n) if i != j])
    marginal = model._marginal_nll
    bfgs = optx.BFGS(rtol=1e-7, atol=1e-7)

    def fn(theta_free, args):
        c, b_init, data = args
        theta = jnp.zeros(n, dtype=theta_free.dtype).at[free_idx].set(theta_free).at[j].set(c)
        val, b_hat = marginal(theta, b_init, *data)
        return val, b_hat

    @jax.jit
    def solve(theta0_free, c, b_init, data):
        sol = optx.minimise(
            fn,
            bfgs,
            y0=theta0_free,
            args=(c, b_init, data),
            has_aux=True,
            max_steps=300,
            throw=False,
        )
        # Recompute objective at the optimum for an exact deviance.
        theta_full = jnp.zeros(n, dtype=theta0_free.dtype).at[free_idx].set(sol.value).at[j].set(c)
        val, _ = marginal(theta_full, b_init, *data)
        return sol.value, val

    return solve, np.asarray(free_idx)


def _get_profile_solver(model, j: int):
    key = (id(model), j)
    if key not in _PROFILE_SOLVERS:
        _PROFILE_SOLVERS[key] = _make_profile_solver(model, j)
    return _PROFILE_SOLVERS[key]


def _refit_fixed(model, j: int, c: float, theta_warm: np.ndarray) -> tuple[float, np.ndarray]:
    """Refit with θ[j] fixed at c, warm-started from `theta_warm`."""
    solve, free_idx = _get_profile_solver(model, j)
    n = model.spec.n_theta
    theta_free_jax, val = solve(
        jnp.asarray(theta_warm[free_idx], dtype=jnp.float64),
        jnp.float64(c),
        jnp.asarray(model._b_hat),
        model._data_jax,
    )
    full = np.empty(n, dtype=np.float64)
    full[free_idx] = np.asarray(theta_free_jax)
    full[j] = c
    return 2.0 * float(val), full


def _profile_one(
    model,
    j: int,
    *,
    max_zeta: float,
    step_init: float,
    max_points: int,
) -> pd.DataFrame:
    """Sweep ζ outward in both directions from θ̂_j."""
    theta_hat = np.asarray(model._theta)
    D_hat = 2.0 * float(model._fun)
    psi_hat = float(theta_hat[j])

    rows = [{"value": psi_hat, "zeta": 0.0, "deviance": D_hat}]

    for direction in (+1, -1):
        warm = theta_hat.copy()
        step = step_init
        c = psi_hat
        for _ in range(max_points):
            c = c + direction * step
            D, warm = _refit_fixed(model, j, c, warm)
            dd = max(D - D_hat, 0.0)
            zeta = direction * np.sqrt(dd)
            rows.append({"value": c, "zeta": zeta, "deviance": D})
            if abs(zeta) >= max_zeta:
                break
            # Adapt step so successive ζ increments are about 0.5.
            if abs(zeta) > 0.05:
                target = 0.5
                step = step * (target / max(abs(zeta - rows[-2]["zeta"]), 0.05))
                step = float(np.clip(step, step_init / 8, step_init * 8))

    df = pd.DataFrame(rows).sort_values("value").reset_index(drop=True)
    return df


def _interpolate_ci(df: pd.DataFrame, crit: float) -> tuple[float, float]:
    """Find values where ζ = ±crit via monotone interpolation."""
    if df["zeta"].is_monotonic_increasing or df["zeta"].diff().dropna().ge(0).all():
        interp = PchipInterpolator(df["zeta"].values, df["value"].values, extrapolate=False)
    else:
        # Sort by zeta defensively (in case of non-monotone trace).
        df2 = df.sort_values("zeta")
        interp = PchipInterpolator(df2["zeta"].values, df2["value"].values, extrapolate=False)

    lo = float(interp(-crit)) if df["zeta"].min() <= -crit else float("nan")
    hi = float(interp(+crit)) if df["zeta"].max() >= +crit else float("nan")
    return lo, hi


def profile(
    model,
    params: list[str] | None = None,
    *,
    level: float = 0.95,
    max_points: int = 12,
    step_init: float = 0.5,
) -> ProfileResult:
    """Profile-likelihood CIs.

    `params` is a list of parameter names (see `_theta_names`). If None, all
    parameters are profiled. `step_init` is the initial step on the
    unconstrained θ scale; it adapts based on observed ζ increments.
    """
    if model._theta is None:
        raise RuntimeError("Fit the model first.")
    names = _theta_names(model)
    if params is None:
        params = names
    crit = float(stats.norm.ppf(0.5 + level / 2.0))
    max_zeta = crit * 1.1

    traces: dict[str, pd.DataFrame] = {}
    intervals_rows = []
    for nm in params:
        if nm not in names:
            raise KeyError(f"Unknown parameter {nm!r}; choices: {names}")
        j = names.index(nm)
        df = _profile_one(model, j, max_zeta=max_zeta, step_init=step_init, max_points=max_points)
        traces[nm] = df
        lo, hi = _interpolate_ci(df, crit)
        intervals_rows.append(
            {
                "parameter": nm,
                "estimate": float(model._theta[j]),
                "lower": lo,
                "upper": hi,
                "level": level,
            }
        )

    intervals = pd.DataFrame(intervals_rows).set_index("parameter")
    return ProfileResult(traces=traces, intervals=intervals)
