"""The `MixedModel` class and the `lmer` / `glmer` entry points."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from . import inference
from .covariance import cov_from_vec, n_chol_params, vec_to_L
from .data import ModelMatrices, build_matrices, to_jax
from .families import Family, Gaussian
from .formula import parse_formula
from .laplace import make_marginal_nll
from .linalg import make_structure
from .nll import ModelSpec, linear_predictor, make_joint_nll
from .optimize import make_solver, run_optimizer

if TYPE_CHECKING:
    from .summary import Summary

# Ensure 64-bit by default; variance estimation in float32 is unreliable.
jax.config.update("jax_enable_x64", True)


@dataclass
class MixedModel:
    formula: str
    family: Family
    data: pd.DataFrame
    matrices: ModelMatrices
    spec: ModelSpec
    _theta: np.ndarray | None = None
    _b_hat: np.ndarray | None = None
    _fun: float | None = None
    _opt_message: str = ""
    _opt_success: bool = False
    _opt_nit: int = 0
    _joint_nll: Any = None
    _marginal_nll: Any = None
    _data_jax: tuple = field(default_factory=tuple)
    _structure: Any = None
    _solver: Any = None  # cached jit'd optimizer

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_formula(
        cls,
        formula: str,
        data: pd.DataFrame,
        *,
        family: Family | None = None,
        weights: np.ndarray | None = None,
    ) -> "MixedModel":
        """Build a :class:`MixedModel` from an lme4-style formula and a DataFrame.

        ``family=`` is a :class:`Family` instance; the link is set on the
        family itself (e.g. ``Bernoulli(ProbitLink())``). Defaults to
        ``Gaussian()`` (identity link).
        """
        fam = Gaussian() if family is None else family
        parsed = parse_formula(formula)
        mm = build_matrices(parsed, data, weights=weights)
        spec = ModelSpec(
            p=mm.p,
            has_dispersion=fam.has_dispersion,
            re_q=tuple(b.q for b in mm.re),
            re_G=tuple(b.G for b in mm.re),
            family_name=fam.name,
        )
        g = make_joint_nll(spec, fam)
        structure = make_structure(spec, fam)
        m_fn = make_marginal_nll(g, structure)
        solver = make_solver(m_fn)
        return cls(
            formula=formula,
            family=fam,
            data=data,
            matrices=mm,
            spec=spec,
            _joint_nll=g,
            _marginal_nll=m_fn,
            _data_jax=to_jax(mm),
            _structure=structure,
            _solver=solver,
        )

    # ------------------------------------------------------------------
    # Initial values
    # ------------------------------------------------------------------

    def initial_theta(self) -> np.ndarray:
        """Sensible starting values: pseudo-OLS for β, dispersion=1, Σ_k = I.

        We transform `y` through the family's link function before the
        least-squares step, so β₀ is on the right scale regardless of which
        link is in use.
        """
        mm = self.matrices
        fam = self.family
        # Pseudo-OLS for β on the link scale. The Link's `to_eta` does the
        # transform; we pre-clip count/positive-real data so that y = 0 maps
        # to a finite (not -∞) starting value.
        if fam.name == "binomial":
            y_for_link = mm.y / np.maximum(mm.weights, 1.0)
        elif fam.link.name in ("log", "inverse", "sqrt"):
            y_for_link = np.maximum(mm.y, 0.5)
        else:
            y_for_link = mm.y
        y_link = np.asarray(fam.link.to_eta(jnp.asarray(y_for_link)))
        beta0, *_ = np.linalg.lstsq(mm.X, y_link, rcond=None)
        parts: list[np.ndarray] = [beta0]

        # Dispersion start.
        if fam.has_dispersion:
            if fam.name == "gaussian" and fam.link.name == "identity":
                resid = mm.y - mm.X @ beta0
                sigma0 = max(np.std(resid, ddof=mm.p), 1e-3)
                log_phi0 = float(np.log(sigma0))  # dispersion = exp(2 log_phi) = σ²
            else:
                log_phi0 = 0.0  # dispersion = 1 by default
            parts.append(np.array([log_phi0]))

        # RE covariance starts: Σ_k = (s·I)² with `s` a small fraction of σ (or 0.5).
        if fam.name == "gaussian" and fam.link.name == "identity":
            resid = mm.y - mm.X @ beta0
            s_default = max(np.std(resid, ddof=mm.p) / 2.0, 1e-2)
        else:
            s_default = 0.5
        for b in mm.re:
            q = b.q
            v = np.zeros(n_chol_params(q))
            rows, cols = np.tril_indices(q)
            for i, (r, c) in enumerate(zip(rows, cols)):
                if r == c:
                    v[i] = np.log(s_default)
            parts.append(v)
        return np.concatenate(parts)

    def initial_b(self) -> jax.Array:
        return jnp.zeros(self.spec.n_b)

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, *, theta0: np.ndarray | None = None) -> "MixedModel":
        """Fit by Laplace maximum likelihood. Returns self for chaining.

        On first call, the entire JAX optimization pipeline is compiled
        (typically <2 s wall-clock). Subsequent fits reuse the compiled
        artifact; warm fits on small models take tens of milliseconds.
        """
        theta0 = theta0 if theta0 is not None else self.initial_theta()
        res = run_optimizer(
            self._solver,
            theta0,
            self.initial_b(),
            self._data_jax,
        )
        self._theta = res.theta
        self._b_hat = res.b_hat
        self._fun = res.fun
        self._opt_message = res.message
        self._opt_success = res.success
        self._opt_nit = res.nit
        return self

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def _check_fit(self):
        if self._theta is None:
            raise RuntimeError("Model has not been fit yet; call .fit().")

    @property
    def theta(self) -> np.ndarray:
        self._check_fit()
        return self._theta  # type: ignore[return-value]

    @property
    def b_hat(self) -> np.ndarray:
        self._check_fit()
        return self._b_hat  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Parameter accessors
    # ------------------------------------------------------------------

    def fixed_effects(self) -> pd.Series:
        """Fixed-effects estimates β̂ as a named Series."""
        self._check_fit()
        beta, _, _ = self.spec.split_theta(jnp.asarray(self._theta))
        return pd.Series(np.asarray(beta), index=self.matrices.fixed_names, name="estimate")

    # `coefficients` is provided as a synonym for `fixed_effects`, for users
    # coming from sklearn/statsmodels. In a mixed-effects model there is no
    # other "coefficient" vector at the model level (random effects are
    # per-group, returned by `random_effects()`).
    coefficients = fixed_effects

    def covariance_matrix(self) -> pd.DataFrame:
        """Variance-covariance matrix of the fixed-effects estimator β̂."""
        from .inference.wald import _vcov_beta

        cov = _vcov_beta(self)
        names = self.matrices.fixed_names
        return pd.DataFrame(np.asarray(cov), index=names, columns=names)

    def standard_errors(self) -> pd.Series:
        """Standard errors of the fixed-effects estimator."""
        cov = self.covariance_matrix().to_numpy()
        se = np.sqrt(np.clip(np.diag(cov), 0.0, None))
        return pd.Series(se, index=self.matrices.fixed_names, name="std_error")

    def coefficients_table(self) -> pd.DataFrame:
        """Fixed-effects table: estimate, SE, z, p, Wald CI."""
        return self.wald().to_frame()

    def sigma(self) -> float | None:
        """Residual standard deviation (Gaussian); None for non-dispersion families."""
        if not self.family.has_dispersion:
            return None
        _, log_phi, _ = self.spec.split_theta(jnp.asarray(self.theta))
        return float(jnp.exp(log_phi))

    def variance_components(self) -> list[dict]:
        """Variance components per RE block.

        Returns one dict per block with keys: name, cov (q×q), sd (q,), corr (q×q).
        """
        self._check_fit()
        _, _, re_params = self.spec.split_theta(jnp.asarray(self.theta))
        out = []
        for b, vec, q in zip(self.matrices.re, re_params, self.spec.re_q):
            cov = np.asarray(cov_from_vec(vec, q))
            sd = np.sqrt(np.diag(cov))
            with np.errstate(invalid="ignore", divide="ignore"):
                corr = cov / np.outer(sd, sd)
            out.append(
                {
                    "name": b.name,
                    "group": b.group_name,
                    "columns": b.col_names,
                    "cov": cov,
                    "sd": sd,
                    "corr": corr,
                }
            )
        return out

    def random_effects(self) -> list[pd.DataFrame]:
        """Conditional modes (BLUPs) per random-effects block."""
        self._check_fit()
        b_blocks = self.spec.split_b(jnp.asarray(self._b_hat))
        out = []
        for b, mat in zip(self.matrices.re, b_blocks):
            df = pd.DataFrame(np.asarray(mat), index=b.group_levels, columns=b.col_names)
            df.index.name = b.group_name
            out.append(df)
        return out

    def fitted(self, *, include_re: bool = True) -> np.ndarray:
        """Fitted values μ̂ on the response scale (i.e. `inv_link(η̂)`)."""
        return self.predict(None, include_re=include_re)

    def residuals(self, type: str = "response") -> np.ndarray:
        """Residuals.

        Parameters
        ----------
        type : {"response", "pearson", "working"}
            * `"response"`: y − μ̂.
            * `"pearson"`: (y − μ̂) / √(Var(μ̂) · ϕ).
            * `"working"`: (y − μ̂) / dμ/dη — the IRLS “working” residual.
        """
        self._check_fit()
        y = self.matrices.y
        mu = self.fitted()
        if type == "response":
            return y - mu
        if type == "pearson":
            var = np.asarray(
                self.family.variance(jnp.asarray(mu), jnp.asarray(self.matrices.weights))
            )
            phi = (self.sigma() ** 2) if self.family.has_dispersion else 1.0
            return (y - mu) / np.sqrt(var * phi)
        if type == "working":
            # dμ/dη = variance(μ) / weights for canonical-link GLMs; for Gaussian it's 1.
            if self.family.name == "gaussian":
                dmu = np.ones_like(mu)
            else:
                var = np.asarray(
                    self.family.variance(jnp.asarray(mu), jnp.asarray(self.matrices.weights))
                )
                dmu = var / self.matrices.weights
            return (y - mu) / dmu
        raise ValueError(f"Unknown residual type {type!r}")

    def predict(
        self, newdata: pd.DataFrame | None = None, *, include_re: bool = True
    ) -> np.ndarray:
        """Linear-predictor (η) predictions on the response scale."""
        self._check_fit()
        if newdata is None:
            X = self.matrices.X
            re_jax = self._data_jax[3]
            b_blocks = self.spec.split_b(jnp.asarray(self._b_hat))
            beta, _, _ = self.spec.split_theta(jnp.asarray(self._theta))
            if include_re:
                eta = linear_predictor(beta, b_blocks, jnp.asarray(X), re_jax)
            else:
                eta = jnp.asarray(X) @ beta
            return np.asarray(self.family.inv_link(eta))
        else:
            # Rebuild matrices on new data using the same formula
            parsed = parse_formula(self.formula)
            mm_new = build_matrices(parsed, newdata)
            beta, _, _ = self.spec.split_theta(jnp.asarray(self._theta))
            re_jax_new = to_jax(mm_new)[3]
            if include_re:
                b_blocks = self.spec.split_b(jnp.asarray(self._b_hat))
                # NB: requires the grouping factor levels in newdata to be a subset
                # of the training levels; otherwise group_idx is invalid. A more
                # robust implementation would zero out unknown groups; left as TODO.
                eta = linear_predictor(beta, b_blocks, jnp.asarray(mm_new.X), re_jax_new)
            else:
                eta = jnp.asarray(mm_new.X) @ beta
            return np.asarray(self.family.inv_link(eta))

    # ------------------------------------------------------------------
    # Likelihood / information criteria
    # ------------------------------------------------------------------

    def log_likelihood(self) -> float:
        """log L̂. Note: for Binomial the binomial coefficient is dropped."""
        self._check_fit()
        return -float(self._fun)

    def deviance(self) -> float:
        """−2 log L̂."""
        return 2.0 * float(self._fun)

    def n_observations(self) -> int:
        """Number of observations used to fit the model."""
        return self.matrices.n

    def degrees_of_freedom(self) -> int:
        """Number of estimated parameters (fixed + variance components + dispersion)."""
        return self.spec.n_theta

    def aic(self) -> float:
        """Akaike Information Criterion: 2 · df − 2 · logL̂."""
        return 2.0 * self.degrees_of_freedom() - 2.0 * self.log_likelihood()

    def bic(self) -> float:
        """Bayesian Information Criterion: log(n) · df − 2 · logL̂."""
        return (
            np.log(self.n_observations()) * self.degrees_of_freedom() - 2.0 * self.log_likelihood()
        )

    @property
    def n(self) -> int:
        return self.matrices.n

    @property
    def converged(self) -> bool:
        return self._opt_success

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def simulate(self, n: int = 1, *, seed: int = 0) -> np.ndarray:
        """Simulate `n` new response vectors from the fitted model.

        Draws fresh `b ~ N(0, Σ̂)` and `y | b ~ family(η̂ + Zb)`. Returns shape
        `(n_obs,)` if `n == 1`, else `(n, n_obs)`.
        """
        from jax import random

        from .covariance import vec_to_L
        from .nll import linear_predictor

        self._check_fit()
        theta = jnp.asarray(self._theta)
        beta, log_phi, re_params = self.spec.split_theta(theta)
        dispersion = jnp.exp(2.0 * log_phi) if self.family.has_dispersion else jnp.array(1.0)
        _, X, weights, re_jax = self._data_jax

        key = random.PRNGKey(seed)
        out = []
        for _ in range(n):
            key, sub = random.split(key)
            keys = random.split(sub, self.spec.n_re_blocks + 1)
            b_blocks = []
            for k, (q, G, v) in enumerate(zip(self.spec.re_q, self.spec.re_G, re_params)):
                L = vec_to_L(v, q)
                z = random.normal(keys[k + 1], shape=(G, q))
                b_blocks.append(z @ L.T)
            eta = linear_predictor(beta, b_blocks, X, re_jax)
            y_new = self.family.simulate(keys[0], eta, weights, dispersion)
            out.append(np.asarray(y_new))
        return out[0] if n == 1 else np.stack(out)

    # ------------------------------------------------------------------
    # Inference shortcuts
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def wald(self, *, level: float = 0.95):
        return inference.wald(self, level=level)

    def profile(self, params: list[str] | None = None, **kw):
        return inference.profile(self, params=params, **kw)

    def bootstrap(self, n: int = 100, **kw):
        return inference.bootstrap(self, n=n, **kw)

    def confidence_intervals(
        self,
        params: list[str] | None = None,
        *,
        level: float = 0.95,
        method: str = "wald",
    ) -> pd.DataFrame:
        """Confidence intervals for model parameters.

        Parameters
        ----------
        params : list of str, optional
            Parameter names to report. For `method="wald"` the names are the
            fixed-effects coefficients; for `"profile"` they can be any of
            the unconstrained θ-vector entries (see :meth:`profile`). If
            None, defaults to all fixed effects (Wald) or all parameters
            (profile).
        method : {"wald", "profile"}
            `"wald"` is the default and matches `lme4`'s default for fixed
            effects. Use `"profile"` for variance components.
        """
        if method == "wald":
            w = self.wald(level=level)
            df = pd.DataFrame(
                {
                    "estimate": w.estimate,
                    "lower": w.ci_lower,
                    "upper": w.ci_upper,
                }
            )
            df.attrs["method"] = "Wald"
            df.attrs["level"] = level
            if params is not None:
                df = df.loc[params]
            return df
        if method == "profile":
            res = self.profile(params=params, level=level)
            return res.intervals[["estimate", "lower", "upper"]].copy()
        raise ValueError(f"Unknown method {method!r}; use 'wald' or 'profile'.")

    def summary(self) -> "Summary":
        from .summary import Summary

        return Summary(self)

    def __repr__(self) -> str:
        if self._theta is None:
            return f"<MixedModel {self.formula!r} (unfit)>"
        return f"<MixedModel {self.formula!r} family={self.family.name} logLik={self.log_likelihood():.3f}>"


# ----------------------------------------------------------------------
# Entry points
# ----------------------------------------------------------------------


# Re-exports
_ = vec_to_L  # keep importable
