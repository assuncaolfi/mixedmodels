# Design notes

This document records the design of `mixedmodels` as it currently stands.
The goal is a Python package that fits linear and generalized linear
mixed-effects models with an API in the spirit of `lme4`, `glmmTMB`, and
`MixedModels.jl`, while staying as a thin layer over mature dependencies
(JAX, formulaic, optimistix, scipy).

## 1. Problem

A mixed-effects model has parameters `θ` (fixed effects, variance
components, dispersion, …) and latent random effects `b`. The joint density
factors as

    p(y, b | θ) = p(y | b, θ) · p(b | θ),     b ~ N(0, Σ_θ).

The marginal log-likelihood requires integrating out `b`. This is
intractable in general; the **Laplace approximation** uses a second-order
expansion of the joint negative log-likelihood `g(b; θ) = −log p(y, b | θ)`
around its mode `b̂(θ)`:

    −log L(θ) ≈ g(b̂; θ) + ½ log det H(θ) − (q/2) log(2π),

where `H = ∂²g/∂b∂bᵀ |_{b=b̂}` and `q = dim(b)`. This reduces fitting to a
nested optimization:

    minimize_θ    [ min_b g(b; θ) ] + ½ log det H(θ).

That is the TMB algorithm (Kristensen et al. 2016, JSS). For LMMs with
Gaussian conjugacy the Laplace approximation is exact; for GLMMs it is the
same Laplace used by `lme4::glmer(nAGQ=1)` and `glmmTMB`.

## 2. Architecture

Everything in `mixedmodels/` is JAX-native; NumPy appears only at I/O
boundaries (formulaic, pandas, scipy interpolation in profile CIs).

```
mixedmodels/
├── __init__.py        public API: MixedModel and the family classes
├── formula.py         parse  y ~ x + (expr | group)
├── data.py            formulaic + pandas -> JAX arrays
├── covariance.py      Σ_θ via log-Cholesky (bijection from ℝ^{q(q+1)/2})
├── families.py        Gaussian/Bernoulli/Binomial/Poisson/Gamma/NegBin × link
├── nll.py             joint nll g(θ, b) assembly
├── laplace.py         inner Newton + Laplace marginal, with implicit-diff
├── linalg.py          HessianStructure: DenseHessian + BlockDiagHessian
├── optimize.py        pure-JAX outer optimizer (optimistix.BFGS, jitted)
├── model.py           MixedModel: from_formula(...).fit(...) + accessors
├── summary.py         pretty-printed summary()
└── inference/
    ├── wald.py        closed-form β-block Wald (Schur complement)
    ├── profile.py     profile-likelihood CIs with ζ-interpolation
    └── bootstrap.py   parametric bootstrap
```

### Joint nll and parameterization

The flat parameter vector `θ` (what the outer optimizer sees) is laid out as

    [ β  |  log_phi  |  vech-log-Cholesky for RE block 1  |  ...  ],

with `log_phi` present iff the family has a dispersion parameter. Latent
`b` is a flat vector concatenating each RE block's `(G_k · q_k)` entries.
Every parameter is unconstrained by construction (log on positive
quantities, log-Cholesky on covariance matrices), so the outer optimizer
sees only ℝⁿ.

### Mode finding with implicit differentiation

`laplace.py::find_mode_implicit` returns `b̂(θ)` with a correct gradient via
the standard one-step trick:

1. Newton iterate on `b` under `stop_gradient` to get a numerical `b_sg`.
2. Do one final Newton step from `b_sg` with `θ` left differentiable:
   `b̂ = b_sg − H⁻¹ ∇g(θ, b_sg)`. Numerically `b̂ = b_sg` because `∇g ≈ 0`
   at the mode, but JAX now sees the dependency `db̂/dθ = −H⁻¹ ∂²g/∂θ∂b`
   exactly (the implicit-function theorem). This makes `jax.grad` of the
   Laplace marginal exact and lets the outer BFGS converge cleanly even on
   GLMMs.

The factor for `½ log det H` is built **again** at the IFT-corrected `b̂`,
so JAX tracks the dependence of H_bb on both `θ` directly and on `b̂(θ)`.
For LMMs the second build is redundant (H_bb doesn't depend on `b`), but
for GLMMs the indirect dependence is non-trivial — omitting it gives an
inconsistent value/gradient pair on models with crossed REs or
observation-level REs (e.g. the cbpp binomial GLMM) and the optimizer
stops short of the true MLE.

### Structured `H_bb`

`linalg.py` exposes a `HessianStructure` interface with `build`, `solve`,
`logdet` methods, and two implementations:

- `BlockDiagHessian` — for models with exactly one `(... | g)` term. The
  Hessian decomposes into `G` independent `q × q` blocks; we compute the
  blocks by a scatter-add of weighted outer products `Ze[i] Ze[i]ᵀ · w_i`
  and use a `vmap`-ed dense Cholesky over groups. `H_bb` is never
  materialized.

- `DenseHessian` — fallback via `jax.hessian`. Used when the model has
  zero or two-plus `(... | g)` terms (i.e. multiple grouping factors,
  whether nested or crossed). Correct, but scales `O((Σ G_k · q_k)³)`.

`make_structure(spec, family)` selects the cheaper applicable backend.

### Outer optimizer

`optimize.py` wraps `optimistix.BFGS` in a single `jax.jit`. The whole fit
runs as one compiled computation — no Python in the hot loop. Each
`MixedModel` instance caches its compiled solver and reuses it across
subsequent `.fit()` calls. Profile-CI refits use the same machinery with
one coordinate of `θ` constrained.

### Wald SE

`inference/wald.py` computes `vcov(β̂)` analytically from the β-block of the
joint Hessian of `g`, using the Schur complement

    I(β) = XᵀWX − (XᵀWZ) H_bb⁻¹ (ZᵀWX),

where `W` is the diagonal of per-observation `∂²nll/∂η²` at `b̂`. This is
the same formula as `lme4` / `glmmTMB` and avoids second-order autodiff
through the full Laplace marginal (which was orders of magnitude slower).

### Profile likelihood

`inference/profile.py` adaptively sweeps each parameter `ψ_j` outward from
`ψ̂_j` until the signed-square-root statistic
`ζ_j(c) = sign(c − ψ̂_j) · √(D(c) − D̂)` crosses the critical value. Each
refit is a one-coord-constrained optimization compiled with optimistix and
warm-started from the previous fit. The ζ-trace is monotone-interpolated
(PCHIP) to find the CI endpoints. All parameters profile on the
unconstrained `θ` scale (`log σ`, log-Cholesky entries) so ζ stays
near-linear and the boundary `σ = 0` shows up as an asymptote rather than
a clipped CI.

### Parametric bootstrap

`inference/bootstrap.py` simulates new responses by drawing fresh
`b ~ N(0, Σ̂)` and `y | b ~ family`, refits, and collects β̂, σ̂, and per-RE
SDs into a `BootstrapResult`. Slow but the reference for finite-sample
inference.

## 3. Dependencies

| Concern | Library |
|---|---|
| Formula parsing | `formulaic` (with a small string preprocessor for `(... \| g)`) |
| Data frames | `pandas` (input only) |
| Arrays + AD | `jax` (hard dependency) |
| Outer optimizer | `optimistix` (BFGS), jit-compiled |
| Inner Newton | hand-rolled in `jax.lax.while_loop` with implicit-diff |
| Sparse linear algebra | none yet; `BlockDiagHessian` exploits structure analytically |
| GLM families | reimplemented in JAX (~150 lines); `statsmodels` is the validation reference |
| Profile interpolation | `scipy.interpolate.PchipInterpolator` |
| Bootstrap parallelism | not yet |

## 4. Families and links

Each family is a frozen dataclass holding a `link: Link` field. The
canonical link is the default. Users compose families explicitly,
e.g. ``Bernoulli()``, ``Bernoulli(ProbitLink())``,
``Gamma(InverseLink())``. There is no string-based resolution — ``family=``
and the link inside it are always concrete objects.

| family | canonical link | other links supported |
|---|---|---|
| `gaussian` | identity | log, inverse |
| `bernoulli` | logit | probit, cloglog |
| `binomial` | logit | probit, cloglog |
| `poisson` | log | identity, sqrt |
| `gamma` | log | identity, inverse |
| `negative_binomial` | log | identity, sqrt |

The canonical-link nll for Bernoulli/Binomial uses the `softplus` form for
numerical stability; non-canonical links fall back to the generic
clipped-log form. Variance functions are link-independent.

## 5. Validation

Everything that involves a fitted model is checked against `glmmTMB` as
the single source of truth. `tests/cases.py` declares one :class:`Case`
per model with its expected fixed effects, random-effect SDs and
correlations, residual scale and log-likelihood; a single parametrized
test in `tests/test_glmmtmb.py` fits each case and asserts agreement to
1e-3 by default. Current cases cover the following feature combinations,
all against `glmmTMB 1.1.14`:

| case | family / link | RE structure | dataset |
|---|---|---|---|
| `sleepstudy_random_intercept` | Gaussian / identity | `(1\|Subject)` | lme4 |
| `sleepstudy_random_slope` | Gaussian / identity | `(1+Days\|Subject)` | lme4 |
| `Dyestuff` | Gaussian / identity | `(1\|Batch)` | lme4 |
| `Penicillin` | Gaussian / identity | `(1\|plate)+(1\|sample)` | lme4 |
| `cbpp_basic` | Binomial / logit | `(1\|herd)` | lme4 |
| `cbpp_obs` | Binomial / logit | `(1\|herd)+(1\|obs)` | lme4 |
| `cbpp_probit` | Binomial / probit | `(1\|herd)` | lme4 |
| `poisson_log` | Poisson / log | `(1\|g)` | seeded synthetic |
| `gamma_log` | Gamma / log | `(1\|g)` | seeded synthetic |
| `negbin_log` | NegBin / log | `(1\|g)` | seeded synthetic |

Reference values were produced once by `scripts/fit_canonical.R` and are
stored verbatim in `tests/cases.py`. Regeneration requires R ≥ 4.4 with
the `glmmTMB` and `jsonlite` packages.

Orthogonal scaffolding tests:

- The lme4-style API (`fixed_effects`, `random_effects`, `vcov`,
  `confidence_intervals`, `bootstrap`, `simulate`, `predict`, `residuals`,
  `aic`/`bic`/`dof`, etc.) is covered in `tests/test_api.py`.
- Inference mechanics (Wald, profile, parametric bootstrap) in
  `tests/test_inference.py`.
- Formula parsing is unit-tested in `tests/test_formula.py`.

## 6. Current limitations

- Crossed REs use the dense `H_bb` path; scaling is `O((Σ G_k · q_k)³)`.
- No structured covariances (AR1, compound symmetry, Toeplitz).
- No zero-inflation, censoring, or truncation.
- The outer optimizer does Laplace ML; no REML.
- No Satterthwaite / Kenward-Roger degrees-of-freedom corrections.
