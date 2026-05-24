# mixedmodels

A Python package for fitting linear and generalized linear mixed-effects
models. 

This package is inspired by [glmmTMB](https://github.com/glmmTMB/glmmTMB) and
powered by JAX. See [`PLAN.md`](PLAN.md) for the design document.

## AI disclaimer

Most of `mixedmodels` was written with a coding agent driving the design and
tests. Correctness is checked by an independent direct-V-matrix LMM optimizer,
by cross-validation of every family against `statsmodels.GLM` in the
no-random-effects limit, and by simulation-recovery tests for every GLMM family
— see [Validation](#validation).

## Quick start

```python
import pandas as pd
from mixedmodels import MixedModel

sleep = pd.read_csv("tests/data/sleepstudy.csv")
model = MixedModel.from_formula( "Reaction ~ Days + (Days | Subject)", sleep)
fit = model.fit()

print(fit.summary())
```

```
Mixed model fit by Laplace maximum likelihood
 Formula: Reaction ~ Days + (Days | Subject)
  Family: gaussian
    Link: identity
       n: 180
    logLik = -854.5069   deviance = 1709.0138
       AIC = 1721.0138        BIC = 1740.1716   df = 6

Random effects:
Groups              Name                    Variance    Std.Dev.  Corr
Subject             Intercept               624.1109     24.9822
                    Days                     34.9985      5.9160    0.01
Residual                                    485.8048     22.0410
Number of obs: 180, groups: Subject 18

Fixed effects:
           estimate  std_error        z        p  ci_lower  ci_upper
term
Intercept  249.8873     6.6330  37.6735   0.0000  236.8869  262.8877
Days        10.6561     1.5072   7.0704   0.0000    7.7022   13.6101
```

On a 16 GB MacBook Air M1, the first fit takes ~1.7 s (XLA compilation
dominates). Subsequent fits of the same model take ~18 ms.

## API

Formulas use `lme4` syntax: `y ~ x + (1 + x | g)`. Multiple `(... | g)`
terms are allowed. With no `(... | g)` terms the model reduces to a GLM.

Families are constructed explicitly, with the link as the (positional or
keyword) argument. Canonical links are the default, so a bare `Gaussian()`,
`Bernoulli()`, etc. is usually what you want. For `Binomial`, pass trial
counts via `weights=`.

| family | canonical link (default) | other links |
|---|---|---|
| `Gaussian` | `IdentityLink` | `LogLink`, `InverseLink` |
| `Bernoulli` | `LogitLink` | `ProbitLink`, `CloglogLink` |
| `Binomial` | `LogitLink` | `ProbitLink`, `CloglogLink` |
| `Poisson` | `LogLink` | `IdentityLink`, `SqrtLink` |
| `Gamma` | `LogLink` | `IdentityLink`, `InverseLink` |
| `NegativeBinomial` | `LogLink` | `IdentityLink`, `SqrtLink` |

```python
from mixedmodels import MixedModel, Bernoulli, ProbitLink

fit = MixedModel.from_formula(
    formula, data, family=Bernoulli(ProbitLink())
).fit()

# Fixed effects
fit.coefficients() ; fit.fixed_effects()    # β̂ as pd.Series
fit.covariance_matrix()                     # vcov(β̂) as pd.DataFrame
fit.standard_errors()
fit.coefficients_table()                    # estimate, SE, z, p, Wald CI

# Random effects / variance components
fit.random_effects()                        # BLUPs (list of DataFrames)
fit.variance_components()                   # variance/covariance/correlation per RE block
fit.sigma()                                 # residual SD (Gaussian); None otherwise

# Fitted values, residuals, predict, simulate
fit.fitted()
fit.residuals(type="response")              # also "pearson", "working"
fit.predict(newdata, include_re=True)
fit.simulate(n=100, seed=0)

# Likelihood / IC
fit.log_likelihood() ; fit.deviance()
fit.aic() ; fit.bic()
fit.n_observations() ; fit.degrees_of_freedom() ; fit.converged

# Inference
fit.confidence_intervals(method="wald")     # or method="profile" for variance components
fit.profile(["log_sigma"])                  # full ζ-traces
fit.bootstrap(n=200, seed=0)
```

## Limitations

- Crossed random effects (two or more `(... | g)` terms whose groups don't
  nest) use a dense `H_bb` factorization. This works correctly but scales
  as `O((Σ G_k · q_k)³)`; fine for hundreds of random effects, slow for
  thousands.
- No structured random-effects covariances (AR1, compound symmetry).
- No zero-inflation, censoring, or truncation.

## Validation

- LMM (Gaussian): each fit is checked against an independent direct
  V-matrix marginal-likelihood optimizer in `tests/test_lmm_sleepstudy.py`
  (matches `lme4`'s ML fit on the same data to ~5 decimals).
- All six families are cross-validated against `statsmodels.GLM` for the
  no-random-effects limit in `tests/test_families.py`.
- Non-canonical links (probit, cloglog, log-Gaussian, sqrt-Poisson) are
  cross-validated against `statsmodels.GLM` in `tests/test_links.py`.
- All GLMM families have simulation-recovery tests with seeded data.
- Multi-grouping-factor LMM and GLMM are tested in `tests/test_multi_re.py`.
