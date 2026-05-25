# mixedmodels

A Python package for fitting linear and generalized linear mixed-effects
models. 

`mixedmodels` is inspired by [glmmTMB](https://github.com/glmmTMB/glmmTMB) and
powered by JAX. 

Most of `mixedmodels` was written with a coding agent driving the design and
tests. Check results against more mature packages. See [`PLAN.md`](PLAN.md) 
for the design document, validation strategies and limitations.

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

On a 16 GB MacBook Air M1, the first fit takes ~1.7s due to compilation. 
Subsequent fits of the same model take ~18 ms.

## API

Formulas use `lme4` syntax: `y ~ x + (1 + x | g)`. Multiple `(... | g)`
terms are allowed. With no `(... | g)` terms the model reduces to a GLM.

Families are constructed explicitly: `Family(Link())`, with canonical 
links by default. For `Binomial`, pass trial counts column name using 
the `weights` argument.

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
fit.confidence_intervals(method="wald")     # also "profile" for variance components
fit.profile(["log_sigma"])                  # full ζ-traces
fit.bootstrap(n=200, seed=0)
```
