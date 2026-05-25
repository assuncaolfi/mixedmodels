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
    logLik = -875.9697   deviance = 1751.9393
       AIC = 1763.9393        BIC = 1783.0971   df = 6

Random effects:
Groups              Name                    Variance    Std.Dev.  Corr
Subject             Intercept               565.5154     23.7806
                    Days                     32.6822      5.7168    0.08
Residual                                    654.9410     25.5918
Number of obs: 180, groups: Subject 18

Fixed effects:
           estimate  std_error        z        p  ci_lower  ci_upper
term
Intercept  251.4051     6.6323  37.9063   0.0000  238.4061  264.4041
Days        10.4673     1.5022   6.9678   0.0000    7.5230   13.4116
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
