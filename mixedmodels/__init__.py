"""TMB-style mixed-effects models in Python, powered by JAX.

Quick start::

    import pandas as pd
    from mixedmodels import MixedModel

    sleep = pd.read_csv("tests/data/sleepstudy.csv")
    fit = MixedModel.from_formula(
        "Reaction ~ Days + (Days | Subject)", sleep
    ).fit()
    print(fit.summary())
"""

import jax

# 64-bit by default — variance estimation in float32 is unreliable.
jax.config.update("jax_enable_x64", True)

from .families import (  # noqa: E402
    Bernoulli,
    Binomial,
    CloglogLink,
    Family,
    Gamma,
    Gaussian,
    IdentityLink,
    InverseLink,
    Link,
    LogitLink,
    LogLink,
    NegativeBinomial,
    Poisson,
    ProbitLink,
    SqrtLink,
)
from .model import MixedModel  # noqa: E402

__all__ = [
    "MixedModel",
    # Families
    "Family",
    "Gaussian",
    "Bernoulli",
    "Binomial",
    "Poisson",
    "Gamma",
    "NegativeBinomial",
    # Links
    "Link",
    "IdentityLink",
    "LogLink",
    "InverseLink",
    "SqrtLink",
    "LogitLink",
    "ProbitLink",
    "CloglogLink",
]

__version__ = "0.2.0"
