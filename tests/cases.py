"""Canonical fit cases, each anchored to a reference fit from `glmmTMB`.

For every case we declare:

  - what `MixedModel.from_formula` call to make,
  - the expected values for fixed effects, random-effects SDs, correlations,
    residual SD, and log-likelihood, taken from running `glmmTMB` in R,
  - per-quantity tolerances.

The reference values were produced once by `scripts/fit_canonical.R`
against `glmmTMB 1.1.14`. Synthetic datasets (Poisson, Gamma, NegBin) were
generated in R with fixed seeds and shipped alongside the lme4 ones; the
CSV under `tests/data/` is exactly what `glmmTMB` was given.

The full log-likelihood includes any constants that the package's
`MixedModel.log_likelihood()` drops (currently just the binomial
coefficient `Σ log C(n, y)` for `family="binomial"`). The test harness
applies the appropriate offset before comparison so the absolute logLik
matches `glmmTMB`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from mixedmodels import (
    Bernoulli,
    Binomial,
    Family,
    Gamma,
    Gaussian,
    NegativeBinomial,
    Poisson,
    ProbitLink,
)

_DATA_DIR = Path(__file__).parent / "data"


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------


def _lme4(name: str) -> pd.DataFrame:
    from statsmodels.datasets import get_rdataset

    return get_rdataset(name, "lme4").data


def _sleepstudy() -> pd.DataFrame:
    return _lme4("sleepstudy")


def _Dyestuff() -> pd.DataFrame:
    return _lme4("Dyestuff")


def _Penicillin() -> pd.DataFrame:
    return _lme4("Penicillin")


def _cbpp(*, with_obs: bool = False) -> pd.DataFrame:
    df = _lme4("cbpp").copy()
    df["period"] = df["period"].astype("category")
    if with_obs:
        df["obs"] = np.arange(len(df))
    return df


def _csv(name: str) -> pd.DataFrame:
    df = pd.read_csv(_DATA_DIR / name)
    # Synthetic CSVs use a stringified factor for the grouping column;
    # turn it back into a categorical so formulaic groups cleanly.
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype("category")
    return df


# ---------------------------------------------------------------------------
# Case definition
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Case:
    """One model declaration with its glmmTMB reference values."""

    name: str
    family: Family
    formula: str
    data_loader: Callable[[], pd.DataFrame]
    expected: dict[str, float]
    tol: dict[str, float] = field(default_factory=dict)
    weights: Callable[[pd.DataFrame], np.ndarray] | None = None

    def default_tol(self, key: str) -> float:
        """Default per-quantity tolerance when none is given explicitly."""
        if key in self.tol:
            return self.tol[key]
        if key == "logLik":
            return 1e-2
        if key.startswith("corr["):
            return 5e-3
        return 1e-3


# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------


CASES: list[Case] = [
    # Gaussian / identity ----------------------------------------------------
    Case(
        name="sleepstudy_random_intercept",
        family=Gaussian(),
        formula="Reaction ~ Days + (1 | Subject)",
        data_loader=_sleepstudy,
        expected={
            "Intercept": 251.4050944577,
            "Days": 10.4672871976,
            "sigma": 30.8954359046,
            "sd[Intercept|Subject]": 36.0120743517,
            "logLik": -897.0393215026,
        },
    ),
    Case(
        name="sleepstudy_random_slope",
        family=Gaussian(),
        formula="Reaction ~ Days + (1 + Days | Subject)",
        data_loader=_sleepstudy,
        expected={
            "Intercept": 251.4043560442,
            "Days": 10.4672262495,
            "sigma": 25.5917590777,
            "sd[Intercept|Subject]": 23.7800740281,
            "sd[Days|Subject]": 5.7166971847,
            "corr[Intercept,Days|Subject]": 0.0813185377,
            "logLik": -875.9696722529,
        },
    ),
    Case(
        name="Dyestuff",
        family=Gaussian(),
        formula="Yield ~ 1 + (1 | Batch)",
        data_loader=_Dyestuff,
        expected={
            "Intercept": 1527.5000262798,
            "sigma": 49.5100987573,
            "sd[Intercept|Batch]": 37.2603992283,
            "logLik": -163.6635299406,
        },
    ),
    Case(
        name="Penicillin",
        family=Gaussian(),
        formula="diameter ~ 1 + (1 | plate) + (1 | sample)",
        data_loader=_Penicillin,
        expected={
            "Intercept": 22.9722183439,
            "sigma": 0.5499320515,
            "sd[Intercept|plate]": 0.8455715303,
            "sd[Intercept|sample]": 1.7706413110,
            "logLik": -166.0941743343,
        },
    ),
    # Binomial / logit -------------------------------------------------------
    Case(
        name="cbpp_basic",
        family=Binomial(),
        formula="incidence ~ period + (1 | herd)",
        data_loader=lambda: _cbpp(with_obs=False),
        weights=lambda df: df["size"].to_numpy(),
        expected={
            "Intercept": -1.3985324664,
            "period[T.2]": -0.9923322929,
            "period[T.3]": -1.1286712975,
            "period[T.4]": -1.5803136871,
            "sd[Intercept|herd]": 0.6422616722,
            "logLik": -92.0262818648,
        },
    ),
    Case(
        name="cbpp_obs",
        family=Binomial(),
        formula="incidence ~ period + (1 | herd) + (1 | obs)",
        data_loader=lambda: _cbpp(with_obs=True),
        weights=lambda df: df["size"].to_numpy(),
        expected={
            "Intercept": -1.5002918362,
            "period[T.2]": -1.2265088896,
            "period[T.3]": -1.3288404777,
            "period[T.4]": -1.8662649466,
            "sd[Intercept|herd]": 0.1839471194,
            "sd[Intercept|obs]": 0.8910784613,
            "logLik": -87.3191555415,
        },
    ),
    # Binomial / probit (non-canonical link) --------------------------------
    Case(
        name="cbpp_probit",
        family=Binomial(ProbitLink()),
        formula="incidence ~ period + (1 | herd)",
        data_loader=lambda: _cbpp(with_obs=False),
        weights=lambda df: df["size"].to_numpy(),
        expected={
            "Intercept": -0.8318499975,
            "period[T.2]": -0.5265962968,
            "period[T.3]": -0.6150715733,
            "period[T.4]": -0.7979442168,
            "sd[Intercept|herd]": 0.3386344759,
            "logLik": -92.5833377032,
        },
    ),
    # Poisson / log (synthetic, seeded) -------------------------------------
    Case(
        name="poisson_log",
        family=Poisson(),
        formula="y ~ x + (1 | g)",
        data_loader=lambda: _csv("poisson_log_data.csv"),
        expected={
            "Intercept": 0.4045874239,
            "x": 0.4182310576,
            "sd[Intercept|g]": 0.5831800024,
            "logLik": -1593.8158179474,
        },
    ),
    # Gamma / log (synthetic, seeded) ---------------------------------------
    Case(
        name="gamma_log",
        family=Gamma(),
        formula="y ~ x + (1 | g)",
        data_loader=lambda: _csv("gamma_log_data.csv"),
        expected={
            "Intercept": 0.9582651672,
            "x": 0.4362097881,
            "sigma": 0.6354688198,
            "sd[Intercept|g]": 0.4221694900,
            "logLik": -2214.2952627826,
        },
    ),
    # Negative binomial (NB2) / log (synthetic, seeded) ---------------------
    Case(
        name="negbin_log",
        family=NegativeBinomial(),
        formula="y ~ x + (1 | g)",
        data_loader=lambda: _csv("negbin_log_data.csv"),
        expected={
            "Intercept": 1.5101126047,
            "x": 0.3045394417,
            "sigma": 2.2235922092,
            "sd[Intercept|g]": 0.4503441381,
            "logLik": -3886.4955193332,
        },
    ),
]


# Silence unused-import lint for re-exports the test module may want later.
_ = (Bernoulli,)
