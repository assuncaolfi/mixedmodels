"""GLM families and links in JAX.

Each family is a small frozen dataclass holding a `link: Link` and knowing how to:
  - map a linear predictor to a mean (via `link.inv_link`),
  - evaluate the per-observation negative log-likelihood,
  - compute the variance function,
  - simulate a response.

Links are independent of families: a `Bernoulli` can use `LogitLink()` (the
canonical default), `ProbitLink()`, or `CloglogLink()`. The whole module is
kept tiny (~250 LoC) and JAX-native because the joint nll runs through
`jax.grad`. `statsmodels` is used as the validation reference but not
imported here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import jax
import jax.numpy as jnp
from jax import random
from jax.scipy.special import gammaln
from jax.scipy.stats import norm as jax_norm

_LOG_2PI = jnp.log(2.0 * jnp.pi)
_EPS = 1e-10  # used to clip probabilities away from {0, 1}


# ---------------------------------------------------------------------------
# Links
# ---------------------------------------------------------------------------


class Link(Protocol):
    name: str

    def inv_link(self, eta: jax.Array) -> jax.Array: ...
    def to_eta(self, y: jax.Array) -> jax.Array: ...


@dataclass(frozen=True)
class IdentityLink:
    name: str = "identity"

    def inv_link(self, eta):
        return eta

    def to_eta(self, y):
        return y


@dataclass(frozen=True)
class LogLink:
    name: str = "log"

    def inv_link(self, eta):
        return jnp.exp(eta)

    def to_eta(self, y):
        return jnp.log(jnp.maximum(y, _EPS))


@dataclass(frozen=True)
class InverseLink:
    name: str = "inverse"

    def inv_link(self, eta):
        return 1.0 / eta

    def to_eta(self, y):
        return 1.0 / jnp.maximum(y, _EPS)


@dataclass(frozen=True)
class SqrtLink:
    name: str = "sqrt"

    def inv_link(self, eta):
        return eta * eta

    def to_eta(self, y):
        return jnp.sqrt(jnp.maximum(y, 0.0))


@dataclass(frozen=True)
class LogitLink:
    name: str = "logit"

    def inv_link(self, eta):
        return jax.nn.sigmoid(eta)

    def to_eta(self, y):
        p = jnp.clip(y, 0.02, 0.98)
        return jnp.log(p / (1.0 - p))


@dataclass(frozen=True)
class ProbitLink:
    name: str = "probit"

    def inv_link(self, eta):
        return jax_norm.cdf(eta)

    def to_eta(self, y):
        p = jnp.clip(y, 0.02, 0.98)
        return jax_norm.ppf(p)


@dataclass(frozen=True)
class CloglogLink:
    name: str = "cloglog"

    def inv_link(self, eta):
        return -jnp.expm1(-jnp.exp(eta))  # 1 − exp(−exp(η))

    def to_eta(self, y):
        p = jnp.clip(y, 0.02, 0.98)
        return jnp.log(-jnp.log1p(-p))


# ---------------------------------------------------------------------------
# Families
# ---------------------------------------------------------------------------


class Family(Protocol):
    name: str
    has_dispersion: bool
    link: Link

    def inv_link(self, eta: jax.Array) -> jax.Array: ...
    def nll(self, y, eta, weights, dispersion) -> jax.Array: ...
    def variance(self, mu, weights) -> jax.Array: ...
    def simulate(self, key, eta, weights, dispersion) -> jax.Array: ...


def _clip_p(p):
    return jnp.clip(p, _EPS, 1.0 - _EPS)


# ----- Gaussian -----------------------------------------------------------


@dataclass(frozen=True)
class Gaussian:
    link: Link = IdentityLink()
    name: str = "gaussian"
    has_dispersion: bool = True

    def inv_link(self, eta):
        return self.link.inv_link(eta)

    def nll(self, y, eta, weights, dispersion):
        mu = self.link.inv_link(eta)
        sigma2 = dispersion / weights
        resid = y - mu
        return 0.5 * (jnp.log(sigma2) + _LOG_2PI + resid * resid / sigma2)

    def variance(self, mu, weights):
        return jnp.ones_like(mu) / weights

    def simulate(self, key, eta, weights, dispersion):
        mu = self.link.inv_link(eta)
        sigma = jnp.sqrt(dispersion / weights)
        return mu + sigma * random.normal(key, eta.shape)


# ----- Bernoulli ----------------------------------------------------------


@dataclass(frozen=True)
class Bernoulli:
    link: Link = LogitLink()
    name: str = "bernoulli"
    has_dispersion: bool = False

    def inv_link(self, eta):
        return self.link.inv_link(eta)

    def nll(self, y, eta, weights, dispersion):
        if isinstance(self.link, LogitLink):
            # Numerically stable for the canonical link.
            return weights * (jax.nn.softplus(eta) - y * eta)
        p = _clip_p(self.link.inv_link(eta))
        return -weights * (y * jnp.log(p) + (1.0 - y) * jnp.log1p(-p))

    def variance(self, mu, weights):
        return mu * (1.0 - mu)

    def simulate(self, key, eta, weights, dispersion):
        p = self.link.inv_link(eta)
        return random.bernoulli(key, p, eta.shape).astype(jnp.float64)


# ----- Binomial (weights are trial counts) --------------------------------


@dataclass(frozen=True)
class Binomial:
    link: Link = LogitLink()
    name: str = "binomial"
    has_dispersion: bool = False

    def inv_link(self, eta):
        return self.link.inv_link(eta)

    def nll(self, y, eta, weights, dispersion):
        if isinstance(self.link, LogitLink):
            return weights * jax.nn.softplus(eta) - y * eta
        p = _clip_p(self.link.inv_link(eta))
        return -(y * jnp.log(p) + (weights - y) * jnp.log1p(-p))

    def variance(self, mu, weights):
        return weights * mu * (1.0 - mu)

    def simulate(self, key, eta, weights, dispersion):
        p = self.link.inv_link(eta)
        n = weights.astype(jnp.int32)
        return random.binomial(key, n=n, p=p).astype(jnp.float64)


# ----- Poisson ------------------------------------------------------------


@dataclass(frozen=True)
class Poisson:
    link: Link = LogLink()
    name: str = "poisson"
    has_dispersion: bool = False

    def inv_link(self, eta):
        return self.link.inv_link(eta)

    def nll(self, y, eta, weights, dispersion):
        mu = self.link.inv_link(eta)
        lam = jnp.maximum(weights * mu, _EPS)
        return lam - y * jnp.log(lam) + gammaln(y + 1.0)

    def variance(self, mu, weights):
        return weights * mu

    def simulate(self, key, eta, weights, dispersion):
        lam = weights * self.link.inv_link(eta)
        return random.poisson(key, lam, eta.shape).astype(jnp.float64)


# ----- Gamma --------------------------------------------------------------


@dataclass(frozen=True)
class Gamma:
    link: Link = LogLink()
    name: str = "gamma"
    has_dispersion: bool = True

    def inv_link(self, eta):
        return self.link.inv_link(eta)

    def nll(self, y, eta, weights, dispersion):
        # Gamma(shape = weights/φ, rate = weights/(φμ)).
        mu = jnp.maximum(self.link.inv_link(eta), _EPS)
        a = weights / dispersion
        return a * jnp.log(mu) + a * y / mu - (a - 1.0) * jnp.log(y) - a * jnp.log(a) + gammaln(a)

    def variance(self, mu, weights):
        return mu * mu / weights

    def simulate(self, key, eta, weights, dispersion):
        a = weights / dispersion
        mu = self.link.inv_link(eta)
        return random.gamma(key, a, mu.shape) * mu / a


# ----- Negative binomial (NB2, size θ = 1/dispersion) ---------------------


@dataclass(frozen=True)
class NegativeBinomial:
    link: Link = LogLink()
    name: str = "negative_binomial"
    has_dispersion: bool = True

    def inv_link(self, eta):
        return self.link.inv_link(eta)

    def nll(self, y, eta, weights, dispersion):
        theta = 1.0 / dispersion
        mu = self.link.inv_link(eta)
        log_mu = jnp.log(jnp.maximum(mu, _EPS))
        log_theta_plus_mu = jnp.logaddexp(jnp.log(theta), log_mu)
        log_p = (
            gammaln(y + theta)
            - gammaln(theta)
            - gammaln(y + 1.0)
            + theta * (jnp.log(theta) - log_theta_plus_mu)
            + y * (log_mu - log_theta_plus_mu)
        )
        return -weights * log_p

    def variance(self, mu, weights):
        return mu  # we expose the Poisson part only; full var depends on dispersion

    def simulate(self, key, eta, weights, dispersion):
        theta = 1.0 / dispersion
        mu = self.link.inv_link(eta)
        k1, k2 = random.split(key)
        lam = random.gamma(k1, theta, mu.shape) * mu / theta
        return random.poisson(k2, lam).astype(jnp.float64)


__all__ = [
    "Family",
    "Gaussian",
    "Bernoulli",
    "Binomial",
    "Poisson",
    "Gamma",
    "NegativeBinomial",
    "Link",
    "IdentityLink",
    "LogLink",
    "InverseLink",
    "SqrtLink",
    "LogitLink",
    "ProbitLink",
    "CloglogLink",
]
