## Fit canonical lme4/glmmTMB examples and emit JSON for use in tests.
## Run with:  Rscript fit_canonical.R

suppressMessages({
  library(lme4)
  library(glmmTMB)
  library(jsonlite)
})

extract <- function(fit) {
  fe <- glmmTMB::fixef(fit)$cond
  vc <- glmmTMB::VarCorr(fit)$cond
  re_sds   <- list()
  re_corrs <- list()
  for (grp in names(vc)) {
    m  <- vc[[grp]]
    sd <- attr(m, "stddev")
    co <- attr(m, "correlation")
    re_sds[[grp]] <- as.list(sd)
    if (length(sd) > 1) {
      cor_list <- list()
      nms <- names(sd)
      for (i in seq_along(nms)) {
        for (j in seq_along(nms)) {
          if (j > i) {
            cor_list[[paste0(nms[i], ",", nms[j])]] <- co[i, j]
          }
        }
      }
      re_corrs[[grp]] <- cor_list
    }
  }
  list(
    family   = family(fit)$family,
    link     = family(fit)$link,
    fixef    = as.list(fe),
    sigma    = unname(sigma(fit)),
    re_sds   = re_sds,
    re_corrs = re_corrs,
    logLik   = as.numeric(logLik(fit)),
    df       = attr(logLik(fit), "df"),
    nobs     = nobs(fit)
  )
}

fits <- list()

## ---- Gaussian / identity ------------------------------------------------

fits$sleepstudy_ri <- extract(glmmTMB(
  Reaction ~ Days + (1 | Subject),
  data = sleepstudy, REML = FALSE
))

fits$sleepstudy_rs <- extract(glmmTMB(
  Reaction ~ Days + (1 + Days | Subject),
  data = sleepstudy, REML = FALSE
))

fits$Dyestuff <- extract(glmmTMB(
  Yield ~ 1 + (1 | Batch),
  data = lme4::Dyestuff, REML = FALSE
))

fits$Penicillin <- extract(glmmTMB(
  diameter ~ 1 + (1 | plate) + (1 | sample),
  data = lme4::Penicillin, REML = FALSE
))

## ---- Binomial / logit (canonical) ---------------------------------------

cbpp <- lme4::cbpp
cbpp$obs <- 1:nrow(cbpp)

fits$cbpp_basic <- extract(glmmTMB(
  cbind(incidence, size - incidence) ~ period + (1 | herd),
  data = cbpp, family = binomial
))

fits$cbpp_obs <- extract(glmmTMB(
  cbind(incidence, size - incidence) ~ period + (1 | herd) + (1 | obs),
  data = cbpp, family = binomial
))

## ---- Binomial / probit (non-canonical link) -----------------------------

fits$cbpp_probit <- extract(glmmTMB(
  cbind(incidence, size - incidence) ~ period + (1 | herd),
  data = cbpp, family = binomial(link = "probit")
))

## ---- Poisson / log (synthetic but seeded) -------------------------------

set.seed(1)
G <- 40; n_per <- 25
g <- rep(seq_len(G), each = n_per)
x <- rnorm(G * n_per)
b <- rnorm(G, sd = 0.6)
eta <- 0.5 + 0.4 * x + b[g]
y <- rpois(G * n_per, exp(eta))
poisson_df <- data.frame(y = y, x = x, g = factor(g))
fits$poisson_log <- extract(glmmTMB(
  y ~ x + (1 | g),
  data = poisson_df, family = poisson
))

## ---- Gamma / log --------------------------------------------------------

set.seed(2)
G <- 40; n_per <- 30
g <- rep(seq_len(G), each = n_per)
x <- rnorm(G * n_per)
b <- rnorm(G, sd = 0.5)
eta <- 1.0 + 0.4 * x + b[g]
mu <- exp(eta)
shape <- 1 / 0.4               # dispersion phi = 0.4
y <- rgamma(G * n_per, shape = shape, scale = mu / shape)
gamma_df <- data.frame(y = y, x = x, g = factor(g))
fits$gamma_log <- extract(glmmTMB(
  y ~ x + (1 | g),
  data = gamma_df, family = Gamma(link = "log")
))

## ---- Negative binomial (NB2) / log --------------------------------------

set.seed(3)
G <- 50; n_per <- 30
g <- rep(seq_len(G), each = n_per)
x <- rnorm(G * n_per)
b <- rnorm(G, sd = 0.5)
eta <- 1.5 + 0.3 * x + b[g]
mu <- exp(eta)
theta <- 1 / 0.4
lam <- rgamma(G * n_per, shape = theta, scale = mu / theta)
y <- rpois(G * n_per, lam)
nb_df <- data.frame(y = y, x = x, g = factor(g))
fits$negbin_log <- extract(glmmTMB(
  y ~ x + (1 | g),
  data = nb_df, family = nbinom2
))

## Write the synthetic data so Python can load the same rows.
write.csv(poisson_df, "/tmp/poisson_log_data.csv", row.names = FALSE)
write.csv(gamma_df,   "/tmp/gamma_log_data.csv",   row.names = FALSE)
write.csv(nb_df,      "/tmp/negbin_log_data.csv",  row.names = FALSE)

writeLines(
  toJSON(fits, pretty = TRUE, auto_unbox = TRUE, digits = 10),
  "/tmp/fits.json"
)
