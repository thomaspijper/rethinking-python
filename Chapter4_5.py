"""Bayesian modeling of height as a function of weight using Laplace approximation. The
model includes a quadratic term to capture non-linear relationships between weight and height.
The code also computes and visualizes the 89% HPDI for both the fitted line (mean) and the
predicted heights of individuals (posterior predictive interval).

Adapted from Rethinking Statistics 2nd edition, Chapter 4.5.1.
"""

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm

from patsy import bs as patsy_bs
from pymc_extras.inference import fit_laplace


### 4.5.1 Polynomial regression ###

# Code 4.64 — reading the data
d: pd.DataFrame = pd.read_csv("Howell1.csv", sep=";")

# Code 4.65 — standardizing the weights and fitting the model using Laplace approximation
# Standardizing the weights makes the numerical optimization more stable, as the parameters
# will be on a more similar scale.
xbar = d["weight"].mean()
sd = d["weight"].std()
weights_t = (d["weight"].to_numpy() - xbar) / sd

with pm.Model() as model_m45:
    a    = pm.Normal("a", mu=178, sigma=20)
    b1    = pm.Lognormal("b1", mu=0, sigma=1)
    b2    = pm.Normal("b2", mu=0, sigma=1)
    sigma = pm.Uniform("sigma", lower=0, upper=50)
    mu = a + b1 * weights_t + b2 * weights_t**2
    pm.Normal("height", mu=mu, sigma=sigma, observed=d["height"].to_numpy())
    idata_m45 = fit_laplace(draws=10_000)

# Code 4.66 — summarizing the posterior samples
print(az.summary(idata_m45, var_names=["a", "b1", "b2", "sigma"], ci_prob=0.89, round_to=2, kind="stats"))

# Code 4.67 — computing mean relationship and the 89% intervals of the mean and the predictions
# weight.seq <- seq(from=-2.2, to=2, length.out=30)   (standardized weight sequence)
# pred_dat   <- list(weight_s=weight.seq, weight_s2=weight.seq^2)
weight_seq_s = np.linspace(-2.2, 2, 30)               # shape (30,) — standardized
sample_a     = idata_m45.posterior["a"].to_numpy().ravel()
sample_b1    = idata_m45.posterior["b1"].to_numpy().ravel()
sample_b2    = idata_m45.posterior["b2"].to_numpy().ravel()
sample_sigma = idata_m45.posterior["sigma"].to_numpy().ravel()

# mu <- link(m4.5, data=pred_dat)
mu_link = (sample_a[:, np.newaxis]
           + sample_b1[:, np.newaxis] * weight_seq_s[np.newaxis, :]
           + sample_b2[:, np.newaxis] * weight_seq_s[np.newaxis, :] ** 2)  # shape (10000, 30)

# mu.mean <- apply(mu, 2, mean)
# mu.PI   <- apply(mu, 2, PI, prob=0.89)
mu_mean = mu_link.mean(axis=0)                                              # shape (30,)
mu_pi   = np.quantile(mu_link, [0.055, 0.945], axis=0)                     # shape (2, 30)

# sim.height <- sim(m4.5, data=pred_dat)
rng = np.random.default_rng(42)
sim_height = rng.normal(mu_link, sample_sigma[:, np.newaxis])               # shape (10000, 30)

# height.PI <- apply(sim.height, 2, PI, prob=0.89)
height_pi = np.quantile(sim_height, [0.055, 0.945], axis=0)                # shape (2, 30)

# Code 4.68 — plotting the data, mean relationship, and the 89% intervals, recreating the
# middle plot in Figure 4.11
weights_s = (d["weight"].to_numpy() - xbar) / sd   # standardized observed weights
fig, ax = plt.subplots()
ax.scatter(weights_s, d["height"], alpha=0.4, s=10)
ax.plot(weight_seq_s, mu_mean, color="black", linewidth=1, label="MAP line")
ax.fill_between(weight_seq_s, mu_pi[0],   mu_pi[1],   alpha=0.4, color="steelblue", label="89% PI for μ")
ax.fill_between(weight_seq_s, height_pi[0], height_pi[1], alpha=0.2, color="steelblue", label="89% PI for heights")
ax.set_xlabel("weight (standardized)")
ax.set_ylabel("height")
ax.set_title("Polynomial regression: MAP + 89% PI (μ) + 89% PI (heights)")
ax.legend()
plt.tight_layout()
plt.show()

# Code 4.69 — the cubic version of the above model + plotting (recreating the right plot in Figure 4.11)
#
# Note that we do things a bit differently this time. Instead of using link() and sim() to get
# the posterior predictions, we use the PyMC-native alternative for link() + sim() using pm.Data and
# pm.sample_posterior_predictive. This is the idiomatic PyMC way to get posterior predictions at new
# input values.
# Key requirements vs the 2nd order polynomial model above:
#   1. pm.Data("weight", ...) — makes the predictor swappable via pm.set_data()
#   2. pm.Data("height_obs", ...) — the observed response must also be pm.Data because PyMC
#      evaluates the full likelihood graph during sample_posterior_predictive, which means mu
#      (shape 30) and the observed array must have the same length; swapping height_obs to a
#      dummy zeros array of length 30 satisfies this — the values themselves are never used
#   3. pm.Deterministic("mu", ...) — registers the mean expression as a named variable so
#      sample_posterior_predictive stores it separately from the noisy height samples;
#      without it, mu is just a Python variable PyMC uses internally but never saves
with pm.Model() as model_m46:
    weight_data = pm.Data("weight", weights_t)                       # swappable predictor
    height_obs  = pm.Data("height_obs", d["height"].to_numpy())     # swappable observed
    a     = pm.Normal("a",     mu=178, sigma=20)
    b1    = pm.Lognormal("b1", mu=0,   sigma=1)
    b2    = pm.Normal("b2",    mu=0,   sigma=1)
    b3    = pm.Normal("b3",    mu=0,   sigma=1)
    sigma = pm.Uniform("sigma", lower=0, upper=50)
    mu    = pm.Deterministic("mu", a + b1 * weight_data             # exposed for PPC
                                     + b2 * weight_data**2
                                     + b3 * weight_data**3)
    pm.Normal("height", mu=mu, sigma=sigma, observed=height_obs)
    idata_m46 = fit_laplace(draws=10_000)

# Swap both weight and height_obs to the prediction grid.
# PyMC evaluates the full likelihood graph during sample_posterior_predictive, so
# height_obs must have the same length as mu (30). The dummy zero values are never
# used — only the posterior draws of a, b1, b2, b3, sigma matter.
weight_seq_s_cubic = np.linspace(-2.2, 2, 30)
with model_m46:
    pm.set_data({"weight":     weight_seq_s_cubic,
                 "height_obs": np.zeros(len(weight_seq_s_cubic))})  # dummy values (shape must match that of weight)
    ppc = pm.sample_posterior_predictive(idata_m46, var_names=["mu", "height"])

# ppc.posterior_predictive["mu"]     — shape (chains, draws, 30): μ evaluated at weight_seq_s_cubic;
#                                       reflects parameter uncertainty only (no σ noise)
# ppc.posterior_predictive["height"] — shape (chains, draws, 30): μ + Normal(0, σ) noise;
#                                       the full posterior predictive distribution
mu_samples_ppc = ppc.posterior_predictive["mu"].stack(sample=("chain", "draw")).values.T      # (10000, 30)
height_samples_ppc = ppc.posterior_predictive["height"].stack(sample=("chain", "draw")).values.T  # (10000, 30)

mu_mean_ppc  = mu_samples_ppc.mean(axis=0)
mu_pi_ppc    = np.quantile(mu_samples_ppc, [0.055, 0.945], axis=0)
height_pi_ppc = np.quantile(height_samples_ppc, [0.055, 0.945], axis=0)

fig, ax = plt.subplots()
ax.scatter((d["weight"].to_numpy() - xbar) / sd, d["height"], alpha=0.4, s=10)
ax.plot(weight_seq_s_cubic, mu_mean_ppc, color="black", linewidth=1, label="MAP line")
ax.fill_between(weight_seq_s_cubic, mu_pi_ppc[0], mu_pi_ppc[1], alpha=0.4, color="steelblue", label="89% PI for μ")
ax.fill_between(weight_seq_s_cubic, height_pi_ppc[0], height_pi_ppc[1], alpha=0.2, color="steelblue", label="89% PI for heights")
ax.set_xlabel("weight (standardized)")
ax.set_ylabel("height")
ax.set_title("Cubic regression (PyMC-native PPC): MAP + 89% PI (μ) + 89% PI (heights)")
ax.legend()
plt.tight_layout()
plt.show()


### 4.5.2 Splines ###

# Code 4.72 — reading the data and printing statistics
d: pd.DataFrame = pd.read_csv("cherry_blossoms.csv", sep=";")
print(d.describe(percentiles=[0.055, 0.945]))

# Code 4.73 — filter out rows with missing doy values, then define knots for the spline basis functions
# R: d2 <- d[complete.cases(d$doy), ]
d2 = d[d["doy"].notna()].sort_values("year").reset_index(drop=True)
num_knots = 15
knot_list = np.quantile(d2["year"], np.linspace(0, 1, num_knots))

# Code 4.74 — construct basis functions for a 3-degree (cubic) spline
# patsy.bs() replicates R's bs() exactly: boundary knots default to range(x),
# interior knots are passed explicitly, include_intercept=True matches intercept=TRUE.
interior_knots = knot_list[1:-1]  # knot_list[-c(1, num_knots)] in R — 13 interior knots
B = np.asarray(patsy_bs(d2["year"].to_numpy(), knots=interior_knots, degree=3, include_intercept=True))

# Code 4.76 — fit the model using Laplace approximation
with pm.Model() as model_m47:
    a     = pm.Normal("a", mu=100, sigma=10)
    w     = pm.Normal("w", mu=0, sigma=10, shape=B.shape[1])
    sigma = pm.Exponential("sigma", lam=1)
    mu = a + B @ w
    pm.Normal("T", mu=mu, sigma=sigma, observed=d2["doy"].to_numpy())
    idata_m47 = fit_laplace(draws=10_000)

w_mean = idata_m47.posterior["w"].mean(dim=["chain", "draw"]).to_numpy()

# Code 4.78 — compute 97% posterior interval for mu at each observation (equivalent to link + PI in R)
sample_a = idata_m47.posterior["a"].to_numpy().ravel()                         # (n_samples,)
sample_w = idata_m47.posterior["w"].to_numpy().reshape(-1, B.shape[1])         # (n_samples, 17)
mu_samples = sample_a[:, None] + (sample_w @ B.T)                          # (n_samples, n_obs)
mu_pi = np.percentile(mu_samples, [1.5, 98.5], axis=0)                     # 97% PI

# Plot: basis functions (top), posterior weighted basis functions (middle), data + 97% PI (bottom)
# This part is an implementation of code 4.75
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
ax1.set_xlim(d2["year"].min(), d2["year"].max())
ax1.set_ylim(0, 1)
ax1.set_ylabel("basis value")
for i in range(B.shape[1]):
    ax1.plot(d2["year"], B[:, i])

# This part is an implementation of code 4.77
ax2.set_xlim(d2["year"].min(), d2["year"].max())
ax2.set_ylim(-6, 6)
ax2.set_ylabel("basis * weight")
for i in range(B.shape[1]):
    ax2.plot(d2["year"], w_mean[i] * B[:, i])

# This part is an implementation of the last two lines of code 4.78
ax3.scatter(d2["year"], d2["doy"], alpha=0.3, s=8, color="steelblue")
ax3.fill_between(d2["year"], mu_pi[0], mu_pi[1], color="black", alpha=0.5)
ax3.set_xlabel("year")
ax3.set_ylabel("Day of year")

plt.tight_layout()
plt.show()