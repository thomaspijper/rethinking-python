"""Bayesian modeling example: Linear regression with unknown mean and standard deviation.

This example demonstrates how to fit a linear regression model with unknown mean and standard deviation,
using a quadratic approximation (Laplace method). It also shows how to extract summary statistics, credible
intervals, and the variance-covariance matrix from the posterior distribution, as well as how to visualize
the joint posterior and the fitted line with uncertainty intervals.

Adapted from Rethinking Statistics 2nd edition, Chapter 4.4.
"""

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
from scipy import stats

from pymc_extras.inference import fit_laplace

# Code 4.37 — reading the data and subsetting to adults, then plotting height vs weight
d: pd.DataFrame = pd.read_csv("Howell1.csv", sep=";")
d2 = d[d["age"] >= 18]

# plot(d2$height ~ d2$weight)
fig, ax = plt.subplots()
ax.scatter(d2["weight"], d2["height"], s=10, alpha=0.6)
ax.set_xlabel("weight")
ax.set_ylabel("height")
plt.tight_layout()
plt.show()


### 4.4.1 The linear model strategy ###

# Code 4.38, 4.39 and 4.41 — simulating 100 lines from the prior distribution
xbar = d2["weight"].mean()
x = np.linspace(d2["weight"].min(), d2["weight"].max(), 200)
N = 100
rng = np.random.default_rng(42)
a       = rng.normal(178, 20, N)
b_norm  = rng.normal(0, 10, N)         # b ~ Normal(0, 10) — allows negative slopes
b_log   = rng.lognormal(0, 1, N)       # log(b) ~ Normal(0,1) i.e. b ~ Lognormal(0,1) — positive only

# Recreating Figure 4.5
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

for ax, b, title in [
    (ax1, b_norm, "b ~ dnorm(0, 10)"),
    (ax2, b_log,  "log(b) ~ dnorm(0, 1)"),
]:
    ax.axhline(0,   linestyle="--", linewidth=0.8, color="black")   # implausible floor
    ax.axhline(272, linestyle="-",  linewidth=0.5, color="black")   # tallest human ever
    ax.set_xlim(d2["weight"].min(), d2["weight"].max())
    ax.set_ylim(-100, 400)
    ax.set_xlabel("weight")
    ax.set_ylabel("height")
    ax.set_title(title)
    for i in range(N):
        ax.plot(x, a[i] + b[i] * (x - xbar), color="black", alpha=0.2, linewidth=0.5)

plt.tight_layout()
plt.show()


### 4.4.2 Finding the posterior distribution ###

# Code 4.42 — fit the model using Laplace approximation
xbar = d2["weight"].mean()
with pm.Model() as model_m43:
    a    = pm.Normal("a", mu=178, sigma=20)
    b    = pm.Lognormal("b", mu=0, sigma=1)
    sigma = pm.Uniform("sigma", lower=0, upper=50)
    mu = a + b * (d2["weight"].to_numpy() - xbar)
    pm.Normal("height", mu=mu, sigma=sigma, observed=d2["height"].to_numpy())
    idata_m43 = fit_laplace(draws=10_000)


### 4.4.3 Interpreting the posterior distribution ###

# Code 4.44 — summary of the posterior distribution
print(az.summary(idata_m43, var_names=["a", "b", "sigma"], ci_prob=0.89, round_to=2, kind="stats"))


# Code 4.45 — variance-covariance matrix from posterior samples
sample_a     = idata_m43.posterior["a"].to_numpy().ravel()
sample_b     = idata_m43.posterior["b"].to_numpy().ravel()
sample_sigma = idata_m43.posterior["sigma"].to_numpy().ravel()
cov_matrix   = np.cov(np.stack([sample_a, sample_b, sample_sigma]))
vcov_laplace = pd.DataFrame(cov_matrix, index=["a", "b", "σ"], columns=["a", "b", "σ"])
print("\nVariance-covariance matrix (Laplace, from posterior samples):")
print(vcov_laplace.to_string(float_format="{:.3f}".format))

# Code 4.46 — plotting the fitted line with data points, recreating Figure 4.6
a_map = sample_a.mean()
b_map = sample_b.mean()
x_line = np.linspace(d2["weight"].min(), d2["weight"].max(), 200)

fig, ax = plt.subplots()
ax.scatter(d2["weight"], d2["height"], color="#1e90ff", alpha=0.4, s=10)
ax.plot(x_line, a_map + b_map * (x_line - xbar), color="black", linewidth=1)
ax.set_xlabel("weight")
ax.set_ylabel("height")
ax.set_title("height ~ weight with MAP regression line")
plt.tight_layout()
plt.show()

# Code 4.47 — inspect the first samples of the posterior distribution
post = idata_m43.posterior.ds.stack(sample=("chain", "draw")).to_dataframe()[["a", "b", "sigma"]]
print(post.head())

# Code 4.48 — fit with samples removed from the dataset to see if the posterior changes
# Change N to see how the posterior changes with different sample sizes
N = 10
dN = d2.iloc[:N]
xbar_reduced = dN["weight"].mean()
with pm.Model() as model_mN:
    a    = pm.Normal("a", mu=178, sigma=20)
    b    = pm.Lognormal("b", mu=0, sigma=1)
    sigma = pm.Uniform("sigma", lower=0, upper=50)
    mu = a + b * (dN["weight"].to_numpy() - xbar_reduced)
    pm.Normal("height", mu=mu, sigma=sigma, observed=dN["height"].to_numpy())
    idata_mN = fit_laplace(draws=10_000)

# Code 4.49 — draw and plot 20 samples from the posterior
rng_mN = np.random.default_rng(42)
n_lines = 20
post_idx = rng_mN.choice(idata_mN.posterior.ds.sizes["draw"] * idata_mN.posterior.ds.sizes["chain"],
                          size=n_lines, replace=False)
post_a = idata_mN.posterior["a"].values.ravel()[post_idx]
post_b = idata_mN.posterior["b"].values.ravel()[post_idx]

x_full = np.linspace(dN["weight"].min(), dN["weight"].max(), 200)

fig, ax = plt.subplots()
ax.scatter(dN["weight"], dN["height"], color="#1e90ff", alpha=0.6, s=10)
ax.set_xlim(dN["weight"].min(), dN["weight"].max())
ax.set_ylim(dN["height"].min(), dN["height"].max())
ax.set_xlabel("weight")
ax.set_ylabel("height")
ax.set_title(f"N = {N}")
for i in range(n_lines):
    ax.plot(x_full, post_a[i] + post_b[i] * (x_full - xbar_reduced),
            color="black", alpha=0.3, linewidth=0.8)
plt.tight_layout()
plt.show()

# Code 4.50 — posterior predictive interval for weight = 50
mu_at_50 = sample_a + sample_b * (50 - xbar)

# Code 4.51 — density plot of posterior samples for mu_at_50
kde_x = np.linspace(mu_at_50.min(), mu_at_50.max(), 500)
fig, ax = plt.subplots()
ax.plot(kde_x, stats.gaussian_kde(mu_at_50)(kde_x), color="#1e90ff", linewidth=2)
ax.set_xlabel("μ | weight=50")
ax.set_ylabel("Density")
ax.set_title("dens(mu_at_50)")
plt.tight_layout()
plt.show()

# Code 4.52 — posterior predictive interval for mu_at_50
pi_mu50 = np.quantile(mu_at_50, [0.055, 0.945])
print(f"PI(mu_at_50, prob=0.89): [{pi_mu50[0]:.4f}, {pi_mu50[1]:.4f}]")

# Code 4.54 — calculating mu for a range of weights
weight_seq = np.arange(25, 71)                    # shape (46,)
mu_link = sample_a[:, np.newaxis] + sample_b[:, np.newaxis] * (weight_seq[np.newaxis, :] - xbar)

# Code 4.56 — calculating the mean and 89% PI for mu at each weight
mu_mean = mu_link.mean(axis=0)                    # shape (46,)
mu_pi   = np.quantile(mu_link, [0.055, 0.945], axis=0)  # shape (2, 46)

# Code 4.55 and 4.57 — recreating Figure 4.9
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Left: plot(height ~ weight, d2, type="n") + 100 posterior μ scatter points
ax1.set_xlabel("weight"); ax1.set_ylabel("height")
ax1.set_title("100 posterior μ lines (link equivalent)")
for i in range(100):
    ax1.scatter(weight_seq, mu_link[i, :], s=2, alpha=0.1, color="#1e90ff")

# Right: plot(height ~ weight, data=d2) + lines(weight.seq, mu.mean) + shade(mu.PI, weight.seq)
ax2.scatter(d2["weight"], d2["height"], color="#1e90ff", alpha=0.5, s=10)
ax2.plot(weight_seq, mu_mean, color="black", linewidth=1)
ax2.fill_between(weight_seq, mu_pi[0], mu_pi[1], alpha=0.3, color="black", label="89% PI for μ")
ax2.set_xlabel("weight"); ax2.set_ylabel("height")
ax2.set_title("MAP line + 89% PI shading")
ax2.legend()

plt.tight_layout()
plt.show()

# Code 4.59 — creating the 89 % prediction interval
#
# For each posterior draw, simulate an observed height at each weight by sampling
# from Normal(mu, sigma). This adds observation noise on top of parameter uncertainty,
# producing the posterior predictive distribution rather than just the posterior of mu.
#
# McElreath uses sim() to do this in R. The PyMC equivalent is pm.sample_posterior_predictive()
# PyMC provides this functionality natively. But, to predict at new inputs (weight_seq
# instead of the training data), the model must declare its input via pm.Data so
# that pm.set_data() can swap in new values at prediction time:
#
#   with model:
#       pm.set_data({"weight": weight_seq.astype(float)})
#       ppc_m43 = pm.sample_posterior_predictive(idata_m43)
#   sim_height_pymc = ppc_m43.posterior_predictive["height"].values.reshape(-1, len(weight_seq))
#   height_pi_pymc  = np.quantile(sim_height_pymc, [0.055, 0.945], axis=0)
#
# Note: this requires the model to use pm.Data("weight", ...) instead of a plain
# numpy array. The posterior samples (idata_m43) do not need to be re-fitted.
#
# For a simple linear model like this the manual approach below is mathematically
# identical and faster. pm.sample_posterior_predictive() becomes more valuable for
# complex models (hierarchical, GLMs, custom likelihoods) where reproducing the
# likelihood by hand would be error-prone.
#
rng_sim = np.random.default_rng(42)
mu_link_full = sample_a[:, np.newaxis] + sample_b[:, np.newaxis] * (weight_seq[np.newaxis, :] - xbar)
sim_height   = rng_sim.normal(mu_link_full, sample_sigma[:, np.newaxis])

# Code 4.60 — calculating the 89% prediction interval for height at each weight
height_pi = np.quantile(sim_height, [0.055, 0.945], axis=0)  # shape (2, 46)

# Code 4.61 — 89% HPDI of mu at each weight and recreating Figure 4.10
mu_hpdi = np.array([az.hdi(mu_link_full[:, i], prob=0.89) for i in range(len(weight_seq))]).T  # shape (2, 46)

fig, ax = plt.subplots()
ax.scatter(d2["weight"], d2["height"], color="#1e90ff", alpha=0.5, s=10)
ax.plot(weight_seq, mu_mean, color="black", linewidth=1, label="MAP line")
ax.fill_between(weight_seq, mu_hpdi[0], mu_hpdi[1], alpha=0.4, color="steelblue", label="89% HPDI for μ")
ax.fill_between(weight_seq, height_pi[0], height_pi[1], alpha=0.2, color="steelblue", label="89% PI for heights")
ax.set_xlabel("weight"); ax.set_ylabel("height")
ax.set_title("MAP line + 89% HPDI (μ) + 89% PI (heights)")
ax.legend()
plt.tight_layout()
plt.show()
