"""Bayesian inference example: Grid approximation and quadratic approximation (Laplace) for a normal model.
This example demonstrates how to compute the posterior distribution for a normal model with unknown mean and
standard deviation, using both a grid approximation and a quadratic approximation (Laplace method). It also
shows how to extract summary statistics, credible intervals, and the variance-covariance matrix from the
posterior distribution.

Adapted from Rethinking Statistics 2nd edition, Chapter 4.3."""

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
from scipy import stats

from pymc_extras.inference import fit_laplace


### 4.3.1 The data ###

# Code 4.7 — read data
d: pd.DataFrame = pd.read_csv("Howell1.csv", sep=";")

# Code 4.8 — show the data
print(d)

# Code 4.9 — summarize the data (specify custom 89% interval boundaries)
print(d.describe(percentiles=[0.055, 0.945]))

# Code 4.11 — subset to adults (age >= 18)
d2 = d[d["age"] >= 18]["height"]


### 4.3.2 The model ###

# Code 4.12, 4.13 and, 4.14
# Plotting of the prior distributions and prior predictive simulation is done below
rng_prior = np.random.default_rng(42)
sample_mu = rng_prior.normal(178, 20,   size=10_000)    # mu    ~ Normal(178, 20)
sample_sigma = rng_prior.uniform(0,   50,  size=10_000)    # sigma ~ Uniform(0, 50)
prior_h = rng_prior.normal(sample_mu, sample_sigma)   # h     ~ Normal(mu, sigma)

# Code 4.14 — wider prior: sigma same, but mu ~ Normal(178, 100)
sample_mu_wide = rng_prior.normal(178, 100, size=10_000)
prior_h_wide = rng_prior.normal(sample_mu_wide, sample_sigma)

# Recreation of Figure 4.3
fig, axes = plt.subplots(2, 2, figsize=(10, 8))

# Top-left: mu ~ Normal(178, 20)
x_mu = np.linspace(100, 250, 500)
axes[0, 0].plot(x_mu, stats.norm.pdf(x_mu, 178, 20), color="#1e90ff")
axes[0, 0].set_xlabel("μ")
axes[0, 0].set_ylabel("Density")
axes[0, 0].set_title("μ ~ dnorm(178, 20)")

# Top-right: sigma ~ Uniform(0, 50)
x_sigma = np.linspace(-10, 60, 500)
axes[0, 1].plot(x_sigma, stats.uniform.pdf(x_sigma, 0, 50), color="#1e90ff")
axes[0, 1].set_xlabel("σ")
axes[0, 1].set_ylabel("Density")
axes[0, 1].set_title("σ ~ dunif(0, 50)")

# Bottom-left: prior predictive h ~ Normal(mu, sigma) with mu ~ Normal(178, 20)
kde_x = np.linspace(prior_h.mean() - 3 * prior_h.std(),
                    prior_h.mean() + 3 * prior_h.std(), 500)
axes[1, 0].plot(kde_x, stats.gaussian_kde(prior_h)(kde_x), color="#1e90ff")
axes[1, 0].set_xlabel("height")
axes[1, 0].set_ylabel("Density")
axes[1, 0].set_title("h ~ dnorm(μ, σ),  μ ~ dnorm(178, 20)")

# Bottom-right: wider prior mu ~ Normal(178, 100)
kde_x_wide = np.linspace(prior_h_wide.mean() - 3 * prior_h_wide.std(),
                         prior_h_wide.mean() + 3 * prior_h_wide.std(), 500)
axes[1, 1].plot(kde_x_wide, stats.gaussian_kde(prior_h_wide)(kde_x_wide), color="#e05c5c")
axes[1, 1].set_xlabel("height")
axes[1, 1].set_ylabel("Density")
axes[1, 1].set_title("h ~ dnorm(μ, σ),  μ ~ dnorm(178, 100)")

plt.suptitle("Prior distributions and prior predictive simulation")
plt.tight_layout()
plt.show()


### 4.3.3 Grid approximation of the posterior distribution ###

# Code 4.16 — the grid approximation
mu_list = np.linspace(150, 160, 100)
sigma_list = np.linspace(7, 9, 100)
mu_grid, sigma_grid = np.meshgrid(mu_list, sigma_list, indexing="ij")
LL = stats.norm.logpdf(
    d2.to_numpy(),
    loc=mu_grid[:, :, np.newaxis],               # shape (100, 100, 1)
    scale=sigma_grid[:, :, np.newaxis],          # shape (100, 100, 1)
).sum(axis=2)                                    # shape (100, 100)

log_prior_mu = stats.norm.logpdf(mu_grid, loc=178, scale=20)
log_prior_sigma = stats.uniform.logpdf(sigma_grid, loc=0, scale=50)
log_prod = LL + log_prior_mu + log_prior_sigma

prob = np.exp(log_prod - log_prod.max())   # normalise for numerical stability

# Code 4.17 — contour lines only
fig, ax = plt.subplots()
ax.contour(mu_list, sigma_list, prob.T, levels=10, cmap="viridis")
ax.set_xlabel("μ"); ax.set_ylabel("σ")
ax.set_title("contour_xyz: posterior contours")
plt.tight_layout()
plt.show()

# Code 4.18 — filled colour image
fig, ax = plt.subplots()
ax.pcolormesh(mu_list, sigma_list, prob.T, cmap="viridis", shading="auto")
ax.set_xlabel("μ"); ax.set_ylabel("σ")
ax.set_title("image_xyz: posterior heatmap")
plt.tight_layout()
plt.show()


### 4.3.4 Sampling from the posterior ###

# Code 4.19 — sample from the posterior distribution
prob_norm = prob / prob.sum()                   # normalise to sum to 1
flat_prob = prob_norm.ravel()                   # shape (10000,)
rng_samp = np.random.default_rng(42)
sample_rows = rng_samp.choice(len(flat_prob), size=10_000, replace=True, p=flat_prob / flat_prob.sum())
sample_mu_g = mu_grid.ravel()[sample_rows]
sample_sig_g = sigma_grid.ravel()[sample_rows]

# Code 4.20 — scatter plot of posterior samples
fig, ax = plt.subplots()
ax.scatter(sample_mu_g, sample_sig_g, s=2, alpha=0.1, color="#1e90ff")
ax.set_xlabel("μ")
ax.set_ylabel("σ")
ax.set_title("Samples from posterior (grid)")
plt.tight_layout()
plt.show()

# Code 4.21 — marginal posterior densities from samples
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
kde_x_mu  = np.linspace(sample_mu_g.min(),  sample_mu_g.max(),  500)
kde_x_sig = np.linspace(sample_sig_g.min(), sample_sig_g.max(), 500)
ax1.plot(kde_x_mu,  stats.gaussian_kde(sample_mu_g)(kde_x_mu),   color="#1e90ff")
ax2.plot(kde_x_sig, stats.gaussian_kde(sample_sig_g)(kde_x_sig), color="#1e90ff")
ax1.set_xlabel("μ");  ax1.set_ylabel("Density"); ax1.set_title("dens(sample.mu)")
ax2.set_xlabel("σ");  ax2.set_ylabel("Density"); ax2.set_title("dens(sample.sigma)")
plt.tight_layout()
plt.show()

# Code 4.22 — 89% posterior compatibility intervals
pi_mu  = np.quantile(sample_mu_g,  [0.055, 0.945])
pi_sig = np.quantile(sample_sig_g, [0.055, 0.945])
print(f"PI(sample.mu):    [{pi_mu[0]:.4f}, {pi_mu[1]:.4f}]")
print(f"PI(sample.sigma): [{pi_sig[0]:.4f}, {pi_sig[1]:.4f}]")

# Code 4.23 — draw 20 data points from the original dataset
rng = np.random.default_rng(42)
d3 = rng.choice(d2.to_numpy(), size=20, replace=False)

# Code 4.24 — grid approximation with reduced data
mu_list2    = np.linspace(150, 170, 200)
sigma_list2 = np.linspace(4, 20, 200)
mu_grid2, sigma_grid2 = np.meshgrid(mu_list2, sigma_list2, indexing="ij")

LL2 = stats.norm.logpdf(
    d3,
    loc=mu_grid2[:, :, np.newaxis],              # shape (200, 200, 1)
    scale=sigma_grid2[:, :, np.newaxis],         # shape (200, 200, 1)
).sum(axis=2)                                    # shape (200, 200)
log_prior_mu2    =    stats.norm.logpdf(mu_grid2,    loc=178, scale=20)
log_prior_sigma2 = stats.uniform.logpdf(sigma_grid2,   loc=0, scale=50)

log_prod2 = LL2 + log_prior_mu2 + log_prior_sigma2
prob2 = np.exp(log_prod2 - log_prod2.max())

prob_norm2 = prob2 / prob2.sum()                   # normalise to sum to 1
flat_prob2 = prob_norm2.ravel()                   # shape (10000,) — row-major (mu varies fastest with indexing="ij")
rng_samp = np.random.default_rng(42)
sample_rows2 = rng_samp.choice(len(flat_prob2), size=10_000, replace=True, p=flat_prob2 / flat_prob2.sum())
sample_mu_g2 = mu_grid2.ravel()[sample_rows2]
sample_sig_g2 = sigma_grid2.ravel()[sample_rows2]

fig, ax = plt.subplots()
ax.scatter(sample_mu_g2, sample_sig_g2, s=2, alpha=0.1, color="#1e90ff")
ax.set_xlabel("μ")
ax.set_ylabel("σ")
ax.set_title("Samples from posterior with reduced data (grid)")
plt.tight_layout()
plt.show()

# Code 4.25 — density plot of posterior samples for sigma
kde_x_sig2 = np.linspace(sample_sig_g2.min(), sample_sig_g2.max(), 500)
mu_s2, sigma_s2 = sample_sig_g2.mean(), sample_sig_g2.std()

fig, ax = plt.subplots()
ax.plot(kde_x_sig2, stats.gaussian_kde(sample_sig_g2)(kde_x_sig2), color="#1e90ff", label="density(sample2.sigma)")
ax.plot(kde_x_sig2, stats.norm.pdf(kde_x_sig2, mu_s2, sigma_s2),
        color="black", linestyle="--", label="Normal fit")
ax.set_xlabel("σ")
ax.set_ylabel("Density")
ax.set_title("dens(sample2.sigma, norm.comp=TRUE)")
ax.legend()
plt.tight_layout()
plt.show()


### 4.3.5 Finding the posterior distribution with quap ###

# Code 4.27 and 4.28 — fit the model using Laplace approximation
# Quadratic/Laplace approximation is not available in PyMC, but it is in the pymc-extras module
with pm.Model() as model_m41:
    mu    = pm.Normal("mu",    mu=178, sigma=20)
    sigma = pm.Uniform("sigma", lower=0, upper=50)
    pm.Normal("height", mu=mu, sigma=sigma, observed=d2.to_numpy())
    idata_m41 = fit_laplace(draws=10_000)

# Code 4.29 — summary of the posterior distribution
print(az.summary(idata_m41, var_names=["mu", "sigma"], ci_prob=0.89, round_to=2, kind="stats"))

# Code 4.31 — the same model with a much narrower prior on mu, to show how the prior affects the posterior
with pm.Model() as model_m42:
    mu    = pm.Normal("mu",    mu=178, sigma=0.1)
    sigma = pm.Uniform("sigma", lower=0, upper=50)
    pm.Normal("height", mu=mu, sigma=sigma, observed=d2.to_numpy())
    idata_m42 = fit_laplace(draws=10_000)

print(az.summary(idata_m42, var_names=["mu", "sigma"], ci_prob=0.89, round_to=2, kind="stats"))


### 4.3.6 Sampling from quap ###

# Code 4.32 — variance-covariance matrix from posterior samples (constrained/original space)
#
# .stack() flattens the chain and draw dimensions into a single sample dimension while keeping
# the data an xarray Dataset. I prefer this over .numpy().ravel() (which merely gives an array),
# because it keeps the context of the data intact. This has a few uses, including:
#   1. It allows you to easily convert the samples into a DataFrame with .to_dataframe()
#   2. vector/matrix parameters are retained, allowing for transposing with .T and other operations
#   3. for categorical models where parameters have named dims, stacked Dataset lets you slice by name
#
# If downstream operations rely on the samples being a numpy.ndarray, you can always convert to that with .values.
#
# BTW, Instead of as a bulk operation, it is also possible to extract the samples for each variable
# directly / separately:
#     sample_mu = idata_m41.posterior["mu"].stack(sample=("chain", "draw"))
#     sample_sigma = idata_m41.posterior["sigma"].stack(sample=("chain", "draw"))
#
post_m41 = idata_m41.posterior.ds.stack(sample=("chain", "draw"))
sample_mu = post_m41["mu"].values
sample_sigma = post_m41["sigma"].values
cov_matrix = np.cov(sample_mu, sample_sigma)
vcov_laplace = pd.DataFrame(cov_matrix, index=["μ", "σ"], columns=["μ", "σ"])
print("\nVariance-covariance matrix (Laplace, from posterior samples):")
print(vcov_laplace.to_string(float_format="{:.10f}".format))

# Code 4.33 — variances on the diagonal
print("\ndiag(vcov(m4.1)) — variances:")
print(np.diag(vcov_laplace.values))

# Code 4.33 continued — correlation matrix (diagonal normalized to 1)
std = np.sqrt(np.diag(vcov_laplace.values))
corr = vcov_laplace.values / np.outer(std, std)
corr_df = pd.DataFrame(corr, index=vcov_laplace.index, columns=vcov_laplace.columns)
print("\ncov2cor(vcov(m4.1)) — correlation matrix:")
print(corr_df.to_string(float_format="{:.10f}".format))
print("\n")

# Code 4.34 — getting the samples of the model into a DataFrame
post = idata_m41.posterior.ds.stack(sample=("chain", "draw")).to_dataframe()[["mu", "sigma"]]

# Code 4.35 — summary of the posterior samples
print(post.describe(percentiles=[0.055, 0.945]))
