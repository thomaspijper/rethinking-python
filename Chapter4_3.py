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


d: pd.DataFrame = pd.read_csv("Howell1.csv", sep=";")
adult_heights = d[d["age"] >= 18]["height"]



### Grid approximation approach to compute posterior over (mu, sigma) on a grid ###

# Define a grid of (mu, sigma) values to evaluate the posterior density
mu_list    = np.linspace(150, 160, 100)
sigma_list = np.linspace(7, 9, 100)

# grid of all (mu, sigma) combinations — shape (10000, 2)
mu_grid, sigma_grid = np.meshgrid(mu_list, sigma_list, indexing="ij")

# log-likelihood: sum of log-normal pdf over all observed heights, for each grid point
# adult_heights shape: (n,) → broadcast against (100, 100, 1)
LL = stats.norm.logpdf(
    adult_heights.to_numpy(),                        # shape (n,)
    loc=mu_grid[:, :, np.newaxis],               # shape (100, 100, 1)
    scale=sigma_grid[:, :, np.newaxis],          # shape (100, 100, 1)
).sum(axis=2)                                    # shape (100, 100)

log_prior_mu    =    stats.norm.logpdf(mu_grid,    loc=178, scale=20)
log_prior_sigma = stats.uniform.logpdf(sigma_grid,   loc=0, scale=50)

log_prod = LL + log_prior_mu + log_prior_sigma
prob     = np.exp(log_prod - log_prod.max())   # normalise for numerical stability
prob_norm = prob / prob.sum()                   # normalise to sum to 1

# Marginal posteriors by summing over the other parameter
prob_mu    = prob_norm.sum(axis=1)              # shape (100,) — marginal over sigma
prob_sigma = prob_norm.sum(axis=0)              # shape (100,) — marginal over mu

# Weighted mean and std from the marginal grids
mean_mu    = np.average(mu_list,    weights=prob_mu)
mean_sigma = np.average(sigma_list, weights=prob_sigma)
std_mu     = np.sqrt(np.average((mu_list    - mean_mu   )**2, weights=prob_mu))
std_sigma  = np.sqrt(np.average((sigma_list - mean_sigma)**2, weights=prob_sigma))

# Compute 89% HPDI from the marginal samples (using the grid as a discrete distribution)
rng = np.random.default_rng(42)
mu_samples    = rng.choice(mu_list,    size=10_000, p=prob_mu    / prob_mu.sum())
sigma_samples = rng.choice(sigma_list, size=10_000, p=prob_sigma / prob_sigma.sum())
mu_lo,    mu_hi    = az.hdi(mu_samples,    hdi_prob=0.89)
sigma_lo, sigma_hi = az.hdi(sigma_samples, hdi_prob=0.89)

# Variance-covariance matrix from the joint grid (exact, analytical)
# Note: off-diagonal correlation = covariance / (std_mu * std_sigma)
cov_mu_sigma = np.sum(prob_norm * (mu_grid - mean_mu) * (sigma_grid - mean_sigma))
vcov_grid = pd.DataFrame(
    [[std_mu**2, cov_mu_sigma],
     [cov_mu_sigma, std_sigma**2]],
    index=["μ", "σ"], columns=["μ", "σ"]
)

print("--- Grid approximation ---")
print(f"μ:  mean={mean_mu:.2f}, std={std_mu:.2f}, 89% HPDI=[{mu_lo:.2f}, {mu_hi:.2f}]")
print(f"σ:  mean={mean_sigma:.2f}, std={std_sigma:.2f}, 89% HPDI=[{sigma_lo:.2f}, {sigma_hi:.2f}]")
print("Variance-covariance matrix (grid):")
print(vcov_grid.to_string(float_format="{:.8f}".format))

# Direct plot of the posterior density — no sampling needed
fig, ax = plt.subplots()
cf = ax.contourf(mu_list, sigma_list, prob.T, levels=20, cmap="viridis")
fig.colorbar(cf, ax=ax, label="Relative posterior density")
ax.set_xlabel("μ"); ax.set_ylabel("σ")
ax.set_title("Posterior distribution (grid approximation)")
plt.tight_layout()
plt.show()



### Quadratic approximation approach to compute posterior over (mu, sigma) on a grid ###

with pm.Model() as model:
    mu    = pm.Normal("mu",    mu=178, sigma=20)
    sigma = pm.Uniform("sigma", lower=0, upper=50)
    pm.Normal("height", mu=mu, sigma=sigma, observed=adult_heights.to_numpy())
    idata = fit_laplace(draws=10_000)

print(az.summary(idata, var_names=["mu", "sigma"], hdi_prob=0.89, round_to=2, kind="stats"))

# Variance-covariance matrix from posterior samples (constrained/original space)
sample_mu    = idata.posterior["mu"].to_numpy().ravel()
sample_sigma = idata.posterior["sigma"].to_numpy().ravel()
cov_matrix   = np.cov(sample_mu, sample_sigma)
vcov_laplace = pd.DataFrame(cov_matrix, index=["μ", "σ"], columns=["μ", "σ"])
print("Variance-covariance matrix (Laplace, from posterior samples):")
print(vcov_laplace.to_string(float_format="{:.6f}".format))

# Summary statistics from the 10,000 posterior samples
# sample_mu and sample_sigma are 1D vectors of paired (μ, σ) draws
mu_lo89,    mu_hi89    = np.percentile(sample_mu,    [5.5, 94.5])
sigma_lo89, sigma_hi89 = np.percentile(sample_sigma, [5.5, 94.5])

print("\n--- Quadratic approximation (calculated from samples of the posterior) ---")
print(f"μ:  mean={sample_mu.mean():.2f}, std={sample_mu.std():.2f}, 89% PI=[{mu_lo89:.2f}, {mu_hi89:.2f}]")
print(f"σ:  mean={sample_sigma.mean():.2f}, std={sample_sigma.std():.2f}, 89% PI=[{sigma_lo89:.2f}, {sigma_hi89:.2f}]")

# 2D joint posterior of mu and sigma
az.plot_pair(idata, var_names=["mu", "sigma"], kind="kde",
             kde_kwargs={"fill_last": True}, marginals=True)
plt.show()
