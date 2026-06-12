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

from pymc_extras.inference import fit_laplace

d: pd.DataFrame = pd.read_csv("Howell1.csv", sep=";")
adult_d = d[d["age"] >= 18]

# # Prior predictive simulation: what lines does the model imply before seeing data?
# # height ~ Normal(a + b*(weight - xbar), sigma)
# # a ~ Normal(178, 20),  b ~ Normal(0, 10)
# np.random.seed(42)
# xbar = adult_d["weight"].mean()
# x = np.linspace(adult_d["weight"].min(), adult_d["weight"].max(), 200)
# N = 100
# rng = np.random.default_rng()
# a = rng.normal(178, 20, N)
# b = rng.lognormal(0, 1, N) # positive slope only, as we expect height to increase with weight
# fig, ax = plt.subplots()
# ax.axhline(0,   linestyle="--", linewidth=0.8, color="black")   # implausible floor
# ax.axhline(272, linestyle="-",  linewidth=0.5, color="black")   # tallest human ever
# ax.set_xlim(adult_d["weight"].min(), adult_d["weight"].max())
# ax.set_ylim(-100, 400)
# ax.set_xlabel("weight")
# ax.set_ylabel("height")
# ax.set_title("b ~ Lognormal(0, 1)")
# for i in range(N):
#     ax.plot(x, a[i] + b[i] * (x - xbar), color="black", alpha=0.2, linewidth=0.5)
# plt.tight_layout()
# plt.show()

# Fit the model using Laplace approximation
xbar = adult_d["weight"].mean()
with pm.Model() as model:
    a    = pm.Normal("a", mu=178, sigma=20)
    b    = pm.Lognormal("b", mu=0, sigma=1)
    sigma = pm.Uniform("sigma", lower=0, upper=50)
    mu = a + b * (adult_d["weight"].to_numpy() - xbar)
    pm.Normal("height", mu=mu, sigma=sigma, observed=adult_d["height"].to_numpy())
    idata = fit_laplace(draws=10_000)

print(az.summary(idata, var_names=["a", "b", "sigma"], hdi_prob=0.89, round_to=2, kind="stats"))

# Variance-covariance matrix from posterior samples (constrained/original space)
sample_a     = idata.posterior["a"].to_numpy().ravel()
sample_b     = idata.posterior["b"].to_numpy().ravel()
sample_sigma = idata.posterior["sigma"].to_numpy().ravel()
cov_matrix   = np.cov(np.stack([sample_a, sample_b, sample_sigma]))
vcov_laplace = pd.DataFrame(cov_matrix, index=["a", "b", "σ"], columns=["a", "b", "σ"])
print("Variance-covariance matrix (Laplace, from posterior samples):")
print(vcov_laplace.to_string(float_format="{:.6f}".format))

# 2D joint posterior of mu and sigma
az.plot_pair(idata, var_names=["a", "b", "sigma"], kind="kde",
             kde_kwargs={"fill_last": True}, marginals=True)
plt.show()

# For each integer weight compute:
#   1. 89% HPDI of mu (mean line) — parameter uncertainty only
#   2. 89% HPDI of individual heights — includes sigma (posterior predictive)
weight_seq = np.arange(int(adult_d["weight"].min()), int(adult_d["weight"].max()) + 1)
mu_lo    = np.empty(len(weight_seq))
mu_hi    = np.empty(len(weight_seq))
hpdi_lo  = np.empty(len(weight_seq))
hpdi_hi  = np.empty(len(weight_seq))
rng = np.random.default_rng()
for i, w in enumerate(weight_seq):
    mu_samples     = sample_a + sample_b * (w - xbar)
    height_samples = rng.normal(mu_samples, sample_sigma)
    mu_lo[i], mu_hi[i]     = az.hdi(mu_samples,     hdi_prob=0.89)
    hpdi_lo[i], hpdi_hi[i] = az.hdi(height_samples, hdi_prob=0.89)

# Plotting the data and the 89% HPDI for the predicted heights at each weight
fig, ax = plt.subplots()
ax.scatter(adult_d["weight"], adult_d["height"], alpha=0.4, s=10)
ax.set_xlabel("Weight")
ax.set_ylabel("Height")
ax.set_title("Height vs Weight (Howell1.csv)")
ax.fill_between(weight_seq, hpdi_lo, hpdi_hi, alpha=0.2, color="steelblue", label="89% HPDI for height of individuals (PI)")
ax.fill_between(weight_seq, mu_lo,   mu_hi,   alpha=0.4, color="steelblue", label="89% HPDI for fitted line")
x = np.linspace(adult_d["weight"].min(), adult_d["weight"].max(), 200)
ax.plot(x, idata.posterior["a"].mean().item() + idata.posterior["b"].mean().item() * (x - xbar), color="black", alpha=0.8, linewidth=1, label="Posterior mean")
ax.legend()
plt.tight_layout()
plt.show()
