"""Bayesian modeling of temperature as a function of year using Laplace approximation. The
model uses a spline basis to capture non-linear relationships between year and temperature.
The code also computes and visualizes the 97% HPDI for the predicted temperatures of individuals
(posterior predictive interval).

Adapted from Rethinking Statistics 2nd edition, Chapter 4.5.2.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm

from pymc_extras.inference import fit_laplace
from scipy.interpolate import BSpline


d: pd.DataFrame = pd.read_csv("cherry_blossoms.csv", sep=";")

# Filter out rows with missing temp values, then define knots for the spline basis functions
d2 = d[d["temp"].notna()]
num_knots = 15
knot_list = np.quantile(d2["year"], np.linspace(0, 1, num_knots))

# Construct basis functions for a 3-degree (cubic) spline
interior_knots = knot_list[1:-1]  # remove first and last (13 interior knots)
t = np.concatenate([[knot_list[0]] * 4, interior_knots, [knot_list[-1]] * 4])
B = BSpline.design_matrix(d2["year"].to_numpy(), t, k=3).toarray()

# Fit the model using Laplace approximation
with pm.Model() as model:
    a     = pm.Normal("a", mu=6, sigma=10)
    w     = pm.Normal("w", mu=0, sigma=1, shape=B.shape[1])
    sigma = pm.Exponential("sigma", lam=1)
    mu = a + B @ w
    pm.Normal("T", mu=mu, sigma=sigma, observed=d2["temp"].to_numpy())
    idata = fit_laplace(draws=10_000)

w_mean = idata.posterior["w"].mean(dim=["chain", "draw"]).to_numpy()

# Compute 97% posterior interval for mu at each observation (equivalent to link + PI in R)
sample_a = idata.posterior["a"].to_numpy().ravel()                         # (n_samples,)
sample_w = idata.posterior["w"].to_numpy().reshape(-1, B.shape[1])         # (n_samples, 17)
mu_samples = sample_a[:, None] + (sample_w @ B.T)                          # (n_samples, n_obs)
mu_pi = np.percentile(mu_samples, [1.5, 98.5], axis=0)                     # 97% PI

# Plot: basis functions (top), posterior weighted basis functions (middle), data + 97% PI (bottom)
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
ax1.set_xlim(d2["year"].min(), d2["year"].max())
ax1.set_ylim(0, 1)
ax1.set_ylabel("basis value")
for i in range(B.shape[1]):
    ax1.plot(d2["year"], B[:, i])

ax2.set_xlim(d2["year"].min(), d2["year"].max())
ax2.set_ylim(-2, 2)
ax2.set_ylabel("basis * weight")
for i in range(B.shape[1]):
    ax2.plot(d2["year"], w_mean[i] * B[:, i])

ax3.scatter(d2["year"], d2["temp"], alpha=0.3, s=8, color="steelblue")
ax3.fill_between(d2["year"], mu_pi[0], mu_pi[1], color="black", alpha=0.5)
ax3.set_xlabel("year")
ax3.set_ylabel("temp")

plt.tight_layout()
plt.show()