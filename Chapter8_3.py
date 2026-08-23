"""Bayesian models of tulip blooms as a function of water and shade, with and without
a water-shade interaction.

Adapted from Rethinking Statistics 2nd edition, Chapter 8.3.
"""

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm

from pymc_extras.inference import fit_laplace


### 8.3.1 A winter flower ###

# Code 8.19 — reading the data
d = pd.read_csv("tulips.csv", sep=";")
print(d.dtypes)
print(d.head())


### 8.3.2 The models ###

# Code 8.20 — center all predictors and response
d["blooms_std"]  = d["blooms"] / d["blooms"].max()
d["water_cent"]  = d["water"]  - d["water"].mean()
d["shade_cent"]  = d["shade"]  - d["shade"].mean()

# 8.21 — Prior probability outside of 0-1 range
a = np.random.default_rng(0).normal(0.5, 1, 10_000)
print(np.mean((a < 0) | (a > 1)))

# 8.22 — Prior probability outside of 0-1 range, with a narrower prior
a = np.random.default_rng(0).normal(0.5, 0.25, 10_000)
print(np.mean((a < 0) | (a > 1)))

# Code 8.23 — Model without interaction
with pm.Model() as model_m84:
    a     = pm.Normal("a",  mu=0.5, sigma=0.25)
    bw    = pm.Normal("bw", mu=0,   sigma=0.25)
    bs    = pm.Normal("bs", mu=0,   sigma=0.25)
    sigma = pm.Exponential("sigma", lam=1)
    mu = a + bw * d["water_cent"] + bs * d["shade_cent"]
    pm.Normal("blooms_std", mu=mu, sigma=sigma, observed=d["blooms_std"].to_numpy())
    idata_m84 = fit_laplace(draws=10_000)

# Code 8.24 — Model with interaction
with pm.Model() as model_m85:
    a     = pm.Normal("a",  mu=0.5, sigma=0.25)
    bw    = pm.Normal("bw", mu=0,   sigma=0.25)
    bs    = pm.Normal("bs", mu=0,   sigma=0.25)
    bws   = pm.Normal("bws", mu=0,  sigma=0.25)
    sigma = pm.Exponential("sigma", lam=1)
    mu = a + bw * d["water_cent"] + bs * d["shade_cent"] + bws * d["water_cent"] * d["shade_cent"]
    pm.Normal("blooms_std", mu=mu, sigma=sigma, observed=d["blooms_std"].to_numpy())
    idata_m85 = fit_laplace(draws=10_000)

# posterior summaries (not in the book)
print(az.summary(idata_m84, var_names=["a", "bw", "bs", "sigma"], ci_prob=0.89, round_to=2, kind="stats"))
print(az.summary(idata_m85, var_names=["a", "bw", "bs", "bws", "sigma"], ci_prob=0.89, round_to=2, kind="stats"))


### 8.3.3 Plotting posterior predictions ###

# Code 8.25 — 3-panel plot of posterior predictions for model m8.4
# The plot for model m8.5 is created as well, recreating Figure 8.7 in the book
water_seq = np.array([-1, 0, 1])
shade_levels = [-1, 0, 1]

post_m84 = idata_m84.posterior.ds.stack(sample=("chain", "draw"))
a84 = post_m84["a"].values
bw84 = post_m84["bw"].values
bs84 = post_m84["bs"].values

post_m85 = idata_m85.posterior.ds.stack(sample=("chain", "draw"))
a85 = post_m85["a"].values
bw85 = post_m85["bw"].values
bs85 = post_m85["bs"].values
bws85 = post_m85["bws"].values

fig, axes = plt.subplots(2, 3, figsize=(10, 7), sharey=True)

for col, s in enumerate(shade_levels):
    idx = d["shade_cent"] == s

    ax = axes[0, col]
    ax.scatter(d.loc[idx, "water_cent"], d.loc[idx, "blooms_std"], color="steelblue", s=20)
    for i in range(20):
        ax.plot(water_seq, a84[i] + bw84[i] * water_seq + bs84[i] * s,
                color="black", alpha=0.3)
    ax.set_xlim(-1.1, 1.1); ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("water"); ax.set_title(f"shade = {s}")
    if col == 0:
        ax.set_ylabel("blooms (m8.4)")

    ax = axes[1, col]
    ax.scatter(d.loc[idx, "water_cent"], d.loc[idx, "blooms_std"], color="steelblue", s=20)
    for i in range(20):
        ax.plot(water_seq, a85[i] + bw85[i] * water_seq + bs85[i] * s + bws85[i] * water_seq * s,
                color="black", alpha=0.3)
    ax.set_xlim(-1.1, 1.1); ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("water"); ax.set_title(f"shade = {s}")
    if col == 0:
        ax.set_ylabel("blooms (m8.5)")

plt.tight_layout()
plt.show()

# Code 8.36 — plot the prior predictions for models m84. and m8.5, recreating Figure 8.8 in the book
# The plotting itself is not included in the book
with model_m84:
    prior_m84 = pm.sample_prior_predictive(draws=20, random_seed=7)
with model_m85:
    prior_m85 = pm.sample_prior_predictive(draws=20, random_seed=7)

pd84 = prior_m84.prior.ds.stack(sample=("chain", "draw"))
a84p  = pd84["a"].values;  bw84p = pd84["bw"].values;  bs84p = pd84["bs"].values

pd85 = prior_m85.prior.ds.stack(sample=("chain", "draw"))
a85p  = pd85["a"].values;  bw85p = pd85["bw"].values
bs85p = pd85["bs"].values; bws85p = pd85["bws"].values

fig, axes = plt.subplots(2, 3, figsize=(10, 7), sharey=True)

for col, s in enumerate(shade_levels):
    for row, (ap, bwp, bsp, bwsp, label) in enumerate([
        (a84p, bw84p, bs84p, None,   "m8.4"),
        (a85p, bw85p, bs85p, bws85p, "m8.5"),
    ]):
        ax = axes[row, col]
        ax.axhline(0, linestyle="--", color="black", linewidth=0.8)
        ax.axhline(1, linestyle="--", color="black", linewidth=0.8)
        for i in range(20):
            mu_i = ap[i] + bwp[i] * water_seq + bsp[i] * s
            if bwsp is not None:
                mu_i += bwsp[i] * water_seq * s
            ax.plot(water_seq, mu_i, color="black", alpha=0.3)
        ax.set_xlim(-1.1, 1.1)
        ax.set_xlabel("water"); ax.set_title(f"shade = {s}")
        if col == 0:
            ax.set_ylabel(f"blooms ({label})")

plt.tight_layout()
plt.show()
