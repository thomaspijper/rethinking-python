"""Bayesian models showing an example of post-treatment bias.

Adapted from Rethinking Statistics 2nd edition, Chapter 6.2.
"""

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm

from pymc_extras.inference import fit_laplace

rng = np.random.default_rng(71)

N = 100  # number of plants

# simulate initial heights
h0 = rng.normal(10, 2, N)

# assign treatments and simulate fungus and growth
treatment = np.repeat([0, 1], N // 2)
fungus = rng.binomial(1, 0.5 - treatment * 0.4, N)
h1 = h0 + rng.normal(5 - 3 * fungus, 1, N)

# compose a clean dataframe
d = pd.DataFrame({"h0": h0, "h1": h1, "treatment": treatment, "fungus": fungus})

# precis(d) equivalent
print(d.describe().round(2))


### 6.2.1 A prior is born ###

# Simulated lognormal proportions
sim_p = rng.lognormal(0, 0.25, 10_000)
print(pd.DataFrame({"sim_p": sim_p}).describe().round(2))

# Fit the model using Laplace approximation, to estimate the growth
with pm.Model() as model:
    p = pm.LogNormal("p", mu=0, sigma=0.25)
    sigma = pm.Exponential("sigma", lam=1)
    # Important: p must be the left operand. Writing d["h0"] * p causes pandas'
    # __mul__ to be called, which fails on a PyMC variable instead of deferring
    # to PyMC's __rmul__. With p on the left, PyMC's __mul__ is called and
    # correctly handles the pandas Series.
    mu = p * d["h0"]
    pm.Normal("h1", mu=mu, sigma=sigma, observed=d["h1"].to_numpy())
    idata_m66 = fit_laplace(draws=10_000)

print(az.summary(idata_m66, var_names=["p", "sigma"], hdi_prob=0.89, round_to=2, kind="stats"))

# Fit the model using Laplace approximation, to estimate the growth with treatment and fungus effects
with pm.Model() as model:
    a = pm.LogNormal("a", mu=0, sigma=0.25)
    bT = pm.Normal("bT", mu=0, sigma=0.5)
    bF = pm.Normal("bF", mu=0, sigma=0.5)
    p = a + bT*d["treatment"] + bF*d["fungus"]
    sigma = pm.Exponential("sigma", lam=1)
    # Important: p must be the left operand. Writing d["h0"] * p causes pandas'
    # __mul__ to be called, which fails on a PyMC variable instead of deferring
    # to PyMC's __rmul__. With p on the left, PyMC's __mul__ is called and
    # correctly handles the pandas Series.
    mu = p * d["h0"]
    pm.Normal("h1", mu=mu, sigma=sigma, observed=d["h1"].to_numpy())
    idata_m67 = fit_laplace(draws=10_000)

print(az.summary(idata_m67, var_names=["a", "bT", "bF", "sigma"], hdi_prob=0.89, round_to=2, kind="stats"))

# Fit the model using Laplace approximation, to estimate the growth with treatment included and fungus excluded
with pm.Model() as model:
    a = pm.LogNormal("a", mu=0, sigma=0.25)
    bT = pm.Normal("bT", mu=0, sigma=0.5)
    p = a + bT*d["treatment"]
    sigma = pm.Exponential("sigma", lam=1)
    # Important: p must be the left operand. Writing d["h0"] * p causes pandas'
    # __mul__ to be called, which fails on a PyMC variable instead of deferring
    # to PyMC's __rmul__. With p on the left, PyMC's __mul__ is called and
    # correctly handles the pandas Series.
    mu = p * d["h0"]
    pm.Normal("h1", mu=mu, sigma=sigma, observed=d["h1"].to_numpy())
    idata_m68 = fit_laplace(draws=10_000)

print(az.summary(idata_m68, var_names=["a", "bT", "sigma"], hdi_prob=0.89, round_to=2, kind="stats"))


# Repeating the steps above, but with moisture M included

rng2 = np.random.default_rng(71)

N2 = 1000
h0_2 = rng2.normal(10, 2, N2)
treatment2 = np.repeat([0, 1], N2 // 2)
M = rng2.binomial(1, 0.5, N2) # Bernoulli is just Binomial with size=1. Default p value for R's rbern() is p=0.5
fungus2 = rng2.binomial(1, 0.5 - treatment2 * 0.4 + 0.4 * M, N2)
h1_2 = h0_2 + rng2.normal(5 + 3 * M, 1, N2)

d2 = pd.DataFrame({"h0": h0_2, "h1": h1_2, "treatment": treatment2, "fungus": fungus2})

print(d2.describe().round(2))

# Fit the model using Laplace approximation, to estimate the growth with treatment and fungus effects
with pm.Model() as model:
    a = pm.LogNormal("a", mu=0, sigma=0.25)
    bT = pm.Normal("bT", mu=0, sigma=0.5)
    bF = pm.Normal("bF", mu=0, sigma=0.5)
    p = a + bT*d2["treatment"] + bF*d2["fungus"]
    sigma = pm.Exponential("sigma", lam=1)
    mu = p * d2["h0"]
    pm.Normal("h1", mu=mu, sigma=sigma, observed=d2["h1"].to_numpy())
    idata_m67_m = fit_laplace(draws=10_000)

print(az.summary(idata_m67_m, var_names=["a", "bT", "bF", "sigma"], hdi_prob=0.89, round_to=2, kind="stats"))

# Fit the model using Laplace approximation, to estimate the growth with treatment included and fungus excluded
with pm.Model() as model:
    a = pm.LogNormal("a", mu=0, sigma=0.25)
    bT = pm.Normal("bT", mu=0, sigma=0.5)
    p = a + bT*d2["treatment"]
    sigma = pm.Exponential("sigma", lam=1)
    mu = p * d2["h0"]
    pm.Normal("h1", mu=mu, sigma=sigma, observed=d2["h1"].to_numpy())
    idata_m68_m = fit_laplace(draws=10_000)

print(az.summary(idata_m68_m, var_names=["a", "bT", "sigma"], hdi_prob=0.89, round_to=2, kind="stats"))
