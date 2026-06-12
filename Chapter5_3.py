"""Bayesian models showing how to handle categorical variables, including binary
and multi-category cases. This example also demonstrates how to compute contrasts
(differences) between categories from posterior samples.

Adapted from Rethinking Statistics 2nd edition, Chapter 5.3.
"""

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import xarray as xr

from pymc_extras.inference import fit_laplace

### Binary categories ###

# Show statistics of prior distributions
rng = np.random.default_rng(0)
mu_female = rng.normal(178, 20, size=10_000)
mu_male = rng.normal(178, 20, size=10_000) + rng.normal(0, 10, size=10_000)
summary = pd.DataFrame({"mu_female": mu_female, "mu_male": mu_male})
print(summary.describe(percentiles=[0.055, 0.945]).loc[["mean", "std", "5.5%", "94.5%"]])

d_h: pd.DataFrame = pd.read_csv("Howell1.csv", sep=";")

# R's vectors are 1-indexed while Python's arrays are 0-indexed. For this reason,
# we use the index variable: 0 = female, 1 = male
sex = np.where(d_h["male"] == 1, 1, 0)

# Fit the model using Laplace approximation. Note the parameter 'shape=2' for a,
# which creates a vector of two intercepts: a[0] for female, a[1] for male.
with pm.Model() as model:
    # a is a vector of two intercepts: a[0] for female, a[1] for male
    a = pm.Normal("a", mu=178, sigma=20, shape=2)
    sigma = pm.Uniform("sigma", lower=0, upper=50)
    mu = a[sex]
    pm.Normal("height", mu=mu, sigma=sigma, observed=d_h["height"].to_numpy())
    idata_m58 = fit_laplace(draws=10_000)

# Unlike precis() in R, vectors are not hidden with az.summary() so we do not need a keyword
# to show vector parameters.
print(az.summary(idata_m58, var_names=["a", "sigma"], hdi_prob=0.89, round_to=2, kind="stats"))

# Compute the female-male difference (contrast) from posterior samples and summarize.
# Note that dims=["chain", "draw"] is needed to recover the correct shape of the
# difference array (10000 samples, not 2)
a_samples = idata_m58.posterior["a"].to_numpy().reshape(-1, 2)  # shape (10000, 2)
idata_m58.posterior["diff_fm"] = xr.DataArray(
    a_samples[:, 0].reshape(idata_m58.posterior["a"].shape[:2])
    - a_samples[:, 1].reshape(idata_m58.posterior["a"].shape[:2]),
    dims=["chain", "draw"],
)
print(az.summary(idata_m58, var_names=["a", "sigma", "diff_fm"], hdi_prob=0.89, round_to=2, kind="stats"))


### Many categories ###

d_m: pd.DataFrame = pd.read_csv("milk.csv", sep=";")

# Create a clade_id variable for indexing the clades, and also get
# the clade names for labeling
d_m["clade_id"], clade_levels = (
    pd.Categorical(d_m["clade"]).codes,
    pd.Categorical(d_m["clade"]).categories,
)
labels = [f"a[{i}]: {name}" for i, name in enumerate(clade_levels)]

# Standardize the kcal.per.g variable
mean_K, std_K = d_m["kcal.per.g"].mean(), d_m["kcal.per.g"].std(ddof=1)
d_m["K"] = (d_m["kcal.per.g"] - mean_K) / std_K

# Fit the model using Laplace approximation
with pm.Model() as model:
    a = pm.Normal("a", mu=0, sigma=0.5, shape=4)
    sigma = pm.Uniform("sigma", lower=0, upper=50)
    mu = a[d_m["clade_id"].to_numpy()]
    pm.Normal("height", mu=mu, sigma=sigma, observed=d_m["K"].to_numpy())
    idata_m59 = fit_laplace(draws=10_000)

# Graphical summary of posterior means and 89% HDI for the 4 clades
az.plot_forest(idata_m59, var_names=["a"], combined=True, hdi_prob=0.89, figsize=(7, 3))
ax = plt.gca()
ax.set_yticklabels(labels[::-1])  # az.plot_forest renders bottom-to-top, so reverse labels
ax.set_xlabel("Expected kcal (std)")
ax.set_title("Posterior means and 89% HDI by clade")
plt.tight_layout()
plt.show()

# Assign primates to 4 randomly made-up "houses" (0-indexed)
# R: sample(rep(1:4, each=8), size=nrow(d)), which means each house appears
# 8 times, randomly shuffled
rng_house = np.random.default_rng(0)
house = rng_house.permutation(np.tile(np.arange(4), 8))[: len(d_m)]
d_m["house"] = house

with pm.Model() as model:
    a = pm.Normal("a", mu=0, sigma=0.5, shape=4)
    h = pm.Normal("h", mu=0, sigma=0.5, shape=4)
    sigma = pm.Exponential("sigma", lam=1)
    mu = a[d_m["clade_id"].to_numpy()] + h[d_m["house"].to_numpy()]
    pm.Normal("K", mu=mu, sigma=sigma, observed=d_m["K"].to_numpy())
    idata_m510 = fit_laplace(draws=10_000)

labels.append("h[0]: Gryffindor")
labels.append("h[1]: Hufflepuff")
labels.append("h[2]: Ravenclaw")
labels.append("h[3]: Slytherin")

# Graphical summary of posterior means and 89% HDI for the
# 4 clades and 4 houses. This likely gives a different result
# than what McElreath's code would give, since Slytherin does
# not stand out as the book implies it should.
az.plot_forest(idata_m510, var_names=["a", "h"], combined=True, hdi_prob=0.89, figsize=(7, 3))
ax = plt.gca()
ax.set_yticklabels(labels[::-1])  # az.plot_forest renders bottom-to-top, so reverse labels
ax.set_xlabel("Expected kcal (std)")
ax.set_title("Posterior means and 89% HDI by clade and house")
plt.tight_layout()
plt.show()
