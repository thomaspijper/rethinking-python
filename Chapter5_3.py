"""Bayesian models showing how to handle categorical variables (binary
and multi-category cases). This example also demonstrates how to compute contrasts
(differences) between categories from posterior samples.

Adapted from Rethinking Statistics 2nd edition, Chapter 5.3.
"""

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm

from pymc_extras.inference import fit_laplace

### 5.3.1 Binary categories ###

# Code 5.45 — load the data
d: pd.DataFrame = pd.read_csv("Howell1.csv", sep=";")
print(d.head())

# Code 5.46 — show statistics of prior distributions
rng = np.random.default_rng(0)
mu_female = rng.normal(178, 20, size=10_000)
mu_male = rng.normal(178, 20, size=10_000) + rng.normal(0, 10, size=10_000)
summary = pd.DataFrame({"mu_female": mu_female, "mu_male": mu_male})
print(summary.describe(percentiles=[0.055, 0.945]).loc[["mean", "std", "5.5%", "94.5%"]])

# Code 5.47 — constuct the index variable
# R's vectors are 1-indexed while Python's arrays are 0-indexed. For this reason,
# we use the index variable: 0 = female, 1 = male
sex = np.where(d["male"] == 1, 1, 0)
print(sex)

# Code 5.48 — fit the model using Laplace approximation. Note the parameter 'shape=2'
# for a, which creates a vector of two intercepts: a[0] for female, a[1] for male.
#
# Unlike precis() in R, vectors are not hidden with az.summary() so we do not need
# the depth=2 argument to show vector parameters.
#
# Also, we include something that is in code 5.49, namely the calculation of diff_fm using
# pm.Deterministic() (so that it is included in the posterior and can be summarized).
with pm.Model() as model_m58:
    # a is a vector of two intercepts: a[0] for female, a[1] for male
    a = pm.Normal("a", mu=178, sigma=20, shape=2)
    sigma = pm.Uniform("sigma", lower=0, upper=50)
    pm.Deterministic("diff_fm", a[0] - a[1])  # female minus male contrast
    mu = a[sex]
    pm.Normal("height", mu=mu, sigma=sigma, observed=d["height"].to_numpy())
    idata_m58 = fit_laplace(draws=10_000)

print(az.summary(idata_m58, var_names=["a", "sigma"], ci_prob=0.89, round_to=2, kind="stats"))

# Code 5.49 — summarize including diff_fm
print(az.summary(idata_m58, var_names=["a", "sigma", "diff_fm"], ci_prob=0.89, round_to=2, kind="stats"))


### 5.3.2 Many categories ###

# Code 5.50 — load the data and print the clade levels
d: pd.DataFrame = pd.read_csv("milk.csv", sep=";")
levels = pd.Categorical(d["clade"]).categories
print(levels)

# Code 5.51 — Create a clade_id variable for indexing the clades
d["clade_id"] = pd.Categorical(d["clade"]).codes

# Code 5.52 — standardize the kcal.per.g variable, fit the model, and plot
mean_K, std_K = d["kcal.per.g"].mean(), d["kcal.per.g"].std(ddof=1)
d["K"] = (d["kcal.per.g"] - mean_K) / std_K

# coords assigns named coordinate values to each dimension. When a variable
# declares dims="clade", PyMC stores these names in the posterior so ArviZ
# can use them as axis labels in plots and summaries automatically.
with pm.Model(coords={"clade": list(levels)}) as model_m59:
    a = pm.Normal("a", mu=0, sigma=0.5, dims="clade")
    sigma = pm.Uniform("sigma", lower=0, upper=50)
    mu = a[d["clade_id"].to_numpy()]
    pm.Normal("height", mu=mu, sigma=sigma, observed=d["K"].to_numpy())
    idata_m59 = fit_laplace(draws=10_000)

# Graphical summary of posterior means and 89% HDI for the 4 clades
az.plot_forest(idata_m59, var_names=["a"], combined=True, ci_probs=[0.5, 0.89])
plt.gcf().set_size_inches(7, 3)
plt.xlabel("Expected kcal (std)")
plt.title("Posterior means and 89% HDI by clade")
plt.tight_layout()
plt.show()

# Code 5.53 — assign primates to 4 randomly made-up "houses" (0-indexed)
# R: sample(rep(1:4, each=8), size=nrow(d)), which means each house appears
# 8 times, randomly shuffled
houses = ["Gryffindor", "Hufflepuff", "Ravenclaw", "Slytherin"]
rng_house = np.random.default_rng(0)
house = rng_house.permutation(np.tile(np.arange(4), 8))[: len(d)]
d["house"] = house

# Code 5.54 — fit the model with both clade and house as predictors
# coords here defines two named dimensions: one for the 4 clades and one for
# the 4 houses.
with pm.Model(coords={"clade": list(levels), "house": houses}) as model_m510:
    a = pm.Normal("a", mu=0, sigma=0.5, dims="clade")
    h = pm.Normal("h", mu=0, sigma=0.5, dims="house")
    sigma = pm.Exponential("sigma", lam=1)
    mu = a[d["clade_id"].to_numpy()] + h[d["house"].to_numpy()]
    pm.Normal("K", mu=mu, sigma=sigma, observed=d["K"].to_numpy())
    idata_m510 = fit_laplace(draws=10_000)

# Graphical summary of posterior means and 89% HDI for the 4 clades and 4 houses
# (code not included in the book). This likely gives a different result than what
# McElreath's code would give, since Slytherin does not stand out as the book implies
# it should.
az.plot_forest(idata_m510, var_names=["a", "h"], combined=True, ci_probs=[0.5, 0.89])
plt.gcf().set_size_inches(7, 3)
plt.xlabel("Expected kcal (std)")
plt.title("Posterior means and 89% HDI by clade and house")
plt.tight_layout()
plt.show()
