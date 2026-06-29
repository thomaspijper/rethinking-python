"""Bayesian models showing the effect of multicollinearity on parameter estimates. The examples also
demonstrate how to visualize pairwise relationships.

Adapted from Rethinking Statistics 2nd edition, Chapter 6.1."""

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import seaborn as sns

from pymc_extras.inference import fit_laplace
from scipy.stats import gaussian_kde

### 6.1.1 Multicolinear legs ###

# Code 6.2 — simulate data for leg lengths and total height
N = 100  # number of individuals
rng = np.random.default_rng(0)
height = rng.normal(10, 2, N)           # sim total height of each
leg_prop = rng.uniform(0.4, 0.5, N)     # leg as proportion of height
leg_left = leg_prop * height + rng.normal(0, 0.02, N)   # sim left leg as proportion + error
leg_right = leg_prop * height + rng.normal(0, 0.02, N)  # sim right leg as proportion + error

d = pd.DataFrame({"height": height, "leg_left": leg_left, "leg_right": leg_right})

# Code 6.3 — fit the model using Laplace approximation, with collinear predictors,
# then summarize
with pm.Model() as model_m61:
    a = pm.Normal("a", mu=10, sigma=100)
    bl = pm.Normal("bl", mu=2, sigma=10)
    br = pm.Normal("br", mu=2, sigma=10)
    sigma = pm.Exponential("sigma", lam=1)
    mu = a + bl * d["leg_left"] + br * d["leg_right"]
    pm.Normal("height", mu=mu, sigma=sigma, observed=d["height"].to_numpy())
    idata_m61 = fit_laplace(draws=10_000)

print(az.summary(idata_m61, var_names=["a", "bl", "br", "sigma"], ci_prob=0.89, round_to=2, kind="stats"))

# Code 6.4 — plot(precis(m6.1)) equivalent: forest plot
# The ci_probs argument can be used to show multiple credible intervals.
# Its default is (0.5, rcParams["stats.ci_prob"]), which is (0.5, 0.89) by default.
az.plot_forest(idata_m61, var_names=["a", "bl", "br", "sigma"])
plt.axvline(0, linestyle="--", color="gray", linewidth=0.8)
plt.xlabel("Posterior estimate")
plt.title("precis(m6.1)")
plt.tight_layout()
plt.show()

# Code 6.5 and 6.6 — recreation of Figure 6.2
# post <- extract.samples(m6.1); plot(bl ~ br, post, ...) equivalent
post = idata_m61.posterior.ds.stack(sample=("chain", "draw"))
bl_samples = post["bl"].values
br_samples = post["br"].values

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

# Left: scatter plot of bl vs br
ax1.scatter(br_samples, bl_samples, alpha=0.1, color="#1e90ff", s=10)
ax1.set_xlabel("br")
ax1.set_ylabel("bl")
ax1.set_title("Posterior samples: bl vs br")

# Right: density of bl + br
bl_br_sum = bl_samples + br_samples
x_range = np.linspace(bl_br_sum.min(), bl_br_sum.max(), 500)
kde = gaussian_kde(bl_br_sum)
ax2.plot(x_range, kde(x_range), color="#1e90ff")
ax2.fill_between(x_range, kde(x_range), alpha=0.2, color="#1e90ff")
ax2.set_xlabel("bl + br")
ax2.set_ylabel("Density")
ax2.set_title("Posterior density: bl + br")

plt.tight_layout()
plt.show()

# Code 6.7 — as the previous model, without the collinear predictors
with pm.Model() as model_m62:
    a = pm.Normal("a", mu=10, sigma=100)
    bl = pm.Normal("bl", mu=2, sigma=10)
    sigma = pm.Exponential("sigma", lam=1)
    mu = a + bl * d["leg_left"]
    pm.Normal("height", mu=mu, sigma=sigma, observed=d["height"].to_numpy())
    idata_m62 = fit_laplace(draws=10_000)

print(az.summary(idata_m62, var_names=["a", "bl", "sigma"], ci_prob=0.89, round_to=2, kind="stats"))


### 6.1.2 Multicollinear milk ###

# Code 6.8 — load the milk data, drop rows with missing values, and standardize the variables
d: pd.DataFrame = pd.read_csv("milk.csv", sep=";")
d = d.dropna().reset_index(drop=True) # Drop rows with missing values
mean_K, std_K = d["kcal.per.g"].mean(), d["kcal.per.g"].std(ddof=1)
mean_F, std_F = d["perc.fat"].mean(), d["perc.fat"].std(ddof=1)
mean_L, std_L = d["perc.lactose"].mean(), d["perc.lactose"].std(ddof=1)
d["K"] = (d["kcal.per.g"] - mean_K) / std_K
d["F"] = (d["perc.fat"] - mean_F) / std_F
d["L"] = (d["perc.lactose"] - mean_L) / std_L


# Code 6.9 — modeling kcal.per.g as a function of perc.fat and perc.lactose
# kcal.per.g regressed on perc.fat
with pm.Model() as model_m63:
    a = pm.Normal("a", mu=0, sigma=0.2)
    bF = pm.Normal("bF", mu=0, sigma=0.5)
    sigma = pm.Exponential("sigma", lam=1)
    mu = a + bF * d["F"]
    pm.Normal("K", mu=mu, sigma=sigma, observed=d["K"].to_numpy())
    idata_m63 = fit_laplace(draws=10_000)

# kcal.per.g regressed on perc.lactose
with pm.Model() as model_m64:
    a = pm.Normal("a", mu=0, sigma=0.2)
    bL = pm.Normal("bL", mu=0, sigma=0.5)
    sigma = pm.Exponential("sigma", lam=1)
    mu = a + bL * d["L"]
    pm.Normal("K", mu=mu, sigma=sigma, observed=d["K"].to_numpy())
    idata_m64 = fit_laplace(draws=10_000)

print(az.summary(idata_m63, var_names=["a", "bF", "sigma"], ci_prob=0.89, round_to=2, kind="stats"))
print(az.summary(idata_m64, var_names=["a", "bL", "sigma"], ci_prob=0.89, round_to=2, kind="stats"))

# Code 6.10 — kcal.per.g regressed on both perc.fat and perc.lactose
with pm.Model() as model_m65:
    a = pm.Normal("a", mu=0, sigma=0.2)
    bF = pm.Normal("bF", mu=0, sigma=0.5)
    bL = pm.Normal("bL", mu=0, sigma=0.5)
    sigma = pm.Exponential("sigma", lam=1)
    mu = a + bF * d["F"] + bL * d["L"]
    pm.Normal("K", mu=mu, sigma=sigma, observed=d["K"].to_numpy())
    idata_m65 = fit_laplace(draws=10_000)

print(az.summary(idata_m65, var_names=["a", "bF", "bL", "sigma"], ci_prob=0.89, round_to=2, kind="stats"))

# Code 6.11 — pairs(~kcal.per.g+perc.fat+perc.lactose, data=d, col=rangi2) equivalent (Figure 6.3)
# To the best of my knowledge, only seaborn's pairplot can achieve a similar result in an easy way
sns.pairplot(
    d[["kcal.per.g", "perc.fat", "perc.lactose"]],
)
plt.suptitle("Pairs plot: kcal.per.g, perc.fat, perc.lactose", y=1.02)
plt.tight_layout()
plt.show()
