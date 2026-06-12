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

N = 100  # number of individuals
rng = np.random.default_rng(123) # This seed will not make the data correspond to the example in the book, but it will be consistent across runs.

height = rng.normal(10, 2, N)           # sim total height of each
leg_prop = rng.uniform(0.4, 0.5, N)     # leg as proportion of height
leg_left = leg_prop * height + rng.normal(0, 0.02, N)   # sim left leg as proportion + error
leg_right = leg_prop * height + rng.normal(0, 0.02, N)  # sim right leg as proportion + error

d = pd.DataFrame({"height": height, "leg_left": leg_left, "leg_right": leg_right})

# Fit the model using Laplace approximation, with collinear predictors
with pm.Model() as model:
    a = pm.Normal("a", mu=10, sigma=100)
    bl = pm.Normal("bl", mu=2, sigma=10)
    br = pm.Normal("br", mu=2, sigma=10)
    sigma = pm.Exponential("sigma", lam=1)
    mu = a + bl * d["leg_left"] + br * d["leg_right"]
    pm.Normal("height", mu=mu, sigma=sigma, observed=d["height"].to_numpy())
    idata_m61 = fit_laplace(draws=10_000)

print(az.summary(idata_m61, var_names=["a", "bl", "br", "sigma"], hdi_prob=0.89, round_to=2, kind="stats"))

params = ["a", "bl", "br", "sigma"]
params.reverse()
summary = az.summary(idata_m61, var_names=params, hdi_prob=0.89, kind="stats")

means = summary["mean"]
lower = summary["hdi_5.5%"]
upper = summary["hdi_94.5%"]

fig, ax = plt.subplots()
y_pos = range(len(params))

ax.scatter(means, y_pos, color="black", zorder=3)
for i, (lo, hi) in enumerate(zip(lower, upper)):
    ax.plot([lo, hi], [i, i], color="black", linewidth=1.5)

ax.set_yticks(list(y_pos))
ax.set_yticklabels(params)
ax.axvline(0, linestyle="--", color="gray", linewidth=0.8)
ax.set_xlabel("Posterior estimate (89% HDI)")
ax.set_title("precis(m6.1)")
plt.tight_layout()
plt.show()

# post <- extract.samples(m6.1); plot(bl ~ br, post, ...) equivalent
post = idata_m61.posterior.stack(sample=("chain", "draw"))
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

# The same as the previous model, without the collinear predictors
with pm.Model() as model:
    a = pm.Normal("a", mu=10, sigma=100)
    bl = pm.Normal("bl", mu=2, sigma=10)
    sigma = pm.Exponential("sigma", lam=1)
    mu = a + bl * d["leg_left"]
    pm.Normal("height", mu=mu, sigma=sigma, observed=d["height"].to_numpy())
    idata_m62 = fit_laplace(draws=10_000)

print(az.summary(idata_m62, var_names=["a", "bl", "sigma"], hdi_prob=0.89, round_to=2, kind="stats"))


### 6.1.2 Multicolinear milk ###

d: pd.DataFrame = pd.read_csv("milk.csv", sep=";")
d = d.dropna().reset_index(drop=True) # Drop rows with missing values

# Standardize the variables
mean_K, std_K = d["kcal.per.g"].mean(), d["kcal.per.g"].std(ddof=1)
mean_F, std_F = d["perc.fat"].mean(), d["perc.fat"].std(ddof=1)
mean_L, std_L = d["perc.lactose"].mean(), d["perc.lactose"].std(ddof=1)
d["K"] = (d["kcal.per.g"] - mean_K) / std_K
d["F"] = (d["perc.fat"] - mean_F) / std_F
d["L"] = (d["perc.lactose"] - mean_L) / std_L

# kcal.per.g regressed on perc.fat
with pm.Model() as model:
    a = pm.Normal("a", mu=0, sigma=0.2)
    bF = pm.Normal("bF", mu=0, sigma=0.5)
    sigma = pm.Exponential("sigma", lam=1)
    mu = a + bF * d["F"]
    pm.Normal("K", mu=mu, sigma=sigma, observed=d["K"].to_numpy())
    idata_m63 = fit_laplace(draws=10_000)

# kcal.per.g regressed on perc.lactose
with pm.Model() as model:
    a = pm.Normal("a", mu=0, sigma=0.2)
    bL = pm.Normal("bL", mu=0, sigma=0.5)
    sigma = pm.Exponential("sigma", lam=1)
    mu = a + bL * d["L"]
    pm.Normal("K", mu=mu, sigma=sigma, observed=d["K"].to_numpy())
    idata_m64 = fit_laplace(draws=10_000)

print(az.summary(idata_m63, var_names=["a", "bF", "sigma"], hdi_prob=0.89, round_to=2, kind="stats"))
print(az.summary(idata_m64, var_names=["a", "bL", "sigma"], hdi_prob=0.89, round_to=2, kind="stats"))

# kcal.per.g regressed on both perc.fat and perc.lactose
with pm.Model() as model:
    a = pm.Normal("a", mu=0, sigma=0.2)
    bF = pm.Normal("bF", mu=0, sigma=0.5)
    bL = pm.Normal("bL", mu=0, sigma=0.5)
    sigma = pm.Exponential("sigma", lam=1)
    mu = a + bF * d["F"] + bL * d["L"]
    pm.Normal("K", mu=mu, sigma=sigma, observed=d["K"].to_numpy())
    idata_m65 = fit_laplace(draws=10_000)

print(az.summary(idata_m65, var_names=["a", "bF", "bL", "sigma"], hdi_prob=0.89, round_to=2, kind="stats"))

# pairs(~kcal.per.g+perc.fat+perc.lactose, data=d, col=rangi2) equivalent (Figure 6.3)
# To the best of my knowledge, only seaborn's pairplot can achieve a similar result
sns.pairplot(
    d[["kcal.per.g", "perc.fat", "perc.lactose"]],
    plot_kws={"color": "#1e90ff", "alpha": 0.6},
    diag_kws={"color": "#1e90ff"},
)
plt.suptitle("Pairs plot: kcal.per.g, perc.fat, perc.lactose", y=1.02)
plt.tight_layout()
plt.show()
