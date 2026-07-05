""" Bayesian modeling of milk energy content as a function of neocortex percent
and log mass, using Laplace approximation. The code also includes prior predictive
checks, posterior predictive checks, and counterfactual simulations.

Adapted from Rethinking Statistics 2nd edition, Chapter 5.2.
"""

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm

from pymc_extras.inference import fit_laplace

# Code 5.28 — Read in the data
d: pd.DataFrame = pd.read_csv("milk.csv", sep=";")

# Code 5.29 — Standardize the variables
mean_K, std_K = d["kcal.per.g"].mean(), d["kcal.per.g"].std(ddof=1)
mean_N, std_N = d["neocortex.perc"].mean(), d["neocortex.perc"].std(ddof=1)
mean_M, std_M = np.log(d["mass"]).mean(), np.log(d["mass"]).std(ddof=1)
d["K"] = (d["kcal.per.g"] - mean_K) / std_K
d["N"] = (d["neocortex.perc"] - mean_N) / std_N
d["M"] = (np.log(d["mass"]) - mean_M) / std_M

# Code 5.30 — Fit the model using Laplace approximation
# With PyMC v6.0.1, this gives a long and uninformative error.
# To prevent this file from failing, the model fitting is commented
# out. The model fitting is repeated later in the file after dropping
# rows with missing values.
#
# with pm.Model() as model_m55_draft:
#     a    = pm.Normal("a", mu=0, sigma=0.2)
#     bN    = pm.Normal("bN", mu=0, sigma=0.5)
#     sigma = pm.Exponential("sigma", lam=1)
#     mu = a + bN * d["N"]
#     pm.Normal("K", mu=mu, sigma=sigma, observed=d["K"].to_numpy())
#     idata_m55_draft = fit_laplace(draws=10_000)

# Code 5.32 — Drop rows with missing values
d = d.dropna().reset_index(drop=True)

# Code 5.33 — Fit the model using Laplace approximation
with pm.Model() as model_m55_draft:
    a    = pm.Normal("a", mu=0, sigma=0.2)
    bN    = pm.Normal("bN", mu=0, sigma=0.5)
    sigma = pm.Exponential("sigma", lam=1)
    mu = a + bN * d["N"]
    pm.Normal("K", mu=mu, sigma=sigma, observed=d["K"].to_numpy())
    idata_m55_draft = fit_laplace(draws=10_000)

# Code 5.34 — Prior predictive check by simulating 50 prior regression lines over N in [-2, 2]
# We create both prior predictive distributions shown in Figure 5.8.
rng = np.random.default_rng(2)
n_lines = 50
N_seq = np.array([-2, 2])
prior_a_silly  = rng.normal(loc=0, scale=1, size=n_lines)
prior_bN_silly = rng.normal(loc=0, scale=1, size=n_lines)
prior_a_less_silly  = rng.normal(loc=0, scale=0.2, size=n_lines)
prior_bN_less_silly = rng.normal(loc=0, scale=0.5, size=n_lines)

# Code 5.35 — Plot the prior regression lines
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

ax = axes[0]
for i in range(n_lines):
    ax.plot(N_seq, prior_a_silly[i] + prior_bN_silly[i] * N_seq, color="black", alpha=0.3)
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_xlabel("Neocortex percent (standardized)")
ax.set_ylabel("kcal per gram (standardized)")
ax.set_title("a ~ dnorm(0, 1)\nbN ~ dnorm(0, 1)")

ax = axes[1]
for i in range(n_lines):
    ax.plot(N_seq, prior_a_less_silly[i] + prior_bN_less_silly[i] * N_seq, color="black", alpha=0.3)
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_xlabel("Neocortex percent (standardized)")
ax.set_ylabel("kcal per gram (standardized)")
ax.set_title("a ~ dnorm(0, 0.2)\nbN ~ dnorm(0, 0.5)")

plt.tight_layout()
plt.show()


# 5.35 — Fit the model using Laplace approximation
with pm.Model() as model_m55:
    a    = pm.Normal("a", mu=0, sigma=0.2)
    bN    = pm.Normal("bN", mu=0, sigma=0.5)
    sigma = pm.Exponential("sigma", lam=1)
    mu = a + bN * d["N"]
    pm.Normal("K", mu=mu, sigma=sigma, observed=d["K"].to_numpy())
    idata_m55 = fit_laplace(draws=10_000)

# 5.36 — Summarize the posterior distribution
print(az.summary(idata_m55, var_names=["a", "bN", "sigma"], ci_prob=0.89, round_to=2, kind="stats"))

# Code 5.37 — Posterior predictions over a range of N values
# Could also be done with PyMC's sample_posterior_predictive, but
# we do it manually here to illustrate the process and to follow
# the book's approach.
#
# Using PyMC requires  model_m55 to be restructured to use
# pm.Data("N_obs", ...) and pm.Deterministic("mu", ...), and then do:
#
# with model_m55:
#     pm.set_data({"N_obs": N_seq})
#     ppc = pm.sample_posterior_predictive(idata_m55, var_names=["mu"])
# mu_arr  = ppc.posterior_predictive["mu"].values.reshape(-1, len(N_seq))
# mu_mean = mu_arr.mean(axis=0)
# mu_pi   = np.percentile(mu_arr, [5.5, 94.5], axis=0)
#
# With complex models, the manual code below would be too complex and the PyMC
# approach would be preferred.
#
post_m55 = idata_m55.posterior.ds.stack(sample=("chain", "draw"))
sample_a  = post_m55["a"].values
sample_bN = post_m55["bN"].values
N_vals = d["N"].to_numpy()
N_seq  = np.linspace(N_vals.min() - 0.15, N_vals.max() + 0.15, 30)
mu_post     = sample_a[:, None] + sample_bN[:, None] * N_seq[None, :]  # shape (10000, 30)
mu_mean     = mu_post.mean(axis=0)
mu_pi       = np.percentile(mu_post, [5.5, 94.5], axis=0)

# We'll define first plot of Figure 5.9 here, but won't show it yet.
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

ax = axes[0]
ax.scatter(d["N"], d["K"])
ax.plot(N_seq, mu_mean, color="black", linewidth=2)
ax.fill_between(N_seq, mu_pi[0], mu_pi[1], color="black", alpha=0.2)
ax.set_xlabel("Neocortex percent (standardized)")
ax.set_ylabel("kcal per gram (standardized)")

# Code 5.38 — Model with bivariate relationship between kilocalories and body mass
with pm.Model() as model_m56:
    a    = pm.Normal("a", mu=0, sigma=0.2)
    bM    = pm.Normal("bM", mu=0, sigma=0.5)
    sigma = pm.Exponential("sigma", lam=1)
    mu = a + bM * d["M"]
    pm.Normal("K", mu=mu, sigma=sigma, observed=d["K"].to_numpy())
    idata_m56 = fit_laplace(draws=10_000)

print(az.summary(idata_m56, var_names=["a", "bM", "sigma"], ci_prob=0.89, round_to=2, kind="stats"))

# Code for creating the second plot of Figure 5.9 (not shown in the book)
post_m56 = idata_m56.posterior.ds.stack(sample=("chain", "draw"))
sample_a  = post_m56["a"].values
sample_bM = post_m56["bM"].values
M_vals = d["M"].to_numpy()
M_seq  = np.linspace(M_vals.min() - 0.15, M_vals.max() + 0.15, 30)
mu_post     = sample_a[:, None] + sample_bM[:, None] * M_seq[None, :]  # shape (10000, 30)
mu_mean     = mu_post.mean(axis=0)
mu_pi       = np.percentile(mu_post, [5.5, 94.5], axis=0)

ax = axes[1]
ax.scatter(d["M"], d["K"])
ax.plot(M_seq, mu_mean, color="black", linewidth=2)
ax.fill_between(M_seq, mu_pi[0], mu_pi[1], color="black", alpha=0.2)
ax.set_xlabel("Log mass (standardized)")
ax.set_ylabel("kcal per gram (standardized)")

# Showing the top part of Figure 5.9.
plt.tight_layout()
plt.show()

# Code 5.39 — Adding both predictor variables
with pm.Model() as model_m57:
    a    = pm.Normal("a", mu=0, sigma=0.2)
    bN    = pm.Normal("bN", mu=0, sigma=0.5)
    bM    = pm.Normal("bM", mu=0, sigma=0.5)
    sigma = pm.Exponential("sigma", lam=1)
    mu = a + bN * d["N"] + bM * d["M"]
    pm.Normal("K", mu=mu, sigma=sigma, observed=d["K"].to_numpy())
    idata_m57 = fit_laplace(draws=10_000)

print(az.summary(idata_m57, var_names=["a", "bN", "bM", "sigma"], ci_prob=0.89, round_to=2, kind="stats"))


# Code 5.40 — Coeftab plot: compare bN and bM across models
# ArviZ 1.x plot_forst() excludes "model" from labellable_dims by design, so model names
# cannot appear as row labels in az.plot_forest. Building the plot manually instead.
from matplotlib.lines import Line2D

cf_models = {
    "m5.5 (N only)": idata_m55,
    "m5.6 (M only)": idata_m56,
    "m5.7 (N + M)": idata_m57,
}
cf_params = ["bN", "bM"]
cf_colors = {"bN": "steelblue", "bM": "tomato"}
fig, ax = plt.subplots(figsize=(7, 4))
y_pos = 0
yticks, ylabels = [], []
for param in cf_params:
    for model_name, idata in cf_models.items():
        post = idata.posterior.ds if hasattr(idata.posterior, "ds") else idata.posterior
        samples = post[param].stack(sample=("chain", "draw")).values if param in post else np.full(100, np.nan)
        if np.all(np.isnan(samples)):
            yticks.append(y_pos); ylabels.append(model_name); y_pos += 1
            continue
        hdi89 = az.hdi(idata, var_names=[param], prob=0.89)[param].values
        hdi50 = az.hdi(idata, var_names=[param], prob=0.50)[param].values
        ax.plot([hdi89[0], hdi89[1]], [y_pos, y_pos], color=cf_colors[param], linewidth=1.5)
        ax.plot([hdi50[0], hdi50[1]], [y_pos, y_pos], color=cf_colors[param], linewidth=4)
        ax.plot(float(np.mean(samples)), y_pos, "o", color="white",
                markeredgecolor=cf_colors[param], markersize=5, zorder=5)
        yticks.append(y_pos); ylabels.append(model_name); y_pos += 1
    y_pos += 0.5
ax.set_yticks(yticks)
ax.set_yticklabels(ylabels, fontsize=9)
ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
ax.legend(handles=[Line2D([0], [0], color=cf_colors[p], linewidth=3, label=p) for p in cf_params],
          loc="lower right")
ax.set_title("Coefficient comparison: m5.5 vs m5.6 vs m5.7")
ax.set_xlabel("Posterior estimate (standardized)")
plt.tight_layout()
plt.show()

# Code 5.41 (and more) — counterfactual plots using m5.7 posterior samples
# We'll make both lower plots of Figure 3.9 in one go
post_m57 = idata_m57.posterior.ds.stack(sample=("chain", "draw"))
sample_a_m57  = post_m57["a"].values
sample_bN_m57 = post_m57["bN"].values
sample_bM_m57 = post_m57["bM"].values

# Vary M, hold N=0
M_cf_seq = np.linspace(M_vals.min() - 0.15, M_vals.max() + 0.15, 30)
mu_varyM = sample_a_m57[:, None] + sample_bN_m57[:, None] * 0 + sample_bM_m57[:, None] * M_cf_seq[None, :]

# Vary N, hold M=0
N_cf_seq = np.linspace(N_vals.min() - 0.15, N_vals.max() + 0.15, 30)
mu_varyN = sample_a_m57[:, None] + sample_bN_m57[:, None] * N_cf_seq[None, :] + sample_bM_m57[:, None] * 0

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

ax = axes[0]
ax.plot(N_cf_seq, mu_varyN.mean(axis=0), color="black", linewidth=2)
ax.fill_between(N_cf_seq, *np.percentile(mu_varyN, [5.5, 94.5], axis=0), color="black", alpha=0.2)
ax.set_xlim(N_vals.min(), N_vals.max())
ax.set_ylim(d["K"].min(), d["K"].max())
ax.set_xlabel("Neocortex percent (standardized)")
ax.set_ylabel("kcal per gram (standardized)")
ax.set_title("Counterfactual: varying N, holding M = 0")

ax = axes[1]
ax.plot(M_cf_seq, mu_varyM.mean(axis=0), color="black", linewidth=2)
ax.fill_between(M_cf_seq, *np.percentile(mu_varyM, [5.5, 94.5], axis=0), color="black", alpha=0.2)
ax.set_xlim(M_vals.min(), M_vals.max())
ax.set_ylim(d["K"].min(), d["K"].max())
ax.set_xlabel("Log mass (standardized)")
ax.set_ylabel("kcal per gram (standardized)")
ax.set_title("Counterfactual: varying M, holding N = 0")

plt.tight_layout()
plt.show()
