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

d: pd.DataFrame = pd.read_csv("milk.csv", sep=";")
d = d.dropna().reset_index(drop=True) # Drop rows with missing values

# Standardize the variables
mean_K, std_K = d["kcal.per.g"].mean(), d["kcal.per.g"].std(ddof=1)
mean_N, std_N = d["neocortex.perc"].mean(), d["neocortex.perc"].std(ddof=1)
mean_M, std_M = np.log(d["mass"]).mean(), np.log(d["mass"]).std(ddof=1)
d["K"] = (d["kcal.per.g"] - mean_K) / std_K
d["N"] = (d["neocortex.perc"] - mean_N) / std_N
d["M"] = (np.log(d["mass"]) - mean_M) / std_M


# Simulate 50 prior regression lines over N in [-2, 2]
rng = np.random.default_rng(2)
n_lines = 50
N_seq = np.array([-2, 2])
prior_a_silly  = rng.normal(loc=0, scale=1, size=n_lines)
prior_bN_silly = rng.normal(loc=0, scale=1, size=n_lines)
prior_a_less_silly  = rng.normal(loc=0, scale=0.2, size=n_lines)
prior_bN_less_silly = rng.normal(loc=0, scale=0.5, size=n_lines)

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

# Fit the model using Laplace approximation
# When NaN values are in the data, this appears to fail silently. There is not error,
# but the output does indicate something went wrong (iteration: 2/2, objective: nan,
# grad: nan).
with pm.Model() as model:
    a    = pm.Normal("a", mu=0, sigma=0.2)
    bN    = pm.Normal("bN", mu=0, sigma=0.5)
    sigma = pm.Exponential("sigma", lam=1)
    mu = a + bN * d["N"]
    pm.Normal("K", mu=mu, sigma=sigma, observed=d["K"].to_numpy())
    idata_m55 = fit_laplace(draws=10_000)

print(az.summary(idata_m55, var_names=["a", "bN", "sigma"], hdi_prob=0.89, round_to=2, kind="stats"))

# Posterior predictions over a range of N values
sample_a  = idata_m55.posterior["a"].to_numpy().ravel()
sample_bN = idata_m55.posterior["bN"].to_numpy().ravel()
N_vals = d["N"].to_numpy()
N_seq  = np.linspace(N_vals.min() - 0.15, N_vals.max() + 0.15, 30)
mu_post     = sample_a[:, None] + sample_bN[:, None] * N_seq[None, :]  # shape (10000, 30)
mu_mean     = mu_post.mean(axis=0)
mu_pi       = np.percentile(mu_post, [5.5, 94.5], axis=0)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

ax = axes[0]
ax.scatter(d["N"], d["K"])
ax.plot(N_seq, mu_mean, color="black", linewidth=2)
ax.fill_between(N_seq, mu_pi[0], mu_pi[1], color="black", alpha=0.2)
ax.set_xlabel("Neocortex percent (standardized)")
ax.set_ylabel("kcal per gram (standardized)")


# Fit the model using Laplace approximation
with pm.Model() as model:
    a    = pm.Normal("a", mu=0, sigma=0.2)
    bM    = pm.Normal("bM", mu=0, sigma=0.5)
    sigma = pm.Exponential("sigma", lam=1)
    mu = a + bM * d["M"]
    pm.Normal("K", mu=mu, sigma=sigma, observed=d["K"].to_numpy())
    idata_m56 = fit_laplace(draws=10_000)

print(az.summary(idata_m56, var_names=["a", "bM", "sigma"], hdi_prob=0.89, round_to=2, kind="stats"))

# Posterior predictions over a range of M values
sample_a  = idata_m56.posterior["a"].to_numpy().ravel()
sample_bM = idata_m56.posterior["bM"].to_numpy().ravel()
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

plt.tight_layout()
plt.show()

# Fit a multivariate model using Laplace approximation
with pm.Model() as model:
    a    = pm.Normal("a", mu=0, sigma=0.2)
    bN    = pm.Normal("bN", mu=0, sigma=0.5)
    bM    = pm.Normal("bM", mu=0, sigma=0.5)
    sigma = pm.Exponential("sigma", lam=1)
    mu = a + bN * d["N"] + bM * d["M"]
    pm.Normal("K", mu=mu, sigma=sigma, observed=d["K"].to_numpy())
    idata_m57 = fit_laplace(draws=10_000)

print(az.summary(idata_m57, var_names=["a", "bN", "bM", "sigma"], hdi_prob=0.89, round_to=2, kind="stats"))

# Coeftab plot: compare bN and bM across models
az.plot_forest(
    [idata_m55, idata_m56, idata_m57],
    model_names=["m5.5 (N only)", "m5.6 (M only)", "m5.7 (N + M)"],
    var_names=["bM", "bN"],
    combined=True,
    hdi_prob=0.89,
    figsize=(7, 3),
)
plt.axvline(0, color="black", linestyle="--", linewidth=0.8)
plt.title("Coefficient comparison: m5.5 vs m5.6 vs m5.7")
plt.tight_layout()
plt.show()

# Counterfactual plots using m5.7 posterior samples
sample_a_m57  = idata_m57.posterior["a"].to_numpy().ravel()
sample_bN_m57 = idata_m57.posterior["bN"].to_numpy().ravel()
sample_bM_m57 = idata_m57.posterior["bM"].to_numpy().ravel()

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
