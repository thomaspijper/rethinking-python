"""Bayesian models of log GDP as a function of terrain ruggedness, with and without
an Africa indicator and a ruggedness-Africa interaction.

Adapted from Rethinking Statistics 2nd edition, Chapters 8.1 and 8.2.
"""

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm

from pymc_extras.inference import fit_laplace


### 8.1.1 Making a rugged model ###

# Code 8.1 — loading the data
d = pd.read_csv("rugged.csv", sep=";")
d["log_gdp"] = np.log(d["rgdppc_2000"])
dd = d[d["rgdppc_2000"].notna()].copy()
dd["log_gdp_std"] = dd["log_gdp"] / dd["log_gdp"].mean()
dd["rugged_std"]  = dd["rugged"]  / dd["rugged"].max()

# Code 8.2 — first candidate model
# 0.215, the mean of rugged_std, centers the predictor so intercept = mean log GDP
with pm.Model() as model_m81:
    a     = pm.Normal("a", mu=1, sigma=1)
    b     = pm.Normal("b", mu=0, sigma=1)
    sigma = pm.Exponential("sigma", lam=1)
    mu = a + b * (dd["rugged_std"].to_numpy() - 0.215)
    pm.Normal("log_gdp_std", mu=mu, sigma=sigma, observed=dd["log_gdp_std"].to_numpy())
    idata_m81 = fit_laplace(draws=10_000, random_seed=1) # Seed added to ensure reproducibility of results

# Code 8.3 — extract and plot priors. This gives the left plot in Figure 8.3.
with model_m81:
    prior = pm.sample_prior_predictive(draws=50, random_seed=7)
prior_draws = prior.prior.ds.stack(sample=("chain", "draw"))
prior_a = prior_draws["a"].values
prior_b = prior_draws["b"].values

rugged_seq = np.linspace(-0.1, 1.1, 30)
fig, ax = plt.subplots()
ax.set_xlim(0, 1)
ax.set_ylim(0.5, 1.5)
ax.axhline(dd["log_gdp_std"].min(), linestyle="--", color="black", linewidth=0.8)
ax.axhline(dd["log_gdp_std"].max(), linestyle="--", color="black", linewidth=0.8)
for i in range(50):
    mu_i = prior_a[i] + prior_b[i] * (rugged_seq - 0.215)
    ax.plot(rugged_seq, mu_i, color="black", alpha=0.3)
ax.set_xlabel("ruggedness")
ax.set_ylabel("log GDP (prop of mean)")
ax.set_title("Prior predictive (m8.1)")
plt.tight_layout()
plt.show()

# Code 8.4 — fraction of lines with slope > 0.6
print(np.mean(np.abs(prior_b) > 0.6))

# Code 8.5 — model with improved priors
with pm.Model() as model_m81:
    a     = pm.Normal("a", mu=1, sigma=0.1)
    b     = pm.Normal("b", mu=0, sigma=0.3)
    sigma = pm.Exponential("sigma", lam=1)
    mu = a + b * (dd["rugged_std"].to_numpy() - 0.215)
    pm.Normal("log_gdp_std", mu=mu, sigma=sigma, observed=dd["log_gdp_std"].to_numpy())
    idata_m81 = fit_laplace(draws=10_000, random_seed=1) # Seed added to ensure reproducibility of results

# Code 8.6 — posterior summary
print(az.summary(idata_m81, var_names=["a", "b", "sigma"], ci_prob=0.89, round_to=2, kind="stats"))


### 8.1.2 Adding an indicator variable isn't enough ###

# Code 8.7 — model with indicator variable for Africa
# 0 = Africa, 1 = not Africa (0-indexed for PyMC)
dd["cid"] = (dd["cont_africa"] != 1).astype(int)

# Code 8.8 — model with indicator variable for Africa
with pm.Model() as model_m82:
    a     = pm.Normal("a", mu=1, sigma=0.1, shape=2)
    b     = pm.Normal("b", mu=0, sigma=0.3)
    sigma = pm.Exponential("sigma", lam=1)
    mu = a[dd["cid"].to_numpy()] + b * (dd["rugged_std"].to_numpy() - 0.215)
    pm.Normal("log_gdp_std", mu=mu, sigma=sigma, observed=dd["log_gdp_std"].to_numpy())
    idata_m82 = fit_laplace(draws=10_000, random_seed=1) # Seed added to ensure reproducibility of results

# Code 8.9 — compare  models m8.1 and m8.2
# We use LOO-CV (via PSIS) instead of WAIC because WAIC is not available in Arviz 1.x
# The result shows that m8.1 is 30 elpd units worse than m8.2. Since the difference is 4
# standard errors from zero (30/7.5 = 4), this is strong evidence m8.2 predicts better
# out-of-sample.
with model_m81:
    pm.compute_log_likelihood(idata_m81)
with model_m82:
    pm.compute_log_likelihood(idata_m82)
print(az.compare({"m8.1": idata_m81, "m8.2": idata_m82}, var_name="log_gdp_std"))

# Code 8.10 — posterior summary
print(az.summary(idata_m82, var_names=["a", "b", "sigma"], ci_prob=0.89, round_to=2, kind="stats"))

# Code 8.11 — posterior contrast between the two intercepts
post_m82 = idata_m82.posterior.ds.stack(sample=("chain", "draw"))
diff_a1_a2 = post_m82["a"].values[0] - post_m82["a"].values[1]
print(np.quantile(diff_a1_a2, [0.055, 0.945]))

# Code 8.12 — posterior plotting
# The plotting code is not in the book, but is included here to recreate Figure 8.4.
rugged_seq = np.linspace(-0.1, 1.1, 30)
a_s = post_m82["a"].values    # (2, n_samples) — row 0: Africa, row 1: not Africa
b_s = post_m82["b"].values    # (n_samples,)

mu_africa = a_s[0] + b_s * (rugged_seq[:, None] - 0.215)  # (30, n_samples)
mu_notafrica = a_s[1] + b_s * (rugged_seq[:, None] - 0.215)

mu_africa_mean = mu_africa.mean(axis=1)
mu_notafrica_mean = mu_notafrica.mean(axis=1)
mu_africa_ci = np.quantile(mu_africa, [0.015, 0.985], axis=1)
mu_notafrica_ci = np.quantile(mu_notafrica, [0.015, 0.985], axis=1)

africa = dd[dd["cid"] == 0]
notafrica = dd[dd["cid"] == 1]

fig, ax = plt.subplots()
ax.scatter(africa["rugged_std"], africa["log_gdp_std"], color="steelblue", s=12, label="Africa")
ax.scatter(notafrica["rugged_std"], notafrica["log_gdp_std"], facecolors="none", edgecolors="black", s=12, label="Not Africa")
ax.plot(rugged_seq, mu_africa_mean, color="steelblue")
ax.plot(rugged_seq, mu_notafrica_mean, color="black")
ax.fill_between(rugged_seq, mu_africa_ci[0], mu_africa_ci[1], color="steelblue", alpha=0.3)
ax.fill_between(rugged_seq, mu_notafrica_ci[0], mu_notafrica_ci[1], color="black", alpha=0.2)
ax.set_xlabel("ruggedness (standardized)")
ax.set_ylabel("log GDP (as proportion of mean)")
ax.legend()
plt.tight_layout()
plt.show()


### 8.1.3 Adding an interaction does work ###

# Code 8.13 — model with interaction between ruggedness and Africa indicator
cid = dd["cid"].to_numpy()
with pm.Model() as model_m83:
    a     = pm.Normal("a", mu=1, sigma=0.1, shape=2)
    b     = pm.Normal("b", mu=0, sigma=0.3, shape=2)
    sigma = pm.Exponential("sigma", lam=1)
    mu = a[cid] + b[cid] * (dd["rugged_std"].to_numpy() - 0.215)
    pm.Normal("log_gdp_std", mu=mu, sigma=sigma, observed=dd["log_gdp_std"].to_numpy())
    idata_m83 = fit_laplace(draws=10_000, random_seed=1) # Seed added to ensure reproducibility of results

# Code 8.14 — posterior summary
print(az.summary(idata_m83, var_names=["a", "b", "sigma"], ci_prob=0.89, round_to=2, kind="stats"))

# Code 8.15 — compare models m8.1, m8.2, and m8.3
with model_m83:
    pm.compute_log_likelihood(idata_m83)
print(az.compare({"m8.1": idata_m81, "m8.2": idata_m82, "m8.3": idata_m83}, var_name="log_gdp_std"))

# Code 8.16 — plot PSIS Pareto k values
# The dashed line at 0.5 marks the threshold above which the Pareto-k diagnostic warns that the LOO
# estimate for that observation is unreliable.
loo_m83 = az.loo(idata_m83, var_name="log_gdp_std", pointwise=True)
fig, ax = plt.subplots()
ax.scatter(range(len(loo_m83.pareto_k)), loo_m83.pareto_k, s=12)
ax.axhline(0.5, linestyle="--", color="black", linewidth=0.8)
ax.set_xlabel("observation index")
ax.set_ylabel("Pareto k")
plt.tight_layout()
plt.show()


### 8.1.4. Plotting the interaction ###

# Code 8.17 — plot the posterior predictions for m8.3. This recreates Figure 8.5
post_m83 = idata_m83.posterior.ds.stack(sample=("chain", "draw"))
a_s83 = post_m83["a"].values   # (2, n_samples)
b_s83 = post_m83["b"].values   # (2, n_samples)

mu_africa83 = a_s83[0] + b_s83[0] * (rugged_seq[:, None] - 0.215)
mu_notafrica83 = a_s83[1] + b_s83[1] * (rugged_seq[:, None] - 0.215)

mu_africa83_mean = mu_africa83.mean(axis=1)
mu_notafrica83_mean = mu_notafrica83.mean(axis=1)
mu_africa83_ci = np.quantile(mu_africa83, [0.015, 0.985], axis=1)
mu_notafrica83_ci = np.quantile(mu_notafrica83, [0.015, 0.985], axis=1)

fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

ax = axes[0]
ax.scatter(africa["rugged_std"], africa["log_gdp_std"], color="steelblue", s=12)
ax.plot(rugged_seq, mu_africa83_mean, color="steelblue", linewidth=2)
ax.fill_between(rugged_seq, mu_africa83_ci[0], mu_africa83_ci[1], color="steelblue", alpha=0.3)
ax.set_xlim(0, 1)
ax.set_xlabel("ruggedness (standardized)")
ax.set_ylabel("log GDP (as proportion of mean)")
ax.set_title("African nations")

ax = axes[1]
ax.scatter(notafrica["rugged_std"], notafrica["log_gdp_std"], facecolors="none", edgecolors="black", s=12)
ax.plot(rugged_seq, mu_notafrica83_mean, color="black", linewidth=2)
ax.fill_between(rugged_seq, mu_notafrica83_ci[0], mu_notafrica83_ci[1], color="black", alpha=0.2)
ax.set_xlim(0, 1)
ax.set_xlabel("ruggedness (standardized)")
ax.set_title("Non-African nations")

plt.tight_layout()
plt.show()


### 8.2 Symmetry of interactions ###

# Code 8.18 — difference between nation inside and outside of Africa with constant ruggedness
# Included is code to plot the delta, recreating Figure 8.6
rugged_seq = np.linspace(-0.2, 1.2, 30)
muA = a_s83[0] + b_s83[0] * (rugged_seq[:, None] - 0.215)  # (30, n_samples)
muN = a_s83[1] + b_s83[1] * (rugged_seq[:, None] - 0.215)
delta = muA - muN

delta_mean = delta.mean(axis=1)
delta_ci   = np.quantile(delta, [0.015, 0.985], axis=1)

fig, ax = plt.subplots()
ax.axhline(0, linestyle="--", color="black", linewidth=0.8)
ax.plot(rugged_seq, delta_mean, color="black")
ax.fill_between(rugged_seq, delta_ci[0], delta_ci[1], color="gray", alpha=0.3)
ax.set_xlim(0, 1)
ax.set_xlabel("ruggedness (standardized)")
ax.set_ylabel("expected difference log GDP (Africa - non-Africa)")
plt.tight_layout()
plt.show()
