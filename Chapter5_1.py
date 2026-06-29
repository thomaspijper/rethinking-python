""" Bayesian modeling of divorce rate as a function of median age at
marriage and marriage rate, using Laplace approximation. The code
also includes residual plots, posterior predictive checks, and
counterfactual simulations.

Adapted from Rethinking Statistics 2nd edition, Chapter 5.1.
"""
import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import statsmodels.api as sm

from pymc_extras.inference import fit_laplace


# Code 5.1 — Read in the data and standardize the variables
#    Note: scipy.stats.zscore() is not used here because we need to keep
#    track of the mean and std for later un-standardization of predictions
#    The book does not un-standardize the predictions, but we will do so
#    for better interpretability
d: pd.DataFrame = pd.read_csv("WaffleDivorce.csv", sep=";")
mean_A, std_A = d["MedianAgeMarriage"].mean(), d["MedianAgeMarriage"].std(ddof=1)
mean_M, std_M = d["Marriage"].mean(),           d["Marriage"].std(ddof=1)
mean_D, std_D = d["Divorce"].mean(),             d["Divorce"].std(ddof=1)
d["A"] = (d["MedianAgeMarriage"] - mean_A) / std_A
d["M"] = (d["Marriage"]          - mean_M) / std_M
d["D"] = (d["Divorce"]           - mean_D) / std_D

# Code 5.2 — Standard deviation of median age at marriage
print(std_A)

# Code 5.3 — Fit the model using Laplace approximation
with pm.Model() as model_m51:
    a     = pm.Normal("a", mu=0, sigma=0.2)
    bA    = pm.Normal("bA", mu=0, sigma=0.5)
    sigma = pm.Exponential("sigma", lam=1)
    mu = a + bA * d["A"]
    pm.Normal("D", mu=mu, sigma=sigma, observed=d["D"].to_numpy())
    idata_m51 = fit_laplace(draws=10_000)
    # extract.prior(m5.1) — draw 50 samples from the prior distributions,

# Code 5.4 — plotting the priors, recreating Figure 5.3
with model_m51:
    idata_m51_prior = pm.sample_prior_predictive(draws=50, random_seed=10)
prior_a = idata_m51_prior.prior["a"].values.ravel()
prior_bA = idata_m51_prior.prior["bA"].values.ravel()
A_seq = np.array([-2, 2])
fig, ax = plt.subplots()
for i in range(50):
    mu_prior_line = prior_a[i] + prior_bA[i] * A_seq
    ax.plot(A_seq, mu_prior_line, color="black", alpha=0.4)
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_xlabel("Median age marriage (standardized)")
ax.set_ylabel("Divorce rate (standardized)")
ax.set_title("Prior predictive simulation")
plt.tight_layout()
plt.show()

# Code 5.5 — plot the posterior predictions, recreating the righthand plot in Figure 5.2
# A_seq <- seq(from=-3, to=3.2, length.out=30)
# mu    <- link(m5.1, data=list(A=A_seq))
# mu.mean <- apply(mu, 2, mean)
# mu.PI   <- apply(mu, 2, PI)
sample_a  = idata_m51.posterior["a"].to_numpy().ravel()
sample_bA = idata_m51.posterior["bA"].to_numpy().ravel()
A_seq     = np.linspace(-3, 3.2, 30)
mu_post   = sample_a[:, None] + sample_bA[:, None] * A_seq[None, :]  # shape (10000, 30)
mu_mean   = mu_post.mean(axis=0)
mu_pi     = np.quantile(mu_post, [0.055, 0.945], axis=0)             # 89% PI, shape (2, 30)

# Convert to original units for plotting
A_seq_orig  = A_seq * std_A + mean_A
mu_mean_orig = mu_mean * std_D + mean_D
mu_pi_orig   = mu_pi   * std_D + mean_D

# plot(D ~ A, data=d) + lines(A_seq, mu.mean) + shade(mu.PI, A_seq)
fig, ax = plt.subplots()
ax.scatter(d["MedianAgeMarriage"], d["Divorce"], color="royalblue")
ax.plot(A_seq_orig, mu_mean_orig, color="black", linewidth=2)
ax.fill_between(A_seq_orig, mu_pi_orig[0], mu_pi_orig[1], color="black", alpha=0.2)
ax.set_xlabel("Median age at marriage (years)")
ax.set_ylabel("Divorce rate (per 1000 adults)")
ax.set_title("Divorce ~ Median age at marriage (m5.1)")
plt.tight_layout()
plt.show()


# Code 5.6 — Fit the model to obtain the model for the lefthanded plot in Figure 5.2.
with pm.Model() as model_m52:
    a    = pm.Normal("a", mu=0, sigma=0.2)
    bM    = pm.Normal("bM", mu=0, sigma=0.5)
    sigma = pm.Exponential("sigma", lam=1)
    mu = a + bM * d["M"]
    pm.Normal("D", mu=mu, sigma=sigma, observed=d["D"].to_numpy())
    idata_m52 = fit_laplace(draws=10_000)

# Plotting the model (not shown in the book)
sample_a  = idata_m52.posterior["a"].to_numpy().ravel()
sample_bM = idata_m52.posterior["bM"].to_numpy().ravel()
M_seq     = np.linspace(-3, 3.2, 30)
mu_post   = sample_a[:, None] + sample_bM[:, None] * M_seq[None, :]
mu_mean   = mu_post.mean(axis=0)
mu_pi     = np.quantile(mu_post, [0.055, 0.945], axis=0)

M_seq_orig   = M_seq  * std_M + mean_M
mu_mean_orig = mu_mean * std_D + mean_D
mu_pi_orig   = mu_pi   * std_D + mean_D

fig, ax = plt.subplots()
ax.scatter(d["Marriage"], d["Divorce"], color="royalblue")
ax.plot(M_seq_orig, mu_mean_orig, color="black", linewidth=2)
ax.fill_between(M_seq_orig, mu_pi_orig[0], mu_pi_orig[1], color="black", alpha=0.2)
ax.set_xlabel("Marriage rate (per 1000 adults)")
ax.set_ylabel("Divorce rate (per 1000 adults)")
ax.set_title("Divorce ~ Marriage rate (m5.2)")
plt.tight_layout()
plt.show()


### 5.1.4 Approximating the posterior ###

# Code 5.10 — Laplace approximation with two predictors (A and M)
with pm.Model() as model_m53:
    a    = pm.Normal("a", mu=0, sigma=0.2)
    bA   = pm.Normal("bA", mu=0, sigma=0.5)
    bM   = pm.Normal("bM", mu=0, sigma=0.5)
    sigma = pm.Exponential("sigma", lam=1)
    mu = a + bA * d["A"] + bM * d["M"]
    pm.Deterministic("mu", mu)
    pm.Normal("D", mu=mu, sigma=sigma, observed=d["D"].to_numpy())
    idata_m53 = fit_laplace(draws=10_000)
print(az.summary(idata_m53, var_names=["a", "bM", "bA", "sigma"], ci_prob=0.89, round_to=2, kind="stats"))

# Code 5.11 — coeftab plot: compare bA and bM across models
# Normally, one would use az.plot_forest() to create a forest plot, but the book's
# layout (model names as row labels) appears not compatible with ArviZ 1.x, so we
# build the plot manually.
models = {
    "m5.1 (A only)": idata_m51,
    "m5.2 (M only)": idata_m52,
    "m5.3 (A + M)": idata_m53,
}
params = ["bA", "bM"]
fig, ax = plt.subplots(figsize=(7, 4))
colors = {"bA": "steelblue", "bM": "tomato"}
y_pos = 0
yticks, ylabels = [], []
for param in params:
    for model_name, idata in models.items():
        post = idata.posterior.ds if hasattr(idata.posterior, "ds") else idata.posterior
        if param not in post:
            samples = np.full(100, np.nan)
        else:
            samples = post[param].values.ravel()
        if np.all(np.isnan(samples)):
            yticks.append(y_pos); ylabels.append(f"{model_name}")
            y_pos += 1
            continue
        hdi89 = az.hdi(idata, var_names=[param], prob=0.89)[param].values
        mean_val = float(np.mean(samples))
        ax.plot([hdi89[0], hdi89[1]], [y_pos, y_pos], color=colors[param], linewidth=2)
        ax.plot(mean_val, y_pos, "o", color="white", markeredgecolor=colors[param],
                markersize=5, zorder=5)
        yticks.append(y_pos); ylabels.append(f"{model_name}")
        y_pos += 1
    y_pos += 0.5  # gap between parameters
ax.set_yticks(yticks)
ax.set_yticklabels(ylabels, fontsize=9)
ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
# legend for parameter colors
from matplotlib.lines import Line2D
legend_elements = [Line2D([0], [0], color=colors[p], linewidth=3, label=p) for p in params]
ax.legend(handles=legend_elements, loc="lower right")
ax.set_title("Coefficient comparison: m5.1 vs m5.2 vs m5.3")
ax.set_xlabel("Posterior estimate (standardized)")
plt.tight_layout()
plt.show()


### 5.1.5.1 Predictor residual plots ###

# Code 5.13 — Laplace approximation for the model where marriage rate M
# is explained by median age at marriage A
with pm.Model() as model_m54_1:
    a    = pm.Normal("a", mu=0, sigma=0.2)
    bAM   = pm.Normal("bAM", mu=0, sigma=0.5)
    sigma = pm.Exponential("sigma", lam=1)
    mu = a + bAM * d["A"]
    pm.Normal("M", mu=mu, sigma=sigma, observed=d["M"].to_numpy())
    idata_m54_1 = fit_laplace(draws=10_000)

# Code 5.14 — compute residuals: observed M minus posterior mean of predicted M
sample_a_m54_1   = idata_m54_1.posterior["a"].to_numpy().ravel()
sample_bAM_m54_1 = idata_m54_1.posterior["bAM"].to_numpy().ravel()
mu_post_m54_1 = sample_a_m54_1[:, None] + sample_bAM_m54_1[:, None] * d["A"].to_numpy()[None, :]
mu_mean_m54_1 = mu_post_m54_1.mean(axis=0)   # shape (n_obs,)
mu_resid_m54_1    = d["M"].to_numpy() - mu_mean_m54_1


# The following code blocks are not in the book, but are used to recreate Figure 5.4
#
# Laplace approximation for the model where marriage rate A
# is explained by median age at marriage M
with pm.Model() as model_m54_2:
    a    = pm.Normal("a", mu=0, sigma=0.2)
    bAM   = pm.Normal("bAM", mu=0, sigma=0.5)
    sigma = pm.Exponential("sigma", lam=1)
    mu = a + bAM * d["M"]
    pm.Normal("A", mu=mu, sigma=sigma, observed=d["A"].to_numpy())
    idata_m54_2 = fit_laplace(draws=10_000)

# Compute residuals: observed A minus posterior mean of predicted A
sample_a_m54_2   = idata_m54_2.posterior["a"].to_numpy().ravel()
sample_bAM_m54_2 = idata_m54_2.posterior["bAM"].to_numpy().ravel()
mu_post_m54_2 = sample_a_m54_2[:, None] + sample_bAM_m54_2[:, None] * d["M"].to_numpy()[None, :]
mu_mean_m54_2 = mu_post_m54_2.mean(axis=0)   # shape (n_obs,)
mu_resid_m54_2    = d["A"].to_numpy() - mu_mean_m54_2

# Plot A vs M with the model line, residuals, and vertical connectors
A_vals = d["A"].to_numpy()
M_vals = d["M"].to_numpy()
A_line_1 = np.linspace(A_vals.min(), A_vals.max(), 100)
M_line_1 = sample_a_m54_1.mean() + sample_bAM_m54_1.mean() * A_line_1
M_line_2 = np.linspace(M_vals.min(), M_vals.max(), 100)
A_line_2 = sample_a_m54_2.mean() + sample_bAM_m54_2.mean() * M_line_2

# Plot residuals against divorce rate with a linear regression overlay
D_vals = d["D"].to_numpy()
ols_fit_1 = sm.OLS(D_vals, sm.add_constant(mu_resid_m54_1)).fit()
ols_fit_2 = sm.OLS(D_vals, sm.add_constant(mu_resid_m54_2)).fit()
resid_line_1 = np.linspace(mu_resid_m54_1.min(), mu_resid_m54_1.max(), 100)
resid_line_2 = np.linspace(mu_resid_m54_2.min(), mu_resid_m54_2.max(), 100)

# Not happy about using yet another module (statsmodels) here, but I could not find this
# functionality in the other modules
ci_1 = ols_fit_1.get_prediction(sm.add_constant(resid_line_1)).summary_frame(alpha=1 - 0.89)
ci_2 = ols_fit_2.get_prediction(sm.add_constant(resid_line_2)).summary_frame(alpha=1 - 0.89)

fig, axes = plt.subplots(2, 2, figsize=(10, 8))

# Top left plot: M vs A with model line and residual connectors
ax = axes[0, 0]
ax.plot(A_line_1, M_line_1, color="black", linewidth=1.5)
for a_i, m_i, m_hat_i in zip(A_vals, M_vals, mu_mean_m54_1):
    ax.plot([a_i, a_i], [m_hat_i, m_i], color="gray", linewidth=0.8)
ax.scatter(A_vals, M_vals, color="royalblue", zorder=3)
ax.set_xlabel("Age at marriage (standardized)")
ax.set_ylabel("Marriage rate (standardized)")
ax.set_title("Residuals of M ~ A")

# Bottom left plot: Divorce rate vs residuals of M ~ A with regression line and CI
ax = axes[1, 0]
ax.scatter(mu_resid_m54_1, D_vals, color="royalblue")
ax.plot(resid_line_1, ci_1["mean"], color="black", linewidth=1.5)
ax.fill_between(resid_line_1, ci_1["mean_ci_lower"], ci_1["mean_ci_upper"], color="black", alpha=0.2)
ax.axvline(0, color="gray", linestyle="--", linewidth=0.8)
ax.set_xlabel("Residual marriage rate (M | A)")
ax.set_ylabel("Divorce rate (standardized)")
ax.set_title("Divorce rate vs. marriage rate residuals")

# Top right plot: A vs M with model line and residual connectors
ax = axes[0, 1]
ax.plot(M_line_2, A_line_2, color="black", linewidth=1.5)
for m_i, a_i, a_hat_i in zip(M_vals, A_vals, mu_mean_m54_2):
    ax.plot([m_i, m_i], [a_hat_i, a_i], color="gray", linewidth=0.8)
ax.scatter(M_vals, A_vals, color="royalblue", zorder=3)
ax.set_xlabel("Marriage rate (standardized)")
ax.set_ylabel("Age at marriage (standardized)")
ax.set_title("Residuals of A ~ M")

# Bottom right plot: Divorce rate vs residuals of A ~ M with regression line and CI
ax = axes[1, 1]
ax.scatter(mu_resid_m54_2, D_vals, color="royalblue")
ax.plot(resid_line_2, ci_2["mean"], color="black", linewidth=1.5)
ax.fill_between(resid_line_2, ci_2["mean_ci_lower"], ci_2["mean_ci_upper"], color="black", alpha=0.2)
ax.axvline(0, color="gray", linestyle="--", linewidth=0.8)
ax.set_xlabel("Residual age at marriage (A | M)")
ax.set_ylabel("Divorce rate (standardized)")
ax.set_title("Divorce rate vs. age residuals")

plt.tight_layout()
plt.show()


### 5.1.5.2 Posterior prediction plots  ###

# Code 5.15 — simulate posterior predictions for the model with both predictors (m5.3)
# Equivalent to R's link(m5.3) + sim(m5.3) with no new data: use PyMC's posterior
# predictive sampling on the original observed data.
with model_m53:
    ppc_m53 = pm.sample_posterior_predictive(idata_m53, var_names=["mu", "D"], random_seed=0)

# mu samples: (chain, draw, n_obs) -> (draws, n_obs)
mu_m53 = ppc_m53.posterior_predictive["mu"].values
mu_m53 = mu_m53.reshape(-1, mu_m53.shape[-1]) # shape (10000, n_obs)
mu_mean_m53 = mu_m53.mean(axis=0)
mu_pi_m53   = np.percentile(mu_m53, [5.5, 94.5], axis=0)

# simulated observations (sigma noise included)
D_sim = ppc_m53.posterior_predictive["D"].values
D_sim = D_sim.reshape(-1, D_sim.shape[-1]) # shape (10000, n_obs)
D_pi  = np.percentile(D_sim, [5.5, 94.5], axis=0)

# Code 5.16 — plot predicted vs observed divorce rate (Figure 5.5 equivalent)
D_obs = d["D"].to_numpy()
fig, ax = plt.subplots()
ax.scatter(D_obs, mu_mean_m53, color="royalblue")
for i in range(len(D_obs)):
    ax.plot([D_obs[i], D_obs[i]], [mu_pi_m53[0, i], mu_pi_m53[1, i]], color="royalblue", linewidth=0.8)
lim = (min(D_obs.min(), mu_pi_m53.min()), max(D_obs.max(), mu_pi_m53.max()))
ax.plot(lim, lim, linestyle="--", color="black", linewidth=0.8)
ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel("Observed divorce")
ax.set_ylabel("Predicted divorce")
plt.tight_layout()
plt.show()


### 5.1.5.3 Counterfactual plots ###

# Code 5.19 — joint model: A -> D <- M and A -> M, fit simultaneously.
# PyMC supports multiple observed variables in a single model — their log-likelihoods are
# summed jointly, which is should be identical to what quap() does with multiple likelihood
# terms.
with pm.Model() as model_53_A:
    # A -> D <- M
    a     = pm.Normal("a",     mu=0, sigma=0.2)
    bM    = pm.Normal("bM",    mu=0, sigma=0.5)
    bA    = pm.Normal("bA",    mu=0, sigma=0.5)
    sigma = pm.Exponential("sigma", lam=1)
    mu  = a + bM * d["M"] + bA * d["A"]
    pm.Normal("D", mu=mu, sigma=sigma, observed=d["D"].to_numpy())

    # A -> M
    aM      = pm.Normal("aM",     mu=0, sigma=0.2)
    bAM     = pm.Normal("bAM",    mu=0, sigma=0.5)
    sigma_M = pm.Exponential("sigma_M", lam=1)
    mu_M    = aM + bAM * d["A"]
    pm.Normal("M", mu=mu_M, sigma=sigma_M, observed=d["M"].to_numpy())

    idata_m53_A = fit_laplace(draws=10_000)

# Code 5.20 and 5.21 — simulate the effect of manipulating A
#
# using pm.sample_posterior_predictive doesn't fit naturally here. The counterfactual
# requires sequential simulation — D's mean must use the simulated M, not the observed
# M. Since model_53_A was fit with observed M feeding into D's likelihood, PyMC PPC would
# use the observed M values for D rather than a fresh draw, breaking the causal chain.
#
# Extract posterior samples from the joint model
s_aM      = idata_m53_A.posterior["aM"].to_numpy().ravel()       # shape (10000,)
s_bAM     = idata_m53_A.posterior["bAM"].to_numpy().ravel()
s_sigma_M = idata_m53_A.posterior["sigma_M"].to_numpy().ravel()
s_a       = idata_m53_A.posterior["a"].to_numpy().ravel()
s_bA      = idata_m53_A.posterior["bA"].to_numpy().ravel()
s_bM      = idata_m53_A.posterior["bM"].to_numpy().ravel()
s_sigma   = idata_m53_A.posterior["sigma"].to_numpy().ravel()

# Simulate M from A, then D from simulated M and A (vars=c("M","D") in R)
# shape (10000, 30) — one row per posterior draw, one column per A_seq value
A_seq = np.linspace(-2, 2, 30)
rng_cf = np.random.default_rng(0)
sim_M = rng_cf.normal(s_aM[:,None] + s_bAM[:,None] * A_seq, s_sigma_M[:,None])
sim_D = rng_cf.normal(s_a[:,None] + s_bA[:,None] * A_seq + s_bM[:,None] * sim_M, s_sigma[:,None])

D_cf_mean, D_cf_pi = sim_D.mean(0), np.percentile(sim_D, [5.5, 94.5], axis=0)
M_cf_mean, M_cf_pi = sim_M.mean(0), np.percentile(sim_M, [5.5, 94.5], axis=0)

# Code 5.22 — plot the counterfactual curves for D and M as A is manipulated, recreating Figure 5.6
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

ax = axes[0]
ax.plot(A_seq, D_cf_mean, color="black")
ax.fill_between(A_seq, D_cf_pi[0], D_cf_pi[1], color="black", alpha=0.2)
ax.set_ylim(-2, 2)
ax.set_xlabel("Manipulated A")
ax.set_ylabel("Counterfactual D")
ax.set_title("Total counterfactual effect of A on D")

ax = axes[1]
ax.plot(A_seq, M_cf_mean, color="black")
ax.fill_between(A_seq, M_cf_pi[0], M_cf_pi[1], color="black", alpha=0.2)
ax.set_ylim(-2, 2)
ax.set_xlabel("Manipulated A")
ax.set_ylabel("Counterfactual M")
ax.set_title("Counterfactual effect of A on M")

plt.tight_layout()
plt.show()

# Code 5.23 — counterfactual: expected causal effect of increasing median age at
# marriage from age 20 to age 30. Standardize using the same mean/std
# as the original data
A_sim2 = (np.array([20, 30]) - mean_A) / std_A

rng_cf2 = np.random.default_rng(0)
mu_M_cf2 = s_aM[:, None] + s_bAM[:, None] * A_sim2[None, :]
sim_M2   = rng_cf2.normal(loc=mu_M_cf2, scale=s_sigma_M[:, None])  # shape (10000, 2)

mu_D_cf2 = s_a[:, None] + s_bA[:, None] * A_sim2[None, :] + s_bM[:, None] * sim_M2
sim_D2   = rng_cf2.normal(loc=mu_D_cf2, scale=s_sigma[:, None])    # shape (10000, 2)

mean_diff = (sim_D2[:, 1] - sim_D2[:, 0]).mean()
print(f"Mean effect on D of increasing A from 20 to 30: {mean_diff:.4f}")

# Code 5.24 — counterfactual: effect of manipulating M directly, holding A fixed at 0
# (its standardized mean). M is set directly, not simulated. The plot recreates Figure 5.7.
M_seq = np.linspace(-2, 2, 30)
A_fixed = 0.0  # standardized mean of A

rng_cf3 = np.random.default_rng(0)
mu_D_cf3 = s_a[:, None] + s_bA[:, None] * A_fixed + s_bM[:, None] * M_seq[None, :]
sim_D3   = rng_cf3.normal(loc=mu_D_cf3, scale=s_sigma[:, None])  # shape (10000, 30)

D_cf3_mean = sim_D3.mean(axis=0)
D_cf3_pi   = np.percentile(sim_D3, [5.5, 94.5], axis=0)

fig, ax = plt.subplots()
ax.plot(M_seq, D_cf3_mean, color="black")
ax.fill_between(M_seq, D_cf3_pi[0], D_cf3_pi[1], color="black", alpha=0.2)
ax.set_ylim(-2, 2)
ax.set_xlabel("Manipulated M")
ax.set_ylabel("Counterfactual D")
ax.set_title("Total counterfactual effect of M on D")
plt.tight_layout()
plt.show()