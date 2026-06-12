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

d: pd.DataFrame = pd.read_csv("WaffleDivorce.csv", sep=";")

# Standardize the variables
#    Note: scipy.stats.zscore() is not used here because we need to keep
#    track of the mean and std for later un-standardization of predictions
#    The book does not un-standardize the predictions, but we will do so
#    for better interpretability
mean_A, std_A = d["MedianAgeMarriage"].mean(), d["MedianAgeMarriage"].std(ddof=1)
mean_M, std_M = d["Marriage"].mean(),           d["Marriage"].std(ddof=1)
mean_D, std_D = d["Divorce"].mean(),             d["Divorce"].std(ddof=1)
d["A"] = (d["MedianAgeMarriage"] - mean_A) / std_A
d["M"] = (d["Marriage"]          - mean_M) / std_M
d["D"] = (d["Divorce"]           - mean_D) / std_D

# Simulate prior predictive lines over A in [-2, 2]
rng = np.random.default_rng(0)
n_lines = 50
A_seq = np.array([-2, 2])
prior_a     = rng.normal(loc=0,   scale=0.2, size=n_lines)
prior_bA    = rng.normal(loc=0,   scale=0.5, size=n_lines)
fig, ax = plt.subplots()
for i in range(n_lines):
    mu_prior = prior_a[i] + prior_bA[i] * A_seq
    ax.plot(A_seq, mu_prior, color="black", alpha=0.4)
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_xlabel("Median age marriage (standardized)")
ax.set_ylabel("Divorce rate (standardized)")
ax.set_title("Prior predictive simulation")
plt.tight_layout()
plt.show()

# Fit the model using Laplace approximation to obtain the model shown
# in the righthanded plot in Figure 5.2
with pm.Model() as model:
    a    = pm.Normal("a", mu=0, sigma=0.2)
    bA    = pm.Normal("bA", mu=0, sigma=0.5)
    sigma = pm.Exponential("sigma", lam=1)
    mu = a + bA * d["A"]
    pm.Normal("D", mu=mu, sigma=sigma, observed=d["D"].to_numpy())
    idata_m51 = fit_laplace(draws=10_000)

# Extract posterior samples for a and bA with shape (10000,)
sample_a  = idata_m51.posterior["a"].to_numpy().ravel()
sample_bA = idata_m51.posterior["bA"].to_numpy().ravel()

# mu[i, j] = predicted mean for draw i at A_seq[j] with shape (10000, 30)
A_seq = np.linspace(-3, 3.2, 30)
mu_post = sample_a[:, None] + sample_bA[:, None] * A_seq[None, :]
mu_mean = mu_post.mean(axis=0)
mu_pi   = np.percentile(mu_post, [5.5, 94.5], axis=0)  # 89% PI, shape (2, 30)

# Build A_seq in original units, standardize for model, then un-standardize predictions
A_seq_orig = np.linspace(d["MedianAgeMarriage"].min(), d["MedianAgeMarriage"].max(), 30)
A_seq_std  = (A_seq_orig - mean_A) / std_A
mu_post_orig = (sample_a[:, None] + sample_bA[:, None] * A_seq_std[None, :]) * std_D + mean_D
mu_mean_orig = mu_post_orig.mean(axis=0)
mu_pi_orig   = np.percentile(mu_post_orig, [5.5, 94.5], axis=0)

# Plot the data and posterior predictions in original units
fig, ax = plt.subplots()
ax.scatter(d["MedianAgeMarriage"], d["Divorce"], color="royalblue")
ax.plot(A_seq_orig, mu_mean_orig, color="black", linewidth=2)
ax.fill_between(A_seq_orig, mu_pi_orig[0], mu_pi_orig[1], color="black", alpha=0.2)
ax.set_xlabel("Median age marriage (years)")
ax.set_ylabel("Divorce rate (per 1000 adults)")
ax.set_title("Posterior predictions: divorce ~ median age at marriage")
plt.tight_layout()
plt.show()

# Fit the model using Laplace approximation to obtain the model shown
# in the lefthanded plot in Figure 5.2 (plot not included in the book,
# so won't plot it as well)
with pm.Model() as model:
    a    = pm.Normal("a", mu=0, sigma=0.2)
    bM    = pm.Normal("bM", mu=0, sigma=0.5)
    sigma = pm.Exponential("sigma", lam=1)
    mu = a + bM * d["M"]
    pm.Normal("D", mu=mu, sigma=sigma, observed=d["D"].to_numpy())
    idata_m52 = fit_laplace(draws=10_000)

# Laplace approximation with two predictors (A and M)
with pm.Model() as model:
    a    = pm.Normal("a", mu=0, sigma=0.2)
    bA   = pm.Normal("bA", mu=0, sigma=0.5)
    bM   = pm.Normal("bM", mu=0, sigma=0.5)
    sigma = pm.Exponential("sigma", lam=1)
    mu = a + bA * d["A"] + bM * d["M"]
    pm.Normal("D", mu=mu, sigma=sigma, observed=d["D"].to_numpy())
    idata_m53 = fit_laplace(draws=10_000)

print(az.summary(idata_m53, var_names=["a", "bM", "bA", "sigma"], hdi_prob=0.89, round_to=2, kind="stats"))

# Coeftab plot: compare bA and bM across models
az.plot_forest(
    [idata_m51, idata_m52, idata_m53],
    model_names=["m5.1 (A only)", "m5.2 (M only)", "m5.3 (A + M)"],
    var_names=["bA", "bM"],
    combined=True,
    hdi_prob=0.89,
    figsize=(7, 3),
)
plt.axvline(0, color="black", linestyle="--", linewidth=0.8)
plt.title("Coefficient comparison: m5.1 vs m5.2 vs m5.3")
plt.tight_layout()
plt.show()


### Predictor residual plots (section 5.1.5.1) ###

# Laplace approximation for the model where marriage rate M
# is explained by median age at marriage A
with pm.Model() as model:
    a    = pm.Normal("a", mu=0, sigma=0.2)
    bAM   = pm.Normal("bAM", mu=0, sigma=0.5)
    sigma = pm.Exponential("sigma", lam=1)
    mu = a + bAM * d["A"]
    pm.Normal("M", mu=mu, sigma=sigma, observed=d["M"].to_numpy())
    idata_m54_1 = fit_laplace(draws=10_000)

# Laplace approximation for the model where marriage rate A
# is explained by median age at marriage M
with pm.Model() as model:
    a    = pm.Normal("a", mu=0, sigma=0.2)
    bAM   = pm.Normal("bAM", mu=0, sigma=0.5)
    sigma = pm.Exponential("sigma", lam=1)
    mu = a + bAM * d["M"]
    pm.Normal("A", mu=mu, sigma=sigma, observed=d["A"].to_numpy())
    idata_m54_2 = fit_laplace(draws=10_000)

# Compute residuals: observed M minus posterior mean of predicted M
sample_a_m54_1   = idata_m54_1.posterior["a"].to_numpy().ravel()
sample_bAM_m54_1 = idata_m54_1.posterior["bAM"].to_numpy().ravel()
mu_post_m54_1 = sample_a_m54_1[:, None] + sample_bAM_m54_1[:, None] * d["A"].to_numpy()[None, :]
mu_mean_m54_1 = mu_post_m54_1.mean(axis=0)   # shape (n_obs,)
mu_resid_m54_1    = d["M"].to_numpy() - mu_mean_m54_1

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

# Not happy about using yet another module (statsmodels) here, but I could not find this functionality in the other modules
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


### Posterior prediction plots (section 5.1.5.2) ###

# Posterior predictions on the original data using the multivariate model
sample_a_m53    = idata_m53.posterior["a"].to_numpy().ravel()
sample_bA_m53   = idata_m53.posterior["bA"].to_numpy().ravel()
sample_bM_m53   = idata_m53.posterior["bM"].to_numpy().ravel()
sample_sig_m53  = idata_m53.posterior["sigma"].to_numpy().ravel()

# Posterior samples of mu for each observed case
mu_m53 = (sample_a_m53[:, None]
          + sample_bA_m53[:, None] * d["A"].to_numpy()[None, :]
          + sample_bM_m53[:, None] * d["M"].to_numpy()[None, :])   # shape (10000, n_obs)
mu_mean_m53 = mu_m53.mean(axis=0)                                  # shape (n_obs)
mu_pi_m53   = np.percentile(mu_m53, [5.5, 94.5], axis=0)           # shape (2, n_obs)

# Add sigma noise to each mu draw to simulate observations
rng_sim = np.random.default_rng(0)
D_sim   = rng_sim.normal(loc=mu_m53, scale=sample_sig_m53[:, None])  # shape (10000, n_obs)
D_pi    = np.percentile(D_sim, [5.5, 94.5], axis=0)                  # shape (2, n_obs)

# Plot predicted vs observed divorce rate (Figure 5.5 equivalent)
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


### Counterfactual plots (section 5.1.5.3) ###

# Joint model: A -> D <- M and A -> M, fit simultaneously with one Laplace approximation.
# PyMC supports multiple observed variables in a single model — their log-likelihoods are
# summed jointly, which is should be identical to what quap() does with multiple likelihood
# terms.
with pm.Model() as model:
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

print(az.summary(idata_m53_A, var_names=["a", "bA", "bM", "sigma", "aM", "bAM", "sigma_M"],
                 hdi_prob=0.89, round_to=2, kind="stats"))

# Extract posterior samples from the joint model
s_aM      = idata_m53_A.posterior["aM"].to_numpy().ravel()       # shape (10000,)
s_bAM     = idata_m53_A.posterior["bAM"].to_numpy().ravel()
s_sigma_M = idata_m53_A.posterior["sigma_M"].to_numpy().ravel()
s_a       = idata_m53_A.posterior["a"].to_numpy().ravel()
s_bA      = idata_m53_A.posterior["bA"].to_numpy().ravel()
s_bM      = idata_m53_A.posterior["bM"].to_numpy().ravel()
s_sigma   = idata_m53_A.posterior["sigma"].to_numpy().ravel()

# Simulate M from A, then D from simulated M and A (vars=c("M","D") in R)
# Both have shape (10000, 30) — one row per posterior draw, one column per A_seq value
A_seq = np.linspace(-2, 2, 30)
rng_cf = np.random.default_rng(0)
mu_M_cf = s_aM[:, None] + s_bAM[:, None] * A_seq[None, :]
sim_M   = rng_cf.normal(loc=mu_M_cf, scale=s_sigma_M[:, None])

mu_D_cf = s_a[:, None] + s_bA[:, None] * A_seq[None, :] + s_bM[:, None] * sim_M
sim_D   = rng_cf.normal(loc=mu_D_cf, scale=s_sigma[:, None])

D_cf_mean = sim_D.mean(axis=0)
D_cf_pi   = np.percentile(sim_D, [5.5, 94.5], axis=0)

M_cf_mean = sim_M.mean(axis=0)
M_cf_pi   = np.percentile(sim_M, [5.5, 94.5], axis=0)

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

# Counterfactual: expected causal effect of increasing median age at
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

# Counterfactual: effect of manipulating M directly, holding A fixed at 0 (its standardized mean)
# M is set directly, not simulated
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
