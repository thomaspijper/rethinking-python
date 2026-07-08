"""Bayesian models showing the problem with parameters, and how to use information criteria to evaluate models.

Adapted from Rethinking Statistics 2nd edition, Chapter 7.

Because section 7.2 uses models from section 7.1, chapter 7 is presented as one file.
"""

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm

from pymc_extras.inference import fit_laplace
from scipy.special import logsumexp
from scipy.stats import entropy


### 7.1 The problem with parameters ###

# Code 7.1 — build data
sppnames    = ["afarensis", "africanus", "habilis", "boisei",
               "rudolfensis", "ergaster", "sapiens"]
brainvolcc  = [438, 452, 612, 521, 752, 871, 1350]
masskg      = [37.0, 35.5, 34.5, 41.5, 55.5, 61.0, 53.5]
d = pd.DataFrame({"species": sppnames, "brain": brainvolcc, "mass": masskg})

# Code 7.2 — standardise the variables
d["mass_std"]  = (d["mass"] - d["mass"].mean()) / d["mass"].std(ddof=1)
d["brain_std"] = d["brain"] / d["brain"].max()
mass_std = d["mass_std"].to_numpy()

# Code 7.3 — a simple linear model of brain size as a function of mass
with pm.Model() as model_m71:
    a = pm.Normal("a", mu=0.5, sigma=1)
    b = pm.Normal("b", mu=0, sigma=10)
    log_sigma = pm.Normal("log_sigma", mu=0, sigma=1)
    mu = a + b * mass_std
    pm.Normal("brain_std", mu=mu, sigma=pm.math.exp(log_sigma), observed=d["brain_std"].to_numpy())
    idata_m71 = fit_laplace(draws=10_000)

# Code 7.5 — not included, because the next code example does the same, wrapped in a function

# Code 7.6 — function to compute R² from posterior predictive samples
def R2_is_bad(model, idata, y_obs, obs_var):
    """Compute R² from posterior predictive samples. Works for any model.
    Mirrors R2_is_bad() from the book, using pm.sample_posterior_predictive
    as the equivalent of sim().
    """
    with model:
        ppc = pm.sample_posterior_predictive(idata, var_names=[obs_var])
    y_pred = ppc.posterior_predictive[obs_var].values  # shape (chains, draws, n_obs)
    r = y_pred.reshape(-1, len(y_obs)).mean(axis=0) - y_obs
    return 1 - np.var(r) / np.var(y_obs)

# This is the result code 7.5 would give
print(f"R² (m7.1) = {R2_is_bad(model_m71, idata_m71, d['brain_std'].to_numpy(), 'brain_std'):.4f}")

# Code 7.7 — quadratic version of the model
with pm.Model() as model_m72:
    a         = pm.Normal("a", mu=0.5, sigma=1)
    b         = pm.Normal("b", mu=0, sigma=10, shape=2)
    log_sigma = pm.Normal("log_sigma", mu=0, sigma=1)
    mu        = a + b[0]*mass_std + b[1]*mass_std**2
    pm.Normal("brain_std", mu=mu, sigma=pm.math.exp(log_sigma), observed=d["brain_std"].to_numpy())
    idata_m72 = fit_laplace(draws=10_000)

print(f"R² (m7.2) = {R2_is_bad(model_m72, idata_m72, d['brain_std'].to_numpy(), 'brain_std'):.4f}")

# Code 7.8 — higher-order versions of the model
# Cubic
with pm.Model() as model_m73:
    a         = pm.Normal("a", mu=0.5, sigma=1)
    b         = pm.Normal("b", mu=0, sigma=10, shape=3)
    log_sigma = pm.Normal("log_sigma", mu=0, sigma=1)
    mu        = a + b[0]*mass_std + b[1]*mass_std**2 + b[2]*mass_std**3
    pm.Normal("brain_std", mu=mu, sigma=pm.math.exp(log_sigma), observed=d["brain_std"].to_numpy())
    idata_m73 = fit_laplace(draws=10_000)

print(f"R² (m7.3) = {R2_is_bad(model_m73, idata_m73, d['brain_std'].to_numpy(), 'brain_std'):.4f}")

# Quartic
with pm.Model() as model_m74:
    a         = pm.Normal("a", mu=0.5, sigma=1)
    b         = pm.Normal("b", mu=0, sigma=10, shape=4)
    log_sigma = pm.Normal("log_sigma", mu=0, sigma=1)
    mu        = a + b[0]*mass_std + b[1]*mass_std**2 + b[2]*mass_std**3 + b[3]*mass_std**4
    pm.Normal("brain_std", mu=mu, sigma=pm.math.exp(log_sigma), observed=d["brain_std"].to_numpy())
    idata_m74 = fit_laplace(draws=10_000)

print(f"R² (m7.4) = {R2_is_bad(model_m74, idata_m74, d['brain_std'].to_numpy(), 'brain_std'):.4f}")

# Quintic
with pm.Model() as model_m75:
    a         = pm.Normal("a", mu=0.5, sigma=1)
    b         = pm.Normal("b", mu=0, sigma=10, shape=5)
    log_sigma = pm.Normal("log_sigma", mu=0, sigma=1)
    mu        = a + b[0]*mass_std + b[1]*mass_std**2 + b[2]*mass_std**3 + b[3]*mass_std**4 + b[4]*mass_std**5
    pm.Normal("brain_std", mu=mu, sigma=pm.math.exp(log_sigma), observed=d["brain_std"].to_numpy())
    idata_m75 = fit_laplace(draws=10_000)

print(f"R² (m7.5) = {R2_is_bad(model_m75, idata_m75, d['brain_std'].to_numpy(), 'brain_std'):.4f}")

# Sextic
with pm.Model() as model_m76:
    a  = pm.Normal("a", mu=0.5, sigma=1)
    b  = pm.Normal("b", mu=0, sigma=10, shape=6)
    mu = (a + b[0]*mass_std + b[1]*mass_std**2 + b[2]*mass_std**3
            + b[3]*mass_std**4 + b[4]*mass_std**5 + b[5]*mass_std**6)
    pm.Normal("brain_std", mu=mu, sigma=0.001, observed=d["brain_std"].to_numpy())
    idata_m76 = fit_laplace(draws=10_000)

print(f"R² (m7.6) = {R2_is_bad(model_m76, idata_m76, d['brain_std'].to_numpy(), 'brain_std'):.4f}")

# Code 7.10
# Here, we are calculating mu and the CI for the a model, but we wrap it in a function so we
# can use it for each model in order to recreate Figure 7.3

def compute_mu_ci(idata, mass_seq, degree, hdi_prob=0.89):
    """Compute posterior mean and percentile interval of mu over a sequence of mass values.
    Equivalent to l <- link(model, data=list(mass_std=mass_seq)); mu <- apply(l,2,mean); ci <- apply(l,2,PI)
    """
    post = idata.posterior.ds.stack(sample=("chain", "draw"))
    a_s  = post["a"].values                   # (n_samples,)
    b_s  = post["b"].values                   # (n_samples,) or (degree, n_samples) or (n_samples, degree)
    if b_s.ndim == 1:
        b_s = b_s[:, np.newaxis]              # normalise to (n_samples, 1)
    n_samples = len(a_s)
    if b_s.shape[0] != n_samples:             # xarray stacks b_dim first → (degree, n_samples)
        b_s = b_s.T                           # transpose to (n_samples, degree)

    # mu_s shape: (n_seq, n_samples)
    mu_s = np.tile(a_s, (len(mass_seq), 1))
    for k in range(degree):
        mu_s += b_s[:, k] * mass_seq[:, np.newaxis] ** (k + 1)

    alpha = (1 - hdi_prob) / 2
    return (mu_s.mean(axis=1),
            np.percentile(mu_s, alpha * 100, axis=1),
            np.percentile(mu_s, (1 - alpha) * 100, axis=1))

# Sequence in original mass units; convert to standardised for model predictions
mean_mass = d["mass"].mean()
std_mass  = d["mass"].std(ddof=1)
max_brain = d["brain"].max()

mass_seq_orig = np.linspace(d["mass"].min(), d["mass"].max(), 100)
mass_seq      = (mass_seq_orig - mean_mass) / std_mass   # standardised, used by models

models = [
    (model_m71, idata_m71, 1, "m7.1"),
    (model_m72, idata_m72, 2, "m7.2"),
    (model_m73, idata_m73, 3, "m7.3"),
    (model_m74, idata_m74, 4, "m7.4"),
    (model_m75, idata_m75, 5, "m7.5"),
    (model_m76, idata_m76, 6, "m7.6"),
]

fig, axes = plt.subplots(3, 2, figsize=(8, 12))
for ax, (model, idata, degree, title) in zip(axes.flatten(), models):
    mu_mean, mu_lo, mu_hi = compute_mu_ci(idata, mass_seq, degree)
    # Convert predictions back to original units (cc)
    ax.scatter(d["mass"], d["brain"], color="black", s=20, zorder=3)
    ax.plot(mass_seq_orig, mu_mean * max_brain, color="black")
    ax.fill_between(mass_seq_orig, mu_lo * max_brain, mu_hi * max_brain, alpha=0.3, color="gray")
    ax.set_title(title)
    ax.set_xlabel("body mass (kg)")
    ax.set_ylabel("brain volume (cc)")
    ax.set_ylim(200, 1700)

plt.tight_layout()
plt.show()


### 7.2 Entropy and Accuracy ###

### 7.2.2 Information and uncertainty ###

# Code 7.12 — Entropy calculation
# This can be computed with numpy...
p = np.array([0.3, 0.7])
print(-np.sum(p * np.log(p)))

#... or with scipy's entropy function, which defaults to natural log.
print(entropy(p))

# Scipy's entropy function can also compute KL divergence (section 7.2.3) if one provides more than one argument
q = np.array([0.25, 0.75])
print(entropy(p, q))


### 7.2.4 Estimating divergence ###

def lppd(model, idata, obs_var="brain_std"):
    """Compute log-pointwise-predictive-density for each observation.

    Equivalent to lppd() from the rethinking package.
    """
    if "log_likelihood" not in idata:  # makes sure we only compute log likelihood if it hasn't already been computed
        with model:
            pm.compute_log_likelihood(idata)
    ll = idata.log_likelihood[obs_var].values  # (chains, draws, n_obs)
    ll_flat = ll.reshape(-1, ll.shape[-1])      # (n_samples, n_obs)
    return logsumexp(ll_flat, axis=0) - np.log(ll_flat.shape[0])

# Code 7.13 — compute lppd for model m7.1
lppd_m71 = lppd(model_m71, idata_m71)
print(f"lppd values (m7.1) = {lppd_m71.round(4)}")
print(f"lppd (m7.1) = {lppd_m71.sum():.4f}")  # Summing gives the total log-probability score for model and data


### 7.2.5 Scoring the right data ###

# Code 7.15 — sapply(list(m7.1,...,m7.6), function(m) sum(lppd(m))) equivalent
lppd_vals = []
for model, idata, title in [
    (model_m71, idata_m71, "m7.1"),
    (model_m72, idata_m72, "m7.2"),
    (model_m73, idata_m73, "m7.3"),
    (model_m74, idata_m74, "m7.4"),
    (model_m75, idata_m75, "m7.5"),
    (model_m76, idata_m76, "m7.6"),
]:
    lppd_vals.append(lppd(model, idata))

lppd_sums = [np.sum(vals) for vals in lppd_vals]
for name, s in zip(['m7.1', 'm7.2', 'm7.3', 'm7.4', 'm7.5', 'm7.6'], lppd_sums):
    print(f"  {name}: {s:.4f}")

#############################################################################################
# sim_train_test() is mentioned in section 7.2.5 but not used in any code example.
# The code below is equivalent to this function — vectorised over n_sims for performance.
#
# True model: y = X[:,0], where [y, x1, x2, ...] ~ MVN(0, Rho)
# Rho has off-diagonal correlations rho[0]=0.15 (y-x1) and rho[1]=-0.4 (y-x2).
# Model k fits an intercept + k-1 predictors (x1...x_{k-1}) with sigma=1 fixed.
# Training deviance  = -2 * lppd  (log-likelihood at MAP on training data)
# Test deviance      = -2 * sum(Normal log-likelihood of test y given MAP mu)
#
# Finally, we use this recreate the left plot in Figure 7.6

from scipy.stats import norm as sp_norm

def sim_train_test_batch(N, k, n_sims, rng, rho=(0.15, -0.4)):
    """Vectorised version of rethinking's sim_train_test().
    Returns (dev_train, dev_test) each of shape (n_sims,).
    """
    n_dim = max(k, 1 + len(rho))

    # Correlation matrix: y correlated with x1 (rho[0]) and x2 (rho[1])
    Rho = np.eye(n_dim)
    for i, r in enumerate(rho):
        Rho[0, i + 1] = r
        Rho[i + 1, 0] = r

    L = np.linalg.cholesky(Rho)

    # Draw (n_sims, N, n_dim) from MVN(0, Rho)
    Z_train = rng.standard_normal((n_sims, N, n_dim))
    Z_test  = rng.standard_normal((n_sims, N, n_dim))
    X_train = Z_train @ L.T   # (n_sims, N, n_dim)
    X_test  = Z_test  @ L.T

    y_train = X_train[:, :, 0]   # (n_sims, N)
    y_test  = X_test[:, :, 0]

    # Design matrices: intercept + k-1 predictors
    ones = np.ones((n_sims, N, 1))
    if k == 1:
        Xk_train = ones
        Xk_test  = ones
    else:
        Xk_train = np.concatenate([ones, X_train[:, :, 1:k]], axis=2)  # (n_sims, N, k)
        Xk_test  = np.concatenate([ones, X_test[:,  :, 1:k]], axis=2)

    # Batched OLS via normal equations: beta = (X'X)^{-1} X'y
    XtX  = np.einsum('sni,snj->sij', Xk_train, Xk_train)  # (n_sims, k, k)
    Xty  = np.einsum('sni,sn->si',  Xk_train, y_train)    # (n_sims, k)
    beta = np.linalg.solve(XtX, Xty[:, :, np.newaxis]).squeeze(-1)  # (n_sims, k)

    mu_train = np.einsum('sni,si->sn', Xk_train, beta)
    mu_test  = np.einsum('sni,si->sn', Xk_test,  beta)

    # Deviance = -2 * sum of log-likelihoods (sigma=1 fixed)
    dev_train = -2 * np.sum(sp_norm.logpdf(y_train, mu_train, 1), axis=1)
    dev_test  = -2 * np.sum(sp_norm.logpdf(y_test,  mu_test,  1), axis=1)
    return dev_train, dev_test

rng_sim = np.random.default_rng(1)
N_sim   = 20
n_sims  = 10_000
kseq    = range(1, 6)

dev = np.zeros((4, 5))   # rows: mean_train, mean_test, sd_train, sd_test
for i, k in enumerate(kseq):
    print(f"Simulating k={k}...")
    dt, dout = sim_train_test_batch(N_sim, k, n_sims, rng_sim)
    dev[0, i] = dt.mean()
    dev[1, i] = dout.mean()
    dev[2, i] = dt.std()
    dev[3, i] = dout.std()

# Plot: filled blue = in-sample, open black = out-of-sample; bars = ±1 SD
fig, ax = plt.subplots(figsize=(6, 5))
ks = np.arange(1, 6)
ax.scatter(ks,       dev[0], color="#1e90ff", zorder=3, s=30, label="in-sample")
ax.scatter(ks + 0.1, dev[1], facecolors="none", edgecolors="black", zorder=3, s=30, label="out-of-sample")

for i in range(5):
    ax.plot([ks[i],       ks[i]],       [dev[0, i] - dev[2, i], dev[0, i] + dev[2, i]], color="#1e90ff")
    ax.plot([ks[i] + 0.1, ks[i] + 0.1], [dev[1, i] - dev[3, i], dev[1, i] + dev[3, i]], color="black")

ax.set_xlabel("number of parameters")
ax.set_ylabel("deviance")
ax.set_title(f"N={N_sim}")
ax.legend()
plt.tight_layout()
plt.show()
#############################################################################################

### 7.5 Model comparison ###

# Section 7.5 uses models from Chapters 5 and 6.
# We refit them here so this file remains self-contained.

# Read in the data
d: pd.DataFrame = pd.read_csv("WaffleDivorce.csv", sep=";")
mean_A, std_A = d["MedianAgeMarriage"].mean(), d["MedianAgeMarriage"].std(ddof=1)
mean_M, std_M = d["Marriage"].mean(), d["Marriage"].std(ddof=1)
mean_D, std_D = d["Divorce"].mean(), d["Divorce"].std(ddof=1)
d["A"] = (d["MedianAgeMarriage"] - mean_A) / std_A
d["M"] = (d["Marriage"] - mean_M) / std_M
d["D"] = (d["Divorce"] - mean_D) / std_D

# m5.1
with pm.Model() as model_m51:
    a     = pm.Normal("a", mu=0, sigma=0.2)
    bA    = pm.Normal("bA", mu=0, sigma=0.5)
    sigma = pm.Exponential("sigma", lam=1)
    mu = a + bA * d["A"]
    pm.Normal("D", mu=mu, sigma=sigma, observed=d["D"].to_numpy())
    idata_m51 = fit_laplace(draws=10_000)
    pm.compute_log_likelihood(idata_m51)

# m5.2
with pm.Model() as model_m52:
    a    = pm.Normal("a", mu=0, sigma=0.2)
    bM    = pm.Normal("bM", mu=0, sigma=0.5)
    sigma = pm.Exponential("sigma", lam=1)
    mu = a + bM * d["M"]
    pm.Normal("D", mu=mu, sigma=sigma, observed=d["D"].to_numpy())
    idata_m52 = fit_laplace(draws=10_000)
    pm.compute_log_likelihood(idata_m52)

# m5.3
with pm.Model() as model_m53:
    a    = pm.Normal("a", mu=0, sigma=0.2)
    bA   = pm.Normal("bA", mu=0, sigma=0.5)
    bM   = pm.Normal("bM", mu=0, sigma=0.5)
    sigma = pm.Exponential("sigma", lam=1)
    mu = a + bA * d["A"] + bM * d["M"]
    pm.Deterministic("mu", mu)
    pm.Normal("D", mu=mu, sigma=sigma, observed=d["D"].to_numpy())
    idata_m53 = fit_laplace(draws=10_000)
    pm.compute_log_likelihood(idata_m53)


# Recreate Chapter 6 data (same seed as Chapter6_2.py)
rng_ch6 = np.random.default_rng(71)
N_ch6 = 100
h0_ch6 = rng_ch6.normal(10, 2, N_ch6)
treatment_ch6 = np.repeat([0, 1], N_ch6 // 2)
fungus_ch6 = rng_ch6.binomial(1, 0.5 - treatment_ch6 * 0.4, N_ch6)
h1_ch6 = h0_ch6 + rng_ch6.normal(5 - 3 * fungus_ch6, 1, N_ch6)
d_ch6 = pd.DataFrame({"h0": h0_ch6, "h1": h1_ch6,
                       "treatment": treatment_ch6, "fungus": fungus_ch6})

# m6.6
with pm.Model() as model_m66:
    p = pm.LogNormal("p", mu=0, sigma=0.25)
    sigma = pm.Exponential("sigma", lam=1)
    mu = p * d_ch6["h0"]
    pm.Normal("h1", mu=mu, sigma=sigma, observed=d_ch6["h1"].to_numpy())
    idata_m66 = fit_laplace(draws=10_000)
    pm.compute_log_likelihood(idata_m66)

# m6.7
with pm.Model() as model_m67:
    a  = pm.LogNormal("a",  mu=0, sigma=0.25)
    bT = pm.Normal("bT", mu=0, sigma=0.5)
    bF = pm.Normal("bF", mu=0, sigma=0.5)
    p  = a + bT * d_ch6["treatment"] + bF * d_ch6["fungus"]
    sigma = pm.Exponential("sigma", lam=1)
    mu = p * d_ch6["h0"]
    pm.Normal("h1", mu=mu, sigma=sigma, observed=d_ch6["h1"].to_numpy())
    idata_m67 = fit_laplace(draws=10_000)
    pm.compute_log_likelihood(idata_m67)

# m6.8
with pm.Model() as model_m68:
    a  = pm.LogNormal("a",  mu=0, sigma=0.25)
    bT = pm.Normal("bT", mu=0, sigma=0.5)
    p  = a + bT * d_ch6["treatment"]
    sigma = pm.Exponential("sigma", lam=1)
    mu = p * d_ch6["h0"]
    pm.Normal("h1", mu=mu, sigma=sigma, observed=d_ch6["h1"].to_numpy())
    idata_m68 = fit_laplace(draws=10_000)
    pm.compute_log_likelihood(idata_m68)

### 7.5.1 Model mis-selection ###

# Code 7.25 — WAIC(m6.7): az.waic() was removed in ArviZ 1.x. az.loo() is the recommended
# replacement, the rationale being that LOO-CV (via PSIS) is more reliable than WAIC for
# finite samples. The output is elpd_loo, p_loo, and Pareto k diagnostic values.
print(az.loo(idata_m67, var_name="h1"))

# Code 7.26 — compare(m6.6, m6.7, m6.8, func=WAIC): az.compare() is the Python equivalent.
# LOO-CV is used instead of WAIC (the default).
# Models are sorted by elpd_loo (highest = best). Columns:
#   rank      — model rank (0 = best)
#   elpd_diff — difference in elpd_loo relative to the best model
#   dse       — standard error of elpd_diff
#   p_worse   — tprobability that each model is worse than the best ranked model
#   diag_diff — potential issues with the ELPD difference
#   diag_elpd — potential issues with the ELPD estimate
#   p         — pIC, estimated effective number of parameters
#   elpd      — ELPD estimated using PSIS-LOO-CV
#   p_loo     — effective number of parameters
#   se        — standard error of elpd_loo
#   weight    — model weight for Bayesian model averaging (sum to 1)
#   warning   — True if any Pareto-k > 0.7 (LOO estimate unreliable)
comparison = az.compare({"m6.6": idata_m66, "m6.7": idata_m67, "m6.8": idata_m68},
                        var_name="h1")
print(comparison)

# Code 7.27 — SE of the difference between m6.7 and m6.8.
# In R: sqrt(n * var(waic_m6.7 - waic_m6.8))
# LOO pointwise values are on the elpd scale (higher = better); WAIC in the book uses
# deviance scale (-2 * elpd, lower = better). The SE of the difference is the same either way.
# Note: az.compare() already reports this as the 'dse' column.
loo_m67 = az.loo(idata_m67, var_name="h1", pointwise=True)
loo_m68 = az.loo(idata_m68, var_name="h1", pointwise=True)
diff_loo = loo_m67.elpd_i.values - loo_m68.elpd_i.values
n = len(diff_loo)
se_diff = np.sqrt(n * np.var(diff_loo))
print(f"SE of LOO difference (m6.7 - m6.8): {se_diff:.4f}")
print(f"(az.compare 'dse' for m6.8: {comparison.loc['m6.8', 'dse']:.4f})")

# Code 7.28 — interval of the difference
# We use McElreath's value of 10.4, even though it is not the same as what we
# found in the previous example
print(40.0 + np.array([-1, 1]) * 10.4 * 2.6)

# Code 7.29 — plot(compare(m6.6, m6.7, m6.8)): az.plot_compare() is the Python equivalent.
# Shows elpd_loo per model with ±1 SE bars. McElreath's R plot additionally shows in-sample
# deviance (open dots) and SE-of-difference triangles; these are not available in ArviZ 1.x.
az.plot_compare(comparison)
plt.title("LOO comparison: m6.6, m6.7, m6.8")
plt.tight_layout()
plt.show()

# Code 7.30 — SE of the difference between m6.6 and m6.8 (not relative to the best model,
# so comparison.loc['dse'] cannot be used here — that is always vs. the best model).
loo_m66 = az.loo(idata_m66, var_name="h1", pointwise=True)
diff_m66_m68 = loo_m66.elpd_i.values - loo_m68.elpd_i.values
print(f"SE of LOO difference (m6.6 - m6.8): {np.sqrt(n * np.var(diff_m66_m68)):.4f}")

# Code 7.31 — compare(m6.6, m6.7, m6.8)@dSE: full pairwise SE-of-difference matrix.
# az.compare() only reports dse vs. the best model; all pairs require manual computation.
_loos = {"m6.6": loo_m66, "m6.7": loo_m67, "m6.8": loo_m68}
_names = list(_loos.keys())
dse_matrix = pd.DataFrame(0.0, index=_names, columns=_names)
for _a in _names:
    for _b in _names:
        if _a != _b:
            _diff = _loos[_a].elpd_i.values - _loos[_b].elpd_i.values
            dse_matrix.loc[_a, _b] = np.sqrt(len(_diff) * np.var(_diff))
print(dse_matrix.round(2))


### 7.5.2 Outliers and other illusions ###

# Code 7.33 — compare(m5.1, m5.2, m5.3, func=PSIS): PSIS-LOO model comparison.
# func=PSIS in R uses the same algorithm as az.compare() (PSIS-LOO-CV).
# Note that, as in the book, this gives warnings about Pareto-k values
comparison_m5 = az.compare({"m5.1": idata_m51, "m5.2": idata_m52, "m5.3": idata_m53},
                            var_name="D")
print(comparison_m5)

# In absence of a WAIC function, code 7.34 is not included

# Code 7.35 — m5.3t: same as m5.3 but with a Student-t likelihood (nu=2) instead of Normal.
# The heavy tails make it more robust to outliers (e.g. Idaho in the divorce data).
# dstudent(2, mu, sigma) in R = pm.StudentT(nu=2, mu=mu, sigma=sigma) in PyMC.
with pm.Model() as model_m53t:
    a     = pm.Normal("a",     mu=0, sigma=0.2)
    bA    = pm.Normal("bA",    mu=0, sigma=0.5)
    bM    = pm.Normal("bM",    mu=0, sigma=0.5)
    sigma = pm.Exponential("sigma", lam=1)
    mu    = a + bA * d["A"] + bM * d["M"]
    pm.StudentT("D", nu=2, mu=mu, sigma=sigma, observed=d["D"].to_numpy())
    idata_m53t = fit_laplace(draws=10_000)
    pm.compute_log_likelihood(idata_m53t)

print(az.summary(idata_m53t, var_names=["a", "bM", "bA", "sigma"], ci_prob=0.89, round_to=2, kind="stats"))

comparison_m53t = az.compare({"m5.3": idata_m53, "m5.3t": idata_m53t}, var_name="D")
print(comparison_m53t)
