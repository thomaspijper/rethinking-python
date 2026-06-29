"""Bayesian inference example: Binomial model.
This example demonstrates how to compute the posterior distribution for a binomial model
with a uniform prior, and how to extract summary statistics and credible intervals from the posterior.

Adapted from Rethinking Statistics 2nd edition, Chapter 3.
"""

import numpy as np
import arviz as az
import matplotlib.pyplot as plt

from scipy.stats import binom


### 3.1 Sampling from a grid-approximation posterior ###

# Code 3.2 — grid approximation for binomial model with uniform prior
p_grid = np.linspace(0, 1, 1000)
prior = np.ones(1000)
likelihood = binom.pmf(6, n=9, p=p_grid)
posterior = likelihood * prior
posterior = posterior / posterior.sum()

# Code 3.3 — pulling samples from the posterior distribution
samples = np.random.choice(p_grid, size=10_000, replace=True, p=posterior)

# Code 3.4 and 3.5 to recreate Figure 3.1 — left: samples vs index; right: density of samples
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].scatter(range(len(samples)), samples, alpha=0.1, s=12)
axes[0].set_xlabel("Sample number")
axes[0].set_ylabel("Proportion water (p)")
x_kde0, y_kde0, _ = az.kde(samples)
axes[1].plot(x_kde0, y_kde0)
axes[1].set_xlabel("Proportion water (p)")
axes[1].set_ylabel("Density")
plt.tight_layout()
plt.show()


### 3.2 Sampling to summarize ###

# Code 3.6 — add up posterior probability where p < 0.5
print(posterior[p_grid < 0.5].sum())

# Code 3.7 — posterior probability where p < 0.5 using samples
print((samples < 0.5).sum()/1e4)

# Code 3.8 — posterior probability between 0.5 and 0.75
print(((samples > 0.5) & (samples < 0.75)).sum() / 1e4)

# Code 3.9 — 80th percentile of the posterior distribution
print(np.quantile(samples, [0.8]))

# Code 3.10 — 80% credible interval (equal-tailed)
print(np.quantile(samples, [0.1, 0.9]))

# Code 3.11 and 3.12 — percentile interval (equal-tailed) for a different posterior
p_grid2 = np.linspace(0, 1, 1000)
posterior2 = binom.pmf(3, n=3, p=p_grid2) * np.ones(1000)
posterior2 = posterior2 / posterior2.sum()
samples2 = np.random.choice(p_grid2, size=10_000, replace=True, p=posterior2)
lo, hi = np.quantile(samples2, [0.25, 0.75])  # PI(prob=0.5) → 25th and 75th percentile
print(f"50% PI: [{lo:.3f}, {hi:.3f}]")

# Code 3.13 — HPDI
print(az.hdi(samples2, prob=0.5))  # narrowest interval containing 50% of the samples

# Code 3.14, 3.15, and 3.16 — Point estimates of the posterior
print(f"MAP (grid):   {p_grid2[np.argmax(posterior2)]:.3f}")   # which.max(posterior)
x_kde, y_kde, _ = az.kde(samples2)
print(f"Mode (chain): {x_kde[np.argmax(y_kde)]:.3f}")          # chainmode(samples, adj=0.01)
print(f"Mean:         {samples2.mean():.3f}")                  # mean(samples)
print(f"Median:       {np.median(samples2):.3f}")              # median(samples)

# Code 3.17 — Loss function: expected absolute deviation from a decision point p = 0.5
print(np.sum(posterior2 * np.abs(0.5 - p_grid2)))

# Code 3.18 and 3.19 — Loss function for a range of decision points
# Minimise loss over all candidate decisions — vectorised with broadcasting
# loss <- sapply(p_grid, function(d) sum(posterior * abs(d - p_grid)))
loss = np.sum(posterior2 * np.abs(p_grid2[:, np.newaxis] - p_grid2[np.newaxis, :]), axis=1)
print(p_grid2[np.argmin(loss)])


### 3.3 Sampling to simulate prediction ###

# Code 3.20 — dbinom(0:2, size=2, prob=0.7) — PMF at k=0,1,2
print(binom.pmf(np.arange(3), n=2, p=0.7))

# Code 3.21 — rbinom(1, size=2, prob=0.7) — 1 random draw from Binomial(n=2, p=0.7)
print(binom.rvs(n=2, size=1, p=0.7))

# Code 3.22 — rbinom(10, size=2, prob=0.7)
print(binom.rvs(n=2, size=10, p=0.7))

# Code 3.23 — dummy_w <- rbinom(1e5, size=2, prob=0.7); table(dummy_w)/1e5
dummy_w = binom.rvs(n=2, size=100_000, p=0.7)
unique, counts = np.unique(dummy_w, return_counts=True)
print("dummy_w")
print("  ".join(f"{v:>8}" for v in unique))
print("  ".join(f"{c/1e5:>8.5f}" for c in counts))

# Code 3.24 — dummy_w <- rbinom(1e5, size=9, prob=0.7); simplehist(dummy_w)
dummy_w2 = binom.rvs(n=9, size=100_000, p=0.7)
fig, ax = plt.subplots()
ax.hist(dummy_w2, bins=range(11), align="left", rwidth=0.8)
ax.set_xlabel("dummy water count")
ax.set_ylabel("Frequency")
plt.tight_layout()
plt.show()
