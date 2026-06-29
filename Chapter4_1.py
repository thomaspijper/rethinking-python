"""Examples of data converging towards normal distributions.

Adapted from Rethinking Statistics 2nd edition, Chapter 4.1."""

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
from scipy import stats

from pymc_extras.inference import fit_laplace


### 4.1.1 Normal by addition ###

# Code 4.1 — Simulating the sum of 16 uniform random variables
rng_pos = np.random.default_rng(42)
pos = rng_pos.uniform(-1, 1, size=(1000, 16)).sum(axis=1)

# hist(pos)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
ax1.hist(pos, bins=30, color="#1e90ff", edgecolor="white")
ax1.set_xlabel("pos")
ax1.set_ylabel("Frequency")
ax1.set_title("Histogram of pos")

# plot(density(pos))
kde_x = np.linspace(pos.min(), pos.max(), 500)
kde = stats.gaussian_kde(pos)
ax2.plot(kde_x, kde(kde_x), color="#1e90ff")
ax2.fill_between(kde_x, kde(kde_x), alpha=0.2, color="#1e90ff")
ax2.set_xlabel("pos")
ax2.set_ylabel("Density")
ax2.set_title("Density plot of pos")

plt.tight_layout()
plt.show()

### 4.1.2 Normal by multiplication ###

# Code 4.3 — Simulating the product of 12 uniform random variables
rng_growth = np.random.default_rng(42)
growth = (1 + rng_growth.uniform(0, 0.1, size=(10_000, 12))).prod(axis=1)

kde_x = np.linspace(growth.min(), growth.max(), 500)
kde = stats.gaussian_kde(growth)
mu_g, sigma_g = growth.mean(), growth.std()

fig, ax = plt.subplots()
ax.plot(kde_x, kde(kde_x), color="#1e90ff", label="density(growth)")
ax.plot(kde_x, stats.norm.pdf(kde_x, mu_g, sigma_g),
        color="black", linestyle="--", label="Normal fit")
ax.set_xlabel("growth")
ax.set_ylabel("Density")
ax.set_title("Density plot of growth with normal fit")
ax.legend()
plt.tight_layout()
plt.show()

# Code 4.4 — Simulating multiplication with small and big effects
rng_big = np.random.default_rng(42)
big   = (1 + rng_big.uniform(0, 0.5,  size=(10_000, 12))).prod(axis=1)
small = (1 + rng_big.uniform(0, 0.01, size=(10_000, 12))).prod(axis=1)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

for ax, data, label, color in [
    (ax1, big,   "big",   "#e05c5c"),
    (ax2, small, "small", "#5ca0e0"),
]:
    kde_x = np.linspace(data.min(), data.max(), 500)
    kde   = stats.gaussian_kde(data)
    mu_d, sigma_d = data.mean(), data.std()

    ax.plot(kde_x, kde(kde_x), color=color, label=f"density({label})")
    ax.plot(kde_x, stats.norm.pdf(kde_x, mu_d, sigma_d),
            color="black", linestyle="--", label="Normal fit")
    ax.set_xlabel(label)
    ax.set_ylabel("Density")
    ax.set_title(f"Density plot of {label} with normal fit")
    ax.legend()

plt.tight_layout()
plt.show()


### 4.1.3 Normal by log-multiplication ###

# Code 4.5 — Simulating the log of the product of 12 uniform random variables
rng_log = np.random.default_rng(42)
log_big = np.log((1 + rng_log.uniform(0, 0.5, size=(10_000, 12))).prod(axis=1))

kde_x = np.linspace(log_big.min(), log_big.max(), 500)
kde   = stats.gaussian_kde(log_big)
mu_lb, sigma_lb = log_big.mean(), log_big.std()

fig, ax = plt.subplots()
ax.plot(kde_x, kde(kde_x), color="#e05c5c", label="density(log.big)")
ax.plot(kde_x, stats.norm.pdf(kde_x, mu_lb, sigma_lb),
        color="black", linestyle="--", label="Normal fit")
ax.set_xlabel("log.big")
ax.set_ylabel("Density")
ax.set_title("Density plot of log.big with normal fit")
ax.legend()
plt.tight_layout()
plt.show()
