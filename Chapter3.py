"""Bayesian inference example: Binomial model.
This example demonstrates how to compute the posterior distribution for a binomial model
with a uniform prior, and how to extract summary statistics and credible intervals from the posterior.

Adapted from Rethinking Statistics 2nd edition, Chapter 3.
"""

import numpy as np
import arviz as az

from scipy.stats import binom
from plotly import graph_objects as go

# Grid approximation for binomial model with uniform prior
p_grid = np.linspace(0, 1, 1000)
prior = np.ones(1000)
likelihood = binom.pmf(8, n=15, p=p_grid)
posterior = likelihood * prior
posterior = posterior / posterior.sum()

# Plotting the posterior distribution
fig = go.Figure()
fig.add_trace(go.Scatter(x=p_grid, y=posterior, mode="lines", name="Posterior"))
fig.update_layout(title="Posterior Distribution", xaxis_title="Probability of success", yaxis_title="Density")
fig.show()

# Pulling samples from the posterior distribution and calculating the 90% HPDI
samples = np.random.choice(p_grid, size=10_000, replace=True, p=posterior)
print(len(samples[(0.2 < samples) & (samples <= 0.8)])/len(samples))
lower, upper = az.hdi(samples,     hdi_prob=0.90)
print(f"90% HPDI: [{lower:.3f}, {upper:.3f}]")

# Simulating predictions from the total posterior
w = binom.rvs(n=15, size=int(1e4), p=samples)
unique, counts = np.unique(w, return_counts=True)
d = dict(zip(unique, counts))
print(d[8]/len(w))

# Plotting the simulated data
fig = go.Figure()
fig.add_trace(go.Histogram(x=w, nbinsx=50, name="Simulated Data"))
fig.update_layout(title="Simulated Binomial Data", xaxis_title="Number of successes", yaxis_title="Frequency")
fig.show()
