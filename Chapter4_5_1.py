"""Bayesian modeling of height as a function of weight using Laplace approximation. The
model includes a quadratic term to capture non-linear relationships between weight and height.
The code also computes and visualizes the 89% HPDI for both the fitted line (mean) and the
predicted heights of individuals (posterior predictive interval).

Adapted from Rethinking Statistics 2nd edition, Chapter 4.5.1.
"""

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm

from pymc_extras.inference import fit_laplace

d: pd.DataFrame = pd.read_csv("Howell1.csv", sep=";")

# Fit the model using Laplace approximation
xbar = d["weight"].mean()
sd = d["weight"].std()

# Standardizing the weights. This has a few reasons:
# - It makes the numerical optimization more stable, as the parameters will be on a more similar scale.
# - It makes the priors more interpretable: a unit change in weights_t corresponds to a change of 1 SD in
#   weight, which is easier to reason about than a change of 1 kg.
weights_t = (d["weight"].to_numpy() - xbar) / sd

# Fit the model using Laplace approximation
with pm.Model() as model:
    a    = pm.Normal("a", mu=178, sigma=20)
    b1    = pm.Lognormal("b1", mu=0, sigma=1)
    b2    = pm.Normal("b2", mu=0, sigma=1)
    sigma = pm.Uniform("sigma", lower=0, upper=50)
    mu = a + b1 * weights_t + b2 * weights_t**2
    pm.Normal("height", mu=mu, sigma=sigma, observed=d["height"].to_numpy())
    idata = fit_laplace(draws=10_000)

print(az.summary(idata, var_names=["a", "b1", "b2", "sigma"], hdi_prob=0.89, round_to=2, kind="stats"))

# For each integer weight compute:
#   1. 89% HPDI of mu (mean line) — parameter uncertainty only
#   2. 89% HPDI of individual heights — includes sigma (posterior predictive)
weight_seq = np.arange(int(d["weight"].min()), int(d["weight"].max()) + 1)
mu_lo    = np.empty(len(weight_seq))
mu_hi    = np.empty(len(weight_seq))
hpdi_lo  = np.empty(len(weight_seq))
hpdi_hi  = np.empty(len(weight_seq))
sample_a     = idata.posterior["a"].to_numpy().ravel()
sample_b1    = idata.posterior["b1"].to_numpy().ravel()
sample_b2    = idata.posterior["b2"].to_numpy().ravel()
sample_sigma = idata.posterior["sigma"].to_numpy().ravel()
rng = np.random.default_rng()
for i, w in enumerate(weight_seq):
    mu_samples     = sample_a + sample_b1 * ((w - xbar)/sd) + sample_b2 * ((w - xbar)/sd)**2
    height_samples = rng.normal(mu_samples, sample_sigma)
    mu_lo[i], mu_hi[i]     = az.hdi(mu_samples,     hdi_prob=0.89)
    hpdi_lo[i], hpdi_hi[i] = az.hdi(height_samples, hdi_prob=0.89)

# Plotting the data and the 89% HPDI for the predicted heights at each weight
fig, ax = plt.subplots()
ax.scatter(d["weight"], d["height"], alpha=0.4, s=10)
ax.set_xlabel("Weight")
ax.set_ylabel("Height")
ax.set_title("Height vs Weight (Howell1.csv)")
ax.fill_between(weight_seq, hpdi_lo, hpdi_hi, alpha=0.2, color="steelblue", label="89% HPDI for height of individuals (PI)")
ax.fill_between(weight_seq, mu_lo,   mu_hi,   alpha=0.4, color="steelblue", label="89% HPDI for fitted line")
x = np.linspace(d["weight"].min(), d["weight"].max(), 200)
ax.plot(x, idata.posterior["a"].mean().item() + idata.posterior["b1"].mean().item() * ((x - xbar)/sd) + idata.posterior["b2"].mean().item() * ((x - xbar)/sd)**2, color="black", alpha=0.8, linewidth=1, label="Posterior mean")
ax.legend()
plt.tight_layout()
plt.show()

# Exercise 4H1 -- calculate expected height and 89 % interval for the following weights: 46.25, 43.72, 64.78, 32,59, and 54,63 kg
weights_exercise = np.array([46.25, 43.72, 64.78, 32.59, 54.63])
for w in weights_exercise:
    mu_samples     = sample_a + sample_b1 * ((w - xbar)/sd) + sample_b2 * ((w - xbar)/sd)**2
    height_samples = rng.normal(mu_samples, sample_sigma)
    mu_lo, mu_hi     = az.hdi(mu_samples,     hdi_prob=0.89)
    hpdi_lo, hpdi_hi = az.hdi(height_samples, hdi_prob=0.89)
    print(f"Weight: {w:.2f} kg \n   -- Expected height for the individual: {mu_samples.mean():.2f} cm "
          f"(89% HPDI: [{mu_lo:.2f}, {mu_hi:.2f}])\n   -- 89% height interval for the individual: "
          f"[{hpdi_lo:.2f}, {hpdi_hi:.2f}]")