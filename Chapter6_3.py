"""Bayesian models showing examples of collider bias.

Adapted from Rethinking Statistics 2nd edition, Chapter 6.3.
"""

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
from scipy.special import expit

from pymc_extras.inference import fit_laplace

### 6.3.1 Collider of false sorrow ###

# The following recreates the dataframe created with the sim_happiness function in the rethinking package
seed     = 1977
N_years  = 1000
max_age  = 65
N_births = 20
aom      = 18  # age of marriage

rng = np.random.default_rng(seed)

A = np.array([], dtype=float)  # age
H = np.array([], dtype=float)  # happiness (fixed trait)
M = np.array([], dtype=float)  # married (0/1)

for t in range(N_years):
    # Age existing individuals
    A = A + 1

    # Add newborns
    A = np.concatenate([A, np.ones(N_births)])
    H = np.concatenate([H, np.linspace(-2, 2, N_births)])  # seq(from=-2, to=2, length.out=N_births)
    M = np.concatenate([M, np.zeros(N_births)])

    # Each eligible individual has a chance to get married: rbern(1, inv_logit(H - 4))
    eligible = (A >= aom) & (M == 0)
    new_marriages = rng.binomial(1, expit(H - 4))  # draw for everyone; only eligible ones are used
    M = np.where(eligible, new_marriages, M)

    # Mortality: remove individuals older than max_age
    alive = A <= max_age
    A = A[alive]
    H = H[alive]
    M = M[alive]

d = pd.DataFrame({"age": A.astype(int), "married": M.astype(int), "happiness": H})

print(d.describe().round(2))
print(f"\nShape: {d.shape}")

# Keep only adults and rescale age to [0, 1]
d2 = d[d["age"] > 17].copy()
d2["A"] = (d2["age"] - 18) / (65 - 18)

# d2$mid <- d2$married + 1  (R is 1-indexed; use married directly as 0-indexed in Python)
mid = d2["married"].to_numpy().astype(int)  # 0 = unmarried, 1 = married

# Model approximating the relationship between age and happiness, accounting for marriage status
with pm.Model() as model:
    a = pm.Normal("a", mu=0, sigma=1, shape=2)  # one intercept per marriage status
    bA = pm.Normal("bA", mu=0, sigma=2)
    sigma = pm.Exponential("sigma", lam=1)
    mu = a[mid] + bA * d2["A"].to_numpy()
    pm.Normal("happiness", mu=mu, sigma=sigma, observed=d2["happiness"].to_numpy())
    idata_m69 = fit_laplace(draws=10_000)

print(az.summary(idata_m69, var_names=["a", "bA", "sigma"], hdi_prob=0.89, round_to=2, kind="stats"))

# Model approximating the relationship between age and happiness without accounting for marriage status
with pm.Model() as model:
    a = pm.Normal("a", mu=0, sigma=1)
    bA = pm.Normal("bA", mu=0, sigma=2)
    sigma = pm.Exponential("sigma", lam=1)
    mu = a + bA * d2["A"].to_numpy()
    pm.Normal("happiness", mu=mu, sigma=sigma, observed=d2["happiness"].to_numpy())
    idata_m610 = fit_laplace(draws=10_000)

print(az.summary(idata_m610, var_names=["a", "bA", "sigma"], hdi_prob=0.89, round_to=2, kind="stats"))


### 6.3.2 The haunted DAG ###

N    = 200  # number of grandparent-parent-child triads
b_GP = 1    # direct effect of G on P
b_GC = 0    # direct effect of G on C
b_PC = 1    # direct effect of P on C
b_U  = 2    # direct effect of U on P and C

rng3 = np.random.default_rng(1)

U = 2 * rng3.binomial(1, 0.5, N) - 1   # rbern(N, 0.5) -> {0,1}, scaled to {-1, 1}
G = rng3.normal(0, 1, N)
P = rng3.normal(b_GP * G + b_U * U, 1, N)
C = rng3.normal(b_PC * P + b_GC * G + b_U * U, 1, N)

d3 = pd.DataFrame({"C": C, "P": P, "G": G, "U": U})

# Laplace approximation of the model with both P and G as predictors of C
with pm.Model() as model:
    a = pm.Normal("a",    mu=0, sigma=1)
    b_PC = pm.Normal("b_PC", mu=0, sigma=1)
    b_GC = pm.Normal("b_GC", mu=0, sigma=1)
    sigma = pm.Exponential("sigma", lam=1)
    mu = a + b_PC * d3["P"] + b_GC * d3["G"]
    pm.Normal("C", mu=mu, sigma=sigma, observed=d3["C"].to_numpy())
    idata_m611 = fit_laplace(draws=10_000)

print(az.summary(idata_m611, var_names=["a", "b_PC", "b_GC", "sigma"], hdi_prob=0.89, round_to=2, kind="stats"))

# Laplace approximation of the model with both P, G, and U as predictors of C
with pm.Model() as model:
    a = pm.Normal("a",    mu=0, sigma=1)
    b_PC = pm.Normal("b_PC", mu=0, sigma=1)
    b_GC = pm.Normal("b_GC", mu=0, sigma=1)
    b_U = pm.Normal("b_U",  mu=0, sigma=1)
    sigma = pm.Exponential("sigma", lam=1)
    mu = a + b_PC * d3["P"] + b_GC * d3["G"] + b_U * d3["U"]
    pm.Normal("C", mu=mu, sigma=sigma, observed=d3["C"].to_numpy())
    idata_m611 = fit_laplace(draws=10_000)

print(az.summary(idata_m611, var_names=["a", "b_PC", "b_GC", "b_U", "sigma"], hdi_prob=0.89, round_to=2, kind="stats"))
