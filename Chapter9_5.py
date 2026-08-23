"""Demonstrates MCMC pathologies diagnosed via pair, trace, and rank plots.

Adapted from Rethinking Statistics 2nd edition, Chapter 9.5.
"""

import arviz as az
import matplotlib.pyplot as plt
import pymc as pm
import numpy as np


def main():
    ### 9.5.3 Taming a wild chain ###

    # Code 9.22 — a wild chain caused by flat/weak priors on very little data
    y = np.array([-1, 1])
    with pm.Model() as model_m92:
        alpha = pm.Normal("alpha", mu=0, sigma=1000)
        sigma = pm.Exponential("sigma", lam=0.0001)
        mu = alpha
        pm.Normal("y", mu=mu, sigma=sigma, observed=y)
        idata_m92 = pm.sample(chains=3, random_seed=11)

    # Code 9.23 — posterior summary
    print(az.summary(idata_m92, var_names=["alpha", "sigma"], ci_prob=0.89, ci_kind="hdi", round_to=2))

    # Pair plot
    # Note the visuals={"divergence": True} parameter, which highlights divergent transitions in
    # the pair plot in red.
    az.plot_pair(idata_m92, var_names=["alpha", "sigma"], visuals={"divergence": True}, marginal=True, marginal_kind="kde").show()

    # Trace plot
    # Problematic parts of the chain are highlighted with black ticks on the x-axis.
    az.plot_trace(idata_m92, var_names=["alpha", "sigma"], sample_dims=["draw"])
    plt.tight_layout()
    plt.show()

    # Rank plot
    # Problematic parts of the chain are overlayed with black dots. Also note the low p values.
    az.plot_rank(idata_m92, var_names=["alpha", "sigma"])
    plt.tight_layout()
    plt.show()

    # Code 9.24 — using weakly informative priors
    # Note that a few divergent transitions remain, but the chains are much better behaved than before.
    with pm.Model() as model_m93:
        alpha = pm.Normal("alpha", mu=1, sigma=10)
        sigma = pm.Exponential("sigma", lam=1)
        mu = alpha
        pm.Normal("y", mu=mu, sigma=sigma, observed=y)
        idata_m93 = pm.sample(chains=3, random_seed=5)
    print(az.summary(idata_m93, var_names=["alpha", "sigma"], ci_prob=0.89, ci_kind="hdi", round_to=2))

    # Pair plot
    az.plot_pair(idata_m93, var_names=["alpha", "sigma"], visuals={"divergence": True}, marginal=True, marginal_kind="kde").show()

    # Trace plot
    az.plot_trace(idata_m93, var_names=["alpha", "sigma"], sample_dims=["draw"])
    plt.tight_layout()
    plt.show()

    # Rank plot
    az.plot_rank(idata_m93, var_names=["alpha", "sigma"])
    plt.tight_layout()
    plt.show()


    ### 9.5.4 Non-identifiable parameters ###

    # Code 9.25 — simulation 100 observations from a Guassian distribution
    rng = np.random.default_rng(41)
    y = rng.normal(loc=0, scale=1, size=100)

    # Code 9.26 — a model with non-identifiable parameters
    # Just as the book mentions, this model will take a long time to run, and it will also
    # complain about some "maximum tree depth" being exceeded.
    #
    # What does that mean? NUTS (PyMC's default sampler) builds a binary tree of leapfrog steps
    # at each iteration, doubling in size until it either detects a U-turn (the trajectory starts
    # curving back on itself, signaling it is time to stop) or hits preset depth limit — max_treedepth,
    # default 10 (so up to 2^10 = 1024 steps).
    #
    # "Exceeded maximum tree depth" means the sampler hit that cap without detecting a U-turn: it was
    # forced to stop early rather than naturally. This is a soft warning, unlike divergences:
    #   * it does not necessarily mean that the posterior draws are biased/wrong;
    #   * it does mean sampling was inefficient — the chain is exploring very slowly, often because
    #     the posterior geometry is very flat or poorly identified.
    #
    with pm.Model() as model_m94:
        a1    = pm.Normal("a1", mu=0, sigma=1000)
        a2    = pm.Normal("a2", mu=0, sigma=1000)
        sigma = pm.Exponential("sigma", lam=1)
        mu = a1 + a2
        pm.Normal("y", mu=mu, sigma=sigma, observed=y)
        idata_m94 = pm.sample(chains=3, random_seed=384)
    print(az.summary(idata_m94, var_names=["a1", "a2", "sigma"], ci_prob=0.89, ci_kind="hdi", round_to=2))

    # Trace plot
    az.plot_trace(idata_m94, var_names=["a1", "a2", "sigma"], sample_dims=["draw"])
    plt.tight_layout()
    plt.show()

    # Rank plot
    az.plot_rank(idata_m94, var_names=["a1", "a2", "sigma"])
    plt.tight_layout()
    plt.show()

    # Code 9.27 — adding weakly regularizing priors to the non-identifiable model
    with pm.Model() as model_m95:
        a1    = pm.Normal("a1", mu=0, sigma=10)
        a2    = pm.Normal("a2", mu=0, sigma=10)
        sigma = pm.Exponential("sigma", lam=1)
        mu = a1 + a2
        pm.Normal("y", mu=mu, sigma=sigma, observed=y)
        idata_m95 = pm.sample(chains=3, random_seed=384)
    print(az.summary(idata_m95, var_names=["a1", "a2", "sigma"], ci_prob=0.89, ci_kind="hdi", round_to=2))

    # Trace plot
    az.plot_trace(idata_m95, var_names=["a1", "a2", "sigma"], sample_dims=["draw"])
    plt.tight_layout()
    plt.show()

    # Rank plot
    #
    # Even though there are no divergences, ESS is high enough, and Rhat is 1.0, the rank plot
    # shows that the chains are still not mixing well. This is a sign that the posterior is still
    # poorly identified. The reason is that a1 and a2 are highly anti-correlated in the
    # posterior, which creates a narrow, ridge-shaped region. The sampler was behaving correctly,
    # but the posterior geometry itself is hard to explore efficiently.
    az.plot_rank(idata_m95, var_names=["a1", "a2", "sigma"])
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # We wrap all code in main() to avoid issues with multiprocessing on Windows.
    main()
