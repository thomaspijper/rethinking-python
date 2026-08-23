"""MCMC applied to a rugged-terrain/GDP interaction model with posterior diagnostics and visualizations.

Adapted from Rethinking Statistics 2nd edition, Chapter 9.4.
"""

import arviz as az
import matplotlib.pyplot as plt
import pandas as pd
import pymc as pm
import numpy as np

from pymc_extras.inference import fit_laplace
from scipy.stats import gaussian_kde, rankdata

def main():
    # Code 9.11 — reading the data
    d = pd.read_csv("rugged.csv", sep=";")
    d["log_gdp"] = np.log(d["rgdppc_2000"])
    dd = d[d["rgdppc_2000"].notna()].copy()
    dd["log_gdp_std"] = dd["log_gdp"] / dd["log_gdp"].mean()
    dd["rugged_std"] = dd["rugged"] / dd["rugged"].max()
    dd["cid"] = (dd["cont_africa"] == 1).astype(int)  # 0 = not Africa, 1 = Africa

    # Code 9.12 — fitting the model using the quadratic approximation
    cid = dd["cid"].to_numpy()
    with pm.Model() as model_m83:
        a     = pm.Normal("a", mu=1,   sigma=0.1, shape=2)
        b     = pm.Normal("b", mu=0,   sigma=0.3, shape=2)
        sigma = pm.Exponential("sigma", lam=1)
        mu = a[cid] + b[cid] * (dd["rugged_std"].to_numpy() - 0.215)
        pm.Normal("log_gdp_std", mu=mu, sigma=sigma, observed=dd["log_gdp_std"].to_numpy())
        idata_m83_q = fit_laplace(draws=10_000, random_seed=1)
    print(az.summary(idata_m83_q, var_names=["a", "b", "sigma"], ci_prob=0.89, round_to=2, kind="stats"))


    ### 9.4.1 Preparation ###

    # Code 9.13 is omitted as dat_slim an ulam() artifact. PyMC uses the arrays directly, so
    # tidying the data is not required.


    ### 9.4.2 Sampling from the posterior ###

    # Code 9.14 — same model as m83 fitted with MCMC (1 chain, as in the book)
    # dat_slim from Code 9.13 is an ulam() artifact; PyMC uses the arrays directly
    with pm.Model() as model_m91:
        a     = pm.Normal("a", mu=1,   sigma=0.1, shape=2)
        b     = pm.Normal("b", mu=0,   sigma=0.3, shape=2)
        sigma = pm.Exponential("sigma", lam=1)
        mu = a[cid] + b[cid] * (dd["rugged_std"].to_numpy() - 0.215)
        pm.Normal("log_gdp_std", mu=mu, sigma=sigma, observed=dd["log_gdp_std"].to_numpy())
        idata_m91 = pm.sample(chains=1, random_seed=1)

    # Code 9.15 — posterior summary
    #
    # Since the posterior is not necessarily Gaussian/symmetric, we use the highest density
    # interval (HDI) instead of the equal-tailed interval (ETI, the default in Arviz).
    #
    # We also remove the kind="stats" argument because it hides the MCMC diagnostics. This reveals
    # ess_bulk and ess_tail, which are the effective sample sizes for the bulk and tail of the
    # posterior distribution, respectively.
    print(az.summary(idata_m91, var_names=["a", "b", "sigma"], ci_prob=0.89, ci_kind="hdi", round_to=2))


    ### 9.4.3 Sampling again, in parallel ###

    # Code 9.16 — using 4 chains and 4 cores
    # Actually, 4 chains divided over 4 cores is the default, so we just remove the chains=1 argument.
    #
    # Important: on Windows it is required to wrap the sampling code in a `if __name__ == "__main__":` block or
    # a function, otherwise it will fail. The reason is that Windows uses spawn instead of fork to create new
    # processes and the spawned processes will re-import the main module. Without a guard, this would trigger
    # recursive process spawning.
    with pm.Model() as model_m91:
        a     = pm.Normal("a", mu=1,   sigma=0.1, shape=2)
        b     = pm.Normal("b", mu=0,   sigma=0.3, shape=2)
        sigma = pm.Exponential("sigma", lam=1)
        mu = a[cid] + b[cid] * (dd["rugged_std"].to_numpy() - 0.215)
        pm.Normal("log_gdp_std", mu=mu, sigma=sigma, observed=dd["log_gdp_std"].to_numpy())
        idata_m91 = pm.sample(random_seed=1)

    # Code 9.17 — there is no direct analogue of show() in PyMC, but the MCMC procedure already
    # prints quite a bit of information. There are also a few other things we can do.
    print(model_m91.str_repr())   # the model structure, including the generative process and the priors
    print(idata_m91.sample_stats) # per-draw NUTS diagnostics (step size, tree depth, …) and various attributes (version numbers, sampling time, tuning steps, etc.)

    # Graphviz can be used to visualize the DAG of the model. It requires the python library graphviz, along
    # with the graphviz binaries installed on your system. Uncomment the following line if these are installed.
    # pm.model_to_graphviz(model_m91).view()

    # Code 9.18 — posterior summary
    print(az.summary(idata_m91, var_names=["a", "b", "sigma"], ci_prob=0.89, ci_kind="hdi", round_to=2))


    ### 9.4.4 Visualizations ###

    # Code 9.19 — pairs plot of the posterior distribution, recreating Figure 9.7.
    az.plot_pair(idata_m91, var_names=["a", "b", "sigma"], marginal=True, marginal_kind="kde").show()

    # Compared to the book, the correlation between parameters is not shown as az.plot_pair() cannot show
    # different content per triangle. If this is desired, one can build the grid manually:
    # upper triangle = scatter, diagonal = KDE, lower triangle = correlation coefficient.
    post = idata_m91.posterior.ds.stack(sample=("chain", "draw"))
    data = {}
    for var in ["a", "b", "sigma"]:
        vals = post[var].values
        if vals.ndim == 1:
            data[var] = vals
        else:
            for i in range(vals.shape[0]):
                data[f"{var}[{i}]"] = vals[i]
    df = pd.DataFrame(data)
    labels = df.columns.tolist()
    n = len(labels)

    fig, axes = plt.subplots(n, n, figsize=(2.2 * n, 2.2 * n))
    for i in range(n):
        for j in range(n):
            ax = axes[i, j]
            if i == j:
                x = df[labels[i]].to_numpy()
                xs = np.linspace(x.min(), x.max(), 200)
                ax.plot(xs, gaussian_kde(x)(xs), color="black")
                ax.set_yticks([])
            elif i < j:
                ax.scatter(df[labels[j]], df[labels[i]], s=2, alpha=0.3, color="steelblue")
            else:
                corr = np.corrcoef(df[labels[j]], df[labels[i]])[0, 1]
                ax.text(0.5, 0.5, f"{corr:.2f}", ha="center", va="center",
                        fontsize=10 + 15 * abs(corr), transform=ax.transAxes)
                ax.set_xticks([]); ax.set_yticks([])
            if i == n - 1:
                ax.set_xlabel(labels[j])
            if j == 0:
                ax.set_ylabel(labels[i])
    plt.tight_layout()
    plt.show()


    ### 9.4.5 Checking the chain ###

    # Code 9.20 — trace plot of the posterior distribution
    # They keyword sample_dims=["draw"] gives each chain its own column instead of overlaying them,
    # if desired.
    # Note: PyMC's default is to drop samples from the tuning/warmup phase, so the trace plot won't show
    # show the warmup samples (or adaptation samples as the book calls them). Inspecting them is possible
    # by setting discard_tuned_samples=False in pm.sample(), which creates an idata.warmup_posterior group
    # in addition to idata.posterior. Combining them for plotting is tricky (it requires .ds access and
    # coordinate offsetting). Note that warmup samples are not valid posterior samples and should not be
    # used for inference.
    az.plot_trace(idata_m91, var_names=["a", "b", "sigma"], sample_dims=["draw"])
    plt.tight_layout()
    plt.show()


    # Code 9.21 — trace rank plot (trankplot)
    # In a rank plot, values of all chains are pooled and ranked from 1 to N, where N is the
    # total number of samples across all chains. The ranks are then split back into their respective
    # chains and plotted as a histogram. If all chains are targeting the same posterior, we expect the
    # ranks in each chain to be uniformly distributed.
    #
    # Recent versions of Arviz use a delta-ECDF plot instead of the classic histogram, which
    # looks very different from the book's trankplot(). Here ECDF = empirical CDF, an estimate of the
    # cumulative distribution function based on observed samples. Ranks are rescaled from [1, N] to
    # [0, 1]. After splitting, the ECDF of each chain is compared to the ECDF expected under uniformity,
    # which gives the delta-ECDF. No y-tick values are shown by default, but a well-mixed chain's curve
    # should hover close to zero.
    #
    # The default method ("mtc_c") runs a multi-chain rank test, whereby suspicious points are
    # highlighted. The displayed p-value tests whether the pooled fractional ranks are consistent with
    # all chains sampling the same (uniform) distribution (with the default α value being 0.01). Note
    # that individual points are only highlighted when p < α (i.e. the global test is already
    # significant); among those, the highlighted points are the ones with the largest Shapley-value
    # contribution to that deviation.
    az.plot_rank(idata_m91, var_names=["a", "b", "sigma"])
    plt.tight_layout()
    plt.show()

    # Of course, we can manually recreate the classic histogram of ranks per chain that the book uses.
    def plot_rank_hist(idata, var_names, bins=20):
        """Classic histogram of ranks per chain, recreating rethinking::trankplot()"""
        post = idata.posterior.ds
        n_chains = post.sizes["chain"]

        rows = []
        for var in var_names:
            da = post[var]
            extra_dims = [d for d in da.dims if d not in ("chain", "draw")]
            if extra_dims:
                for i in range(da.sizes[extra_dims[0]]):
                    rows.append((f"{var}[{i}]", da.isel({extra_dims[0]: i})))
            else:
                rows.append((var, da))

        fig, axes = plt.subplots(len(rows), 1, figsize=(8, 2.5 * len(rows)))
        for ax, (label, da) in zip(axes, rows):
            vals = da.values                                     # (chain, draw)
            ranks = rankdata(vals.ravel()).reshape(vals.shape)   # rank across all chains combined
            for c in range(n_chains):
                counts, edges = np.histogram(ranks[c], bins=bins)
                ax.stairs(counts, edges, baseline=None, linewidth=1.5, label=f"chain {c}")
            ax.set_ylabel(label)
            ax.set_yticks([])
        axes[0].legend(fontsize=8, loc="upper right")
        axes[-1].set_xlabel("rank")
        plt.tight_layout()
        plt.show()

    plot_rank_hist(idata_m91, ["a", "b", "sigma"])


if __name__ == "__main__":
    # We wrap all code in main() to avoid issues with multiprocessing on Windows.
    main()