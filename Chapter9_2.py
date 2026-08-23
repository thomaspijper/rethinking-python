"""The single code example from section 9.2 of Rethinking Statistics 2nd edition,
demonstrating limitations of the Metropolis and Gibbs samplers in high-dimensional
problems.

Adapted from Rethinking Statistics 2nd edition, Chapter 9.2.
"""

import matplotlib.pyplot as plt
import numpy as np

from scipy.stats import gaussian_kde


### 9.2.2 High-dimensional problems ###

# Code 9.4 — sampling from a high dimension distribution
# Shown are D = 1, 10, 100, and 1000. This recreates Figure 9.4
T = 1000
rng = np.random.default_rng(0)

fig, ax = plt.subplots()
for D in [1, 10, 100, 1000]:
    Rd = np.linalg.norm(rng.standard_normal((T, D)), axis=1)
    x  = np.linspace(Rd.min()-1, Rd.max(), 200)
    ax.plot(x, gaussian_kde(Rd)(x), label=f"D={D}")
ax.set_xlabel("radial distance"); ax.set_ylabel("density")
ax.legend()
plt.tight_layout()
plt.show()
