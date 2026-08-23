"""The pattern in which King Markov visited the islands in his kingdom.

Adapted from Rethinking Statistics 2nd edition, Chapter 9.1.
"""

import matplotlib.pyplot as plt
import numpy as np


# Code 9.1 — Metropolis algorithm: island-hopping simulation
rng = np.random.default_rng(0)
num_weeks = 100_000
positions = np.empty(num_weeks, dtype=int)
current = 10
for i in range(num_weeks):
    positions[i] = current
    proposal = current + rng.choice([-1, 1])
    if proposal < 1:  proposal = 10
    if proposal > 10: proposal = 1
    prob_move = proposal / current
    if rng.uniform() < prob_move:
        current = proposal

# Code 9.2 and 9.3 — visualise the chain
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

ax1.plot(np.arange(1, 101), positions[:100], "o-", markersize=3)
ax1.set_xlabel("week"); ax1.set_ylabel("island")

islands, counts = np.unique(positions, return_counts=True)
ax2.bar(islands, counts)
ax2.set_xlabel("island"); ax2.set_ylabel("number of weeks")

plt.tight_layout()
plt.show()
