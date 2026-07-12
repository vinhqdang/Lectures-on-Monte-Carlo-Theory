#!/usr/bin/env python3
"""
Figure generation for Chapter 9: FDR control using compound e-values.

Generates:
  (a) ebh_illustration.pdf  -- e-BH procedure applied to a toy set of 20
      e-values: sorted e-values compared against the e-BH threshold curve
      K/(alpha*k), with rejected/non-rejected hypotheses marked.
  (b) fdr_simulation.pdf    -- empirical FDR of the e-BH procedure across many
      repeated trials, with some true nulls and some true alternatives,
      showing that the running-average FDR stays below the target alpha.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

rng = np.random.default_rng(2024)

# ----------------------------------------------------------------------
# Figure (a): e-BH procedure on a toy set of 20 e-values
# ----------------------------------------------------------------------

K = 20
alpha = 0.1

# Construct a toy set of e-values: a handful of large ("real signal") values
# and many values scattered near/below 1 (consistent with the null).
e_values = np.array([
    250.0, 110.0, 72.0, 55.0, 20.0, 9.5, 3.2, 2.1, 1.8, 1.6,
    1.4, 1.3, 1.1, 1.0, 0.9, 0.8, 0.7, 0.6, 0.35, 0.2
])
assert len(e_values) == K

order = np.argsort(-e_values)          # descending order
e_sorted = e_values[order]
k_index = np.arange(1, K + 1)

# e-BH rejection rule: reject the largest k such that k * e_[k] / K >= 1/alpha
threshold_curve = K / (alpha * k_index)          # e_[k] must be >= this
satisfies = e_sorted >= threshold_curve
if satisfies.any():
    k_star = np.max(np.where(satisfies)[0]) + 1
else:
    k_star = 0

rejected = np.zeros(K, dtype=bool)
rejected[:k_star] = True

fig, ax = plt.subplots(figsize=(7.5, 5.0))
colors = ['#d62728' if r else '#1f77b4' for r in rejected]
ax.scatter(k_index, e_sorted, c=colors, s=55, zorder=3,
           label='_nolegend_')
ax.plot(k_index, threshold_curve, '--', color='gray', linewidth=2,
        label=r'threshold $K/(\alpha k)$')
ax.axvline(k_star + 0.5, color='black', linestyle=':', linewidth=1.2)
ax.set_yscale('log')
ax.set_xlabel('rank $k$ (sorted, largest e-value first)')
ax.set_ylabel(r'$e_{[k]}$ (log scale)')
ax.set_title(f'e-BH procedure on 20 e-values ($K=20$, $\\alpha=0.1$): '
             f'$k^*={k_star}$ rejections')

# Manual legend entries for rejected/not-rejected
from matplotlib.lines import Line2D
legend_elems = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#d62728',
           markersize=9, label='rejected'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#1f77b4',
           markersize=9, label='not rejected'),
    Line2D([0], [0], color='gray', linestyle='--', linewidth=2,
           label=r'threshold $K/(\alpha k)$'),
]
ax.legend(handles=legend_elems, loc='upper right')
ax.grid(alpha=0.3, which='both')
fig.tight_layout()
fig.savefig('ebh_illustration.pdf')
plt.close(fig)

print(f"Figure (a): k* = {k_star}, rejected e-values = {e_sorted[:k_star]}")

# ----------------------------------------------------------------------
# Figure (b): empirical FDR of e-BH across repeated trials
# ----------------------------------------------------------------------

K2 = 30          # number of hypotheses per trial
pi0 = 0.7        # fraction of true nulls
n_null = int(round(pi0 * K2))
n_alt = K2 - n_null
mu_alt = 2.0     # mean shift under the alternative
alpha2 = 0.2     # target FDR level
lam = mu_alt     # e-value tuned to the (assumed known) alternative effect size
n_trials = 4000

fdp_trials = np.empty(n_trials)

for t in range(n_trials):
    # null coordinates: Z ~ N(0,1);  alternative coordinates: Z ~ N(mu_alt,1)
    z_null = rng.normal(0.0, 1.0, size=n_null)
    z_alt = rng.normal(mu_alt, 1.0, size=n_alt)
    z = np.concatenate([z_null, z_alt])
    is_null = np.concatenate([np.ones(n_null, dtype=bool),
                               np.zeros(n_alt, dtype=bool)])

    # e-value for testing H0: mean <= 0 vs H1: mean > 0
    e = np.exp(lam * z - lam**2 / 2.0)

    order2 = np.argsort(-e)
    e_sorted2 = e[order2]
    is_null_sorted = is_null[order2]
    k_idx2 = np.arange(1, K2 + 1)
    sat = (k_idx2 * e_sorted2 / K2) >= (1.0 / alpha2)
    if sat.any():
        kstar2 = np.max(np.where(sat)[0]) + 1
    else:
        kstar2 = 0

    if kstar2 == 0:
        fdp_trials[t] = 0.0
    else:
        false_disc = np.sum(is_null_sorted[:kstar2])
        fdp_trials[t] = false_disc / kstar2

running_fdr = np.cumsum(fdp_trials) / np.arange(1, n_trials + 1)
final_fdr = running_fdr[-1]

fig2, ax2 = plt.subplots(figsize=(7.5, 5.0))
trial_axis = np.arange(1, n_trials + 1)
ax2.scatter(trial_axis[::8], fdp_trials[::8], s=8, color='#9ecae1',
            alpha=0.6, label='per-trial FDP', zorder=1)
ax2.plot(trial_axis, running_fdr, color='#1f77b4',
         linewidth=2.2, label='running-average FDR of e-BH', zorder=3)
ax2.axhline(alpha2, color='#d62728', linestyle='--', linewidth=2,
            label=rf'target level $\alpha={alpha2}$', zorder=2)
ax2.annotate(f'final avg FDR $\\approx$ {final_fdr:.3f}',
             xy=(n_trials, final_fdr),
             xytext=(n_trials * 0.45, alpha2 * 1.35),
             arrowprops=dict(arrowstyle='->', color='black'))
ax2.set_xlabel('number of trials')
ax2.set_ylabel('FDP per trial / running-average FDR')
ax2.set_title(f'Empirical FDR of e-BH over {n_trials} trials '
              f'($K={K2}$, $\\pi_0={pi0}$, $\\mu={mu_alt}$)')
ax2.set_ylim(0, max(0.05, alpha2 * 1.8))
ax2.legend(loc='upper right', framealpha=0.9)
ax2.grid(alpha=0.3)
fig2.tight_layout()
fig2.savefig('fdr_simulation.pdf')
plt.close(fig2)

print(f"Figure (b): empirical average FDR over {n_trials} trials = {final_fdr:.4f} "
      f"(target alpha = {alpha2})")
