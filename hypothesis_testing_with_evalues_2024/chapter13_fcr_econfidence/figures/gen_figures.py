"""
Generate figures for Chapter 13 slides:
  "False coverage rate control using e-confidence intervals"
  (Ramdas & Wang, "Hypothesis Testing with E-Values", arXiv:2410.23614)

Running numerical example used throughout the chapter: m = 6 coins are each
flipped n = 20 times (independently). Unknown to the "scientist," every coin
is secretly fair (true bias theta* = 0.5). The observed head-counts are

    Coin:      A    B    C    D    E    F
    heads:    15    5   11    9    8   12
    n_hat:  0.75 0.25 0.55 0.45 0.40 0.60

Confidence intervals throughout are *exact* Clopper-Pearson binomial
intervals. By the trivial embedding of Section 13.1 (every valid CI is
also an e-CI, via E(theta,alpha) = 1{theta not in C(alpha)}/alpha), these
doubly serve as e-CIs; we use them to make the FCR / e-BY story fully
numerical.

Figure (a) fig_econfidence_intervals.pdf:
    The 6 naive 95% e-CIs, with the true value theta*=0.5 marked. The two
    "most extreme" coins (A, B) are selected for reporting by an
    (unspecified / data-dependent) selection rule, and their intervals both
    happen to miss 0.5 at the naive level. The e-BY-corrected 98.33% e-CIs
    for exactly those two coins are overlaid to show how the correction
    restores coverage.

Figure (b) fig_fcr_simulation.pdf:
    A repeated-trials simulation (K=6 independent coins, L=2 selected per
    trial by "most extreme observed bias") comparing the empirical FCR of
    (i) naively reporting nominal (1-delta) CIs for the selected coins vs.
    (ii) the e-BY correction alpha_i = delta*|S|/K. Shows the naive FCR
    running well above the target level delta=0.05, while the corrected
    FCR stays controlled at or below delta, as guaranteed by Theorem 13.7.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import beta as beta_dist

plt.rcParams.update({
    "font.size": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

COL_GRAY = "#8c8c8c"
COL_BLUE = "#1f6f9c"
COL_RED = "#c0392b"
COL_GREEN = "#1e8449"


def clopper_pearson(x, n, conf):
    """Exact (1-alpha)-CI for a Binomial(n, theta) proportion, alpha = 1-conf."""
    a = 1.0 - conf
    lo = 0.0 if x == 0 else beta_dist.ppf(a / 2, x, n - x + 1)
    hi = 1.0 if x == n else beta_dist.ppf(1 - a / 2, x + 1, n - x)
    return lo, hi


# ---------------------------------------------------------------------------
# Figure (a): the 6 coins, naive e-CIs, selection, and e-BY correction
# ---------------------------------------------------------------------------
coins = ["A", "B", "C", "D", "E", "F"]
n_flips = 20
heads = np.array([15, 5, 11, 9, 8, 12])
phat = heads / n_flips
theta_true = 0.5
delta = 0.05
K = len(coins)
selected = [0, 1]  # coins A, B: the two most extreme |phat-0.5|
L = len(selected)
alpha_i_eby = delta * L / K

naive_lo, naive_hi = [], []
for x in heads:
    lo, hi = clopper_pearson(x, n_flips, 1 - delta)
    naive_lo.append(lo)
    naive_hi.append(hi)
naive_lo, naive_hi = np.array(naive_lo), np.array(naive_hi)

eby_lo, eby_hi = {}, {}
for i in selected:
    lo, hi = clopper_pearson(heads[i], n_flips, 1 - alpha_i_eby)
    eby_lo[i], eby_hi[i] = lo, hi

fig, ax = plt.subplots(figsize=(8.6, 4.6))
x_pos = np.arange(K)

for i in range(K):
    covers = naive_lo[i] <= theta_true <= naive_hi[i]
    color = COL_GRAY if i not in selected else COL_RED
    ax.plot([x_pos[i], x_pos[i]], [naive_lo[i], naive_hi[i]],
            color=color, lw=3, solid_capstyle="round",
            zorder=2, alpha=0.85 if i not in selected else 1.0)
    ax.plot(x_pos[i], phat[i], "o", color="black", zorder=4, ms=6)
    label_y = -0.13
    ax.text(x_pos[i], label_y, f"{'MISSES' if not covers else 'covers'} 0.5",
            ha="center", fontsize=8.5,
            color=COL_RED if not covers else "#555555")

# overlay e-BY corrected intervals for the two selected coins, offset slightly
offset = 0.16
for i in selected:
    ax.plot([x_pos[i] + offset, x_pos[i] + offset], [eby_lo[i], eby_hi[i]],
            color=COL_BLUE, lw=3, solid_capstyle="round", zorder=3)
    ax.plot(x_pos[i] + offset, phat[i], "o", color=COL_BLUE, zorder=5, ms=5)

ax.axhline(theta_true, color=COL_GREEN, ls="--", lw=1.4, zorder=1,
           label=r"true bias $\theta^*=0.5$ (all coins secretly fair)")

ax.set_xticks(x_pos)
ax.set_xticklabels([f"Coin {c}\n({h}/{n_flips} heads)" for c, h in zip(coins, heads)])
ax.set_ylim(-0.22, 1.05)
ax.set_ylabel(r"coin bias $\theta$")
ax.set_title("Naive 95% e-CIs (gray/red) vs. e-BY-corrected 98.33% e-CIs (blue)\nfor the two selected 'most extreme' coins A, B")

handles = [
    plt.Line2D([0], [0], color=COL_GRAY, lw=3, label="naive 95% e-CI (not selected)"),
    plt.Line2D([0], [0], color=COL_RED, lw=3, label="naive 95% e-CI (selected A, B)"),
    plt.Line2D([0], [0], color=COL_BLUE, lw=3, label=r"e-BY $(1-\alpha_i)$-e-CI, $\alpha_i=\delta|S|/K$"),
    plt.Line2D([0], [0], color=COL_GREEN, ls="--", lw=1.4, label=r"true $\theta^*=0.5$"),
]
ax.legend(handles=handles, loc="upper right", fontsize=8, framealpha=0.9, ncol=1)
fig.tight_layout()
fig.savefig("fig_econfidence_intervals.pdf")
plt.close(fig)

print("Figure (a) numbers:")
for i, c in enumerate(coins):
    print(f"  Coin {c}: phat={phat[i]:.2f}  naive 95% CI=[{naive_lo[i]:.3f},{naive_hi[i]:.3f}]"
          f"  covers0.5={naive_lo[i] <= theta_true <= naive_hi[i]}")
for i in selected:
    print(f"  Coin {coins[i]}: e-BY 98.33% CI=[{eby_lo[i]:.3f},{eby_hi[i]:.3f}]"
          f"  covers0.5={eby_lo[i] <= theta_true <= eby_hi[i]}")


# ---------------------------------------------------------------------------
# Figure (b): repeated-trials FCR simulation
# ---------------------------------------------------------------------------
rng = np.random.default_rng(2024)
n_trials = 20000

fcp_naive = np.empty(n_trials)
fcp_corrected = np.empty(n_trials)

for t in range(n_trials):
    x = rng.binomial(n_flips, theta_true, size=K)
    ph = x / n_flips
    dev = np.abs(ph - 0.5)
    order = np.argsort(-dev)
    sel = order[:L]
    alpha_i = delta * L / K  # e-BY: alpha_i = delta|S|/K (here |S|=L is fixed)

    miss_naive = 0
    miss_corr = 0
    for i in sel:
        lo_n, hi_n = clopper_pearson(x[i], n_flips, 1 - delta)
        lo_c, hi_c = clopper_pearson(x[i], n_flips, 1 - alpha_i)
        if not (lo_n <= theta_true <= hi_n):
            miss_naive += 1
        if not (lo_c <= theta_true <= hi_c):
            miss_corr += 1
    fcp_naive[t] = miss_naive / L
    fcp_corrected[t] = miss_corr / L

running_naive = np.cumsum(fcp_naive) / np.arange(1, n_trials + 1)
running_corrected = np.cumsum(fcp_corrected) / np.arange(1, n_trials + 1)

print(f"\nFigure (b) numbers (n_trials={n_trials}):")
print(f"  naive FCR estimate:      {running_naive[-1]:.4f}")
print(f"  e-BY corrected FCR estimate: {running_corrected[-1]:.4f}")
print(f"  target delta = {delta}")

fig, ax = plt.subplots(figsize=(8.2, 4.6))
t_axis = np.arange(1, n_trials + 1)
ax.plot(t_axis, running_naive, color=COL_RED, lw=1.8,
        label=f"naive: report nominal $(1-\\delta)$-CI (final FCR $\\approx$ {running_naive[-1]:.3f})")
ax.plot(t_axis, running_corrected, color=COL_BLUE, lw=1.8,
        label=f"e-BY: $\\alpha_i=\\delta|S|/K$ (final FCR $\\approx$ {running_corrected[-1]:.3f})")
ax.axhline(delta, color=COL_GREEN, ls="--", lw=1.4,
           label=r"target level $\delta = 0.05$")
ax.set_xscale("log")
ax.set_xlabel("number of simulated trials (log scale)")
ax.set_ylabel("running-average FCR estimate")
ax.set_title(r"FCR of selecting the $L{=}2$ most extreme of $K{=}6$ independent coins"
             "\nacross repeated trials: naive vs. e-BY-corrected")
ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
ax.set_ylim(0, max(0.02, running_naive.max() * 1.15))
fig.tight_layout()
fig.savefig("fig_fcr_simulation.pdf")
plt.close(fig)

print("\nDone. Wrote fig_econfidence_intervals.pdf and fig_fcr_simulation.pdf")
