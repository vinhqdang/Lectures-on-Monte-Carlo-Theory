#!/usr/bin/env python3
"""
gen_figures.py

Generates the figures used in the Chapter 8 (Handling multiple e-values)
Beamer slides for "Hypothesis Testing with E-Values" (Ramdas & Wang, 2024).

Figure 1 (fig_average_dependence.pdf):
    Demonstrates, by simulation, that the ARITHMETIC MEAN of K e-values
    remains a valid e-value (empirical mean <= 1 under the null) even when
    the e-values are strongly, adversarially dependent -- constructed here
    via a shared latent "shock" variable W that simultaneously inflates all
    K e-values together. In contrast, the SUM and the PRODUCT of the same
    dependent e-values are *not* valid e-values in general: their empirical
    means drift far above 1.

Figure 2 (fig_product_vs_average.pdf):
    Illustrates the "mean-variance trade-off" of Section 8.4: for K
    INDEPENDENT e-values (each worth 4 w.p. 1/2 and 0 w.p. 1/2 under the
    alternative, so each has mean 2), the PRODUCT grows exponentially in
    expectation but collapses to 0 with probability 1 - 2^{-K} (i.e. its
    typical/median behaviour is far worse than its mean suggests), whereas
    the AVERAGE grows only linearly but concentrates reliably around its
    mean by the law of large numbers -- i.e. it is the "safe" merge.

Usage:
    python3 gen_figures.py
Outputs PDFs into the current directory (figures/).
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rng = np.random.default_rng(20240813)

plt.rcParams.update({
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# ============================================================
# Figure 1: averaging survives arbitrary dependence
# ============================================================

K = 5
p = 0.1  # probability of the common "shock" W = 1

# Choose "high" values a_k (realized when the shock hits, W=1) so that
# each E_k, marginalized over W, has mean exactly 1:
#   E[E_k] = p*a_k + (1-p)*b_k = 1  =>  b_k = (1 - p*a_k) / (1-p)
a = np.array([8.0, 6.0, 10.0, 5.0, 9.0])
b = (1 - p * a) / (1 - p)
assert np.all(b >= 0), "baseline values must stay nonnegative"

N = 20000  # number of simulated trials ("repetitions of the experiment")

W = rng.binomial(1, p, size=N)  # shared shock: dependence engine
# realize each e-value: if W=1, e_k = a_k (all K jump together -- strong,
# adversarial dependence); if W=0, e_k = b_k.
E = np.where(W[:, None] == 1, a[None, :], b[None, :])  # shape (N, K)

avg = E.mean(axis=1)
summ = E.sum(axis=1)
prod = E.prod(axis=1)
mx = E.max(axis=1)

# running (cumulative) empirical means, to show convergence
run_avg = np.cumsum(avg) / np.arange(1, N + 1)
run_sum = np.cumsum(summ) / np.arange(1, N + 1)
run_prod = np.cumsum(prod) / np.arange(1, N + 1)
run_max = np.cumsum(mx) / np.arange(1, N + 1)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))

ax = axes[0]
trials = np.arange(1, N + 1)
ax.plot(trials, run_avg, color="#1b7837", lw=1.8, label=r"average $\mathbb{M}_K(\mathbf{E})$")
ax.plot(trials, run_sum, color="#b2182b", lw=1.4, ls="--", label=r"sum $\sum_k E_k$")
ax.plot(trials, run_max, color="#2166ac", lw=1.4, ls="-.", label=r"max$_k E_k$")
ax.axhline(1.0, color="black", lw=1.0, ls=":")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("number of trials $N$ (running empirical mean)")
ax.set_ylabel(r"empirical mean of merged value")
ax.set_title("Under strong (shared-shock) dependence\n" + r"only the average has mean $\leq 1$")
ax.legend(fontsize=8.5, loc="upper right")

ax2 = axes[1]
labels = ["average", "sum", "product", "max"]
finals = [run_avg[-1], run_sum[-1], run_prod[-1], run_max[-1]]
colors = ["#1b7837", "#b2182b", "#762a83", "#2166ac"]
bars = ax2.bar(labels, finals, color=colors, alpha=0.85)
ax2.set_yscale("log")
ax2.axhline(1.0, color="black", lw=1.2, ls=":", label=r"$e$-value threshold: mean $\leq 1$")
for rect, val in zip(bars, finals):
    ax2.text(rect.get_x() + rect.get_width() / 2, val * 1.15, f"{val:.2f}",
              ha="center", va="bottom", fontsize=9)
ax2.set_ylabel(r"empirical mean after $N={:d}$ trials".format(N))
ax2.set_title(f"Merging functions compared ($K={K}$, shared-shock dependence)")
ax2.legend(fontsize=8.5, loc="upper left")

fig.suptitle(
    r"Averaging is safe under arbitrary dependence: $\mathbb{E}[\mathbb{M}_K(\mathbf{E})] \leq 1$ always holds",
    fontsize=11.5, y=1.02,
)
fig.tight_layout()
fig.savefig("fig_average_dependence.pdf", bbox_inches="tight")
plt.close(fig)

print("Wrote fig_average_dependence.pdf")
print(f"  Final empirical means over N={N} trials: "
      f"avg={run_avg[-1]:.3f}, sum={run_sum[-1]:.3f}, "
      f"prod={run_prod[-1]:.3f}, max={run_max[-1]:.3f}")

# ============================================================
# Figure 2: mean-variance trade-off, product vs average
# ============================================================

# Each independent e-value E_k (under the alternative Q) takes value 4
# w.p. 1/2 and 0 w.p. 1/2, so E^Q[E_k] = 2 (a "powered" e-value, as in
# Example 8.17 of the book).
Ks = np.arange(1, 26)
N2 = 4000
val_high, val_low, prob_high = 4.0, 0.0, 0.5

Kmax = Ks[-1]
draws = rng.binomial(1, prob_high, size=(N2, Kmax))  # 1 = "high" (=4), 0 = "low" (=0)
e_vals = np.where(draws == 1, val_high, val_low)  # shape (N2, Kmax)

median_prod, mean_avg, median_avg = [], [], []
frac_zero_prod = []
for K_ in Ks:
    e_K = e_vals[:, :K_]
    prod_K = np.prod(e_K, axis=1)
    avg_K = np.mean(e_K, axis=1)
    median_prod.append(np.median(prod_K))
    mean_avg.append(avg_K.mean())
    median_avg.append(np.median(avg_K))
    frac_zero_prod.append(np.mean(prod_K == 0))

# The theoretical mean of the product is exact: E[E_k] = 2 for each
# independent factor, so E[Pi_K] = 2^K. A *Monte Carlo* estimate of this
# mean is hopeless for large K, since it requires observing the
# probability-2^{-K} event that all K coins land "high" -- exactly the
# point of this figure (the mean is dominated by an astronomically rare
# outcome), so we plot the exact formula rather than the noisy simulated
# mean.
mean_prod_theory = (val_high * prob_high) ** Ks  # = 2**Ks
median_prod = np.array(median_prod)
mean_avg = np.array(mean_avg)
median_avg = np.array(median_avg)
frac_zero_prod = np.array(frac_zero_prod)

fig2, axes2 = plt.subplots(1, 2, figsize=(11, 4.3))

ax = axes2[0]
ax.plot(Ks, mean_prod_theory, color="#762a83", lw=1.8, marker="o", ms=3,
        label=r"$\mathbb{E}^{\mathbb{Q}}[\Pi_K(\mathbf{E})] = 2^K$  (exact)")
ax.plot(Ks, np.maximum(median_prod, 1e-3), color="#762a83", lw=1.4, ls="--", marker="s", ms=3,
        label=r"simulated median of $\Pi_K(\mathbf{E})$  (typically $0$)")
ax.plot(Ks, mean_avg, color="#1b7837", lw=1.8, marker="o", ms=3,
        label=r"$\mathbb{E}^{\mathbb{Q}}[\mathbb{M}_K(\mathbf{E})]$  (theory: $2$)")
ax.plot(Ks, median_avg, color="#1b7837", lw=1.4, ls="--", marker="s", ms=3,
        label=r"median of $\mathbb{M}_K(\mathbf{E})$  (stays near $2$)")
ax.set_yscale("log")
ax.set_xlabel(r"number of independent e-values $K$")
ax.set_ylabel("value (log scale)")
ax.set_title("Product's mean explodes, but its\ntypical (median) value collapses to 0")
ax.legend(fontsize=7.8, loc="center left")

ax2 = axes2[1]
ax2.plot(Ks, frac_zero_prod * 100, color="#b2182b", lw=2.0, marker="o", ms=3)
ax2.set_xlabel(r"number of independent e-values $K$")
ax2.set_ylabel(r"$\mathbb{P}(\Pi_K(\mathbf{E}) = 0)$  (\%)")
ax2.set_title(r"Chance the product is a total loss: $1-2^{-K}$" "\n(the average never collapses like this)")
ax2.set_ylim(0, 102)

fig2.suptitle(
    "Mean-variance trade-off (Section 8.4): product = high mean, high risk; "
    "average = modest mean, reliable",
    fontsize=10.8, y=1.03,
)
fig2.tight_layout()
fig2.savefig("fig_product_vs_average.pdf", bbox_inches="tight")
plt.close(fig2)

print("Wrote fig_product_vs_average.pdf")
print(f"  At K={Kmax}: exact mean(product)={mean_prod_theory[-1]:.3g}, "
      f"simulated median(product)={median_prod[-1]:.3g}, "
      f"mean(avg)={mean_avg[-1]:.3g}, median(avg)={median_avg[-1]:.3g}, "
      f"P(product=0)={frac_zero_prod[-1]*100:.1f}%")
