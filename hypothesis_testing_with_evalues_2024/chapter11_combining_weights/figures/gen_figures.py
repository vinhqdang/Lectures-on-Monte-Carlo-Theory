"""
Generate figures for Chapter 11 slides:
  "Combining e-values and using e-values as weights"
  (Ramdas & Wang, "Hypothesis Testing with E-Values", arXiv:2410.23614)

Figure 1 (fig_merging.pdf):
    The running numerical example of Section 11.2. Three e-values
    E1 = 0.6, E2 = 2.4, E3 = 1.1 are combined by two different
    admissible e-merging functions M_lambda:
      - the simple average (lambda_0 = 0, lambda_k = 1/3 each)
      - a "skeptical" weighted average with an intercept
        (lambda_0 = 0.4, lambda_k = 0.2 each)
    Both are valid e-values on average, and we visualize how the
    intercept term lambda_0 shrinks the combined value toward 1.

Figure 2 (fig_weights.pdf):
    Section 11.4's e-value-as-weight idea. For a fixed nominal
    threshold alpha = 0.05, an e-value E_k used as a weight turns
    the effective rejection threshold for the raw p-value P_k into
    alpha * E_k (capped at 1), since P_k/E_k <= alpha  <=>  P_k <= alpha*E_k.
    We show a toy panel of tests with different e-value weights and
    the resulting (loosened or tightened) thresholds.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.size": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# ---------------------------------------------------------------------------
# Figure 1: merging three e-values with two different e-merging functions
# ---------------------------------------------------------------------------
E = np.array([0.6, 2.4, 1.1])
labels_E = [r"$E_1=0.6$", r"$E_2=2.4$", r"$E_3=1.1$"]

# Simple average: lambda_0 = 0, lambda_k = 1/3
M_avg = np.mean(E)

# Skeptical weighted average: lambda_0 = 0.4, lambda_k = 0.2 each
lam0 = 0.4
lam = np.array([0.2, 0.2, 0.2])
M_skeptical = lam0 + np.sum(lam * E)

fig, ax = plt.subplots(figsize=(7.5, 4.2))

bar_labels = labels_E + [r"$M_{\mathrm{avg}}$", r"$M_{\boldsymbol{\lambda}}$"]
values = list(E) + [M_avg, M_skeptical]
colors = ["#4C72B0", "#4C72B0", "#4C72B0", "#DD8452", "#55A868"]

bars = ax.bar(bar_labels, values, color=colors, edgecolor="black", linewidth=0.6, width=0.6)
ax.axhline(1.0, color="gray", linestyle="--", linewidth=1.2)
ax.text(4.55, 1.05, r"$e=1$ (no evidence)", color="gray", fontsize=10, ha="right")

for b, v in zip(bars, values):
    ax.text(b.get_x() + b.get_width()/2, v + 0.05, f"{v:.3f}", ha="center", fontsize=10)

ax.set_ylabel("e-value")
ax.set_title("Combining three e-values via two admissible e-merging functions")
ax.set_ylim(0, 3.0)
fig.tight_layout()
fig.savefig("fig_merging.pdf")
plt.close(fig)

# ---------------------------------------------------------------------------
# Figure 2: e-values used as weights loosen/tighten a p-value threshold
# ---------------------------------------------------------------------------
alpha = 0.05
weights = np.array([0.2, 0.5, 1.0, 2.0, 4.0, 8.0])
eff_threshold = np.minimum(alpha * weights, 1.0)

fig, ax = plt.subplots(figsize=(8.0, 4.2))
x = np.arange(len(weights))

bars = ax.bar(x, eff_threshold, color="#4C72B0", edgecolor="black", linewidth=0.6, width=0.55)
ax.axhline(alpha, color="gray", linestyle="--", linewidth=1.2)
ax.text(len(weights) - 0.4, alpha + 0.015, r"unweighted $\alpha=0.05$", color="gray",
        fontsize=10, ha="right")

for xi, (w, t) in zip(x, zip(weights, eff_threshold)):
    ax.text(xi, t + 0.01, f"{t:.3f}", ha="center", fontsize=10)

ax.set_xticks(x)
ax.set_xticklabels([f"$E_k={w:g}$" for w in weights])
ax.set_ylabel(r"effective threshold $\alpha \cdot E_k \wedge 1$")
ax.set_title(r"Using an e-value as a weight: effective threshold for $P_k$ at nominal $\alpha=0.05$")
ax.set_ylim(0, 1.05)
fig.tight_layout()
fig.savefig("fig_weights.pdf")
plt.close(fig)

print("Wrote fig_merging.pdf and fig_weights.pdf")
