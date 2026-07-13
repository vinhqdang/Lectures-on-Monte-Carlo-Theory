"""
Generate figures for Chapter 2 (Background to Bayesian Machine Learning
in Quantitative Finance) slides on:
Bayesian Machine Learning in Quantitative Finance (Mongwe, Mbuvha & Marwala, 2025)

Figures produced (all saved as vector PDF):
  1. fig_freq_vs_bayes.pdf -- Side-by-side comparison of a frequentist point
                              estimate vs. a full Bayesian posterior
                              distribution, on a tiny coin-flip toy problem
                              (5 flips: H, T, H, H, T -> 3 heads, 2 tails).
  2. fig_roc_auc.pdf       -- ROC curve for a toy binary classifier, with the
                              Area Under the Curve (AUC) computed by the
                              trapezoidal rule and marked on the plot.
"""

import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta as beta_dist

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "figure.dpi": 150,
})

# =============================================================================
# Figure 1: Frequentist point estimate vs. Bayesian full posterior
# =============================================================================
# Toy parameter-estimation problem: theta = P(coin lands heads).
# Data: 5 flips = H, T, H, H, T  ->  3 heads, 2 tails.
data = [1, 0, 1, 1, 0]
n = len(data)
n_heads = sum(data)
n_tails = n - n_heads

# --- Frequentist side: Maximum Likelihood Estimate (MLE) ---
theta_mle = n_heads / n  # = 0.6, a single number, no uncertainty attached

# --- Bayesian side: Beta(1,1) uniform prior -> Beta(1+3, 1+2) posterior ---
a_prior, b_prior = 1.0, 1.0
a_post, b_post = a_prior + n_heads, b_prior + n_tails  # Beta(4,3)

theta_grid = np.linspace(0.001, 0.999, 500)
post_pdf = beta_dist.pdf(theta_grid, a_post, b_post)
post_mean = a_post / (a_post + b_post)
ci_low, ci_high = beta_dist.ppf(0.025, a_post, b_post), beta_dist.ppf(0.975, a_post, b_post)

fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))

# --- Left panel: frequentist point estimate ---
ax = axes[0]
ax.axhline(0, color="black", lw=1)
ax.plot([theta_mle], [0], marker="o", markersize=14, color="#C44E52", zorder=5)
ax.annotate(rf"$\hat{{\theta}}_{{MLE}} = {theta_mle:.2f}$",
            xy=(theta_mle, 0), xytext=(theta_mle, 0.35),
            ha="center", fontsize=12, color="#C44E52",
            arrowprops=dict(arrowstyle="->", color="#C44E52", lw=1.2))
ax.set_xlim(0, 1)
ax.set_ylim(-0.3, 1.0)
ax.set_yticks([])
ax.set_xlabel(r"$\theta$ = probability the coin lands heads")
ax.set_title("Frequentist: a single point estimate\n(no uncertainty attached)")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_visible(False)
for x in np.linspace(0, 1, 6):
    ax.plot([x, x], [-0.03, 0.03], color="black", lw=1)
    ax.text(x, -0.12, f"{x:.1f}", ha="center", fontsize=8.5)

# --- Right panel: Bayesian full posterior distribution ---
ax = axes[1]
ax.plot(theta_grid, post_pdf, color="#4C72B0", lw=2.5,
        label=rf"Posterior: Beta({a_post:.0f},{b_post:.0f})")
ax.fill_between(theta_grid, post_pdf, color="#4C72B0", alpha=0.15)
ax.axvspan(ci_low, ci_high, color="#4C72B0", alpha=0.10,
           label=f"95% credible interval\n[{ci_low:.2f}, {ci_high:.2f}]")
ax.axvline(post_mean, color="#2F4C6B", ls="--", lw=1.5)
ax.annotate(f"posterior mean = {post_mean:.2f}",
            xy=(post_mean, beta_dist.pdf(post_mean, a_post, b_post)),
            xytext=(post_mean - 0.35, beta_dist.pdf(post_mean, a_post, b_post) + 0.25),
            fontsize=10, color="#2F4C6B",
            arrowprops=dict(arrowstyle="->", color="#2F4C6B", lw=1))
ax.set_xlim(0, 1)
ax.set_ylim(bottom=0)
ax.set_xlabel(r"$\theta$ = probability the coin lands heads")
ax.set_ylabel("posterior density")
ax.set_title("Bayesian: a full distribution over $\\theta$\n(quantifies the uncertainty)")
ax.legend(loc="upper left", fontsize=8.5, frameon=True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.suptitle("Toy parameter estimation: 5 coin flips (H,T,H,H,T)", fontsize=13, y=1.02)
fig.tight_layout()
fig.savefig("fig_freq_vs_bayes.pdf", bbox_inches="tight")
plt.close(fig)

print("Figure 1 summary:")
print(f"  Data: {n_heads} heads, {n_tails} tails out of {n} flips")
print(f"  Frequentist MLE: theta_hat = {theta_mle:.4f}")
print(f"  Bayesian posterior: Beta({a_post:.0f},{b_post:.0f}), mean = {post_mean:.4f}, "
      f"95% CI = [{ci_low:.4f}, {ci_high:.4f}]")

# =============================================================================
# Figure 2: ROC curve for a toy binary classifier, with AUC marked
# =============================================================================
# Toy dataset: 10 examples with true labels and model scores (predicted
# probability of the positive class).
rng = np.random.default_rng(42)
y_true = np.array([1, 1, 1, 1, 0, 1, 0, 0, 0, 0])
y_score = np.array([0.92, 0.85, 0.78, 0.66, 0.61, 0.55, 0.48, 0.42, 0.30, 0.11])

thresholds = np.concatenate(([1.1], np.sort(y_score)[::-1], [-0.1]))
tpr_list = []
fpr_list = []
P = np.sum(y_true == 1)
N = np.sum(y_true == 0)

for t in thresholds:
    y_pred = (y_score >= t).astype(int)
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    tpr_list.append(tp / P)
    fpr_list.append(fp / N)

fpr = np.array(fpr_list)
tpr = np.array(tpr_list)

# Sort by FPR for a monotone curve, then integrate with the trapezoidal rule
order = np.argsort(fpr)
fpr_sorted = fpr[order]
tpr_sorted = tpr[order]
auc = np.trapezoid(tpr_sorted, fpr_sorted) if hasattr(np, "trapezoid") else np.trapz(tpr_sorted, fpr_sorted)

fig, ax = plt.subplots(figsize=(6.4, 5.4))
ax.plot(fpr_sorted, tpr_sorted, color="#4C72B0", lw=2.5, marker="o", markersize=5,
        label="ROC curve (toy classifier)")
ax.fill_between(fpr_sorted, tpr_sorted, color="#4C72B0", alpha=0.15)
ax.plot([0, 1], [0, 1], color="gray", lw=1.5, ls="--", label="Random guessing (AUC = 0.50)")
ax.text(0.55, 0.30, f"AUC = {auc:.3f}", fontsize=14, color="#2F4C6B",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#EAF0F8", edgecolor="#4C72B0"))
ax.set_xlim(-0.02, 1.02)
ax.set_ylim(-0.02, 1.02)
ax.set_xlabel("False Positive Rate (FPR)")
ax.set_ylabel("True Positive Rate (TPR)")
ax.set_title("ROC curve and Area Under the Curve (AUC)\nfor a toy binary classifier")
ax.legend(loc="lower right", fontsize=9.5, frameon=True)
ax.set_aspect("equal")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.tight_layout()
fig.savefig("fig_roc_auc.pdf", bbox_inches="tight")
plt.close(fig)

print("\nFigure 2 summary:")
print(f"  y_true  = {y_true.tolist()}")
print(f"  y_score = {y_score.tolist()}")
print(f"  AUC (trapezoidal rule) = {auc:.4f}")

print("\nFigures written: fig_freq_vs_bayes.pdf, fig_roc_auc.pdf")
