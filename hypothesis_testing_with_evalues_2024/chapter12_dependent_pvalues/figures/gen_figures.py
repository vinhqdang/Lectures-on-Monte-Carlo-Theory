"""
Generate figures for Chapter 12 slides:
  "Using e-values to combine dependent p-values"
  (Ramdas & Wang, "Hypothesis Testing with E-Values", arXiv:2410.23614)

Figure 1 (fig_meanpvalue_validity.pdf):
    Confirms empirically that 2 * (arithmetic mean of K p-values) is a valid
    p-value even under STRONG, worst-case-style dependence that we construct
    on purpose (a "cyclic shift" / comonotonic-type copula: P_k is a
    deterministic function of a single common uniform U, so the K p-values
    are about as dependent as they can be while remaining marginally
    Uniform(0,1)). We show (a) the histogram of 2*mean(P) and (b) its
    empirical CDF against the diagonal Uniform(0,1) CDF -- validity means the
    empirical CDF must lie at or below the diagonal everywhere.

Figure 2 (fig_exchangeable_power.pdf):
    Compares the power of two valid merging rules for K *exchangeable*
    p-values, both built from the same convex calibrator f(p) = (2-2p)_+:
      - the "arbitrary dependence" rule of Theorem 12.17 (single average
        over all K p-values -- valid for ANY dependence structure);
      - the "exchangeable" rule of Theorem 12.27 (running/prefix average,
        taking the max over prefixes 1..K -- only valid when the p-values
        are exchangeable, but strictly more powerful).
    Power is measured by simulating p-values from Beta(a,1) (a<1 favors
    small p-values, i.e. there is real signal) and reporting rejection rate
    at a fixed significance level alpha.
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

rng = np.random.default_rng(2024)


def f_calibrator(x):
    """The convex calibrator f(p) = (2 - 2p)_+ used throughout Chapter 12
    (Example 12.21): it satisfies int_0^1 f = 1 and f(p) = 0 for p >= 1,
    so it is an admissible calibrator turning a p-value into an e-value."""
    return np.clip(2.0 - 2.0 * x, 0.0, None)


def merged_pvalue(p_row, use_max_prefix):
    """Compute the merged p-value F(p) via bisection on epsilon, using
    Theorem 12.17 (use_max_prefix=False, single average over all K terms)
    or Theorem 12.27 (use_max_prefix=True, max over prefix-averages), both
    instantiated with the calibrator f_calibrator above.

    F(p) = inf{ eps in (0,1] : criterion(eps) >= 1 },
    where criterion(eps) = mean_k f(p_k/eps)                (Thm 12.17)
                          = max_{l=1..K} mean_{k<=l} f(p_k/eps)  (Thm 12.27)
    criterion(eps) is nondecreasing in eps, so bisection applies.
    """
    K = len(p_row)

    def criterion(eps):
        vals = f_calibrator(p_row / eps)
        if use_max_prefix:
            cums = np.cumsum(vals)
            ells = np.arange(1, K + 1)
            return np.max(cums / ells)
        return np.mean(vals)

    lo, hi = 1e-12, 1.0
    while criterion(hi) < 1.0 and hi < 1e6:
        hi *= 2.0
    if criterion(hi) < 1.0:
        return 1.0  # criterion never reaches 1: merged p-value is (capped at) 1
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if criterion(mid) >= 1.0:
            hi = mid
        else:
            lo = mid
    return min(hi, 1.0)


# ---------------------------------------------------------------------------
# Figure 1: validity of 2*mean(p) under strong (worst-case-style) dependence
# ---------------------------------------------------------------------------
K = 8
N = 50000
rho = 0.95  # strength of the constructed dependence, close to the extremal case

# Strong-dependence (Gaussian-copula) construction: half of the p-values
# load POSITIVELY and half load NEGATIVELY on a single common factor W, so
# that when one half's p-values tend to be large, the other half's tend to
# be small -- the same qualitative mechanism as the book's extremal example
# (P_1=U, P_2=1-U, whose average is the constant 1/2, showing the factor 2
# cannot be improved). Each P_k is still marginally exactly Uniform(0,1);
# only the joint (strongly dependent, here anti-dependent across the two
# halves) structure is unusual.
W = rng.standard_normal(N)
eps = rng.standard_normal((N, K))
loadings = np.where(np.arange(K) < K // 2, rho, -rho)
Z = loadings[None, :] * W[:, None] + np.sqrt(1 - rho**2) * eps
from scipy.stats import norm
P = norm.cdf(Z)
mean_P = P.mean(axis=1)
twice_mean = np.minimum(2.0 * mean_P, 1.0)

alphas = np.linspace(0.001, 1.0, 400)
ecdf = np.array([(twice_mean <= a).mean() for a in alphas])

fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))

axes[0].hist(twice_mean, bins=50, density=True, color="#4C72B0",
             alpha=0.85, edgecolor="white")
axes[0].axhline(1.0, color="gray", linestyle="--", linewidth=1.2)
axes[0].set_xlabel(r"$2\bar P = 2 \cdot \mathrm{mean}(P_1,\dots,P_K)$")
axes[0].set_ylabel("density")
axes[0].set_title(f"Histogram of $2\\bar P$, $K={K}$ strongly anti-dependent p-values")

axes[1].plot(alphas, ecdf, color="#C44E52", linewidth=2.2,
             label=r"empirical $\mathbb{P}(2\bar P \leq \alpha)$")
axes[1].plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1.2,
             label=r"Uniform(0,1) CDF (validity boundary)")
axes[1].set_xlabel(r"$\alpha$")
axes[1].set_ylabel(r"$\mathbb{P}(2\bar P \leq \alpha)$")
axes[1].set_title("Type-I error stays $\\leq\\alpha$ under strong dependence")
axes[1].legend(loc="upper left", fontsize=9.5)
axes[1].set_xlim(0, 1)
axes[1].set_ylim(0, 1)

fig.tight_layout()
fig.savefig("fig_meanpvalue_validity.pdf")
plt.close(fig)

# ---------------------------------------------------------------------------
# Figure 2: power comparison, arithmetic-mean merge vs exchangeable merge
# ---------------------------------------------------------------------------
K2 = 5
N2 = 4000
alpha = 0.05
a_grid = np.array([1.0, 0.85, 0.7, 0.55, 0.4, 0.3, 0.22, 0.16, 0.12, 0.09])

power_arith = []
power_exch = []
for a in a_grid:
    p_data = rng.beta(a, 1.0, size=(N2, K2))
    # Randomly permute each row: harmless for iid draws, but makes explicit
    # that we only use exchangeability (not a known ordering/independence).
    idx = rng.permuted(np.tile(np.arange(K2), (N2, 1)), axis=1)
    p_data = np.take_along_axis(p_data, idx, axis=1)

    merged_arith = np.array([merged_pvalue(row, use_max_prefix=False) for row in p_data])
    merged_exch = np.array([merged_pvalue(row, use_max_prefix=True) for row in p_data])

    power_arith.append(np.mean(merged_arith <= alpha))
    power_exch.append(np.mean(merged_exch <= alpha))

power_arith = np.array(power_arith)
power_exch = np.array(power_exch)

fig2, ax = plt.subplots(figsize=(7.5, 5.2))
ax.plot(a_grid, power_arith, "o-", color="#4C72B0", linewidth=2,
         label="Arbitrary-dependence rule (Thm 12.17, full average)")
ax.plot(a_grid, power_exch, "s-", color="#C44E52", linewidth=2,
         label="Exchangeable rule (Thm 12.27, max over prefixes)")
ax.axhline(alpha, color="gray", linestyle=":", linewidth=1.2)
ax.text(a_grid[0], alpha + 0.012, rf"target level $\alpha={alpha}$", color="gray", fontsize=9.5)
ax.invert_xaxis()
ax.set_xlabel(r"Beta$(a,1)$ shape parameter $a$   ($a=1$: null; smaller $a$: stronger signal)")
ax.set_ylabel(rf"power $=\ \mathbb{{P}}(\text{{merged }} p \leq \alpha={alpha})$")
ax.set_title(f"Power comparison for $K={K2}$ exchangeable p-values")
ax.legend(loc="upper left", fontsize=9.5)
ax.set_ylim(0, 1.02)

fig2.tight_layout()
fig2.savefig("fig_exchangeable_power.pdf")
plt.close(fig2)

print("Wrote fig_meanpvalue_validity.pdf and fig_exchangeable_power.pdf")
print(f"Type-I error at a=1: arithmetic-mean rule = {power_arith[0]:.4f}, "
      f"exchangeable rule = {power_exch[0]:.4f} (both should be <= alpha={alpha})")
