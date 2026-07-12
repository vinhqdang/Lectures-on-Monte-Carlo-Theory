"""
gen_figures.py
Generates all figures for Chapter 1 (Introduction) slides on
"Hypothesis Testing with E-Values" (Ramdas & Wang, 2024).

Running example throughout: testing whether a coin is fair,
H0: theta = 0.5, using n = 20 flips with 14 observed heads,
and the likelihood-ratio e-value against the alternative theta = 0.7.

Run with:  python3 gen_figures.py
Saves all figures as vector PDFs into this directory.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from math import comb

rng = np.random.default_rng(2024)

# ---------------------------------------------------------------------------
# Running-example constants
# ---------------------------------------------------------------------------
P_NULL = 0.5     # H0: fair coin
Q_ALT = 0.7      # alternative used to build the likelihood-ratio e-value
N = 20           # number of flips in the worked example
S_OBS = 14       # observed number of heads

def lr_evalue(s, n, p=P_NULL, q=Q_ALT):
    """Likelihood-ratio e-value E_n = (q/p)^s * ((1-q)/(1-p))^(n-s)."""
    return (q / p) ** s * ((1 - q) / (1 - p)) ** (n - s)

# Print the worked-example arithmetic so it can be pasted verbatim into slides
e_obs = lr_evalue(S_OBS, N)
print("=== Worked example: n=20 flips, 14 heads, H0: theta=0.5 vs theta=0.7 ===")
print(f"E_20 = (0.7/0.5)^14 * (0.3/0.5)^6 = 1.4^14 * 0.6^6 = {1.4**14:.4f} * {0.6**6:.6f} = {e_obs:.4f}")

# One-sided exact binomial p-value: P(S_20 >= 14 | theta=0.5)
p_obs = sum(comb(N, k) * 0.5 ** N for k in range(S_OBS, N + 1))
print(f"Exact one-sided p-value P(S_20 >= 14 | H0) = {p_obs:.4f}")

# ---------------------------------------------------------------------------
# Figure 1: p-values vs e-values under the null (repeated experiments)
# ---------------------------------------------------------------------------
n_reps = 20000
n_flips = N

S = rng.binomial(n_flips, P_NULL, size=n_reps)

# p-values: exact one-sided binomial tail probability P(Bin(n,0.5) >= s)
# Precompute the survival function for s = 0..n
tail = np.array([sum(comb(n_flips, k) * 0.5 ** n_flips for k in range(s, n_flips + 1))
                  for s in range(n_flips + 1)])
pvals = tail[S]

# e-values: likelihood-ratio e-value against theta=0.7
evals = lr_evalue(S, n_flips)

print(f"\nMonte Carlo check under H0 (n_reps={n_reps}):")
print(f"  mean(p-value) = {pvals.mean():.4f}  (should be ~0.5, since P is stochastically >= U(0,1))")
print(f"  mean(e-value) = {evals.mean():.4f}  (should be <= 1, by definition of an e-variable)")
print(f"  P(e-value > 20) = {np.mean(evals > 20):.4f}  (Markov: should be <= 1/20 = 0.05)")

fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6))

axes[0].hist(pvals, bins=30, range=(0, 1), color='#4C72B0', edgecolor='white', density=True)
axes[0].axhline(1.0, color='black', linestyle='--', linewidth=1, label='Uniform(0,1) density')
axes[0].set_title('P-values under $H_0$\n(stochastically $\\geq$ Uniform(0,1))')
axes[0].set_xlabel('p-value')
axes[0].set_ylabel('density')
axes[0].legend(fontsize=8)

# For the e-value histogram, clip a long right tail for display but report the mean computed
# on the raw (unclipped) data above.
clip = 8
axes[1].hist(np.clip(evals, 0, clip), bins=30, range=(0, clip), color='#DD8452',
             edgecolor='white', density=True)
axes[1].axvline(evals.mean(), color='black', linestyle='--', linewidth=1,
                 label=f'mean = {evals.mean():.3f} $\\leq 1$')
axes[1].set_title('E-values under $H_0$\n(right-skewed, mean $\\leq 1$)')
axes[1].set_xlabel('e-value (clipped at 8 for display)')
axes[1].legend(fontsize=8)

fig.suptitle('Same 20-flip experiment, repeated 20{,}000 times under $H_0$: '
              'p-value vs.\\ e-value', fontsize=10)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig('fig_pvalue_vs_evalue_hist.pdf')
plt.close(fig)

# ---------------------------------------------------------------------------
# Figure 2: betting wealth process (supermartingale under the null)
# ---------------------------------------------------------------------------
n_rounds = 300
n_paths = 8

fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))

# Left panel: under H0 (theta = 0.5), wealth = running product of per-flip e-values
alpha = 0.05
threshold = 1.0 / alpha
for i in range(n_paths):
    flips = rng.binomial(1, P_NULL, size=n_rounds)
    per_flip_e = np.where(flips == 1, Q_ALT / P_NULL, (1 - Q_ALT) / (1 - P_NULL))
    wealth = np.cumprod(per_flip_e)
    axes[0].plot(wealth, linewidth=0.9, alpha=0.85)
axes[0].axhline(threshold, color='black', linestyle='--', linewidth=1,
                 label=f'reject if wealth $> 1/\\alpha = {threshold:.0f}$')
axes[0].set_yscale('log')
axes[0].set_title('Wealth under $H_0$ (fair coin)\n(supermartingale: rarely crosses threshold)')
axes[0].set_xlabel('round (flip number)')
axes[0].set_ylabel('wealth (log scale)')
axes[0].legend(fontsize=7, loc='upper left')

# Right panel: under the true alternative (theta = 0.7), wealth grows exponentially
for i in range(n_paths):
    flips = rng.binomial(1, Q_ALT, size=n_rounds)
    per_flip_e = np.where(flips == 1, Q_ALT / P_NULL, (1 - Q_ALT) / (1 - P_NULL))
    wealth = np.cumprod(per_flip_e)
    axes[1].plot(wealth, linewidth=0.9, alpha=0.85)
axes[1].axhline(threshold, color='black', linestyle='--', linewidth=1,
                  label=f'reject if wealth $> 1/\\alpha = {threshold:.0f}$')
axes[1].set_yscale('log')
axes[1].set_title('Wealth under the alternative (biased coin, $\\theta=0.7$)\n'
                    '(wealth grows exponentially)')
axes[1].set_xlabel('round (flip number)')
axes[1].legend(fontsize=7, loc='upper left')

fig.suptitle('Betting interpretation: Skeptic\'s wealth process, 8 sample paths', fontsize=10)
fig.tight_layout(rect=[0, 0, 1, 0.90])
fig.savefig('fig_betting_wealth.pdf')
plt.close(fig)

# Numerically confirm the supermartingale / Ville-inequality property
n_reps2 = 5000
max_wealth = np.zeros(n_reps2)
for r in range(n_reps2):
    flips = rng.binomial(1, P_NULL, size=n_rounds)
    per_flip_e = np.where(flips == 1, Q_ALT / P_NULL, (1 - Q_ALT) / (1 - P_NULL))
    wealth = np.cumprod(per_flip_e)
    max_wealth[r] = wealth.max()
print(f"\nVille's inequality check under H0 ({n_reps2} reps, {n_rounds} rounds each):")
print(f"  P(max wealth over time > 1/alpha={threshold:.0f}) = {np.mean(max_wealth > threshold):.4f} "
      f"(should be <= alpha = {alpha})")

# ---------------------------------------------------------------------------
# Figure 3: e-process growth, single run, null vs alternative (log scale)
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.0, 3.6))
flips_null = rng.binomial(1, P_NULL, size=n_rounds)
flips_alt = rng.binomial(1, Q_ALT, size=n_rounds)
e_null = np.cumprod(np.where(flips_null == 1, Q_ALT / P_NULL, (1 - Q_ALT) / (1 - P_NULL)))
e_alt = np.cumprod(np.where(flips_alt == 1, Q_ALT / P_NULL, (1 - Q_ALT) / (1 - P_NULL)))
ax.plot(np.log(e_null), label='data from $H_0$ ($\\theta=0.5$)', color='#4C72B0')
ax.plot(np.log(e_alt), label='data from alternative ($\\theta=0.7$)', color='#DD8452')
ax.axhline(np.log(threshold), color='black', linestyle='--', linewidth=1,
            label='$\\log(1/\\alpha)$ threshold')
ax.set_xlabel('round (flip number)')
ax.set_ylabel('$\\log$(e-process)')
ax.set_title('Growth of the likelihood-ratio e-process $M_n$')
ax.legend(fontsize=7)
fig.tight_layout()
fig.savefig('fig_eprocess_growth.pdf')
plt.close(fig)

print("\nAll figures written to the figures/ directory:")
print("  fig_pvalue_vs_evalue_hist.pdf")
print("  fig_betting_wealth.pdf")
print("  fig_eprocess_growth.pdf")
