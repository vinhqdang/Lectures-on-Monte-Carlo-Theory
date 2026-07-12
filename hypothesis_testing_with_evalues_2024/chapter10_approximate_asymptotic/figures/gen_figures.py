"""
gen_figures.py
Generates the two figures used in Chapter 10 (Approximate and Asymptotic
E-Values) slides:

  (a) fig_clt_convergence.pdf
      Shows how the actual null-rejection rate of the CLT-based asymptotic
      e-value E^(n) = exp(lambda * sqrt(n) * Xbar / S - lambda^2/2)
      (Example 10.10 / Theorem 10.11) converges to the nominal level alpha
      as the sample size n grows, when the underlying population is a
      centered Exponential (highly non-Gaussian at small n).

  (b) fig_exact_vs_approximate.pdf
      Compares an exact e-value (E[E] <= 1 exactly, Example 10.12,
      X^2/sigma^2 for the bounded-variance null) against an approximate
      e-value that only satisfies E[E] <= 1 + eps (Definition 10.1) under a
      contaminated / misspecified data-generating law, shading the region
      [1, 1+eps] that the theoretical bound guarantees the approximate
      e-value's expectation must lie within.

Both figures are saved as self-contained PDFs in this directory. The script
also prints the concrete numbers used in the "running numerical example" of
the slides (n = 30, 100, 1000).
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rng = np.random.default_rng(2024)

# ----------------------------------------------------------------------
# Shared style
# ----------------------------------------------------------------------
plt.rcParams.update({
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

COLOR_MAIN = "#1f4e79"
COLOR_ALT = "#c0392b"
COLOR_NOM = "#555555"
COLOR_BAND = "#c0392b"


# ----------------------------------------------------------------------
# Figure (a): convergence of actual rejection rate to nominal level
# ----------------------------------------------------------------------
def clt_evalue(x, lam):
    """E^(n) = exp(lambda * sqrt(n) * xbar / S - lambda^2 / 2),
    S = sqrt(mean(x^2)) as in (10.3)-(10.4)."""
    n = x.shape[-1]
    xbar = x.mean(axis=-1)
    S = np.sqrt((x ** 2).mean(axis=-1))
    return np.exp(lam * np.sqrt(n) * xbar / S - lam ** 2 / 2.0)


def simulate_rejection_rate(n, lam, alpha, n_reps, rng):
    """Population: X = Y - 1 with Y ~ Exp(rate=1), so E[X] = 0, Var(X) = 1,
    but X is strongly right-skewed (skewness = 2) -- a genuinely
    non-Gaussian null population."""
    Y = rng.exponential(scale=1.0, size=(n_reps, n))
    X = Y - 1.0
    E = clt_evalue(X, lam)
    threshold = 1.0 / alpha  # reject H0 when E^(n) >= 1/alpha
    return np.mean(E >= threshold), E


alpha = 0.05
lam = 1.5
n_reps = 200_000
ns = [5, 10, 20, 30, 50, 100, 300, 1000, 3000, 10000]

rejection_rates = []
mean_evalues = []
for n in ns:
    rate, E = simulate_rejection_rate(n, lam, alpha, n_reps, rng)
    rejection_rates.append(rate)
    mean_evalues.append(E.mean())

fig, ax = plt.subplots(figsize=(7.0, 4.6))
ax.axhline(alpha, color=COLOR_NOM, linestyle="--", linewidth=1.5,
           label=f"nominal level $\\alpha={alpha}$")
ax.plot(ns, rejection_rates, "o-", color=COLOR_MAIN, linewidth=2,
        markersize=6, label="actual rejection rate $\\widehat{\\mathbb{P}}(E^{(n)} \\geq 1/\\alpha)$")
ax.set_xscale("log")
ax.set_xlabel("sample size $n$ (log scale)")
ax.set_ylabel("rejection rate under $H_0$")
ax.set_title("CLT-based asymptotic e-value: actual level $\\to$ nominal level")
ax.set_ylim(0, max(rejection_rates) * 1.25 + 0.01)
ax.legend(loc="upper right", frameon=False, fontsize=9)
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig("fig_clt_convergence.pdf")
plt.close(fig)

print("=== Figure (a): CLT convergence data ===")
print(f"{'n':>7} | {'mean E[E^(n)]':>15} | {'rejection rate':>15}")
for n, m, r in zip(ns, mean_evalues, rejection_rates):
    print(f"{n:>7} | {m:>15.4f} | {r:>15.5f}")

# ----------------------------------------------------------------------
# Running numerical example printed for the slides: n = 30, 100, 1000
# Single realized dataset (fixed seed) + the actual E^(n) value
# ----------------------------------------------------------------------
print("\n=== Running example: single realized dataset, lambda =", lam, "===")
rng_example = np.random.default_rng(7)
for n in [30, 100, 1000]:
    Y = rng_example.exponential(scale=1.0, size=n)
    X = Y - 1.0
    xbar = X.mean()
    S = np.sqrt((X ** 2).mean())
    E_val = np.exp(lam * np.sqrt(n) * xbar / S - lam ** 2 / 2.0)
    z_stat = np.sqrt(n) * xbar / S
    print(f"n={n:5d}: xbar={xbar: .4f}, S={S:.4f}, "
          f"sqrt(n)*xbar/S={z_stat: .4f}, E^(n)={E_val:.4f}")

# ----------------------------------------------------------------------
# Figure (b): exact vs. approximate e-value with shaded error bound
# ----------------------------------------------------------------------
# Null model: P = {P : E[X] = 0, Var(X) <= sigma0^2}, sigma0 = 1.
# Exact e-value (Example 10.12): E_exact = X^2 / sigma0^2, satisfies
#   E[E_exact] <= 1 exactly, for ANY P with Var(X) <= sigma0^2.
#
# Approximate e-value: the practitioner uses a plug-in variance estimate
# sigma_hat^2 that is contaminated / biased by a factor (1 - eps) relative
# to sigma0^2 (e.g., a slightly stale variance estimate), so
#   E_approx = X^2 / sigma_hat^2 = X^2 / ((1-eps) sigma0^2).
# Under the true null variance sigma0^2, this satisfies (Definition 10.1)
#   E[E_approx] = E[X^2]/((1-eps)sigma0^2) <= 1/(1-eps) ~= 1 + eps + eps^2 + ...
# so it is an (eps, 0)-approximate e-variable with a KNOWN bound 1+eps'.
sigma0 = 1.0
eps_grid = np.linspace(0.0, 0.30, 16)
n_mc = 400_000

exact_means = []
approx_means = []
approx_bounds = []
for eps in eps_grid:
    X = rng.normal(loc=0.0, scale=sigma0, size=n_mc)
    E_exact = (X ** 2) / sigma0 ** 2
    sigma_hat2 = (1 - eps) * sigma0 ** 2
    E_approx = (X ** 2) / sigma_hat2
    exact_means.append(E_exact.mean())
    approx_means.append(E_approx.mean())
    approx_bounds.append(1.0 / (1 - eps))  # exact theoretical value of E[E_approx]

fig, ax = plt.subplots(figsize=(7.0, 4.6))
ax.fill_between(eps_grid, 1.0, approx_bounds, color=COLOR_BAND, alpha=0.18,
                 label=r"guaranteed band $[1,\ 1+\varepsilon]$ (approx. bound)")
ax.plot(eps_grid, exact_means, "-", color=COLOR_MAIN, linewidth=2.5,
        label=r"exact e-value: $\mathbb{E}[E_{\mathrm{exact}}] \le 1$")
ax.plot(eps_grid, approx_means, "o--", color=COLOR_ALT, linewidth=2,
        markersize=5,
        label=r"approximate e-value: $\mathbb{E}[E_{\mathrm{approx}}] \le 1+\varepsilon$")
ax.plot(eps_grid, approx_bounds, ":", color=COLOR_ALT, linewidth=1.2, alpha=0.8)
ax.set_xlabel(r"misspecification level $\varepsilon$ (variance plug-in error)")
ax.set_ylabel(r"$\mathbb{E}[E]$ under the null")
ax.set_title("Exact e-value vs. $(\\varepsilon,0)$-approximate e-value")
ax.legend(loc="upper left", frameon=False, fontsize=9)
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig("fig_exact_vs_approximate.pdf")
plt.close(fig)

print("\n=== Figure (b): exact vs approximate e-value data ===")
print(f"{'eps':>6} | {'E[E_exact]':>11} | {'E[E_approx]':>12} | {'1/(1-eps)':>10}")
for eps, em, am, b in zip(eps_grid, exact_means, approx_means, approx_bounds):
    print(f"{eps:6.3f} | {em:11.4f} | {am:12.4f} | {b:10.4f}")

print("\nSaved fig_clt_convergence.pdf and fig_exact_vs_approximate.pdf")
