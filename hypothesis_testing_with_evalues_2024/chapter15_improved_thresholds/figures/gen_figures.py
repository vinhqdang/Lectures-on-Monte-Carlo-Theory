#!/usr/bin/env python3
"""
gen_figures.py
Generates all figures for Chapter 15 (Improved e-value thresholds under
additional conditions) slides, based on:
  Ramdas & Wang, "Hypothesis Testing with E-Values" (arXiv:2410.23614), Ch. 15.

Figures produced (all saved as vector PDF in this directory):
  fig_comonotonic.pdf   -- conceptual picture of comonotonic random variables
                           (Def. 15.3): a family of e-variables E_mu, mu>0,
                           that are all increasing functions of a common Z,
                           and their supremum (Section 15.2 / Example 15.6).
  fig_Rgamma.pdf        -- R_gamma(E) vs gamma in (0,1] comparing the plain
                           Markov bound (E_0 = all e-variables) against the
                           improved bounds for decreasing density (E_D),
                           unimodal density (E_U), and log-normal (E_LN)
                           (Theorem 15.7, Theorem 15.10).
  fig_Talpha.pdf        -- the resulting thresholds T_alpha(E) vs alpha in
                           (0, 0.2]: naive 1/alpha vs the improved thresholds
                           for E_D, E_U, E_LN. This is the headline
                           "improved threshold is smaller/less conservative"
                           figure.
  fig_power_bars.pdf    -- bar chart: statistical power gained in the running
                           numerical example (coin/Gaussian mean-shift
                           e-variable, alpha=0.05, mu=0.2, n=100) when the
                           naive threshold 1/alpha=20 is replaced by the
                           improved thresholds T_alpha(E_D)=10 and
                           T_alpha(E_LN)=3.87.

Run: python3 gen_figures.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import brentq

plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 10.5,
    "figure.dpi": 150,
})

COLORS = {
    "E0":  "#444444",
    "ED":  "#3b6ea5",
    "EU":  "#c1666b",
    "ELN": "#5a8f5a",
}

# ---------------------------------------------------------------------------
# Helper functions implementing the exact formulas of Chapter 15.
# ---------------------------------------------------------------------------

def R_gamma_E0(gamma):
    """Markov bound: R_gamma(full set of e-variables) = gamma (Prop 2.1)."""
    return gamma


def R_gamma_ED(gamma):
    """Theorem 15.7(i): decreasing density on its support.
    R_gamma(E_D) = gamma/2 for gamma != 1, and R_1(E_D) = 1."""
    gamma = np.asarray(gamma, dtype=float)
    out = gamma / 2.0
    out = np.where(gamma >= 1.0, 1.0, out)
    return out


def R_gamma_EU(gamma):
    """Theorem 15.7(iii): unimodal density on [0, infty).
    R_gamma(E_U) = (gamma/2) v (2 gamma - 1)."""
    gamma = np.asarray(gamma, dtype=float)
    return np.maximum(gamma / 2.0, 2.0 * gamma - 1.0)


def R_gamma_ELN(gamma):
    """Theorem 15.10(v): E has a log-normal distribution.
    R_gamma(E_LN) = Phi(-sqrt(-2 log gamma)) for gamma != 1, R_1 = 1."""
    gamma = np.asarray(gamma, dtype=float)
    out = np.ones_like(gamma)
    mask = gamma < 1.0
    g = gamma[mask]
    out[mask] = norm.cdf(-np.sqrt(-2.0 * np.log(g)))
    return out


def T_alpha_from_R(R_func, alpha, lo=1e-8, hi=0.999999999):
    """Invert gamma -> R_gamma(E) = alpha to get gamma*, then T = 1/gamma*."""
    f = lambda g: R_func(np.array([g]))[0] - alpha
    gamma_star = brentq(f, lo, hi)
    return 1.0 / gamma_star


# ---------------------------------------------------------------------------
# Figure 1: comonotonicity, illustrated concretely with the likelihood-ratio
# family E_mu = exp(mu S_n - n mu^2/2), mu > 0 (Eq. 15.1 / Example 15.6),
# which are all increasing functions of the common statistic Z = S_n.
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))

ax = axes[0]
z = np.linspace(-3, 6, 400)
n = 10
mus = [0.15, 0.3, 0.5]
colors = ["#3b6ea5", "#c1666b", "#5a8f5a"]
for mu, c in zip(mus, colors):
    e_mu = np.exp(mu * z - n * mu**2 / 2)
    ax.plot(z, e_mu, color=c, lw=2.2, label=rf"$E_\mu(Z)$, $\mu={mu}$")
env = np.exp(np.maximum(z, 0) ** 2 / (2 * n))
ax.plot(z, env, "k--", lw=2.0, label=r"$\sup_{\mu>0} E_\mu(Z)$")
ax.set_xlabel(r"common driver $Z = S_n$")
ax.set_ylabel(r"e-variable value")
ax.set_ylim(0, 8)
ax.set_title("Comonotonic e-variables: all increasing in $Z$")
ax.legend(loc="upper left", frameon=True)
ax.grid(alpha=0.25)

ax2 = axes[1]
rng = np.random.default_rng(15)
z_pts = rng.normal(0, 1, 60)
z_pts.sort()
celsius = z_pts * 8 + 15
fahrenheit = celsius * 9 / 5 + 32
ax2.plot(celsius, fahrenheit, "o-", color="#3b6ea5", ms=4, lw=1.5)
ax2.set_xlabel(r"Thermometer reading in $^\circ$C")
ax2.set_ylabel(r"Same weather, in $^\circ$F")
ax2.set_title("Analogy: comonotonic = always move together")
ax2.grid(alpha=0.25)

fig.tight_layout()
fig.savefig("fig_comonotonic.pdf")
plt.close(fig)

# ---------------------------------------------------------------------------
# Figure 2: R_gamma(E) vs gamma for the four classes (Theorem 15.7, 15.10).
# ---------------------------------------------------------------------------
gamma_grid = np.linspace(1e-6, 1.0, 800)

fig, ax = plt.subplots(figsize=(7.0, 5.2))
ax.plot(gamma_grid, R_gamma_E0(gamma_grid), color=COLORS["E0"], lw=2.2,
        label=r"$R_\gamma(\mathfrak{E})$ -- Markov / no assumption")
ax.plot(gamma_grid, R_gamma_ED(gamma_grid), color=COLORS["ED"], lw=2.4,
        label=r"$R_\gamma(\mathcal{E}_{\mathrm{D}})$ -- decreasing density")
ax.plot(gamma_grid, R_gamma_EU(gamma_grid), color=COLORS["EU"], lw=2.2,
        ls="--", label=r"$R_\gamma(\mathcal{E}_{\mathrm{U}})$ -- unimodal density")
ax.plot(gamma_grid, R_gamma_ELN(gamma_grid), color=COLORS["ELN"], lw=2.2,
        ls=":", label=r"$R_\gamma(\mathcal{E}_{\mathrm{LN}})$ -- log-normal")
ax.set_xlabel(r"$\gamma$")
ax.set_ylabel(r"$R_\gamma(\mathcal{E}) = \sup_{E \in \mathcal{E}} \mathbb{P}(E \geq 1/\gamma)$")
ax.set_title("Worst-case type-I error under extra conditions")
ax.legend(loc="upper left", frameon=True)
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig("fig_Rgamma.pdf")
plt.close(fig)

# ---------------------------------------------------------------------------
# Figure 3: T_alpha(E) vs alpha -- the headline comparison of thresholds.
# ---------------------------------------------------------------------------
alphas = np.linspace(0.002, 0.2, 200)

T_E0 = 1.0 / alphas
T_ED = 1.0 / (2.0 * alphas)
T_EU = np.array([T_alpha_from_R(R_gamma_EU, a) for a in alphas])
T_ELN = np.array([T_alpha_from_R(R_gamma_ELN, a) for a in alphas])

fig, ax = plt.subplots(figsize=(7.0, 5.2))
ax.plot(alphas, T_E0, color=COLORS["E0"], lw=2.2,
        label=r"naive Markov threshold $1/\alpha$")
ax.plot(alphas, T_ED, color=COLORS["ED"], lw=2.4,
        label=r"$T_\alpha(\mathcal{E}_{\mathrm{D}}) = 1/(2\alpha)$ (decreasing density)")
ax.plot(alphas, T_EU, color=COLORS["EU"], lw=2.2, ls="--",
        label=r"$T_\alpha(\mathcal{E}_{\mathrm{U}})$ (unimodal density)")
ax.plot(alphas, T_ELN, color=COLORS["ELN"], lw=2.2, ls=":",
        label=r"$T_\alpha(\mathcal{E}_{\mathrm{LN}})$ (log-normal)")
ax.axvline(0.05, color="gray", lw=1.0, ls="-.")
ax.set_xlabel(r"$\alpha$")
ax.set_ylabel(r"threshold $T_\alpha(\mathcal{E})$ (reject when $E \geq T_\alpha$)")
ax.set_title("Improved (smaller) rejection thresholds vs. the naive $1/\\alpha$")
ax.set_ylim(0, 100)
ax.legend(loc="upper right", frameon=True)
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig("fig_Talpha.pdf")
plt.close(fig)

# Print the key numbers used in the running example (alpha = 0.05).
alpha0 = 0.05
T0 = 1 / alpha0
TD = 1 / (2 * alpha0)
TLN = T_alpha_from_R(R_gamma_ELN, alpha0)
print(f"alpha = {alpha0}")
print(f"  naive threshold        1/alpha        = {T0:.3f}")
print(f"  decreasing-density T_a(E_D) = 1/(2a)   = {TD:.3f}")
print(f"  log-normal T_a(E_LN)                   = {TLN:.4f}")

# ---------------------------------------------------------------------------
# Figure 4: power gained in the running numerical example.
#
# Running example: coin-fairness-style Gaussian mean-shift e-variable
#   E_mu(S_n) = exp(mu * S_n - n * mu^2 / 2),   S_n ~ N(0, n) under H0.
# For a FIXED candidate mu, log E_mu ~ N(-n mu^2/2, n mu^2) under H0, i.e.
# E_mu is exactly log-normal -> falls in E_LN (Section 15.4) and, since
# log-normal densities are unimodal, also in E_U (Section 15.3).
#
# We reject when E_mu >= T, i.e. when S_n >= S*(T) := log(T)/mu + n*mu/2.
# Under a true alternative mean mu_true = mu, S_n ~ N(n*mu, n), so
#   power(T) = P(S_n >= S*(T)) = 1 - Phi((S*(T) - n*mu) / sqrt(n)).
# ---------------------------------------------------------------------------
mu = 0.2
n = 100
alpha0 = 0.05


def crit_value(T):
    return np.log(T) / mu + n * mu / 2.0


def power(T, mu_true=mu):
    Sstar = crit_value(T)
    return 1.0 - norm.cdf((Sstar - n * mu_true) / np.sqrt(n))


T_naive = 1 / alpha0
T_dens = 1 / (2 * alpha0)
T_logn = T_alpha_from_R(R_gamma_ELN, alpha0)

labels = [r"naive $1/\alpha=20$", r"$\mathcal{E}_{\mathrm{D}}$: $T=10$",
          r"$\mathcal{E}_{\mathrm{LN}}$: $T\approx 3.87$"]
thresholds = [T_naive, T_dens, T_logn]
powers = [power(T) for T in thresholds]
print("Power at mu_true=0.2, n=100, alpha=0.05:",
      [f"{p:.3f}" for p in powers])

fig, ax = plt.subplots(figsize=(6.6, 4.8))
bar_colors = [COLORS["E0"], COLORS["ED"], COLORS["ELN"]]
bars = ax.bar(labels, powers, color=bar_colors, width=0.55)
for b, p in zip(bars, powers):
    ax.text(b.get_x() + b.get_width() / 2, p + 0.015, f"{p*100:.1f}\\%",
            ha="center", va="bottom", fontsize=11)
ax.set_ylabel("power against $\\mu=0.2$ (n=100, $\\alpha=0.05$)")
ax.set_ylim(0, 0.8)
ax.set_title("Same validity guarantee, more power")
ax.grid(alpha=0.25, axis="y")
fig.tight_layout()
fig.savefig("fig_power_bars.pdf")
plt.close(fig)

print("All figures written to the current directory.")
