#!/usr/bin/env python3
"""
gen_figures.py -- Figures for Chapter 2: Validity -- E-values under the Null
"Hypothesis Testing with E-Values" by Ramdas & Wang (arXiv:2410.23614)

Generates (as vector PDFs, into this figures/ directory):
  fig1_calibrators.pdf         -- p-to-e and e-to-p calibrator functions
  fig2_markov.pdf              -- Markov's inequality for e-variables, simulated
  fig3_coin_binomial.pdf       -- running example: Binomial(20,0.5) null pmf,
                                   observed x=14, p-value / e-value illustration
  fig4_evalue_pvalue_normal.pdf-- e-values (several deltas) vs p-value, two-sided
                                   normal test, log10 scale (analogous to the
                                   book's Figure 2.2, redrawn independently)

Run with: python3 gen_figures.py
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

plt.rcParams.update({
    'font.size': 11,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'figure.autolayout': True,
})

rng = np.random.default_rng(20240712)

# ─────────────────────────────────────────────────────────────────────────
# Figure 1: Calibrator functions (p-to-e family + the unique e-to-p one)
# ─────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))

p = np.linspace(1e-3, 1, 800)

ax = axes[0]
for kappa, style in [(0.1, '-'), (0.3, '-'), (0.5, '-'), (0.7, '-'), (0.9, '-')]:
    f = kappa * p ** (kappa - 1)
    ax.plot(p, f, style, label=fr'$\kappa={kappa}$ (power, Eq. 2.1)', alpha=0.85)

f_23 = 2 * (1 - p)
f_24 = p ** (-0.5) - 1
f_25 = -np.log(p)
ax.plot(p, f_23, 'k--', lw=2, label=r'$f(p)=2(1-p)$ (Eq. 2.3)')
ax.plot(p, f_24, 'r--', lw=2, label=r'$f(p)=p^{-1/2}-1$ (Eq. 2.4)')
ax.plot(p, f_25, 'b--', lw=2, label=r'$f(p)=-\log p$ (Eq. 2.5)')

# mark our running example: p = 0.0577 -> e = 3.16
p0 = 0.057659149169921875
e0 = p0 ** (-0.5) - 1
ax.plot([p0], [e0], 'r*', markersize=14, zorder=5)
ax.annotate(f'coin example\n$p={p0:.3f}\\to e={e0:.2f}$',
            xy=(p0, e0), xytext=(0.35, 8),
            arrowprops=dict(arrowstyle='->', color='gray'), fontsize=9)

ax.set_xlabel(r'$p$-value $p$')
ax.set_ylabel(r'calibrated e-value $f(p)$')
ax.set_title('p-to-e calibrators (Definition 2.3(i))')
ax.set_ylim(0, 12)
ax.legend(fontsize=7, loc='upper right')

ax = axes[1]
t = np.linspace(0.05, 6, 800)
f_e2p = np.minimum(1, 1 / t)
ax.plot(t, f_e2p, 'g-', lw=2.5, label=r'$f(t)=\min(1,1/t)$')
ax.axhline(1, color='gray', lw=0.7)
ax.axvline(1, color='gray', lw=0.7, ls=':')

# mark our running example: e = 5.184 -> p = 0.1929
e1 = 5.184417904532337
p1 = min(1, 1 / e1)
ax.plot([e1], [p1], 'r*', markersize=14, zorder=5)
ax.annotate(f'coin example\n$e={e1:.2f}\\to p={p1:.3f}$',
            xy=(e1, p1), xytext=(2.6, 0.55),
            arrowprops=dict(arrowstyle='->', color='gray'), fontsize=9)

ax.set_xlabel(r'e-value $t$')
ax.set_ylabel(r'calibrated p-value $f(t)$')
ax.set_title('The unique admissible e-to-p calibrator\n(Proposition 2.4, from Markov\'s inequality)')
ax.legend(fontsize=9, loc='upper right')
ax.set_ylim(-0.02, 1.05)

fig.savefig('fig1_calibrators.pdf')
plt.close(fig)

# ─────────────────────────────────────────────────────────────────────────
# Figure 2: Markov's inequality for e-variables, simulated
# ─────────────────────────────────────────────────────────────────────────
# Simulate a "typical" e-variable E >= 0 with E[E] = 1 under the null:
# E = exp(lambda*Z - lambda^2/2), Z ~ N(0,1), lambda = 1  (an exact e-variable)
n_sim = 400_000
lam = 1.0
Z = rng.standard_normal(n_sim)
E_sim = np.exp(lam * Z - lam ** 2 / 2)

# Also the two-point "extremal" e-variable for which Markov holds with
# equality: E take value 0 w.p. 1-1/c and 1/c... but to show equality
# at a specific alpha we use E in {0, 1/alpha0} with mean 1.
alpha0 = 0.1
E_extreme = rng.choice([0.0, 1 / alpha0], size=n_sim, p=[1 - alpha0, alpha0])

alphas = np.linspace(0.005, 0.999, 300)
thresholds = 1 / alphas

tail_prob_sim = np.array([(E_sim >= th).mean() for th in thresholds])
tail_prob_extreme = np.array([(E_extreme >= th).mean() for th in thresholds])

fig, ax = plt.subplots(figsize=(7.2, 5))
ax.plot(alphas, alphas, 'k-', lw=2, label=r'Markov bound: $\mathbb{P}(E\geq 1/\alpha)\leq \alpha$')
ax.plot(alphas, tail_prob_sim, 'C0-', lw=2,
        label=r'simulated: $E=\exp(Z-1/2)$, $Z\sim N(0,1)$')
ax.plot(alphas, tail_prob_extreme, 'C3--', lw=2,
        label=r'extremal $E\in\{0,1/\alpha_0\}$, $\alpha_0=0.1$ (equality only at $\alpha=\alpha_0$)')
ax.plot([alpha0], [alpha0], 'C3o', markersize=8, zorder=5)
ax.set_xlabel(r'$\alpha$')
ax.set_ylabel(r'$\mathbb{P}(E \geq 1/\alpha)$')
ax.set_title("Markov's inequality for e-variables (Proposition 2.1)")
ax.legend(fontsize=9, loc='upper left')
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
fig.savefig('fig2_markov.pdf')
plt.close(fig)

# ─────────────────────────────────────────────────────────────────────────
# Figure 3: Running coin example -- Binomial(20, 0.5) null pmf
# ─────────────────────────────────────────────────────────────────────────
n, theta0, x_obs = 20, 0.5, 14
k = np.arange(0, n + 1)
pmf = stats.binom.pmf(k, n, theta0)
p_obs = stats.binom.pmf(x_obs, n, theta0)
one_sided_mask = k >= x_obs
two_sided_mask = pmf <= p_obs + 1e-15

fig, ax = plt.subplots(figsize=(8, 4.6))
colors = ['C3' if two_sided_mask[i] else 'C0' for i in range(len(k))]
ax.bar(k, pmf, color=colors, edgecolor='black', linewidth=0.4)
ax.axvline(x_obs, color='black', ls='--', lw=1.5)
ax.text(x_obs + 0.3, max(pmf) * 0.9, f'observed $x=14$', fontsize=10)
ax.set_xlabel('number of heads $k$ out of $n=20$ flips')
ax.set_ylabel(r'$\mathbb{P}_{\theta_0=0.5}(X=k)$')
ax.set_title(r'Running example: $H_0:\theta=0.5$, $n=20$, observed 14 heads'
             '\n(red bars: outcomes at least as extreme, two-sided p-value shown)')
p_two = pmf[two_sided_mask].sum()
p_one = pmf[one_sided_mask].sum()
ax.annotate(f'two-sided $p$-value $\\approx {p_two:.3f}$\none-sided $p$-value $\\approx {p_one:.3f}$',
            xy=(0.98, 0.95), xycoords='axes fraction', ha='right', va='top',
            fontsize=10, bbox=dict(boxstyle='round', fc='white', ec='gray'))
fig.savefig('fig3_coin_binomial.pdf')
plt.close(fig)

# ─────────────────────────────────────────────────────────────────────────
# Figure 4: e-values (several deltas) vs p-value, two-sided normal test
#           (independent redraw, analogous in spirit to book Fig. 2.2)
# ─────────────────────────────────────────────────────────────────────────
z = np.linspace(-6, 6, 1200)
p_val = 2 * (1 - stats.norm.cdf(np.abs(z)))  # two-sided p-value as fn of Z=z
p_val = np.clip(p_val, 1e-300, 1)

fig, ax = plt.subplots(figsize=(7.6, 5.4))
ax.plot(z, np.log10(p_val), 'k-', lw=2.5, label='p-value (two-sided z-test)')

for delta, color in [(1, 'C0'), (3, 'C1'), (5, 'C2')]:
    e_val = np.exp(delta * z - delta ** 2 / 2)
    ax.plot(z, np.log10(e_val), '--', color=color, lw=1.8,
            label=fr'e-value, $\delta={delta}$')

for beta in [2, 1.5, 1, 0.5]:
    ax.axhline(beta, color='orange', ls=':', lw=0.8)
for a in [0.05, 0.01]:
    ax.axhline(np.log10(a), color='gray', ls=':', lw=0.8)

ax.set_xlabel(r'test statistic $Z = z$')
ax.set_ylabel(r'$\log_{10}$ of e-values and p-values')
ax.set_title('E-values vs. p-values, two-sided normal test\n(own redraw, cf. book Fig. 2.2)')
ax.legend(fontsize=9, loc='upper center', ncol=2)
ax.set_ylim(-6, 6)
fig.savefig('fig4_evalue_pvalue_normal.pdf')
plt.close(fig)

print("All figures written to figures/ as PDF.")
