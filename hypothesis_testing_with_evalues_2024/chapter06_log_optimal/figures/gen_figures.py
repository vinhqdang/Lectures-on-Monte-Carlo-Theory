"""
gen_figures.py
Generates figures for Chapter 6: The log-optimal e-value and reverse
information projection (Ramdas & Wang, "Hypothesis Testing with E-Values").

Figure 1 (fig_epower_domination.pdf):
    Coin-fairness running example. Null: theta <= 0.5 (fair-or-tails-biased
    coin), point alternative Q = Bernoulli(theta1). Compares the e-power
    (expected log-growth, in nats) of the numeraire / log-optimal e-value,
    which uses the full sample of n tosses, against a sample-splitting
    "universal inference" style e-value that spends half the sample on
    estimating the null boundary and only accumulates evidence on the
    other half. Both curves are exactly n*KL(theta1||0.5) and
    (n/2)*KL(theta1||0.5) respectively, so the numeraire dominates by
    construction -- illustrating Section 6.5 (numeraire dominates
    universal inference) concretely.

Figure 2 (fig_ripr_projection.pdf):
    Schematic visualization of the reverse information projection (RIPr).
    A convex region represents (the bipolar of) the null P; a point Q
    lies outside it; the RIPr P* is the "closest" point in a
    KL-divergence sense, analogous to Figure 6.12 in the book.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

plt.rcParams['font.size'] = 11
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3


# ----------------------------------------------------------------------
# Figure 1: numeraire e-power vs split-sample universal-inference e-power
# ----------------------------------------------------------------------

def kl_bernoulli(theta1, theta0):
    """KL(Bernoulli(theta1) || Bernoulli(theta0)) in nats."""
    theta1 = np.asarray(theta1, dtype=float)
    eps = 1e-12
    theta1c = np.clip(theta1, eps, 1 - eps)
    return (theta1c * np.log(theta1c / theta0)
            + (1 - theta1c) * np.log((1 - theta1c) / (1 - theta0)))


theta0 = 0.5
n = 20
theta1_grid = np.linspace(0.51, 0.95, 300)

kl_vals = kl_bernoulli(theta1_grid, theta0)
epower_numeraire = n * kl_vals            # uses all n tosses
epower_ui_split = (n / 2) * kl_vals       # only the test half accrues evidence

fig, ax = plt.subplots(figsize=(7.2, 4.6))
ax.plot(theta1_grid, epower_numeraire, color='#1b5e8a', lw=2.4,
        label=r'Numeraire $E^*$: e-power $= n \cdot \mathrm{KL}(\theta_1 \| 0.5)$')
ax.plot(theta1_grid, epower_ui_split, color='#c0392b', lw=2.4, ls='--',
        label=r'Split universal inference: e-power $= \frac{n}{2}\cdot \mathrm{KL}(\theta_1 \| 0.5)$')
ax.fill_between(theta1_grid, epower_ui_split, epower_numeraire,
                color='#1b5e8a', alpha=0.08)

# mark the running numeric example theta1 = 0.7, n = 20
theta1_star = 0.7
kl_star = kl_bernoulli(theta1_star, theta0)
ax.scatter([theta1_star], [n * kl_star], color='#1b5e8a', zorder=5, s=45)
ax.scatter([theta1_star], [(n / 2) * kl_star], color='#c0392b', zorder=5, s=45)
ax.annotate(f'$n\\cdot$KL $= {n*kl_star:.2f}$\n$(E^* \\approx {np.exp(n*kl_star):.2f})$',
            xy=(theta1_star, n * kl_star), xytext=(theta1_star + 0.03, n * kl_star + 0.15),
            fontsize=9.5, color='#1b5e8a')
ax.annotate(f'$\\frac{{n}}{{2}}\\cdot$KL $= {(n/2)*kl_star:.2f}$\n$(E_{{UI}} \\approx {np.exp((n/2)*kl_star):.2f})$',
            xy=(theta1_star, (n / 2) * kl_star), xytext=(theta1_star + 0.03, (n / 2) * kl_star - 0.55),
            fontsize=9.5, color='#c0392b')
ax.axvline(theta1_star, color='gray', lw=0.8, ls=':')

ax.set_xlabel(r'Alternative coin bias $\theta_1$ (data-generating $\mathbb{Q} = \mathrm{Bernoulli}(\theta_1)$)')
ax.set_ylabel('E-power (nats), $n = 20$ tosses')
ax.set_title('Numeraire dominates universal inference: coin-fairness test\n'
             r'$H_0: \theta \leq 0.5$ vs. point alternative $\theta_1$', fontsize=11.5)
ax.legend(loc='upper left', fontsize=9.5, framealpha=0.9)
ax.set_xlim(0.5, 0.95)
ax.set_ylim(bottom=0)
fig.tight_layout()
fig.savefig('fig_epower_domination.pdf')
plt.close(fig)


# ----------------------------------------------------------------------
# Figure 2: schematic of the reverse information projection
# ----------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(7.2, 5.0))
ax.set_xlim(-3.4, 4.6)
ax.set_ylim(-2.6, 3.0)
ax.axis('off')

# Convex region representing (the bipolar of) the null P
theta = np.linspace(0, 2 * np.pi, 400)
cx, cy = 1.6, 0.2
rx, ry = 1.9, 1.9
wobble = 1 + 0.10 * np.sin(4 * theta) + 0.05 * np.cos(3 * theta + 0.7)
region_x = cx + rx * wobble * np.cos(theta)
region_y = cy + ry * wobble * np.sin(theta)
ax.fill(region_x, region_y, color='#9fb8c8', alpha=0.35, zorder=1)
ax.plot(region_x, region_y, color='#1b5e8a', lw=1.8, zorder=2)
ax.text(cx, cy + 1.55, r'$\mathcal{P}^{\circ\circ}$ (effective null, $\supseteq \mathrm{Conv}(\mathcal{P})$)',
        ha='center', fontsize=10, color='#1b5e8a')
ax.text(cx, cy - 1.7, r'$\mathrm{Conv}(\mathcal{P})$ lies inside', ha='center',
        fontsize=8.5, color='#1b5e8a', style='italic')

# point Q outside the region
Qx, Qy = -2.3, 0.7
ax.scatter([Qx], [Qy], color='#c0392b', s=70, zorder=5)
ax.text(Qx - 0.15, Qy + 0.35, r'$\mathbb{Q}$', fontsize=15, color='#c0392b', ha='center')

# projection point P* on boundary (closest point in KL sense, schematic)
# pick the boundary point of the region closest to Q along the line direction
dirvec = np.array([cx - Qx, cy - Qy])
dirvec = dirvec / np.linalg.norm(dirvec)
# find approx boundary intersection point nearest Q
pts = np.stack([region_x, region_y], axis=1)
dists_along = (pts - np.array([Qx, Qy])) @ dirvec
proj_perp = np.linalg.norm((pts - np.array([Qx, Qy])) - np.outer(dists_along, dirvec), axis=1)
candidate = np.argmin(proj_perp + 0.001 * np.abs(dists_along - np.min(dists_along[dists_along > 0])))
mask = dists_along > 0
idx = np.argmin(np.where(mask, proj_perp, np.inf))
Px, Py = region_x[idx], region_y[idx]

ax.plot([Qx, Px], [Qy, Py], color='#333333', lw=1.6, ls='-', zorder=4)
ax.scatter([Px], [Py], color='#e08e00', s=70, zorder=6, edgecolor='black', linewidth=0.6)
ax.text(Px + 0.05, Py - 0.45, r'$\mathbb{P}^*$ (RIPr)', fontsize=12, color='#e08e00', ha='left')

mid = np.array([(Qx + Px) / 2, (Qy + Py) / 2])
ax.text(mid[0] - 0.15, mid[1] + 0.35, r'$\mathrm{KL}(\mathbb{Q},\mathbb{P}^*)$'
        '\n' r'$= \inf_{\mathbb{P}\in\mathcal{P}^{\circ\circ}} \mathrm{KL}(\mathbb{Q},\mathbb{P})$',
        fontsize=9.5, color='#333333', ha='center')

# arrow annotation
ax.annotate('', xy=(Px, Py), xytext=(Qx, Qy),
            arrowprops=dict(arrowstyle='->', color='#333333', lw=1.4))

# numeraire label
ax.text(1.0, -2.35,
        r'Numeraire $E^* = \frac{d\mathbb{Q}}{d\mathbb{P}^*}$   and   '
        r'e-power $= \mathbb{E}^{\mathbb{Q}}[\log E^*] = \mathrm{KL}(\mathbb{Q},\mathbb{P}^*)$',
        ha='center', fontsize=10.5,
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#fdf6e3', edgecolor='#e08e00'))

ax.set_title('Reverse information projection: $\\mathbb{Q}$ projected onto the effective null $\\mathcal{P}^{\\circ\\circ}$',
             fontsize=12)

fig.tight_layout()
fig.savefig('fig_ripr_projection.pdf')
plt.close(fig)

print("Wrote fig_epower_domination.pdf and fig_ripr_projection.pdf")
