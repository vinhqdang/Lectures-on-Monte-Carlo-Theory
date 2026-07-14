#!/usr/bin/env python3
"""
gen_figures.py -- Figures for Chapter 12: Infimal Convolution
(Bauschke & Combettes, Convex Analysis and Monotone Operator Theory in
Hilbert Spaces, 2nd ed., CMS Books in Mathematics, Springer, 2017)

Reproduces, from scratch, every figure used in chapter12_slides.tex:

  fig_moreau_envelope.pdf   -- f(x) = |x| and its Moreau envelopes ^gamma f
                               for several gamma (the "Huber" smoothing).
  fig_prox_soft_threshold.pdf -- the proximity operator Prox_{gamma|.|}(x),
                               i.e. the soft-thresholding operator, for
                               several gamma.
  fig_moreau_prox_combined.pdf -- both plots side by side (used as a
                               single wide slide figure).

Run:  python3 gen_figures.py
Requires: numpy, matplotlib (Agg backend, no display needed).
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    'font.size': 11,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.spines.top': False,
    'axes.spines.right': False,
})


def soft_threshold(x, gamma):
    """Prox_{gamma |.|}(x) = sign(x) * max(|x| - gamma, 0)."""
    return np.sign(x) * np.maximum(np.abs(x) - gamma, 0.0)


def moreau_envelope_abs(x, gamma):
    """Moreau envelope of f = |.| with parameter gamma (Huber function):
       ^gamma f(x) = x^2/(2 gamma)   if |x| <= gamma
                   = |x| - gamma/2   if |x| >  gamma
    """
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    small = np.abs(x) <= gamma
    out[small] = x[small] ** 2 / (2.0 * gamma)
    out[~small] = np.abs(x[~small]) - gamma / 2.0
    return out


# -----------------------------------------------------------------
# Figure 1: f = |.| and its Moreau envelopes for several gamma
# -----------------------------------------------------------------
x = np.linspace(-4, 4, 2000)
gammas = [0.5, 1.0, 2.0]
colors = ['#1b9e77', '#d95f02', '#7570b3']

fig, ax = plt.subplots(figsize=(6.0, 4.5))
ax.plot(x, np.abs(x), 'k--', linewidth=2.0, label=r'$f(x)=|x|$')
for gamma, c in zip(gammas, colors):
    ax.plot(x, moreau_envelope_abs(x, gamma), color=c, linewidth=2.0,
            label=rf'${{}}^{{{gamma:g}}}f(x)$, $\gamma={gamma:g}$')
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'value')
ax.set_title('Moreau envelopes of $f=|\\cdot|$')
ax.legend(loc='upper center', fontsize=9)
ax.set_xlim(-4, 4)
ax.set_ylim(-0.2, 4.2)
fig.tight_layout()
fig.savefig('fig_moreau_envelope.pdf')
plt.close(fig)

# -----------------------------------------------------------------
# Figure 2: proximity operator (soft-thresholding) for several gamma
# -----------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.0, 4.5))
ax.plot(x, x, 'k:', linewidth=1.2, label=r'identity ($\gamma=0$)')
for gamma, c in zip(gammas, colors):
    ax.plot(x, soft_threshold(x, gamma), color=c, linewidth=2.0,
            label=rf'$\mathrm{{Prox}}_{{\gamma f}}(x)$, $\gamma={gamma:g}$')
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$\mathrm{Prox}_{\gamma f}(x)$')
ax.set_title('Proximity operator of $f=|\\cdot|$ (soft-thresholding)')
ax.legend(loc='upper left', fontsize=9)
ax.axhline(0, color='gray', linewidth=0.6)
ax.axvline(0, color='gray', linewidth=0.6)
ax.set_xlim(-4, 4)
ax.set_ylim(-4, 4)
ax.set_aspect('equal', adjustable='box')
fig.tight_layout()
fig.savefig('fig_prox_soft_threshold.pdf')
plt.close(fig)

# -----------------------------------------------------------------
# Figure 3: combined side-by-side (handy for a single-figure slide)
# -----------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3))

ax = axes[0]
ax.plot(x, np.abs(x), 'k--', linewidth=2.0, label=r'$f(x)=|x|$')
for gamma, c in zip(gammas, colors):
    ax.plot(x, moreau_envelope_abs(x, gamma), color=c, linewidth=2.0,
            label=rf'$\gamma={gamma:g}$')
ax.set_xlabel(r'$x$')
ax.set_title(r'Moreau envelope ${}^{\gamma}\!f$')
ax.legend(fontsize=8, loc='upper center')
ax.set_xlim(-4, 4)
ax.set_ylim(-0.2, 4.2)

ax = axes[1]
ax.plot(x, x, 'k:', linewidth=1.2, label=r'identity')
for gamma, c in zip(gammas, colors):
    ax.plot(x, soft_threshold(x, gamma), color=c, linewidth=2.0,
            label=rf'$\gamma={gamma:g}$')
ax.axhline(0, color='gray', linewidth=0.6)
ax.axvline(0, color='gray', linewidth=0.6)
ax.set_xlabel(r'$x$')
ax.set_title(r'$\mathrm{Prox}_{\gamma f}$ (soft-thresholding)')
ax.legend(fontsize=8, loc='upper left')
ax.set_xlim(-4, 4)
ax.set_ylim(-4, 4)
ax.set_aspect('equal', adjustable='box')

fig.tight_layout()
fig.savefig('fig_moreau_prox_combined.pdf')
plt.close(fig)

print("Wrote fig_moreau_envelope.pdf, fig_prox_soft_threshold.pdf, "
      "fig_moreau_prox_combined.pdf")
