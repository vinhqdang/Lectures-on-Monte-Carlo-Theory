#!/usr/bin/env python3
"""
gen_figures.py
Generates figures for Chapter 1: Background
(Bauschke & Combettes, "Convex Analysis and Monotone Operator Theory in
Hilbert Spaces", 2nd ed., CMS Books in Mathematics, Springer, 2017).

Figures produced (saved as vector PDF in this directory):
  fig_net_convergence.pdf   -- the net (x_a) from Exercise 1.4: A = Z directed
                                by <=, x_a = a for a <= 0 and x_a = 1/a for
                                a > 0.  Shows that the net converges to 0
                                (tail behaviour as a -> +infinity) even though
                                the net, taken as a whole, is unbounded below.
  fig_lsc_epigraph.pdf      -- two indicator-type functions on R illustrating
                                Example 1.25 / Lemma 1.24: the epigraph of a
                                lower semicontinuous function is closed, while
                                the epigraph of a non-lower-semicontinuous
                                function is not closed (a point is "missing").
  fig_order_zorn.pdf        -- a Hasse-type diagram of a partially ordered set
                                showing a chain, an upper bound for the chain,
                                and a maximal element, illustrating Section 1.3
                                (Order) and Zorn's Lemma (Fact 1.1).
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ---------------------------------------------------------------------------
# Figure 1: a net that is unbounded yet convergent (Exercise 1.4)
# ---------------------------------------------------------------------------
a_neg = np.arange(-12, 1)                 # a = -12, ..., -1, 0
a_pos = np.arange(1, 13)                  # a = 1, ..., 12

x_neg = a_neg.astype(float)               # x_a = a for a <= 0
x_pos = 1.0 / a_pos                       # x_a = 1/a for a > 0

fig, ax = plt.subplots(figsize=(7.2, 4.6))

ax.axhline(0, color='gray', linewidth=0.7, zorder=0)
ax.axvline(0, color='gray', linewidth=0.7, zorder=0)

ax.plot(a_neg, x_neg, 'o-', color='crimson', markersize=5, linewidth=1.3,
        label=r'$a \leq 0$: $\;x_a = a$ (diverges to $-\infty$)')
ax.plot(a_pos, x_pos, 'o-', color='steelblue', markersize=5, linewidth=1.3,
        label=r'$a > 0$: $\;x_a = 1/a \to 0$')

ax.annotate('the net converges here:\n"eventually in every neighborhood of 0"'
            '\nonly the tail ' r'$a\to+\infty$' ' matters',
            xy=(9, 1.0 / 9), xytext=(2.0, 2.6),
            arrowprops=dict(arrowstyle='-|>', color='black', lw=1.1),
            fontsize=9, ha='left')

ax.annotate('unbounded left tail\n(does not affect convergence)',
            xy=(-9, -9), xytext=(-11.5, 2.0),
            arrowprops=dict(arrowstyle='-|>', color='black', lw=1.1),
            fontsize=9, ha='left')

ax.set_xlabel(r'index $a \in A = \mathbb{Z}$ (directed by $\leqslant$)')
ax.set_ylabel(r'$x_a$')
ax.set_title(r'A net $(x_a)_{a\in A}$ that is unbounded but converges to $0$'
             '\n(Exercise 1.4)')
ax.set_ylim(-12.5, 4.5)
ax.legend(loc='lower right', fontsize=8.5, framealpha=0.9)
fig.tight_layout()
fig.savefig('fig_net_convergence.pdf')
plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2: closed epigraph (lower semicontinuous) vs. non-closed epigraph
# (not lower semicontinuous), instantiating Example 1.25 / Lemma 1.24 with
# the indicator function iota_C for C = {0} (closed) and C = ]0,1[ (not
# closed).
# ---------------------------------------------------------------------------
fig2, axes = plt.subplots(1, 2, figsize=(10.4, 4.6))

xs = np.linspace(-1.5, 2.0, 800)

# --- Left panel: iota_{{0}} -- lower semicontinuous, epigraph closed -------
ax = axes[0]
M = 3.2  # plotting cutoff standing in for +infinity
y_left = np.where(np.isclose(xs, 0.0, atol=1e-6), 0.0, M)
mask = np.abs(xs) > 0.02
ax.plot(xs[mask], np.full(mask.sum(), M), color='steelblue', linewidth=2.2)
ax.fill_between(xs[mask], np.full(mask.sum(), M), M + 1.0,
                 color='steelblue', alpha=0.15)
ax.plot([0], [0], marker='o', color='steelblue', markersize=9, zorder=5)
ax.vlines(0, 0, M + 1.0, color='steelblue', alpha=0.15, linewidth=8)
ax.set_title(r'$\iota_{\{0\}}$: $C=\{0\}$ is closed' '\n' r'lower semicontinuous, $\mathrm{epi}\,f$ closed')
ax.set_ylim(-0.5, M + 1.0)
ax.set_xlim(-1.5, 2.0)
ax.set_xlabel('$x$')
ax.set_ylabel(r'$\xi$  (with $\mathrm{epi}\,f$ shaded)')
ax.axhline(0, color='gray', linewidth=0.5)
ax.axvline(0, color='gray', linewidth=0.5)
ax.annotate(r'$f(0)=0$ is filled in:''\nno point is missing from $\mathrm{epi}\,f$',
            xy=(0, 0), xytext=(0.25, -0.4), fontsize=8.5)

# --- Right panel: iota_{]0,1[} -- NOT lower semicontinuous, epigraph not closed
ax = axes[1]
inside = (xs > 0) & (xs < 1)
outside = ~inside
ax.plot(xs[outside], np.full(outside.sum(), M), color='crimson', linewidth=2.2)
ax.fill_between(xs[outside], np.full(outside.sum(), M), M + 1.0,
                 color='crimson', alpha=0.15)
ax.hlines(0, 0, 1, color='crimson', linewidth=2.2)
ax.fill_between(xs[inside], 0, M + 1.0, color='crimson', alpha=0.15)
# open circles at x=0 and x=1 at height 0 -- the "missing" boundary points
ax.plot([0, 1], [0, 0], marker='o', markerfacecolor='white',
        markeredgecolor='crimson', markersize=9, linestyle='None', zorder=5)
ax.plot([0, 1], [M, M], marker='o', color='crimson', markersize=7, zorder=5)
ax.set_title(r'$\iota_{]0,1[}$: $C=\,]0,1[\,$ is not closed' '\n'
             r'NOT lower semicontinuous, $\mathrm{epi}\,f$ not closed')
ax.set_ylim(-0.5, M + 1.0)
ax.set_xlim(-1.5, 2.0)
ax.set_xlabel('$x$')
ax.axhline(0, color='gray', linewidth=0.5)
ax.axvline(0, color='gray', linewidth=0.5)
ax.annotate('open circles: the point\n' r'$(0,0)\notin\mathrm{epi}\,f$ is missing'
            '\n(sequence from inside approaches it)',
            xy=(0, 0), xytext=(0.55, -0.42), fontsize=8.5)

fig2.suptitle(r'Epigraph of the indicator function $\iota_C$ is closed $\Leftrightarrow$ $C$ is closed  (Example 1.25)')
fig2.tight_layout(rect=[0, 0, 1, 0.94])
fig2.savefig('fig_lsc_epigraph.pdf')
plt.close(fig2)


# ---------------------------------------------------------------------------
# Figure 3: a Hasse diagram illustrating a partially ordered set, a chain,
# an upper bound, and a maximal element (Section 1.3, Fact 1.1 Zorn's Lemma).
# ---------------------------------------------------------------------------
fig3, ax = plt.subplots(figsize=(7.0, 5.2))

# Nodes: (label, x, y)
nodes = {
    'a': (0.0, 0.0),
    'b': (-1.2, 1.0),
    'c': (1.2, 1.0),
    'd': (-1.2, 2.0),
    'e': (0.0, 2.0),
    'f': (1.2, 2.0),
    'g': (0.0, 3.2),   # maximal element, upper bound of the chain a<b<d<g
}

edges = [('a', 'b'), ('a', 'c'), ('b', 'd'), ('b', 'e'), ('c', 'e'), ('c', 'f'),
         ('d', 'g'), ('e', 'g'), ('f', 'g')]

for (u, v) in edges:
    x1, y1 = nodes[u]
    x2, y2 = nodes[v]
    ax.plot([x1, x2], [y1, y2], color='gray', linewidth=1.3, zorder=1)

chain = ['a', 'b', 'd', 'g']
chain_color = 'crimson'
for (u, v) in zip(chain[:-1], chain[1:]):
    x1, y1 = nodes[u]
    x2, y2 = nodes[v]
    ax.plot([x1, x2], [y1, y2], color=chain_color, linewidth=3.2, zorder=2,
             alpha=0.55)

for label, (x, y) in nodes.items():
    face = 'gold' if label == 'g' else ('mistyrose' if label in chain else 'white')
    ax.plot(x, y, marker='o', markersize=26, markerfacecolor=face,
            markeredgecolor='black', zorder=3)
    ax.annotate(label, xy=(x, y), ha='center', va='center', fontsize=12,
                zorder=4, fontweight='bold')

ax.annotate('chain: $a \\prec b \\prec d \\prec g$\n(totally ordered subset)',
            xy=(-0.6, 1.5), xytext=(-3.4, 1.3), color=chain_color, fontsize=9,
            arrowprops=dict(arrowstyle='-', color=chain_color, lw=0.8))

ax.annotate('$g$ is an upper bound\nof the chain, and also\na maximal element\n(Zorn, Fact 1.1)',
            xy=(0.0, 3.2), xytext=(1.6, 3.6), color='darkgoldenrod', fontsize=9,
            arrowprops=dict(arrowstyle='-|>', color='darkgoldenrod', lw=1.0))

ax.set_xlim(-3.8, 3.4)
ax.set_ylim(-0.7, 4.3)
ax.axis('off')
ax.set_title('A partially ordered set $(A,\\preccurlyeq)$: a chain, an upper\n'
             'bound, and a maximal element (Section 1.3)')

legend_elems = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='white',
           markeredgecolor='black', markersize=12, label='element of $A$'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='mistyrose',
           markeredgecolor='black', markersize=12, label='element of the chain'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='gold',
           markeredgecolor='black', markersize=12, label='maximal element'),
]
ax.legend(handles=legend_elems, loc='lower right', fontsize=8.5, framealpha=0.9)

fig3.tight_layout()
fig3.savefig('fig_order_zorn.pdf')
plt.close(fig3)

print("Generated: fig_net_convergence.pdf, fig_lsc_epigraph.pdf, fig_order_zorn.pdf")
