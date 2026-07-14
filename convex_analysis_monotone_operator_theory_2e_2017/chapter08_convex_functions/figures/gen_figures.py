#!/usr/bin/env python3
"""
gen_figures.py
Generates figures for Chapter 8: Convex Functions
(Bauschke & Combettes, "Convex Analysis and Monotone Operator Theory in
Hilbert Spaces", 2nd ed., CMS Books in Mathematics, Springer, 2017).

Figures produced (saved as vector PDF in this directory):

  fig_chord_convex.pdf
      f(x) = x^2 on [-2,2] with the chord joining (-1.5, f(-1.5)) and
      (1, f(1)).  The chord lies entirely on or above the graph, which is
      exactly the geometric definition of convexity.  A concrete point
      alpha = 0.4 is marked to show the numeric inequality
      f(alpha*x + (1-alpha)*y) <= alpha f(x) + (1-alpha) f(y).

  fig_chord_nonconvex.pdf
      g(x) = x^4 - 3x^2 (a "double well", non-convex) on [-2.2,2.2] with
      the chord joining two points on the two different wells.  Part of
      the chord dips BELOW the graph, which is exactly what convexity
      forbids; the violating point is marked numerically.

  fig_epigraph.pdf
      The epigraph epi f = {(x,eta) : eta >= f(x)} of f(x) = x^2,
      shaded, together with the graph of f.  Illustrates Proposition 8.4:
      f is convex iff epi f is a convex subset of H x R.

  fig_huber.pdf
      The Huber function (Example 8.44) with rho = 1, contrasted with
      x^2/2 and rho|x| - rho^2/2 (its two "ingredients"), showing that it
      is continuous and convex everywhere (a running example used for the
      topological-properties part of the chapter).
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.size': 12,
    'axes.linewidth': 1.0,
    'lines.linewidth': 2.0,
})

# ===========================================================================
# Figure 1: convex function -- chord lies on/above the graph
# ===========================================================================
fig, ax = plt.subplots(figsize=(6, 4.5))

x = np.linspace(-2, 2, 400)
f = x**2
ax.plot(x, f, color='navy', label=r'$f(x) = x^2$')

x0, x1 = -1.5, 1.0
y0, y1 = x0**2, x1**2
ax.plot([x0, x1], [y0, y1], color='crimson', linestyle='--', linewidth=1.8,
        label='chord joining the two points')
ax.scatter([x0, x1], [y0, y1], color='crimson', zorder=5)

alpha = 0.4
xm = alpha * x0 + (1 - alpha) * x1
chord_val = alpha * y0 + (1 - alpha) * y1
graph_val = xm**2
ax.scatter([xm], [graph_val], color='black', zorder=6, marker='o', s=45)
ax.scatter([xm], [chord_val], color='crimson', zorder=6, marker='s', s=45)
ax.annotate(
    rf'$f(\alpha x+(1-\alpha)y)={graph_val:.2f}$' + '\n' +
    rf'$\alpha f(x)+(1-\alpha)f(y)={chord_val:.2f}$',
    xy=(xm, graph_val), xytext=(xm + 0.15, graph_val + 1.6),
    fontsize=9.5,
    arrowprops=dict(arrowstyle='->', color='gray'))

ax.text(x0, y0 + 0.25, r'$(x,f(x))$', fontsize=10, ha='center')
ax.text(x1, y1 + 0.25, r'$(y,f(y))$', fontsize=10, ha='center')
ax.set_title(r'Convex: the chord lies on or above the graph')
ax.set_xlabel('$x$')
ax.set_ylabel('value')
ax.legend(loc='upper center', fontsize=9.5)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig('fig_chord_convex.pdf')
plt.close(fig)

# ===========================================================================
# Figure 2: non-convex function -- chord dips below the graph
# ===========================================================================
fig, ax = plt.subplots(figsize=(6, 4.5))

x = np.linspace(-2.2, 2.2, 400)
g = x**4 - 3 * x**2
ax.plot(x, g, color='navy', label=r'$g(x) = x^4 - 3x^2$')

x0, x1 = -1.6, 1.6
y0, y1 = x0**4 - 3 * x0**2, x1**4 - 3 * x1**2
ax.plot([x0, x1], [y0, y1], color='crimson', linestyle='--', linewidth=1.8,
        label='chord joining the two points')
ax.scatter([x0, x1], [y0, y1], color='crimson', zorder=5)

alpha = 0.5
xm = alpha * x0 + (1 - alpha) * x1
chord_val = alpha * y0 + (1 - alpha) * y1
graph_val = xm**4 - 3 * xm**2
ax.scatter([xm], [graph_val], color='black', zorder=6, marker='o', s=45)
ax.scatter([xm], [chord_val], color='crimson', zorder=6, marker='s', s=45)
ax.annotate(
    rf'$g(\alpha x+(1-\alpha)y)={graph_val:.2f} > $' + '\n' +
    rf'$\alpha g(x)+(1-\alpha)g(y)={chord_val:.2f}$'
    '\n(convexity inequality FAILS)',
    xy=(xm, chord_val), xytext=(xm + 0.15, chord_val - 3.2),
    fontsize=9,
    arrowprops=dict(arrowstyle='->', color='gray'))

ax.set_title(r'Non-convex: the chord dips below the graph')
ax.set_xlabel('$x$')
ax.set_ylabel('value')
ax.legend(loc='upper center', fontsize=9.5)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig('fig_chord_nonconvex.pdf')
plt.close(fig)

# ===========================================================================
# Figure 3: epigraph of f(x) = x^2
# ===========================================================================
fig, ax = plt.subplots(figsize=(6, 5))

x = np.linspace(-2.2, 2.2, 400)
f = x**2
ax.fill_between(x, f, 4.9, color='steelblue', alpha=0.35,
                 label=r'epi $f = \{(x,\eta): \eta \geq f(x)\}$')
ax.plot(x, f, color='navy', linewidth=2.2, label=r'graph of $f(x)=x^2$')

# two points inside epi f and their convex combination
p1 = (-1.3, 3.5)
p2 = (1.6, 4.0)
ax.scatter(*p1, color='black', zorder=5)
ax.scatter(*p2, color='black', zorder=5)
alpha = 0.3
pm = (alpha * p1[0] + (1 - alpha) * p2[0], alpha * p1[1] + (1 - alpha) * p2[1])
ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color='crimson', linestyle='--',
        linewidth=1.6)
ax.scatter(*pm, color='crimson', zorder=6, marker='D', s=45,
           label=r'$\alpha(x,\xi)+(1-\alpha)(y,\eta)\in\mathrm{epi}\,f$')

ax.set_xlim(-2.2, 2.2)
ax.set_ylim(-0.3, 5)
ax.set_xlabel(r'$x$ (or $\mathcal{H}$)')
ax.set_ylabel(r'$\eta$ (or $\mathbb{R}$)')
ax.set_title(r'Proposition 8.4: $f$ convex $\Leftrightarrow$ epi $f$ convex')
ax.legend(loc='upper center', fontsize=8.7)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig('fig_epigraph.pdf')
plt.close(fig)

# ===========================================================================
# Figure 4: the Huber function (Example 8.44), rho = 1
# ===========================================================================
fig, ax = plt.subplots(figsize=(6, 4.5))

rho = 1.0
x = np.linspace(-3, 3, 600)
huber = np.where(np.abs(x) > rho, rho * np.abs(x) - rho**2 / 2, x**2 / 2)
quad = x**2 / 2
lin_pos = rho * x - rho**2 / 2
lin_neg = -rho * x - rho**2 / 2

ax.plot(x, quad, color='gray', linestyle=':', linewidth=1.4,
        label=r'$|x|^2/2$ (used for $|x|\leq\rho$)')
ax.plot(x, lin_pos, color='gray', linestyle='-.', linewidth=1.2)
ax.plot(x, lin_neg, color='gray', linestyle='-.', linewidth=1.2,
        label=r'$\rho|x|-\rho^2/2$ (used for $|x|>\rho$)')
ax.plot(x, huber, color='darkorange', linewidth=2.6, label='Huber function $f$')
ax.axvline(rho, color='black', linestyle=':', linewidth=0.8)
ax.axvline(-rho, color='black', linestyle=':', linewidth=0.8)

ax.set_ylim(-0.3, 4)
ax.set_title(r'Huber function ($\rho=1$): continuous and convex everywhere')
ax.set_xlabel('$x$')
ax.set_ylabel('$f(x)$')
ax.legend(loc='upper center', fontsize=9)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig('fig_huber.pdf')
plt.close(fig)

print("Wrote fig_chord_convex.pdf, fig_chord_nonconvex.pdf, "
      "fig_epigraph.pdf, fig_huber.pdf")
