#!/usr/bin/env python3
"""
gen_figures.py
Generates all figures for Chapter 10 (Convex Functions: Variants) slides,
Bauschke & Combettes, "Convex Analysis and Monotone Operator Theory in
Hilbert Spaces", 2nd ed.

Figures produced (all saved as vector PDF in this directory):
  1. fig_hierarchy.pdf          -- linear vs. convex vs. strongly convex
                                    vs. uniformly convex 1-D functions.
  2. fig_strong_gap.pdf         -- explicit "curvature gap" picture for
                                    f(x) = x^2, illustrating the strong
                                    convexity inequality with real numbers.
  3. fig_quasiconvex.pdf        -- a quasiconvex-but-not-convex function
                                    (f(x) = |x|/(|x|+1)) together with a
                                    sublevel set, and the book's own
                                    Exercise 10.12 function
                                    (2x if x<=0, x if x>0), which is not
                                    convex yet uniformly quasiconvex.

Plain python3 + matplotlib only (Agg backend, no display needed).
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.size": 11,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# ----------------------------------------------------------------------
# Figure 1: linear / convex(non-strict) / strongly convex / uniformly
#            convex (non-strongly) functions on the same axes.
# ----------------------------------------------------------------------
x = np.linspace(-2, 2, 801)

f_lin = 1.5 * x                    # linear:            sublinear, not convex-strict
f_conv = np.abs(x)                 # convex, NOT strictly convex (affine on each ray)
f_strong = x ** 2                  # strongly convex with constant beta = 2
f_unif = 0.25 * x ** 4              # uniformly convex (modulus grows faster than
                                    # quadratic) but NOT strongly convex -- see
                                    # Exercise 10.8 of the book (f = |.|^4).

fig, ax = plt.subplots(figsize=(7.2, 5.0))
ax.plot(x, f_lin, label=r"linear: $f(x)=1.5x$ (sublinear, not convex-strict)",
        color="#4C72B0", lw=2.2)
ax.plot(x, f_conv, label=r"convex, not strictly convex: $f(x)=|x|$",
        color="#DD8452", lw=2.2)
ax.plot(x, f_strong, label=r"strongly convex ($\beta=2$): $f(x)=x^2$",
        color="#55A868", lw=2.4)
ax.plot(x, f_unif, label=r"uniformly convex, not strongly: $f(x)=\frac{1}{4} x^4$",
        color="#C44E52", lw=2.4)

ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$f(x)$")
ax.set_title("Linear $\\;\\subsetneq\\;$ convex $\\;\\subsetneq\\;$ strictly convex\n"
             "strongly convex $\\;\\subsetneq\\;$ uniformly convex $\\;\\subsetneq\\;$ strictly convex")
ax.set_ylim(-1, 5)
ax.legend(loc="upper center", fontsize=8.5, framealpha=0.95)
fig.tight_layout()
fig.savefig("fig_hierarchy.pdf")
plt.close(fig)

# ----------------------------------------------------------------------
# Figure 2: explicit strong-convexity "gap" picture for f(x) = x^2.
#   Shows the chord between x=-1 and y=3, the function value at the
#   midpoint, and the guaranteed quadratic gap
#   alpha f(x) + (1-alpha) f(y) - f(alpha x + (1-alpha) y)
#      >= alpha(1-alpha)(beta/2) ||x-y||^2
#   evaluated at alpha = 1/2, beta = 2, x = -1, y = 3.
# ----------------------------------------------------------------------
xx = np.linspace(-2.5, 3.5, 600)
f = xx ** 2

x0, y0, alpha = -1.0, 3.0, 0.5
beta = 2.0
mid = alpha * x0 + (1 - alpha) * y0
chord_val = alpha * (x0 ** 2) + (1 - alpha) * (y0 ** 2)
f_mid = mid ** 2
gap = chord_val - f_mid
guaranteed_gap = alpha * (1 - alpha) * (beta / 2) * (x0 - y0) ** 2

fig, ax = plt.subplots(figsize=(7.2, 5.2))
ax.plot(xx, f, color="#4C72B0", lw=2.4, label=r"$f(x)=x^2$")
ax.plot([x0, y0], [x0 ** 2, y0 ** 2], color="#DD8452", lw=2, ls="--",
        marker="o", label="chord joining $(x,f(x))$ and $(y,f(y))$")
ax.plot([mid, mid], [f_mid, chord_val], color="#55A868", lw=3,
        solid_capstyle="butt",
        label=r"gap $=\alpha f(x)+(1-\alpha)f(y)-f(\alpha x+(1-\alpha)y)$")
ax.scatter([mid], [f_mid], color="#C44E52", zorder=5)
ax.scatter([mid], [chord_val], color="#DD8452", zorder=5)

ax.annotate(f"$f(x)=f(-1)={x0**2:.0f}$", (x0, x0 ** 2), textcoords="offset points",
            xytext=(-45, 8), fontsize=9)
ax.annotate(f"$f(y)=f(3)={y0**2:.0f}$", (y0, y0 ** 2), textcoords="offset points",
            xytext=(-10, 8), fontsize=9)
ax.annotate(f"chord value $={chord_val:.2f}$", (mid, chord_val), textcoords="offset points",
            xytext=(8, 4), fontsize=9)
ax.annotate(f"$f(\\mathrm{{mid}})=f(1)={f_mid:.2f}$", (mid, f_mid), textcoords="offset points",
            xytext=(8, -18), fontsize=9)

ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$f(x)$")
ax.set_title(
    r"$\alpha=\frac{1}{2},\; x=-1,\; y=3$:   actual gap $=%.2f\;\geq\;$ guaranteed "
    r"$\alpha(1-\alpha)\frac{\beta}{2}\|x-y\|^2=%.2f$" % (gap, guaranteed_gap),
    fontsize=10)
ax.legend(loc="upper center", fontsize=8, framealpha=0.95)
fig.tight_layout()
fig.savefig("fig_strong_gap.pdf")
plt.close(fig)

# ----------------------------------------------------------------------
# Figure 3: quasiconvex-but-not-convex functions.
#   Left panel:  f(x) = |x| / (|x|+1)  (Exercise 10.13): strictly
#                quasiconvex, NOT convex (concave on each ray).
#                A sublevel set lev_{<= xi} f is shown as a shaded
#                interval to make "convex sublevel sets" concrete.
#   Right panel: the book's own Exercise 10.12 function
#                f(x) = 2x if x <= 0, x if x > 0: monotone increasing
#                hence quasiconvex (Example 10.22), but NOT convex
#                (the slope drops from 2 to 1 at x=0).
# ----------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.6))

# -- left panel --
xa = np.linspace(-6, 6, 1200)
fa = np.abs(xa) / (np.abs(xa) + 1.0)
ax = axes[0]
ax.plot(xa, fa, color="#4C72B0", lw=2.4, label=r"$f(x)=\dfrac{|x|}{|x|+1}$")

xi = 0.5
ax.axhline(xi, color="#C44E52", lw=1.4, ls="--", label=r"level $\xi=0.5$")
# lev_{<= xi} f = {x : |x|/(|x|+1) <= xi} = [-xi/(1-xi), xi/(1-xi)]
bound = xi / (1 - xi)
mask = xa <= bound
mask &= xa >= -bound
ax.fill_between(xa, 0, fa, where=mask, color="#55A868", alpha=0.3,
                 label=r"sublevel set $\mathrm{lev}_{\leqslant \xi} f=[-1,1]$ (convex)")

# concavity witness: chord below curve on x>0 ray
x1, x2 = 0.3, 4.0
y1, y2 = x1 / (x1 + 1), x2 / (x2 + 1)
ax.plot([x1, x2], [y1, y2], color="#DD8452", lw=1.8, ls=":", marker="o", ms=4,
        label="chord lies BELOW graph $\\Rightarrow$ concave arm $\\Rightarrow$ not convex")

ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$f(x)$")
ax.set_title("Quasiconvex, not convex\n(Exercise 10.13)")
ax.legend(loc="lower right", fontsize=7.3, framealpha=0.95)
ax.set_ylim(-0.05, 1.05)

# -- right panel --
xb = np.linspace(-3, 3, 1200)
fb = np.where(xb <= 0, 2 * xb, xb)
ax = axes[1]
ax.plot(xb, fb, color="#4C72B0", lw=2.4,
        label=r"$f(x)=2x\ (x\leqslant0),\;\; x\ (x>0)$")

xL, xR = -1.0, 1.0
yL, yR = 2 * xL, xR
ax.plot([xL, xR], [yL, yR], color="#DD8452", lw=1.8, ls=":", marker="o", ms=5,
        label="chord from $x=-1$ to $y=1$")
mid_x = 0.0
chord_mid = 0.5 * yL + 0.5 * yR
ax.scatter([mid_x], [fb[np.argmin(np.abs(xb - mid_x))]], color="#C44E52", zorder=5)
ax.annotate(f"$f(0)=0 > $ chord midpoint $={chord_mid:.1f}$\n$\\Rightarrow$ NOT convex",
            (mid_x, 0), textcoords="offset points", xytext=(10, -35), fontsize=8)
ax.annotate("increasing on $\\mathbb{R}$\n$\\Rightarrow$ quasiconvex\n(Example 10.22)",
            (1.6, 1.6), fontsize=8, color="#2b6a2b")

ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$f(x)$")
ax.set_title("Quasiconvex, not convex\n(Exercise 10.12)")
ax.legend(loc="upper left", fontsize=7.5, framealpha=0.95)

fig.tight_layout()
fig.savefig("fig_quasiconvex.pdf")
plt.close(fig)

print("Wrote fig_hierarchy.pdf, fig_strong_gap.pdf, fig_quasiconvex.pdf")
