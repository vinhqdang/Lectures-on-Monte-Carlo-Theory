#!/usr/bin/env python3
"""
Figure generation for Chapter 5: The Inertial Krasnosel'skii-Mann Iteration.

Produces (as vector PDFs):
  (a) km_vs_inertial_km.pdf   -- plain KM vs. inertial KM on the running
                                  example (projection onto the unit disk,
                                  x0 = (3,4)), showing faster convergence
                                  of the inertial variant.
  (b) heavy_ball_schematic.pdf -- a "heavy ball rolling downhill toward a
                                  target" schematic motivating momentum /
                                  inertia.
  (c) rn_convergence.pdf       -- semilog plot of |r_n - 1| for both
                                  methods, i.e. distance-to-fixed-point
                                  along the ray, versus iteration number.

Plain python3 + matplotlib (Agg backend), no external dependencies.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.size": 11,
    "axes.grid": True,
    "grid.alpha": 0.3,
})


# ----------------------------------------------------------------------
# The nonexpansive operator T = projection onto the closed unit disk
# ----------------------------------------------------------------------
def T(x):
    n = np.linalg.norm(x)
    return x if n <= 1.0 else x / n


def plain_km(x0, lam=0.5, n_iter=12):
    x = np.array(x0, dtype=float)
    hist = [x.copy()]
    for _ in range(n_iter):
        x = (1 - lam) * x + lam * T(x)
        hist.append(x.copy())
    return np.array(hist)


def inertial_km(x0, theta=0.2, lam=0.5, n_iter=8):
    """
    yn = xn + theta_n (xn - x_{n-1})
    x_{n+1} = (1-lam) yn + lam T(yn)
    with x_{-1} = x0 for the very first step (colinear start).
    """
    xprev = np.array(x0, dtype=float)
    x = np.array(x0, dtype=float)
    hist = [x.copy()]
    for _ in range(n_iter):
        y = x + theta * (x - xprev)
        xnew = (1 - lam) * y + lam * T(y)
        xprev = x
        x = xnew
        hist.append(x.copy())
    return np.array(hist)


x0 = np.array([3.0, 4.0])
lam = 0.5
theta = 0.2

hist_plain = plain_km(x0, lam=lam, n_iter=12)
hist_iner = inertial_km(x0, theta=theta, lam=lam, n_iter=8)

r_plain = np.linalg.norm(hist_plain, axis=1)
r_iner = np.linalg.norm(hist_iner, axis=1)


# ----------------------------------------------------------------------
# Figure (a): trajectories in the plane, unit disk, both methods
# ----------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))

circle_theta = np.linspace(0, 2 * np.pi, 400)
circle_x, circle_y = np.cos(circle_theta), np.sin(circle_theta)

for ax, hist, r, title, color in [
    (axes[0], hist_plain, r_plain, "Plain KM ($\\lambda_n \\equiv 0.5$)", "tab:blue"),
    (axes[1], hist_iner, r_iner, "Inertial KM ($\\theta_n = 0.2$, $\\lambda_n \\equiv 0.5$)", "tab:red"),
]:
    ax.plot(circle_x, circle_y, "k--", lw=1.2, label="unit circle $\\partial C$")
    ax.fill(circle_x, circle_y, color="gray", alpha=0.08)
    ax.plot(hist[:, 0], hist[:, 1], "o-", color=color, ms=5, lw=1.6,
            label="iterates $x_n$")
    ax.plot([0.6], [0.8], marker="*", color="black", ms=16, zorder=5,
            label="fixed point $(0.6,0.8)$")
    ax.plot([0], [0], marker="+", color="gray", ms=8)
    for i, (px, py) in enumerate(hist):
        if i <= 4 or i == len(hist) - 1:
            ax.annotate(f"$x_{{{i}}}$", (px, py), textcoords="offset points",
                        xytext=(6, 6), fontsize=8)
    ax.set_xlim(-0.6, 5.3)
    ax.set_ylim(-0.4, 5.3)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("$x^{(1)}$")
    ax.set_ylabel("$x^{(2)}$")
    ax.legend(loc="upper right", fontsize=8)

fig.suptitle("Running example: projection onto the unit disk, $x_0=(3,4)$",
             fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig("km_vs_inertial_km.pdf")
plt.close(fig)


# ----------------------------------------------------------------------
# Figure (c): convergence of |r_n - 1| (distance to the fixed point,
# along the ray) versus iteration number, semilog scale
# ----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.5, 5))
n_plain = np.arange(len(r_plain))
n_iner = np.arange(len(r_iner))

ax.semilogy(n_plain, np.abs(r_plain - 1), "o-", color="tab:blue",
            label="Plain KM: $|r_n-1| = 4\\cdot(0.5)^n$")
ax.semilogy(n_iner, np.abs(r_iner - 1), "s-", color="tab:red",
            label="Inertial KM: $|r_n-1|$ (faster decay)")
ax.axhline(1e-3, color="gray", ls=":", lw=1.2, label="tolerance $10^{-3}$")

ax.annotate("plain KM: 12 iters", xy=(12, abs(r_plain[12] - 1)),
            xytext=(12.3, 0.02),
            arrowprops=dict(arrowstyle="->", color="tab:blue"),
            color="tab:blue", fontsize=9)
ax.annotate("inertial KM: 8 iters", xy=(8, abs(r_iner[8] - 1)),
            xytext=(3.0, 0.0003),
            arrowprops=dict(arrowstyle="->", color="tab:red"),
            color="tab:red", fontsize=9)

ax.set_xlabel("iteration $n$")
ax.set_ylabel("$|r_n - 1|$  (distance to fixed point along the ray)")
ax.set_title("Inertia reaches tolerance $10^{-3}$ in fewer iterations")
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig("rn_convergence.pdf")
plt.close(fig)


# ----------------------------------------------------------------------
# Figure (b): heavy-ball-rolling-downhill schematic motivating momentum
# ----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 5))

# A bumpy "valley" landscape (potential well) with a small local wiggle
xs = np.linspace(-4.5, 4.5, 600)
landscape = 0.18 * xs**2 + 0.35 * np.sin(1.7 * xs) * np.exp(-0.15 * xs**2)
ax.plot(xs, landscape, color="saddlebrown", lw=2.5, zorder=1)
ax.fill_between(xs, landscape, landscape.min() - 1.0, color="tan", alpha=0.35, zorder=0)

# target at bottom of the valley
target_x = 0.05
target_y = np.interp(target_x, xs, landscape)
ax.plot(target_x, target_y, marker="*", color="black", ms=20, zorder=5)
ax.annotate("target / fixed point", (target_x, target_y),
            xytext=(1.0, target_y - 1.3),
            arrowprops=dict(arrowstyle="->"), fontsize=10)

# Path 1: plain gradient-descent-like ball -- gets stuck / slow through the bump
gd_x = np.array([-4.2, -3.4, -2.75, -2.35, -2.15, -2.05, -2.0])
gd_y = np.interp(gd_x, xs, landscape) + 0.05
ax.plot(gd_x, gd_y, "o-", color="tab:blue", ms=6, lw=2,
         label="no inertia: stalls at the small bump")

# Path 2: heavy ball with momentum -- coasts over the bump and reaches bottom
hb_x = np.array([-4.2, -3.1, -1.9, -0.6, 0.6, 0.15, 0.05])
hb_y = np.interp(hb_x, xs, landscape) + 0.05
ax.plot(hb_x, hb_y, "s-", color="tab:red", ms=6, lw=2,
         label="with inertia: coasts through the bump")

# velocity arrow illustration on the heavy ball path
for i in range(len(hb_x) - 1):
    ax.annotate("", xy=(hb_x[i+1], hb_y[i+1]), xytext=(hb_x[i], hb_y[i]),
                arrowprops=dict(arrowstyle="-|>", color="tab:red", alpha=0.35, lw=1))

ax.text(-4.3, landscape.max() + 0.3,
        "Heavy ball with momentum: past velocity carries it\n"
        "through small bumps that would stall a memoryless method.",
        fontsize=9.5, va="top")

ax.set_xlim(-4.7, 4.7)
ax.set_ylim(landscape.min() - 0.6, landscape.max() + 1.0)
ax.set_xlabel("position $x$")
ax.set_ylabel("potential $\\varphi(x)$")
ax.set_title("Motivation: the heavy-ball / momentum analogy")
ax.legend(loc="lower right", fontsize=9)
fig.tight_layout()
fig.savefig("heavy_ball_schematic.pdf")
plt.close(fig)

print("Wrote km_vs_inertial_km.pdf, rn_convergence.pdf, heavy_ball_schematic.pdf")
print("plain KM iters to 1e-3:", len(r_plain) - 1, " last r:", r_plain[-1])
print("inertial KM iters to 1e-3:", len(r_iner) - 1, " last r:", r_iner[-1])
