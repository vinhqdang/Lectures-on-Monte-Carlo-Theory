#!/usr/bin/env python3
"""
gen_figures.py

Generates the figures used in the Chapter 5 slides on Fejer monotonicity and
fixed point iterations (Bauschke & Combettes, 2nd ed., Chapter 5).

Running example reused throughout the chapter (and shared with the
Krasnosel'skii-Mann book's chapter03 slides in this repository):

    H = R^2,  C = closed unit disk,  T = P_C  (metric projection onto C),
    x0 = (3, 4),   Fix(T) = C,  and the KM sequence with x0 on the ray
    through the boundary point (0.6, 0.8) has fixed target x* = (0.6, 0.8),
    constant relaxation lambda_n = 0.5.

    For ||x|| > 1:  T(x) = x / ||x||   (radial projection onto the boundary)
    For ||x|| <= 1: T(x) = x

Because x0 = 5*(0.6,0.8) lies on the ray through the target point, every KM
iterate x_n stays on that ray: writing x_n = r_n * (0.6,0.8), the vector
recursion x_{n+1} = x_n + lambda(Tx_n - x_n) collapses to the scalar
recursion

        r_{n+1} = (1-lambda) r_n + lambda ,   r_0 = 5, lambda = 0.5,

reproducing r_0=5, r_1=3, r_2=2, r_3=1.5, r_4=1.25, ... -> 1, i.e.
x_n -> (0.6,0.8).

All figures are produced directly from these recursions (or from an
explicit auxiliary construction for the abstract Fejer-monotone and
quasi-Fejer illustrations), so every number quoted in the slides can be
reproduced by running this script. Figures are saved as vector PDFs.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 13,
    "axes.titlesize": 13,
    "legend.fontsize": 10.5,
    "lines.linewidth": 2.0,
    "figure.autolayout": True,
})

X0 = np.array([3.0, 4.0])
XSTAR = np.array([0.6, 0.8])


def T_disk(x):
    """Metric projection onto the closed unit disk in R^2."""
    n = np.linalg.norm(x)
    if n > 1.0:
        return x / n
    return x.copy()


def km_orbit(lmbda_fn, n_steps, x0=X0, T=T_disk):
    """
    Run x_{n+1} = x_n + lambda_n (T(x_n) - x_n), lambda_n = lmbda_fn(n).
    Returns array of shape (n_steps+1, 2): x_0, ..., x_{n_steps}.
    """
    x = x0.copy()
    pts = [x.copy()]
    for n in range(n_steps):
        lam = lmbda_fn(n)
        x = x + lam * (T(x) - x)
        pts.append(x.copy())
    return np.array(pts)


# ----------------------------------------------------------------------
# Figure (a): an abstract Fejer monotone sequence -- distances to the
# target point x are non-increasing, but NOT necessarily strictly
# decreasing (plateaus are allowed and shown here on purpose).
# ----------------------------------------------------------------------
def fig_fejer_monotone_distances():
    # A hand-built sequence of distances d_n = ||x_n - x|| that is
    # non-increasing with genuine plateaus, converging to a limit > 0
    # (recall: Fejer monotonicity does NOT force convergence to 0 --
    # only that d_n decreases to some limit, e.g. Example 5.6).
    d = np.array([5.0, 5.0, 4.2, 4.2, 4.2, 3.4, 2.9, 2.9,
                  2.3, 2.3, 1.9, 1.7, 1.7, 1.7, 1.5, 1.4,
                  1.35, 1.32, 1.32, 1.32])
    n = np.arange(len(d))
    assert np.all(np.diff(d) <= 1e-12), "sequence must be non-increasing"

    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    ax.step(n, d, where="post", color="#1f77b4", lw=1.4, alpha=0.6)
    ax.plot(n, d, "o", color="#1f77b4", ms=6,
            label=r"$d_n = \|x_n - x\|$, one fixed $x \in C$")
    ax.axhline(d[-1], color="#d62728", ls="--", lw=1.5,
               label=rf"limit $\approx {d[-1]:.2f}$ (need not be $0$)")
    ax.set_xlabel(r"$n$")
    ax.set_ylabel(r"$\|x_n - x\|$")
    ax.set_title("Fejér monotonicity: never farther, but flat spells allowed")
    ax.set_ylim(0, 5.6)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.savefig("fejer_monotone_distances.pdf")
    plt.close(fig)


# ----------------------------------------------------------------------
# Figure (b): KM iteration trajectory for the disk-projection example,
# lambda_n = 0.5 constant -- the running example of the chapter.
# ----------------------------------------------------------------------
def fig_km_disk_trajectory():
    fig, ax = plt.subplots(figsize=(5.8, 5.8))

    theta = np.linspace(0, 2 * np.pi, 400)
    ax.fill(np.cos(theta), np.sin(theta), color="0.93", zorder=0)
    ax.plot(np.cos(theta), np.sin(theta), color="0.55", lw=1.4,
            linestyle="--", label=r"$C = $ closed unit disk $= \mathrm{Fix}\,T$")

    pts = km_orbit(lambda n: 0.5, 8)
    ax.plot(pts[:, 0], pts[:, 1], "o-", color="#d62728", ms=6,
            label=r"$x_{n+1}=x_n+0.5(Tx_n-x_n)$")

    for k, p in enumerate(pts[:5]):
        r = np.linalg.norm(p)
        ax.annotate(rf"$x_{{{k}}}$, $r_{{{k}}}={r:g}$", p,
                    textcoords="offset points", xytext=(8, 6), fontsize=9.5)

    ax.plot(*X0, marker="*", ms=16, color="black", zorder=5)
    ax.plot(*XSTAR, marker="X", ms=11, color="black", zorder=5)
    ax.annotate(r"$x^\ast=(0.6,0.8)\in\mathrm{Fix}\,T$", XSTAR,
                textcoords="offset points", xytext=(10, -16), fontsize=11)

    ax.set_xlabel(r"$x^{(1)}$")
    ax.set_ylabel(r"$x^{(2)}$")
    ax.set_title("Running example: KM iteration is Fejér monotone w.r.t. Fix $T$")
    ax.set_aspect("equal")
    ax.legend(loc="upper left", framealpha=0.9, fontsize=9.5)
    ax.grid(alpha=0.25)
    fig.savefig("km_disk_trajectory.pdf")
    plt.close(fig)

    r = np.linalg.norm(pts, axis=1)
    print("KM disk-trajectory radii r_n =", np.round(r, 4))


# ----------------------------------------------------------------------
# Figure (c): quasi-Fejer monotone sequence -- squared distances may
# increase a little at each step, but only by a summable amount eps_n;
# the sequence still converges (Lemma 5.31 / Theorem 5.33).
# ----------------------------------------------------------------------
def fig_quasi_fejer():
    rng = np.random.default_rng(0)
    N = 40
    eps = 0.6 / (np.arange(1, N + 1) ** 1.6)          # summable errors
    d2 = np.zeros(N + 1)
    d2[0] = 25.0
    true_decrease = 0.35
    for k in range(N):
        # true Fejer-type contraction plus a random perturbation bounded by eps_k
        wiggle = rng.uniform(-1, 1) * eps[k]
        d2[k + 1] = max(0.0, d2[k] - true_decrease + eps[k] + wiggle - eps[k])
        d2[k + 1] = min(d2[k + 1], d2[k] + eps[k])     # enforce (5.41)-type bound

    d = np.sqrt(d2)
    n = np.arange(N + 1)

    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    ax.plot(n, d2, "o-", color="#9467bd", ms=4.5,
            label=r"$\|x_{n+1}-x\|^2$ (quasi-Fejér, can bump up)")
    env = 25.0 - true_decrease * n + np.concatenate(([0], np.cumsum(eps)))
    ax.plot(n, np.clip(env, 0, None), "--", color="0.5", lw=1.3,
            label=r"envelope $d_0^2 - (\mathrm{net\ decrease}) + \sum_{k<n}\varepsilon_k$")
    ax.set_xlabel(r"$n$")
    ax.set_ylabel(r"$\|x_n - x\|^2$")
    ax.set_title(r"Quasi-Fejér: $\|x_{n+1}-x\|^2\leq\|x_n-x\|^2+\varepsilon_n$, $\sum\varepsilon_n<\infty$")
    ax.legend(fontsize=9.5)
    ax.grid(alpha=0.3)
    fig.savefig("quasi_fejer.pdf")
    plt.close(fig)


# ----------------------------------------------------------------------
# Figure (d): Baillon's nonlinear ergodic theorem -- Picard iterates of
# a nonexpansive (but not averaged) rotation do NOT converge, yet their
# Cesaro averages converge to the (unique) fixed point 0.
# ----------------------------------------------------------------------
def fig_baillon_ergodic():
    # T = rotation by 90 degrees: nonexpansive (isometry), Fix T = {0}.
    def T_rot(x):
        return np.array([-x[1], x[0]])

    x0 = np.array([1.0, 0.0])
    N = 24
    pts = [x0.copy()]
    x = x0.copy()
    for _ in range(N):
        x = T_rot(x)
        pts.append(x.copy())
    pts = np.array(pts)  # Picard iterates T^k x0, k=0..N

    cesaro = np.cumsum(pts, axis=0) / (np.arange(len(pts)).reshape(-1, 1) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 5.0))

    ax = axes[0]
    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(theta), np.sin(theta), color="0.7", lw=1, ls="--")
    ax.plot(pts[:9, 0], pts[:9, 1], "o", color="#2ca02c", ms=7,
            label=r"Picard iterates $T^k x_0$ (cycle, never converge)")
    ax.plot(0, 0, marker="X", ms=12, color="black", zorder=5,
            label=r"$\mathrm{Fix}\,T=\{0\}$")
    ax.set_aspect("equal")
    ax.set_title(r"$T^k x_0$ rotates forever: $(x_n)$ diverges")
    ax.set_xlabel(r"$x^{(1)}$"); ax.set_ylabel(r"$x^{(2)}$")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(np.arange(len(cesaro)), np.linalg.norm(cesaro, axis=1),
            "o-", color="#ff7f0e", ms=5,
            label=r"$\left\|\frac{1}{n+1}\sum_{k=0}^n T^k x_0\right\|$")
    ax.set_xlabel(r"$n$")
    ax.set_ylabel("norm of Cesàro average")
    ax.set_title("Baillon: Cesàro averages converge to $\\mathrm{Fix}\\,T$")
    ax.legend(fontsize=9.5)
    ax.grid(alpha=0.3)

    fig.savefig("baillon_ergodic.pdf")
    plt.close(fig)


if __name__ == "__main__":
    fig_fejer_monotone_distances()
    fig_km_disk_trajectory()
    fig_quasi_fejer()
    fig_baillon_ergodic()
    print("All figures written to figures/")
