#!/usr/bin/env python3
"""
gen_figures.py

Generates the figures used in Chapter 3 slides on the Krasnosel'skii-Mann
(KM) iteration.  Running example throughout: T = metric projection onto the
closed unit disk in R^2, x0 = (3,4), unique fixed point x* = (0.6, 0.8).

For x with ||x|| > 1,  T(x) = x / ||x||  (projects radially onto the unit
circle); for ||x|| <= 1, T(x) = x.  Because x0 is a positive multiple of the
fixed-point direction u = (0.6, 0.8), every KM iterate stays on the ray
{r u : r > 0}, and writing x_n = r_n u the whole vector recursion collapses
to the scalar recursion

        r_{n+1} = (1 - lambda_n) r_n + lambda_n ,      r_0 = 5 ,

(since T(x_n) = u, i.e. "radius" 1, whenever r_n > 1).  Equivalently, with
e_n := ||x_n - x*|| = r_n - 1,

        e_{n+1} = (1 - lambda_n) e_n .

All plots below are produced directly from this recursion / the R^2 vector
iteration, so the numbers drawn match the numbers quoted in the slides
exactly.

Saves all figures as vector PDFs in the current directory (figures/).
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


def T(x):
    """Metric projection onto the closed unit disk in R^2."""
    n = np.linalg.norm(x)
    if n > 1.0:
        return x / n
    return x.copy()


def km_orbit(lmbda_fn, n_steps, x0=X0):
    """
    Run the KM iteration x_{n+1} = (1-lambda_n) x_n + lambda_n T(x_n).
    lmbda_fn(n) returns lambda_n for step index n = 0, 1, 2, ...
    Returns array of shape (n_steps+1, 2) of iterates x_0, ..., x_{n_steps}.
    """
    x = x0.copy()
    pts = [x.copy()]
    for n in range(n_steps):
        lam = lmbda_fn(n)
        x = (1 - lam) * x + lam * T(x)
        pts.append(x.copy())
    return np.array(pts)


# ----------------------------------------------------------------------
# Figure (a): trajectories in R^2 for constant lambda = 0.3, 0.5, 0.8
# ----------------------------------------------------------------------
def fig_trajectories():
    fig, ax = plt.subplots(figsize=(5.6, 5.6))

    theta = np.linspace(0, 2 * np.pi, 400)
    ax.plot(np.cos(theta), np.sin(theta), color="0.6", lw=1.4,
             linestyle="--", label=r"unit circle $\partial C$")
    ax.fill(np.cos(theta), np.sin(theta), color="0.93", zorder=0)

    colors = {0.3: "#1f77b4", 0.5: "#d62728", 0.8: "#2ca02c"}
    for lam in [0.3, 0.5, 0.8]:
        pts = km_orbit(lambda n: lam, 10)
        ax.plot(pts[:, 0], pts[:, 1], "o-", color=colors[lam], ms=5,
                 label=rf"$\lambda_n \equiv {lam}$")

    ax.plot(*X0, marker="*", ms=16, color="black", zorder=5)
    ax.annotate(r"$x_0=(3,4)$", X0, textcoords="offset points",
                xytext=(8, 6), fontsize=11)
    ax.plot(*XSTAR, marker="X", ms=11, color="black", zorder=5)
    ax.annotate(r"$x^*=(0.6,0.8)$", XSTAR, textcoords="offset points",
                xytext=(8, -14), fontsize=11)

    ax.set_xlabel(r"$x^{(1)}$")
    ax.set_ylabel(r"$x^{(2)}$")
    ax.set_title("KM trajectories for the disk-projection example")
    ax.set_aspect("equal")
    ax.legend(loc="upper left", framealpha=0.9)
    ax.grid(alpha=0.25)
    fig.savefig("km_trajectories.pdf")
    plt.close(fig)


# ----------------------------------------------------------------------
# Figure (b): ||x_n - x*|| vs n on log scale -- geometric decay
# ----------------------------------------------------------------------
def fig_convergence_rate():
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    colors = {0.3: "#1f77b4", 0.5: "#d62728", 0.8: "#2ca02c"}
    for lam in [0.3, 0.5, 0.8]:
        pts = km_orbit(lambda n: lam, 12)
        err = np.linalg.norm(pts - XSTAR, axis=1)
        ax.semilogy(range(len(err)), err, "o-", color=colors[lam], ms=5,
                    label=rf"$\lambda_n \equiv {lam}$  (rate $(1-\lambda)^n$)")
    ax.set_xlabel(r"iteration $n$")
    ax.set_ylabel(r"$\|x_n - x^*\|$  (log scale)")
    ax.set_title("Geometric decay for constant relaxation")
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    fig.savefig("km_convergence_rate.pdf")
    plt.close(fig)


# ----------------------------------------------------------------------
# Figure (c): condition sum lambda_n(1-lambda_n) = infinity vs < infinity
# ----------------------------------------------------------------------
def fig_condition_violation():
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.4))

    n_steps = 24

    # GOOD: lambda_n = 1/(n+2)  => sum lambda_n(1-lambda_n) diverges (harmonic-like)
    good_lam = lambda n: 1.0 / (n + 2)
    pts_good = km_orbit(good_lam, n_steps)
    err_good = np.linalg.norm(pts_good - XSTAR, axis=1)

    # BAD: lambda_n = 1/(n+2)^2 => sum lambda_n < infinity, so sum lambda_n(1-lambda_n) < infinity
    bad_lam = lambda n: 1.0 / (n + 2) ** 2
    pts_bad = km_orbit(bad_lam, n_steps)
    err_bad = np.linalg.norm(pts_bad - XSTAR, axis=1)

    ns = np.arange(n_steps + 1)
    ax = axes[0]
    ax.plot(ns, err_good, "o-", color="#2ca02c", ms=4,
             label=r"$\lambda_n=\frac{1}{n+2}$" + "\n" + r"$\sum \lambda_n(1-\lambda_n)=\infty$ (satisfied)")
    ax.plot(ns, err_bad, "s-", color="#d62728", ms=4,
             label=r"$\lambda_n=\frac{1}{(n+2)^2}$" + "\n" + r"$\sum \lambda_n(1-\lambda_n)<\infty$ (violated)")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel(r"iteration $n$")
    ax.set_ylabel(r"$\|x_n - x^*\|$")
    ax.set_title("Linear scale: the bad sequence stalls")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    ax = axes[1]
    cum_good = np.cumsum([good_lam(n) * (1 - good_lam(n)) for n in range(n_steps)])
    cum_bad = np.cumsum([bad_lam(n) * (1 - bad_lam(n)) for n in range(n_steps)])
    ax.plot(np.arange(1, n_steps + 1), cum_good, "o-", color="#2ca02c", ms=4,
             label=r"$\sum_{k=0}^{n} \lambda_k(1-\lambda_k)$, good seq.")
    ax.plot(np.arange(1, n_steps + 1), cum_bad, "s-", color="#d62728", ms=4,
             label=r"$\sum_{k=0}^{n} \lambda_k(1-\lambda_k)$, bad seq.")
    ax.set_xlabel(r"$n$")
    ax.set_ylabel("partial sum")
    ax.set_title(r"Partial sums of $\lambda_k(1-\lambda_k)$")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    fig.suptitle(r"Reich's condition $\sum \lambda_n(1-\lambda_n)=\infty$: satisfied vs. violated")
    fig.savefig("km_condition_violation.pdf")
    plt.close(fig)


# ----------------------------------------------------------------------
# Figure (d): perturbed iteration still converges to the same fixed point
# ----------------------------------------------------------------------
def fig_perturbed():
    fig, ax = plt.subplots(figsize=(6.4, 4.6))

    lam = 0.5
    n_steps = 14

    # unperturbed
    pts_clean = km_orbit(lambda n: lam, n_steps)
    err_clean = np.linalg.norm(pts_clean - XSTAR, axis=1)

    # perturbed: x_{n+1} = (1-lambda) x_n + lambda (T(x_n) + e_n),  e_n = (0.01/(n+1)) (1,0)
    x = X0.copy()
    pts_pert = [x.copy()]
    for n in range(n_steps):
        e_n = np.array([0.01 / (n + 1), 0.0])
        x = (1 - lam) * x + lam * (T(x) + e_n)
        pts_pert.append(x.copy())
    pts_pert = np.array(pts_pert)
    err_pert = np.linalg.norm(pts_pert - XSTAR, axis=1)

    ns = np.arange(n_steps + 1)
    ax.semilogy(ns, err_clean, "o-", color="#7f7f7f", ms=5,
                label=r"unperturbed, $\lambda_n\equiv 0.5$")
    ax.semilogy(ns, err_pert, "^-", color="#9467bd", ms=6,
                label=r"perturbed, $e_n=\frac{0.01}{n+1}(1,0)$")
    ax.set_xlabel(r"iteration $n$")
    ax.set_ylabel(r"$\|x_n - x^*\|$  (log scale)")
    ax.set_title("Perturbed KM iteration: same limit, small extra wobble")
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    fig.savefig("km_perturbed.pdf")
    plt.close(fig)


if __name__ == "__main__":
    fig_trajectories()
    fig_convergence_rate()
    fig_condition_violation()
    fig_perturbed()
    print("All figures written to", __file__)
