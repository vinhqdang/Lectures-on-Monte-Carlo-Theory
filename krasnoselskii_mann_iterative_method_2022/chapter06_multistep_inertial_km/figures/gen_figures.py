#!/usr/bin/env python3
"""
Figure generator for Chapter 6 -- The Multi-step Inertial
Krasnosel'skii-Mann Iteration.

Produces (as vector PDF):
  fig_running_example.pdf   -- single-step vs. multi-step (s=2) inertial
                                KM on the running example (projection onto
                                the closed unit disk, x0 = (3,4)).
  fig_convergence_error.pdf -- semilog plot of the distance to the fixed
                                point (0.6, 0.8) vs. iteration number for
                                both schemes.
  fig_fixed_weights.pdf     -- the pre-specified ("does not depend on the
                                iterative sequence") exponential memory
                                weights w(lag) = (1-mu) mu^lag of Example
                                6.2 / algorithm (6.19)-(6.21), for a few
                                choices of mu -- illustrating a parameter
                                sequence that can be scheduled in advance.

All figures are saved directly into the figures/ directory as PDF.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------------------------------------------
# Running example: T = projection onto the closed unit disk in R^2
# ----------------------------------------------------------------

def T(x):
    """Projection onto the closed unit disk (nonexpansive)."""
    n = np.linalg.norm(x)
    return x / n if n > 1.0 else x.copy()


x0 = np.array([3.0, 4.0])
x_star = np.array([0.6, 0.8])  # = x0 / ||x0||, the fixed point the KM
                                 # iterates below are driven to.
LAMBDA = 0.5                    # constant KM averaging parameter lambda_n = 0.5


def single_step_inertial_km(a=0.10, n_iter=60):
    """Chapter-5 style single-step inertial KM (recap):
         y_n = x_n + a_n (x_n - x_{n-1})
         x_{n+1} = (1-lambda) y_n + lambda T(y_n)
    with a constant inertial parameter a_n = a.
    """
    xs = [x0.copy(), x0.copy()]  # x_{-1} = x_0
    for n in range(1, n_iter + 1):
        xn, xnm1 = xs[-1], xs[-2]
        yn = xn + a * (xn - xnm1)
        xnp1 = (1 - LAMBDA) * yn + LAMBDA * T(yn)
        xs.append(xnp1)
    return np.array(xs[1:])  # drop the duplicated x_{-1}


def multistep_inertial_km(theta1=0.13, theta2=0.01, n_iter=60):
    """Multi-step inertial KM (6.5) with s=2, S_n = {0,1}, b_{n,k}=a_{n,k}
    (Remark 6.4 case, so z_n = y_n):
         y_n = x_n + theta1 (x_n - x_{n-1}) + theta2 (x_{n-1} - x_{n-2})
         x_{n+1} = (1-lambda) y_n + lambda T(y_n)
    """
    xs = [x0.copy(), x0.copy(), x0.copy()]  # x_{-2}=x_{-1}=x_0
    for n in range(1, n_iter + 1):
        xn, xnm1, xnm2 = xs[-1], xs[-2], xs[-3]
        yn = xn + theta1 * (xn - xnm1) + theta2 * (xnm1 - xnm2)
        xnp1 = (1 - LAMBDA) * yn + LAMBDA * T(yn)
        xs.append(xnp1)
    return np.array(xs[2:])  # drop the two duplicated warm-up points


single = single_step_inertial_km()
multi = multistep_inertial_km()

err_single = np.linalg.norm(single - x_star, axis=1)
err_multi = np.linalg.norm(multi - x_star, axis=1)

tol = 1e-3


def first_below(err, tol):
    """Smallest n (1-indexed) after which the error stays below tol
    for every subsequent iterate (guards against non-monotonic overshoot)."""
    below = err < tol
    for i in range(len(below)):
        if below[i:].all():
            return i + 1
    return -1


n_single = first_below(err_single, tol)
n_multi = first_below(err_multi, tol)
print(f"single-step inertial KM reaches tol={tol}: n = {n_single}, "
      f"error = {err_single[n_single-1]:.3e}")
print(f"multi-step inertial KM  reaches tol={tol}: n = {n_multi}, "
      f"error = {err_multi[n_multi-1]:.3e}")
for n in range(1, 13):
    print(f"n={n:2d}: single=({single[n-1,0]:.6f},{single[n-1,1]:.6f}) "
          f"err={err_single[n-1]:.6f}   "
          f"multi=({multi[n-1,0]:.6f},{multi[n-1,1]:.6f}) "
          f"err={err_multi[n-1]:.6f}")

# ----------------------------------------------------------------
# Figure (a): trajectories in R^2 on the running example
# ----------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.4, 6.0))

circle = plt.Circle((0, 0), 1.0, fill=False, linestyle="--",
                     color="gray", linewidth=1.2, label=r"unit circle $\partial C$")
ax.add_patch(circle)

n_show = 10
ax.plot(single[:n_show, 0], single[:n_show, 1], "o-", color="#1f77b4",
         markersize=5, linewidth=1.4, label="single-step inertial KM (Ch.\\,5)")
ax.plot(multi[:n_show, 0], multi[:n_show, 1], "s-", color="#d62728",
         markersize=5, linewidth=1.4, label="multi-step inertial KM ($s=2$)")
ax.plot([x0[0]], [x0[1]], "k*", markersize=14, label=r"$x_0=(3,4)$")
ax.plot([x_star[0]], [x_star[1]], "kD", markersize=8,
        label=r"fixed point $(0.6,0.8)$")

ax.set_xlabel("$x^{(1)}$")
ax.set_ylabel("$x^{(2)}$")
ax.set_title("Running example: projection onto the unit disk")
ax.set_xlim(-0.5, 3.5)
ax.set_ylim(-0.5, 4.5)
ax.set_aspect("equal")
ax.legend(loc="upper right", fontsize=8)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("fig_running_example.pdf")
plt.close(fig)

# ----------------------------------------------------------------
# Figure (b): semilog convergence-error comparison
# ----------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.2, 4.6))
n_plot = 40
ax.semilogy(range(1, n_plot + 1), err_single[:n_plot], "o-",
            color="#1f77b4", markersize=4, linewidth=1.3,
            label="single-step inertial KM")
ax.semilogy(range(1, n_plot + 1), err_multi[:n_plot], "s-",
            color="#d62728", markersize=4, linewidth=1.3,
            label="multi-step inertial KM ($s=2$)")
ax.axhline(tol, color="gray", linestyle=":", linewidth=1.2,
           label=f"tolerance $=10^{{{int(np.log10(tol))}}}$")
ax.set_xlabel("iteration $n$")
ax.set_ylabel(r"$\|x_n - x^\star\|$")
ax.set_title("Convergence error vs. iteration")
ax.legend(fontsize=9)
ax.grid(alpha=0.3, which="both")
fig.tight_layout()
fig.savefig("fig_convergence_error.pdf")
plt.close(fig)

# ----------------------------------------------------------------
# Figure (c): fixed, pre-specified exponential memory weights
#   w(lag) = (1-mu) mu^lag   (Example 6.2, used in (6.19)-(6.21))
#   These do NOT depend on the iterative sequence {x_n}: they can be
#   tabulated once, in advance, for any n.
# ----------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.2, 4.6))
lags = np.arange(0, 16)
for mu, color in zip([0.3, 0.5, 0.7], ["#2ca02c", "#9467bd", "#ff7f0e"]):
    w = (1 - mu) * mu ** lags
    w[0] = 0  # k=0 term uses mu^n normalization separately in (6.20)-(6.21);
              # show only the k>=1 "recency-decay" weights here.
    markerline, stemlines, baseline = ax.stem(
        lags, w, basefmt=" ", label=fr"$\mu={mu}$")
    plt.setp(markerline, color=color, markersize=5)
    plt.setp(stemlines, color=color, linewidth=1.2)
ax.set_xlabel(r"lag $k$ (how many steps back, $n-k$)")
ax.set_ylabel(r"weight $(1-\mu)\,\mu^{\,k}$")
ax.set_title("Pre-specified inertial weights (Example 6.2 / Sec. 6.2)")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("fig_fixed_weights.pdf")
plt.close(fig)

print("Saved: fig_running_example.pdf, fig_convergence_error.pdf, "
      "fig_fixed_weights.pdf")
