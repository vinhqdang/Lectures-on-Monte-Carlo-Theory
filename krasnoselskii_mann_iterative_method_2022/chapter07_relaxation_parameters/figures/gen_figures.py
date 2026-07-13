#!/usr/bin/env python3
"""
Figures for Chapter 7 -- Relaxation Parameters of the Krasnosel'skii-Mann Iteration.

(a) fig_convergence.pdf
    Running example (unit-disk projection, x0 = (3,4)): compares the KM
    iteration with a CONSTANT relaxation parameter lambda = 0.5 against the
    KM iteration driven by the (hindsight) optimal relaxation parameter
    lambda_hat_{p,n} of Eq. (7.9), evaluated at the true fixed point
    p = x* = (0.6, 0.8).

(b) fig_vi_diagram.pdf
    A geometric picture of a variational inequality problem (7.16):
    find x* in C such that <F(x*), y - x*> >= 0 for all y in C,
    for C = the closed unit disk and F(x) = x - q, the gradient of
    f(x) = (1/2)||x-q||^2.  The solution x* is exactly the projection of q
    onto C -- the same point (0.6, 0.8) used in the running example.

Run:  python3 gen_figures.py   (writes PDFs into this directory)
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------
# The operator T of the running example: metric projection onto the
# closed unit disk C = {x in R^2 : ||x|| <= 1}.
# ----------------------------------------------------------------------
def P_C(x):
    x = np.asarray(x, dtype=float)
    nrm = np.linalg.norm(x)
    return x if nrm <= 1.0 else x / nrm


def km_iteration(x0, lam_fn, T=P_C, n_iter=10):
    """x_{n+1} = (1 - lambda_n) x_n + lambda_n T(x_n)."""
    xs = [np.asarray(x0, dtype=float)]
    for n in range(n_iter):
        xn = xs[-1]
        lam = lam_fn(n, xn)
        xs.append((1 - lam) * xn + lam * T(xn))
    return xs


def lam_hat_p(xn, T, p):
    """Eq. (7.9): n-th optimal relaxation parameter w.r.t. known fixed point p."""
    Txn = T(xn)
    num = np.dot(xn - p, xn - p) - np.dot(Txn - p, Txn - p)
    den = 2 * np.dot(xn - Txn, xn - Txn)
    return 1.0 if den == 0 else 0.5 + num / den


def lam_approx(xn, T):
    """Eq. (7.12): the practical approximation, using T(xn) and T(T(xn))
    in place of the (unknown) fixed point p."""
    Txn = T(xn)
    T2xn = T(Txn)
    num = np.dot(xn - Txn, xn - Txn) - np.dot(Txn - T2xn, Txn - T2xn)
    diff = (xn - Txn) - (Txn - T2xn)
    den = 2 * np.dot(diff, diff)
    return 1.0 if den == 0 else 0.5 + num / den


x0 = np.array([3.0, 4.0])
xstar = np.array([0.6, 0.8])  # = x0 / ||x0||; the fixed point reached along this ray

# ----------------------------------------------------------------------
# Figure (a): constant relaxation vs. (hindsight) optimal relaxation
# ----------------------------------------------------------------------
xs_const = km_iteration(x0, lambda n, xn: 0.5, n_iter=10)
xs_opt = km_iteration(x0, lambda n, xn: lam_hat_p(xn, P_C, xstar), n_iter=10)

err_const = [np.linalg.norm(x - xstar) for x in xs_const]
err_opt = [np.linalg.norm(x - xstar) for x in xs_opt]

fig, ax = plt.subplots(figsize=(6.4, 4.2))
ns = np.arange(len(err_const))
ax.semilogy(ns, np.maximum(err_const, 1e-16), 'o-', color='#1f77b4', lw=1.8,
            label=r'constant $\lambda_n \equiv 0.5$')
ax.semilogy(np.arange(len(err_opt)), np.maximum(err_opt, 1e-16), 's-',
            color='#d62728', lw=1.8,
            label=r'optimal $\lambda_n=\widehat\lambda_{p,n}$  (Eq. 7.9)')
ax.set_xlabel(r'iteration $n$')
ax.set_ylabel(r'$\|x_n - x^\ast\|$  (log scale)')
ax.set_title("Running example: unit-disk projection, $x_0=(3,4)$")
ax.legend(loc='upper right', fontsize=9)
ax.grid(True, which='both', alpha=0.3)
ax.annotate('exact convergence\nalready at $n=1$', xy=(1, 1e-15), xytext=(3, 1e-6),
            fontsize=8, color='#d62728',
            arrowprops=dict(arrowstyle='->', color='#d62728'))
fig.tight_layout()
fig.savefig('figures/fig_convergence.pdf')
plt.close(fig)

# ----------------------------------------------------------------------
# Figure (b): geometric picture of a variational inequality problem
# ----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 6))
theta = np.linspace(0, 2 * np.pi, 200)
ax.plot(np.cos(theta), np.sin(theta), color='black', lw=1.5)
ax.fill(np.cos(theta), np.sin(theta), color='#cfe2f3', alpha=0.6, zorder=0)
ax.text(0, 0, r'$C$', fontsize=13, ha='center', va='center')

q = np.array([3.0, 4.0])
F_xstar = xstar - q  # F(x) = x - q, the gradient of (1/2)||x-q||^2

ax.plot(*q, 'k*', markersize=14)
ax.annotate(r'$q=(3,4)$', xy=q, xytext=(q[0] + 0.15, q[1] + 0.05), fontsize=11)
ax.plot(*xstar, 'ko', markersize=6)
ax.annotate(r'$x^\ast=(0.6,0.8)$', xy=xstar,
            xytext=(xstar[0] + 0.2, xstar[1] - 0.45), fontsize=11)

ax.plot([q[0], xstar[0]], [q[1], xstar[1]], 'k--', lw=0.8, alpha=0.6)

scale = 0.35
ax.annotate('', xy=xstar + scale * F_xstar, xytext=xstar,
            arrowprops=dict(arrowstyle='->', color='#d62728', lw=2))
ax.annotate(r'$F(x^\ast)=x^\ast-q$', xy=xstar + scale * F_xstar,
            xytext=(xstar[0] + scale * F_xstar[0] - 1.9,
                    xstar[1] + scale * F_xstar[1] - 0.15),
            color='#d62728', fontsize=11)

ys = np.array([[0.0, 0.0], [-0.8, 0.6], [1.0, 0.0], [-1.0, 0.0]])
for y in ys:
    ax.annotate('', xy=y, xytext=xstar,
                arrowprops=dict(arrowstyle='->', color='#2ca02c', lw=1.3, alpha=0.85))
    ax.plot(*y, 'g.', markersize=7)
ax.annotate(r'$y-x^\ast$ for sample $y\in C$', xy=(-0.9, 0.75), color='#2ca02c', fontsize=9)

ax.set_xlim(-1.8, 3.6)
ax.set_ylim(-1.8, 4.6)
ax.set_aspect('equal')
ax.set_title("Variational inequality:  " r"$\langle F(x^\ast), y-x^\ast\rangle \geq 0\ \ \forall y \in C$")
ax.text(-1.7, -1.55, r'$C=$ unit disk, $F(x)=x-q=\nabla\left(\frac{1}{2}\|x-q\|^2\right)$', fontsize=9)
ax.set_xticks([])
ax.set_yticks([])
for spine in ax.spines.values():
    spine.set_visible(False)
fig.tight_layout()
fig.savefig('figures/fig_vi_diagram.pdf')
plt.close(fig)

print("Wrote figures/fig_convergence.pdf and figures/fig_vi_diagram.pdf")
print()
print("Numeric check (running example):")
print(f"{'n':>2}  {'x_n (const 0.5)':>22}  {'err':>10} |  {'x_n (optimal)':>22}  {'err':>10}")
for n in range(6):
    xc, xo = xs_const[n], xs_opt[n]
    print(f"{n:>2}  ({xc[0]:6.4f},{xc[1]:6.4f})       {np.linalg.norm(xc-xstar):10.6f} | "
          f"({xo[0]:6.4f},{xo[1]:6.4f})       {np.linalg.norm(xo-xstar):10.6f}")
