"""
gen_figures.py -- Chapter 15: Fenchel-Rockafellar Duality
Bauschke & Combettes, "Convex Analysis and Monotone Operator Theory
in Hilbert Spaces", 2nd ed., CMS Books in Mathematics, Springer 2017.

Generates all figures for the Beamer slides using matplotlib (Agg backend).
Every figure is saved as a vector PDF into this "figures" directory.

Running numeric example used throughout the slides:

  Primal (no operator, Definition 15.10):
      f(x) = (1/2) x^2,   g(x) = (1/2)(x-4)^2       (H = R)
      minimize f(x) + g(x)   ->   x* = 2,  mu = 4

  Dual (Definition 15.10):
      minimize  f*(-u) + g*(u) = u^2 + 4u           ->  u* = -2, mu* = -4

  Primal (with operator L, Definition 15.19 / Theorem 15.23):
      f(x) = (1/2)||x||^2 on R^2,  g(y) = iota_{4}(y) on R,  L(x1,x2)=x1+x2
      minimize f(x) + g(Lx)  s.t. x1+x2 = 4          ->  x* = (2,2), mu = 4

  Dual:
      minimize f*(-L*v) + g*(v) = v^2 + 4v           ->  v* = -2, mu* = -4

Both formulations give the SAME numbers: mu = 4 = -mu*, confirming
strong duality (zero duality gap) via Theorem 15.23.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

FIGDIR = os.path.dirname(os.path.abspath(__file__))


def savefig(name):
    path = os.path.join(FIGDIR, name)
    plt.savefig(path, bbox_inches='tight')
    plt.close('all')
    print(f"  saved {path}")


# ---------------------------------------------------------------------------
# Figure 1: Fenchel duality (no operator) on the running example
#   Left:  primal objective f(x)+g(x) in x-space
#   Right: dual objective f*(-u)+g*(u) in u-space
# ---------------------------------------------------------------------------
def fig_primal_dual_quadratics():
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))

    x = np.linspace(-2, 6, 400)
    f = 0.5 * x**2
    g = 0.5 * (x - 4)**2
    s = f + g

    ax = axes[0]
    ax.plot(x, f, '--', color='royalblue', lw=1.8, label=r'$f(x)=\tfrac12 x^2$')
    ax.plot(x, g, '--', color='darkorange', lw=1.8, label=r'$g(x)=\tfrac12 (x-4)^2$')
    ax.plot(x, s, '-', color='black', lw=2.4, label=r'$f(x)+g(x)$')
    ax.axvline(2.0, color='green', lw=1, ls=':')
    ax.plot([2.0], [4.0], 'o', color='green', ms=8, zorder=5)
    ax.annotate(r'$x^\star=2$' + '\n' + r'$\mu = 4$',
                xy=(2.0, 4.0), xytext=(2.6, 9.0),
                arrowprops=dict(arrowstyle='->', color='green'),
                color='green', fontsize=11)
    ax.set_xlabel(r'$x$')
    ax.set_ylabel('value')
    ax.set_title('Primal: minimize $f(x)+g(x)$')
    ax.legend(loc='upper center', fontsize=9)
    ax.set_ylim(-1, 20)

    u = np.linspace(-6, 2, 400)
    h = u**2 + 4 * u  # f*(-u) + g*(u)

    ax2 = axes[1]
    ax2.plot(u, h, '-', color='purple', lw=2.4,
             label=r'$f^*(-u)+g^*(u) = u^2+4u$')
    ax2.axvline(-2.0, color='green', lw=1, ls=':')
    ax2.plot([-2.0], [-4.0], 'o', color='green', ms=8, zorder=5)
    ax2.annotate(r'$u^\star=-2$' + '\n' + r'$\mu^* = -4$',
                 xy=(-2.0, -4.0), xytext=(-5.5, 6.0),
                 arrowprops=dict(arrowstyle='->', color='green'),
                 color='green', fontsize=11)
    ax2.axhline(-4.0, color='gray', lw=0.8, ls='--')
    ax2.set_xlabel(r'$u$')
    ax2.set_ylabel('value')
    ax2.set_title('Dual: minimize $f^*(-u)+g^*(u)$')
    ax2.legend(loc='upper center', fontsize=9)

    fig.suptitle(r'$\mu = 4 = -\mu^*$: strong duality, zero gap', y=1.03)
    savefig('fig_primal_dual_quadratics.pdf')


# ---------------------------------------------------------------------------
# Figure 2: The same example written with an operator L (Fenchel-Rockafellar)
#   minimize (1/2)||x||^2  s.t.  x1+x2 = 4
# ---------------------------------------------------------------------------
def fig_constrained_qp():
    fig, ax = plt.subplots(figsize=(5.4, 5.0))

    x1 = np.linspace(-1, 5, 400)
    x2 = np.linspace(-1, 5, 400)
    X1, X2 = np.meshgrid(x1, x2)
    F = 0.5 * (X1**2 + X2**2)

    levels = [0.5, 1, 2, 4, 6, 8, 10, 12]
    cs = ax.contour(X1, X2, F, levels=levels, cmap='Blues', linewidths=1.0)
    ax.clabel(cs, inline=True, fontsize=7, fmt='%.1f')

    line_x = np.linspace(-1, 5, 100)
    line_y = 4 - line_x
    ax.plot(line_x, line_y, color='darkorange', lw=2.5,
            label=r'constraint $x_1+x_2=4$  (i.e. $Lx=4$)')

    ax.plot([2], [2], 'o', color='green', ms=10, zorder=5)
    ax.annotate(r'$x^\star=(2,2)$' + '\n' + r'$f(x^\star)=4$',
                xy=(2, 2), xytext=(2.7, 0.2),
                arrowprops=dict(arrowstyle='->', color='green'),
                color='green', fontsize=11)

    # gradient of f at x* is normal to the level circle and parallel to (1,1)
    ax.annotate('', xy=(2.9, 2.9), xytext=(2, 2),
                arrowprops=dict(arrowstyle='->', color='crimson', lw=1.8))
    ax.text(3.0, 2.65, r'$\nabla f(x^\star)\parallel(1,1)$', color='crimson',
            fontsize=9)

    ax.set_xlabel(r'$x_1$')
    ax.set_ylabel(r'$x_2$')
    ax.set_title(r'$\min\ \tfrac12\|x\|^2$ s.t. $x_1+x_2=4$   ($f+g\circ L$ form)')
    ax.legend(loc='upper right', fontsize=8)
    ax.set_xlim(-1, 5)
    ax.set_ylim(-1, 5)
    ax.set_aspect('equal')
    savefig('fig_constrained_qp.pdf')


# ---------------------------------------------------------------------------
# Figure 3: Weak duality / duality gap -- general schematic
# ---------------------------------------------------------------------------
def fig_duality_gap_diagram():
    fig, ax = plt.subplots(figsize=(7.5, 2.6))

    ax.axhline(0, color='black', lw=1)

    # weak duality picture: -mu* <= mu always
    ax.plot([-1.2], [0], marker='|', ms=20, color='purple')
    ax.text(-1.2, 0.25, r'$-\mu^{*}$', ha='center', color='purple', fontsize=12)

    ax.plot([1.2], [0], marker='|', ms=20, color='royalblue')
    ax.text(1.2, 0.25, r'$\mu$', ha='center', color='royalblue', fontsize=12)

    ax.annotate('', xy=(1.1, 0), xytext=(-1.1, 0),
                arrowprops=dict(arrowstyle='-', color='gray', lw=8, alpha=0.35))
    ax.text(0, -0.35, r'always true: $\mu \geq -\mu^{*}$   (weak duality, Prop.\ 15.9 / 15.18)',
            ha='center', fontsize=10)

    ax.plot([3.6], [0], marker='|', ms=20, color='purple')
    ax.plot([3.6], [0], marker='|', ms=20, color='royalblue', mfc='none')
    ax.text(3.6, 0.25, r'$\mu = -\mu^{*}$', ha='center', color='green', fontsize=12,
            fontweight='bold')
    ax.text(3.6, -0.35, r'our example: $4 = -(-4)$' + '\n' + r'$\Delta(f,g,L)=\mu+\mu^*=0$',
            ha='center', fontsize=9.5, color='green')

    ax.set_xlim(-2.6, 5.4)
    ax.set_ylim(-0.9, 0.7)
    ax.axis('off')
    ax.set_title('Duality gap $\\Delta(f,g,L)$: weak duality vs.\\ our zero-gap example',
                  fontsize=11)
    savefig('fig_duality_gap_diagram.pdf')


# ---------------------------------------------------------------------------
# Figure 4: von Neumann minimax numeric example (Corollary 15.30, Section 15.5)
#   2x2 zero-sum game, payoff matrix A = [[1,-1],[-1,1]]
#   min_x max_v x^T A v  = max_v min_x x^T A v = 0
#   at x = v = (1/2, 1/2)
# ---------------------------------------------------------------------------
def fig_von_neumann_minimax():
    A = np.array([[1.0, -1.0], [-1.0, 1.0]])

    # sweep mixed strategies x=(t,1-t) and v=(s,1-s)
    t = np.linspace(0, 1, 200)
    s = np.linspace(0, 1, 200)

    # phi(x) = max_v x^T A v  (row player's worst case for fixed x)
    phi = []
    for tt in t:
        x = np.array([tt, 1 - tt])
        vals = [x @ A @ np.array([ss, 1 - ss]) for ss in [0.0, 1.0]]
        phi.append(max(vals))
    phi = np.array(phi)

    # psi(v) = min_x x^T A v  (column player's worst case for fixed v)
    psi = []
    for ss in s:
        v = np.array([ss, 1 - ss])
        vals = [np.array([xx, 1 - xx]) @ A @ v for xx in [0.0, 1.0]]
        psi.append(min(vals))
    psi = np.array(psi)

    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    ax.plot(t, phi, color='royalblue', lw=2.2,
            label=r'$\varphi(x)=\max_{v\in D}\langle Lx\mid v\rangle$')
    ax.plot(s, psi, color='darkorange', lw=2.2,
            label=r'$\psi(v)=\min_{x\in C}\langle Lx\mid v\rangle$')

    tstar = 0.5
    ax.plot([tstar], [0.0], 'o', color='green', ms=9, zorder=5)
    ax.annotate(r'$x^\star=v^\star=(\tfrac12,\tfrac12)$' + '\n' + 'value $=0$',
                xy=(tstar, 0.0), xytext=(0.55, 0.55),
                arrowprops=dict(arrowstyle='->', color='green'), color='green',
                fontsize=10)

    ax.axhline(0, color='gray', lw=0.7, ls='--')
    ax.set_xlabel(r'mixing weight on first pure strategy')
    ax.set_ylabel('payoff')
    ax.set_title(r'von Neumann: $\min_{x\in C}\max_{v\in D}\langle Lx\mid v\rangle'
                 r'=\max_{v\in D}\min_{x\in C}\langle Lx\mid v\rangle$')
    ax.legend(loc='upper right', fontsize=9)
    savefig('fig_von_neumann_minimax.pdf')


if __name__ == '__main__':
    fig_primal_dual_quadratics()
    fig_constrained_qp()
    fig_duality_gap_diagram()
    fig_von_neumann_minimax()
    print("All figures generated.")
