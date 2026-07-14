#!/usr/bin/env python3
"""
Figure generator for Chapter 19 (Duality in Convex Optimization) slides.

Both problems below are solved TWICE: once by a generic numerical solver
(scipy.optimize) applied directly to the stated primal / dual problem, and
once by the closed-form recipe derived in the slides (Prop. 19.5 / Cor.
19.23 for the equality-constrained example; Prop. 19.25 / Cor. 19.30 for
the inequality-constrained example).  The two always agree, and in both
cases the primal optimal value mu and the dual optimal value mu* satisfy
mu = -mu* (zero duality gap), exactly as in Theorem 19.1 / Corollary 19.19.

Generates, as vector PDFs in the current directory:

  1. fig_equality_duality.pdf
       Running example of Section 19.3 (Corollary 19.23): the best
       approximation / least-norm problem
           minimize_{x in R^2}  (1/2)||x-z||^2   s.t.  <x|a> = b
       (projecting z onto a line) next to its scalar dual
           minimize_{nu in R}   (1/2)||a||^2 nu^2 + nu(b - <z|a>).
       Left panel: the plane, the line, z, and the projection x-bar.
       Right panel: the dual objective h(nu), its minimizer nu-bar, and
       the matching optimal values (mu = -mu*).

  2. fig_inequality_duality.pdf
       Running example of Section 19.4 (Corollary 19.30): the
       inequality-constrained problem
           minimize_{x in R^2}  (1/2)||x||^2   s.t.  <a|x> >= b
       (nearest point to the origin in a half-plane) next to its scalar
       dual (nu >= 0)
           minimize_{nu >= 0}   (1/2)||a||^2 nu^2 - b nu.
       Left panel: feasible half-plane, contours of the objective, and
       the constrained optimum x-bar.  Right panel: the dual objective
       d(nu) restricted to nu >= 0, its minimizer nu-bar, and the
       matching optimal values (mu = -mu*).

Plain python3 + matplotlib (Agg backend) + numpy + scipy.optimize only.
No LaTeX rendering dependency beyond matplotlib's own mathtext.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize, minimize_scalar

plt.rcParams.update({
    "font.size": 11,
    "axes.grid": True,
    "grid.alpha": 0.3,
})


def fig_equality_duality():
    # ---- data for the running example (Corollary 19.23, m=1) ----
    z = np.array([3.0, 1.0])
    a = np.array([1.0, 1.0])
    b = 2.0

    # Primal:  minimize (1/2)||x-z||^2  s.t.  <a|x> = b   -- solved numerically
    f_primal = lambda x: 0.5 * np.sum((x - z) ** 2)
    cons = ({"type": "eq", "fun": lambda x: a @ x - b},)
    res_p = minimize(f_primal, x0=np.array([0.0, 0.0]), constraints=cons)
    x_bar_num = res_p.x
    mu_num = res_p.fun

    # Closed form (recipe of Prop. 19.4/19.5 + Cor. 19.23):
    #   nu-bar = (<z|a> - b) / ||a||^2,   x-bar = z - nu-bar * a
    nu_bar = (z @ a - b) / (a @ a)
    x_bar = z - nu_bar * a
    mu = f_primal(x_bar)

    # Dual:  minimize_{nu in R} h(nu) = (1/2)||a||^2 nu^2 + nu(b - <z|a>)
    h = lambda nu: 0.5 * (a @ a) * nu ** 2 + nu * (b - z @ a)
    res_d = minimize_scalar(h)
    nu_num = res_d.x
    mustar_num = res_d.fun

    print("=== Equality-constrained running example ===")
    print(f"  numeric primal:  x_bar={x_bar_num}, mu={mu_num:.6f}")
    print(f"  closed-form   :  x_bar={x_bar}, mu={mu:.6f}, nu_bar={nu_bar:.6f}")
    print(f"  numeric dual  :  nu_bar={nu_num:.6f}, mu*={mustar_num:.6f}")
    print(f"  mu + mu* = {mu + mustar_num:.8f}  (should be 0)")

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6))

    # --- left panel: primal geometry ---
    ax = axes[0]
    t = np.linspace(-2, 6, 200)
    # line <a|x> = b, i.e. x1 + x2 = 2  ->  x2 = b - x1
    line_x1 = t
    line_x2 = b - t
    ax.plot(line_x1, line_x2, color="tab:blue", lw=2.5,
            label=r"constraint $\langle a\,|\,x\rangle = b$")
    ax.plot(*z, "o", color="tab:green", ms=9, zorder=5)
    ax.annotate(r"$z=(3,1)$", xy=tuple(z), xytext=(z[0] + 0.3, z[1] + 0.5))
    ax.plot(*x_bar, "s", color="tab:red", ms=9, zorder=5)
    ax.annotate(r"$\bar x = P_{\{a\}^\top=b}(z) = (%.1f,%.1f)$" % tuple(x_bar),
                xy=tuple(x_bar), xytext=(x_bar[0] - 2.6, x_bar[1] - 1.6),
                arrowprops=dict(arrowstyle="->", color="black"))
    ax.plot([z[0], x_bar[0]], [z[1], x_bar[1]], "--", color="gray", lw=1.5)
    # a few level circles of (1/2)||x-z||^2
    theta = np.linspace(0, 2 * np.pi, 200)
    for r in [np.sqrt(2 * mu), np.sqrt(2 * mu) * 1.6]:
        ax.plot(z[0] + r * np.cos(theta), z[1] + r * np.sin(theta),
                color="tab:green", lw=0.8, alpha=0.5)
    ax.set_xlim(-2, 6)
    ax.set_ylim(-3, 5)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.set_title("Primal: best approximation to $z$\non the line, $\\mu=%.1f$" % mu)
    ax.legend(loc="lower left", fontsize=9)

    # --- right panel: dual function ---
    ax = axes[1]
    nus = np.linspace(-2, 4, 300)
    ax.plot(nus, h(nus), color="tab:purple", lw=2.5,
            label=r"$h(\nu)=\frac{1}{2}\|a\|^2\nu^2+\nu(b-\langle z\,|\,a\rangle)$")
    ax.plot([nu_bar], [mu], "o", color="tab:red", ms=0)  # placeholder, invisible
    ax.plot([nu_bar], [h(nu_bar)], "o", color="tab:red", ms=9, zorder=5)
    ax.annotate(r"$\bar\nu=%.1f,\ \mu^*=%.1f$" % (nu_bar, h(nu_bar)),
                xy=(nu_bar, h(nu_bar)), xytext=(nu_bar + 0.4, h(nu_bar) + 3),
                arrowprops=dict(arrowstyle="->", color="black"))
    ax.axhline(mu, color="tab:green", ls="--", lw=1.5,
               label=r"$\mu=%.1f$ (primal optimal value)" % mu)
    ax.set_xlabel(r"$\nu$")
    ax.set_ylabel(r"$h(\nu)$")
    ax.set_title(r"Dual: $\mu=-\mu^*$  (%.1f $= -(%.1f)$)" % (mu, h(nu_bar)))
    ax.legend(loc="upper center", fontsize=8.5)

    fig.suptitle("Running example (Cor. 19.23): equality-constrained "
                 "least-norm problem and its dual")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig("fig_equality_duality.pdf")
    plt.close(fig)


def fig_inequality_duality():
    # ---- data for the running example (Cor. 19.30, single inequality) ----
    a = np.array([3.0, 4.0])
    b = 10.0

    # Primal: minimize (1/2)||x||^2  s.t.  <a|x> >= b   -- solved numerically
    f_primal = lambda x: 0.5 * np.sum(x ** 2)
    cons = ({"type": "ineq", "fun": lambda x: a @ x - b},)
    res_p = minimize(f_primal, x0=np.array([1.0, 1.0]), constraints=cons)
    x_bar_num = res_p.x
    mu_num = res_p.fun

    # Closed form: x-bar = (b/||a||^2) a   (projection of 0 onto the
    # active hyperplane <a|x>=b)
    x_bar = (b / (a @ a)) * a
    mu = f_primal(x_bar)

    # Dual: minimize_{nu>=0} d(nu) = (1/2)||a||^2 nu^2 - b nu
    d = lambda nu: 0.5 * (a @ a) * nu ** 2 - b * nu
    res_d = minimize_scalar(d, bounds=(0, 10), method="bounded")
    nu_num = res_d.x
    mustar_num = res_d.fun
    nu_bar = b / (a @ a)  # closed form:  d'(nu)=||a||^2 nu - b = 0

    print("=== Inequality-constrained running example ===")
    print(f"  numeric primal:  x_bar={x_bar_num}, mu={mu_num:.6f}")
    print(f"  closed-form   :  x_bar={x_bar}, mu={mu:.6f}")
    print(f"  numeric dual  :  nu_bar={nu_num:.6f}, mu*={mustar_num:.6f}")
    print(f"  closed-form   :  nu_bar={nu_bar:.6f}, mu*={d(nu_bar):.6f}")
    print(f"  mu + mu* = {mu + d(nu_bar):.8f}  (should be 0)")
    print(f"  complementary slackness: nu_bar*(b-<a|x_bar>) = "
          f"{nu_bar * (b - a @ x_bar):.8f} (should be 0)")

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6))

    # --- left panel: primal geometry ---
    ax = axes[0]
    x1 = np.linspace(-1, 5, 300)
    x2 = np.linspace(-1, 5, 300)
    X1, X2 = np.meshgrid(x1, x2)
    F = 0.5 * (X1 ** 2 + X2 ** 2)
    cs = ax.contour(X1, X2, F, levels=[0.5, 1, 2, 3, 4, 5], colors="tab:green",
                     linewidths=0.9, alpha=0.7)
    ax.clabel(cs, inline=True, fontsize=7, fmt=r"%.1f")
    # feasible half-plane 3x1+4x2 >= 10
    line_x2 = (b - 3 * x1) / 4
    ax.plot(x1, line_x2, color="tab:blue", lw=2.5,
            label=r"boundary $\langle a\,|\,x\rangle = b$")
    ax.fill_between(x1, line_x2, 5, color="tab:blue", alpha=0.08,
                     label=r"feasible set $\langle a\,|\,x\rangle \geq b$")
    ax.plot(0, 0, "x", color="black", ms=9, mew=2, zorder=5)
    ax.annotate("origin (unconstrained min)", xy=(0, 0), xytext=(-0.9, -0.9))
    ax.plot(*x_bar, "s", color="tab:red", ms=9, zorder=5)
    ax.annotate(r"$\bar x=(%.1f,%.1f)$" % tuple(x_bar),
                xy=tuple(x_bar), xytext=(x_bar[0] + 0.3, x_bar[1] - 1.2),
                arrowprops=dict(arrowstyle="->", color="black"))
    ax.set_xlim(-1, 5)
    ax.set_ylim(-1, 5)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.set_title("Primal: nearest point to $0$\nin a half-plane, $\\mu=%.1f$" % mu)
    ax.legend(loc="upper right", fontsize=8)

    # --- right panel: dual function on nu >= 0 ---
    ax = axes[1]
    nus = np.linspace(0, 1.0, 300)
    ax.plot(nus, d(nus), color="tab:purple", lw=2.5,
            label=r"$d(\nu)=\frac{1}{2}\|a\|^2\nu^2-b\nu,\ \ \nu\geq 0$")
    ax.plot([nu_bar], [d(nu_bar)], "o", color="tab:red", ms=9, zorder=5)
    ax.annotate(r"$\bar\nu=%.2f,\ \mu^*=%.1f$" % (nu_bar, d(nu_bar)),
                xy=(nu_bar, d(nu_bar)), xytext=(nu_bar + 0.15, d(nu_bar) + 1.2),
                arrowprops=dict(arrowstyle="->", color="black"))
    ax.axhline(mu, color="tab:green", ls="--", lw=1.5,
               label=r"$\mu=%.1f$ (primal optimal value)" % mu)
    ax.axvline(0, color="gray", lw=1)
    ax.set_xlabel(r"$\nu$")
    ax.set_ylabel(r"$d(\nu)$")
    ax.set_title(r"Dual: $\mu=-\mu^*$  (%.1f $= -(%.1f)$)" % (mu, d(nu_bar)))
    ax.legend(loc="upper center", fontsize=8.5)

    fig.suptitle("Running example (Cor. 19.30): inequality-constrained "
                 "problem and its dual")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig("fig_inequality_duality.pdf")
    plt.close(fig)


if __name__ == "__main__":
    fig_equality_duality()
    fig_inequality_duality()
    print("\nFigures written:")
    print("  fig_equality_duality.pdf")
    print("  fig_inequality_duality.pdf")
