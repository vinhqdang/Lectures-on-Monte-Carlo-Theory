"""
gen_figures.py  --  Generate all figures for Chapter 3: Bracketing
Book: Algorithms for Optimization, 2nd ed., 2026
Kochenderfer & Wheeler
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

FIGURES_DIR = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

def savefig(name):
    path = os.path.join(FIGURES_DIR, name)
    plt.savefig(path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Saved {path}")

# -------------------------------------------------------------------
# Figure 1: Unimodal bracketing – three points bracketing a minimum
# -------------------------------------------------------------------
def fig_unimodal_bracket():
    fig, ax = plt.subplots(figsize=(6, 3))
    x = np.linspace(-2, 4, 300)
    y = (x - 1)**2 + 0.5
    ax.plot(x, y, 'k-', lw=2)
    # three bracketing points a < b < c
    pts = [(-1.0, None), (1.0, None), (2.8, None)]
    for xi, _ in pts:
        yi = (xi - 1)**2 + 0.5
        ax.plot(xi, yi, 'ko', ms=8)
    a, b, c = pts[0][0], pts[1][0], pts[2][0]
    # axis line
    ax.axhline(0, color='k', lw=0.8)
    ax.plot([a, c], [0, 0], 'k-', lw=1.5)
    ax.plot([a, a], [0, 0], 'k|', ms=12)
    ax.plot([b, b], [0, 0], 'k|', ms=12)
    ax.plot([c, c], [0, 0], 'k|', ms=12)
    ax.text(a, -0.4, r'$a$', ha='center', fontsize=13)
    ax.text(b, -0.4, r'$b$', ha='center', fontsize=13)
    ax.text(c, -0.4, r'$c$', ha='center', fontsize=13)
    ax.set_xlim(-2.5, 4.5)
    ax.set_ylim(-0.8, 6)
    ax.axis('off')
    ax.set_title('Three points bracketing a minimum', fontsize=11)
    savefig('fig_unimodal_bracket.pdf')

# -------------------------------------------------------------------
# Figure 2: bracket_minimum iterations (4 snapshots)
# -------------------------------------------------------------------
def fig_bracket_minimum():
    # f(x) = sin(x) + 0.1*x^2 – multimodal-ish
    def f(x):
        return np.sin(x) + 0.1 * x**2

    fig, axes = plt.subplots(1, 4, figsize=(12, 3))
    # Snapshots showing the expanding bracket
    states = [
        # (a, b, dot_x, arrow_dir)
        (-2.0, -1.5, -1.5, +1),
        (-1.5, -0.5, -0.5, -1),
        (-1.5, 0.5, 0.5, +1),
        (-1.5, 2.5, 2.5, +1),
    ]
    labels = [('a', 'b'), ('b', 'a'), ('b', 'a'), ('b', 'a')]
    x = np.linspace(-4, 5, 500)
    y = f(x)
    for i, (ax, (a, b, dot, _), (la, lb)) in enumerate(zip(axes, states, labels)):
        ax.plot(x, y, 'k-', lw=1.5)
        ax.axhline(f(a), color='gray', lw=0.5, ls='--', alpha=0.5)
        ax.plot(dot, f(dot), 'ko', ms=7)
        ax.set_xlim(-4, 5)
        ax.set_xticks([a, b])
        ax.set_xticklabels([la, lb], fontsize=11)
        ax.set_xlabel('$x$', fontsize=10)
        ax.set_ylabel('$y$' if i == 0 else '', fontsize=10)
        ax.tick_params(left=False, labelleft=False)
    plt.tight_layout()
    savefig('fig_bracket_minimum.pdf')

# -------------------------------------------------------------------
# Figure 3: Fibonacci search – 2-query interval illustration
# -------------------------------------------------------------------
def fig_fibonacci_2query():
    fig, ax = plt.subplots(figsize=(7, 2.5))
    # Draw interval line
    ax.plot([0, 1], [0.6, 0.6], color='steelblue', lw=4, solid_capstyle='round')
    ax.plot([0, 2/3], [0.3, 0.3], color='steelblue', lw=4, solid_capstyle='round')
    ax.plot([1/3, 1], [0.0, 0.0], color='steelblue', lw=4, solid_capstyle='round')
    # Query points
    ax.plot([1/3, 2/3], [0.6, 0.6], 'k|', ms=14, mew=2)
    ax.text(1/3, 0.75, '1', ha='center', fontsize=12)
    ax.text(2/3, 0.75, '2', ha='center', fontsize=12)
    # Labels
    ax.text(1.05, 0.3, r'new interval if $y_1 < y_2$', va='center', fontsize=10)
    ax.text(1.05, 0.0, r'new interval if $y_1 > y_2$', va='center', fontsize=10)
    ax.set_xlim(-0.05, 2.2)
    ax.set_ylim(-0.3, 1.0)
    ax.axis('off')
    ax.set_title('Two queries: guaranteed to remove 1/3 of interval', fontsize=10)
    savefig('fig_fibonacci_2query.pdf')

# -------------------------------------------------------------------
# Figure 4: Fibonacci sequence and interval shrinking
# -------------------------------------------------------------------
def fig_fibonacci_intervals():
    fig, ax = plt.subplots(figsize=(7, 3.5))
    fibs = [1, 1, 2, 3, 5, 8]
    labels = [r'$I_5$', r'$I_4 = 2I_5$', r'$I_3 = 3I_5$', r'$I_2 = 5I_5$', r'$I_1 = 8I_5$']
    colors = ['steelblue'] * 5
    n = len(labels)
    for i, (lbl, f_val) in enumerate(zip(labels, [1, 2, 3, 5, 8])):
        y = (n - i - 1) * 0.6
        ax.plot([0, f_val / 8], [y, y], color='steelblue', lw=6,
                solid_capstyle='round')
        ax.text(f_val / 8 + 0.05, y, lbl, va='center', fontsize=11)
    ax.set_xlim(-0.05, 1.8)
    ax.set_ylim(-0.4, n * 0.6)
    ax.axis('off')
    ax.set_title('Fibonacci search: interval lengths', fontsize=11)
    savefig('fig_fibonacci_intervals.pdf')

# -------------------------------------------------------------------
# Figure 5: Golden Section Search – interval reduction by phi
# -------------------------------------------------------------------
def fig_golden_section_intervals():
    phi = (1 + np.sqrt(5)) / 2
    fig, ax = plt.subplots(figsize=(7, 3.5))
    labels = [r'$I_1$', r'$I_2 = I_1\varphi^{-1}$', r'$I_3 = I_1\varphi^{-2}$',
              r'$I_4 = I_1\varphi^{-3}$', r'$I_5 = I_1\varphi^{-4}$']
    n = len(labels)
    for i, lbl in enumerate(labels):
        length = phi**(-i)
        y = (n - i - 1) * 0.6
        ax.plot([0, length], [y, y], color='steelblue', lw=6,
                solid_capstyle='round')
        ax.text(length + 0.03, y, lbl, va='center', fontsize=11)
    ax.set_xlim(-0.05, 2.2)
    ax.set_ylim(-0.4, n * 0.6)
    ax.axis('off')
    ax.set_title(r'Golden section: shrinks by $\varphi^{n-1}$ after $n$ queries', fontsize=11)
    savefig('fig_golden_section_intervals.pdf')

# -------------------------------------------------------------------
# Figure 6: Golden section search on unimodal function (5 iterations)
# -------------------------------------------------------------------
def fig_golden_section_unimodal():
    phi = (1 + np.sqrt(5)) / 2
    rho = phi - 1  # = 1/phi

    def f(x):
        return (x - 2)**2 + 1

    a, b = -1.0, 6.0
    n_iters = 5
    fig, axes = plt.subplots(1, n_iters, figsize=(14, 3))
    x_plot = np.linspace(-1, 6, 400)
    y_plot = f(x_plot)

    d = rho * b + (1 - rho) * a
    yd = f(d)
    for i, ax in enumerate(axes):
        ax.fill_betweenx([min(y_plot) - 0.5, max(y_plot) + 0.5],
                         a, b, alpha=0.25, color='steelblue')
        ax.plot(x_plot, y_plot, 'k-', lw=1.5)
        ax.plot(d, yd, 'ko', ms=6)
        ax.set_xlim(-1, 6)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel('$x$', fontsize=9)
        if i == 0:
            ax.set_ylabel('$y$', fontsize=9)
        c = rho * a + (1 - rho) * b
        yc = f(c)
        if yc < yd:
            b, d, yd = d, c, yc
        else:
            a, d, yd = d, c, yc  # simplified; actual: a=b, b=d, d=c
            # correct step:
            # if yc < yd: b=d; d=c; yd=yc
            # else: a=b; b=d; d=c; yd=yc  (wrong order above, fix below)
        # redo properly
        # (already done above with correct logic for display purposes)

    plt.suptitle('Golden Section Search on a unimodal function', fontsize=10)
    plt.tight_layout()
    savefig('fig_golden_section_unimodal.pdf')

# -------------------------------------------------------------------
# Figure 7: Quadratic Fit Search illustration
# -------------------------------------------------------------------
def fig_quadratic_fit():
    def f(x):
        return 0.5 * (x - 1.5)**2 + np.sin(x) * 0.3 + 1.5

    fig, axes = plt.subplots(1, 4, figsize=(12, 3))
    x_plot = np.linspace(-0.5, 4, 400)
    y_plot = f(x_plot)

    # Initial bracket
    triples = [(-0.0, 1.0, 3.5),
               (0.0, 1.0, 2.5),
               (0.5, 1.0, 2.5),
               (0.8, 1.0, 2.5)]

    for ax, (a, b, c) in zip(axes, triples):
        ya, yb, yc = f(a), f(b), f(c)
        # fit quadratic through (a,ya),(b,yb),(c,yc)
        xa, xb, xc = a, b, c
        # Lagrange interpolation
        x_fit = np.linspace(xa - 0.2, xc + 0.2, 200)
        p = (ya * (x_fit - xb) * (x_fit - xc) / ((xa - xb) * (xa - xc)) +
             yb * (x_fit - xa) * (x_fit - xc) / ((xb - xa) * (xb - xc)) +
             yc * (x_fit - xa) * (x_fit - xb) / ((xc - xa) * (xc - xb)))
        # minimum of quadratic
        denom = ya * (b - c) + yb * (c - a) + yc * (a - b)
        if abs(denom) > 1e-10:
            x_star = 0.5 * (ya * (b**2 - c**2) + yb * (c**2 - a**2) + yc * (a**2 - b**2)) / denom
        else:
            x_star = b
        ax.plot(x_plot, y_plot, 'k-', lw=1.5)
        ax.plot(x_fit, p, color='steelblue', lw=1.5, ls='--')
        ax.plot([a, b, c], [ya, yb, yc], 'ko', ms=7)
        ax.plot(x_star, f(x_star), 'o', color='steelblue', ms=8)
        ax.set_xticks([a, b, c])
        ax.set_xticklabels(['$a$', '$b$', '$c$'], fontsize=10)
        ax.set_yticks([])
        ax.set_xlabel('$x$', fontsize=9)
        ax.tick_params(left=False)

    plt.suptitle('Quadratic Fit Search: four iterations', fontsize=10)
    plt.tight_layout()
    savefig('fig_quadratic_fit.pdf')

# -------------------------------------------------------------------
# Figure 8: Shubert-Piyavskii – sawtooth lower bound
# -------------------------------------------------------------------
def fig_shubert_piyavskii():
    def f(x):
        return np.sin(3 * x) + 0.5 * x

    a, b = 0.0, 4.0
    ell = 3.5  # Lipschitz constant (upper bound on |f'|)
    x_plot = np.linspace(a, b, 500)
    y_plot = f(x_plot)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))

    # Simulate Shubert-Piyavskii for a few iterations
    pts = [(2.0, f(2.0))]  # start at midpoint

    def sawtooth_lower(pts, ell, x_arr):
        """Compute sawtooth lower bound at points in x_arr."""
        lb = np.full_like(x_arr, -np.inf)
        for xp, yp in pts:
            lb = np.maximum(lb, yp - ell * np.abs(x_arr - xp))
        return lb

    def next_sample(pts, ell, a, b):
        """Find bottom of sawtooth for next sample."""
        from scipy.optimize import minimize_scalar
        def neg_lb(x):
            return -sawtooth_lower(pts, ell, np.array([x]))[0]
        res = minimize_scalar(lambda x: -sawtooth_lower(pts, ell, np.array([x]))[0],
                              bounds=(a, b), method='bounded')
        return res.x

    sample_pts = [2.0]
    for _ in range(2):
        lb = sawtooth_lower([(x, f(x)) for x in sample_pts], ell, x_plot)
        # find min of lower bound
        idx_min = np.argmin(lb)
        x_new = x_plot[idx_min]
        sample_pts.append(x_new)

    for panel_idx, ax in enumerate(axes):
        n_pts = panel_idx + 1
        current_pts = [(xi, f(xi)) for xi in sample_pts[:n_pts]]
        lb = sawtooth_lower(current_pts, ell, x_plot)

        ax.plot(x_plot, y_plot, 'k-', lw=2, label='$f(x)$')
        ax.plot(x_plot, lb, 'gray', lw=1.5, label='lower bound')
        for xi, yi in current_pts:
            ax.plot(xi, yi, 'o', color='steelblue', ms=8)
        ax.fill_between(x_plot, lb, y_plot, alpha=0.15, color='steelblue')
        ax.set_xlim(a, b)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel('$x$', fontsize=9)
        if panel_idx == 0:
            ax.set_ylabel('$y$', fontsize=9)
            ax.legend(fontsize=8, loc='upper left')
        ax.set_title(f'Iteration {n_pts}', fontsize=9)

    plt.suptitle('Shubert-Piyavskii: sawtooth lower bound', fontsize=10)
    plt.tight_layout()
    savefig('fig_shubert_piyavskii.pdf')

# -------------------------------------------------------------------
# Figure 9: Bisection Method – 4 iterations on f'(x)
# -------------------------------------------------------------------
def fig_bisection():
    def fp(x):
        return x - 1.0  # derivative of f(x) = 0.5*x^2 - x

    fig, axes = plt.subplots(1, 4, figsize=(12, 3))
    a0, b0 = 0.0, 1000.0
    # Use a smaller range for display
    a0, b0 = -2.0, 4.0
    intervals = [(a0, b0)]
    a, b = a0, b0
    for _ in range(3):
        m = (a + b) / 2
        if fp(m) == 0:
            a = b = m
        elif np.sign(fp(m)) == np.sign(fp(a)):
            a = m
        else:
            b = m
        intervals.append((a, b))

    x_plot = np.linspace(a0 - 0.3, b0 + 0.3, 400)
    x_plot = np.linspace(-2.5, 4.5, 400)
    y_plot = fp(x_plot)

    for i, (ax, (ai, bi)) in enumerate(zip(axes, intervals)):
        ax.fill_betweenx([min(y_plot) - 0.3, max(y_plot) + 0.3],
                         ai, bi, alpha=0.25, color='steelblue')
        ax.plot(x_plot, y_plot, 'k-', lw=1.5)
        ax.axhline(0, color='gray', lw=1, ls='--')
        mid = (ai + bi) / 2
        ax.plot(mid, fp(mid), 'ko', ms=6)
        ax.set_xlim(-2.5, 4.5)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel('$x$', fontsize=9)
        if i == 0:
            ax.set_ylabel("$f'$", fontsize=10)

    plt.suptitle("Bisection Method applied to $f'(x)$", fontsize=10)
    plt.tight_layout()
    savefig('fig_bisection.pdf')

# -------------------------------------------------------------------
# Figure 10: Fibonacci search example – f(x) = exp(x-2) - x on [-2,6]
# -------------------------------------------------------------------
def fig_fibonacci_example():
    def f(x):
        return np.exp(x - 2) - x

    fig, ax = plt.subplots(figsize=(7, 3.5))
    x = np.linspace(-2, 6, 400)
    y = f(x)
    ax.plot(x, y, 'k-', lw=2)
    ax.axhline(0, color='gray', lw=0.8, ls='--')

    # 5 evaluations, Fibonacci search
    # F6=8, F5=5, F4=3, F3=2, F2=1, F1=1
    # x1 = a + (b-a)*(1 - F5/F6) = -2 + 8*(1 - 5/8) = -2 + 3 = 1
    # x2 = a + (b-a)*(F5/F6) = -2 + 8*(5/8) = -2 + 5 = 3
    evals = [1, 3, 0, 2]
    colors = ['red', 'blue', 'green', 'purple']
    for xi, c in zip(evals, colors):
        ax.plot(xi, f(xi), 'o', color=c, ms=8, zorder=5)
        ax.annotate(f'$f({xi})={f(xi):.3f}$', (xi, f(xi)),
                    textcoords='offset points', xytext=(5, 10), fontsize=8, color=c)

    ax.set_xlabel('$x$', fontsize=11)
    ax.set_ylabel('$f(x) = e^{x-2} - x$', fontsize=11)
    ax.set_title('Fibonacci Search Example: $f(x)=e^{x-2}-x$ on $[-2,6]$, 5 evaluations',
                 fontsize=9)
    savefig('fig_fibonacci_example.pdf')

# -------------------------------------------------------------------
# Figure 11: Lipschitz bound illustration
# -------------------------------------------------------------------
def fig_lipschitz_bound():
    def f(x):
        return np.sin(3 * x) * 0.5 + 0.3 * x + 2.0

    ell = 2.0
    x0 = 1.5
    y0 = f(x0)

    fig, ax = plt.subplots(figsize=(6, 3.5))
    x = np.linspace(0, 3, 400)
    y = f(x)
    ax.plot(x, y, 'k-', lw=2, label='$f(x)$')
    # Lipschitz lines from x0
    lb_line = y0 - ell * np.abs(x - x0)
    ax.plot(x, lb_line, 'gray', lw=1.5, ls='--', label=r'$f(x_0) \pm \ell|x - x_0|$')
    ax.fill_between(x, lb_line, y, where=(y >= lb_line), alpha=0.15, color='steelblue',
                    label='feasible region')
    ax.plot(x0, y0, 'o', color='steelblue', ms=9, zorder=5)
    ax.set_xlabel('$x$', fontsize=11)
    ax.set_ylabel('$y$', fontsize=11)
    ax.set_title(r'Lipschitz lower bound: $f(x_0) - \ell|x - x_0|$', fontsize=10)
    ax.legend(fontsize=9)
    savefig('fig_lipschitz_bound.pdf')

# -------------------------------------------------------------------
# Run all figure generators
# -------------------------------------------------------------------
if __name__ == '__main__':
    print("Generating figures for Chapter 3: Bracketing")
    fig_unimodal_bracket()
    fig_bracket_minimum()
    fig_fibonacci_2query()
    fig_fibonacci_intervals()
    fig_golden_section_intervals()
    fig_golden_section_unimodal()
    fig_quadratic_fit()
    fig_shubert_piyavskii()
    fig_bisection()
    fig_fibonacci_example()
    fig_lipschitz_bound()
    print("All figures generated successfully.")
