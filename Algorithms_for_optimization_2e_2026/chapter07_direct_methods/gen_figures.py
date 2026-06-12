"""
gen_figures.py  --  Generate all figures for Chapter 7: Direct Methods
Algorithms for Optimization, 2nd ed., Kochenderfer & Wheeler (2026)

Run with:
    conda run -n py313 python3 gen_figures.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import matplotlib.gridspec as gridspec

OUT = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(OUT, exist_ok=True)

def savefig(name, fig=None, dpi=150):
    path = os.path.join(OUT, name)
    (fig or plt).savefig(path, bbox_inches='tight', dpi=dpi)
    plt.close('all')
    print(f"  Saved {path}")

# ─────────────────────────────────────────────────────────────────────────────
# Helper: Rosenbrock & Quadratic test functions
# ─────────────────────────────────────────────────────────────────────────────
def rosenbrock(x, y):
    return (1 - x)**2 + 100*(y - x**2)**2

def banana(x, y):
    return rosenbrock(x, y)

def simple_quad(x, y):
    return x**2 + 4*y**2 + x*y

# ─────────────────────────────────────────────────────────────────────────────
# Figure 1: Cyclic Coordinate Search illustration
# ─────────────────────────────────────────────────────────────────────────────
def fig_cyclic_coordinate():
    fig, ax = plt.subplots(figsize=(5, 4))
    x = np.linspace(-2, 2, 300)
    y = np.linspace(-1, 3, 300)
    X, Y = np.meshgrid(x, y)
    Z = rosenbrock(X, Y)
    ax.contour(X, Y, Z, levels=np.logspace(0, 3.5, 18), colors='steelblue', linewidths=0.7)

    # Simulate cyclic coordinate search path
    pts = [np.array([-1.5, 2.0])]
    def line_min_x1(pt):
        ts = np.linspace(-2, 2, 400)
        vals = [rosenbrock(t, pt[1]) for t in ts]
        return np.array([ts[np.argmin(vals)], pt[1]])
    def line_min_x2(pt):
        ts = np.linspace(-1, 3, 400)
        vals = [rosenbrock(pt[0], t) for t in ts]
        return np.array([pt[0], ts[np.argmin(vals)]])

    p = pts[0].copy()
    for _ in range(5):
        p_new = line_min_x1(p)
        pts.append(p_new.copy())
        p = line_min_x2(p_new)
        pts.append(p.copy())

    pts = np.array(pts)
    ax.plot(pts[:, 0], pts[:, 1], 'o-', color='crimson', markersize=4, linewidth=1.5, label='Search path')
    ax.plot(pts[0, 0], pts[0, 1], 's', color='orange', markersize=7, zorder=5, label='Start')
    ax.plot(1, 1, '*', color='lime', markersize=10, zorder=5, label='Minimum')
    ax.set_xlabel(r'$x_1$', fontsize=12)
    ax.set_ylabel(r'$x_2$', fontsize=12)
    ax.set_title('Cyclic Coordinate Search\n(Rosenbrock function)', fontsize=11)
    ax.legend(fontsize=8)
    ax.set_xlim(-2, 2)
    ax.set_ylim(-1, 3)
    savefig("fig_cyclic_coordinate.pdf", fig)

# ─────────────────────────────────────────────────────────────────────────────
# Figure 2: Cyclic coordinate can get stuck (diagonal valley)
# ─────────────────────────────────────────────────────────────────────────────
def fig_cyclic_stuck():
    fig, ax = plt.subplots(figsize=(4.5, 4))
    x = np.linspace(-2, 4, 300)
    y = np.linspace(-2, 4, 300)
    X, Y = np.meshgrid(x, y)
    # Rotated ellipse
    theta = np.pi / 4
    Xr = X * np.cos(theta) + Y * np.sin(theta)
    Yr = -X * np.sin(theta) + Y * np.cos(theta)
    Z = Xr**2 / 0.5 + Yr**2 / 4
    ax.contour(X, Y, Z, levels=12, colors='steelblue', linewidths=0.7)
    # Show stuck zig-zag
    p = np.array([-1.5, 3.0])
    pts = [p.copy()]
    for _ in range(6):
        # minimize along x1
        ts = np.linspace(-2, 4, 400)
        vals = [(t*np.cos(theta)+p[1]*np.sin(theta))**2/0.5 + (-t*np.sin(theta)+p[1]*np.cos(theta))**2/4 for t in ts]
        p = np.array([ts[np.argmin(vals)], p[1]])
        pts.append(p.copy())
        # minimize along x2
        vals = [(p[0]*np.cos(theta)+t*np.sin(theta))**2/0.5 + (-p[0]*np.sin(theta)+t*np.cos(theta))**2/4 for t in ts]
        p = np.array([p[0], ts[np.argmin(vals)]])
        pts.append(p.copy())
    pts = np.array(pts)
    ax.plot(pts[:, 0], pts[:, 1], 'o-', color='crimson', markersize=4, linewidth=1.5)
    ax.plot(pts[0, 0], pts[0, 1], 's', color='orange', markersize=7, zorder=5, label='Start')
    ax.plot(0, 0, '*', color='lime', markersize=10, zorder=5, label='Minimum')
    ax.set_xlabel(r'$x_1$', fontsize=12)
    ax.set_ylabel(r'$x_2$', fontsize=12)
    ax.set_title('Cyclic Coordinate: Slow on\nDiagonal Valleys', fontsize=11)
    ax.legend(fontsize=8)
    savefig("fig_cyclic_stuck.pdf", fig)

# ─────────────────────────────────────────────────────────────────────────────
# Figure 3: Powell's Method illustration
# ─────────────────────────────────────────────────────────────────────────────
def fig_powell():
    fig, ax = plt.subplots(figsize=(5, 4))
    x = np.linspace(-2, 2, 300)
    y = np.linspace(-1, 3, 300)
    X, Y = np.meshgrid(x, y)
    Z = rosenbrock(X, Y)
    ax.contour(X, Y, Z, levels=np.logspace(0, 3.5, 18), colors='steelblue', linewidths=0.7)

    # Powell's method: start, show conjugate directions building
    p0 = np.array([-1.2, 1.8])
    ax.annotate('', xy=p0 + np.array([0.6, 0.0]), xytext=p0,
                arrowprops=dict(arrowstyle='->', color='red', lw=2))
    ax.annotate('', xy=p0 + np.array([0.0, -0.5]), xytext=p0,
                arrowprops=dict(arrowstyle='->', color='blue', lw=2))
    # Conjugate direction
    ax.annotate('', xy=p0 + np.array([0.5, -0.4]), xytext=p0,
                arrowprops=dict(arrowstyle='->', color='green', lw=2))

    ax.text(p0[0] + 0.65, p0[1] + 0.05, r'$\mathbf{e}_1$', color='red', fontsize=11)
    ax.text(p0[0] + 0.05, p0[1] - 0.55, r'$\mathbf{e}_2$', color='blue', fontsize=11)
    ax.text(p0[0] + 0.52, p0[1] - 0.5, r'$\mathbf{d}_{conj}$', color='green', fontsize=10)
    ax.plot(*p0, 'o', color='orange', markersize=8, zorder=5, label='Current point')
    ax.plot(1, 1, '*', color='lime', markersize=10, zorder=5, label='Minimum')
    ax.set_xlabel(r'$x_1$', fontsize=12)
    ax.set_ylabel(r'$x_2$', fontsize=12)
    ax.set_title("Powell's Method:\nConjugate Directions", fontsize=11)
    ax.legend(fontsize=8)
    ax.set_xlim(-2, 2)
    ax.set_ylim(-1, 3)
    savefig("fig_powell.pdf", fig)

# ─────────────────────────────────────────────────────────────────────────────
# Figure 4: Hooke-Jeeves iterations
# ─────────────────────────────────────────────────────────────────────────────
def hooke_jeeves(f, x0, alpha=0.5, eps=1e-4):
    x = np.array(x0, dtype=float)
    n = len(x)
    traj = [x.copy()]
    while alpha > eps:
        improved = False
        x_base = x.copy()
        x_trial = x.copy()
        for i in range(n):
            for sign in [1, -1]:
                x_new = x_trial.copy()
                x_new[i] += sign * alpha
                if f(x_new) < f(x_trial):
                    x_trial = x_new
                    improved = True
                    break
        if improved:
            # Pattern move
            x_pattern = x_trial + (x_trial - x_base)
            x = x_trial.copy()
            traj.append(x.copy())
        else:
            alpha *= 0.5
    return np.array(traj)

def fig_hooke_jeeves():
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    x = np.linspace(-2, 2, 300)
    y = np.linspace(-1, 3, 300)
    X, Y = np.meshgrid(x, y)
    Z = rosenbrock(X, Y)

    starts = [np.array([-1.2, 1.8]), np.array([-0.5, 0.5]), np.array([0.5, 2.0])]
    alphas = [0.8, 0.4, 0.2]
    for ax, x0, a in zip(axes, starts, alphas):
        ax.contour(X, Y, Z, levels=np.logspace(0, 3.5, 15), colors='steelblue', linewidths=0.6)
        f2d = lambda v: rosenbrock(v[0], v[1])
        traj = hooke_jeeves(f2d, x0, alpha=a, eps=a/10)
        if len(traj) > 1:
            ax.plot(traj[:, 0], traj[:, 1], 'o-', color='crimson', markersize=4, linewidth=1.3)
        ax.plot(x0[0], x0[1], 's', color='orange', markersize=7, zorder=5)
        ax.plot(1, 1, '*', color='lime', markersize=9, zorder=5)
        ax.set_xlabel(r'$x_1$', fontsize=10)
        ax.set_ylabel(r'$x_2$', fontsize=10)
        ax.set_xlim(-2, 2); ax.set_ylim(-1, 3)
    axes[0].set_title('Hooke-Jeeves: Early Iterations', fontsize=10)
    axes[1].set_title('Hooke-Jeeves: Mid Iterations', fontsize=10)
    axes[2].set_title('Hooke-Jeeves: Later Iterations', fontsize=10)
    fig.suptitle('Hooke-Jeeves Method on Rosenbrock', fontsize=11, y=1.01)
    savefig("fig_hooke_jeeves.pdf", fig)

# ─────────────────────────────────────────────────────────────────────────────
# Figure 5: Positive Spanning Set illustration
# ─────────────────────────────────────────────────────────────────────────────
def fig_positive_spanning():
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2))

    # Panel 1: Only spans a cone (2 directions, same side)
    ax = axes[0]
    ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    ax.axhline(0, color='gray', lw=0.5); ax.axvline(0, color='gray', lw=0.5)
    dirs = [(1, 0.5), (0.5, 1)]
    for d in dirs:
        ax.annotate('', xy=d, xytext=(0, 0),
                    arrowprops=dict(arrowstyle='->', color='navy', lw=2))
    # Shade the cone
    theta1 = np.arctan2(0.5, 1)
    theta2 = np.arctan2(1, 0.5)
    thetas = np.linspace(theta1, theta2, 50)
    ax.fill(np.append([0], np.cos(thetas)), np.append([0], np.sin(thetas)),
            alpha=0.2, color='blue')
    ax.set_title('Only spans cone', fontsize=9)
    ax.set_xlabel(r'$x_1$'); ax.set_ylabel(r'$x_2$')

    # Panel 2: Only positively spans 1D
    ax = axes[1]
    ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    ax.axhline(0, color='gray', lw=0.5); ax.axvline(0, color='gray', lw=0.5)
    dirs = [(1, 0), (-1, 0)]
    for d in dirs:
        ax.annotate('', xy=d, xytext=(0, 0),
                    arrowprops=dict(arrowstyle='->', color='navy', lw=2))
    ax.axhline(0, color='blue', lw=2, alpha=0.4)
    ax.set_title('Positively spans 1D only', fontsize=9)
    ax.set_xlabel(r'$x_1$'); ax.set_ylabel(r'$x_2$')

    # Panel 3: Positively spans R^2
    ax = axes[2]
    ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    ax.axhline(0, color='gray', lw=0.5); ax.axvline(0, color='gray', lw=0.5)
    dirs = [(1, 0), (0, 1), (-1, -1)]
    for d in dirs:
        n = np.linalg.norm(d)
        ax.annotate('', xy=np.array(d)/n, xytext=(0, 0),
                    arrowprops=dict(arrowstyle='->', color='navy', lw=2))
    circle = plt.Circle((0, 0), 1.0, color='blue', alpha=0.1, fill=True)
    ax.add_patch(circle)
    ax.set_title(r'Positively spans $\mathbb{R}^2$', fontsize=9)
    ax.set_xlabel(r'$x_1$'); ax.set_ylabel(r'$x_2$')

    fig.suptitle('Positive Spanning Sets', fontsize=11)
    plt.tight_layout()
    savefig("fig_positive_spanning.pdf", fig)

# ─────────────────────────────────────────────────────────────────────────────
# Figure 6: Generalized Pattern Search — mesh lattice
# ─────────────────────────────────────────────────────────────────────────────
def fig_pattern_search_lattice():
    fig, ax = plt.subplots(figsize=(5, 4))
    x = np.linspace(-2, 2, 200)
    y = np.linspace(-2, 2, 200)
    X, Y = np.meshgrid(x, y)
    Z = simple_quad(X, Y)
    ax.contour(X, Y, Z, levels=15, colors='lightsteelblue', linewidths=0.6)

    # Draw mesh lattice
    step = 0.4
    for xi in np.arange(-2, 2.1, step):
        for yi in np.arange(-2, 2.1, step):
            ax.plot(xi, yi, '.', color='lightgray', markersize=3, zorder=1)

    # Search trajectory
    pts = [np.array([-1.2, 1.0])]
    f2d = lambda v: simple_quad(v[0], v[1])
    p = pts[0].copy()
    step_size = 0.4
    for _ in range(8):
        best = p.copy()
        best_val = f2d(p)
        for d in [(1,0),(-1,0),(0,1),(0,-1)]:
            cand = p + step_size * np.array(d)
            if f2d(cand) < best_val:
                best = cand
                best_val = f2d(cand)
        if np.allclose(best, p):
            step_size *= 0.5
        else:
            p = best
            pts.append(p.copy())

    pts = np.array(pts)
    ax.plot(pts[:, 0], pts[:, 1], 'o-', color='crimson', markersize=5, linewidth=1.5, zorder=3)
    ax.plot(pts[0, 0], pts[0, 1], 's', color='orange', markersize=8, zorder=5, label='Start')
    ax.plot(0, 0, '*', color='lime', markersize=10, zorder=5, label='Min')
    ax.set_xlabel(r'$x_1$', fontsize=12)
    ax.set_ylabel(r'$x_2$', fontsize=12)
    ax.set_title('Generalized Pattern Search\non Scaled Lattice', fontsize=11)
    ax.legend(fontsize=9)
    ax.set_xlim(-2, 2); ax.set_ylim(-2, 2)
    savefig("fig_pattern_search.pdf", fig)

# ─────────────────────────────────────────────────────────────────────────────
# Figure 7: Nelder-Mead simplex operations
# ─────────────────────────────────────────────────────────────────────────────
def fig_nelder_mead_ops():
    fig, axes = plt.subplots(2, 2, figsize=(8, 6))
    ax_names = ['Reflection', 'Expansion', 'Contraction', 'Shrinkage']

    # Base simplex: xl (lowest), xs (second highest), xh (highest)
    xl = np.array([0.5, 0.2])
    xs = np.array([1.5, 0.2])
    xh = np.array([1.0, 1.4])
    centroid = (xl + xs) / 2  # centroid of all except highest

    alpha = 1.0
    beta  = 2.0
    gamma = 0.5

    xr = centroid + alpha * (centroid - xh)
    xe = centroid + beta  * (xr - centroid)
    xc = centroid + gamma * (xh - centroid)

    colors = {'xl': 'royalblue', 'xs': 'steelblue', 'xh': 'crimson',
              'xbar': 'gray', 'new': 'green'}

    def draw_tri(ax, p1, p2, p3, new_pt, new_label, color_new, title):
        tri = plt.Polygon([p1, p2, p3], fill=False, edgecolor='steelblue', linewidth=2)
        ax.add_patch(tri)
        new_tri = plt.Polygon([p1, p2, new_pt], fill=False,
                               edgecolor=color_new, linewidth=1.5, linestyle='--')
        ax.add_patch(new_tri)
        ax.plot(*xh, 'ro', markersize=8, zorder=5)
        ax.plot(*xl, 'bo', markersize=8, zorder=5)
        ax.plot(*xs, 'bs', markersize=6, zorder=5)
        ax.plot(*centroid, 'k+', markersize=10, markeredgewidth=2, zorder=5)
        ax.plot(*new_pt, 'g^', markersize=9, zorder=5)
        ax.annotate(r'$\mathbf{x}_h$', xh, fontsize=9, color='red',
                    xytext=(xh[0]+0.05, xh[1]+0.05))
        ax.annotate(r'$\mathbf{x}_l$', xl, fontsize=9, color='blue',
                    xytext=(xl[0]-0.18, xl[1]-0.12))
        ax.annotate(r'$\bar{\mathbf{x}}$', centroid, fontsize=9, color='black',
                    xytext=(centroid[0]+0.03, centroid[1]+0.05))
        ax.annotate(new_label, new_pt, fontsize=9, color='green',
                    xytext=(new_pt[0]+0.05, new_pt[1]+0.05))
        ax.set_title(title, fontsize=10)
        ax.set_xlim(-0.2, 2.2); ax.set_ylim(-0.3, 2.2)
        ax.set_aspect('equal')
        ax.axis('off')

    draw_tri(axes[0,0], xl, xs, xh, xr, r'$\mathbf{x}_r$', 'green', 'Reflection')
    draw_tri(axes[0,1], xl, xs, xh, xe, r'$\mathbf{x}_e$', 'darkgreen', 'Expansion')
    draw_tri(axes[1,0], xl, xs, xh, xc, r'$\mathbf{x}_c$', 'orange', 'Contraction')

    # Shrinkage: all points move toward xl
    ax = axes[1,1]
    xl_new = xl
    xs_new = (xs + xl) / 2
    xh_new = (xh + xl) / 2
    orig_tri = plt.Polygon([xl, xs, xh], fill=False, edgecolor='steelblue', linewidth=2)
    new_tri  = plt.Polygon([xl_new, xs_new, xh_new], fill=False,
                            edgecolor='orange', linewidth=1.5, linestyle='--')
    ax.add_patch(orig_tri)
    ax.add_patch(new_tri)
    ax.plot(*xl, 'bo', markersize=8, zorder=5)
    ax.plot(*xs_new, 'g^', markersize=8, zorder=5)
    ax.plot(*xh_new, 'g^', markersize=8, zorder=5)
    ax.annotate(r'$\mathbf{x}_l$', xl, fontsize=9, color='blue',
                xytext=(xl[0]-0.18, xl[1]-0.12))
    ax.set_title('Shrinkage', fontsize=10)
    ax.set_xlim(-0.2, 2.2); ax.set_ylim(-0.3, 2.2)
    ax.set_aspect('equal')
    ax.axis('off')

    fig.suptitle('Nelder-Mead Simplex Operations', fontsize=12, fontweight='bold')
    plt.tight_layout()
    savefig("fig_nelder_mead_ops.pdf", fig)

# ─────────────────────────────────────────────────────────────────────────────
# Figure 8: Nelder-Mead iterations on Rosenbrock
# ─────────────────────────────────────────────────────────────────────────────
def nelder_mead_py(f, simplex, eps=1e-6, max_iter=500,
                   alpha=1.0, beta=2.0, gamma=0.5):
    """Pure-Python Nelder-Mead for illustration."""
    S = [np.array(s, dtype=float) for s in simplex]
    history = [S[0].copy()]
    for _ in range(max_iter):
        y_arr = np.array([f(s) for s in S])
        p = np.argsort(y_arr)
        S = [S[i] for i in p]
        y_arr = y_arr[p]
        if np.std(y_arr) < eps:
            break
        xl, yl = S[0], y_arr[0]
        xh, yh = S[-1], y_arr[-1]
        xs, ys = S[-2], y_arr[-2]
        xm = np.mean(S[:-1], axis=0)
        xr = xm + alpha * (xm - xh)
        yr = f(xr)
        if yr < yl:
            xe = xm + beta * (xr - xm)
            ye = f(xe)
            S[-1] = xe if ye < yr else xr
        elif yr >= ys:
            if yr < yh:
                xh, yh = xr, yr
                S[-1] = xr
            xc = xm + gamma * (xh - xm)
            yc = f(xc)
            if yc > yh:
                for i in range(1, len(S)):
                    S[i] = (S[i] + xl) / 2
            else:
                S[-1] = xc
        else:
            S[-1] = xr
        history.append(S[0].copy())
    return S[0], np.array(history)

def fig_nelder_mead_iters():
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    axes = axes.flatten()
    x = np.linspace(-2, 2, 300)
    y = np.linspace(-1, 3, 300)
    X, Y = np.meshgrid(x, y)
    Z = rosenbrock(X, Y)

    f2d = lambda v: rosenbrock(v[0], v[1])
    init_simplex = [[-1.2, 1.8], [-0.8, 1.8], [-1.2, 2.2]]
    _, history = nelder_mead_py(f2d, init_simplex, eps=1e-8, max_iter=200)

    snap_iters = [0, 5, 15, 30, 60, len(history)-1]
    for k, (ax, it) in enumerate(zip(axes, snap_iters)):
        ax.contour(X, Y, Z, levels=np.logspace(0, 3.5, 15),
                   colors='steelblue', linewidths=0.5)
        idx = min(it, len(history)-1)
        ax.plot(history[:idx+1, 0], history[:idx+1, 1],
                'o-', color='crimson', markersize=3, linewidth=1, alpha=0.7)
        if idx < len(history):
            ax.plot(history[idx, 0], history[idx, 1], 's', color='orange',
                    markersize=6, zorder=5)
        ax.plot(1, 1, '*', color='lime', markersize=9, zorder=5)
        ax.set_xlim(-2, 2); ax.set_ylim(-1, 3)
        ax.set_title(f'Iteration {idx}', fontsize=9)
        ax.set_xlabel(r'$x_1$', fontsize=8)
        ax.set_ylabel(r'$x_2$', fontsize=8)
    fig.suptitle('Nelder-Mead on Rosenbrock (12 iterations shown)', fontsize=11)
    plt.tight_layout()
    savefig("fig_nelder_mead_iters.pdf", fig)

# ─────────────────────────────────────────────────────────────────────────────
# Figure 9: DIRECT method — interval partitioning
# ─────────────────────────────────────────────────────────────────────────────
def branin(x1, x2):
    """Branin function mapped to [0,1]^2 from x1 in [-5,10], x2 in [0,15]"""
    xx1 = x1 * 15 - 5
    xx2 = x2 * 15
    a = 1; b = 5.1/(4*np.pi**2); c = 5/np.pi; r = 6; s = 10; t = 1/(8*np.pi)
    return a*(xx2 - b*xx1**2 + c*xx1 - r)**2 + s*(1-t)*np.cos(xx1) + s

def fig_direct_partitioning():
    """Show DIRECT partitioning after a few iterations."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Left: show partition boxes after ~5 iterations (schematic)
    ax = axes[0]
    x = np.linspace(0, 1, 200)
    y = np.linspace(0, 1, 200)
    X, Y = np.meshgrid(x, y)
    Z = branin(X, Y)
    ax.pcolormesh(X, Y, Z, cmap='viridis', shading='auto')
    ax.contour(X, Y, Z, levels=12, colors='white', linewidths=0.4, alpha=0.5)

    # Draw partition lines (schematic of DIRECT after a few splits)
    for xv in [1/3, 2/3]:
        ax.axvline(xv, color='white', lw=0.8, alpha=0.7)
    for yv in [1/3, 2/3]:
        ax.plot([1/3, 2/3], [yv, yv], 'w-', lw=0.8, alpha=0.7)
    # Center points
    centers_x = [1/6, 3/6, 5/6, 3/6, 3/6]
    centers_y = [3/6, 3/6, 3/6, 1/6, 5/6]
    ax.plot(centers_x, centers_y, 'w.', markersize=6)
    ax.set_xlabel(r'$x_1$', fontsize=11)
    ax.set_ylabel(r'$x_2$', fontsize=11)
    ax.set_title('DIRECT: Interval Partitions\n(Branin function)', fontsize=10)

    # Right: (r, f(c)) scatter — identify potentially optimal intervals
    ax = axes[1]
    np.random.seed(42)
    rs = np.array([0.0, 0.05, 0.1, 0.15, 0.18, 0.20, 0.25, 0.30, 0.05,
                   0.10, 0.20, 0.08, 0.15])
    fcs = np.array([25, 80, 45, 120, 200, 90, 150, 300, 60, 35, 75, 100, 180])
    ax.scatter(rs, fcs, color='navy', s=30, zorder=3)
    # Lower convex hull
    from scipy.spatial import ConvexHull
    pts_r = np.column_stack([rs, fcs])
    # Manual lower hull
    order = np.argsort(rs)
    xs_h = rs[order]; ys_h = fcs[order]
    # Pick lower-left frontier
    frontier_idx = []
    min_y = np.inf
    for i in range(len(xs_h)-1, -1, -1):
        if ys_h[i] < min_y:
            min_y = ys_h[i]
            frontier_idx.append(i)
    frontier_idx = sorted(frontier_idx)
    ax.plot(xs_h[frontier_idx], ys_h[frontier_idx], 'r-o', markersize=5,
            label='Lower convex hull')
    ax.set_xlabel(r'$r$ (interval half-width)', fontsize=11)
    ax.set_ylabel(r'$f(\mathbf{c})$ (center value)', fontsize=11)
    ax.set_title('DIRECT: Potentially Optimal\nIntervals', fontsize=10)
    ax.legend(fontsize=9)
    plt.tight_layout()
    savefig("fig_direct_partitioning.pdf", fig)

# ─────────────────────────────────────────────────────────────────────────────
# Figure 10: DIRECT splitting illustration
# ─────────────────────────────────────────────────────────────────────────────
def fig_direct_splitting():
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))

    ax = axes[0]
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    # Original rectangle
    rect = plt.Rectangle((0.1, 0.2), 0.8, 0.6, fill=False,
                          edgecolor='navy', linewidth=2)
    ax.add_patch(rect)
    ax.plot(0.5, 0.5, 'ko', markersize=8, zorder=5)
    ax.text(0.5, 0.52, 'center', ha='center', fontsize=8)
    ax.set_title('Before Splitting', fontsize=10)
    ax.set_xlabel(r'$x_1$'); ax.set_ylabel(r'$x_2$')

    ax = axes[1]
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    # Split into 3 along longest dimension (x1)
    thirds = [0.1 + k*0.8/3 for k in range(4)]
    for i in range(3):
        rect = plt.Rectangle((thirds[i], 0.2), 0.8/3, 0.6, fill=False,
                              edgecolor='navy', linewidth=1.5)
        ax.add_patch(rect)
        cx = (thirds[i] + thirds[i+1]) / 2
        ax.plot(cx, 0.5, 'ro', markersize=6, zorder=5)
    ax.set_title('After Splitting Along Longest Dim.', fontsize=10)
    ax.set_xlabel(r'$x_1$'); ax.set_ylabel(r'$x_2$')

    fig.suptitle('DIRECT: Rectangle Splitting', fontsize=11)
    plt.tight_layout()
    savefig("fig_direct_splitting.pdf", fig)

# ─────────────────────────────────────────────────────────────────────────────
# Figure 11: Comparison of methods — convergence on Rosenbrock
# ─────────────────────────────────────────────────────────────────────────────
def fig_convergence_comparison():
    from scipy.optimize import minimize
    import warnings; warnings.filterwarnings('ignore')

    f2d = lambda v: rosenbrock(v[0], v[1])
    x0 = np.array([-1.2, 1.0])

    histories = {}
    for method in ['Nelder-Mead', 'Powell']:
        hist = []
        def cb(xk):
            hist.append(f2d(xk))
        minimize(f2d, x0, method=method, callback=cb,
                 options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-10})
        hist = [f2d(x0)] + hist
        histories[method] = hist

    # Cyclic coordinate (manual)
    def cyclic_hist(f, x0, n_iters=30):
        from scipy.optimize import minimize_scalar
        x = np.array(x0, dtype=float)
        n = len(x)
        hist = [f(x)]
        for _ in range(n_iters):
            for i in range(n):
                def fi(t):
                    xc = x.copy(); xc[i] = t; return f(xc)
                res = minimize_scalar(fi, bounds=(x[i]-3, x[i]+3), method='bounded')
                x[i] = res.x
            hist.append(f(x))
        return hist

    histories['Cyclic Coord.'] = cyclic_hist(f2d, x0, n_iters=40)

    fig, ax = plt.subplots(figsize=(6, 4))
    for name, hist in histories.items():
        ax.semilogy(hist, label=name, linewidth=1.8)
    ax.set_xlabel('Iterations', fontsize=11)
    ax.set_ylabel(r'$f(\mathbf{x})$', fontsize=11)
    ax.set_title('Convergence Comparison on Rosenbrock', fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, which='both', alpha=0.3)
    savefig("fig_convergence_comparison.pdf", fig)

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("Generating figures for Chapter 7: Direct Methods ...")
    fig_cyclic_coordinate()
    fig_cyclic_stuck()
    fig_powell()
    fig_hooke_jeeves()
    fig_positive_spanning()
    fig_pattern_search_lattice()
    fig_nelder_mead_ops()
    fig_nelder_mead_iters()
    fig_direct_partitioning()
    fig_direct_splitting()
    fig_convergence_comparison()
    print("All figures saved to:", OUT)
