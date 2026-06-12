"""
gen_figures.py  —  Chapter 15: Multiobjective Optimization
Generates all figures needed for chapter15_slides.tex
Saves PDFs to ./figures/
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Ellipse, FancyArrowPatch
import numpy as np
from scipy.optimize import minimize, linprog
import os
import sys

# ── output directory ──────────────────────────────────────────────────────────
FIGDIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIGDIR, exist_ok=True)

def savefig(name):
    path = os.path.join(FIGDIR, name)
    plt.savefig(path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  saved {path}")


# ── colour palette ────────────────────────────────────────────────────────────
BLUE  = "#2166ac"
LBLUE = "#a6cee3"
RED   = "#d6604d"
GREEN = "#4dac26"
GRAY  = "#888888"

# =============================================================================
# Fig 1 — Dominance regions (single vs multi-objective)
# =============================================================================
def fig_dominance():
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))

    # --- single objective ---
    ax = axes[0]
    ax.set_xlim(0, 4); ax.set_ylim(-0.5, 0.5)
    ax.axhline(0, color='k', lw=1)
    ax.annotate('', xy=(4, 0), xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', lw=1.5, color='k'))
    ax.plot(2.0, 0, 'ko', ms=8)
    ax.annotate('$f(\\mathbf{x})$', xy=(2.0, 0), xytext=(2.0, 0.15),
                ha='center', fontsize=11)
    ax.annotate('', xy=(0.1, 0), xytext=(1.9, 0),
                arrowprops=dict(arrowstyle='<->', lw=1.5, color=BLUE))
    ax.text(0.95, -0.2, "$f(\\mathbf{x}')<f(\\mathbf{x})$\n$\\mathbf{x}'$ is better",
            ha='center', fontsize=9, color=BLUE)
    ax.annotate('', xy=(3.9, 0), xytext=(2.1, 0),
                arrowprops=dict(arrowstyle='<->', lw=1.5, color=RED))
    ax.text(3.0, -0.2, "$f(\\mathbf{x}')>f(\\mathbf{x})$\n$\\mathbf{x}'$ is worse",
            ha='center', fontsize=9, color=RED)
    ax.text(3.9, 0.05, '$y$', fontsize=11)
    ax.set_title('Single Objective', fontsize=12)
    ax.axis('off')

    # --- multiple objectives ---
    ax = axes[1]
    ax.set_xlim(-0.2, 3.5); ax.set_ylim(-0.2, 3.5)
    ax.annotate('', xy=(3.5, 0), xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', lw=1.5, color='k'))
    ax.annotate('', xy=(0, 3.5), xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', lw=1.5, color='k'))
    ax.text(3.4, -0.2, '$y_1$', fontsize=12)
    ax.text(-0.15, 3.4, '$y_2$', fontsize=12)
    fx, fy = 1.8, 1.8
    ax.plot(fx, fy, 'ko', ms=8, zorder=5)
    ax.text(fx+0.1, fy+0.1, '$\\mathbf{f}(\\mathbf{x})$', fontsize=10)
    # blue (better) quadrant
    ax.fill_between([0, fx], [0, 0], [fy, fy], color=LBLUE, alpha=0.7, label="$\\mathbf{x}'$ is better")
    ax.text(0.7, 0.9,
            "$\\mathbf{f}(\\mathbf{x}')<\\mathbf{f}(\\mathbf{x})$\n$\\mathbf{x}'$ is better",
            fontsize=9, color='navy', va='center')
    # red (worse) quadrant
    ax.fill_between([fx, 3.3], [fy, fy], [3.3, 3.3], color='#f4a582', alpha=0.7)
    ax.text(2.5, 2.5,
            "$\\mathbf{f}(\\mathbf{x}')>\\mathbf{f}(\\mathbf{x})$\n$\\mathbf{x}'$ is worse",
            fontsize=9, color='darkred', va='center')
    ax.set_title('Multiple Objectives', fontsize=12)
    ax.set_aspect('equal')
    plt.tight_layout()
    savefig("fig_dominance.pdf")


# =============================================================================
# Fig 2 — Pareto frontier concept (criterion space)
# =============================================================================
def fig_pareto_frontier():
    fig, ax = plt.subplots(figsize=(5, 4))
    theta = np.linspace(0, np.pi/2, 200)
    # kidney-shaped feasible region
    t = np.linspace(0, 2*np.pi, 500)
    rx, ry = 2.0, 1.5
    cx, cy = 2.5, 2.0
    x = cx + rx*np.cos(t) + 0.4*np.cos(2*t)
    y = cy + ry*np.sin(t) + 0.3*np.sin(2*t)
    ax.fill(x, y, color=LBLUE, alpha=0.5, label='Criterion space $\\mathcal{Y}$')
    ax.plot(x, y, color=BLUE, lw=1.5)

    # Pareto frontier (lower-left boundary)
    front_t = np.linspace(np.pi*0.7, np.pi*1.1, 80)
    fx = cx + rx*np.cos(front_t) + 0.4*np.cos(2*front_t)
    fy = cy + ry*np.sin(front_t) + 0.3*np.sin(2*front_t)
    ax.plot(fx, fy, color=BLUE, lw=4, label='Pareto frontier', solid_capstyle='round')

    ax.set_xlabel('$y_1$', fontsize=13)
    ax.set_ylabel('$y_2$', fontsize=13)
    ax.set_xlim(0, 5.2); ax.set_ylim(0, 4.2)
    ax.legend(fontsize=10, loc='upper right')
    ax.set_title('Criterion Space and Pareto Frontier', fontsize=12)
    plt.tight_layout()
    savefig("fig_pareto_frontier.pdf")


# =============================================================================
# Fig 3 — Aircraft collision avoidance example
# =============================================================================
def fig_collision_avoidance():
    np.random.seed(42)
    fig, ax = plt.subplots(figsize=(5, 4))

    # Pareto frontier  (concave in collision–alert space)
    t_front = np.linspace(0, 1, 100)
    front_x = t_front**0.6 * 5          # alert rate
    front_y = (1 - t_front)**0.6 * 5   # collision rate
    ax.fill_between(front_x, front_y, front_y + 0.35,
                    color=LBLUE, alpha=0.9, lw=0)
    ax.plot(front_x, front_y, color=BLUE, lw=2.5)

    # scattered dominated points
    dom_x = np.random.uniform(0.5, 5.0, 60)
    dom_y = np.random.uniform(0.5, 5.0, 60)
    # keep only those above the frontier
    mask = dom_y > (1 - (dom_x/5)**1.667)**0.6 * 5 + 0.4
    ax.scatter(dom_x[mask], dom_y[mask], c='k', s=18, zorder=3)

    ax.text(3.5, 3.5, 'Pareto dominated\n(suboptimal)', fontsize=9, ha='center')
    ax.text(4.2, 1.8, 'Criterion space\n(denoted $\\mathcal{Y}$)', fontsize=9, ha='center')
    ax.text(2.2, 0.3, 'Pareto Frontier', fontsize=10, color=BLUE, ha='center')
    ax.plot(0.05, 0.05, '*', ms=12, color='gold', zorder=5)
    ax.text(0.2, 0.15, 'Ideal', fontsize=9)

    ax.set_xlabel('Alert rate', fontsize=12)
    ax.set_ylabel('Collision rate', fontsize=12)
    ax.set_xlim(0, 5.5); ax.set_ylim(0, 5.5)
    ax.set_title('Example: Aircraft Collision Avoidance', fontsize=11)
    plt.tight_layout()
    savefig("fig_collision_avoidance.pdf")


# =============================================================================
# Fig 4 — Constraint method illustration
# =============================================================================
def fig_constraint_method():
    fig, ax = plt.subplots(figsize=(5, 4.5))
    t = np.linspace(0, 2*np.pi, 500)
    rx, ry = 2.2, 1.8
    cx, cy = 2.8, 2.5
    x = cx + rx*np.cos(t) + 0.5*np.cos(2*t)
    y = cy + ry*np.sin(t) + 0.4*np.sin(2*t)
    ax.fill(x, y, color=LBLUE, alpha=0.5)
    ax.plot(x, y, color=BLUE, lw=1.5)

    # Pareto frontier
    front_t = np.linspace(np.pi*0.68, np.pi*1.12, 80)
    fx = cx + rx*np.cos(front_t) + 0.5*np.cos(2*front_t)
    fy = cy + ry*np.sin(front_t) + 0.4*np.sin(2*front_t)
    ax.plot(fx, fy, color=BLUE, lw=4)

    # epsilon-constraint lines
    c2_large = 3.5
    c2_small = 1.5
    ax.axhline(c2_large, color=RED, lw=1.5, ls='--', label=f'$c_2$ large')
    ax.axhline(c2_small, color=GREEN, lw=1.5, ls='--', label=f'$c_2$ small')

    # optimal points
    # find intersection of c2=c2_large with pareto front
    for c2, col in [(c2_large, RED), (c2_small, GREEN)]:
        diffs = np.abs(fy - c2)
        idx = np.argmin(diffs)
        ax.plot(fx[idx], fy[idx], 'o', color=col, ms=9, zorder=5)
        ax.axvline(fx[idx], color=col, lw=0.8, ls=':')

    ax.text(-0.1, c2_large+0.1, '$c_2$ large', color=RED, fontsize=9)
    ax.text(-0.1, c2_small-0.3, '$c_2$ small', color=GREEN, fontsize=9)
    ax.set_xlabel('$y_1$', fontsize=13)
    ax.set_ylabel('$y_2$', fontsize=13)
    ax.text(cx+0.3, cy+0.2, '$\\mathcal{Y}$', fontsize=16, color=BLUE)
    ax.set_xlim(-0.3, 5.5); ax.set_ylim(-0.3, 5.0)
    ax.set_title('Constraint Method', fontsize=12)
    plt.tight_layout()
    savefig("fig_constraint_method.pdf")


# =============================================================================
# Fig 5 — Weighted sum method
# =============================================================================
def fig_weighted_sum():
    fig, ax = plt.subplots(figsize=(5, 4.5))
    t = np.linspace(0, 2*np.pi, 500)
    rx, ry = 2.2, 1.8
    cx, cy = 2.8, 2.5
    x = cx + rx*np.cos(t) + 0.5*np.cos(2*t)
    y = cy + ry*np.sin(t) + 0.4*np.sin(2*t)
    ax.fill(x, y, color=LBLUE, alpha=0.4)
    ax.plot(x, y, color=BLUE, lw=1.2)

    front_t = np.linspace(np.pi*0.68, np.pi*1.12, 100)
    fx = cx + rx*np.cos(front_t) + 0.5*np.cos(2*front_t)
    fy = cy + ry*np.sin(front_t) + 0.4*np.sin(2*front_t)
    ax.plot(fx, fy, color=BLUE, lw=4)
    ax.text(cx+0.3, cy+0.2, '$\\mathcal{Y}$', fontsize=16, color=BLUE)

    # Draw two weight vectors (iso-lines of the weighted objective)
    for w1, label, col in [(0.3, '$w_1$', GRAY), (0.7, '$w_1$', GRAY)]:
        w2 = 1 - w1
        # iso-value line: w1*y1 + w2*y2 = const through Pareto point
        # find optimal point on frontier
        vals = w1*np.array(fx) + w2*np.array(fy)
        idx = np.argmin(vals)
        c = vals[idx]
        # draw line
        y1_range = np.linspace(0, 5.5, 10)
        y2_line = (c - w1*y1_range) / w2
        ax.plot(y1_range, y2_line, '--', color=col, lw=1, alpha=0.7)
        ax.plot(fx[idx], fy[idx], 'o', color=col, ms=8, zorder=5)

    ax.annotate('$w_1=0$', xy=(0.1, 4.5), fontsize=10, color=GRAY)
    ax.annotate('$w_1=1$', xy=(0.1, 0.3), fontsize=10, color=GRAY)
    ax.set_xlabel('$y_1$', fontsize=13)
    ax.set_ylabel('$y_2$', fontsize=13)
    ax.set_xlim(-0.3, 5.5); ax.set_ylim(-0.3, 5.0)
    ax.set_title('Weighted Sum Method', fontsize=12)
    plt.tight_layout()
    savefig("fig_weighted_sum.pdf")


# =============================================================================
# Fig 6 — Goal programming (p = 1, 2, inf)
# =============================================================================
def fig_goal_programming():
    fig, ax = plt.subplots(figsize=(5, 4.5))
    t = np.linspace(0, 2*np.pi, 500)
    rx, ry = 2.2, 1.8
    cx, cy = 2.8, 2.5
    x = cx + rx*np.cos(t) + 0.5*np.cos(2*t)
    y = cy + ry*np.sin(t) + 0.4*np.sin(2*t)
    ax.fill(x, y, color=LBLUE, alpha=0.4)
    ax.plot(x, y, color=BLUE, lw=1.2)

    front_t = np.linspace(np.pi*0.68, np.pi*1.12, 100)
    fx = cx + rx*np.cos(front_t) + 0.5*np.cos(2*front_t)
    fy = cy + ry*np.sin(front_t) + 0.4*np.sin(2*front_t)
    ax.plot(fx, fy, color=BLUE, lw=3)

    ygoal = np.array([0.5, 0.4])
    ax.plot(*ygoal, 'k.', ms=10)
    ax.text(ygoal[0]-0.5, ygoal[1]-0.3, '$\\mathbf{y}^{\\mathrm{goal}}$', fontsize=11)

    # Find optimal for p=1, 2, inf on frontier
    from scipy.optimize import minimize_scalar
    colors_p = {'$p=1$': RED, '$p=2$': GREEN, '$p=\\infty$': 'purple'}
    for (plabel, col), p_val in zip(colors_p.items(), [1, 2, 100]):
        best_idx = np.argmin([np.sum(np.abs(np.array([fx[i], fy[i]]) - ygoal)**p_val)**(1/p_val)
                              if p_val < 90 else np.max(np.abs(np.array([fx[i], fy[i]]) - ygoal))
                              for i in range(len(fx))])
        ax.plot(fx[best_idx], fy[best_idx], 'o', color=col, ms=9, zorder=5, label=plabel)

    ax.set_xlabel('$y_1$', fontsize=13)
    ax.set_ylabel('$y_2$', fontsize=13)
    ax.set_xlim(-0.3, 5.5); ax.set_ylim(-0.3, 5.0)
    ax.legend(fontsize=10, loc='upper right')
    ax.set_title('Goal Programming ($L_p$ norm to $\\mathbf{y}^{\\mathrm{goal}}$)', fontsize=11)
    plt.tight_layout()
    savefig("fig_goal_programming.pdf")


# =============================================================================
# Fig 7 — Nondomination levels (NSGA-II style)
# =============================================================================
def fig_nondomination_levels():
    np.random.seed(7)
    # Generate random 2D points
    n = 60
    ys = np.random.rand(n, 2) * 4 + 0.2

    def dominates(a, b):
        return np.all(a <= b) and np.any(a < b)

    def nondom_levels(ys):
        m = len(ys)
        levels = np.zeros(m, dtype=int)
        L = 0
        while np.any(levels == 0):
            L += 1
            for i in range(m):
                if levels[i] == 0:
                    dominated = False
                    for j in range(m):
                        if (levels[j] == 0 or levels[j] == L) and j != i:
                            if dominates(ys[j], ys[i]):
                                dominated = True
                                break
                    if not dominated:
                        levels[i] = L
        return levels

    levels = nondom_levels(ys)
    max_level = levels.max()

    fig, ax = plt.subplots(figsize=(5, 4))
    cmap = plt.cm.Blues_r
    for lvl in range(1, min(max_level+1, 11)):
        mask = levels == lvl
        color = cmap(0.1 + 0.7*(lvl-1)/(max(max_level-1, 1)))
        ax.scatter(ys[mask, 0], ys[mask, 1], color=color, s=40, zorder=3)
        # Connect nondom front
        pts = ys[mask]
        pts = pts[np.argsort(pts[:, 0])]
        ax.plot(pts[:, 0], pts[:, 1], color=color, lw=1, alpha=0.6)

    # Annotate level 1, 2, max
    for lvl, label in [(1, 'Level 1'), (2, 'Level 2'), (max_level, f'Level {max_level}')]:
        mask = levels == lvl
        if mask.sum() > 0:
            mx, my = ys[mask].mean(axis=0)
            color = cmap(0.1 + 0.7*(lvl-1)/(max(max_level-1, 1)))
            ax.text(mx+0.1, my+0.15, label, color=color, fontsize=9)

    ax.set_xlabel('$y_1$', fontsize=13)
    ax.set_ylabel('$y_2$', fontsize=13)
    ax.set_title('Nondomination Levels', fontsize=12)
    plt.tight_layout()
    savefig("fig_nondomination_levels.pdf")


# =============================================================================
# Fig 8 — NSGA-II evolution (4 generations)
# =============================================================================
def fig_nsga_evolution():
    np.random.seed(0)
    # Circle function: minimize [||x||, ||x - [1,1]||]
    def circle_f(X):
        return np.array([np.linalg.norm(X), np.linalg.norm(X - np.array([1.0, 1.0]))])

    def dominates(a, b):
        return np.all(a <= b) and np.any(a < b)

    # true Pareto: points on the segment between [0,0] and [1,1]
    t_pareto = np.linspace(0, 1, 200)
    pareto_y1 = np.sqrt(2)*t_pareto
    pareto_y2 = np.sqrt(2)*(1 - t_pareto)

    fig, axes = plt.subplots(1, 4, figsize=(12, 3))
    pop = np.random.rand(40, 2) * 2 - 0.5

    for gen, ax in enumerate(axes):
        ys = np.array([circle_f(p) for p in pop])
        ax.plot(pareto_y1, pareto_y2, color=BLUE, lw=2, label='Pareto front')
        ax.scatter(ys[:, 0], ys[:, 1], c='k', s=25, zorder=3)
        ax.set_title(f'Gen {gen+1}', fontsize=11)
        ax.set_xlabel('$y_1$')
        ax.set_ylabel('$y_2$')
        ax.set_xlim(-0.1, 2.1); ax.set_ylim(-0.1, 2.1)

        # Simple selection+mutation toward Pareto
        for i in range(len(pop)):
            if np.random.rand() < 0.5:
                pop[i] += np.random.randn(2) * 0.15
        pop = np.clip(pop, -0.5, 1.8)

    plt.suptitle('Vector Evaluated GA: 4 Generations', fontsize=12)
    plt.tight_layout()
    savefig("fig_nsga_evolution.pdf")


# =============================================================================
# Fig 9 — Pareto filter
# =============================================================================
def fig_pareto_filter():
    np.random.seed(42)
    # Circle function
    def f_circle(x):
        return np.array([np.linalg.norm(x), np.linalg.norm(x - np.array([1.0, 1.0]))])

    def dominates(a, b):
        return np.all(a <= b) and np.any(a < b)

    pop = np.random.rand(50, 2) * 2
    ys = np.array([f_circle(p) for p in pop])

    # naive pareto
    pareto_idx = [i for i in range(len(ys))
                  if not any(dominates(ys[j], ys[i]) for j in range(len(ys)) if j != i)]

    t = np.linspace(0, 1, 200)
    pareto_true_y1 = np.sqrt(2)*t
    pareto_true_y2 = np.sqrt(2)*(1 - t)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(pareto_true_y1, pareto_true_y2, color=BLUE, lw=2.5, label='True Pareto front')
    ax.scatter(ys[:, 0], ys[:, 1], c='k', s=20, alpha=0.5, label='Population')
    ax.scatter(ys[pareto_idx, 0], ys[pareto_idx, 1], c=RED, s=60, zorder=4,
               label='Pareto filter')
    pts = ys[pareto_idx]
    pts = pts[np.argsort(pts[:, 0])]
    ax.plot(pts[:, 0], pts[:, 1], '-', color=RED, lw=1.5, zorder=3)
    ax.set_xlabel('$y_1$', fontsize=13)
    ax.set_ylabel('$y_2$', fontsize=13)
    ax.set_xlim(-0.1, 2.0); ax.set_ylim(-0.1, 2.0)
    ax.legend(fontsize=9, loc='upper right')
    ax.set_title('Pareto Filter', fontsize=12)
    plt.tight_layout()
    savefig("fig_pareto_filter.pdf")


# =============================================================================
# Fig 10 — Fitness sharing (niche techniques)
# =============================================================================
def fig_fitness_sharing():
    np.random.seed(5)
    def f_circle(x):
        return np.array([np.linalg.norm(x), np.linalg.norm(x - np.array([1.0, 1.0]))])

    def dominates(a, b):
        return np.all(a <= b) and np.any(a < b)

    pop = np.random.rand(60, 2) * 1.6
    ys = np.array([f_circle(p) for p in pop])

    sigma = 0.3
    # fitness sharing: penalize crowded points
    shared_ys = ys.copy()
    for i in range(len(ys)):
        neighbors = sum(1 for j in range(len(ys))
                        if j != i and np.linalg.norm(ys[i] - ys[j]) < sigma)
        shared_ys[i] = ys[i] * (1 + 0.15 * neighbors)

    pareto_idx = [i for i in range(len(shared_ys))
                  if not any(dominates(shared_ys[j], shared_ys[i])
                             for j in range(len(shared_ys)) if j != i)]

    t = np.linspace(0, 1, 200)
    py1 = np.sqrt(2)*t; py2 = np.sqrt(2)*(1-t)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(py1, py2, color=BLUE, lw=2.5, label='True front')
    ax.scatter(ys[:, 0], ys[:, 1], c='k', s=20, alpha=0.4)
    ax.scatter(ys[pareto_idx, 0], ys[pareto_idx, 1], c=GREEN, s=55, zorder=4,
               label='After fitness sharing')
    pts = ys[pareto_idx]; pts = pts[np.argsort(pts[:, 0])]
    ax.plot(pts[:, 0], pts[:, 1], '-', color=GREEN, lw=1.5)
    ax.set_xlabel('$y_1$', fontsize=13); ax.set_ylabel('$y_2$', fontsize=13)
    ax.set_xlim(-0.1, 1.8); ax.set_ylim(-0.1, 1.8)
    ax.legend(fontsize=9)
    ax.set_title('Fitness Sharing (Niche Technique)', fontsize=12)
    plt.tight_layout()
    savefig("fig_fitness_sharing.pdf")


# =============================================================================
# Fig 11 — Preference elicitation weight polytope
# =============================================================================
def fig_preference_elicitation():
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))

    for ax, (title, extra) in zip(axes, [
        ('Initial $\\mathcal{W}$', False),
        ('After 1 query', True),
        ('After 2 queries', True),
    ]):
        w1 = np.linspace(0, 1, 300)
        w2 = 1 - w1
        ax.fill_between(w1, 0, w2, color=LBLUE, alpha=0.6)
        ax.plot(w1, w2, color=BLUE, lw=2)
        ax.set_xlim(0, 1.1); ax.set_ylim(0, 1.1)
        ax.set_xlabel('$w_1$', fontsize=12); ax.set_ylabel('$w_2$', fontsize=12)
        ax.set_title(title, fontsize=11)

        if extra:
            # Add a constraint line
            slope = np.random.uniform(-1, -0.4)
            intercept = np.random.uniform(0.4, 0.8)
            w1_line = np.linspace(0, 1, 100)
            w2_line = slope * w1_line + intercept
            mask = (w2_line >= 0) & (w2_line <= 1)
            ax.plot(w1_line[mask], w2_line[mask], '--', color=RED, lw=1.5)

        ax.text(0.35, 0.35, '$\\mathcal{W}$', fontsize=16, color=BLUE)

    plt.suptitle('Preference Elicitation: Reducing Weight Polytope', fontsize=12)
    plt.tight_layout()
    savefig("fig_preference_elicitation.pdf")


# =============================================================================
# Fig 12 — Exercise 15.3 example: {[1,2],[2,1],[2,2],[1,1]}
# =============================================================================
def fig_exercise_example():
    points = np.array([[1,2],[2,1],[2,2],[1,1]])
    labels = ['[1,2]','[2,1]','[2,2]','[1,1]']

    def dominates(a, b):
        return np.all(a <= b) and np.any(a < b)

    colors = []
    for i, p in enumerate(points):
        dom = any(dominates(points[j], p) for j in range(len(points)) if j != i)
        if not dom:
            # check weakly
            weak = any(np.all(points[j] < p) for j in range(len(points)) if j != i)
            colors.append(BLUE if not weak else GREEN)
        else:
            colors.append('k')

    fig, ax = plt.subplots(figsize=(4, 4))
    for i, (p, lbl, col) in enumerate(zip(points, labels, colors)):
        ax.plot(p[0], p[1], 'o', color=col, ms=10, zorder=4)
        ax.text(p[0]+0.05, p[1]+0.08, lbl, fontsize=10)

    # connect Pareto front
    pareto = [p for p, c in zip(points, colors) if c == BLUE]
    pareto_arr = np.array(sorted(pareto, key=lambda p: p[0]))
    ax.plot(pareto_arr[:,0], pareto_arr[:,1], '-', color=BLUE, lw=2)

    ax.set_xlabel('$y_1$', fontsize=13); ax.set_ylabel('$y_2$', fontsize=13)
    ax.set_xlim(0.5, 3); ax.set_ylim(0.5, 3)
    ax.set_title('Exercise 15.3: Pareto-Optimal Points', fontsize=11)
    legend_handles = [
        mpatches.Patch(color=BLUE, label='Pareto optimal'),
        mpatches.Patch(color=GREEN, label='Weakly Pareto optimal'),
        mpatches.Patch(color='k', label='Dominated'),
    ]
    ax.legend(handles=legend_handles, fontsize=8, loc='upper right')
    plt.tight_layout()
    savefig("fig_exercise_example.pdf")


# =============================================================================
# Fig 13 — Exercise 15.7: Pareto curve for [x^2, (x-2)^2]
# =============================================================================
def fig_exercise_pareto_curve():
    c_vals = np.linspace(0, 4, 300)
    y1 = c_vals          # = c (= x^2 at optimum)
    y2 = np.where(c_vals >= 4, 0, (np.sqrt(c_vals) - 2)**2)

    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    ax.plot(y1, y2, color=BLUE, lw=2.5)
    ax.set_xlabel('$y_1 = x^2$', fontsize=13)
    ax.set_ylabel('$y_2 = (x-2)^2$', fontsize=13)
    ax.set_title('Ex. 15.7: Pareto Curve via Constraint Method', fontsize=11)
    ax.set_xlim(0, 4.2); ax.set_ylim(-0.1, 4.5)
    plt.tight_layout()
    savefig("fig_exercise_pareto_curve.pdf")


# =============================================================================
# Fig 14 — Weighted min-max method illustration
# =============================================================================
def fig_weighted_minmax():
    fig, ax = plt.subplots(figsize=(5, 4.5))
    t = np.linspace(0, 2*np.pi, 500)
    rx, ry = 2.2, 1.8
    cx, cy = 2.8, 2.5
    x = cx + rx*np.cos(t) + 0.5*np.cos(2*t)
    y = cy + ry*np.sin(t) + 0.4*np.sin(2*t)
    ax.fill(x, y, color=LBLUE, alpha=0.4)
    ax.plot(x, y, color=BLUE, lw=1.2)

    front_t = np.linspace(np.pi*0.68, np.pi*1.12, 100)
    fx = cx + rx*np.cos(front_t) + 0.5*np.cos(2*front_t)
    fy = cy + ry*np.sin(front_t) + 0.4*np.sin(2*front_t)
    ax.plot(fx, fy, color=BLUE, lw=4)

    ygoal = np.array([0.5, 0.4])
    ax.plot(*ygoal, 'k.', ms=10)
    ax.text(ygoal[0]-0.5, ygoal[1]-0.3, '$\\mathbf{y}^{\\mathrm{goal}}$', fontsize=11)

    # Chebyshev contours: max(w_i*(f_i - y_goal_i)) = const
    ws = [0.5, 0.5]
    for rho in [0.8, 1.5, 2.2]:
        # corners of Chebyshev diamond
        pts_x = [ygoal[0] + rho/ws[0], ygoal[0], ygoal[0] - rho/ws[0], ygoal[0]]
        pts_y = [ygoal[1], ygoal[1] + rho/ws[1], ygoal[1], ygoal[1] - rho/ws[1]]
        pts_x.append(pts_x[0]); pts_y.append(pts_y[0])
        ax.plot(pts_x, pts_y, '--', color=GRAY, lw=1, alpha=0.5)

    ax.set_xlabel('$y_1$', fontsize=13); ax.set_ylabel('$y_2$', fontsize=13)
    ax.set_xlim(-0.3, 5.5); ax.set_ylim(-0.3, 5.0)
    ax.text(cx+0.3, cy+0.2, '$\\mathcal{Y}$', fontsize=16, color=BLUE)
    ax.set_title('Weighted Min-Max (Tchebycheff) Method', fontsize=11)
    plt.tight_layout()
    savefig("fig_weighted_minmax.pdf")


# =============================================================================
# run all
# =============================================================================
if __name__ == "__main__":
    print("Generating figures for Chapter 15: Multiobjective Optimization")
    fig_dominance()
    fig_pareto_frontier()
    fig_collision_avoidance()
    fig_constraint_method()
    fig_weighted_sum()
    fig_goal_programming()
    fig_nondomination_levels()
    fig_nsga_evolution()
    fig_pareto_filter()
    fig_fitness_sharing()
    fig_preference_elicitation()
    fig_exercise_example()
    fig_exercise_pareto_curve()
    fig_weighted_minmax()
    print("All figures saved to", FIGDIR)
