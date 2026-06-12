"""
gen_figures.py  –  Chapter 9: Population Methods
Generates all figures needed for chapter09_slides.tex.
Requires: matplotlib, numpy, scipy, pymupdf (fitz)
Run: conda run -n py313 python3 gen_figures.py
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

# ── output directory ───────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(SCRIPT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

def savefig(name):
    path = os.path.join(FIG_DIR, name)
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  saved {path}")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 1 – Population iteration overview
# ══════════════════════════════════════════════════════════════════════════════
def fig_population_iteration():
    fig, ax = plt.subplots(figsize=(8, 2.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3)
    ax.axis("off")

    boxes = [
        (1.0, "Initial\nPopulation", "#4C72B0"),
        (3.5, "Evaluate\nObjective", "#55A868"),
        (6.0, "Generate\nNew Pop.", "#C44E52"),
        (8.5, "Converged?", "#8172B2"),
    ]
    for x, label, color in boxes:
        rect = mpatches.FancyBboxPatch((x - 0.85, 0.8), 1.7, 1.2,
                                        boxstyle="round,pad=0.1",
                                        linewidth=1.5, edgecolor="black",
                                        facecolor=color, alpha=0.85)
        ax.add_patch(rect)
        ax.text(x, 1.4, label, ha="center", va="center",
                fontsize=9, color="white", fontweight="bold")

    for x in [1.85, 4.35, 6.85]:
        ax.annotate("", xy=(x + 0.3, 1.4), xytext=(x, 1.4),
                    arrowprops=dict(arrowstyle="->", lw=1.5))

    # feedback arrow
    ax.annotate("", xy=(1.0, 0.8), xytext=(8.5, 0.8),
                arrowprops=dict(arrowstyle="->", lw=1.5,
                                connectionstyle="arc3,rad=-0.3",
                                color="#888888"))
    ax.text(4.75, 0.15, "not converged", ha="center", fontsize=8, color="#555555")
    ax.text(9.35, 1.4, "stop", ha="left", fontsize=9)

    fig.suptitle("Population Iteration Loop", fontsize=11, fontweight="bold")
    savefig("fig_population_iteration.pdf")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 2 – Selection methods (truncation / tournament / roulette)
# ══════════════════════════════════════════════════════════════════════════════
def fig_selection_methods():
    rng = np.random.default_rng(42)
    m = 7
    y_vals = rng.uniform(1, 5, m)
    colors = plt.cm.viridis(np.linspace(0, 1, m))
    x = np.arange(m)

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2))
    titles = ["Truncation (k=3)", "Tournament (k=3)", "Roulette Wheel"]

    for ax, title in zip(axes, titles):
        ax.bar(x, y_vals, color=colors, edgecolor="k", linewidth=0.6)
        ax.set_xlabel("individual", fontsize=9)
        ax.set_ylabel("y", fontsize=9)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xticks([])

    # Truncation: highlight top 3
    top3 = np.argsort(y_vals)[:3]
    for i in top3:
        axes[0].bar(i, y_vals[i], color=colors[i], edgecolor="red",
                    linewidth=2.0)

    # Tournament: highlight random subsets
    subset = rng.choice(m, 3, replace=False)
    winner = subset[np.argmin(y_vals[subset])]
    for i in subset:
        axes[1].bar(i, y_vals[i], color=colors[i], edgecolor="navy",
                    linewidth=1.5)
    axes[1].bar(winner, y_vals[winner], color=colors[winner], edgecolor="red",
                linewidth=2.5)

    # Roulette: fitness-proportional (minimization: invert)
    fitness = np.max(y_vals) - y_vals
    fitness = fitness / fitness.sum()
    axes[2].bar(x, fitness, color=colors, edgecolor="k", linewidth=0.6)
    axes[2].set_ylabel("likelihood", fontsize=9)

    plt.tight_layout()
    savefig("fig_selection_methods.pdf")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 3 – Crossover schemes
# ══════════════════════════════════════════════════════════════════════════════
def fig_crossover_schemes():
    n = 30
    a = np.zeros(n)   # parent A = blue
    b = np.ones(n)    # parent B = red

    fig, axes = plt.subplots(3, 1, figsize=(9, 4.5))
    titles = ["Single-Point Crossover", "Two-Point Crossover", "Uniform Crossover"]
    labels = ["parent A", "parent B", "child"]
    blue = "#4C72B0"
    red = "#C44E52"

    # Single-point
    cp = 18
    c1 = np.concatenate([a[:cp], b[cp:]])
    for row_i, (arr, lbl) in enumerate([(a, "parent A"), (b, "parent B"), (c1, "child")]):
        cols = [blue if v == 0 else red for v in arr]
        for j, col in enumerate(cols):
            axes[0].add_patch(plt.Circle((j, 2 - row_i), 0.38, color=col))
    axes[0].axvline(cp - 0.5, color="black", linewidth=1.5, linestyle="--")
    axes[0].text(cp - 0.5, -0.8, "crossover point", ha="center", fontsize=8)

    # Two-point
    cp1, cp2 = 10, 22
    c2 = np.concatenate([a[:cp1], b[cp1:cp2], a[cp2:]])
    for row_i, arr in enumerate([a, b, c2]):
        cols = [blue if v == 0 else red for v in arr]
        for j, col in enumerate(cols):
            axes[1].add_patch(plt.Circle((j, 2 - row_i), 0.38, color=col))
    axes[1].axvline(cp1 - 0.5, color="black", linewidth=1.5, linestyle="--")
    axes[1].axvline(cp2 - 0.5, color="black", linewidth=1.5, linestyle="--")

    # Uniform (p=0.5)
    rng = np.random.default_rng(7)
    mask = rng.random(n) > 0.5
    cu = np.where(mask, b, a)
    for row_i, arr in enumerate([a, b, cu]):
        cols = [blue if v == 0 else red for v in arr]
        for j, col in enumerate(cols):
            axes[2].add_patch(plt.Circle((j, 2 - row_i), 0.38, color=col))

    for ax, title in zip(axes, titles):
        ax.set_xlim(-1, n + 0.5)
        ax.set_ylim(-1.3, 3.0)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(title, fontsize=10, fontweight="bold", loc="left")
        for row_i, lbl in enumerate(labels):
            ax.text(-0.8, 2 - row_i, lbl, ha="right", va="center", fontsize=8)

    plt.tight_layout()
    savefig("fig_crossover_schemes.pdf")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 4 – Genetic algorithm evolution on Michalewicz function
# ══════════════════════════════════════════════════════════════════════════════
def michalewicz(x, m=10):
    d = len(x)
    return -sum(np.sin(x[i]) * np.sin((i + 1) * x[i] ** 2 / np.pi) ** (2 * m)
                for i in range(d))


def fig_ga_evolution():
    # 2D version: minimise ||x||
    f = lambda x: np.linalg.norm(x)
    rng = np.random.default_rng(0)

    pop_size = 30
    n_gen = 4
    lo, hi = -3.0, 3.0

    # grid for background
    gx = np.linspace(lo, hi, 120)
    gy = np.linspace(lo, hi, 120)
    GX, GY = np.meshgrid(gx, gy)
    Z = np.sqrt(GX**2 + GY**2)

    pop = rng.uniform(lo, hi, (pop_size, 2))
    k_trunc = 10

    fig, axes = plt.subplots(1, n_gen, figsize=(11, 3))

    for gen_i, ax in enumerate(axes):
        ax.contourf(GX, GY, Z, levels=20, cmap="viridis", alpha=0.7)
        ax.scatter(pop[:, 0], pop[:, 1], c="black", s=20, zorder=5)
        ax.set_title(f"Gen {gen_i}", fontsize=9)
        ax.set_xlabel("$x_1$", fontsize=8)
        if gen_i == 0:
            ax.set_ylabel("$x_2$", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)

        # evolve: truncation selection + single-point crossover + Gaussian mutation
        if gen_i < n_gen - 1:
            y = np.array([f(p) for p in pop])
            parents_idx = np.argsort(y)[:k_trunc]
            new_pop = []
            while len(new_pop) < pop_size:
                a_idx, b_idx = rng.choice(parents_idx, 2, replace=False)
                cp = rng.integers(1, 2)
                child = np.concatenate([pop[a_idx, :cp], pop[b_idx, cp:]])
                child += rng.normal(0, 0.3, child.shape)
                child = np.clip(child, lo, hi)
                new_pop.append(child)
            pop = np.array(new_pop)

    plt.suptitle("Genetic Algorithm on $f(\\mathbf{x})=\\|\\mathbf{x}\\|$",
                 fontsize=10, fontweight="bold")
    plt.tight_layout()
    savefig("fig_ga_evolution.pdf")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 5 – Differential evolution steps
# ══════════════════════════════════════════════════════════════════════════════
def ackley(x):
    n = len(x)
    a, b, c = 20, 0.2, 2 * np.pi
    s1 = np.sum(x**2) / n
    s2 = np.sum(np.cos(c * x)) / n
    return -a * np.exp(-b * np.sqrt(s1)) - np.exp(s2) + a + np.e


def fig_differential_evolution():
    rng = np.random.default_rng(3)
    f = lambda x: ackley(x)

    lo, hi = -4.0, 4.0
    gx = np.linspace(lo, hi, 120)
    gy = np.linspace(lo, hi, 120)
    GX, GY = np.meshgrid(gx, gy)
    Z = np.array([[ackley(np.array([GX[i, j], GY[i, j]]))
                   for j in range(120)] for i in range(120)])

    pop_size = 20
    p = 0.5
    w = 0.5
    n_gen = 4

    pop = rng.uniform(lo, hi, (pop_size, 2))

    fig, axes = plt.subplots(1, n_gen, figsize=(11, 3))
    for gen_i, ax in enumerate(axes):
        ax.contourf(GX, GY, Z, levels=20, cmap="plasma", alpha=0.65)
        ax.scatter(pop[:, 0], pop[:, 1], c="black", s=20, zorder=5)
        ax.set_title(f"Gen {gen_i}", fontsize=9)
        ax.set_xlabel("$x_1$", fontsize=8)
        if gen_i == 0:
            ax.set_ylabel("$x_2$", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)

        if gen_i < n_gen - 1:
            new_pop = np.copy(pop)
            for i in range(pop_size):
                idxs = [j for j in range(pop_size) if j != i]
                a_i, b_i, c_i = rng.choice(idxs, 3, replace=False)
                interim = pop[a_i] + w * (pop[b_i] - pop[c_i])
                mask = rng.random(2) < p
                if not mask.any():
                    mask[rng.integers(2)] = True
                child = np.where(mask, interim, pop[i])
                if f(child) < f(pop[i]):
                    new_pop[i] = child
            pop = new_pop

    plt.suptitle("Differential Evolution on Ackley's Function",
                 fontsize=10, fontweight="bold")
    plt.tight_layout()
    savefig("fig_differential_evolution.pdf")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 6 – PSO trajectory
# ══════════════════════════════════════════════════════════════════════════════
def wheelers_ridge(x):
    return -np.exp(-(x[0] - 1)**2 - (x[1] - 1)**2) \
           - np.exp(-(x[0] + 1)**2 - (x[1] + 1)**2) \
           + 2.0

def fig_pso():
    rng = np.random.default_rng(11)
    lo, hi = -3.0, 3.0
    gx = np.linspace(lo, hi, 100)
    gy = np.linspace(lo, hi, 100)
    GX, GY = np.meshgrid(gx, gy)
    Z = np.array([[wheelers_ridge(np.array([GX[i, j], GY[i, j]]))
                   for j in range(100)] for i in range(100)])

    n_particles = 20
    w_inertia = 0.1
    c1 = 0.25
    c2 = 2.0
    n_gen = 4

    pos = rng.uniform(lo, hi, (n_particles, 2))
    vel = rng.normal(0, 0.5, (n_particles, 2))
    x_best = pos.copy()
    y_best = np.array([wheelers_ridge(p) for p in pos])
    global_best_idx = np.argmin(y_best)
    global_best = x_best[global_best_idx].copy()

    fig, axes = plt.subplots(1, n_gen, figsize=(11, 3))

    for gen_i, ax in enumerate(axes):
        ax.contourf(GX, GY, Z, levels=20, cmap="viridis", alpha=0.65)
        ax.scatter(pos[:, 0], pos[:, 1], c="black", s=20, zorder=5)
        ax.scatter([global_best[0]], [global_best[1]], c="red",
                   s=60, marker="*", zorder=6)
        ax.set_title(f"Gen {gen_i}", fontsize=9)
        ax.set_xlabel("$x_1$", fontsize=8)
        if gen_i == 0:
            ax.set_ylabel("$x_2$", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)

        if gen_i < n_gen - 1:
            r1 = rng.random((n_particles, 2))
            r2 = rng.random((n_particles, 2))
            vel = (w_inertia * vel
                   + c1 * r1 * (x_best - pos)
                   + c2 * r2 * (global_best - pos))
            pos = pos + vel
            pos = np.clip(pos, lo, hi)
            for i in range(n_particles):
                y = wheelers_ridge(pos[i])
                if y < y_best[i]:
                    y_best[i] = y
                    x_best[i] = pos[i].copy()
            global_best_idx = np.argmin(y_best)
            global_best = x_best[global_best_idx].copy()

    plt.suptitle("Particle Swarm Optimization on Wheeler's Ridge",
                 fontsize=10, fontweight="bold")
    plt.tight_layout()
    savefig("fig_pso_evolution.pdf")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 7 – Firefly algorithm
# ══════════════════════════════════════════════════════════════════════════════
def fig_firefly():
    rng = np.random.default_rng(5)
    lo, hi = -3.0, 3.0
    f = lambda x: np.sum(x**2)
    alpha = 0.3
    beta0 = 1.0
    gamma = 1.0

    gx = np.linspace(lo, hi, 80)
    gy = np.linspace(lo, hi, 80)
    GX, GY = np.meshgrid(gx, gy)
    Z = GX**2 + GY**2

    n = 20
    pop = rng.uniform(lo, hi, (n, 2))
    n_gen = 4

    fig, axes = plt.subplots(1, n_gen, figsize=(11, 3))
    for gen_i, ax in enumerate(axes):
        ax.contourf(GX, GY, Z, levels=15, cmap="YlOrRd_r", alpha=0.7)
        y_vals = np.array([f(p) for p in pop])
        sizes = 80 * (1 - y_vals / y_vals.max()) + 10
        ax.scatter(pop[:, 0], pop[:, 1], c="gold", s=sizes,
                   edgecolors="orange", zorder=5)
        ax.set_title(f"Gen {gen_i}", fontsize=9)
        ax.set_xlabel("$x_1$", fontsize=8)
        if gen_i == 0:
            ax.set_ylabel("$x_2$", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)

        if gen_i < n_gen - 1:
            new_pop = pop.copy()
            for i in range(n):
                for j in range(n):
                    if f(pop[j]) < f(pop[i]):
                        r = np.linalg.norm(pop[i] - pop[j])
                        beta = beta0 * np.exp(-gamma * r**2)
                        new_pop[i] = (new_pop[i]
                                      + beta * (pop[j] - new_pop[i])
                                      + alpha * rng.normal(0, 1, 2))
            pop = np.clip(new_pop, lo, hi)

    plt.suptitle("Firefly Algorithm on $f(\\mathbf{x})=\\|\\mathbf{x}\\|^2$",
                 fontsize=10, fontweight="bold")
    plt.tight_layout()
    savefig("fig_firefly_evolution.pdf")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 8 – Cuckoo search
# ══════════════════════════════════════════════════════════════════════════════
def rosenbrock(x):
    return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2


def fig_cuckoo_search():
    rng = np.random.default_rng(99)
    lo, hi = -2.5, 2.5
    gx = np.linspace(lo, hi, 100)
    gy = np.linspace(lo, hi, 100)
    GX, GY = np.meshgrid(gx, gy)
    Z = np.array([[rosenbrock(np.array([GX[i, j], GY[i, j]]))
                   for j in range(100)] for i in range(100)])
    Z_log = np.log1p(Z)

    n = 15
    p_abandon = 0.25
    pop = [(rng.uniform(lo, hi, 2), rosenbrock(rng.uniform(lo, hi, 2)))
           for _ in range(n)]
    pop = [(x, rosenbrock(x)) for x, _ in pop]
    pop.sort(key=lambda t: t[1])

    n_gen = 4
    fig, axes = plt.subplots(1, n_gen, figsize=(11, 3))

    for gen_i, ax in enumerate(axes):
        ax.contourf(GX, GY, Z_log, levels=20, cmap="plasma", alpha=0.65)
        pts = np.array([t[0] for t in pop])
        ax.scatter(pts[:, 0], pts[:, 1], c="white", s=25, zorder=5,
                   edgecolors="black", linewidths=0.5)
        ax.set_title(f"Gen {gen_i}", fontsize=9)
        ax.set_xlabel("$x_1$", fontsize=8)
        if gen_i == 0:
            ax.set_ylabel("$x_2$", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)

        if gen_i < n_gen - 1:
            # Levy flight step
            new_pop = []
            for i in range(n):
                j = rng.integers(n)
                # Cauchy flight
                step = rng.standard_cauchy(2) * 0.3
                x_new = np.clip(pop[j][0] + step, lo, hi)
                y_new = rosenbrock(x_new)
                if y_new < pop[i][1]:
                    new_pop.append((x_new, y_new))
                else:
                    new_pop.append(pop[i])
            # abandon fraction
            m_abandon = max(1, int(p_abandon * n))
            new_pop.sort(key=lambda t: t[1])
            for i in range(n - m_abandon, n):
                x_new = np.clip(rng.uniform(lo, hi, 2), lo, hi)
                new_pop[i] = (x_new, rosenbrock(x_new))
            pop = sorted(new_pop, key=lambda t: t[1])

    plt.suptitle("Cuckoo Search on Rosenbrock's Function",
                 fontsize=10, fontweight="bold")
    plt.tight_layout()
    savefig("fig_cuckoo_search.pdf")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 9 – Hybrid methods: Lamarckian vs Baldwinian
# ══════════════════════════════════════════════════════════════════════════════
def fig_hybrid_methods():
    f = lambda x: -np.exp(-x**2) - 2 * np.exp(-(x - 3)**2)
    x_arr = np.linspace(-3, 6, 400)
    y_arr = f(x_arr)

    # population near x=0
    rng = np.random.default_rng(2)
    pop = rng.normal(0.3, 0.5, 7)
    pop_y = f(pop)

    # local search destination (gradient descent towards local min near x~0)
    pop_ls = pop - 0.8 * (pop - 0.0) + rng.normal(0, 0.05, len(pop))
    pop_ls_y = f(pop_ls)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.5), sharey=True)

    for ax, title, do_move in [(ax1, "Lamarckian", True),
                                (ax2, "Baldwinian", False)]:
        ax.plot(x_arr, y_arr, "k-", linewidth=2, zorder=1)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("$x$", fontsize=10)
        ax.set_ylabel("$f(x)$", fontsize=10)
        ax.set_xlim(-3, 6)
        ax.set_ylim(-2.5, 0.5)

        if do_move:
            # Lamarckian: individuals physically moved to local min
            for xi, yi, xls, yls in zip(pop, pop_y, pop_ls, pop_ls_y):
                ax.plot(xls, yls, "ko", ms=7, zorder=5)
        else:
            # Baldwinian: individuals stay put, but scored by local min value
            for xi, yi, xls, yls in zip(pop, pop_y, pop_ls, pop_ls_y):
                ax.plot(xi, yi, "ko", ms=7, zorder=5)
                ax.plot(xls, yls, "o", color="#4C72B0", ms=5,
                        markerfacecolor="none", zorder=4)
                ax.plot([xi, xls], [yi, yls], color="#4C72B0",
                        linewidth=0.8, zorder=3, alpha=0.7)

    plt.tight_layout()
    savefig("fig_hybrid_methods.pdf")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 10 – DE variant comparison (best objective vs iteration)
# ══════════════════════════════════════════════════════════════════════════════
def fig_de_variants():
    rng = np.random.default_rng(42)
    n_iter = 20
    n_trials = 50
    pop_size = 20
    lo, hi = -5.0, 5.0

    def run_de(p, w, a_is_x):
        results = []
        for _ in range(n_trials):
            pop = rng.uniform(lo, hi, (pop_size, 2))
            best_hist = []
            for _ in range(n_iter):
                y = np.array([ackley(x) for x in pop])
                best_hist.append(y.min())
                new_pop = np.copy(pop)
                for i in range(pop_size):
                    idxs = [j for j in range(pop_size) if j != i]
                    a_i, b_i, c_i = rng.choice(idxs, 3, replace=False)
                    a_pt = pop[i] if a_is_x else pop[a_i]
                    interim = a_pt + w * (pop[b_i] - pop[c_i])
                    mask = rng.random(2) < p
                    if not mask.any():
                        mask[rng.integers(2)] = True
                    child = np.where(mask, interim, pop[i])
                    if ackley(child) < ackley(pop[i]):
                        new_pop[i] = child
                pop = new_pop
            results.append(best_hist)
        return np.array(results)

    configs = [
        (0.5, 0.0, False, "$p=0.5,w=0.0$", "#e74c3c"),
        (0.5, 0.5, True,  "$p=0.5,w=0.5,\\mathbf{a}=\\mathbf{x}$", "#f39c12"),
        (0.5, 0.5, False, "$p=0.5,w=0.5$", "#2ecc71"),
        (0.9, 0.5, False, "$p=0.9,w=0.5$", "#3498db"),
        (1.0, 0.5, False, "$p=1.0,w=0.5$", "#9b59b6"),
    ]

    fig, ax = plt.subplots(figsize=(7, 4))
    for p, w, a_x, label, color in configs:
        data = run_de(p, w, a_x)
        mean = data.mean(axis=0)
        std = data.std(axis=0)
        iters = np.arange(n_iter)
        ax.plot(iters, mean, color=color, linewidth=2, label=label)
        ax.fill_between(iters, mean - std, mean + std, color=color, alpha=0.15)

    ax.set_xlabel("iteration", fontsize=10)
    ax.set_ylabel("best objective function value", fontsize=10)
    ax.set_title("Differential Evolution Variants on Ackley's Function",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    savefig("fig_de_variants.pdf")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Generating figures for Chapter 9: Population Methods...")
    fig_population_iteration()
    fig_selection_methods()
    fig_crossover_schemes()
    fig_ga_evolution()
    fig_differential_evolution()
    fig_pso()
    fig_firefly()
    fig_cuckoo_search()
    fig_hybrid_methods()
    fig_de_variants()
    print("All figures saved to:", FIG_DIR)
