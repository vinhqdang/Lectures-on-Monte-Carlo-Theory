"""
gen_figures.py  —  Generate all figures for Chapter 1 (TSP) slides.
Run with:  conda run -n py313 python3 gen_figures.py
Output:    figures/*.pdf  (included by LaTeX)
"""

import os
import math
import itertools
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import matplotlib.patheffects as pe

# Make sure we save into the figures/ subdirectory relative to this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(SCRIPT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

def savefig(name, fig=None, dpi=150):
    path = os.path.join(FIG_DIR, name)
    if fig is None:
        plt.savefig(path, bbox_inches='tight', dpi=dpi)
        plt.close()
    else:
        fig.savefig(path, bbox_inches='tight', dpi=dpi)
        plt.close(fig)
    print(f"  Saved: {path}")

# ─────────────────────────────────────────────────────────────────────────────
# Fig 1 — 5-city TSP tour (Inverness, Aberdeen, Dundee, Edinburgh, Glasgow)
# ─────────────────────────────────────────────────────────────────────────────
def fig_5city_tour():
    cities = {
        'Inverness': (1.0, 4.0),
        'Aberdeen':  (2.5, 3.0),
        'Dundee':    (2.0, 2.0),
        'Edinburgh': (2.5, 0.5),
        'Glasgow':   (0.5, 0.5),
    }
    tour = ['Inverness', 'Aberdeen', 'Dundee', 'Edinburgh', 'Glasgow', 'Inverness']

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.set_xlim(-0.3, 3.2)
    ax.set_ylim(-0.3, 4.6)
    ax.axis('off')
    ax.set_aspect('equal')

    # Draw tour edges with arrows
    for i in range(len(tour)-1):
        x0, y0 = cities[tour[i]]
        x1, y1 = cities[tour[i+1]]
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle='->', color='steelblue', lw=1.5))

    # Draw nodes
    for name, (x, y) in cities.items():
        ax.plot(x, y, 'o', color='white', markeredgecolor='steelblue',
                markeredgewidth=2, markersize=10, zorder=5)
        ax.text(x + 0.13, y + 0.12, name, fontsize=9, va='bottom')

    ax.set_title("A valid 5-city TSP tour\n(Hamiltonian Circuit)", fontsize=10)
    savefig("fig_5city_tour.pdf")

# ─────────────────────────────────────────────────────────────────────────────
# Fig 2 — Factorial growth: number of permutations vs number of cities
# ─────────────────────────────────────────────────────────────────────────────
def fig_factorial_growth():
    ns = list(range(2, 21))
    perms = [math.factorial(n-1) / 2 for n in ns]   # (n-1)!/2  unique tours

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.semilogy(ns, perms, 'o-', color='steelblue', lw=2, markersize=5)
    ax.set_xlabel("Number of cities $n$", fontsize=11)
    ax.set_ylabel("Unique tours $(n-1)!/2$", fontsize=11)
    ax.set_title("Combinatorial explosion of TSP search space", fontsize=11)
    ax.grid(True, which='both', alpha=0.3)

    # annotate a few key values
    for n in [5, 10, 15, 20]:
        v = math.factorial(n-1)//2
        ax.annotate(f"n={n}\n{v:,}", xy=(n, v),
                    xytext=(n+0.4, v*1.5),
                    fontsize=7, color='darkblue',
                    arrowprops=dict(arrowstyle='->', color='gray', lw=0.8))

    plt.tight_layout()
    savefig("fig_factorial_growth.pdf")

# ─────────────────────────────────────────────────────────────────────────────
# Fig 3 — Distance matrix for a small 4-city example
# ─────────────────────────────────────────────────────────────────────────────
def fig_distance_matrix():
    cities = ['A', 'B', 'C', 'D']
    coords = np.array([[0.0, 0.0],
                       [3.0, 0.0],
                       [3.0, 2.0],
                       [0.0, 2.0]])

    n = len(cities)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            D[i, j] = np.linalg.norm(coords[i] - coords[j])

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))

    # Left: city map
    ax = axes[0]
    ax.scatter(coords[:, 0], coords[:, 1], s=200, color='steelblue', zorder=5)
    for i, c in enumerate(cities):
        ax.text(coords[i, 0] + 0.1, coords[i, 1] + 0.1, c, fontsize=13, fontweight='bold')
    # draw all edges lightly
    for i in range(n):
        for j in range(i+1, n):
            ax.plot([coords[i,0], coords[j,0]], [coords[i,1], coords[j,1]],
                    'k-', alpha=0.2, lw=1)
    ax.set_xlim(-0.5, 3.8)
    ax.set_ylim(-0.5, 2.8)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title("City coordinates", fontsize=10)

    # Right: distance matrix as table
    ax2 = axes[1]
    ax2.axis('off')
    table_data = [[f"{D[i,j]:.2f}" for j in range(n)] for i in range(n)]
    col_labels = cities
    row_labels = cities
    tbl = ax2.table(cellText=table_data,
                    rowLabels=row_labels,
                    colLabels=col_labels,
                    loc='center',
                    cellLoc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1.4, 1.8)
    # colour diagonal
    for i in range(n):
        tbl[(i+1, i)].set_facecolor('#d0e8f0')
    ax2.set_title("Distance matrix $D$", fontsize=10, pad=20)

    fig.suptitle("4-city TSP: coordinates and distance matrix", fontsize=11, y=1.02)
    plt.tight_layout()
    savefig("fig_distance_matrix.pdf")

# ─────────────────────────────────────────────────────────────────────────────
# Fig 4 — Nearest Neighbour heuristic step-by-step (7-city example)
# ─────────────────────────────────────────────────────────────────────────────
def fig_nn_steps():
    # City positions as in book Fig 1.2
    cities = {
        'a': (2.5, 4.0),
        'b': (3.5, 3.0),
        'c': (5.0, 4.0),
        'd': (5.5, 3.0),
        'e': (6.0, 2.0),
        'f': (3.0, 1.0),
        'g': (1.0, 2.0),
    }
    city_names = list(cities.keys())
    coords = {k: np.array(v) for k, v in cities.items()}

    # NN tour starting from E
    nn_tour = ['e', 'd', 'c', 'b', 'f', 'g', 'a']  # as shown in book

    steps = [
        # (title, visited, current, edges)
        ("(a) Problem: 7 cities", [], None, []),
        ("(b) Start at E (selected at random)", ['e'], 'e', []),
        ("(c) Nearest to E is D", ['e', 'd'], 'd', [('e','d')]),
        ("(d) Nearest unvisited: C, then B, F, G", ['e','d','c','b','f','g'], 'g', [('e','d'),('d','c'),('c','b'),('b','f'),('f','g')]),
        ("(e) Last unvisited: A added", ['e','d','c','b','f','g','a'], 'a', [('e','d'),('d','c'),('c','b'),('b','f'),('f','g'),('g','a')]),
        ("(f) Complete NN tour (sub-optimal)", ['e','d','c','b','f','g','a'], None, [('e','d'),('d','c'),('c','b'),('b','f'),('f','g'),('g','a'),('a','e')]),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()

    for idx, (title, visited, current, edges) in enumerate(steps):
        ax = axes[idx]
        ax.set_xlim(0, 7.5)
        ax.set_ylim(0, 5.2)
        ax.axis('off')
        ax.set_aspect('equal')
        ax.set_title(title, fontsize=8.5, pad=3)

        # Draw edges
        for (u, v) in edges:
            p0 = coords[u]
            p1 = coords[v]
            ax.plot([p0[0], p1[0]], [p0[1], p1[1]], 'steelblue', lw=1.5, zorder=2)

        # Draw nodes
        for name, pos in coords.items():
            color = '#aaaaaa' if name in visited else 'white'
            ec = 'steelblue'
            ax.plot(pos[0], pos[1], 'o', color=color,
                    markeredgecolor=ec, markeredgewidth=2, markersize=16, zorder=3)
            ax.text(pos[0], pos[1], name, ha='center', va='center',
                    fontsize=9, fontweight='bold', zorder=4)

    plt.suptitle("Nearest Neighbour Heuristic — Step by Step", fontsize=12, y=1.01)
    plt.tight_layout()
    savefig("fig_nn_steps.pdf")

# ─────────────────────────────────────────────────────────────────────────────
# Fig 5 — 2-opt swap illustration
# ─────────────────────────────────────────────────────────────────────────────
def fig_2opt_swap():
    # Tour ABEDCFG → after reversing EDC → ABCDEG (illustrative)
    cities_before = {
        'a': (2.5, 4.0),
        'b': (3.5, 3.0),
        'c': (5.0, 4.0),
        'd': (5.5, 3.0),
        'e': (6.0, 2.0),
        'f': (3.0, 1.0),
        'g': (1.0, 2.0),
    }
    coords = {k: np.array(v) for k, v in cities_before.items()}

    tour_before = ['a', 'b', 'e', 'd', 'c', 'f', 'g', 'a']   # sub-optimal
    tour_after  = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'a']   # after 2-opt

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))

    for ax, tour, title, color in [
        (axes[0], tour_before, "Before 2-opt swap\n(tour A-B-E-D-C-F-G)", 'tomato'),
        (axes[1], tour_after,  "After 2-opt swap\n(tour A-B-C-D-E-F-G)",  'steelblue'),
    ]:
        ax.set_xlim(0, 7.5)
        ax.set_ylim(0, 5.2)
        ax.axis('off')
        ax.set_aspect('equal')
        ax.set_title(title, fontsize=10)

        for i in range(len(tour)-1):
            p0 = coords[tour[i]]
            p1 = coords[tour[i+1]]
            ax.annotate("", xy=p1, xytext=p0,
                        arrowprops=dict(arrowstyle='->', color=color, lw=1.5))

        for name, pos in coords.items():
            ax.plot(pos[0], pos[1], 'o', color='white',
                    markeredgecolor=color, markeredgewidth=2, markersize=18, zorder=3)
            ax.text(pos[0], pos[1], name, ha='center', va='center',
                    fontsize=9, fontweight='bold', zorder=4)

    plt.suptitle("2-opt: reversing a segment to shorten the tour", fontsize=11)
    plt.tight_layout()
    savefig("fig_2opt_swap.pdf")

# ─────────────────────────────────────────────────────────────────────────────
# Fig 6 — Results comparison table (from book Tables 1.2 and 1.3)
# ─────────────────────────────────────────────────────────────────────────────
def fig_results_table():
    # Data from Table 1.3 (book p.20): visits, NearestN dist, TwoOpt dist, Hybrid dist
    visits = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,20,25,30,35,40,45,50]
    nn_dist  = [64.11,68.93,69.02,93.23,112.35,151.8,169.27,195.47,254.66,328.77,
                484.15,661.8,665.11,677.64,815.5,887.28,946.7,883.8,1032.79,1064.67,1236.75,
                None]
    two_dist = [64.11,68.93,69.14,93.87,111.04,148.53,193.92,217.22,288.79,356.88,
                418.87,438.6,556.84,571.55,566.39,703.19,764.83,868.17,889.38,963.47,
                993.1,1046.69]
    hyb_dist = [64.11,68.93,69.02,93.23,108.77,125.68,169.27,195.47,249.85,328.77,
                389.17,401.55,514.51,521.86,534.4,664.34,725.15,833.44,867.89,919.61,
                925.88,950.49]

    # Use only the subset with all three values
    v_plot = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,20,25,30,35,40,45,50]
    nn_p  = [64.11,68.93,69.02,93.23,112.35,151.8,169.27,195.47,254.66,328.77,
             484.15,661.8,665.11,677.64,815.5,887.28,946.7,883.8,1032.79,1064.67,1236.75,1236.75]
    two_p = [64.11,68.93,69.14,93.87,111.04,148.53,193.92,217.22,288.79,356.88,
             418.87,438.6,556.84,571.55,566.39,703.19,764.83,868.17,889.38,963.47,993.1,1046.69]
    hyb_p = [64.11,68.93,69.02,93.23,108.77,125.68,169.27,195.47,249.85,328.77,
             389.17,401.55,514.51,521.86,534.4,664.34,725.15,833.44,867.89,919.61,925.88,950.49]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(v_plot, nn_p,  'o--', color='tomato',     lw=2, markersize=5, label='Nearest Neighbour')
    ax.plot(v_plot, two_p, 's-',  color='steelblue',  lw=2, markersize=5, label='2-opt')
    ax.plot(v_plot, hyb_p, '^-',  color='green',      lw=2, markersize=5, label='Hybrid (NN + 2-opt)')
    ax.set_xlabel("Number of visits", fontsize=11)
    ax.set_ylabel("Tour distance (km)", fontsize=11)
    ax.set_title("Case study: tour distances by algorithm (Santa Claus dataset)", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    savefig("fig_results_comparison.pdf")

# ─────────────────────────────────────────────────────────────────────────────
# Fig 7 — NP-hardness / complexity landscape
# ─────────────────────────────────────────────────────────────────────────────
def fig_np_hardness():
    fig, ax = plt.subplots(figsize=(6, 3.5))

    # Draw nested set diagram
    from matplotlib.patches import Ellipse
    p_ellipse = Ellipse((0.5, 0.5), 0.35, 0.55, color='lightblue', alpha=0.7, zorder=2)
    np_ellipse = Ellipse((0.5, 0.5), 0.70, 0.80, color='lightyellow', alpha=0.7, zorder=1)
    npc_ellipse = Ellipse((0.72, 0.42), 0.32, 0.40, color='#ffd0d0', alpha=0.6, zorder=3)

    ax.add_patch(np_ellipse)
    ax.add_patch(p_ellipse)
    ax.add_patch(npc_ellipse)

    ax.text(0.50, 0.50, 'P', fontsize=14, ha='center', va='center', fontweight='bold', zorder=5)
    ax.text(0.28, 0.78, 'NP', fontsize=13, ha='center', va='center', fontweight='bold', zorder=5)
    ax.text(0.72, 0.42, 'NP-hard\n(incl. TSP)', fontsize=9, ha='center', va='center',
            fontweight='bold', zorder=5)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title("Complexity classes: TSP is NP-hard\n(assuming P $\\neq$ NP)", fontsize=11)

    # Add note
    ax.text(0.5, 0.03,
            "No known polynomial-time algorithm; best exact solvers use exponential time in the worst case.",
            ha='center', fontsize=8, style='italic', color='#444444')

    savefig("fig_np_hardness.pdf")

# ─────────────────────────────────────────────────────────────────────────────
# Fig 8 — Class hierarchy (software engineering)
# ─────────────────────────────────────────────────────────────────────────────
def fig_class_diagram():
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.axis('off')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)

    def draw_box(ax, x, y, w, h, title, methods, color='#d0e8f8'):
        rect = plt.Rectangle((x, y), w, h, linewidth=1.5, edgecolor='steelblue',
                              facecolor=color, zorder=3)
        ax.add_patch(rect)
        # title bar
        title_rect = plt.Rectangle((x, y+h-0.55), w, 0.55, linewidth=0,
                                   facecolor='steelblue', alpha=0.4, zorder=4)
        ax.add_patch(title_rect)
        ax.text(x + w/2, y+h-0.27, title, ha='center', va='center',
                fontsize=9, fontweight='bold', zorder=5)
        for i, m in enumerate(methods):
            ax.text(x + 0.12, y+h-0.85 - i*0.38, m, ha='left', va='center',
                    fontsize=7.5, zorder=5)

    # TSPProblem
    draw_box(ax, 0.2, 0.5, 2.8, 3.0, 'TSPProblem',
             ['currentSolution: List', 'start: Visit', '+setRoute(route)',
              '+addVisit(v)', '+getSize(): int', '+getDistance(): float'])

    # TSPSolver (abstract)
    draw_box(ax, 3.8, 1.5, 2.4, 2.0, '<<abstract>>\nTSPSolver',
             ['+solve()', '+setProblem(p)'],
             color='#fff0d0')

    # Subclasses
    draw_box(ax, 2.8, 0.2, 1.6, 1.0, 'NearestNeighbour', ['+solve()'], color='#d8f0d8')
    draw_box(ax, 5.0, 0.2, 1.4, 1.0, 'TwoOpt', ['+solve()'], color='#d8f0d8')
    draw_box(ax, 7.0, 0.2, 1.6, 1.0, 'Exhaustive', ['+solve()'], color='#d8f0d8')

    # Arrows: TSPProblem → TSPSolver (association)
    ax.annotate("", xy=(3.8, 2.5), xytext=(3.0, 2.5),
                arrowprops=dict(arrowstyle='->', color='steelblue', lw=1.5))

    # Arrows: subclasses inherit from TSPSolver
    for xc in [3.6, 5.7, 7.8]:
        ax.annotate("", xy=(xc if xc < 5 else 5.0 if xc < 6 else 5.2, 1.5),
                    xytext=(xc, 1.2),
                    arrowprops=dict(arrowstyle='-|>', color='steelblue', lw=1.5))

    ax.annotate("", xy=(3.6, 1.5), xytext=(3.6, 1.2),
                arrowprops=dict(arrowstyle='-|>', color='steelblue', lw=1.5))
    ax.annotate("", xy=(5.7, 1.5), xytext=(5.7, 1.2),
                arrowprops=dict(arrowstyle='-|>', color='steelblue', lw=1.5))
    ax.annotate("", xy=(7.0, 1.5), xytext=(7.8, 1.2),
                arrowprops=dict(arrowstyle='-|>', color='steelblue', lw=1.5))

    ax.set_title("Software design: separating problem from heuristic", fontsize=11, y=0.98)
    savefig("fig_class_diagram.pdf")

# ─────────────────────────────────────────────────────────────────────────────
# Fig 9 — Exhaustive search tree for small TSP
# ─────────────────────────────────────────────────────────────────────────────
def fig_exhaustive_tree():
    """Small partial search tree for a 4-city exhaustive search."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axis('off')
    ax.set_xlim(-0.5, 8.5)
    ax.set_ylim(-0.5, 5)

    def node(ax, x, y, label, color='white'):
        c = plt.Circle((x, y), 0.35, color=color, ec='steelblue', lw=1.5, zorder=3)
        ax.add_patch(c)
        ax.text(x, y, label, ha='center', va='center', fontsize=9, fontweight='bold', zorder=4)

    def edge(ax, x0, y0, x1, y1):
        ax.plot([x0, x1], [y0, y1], 'k-', lw=1, alpha=0.5, zorder=2)

    # Root
    node(ax, 4, 4.3, 'A')

    # Level 1: B, C, D
    l1 = [(1.5, 3.0, 'B'), (4.0, 3.0, 'C'), (6.5, 3.0, 'D')]
    for x, y, l in l1:
        edge(ax, 4, 4.3, x, y)
        node(ax, x, y, l)

    # Level 2
    l2 = [
        (0.5, 1.8, 'C'), (1.5, 1.8, 'D'),   # from B
        (3.0, 1.8, 'B'), (4.0, 1.8, 'D'),   # from C -- but skip some for clarity
        (5.8, 1.8, 'B'), (6.5, 1.8, 'C'),   # from D
    ]
    parents = [(1.5,3.0),(1.5,3.0),(4.0,3.0),(4.0,3.0),(6.5,3.0),(6.5,3.0)]
    for (x,y,l), (px,py) in zip(l2, parents):
        edge(ax, px, py, x, y)
        node(ax, x, y, l)

    # Level 3 (leaves) — just under B→C and B→D
    leaves = [(0.2, 0.7, 'D'), (0.8, 0.7, '…'),
              (1.2, 0.7, 'C'), (1.8, 0.7, '…'),
              (2.7, 0.7, '…'), (3.3, 0.7, '…'),
              (3.7, 0.7, '…'), (4.3, 0.7, '…'),
              (5.5, 0.7, '…'), (6.1, 0.7, '…'),
              (6.2, 0.7, '…'), (6.8, 0.7, '…')]
    leaf_parents = [(0.5,1.8),(0.5,1.8),(1.5,1.8),(1.5,1.8),
                    (3.0,1.8),(3.0,1.8),(4.0,1.8),(4.0,1.8),
                    (5.8,1.8),(5.8,1.8),(6.5,1.8),(6.5,1.8)]
    for (x,y,l),(px,py) in zip(leaves, leaf_parents):
        edge(ax, px, py, x, y)
        node(ax, x, y, l, color='#e8f8e8' if l != '…' else '#f5f5f5')

    ax.text(4, -0.2,
            "Each path root→leaf is a complete tour. With n cities there are (n-1)!/2 distinct tours.",
            ha='center', fontsize=9, style='italic', color='#333333')
    ax.set_title("Partial search tree for 4-city exhaustive search (starting from A)", fontsize=11)
    savefig("fig_exhaustive_tree.pdf")

# ─────────────────────────────────────────────────────────────────────────────
# Fig 10 — Hybrid algorithm flow
# ─────────────────────────────────────────────────────────────────────────────
def fig_hybrid_flow():
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.axis('off')
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 5)

    boxes = [
        (3.0, 4.3, "Start", '#d0e8f8'),
        (3.0, 3.4, "Generate initial solution\n(Nearest Neighbour)", '#d8f0d8'),
        (3.0, 2.4, "Apply 2-opt local search\nuntil no improvement", '#fff0d0'),
        (3.0, 1.4, "Record best tour\n& distance", '#fde8d8'),
        (3.0, 0.4, "Return best solution", '#d0e8f8'),
    ]
    for (x, y, label, color) in boxes:
        w, h = 3.0, 0.65
        rect = plt.Rectangle((x-w/2, y-h/2), w, h,
                              linewidth=1.5, edgecolor='steelblue',
                              facecolor=color, zorder=3, clip_on=False)
        ax.add_patch(rect)
        ax.text(x, y, label, ha='center', va='center', fontsize=9, zorder=4)

    # arrows between boxes
    ys = [b[1] for b in boxes]
    for i in range(len(ys)-1):
        ax.annotate("", xy=(3.0, ys[i+1]+0.33), xytext=(3.0, ys[i]-0.33),
                    arrowprops=dict(arrowstyle='->', color='steelblue', lw=1.5))

    ax.set_title("Hybrid algorithm: NN initialisation + 2-opt refinement", fontsize=10, y=1.0)
    savefig("fig_hybrid_flow.pdf")

# ─────────────────────────────────────────────────────────────────────────────
# Run all
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("Generating figures for Chapter 1 (TSP)...")
    fig_5city_tour()
    fig_factorial_growth()
    fig_distance_matrix()
    fig_nn_steps()
    fig_2opt_swap()
    fig_results_table()
    fig_np_hardness()
    fig_class_diagram()
    fig_exhaustive_tree()
    fig_hybrid_flow()
    print("Done. All figures written to", FIG_DIR)
