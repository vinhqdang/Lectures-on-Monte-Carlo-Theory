"""
gen_figures.py  –  Generate all figures for Chapter 4 slides
(Heuristics for the Vehicle Routing Problem)
Uses matplotlib (backend='Agg') + pymupdf for PDF crops.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import numpy as np
import os, sys

OUTDIR = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(OUTDIR, exist_ok=True)

def savefig(name):
    plt.savefig(os.path.join(OUTDIR, name), dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  saved {name}')

# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 – Clarke-Wright Savings Algorithm illustration
# ─────────────────────────────────────────────────────────────────────────────
def fig_clarke_wright():
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    depot = np.array([0, 0])
    customers = {
        1: np.array([2, 3]),
        2: np.array([4, 2]),
        3: np.array([-2, 3]),
        4: np.array([-3, 1]),
    }
    colors = ['#e74c3c', '#2980b9', '#27ae60', '#f39c12']

    for ax_idx, (ax, title, routes) in enumerate(zip(axes,
        ['Step 1: All routes via depot',
         'Step 2: Merge best saving (1,2)',
         'Step 3: Merge (3,4)'],
        [
            [(depot, customers[1], depot),
             (depot, customers[2], depot),
             (depot, customers[3], depot),
             (depot, customers[4], depot)],
            [(depot, customers[1], customers[2], depot),
             (depot, customers[3], depot),
             (depot, customers[4], depot)],
            [(depot, customers[1], customers[2], depot),
             (depot, customers[3], customers[4], depot)],
        ])):
        ax.set_title(title, fontsize=10, fontweight='bold')
        # plot routes
        route_colors = ['#e74c3c','#2980b9','#27ae60','#f39c12']
        for r_idx, route in enumerate(routes):
            xs = [p[0] for p in route]
            ys = [p[1] for p in route]
            ax.plot(xs, ys, '-o', color=route_colors[r_idx % len(route_colors)],
                    linewidth=2, markersize=7, zorder=2)
        # depot
        ax.plot(*depot, 's', color='black', markersize=14, zorder=3)
        ax.text(depot[0]+0.1, depot[1]-0.3, 'Depot', fontsize=9, fontweight='bold')
        # customer labels
        for cid, pos in customers.items():
            ax.text(pos[0]+0.1, pos[1]+0.15, f'$c_{cid}$', fontsize=11)
        ax.set_xlim(-4.5, 5.5)
        ax.set_ylim(-1, 4.5)
        ax.set_aspect('equal')
        ax.axis('off')

    fig.suptitle('Clarke-Wright Savings Algorithm — Step-by-Step Merging',
                 fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()
    savefig('cw_savings_steps.pdf')


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 – Nearest Neighbour heuristic illustration
# ─────────────────────────────────────────────────────────────────────────────
def fig_nearest_neighbour():
    np.random.seed(42)
    n = 8
    pts = np.random.rand(n, 2) * 10
    depot_idx = 0
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # left: raw nodes
    ax = axes[0]
    ax.scatter(pts[:, 0], pts[:, 1], s=120, color='steelblue', zorder=3)
    ax.scatter(pts[depot_idx, 0], pts[depot_idx, 1], s=200, marker='s',
               color='black', zorder=4)
    ax.text(pts[depot_idx, 0]+0.2, pts[depot_idx, 1]+0.2, 'D', fontsize=11, fontweight='bold')
    for i in range(1, n):
        ax.text(pts[i, 0]+0.2, pts[i, 1]+0.2, str(i), fontsize=10)
    ax.set_title('8 customers + depot', fontsize=11)
    ax.axis('off')

    # right: NN tour
    ax = axes[1]
    visited = [False]*n
    tour = [depot_idx]
    visited[depot_idx] = True
    current = depot_idx
    for _ in range(n-1):
        best_d, best_j = 1e9, -1
        for j in range(n):
            if not visited[j]:
                d = np.linalg.norm(pts[current] - pts[j])
                if d < best_d:
                    best_d, best_j = d, j
        visited[best_j] = True
        tour.append(best_j)
        current = best_j
    tour.append(depot_idx)

    xs = pts[tour, 0]
    ys = pts[tour, 1]
    ax.plot(xs, ys, '-o', color='#e74c3c', linewidth=2, markersize=8, zorder=2)
    ax.scatter(pts[depot_idx, 0], pts[depot_idx, 1], s=200, marker='s',
               color='black', zorder=4)
    ax.text(pts[depot_idx, 0]+0.2, pts[depot_idx, 1]+0.2, 'D', fontsize=11, fontweight='bold')
    for i in range(1, n):
        ax.text(pts[i, 0]+0.2, pts[i, 1]+0.2, str(i), fontsize=10)
    for step_idx in range(len(tour)-1):
        mid = (pts[tour[step_idx]] + pts[tour[step_idx+1]]) / 2
        ax.text(mid[0], mid[1], str(step_idx+1), fontsize=7, color='navy', alpha=0.7)
    ax.set_title('Nearest Neighbour Tour', fontsize=11)
    ax.axis('off')

    fig.suptitle('Nearest Neighbour Constructive Heuristic', fontsize=12, fontweight='bold')
    plt.tight_layout()
    savefig('nearest_neighbour.pdf')


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3 – Sweep algorithm illustration
# ─────────────────────────────────────────────────────────────────────────────
def fig_sweep():
    np.random.seed(7)
    n = 12
    angles = np.sort(np.random.rand(n) * 2 * np.pi)
    radii = 2.5 + np.random.rand(n) * 2
    xs = radii * np.cos(angles)
    ys = radii * np.sin(angles)
    depot = np.array([0, 0])

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    # left: raw customers + sweep line
    ax = axes[0]
    ax.scatter(xs, ys, s=100, color='steelblue', zorder=3)
    ax.scatter(*depot, s=200, marker='s', color='black', zorder=4)
    ax.text(0.1, 0.2, 'D', fontsize=11, fontweight='bold')
    for i in range(n):
        ax.text(xs[i]+0.15, ys[i]+0.15, str(i+1), fontsize=9)
    sweep_angle = np.pi / 3
    sweep_len = 6
    ax.annotate('', xy=(sweep_len*np.cos(sweep_angle), sweep_len*np.sin(sweep_angle)),
                xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color='red', lw=2))
    ax.set_title('Sweep line (radial scan from depot)', fontsize=10)
    ax.set_aspect('equal')
    ax.axis('off')

    # right: routes after sweep clustering
    ax = axes[1]
    # colour clusters
    cluster_colors = ['#e74c3c', '#2980b9', '#27ae60']
    boundaries = [0, 4, 8, 12]
    for c, (s, e) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        idx = list(range(s, e))
        route_x = [depot[0]] + [xs[i] for i in idx] + [depot[0]]
        route_y = [depot[1]] + [ys[i] for i in idx] + [depot[1]]
        ax.plot(route_x, route_y, '-o', color=cluster_colors[c], linewidth=2,
                markersize=8, label=f'Route {c+1}')
        for i in idx:
            ax.text(xs[i]+0.15, ys[i]+0.15, str(i+1), fontsize=9)
    ax.scatter(*depot, s=200, marker='s', color='black', zorder=4)
    ax.text(0.1, 0.2, 'D', fontsize=11, fontweight='bold')
    ax.set_title('Three routes after sweep clustering', fontsize=10)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.legend(fontsize=9, loc='lower right')

    fig.suptitle('Sweep Algorithm — Rotate and Cluster', fontsize=12, fontweight='bold')
    plt.tight_layout()
    savefig('sweep_algorithm.pdf')


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4 – 2-opt move illustration
# ─────────────────────────────────────────────────────────────────────────────
def fig_2opt():
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    pts = np.array([
        [0, 0], [1, 3], [3, 4], [5, 3],
        [6, 1], [4, -1], [2, -1], [0, 0]
    ], dtype=float)

    # before 2-opt (with crossing)
    before = [0, 1, 4, 3, 2, 5, 6, 0]
    ax = axes[0]
    xb = [pts[i, 0] for i in before]
    yb = [pts[i, 1] for i in before]
    ax.plot(xb, yb, 'b-o', linewidth=2, markersize=9)
    ax.plot([pts[1,0], pts[4,0]], [pts[1,1], pts[4,1]], 'r--', linewidth=2.5, label='crossing edges')
    ax.plot([pts[2,0], pts[5,0]], [pts[2,1], pts[5,1]], 'r--', linewidth=2.5)
    for i in range(1, 7):
        ax.text(pts[i, 0]+0.1, pts[i, 1]+0.15, str(i), fontsize=11)
    ax.text(pts[0, 0]-0.3, pts[0, 1]-0.2, 'D', fontsize=11, fontweight='bold')
    ax.set_title('Before 2-opt (crossing edges in red)', fontsize=10)
    ax.axis('off')
    ax.legend(fontsize=9)

    # after 2-opt
    after = [0, 1, 2, 3, 4, 5, 6, 0]
    ax = axes[1]
    xa = [pts[i, 0] for i in after]
    ya = [pts[i, 1] for i in after]
    ax.plot(xa, ya, 'g-o', linewidth=2, markersize=9)
    for i in range(1, 7):
        ax.text(pts[i, 0]+0.1, pts[i, 1]+0.15, str(i), fontsize=11)
    ax.text(pts[0, 0]-0.3, pts[0, 1]-0.2, 'D', fontsize=11, fontweight='bold')
    ax.set_title('After 2-opt (no crossing, shorter route)', fontsize=10)
    ax.axis('off')

    fig.suptitle('2-opt Move: Remove Two Edges and Reconnect', fontsize=12, fontweight='bold')
    plt.tight_layout()
    savefig('two_opt.pdf')


# ─────────────────────────────────────────────────────────────────────────────
# Figure 5 – Or-opt move illustration
# ─────────────────────────────────────────────────────────────────────────────
def fig_or_opt():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    pts_route1 = np.array([[0,0],[1,2],[3,3],[5,2],[6,0],[4,-1],[2,-1],[0,0]], dtype=float)
    pts_route2 = np.array([[7,0],[8,2],[9,1],[7,0]], dtype=float)

    ax = axes[0]
    ax.plot(pts_route1[:,0], pts_route1[:,1], 'b-o', linewidth=2, markersize=9, label='Route 1')
    ax.plot(pts_route2[:,0], pts_route2[:,1], 'g-o', linewidth=2, markersize=9, label='Route 2')
    labels1 = ['D','1','2','3','4','5','6']
    labels2 = ['D2','7','8']
    for i, lb in enumerate(labels1):
        ax.text(pts_route1[i,0]+0.1, pts_route1[i,1]+0.2, lb, fontsize=10)
    for i, lb in enumerate(labels2):
        ax.text(pts_route2[i,0]+0.1, pts_route2[i,1]+0.2, lb, fontsize=10)
    ax.set_title('Before Or-opt: node 3 in Route 1', fontsize=10)
    ax.axis('off')
    ax.legend(fontsize=9)

    # after: node 3 relocated to route 2
    ax = axes[1]
    new_r1 = np.array([[0,0],[1,2],[5,2],[6,0],[4,-1],[2,-1],[0,0]], dtype=float)
    new_r2 = np.array([[7,0],[8,2],[3,3],[9,1],[7,0]], dtype=float)
    ax.plot(new_r1[:,0], new_r1[:,1], 'b-o', linewidth=2, markersize=9, label='Route 1 (updated)')
    ax.plot(new_r2[:,0], new_r2[:,1], 'g-o', linewidth=2, markersize=9, label='Route 2 (updated)')
    labels_r1 = ['D','1','3','4','5','6']
    labels_r2 = ['D2','7','3*','8']
    for i, lb in enumerate(labels_r1):
        ax.text(new_r1[i,0]+0.1, new_r1[i,1]+0.2, lb, fontsize=10)
    for i, lb in enumerate(labels_r2):
        ax.text(new_r2[i,0]+0.1, new_r2[i,1]+0.2, lb, fontsize=10)
    ax.annotate('node 3\nrelocated', xy=(3,3), xytext=(3.5, 4.2),
                arrowprops=dict(arrowstyle='->', color='red'), fontsize=9, color='red')
    ax.set_title('After Or-opt: node 3 inserted into Route 2', fontsize=10)
    ax.axis('off')
    ax.legend(fontsize=9)

    fig.suptitle('Or-opt Move: Relocate One (or Two) Customers Between Routes',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    savefig('or_opt.pdf')


# ─────────────────────────────────────────────────────────────────────────────
# Figure 6 – Tabu Search diagram (solution space with tabu list)
# ─────────────────────────────────────────────────────────────────────────────
def fig_tabu_search():
    np.random.seed(0)
    fig, ax = plt.subplots(figsize=(9, 5))

    # simulate a solution-quality landscape
    x = np.linspace(0, 10, 300)
    y = (np.sin(x) + 0.5*np.sin(2.3*x) + 0.3*np.cos(4*x)) * 2 + 8

    ax.plot(x, y, 'k-', linewidth=2)
    ax.fill_between(x, y.min()-0.5, y, alpha=0.12, color='steelblue')

    # path of tabu search
    ts_x = [0.5, 1.2, 2.0, 2.8, 3.5, 4.0, 4.7, 5.4, 6.1, 6.8, 7.5, 8.2, 9.0]
    ts_y = [np.interp(xi, x, y) for xi in ts_x]
    ax.plot(ts_x, ts_y, 'ro--', linewidth=1.8, markersize=8, label='Tabu Search path')

    # mark best known and tabu moves
    best_idx = np.argmin(ts_y)
    ax.scatter(ts_x[best_idx], ts_y[best_idx], s=200, color='gold',
               edgecolors='black', zorder=5, label=f'Best known = {ts_y[best_idx]:.2f}')
    ax.annotate('Best\nknown', xy=(ts_x[best_idx], ts_y[best_idx]),
                xytext=(ts_x[best_idx]+0.5, ts_y[best_idx]-1.5),
                arrowprops=dict(arrowstyle='->', color='darkred'), fontsize=9, color='darkred')

    # tabu list box
    ax.text(0.5, 10.8, 'Tabu list (recent moves — forbidden):', fontsize=9, style='italic')
    for i, xi in enumerate(ts_x[:4]):
        ax.add_patch(mpatches.FancyBboxPatch((xi-0.25, 10.0), 0.5, 0.6,
                     boxstyle='round,pad=0.05', fc='#fadbd8', ec='red', lw=1.2))
        ax.text(xi, 10.3, f'$m_{i+1}$', fontsize=9, ha='center', color='darkred')

    ax.set_xlabel('Solution space (neighbourhood moves)', fontsize=11)
    ax.set_ylabel('Objective value (total distance)', fontsize=11)
    ax.set_title('Tabu Search: Escape Local Optima via Forbidden-Move Memory',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=10, loc='upper right')
    ax.set_ylim(y.min()-1, 12.5)
    plt.tight_layout()
    savefig('tabu_search.pdf')


# ─────────────────────────────────────────────────────────────────────────────
# Figure 7 – Simulated Annealing: acceptance probability vs temperature
# ─────────────────────────────────────────────────────────────────────────────
def fig_simulated_annealing():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # left: acceptance probability vs delta cost at different temperatures
    delta = np.linspace(0, 50, 300)
    for T, lbl, col in [(200, 'T=200 (hot)', '#e74c3c'),
                         (50, 'T=50', '#f39c12'),
                         (10, 'T=10', '#2980b9'),
                         (2, 'T=2 (cold)', '#27ae60')]:
        prob = np.exp(-delta / T)
        axes[0].plot(delta, prob, color=col, linewidth=2, label=lbl)
    axes[0].axhline(0.5, ls='--', color='gray', linewidth=1)
    axes[0].set_xlabel(r'Worsening $\Delta C = C_{new} - C_{current}$', fontsize=10)
    axes[0].set_ylabel(r'Acceptance probability $\exp(-\Delta C / T)$', fontsize=10)
    axes[0].set_title('Higher temperature → more worsening moves accepted', fontsize=10)
    axes[0].legend(fontsize=9)
    axes[0].set_ylim(0, 1.05)

    # right: cooling schedule
    t_steps = np.arange(0, 200)
    T0 = 200
    alpha = 0.97
    temps = T0 * alpha**t_steps
    axes[1].plot(t_steps, temps, 'b-', linewidth=2.5)
    axes[1].set_xlabel('Iteration', fontsize=10)
    axes[1].set_ylabel('Temperature $T$', fontsize=10)
    axes[1].set_title(r'Geometric cooling: $T_{k+1} = \alpha \cdot T_k$, $\alpha=0.97$', fontsize=10)
    axes[1].fill_between(t_steps, 0, temps, alpha=0.2, color='blue')
    axes[1].annotate('Exploration\n(high T)', xy=(10, 150), fontsize=10, color='navy')
    axes[1].annotate('Exploitation\n(low T)', xy=(140, 20), fontsize=10, color='darkblue')

    fig.suptitle('Simulated Annealing — Temperature Controls Solution Acceptance',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    savefig('simulated_annealing.pdf')


# ─────────────────────────────────────────────────────────────────────────────
# Figure 8 – Genetic Algorithm: crossover on VRP routes
# ─────────────────────────────────────────────────────────────────────────────
def fig_genetic_algorithm():
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axis('off')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)

    def draw_route(ax, x, y, route_segs, color, label):
        for seg in route_segs:
            ax.text(x, y, label, fontsize=10, fontweight='bold', color=color)
            txt = '  |  '.join(['–'.join(map(str, s)) for s in seg])
            ax.text(x+1.5, y, txt, fontsize=10, color='black',
                    bbox=dict(boxstyle='round,pad=0.3', fc=color, alpha=0.2, ec=color))

    # Parent 1
    ax.text(0.2, 6.2, 'Parent 1:', fontsize=10, fontweight='bold', color='#2980b9')
    ax.text(1.5, 6.2, '[D→1→2→3→D]   [D→4→5→D]', fontsize=10,
            bbox=dict(boxstyle='round,pad=0.3', fc='#aed6f1', ec='#2980b9'))

    # Parent 2
    ax.text(0.2, 5.0, 'Parent 2:', fontsize=10, fontweight='bold', color='#c0392b')
    ax.text(1.5, 5.0, '[D→3→1→4→D]   [D→2→5→D]', fontsize=10,
            bbox=dict(boxstyle='round,pad=0.3', fc='#f1948a', ec='#c0392b'))

    # Arrow
    ax.annotate('', xy=(5, 3.8), xytext=(5, 4.5),
                arrowprops=dict(arrowstyle='->', lw=2.5, color='darkgreen'))
    ax.text(5.2, 4.1, 'OX Crossover', fontsize=10, color='darkgreen', fontweight='bold')

    # Children
    ax.text(0.2, 3.0, 'Child 1:', fontsize=10, fontweight='bold', color='#8e44ad')
    ax.text(1.5, 3.0, '[D→1→2→3→D]   [D→4→5→D]  (inherits P1 segment, fixes P2 order)',
            fontsize=9, bbox=dict(boxstyle='round,pad=0.3', fc='#d7bde2', ec='#8e44ad'))

    ax.text(0.2, 1.8, 'Child 2:', fontsize=10, fontweight='bold', color='#d35400')
    ax.text(1.5, 1.8, '[D→3→1→4→D]   [D→2→5→D]  (inherits P2 segment, fixes P1 order)',
            fontsize=9, bbox=dict(boxstyle='round,pad=0.3', fc='#fad7a0', ec='#d35400'))

    ax.annotate('', xy=(5, 1.2), xytext=(5, 1.6),
                arrowprops=dict(arrowstyle='->', lw=2.5, color='black'))
    ax.text(1.5, 0.6,
            'Repair infeasible routes (capacity check) → Mutation (swap 2 customers) → Next gen',
            fontsize=9, color='black',
            bbox=dict(boxstyle='round,pad=0.3', fc='#f9f9f9', ec='gray'))

    ax.set_title('Genetic Algorithm: Order Crossover (OX) on VRP Chromosomes',
                 fontsize=11, fontweight='bold')
    plt.tight_layout()
    savefig('genetic_algorithm.pdf')


# ─────────────────────────────────────────────────────────────────────────────
# Figure 9 – Ant Colony Optimization: pheromone update
# ─────────────────────────────────────────────────────────────────────────────
def fig_aco():
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    # left: graph with pheromone levels
    ax = axes[0]
    nodes = {0: (0,0), 1: (2,3), 2: (4,1), 3: (3,4), 4: (6,2), 5: (5,5)}
    pheromones = {(0,1):3.0, (0,2):1.0, (1,3):2.5, (1,2):0.5,
                  (2,4):2.0, (3,4):1.5, (3,5):3.5, (4,5):1.0}
    for (u,v), ph in pheromones.items():
        x_vals = [nodes[u][0], nodes[v][0]]
        y_vals = [nodes[u][1], nodes[v][1]]
        ax.plot(x_vals, y_vals, 'b-', linewidth=ph*1.5, alpha=0.6)
        mx, my = (nodes[u][0]+nodes[v][0])/2, (nodes[u][1]+nodes[v][1])/2
        ax.text(mx, my, f'{ph:.1f}', fontsize=8, color='navy', ha='center')
    for nid, (x,y) in nodes.items():
        ax.scatter(x, y, s=150, color='steelblue', zorder=4)
        lbl = 'D' if nid == 0 else str(nid)
        ax.text(x+0.15, y+0.2, lbl, fontsize=11, fontweight='bold')
    ax.set_title('Edge width ∝ pheromone level τ(i,j)', fontsize=10)
    ax.axis('off')

    # right: iteration vs best tour length
    ax = axes[1]
    iters = np.arange(1, 101)
    np.random.seed(3)
    base = 100
    improvement = base - 25 * (1 - np.exp(-iters/30)) + 5*np.random.randn(100).cumsum()*0.05
    ax.plot(iters, improvement, 'r-', linewidth=2, label='Best tour length')
    ax.set_xlabel('ACO Iteration', fontsize=10)
    ax.set_ylabel('Best solution cost', fontsize=10)
    ax.set_title('ACO convergence over iterations', fontsize=10)
    ax.legend(fontsize=9)

    fig.suptitle('Ant Colony Optimisation — Pheromone-Guided Probabilistic Search',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    savefig('aco.pdf')


# ─────────────────────────────────────────────────────────────────────────────
# Figure 10 – Large Neighbourhood Search: destroy & repair
# ─────────────────────────────────────────────────────────────────────────────
def fig_lns():
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    depot = np.array([0, 0])
    customers = np.array([
        [2, 3], [4, 2], [-2, 3], [-3, 1],
        [3, -2], [-1, -3], [1, 4], [4, 4]
    ])
    colors = ['#e74c3c','#2980b9','#27ae60']

    routes_before = [[0,1,2,7], [0,3,4,5], [0,6]]  # 0=depot offset
    routes_removed = [[0,1,7], [0,3,5], [0,6]]      # customers 2,4 removed
    routes_after = [[0,1,2,7], [0,4,3,5], [0,6]]

    def plot_routes(ax, routes, pts, removed=None, title=''):
        for r_idx, route in enumerate(routes):
            xs = [depot[0] if i==0 else customers[i-1][0] for i in route] + [depot[0]]
            ys = [depot[1] if i==0 else customers[i-1][1] for i in route] + [depot[1]]
            ax.plot(xs, ys, '-o', color=colors[r_idx % 3], linewidth=2, markersize=8)
        ax.scatter(*depot, s=200, marker='s', color='black', zorder=5)
        ax.text(depot[0]+0.1, depot[1]+0.2, 'D', fontsize=10, fontweight='bold')
        for i, c in enumerate(customers):
            if removed and (i+1) in removed:
                ax.scatter(c[0], c[1], s=150, marker='x', color='red', zorder=5, linewidths=2.5)
                ax.text(c[0]+0.1, c[1]+0.2, f'{i+1}*', fontsize=9, color='red')
            else:
                ax.text(c[0]+0.1, c[1]+0.2, str(i+1), fontsize=9)
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.axis('off')
        ax.set_xlim(-4.5, 5.5)
        ax.set_ylim(-4.5, 5.5)

    plot_routes(axes[0], routes_before, customers, title='Current solution')
    plot_routes(axes[1], routes_removed, customers, removed=[2,4], title='Destroy: remove cust. 2,4')
    plot_routes(axes[2], routes_after, customers, title='Repair: reinsert optimally')

    fig.suptitle('Large Neighbourhood Search (LNS) — Destroy and Repair',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    savefig('lns.pdf')


# ─────────────────────────────────────────────────────────────────────────────
# Figure 11 – Quality vs Time scatter (from book Fig 4.3 reproduction)
# ─────────────────────────────────────────────────────────────────────────────
def fig_quality_time_scatter():
    # Approximate data from book Figure 4.3
    algorithms = {
        'CLM01': (600, 1.70),
        'PRO7-B': (800, 1.45),
        'CM12-C': (300, 1.40),
        'CGW10-A': (80, 1.20),
        'CGW10-B': (15, 1.25),
        'MB07-B': (12, 1.22),
        'RDRU4':  (600, 1.20),
        'CM12-B': (700, 1.15),
        'T05':    (120, 1.05),
        'F09':    (200, 0.65),
        'MB07-A': (800, 0.85),
        'ZK10-B': (1500, 0.42),
        'NB09-B': (1200, 0.38),
        'SUO13-B':(5000, 0.35),
        'GGW11-C':(2500, 0.45),
        'PRO7-A': (600, 0.90),
        'SUO13-A':(30000, 0.33),
        'ZK10-A': (12000, 0.27),
        'NB09-A': (10000, 0.28),
        'GGW11-B':(2500, 0.35),
        'JCL12-B':(8000, 0.38),
        'CM12-A': (7000, 0.40),
        'CGW11-B':(9000, 0.42),
        'VCGLR12-A':(3000,0.22),
        'VCGLR12-B':(800, 0.28),
        'CGW11-A':(9000, 0.20),
        'JCL12-A':(80000,0.18),
        'GGW11-A':(100000,0.25),
    }
    fig, ax = plt.subplots(figsize=(9, 6))
    times = np.array([v[0] for v in algorithms.values()])
    gaps  = np.array([v[1] for v in algorithms.values()])
    ax.scatter(times, gaps, s=60, color='black', zorder=4)
    for name, (t, g) in algorithms.items():
        ax.annotate(name, (t, g), fontsize=6, textcoords='offset points',
                    xytext=(4, 2), color='black')
    ax.set_xscale('log')
    ax.set_xlabel('Normalised running time (s, log scale)', fontsize=11)
    ax.set_ylabel('Average gap to best-known (%)', fontsize=11)
    ax.set_title('Solution quality vs. computation time\n(GWKC benchmark instances, reproduction of book Fig. 4.3)',
                 fontsize=11, fontweight='bold')
    ax.invert_xaxis()
    plt.tight_layout()
    savefig('quality_time_scatter.pdf')


# ─────────────────────────────────────────────────────────────────────────────
# Figure 12 – Bar chart of average gaps for selected metaheuristics
# ─────────────────────────────────────────────────────────────────────────────
def fig_gap_comparison():
    methods = ['CLM01\n(TS)', 'TV03\n(SA)', 'RDH04\n(SA/TS)', 'T05\n(TS)',
               'MB07\n(SA)', 'PRO7\n(GA)', 'NB09\n(ALNS)', 'P09\n(ALNS)',
               'ZK10\n(TS)', 'GGW10\n(HGA)', 'VCGLR12\n(ALNS)', 'GGW11\n(HGA)',
               'CGW11\n(HGA)', 'CM12\n(TS)', 'JCL12\n(EA)', 'SUO13\n(HGA)']
    gaps = [1.79, 2.06, 0.93, 0.93, 0.33, 0.82, 0.63, 0.63, 0.22, 0.12, 0.16, 0.19, 0.35, 0.56, 0.35, 0.40]
    colors_bar = ['#e74c3c' if g > 1 else '#f39c12' if g > 0.5 else '#27ae60' for g in gaps]

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(methods, gaps, color=colors_bar, edgecolor='black', linewidth=0.7)
    ax.axhline(0.5, ls='--', color='gray', linewidth=1.2, label='0.5% threshold')
    for bar, gap in zip(bars, gaps):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{gap:.2f}', ha='center', va='bottom', fontsize=8)
    ax.set_ylabel('Average gap to best-known solution (%)', fontsize=11)
    ax.set_title('Average Gap (%) on GWKC Benchmark — Selected Metaheuristics (2001–2013)',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_ylim(0, 2.5)
    ax.tick_params(axis='x', labelsize=8)

    red_patch = mpatches.Patch(color='#e74c3c', label='Gap > 1%')
    orange_patch = mpatches.Patch(color='#f39c12', label='0.5–1%')
    green_patch = mpatches.Patch(color='#27ae60', label='< 0.5%')
    ax.legend(handles=[red_patch, orange_patch, green_patch], fontsize=9, loc='upper right')

    plt.tight_layout()
    savefig('gap_comparison.pdf')


# ─────────────────────────────────────────────────────────────────────────────
# Figure 13 – Decomposition strategies (from book Fig 4.2)
# ─────────────────────────────────────────────────────────────────────────────
def fig_decomposition():
    fig, axes = plt.subplots(1, 3, figsize=(13, 5))
    np.random.seed(12)
    n = 16
    pts = np.random.rand(n, 2) * 8
    depot = np.array([4, 4])

    titles = ['(a) Circular sectors', '(b) Horizontal bands', '(c) Depot separation']
    cmaps = [['#aed6f1','#fadbd8','#d5f5e3','#fef9e7'],
             ['#aed6f1','#fadbd8','#d5f5e3'],
             ['#aed6f1','#fadbd8','#d5f5e3']]

    for ax_idx, (ax, title) in enumerate(zip(axes, titles)):
        ax.scatter(depot[0], depot[1], s=200, marker='s', color='black', zorder=5)
        ax.text(depot[0]+0.1, depot[1]+0.2, 'D', fontsize=11, fontweight='bold')
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(title, fontsize=10, fontweight='bold')

        if ax_idx == 0:
            # sectors
            angles_bound = [0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi]
            sector_cols = ['#aed6f1','#fadbd8','#d5f5e3','#fef9e7']
            for si in range(4):
                theta = np.linspace(angles_bound[si], angles_bound[si+1], 50)
                r = 6
                wedge_x = [depot[0]] + list(depot[0]+r*np.cos(theta)) + [depot[0]]
                wedge_y = [depot[1]] + list(depot[1]+r*np.sin(theta)) + [depot[1]]
                ax.fill(wedge_x, wedge_y, alpha=0.3, color=sector_cols[si])
            for i, p in enumerate(pts):
                angle = np.arctan2(p[1]-depot[1], p[0]-depot[0]) % (2*np.pi)
                si = min(int(angle / (np.pi/2)), 3)
                ax.scatter(p[0], p[1], s=80, color=sector_cols[si], edgecolors='black', zorder=4)

        elif ax_idx == 1:
            # bands
            band_cols = ['#aed6f1','#fadbd8','#d5f5e3']
            bounds = [0, 2.7, 5.3, 8]
            for bi in range(3):
                ax.axhspan(bounds[bi], bounds[bi+1], alpha=0.2, color=band_cols[bi])
            for i, p in enumerate(pts):
                bi = 0 if p[1] < 2.7 else (1 if p[1] < 5.3 else 2)
                ax.scatter(p[0], p[1], s=80, color=band_cols[bi], edgecolors='black', zorder=4)

        else:
            # depots
            depot2 = np.array([1, 7])
            depot3 = np.array([7, 1])
            extra_depots = [depot, depot2, depot3]
            dep_colors = ['#aed6f1','#fadbd8','#d5f5e3']
            for ei, (ed, dc) in enumerate(zip(extra_depots, dep_colors)):
                ax.scatter(ed[0], ed[1], s=200, marker='s', color='black', zorder=5)
                label = 'D1' if ei==0 else f'D{ei+1}'
                ax.text(ed[0]+0.1, ed[1]+0.2, label, fontsize=9, fontweight='bold')
            for i, p in enumerate(pts):
                dists = [np.linalg.norm(p - ed) for ed in extra_depots]
                bi = np.argmin(dists)
                ax.scatter(p[0], p[1], s=80, color=dep_colors[bi], edgecolors='black', zorder=4)

    fig.suptitle('Decomposition Strategies for Large VRP Instances',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    savefig('decomposition.pdf')


# ─────────────────────────────────────────────────────────────────────────────
# Crop figure from PDF using pymupdf (book Fig 4.2 — decomposition strategies)
# ─────────────────────────────────────────────────────────────────────────────
def crop_book_figures():
    try:
        import fitz  # pymupdf
        pdf_path = os.path.join(
            os.path.dirname(__file__), '..',
            'Vehicle Routing_ Problems, Methods, and Applications, Second Edition 2014.pdf'
        )
        pdf_path = os.path.normpath(pdf_path)
        if not os.path.exists(pdf_path):
            print(f'  PDF not found at {pdf_path}, skipping crop')
            return

        doc = fitz.open(pdf_path)
        # Page 96 (0-indexed: 95) has Figure 4.2 (decomposition strategies)
        # Page 103 (0-indexed: 102) has Figure 4.3 (quality vs time scatter)
        crop_specs = [
            (95, fitz.Rect(50, 300, 550, 650), 'fig42_decomposition_crop.pdf'),
            (102, fitz.Rect(50, 50, 550, 700), 'fig43_quality_time_crop.pdf'),
        ]
        for page_idx, rect, outname in crop_specs:
            if page_idx < len(doc):
                page = doc[page_idx]
                mat = fitz.Matrix(2.0, 2.0)
                pix = page.get_pixmap(matrix=mat, clip=rect)
                out_path = os.path.join(OUTDIR, outname.replace('.pdf', '.png'))
                pix.save(out_path)
                print(f'  cropped {outname} -> {out_path}')
        doc.close()
    except ImportError:
        print('  pymupdf not available, skipping PDF crops')
    except Exception as e:
        print(f'  PDF crop error: {e}')


if __name__ == '__main__':
    print('Generating figures...')
    fig_clarke_wright()
    fig_nearest_neighbour()
    fig_sweep()
    fig_2opt()
    fig_or_opt()
    fig_tabu_search()
    fig_simulated_annealing()
    fig_genetic_algorithm()
    fig_aco()
    fig_lns()
    fig_quality_time_scatter()
    fig_gap_comparison()
    fig_decomposition()
    crop_book_figures()
    print('All figures generated.')
