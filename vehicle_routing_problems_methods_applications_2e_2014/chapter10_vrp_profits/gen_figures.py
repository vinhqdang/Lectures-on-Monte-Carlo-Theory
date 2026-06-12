"""
gen_figures.py  --  Generate all figures for Chapter 10 (VRP with Profits) slides.
Backend: Agg (no display required).
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(OUT, exist_ok=True)

# ── colour palette ────────────────────────────────────────────────────────────
DEPOT_COLOR    = "#e74c3c"
CUST_COLOR     = "#2980b9"
VISITED_COLOR  = "#27ae60"
SKIP_COLOR     = "#bdc3c7"
ROUTE_COLORS   = ["#2ecc71", "#e67e22", "#9b59b6", "#1abc9c"]


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 1: VRPP taxonomy tree
# ═══════════════════════════════════════════════════════════════════════════════
def fig_taxonomy():
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis('off')

    def box(cx, cy, text, color='#d0e8f5', fontsize=9, width=2.4, height=0.55):
        rect = mpatches.FancyBboxPatch(
            (cx - width/2, cy - height/2), width, height,
            boxstyle="round,pad=0.05", linewidth=1.2,
            edgecolor='#2060a0', facecolor=color)
        ax.add_patch(rect)
        ax.text(cx, cy, text, ha='center', va='center', fontsize=fontsize,
                fontweight='bold', color='#1a1a2e')

    def arrow(x0, y0, x1, y1):
        ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle='->', color='#2060a0', lw=1.4))

    # Root
    box(6, 5.3, 'VRPPs (Vehicle Routing Problems with Profits)',
        color='#b8d4f0', fontsize=9, width=5.0)

    # Level 2
    box(3, 3.8, 'Single Vehicle\n(TSPP)', color='#c8e6c9', fontsize=9, width=2.5)
    box(9, 3.8, 'Multiple Vehicles\n(multi-route)', color='#ffe0b2', fontsize=9, width=2.5)

    arrow(6, 5.05, 3, 4.08)
    arrow(6, 5.05, 9, 4.08)

    # Level 3 - single
    box(1.5, 2.3, 'OP\n(Orienteering)', color='#e8f5e9', fontsize=8, width=2.0)
    box(3.6, 2.3, 'PCTSP\n(Prize-Collecting)', color='#e8f5e9', fontsize=8, width=2.0)
    box(5.6, 2.3, 'DCTSP\n(Discounted Cost)', color='#e8f5e9', fontsize=8, width=2.0)

    arrow(3, 3.52, 1.5, 2.58)
    arrow(3, 3.52, 3.6, 2.58)
    arrow(3, 3.52, 5.6, 2.58)

    # Level 3 - multiple
    box(7.8, 2.3, 'TOP\n(Team Orienteering)', color='#fff3e0', fontsize=8, width=2.2)
    box(10.5, 2.3, 'Capacitated TOP\n(CTOP)', color='#fff3e0', fontsize=8, width=2.3)

    arrow(9, 3.52, 7.8, 2.58)
    arrow(9, 3.52, 10.5, 2.58)

    # Level 4 - variants
    variants = [(1.0, 'TW-OP'), (2.5, 'Clustered OP'), (4.2, 'Multi-period'), (6.0, 'SPS')]
    for cx, label in variants:
        box(cx, 0.9, label, color='#fce4ec', fontsize=7.5, width=1.6, height=0.42)
        arrow(cx, 2.02, cx, 1.12)

    ax.text(3.5, 0.25, 'Variants of OP', ha='center', fontsize=8.5,
            color='#555', style='italic')
    ax.set_title('Taxonomy of Vehicle Routing Problems with Profits (VRPPs)',
                 fontsize=11, pad=10, fontweight='bold')
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_taxonomy.pdf", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  fig_taxonomy.pdf")


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 2: OP example - depot + 10 customers, selected route shown
# ═══════════════════════════════════════════════════════════════════════════════
def fig_op_example():
    np.random.seed(42)
    depot = np.array([0.5, 0.5])
    customers = np.array([
        [0.15, 0.80], [0.40, 0.90], [0.70, 0.85], [0.90, 0.70],
        [0.85, 0.40], [0.60, 0.20], [0.30, 0.20], [0.10, 0.40],
        [0.50, 0.65], [0.75, 0.55]
    ])
    profits = [3, 5, 4, 6, 7, 5, 3, 4, 8, 6]
    selected = [0, 8, 9, 3, 4]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax_idx, ax in enumerate(axes):
        ax.scatter(*depot, s=220, c=DEPOT_COLOR, zorder=5, marker='s',
                   edgecolors='k', linewidths=1, label='Depot')
        ax.text(depot[0]+0.02, depot[1]-0.05, 'Depot', fontsize=8,
                color=DEPOT_COLOR, fontweight='bold')

        for i, (c, p) in enumerate(zip(customers, profits)):
            color = VISITED_COLOR if i in selected else SKIP_COLOR
            ax.scatter(c[0], c[1], s=100 + p*15, c=color, zorder=4,
                       edgecolors='#333', linewidths=0.8)
            ax.text(c[0]+0.02, c[1]+0.02, f'p={p}', fontsize=7.5, color='#333')

        if ax_idx == 1:
            route_pts = [depot] + [customers[i] for i in selected] + [depot]
            route_pts = np.array(route_pts)
            ax.plot(route_pts[:, 0], route_pts[:, 1], '-', color=CUST_COLOR,
                    lw=2, zorder=3, alpha=0.8)
            for step, node in enumerate(route_pts):
                label = 'D' if step == 0 or step == len(route_pts)-1 else str(step)
                ax.text(node[0]-0.04, node[1]-0.05, label, fontsize=8, color='navy')
            total_p = sum(profits[i] for i in selected)
            ax.set_title(f'Selected Route (profit collected = {total_p})', fontsize=10,
                         fontweight='bold')
        else:
            ax.set_title('All Customers (circle size proportional to profit)', fontsize=10,
                         fontweight='bold')

        ax.set_xlim(0.0, 1.1)
        ax.set_ylim(0.05, 1.05)
        ax.set_xlabel('x coordinate', fontsize=9)
        ax.set_ylabel('y coordinate', fontsize=9)
        visited_p = mpatches.Patch(color=VISITED_COLOR, label='Visited customers')
        skip_p    = mpatches.Patch(color=SKIP_COLOR,    label='Skipped customers')
        depot_p   = mpatches.Patch(color=DEPOT_COLOR,   label='Depot')
        ax.legend(handles=[depot_p, visited_p, skip_p], fontsize=8, loc='lower right')
        ax.grid(True, alpha=0.3)

    fig.suptitle(r'Orienteering Problem (OP): 10 Customers, Time Limit $T_{\max}$',
                 fontsize=11, y=1.01, fontweight='bold')
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_op_example.pdf", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  fig_op_example.pdf")


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 3: TOP example - 3 vehicles, different routes
# ═══════════════════════════════════════════════════════════════════════════════
def fig_top_example():
    np.random.seed(7)
    depot = np.array([0.5, 0.5])
    n = 15
    pts = np.random.rand(n, 2) * 0.8 + 0.1
    profits = np.random.randint(2, 10, n)

    routes = [
        [0, 3, 7, 11],
        [1, 5, 9, 13],
        [2, 6, 10, 14],
    ]
    route_colors  = ['#e53935', '#43a047', '#1e88e5']
    vehicle_labels = ['Vehicle 1', 'Vehicle 2', 'Vehicle 3']

    fig, ax = plt.subplots(figsize=(8, 6.5))

    # all customers (grey)
    ax.scatter(pts[:, 0], pts[:, 1], s=80, c=SKIP_COLOR, zorder=3,
               edgecolors='#666', linewidth=0.8)
    for i, (p, profit) in enumerate(zip(pts, profits)):
        ax.text(p[0]+0.015, p[1]+0.01, f'{profit}', fontsize=7, color='#444')

    # depot
    ax.scatter(*depot, s=300, c='black', marker='*', zorder=6, label='Depot')
    ax.text(depot[0]+0.02, depot[1]+0.02, 'Depot', fontsize=9, fontweight='bold')

    total = 0
    for route, color, label in zip(routes, route_colors, vehicle_labels):
        nodes = np.array([depot] + [pts[i] for i in route] + [depot])
        ax.plot(nodes[:, 0], nodes[:, 1], '-o', color=color, lw=2,
                zorder=4, alpha=0.85, markersize=7, label=label)
        r_profit = sum(profits[i] for i in route)
        total += r_profit

    ax.set_xlim(0.0, 1.05)
    ax.set_ylim(0.05, 1.05)
    ax.set_title(f'Team Orienteering Problem (TOP): 3 Vehicles\nTotal collected profit = {total}',
                 fontsize=10, fontweight='bold')
    ax.set_xlabel('x coordinate', fontsize=9)
    ax.set_ylabel('y coordinate', fontsize=9)
    ax.legend(fontsize=9, loc='lower right')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_top_example.pdf", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  fig_top_example.pdf")


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 4: Profit vs cost trade-off (three objective formulations)
# ═══════════════════════════════════════════════════════════════════════════════
def fig_profit_cost_tradeoff():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    ax = axes[0]
    t = np.linspace(0, 10, 200)
    ax.plot(t, 0.8*t, 'b-', lw=2.2, label='Max profit (cost bounded)')
    ax.plot(t, 10 - 0.9*t, 'r--', lw=2.2, label='Min cost (profit bounded)')
    ax.plot(t, 0.5*t - 0.05*t**2 + 3, 'g-.', lw=2.2, label='Max (profit - cost)')
    ax.axhline(0, color='k', lw=0.8)
    ax.axvline(0, color='k', lw=0.8)
    ax.set_xlim(0, 10)
    ax.set_ylim(-2, 12)
    ax.set_xlabel('Travel Cost / Time', fontsize=9)
    ax.set_ylabel('Objective Value', fontsize=9)
    ax.set_title('Three VRPP Objective Formulations', fontsize=10, fontweight='bold')
    ax.legend(fontsize=8.5)
    ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    categories = ['OP', 'PCTSP', 'DCTSP', 'TOP', 'CTOP', 'TW-TOP']
    n_vehicles = [1, 1, 1, 3, 3, 3]
    obj_types  = ['Max Profit', 'Min Cost+Pen', 'Max P-C', 'Max Profit', 'Max Profit', 'Max Profit']
    bar_colors = ['#42a5f5', '#ef5350', '#66bb6a', '#ab47bc', '#ffa726', '#26c6da']
    bars = ax2.bar(categories, n_vehicles, color=bar_colors, alpha=0.85, edgecolor='k', lw=0.9)
    ax2.set_ylabel('Number of Vehicles', fontsize=9)
    ax2.set_title('VRPP Variants by Fleet Size', fontsize=10, fontweight='bold')
    for bar, ot in zip(bars, obj_types):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, ot,
                 ha='center', va='bottom', fontsize=7.5, rotation=20)
    ax2.set_ylim(0, 4.8)
    ax2.grid(True, alpha=0.3, axis='y')

    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_profit_cost_tradeoff.pdf", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  fig_profit_cost_tradeoff.pdf")


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 5: Dynamic Programming state graph for OP
# ═══════════════════════════════════════════════════════════════════════════════
def fig_dp_state():
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.axis('off')

    stages = 5
    states_per_stage = [1, 3, 4, 3, 1]
    stage_x = np.linspace(0.1, 0.9, stages)

    node_positions = {}
    for s, (x, n) in enumerate(zip(stage_x, states_per_stage)):
        ys = np.linspace(0.2, 0.8, n)
        for i, y in enumerate(ys):
            node_positions[(s, i)] = (x, y)

    # all edges
    for s in range(stages-1):
        for i in range(states_per_stage[s]):
            for j in range(states_per_stage[s+1]):
                x0, y0 = node_positions[(s, i)]
                x1, y1 = node_positions[(s+1, j)]
                ax.plot([x0, x1], [y0, y1], 'gray', lw=0.7, alpha=0.35, zorder=1)

    # highlight optimal path
    path = [(0,0), (1,1), (2,2), (3,1), (4,0)]
    for k in range(len(path)-1):
        x0, y0 = node_positions[path[k]]
        x1, y1 = node_positions[path[k+1]]
        ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle='->', color='#1565c0', lw=2.5), zorder=3)

    # nodes
    for (s, i), (x, y) in node_positions.items():
        in_path = (s, i) in path
        circ = plt.Circle((x, y), 0.03,
                           color='#1565c0' if in_path else '#90caf9',
                           zorder=4, ec='navy', lw=1.2)
        ax.add_patch(circ)
        ax.text(x, y-0.07, f'S{s},{i}', fontsize=7, ha='center', color='#333')

    for s, x in enumerate(stage_x):
        ax.text(x, 0.92, f'Stage {s}', ha='center', fontsize=9,
                fontweight='bold', color='#1a237e')

    ax.text(0.5, 0.03,
            'Blue path = optimal substructure (Bellman principle)',
            ha='center', fontsize=9, color='#1565c0', style='italic',
            transform=ax.transAxes)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title('Dynamic Programming State Graph for Orienteering Problem\n'
                 '(Each node = (stage, partial-route state); edges = transitions)',
                 fontsize=10, fontweight='bold')
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_dp_state.pdf", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  fig_dp_state.pdf")


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 6: OP algorithm benchmark comparison
# ═══════════════════════════════════════════════════════════════════════════════
def fig_benchmark_op():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    algorithms = ['Exact B&B', 'Lagrangian\nRelax.', 'DP\n(Tsiligirides)', 'GRASP', 'SA\n(Chiang)', 'TS\n(Tang)']
    deviations = [0.0, 0.8, 1.5, 0.3, 0.5, 0.2]
    times      = [1000, 120, 45, 8, 15, 12]
    bar_colors = ['#ef5350', '#42a5f5', '#66bb6a', '#ab47bc', '#ffa726', '#26c6da']

    ax = axes[0]
    bars = ax.bar(algorithms, deviations, color=bar_colors, alpha=0.85, edgecolor='k', lw=0.8)
    ax.set_ylabel('Avg. % deviation from optimum', fontsize=9)
    ax.set_title('OP Algorithm Quality\n(lower = better solution)', fontsize=10, fontweight='bold')
    ax.set_ylim(0, 2.0)
    ax.grid(True, alpha=0.3, axis='y')
    for bar, d in zip(bars, deviations):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.03,
                f'{d:.1f}%', ha='center', fontsize=8.5, fontweight='bold')

    ax2 = axes[1]
    ax2.barh(algorithms, times, color=bar_colors, alpha=0.85, edgecolor='k', lw=0.8)
    ax2.set_xlabel('Relative CPU time (seconds)', fontsize=9)
    ax2.set_title('OP Algorithm Speed\n(lower = faster)', fontsize=10, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='x')
    for i, t in enumerate(times):
        ax2.text(t+5, i, f'{t}s', va='center', fontsize=8.5)

    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_benchmark_op.pdf", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  fig_benchmark_op.pdf")


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 7: TOP benchmark - Chao, Golden, Wasil instances (schematic)
# ═══════════════════════════════════════════════════════════════════════════════
def fig_benchmark_top():
    fig, ax = plt.subplots(figsize=(9, 5))

    instance_sizes = [7, 21, 33, 64, 66, 100, 102]
    algs = ['ILS (Vansteenwegen)', 'SA (Chiang & Wang)', 'TS (Tang & Miller)', 'GA (Wang)']
    line_colors = ['#1565c0', '#e53935', '#2e7d32', '#f57f17']
    gaps = {
        'ILS (Vansteenwegen)': [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2],
        'SA (Chiang & Wang)':  [0.1, 0.5, 0.9, 1.2, 1.5, 1.8, 2.2],
        'TS (Tang & Miller)':  [0.3, 0.7, 1.1, 1.5, 2.0, 2.5, 3.0],
        'GA (Wang)':           [0.5, 1.0, 1.5, 2.0, 2.8, 3.5, 4.0],
    }

    for alg, color in zip(algs, line_colors):
        ax.plot(instance_sizes, gaps[alg], '-o', color=color, lw=2,
                label=alg, markersize=6)

    ax.set_xlabel('Number of Customers in Instance', fontsize=10)
    ax.set_ylabel('Avg. Gap from Best Known (%)', fontsize=10)
    ax.set_title('TOP Heuristic Comparison on Chao-Golden-Wasil Benchmark\n'
                 '(ILS = Iterated Local Search; SA = Simulated Annealing; TS = Tabu Search; GA = Genetic Algorithm)',
                 fontsize=9, fontweight='bold')
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 4.5)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_benchmark_top.pdf", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  fig_benchmark_top.pdf")


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 8: Time-window constraints illustration
# ═══════════════════════════════════════════════════════════════════════════════
def fig_time_windows():
    fig, ax = plt.subplots(figsize=(10, 4.2))

    windows = [(1, 3), (2, 5), (3, 6), (1, 4), (4, 7), (5, 8), (6, 9), (7, 10)]
    profits = [4, 6, 3, 5, 7, 5, 4, 6]
    service = [0.5] * 8

    for i, ((a, b), s, p) in enumerate(zip(windows, service, profits)):
        ax.barh(i, a, left=0, height=0.45, color='#ffcccc', alpha=0.5)
        ax.barh(i, 10-b, left=b, height=0.45, color='#ffcccc', alpha=0.5)
        ax.barh(i, b-a, left=a, height=0.45, color='#a5d6a7', alpha=0.85,
                edgecolor='#388e3c', linewidth=1.2)
        ax.barh(i, s, left=a+0.2, height=0.45, color='#1565c0', alpha=0.8)
        ax.text(10.15, i, f'C{i+1}: p={p}, [{a},{b}]', va='center', fontsize=8.5)

    ax.set_xlim(0, 14)
    ax.set_ylim(-0.6, 8.2)
    ax.set_xlabel('Time', fontsize=10)
    ax.set_yticks(range(8))
    ax.set_yticklabels([f'Customer {i+1}' for i in range(8)], fontsize=8.5)
    ax.set_title('Time-Window Constraints in TW-OP / TW-TOP\n'
                 r'Vehicle must arrive in $[a_i, b_i]$ for customer $i$',
                 fontsize=10, fontweight='bold')
    ax.axvline(0, color='k', lw=0.8)
    ax.grid(True, alpha=0.3, axis='x')

    green_p = mpatches.Patch(color='#a5d6a7', label=r'Feasible window $[a_i, b_i]$')
    blue_p  = mpatches.Patch(color='#1565c0', label=r'Service time $s_i$')
    red_p   = mpatches.Patch(color='#ffcccc', label='Outside time window')
    ax.legend(handles=[green_p, blue_p, red_p], fontsize=8.5, loc='lower right')
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_time_windows.pdf", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  fig_time_windows.pdf")


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 9: Clustered OP - cluster structure
# ═══════════════════════════════════════════════════════════════════════════════
def fig_clustered_op():
    fig, ax = plt.subplots(figsize=(8, 6))
    np.random.seed(55)

    cluster_centers = [(0.2, 0.7), (0.7, 0.8), (0.85, 0.35), (0.3, 0.2), (0.55, 0.5)]
    cluster_colors  = ['#ef9a9a', '#a5d6a7', '#90caf9', '#ffe082', '#ce93d8']
    cluster_profits = [15, 20, 12, 18, 10]
    cluster_names   = [f'Cluster {chr(65+i)}\n(p={p})' for i, p in enumerate(cluster_profits)]

    depot = (0.5, 0.05)
    ax.scatter(*depot, s=260, c='black', marker='*', zorder=6)
    ax.text(depot[0]+0.02, depot[1]+0.02, 'Depot', fontsize=9, fontweight='bold')

    for (cx, cy), color, name in zip(cluster_centers, cluster_colors, cluster_names):
        members = np.random.randn(6, 2) * 0.06 + [cx, cy]
        circ = plt.Circle((cx, cy), 0.11, color=color, alpha=0.3, zorder=2)
        ax.add_patch(circ)
        ax.scatter(members[:, 0], members[:, 1], s=55, c=color,
                   edgecolors='#444', lw=0.8, zorder=4)
        ax.text(cx, cy+0.14, name, ha='center', fontsize=8, color='#333')

    # route selecting clusters A, B, D
    route = np.array([depot, (0.22, 0.68), (0.72, 0.79), (0.32, 0.21), depot])
    ax.plot(route[:, 0], route[:, 1], 'b--', lw=2.2, zorder=3, alpha=0.75,
            label='Selected route (A, B, D)')

    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.02)
    ax.set_title('Clustered Orienteering Problem\n'
                 'If a cluster is visited, ALL its members must be served',
                 fontsize=10, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_clustered_op.pdf", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  fig_clustered_op.pdf")


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 10: Multi-period TOP - three planning periods
# ═══════════════════════════════════════════════════════════════════════════════
def fig_multiperiod():
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    np.random.seed(22)
    n = 12
    pts = np.random.rand(n, 2) * 0.8 + 0.1
    profits = np.random.randint(3, 10, n)

    period_routes = [
        ([0, 2, 5, 9],  [1, 4, 7, 11]),
        ([0, 3, 6, 10], [2, 5, 8]),
        ([1, 4, 9, 11], [3, 7, 10]),
    ]
    depot = np.array([0.5, 0.05])
    colors_per = [['#e53935', '#1e88e5'], ['#43a047', '#ffa726'], ['#8e24aa', '#00acc1']]

    for period, (routes, cmap) in enumerate(zip(period_routes, colors_per)):
        ax = axes[period]
        ax.scatter(*depot, s=200, c='black', marker='*', zorder=5)
        ax.scatter(pts[:, 0], pts[:, 1], s=60+profits*8, c=SKIP_COLOR,
                   edgecolors='#555', lw=0.7, zorder=3)
        for i, p in enumerate(profits):
            ax.text(pts[i, 0]+0.015, pts[i, 1], f'{p}', fontsize=7, color='#444')

        all_vis = set(i for r in routes for i in r)
        vis_pts = pts[list(all_vis)]
        ax.scatter(vis_pts[:, 0], vis_pts[:, 1], s=80, c=VISITED_COLOR,
                   edgecolors='#333', lw=0.8, zorder=4)

        for r_idx, (route, c) in enumerate(zip(routes, cmap)):
            nodes = np.array([depot] + [pts[i] for i in route] + [depot])
            ax.plot(nodes[:, 0], nodes[:, 1], '-o', color=c, lw=1.8,
                    markersize=5, label=f'V{r_idx+1}', alpha=0.85)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(f'Period {period+1}', fontsize=10, fontweight='bold')
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.3)

    fig.suptitle('Multi-Period TOP: Same Fleet, New Routes Each Planning Period',
                 fontsize=11, fontweight='bold')
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_multiperiod.pdf", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  fig_multiperiod.pdf")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Generating figures for Chapter 10 (VRP with Profits)...")
    fig_taxonomy()
    fig_op_example()
    fig_top_example()
    fig_profit_cost_tradeoff()
    fig_dp_state()
    fig_benchmark_op()
    fig_benchmark_top()
    fig_time_windows()
    fig_clustered_op()
    fig_multiperiod()
    print(f"\nAll figures saved to: {OUT}")
