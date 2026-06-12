"""
gen_figures.py  —  Generate all figures for Chapter 5 VRPTW slides.
Run with:  conda run -n py313 python3 gen_figures.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(OUT, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 — VRPTW overview: depot + customers with time windows
# ─────────────────────────────────────────────────────────────────────────────
def fig_vrptw_overview():
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.set_xlim(-0.5, 10.5); ax.set_ylim(-0.5, 8.5)
    ax.set_aspect('equal'); ax.axis('off')
    ax.set_title("VRPTW: Depot, Customers and Time Windows", fontsize=13, fontweight='bold')

    depot = (5, 4)
    customers = {
        1: (1, 7),  2: (3, 7),  3: (7, 7),  4: (9, 7),
        5: (1, 1),  6: (3, 1),  7: (7, 1),  8: (9, 1),
        9: (5, 6), 10: (5, 2),
    }
    tw = {
        1: "[0,30]",  2: "[10,40]", 3: "[20,50]", 4: "[5,35]",
        5: "[15,45]", 6: "[0,60]",  7: "[30,60]", 8: "[10,50]",
        9: "[0,20]",  10:"[25,55]",
    }
    colors_route = ['#1f77b4', '#ff7f0e', '#2ca02c']
    route1 = [depot, customers[1], customers[2], customers[9], depot]
    route2 = [depot, customers[3], customers[4], customers[8], depot]
    route3 = [depot, customers[5], customers[6], customers[10], customers[7], depot]

    for route, col in zip([route1, route2, route3], colors_route):
        xs = [p[0] for p in route]; ys = [p[1] for p in route]
        ax.plot(xs, ys, '-o', color=col, linewidth=1.8, markersize=5, alpha=0.7, zorder=2)

    ax.plot(*depot, 's', color='black', markersize=14, zorder=5)
    ax.text(depot[0], depot[1]+0.4, "Depot", ha='center', fontsize=9, fontweight='bold')

    for cid, pos in customers.items():
        ax.plot(*pos, 'o', color='steelblue', markersize=11, zorder=4)
        ax.text(pos[0], pos[1]+0.35, f"C{cid}", ha='center', fontsize=7.5, fontweight='bold')
        ax.text(pos[0], pos[1]-0.55, tw[cid], ha='center', fontsize=6.5, color='darkred')

    legend_elements = [
        mpatches.Patch(facecolor='black', label='Depot (square)'),
        mpatches.Patch(facecolor='steelblue', label='Customer (circle)'),
        mpatches.Patch(facecolor='white', edgecolor='darkred', label='[a,b] = time window'),
        mpatches.Patch(facecolor='#1f77b4', alpha=0.7, label='Route 1'),
        mpatches.Patch(facecolor='#ff7f0e', alpha=0.7, label='Route 2'),
        mpatches.Patch(facecolor='#2ca02c', alpha=0.7, label='Route 3'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "vrptw_overview.pdf"), dpi=150)
    plt.close(fig)
    print("Saved vrptw_overview.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 — Hard vs. Soft time windows illustration
# ─────────────────────────────────────────────────────────────────────────────
def fig_hard_soft_tw():
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for ax, title, color, label in zip(
        axes,
        ["Hard Time Window", "Soft Time Window"],
        ['#d62728', '#2ca02c'],
        ["Service MUST occur in [a,b]\n(infeasible outside)", "Penalty for violating [a,b]\n(feasible but costly)"]
    ):
        ax.set_xlim(0, 12); ax.set_ylim(-0.3, 3)
        ax.axis('off')
        ax.set_title(title, fontsize=12, fontweight='bold', color=color)

        ax.annotate("", xy=(11.5, 1), xytext=(0.5, 1),
                    arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
        ax.text(6, 0.3, "Time", ha='center', fontsize=10)

        ax.axvspan(3, 8, alpha=0.25, color=color, ymin=0.35, ymax=0.85)
        ax.text(3, 1.18, 'a', ha='center', fontsize=10, color=color)
        ax.text(8, 1.18, 'b', ha='center', fontsize=10, color=color)
        ax.plot([3, 3], [0.7, 1.3], color=color, lw=2)
        ax.plot([8, 8], [0.7, 1.3], color=color, lw=2)

        ax.plot(2,   1, 'x', color='#1f77b4', ms=10, mew=2.5)
        ax.plot(5.5, 1, '*', color='green', ms=14)
        ax.plot(9.5, 1, 'x', color='#d62728', ms=10, mew=2.5)

        ax.text(2,   1.5, "Too early", ha='center', fontsize=8)
        ax.text(5.5, 1.5, "On time",   ha='center', fontsize=8)
        ax.text(9.5, 1.5, "Too late",  ha='center', fontsize=8)
        ax.text(6, 2.5, label, ha='center', fontsize=9, style='italic',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    fig.suptitle("Hard vs. Soft Time Windows", fontsize=13, fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "hard_soft_tw.pdf"), dpi=150)
    plt.close(fig)
    print("Saved hard_soft_tw.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3 — Column Generation / Branch-and-Price schematic
# ─────────────────────────────────────────────────────────────────────────────
def fig_branch_price():
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 10); ax.set_ylim(0, 8)
    ax.axis('off')
    ax.set_title("Branch-and-Price Framework for VRPTW", fontsize=13, fontweight='bold')

    def box(ax, x, y, w, h, text, facecolor='#aec7e8', fontsize=9):
        rect = mpatches.FancyBboxPatch((x - w/2, y - h/2), w, h,
                                       boxstyle="round,pad=0.1",
                                       facecolor=facecolor, edgecolor='navy', lw=1.5)
        ax.add_patch(rect)
        ax.text(x, y, text, ha='center', va='center', fontsize=fontsize,
                multialignment='center')

    def arrow(ax, x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

    box(ax, 5, 7.2, 4.5, 0.9,
        "LP Relaxation (Master Problem)\n"
        "min sum c_r * lambda_r  s.t. coverage constraints", '#aec7e8', 8)
    arrow(ax, 5, 6.75, 5, 6.15)
    box(ax, 5, 5.7, 4.5, 0.9,
        "Solve Sub-problem (SPPRC)\nFind column (route) with negative reduced cost", '#c7e9c0', 8)
    arrow(ax, 5, 5.25, 5, 4.65)
    box(ax, 5, 4.2, 4.5, 0.9,
        "Add new column(s) to Master Problem\nRe-solve LP Relaxation", '#aec7e8', 8)

    ax.annotate("", xy=(7.8, 5.7), xytext=(7.8, 4.2),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.5,
                                connectionstyle='arc3,rad=0.5'))
    ax.text(8.7, 4.95, "Iterate", fontsize=8, color='gray', ha='center')
    ax.text(8.7, 4.6, "(column\ngeneration)", fontsize=7, color='gray', ha='center')

    arrow(ax, 5, 3.75, 5, 3.15)
    box(ax, 5, 2.7, 4.5, 0.9,
        "Branching Decision\n(integrality violated? branch on arc/route)", '#ffbb78', 8)
    arrow(ax, 3.5, 2.25, 2.2, 1.6)
    arrow(ax, 6.5, 2.25, 7.8, 1.6)
    box(ax, 2, 1.2, 2.8, 0.7, "Branch node (left)", '#aec7e8', 8)
    box(ax, 8, 1.2, 2.8, 0.7, "Branch node (right)", '#aec7e8', 8)

    ax.text(5, 0.4, "At each node: re-solve LP with column generation.\nPrune if LP bound >= best integer solution found.",
            ha='center', fontsize=8, style='italic',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "branch_price.pdf"), dpi=150)
    plt.close(fig)
    print("Saved branch_price.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4 — SPPRC resource extension illustration
# ─────────────────────────────────────────────────────────────────────────────
def fig_spprc():
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_xlim(0, 10); ax.set_ylim(0, 6)
    ax.axis('off')
    ax.set_title("Shortest Path with Resource Constraints (SPPRC)", fontsize=12, fontweight='bold')

    nodes = {'0 (depot)': (1, 3), 'A': (3, 5), 'B': (3, 1),
             'C': (6, 5), 'D': (6, 1), 'n+1 (depot)': (9, 3)}
    tw = {'0 (depot)': '[0,0]', 'A': '[5,20]', 'B': '[10,30]',
          'C': '[15,40]', 'D': '[20,45]', 'n+1 (depot)': '[0,100]'}
    dem = {'A': 'q=3', 'B': 'q=4', 'C': 'q=2', 'D': 'q=5'}

    for name, pos in nodes.items():
        col = 'black' if 'depot' in name else 'steelblue'
        ax.plot(*pos, 'o', color=col, ms=18, zorder=3)
        ax.text(pos[0], pos[1]+0.5, name, ha='center', fontsize=8, fontweight='bold')
        ax.text(pos[0], pos[1]-0.6, tw[name], ha='center', fontsize=7, color='darkred')
        if name in dem:
            ax.text(pos[0]+0.6, pos[1], dem[name], ha='center', fontsize=7, color='purple')

    edges = [('0 (depot)', 'A'), ('0 (depot)', 'B'), ('A', 'C'), ('A', 'D'),
             ('B', 'C'), ('B', 'D'), ('C', 'n+1 (depot)'), ('D', 'n+1 (depot)')]
    for u, v in edges:
        p1, p2 = nodes[u], nodes[v]
        ax.annotate("", xy=p2, xytext=p1,
                    arrowprops=dict(arrowstyle='->', color='gray', lw=1.2))

    ax.text(5, 0.3,
            "Resource label at each node: (earliest arrival time, load so far)\n"
            "Extension: F(tau, q) -> ( max(tau + t_ij, a_j), q + q_j ).  Discard if tau > b_j or q > Q.",
            ha='center', fontsize=8, style='italic',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "spprc.pdf"), dpi=150)
    plt.close(fig)
    print("Saved spprc.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 5 — Solomon insertion heuristic worked example (I1)
# ─────────────────────────────────────────────────────────────────────────────
def fig_insertion_heuristic():
    fig, axes = plt.subplots(1, 3, figsize=(13, 5))
    depot = np.array([5, 5])
    customers = {
        1: np.array([2, 8]),
        2: np.array([8, 8]),
        3: np.array([2, 2]),
        4: np.array([8, 2]),
    }
    tw = {1: '[0,15]', 2: '[5,25]', 3: '[10,30]', 4: '[0,20]'}

    stages = [
        ([1],     "Step 1: Seed customer 1\n(closest to depot or earliest deadline)"),
        ([1, 2],  "Step 2: Insert customer 2\n(best c2 position after 1)"),
        ([1, 2, 3], "Step 3: Insert customer 3\n(best feasible position)"),
    ]

    for ax, (route_ids, title) in zip(axes, stages):
        ax.set_xlim(0, 10); ax.set_ylim(0, 10)
        ax.set_aspect('equal'); ax.axis('off')
        ax.set_title(title, fontsize=8.5, fontweight='bold')

        full_route = [depot] + [customers[i] for i in route_ids] + [depot]
        xs = [p[0] for p in full_route]; ys = [p[1] for p in full_route]
        ax.plot(xs, ys, 'b-o', linewidth=1.8, markersize=7)

        ax.plot(*depot, 's', color='black', ms=12, zorder=5)
        ax.text(depot[0], depot[1]+0.5, 'Depot', ha='center', fontsize=8)

        for cid in range(1, 5):
            pos = customers[cid]
            if cid in route_ids:
                ax.plot(*pos, 'o', color='steelblue', ms=11, zorder=4)
            else:
                ax.plot(*pos, 'o', color='lightgray', ms=11, zorder=4)
            ax.text(pos[0], pos[1]+0.5, f'C{cid}', ha='center', fontsize=8)
            ax.text(pos[0], pos[1]-0.6, tw[cid], ha='center', fontsize=7, color='darkred')

    fig.suptitle("Solomon I1 Insertion Heuristic: Step-by-Step Example\n"
                 "(Gray = not yet inserted; Blue = in current route; Black square = depot)",
                 fontsize=10, fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "insertion_heuristic.pdf"), dpi=150)
    plt.close(fig)
    print("Saved insertion_heuristic.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 6 — 2-opt move
# ─────────────────────────────────────────────────────────────────────────────
def fig_2opt():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    def draw_route(ax, positions, labels, title, color='steelblue'):
        ax.set_aspect('equal'); ax.axis('off')
        ax.set_title(title, fontsize=10, fontweight='bold')
        n = len(positions)
        for i in range(n):
            p1, p2 = positions[i], positions[(i+1) % n]
            ax.annotate("", xy=p2, xytext=p1,
                        arrowprops=dict(arrowstyle='->', color=color, lw=1.5))
        for pos, lab in zip(positions, labels):
            ax.plot(*pos, 'o', color=color, ms=13, zorder=3)
            ax.text(pos[0], pos[1]+0.3, lab, ha='center', fontsize=9, fontweight='bold')
        ax.set_xlim(-0.5, 5.5); ax.set_ylim(-0.5, 5.5)

    pos_before = [(0,2), (1,4), (3,4), (4,2), (3,0), (1,0)]
    pos_after  = [(0,2), (1,4), (3,0), (4,2), (3,4), (1,0)]
    labels = ['0\n(depot)', 'C1', 'C2', 'C3', 'C4', 'C5']

    draw_route(axes[0], pos_before, labels, "Before 2-opt\n(crossing edges C2-C3 and C4-C5)", 'steelblue')
    axes[0].plot([pos_before[2][0], pos_before[4][0]],
                 [pos_before[2][1], pos_before[4][1]], '--', color='red', lw=1.5, label='edges to remove')
    axes[0].legend(loc='lower right', fontsize=8)

    draw_route(axes[1], pos_after, labels, "After 2-opt\n(segment reversed: shorter total distance)", '#2ca02c')
    axes[1].text(2.5, -0.3, "Segment between removed edges is reversed.\nAccept if new total distance is shorter.",
                 ha='center', fontsize=8, style='italic')

    fig.suptitle("The 2-opt Local Search Move", fontsize=12, fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "two_opt.pdf"), dpi=150)
    plt.close(fig)
    print("Saved two_opt.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 7 — Or-opt (relocate single customer)
# ─────────────────────────────────────────────────────────────────────────────
def fig_oropt():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    def draw_two_routes(ax, r1, r2, labels1, labels2, title):
        ax.set_aspect('equal'); ax.axis('off')
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.set_xlim(-0.5, 6); ax.set_ylim(-0.5, 5.5)
        for route, color in [(r1, 'steelblue'), (r2, '#ff7f0e')]:
            n = len(route)
            for i in range(n-1):
                p1, p2 = route[i], route[i+1]
                ax.annotate("", xy=p2, xytext=p1,
                            arrowprops=dict(arrowstyle='->', color=color, lw=1.5))
            for pos in route:
                ax.plot(*pos, 'o', color=color, ms=12, zorder=3)
        all_pts = r1 + r2
        all_labs = labels1 + labels2
        for pos, lab in zip(all_pts, all_labs):
            ax.text(pos[0], pos[1]+0.3, lab, ha='center', fontsize=8, fontweight='bold')

    depot = (0.5, 2.5)
    r1_before = [depot, (1.5, 4.5), (3.5, 4.5), (4.5, 4.5), depot]
    r2_before = [depot, (1.5, 0.5), (3.5, 0.5), depot]
    labs1_before = ['dep', 'A', 'B', 'C', 'dep']
    labs2_before = ['dep', 'D', 'E', 'dep']

    draw_two_routes(axes[0], r1_before, r2_before, labs1_before, labs2_before,
                    "Before Or-opt\nRoute 1: dep-A-B-C | Route 2: dep-D-E")

    r1_after = [depot, (1.5, 4.5), (3.5, 4.5), depot]
    r2_after = [depot, (1.5, 0.5), (3.5, 0.5), (4.5, 0.5), depot]
    labs1_after = ['dep', 'A', 'B', 'dep']
    labs2_after = ['dep', 'D', 'E', 'C', 'dep']

    draw_two_routes(axes[1], r1_after, r2_after, labs1_after, labs2_after,
                    "After Or-opt\nC moved to Route 2: dep-D-E-C")

    fig.suptitle("Or-opt Move: Relocate One Customer Between Routes", fontsize=12, fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "oropt.pdf"), dpi=150)
    plt.close(fig)
    print("Saved oropt.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 8 — Cross-exchange neighbourhood illustration
# ─────────────────────────────────────────────────────────────────────────────
def fig_cross_exchange():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    def draw_segs(ax, r1, r2, title, labs1, labs2):
        ax.set_aspect('equal'); ax.axis('off')
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.set_xlim(-0.5, 6.5); ax.set_ylim(-0.5, 6)
        for route, color in [(r1, 'steelblue'), (r2, '#ff7f0e')]:
            for i in range(len(route)-1):
                p1, p2 = route[i], route[i+1]
                ax.annotate("", xy=p2, xytext=p1,
                            arrowprops=dict(arrowstyle='->', color=color, lw=1.5))
            for pos in route:
                ax.plot(*pos, 'o', color=color, ms=12, zorder=3)
        for pos, lab in zip(r1, labs1):
            ax.text(pos[0], pos[1]+0.35, lab, ha='center', fontsize=8)
        for pos, lab in zip(r2, labs2):
            ax.text(pos[0], pos[1]-0.55, lab, ha='center', fontsize=8)

    depot = (3, 3)
    r1_before = [depot, (1,5), (2,5), (4,5), (5,5), depot]
    r2_before = [depot, (1,1), (2,1), (4,1), (5,1), depot]
    draw_segs(axes[0], r1_before, r2_before,
              "Before Cross-Exchange\nBlue (top) | Orange (bottom)",
              ['dep','A','B','C','D','dep'], ['dep','E','F','G','H','dep'])

    r1_after = [depot, (1,5), (2,5), (4,1), (5,1), depot]
    r2_after = [depot, (1,1), (2,1), (4,5), (5,5), depot]
    draw_segs(axes[1], r1_after, r2_after,
              "After Cross-Exchange\nSegments [C,D] and [G,H] swapped",
              ['dep','A','B','G','H','dep'], ['dep','E','F','C','D','dep'])

    fig.suptitle("Cross-Exchange Neighbourhood", fontsize=12, fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "cross_exchange.pdf"), dpi=150)
    plt.close(fig)
    print("Saved cross_exchange.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 9 — LNS destroy-and-repair illustration
# ─────────────────────────────────────────────────────────────────────────────
def fig_lns():
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    depot = np.array([5, 5])
    np.random.seed(42)
    n = 10
    pts = np.random.uniform(1, 9, (n, 2))
    cust = {i+1: pts[i] for i in range(n)}
    removed = {3, 6, 8}

    titles = ["Original Solution\n(complete routes)",
              "Destroy Phase\n(remove customers 3,6,8 shown as red crosses)",
              "Repair Phase\n(re-insert removed customers optimally)"]
    route_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    routes = [[1,2,3], [4,5,6], [7,8,9,10]]

    for ax, title in zip(axes, titles):
        ax.set_xlim(0, 10); ax.set_ylim(0, 10)
        ax.set_aspect('equal'); ax.axis('off')
        ax.set_title(title, fontsize=9, fontweight='bold')

        ax.plot(*depot, 's', color='black', ms=12, zorder=5)
        ax.text(depot[0], depot[1]+0.5, 'Depot', ha='center', fontsize=8)

        for ridx, (route, col) in enumerate(zip(routes, route_colors)):
            if "Destroy" in title:
                visible = [c for c in route if c not in removed]
            else:
                visible = route

            pts_route = [depot] + [cust[c] for c in visible] + [depot]
            xs = [p[0] for p in pts_route]; ys = [p[1] for p in pts_route]
            ax.plot(xs, ys, '-', color=col, lw=1.8, alpha=0.7)

            for c in route:
                pos = cust[c]
                if c in removed and "Destroy" in title:
                    ax.plot(*pos, 'x', color='red', ms=12, mew=2.5, zorder=4)
                else:
                    ax.plot(*pos, 'o', color=col, ms=10, zorder=4)
                ax.text(pos[0], pos[1]+0.45, f'C{c}', ha='center', fontsize=7)

    fig.suptitle("Large Neighbourhood Search (LNS): Destroy and Repair", fontsize=12, fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "lns.pdf"), dpi=150)
    plt.close(fig)
    print("Saved lns.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 10 — Tabu Search trajectory
# ─────────────────────────────────────────────────────────────────────────────
def fig_tabu():
    fig, ax = plt.subplots(figsize=(10, 5))
    np.random.seed(7)
    n = 30
    xs = np.arange(n)
    obj = 200 + 50*np.sin(xs/3) + 30*np.sin(xs/1.5) - 1.5*xs + np.random.normal(0, 5, n)
    obj = np.clip(obj, 80, 250)
    best_so_far = np.minimum.accumulate(obj)

    ax.plot(xs, obj, 'b-o', ms=5, lw=1.5, label='Current solution value')
    ax.plot(xs, best_so_far, 'r--', lw=2, label='Best solution found so far')

    tabu_iter = [5, 12, 20]
    for ti in tabu_iter:
        ax.axvline(ti, color='orange', lw=1, linestyle=':', alpha=0.7)
        ax.text(ti+0.3, obj[ti]+5, 'tabu\nmove', fontsize=7, color='darkorange')

    ax.set_xlabel("Iteration", fontsize=11)
    ax.set_ylabel("Objective (total distance)", fontsize=11)
    ax.set_title("Tabu Search Trajectory: Escaping Local Optima", fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.text(15, 225,
            "Key: Tabu search ALLOWS worsening moves\n"
            "to escape local optima (orange dashed lines).\n"
            "The best solution is tracked separately (red).",
            fontsize=8, bbox=dict(boxstyle='round', facecolor='lightyellow'))
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "tabu_trajectory.pdf"), dpi=150)
    plt.close(fig)
    print("Saved tabu_trajectory.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 11 — Metaheuristic comparison bar chart
# ─────────────────────────────────────────────────────────────────────────────
def fig_metaheuristic_comparison():
    methods = ['SA\n(BVH04)', 'TS\n(MB05)', 'GLS\n(PR07)', 'ILS\n(HY08)',
               'LNS\n(RT09)', 'EA\n(PDR09)', 'ILS+LNS\n(NBD10)', 'Hybrid\n(VCGP13)']
    cnv_pct = [0.12, 0.08, 0.06, 0.05, 0.02, 0.03, 0.02, 0.01]
    veh_pct = [0.8,  0.5,  0.3,  0.4,  0.1,  0.2,  0.1,  0.05]

    x = np.arange(len(methods))
    width = 0.38
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - width/2, cnv_pct, width, label='Distance gap above best known (%)',
           color='steelblue', alpha=0.85)
    ax.bar(x + width/2, veh_pct, width, label='Vehicle count gap (%)',
           color='#ff7f0e', alpha=0.85)

    ax.set_xlabel("Method (reference)", fontsize=11)
    ax.set_ylabel("Gap above best known solution (%)", fontsize=11)
    ax.set_title("Illustrative Comparison of Metaheuristics on Solomon Benchmarks\n"
                 "(Lower is better -- approximate values for illustration)",
                 fontsize=11, fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(methods, fontsize=9)
    ax.legend(fontsize=9)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "metaheuristic_comparison.pdf"), dpi=150)
    plt.close(fig)
    print("Saved metaheuristic_comparison.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 12 — Solomon benchmark instance classes
# ─────────────────────────────────────────────────────────────────────────────
def fig_solomon_instances():
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    np.random.seed(0)
    classes = [
        ('C1: Clustered, Tight TW',  'clustered', 'tight'),
        ('C2: Clustered, Wide TW',   'clustered', 'wide'),
        ('R1: Random, Tight TW',     'random',    'tight'),
        ('R2: Random, Wide TW',      'random',    'wide'),
        ('RC1: Mixed, Tight TW',     'mixed',     'tight'),
        ('RC2: Mixed, Wide TW',      'mixed',     'wide'),
    ]

    def gen_pts(kind, n=25):
        if kind == 'random':
            return np.random.uniform(0, 1, (n, 2))
        elif kind == 'clustered':
            centers = np.random.uniform(0.2, 0.8, (4, 2))
            pts = []
            for _ in range(n):
                c = centers[np.random.randint(4)]
                pts.append(c + np.random.normal(0, 0.08, 2))
            return np.clip(np.array(pts), 0, 1)
        else:
            half = n // 2
            r = np.random.uniform(0, 1, (half, 2))
            centers = np.random.uniform(0.2, 0.8, (2, 2))
            c_pts = []
            for _ in range(n - half):
                c = centers[np.random.randint(2)]
                c_pts.append(c + np.random.normal(0, 0.1, 2))
            return np.clip(np.vstack([r, np.array(c_pts)]), 0, 1)

    for ax, (title, kind, tw) in zip(axes.flat, classes):
        pts = gen_pts(kind)
        tw_half = 0.08 if tw == 'tight' else 0.25
        ax.scatter(pts[:, 0], pts[:, 1], c='steelblue', s=40, zorder=3)
        ax.plot(0.5, 0.5, 's', color='black', ms=10, zorder=5)
        for p in pts:
            tw_start = max(0, p[0] - tw_half)
            tw_end   = min(1, p[0] + tw_half)
            ax.plot([tw_start, tw_end], [p[1], p[1]], '-', color='salmon',
                    lw=1, alpha=0.6, zorder=1)
        ax.set_title(title, fontsize=9, fontweight='bold')
        ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, 1.05)
        ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle("Solomon (1987) Benchmark Instance Classes\n"
                 "(Black square = depot; blue dots = customers; red bars = time windows)",
                 fontsize=11, fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "solomon_instances.pdf"), dpi=150)
    plt.close(fig)
    print("Saved solomon_instances.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 13 — Path relinking schematic
# ─────────────────────────────────────────────────────────────────────────────
def fig_path_relinking():
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_xlim(0, 10); ax.set_ylim(0, 6)
    ax.axis('off')
    ax.set_title("Path Relinking: Connecting Two Elite Solutions", fontsize=12, fontweight='bold')

    ax.plot(1, 3, 'o', color='steelblue', ms=20, zorder=5)
    ax.text(1, 3, 'x', ha='center', va='center', fontsize=10, color='white', fontweight='bold')
    ax.text(1, 2.2, 'Elite solution 1\n(initiating)', ha='center', fontsize=8)

    ax.plot(9, 3, 'o', color='#2ca02c', ms=20, zorder=5)
    ax.text(9, 3, 'x*', ha='center', va='center', fontsize=9, color='white', fontweight='bold')
    ax.text(9, 2.2, 'Elite solution 2\n(guiding)', ha='center', fontsize=8)

    path_x = [1, 2.5, 4, 5.5, 7, 9]
    path_y = [3, 4.2, 3.8, 4.5, 3.2, 3]
    ax.plot(path_x, path_y, 'k--', lw=1.5, zorder=2)
    for xi, yi in zip(path_x[1:-1], path_y[1:-1]):
        ax.plot(xi, yi, 'o', color='#ff7f0e', ms=12, zorder=3)

    ax.plot(5.5, 4.5, '*', color='gold', ms=18, zorder=4)
    ax.text(5.5, 5.1, 'Best solution\non path', ha='center', fontsize=8, color='darkgoldenrod')

    ax.text(5, 0.8,
            "Path relinking explores intermediate solutions between two elite\n"
            "solutions by gradually introducing attributes of one into the other.",
            ha='center', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "path_relinking.pdf"), dpi=150)
    plt.close(fig)
    print("Saved path_relinking.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 14 — Split delivery illustration
# ─────────────────────────────────────────────────────────────────────────────
def fig_split_delivery():
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    depot = np.array([5, 5])
    cust_A = np.array([2, 7])
    cust_B = np.array([8, 7])

    for ax, title, split in zip(axes,
        ["Standard VRPTW\n(demand 8 must go in ONE visit)",
         "Split Delivery VRPTW\n(demand 8 split: 5+3 across two vehicles)"],
        [False, True]):
        ax.set_xlim(0, 10); ax.set_ylim(0, 10)
        ax.set_aspect('equal'); ax.axis('off')
        ax.set_title(title, fontsize=9, fontweight='bold')
        ax.plot(*depot, 's', color='black', ms=14, zorder=5)
        ax.text(depot[0], depot[1]+0.5, 'Depot', ha='center', fontsize=9)

        for pos, lab, dem in [(cust_A, 'A', 'demand=8'), (cust_B, 'B', 'demand=3')]:
            ax.plot(*pos, 'o', color='steelblue', ms=14, zorder=4)
            ax.text(pos[0], pos[1]+0.5, lab, ha='center', fontsize=9, fontweight='bold')
            ax.text(pos[0], pos[1]-0.7, dem, ha='center', fontsize=8, color='darkred')

        if not split:
            ax.annotate("", xy=cust_A, xytext=depot,
                        arrowprops=dict(arrowstyle='->', color='steelblue', lw=2))
            ax.annotate("", xy=depot, xytext=cust_A,
                        arrowprops=dict(arrowstyle='->', color='steelblue', lw=2,
                                        connectionstyle='arc3,rad=0.15'))
            ax.text(3.5, 6.8, 'Vehicle 1\n(capacity 10\nload=8)', fontsize=7.5, ha='center',
                    bbox=dict(facecolor='lightblue', alpha=0.7))
        else:
            ax.annotate("", xy=cust_A, xytext=depot,
                        arrowprops=dict(arrowstyle='->', color='steelblue', lw=2))
            ax.annotate("", xy=depot, xytext=cust_A,
                        arrowprops=dict(arrowstyle='->', color='steelblue', lw=2,
                                        connectionstyle='arc3,rad=0.15'))
            ax.text(3.2, 6.8, 'Veh 1\nload=5', fontsize=7.5, ha='center',
                    bbox=dict(facecolor='lightblue', alpha=0.7))
            ax.annotate("", xy=cust_A, xytext=depot,
                        arrowprops=dict(arrowstyle='->', color='#ff7f0e', lw=2,
                                        connectionstyle='arc3,rad=-0.3'))
            ax.text(5.5, 8.2, 'Veh 2\nload=3\n(remainder)', fontsize=7.5, ha='center',
                    bbox=dict(facecolor='#ffe8b0', alpha=0.7))

    fig.suptitle("Split Delivery: One Customer Served by Multiple Vehicles",
                 fontsize=11, fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "split_delivery.pdf"), dpi=150)
    plt.close(fig)
    print("Saved split_delivery.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 15 — Stochastic VRPTW: recourse action
# ─────────────────────────────────────────────────────────────────────────────
def fig_stochastic():
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    depot = np.array([5, 5])
    customers = [np.array([2, 8]), np.array([4, 8]), np.array([7, 7]),
                 np.array([3, 3]), np.array([7, 3])]

    for ax, title, fail_idx in zip(axes,
        ["Planned Route\n(stochastic demand not yet realised)",
         "Recourse Action\n(Customer C3 demand exceeds capacity: return to depot)"],
        [None, 2]):
        ax.set_xlim(0, 10); ax.set_ylim(0, 10)
        ax.set_aspect('equal'); ax.axis('off')
        ax.set_title(title, fontsize=9, fontweight='bold')
        ax.plot(*depot, 's', color='black', ms=14, zorder=5)
        ax.text(depot[0], depot[1]+0.5, 'Depot', ha='center', fontsize=9)

        route = [depot] + customers + [depot]
        for i in range(len(route)-1):
            p1, p2 = route[i], route[i+1]
            color = 'red' if (fail_idx is not None and i == fail_idx) else 'steelblue'
            ax.annotate("", xy=p2, xytext=p1,
                        arrowprops=dict(arrowstyle='->', color=color, lw=1.8))
        for j, pos in enumerate(customers):
            ax.plot(*pos, 'o', color='steelblue', ms=12, zorder=4)
            ax.text(pos[0], pos[1]+0.4, f'C{j+1}', ha='center', fontsize=8, fontweight='bold')

        if fail_idx is not None:
            p_fail = customers[fail_idx]
            ax.annotate("", xy=depot, xytext=p_fail,
                        arrowprops=dict(arrowstyle='->', color='red', lw=2,
                                        connectionstyle='arc3,rad=0.4'))
            ax.text(7.5, 6.5, "Return to depot!\n(vehicle full)", fontsize=8,
                    color='red', ha='center',
                    bbox=dict(facecolor='mistyrose', alpha=0.8))

    fig.suptitle("Stochastic VRPTW: Planned Route vs. Recourse When Capacity Exceeded",
                 fontsize=10, fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "stochastic_vrptw.pdf"), dpi=150)
    plt.close(fig)
    print("Saved stochastic_vrptw.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 16 — Dual-variable stabilisation (column generation convergence)
# ─────────────────────────────────────────────────────────────────────────────
def fig_cg_convergence():
    fig, ax = plt.subplots(figsize=(9, 5))
    np.random.seed(3)
    iters = np.arange(1, 31)
    # Simulate oscillating duals without stabilisation vs. smooth with stabilisation
    no_stab = 100 + 40*np.sin(iters*0.7) * np.exp(-iters*0.05) + np.random.normal(0,3,30)
    with_stab = 100 + 15*np.sin(iters*0.4) * np.exp(-iters*0.1) + np.random.normal(0,1,30)
    no_stab   = np.clip(no_stab, 70, 160)
    with_stab = np.clip(with_stab, 85, 130)

    ax.plot(iters, no_stab,   'r-o', ms=5, lw=1.5, label='Without stabilisation (oscillating)')
    ax.plot(iters, with_stab, 'b-o', ms=5, lw=1.5, label='With dual stabilisation (smooth)')
    ax.axhline(100, color='gray', lw=1.5, linestyle='--', label='Optimal LP value')

    ax.set_xlabel("Column generation iteration", fontsize=11)
    ax.set_ylabel("LP bound value", fontsize=11)
    ax.set_title("Column Generation Convergence:\nDual Stabilisation Reduces Oscillation",
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.4)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "cg_convergence.pdf"), dpi=150)
    plt.close(fig)
    print("Saved cg_convergence.pdf")


if __name__ == "__main__":
    fig_vrptw_overview()
    fig_hard_soft_tw()
    fig_branch_price()
    fig_spprc()
    fig_insertion_heuristic()
    fig_2opt()
    fig_oropt()
    fig_cross_exchange()
    fig_lns()
    fig_tabu()
    fig_metaheuristic_comparison()
    fig_solomon_instances()
    fig_path_relinking()
    fig_split_delivery()
    fig_stochastic()
    fig_cg_convergence()
    print("\nAll figures generated successfully.")
