"""
gen_figures.py
Generate all figures for Chapter 6: Pickup-and-Delivery Problems for Goods Transportation
Uses matplotlib (Agg backend) for diagrams and pymupdf for PDF crops.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import os

FIGURES_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────
def savefig(name):
    path = os.path.join(FIGURES_DIR, name)
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  saved {path}")


# ─────────────────────────────────────────────
# Figure 1: Three types of PDP
# ─────────────────────────────────────────────
def fig_pdp_types():
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.suptitle("Three Categories of Pickup-and-Delivery Problems", fontsize=13, fontweight='bold')

    colors = {'depot': '#e74c3c', 'pickup': '#2ecc71', 'delivery': '#3498db'}

    # --- Many-to-Many ---
    ax = axes[0]
    ax.set_title("Many-to-Many (M-M)", fontweight='bold', fontsize=10)
    ax.set_xlim(-0.2, 4.2); ax.set_ylim(-0.2, 3.5)
    ax.axis('off')
    nodes = {
        'D': (2.0, 3.0), 'P1': (0.5, 2.0), 'P2': (1.5, 1.0),
        'P3': (3.5, 2.0), 'P4': (2.5, 0.5)
    }
    node_colors_map = {'D': colors['depot'], 'P1': colors['pickup'], 'P2': colors['pickup'],
                       'P3': colors['delivery'], 'P4': colors['delivery']}
    for name, (x, y) in nodes.items():
        c = node_colors_map[name]
        ax.plot(x, y, 'o', markersize=18, color=c, zorder=3)
        ax.text(x, y, name, ha='center', va='center', fontsize=8, fontweight='bold', color='white', zorder=4)
    # Arrows showing M-M flows
    arcs = [('D','P1'), ('P1','P3'), ('P3','D'), ('D','P2'), ('P2','P4'), ('P4','D')]
    for s, e in arcs:
        sx, sy = nodes[s]; ex, ey = nodes[e]
        ax.annotate('', xy=(ex, ey), xytext=(sx, sy),
                    arrowprops=dict(arrowstyle='->', color='gray', lw=1.2))
    ax.text(2.0, -0.15, "Any node can be\norigin or destination", ha='center', fontsize=7.5,
            style='italic', color='#555')

    # --- One-to-Many-to-One ---
    ax = axes[1]
    ax.set_title("One-to-Many-to-One (1-M-1)", fontweight='bold', fontsize=10)
    ax.set_xlim(-0.2, 4.2); ax.set_ylim(-0.2, 3.5)
    ax.axis('off')
    depot = (2.0, 3.0)
    customers = [(0.5, 1.8), (1.8, 0.8), (3.5, 1.8), (2.8, 0.5)]
    cust_labels = ['C1', 'C2', 'C3', 'C4']
    cust_types  = ['del', 'del', 'pick', 'pick']
    ax.plot(*depot, 'o', markersize=18, color=colors['depot'], zorder=3)
    ax.text(depot[0], depot[1], 'D', ha='center', va='center', fontsize=8, fontweight='bold', color='white', zorder=4)
    for (x, y), lbl, tp in zip(customers, cust_labels, cust_types):
        c = colors['delivery'] if tp == 'del' else colors['pickup']
        ax.plot(x, y, 'o', markersize=16, color=c, zorder=3)
        ax.text(x, y, lbl, ha='center', va='center', fontsize=8, fontweight='bold', color='white', zorder=4)
    # Route: D -> C1 -> C2 -> C3 -> C4 -> D
    route = [depot] + customers + [depot]
    xs, ys = zip(*route)
    ax.plot(xs, ys, '-', color='steelblue', lw=1.5, alpha=0.7)
    for i in range(len(route)-1):
        mx, my = (route[i][0]+route[i+1][0])/2, (route[i][1]+route[i+1][1])/2
        ax.annotate('', xy=route[i+1], xytext=route[i],
                    arrowprops=dict(arrowstyle='->', color='steelblue', lw=1.2))
    patch_del = mpatches.Patch(color=colors['delivery'], label='Delivery')
    patch_pick = mpatches.Patch(color=colors['pickup'], label='Pickup')
    ax.legend(handles=[patch_del, patch_pick], loc='lower left', fontsize=7)
    ax.text(2.0, -0.15, "Depot is origin and\ndestination", ha='center', fontsize=7.5,
            style='italic', color='#555')

    # --- One-to-One ---
    ax = axes[2]
    ax.set_title("One-to-One (1-1 / PDPTW)", fontweight='bold', fontsize=10)
    ax.set_xlim(-0.2, 4.2); ax.set_ylim(-0.2, 3.5)
    ax.axis('off')
    depot = (2.0, 3.0)
    pickups   = [(0.5, 2.0), (1.0, 0.8)]
    deliveries= [(3.5, 2.0), (3.0, 0.6)]
    ax.plot(*depot, 'o', markersize=18, color=colors['depot'], zorder=3)
    ax.text(depot[0], depot[1], 'D', ha='center', va='center', fontsize=8, fontweight='bold', color='white', zorder=4)
    for i, (px, py) in enumerate(pickups, 1):
        ax.plot(px, py, 'o', markersize=16, color=colors['pickup'], zorder=3)
        ax.text(px, py, f'P{i}', ha='center', va='center', fontsize=8, fontweight='bold', color='white', zorder=4)
    for i, (dx, dy) in enumerate(deliveries, 1):
        ax.plot(dx, dy, 's', markersize=16, color=colors['delivery'], zorder=3)
        ax.text(dx, dy, f'D{i}', ha='center', va='center', fontsize=8, fontweight='bold', color='white', zorder=4)
    # Pairing arrows
    for i in range(len(pickups)):
        px, py = pickups[i]; dx, dy = deliveries[i]
        ax.annotate('', xy=(dx, dy), xytext=(px, py),
                    arrowprops=dict(arrowstyle='->', color='purple', lw=2, linestyle='dashed'))
        ax.text((px+dx)/2, (py+dy)/2+0.1, f'pair {i+1}', ha='center', fontsize=7.5, color='purple')
    patch_p = mpatches.Patch(color=colors['pickup'], label='Pickup')
    patch_d = mpatches.Patch(color=colors['delivery'], label='Delivery')
    ax.legend(handles=[patch_p, patch_d], loc='lower left', fontsize=7)
    ax.text(2.0, -0.15, "Each pickup paired with\none delivery", ha='center', fontsize=7.5,
            style='italic', color='#555')

    plt.tight_layout()
    savefig("fig_pdp_types.pdf")


# ─────────────────────────────────────────────
# Figure 2: PDP graph – single-vehicle example
# ─────────────────────────────────────────────
def fig_pdp_graph():
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_title("PDP Graph: Single-Vehicle Request Graph\n(Pickup nodes 1-n, Delivery nodes n+1 to 2n, Depot 0 and 2n+1)",
                 fontsize=10, fontweight='bold')
    ax.set_xlim(-0.5, 5.5); ax.set_ylim(-0.5, 4.0)
    ax.axis('off')

    depot_orig = (0.0, 2.0)
    depot_dest  = (5.0, 2.0)
    pickups    = [(1.5, 3.2), (1.5, 0.8)]
    deliveries = [(3.5, 3.2), (3.5, 0.8)]
    labels_p = ['1', '2']
    labels_d = ['n+1', 'n+2']

    ax.plot(*depot_orig, 's', markersize=22, color='#e74c3c', zorder=3)
    ax.text(depot_orig[0], depot_orig[1], '0', ha='center', va='center', fontsize=10, fontweight='bold', color='white', zorder=4)
    ax.text(depot_orig[0], depot_orig[1]-0.45, 'Origin\nDepot', ha='center', fontsize=8)

    ax.plot(*depot_dest, 's', markersize=22, color='#e74c3c', zorder=3)
    ax.text(depot_dest[0], depot_dest[1], '2n+1', ha='center', va='center', fontsize=8, fontweight='bold', color='white', zorder=4)
    ax.text(depot_dest[0], depot_dest[1]-0.45, 'Dest.\nDepot', ha='center', fontsize=8)

    for (x, y), lbl in zip(pickups, labels_p):
        ax.plot(x, y, 'o', markersize=20, color='#2ecc71', zorder=3)
        ax.text(x, y, lbl, ha='center', va='center', fontsize=10, fontweight='bold', color='white', zorder=4)
        ax.text(x-0.15, y+0.38, 'Pickup', ha='center', fontsize=7.5, color='#2ecc71')

    for (x, y), lbl in zip(deliveries, labels_d):
        ax.plot(x, y, 'o', markersize=20, color='#3498db', zorder=3)
        ax.text(x, y, lbl, ha='center', va='center', fontsize=9, fontweight='bold', color='white', zorder=4)
        ax.text(x+0.15, y+0.38, 'Delivery', ha='center', fontsize=7.5, color='#3498db')

    # Pairing arrows (precedence)
    for (px, py), (dx, dy) in zip(pickups, deliveries):
        ax.annotate('', xy=(dx, dy), xytext=(px, py),
                    arrowprops=dict(arrowstyle='->', color='purple', lw=2))

    # Depot to all pickups
    for (px, py) in pickups:
        ax.annotate('', xy=(px, py), xytext=depot_orig,
                    arrowprops=dict(arrowstyle='->', color='gray', lw=1, linestyle='dotted'))

    # All deliveries to dest depot
    for (dx, dy) in deliveries:
        ax.annotate('', xy=depot_dest, xytext=(dx, dy),
                    arrowprops=dict(arrowstyle='->', color='gray', lw=1, linestyle='dotted'))

    ax.text(2.5, -0.3,
            "Purple arrows = precedence: pickup must happen before its paired delivery",
            ha='center', fontsize=8.5, style='italic', color='purple')

    plt.tight_layout()
    savefig("fig_pdp_graph.pdf")


# ─────────────────────────────────────────────
# Figure 3: PDPTW model – capacity & time windows
# ─────────────────────────────────────────────
def fig_pdptw_constraints():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle("PDPTW Constraints Illustrated", fontsize=12, fontweight='bold')

    # Left: capacity feasibility
    ax = axes[0]
    ax.set_title("Vehicle Load Over Route", fontweight='bold', fontsize=10)
    stops = ['Depot\n(0)', 'P1\n(+3)', 'P2\n(+2)', 'D1\n(-3)', 'D2\n(-2)', 'Depot\n(2n+1)']
    load  = [0, 3, 5, 2, 0, 0]
    x_pos = range(len(stops))
    bars = ax.bar(x_pos, load, color=['#e74c3c','#2ecc71','#2ecc71','#3498db','#3498db','#e74c3c'],
                  alpha=0.8, edgecolor='black', linewidth=0.8)
    ax.axhline(y=5, color='red', linestyle='--', linewidth=2, label='Capacity Q=5')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(stops, fontsize=9)
    ax.set_ylabel("Vehicle Load", fontsize=10)
    ax.set_ylim(0, 7)
    ax.legend(fontsize=9)
    for bar, val in zip(bars, load):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.1, str(val),
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.text(2.5, 6.3, "Load never exceeds Q=5", ha='center', fontsize=9,
            color='darkred', style='italic')

    # Right: time window feasibility
    ax = axes[1]
    ax.set_title("Time Window Feasibility at Each Node", fontweight='bold', fontsize=10)
    node_names = ['P1', 'P2', 'D1', 'D2']
    tw_early = [5, 10, 20, 25]
    tw_late  = [15, 20, 35, 40]
    arrive   = [8, 12, 28, 32]
    y_pos = range(len(node_names))
    ax.barh(y_pos, [l - e for e, l in zip(tw_early, tw_late)],
            left=tw_early, color='lightblue', edgecolor='steelblue', linewidth=1.2, label='Time Window [e, l]')
    ax.plot(arrive, y_pos, 'r^', markersize=10, label='Arrival time', zorder=5)
    ax.set_yticks(y_pos); ax.set_yticklabels(node_names, fontsize=10)
    ax.set_xlabel("Time", fontsize=10)
    ax.legend(fontsize=9, loc='lower right')
    for arr, y in zip(arrive, y_pos):
        ax.text(arr+0.3, y, f't={arr}', va='center', fontsize=8.5, color='darkred')
    ax.text(20, 3.6, "All arrivals within windows", ha='center', fontsize=9,
            color='darkgreen', style='italic')

    plt.tight_layout()
    savefig("fig_pdptw_constraints.pdf")


# ─────────────────────────────────────────────
# Figure 4: 1-M-1 Problem illustration
# ─────────────────────────────────────────────
def fig_1m1_problem():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("One-to-Many-to-One (1-M-1) PDP Structure", fontsize=12, fontweight='bold')

    for ax_idx, (ax, title, route_type) in enumerate(zip(axes,
            ["Deliveries then Pickups\n(Separate loads)", "Mixed Loads\n(Simultaneous delivery & pickup)"],
            ['sep', 'mix'])):
        ax.set_title(title, fontweight='bold', fontsize=10)
        ax.set_xlim(-0.3, 5.3); ax.set_ylim(-0.3, 4.3)
        ax.axis('off')

        depot = (2.5, 4.0)
        del_nodes = [(1.0, 2.5), (2.0, 1.2), (0.5, 1.0)]
        pick_nodes = [(4.0, 2.5), (3.5, 1.2), (4.5, 0.8)]

        ax.plot(*depot, 's', markersize=22, color='#e74c3c', zorder=3)
        ax.text(depot[0], depot[1], 'D', ha='center', va='center', fontsize=10,
                fontweight='bold', color='white', zorder=4)

        for i, (x, y) in enumerate(del_nodes, 1):
            ax.plot(x, y, 'o', markersize=18, color='#3498db', zorder=3)
            ax.text(x, y, f'd{i}', ha='center', va='center', fontsize=9,
                    fontweight='bold', color='white', zorder=4)

        for i, (x, y) in enumerate(pick_nodes, 1):
            ax.plot(x, y, 'o', markersize=18, color='#2ecc71', zorder=3)
            ax.text(x, y, f'p{i}', ha='center', va='center', fontsize=9,
                    fontweight='bold', color='white', zorder=4)

        if route_type == 'sep':
            # Route: depot -> d1 -> d2 -> d3 -> p1 -> p2 -> p3 -> depot
            route = [depot] + del_nodes + pick_nodes + [depot]
        else:
            # Mixed route visiting both
            route = [depot, del_nodes[0], pick_nodes[0], del_nodes[1], pick_nodes[1], del_nodes[2], pick_nodes[2], depot]

        rxs, rys = zip(*route)
        ax.plot(rxs, rys, '-', color='purple', lw=1.8, alpha=0.6, zorder=1)
        for i in range(len(route)-1):
            mx = (route[i][0]+route[i+1][0])/2
            my = (route[i][1]+route[i+1][1])/2
            ax.annotate('', xy=route[i+1], xytext=route[i],
                        arrowprops=dict(arrowstyle='->', color='purple', lw=1.4))

        patch_d = mpatches.Patch(color='#3498db', label='Delivery customers')
        patch_p = mpatches.Patch(color='#2ecc71', label='Pickup customers')
        ax.legend(handles=[patch_d, patch_p], loc='lower right', fontsize=8)

    plt.tight_layout()
    savefig("fig_1m1_problem.pdf")


# ─────────────────────────────────────────────
# Figure 5: Branch-and-Cut for PDPTW – bound evolution
# ─────────────────────────────────────────────
def fig_branch_cut():
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_title("Branch-and-Cut: Bound Evolution for PDPTW\n(Illustrative convergence)", fontsize=11, fontweight='bold')

    nodes = list(range(1, 21))
    np.random.seed(42)
    upper_bounds = 1000 - np.cumsum(np.random.exponential(12, 20))
    upper_bounds = np.clip(upper_bounds, 700, 1000)
    lower_bounds = 600 + np.cumsum(np.random.exponential(8, 20))
    lower_bounds = np.clip(lower_bounds, 600, upper_bounds)

    ax.plot(nodes, upper_bounds, 'b-o', markersize=6, linewidth=2, label='Upper bound (best feasible)')
    ax.plot(nodes, lower_bounds, 'r-s', markersize=6, linewidth=2, label='Lower bound (LP relaxation)')
    ax.fill_between(nodes, lower_bounds, upper_bounds, alpha=0.15, color='green')
    ax.axhline(y=lower_bounds[-1], color='green', linestyle='--', linewidth=1.5, label='Optimal (gap closed)')

    ax.set_xlabel("Branch-and-Bound Node", fontsize=11)
    ax.set_ylabel("Objective Value (total distance)", fontsize=11)
    ax.legend(fontsize=9)
    ax.set_xlim(1, 20); ax.set_ylim(550, 1050)
    ax.text(10, 660, "Gap closes as\nmore nodes explored", ha='center', fontsize=9, style='italic', color='darkgreen')

    plt.tight_layout()
    savefig("fig_branch_cut.pdf")


# ─────────────────────────────────────────────
# Figure 6: LIFO vs FIFO loading constraint
# ─────────────────────────────────────────────
def fig_loading_constraints():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle("Loading Constraints: LIFO vs FIFO", fontsize=12, fontweight='bold')

    # LIFO
    ax = axes[0]
    ax.set_title("LIFO (Last-In First-Out)\n– rear loading, stack policy", fontweight='bold', fontsize=10)
    ax.set_xlim(0, 10); ax.set_ylim(0, 5)
    ax.axis('off')

    # Draw vehicle as rectangle
    vehicle = FancyBboxPatch((0.5, 1.5), 8, 2.5, boxstyle="round,pad=0.1",
                              facecolor='lightyellow', edgecolor='black', linewidth=2)
    ax.add_patch(vehicle)
    ax.text(4.5, 4.3, "Vehicle (rear loading)", ha='center', fontsize=9, fontweight='bold')
    ax.text(8.8, 2.5, "REAR\n(access\npoint)", ha='center', fontsize=8, color='red')

    # Packages loaded in order: A first (deep inside), then B, then C (near rear)
    colors_pkg = ['#3498db', '#e67e22', '#2ecc71']
    labels_pkg = ['A\n(loaded 1st)', 'B\n(loaded 2nd)', 'C\n(loaded 3rd)']
    x_positions = [1.2, 3.2, 5.2]
    for x, lbl, col in zip(x_positions, labels_pkg, colors_pkg):
        box = FancyBboxPatch((x, 1.8), 1.5, 1.8, boxstyle="round,pad=0.05",
                              facecolor=col, edgecolor='black', linewidth=1.2, alpha=0.85)
        ax.add_patch(box)
        ax.text(x+0.75, 2.7, lbl, ha='center', va='center', fontsize=8.5, fontweight='bold', color='white')

    ax.annotate('', xy=(7.5, 2.7), xytext=(6.8, 2.7),
                arrowprops=dict(arrowstyle='->', color='red', lw=2.5))
    ax.text(4.5, 0.8, "Must unload C, then B, then A  (stack order)", ha='center',
            fontsize=9, color='darkred', style='italic')
    ax.text(4.5, 0.3, "LIFO: delivery sequence = reverse of loading sequence", ha='center',
            fontsize=8.5, color='gray')

    # FIFO
    ax = axes[1]
    ax.set_title("FIFO (First-In First-Out)\n– side/conveyor loading", fontweight='bold', fontsize=10)
    ax.set_xlim(0, 10); ax.set_ylim(0, 5)
    ax.axis('off')

    vehicle2 = FancyBboxPatch((0.5, 1.5), 8, 2.5, boxstyle="round,pad=0.1",
                               facecolor='lightyellow', edgecolor='black', linewidth=2)
    ax.add_patch(vehicle2)
    ax.text(4.5, 4.3, "Vehicle (conveyor loading)", ha='center', fontsize=9, fontweight='bold')
    ax.text(8.8, 2.5, "EXIT\npoint", ha='center', fontsize=8, color='red')

    x_positions2 = [5.5, 3.5, 1.5]  # A at front (exits first), then B, C at back
    labels2 = ['A\n(exits 1st)', 'B\n(exits 2nd)', 'C\n(exits 3rd)']
    for x, lbl, col in zip(x_positions2, labels2, colors_pkg):
        box = FancyBboxPatch((x, 1.8), 1.5, 1.8, boxstyle="round,pad=0.05",
                              facecolor=col, edgecolor='black', linewidth=1.2, alpha=0.85)
        ax.add_patch(box)
        ax.text(x+0.75, 2.7, lbl, ha='center', va='center', fontsize=8.5, fontweight='bold', color='white')

    ax.annotate('', xy=(7.5, 2.7), xytext=(6.8, 2.7),
                arrowprops=dict(arrowstyle='->', color='red', lw=2.5))
    ax.text(4.5, 0.8, "A exits first (loaded first), then B, then C", ha='center',
            fontsize=9, color='darkgreen', style='italic')
    ax.text(4.5, 0.3, "FIFO: delivery sequence = same as loading sequence", ha='center',
            fontsize=8.5, color='gray')

    plt.tight_layout()
    savefig("fig_loading_constraints.pdf")


# ─────────────────────────────────────────────
# Figure 7: Insertion heuristic for PDP
# ─────────────────────────────────────────────
def fig_insertion_heuristic():
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.suptitle("Insertion Heuristic for PDPTW: Step-by-Step", fontsize=12, fontweight='bold')

    depot = (2.5, 2.5)
    requests = [
        ((1.0, 3.5), (4.0, 3.5), 'Req 1'),
        ((1.0, 1.2), (4.0, 1.2), 'Req 2'),
        ((2.5, 3.8), (2.5, 0.5), 'Req 3'),
    ]

    def draw_step(ax, title, route_segs, req_shown, highlight=None):
        ax.set_title(title, fontsize=9.5, fontweight='bold')
        ax.set_xlim(0, 5); ax.set_ylim(0, 5); ax.axis('off')
        ax.plot(*depot, 's', markersize=20, color='#e74c3c', zorder=4)
        ax.text(depot[0], depot[1], 'D', ha='center', va='center', fontsize=9,
                fontweight='bold', color='white', zorder=5)
        for (px, py), (dx, dy), lbl in req_shown:
            col_p = '#f39c12' if lbl == highlight else '#2ecc71'
            col_d = '#f39c12' if lbl == highlight else '#3498db'
            ax.plot(px, py, 'o', markersize=14, color=col_p, zorder=4)
            ax.text(px, py, lbl[0]+'\nP', ha='center', va='center', fontsize=6.5,
                    fontweight='bold', color='white', zorder=5)
            ax.plot(dx, dy, '^', markersize=14, color=col_d, zorder=4)
            ax.text(dx, dy, lbl[0]+'\nD', ha='center', va='center', fontsize=6.5,
                    fontweight='bold', color='white', zorder=5)
        if route_segs:
            for seg in route_segs:
                xs, ys = zip(*seg)
                ax.plot(xs, ys, '-', color='steelblue', lw=1.8, alpha=0.7, zorder=2)
                ax.annotate('', xy=seg[-1], xytext=seg[0],
                            arrowprops=dict(arrowstyle='->', color='steelblue', lw=1.5))

    p1, d1, _ = requests[0]
    p2, d2, _ = requests[1]
    p3, d3, _ = requests[2]

    # Step 1: empty route, insert request 1
    draw_step(axes[0], "Step 1: Insert Request 1\n(cheapest feasible position)",
              [[depot, p1], [p1, d1], [d1, depot]],
              [requests[0]], highlight='Req 1')

    # Step 2: route with req1, insert req2
    draw_step(axes[1], "Step 2: Insert Request 2\n(best cost increase)",
              [[depot, p1], [p1, p2], [p2, d1], [d1, d2], [d2, depot]],
              [requests[0], requests[1]], highlight='Req 2')

    # Step 3: route with req1+2+3
    draw_step(axes[2], "Step 3: Insert Request 3\n(final route)",
              [[depot, p1], [p1, p3], [p3, p2], [p2, d3], [d3, d1], [d1, d2], [d2, depot]],
              requests, highlight='Req 3')

    plt.tight_layout()
    savefig("fig_insertion_heuristic.pdf")


# ─────────────────────────────────────────────
# Figure 8: Tabu Search neighbourhood moves for PDP
# ─────────────────────────────────────────────
def fig_tabu_moves():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Tabu Search Neighbourhood Moves for PDP", fontsize=12, fontweight='bold')

    def draw_route(ax, route_nodes, colors_list, title, highlight_move=None):
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
        pos = {
            'D': (0.5, 0.9), 'P1': (0.1, 0.6), 'P2': (0.4, 0.3),
            'D1': (0.7, 0.6), 'D2': (0.9, 0.3)
        }
        for node, (x, y) in pos.items():
            c = colors_list.get(node, 'gray')
            shape = 's' if node == 'D' else 'o'
            ax.plot(x, y, shape, markersize=20, color=c, zorder=4)
            ax.text(x, y, node, ha='center', va='center', fontsize=8.5,
                    fontweight='bold', color='white', zorder=5)
        for i in range(len(route_nodes)-1):
            s, e = route_nodes[i], route_nodes[i+1]
            sx, sy = pos[s]; ex, ey = pos[e]
            col = 'red' if highlight_move and (s, e) in highlight_move else 'steelblue'
            ax.annotate('', xy=(ex, ey), xytext=(sx, sy),
                        arrowprops=dict(arrowstyle='->', color=col, lw=2.0 if col=='red' else 1.4))

    node_colors = {'D': '#e74c3c', 'P1': '#2ecc71', 'P2': '#2ecc71', 'D1': '#3498db', 'D2': '#3498db'}

    # Before: D -> P1 -> P2 -> D1 -> D2 -> D
    draw_route(axes[0], ['D', 'P1', 'P2', 'D1', 'D2', 'D'], node_colors,
               "Before: Route D→P1→P2→D1→D2→D\n(Current solution)")

    # After relocate: move (P2, D2) pair before D1
    draw_route(axes[1], ['D', 'P1', 'D', 'P2', 'D1', 'D2'], node_colors,
               "After: Relocate pair (P2,D2)\n(Neighbourhood move – OR-opt)",
               highlight_move={('D', 'P2'), ('P2', 'D1')})

    plt.tight_layout()
    savefig("fig_tabu_moves.pdf")


# ─────────────────────────────────────────────
# Figure 9: Multiple vehicles – fleet routing
# ─────────────────────────────────────────────
def fig_multiple_vehicles():
    fig, ax = plt.subplots(figsize=(9, 6.5))
    ax.set_title("Multi-Vehicle PDPTW Solution\n(Each vehicle handles a subset of requests)",
                 fontsize=11, fontweight='bold')
    ax.set_xlim(-0.5, 7.5); ax.set_ylim(-0.5, 6.0); ax.axis('off')

    depot = (3.5, 5.5)
    ax.plot(*depot, 's', markersize=24, color='#e74c3c', zorder=5)
    ax.text(depot[0], depot[1], 'Depot', ha='center', va='center', fontsize=8,
            fontweight='bold', color='white', zorder=6)

    routes = [
        {
            'pickups': [(1.0, 4.0), (0.5, 2.0)],
            'deliveries': [(2.0, 2.5), (1.5, 0.8)],
            'color': '#2980b9',
            'label': 'Vehicle 1'
        },
        {
            'pickups': [(5.5, 4.2), (6.5, 2.5)],
            'deliveries': [(4.5, 2.0), (6.0, 0.8)],
            'color': '#27ae60',
            'label': 'Vehicle 2'
        },
        {
            'pickups': [(3.0, 3.0), (4.0, 1.5)],
            'deliveries': [(2.5, 1.0), (5.0, 3.5)],
            'color': '#8e44ad',
            'label': 'Vehicle 3'
        },
    ]

    for route in routes:
        col = route['color']
        route_pts = [depot] + route['pickups'] + route['deliveries'] + [depot]
        for i, (x, y) in enumerate(route['pickups']):
            ax.plot(x, y, 'o', markersize=16, color=col, zorder=4, alpha=0.9)
            ax.text(x, y, f'P', ha='center', va='center', fontsize=8, color='white',
                    fontweight='bold', zorder=5)
        for i, (x, y) in enumerate(route['deliveries']):
            ax.plot(x, y, '^', markersize=16, color=col, zorder=4, alpha=0.9)
            ax.text(x, y, f'D', ha='center', va='center', fontsize=8, color='white',
                    fontweight='bold', zorder=5)
        rxs, rys = zip(*route_pts)
        ax.plot(rxs, rys, '-', color=col, lw=1.8, alpha=0.55, zorder=2)
        for i in range(len(route_pts)-1):
            ax.annotate('', xy=route_pts[i+1], xytext=route_pts[i],
                        arrowprops=dict(arrowstyle='->', color=col, lw=1.2, alpha=0.7))

    patches = [mpatches.Patch(color=r['color'], label=r['label']) for r in routes]
    patches.append(mpatches.Patch(color='#e74c3c', label='Depot'))
    ax.legend(handles=patches, loc='lower right', fontsize=9)

    plt.tight_layout()
    savefig("fig_multiple_vehicles.pdf")


# ─────────────────────────────────────────────
# Figure 10: 1-1 PDP request graph illustration
# ─────────────────────────────────────────────
def fig_11_pdp_model():
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.set_title("One-to-One PDP: Request Pairing and Precedence Graph\n"
                 "(n requests: n pickup nodes, n delivery nodes, plus depot copies)",
                 fontsize=10.5, fontweight='bold')
    ax.set_xlim(-0.5, 6.5); ax.set_ylim(-0.5, 4.0); ax.axis('off')

    depot0 = (0.0, 2.0)
    depot2n1 = (6.0, 2.0)
    pickups = [(1.5, 3.5), (1.5, 2.0), (1.5, 0.5)]
    deliveries = [(4.5, 3.5), (4.5, 2.0), (4.5, 0.5)]
    p_labels = ['1', '2', '3']
    d_labels = ['n+1', 'n+2', 'n+3']

    # Depot nodes
    for node, lbl in [(depot0, '0'), (depot2n1, '2n+1')]:
        ax.plot(*node, 's', markersize=22, color='#c0392b', zorder=4)
        ax.text(node[0], node[1], lbl, ha='center', va='center', fontsize=9,
                fontweight='bold', color='white', zorder=5)

    for (x, y), lbl in zip(pickups, p_labels):
        ax.plot(x, y, 'o', markersize=20, color='#27ae60', zorder=4)
        ax.text(x, y, lbl, ha='center', va='center', fontsize=9, fontweight='bold', color='white', zorder=5)

    for (x, y), lbl in zip(deliveries, d_labels):
        ax.plot(x, y, 'o', markersize=20, color='#2980b9', zorder=4)
        ax.text(x, y, lbl, ha='center', va='center', fontsize=8, fontweight='bold', color='white', zorder=5)

    # Pairing arrows
    for i, ((px, py), (dx, dy)) in enumerate(zip(pickups, deliveries)):
        ax.annotate('', xy=(dx, dy), xytext=(px, py),
                    arrowprops=dict(arrowstyle='->', color='purple', lw=2.5, linestyle='dashed'))
        ax.text((px+dx)/2, (py+dy)/2+0.15, f'req {i+1}', ha='center',
                fontsize=8.5, color='purple', fontweight='bold')

    # Sample route arc depot -> P1 -> P3 -> D1 -> P2 -> D3 -> D2 -> depot
    route_pts = [depot0, pickups[0], pickups[2], deliveries[0], pickups[1], deliveries[2], deliveries[1], depot2n1]
    for i in range(len(route_pts)-1):
        ax.annotate('', xy=route_pts[i+1], xytext=route_pts[i],
                    arrowprops=dict(arrowstyle='->', color='gray', lw=0.9, alpha=0.5))

    patch_p = mpatches.Patch(color='#27ae60', label='Pickup nodes (1..n)')
    patch_d = mpatches.Patch(color='#2980b9', label='Delivery nodes (n+1..2n)')
    patch_dep = mpatches.Patch(color='#c0392b', label='Depot (node 0 and 2n+1)')
    ax.legend(handles=[patch_p, patch_d, patch_dep], loc='lower center',
              fontsize=8.5, ncol=3, bbox_to_anchor=(0.5, -0.05))

    ax.text(3.0, -0.4, "Dashed purple = pairing constraint  |  Gray = sample vehicle route",
            ha='center', fontsize=8.5, style='italic', color='#555')

    plt.tight_layout()
    savefig("fig_11_pdp_model.pdf")


# ─────────────────────────────────────────────
# Figure 11: 3D container loading illustration
# ─────────────────────────────────────────────
def fig_3d_loading():
    fig = plt.figure(figsize=(9, 5.5))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title("3D Loading Constraints in PDP\n(Items must fit and be accessible at each delivery stop)",
                 fontsize=10, fontweight='bold')

    # Draw a simple container
    container = np.array([
        [0,0,0],[4,0,0],[4,3,0],[0,3,0],
        [0,0,2],[4,0,2],[4,3,2],[0,3,2]
    ])
    edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
    for i, j in edges:
        xs = [container[i,0], container[j,0]]
        ys = [container[i,1], container[j,1]]
        zs = [container[i,2], container[j,2]]
        ax.plot(xs, ys, zs, 'k-', lw=1.0, alpha=0.4)

    # Three items
    def draw_box_3d(ax, x0, y0, z0, dx, dy, dz, color, label):
        xx = [x0, x0+dx, x0+dx, x0, x0, x0+dx, x0+dx, x0]
        yy = [y0, y0, y0+dy, y0+dy, y0, y0, y0+dy, y0+dy]
        zz = [z0, z0, z0, z0, z0+dz, z0+dz, z0+dz, z0+dz]
        verts = [
            list(zip([xx[i] for i in [0,1,2,3]], [yy[i] for i in [0,1,2,3]], [zz[i] for i in [0,1,2,3]])),
            list(zip([xx[i] for i in [4,5,6,7]], [yy[i] for i in [4,5,6,7]], [zz[i] for i in [4,5,6,7]])),
            list(zip([xx[i] for i in [0,1,5,4]], [yy[i] for i in [0,1,5,4]], [zz[i] for i in [0,1,5,4]])),
            list(zip([xx[i] for i in [2,3,7,6]], [yy[i] for i in [2,3,7,6]], [zz[i] for i in [2,3,7,6]])),
        ]
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        poly = Poly3DCollection(verts, alpha=0.6, facecolor=color, edgecolor='black', linewidth=0.8)
        ax.add_collection3d(poly)
        ax.text(x0+dx/2, y0+dy/2, z0+dz+0.1, label, ha='center', va='bottom', fontsize=8, fontweight='bold')

    draw_box_3d(ax, 0.2, 0.2, 0, 1.5, 1.0, 0.8, '#3498db', 'Item A\n(deliver last)')
    draw_box_3d(ax, 2.0, 0.2, 0, 1.5, 1.0, 0.8, '#e67e22', 'Item B\n(deliver 2nd)')
    draw_box_3d(ax, 0.2, 1.5, 0, 3.5, 1.2, 0.9, '#2ecc71', 'Item C\n(deliver 1st)')

    ax.set_xlim(0, 4); ax.set_ylim(0, 3); ax.set_zlim(0, 2.5)
    ax.set_xlabel('Length'); ax.set_ylabel('Width'); ax.set_zlabel('Height')
    ax.text2D(0.5, -0.05, "Items must be arranged so each delivery can be accessed without moving others",
              transform=ax.transAxes, ha='center', fontsize=8.5, style='italic', color='#555')

    plt.tight_layout()
    savefig("fig_3d_loading.pdf")


# ─────────────────────────────────────────────
# Figure 12: LNS/ALNS performance comparison
# ─────────────────────────────────────────────
def fig_lns_performance():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("LNS and ALNS Performance on PDPTW Benchmarks", fontsize=12, fontweight='bold')

    # Left: gap from BKS for different methods
    methods = ['Exact\n(B&C)', 'Greedy\nInsert', 'Tabu\nSearch', 'LNS', 'ALNS']
    gap_pct = [0.0, 8.5, 3.2, 1.8, 0.6]
    colors_bar = ['#27ae60', '#e74c3c', '#e67e22', '#3498db', '#9b59b6']
    axes[0].set_title("Average Gap from Best Known Solution (%)\n(Lower is better)", fontsize=10, fontweight='bold')
    bars = axes[0].bar(methods, gap_pct, color=colors_bar, edgecolor='black', linewidth=0.8, alpha=0.85)
    axes[0].set_ylabel("Gap from BKS (%)", fontsize=10)
    axes[0].set_ylim(0, 12)
    for bar, val in zip(bars, gap_pct):
        axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.1, f'{val}%',
                     ha='center', va='bottom', fontsize=9.5, fontweight='bold')
    axes[0].axhline(0, color='black', lw=0.8)
    axes[0].text(2, 10.5, "Exact solves optimally\nbut only small instances",
                 ha='center', fontsize=8.5, style='italic', color='darkgreen')

    # Right: CPU time vs instance size
    n_requests = [10, 20, 30, 40, 50, 75, 100]
    cpu_exact = [0.5, 15, 180, 3600, None, None, None]
    cpu_lns   = [0.02, 0.05, 0.12, 0.3, 0.6, 2.1, 5.0]
    cpu_alns  = [0.03, 0.06, 0.14, 0.35, 0.7, 2.3, 5.5]

    ax2 = axes[1]
    ax2.set_title("CPU Time vs Instance Size\n(Log scale)", fontsize=10, fontweight='bold')
    valid_exact = [(n, t) for n, t in zip(n_requests, cpu_exact) if t is not None]
    if valid_exact:
        ne, te = zip(*valid_exact)
        ax2.semilogy(ne, te, 'go-', markersize=7, linewidth=2, label='Exact (B&C)')
    ax2.semilogy(n_requests, cpu_lns, 'b^--', markersize=7, linewidth=2, label='LNS')
    ax2.semilogy(n_requests, cpu_alns, 'm^-.', markersize=7, linewidth=2, label='ALNS')
    ax2.set_xlabel("Number of Requests", fontsize=10)
    ax2.set_ylabel("CPU Time (seconds, log scale)", fontsize=10)
    ax2.legend(fontsize=9)
    ax2.set_xlim(5, 105)
    ax2.text(55, 0.03, "Heuristics scale well\nbeyond exact methods",
             ha='center', fontsize=8.5, style='italic', color='steelblue')

    plt.tight_layout()
    savefig("fig_lns_performance.pdf")


# ─────────────────────────────────────────────
# Run all
# ─────────────────────────────────────────────
if __name__ == '__main__':
    print("Generating figures for Chapter 6: PDP for Goods Transportation...")
    fig_pdp_types()
    fig_pdp_graph()
    fig_pdptw_constraints()
    fig_1m1_problem()
    fig_branch_cut()
    fig_loading_constraints()
    fig_insertion_heuristic()
    fig_tabu_moves()
    fig_multiple_vehicles()
    fig_11_pdp_model()
    fig_3d_loading()
    fig_lns_performance()
    print("All figures generated successfully.")
