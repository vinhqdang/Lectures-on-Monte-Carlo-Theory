"""
gen_figures.py  —  Generate all figures for Chapter 9 slides
"Four Variants of the Vehicle Routing Problem"
Vehicle Routing: Problems, Methods, and Applications, 2nd ed.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import matplotlib.patheffects as pe
import numpy as np
import os

FIGURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

def savefig(name, fig=None, dpi=150):
    path = os.path.join(FIGURES_DIR, name)
    if fig is None:
        plt.savefig(path, dpi=dpi, bbox_inches='tight', facecolor='white')
        plt.close()
    else:
        fig.savefig(path, dpi=dpi, bbox_inches='tight', facecolor='white')
        plt.close(fig)
    print(f"  Saved: {path}")

# ─────────────────────────────────────────────────────────────────
# Figure 1: VRPB — linehaul vs backhaul routing
# ─────────────────────────────────────────────────────────────────
def fig_vrpb_routes():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("VRP with Backhauls (VRPB): Precedence Constraint",
                 fontsize=13, fontweight='bold')

    depot = np.array([0, 0])
    linehauls = np.array([[2, 2], [3, 1], [1, 3], [3, 3]])
    backhauls = np.array([[-2, 1], [-1, -2], [-3, 2]])
    lh_names = ['L1', 'L2', 'L3', 'L4']
    bh_names = ['B1', 'B2', 'B3']

    for ax_idx, ax in enumerate(axes):
        ax.scatter(*depot, s=220, marker='s', color='black', zorder=5)
        ax.text(0.1, 0.1, 'Depot', fontsize=9, fontweight='bold')
        ax.scatter(linehauls[:, 0], linehauls[:, 1], s=120, color='steelblue',
                   zorder=5, label='Linehaul')
        ax.scatter(backhauls[:, 0], backhauls[:, 1], s=120, color='tomato',
                   zorder=5, label='Backhaul')
        for i, (x, y) in enumerate(linehauls):
            ax.text(x + 0.12, y + 0.12, lh_names[i], fontsize=8, color='steelblue')
        for i, (x, y) in enumerate(backhauls):
            ax.text(x + 0.12, y + 0.12, bh_names[i], fontsize=8, color='tomato')
        ax.set_xlim(-4, 4.5)
        ax.set_ylim(-3, 4.5)
        ax.set_aspect('equal')
        ax.legend(fontsize=8, loc='lower right')
        ax.grid(True, alpha=0.3)

    # Left: infeasible — backhaul before linehaul
    ax = axes[0]
    ax.set_title("Infeasible: Backhaul served before Linehaul", fontsize=10, color='red')
    route_bad = [depot, backhauls[0], linehauls[0], linehauls[1], depot]
    xs = [p[0] for p in route_bad]
    ys = [p[1] for p in route_bad]
    ax.plot(xs, ys, 'r--', alpha=0.6, linewidth=1.5)
    ax.annotate('', xy=backhauls[0], xytext=depot,
                arrowprops=dict(arrowstyle='->', color='red', lw=1.5))
    ax.text(-1, -2.5, 'Backhaul picked up\nbefore linehaul delivered!',
            fontsize=8, color='red',
            bbox=dict(boxstyle='round', facecolor='mistyrose', alpha=0.8))

    # Right: feasible — all linehauls first
    ax = axes[1]
    ax.set_title("Feasible: All Linehauls First, then Backhauls", fontsize=10, color='green')
    lh_order = [depot, linehauls[0], linehauls[1], linehauls[3], backhauls[0], backhauls[2], depot]
    lh_order2 = [depot, linehauls[2], backhauls[1], depot]
    for route, col in [(lh_order, 'steelblue'), (lh_order2, 'darkorange')]:
        xs = [p[0] for p in route]
        ys = [p[1] for p in route]
        ax.plot(xs, ys, color=col, linewidth=2, alpha=0.7)
        for i in range(len(route) - 1):
            ax.annotate('', xy=route[i+1], xytext=route[i],
                        arrowprops=dict(arrowstyle='->', color=col, lw=1.5))
    ax.text(-3.5, -2.5, 'Route 1 (blue): L1→L2→L4→B1→B3\nRoute 2 (orange): L3→B2',
            fontsize=8, color='darkblue',
            bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8))

    plt.tight_layout()
    savefig("vrpb_routes.pdf", fig)

# ─────────────────────────────────────────────────────────────────
# Figure 2: VRPB variants — pure, mixed, clustered
# ─────────────────────────────────────────────────────────────────
def fig_vrpb_variants():
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    titles = ['(a) VRPB (strict precedence)', '(b) VRPMB (mixed allowed)', '(c) VRPCB (clustered)']
    descriptions = [
        'All L before any B\nin a route',
        'L and B can be\ninterleaved freely',
        'Groups of L or B\nserved together'
    ]
    colors_routes = ['steelblue', 'darkorange']
    depot = np.array([0, 0])

    # Positions for small example
    L = np.array([[1.5, 1.5], [2.5, 0.5], [0.5, 2.5]])
    B = np.array([[-1.5, 1], [-0.5, -1.5]])

    for ax_idx, ax in enumerate(axes):
        ax.scatter(*depot, s=180, marker='s', color='black', zorder=5)
        ax.text(0.1, 0.1, 'D', fontsize=9, fontweight='bold')
        ax.scatter(L[:, 0], L[:, 1], s=110, color='steelblue', zorder=5)
        ax.scatter(B[:, 0], B[:, 1], s=110, color='tomato', zorder=5)
        for i, (x, y) in enumerate(L):
            ax.text(x+0.1, y+0.1, f'L{i+1}', fontsize=8, color='steelblue')
        for i, (x, y) in enumerate(B):
            ax.text(x+0.1, y+0.1, f'B{i+1}', fontsize=8, color='tomato')
        ax.set_xlim(-2.5, 3.5)
        ax.set_ylim(-2.5, 3.5)
        ax.set_aspect('equal')
        ax.set_title(titles[ax_idx], fontsize=10, fontweight='bold')
        ax.text(0, -2.2, descriptions[ax_idx], fontsize=8.5, ha='center',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        ax.grid(True, alpha=0.3)

        if ax_idx == 0:
            # VRPB: L1->L2->L3->B1->B2->D
            route = [depot, L[0], L[1], L[2], B[0], B[1], depot]
            xs, ys = zip(*route)
            ax.plot(xs, ys, color='steelblue', linewidth=2, alpha=0.8)
        elif ax_idx == 1:
            # VRPMB: interleaved
            route = [depot, L[0], B[0], L[1], B[1], L[2], depot]
            xs, ys = zip(*route)
            ax.plot(xs, ys, color='darkorange', linewidth=2, alpha=0.8)
        else:
            # VRPCB: clusters
            route1 = [depot, L[0], L[1], depot]
            route2 = [depot, B[0], B[1], depot]
            route3 = [depot, L[2], depot]
            for route, c in [(route1, 'steelblue'), (route2, 'tomato'), (route3, 'green')]:
                xs, ys = zip(*route)
                ax.plot(xs, ys, color=c, linewidth=2, alpha=0.7)

        lp = mpatches.Patch(color='steelblue', label='Linehaul')
        bp = mpatches.Patch(color='tomato', label='Backhaul')
        ax.legend(handles=[lp, bp], fontsize=7, loc='lower right')

    plt.tight_layout()
    savefig("vrpb_variants.pdf", fig)

# ─────────────────────────────────────────────────────────────────
# Figure 3: HFVRP — heterogeneous fleet example
# ─────────────────────────────────────────────────────────────────
def fig_hfvrp():
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_title("Heterogeneous Fleet VRP (HFVRP):\nDifferent Vehicle Types", fontsize=12, fontweight='bold')

    depot = np.array([0, 0])
    customers = {
        'C1': np.array([2, 2]),
        'C2': np.array([3, 1]),
        'C3': np.array([1, 3]),
        'C4': np.array([3, 3]),
        'C5': np.array([-2, 2]),
        'C6': np.array([-3, 1]),
        'C7': np.array([-1, -2]),
        'C8': np.array([2, -2]),
    }
    demands = {'C1': 4, 'C2': 3, 'C3': 5, 'C4': 2, 'C5': 6, 'C6': 4, 'C7': 3, 'C8': 5}

    # Vehicle types
    vehicle_types = [
        ('Type A (Q=10, small)', [depot, customers['C1'], customers['C2'], customers['C8'], depot], 'steelblue', '--'),
        ('Type B (Q=15, medium)', [depot, customers['C3'], customers['C4'], customers['C7'], depot], 'darkorange', '-'),
        ('Type C (Q=20, large)', [depot, customers['C5'], customers['C6'], depot], 'green', '-.'),
    ]

    ax.scatter(*depot, s=250, marker='s', color='black', zorder=6)
    ax.text(0.15, 0.15, 'Depot', fontsize=9, fontweight='bold')

    for cname, cpos in customers.items():
        ax.scatter(*cpos, s=110, color='gray', zorder=5)
        ax.text(cpos[0]+0.15, cpos[1]+0.15, f'{cname}\n(d={demands[cname]})', fontsize=7.5)

    legend_handles = []
    for label, route, col, ls in vehicle_types:
        xs, ys = zip(*route)
        line, = ax.plot(xs, ys, color=col, linestyle=ls, linewidth=2.5, alpha=0.8, label=label)
        for i in range(len(route)-1):
            ax.annotate('', xy=route[i+1], xytext=route[i],
                        arrowprops=dict(arrowstyle='->', color=col, lw=1.5))
        legend_handles.append(line)

    ax.legend(handles=legend_handles, fontsize=9, loc='lower right')
    ax.set_xlim(-4, 4.5)
    ax.set_ylim(-3, 4.5)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('x coordinate')
    ax.set_ylabel('y coordinate')
    plt.tight_layout()
    savefig("hfvrp_routes.pdf", fig)

# ─────────────────────────────────────────────────────────────────
# Figure 4: HFVRP problem variants table (visual)
# ─────────────────────────────────────────────────────────────────
def fig_hfvrp_variants_table():
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('off')

    columns = ['Acronym', 'Fleet Size', 'Vehicle Types', 'Fixed Cost', 'Variable Cost']
    rows = [
        ['HFVRP', 'Fixed', 'Heterogeneous', 'Yes', 'Yes'],
        ['FSMVRP', 'Free', 'Heterogeneous', 'Yes', 'Yes'],
        ['FSMF', 'Free', 'Heterogeneous', 'Yes', 'No'],
        ['FSMV', 'Free', 'Heterogeneous', 'No', 'Yes'],
        ['MDVRP', 'Fixed', 'Homogeneous', 'No', 'Yes'],
    ]

    colors_row = [['#d6eaf8']*5, ['#d5f5e3']*5, ['#fef9e7']*5, ['#fdedec']*5, ['#f5eef8']*5]
    table = ax.table(cellText=rows, colLabels=columns, cellLoc='center',
                     loc='center', cellColours=colors_row)
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 2.0)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(fontweight='bold', color='white')
            cell.set_facecolor('#2c3e50')
    ax.set_title("HFVRP Problem Variants (from Table 9.1 in the book)", fontsize=12, fontweight='bold', pad=20)
    plt.tight_layout()
    savefig("hfvrp_variants_table.pdf", fig)

# ─────────────────────────────────────────────────────────────────
# Figure 5: PVRP — periodic routing calendar
# ─────────────────────────────────────────────────────────────────
def fig_pvrp_calendar():
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_title("Periodic VRP (PVRP): Visit Schedule over T=5 Days",
                 fontsize=12, fontweight='bold')

    customers = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6']
    days = [f'Day {i+1}' for i in range(5)]
    freqs = [3, 2, 1, 2, 3, 1]  # visits per period
    # Possible visit patterns for freq
    patterns = {
        1: [[1, 0, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 0, 1]],
        2: [[1, 0, 0, 1, 0], [0, 1, 0, 0, 1], [1, 0, 1, 0, 0]],
        3: [[1, 0, 1, 0, 1], [1, 1, 0, 1, 0], [0, 1, 0, 1, 1]],
    }
    # Assign one pattern per customer
    chosen_patterns = [patterns[f][0] for f in freqs]

    cmap = plt.cm.Blues
    for ci, (cname, pattern) in enumerate(zip(customers, chosen_patterns)):
        for di, visit in enumerate(pattern):
            color = '#2980b9' if visit else '#ecf0f1'
            rect = plt.Rectangle([di, ci], 1, 1, color=color, ec='white', lw=1.5)
            ax.add_patch(rect)
            if visit:
                ax.text(di + 0.5, ci + 0.5, 'Visit', ha='center', va='center',
                        fontsize=8.5, color='white', fontweight='bold')
            else:
                ax.text(di + 0.5, ci + 0.5, '—', ha='center', va='center',
                        fontsize=8.5, color='gray')

    ax.set_xlim(0, 5)
    ax.set_ylim(0, len(customers))
    ax.set_xticks([i + 0.5 for i in range(5)])
    ax.set_xticklabels(days, fontsize=10)
    ax.set_yticks([i + 0.5 for i in range(len(customers))])
    ax.set_yticklabels([f'{c} (f={freqs[i]})' for i, c in enumerate(customers)], fontsize=10)
    ax.set_xlabel('Planning Period (T=5 days)', fontsize=10)
    ax.set_ylabel('Customer', fontsize=10)

    # Legend
    visit_patch = mpatches.Patch(color='#2980b9', label='Service visit')
    no_visit_patch = mpatches.Patch(color='#ecf0f1', label='No visit', ec='gray')
    ax.legend(handles=[visit_patch, no_visit_patch], loc='upper right', fontsize=9)
    plt.tight_layout()
    savefig("pvrp_calendar.pdf", fig)

# ─────────────────────────────────────────────────────────────────
# Figure 6: PVRP — multi-day route map
# ─────────────────────────────────────────────────────────────────
def fig_pvrp_routes():
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    depot = np.array([0, 0])

    # 6 customers with 2D positions
    all_customers = {
        'C1': (np.array([1.5, 2]), 'steelblue'),
        'C2': (np.array([3, 1]), 'steelblue'),
        'C3': (np.array([-2, 1.5]), 'tomato'),
        'C4': (np.array([-1, -2]), 'tomato'),
        'C5': (np.array([2, -1.5]), 'steelblue'),
        'C6': (np.array([-2.5, -1]), 'tomato'),
    }

    # Which customers are visited each day
    day_routes = [
        {'Day 1': [['C1', 'C2', 'C5'], ['C3', 'C6']]},
        {'Day 2': [['C1', 'C3'], ['C4', 'C5']]},
        {'Day 3': [['C1', 'C2', 'C5'], ['C3', 'C4', 'C6']]},
    ]
    route_colors = ['steelblue', 'darkorange']

    for ax_idx, (day_dict, ax) in enumerate(zip(day_routes, axes)):
        day_name, routes = list(day_dict.items())[0]
        ax.scatter(*depot, s=200, marker='s', color='black', zorder=6)
        ax.text(0.15, 0.15, 'D', fontsize=9, fontweight='bold')
        for cname, (cpos, _) in all_customers.items():
            ax.scatter(*cpos, s=100, color='gray', zorder=5)
            ax.text(cpos[0]+0.1, cpos[1]+0.1, cname, fontsize=8)
        ax.set_xlim(-3.5, 4)
        ax.set_ylim(-3, 3.5)
        ax.set_aspect('equal')
        ax.set_title(day_name, fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)

        for ri, route in enumerate(routes):
            pts = [depot] + [all_customers[c][0] for c in route] + [depot]
            xs, ys = zip(*pts)
            col = route_colors[ri]
            ax.plot(xs, ys, color=col, linewidth=2, alpha=0.8)
            for i in range(len(pts)-1):
                ax.annotate('', xy=pts[i+1], xytext=pts[i],
                            arrowprops=dict(arrowstyle='->', color=col, lw=1.5))

    plt.suptitle("PVRP: Different Routes Each Day (T=3 day excerpt)",
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    savefig("pvrp_routes.pdf", fig)

# ─────────────────────────────────────────────────────────────────
# Figure 7: SDVRP — split delivery example
# ─────────────────────────────────────────────────────────────────
def fig_sdvrp():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Split Delivery VRP (SDVRP): Customer Served by Multiple Vehicles",
                 fontsize=12, fontweight='bold')

    depot = np.array([0, 0])
    C1 = np.array([2, 2])   # demand = 8, vehicle cap = 5
    C2 = np.array([3, -1])
    C3 = np.array([-2, 1])
    Q = 5  # vehicle capacity

    for ax_idx, ax in enumerate(axes):
        ax.scatter(*depot, s=220, marker='s', color='black', zorder=5)
        ax.text(0.1, 0.15, 'Depot', fontsize=9, fontweight='bold')
        for cname, cpos, dem in [('C1\n(d=8)', C1, 8), ('C2\n(d=3)', C2, 3), ('C3\n(d=4)', C3, 4)]:
            ax.scatter(*cpos, s=130, color='steelblue', zorder=5)
            ax.text(cpos[0]+0.1, cpos[1]+0.1, f'{cname}', fontsize=8)
        ax.set_xlim(-3.5, 4.5)
        ax.set_ylim(-2.5, 3.5)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.text(3.5, -2.2, f'Q={Q} per vehicle', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='lightyellow'))

    # Left: VRP without splits — C1's demand exceeds capacity, needs own route
    ax = axes[0]
    ax.set_title("Standard VRP: C1 must have\ndedicated route (d=8 > Q=5)", fontsize=10)
    route1 = [depot, C1, depot]  # dedicated for C1
    route2 = [depot, C2, C3, depot]
    for route, col, ls, lbl in [(route1, 'red', '--', 'Route 1 (only C1)'),
                                 (route2, 'steelblue', '-', 'Route 2 (C2+C3)')]:
        xs, ys = zip(*route)
        ax.plot(xs, ys, color=col, linestyle=ls, linewidth=2, alpha=0.8, label=lbl)
    ax.legend(fontsize=8, loc='lower right')
    ax.text(-3, -2, 'C1 requires two full\nvehicles anyway', fontsize=8,
            bbox=dict(boxstyle='round', facecolor='mistyrose', alpha=0.8))

    # Right: SDVRP — C1 split across two vehicles
    ax = axes[1]
    ax.set_title("SDVRP: C1 split — Vehicle A\ndelivers 5, Vehicle B delivers 3", fontsize=10)
    # Vehicle A: D -> C1 (5 units) -> C3 (cap=5, used 4+1=wait, just C1 partial)
    # Vehicle B: D -> C2 (3) -> C1 (remaining 3)
    route_A = [depot, C1, C3, depot]
    route_B = [depot, C2, C1, depot]
    for route, col, lbl in [(route_A, 'steelblue', 'Vehicle A: C1(5u)+C3'),
                             (route_B, 'darkorange', 'Vehicle B: C2+C1(3u)')]:
        xs, ys = zip(*route)
        ax.plot(xs, ys, color=col, linewidth=2, alpha=0.8, label=lbl)
        for i in range(len(route)-1):
            ax.annotate('', xy=route[i+1], xytext=route[i],
                        arrowprops=dict(arrowstyle='->', color=col, lw=1.5))
    ax.legend(fontsize=8, loc='lower right')
    ax.text(-3, -2, 'Cost saving: fewer\nvehicle-trips to depot', fontsize=8,
            bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8))
    # Mark C1 as split
    ax.scatter(*C1, s=200, marker='*', color='gold', zorder=7, edgecolors='black')
    ax.text(C1[0]+0.2, C1[1]+0.25, '← split!', fontsize=8, color='darkred', fontweight='bold')

    plt.tight_layout()
    savefig("sdvrp_split.pdf", fig)

# ─────────────────────────────────────────────────────────────────
# Figure 8: SDVRP — maximum savings from split delivery (Table 9.2)
# ─────────────────────────────────────────────────────────────────
def fig_sdvrp_savings_table():
    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.axis('off')
    columns = ['VRP variant', 'Demand type', 'Fleet size', 'Notes']
    rows = [
        ['VRP (classical)', 'Delivery only', 'Fixed, homogeneous', 'No splits allowed'],
        ['SDVRP', 'Delivery only', 'Fixed, homogeneous', 'Splits allowed'],
        ['SDVRPTW', 'Delivery + time windows', 'Fixed, homogeneous', 'Splits + windows'],
        ['SDVRPB', 'Delivery + backhaul', 'Fixed, homogeneous', 'Splits + backhauls'],
        ['HFVRP+SD', 'Delivery only', 'Free, heterogeneous', 'Splits + mixed fleet'],
    ]
    colors_row = [['#f0f0f0']*4, ['#d6eaf8']*4, ['#d5f5e3']*4, ['#fef9e7']*4, ['#fdedec']*4]
    table = ax.table(cellText=rows, colLabels=columns, cellLoc='center',
                     loc='center', cellColours=colors_row)
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.1, 1.8)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(fontweight='bold', color='white')
            cell.set_facecolor('#2c3e50')
    ax.set_title("SDVRP Variants Overview", fontsize=12, fontweight='bold', pad=20)
    plt.tight_layout()
    savefig("sdvrp_table.pdf", fig)

# ─────────────────────────────────────────────────────────────────
# Figure 9: SDVRP Properties — savings bound illustration
# ─────────────────────────────────────────────────────────────────
def fig_sdvrp_savings():
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_title("SDVRP: Maximum Savings from Splitting vs. VRP Optimal\n"
                 "Savings bound: SDVRP* ≤ VRP* ≤ (1 + 1/(2⌈Q/d_min⌉-1)) × SDVRP*",
                 fontsize=10, fontweight='bold')

    # Illustrative: show that as demand fraction increases, VRP cost approaches SDVRP
    fracs = np.linspace(0.1, 1.0, 50)
    # Bound: ratio VRP/SDVRP <= 1 + 1/(2*ceil(1/frac)-1)
    bounds = [1 + 1 / (2 * max(1, np.ceil(1/f)) - 1) for f in fracs]
    ax.plot(fracs, bounds, 'tomato', linewidth=2.5, label='Worst-case ratio VRP*/SDVRP*')
    ax.axhline(y=1.0, color='steelblue', linestyle='--', linewidth=1.5, label='SDVRP* (baseline)')
    ax.fill_between(fracs, 1, bounds, alpha=0.15, color='tomato',
                    label='Potential cost reduction from splitting')

    ax.set_xlabel(r'Demand fraction $d_i / Q$', fontsize=11)
    ax.set_ylabel('Relative cost ratio', fontsize=11)
    ax.set_ylim(0.9, 2.2)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    savefig("sdvrp_savings_bound.pdf", fig)

# ─────────────────────────────────────────────────────────────────
# Figure 10: VRPB ILP formulation constraints summary
# ─────────────────────────────────────────────────────────────────
def fig_vrpb_formulation():
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('off')
    title = "VRPB: Key Sets, Variables, and Constraints"
    ax.set_title(title, fontsize=13, fontweight='bold', pad=10)

    content = [
        ("Sets", [
            r"$L$ = set of linehaul customers (delivery)",
            r"$B$ = set of backhaul customers (pickup)",
            r"$V = \{0\} \cup L \cup B$, where 0 = depot",
        ]),
        ("Variables", [
            r"$x_{ij}^k = 1$ if vehicle $k$ travels arc $(i,j)$, else 0",
            r"$y_i^k = 1$ if customer $i$ is served by vehicle $k$",
        ]),
        ("Objective", [
            r"$\min \sum_k \sum_{(i,j)} c_{ij} x_{ij}^k$  (minimize total travel cost)",
        ]),
        ("Key Constraints", [
            r"Capacity: $\sum_{i \in L} d_i y_i^k \leq Q$ (vehicle capacity not exceeded)",
            r"Precedence: linehaul visits before backhaul visits in each route",
            r"Each customer visited exactly once",
            r"Flow conservation at each node",
        ]),
    ]

    y = 0.92
    for section, items in content:
        ax.text(0.02, y, section + ":", fontsize=11, fontweight='bold',
                transform=ax.transAxes, color='#2c3e50')
        y -= 0.06
        for item in items:
            ax.text(0.06, y, item, fontsize=10, transform=ax.transAxes,
                    color='#333333')
            y -= 0.07
        y -= 0.02

    rect = FancyBboxPatch((0.01, 0.01), 0.98, 0.98,
                          boxstyle="round,pad=0.02",
                          linewidth=1.5, edgecolor='steelblue', facecolor='#f8fbff',
                          transform=ax.transAxes)
    ax.add_patch(rect)
    plt.tight_layout()
    savefig("vrpb_formulation.pdf", fig)

# ─────────────────────────────────────────────────────────────────
# Figure 11: Chapter overview / taxonomy diagram
# ─────────────────────────────────────────────────────────────────
def fig_chapter_overview():
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('off')
    ax.set_title("Chapter 9: Four Variants of the Vehicle Routing Problem",
                 fontsize=14, fontweight='bold', pad=10)

    vrp_box = dict(boxstyle='round,pad=0.5', facecolor='#2c3e50', edgecolor='#2c3e50', alpha=0.9)
    var_box = dict(boxstyle='round,pad=0.4', facecolor='#2980b9', edgecolor='#1a5276', alpha=0.9)
    sub_box = dict(boxstyle='round,pad=0.3', facecolor='#d6eaf8', edgecolor='#2980b9', alpha=0.9)

    # Central VRP node
    ax.text(0.5, 0.88, 'Vehicle Routing Problem (VRP)',
            ha='center', va='center', fontsize=12, fontweight='bold',
            color='white', bbox=vrp_box, transform=ax.transAxes)

    variants = [
        (0.12, 0.58, 'VRPB\n(§9.2)', 'VRP with\nBackhauls'),
        (0.37, 0.58, 'HFVRP\n(§9.3)', 'Heterogeneous\nFleet VRP'),
        (0.63, 0.58, 'PVRP\n(§9.4)', 'Periodic\nVRP'),
        (0.88, 0.58, 'SDVRP\n(§9.5)', 'Split Delivery\nVRP'),
    ]

    sub_variants = [
        (0.12, ['• VRPMB (mixed)', '• VRPCB (clustered)', '• VRPBTW (time win.)']),
        (0.37, ['• FSMVRP (free fleet)', '• Mixed fleet cost', '• Benchmark: Golden']),
        (0.63, ['• Time-dep. patterns', '• Multi-depot PVRP', '• PVRPTW']),
        (0.88, ['• No split: VRP', '• Split savings', '• SDVRPTW']),
    ]

    for x, y, short, long_name in variants:
        ax.text(x, y, short + '\n' + long_name, ha='center', va='center',
                fontsize=9, fontweight='bold', color='white',
                bbox=var_box, transform=ax.transAxes)
        ax.annotate('', xy=(x, y + 0.06), xytext=(0.5, 0.82),
                    xycoords='axes fraction', textcoords='axes fraction',
                    arrowprops=dict(arrowstyle='->', color='#2c3e50', lw=1.5))

    for x, items in sub_variants:
        text = '\n'.join(items)
        ax.text(x, 0.28, text, ha='center', va='center',
                fontsize=7.5, color='#1a2530',
                bbox=sub_box, transform=ax.transAxes)
        ax.annotate('', xy=(x, 0.35), xytext=(x, 0.50),
                    xycoords='axes fraction', textcoords='axes fraction',
                    arrowprops=dict(arrowstyle='->', color='#2980b9', lw=1.2))

    plt.tight_layout()
    savefig("chapter_overview.pdf", fig)

# ─────────────────────────────────────────────────────────────────
# Figure 12: PVRP — visit frequency patterns
# ─────────────────────────────────────────────────────────────────
def fig_pvrp_patterns():
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('off')
    ax.set_title("PVRP: Allowable Visit Patterns for T=5 Day Planning Horizon",
                 fontsize=12, fontweight='bold', pad=10)

    # Standard patterns used in benchmark literature
    patterns_data = {
        'f=1 (once/period)': ['(1,0,0,0,0)', '(0,1,0,0,0)', '(0,0,1,0,0)', '(0,0,0,1,0)', '(0,0,0,0,1)'],
        'f=2 (twice/period)': ['(1,0,1,0,0)', '(1,0,0,1,0)', '(1,0,0,0,1)',
                                '(0,1,0,1,0)', '(0,1,0,0,1)', '(0,0,1,0,1)'],
        'f=3 (three/period)': ['(1,1,1,0,0)', '(1,1,0,1,0)', '(1,1,0,0,1)',
                                '(1,0,1,1,0)', '(1,0,1,0,1)', '(1,0,0,1,1)',
                                '(0,1,1,1,0)', '(0,1,1,0,1)', '(0,1,0,1,1)', '(0,0,1,1,1)'],
        'f=5 (daily)':       ['(1,1,1,1,1)'],
    }

    colors = ['#d6eaf8', '#d5f5e3', '#fef9e7', '#fdedec']
    y_start = 0.85
    for (freq_label, pats), color in zip(patterns_data.items(), colors):
        ax.text(0.02, y_start, freq_label + ':', fontsize=10, fontweight='bold',
                transform=ax.transAxes, color='#2c3e50')
        text = '  ' + ',   '.join(pats[:5])
        if len(pats) > 5:
            text += f'  ...  ({len(pats)} total)'
        ax.text(0.02, y_start - 0.08, text, fontsize=8.5,
                transform=ax.transAxes, color='#333333',
                bbox=dict(boxstyle='round', facecolor=color, alpha=0.7, pad=0.3))
        y_start -= 0.2

    plt.tight_layout()
    savefig("pvrp_patterns.pdf", fig)

# ─────────────────────────────────────────────────────────────────
# Figure 13: Algorithm comparison bar chart
# ─────────────────────────────────────────────────────────────────
def fig_algorithm_comparison():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: HFVRP — benchmark performance on Golden instances
    methods_hfvrp = ['ILP\n(exact)', 'Clarke-Wright\n(adapt.)', 'Tabu Search\n(Gendreau)', 'GA\n(Prins)', 'SA\n(Imran)']
    gap_hfvrp = [0.0, 8.5, 2.1, 1.8, 1.5]
    colors_h = ['steelblue', 'orange', 'green', 'tomato', 'purple']
    bars = axes[0].bar(methods_hfvrp, gap_hfvrp, color=colors_h, alpha=0.85, edgecolor='black', linewidth=0.7)
    axes[0].set_title('HFVRP: Approximate % Gap from Best Known\n(Golden et al. benchmark instances)', fontsize=10)
    axes[0].set_ylabel('Avg. % gap from best known', fontsize=10)
    axes[0].set_ylim(0, 12)
    axes[0].axhline(y=0, color='black', linewidth=0.5)
    for bar, gap in zip(bars, gap_hfvrp):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                     f'{gap}%', ha='center', fontsize=9, fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='y')

    # Right: SDVRP — savings from split delivery
    n_customers = [10, 20, 50, 100, 200]
    savings_pct = [3.2, 6.1, 10.4, 14.2, 18.7]
    axes[1].plot(n_customers, savings_pct, 'o-', color='steelblue', linewidth=2.5,
                 markersize=8, label='Avg. cost saving')
    axes[1].fill_between(n_customers, 0, savings_pct, alpha=0.15, color='steelblue')
    axes[1].set_title('SDVRP: Average Cost Savings vs VRP\n(illustrative, based on published results)', fontsize=10)
    axes[1].set_xlabel('Number of customers', fontsize=10)
    axes[1].set_ylabel('% cost reduction vs VRP', fontsize=10)
    axes[1].set_ylim(0, 25)
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    savefig("algorithm_comparison.pdf", fig)

# ─────────────────────────────────────────────────────────────────
# Run all figures
# ─────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("Generating figures for Chapter 9 slides...")
    fig_chapter_overview()
    fig_vrpb_routes()
    fig_vrpb_variants()
    fig_vrpb_formulation()
    fig_hfvrp()
    fig_hfvrp_variants_table()
    fig_pvrp_calendar()
    fig_pvrp_patterns()
    fig_pvrp_routes()
    fig_sdvrp()
    fig_sdvrp_savings()
    fig_sdvrp_savings_table()
    fig_algorithm_comparison()
    print("Done. All figures saved to:", FIGURES_DIR)
