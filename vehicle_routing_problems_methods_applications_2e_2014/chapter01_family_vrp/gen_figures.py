"""
gen_figures.py  —  Generate all figures for Chapter 1: The Family of VRP
Requires: matplotlib, numpy, scipy
Run with:  conda run -n py313 python3 gen_figures.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

FIGURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

def savefig(name, dpi=150, bbox_inches='tight'):
    path = os.path.join(FIGURES_DIR, name)
    plt.savefig(path, dpi=dpi, bbox_inches=bbox_inches, facecolor='white')
    plt.close()
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1: Basic CVRP Example — depot + 6 customers, 2 routes
# ─────────────────────────────────────────────────────────────────────────────
def fig_cvrp_example():
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_aspect('equal')

    depot = np.array([5, 5])
    customers = {
        1: np.array([2, 7]),
        2: np.array([1, 4]),
        3: np.array([3, 2]),
        4: np.array([7, 2]),
        5: np.array([8, 4]),
        6: np.array([8, 7]),
    }
    demands = {1: 3, 2: 4, 3: 3, 4: 4, 5: 3, 6: 3}

    route1 = [depot, customers[1], customers[2], customers[3], depot]
    route2 = [depot, customers[4], customers[5], customers[6], depot]

    colors = ['#2196F3', '#F44336']
    styles = ['-', '--']

    for route, color, style, label in zip([route1, route2], colors, styles,
                                          ['Route 1 (load=10)', 'Route 2 (load=10)']):
        xs = [p[0] for p in route]
        ys = [p[1] for p in route]
        ax.plot(xs, ys, color=color, linewidth=2.2, linestyle=style,
                label=label, alpha=0.85, zorder=2)
        for i in range(len(route) - 1):
            dx = route[i+1][0] - route[i][0]
            dy = route[i+1][1] - route[i][1]
            mx = route[i][0] + 0.45 * dx
            my = route[i][1] + 0.45 * dy
            ax.annotate('', xy=(mx + 0.01*dx, my + 0.01*dy),
                        xytext=(mx - 0.01*dx, my - 0.01*dy),
                        arrowprops=dict(arrowstyle='->', color=color, lw=1.8))

    ax.scatter(*depot, s=220, color='black', marker='s', zorder=5)
    ax.annotate('Depot (0)', depot, textcoords='offset points',
                xytext=(8, 6), fontsize=11, fontweight='bold')

    for cid, pos in customers.items():
        ax.scatter(*pos, s=160, color='#FF9800', zorder=5, edgecolors='black', linewidths=1.2)
        ax.annotate(f'C{cid}\n(d={demands[cid]})', pos,
                    textcoords='offset points', xytext=(8, 4), fontsize=9.5)

    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(0, 9.5)
    ax.set_title('CVRP Example: 6 Customers, 2 Vehicles, Capacity Q = 10',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('x-coordinate', fontsize=11)
    ax.set_ylabel('y-coordinate', fontsize=11)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.text(0.02, 0.02,
            'Total cost = sum of all arc distances\nEach route load <= Q = 10',
            transform=ax.transAxes, fontsize=9,
            verticalalignment='bottom',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow',
                      edgecolor='gray', alpha=0.8))
    savefig('cvrp_example.png')


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2: VRP Taxonomy / Family Tree
# ─────────────────────────────────────────────────────────────────────────────
def fig_vrp_taxonomy():
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8.5)
    ax.axis('off')
    fig.patch.set_facecolor('#FAFAFA')

    def box(ax, x, y, w, h, text, color='#BBDEFB', fontsize=9.5, bold=False, fgcolor='black'):
        rect = mpatches.FancyBboxPatch((x - w/2, y - h/2), w, h,
                                        boxstyle='round,pad=0.08',
                                        facecolor=color, edgecolor='#37474F',
                                        linewidth=1.5, zorder=3)
        ax.add_patch(rect)
        weight = 'bold' if bold else 'normal'
        ax.text(x, y, text, ha='center', va='center', fontsize=fontsize,
                fontweight=weight, zorder=4, multialignment='center', color=fgcolor)

    def arrow(ax, x1, y1, x2, y2):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color='#37474F', lw=1.4),
                    zorder=2)

    box(ax, 7, 8.0, 3.0, 0.70, 'VRP\n(Vehicle Routing Problem)',
        color='#1565C0', fontsize=11, bold=True, fgcolor='white')
    box(ax, 7, 6.8, 2.6, 0.60, 'CVRP\n(Capacitated VRP)',
        color='#1976D2', fontsize=10, bold=True, fgcolor='white')
    arrow(ax, 7, 7.65, 7, 7.10)

    variants = [
        (1.2, 5.3, 'VRPTW\n(Time Windows)'),
        (3.2, 5.3, 'PDP / PDPTW\n(Pickup & Delivery)'),
        (5.3, 5.3, 'SVRP\n(Stochastic)'),
        (7.2, 5.3, 'DVRP\n(Dynamic)'),
        (9.2, 5.3, 'VRPP\n(Profits)'),
        (11.2, 5.3, 'SDVRP\n(Split Deliveries)'),
        (13.0, 5.3, 'HVRP\n(Heterogeneous\nFleet)'),
    ]
    for (x, y, label) in variants:
        box(ax, x, y, 1.85, 0.80, label, color='#42A5F5', fontsize=8.5, fgcolor='white')
        arrow(ax, 7, 6.50, x, 5.70)

    sub_variants = [
        (1.2, 3.8, 'PVRP\n(Periodic)'),
        (3.2, 3.8, 'MDVRP\n(Multi-Depot)'),
        (5.3, 3.8, 'VRPB\n(Backhauls)'),
        (7.2, 3.8, 'IRP\n(Inventory\nRouting)'),
        (9.2, 3.8, 'VRPBTW\n(Backhauls +\nTime Windows)'),
        (11.2, 3.8, 'E-VRP\n(Electric\nVehicles)'),
    ]
    for (x, y, label) in sub_variants:
        box(ax, x, y, 1.85, 0.80, label, color='#90CAF9', fontsize=8.5)

    ax.text(7, 2.7,
            'Each variant adds one or more real-world constraints to the basic CVRP.\n'
            'Many logistics and distribution problems reduce to one of these variants.',
            ha='center', va='center', fontsize=10, style='italic', color='#37474F')

    ax.set_title('The Family of Vehicle Routing Problems (VRP)',
                 fontsize=14, fontweight='bold', pad=12)
    savefig('vrp_taxonomy.png')


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3: CVRP Graph representation
# ─────────────────────────────────────────────────────────────────────────────
def fig_cvrp_graph():
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.set_aspect('equal')
    ax.axis('off')

    n = 5
    depot = np.array([0.0, 0.0])
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    r = 2.5
    customers = {i+1: np.array([r * np.cos(a), r * np.sin(a)])
                 for i, a in enumerate(angles)}
    all_nodes = {0: depot}
    all_nodes.update(customers)

    drawn = set()
    for i in range(n+1):
        for j in range(n+1):
            if i != j and (j, i) not in drawn:
                p1 = all_nodes[i]
                p2 = all_nodes[j]
                ax.annotate('', xy=p2, xytext=p1,
                            arrowprops=dict(arrowstyle='->', color='#BDBDBD',
                                            lw=0.9, connectionstyle='arc3,rad=0.08'))
                drawn.add((i, j))

    for nid, pos in all_nodes.items():
        if nid == 0:
            ax.scatter(*pos, s=300, color='black', marker='s', zorder=5)
            ax.text(pos[0], pos[1] - 0.45, 'Depot\n(node 0)',
                    ha='center', fontsize=9, fontweight='bold')
        else:
            ax.scatter(*pos, s=200, color='#FF9800', zorder=5,
                       edgecolors='black', linewidths=1.2)
            ax.text(pos[0], pos[1] + 0.3, f'C{nid}',
                    ha='center', fontsize=9.5, fontweight='bold')

    ax.set_xlim(-3.5, 3.8)
    ax.set_ylim(-3.5, 3.5)
    ax.set_title('Complete Directed Graph G=(V, A) for the CVRP\n'
                 'V = {0, 1, ..., n}, depot = node 0, customers = nodes 1..n',
                 fontsize=11, fontweight='bold')
    savefig('cvrp_graph.png')


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4: VRPTW — time windows illustration
# ─────────────────────────────────────────────────────────────────────────────
def fig_time_windows():
    fig, ax = plt.subplots(figsize=(9, 4))

    customers_tw = ['C1', 'C2', 'C3', 'C4', 'C5']
    early = [10, 20, 35, 50, 65]
    late  = [25, 40, 55, 70, 85]
    service = [5, 5, 5, 5, 5]
    arrival = [12, 28, 38, 55, 70]

    y_pos = list(range(len(customers_tw)))

    for i, (e, l, s, c, arr) in enumerate(zip(early, late, service,
                                               customers_tw, arrival)):
        ax.barh(i, l - e, left=e, height=0.4, color='#90CAF9',
                edgecolor='#1565C0', linewidth=1.3,
                label='Time window' if i == 0 else '')
        ax.barh(i, s, left=arr, height=0.4, color='#EF9A9A',
                edgecolor='#B71C1C', linewidth=1.0,
                label='Service time' if i == 0 else '')
        ax.vlines(arr, i - 0.25, i + 0.25, color='#1B5E20', linewidth=2)
        ax.text(e, i + 0.28, f'[{e}, {l}]', fontsize=8.5, va='bottom',
                color='#1565C0', fontweight='bold')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(customers_tw, fontsize=11)
    ax.set_xlabel('Time', fontsize=11)
    ax.set_title('VRPTW: Time Windows [a_i, b_i] and Service Times s_i',
                 fontsize=12, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    ax.set_xlim(0, 100)
    ax.grid(True, axis='x', alpha=0.3)
    ax.text(0.02, 0.04,
            'Green line = vehicle arrival time\n'
            'Blue bar = allowed arrival window [a_i, b_i]\n'
            'Red bar = service duration s_i',
            transform=ax.transAxes, fontsize=8.5,
            verticalalignment='bottom',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow',
                      edgecolor='gray', alpha=0.8))
    plt.tight_layout()
    savefig('vrptw_time_windows.png')


# ─────────────────────────────────────────────────────────────────────────────
# Figure 5: Pickup and Delivery
# ─────────────────────────────────────────────────────────────────────────────
def fig_pickup_delivery():
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_aspect('equal')

    depot = np.array([5.0, 5.0])
    pickups  = {1: np.array([2.0, 7.0]), 2: np.array([1.0, 3.0]),
                3: np.array([7.0, 2.0])}
    delivers = {1: np.array([8.0, 7.0]), 2: np.array([8.0, 3.0]),
                3: np.array([3.0, 2.0])}

    colors_p = ['#1976D2', '#388E3C', '#F57C00']
    for i, color in zip([1, 2, 3], colors_p):
        p = pickups[i]
        d = delivers[i]
        ax.annotate('', xy=d, xytext=p,
                    arrowprops=dict(arrowstyle='->', color=color, lw=2.0,
                                    connectionstyle='arc3,rad=0.25'))
        mid = (p + d) / 2
        ax.text(mid[0], mid[1] + 0.3, f'Request {i}', fontsize=9,
                color=color, ha='center', fontweight='bold')

    for i, color in zip([1, 2, 3], colors_p):
        route = [depot, pickups[i], delivers[i], depot]
        xs = [pt[0] for pt in route]
        ys = [pt[1] for pt in route]
        ax.plot(xs, ys, '--', color=color, alpha=0.3, linewidth=1.2)

    ax.scatter(*depot, s=250, color='black', marker='s', zorder=5)
    ax.text(depot[0], depot[1] - 0.5, 'Depot', ha='center',
            fontsize=10, fontweight='bold')

    for i in [1, 2, 3]:
        p = pickups[i]
        d = delivers[i]
        ax.scatter(*p, s=160, color='#42A5F5', zorder=5,
                   edgecolors='black', lw=1.2)
        ax.scatter(*d, s=160, color='#EF5350', zorder=5,
                   marker='^', edgecolors='black', lw=1.2)
        ax.text(p[0] - 0.35, p[1], f'P{i}', fontsize=9.5,
                color='#1565C0', fontweight='bold')
        ax.text(d[0] + 0.1, d[1], f'D{i}', fontsize=9.5,
                color='#B71C1C', fontweight='bold')

    p_patch = mpatches.Patch(color='#42A5F5', label='Pickup location (P)')
    d_patch = mpatches.Patch(color='#EF5350', label='Delivery location (D)')
    ax.legend(handles=[p_patch, d_patch], loc='lower right', fontsize=9)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 9)
    ax.set_title('Pickup and Delivery Problem (PDP)\n'
                 'Each request: item picked up at P_i, delivered to D_i',
                 fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)
    savefig('pickup_delivery.png')


# ─────────────────────────────────────────────────────────────────────────────
# Figure 6: TSP vs VRP comparison
# ─────────────────────────────────────────────────────────────────────────────
def fig_tsp_vs_vrp():
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    np.random.seed(7)

    pts = np.random.rand(8, 2) * 8 + 1
    depot_pt = pts[0]

    ax = axes[0]
    order = [0, 2, 5, 7, 6, 4, 3, 1, 0]
    xs = pts[order, 0]
    ys = pts[order, 1]
    ax.plot(xs, ys, '-o', color='#1976D2', linewidth=2, markersize=8,
            markerfacecolor='#FF9800', markeredgecolor='black', zorder=3)
    ax.scatter(*depot_pt, s=250, color='black', marker='s', zorder=5)
    ax.text(depot_pt[0], depot_pt[1] + 0.4, 'Depot',
            ha='center', fontsize=9, fontweight='bold')
    for i, p in enumerate(pts):
        if i > 0:
            ax.text(p[0] + 0.2, p[1], f'C{i}', fontsize=8.5)
    ax.set_title('TSP -- Single Vehicle\nOne tour visits all customers',
                 fontsize=11, fontweight='bold')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    ax = axes[1]
    r1 = [0, 2, 5, 7]
    r2 = [0, 6, 4, 3, 1]
    for r, c, lbl in zip([r1, r2], ['#2196F3', '#F44336'],
                          ['Route 1', 'Route 2']):
        route_pts = list(pts[r]) + [depot_pt]
        rxs = [p[0] for p in route_pts]
        rys = [p[1] for p in route_pts]
        ax.plot(rxs, rys, '-o', color=c, linewidth=2, markersize=8,
                markerfacecolor='#FF9800', markeredgecolor='black',
                zorder=3, label=lbl)
    ax.scatter(*depot_pt, s=250, color='black', marker='s', zorder=5)
    ax.text(depot_pt[0], depot_pt[1] + 0.4, 'Depot',
            ha='center', fontsize=9, fontweight='bold')
    for i, p in enumerate(pts):
        if i > 0:
            ax.text(p[0] + 0.2, p[1], f'C{i}', fontsize=8.5)
    ax.set_title('VRP -- Multiple Vehicles\nTwo routes, each starting/ending at depot',
                 fontsize=11, fontweight='bold')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    ax.legend(fontsize=9, loc='lower right')

    fig.suptitle('TSP is a special case of VRP: one vehicle, no capacity constraint',
                 fontsize=12, fontweight='bold', y=1.01)
    plt.tight_layout()
    savefig('tsp_vs_vrp.png')


# ─────────────────────────────────────────────────────────────────────────────
# Figure 7: Heterogeneous Fleet
# ─────────────────────────────────────────────────────────────────────────────
def fig_heterogeneous_fleet():
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.axis('off')
    ax.set_xlim(0, 9)
    ax.set_ylim(0, 4.5)

    vehicles = [
        ('Small Van',    10, '#90CAF9', '#263238', 1),
        ('Medium Truck', 20, '#42A5F5', 'white',   2),
        ('Large Truck',  40, '#1565C0', 'white',   3),
    ]

    for i, (name, cap, bcolor, fcolor, idx) in enumerate(vehicles):
        y = 3.8 - i * 1.1
        w = cap * 0.10
        rect = mpatches.FancyBboxPatch((0.4, y - 0.28), w, 0.56,
                                        boxstyle='round,pad=0.05',
                                        facecolor=bcolor, edgecolor='#263238',
                                        linewidth=1.5, zorder=3)
        ax.add_patch(rect)
        ax.text(0.4 + w/2, y, f'{name}\nCapacity Q={cap}',
                ha='center', va='center', fontsize=10,
                color=fcolor, fontweight='bold', zorder=4)
        ax.text(5.2, y,
                f'Fixed cost: ${idx * 50}/route    '
                f'Variable cost: ${4 - idx:.0f}/km',
                fontsize=9.5, va='center')

    ax.set_title('Heterogeneous Fleet VRP (HVRP)\n'
                 'Vehicles differ in capacity, fixed cost, and variable travel cost',
                 fontsize=12, fontweight='bold')
    ax.text(0.5, 0.45,
            'Key challenge: assign customers to vehicle types to minimise total cost\n'
            '(larger capacity => higher fixed cost but fewer trips needed)',
            fontsize=9.5, va='center',
            transform=ax.transAxes,
            bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow',
                      edgecolor='gray', alpha=0.8))
    savefig('heterogeneous_fleet.png')


# ─────────────────────────────────────────────────────────────────────────────
# Figure 8: Stochastic VRP
# ─────────────────────────────────────────────────────────────────────────────
def fig_stochastic_vrp():
    try:
        from scipy import stats
        has_scipy = True
    except ImportError:
        has_scipy = False

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    ax = axes[0]
    np.random.seed(0)
    demands = np.random.normal(10, 2.5, 1000)
    ax.hist(demands, bins=30, color='#42A5F5', edgecolor='white',
            alpha=0.85, density=True)
    ax.axvline(10, color='#B71C1C', lw=2.2, label='Mean demand = 10')
    ax.axvline(15, color='#F57C00', lw=2, linestyle='--', label='Capacity Q = 15')
    if has_scipy:
        xs = np.linspace(2, 20, 200)
        ax.plot(xs, stats.norm.pdf(xs, 10, 2.5), 'k-', lw=2)
    ax.set_xlabel('Customer demand d_i', fontsize=11)
    ax.set_ylabel('Probability density', fontsize=11)
    ax.set_title('Stochastic Demand\nCustomer demand is a random variable',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.axis('off')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    stages = [
        (0.12, 0.65, 'Stage 1\n(Before departure)',
         'Plan routes based on\nexpected demands', '#BBDEFB'),
        (0.48, 0.65, 'Stage 2\n(During route)',
         'Observe actual demand;\nmay exceed capacity', '#FFE082'),
        (0.84, 0.65, 'Recourse Action\n(If overloaded)',
         'Emergency return to depot;\nload more goods', '#FFCDD2'),
    ]
    for x, y, title, desc, color in stages:
        rect = mpatches.FancyBboxPatch((x - 0.13, y - 0.22), 0.26, 0.44,
                                        boxstyle='round,pad=0.03',
                                        facecolor=color, edgecolor='#37474F',
                                        linewidth=1.4, zorder=3,
                                        transform=ax.transAxes)
        ax.add_patch(rect)
        ax.text(x, y + 0.07, title, ha='center', va='center', fontsize=8.5,
                fontweight='bold', transform=ax.transAxes)
        ax.text(x, y - 0.09, desc, ha='center', va='center', fontsize=8,
                transform=ax.transAxes)

    for x1, x2 in [(0.25, 0.35), (0.61, 0.71)]:
        ax.annotate('', xy=(x2, 0.65), xytext=(x1, 0.65),
                    xycoords='axes fraction', textcoords='axes fraction',
                    arrowprops=dict(arrowstyle='->', color='#37474F', lw=1.8))

    ax.set_title('Two-Stage Stochastic VRP', fontsize=11,
                 fontweight='bold', pad=10)
    ax.text(0.5, 0.22,
            'Objective: minimise expected total cost\n'
            '(planned routes + expected recourse penalties)',
            ha='center', va='center', fontsize=9.5,
            transform=ax.transAxes,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow',
                      edgecolor='gray', alpha=0.8))

    plt.tight_layout()
    savefig('stochastic_vrp.png')


# ─────────────────────────────────────────────────────────────────────────────
# Figure 9: Periodic VRP
# ─────────────────────────────────────────────────────────────────────────────
def fig_periodic_vrp():
    fig, ax = plt.subplots(figsize=(10, 5))

    n_days = 5
    n_cust = 6
    visit_patterns = {
        'C1 (freq=2)': [1, 0, 1, 0, 0],
        'C2 (freq=3)': [1, 0, 0, 1, 1],
        'C3 (freq=1)': [0, 0, 1, 0, 0],
        'C4 (freq=2)': [0, 1, 0, 1, 0],
        'C5 (freq=3)': [1, 1, 0, 0, 1],
        'C6 (freq=1)': [0, 0, 0, 1, 0],
    }

    cust_names = list(visit_patterns.keys())
    for i, (cname, pattern) in enumerate(visit_patterns.items()):
        y = n_cust - i - 1
        for d, visit in enumerate(pattern):
            if visit:
                ax.scatter(d + 1, y, s=200, color='#1976D2', marker='s',
                           zorder=4, edgecolors='black', lw=1.2)
                ax.text(d + 1, y, 'V', ha='center', va='center',
                        fontsize=9, color='white', fontweight='bold', zorder=5)
            else:
                ax.scatter(d + 1, y, s=200, color='#EEEEEE', marker='s',
                           zorder=3, edgecolors='#9E9E9E', lw=0.8)

    ax.set_xticks(range(1, n_days + 1))
    ax.set_xticklabels([f'Day {d}' for d in range(1, n_days + 1)], fontsize=11)
    ax.set_yticks(range(n_cust))
    ax.set_yticklabels(list(reversed(cust_names)), fontsize=10)
    ax.set_xlabel('Planning horizon (days)', fontsize=11)
    ax.set_ylabel('Customer', fontsize=11)
    ax.set_title('Periodic VRP (PVRP): Visit Schedule over a 5-Day Horizon\n'
                 'Each customer has a required visit frequency; planner assigns visit days',
                 fontsize=11, fontweight='bold')
    ax.set_xlim(0.3, 5.7)
    ax.set_ylim(-0.7, 5.7)
    ax.grid(True, alpha=0.2)
    ax.text(0.02, 0.04,
            'Blue square (V) = customer is visited on that day\n'
            'Goal: design routes for each day so total cost over planning horizon is minimised',
            transform=ax.transAxes, fontsize=9,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow',
                      edgecolor='gray', alpha=0.8))
    plt.tight_layout()
    savefig('periodic_vrp.png')


# ─────────────────────────────────────────────────────────────────────────────
# Figure 10: MTZ subtour elimination
# ─────────────────────────────────────────────────────────────────────────────
def fig_mtz_subtour():
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    nodes_main   = {0: (3.0, 3.0), 1: (1.5, 4.5), 2: (4.5, 4.5)}
    nodes_subtour= {3: (1.5, 1.5), 4: (4.5, 1.5)}

    ax = axes[0]
    ax.set_aspect('equal')
    ax.axis('off')

    for nid, pos in nodes_main.items():
        color  = 'black' if nid == 0 else '#FF9800'
        marker = 's' if nid == 0 else 'o'
        ax.scatter(*pos, s=200, color=color, marker=marker, zorder=5,
                   edgecolors='black', lw=1.2)
        lbl = 'Depot' if nid == 0 else f'C{nid}'
        ax.text(pos[0], pos[1] + 0.35, lbl, ha='center',
                fontsize=9.5, fontweight='bold')

    for nid, pos in nodes_subtour.items():
        ax.scatter(*pos, s=200, color='#EF5350', zorder=5, marker='o',
                   edgecolors='black', lw=1.2)
        ax.text(pos[0], pos[1] + 0.35, f'C{nid}', ha='center',
                fontsize=9.5, fontweight='bold')

    for (i, j) in [(0,1),(1,2),(2,0)]:
        p1 = np.array(nodes_main[i])
        p2 = np.array(nodes_main[j])
        ax.annotate('', xy=p2, xytext=p1,
                    arrowprops=dict(arrowstyle='->', color='#1976D2', lw=2.0,
                                    connectionstyle='arc3,rad=0.1'))

    for (i, j) in [(3,4),(4,3)]:
        p1 = np.array(nodes_subtour[i])
        p2 = np.array(nodes_subtour[j])
        ax.annotate('', xy=p2, xytext=p1,
                    arrowprops=dict(arrowstyle='->', color='#B71C1C', lw=2.0,
                                    connectionstyle='arc3,rad=0.25'))

    ax.text(3, 0.8, 'INVALID: subtour {C3,C4} disconnected from depot',
            ha='center', fontsize=9, color='#B71C1C', style='italic')
    ax.set_title('Without MTZ: subtours are allowed', fontsize=10,
                 fontweight='bold', color='#B71C1C')
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 6)

    ax = axes[1]
    ax.set_aspect('equal')
    ax.axis('off')

    all_nodes = {0: (3.0, 3.0), 1: (1.5, 4.5), 2: (4.5, 4.5),
                 3: (1.5, 1.5), 4: (4.5, 1.5)}

    for nid, pos in all_nodes.items():
        color  = 'black' if nid == 0 else '#FF9800'
        marker = 's' if nid == 0 else 'o'
        ax.scatter(*pos, s=200, color=color, marker=marker, zorder=5,
                   edgecolors='black', lw=1.2)
        lbl = 'Depot\n(u=0)' if nid == 0 else f'C{nid}\n(u={nid*7})'
        ax.text(pos[0], pos[1] + 0.3, lbl, ha='center',
                fontsize=8.5, fontweight='bold')

    for (i, j) in [(0,1),(1,3),(3,4),(4,2),(2,0)]:
        p1 = np.array(all_nodes[i])
        p2 = np.array(all_nodes[j])
        ax.annotate('', xy=p2, xytext=p1,
                    arrowprops=dict(arrowstyle='->', color='#388E3C', lw=2.0,
                                    connectionstyle='arc3,rad=0.1'))

    ax.set_title('With MTZ: single connected route', fontsize=10,
                 fontweight='bold', color='#1B5E20')
    ax.text(3, 0.5, 'Route: 0 -> 1 -> 3 -> 4 -> 2 -> 0',
            ha='center', fontsize=9.5, color='#388E3C')
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 6)

    fig.suptitle('MTZ Constraints: Eliminating Subtours in VRP/TSP Formulations',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    savefig('mtz_subtour.png')


# ─────────────────────────────────────────────────────────────────────────────
# Figure 11: Solomon clustered instance (C-type)
# ─────────────────────────────────────────────────────────────────────────────
def fig_solomon_instance():
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_aspect('equal')

    np.random.seed(101)
    depot = np.array([40.0, 50.0])
    cluster1 = np.random.normal([20, 65], 5, (5, 2))
    cluster2 = np.random.normal([60, 70], 5, (5, 2))
    cluster3 = np.random.normal([55, 25], 5, (5, 2))
    all_cust = np.vstack([cluster1, cluster2, cluster3])

    ax.scatter(*depot, s=280, color='black', marker='s', zorder=5)
    ax.text(depot[0] + 1.5, depot[1] + 2.5, 'Depot',
            fontsize=10, fontweight='bold')

    cluster_colors = ['#1976D2'] * 5 + ['#388E3C'] * 5 + ['#F57C00'] * 5
    for i, (pt, col) in enumerate(zip(all_cust, cluster_colors)):
        ax.scatter(*pt, s=140, color=col, zorder=4, edgecolors='black', lw=0.9)
        ax.text(pt[0] + 0.8, pt[1], f'{i+1}', fontsize=8)

    for cluster, color in zip([cluster1, cluster2, cluster3],
                               ['#1976D2', '#388E3C', '#F57C00']):
        order = np.argsort(np.arctan2(
            cluster[:, 1] - cluster[:, 1].mean(),
            cluster[:, 0] - cluster[:, 0].mean()))
        route_pts = [depot] + list(cluster[order]) + [depot]
        rxs = [p[0] for p in route_pts]
        rys = [p[1] for p in route_pts]
        ax.plot(rxs, rys, '-', color=color, linewidth=1.8, alpha=0.7)

    c1 = mpatches.Patch(color='#1976D2', label='Cluster 1 (Route 1)')
    c2 = mpatches.Patch(color='#388E3C', label='Cluster 2 (Route 2)')
    c3 = mpatches.Patch(color='#F57C00', label='Cluster 3 (Route 3)')
    ax.legend(handles=[c1, c2, c3], loc='lower right', fontsize=9)
    ax.set_xlim(5, 80)
    ax.set_ylim(5, 85)
    ax.set_xlabel('x-coordinate', fontsize=11)
    ax.set_ylabel('y-coordinate', fontsize=11)
    ax.set_title("Solomon C-type Benchmark Instance (Clustered Customers)\n"
                 "3 natural clusters => 3 vehicle routes",
                 fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)
    savefig('solomon_instance.png')


# ─────────────────────────────────────────────────────────────────────────────
# Figure 12: Split deliveries
# ─────────────────────────────────────────────────────────────────────────────
def fig_split_deliveries():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    depot = np.array([5.0, 4.0])
    customers_sd = {1: np.array([2.0, 6.0]),
                    2: np.array([8.0, 6.0]),
                    3: np.array([5.0, 1.0])}
    demands_sd = {1: 12, 2: 10, 3: 8}
    Q = 15

    routes_no_split = [
        [depot, customers_sd[1], depot],
        [depot, customers_sd[2], depot],
        [depot, customers_sd[3], depot],
    ]
    routes_split = [
        [depot, customers_sd[1], customers_sd[3], depot],
        [depot, customers_sd[2], customers_sd[1], depot],
    ]

    for ax_i, (ax, title, routes) in enumerate(zip(
            axes,
            ['Without Split Deliveries\n(3 routes needed)', 'With Split Deliveries\n(2 routes sufficient)'],
            [routes_no_split, routes_split])):
        colors = ['#1976D2', '#388E3C', '#F57C00']
        for r_idx, route in enumerate(routes):
            xs = [p[0] for p in route]
            ys = [p[1] for p in route]
            ax.plot(xs, ys, '-o', color=colors[r_idx], linewidth=2.0,
                    markersize=6, label=f'Route {r_idx+1}', alpha=0.8)

        ax.scatter(*depot, s=220, color='black', marker='s', zorder=5)
        ax.text(depot[0], depot[1] + 0.4, 'Depot', ha='center',
                fontsize=9.5, fontweight='bold')
        for cid, pos in customers_sd.items():
            ax.scatter(*pos, s=160, color='#FF9800', zorder=4,
                       edgecolors='black', lw=1.2)
            ax.text(pos[0] - 0.6, pos[1], f'C{cid}\n(d={demands_sd[cid]})',
                    fontsize=8.5)

        ax.set_xlim(0, 10)
        ax.set_ylim(-0.5, 8)
        ax.set_title(title, fontsize=10.5, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8.5, loc='lower right')
        ax.set_aspect('equal')

    fig.suptitle(f'Split Deliveries: customer demand can be served by multiple vehicles (Q={Q})',
                 fontsize=11, fontweight='bold')
    plt.tight_layout()
    savefig('split_deliveries.png')


# ─────────────────────────────────────────────────────────────────────────────
# Figure 13: Dynamic VRP — online vs offline illustration
# ─────────────────────────────────────────────────────────────────────────────
def fig_dynamic_vrp():
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axis('off')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5.5)

    timeline_y = 2.5
    ax.axhline(timeline_y, xmin=0.05, xmax=0.95, color='#37474F', lw=2.5)
    ax.annotate('', xy=(9.6, timeline_y), xytext=(0.4, timeline_y),
                arrowprops=dict(arrowstyle='->', color='#37474F', lw=2))
    ax.text(9.8, timeline_y, 'Time', fontsize=11, va='center', fontweight='bold')

    events = [
        (1.0, 't=0\nRoutes planned\n(known customers)', '#1976D2', 4.2),
        (3.0, 't=1\nNew request\narrives', '#388E3C', 1.0),
        (5.0, 't=2\nRoute\nre-optimised', '#F57C00', 4.2),
        (7.0, 't=3\nAnother request\narrives', '#E53935', 1.0),
        (9.0, 't=4\nFinal routes\nexecuted', '#7B1FA2', 4.2),
    ]

    for x, label, color, y_text in events:
        ax.vlines(x, timeline_y - 0.15, timeline_y + 0.15,
                  color=color, lw=2.5, zorder=5)
        ax.scatter(x, timeline_y, s=100, color=color, zorder=6, edgecolors='black')
        ax.text(x, y_text, label, ha='center', va='center', fontsize=8.5,
                color=color, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                          edgecolor=color, alpha=0.9))
        ax.annotate('', xy=(x, timeline_y + (0.2 if y_text > timeline_y else -0.2)),
                    xytext=(x, y_text + (-0.4 if y_text > timeline_y else 0.4)),
                    arrowprops=dict(arrowstyle='->', color=color, lw=1.2))

    ax.text(5, 0.3,
            'Key challenge: incorporate new information (requests, cancellations, traffic) into routes in real time',
            ha='center', va='center', fontsize=9.5, style='italic',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow',
                      edgecolor='gray', alpha=0.8))
    ax.set_title('Dynamic VRP (DVRP): Requests Arrive During Execution\n'
                 'Routes must be continuously updated as new information arrives',
                 fontsize=11, fontweight='bold')
    savefig('dynamic_vrp.png')


# ─────────────────────────────────────────────────────────────────────────────
# Run all
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("Generating figures for Chapter 1: The Family of VRP ...")
    fig_cvrp_example()
    fig_vrp_taxonomy()
    fig_cvrp_graph()
    fig_time_windows()
    fig_pickup_delivery()
    fig_tsp_vs_vrp()
    fig_heterogeneous_fleet()
    fig_stochastic_vrp()
    fig_periodic_vrp()
    fig_mtz_subtour()
    fig_solomon_instance()
    fig_split_deliveries()
    fig_dynamic_vrp()
    print("\nAll figures generated successfully.")
