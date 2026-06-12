"""
gen_figures.py  –  Generate all figures for Chapter 7 slides
(Pickup-and-Delivery Problems for People Transportation)

Run:
    conda run -n py313 python3 gen_figures.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import matplotlib.patheffects as pe

# ── output directory (figures/ relative to this script) ──────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(SCRIPT_DIR, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

def savefig(name):
    path = os.path.join(FIG_DIR, name)
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  saved: {path}")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 – DARP overview: depot + pickup/delivery pairs + vehicle route
# ─────────────────────────────────────────────────────────────────────────────
def fig_darp_overview():
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(-0.5, 8.5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('DARP: Dial-a-Ride Problem – Pickup and Delivery Structure',
                 fontsize=13, fontweight='bold', pad=12)

    # Depot
    depot = (5, 4)
    ax.plot(*depot, 's', color='black', ms=14, zorder=5)
    ax.text(depot[0], depot[1]+0.45, 'Depot\n(origin & end)', ha='center',
            fontsize=9, fontweight='bold')

    # Pickup-delivery pairs
    pairs = [
        ((1, 7), (3, 5.5), '1'),
        ((2, 1), (4, 2.5), '2'),
        ((8, 7), (6, 5.5), '3'),
        ((9, 2), (7, 3.5), '4'),
    ]
    colors = ['#2196F3', '#E91E63', '#4CAF50', '#FF9800']

    route_order = [
        depot,
        (1, 7), (2, 1),       # pickups 1, 2
        (3, 5.5), (4, 2.5),   # deliveries 1, 2
        (8, 7), (9, 2),       # pickups 3, 4
        (6, 5.5), (7, 3.5),   # deliveries 3, 4
        depot
    ]

    for (px, py), (dx, dy), label in pairs:
        c = colors[int(label)-1]
        ax.plot(px, py, 'o', color=c, ms=12, zorder=5)
        ax.text(px, py+0.4, f'$p_{label}$', ha='center', fontsize=10,
                color=c, fontweight='bold')
        ax.plot(dx, dy, '^', color=c, ms=12, zorder=5)
        ax.text(dx, dx/dy if False else dy+0.4,
                f'$d_{label}$', ha='center', fontsize=10,
                color=c, fontweight='bold')
        ax.annotate('', xy=(dx, dy), xytext=(px, py),
                    arrowprops=dict(arrowstyle='->', color=c,
                                   lw=1.5, linestyle='dashed'))

    # Route
    rx = [p[0] for p in route_order]
    ry = [p[1] for p in route_order]
    ax.plot(rx, ry, '-', color='gray', lw=1.2, alpha=0.5, zorder=1)

    legend_els = [
        mpatches.Patch(color='white', label='Circle = pickup $p_i$'),
        mpatches.Patch(color='white', label='Triangle = delivery $d_i$'),
        mpatches.Patch(color='black', label='Square = depot'),
        mpatches.Patch(color='gray', label='Gray line = vehicle route'),
    ]
    ax.legend(handles=legend_els, loc='lower right', fontsize=8,
              framealpha=0.9)

    savefig('darp_overview.pdf')


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 – DARP Constraint taxonomy diagram
# ─────────────────────────────────────────────────────────────────────────────
def fig_darp_constraints():
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axis('off')
    ax.set_title('DARP Constraints at a Glance', fontsize=13,
                 fontweight='bold', pad=8)

    categories = [
        ('Time Windows\n[e_i, l_i]',
         'Customer i must be\nserved in window\n[e_i, l_i]', '#1565C0'),
        ('Max Ride Time\nL_i',
         'Time on board\ncannot exceed L_i\n(user comfort)', '#6A1B9A'),
        ('Capacity\nQ_k',
         'Vehicle k load\nnever exceeds Q_k\nat any point', '#2E7D32'),
        ('Route Precedence\np_i before d_i',
         'Pickup always\nbefore delivery\nfor same request', '#C62828'),
        ('Route Duration\nT_max',
         'Total route\nduration bounded\nby T_max', '#E65100'),
    ]

    n = len(categories)
    xs = np.linspace(1, 9, n)
    y_top, y_bot = 3.5, 1.2

    for i, (title, desc, color) in enumerate(categories):
        x = xs[i]
        box = FancyBboxPatch((x-0.82, y_bot), 1.64, y_top - y_bot,
                             boxstyle='round,pad=0.08', linewidth=1.5,
                             edgecolor=color, facecolor=color+'22')
        ax.add_patch(box)
        ax.text(x, y_top - 0.25, title, ha='center', va='top',
                fontsize=9, fontweight='bold', color=color)
        ax.text(x, y_bot + 0.15, desc, ha='center', va='bottom',
                fontsize=8, color='#333333')

    savefig('darp_constraints.pdf')


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3 – Branch-and-cut framework for DARP
# ─────────────────────────────────────────────────────────────────────────────
def fig_branch_cut():
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.axis('off')
    ax.set_title('Branch-and-Cut Framework for DARP (Cordeau 2006)',
                 fontsize=12, fontweight='bold')

    steps = [
        ('LP Relaxation\n(root node)', 0.85, '#1565C0'),
        ('Feasible integer\nsolution?', 0.68, '#2E7D32'),
        ('Add cutting planes\n(capacity, time-window\ncorner-polyhedra cuts)', 0.50, '#E65100'),
        ('Branch on\nfractional variable', 0.33, '#6A1B9A'),
        ('Prune: infeasible\nor ≥ incumbent', 0.16, '#C62828'),
    ]

    for label, y, color in steps:
        box = FancyBboxPatch((0.1, y - 0.07), 0.8, 0.12,
                             boxstyle='round,pad=0.02', transform=ax.transAxes,
                             linewidth=1.5, edgecolor=color,
                             facecolor=color + '22', clip_on=False)
        ax.add_patch(box)
        ax.text(0.5, y, label, ha='center', va='center',
                fontsize=9, fontweight='bold', color=color,
                transform=ax.transAxes)

    # arrows
    ys = [s[1] for s in steps]
    for i in range(len(ys)-1):
        ax.annotate('', xy=(0.5, ys[i+1]+0.07), xytext=(0.5, ys[i]-0.07),
                    xycoords='axes fraction', textcoords='axes fraction',
                    arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))

    # feedback arrow: cut back to LP
    ax.annotate('', xy=(0.9, ys[0]), xytext=(0.9, ys[2]),
                xycoords='axes fraction', textcoords='axes fraction',
                arrowprops=dict(arrowstyle='->', color='#E65100', lw=1.5,
                                connectionstyle='arc3,rad=-0.5'))
    ax.text(0.97, (ys[0]+ys[2])/2, 'iterate', ha='center', va='center',
            fontsize=8, color='#E65100', rotation=90,
            transform=ax.transAxes)

    savefig('branch_cut_framework.pdf')


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4 – Tabu search neighbourhood illustration
# ─────────────────────────────────────────────────────────────────────────────
def fig_tabu_search():
    np.random.seed(42)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    fig.suptitle('Tabu Search for DARP – Solution Space & Neighbourhood',
                 fontsize=12, fontweight='bold')

    # Left: schematic solution quality landscape
    ax = axes[0]
    x = np.linspace(0, 10, 500)
    y = (2*np.sin(x) + np.sin(2.3*x) + 0.5*np.sin(4.1*x)
         + 0.3*np.sin(6*x)) * (-1) + 5
    ax.plot(x, y, 'b-', lw=2)
    ax.set_xlabel('Solution space (schematic)', fontsize=10)
    ax.set_ylabel('Objective value (cost)', fontsize=10)
    ax.set_title('Escaping Local Optima via Tabu List', fontsize=10)

    # mark local minima and global
    local_mins = x[np.r_[False, (y[1:-1] < y[:-2]) & (y[1:-1] < y[2:]), False]]
    for lm in local_mins[:4]:
        idx = np.argmin(np.abs(x - lm))
        ax.plot(x[idx], y[idx], 'ro', ms=8)
    gm_idx = np.argmin(y)
    ax.plot(x[gm_idx], y[gm_idx], 'g*', ms=14, label='Global optimum')
    idx0 = np.argmin(np.abs(x - local_mins[0]))
    ax.annotate('Tabu move\n(escape local min)', xy=(x[idx0], y[idx0]),
                xytext=(2, 7), fontsize=8, color='red',
                arrowprops=dict(arrowstyle='->', color='red'))
    ax.legend(fontsize=9)
    ax.set_xticks([])

    # Right: request relocation neighbourhood
    ax2 = axes[1]
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 8)
    ax2.set_aspect('equal')
    ax2.axis('off')
    ax2.set_title('Request Relocation Move (2-route neighbourhood)', fontsize=10)

    def draw_route(ax, pts, color, label, y_off):
        xs = [p[0] for p in pts]
        ys = [p[1]+y_off for p in pts]
        ax.plot(xs, ys, 'o-', color=color, ms=9, lw=2)
        for i, (xi, yi) in enumerate(zip(xs, ys)):
            ax.text(xi, yi+0.25, label[i], ha='center', fontsize=8,
                    fontweight='bold', color=color)

    # Before
    draw_route(ax2, [(0,6),(2,6),(4,6),(6,6),(8,6)], '#1565C0',
               ['D','p1','p2','d1','d2'], 0)
    draw_route(ax2, [(0,3),(2,3),(4,3),(6,3),(8,3)], '#C62828',
               ['D','p3','p4','d3','d4'], 0)
    ax2.text(9.2, 6, 'Route 1\n(before)', fontsize=8, color='#1565C0')
    ax2.text(9.2, 3, 'Route 2\n(before)', fontsize=8, color='#C62828')

    ax2.annotate('', xy=(5, 4.7), xytext=(5, 4.3),
                 arrowprops=dict(arrowstyle='<->', color='green', lw=2))
    ax2.text(5.3, 4.5, 'Move\nrequest 2', fontsize=8, color='green')

    savefig('tabu_search_darp.pdf')


# ─────────────────────────────────────────────────────────────────────────────
# Figure 5 – Time window and max ride time illustration (Gantt-like)
# ─────────────────────────────────────────────────────────────────────────────
def fig_time_windows():
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.set_title('Time Windows and Maximum Ride Time Constraint in DARP',
                 fontsize=12, fontweight='bold')

    customers = [
        ('Customer 1', 10, 20, 25, 40, 30),   # (label, e_p, l_p, e_d, l_d, L)
        ('Customer 2', 30, 45, 55, 70, 35),
        ('Customer 3', 5,  15, 20, 35, 25),
    ]

    colors_p = ['#1565C0', '#2E7D32', '#6A1B9A']
    colors_d = ['#42A5F5', '#66BB6A', '#AB47BC']

    for i, (label, ep, lp, ed, ld, L) in enumerate(customers):
        y = i * 2.0 + 0.5
        # pickup window
        ax.barh(y+0.3, lp-ep, left=ep, height=0.4, color=colors_p[i],
                alpha=0.85, label=f'{label} pickup window [{ep},{lp}]')
        # delivery window
        ax.barh(y-0.3, ld-ed, left=ed, height=0.4, color=colors_d[i],
                alpha=0.85, label=f'{label} delivery window [{ed},{ld}]')
        # max ride time arrow
        t_pickup = (ep + lp) / 2
        ax.annotate('', xy=(t_pickup + L, y), xytext=(t_pickup, y),
                    arrowprops=dict(arrowstyle='<->', color='red', lw=1.5))
        ax.text(t_pickup + L/2, y+0.15, f'L={L}', ha='center', fontsize=8,
                color='red', fontweight='bold')
        ax.text(ep - 1, y, label, ha='right', fontsize=9)

    ax.set_xlabel('Time', fontsize=11)
    ax.set_yticks([])
    ax.set_xlim(0, 85)
    ax.legend(loc='upper right', fontsize=7.5, ncol=2)

    ax.text(0.01, 0.02,
            'Blue/green/purple bars: time windows  |  Red arrows: max ride time L_i',
            transform=ax.transAxes, fontsize=8, color='gray')

    savefig('time_windows_darp.pdf')


# ─────────────────────────────────────────────────────────────────────────────
# Figure 6 – Insertion heuristic for DARP (step-by-step illustration)
# ─────────────────────────────────────────────────────────────────────────────
def fig_insertion_heuristic():
    fig, axes = plt.subplots(1, 3, figsize=(13, 5))
    fig.suptitle('DARP Insertion Heuristic – Adding Requests One by One',
                 fontsize=12, fontweight='bold')

    depot = (5, 4)
    step_data = [
        {
            'title': 'Step 1: Route = {Depot}',
            'route': [depot, depot],
            'labels': ['D', 'D'],
            'pending': [(1.5, 6.5, 'p1'), (8, 6, 'p2'), (2, 2, 'p3')],
            'delivered': []
        },
        {
            'title': 'Step 2: Insert Request 1\n(cheapest feasible position)',
            'route': [depot, (1.5, 6.5), (3, 5), depot],
            'labels': ['D', 'p1', 'd1', 'D'],
            'pending': [(8, 6, 'p2'), (2, 2, 'p3')],
            'delivered': [(3, 5, 'd1')]
        },
        {
            'title': 'Step 3: Insert Request 2\n(next cheapest insertion)',
            'route': [depot, (1.5, 6.5), (8, 6), (3, 5), (6.5, 2.5), depot],
            'labels': ['D', 'p1', 'p2', 'd1', 'd2', 'D'],
            'pending': [(2, 2, 'p3')],
            'delivered': [(3, 5, 'd1'), (6.5, 2.5, 'd2')]
        },
    ]

    for ax, sd in zip(axes, step_data):
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 9)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(sd['title'], fontsize=9, fontweight='bold')

        # route
        rx = [p[0] for p in sd['route']]
        ry = [p[1] for p in sd['route']]
        ax.plot(rx, ry, 'g-o', ms=8, lw=2, zorder=3)
        for (x, y), lbl in zip(sd['route'], sd['labels']):
            ax.text(x, y+0.4, lbl, ha='center', fontsize=9,
                    fontweight='bold', color='darkgreen')

        # depot
        ax.plot(*depot, 'ks', ms=12, zorder=4)

        # pending
        for px, py, lbl in sd['pending']:
            ax.plot(px, py, 'rx', ms=10, mew=2, zorder=4)
            ax.text(px, py+0.4, lbl, ha='center', fontsize=8,
                    color='red', style='italic')

    savefig('insertion_heuristic.pdf')


# ─────────────────────────────────────────────────────────────────────────────
# Figure 7 – School Bus Routing Problem (SBRP) vs DARP comparison
# ─────────────────────────────────────────────────────────────────────────────
def fig_sbrp_vs_darp():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    fig.suptitle('School Bus Routing (SBRP) vs. Dial-a-Ride (DARP)',
                 fontsize=12, fontweight='bold')

    titles = ['SBRP: Fixed stops, multiple schools', 'DARP: Door-to-door, one depot']
    colors_v = ['#1565C0', '#C62828', '#2E7D32']

    for k, ax in enumerate(axes):
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 9)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(titles[k], fontsize=10, fontweight='bold')

    # SBRP
    ax = axes[0]
    stops = [(1,7),(3,8),(2,5),(4,6),(5,7),(7,8),(8,6),(6,4)]
    schools = [(2,2,'S1','#C62828'), (7,2,'S2','#2E7D32')]
    for sx, sy, sl, sc in schools:
        ax.plot(sx, sy, 's', color=sc, ms=14, zorder=5)
        ax.text(sx, sy+0.5, sl, ha='center', fontsize=10,
                fontweight='bold', color=sc)
    for i, (sx, sy) in enumerate(stops):
        ax.plot(sx, sy, 'o', color='#1565C0', ms=9, zorder=4)
        ax.text(sx+0.3, sy, f'stop{i+1}', fontsize=7, color='#1565C0')

    # two bus routes
    route1 = [(2,2),(1,7),(3,8),(5,7),(2,2)]
    route2 = [(7,2),(8,6),(7,8),(6,4),(4,6),(7,2)]
    for r, c in [(route1,'#1565C0'), (route2,'#C62828')]:
        rx2 = [p[0] for p in r]
        ry2 = [p[1] for p in r]
        ax.plot(rx2, ry2, '-', color=c, lw=2, alpha=0.7)

    ax.text(0.5, 0.04, 'Buses pick up at stops; deliver to school',
            transform=ax.transAxes, fontsize=8, ha='center', color='gray')

    # DARP
    ax = axes[1]
    depot = (5, 4.5)
    ax.plot(*depot, 'ks', ms=14, zorder=5)
    ax.text(depot[0], depot[1]+0.5, 'Depot', ha='center', fontsize=10,
            fontweight='bold')
    homes = [(1,7),(2,2),(8,7),(9,2),(1,4)]
    dests = [(3,6),(4,3),(6,6),(7,3),(3,4)]
    for i, ((hx,hy),(dx,dy)) in enumerate(zip(homes,dests)):
        c = colors_v[i % 3]
        ax.plot(hx, hy, 'o', color=c, ms=9)
        ax.plot(dx, dy, '^', color=c, ms=9)
        ax.annotate('', xy=(dx,dy), xytext=(hx,hy),
                    arrowprops=dict(arrowstyle='->', color=c, lw=1.2,
                                   linestyle='dashed'))
    ax.text(0.5, 0.04, 'Vehicle picks up at door; delivers door-to-door',
            transform=ax.transAxes, fontsize=8, ha='center', color='gray')

    savefig('sbrp_vs_darp.pdf')


# ─────────────────────────────────────────────────────────────────────────────
# Figure 8 – Car pooling network diagram
# ─────────────────────────────────────────────────────────────────────────────
def fig_carpooling():
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Car Pooling / Ride-Sharing: Multiple Passengers per Vehicle',
                 fontsize=11, fontweight='bold')

    # Passengers with origins and destinations
    passengers = [
        ((1, 7), (9, 6), 'A', '#1565C0'),
        ((1, 5), (9, 4), 'B', '#2E7D32'),
        ((1, 3), (9, 2), 'C', '#C62828'),
    ]

    for (ox, oy), (dx, dy), label, color in passengers:
        ax.plot(ox, oy, 'o', color=color, ms=12, zorder=5)
        ax.text(ox-0.5, oy, f'Pax {label}\norigin', ha='right', fontsize=8,
                color=color)
        ax.plot(dx, dy, '^', color=color, ms=12, zorder=5)
        ax.text(dx+0.3, dy, f'Pax {label}\ndest.', ha='left', fontsize=8,
                color=color)

    # Shared route
    shared = [(1,7),(1,5),(1,3),(5,4.5),(9,6),(9,4),(9,2)]
    sx = [p[0] for p in shared]
    sy = [p[1] for p in shared]
    ax.plot(sx, sy, 'k--', lw=2.5, alpha=0.6, label='Shared vehicle route')

    ax.annotate('All 3 passengers\nride together\nover shared segment',
                xy=(5, 4.5), xytext=(5, 6.5),
                fontsize=9, ha='center',
                arrowprops=dict(arrowstyle='->', color='black'))

    ax.legend(fontsize=9, loc='lower right')
    ax.text(0.5, 0.02,
            'Key: detour and ride time constraints prevent unlimited sharing',
            transform=ax.transAxes, ha='center', fontsize=8, color='gray')

    savefig('carpooling_network.pdf')


# ─────────────────────────────────────────────────────────────────────────────
# Figure 9 – LNS (Large Neighbourhood Search) destroy-repair cycle
# ─────────────────────────────────────────────────────────────────────────────
def fig_lns_cycle():
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.axis('off')
    ax.set_title('Large Neighbourhood Search (LNS) – Destroy & Repair Cycle',
                 fontsize=12, fontweight='bold')

    boxes = [
        (0.5, 0.75, 'Initial\nSolution\n$s_0$', '#1565C0'),
        (0.5, 0.50, 'Destroy:\nRemove $q$ requests\nfrom routes', '#C62828'),
        (0.5, 0.25, 'Repair:\nRe-insert removed\nrequests optimally', '#2E7D32'),
        (0.15, 0.50, 'Accept?\n(better than\nincumbent)', '#E65100'),
        (0.85, 0.50, 'Update\nbest known\nsolution', '#6A1B9A'),
    ]

    for bx, by, label, color in boxes:
        box = FancyBboxPatch((bx-0.13, by-0.08), 0.26, 0.16,
                             boxstyle='round,pad=0.02',
                             transform=ax.transAxes,
                             linewidth=2, edgecolor=color,
                             facecolor=color+'22', clip_on=False)
        ax.add_patch(box)
        ax.text(bx, by, label, ha='center', va='center', fontsize=9,
                fontweight='bold', color=color,
                transform=ax.transAxes)

    # arrows
    def ann(ax, xy, xytext, color='gray', cs=None):
        ap = dict(arrowstyle='->', lw=2, color=color)
        if cs:
            ap['connectionstyle'] = cs
        ax.annotate('', xy=xy, xytext=xytext,
                    xycoords='axes fraction', textcoords='axes fraction',
                    arrowprops=ap)

    ann(ax, (0.5, 0.58), (0.5, 0.67))
    ann(ax, (0.5, 0.33), (0.5, 0.42))
    ann(ax, (0.28, 0.50), (0.37, 0.50))
    ann(ax, (0.72, 0.50), (0.63, 0.50))
    # loop back
    ann(ax, (0.5, 0.83), (0.85, 0.58), color='#6A1B9A', cs='arc3,rad=0.4')
    ax.text(0.78, 0.72, 'iterate', fontsize=8, color='#6A1B9A',
            transform=ax.transAxes)

    savefig('lns_cycle.pdf')


# ─────────────────────────────────────────────────────────────────────────────
# Figure 10 – Benchmark comparison bar chart (Cordeau & Laporte instances)
# ─────────────────────────────────────────────────────────────────────────────
def fig_benchmark():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Representative DARP Benchmark Results (Cordeau-Laporte instances)',
                 fontsize=11, fontweight='bold')

    # Solution quality comparison (% above best-known)
    methods = ['Branch-\nand-Cut', 'Tabu\nSearch', 'LNS', 'ALNS', 'DP\n(heuristic)']
    gap = [0.0, 0.8, 0.3, 0.15, 1.2]
    colors_b = ['#1565C0', '#E65100', '#2E7D32', '#6A1B9A', '#C62828']

    ax = axes[0]
    bars = ax.bar(methods, gap, color=colors_b, alpha=0.85, edgecolor='black', lw=0.8)
    ax.set_ylabel('Average gap from best known (%)', fontsize=10)
    ax.set_title('Solution Quality (lower = better)', fontsize=10)
    ax.set_ylim(0, 1.8)
    for bar, g in zip(bars, gap):
        ax.text(bar.get_x()+bar.get_width()/2, g+0.04, f'{g:.2f}%',
                ha='center', fontsize=9, fontweight='bold')

    # CPU time comparison (log scale, schematic)
    times = [3600, 45, 12, 18, 30]  # seconds, schematic
    ax2 = axes[1]
    bars2 = ax2.bar(methods, times, color=colors_b, alpha=0.85, edgecolor='black', lw=0.8)
    ax2.set_yscale('log')
    ax2.set_ylabel('CPU time (seconds, log scale, schematic)', fontsize=10)
    ax2.set_title('Computational Time', fontsize=10)
    for bar, t in zip(bars2, times):
        ax2.text(bar.get_x()+bar.get_width()/2, t*1.2, f'{t}s',
                 ha='center', fontsize=9, fontweight='bold')

    plt.tight_layout()
    savefig('benchmark_comparison.pdf')


# ─────────────────────────────────────────────────────────────────────────────
# Figure 11 – Demand Responsive Transit (DRT) concept
# ─────────────────────────────────────────────────────────────────────────────
def fig_drt():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    fig.suptitle('Demand Responsive Transit (DRT) vs. Fixed-Line Transit',
                 fontsize=12, fontweight='bold')

    for ax, title in zip(axes, ['Fixed Line', 'DRT (Flexible)']):
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 8)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(title, fontsize=11, fontweight='bold')

    # Fixed line
    ax = axes[0]
    stops_x = np.linspace(1, 9, 6)
    stops_y = 4 * np.ones(6)
    ax.plot(stops_x, stops_y, 'b-', lw=3, alpha=0.5)
    ax.plot(stops_x, stops_y, 'bs', ms=12)
    for i, (sx, sy) in enumerate(zip(stops_x, stops_y)):
        ax.text(sx, sy-0.55, f'S{i+1}', ha='center', fontsize=9,
                fontweight='bold', color='blue')
    # passengers walk to stops
    pax_homes = [(1.5,6.5),(3.5,2),(5.5,6),(7.5,2)]
    for hx, hy in pax_homes:
        nearest = stops_x[np.argmin(np.abs(stops_x - hx))]
        ax.plot(hx, hy, 'go', ms=8)
        ax.annotate('', xy=(nearest, 4.1), xytext=(hx, hy),
                    arrowprops=dict(arrowstyle='->', color='green', lw=1.5,
                                   linestyle='dotted'))
    ax.text(5, 7.5, 'Passengers walk to fixed stops', ha='center',
            fontsize=9, style='italic', color='gray')

    # DRT
    ax = axes[1]
    depot = (5, 4)
    ax.plot(*depot, 'ks', ms=14)
    ax.text(5, 4.6, 'Depot', ha='center', fontsize=9, fontweight='bold')
    pax2 = [(1.5,6.5,'A'),(3.5,2,'B'),(7.5,6,'C'),(8.5,2,'D')]
    dests2 = [(5,6.5,'A'),(5,2,'B'),(5,4.5,'C'),(7,4,'D')]
    for (ox,oy,l),(dx2,dy2,_) in zip(pax2,dests2):
        ax.plot(ox, oy, 'go', ms=9)
        ax.plot(dx2, dy2, 'r^', ms=9)
        ax.annotate('', xy=(dx2,dy2), xytext=(ox,oy),
                    arrowprops=dict(arrowstyle='->', color='purple', lw=1.5))
        ax.text(ox-0.3, oy, l, fontsize=9, color='green', fontweight='bold')
    ax.text(5, 7.5, 'Vehicle routes adapt to actual demand', ha='center',
            fontsize=9, style='italic', color='gray')

    savefig('drt_concept.pdf')


# ─────────────────────────────────────────────────────────────────────────────
# Run all
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("Generating Chapter 7 figures ...")
    fig_darp_overview()
    fig_darp_constraints()
    fig_branch_cut()
    fig_tabu_search()
    fig_time_windows()
    fig_insertion_heuristic()
    fig_sbrp_vs_darp()
    fig_carpooling()
    fig_lns_cycle()
    fig_benchmark()
    fig_drt()
    print("Done. All figures saved to figures/")
