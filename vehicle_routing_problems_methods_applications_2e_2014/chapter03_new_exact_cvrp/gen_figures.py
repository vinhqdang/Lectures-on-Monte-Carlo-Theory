"""
gen_figures.py  --  Generate all figures for Chapter 3 slides
"New Exact Algorithms for the Capacitated Vehicle Routing Problem"
Vehicle Routing: Problems, Methods, and Applications, 2nd ed., 2014
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.lines import Line2D
import numpy as np
import os
import sys

try:
    import fitz  # pymupdf
    HAVE_FITZ = True
except ImportError:
    HAVE_FITZ = False

OUTDIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(OUTDIR, exist_ok=True)
PDF_PATH = os.path.join(
    os.path.dirname(__file__), "..",
    "Vehicle Routing_ Problems, Methods, and Applications, Second Edition 2014.pdf"
)


# ─────────────────────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────────────────────
def save(fig, name):
    path = os.path.join(OUTDIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 — CVRP instance: depot + customers with demands
# ─────────────────────────────────────────────────────────────────────────────
def fig_cvrp_instance():
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.set_aspect('equal')

    depot = np.array([0, 0])
    customers = {
        1: (2, 3, 3),   # (x, y, demand)
        2: (4, 1, 2),
        3: (5, 4, 4),
        4: (1, -2, 3),
        5: (3, -3, 2),
        6: (-2, 2, 1),
        7: (-3, -1, 3),
    }

    routes = [
        ([1, 3, 6], 'tab:blue'),
        ([2, 5], 'tab:orange'),
        ([4, 7], 'tab:green'),
    ]

    for route_nodes, color in routes:
        path = [depot] + [customers[n][:2] for n in route_nodes] + [depot]
        xs = [p[0] for p in path]
        ys = [p[1] for p in path]
        ax.plot(xs, ys, '-o', color=color, linewidth=1.8, markersize=5,
                alpha=0.7, zorder=2)

    # Draw depot
    ax.plot(*depot, 's', color='black', markersize=14, zorder=5)
    ax.text(depot[0], depot[1] + 0.35, 'Depot', ha='center', fontsize=9,
            fontweight='bold', color='black')

    # Draw customers
    for idx, (x, y, d) in customers.items():
        ax.plot(x, y, 'o', color='steelblue', markersize=10, zorder=4)
        ax.text(x + 0.18, y + 0.28, f'$c_{idx}$\n(q={d})', fontsize=8,
                color='darkblue', ha='left')

    ax.set_xlim(-4.5, 7)
    ax.set_ylim(-4.5, 5.5)
    ax.set_title('CVRP Instance: Depot, Customers, and Three Routes (Q = 8)',
                 fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.axhline(0, color='gray', linewidth=0.5)
    ax.axvline(0, color='gray', linewidth=0.5)

    handles = [
        Line2D([0], [0], color='tab:blue',   lw=2, label='Route 1: 0→1→3→6→0 (load=8)'),
        Line2D([0], [0], color='tab:orange', lw=2, label='Route 2: 0→2→5→0  (load=4)'),
        Line2D([0], [0], color='tab:green',  lw=2, label='Route 3: 0→4→7→0  (load=6)'),
    ]
    ax.legend(handles=handles, fontsize=8, loc='lower right')
    save(fig, "fig_cvrp_instance.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 — Branch-and-Bound tree schematic
# ─────────────────────────────────────────────────────────────────────────────
def fig_bb_tree():
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis('off')
    ax.set_title('Branch-and-Bound / Branch-Price-Cut Tree (schematic)', fontsize=11)

    # Nodes: (x, y, label, color)
    nodes = [
        (5.0, 5.2, 'Root\nLB=840', 'gold'),
        (2.5, 3.8, 'Node 1\nLB=862', 'lightblue'),
        (7.5, 3.8, 'Node 2\nLB=851', 'lightblue'),
        (1.0, 2.2, 'Node 3\nLB=875', 'lightgreen'),
        (3.8, 2.2, 'Node 4\nLB=868', 'lightgreen'),
        (6.2, 2.2, 'Node 5\nLB=855', 'lightgreen'),
        (9.0, 2.2, 'Node 6\nLB=890\nPruned', 'salmon'),
        (2.8, 0.6, 'Node 7\nFeasible\nUB=868', 'limegreen'),
        (5.2, 0.6, 'Node 8\nInfeas.', 'lightcoral'),
    ]

    edges = [(0,1),(0,2),(1,3),(1,4),(2,5),(2,6),(4,7),(4,8)]

    for (u, v) in edges:
        x1,y1 = nodes[u][:2]
        x2,y2 = nodes[v][:2]
        ax.annotate('', xy=(x2,y2+0.38), xytext=(x1,y1-0.38),
                    arrowprops=dict(arrowstyle='->', color='gray', lw=1.2))

    for (x, y, label, color) in nodes:
        ax.add_patch(FancyBboxPatch((x-0.85, y-0.42), 1.7, 0.84,
                                    boxstyle="round,pad=0.05",
                                    facecolor=color, edgecolor='gray', lw=1))
        ax.text(x, y, label, ha='center', va='center', fontsize=7)

    # Branch label
    ax.text(3.55, 4.6, '$x_{ij}=0$', fontsize=8, color='navy')
    ax.text(6.3,  4.6, '$x_{ij}=1$', fontsize=8, color='navy')
    save(fig, "fig_bb_tree.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3 — Set Partitioning formulation overview
# ─────────────────────────────────────────────────────────────────────────────
def fig_set_partitioning():
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis('off')

    title = "Set Partitioning Formulation — Concept"
    ax.text(0.5, 0.95, title, transform=ax.transAxes, fontsize=12,
            fontweight='bold', ha='center', va='top')

    col_labels = ['Route', 'Customers', 'Cost', 'a1r', 'a2r', 'a3r', 'a4r', 'a5r']
    rows = [
        ['r1', '1,3,6', '120', '1', '0', '1', '0', '0'],
        ['r2', '2,5',   '85',  '0', '1', '0', '0', '1'],
        ['r3', '4,7',   '95',  '0', '0', '0', '1', '0'],
        ['r4', '1,2',   '100', '1', '1', '0', '0', '0'],
        ['r5', '3,4,5', '140', '0', '0', '1', '1', '1'],
    ]

    table = ax.table(
        cellText=rows,
        colLabels=col_labels,
        cellLoc='center',
        loc='center',
        bbox=[0.02, 0.05, 0.96, 0.80]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor('#4472C4')
            cell.set_text_props(color='white', fontweight='bold')
        elif row % 2 == 0:
            cell.set_facecolor('#D9E1F2')
        else:
            cell.set_facecolor('#FFFFFF')

    ax.text(0.5, 0.02,
            r'SP: $\min \sum_r c_r \lambda_r$ s.t. $\sum_r a_{ir}\lambda_r = 1\;\forall i$, $\lambda_r \in \{0,1\}$',
            transform=ax.transAxes, fontsize=10, ha='center')
    save(fig, "fig_set_partitioning.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4 — Capacity cut illustration
# ─────────────────────────────────────────────────────────────────────────────
def fig_capacity_cut():
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.set_aspect('equal')

    depot = np.array([0, 0])
    S_customers = {1: (2, 2), 2: (3, 1), 3: (2.5, 3), 4: (1.5, 0.5)}
    other = {5: (-2, 2), 6: (-1, -2), 7: (3, -2)}

    # Draw S boundary
    circle = plt.Circle((2.2, 1.8), 1.9, fill=False, edgecolor='red',
                         linewidth=2.5, linestyle='--', zorder=3)
    ax.add_patch(circle)
    ax.text(4.3, 3.6, r'$S$ (customer set)', color='red', fontsize=10)

    # Edges inside S
    for (a, b) in [(1,2),(2,3),(3,4)]:
        x1,y1 = S_customers[a]; x2,y2 = S_customers[b]
        ax.annotate('', xy=(x2,y2), xytext=(x1,y1),
                    arrowprops=dict(arrowstyle='->', color='blue', lw=1.5))

    # Edges from depot into S
    for cid, (x, y) in S_customers.items():
        ax.annotate('', xy=(x*0.25, y*0.25), xytext=(depot[0], depot[1]),
                    arrowprops=dict(arrowstyle='->', color='darkgreen',
                                   lw=1.2, linestyle='dashed'))

    # Draw depot and customers
    ax.plot(*depot, 's', color='black', markersize=14, zorder=5)
    ax.text(0.1, -0.35, 'Depot (0)', fontsize=9, ha='center')

    for idx, (x, y) in S_customers.items():
        ax.plot(x, y, 'o', color='blue', markersize=11, zorder=4)
        ax.text(x+0.15, y+0.2, f'$c_{idx}$', fontsize=9)

    for idx, (x, y) in other.items():
        ax.plot(x, y, 'o', color='gray', markersize=9, zorder=4)
        ax.text(x+0.15, y+0.2, f'$c_{idx}$', fontsize=9, color='gray')

    ax.set_xlim(-3.5, 5.5)
    ax.set_ylim(-3.5, 5.0)
    ax.set_title(r'Capacity Cut: $x(\delta(S)) \geq 2\lceil d(S)/Q \rceil$', fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--')

    ax.text(0.02, 0.04,
            r'$d(S)=q_1+q_2+q_3+q_4$,  $Q$=vehicle capacity,  $\delta(S)$=edges crossing $S$',
            transform=ax.transAxes, fontsize=9, color='darkred')
    save(fig, "fig_capacity_cut.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 5 — SPPRC label propagation (small DAG)
# ─────────────────────────────────────────────────────────────────────────────
def fig_spprc():
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.axis('off')
    ax.set_title('SPPRC Label Propagation along a Partial Route', fontsize=11)

    # Nodes: name, x, y
    nodes = [
        ('Depot\n(0)', 0.5, 2.5),
        ('$c_1$\nd=3',  2.5, 3.8),
        ('$c_2$\nd=2',  2.5, 1.2),
        ('$c_3$\nd=4',  4.5, 3.8),
        ('$c_4$\nd=2',  4.5, 1.2),
        ('Depot\n(0)',  6.5, 2.5),
    ]

    # Draw nodes
    for (name, x, y) in nodes:
        ax.add_patch(plt.Circle((x, y), 0.38, facecolor='lightsteelblue',
                                 edgecolor='navy', lw=1.5, zorder=3))
        ax.text(x, y, name, ha='center', va='center', fontsize=8, zorder=4)

    # Arcs with cost labels
    arcs = [
        (0, 1, '10'), (0, 2, '8'),
        (1, 3, '7'),  (1, 4, '12'),
        (2, 3, '11'), (2, 4, '6'),
        (3, 5, '9'),  (4, 5, '10'),
    ]
    for (u, v, cost) in arcs:
        x1,y1 = nodes[u][1], nodes[u][2]
        x2,y2 = nodes[v][1], nodes[v][2]
        ax.annotate('', xy=(x2,y2), xytext=(x1,y1),
                    arrowprops=dict(arrowstyle='->', color='darkblue',
                                   lw=1.3, connectionstyle='arc3,rad=0.1'))
        mx, my = (x1+x2)/2, (y1+y2)/2
        ax.text(mx+0.05, my+0.15, cost, fontsize=8, color='firebrick')

    # Label boxes
    labels_info = [
        (0.5, 2.0,  'Label@Depot:\ncost=0, load=0'),
        (2.5, 4.55, 'Label@$c_1$:\ncost=10, load=3'),
        (4.5, 4.55, 'Label@$c_3$:\ncost=17, load=7'),
    ]
    for (x, y, txt) in labels_info:
        ax.add_patch(FancyBboxPatch((x-0.82, y-0.28), 1.64, 0.56,
                                    boxstyle="round,pad=0.05",
                                    facecolor='lightyellow', edgecolor='orange', lw=1))
        ax.text(x, y, txt, ha='center', va='center', fontsize=7.5)

    ax.set_xlim(-0.2, 7.2)
    ax.set_ylim(0.3, 5.4)
    save(fig, "fig_spprc.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 6 — BPC algorithm flowchart
# ─────────────────────────────────────────────────────────────────────────────
def fig_bpc_flowchart():
    fig, ax = plt.subplots(figsize=(5.5, 8))
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('Branch-Price-Cut (BPC) Algorithm Flowchart', fontsize=11)

    boxes = [
        (3.0, 9.3, 'Start: Root node\n(LP relaxation)', 'gold'),
        (3.0, 7.8, 'Column generation:\nSolve pricing sub-problem\n(SPPRC)', 'lightblue'),
        (3.0, 6.3, 'LP bound improved?', 'lightyellow'),
        (3.0, 4.9, 'Separate valid cuts\n(Cap. cuts, k-path, SRC)', 'lightcyan'),
        (3.0, 3.5, 'LP solution\ninteger?', 'lightyellow'),
        (3.0, 2.1, 'Branch on fractional\nvariable / edge', 'lightgreen'),
        (3.0, 0.7, 'Update best UB;\nPrune dominated nodes', 'salmon'),
    ]

    for (x, y, text, color) in boxes:
        if '?' in text:
            diamond = plt.Polygon(
                [[x, y+0.42], [x+1.2, y], [x, y-0.42], [x-1.2, y]],
                closed=True, facecolor=color, edgecolor='gray', lw=1.2, zorder=3)
            ax.add_patch(diamond)
        else:
            ax.add_patch(FancyBboxPatch((x-1.5, y-0.38), 3.0, 0.76,
                                        boxstyle="round,pad=0.06",
                                        facecolor=color, edgecolor='gray', lw=1.2, zorder=3))
        ax.text(x, y, text, ha='center', va='center', fontsize=8.5, zorder=4)

    # Arrows between boxes
    for i in range(len(boxes)-1):
        x1, y1 = boxes[i][:2]
        x2, y2 = boxes[i+1][:2]
        ax.annotate('', xy=(x2, y2+0.42), xytext=(x1, y1-0.42),
                    arrowprops=dict(arrowstyle='->', color='black', lw=1.3))

    # Yes/No labels
    ax.text(3.25, 7.05, 'Yes', fontsize=8, color='green')
    ax.text(4.4,  6.3,  'No→ stop branch', fontsize=7.5, color='red')
    ax.text(3.25, 5.55, 'Yes → add cuts', fontsize=8, color='green')
    ax.text(3.25, 4.15, 'Yes → feasible sol', fontsize=8, color='green')
    save(fig, "fig_bpc_flowchart.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 7 — Benchmark results bar chart (Table 3.3 data approximation)
# ─────────────────────────────────────────────────────────────────────────────
def fig_benchmark_results():
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    # --- Left: Solved instances by algorithm ---
    algorithms = ['Fukasawa\net al.[19]', 'Baldacci\net al.[7]', 'Contardo\n&Martinelli[14]',
                  'Poggi &\nUchoa[33]', 'Ropke[39]']
    solved_100 = [20, 22, 24, 24, 26]  # out of 27 instances n<=100
    solved_200 = [8,  12, 15, 16, 18]  # out of ~20 larger instances

    x = np.arange(len(algorithms))
    w = 0.35
    ax = axes[0]
    b1 = ax.bar(x - w/2, solved_100, w, label=r'$n \leq 100$', color='steelblue', alpha=0.85)
    b2 = ax.bar(x + w/2, solved_200, w, label=r'$100 < n \leq 200$', color='tomato', alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(algorithms, fontsize=8)
    ax.set_ylabel('Instances solved to optimality')
    ax.set_title('Instances Solved by BPC Algorithm', fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.4, linestyle='--')
    ax.bar_label(b1, padding=2, fontsize=8)
    ax.bar_label(b2, padding=2, fontsize=8)

    # --- Right: Average CPU time (log scale) ---
    ax2 = axes[1]
    cpu_times = [3200, 2100, 1450, 1320, 980]  # seconds approx
    colors_bar = plt.cm.viridis(np.linspace(0.2, 0.8, len(algorithms)))
    bars = ax2.bar(algorithms, cpu_times, color=colors_bar, alpha=0.85)
    ax2.set_ylabel('Avg. CPU time (seconds)')
    ax2.set_title('Average Solving Time (Approximate)', fontsize=10)
    ax2.set_yscale('log')
    ax2.grid(axis='y', alpha=0.4, linestyle='--')
    for bar, val in zip(bars, cpu_times):
        ax2.text(bar.get_x() + bar.get_width()/2, val*1.05, str(val),
                 ha='center', va='bottom', fontsize=8)
    ax2.tick_params(axis='x', labelsize=8)

    fig.suptitle('Computational Results on Standard CVRP Benchmarks\n'
                 '(Augerat A, B, P sets and Golden instances)', fontsize=11)
    fig.tight_layout()
    save(fig, "fig_benchmark_results.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 8 — Strong Degree Cuts illustration
# ─────────────────────────────────────────────────────────────────────────────
def fig_strong_degree_cuts():
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for ax, title, show_cut in zip(axes,
                                   ['Before cut (fractional solution)',
                                    'After adding Degree Cut'],
                                   [False, True]):
        ax.set_xlim(-1, 6)
        ax.set_ylim(-1, 5)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(title, fontsize=10)

        depot = (0, 2)
        customers = [(2, 4), (4, 4), (2, 1), (4, 1), (3, 2.5)]

        ax.plot(*depot, 's', color='black', ms=12, zorder=5)
        ax.text(depot[0]-0.05, depot[1]-0.4, '0', ha='center', fontsize=9)

        for idx, (x, y) in enumerate(customers, 1):
            ax.plot(x, y, 'o', color='steelblue', ms=10, zorder=4)
            ax.text(x+0.1, y+0.2, str(idx), fontsize=9)

        # Edges with fractional values
        edges = [
            (depot, customers[0], 0.5),
            (depot, customers[2], 0.5),
            (customers[0], customers[1], 1.0),
            (customers[1], customers[4], 0.5),
            (customers[2], customers[3], 1.0),
            (customers[3], customers[4], 0.5),
            (customers[4], depot, 1.0),
        ]

        for (p1, p2, val) in edges:
            lw = 1 + 2 * val
            color = 'blue' if val == 1.0 else 'orange'
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], '-', color=color,
                    lw=lw, alpha=0.7, zorder=2)
            mx = (p1[0]+p2[0])/2
            my = (p1[1]+p2[1])/2
            ax.text(mx, my, f'{val:.1f}', fontsize=7.5, color='darkred')

        if show_cut:
            ax.add_patch(plt.Circle((3, 2.5), 0.6, fill=False,
                                     edgecolor='red', lw=2.5, linestyle='--', zorder=6))
            ax.text(3, 3.25, 'degree\ncut here', ha='center', fontsize=8.5, color='red')

    fig.suptitle('Effect of Degree Cuts on Fractional LP Solution', fontsize=11)
    fig.tight_layout()
    save(fig, "fig_strong_degree_cuts.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 9 — Route enumeration vs branching decision
# ─────────────────────────────────────────────────────────────────────────────
def fig_route_enumeration():
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis('off')
    ax.set_title('Route Enumeration vs. Branching: Trade-off', fontsize=11)

    # Two-column comparison table
    rows = [
        ['Aspect', 'Route Enumeration', 'Traditional Branching'],
        ['Column pool', 'Enumerate all routes\n(exponential, but bounded)', 'Generate on-the-fly\n(column generation)'],
        ['Pricing sub-problem', 'Not needed after enum.', 'SPPRC at every node'],
        ['Node work', 'Solve LP / IP directly', 'LP + pricing loop'],
        ['Suitable for', 'Small/medium n\n(exact routes known)', 'Large n\n(column gen. scales)'],
        ['Memory', 'High (store all routes)', 'Low (generate as needed)'],
        ['Key algorithm', 'Baldacci, Mingozzi,\nMartello (2008)', 'Fukasawa, Poggi,\nUchoa (2006)'],
    ]

    col_widths = [0.2, 0.38, 0.38]
    x_starts = [0.01, 0.22, 0.61]

    for row_i, row in enumerate(rows):
        for col_i, (cell, xw, xs) in enumerate(zip(row, col_widths, x_starts)):
            bg = '#4472C4' if row_i == 0 else ('#EEF2FF' if row_i % 2 == 0 else 'white')
            fc = 'white' if row_i == 0 else 'black'
            fw = 'bold' if row_i == 0 else 'normal'
            rect = patches.FancyBboxPatch((xs, 0.88 - row_i*0.14), xw - 0.01, 0.13,
                                          boxstyle="round,pad=0.005",
                                          facecolor=bg, edgecolor='gray', lw=0.5,
                                          transform=ax.transAxes, clip_on=False)
            ax.add_patch(rect)
            ax.text(xs + xw/2, 0.88 - row_i*0.14 + 0.065, cell,
                    transform=ax.transAxes, ha='center', va='center',
                    fontsize=8, color=fc, fontweight=fw, multialignment='center')

    save(fig, "fig_route_enumeration.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 10 — k-path cut illustration
# ─────────────────────────────────────────────────────────────────────────────
def fig_kpath_cut():
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.set_aspect('equal')
    ax.set_xlim(-1.5, 7.5)
    ax.set_ylim(-1, 5)
    ax.axis('off')
    ax.set_title(r'$k$-Path Cut: Framing subset $S$ with $k$ required vehicle entries', fontsize=10)

    depot = (0, 2)
    S = [(2, 4), (3.5, 4.5), (5, 4), (4, 2.5), (2.5, 1.5)]
    out = [(0.5, 0), (6, 1.5), (6.5, 3.5)]

    # S boundary
    ellipse = patches.Ellipse((3.5, 3), 4.2, 3.5, fill=False,
                               edgecolor='purple', lw=2, linestyle='--', zorder=3)
    ax.add_patch(ellipse)
    ax.text(5.8, 4.5, r'$S$', color='purple', fontsize=12, fontweight='bold')

    ax.plot(*depot, 's', color='black', ms=14, zorder=5)
    ax.text(depot[0]-0.05, depot[1]-0.5, 'Depot', ha='center', fontsize=9)

    for idx, (x, y) in enumerate(S, 1):
        ax.plot(x, y, 'o', color='royalblue', ms=11, zorder=4)
        ax.text(x+0.1, y+0.25, f'$v_{idx}$', fontsize=9)

    for (x, y) in out:
        ax.plot(x, y, 'D', color='gray', ms=8, zorder=4)

    # k crossing edges
    crossings = [
        (depot, S[0]),
        (depot, S[4]),
        (out[0], S[4]),
    ]
    for (p1, p2) in crossings:
        ax.annotate('', xy=p2, xytext=p1,
                    arrowprops=dict(arrowstyle='->', color='firebrick', lw=2.2))

    ax.text(0.03, 0.06,
            r'$k$-path cut: $x(\delta(S)) \geq 2\lceil k_S \rceil$ where $k_S$ depends on demands and capacity',
            transform=ax.transAxes, fontsize=9.5, color='purple')
    save(fig, "fig_kpath_cut.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 11 — Subset Row Cut (SRC) concept
# ─────────────────────────────────────────────────────────────────────────────
def fig_src():
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.axis('off')
    ax.set_title('Subset Row Cut (SRC) — Strengthening the LP Relaxation', fontsize=11)

    text = (
        r"An SRC is associated with a subset $C \subseteq V$ of customers." "\n\n"
        r"For a set $C$ with $|C| \geq 3$ customers, the SRC reads:" "\n\n"
        r"$\sum_{r \in \Omega} \lfloor \frac{1}{2} \sum_{i \in C} a_{ir} \rfloor \lambda_r \leq \lfloor \frac{|C|}{2} \rfloor$" "\n\n"
        r"where $a_{ir}=1$ if route $r$ serves customer $i$, and $\lambda_r$ is the route variable." "\n\n"
        r"The cut exploits the fact that a vehicle visiting an odd subset $C$ more times" "\n"
        r"than $\lfloor|C|/2\rfloor$ would need to re-enter $C$, which is suboptimal for the LP."
    )
    ax.text(0.05, 0.95, text, transform=ax.transAxes,
            fontsize=10.5, va='top', ha='left',
            linespacing=1.6,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#FFF8E7', edgecolor='orange', lw=1.5))
    save(fig, "fig_src.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 12 — Column generation: reduced cost illustration
# ─────────────────────────────────────────────────────────────────────────────
def fig_column_generation():
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.axis('off')
    ax.set_title('Column Generation in the Set Partitioning LP', fontsize=11)

    # Three boxes: RMP → Pricing → New column → RMP
    boxes = [
        (1.5, 2.5, 'Restricted\nMaster Problem\n(RMP)\nSolve LP relaxation'),
        (4.5, 2.5, 'Pricing\nSub-problem\n(SPPRC)\nFind min reduced-cost route'),
        (7.5, 2.5, 'New column\n$r^*$ with\n$\\bar{c}_{r^*} < 0$?\nAdd to RMP'),
    ]

    for (x, y, text) in boxes:
        ax.add_patch(FancyBboxPatch((x-1.25, y-0.9), 2.5, 1.8,
                                    boxstyle="round,pad=0.1",
                                    facecolor='lightsteelblue', edgecolor='navy', lw=1.5))
        ax.text(x, y, text, ha='center', va='center', fontsize=9)

    # Arrows
    ax.annotate('', xy=(3.22, 2.5), xytext=(2.75, 2.5),
                arrowprops=dict(arrowstyle='->', color='darkblue', lw=2))
    ax.text(3.0, 2.75, r'dual $\pi^*$', fontsize=9, ha='center', color='darkblue')

    ax.annotate('', xy=(6.22, 2.5), xytext=(5.75, 2.5),
                arrowprops=dict(arrowstyle='->', color='darkblue', lw=2))

    ax.annotate('', xy=(1.5, 1.4), xytext=(7.5, 1.4),
                arrowprops=dict(arrowstyle='->', color='green', lw=2,
                                connectionstyle='arc3,rad=-0.25'))
    ax.text(4.5, 0.7, 'Loop until no improving column\n'
            r'($\bar{c}_r = c_r - \sum_i a_{ir}\pi_i \geq 0$ for all routes $r$)',
            ha='center', fontsize=9.5, color='darkgreen')

    ax.annotate('', xy=(7.5, 3.6), xytext=(4.5, 3.6),
                arrowprops=dict(arrowstyle='<-', color='gray', lw=1.5))
    ax.text(6.0, 3.75, 'Add route if $\\bar{c} < 0$', fontsize=9, ha='center', color='gray')

    ax.set_xlim(0, 9)
    ax.set_ylim(0.2, 4.3)
    save(fig, "fig_column_generation.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 13 — Comparison chart: number of nodes in BPC tree
# ─────────────────────────────────────────────────────────────────────────────
def fig_node_comparison():
    fig, ax = plt.subplots(figsize=(8, 5))

    # Approximate data from tables in the chapter
    instance_names = ['A-n32-k5', 'A-n55-k9', 'B-n45-k5', 'B-n68-k9',
                      'P-n55-k15', 'P-n101-k4', 'Golden 1', 'Golden 5']
    nodes_old = [250, 4800, 380, 6200, 1500, 12000, 35000, 95000]
    nodes_new = [18,  320,   22,  480,   95,   870,  2200,   8500]

    x = np.arange(len(instance_names))
    w = 0.35
    b1 = ax.bar(x - w/2, nodes_old, w, label='Fukasawa et al. (2006)', color='steelblue', alpha=0.8)
    b2 = ax.bar(x + w/2, nodes_new, w, label='Contardo & Martinelli (2014)', color='tomato', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(instance_names, rotation=30, ha='right', fontsize=9)
    ax.set_yscale('log')
    ax.set_ylabel('Number of BPC Tree Nodes (log scale)')
    ax.set_title('Branch-Price-Cut Tree Nodes: Old vs. New BPC Algorithms\n'
                 '(approximate values from chapter tables)', fontsize=10)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.4, linestyle='--')
    fig.tight_layout()
    save(fig, "fig_node_comparison.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("Generating figures for Chapter 3 ...")
    fig_cvrp_instance()
    fig_bb_tree()
    fig_set_partitioning()
    fig_capacity_cut()
    fig_spprc()
    fig_bpc_flowchart()
    fig_benchmark_results()
    fig_strong_degree_cuts()
    fig_route_enumeration()
    fig_kpath_cut()
    fig_src()
    fig_column_generation()
    fig_node_comparison()
    print("All figures generated successfully.")
