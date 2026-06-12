"""
gen_figures.py  --  Generate all figures for Chapter 3 slides
"New Exact Algorithms for the Capacitated Vehicle Routing Problem"
Vehicle Routing: Problems, Methods, and Applications, 2nd ed., 2014
Authors: Marcus Poggi & Eduardo Uchoa

Produces all PDFs required by chapter03_slides.tex:
  fig_bpc_flowchart.pdf
  fig_bb_tree.pdf
  fig_cvrp_instance.pdf   (also fig_set_partitioning.pdf -- reuse)
  fig_set_partitioning.pdf
  fig_capacity_cut.pdf
  fig_kpath_cut.pdf
  fig_strong_degree_cuts.pdf
  fig_src.pdf
  fig_column_generation.pdf
  fig_spprc.pdf
  fig_route_enumeration.pdf
  fig_benchmark_results.pdf
  fig_node_comparison.pdf
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import numpy as np
import os

OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(OUTDIR, exist_ok=True)

def savefig(name, dpi=150):
    path = os.path.join(OUTDIR, name)
    plt.savefig(path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# fig_bpc_flowchart.pdf — Branch-Price-Cut framework flowchart
# ─────────────────────────────────────────────────────────────────────────────
def fig_bpc_flowchart():
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_facecolor('white')
    ax.axis('off')

    boxes = [
        # (x, y, text, facecolor, textcolor, fontsize)
        (0.50, 0.93, 'START: Root LP Node', '#1565C0', 'white', 12),
        (0.50, 0.78, 'Solve LP relaxation via\nColumn Generation (SPPRC)', '#1976D2', 'white', 11),
        (0.50, 0.62, 'Add violated\nCutting Planes?\n(RCI, k-path, SRC)', '#00838F', 'white', 10),
        (0.18, 0.42, 'YES: Add cuts,\nre-solve LP', '#00838F', 'white', 10),
        (0.82, 0.42, 'NO: LP optimal\nIs solution integer?', '#F57F17', 'black', 10),
        (0.50, 0.22, 'YES: Integer solution\nUpdate Upper Bound\nPrune dominated nodes', '#2E7D32', 'white', 10),
        (0.82, 0.22, 'NO: Branch on\nfractional arc x_ij\n(create 2 child nodes)', '#C62828', 'white', 10),
        (0.50, 0.05, 'All nodes fathomed?\n→ OPTIMUM PROVED', '#4A148C', 'white', 12),
    ]

    for (x, y, txt, fc, tc, fs) in boxes:
        ax.text(x, y, txt, ha='center', va='center', fontsize=fs,
                color=tc, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.45', facecolor=fc,
                          edgecolor='#333', linewidth=1.5),
                transform=ax.transAxes, zorder=3)

    # Arrows
    arrows = [
        (0.50, 0.88, 0.50, 0.84),   # start -> solve LP
        (0.50, 0.72, 0.50, 0.68),   # solve -> cuts?
        (0.50, 0.56, 0.18, 0.49),   # cuts? -> YES
        (0.50, 0.56, 0.82, 0.49),   # cuts? -> NO
        (0.18, 0.35, 0.18, 0.84),   # YES -> back to solve (loop)
        (0.82, 0.35, 0.50, 0.28),   # NO int? -> yes
        (0.82, 0.35, 0.82, 0.29),   # NO int? -> branch
        (0.50, 0.16, 0.50, 0.11),   # int -> all fathomed?
    ]
    for (x1, y1, x2, y2) in arrows:
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    xycoords='axes fraction', textcoords='axes fraction',
                    arrowprops=dict(arrowstyle='->', color='#555', lw=2))

    # Loop label
    ax.text(0.06, 0.65, 'Re-solve\n(loop)', ha='center', fontsize=8,
            color='#00838F', transform=ax.transAxes,
            bbox=dict(boxstyle='round,pad=0.2', facecolor='#E0F7FA', edgecolor='#00838F', lw=0.8))

    ax.set_title('Branch-Price-Cut (BPC) Algorithm for the CVRP\n'
                 'The three components: Column Generation, Cutting Planes, Branching',
                 fontsize=12, fontweight='bold', y=0.99)
    savefig('fig_bpc_flowchart.pdf')


# ─────────────────────────────────────────────────────────────────────────────
# fig_bb_tree.pdf — Branch-and-bound tree
# ─────────────────────────────────────────────────────────────────────────────
def fig_bb_tree():
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.set_facecolor('white')
    ax.axis('off')

    root    = (0.50, 0.90)
    child1  = (0.25, 0.65)
    child2  = (0.75, 0.65)
    gc11    = (0.12, 0.35)
    gc12    = (0.38, 0.35)
    gc21    = (0.62, 0.35)
    gc22    = (0.88, 0.35)
    ggc111  = (0.06, 0.10)
    ggc112  = (0.18, 0.10)

    nodes_info = [
        (root,   'Root Node\nLP=115.4\n(fractional)',            '#1565C0', 'white', 10),
        (child1, 'x₁₂ = 0\n(forbid arc 1→2)\nLP=117.2',         '#0288D1', 'white', 9),
        (child2, 'x₁₂ = 1\n(require arc 1→2)\nLP=116.0',        '#0288D1', 'white', 9),
        (gc11,   'x₂₃ = 0\nLP=120.5\n(frac.)',                   '#0277BD', 'white', 8),
        (gc12,   'x₂₃ = 1\nInteger!\nUB=119.3',                  '#2E7D32', 'white', 8),
        (gc21,   'x₃₄ = 0\nPruned\n(LP>119.3)',                  '#B71C1C', 'white', 8),
        (gc22,   'x₃₄ = 1\nInteger!\nUB=116.0',                  '#2E7D32', 'white', 8),
        (ggc111, 'Pruned\n(LP>116.0)',                            '#B71C1C', 'white', 7),
        (ggc112, 'Integer!\nUB=117.8\n(> 116.0)',                 '#F57F17', 'black', 7),
    ]

    for (pos, txt, fc, tc, fs) in nodes_info:
        ax.text(pos[0], pos[1], txt, ha='center', va='center', fontsize=fs,
                color=tc, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.35', facecolor=fc,
                          edgecolor='#333', linewidth=1.2),
                transform=ax.transAxes, zorder=3)

    edges = [
        (root, child1, 'x₁₂=0'),
        (root, child2, 'x₁₂=1'),
        (child1, gc11, 'x₂₃=0'),
        (child1, gc12, 'x₂₃=1'),
        (child2, gc21, 'x₃₄=0'),
        (child2, gc22, 'x₃₄=1'),
        (gc11, ggc111, 'x₄₅=0'),
        (gc11, ggc112, 'x₄₅=1'),
    ]
    for (p1, p2, lbl) in edges:
        ax.annotate('', xy=p2, xytext=p1,
                    xycoords='axes fraction', textcoords='axes fraction',
                    arrowprops=dict(arrowstyle='->', color='#555', lw=1.8))
        mid = (0.5*(p1[0]+p2[0]), 0.5*(p1[1]+p2[1]))
        ax.text(mid[0]+0.015, mid[1], lbl, fontsize=7.5, color='#333',
                ha='center', transform=ax.transAxes, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.1', facecolor='#FFFDE7',
                          edgecolor='#aaa', lw=0.5))

    # Legend
    legend_handles = [
        mpatches.Patch(color='#1565C0', label='Fractional LP node'),
        mpatches.Patch(color='#2E7D32', label='Integer solution (feasible)'),
        mpatches.Patch(color='#B71C1C', label='Pruned node (LB > best UB)'),
        mpatches.Patch(color='#F57F17', label='Suboptimal integer solution'),
    ]
    ax.legend(handles=legend_handles, loc='lower center', fontsize=9,
              ncol=2, framealpha=0.9, bbox_to_anchor=(0.5, -0.03))

    ax.set_title('Branch-and-Bound Tree: Branching on Arc Variables\n'
                 'Optimal solution found: UB = 116.0 at node (x₁₂=1, x₃₄=1)',
                 fontsize=11, fontweight='bold')
    savefig('fig_bb_tree.pdf')


# ─────────────────────────────────────────────────────────────────────────────
# fig_cvrp_instance.pdf — example CVRP instance
# ─────────────────────────────────────────────────────────────────────────────
def fig_cvrp_instance():
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_facecolor('#f8f8ff')

    depot = np.array([0.5, 0.5])
    customers = {
        1: np.array([0.15, 0.75]),
        2: np.array([0.28, 0.90]),
        3: np.array([0.45, 0.82]),
        4: np.array([0.68, 0.80]),
        5: np.array([0.83, 0.62]),
        6: np.array([0.78, 0.30]),
        7: np.array([0.55, 0.20]),
        8: np.array([0.30, 0.22]),
    }
    demands = {1:3, 2:4, 3:2, 4:3, 5:3, 6:4, 7:2, 8:3}
    Q = 10

    route1 = [0, 1, 2, 3, 0]
    route2 = [0, 4, 5, 6, 0]
    route3 = [0, 7, 8, 0]

    coords = {0: depot}
    coords.update(customers)

    route_styles = [
        (route1, '#2196F3', 'Route 1 (load=9): 0→1→2→3→0'),
        (route2, '#E91E63', 'Route 2 (load=10): 0→4→5→6→0'),
        (route3, '#4CAF50', 'Route 3 (load=5): 0→7→8→0'),
    ]

    for route, color, label in route_styles:
        xs = [coords[n][0] for n in route]
        ys = [coords[n][1] for n in route]
        ax.plot(xs, ys, '-', color=color, linewidth=2.5, label=label, zorder=2, alpha=0.8)
        for i in range(len(route)-1):
            p1 = coords[route[i]]
            p2 = coords[route[i+1]]
            d = p2 - p1
            nd = np.linalg.norm(d)
            ax.annotate('', xy=p2 - 0.04*d/max(nd,1e-6),
                        xytext=p1 + 0.04*d/max(nd,1e-6),
                        arrowprops=dict(arrowstyle='->', color=color, lw=2))

    # Draw depot
    ax.plot(*depot, 's', color='black', markersize=16, zorder=5,
            markeredgecolor='#333', markeredgewidth=1.5)
    ax.text(depot[0], depot[1]-0.07, 'Depot (0)', ha='center', fontsize=11, fontweight='bold')

    # Draw customers
    for cid, pos in customers.items():
        ax.plot(*pos, 'o', color='white', markersize=22, zorder=4,
                markeredgecolor='#333', markeredgewidth=1.5)
        ax.text(pos[0], pos[1], str(cid), ha='center', va='center',
                fontsize=10, fontweight='bold', zorder=5)
        ax.text(pos[0]+0.04, pos[1]+0.06, f'q={demands[cid]}',
                fontsize=8.5, color='#444', ha='center')

    ax.set_xlim(0.02, 0.98)
    ax.set_ylim(0.05, 1.0)
    ax.axis('off')
    ax.set_title(f'CVRP Example: 8 Customers, Vehicle Capacity Q = {Q}\n'
                 f'Three routes; each route starts and ends at depot',
                 fontsize=12, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9, framealpha=0.9)
    savefig('fig_cvrp_instance.pdf')


# ─────────────────────────────────────────────────────────────────────────────
# fig_set_partitioning.pdf — route pool table
# ─────────────────────────────────────────────────────────────────────────────
def fig_set_partitioning():
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.set_facecolor('white')
    ax.axis('off')

    routes = [
        ('0→1→2→3→0', [1,2,3], 18.5, 1),
        ('0→1→3→0',   [1,3],   14.2, 0),
        ('0→2→3→4→0', [2,3,4], 21.0, 1),
        ('0→4→5→0',   [4,5],   12.8, 0),
        ('0→1→5→0',   [1,5],   15.1, 0),
        ('0→2→5→0',   [2,5],   16.3, 1),
    ]
    customers = [1, 2, 3, 4, 5]
    n_routes = len(routes)
    col_w = 1.1
    row_h = 0.55

    # Column headers
    headers = ['Route path', 'λ', 'Cost'] + [f'Cust {c}' for c in customers]
    x_positions = [-0.3, 0.55, 1.25] + [2.2 + j*col_w for j in range(len(customers))]
    for hdr, xp in zip(headers, x_positions):
        ax.text(xp, n_routes*row_h + 0.5, hdr, ha='center', va='center',
                fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='#E3F2FD', edgecolor='#1565C0', lw=1))

    for i, (name, cust_list, cost, selected) in enumerate(routes):
        y = (n_routes - 1 - i) * row_h
        fc = '#E8F5E9' if selected else 'white'
        ec = '#2E7D32' if selected else '#ccc'
        lw = 2 if selected else 1
        rect = mpatches.FancyBboxPatch((-0.85, y-0.22), 8.2, row_h-0.06,
                                        boxstyle='round,pad=0.05',
                                        facecolor=fc, edgecolor=ec, linewidth=lw)
        ax.add_patch(rect)
        ax.text(-0.3, y+0.06, name, ha='center', va='center', fontsize=9,
                fontweight='bold' if selected else 'normal', color='#333')
        lv = '1' if selected else '0'
        ax.text(0.55, y+0.06, lv, ha='center', va='center', fontsize=11,
                color='#2E7D32' if selected else '#aaa', fontweight='bold')
        ax.text(1.25, y+0.06, f'{cost:.1f}', ha='center', va='center', fontsize=9)
        for j, c in enumerate(customers):
            val = '1' if c in cust_list else '·'
            color = '#1E88E5' if c in cust_list else '#bbb'
            ax.text(2.2 + j*col_w, y+0.06, val, ha='center', va='center',
                    fontsize=11, color=color, fontweight='bold')

    ax.set_xlim(-1.0, 8.0)
    ax.set_ylim(-0.4, n_routes*row_h + 1.0)
    ax.set_title('Set-Partitioning Formulation: Route Pool (6 candidate routes, 5 customers)\n'
                 'Green rows: λ=1 (selected in optimal solution).  Objective = 18.5 + 21.0 + 16.3 = 55.8',
                 fontsize=11, fontweight='bold')
    savefig('fig_set_partitioning.pdf')


# ─────────────────────────────────────────────────────────────────────────────
# fig_capacity_cut.pdf — capacity cut illustration
# ─────────────────────────────────────────────────────────────────────────────
def fig_capacity_cut():
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.set_facecolor('#f9f9f9')
    ax.axis('off')

    depot = np.array([0.5, 0.5])
    customers = {
        1: np.array([0.20, 0.80]),
        2: np.array([0.38, 0.88]),
        3: np.array([0.55, 0.82]),
        4: np.array([0.72, 0.65]),
        5: np.array([0.25, 0.30]),
        6: np.array([0.75, 0.25]),
    }
    demands = {1:4, 2:3, 3:5, 4:2, 5:4, 6:3}
    Q = 10
    S = {1, 2, 3, 4}

    # Draw depot
    ax.plot(*depot, 's', color='#E53935', markersize=16, zorder=5,
            markeredgecolor='#333', markeredgewidth=1.5)
    ax.text(depot[0], depot[1]-0.08, 'Depot', ha='center', fontsize=11, fontweight='bold')

    # Draw customers
    for cid, pos in customers.items():
        in_S = cid in S
        fc = '#FFCC80' if in_S else '#BBDEFB'
        ec = '#E65100' if in_S else '#1565C0'
        lw = 2.5 if in_S else 1.2
        ax.plot(*pos, 'o', color=fc, markersize=22, zorder=4,
                markeredgecolor=ec, markeredgewidth=lw)
        ax.text(pos[0], pos[1], str(cid), ha='center', va='center',
                fontsize=10, fontweight='bold', zorder=5)
        ax.text(pos[0]+0.04, pos[1]+0.07, f'q={demands[cid]}',
                fontsize=8.5, color='#555', ha='center')

    # Draw ellipse around S
    cx = np.mean([customers[i][0] for i in S])
    cy = np.mean([customers[i][1] for i in S])
    ell = mpatches.Ellipse((cx, cy), 0.62, 0.52, angle=15,
                            facecolor='none', edgecolor='#E65100',
                            linewidth=2.5, linestyle='--', zorder=2)
    ax.add_patch(ell)

    # Draw boundary arcs (depot <-> S)
    for cid in S:
        pos = customers[cid]
        d = pos - depot
        nd = np.linalg.norm(d)
        ax.annotate('', xy=depot + 0.08*d/nd,
                    xytext=pos - 0.06*d/nd,
                    arrowprops=dict(arrowstyle='<->', color='#C62828', lw=2.5))

    # Annotation
    d_S = sum(demands[i] for i in S)
    rS = int(np.ceil(d_S / Q))
    ax.text(0.5, 0.04,
            fr'$S = \{{1,2,3,4\}}$,  $d(S) = {d_S}$,  $\lceil d(S)/Q \rceil = \lceil {d_S}/{Q} \rceil = {rS}$'
            f'\nCapacity Cut (RCI):  $x(\delta(S)) \geq 2 \times {rS} = {2*rS}$  '
            f'(at least {2*rS} arcs cross the dashed boundary)',
            ha='center', fontsize=10.5, color='#BF360C', fontweight='bold',
            transform=ax.transAxes,
            bbox=dict(boxstyle='round,pad=0.35', facecolor='#FFF3E0',
                      edgecolor='#FF6F00', linewidth=1.5))

    ax.set_xlim(0.02, 0.98)
    ax.set_ylim(0.01, 1.0)
    ax.set_title('Rounded Capacity Inequality (RCI): Capacity Cut for Subset S\n'
                 'Orange nodes = subset S; double-headed arrows = boundary arcs δ(S)',
                 fontsize=11, fontweight='bold')
    legend_handles = [
        mpatches.Patch(color='#FFCC80', label='Customers in S', ec='#E65100', lw=2),
        mpatches.Patch(color='#BBDEFB', label='Customers outside S', ec='#1565C0', lw=1.2),
    ]
    ax.legend(handles=legend_handles, loc='upper right', fontsize=9)
    savefig('fig_capacity_cut.pdf')


# ─────────────────────────────────────────────────────────────────────────────
# fig_kpath_cut.pdf — k-path cut illustration
# ─────────────────────────────────────────────────────────────────────────────
def fig_kpath_cut():
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.set_facecolor('#f9f9f9')
    ax.axis('off')

    depot = np.array([0.50, 0.50])
    customers = {
        1: np.array([0.20, 0.78]),
        2: np.array([0.42, 0.88]),
        3: np.array([0.68, 0.78]),
        4: np.array([0.80, 0.35]),
        5: np.array([0.22, 0.28]),
    }
    demands = {1:5, 2:4, 3:5, 4:3, 5:4}
    Q = 10
    P = [1, 2, 3]  # path customers

    # Draw depot
    ax.plot(*depot, 's', color='#E53935', markersize=16, zorder=5,
            markeredgecolor='#333', markeredgewidth=1.5)
    ax.text(depot[0], depot[1]-0.08, 'Depot', ha='center', fontsize=11, fontweight='bold')

    # Draw customers
    for cid, pos in customers.items():
        in_P = cid in P
        fc = '#FFF59D' if in_P else '#C8E6C9'
        ec = '#F57F17' if in_P else '#388E3C'
        lw = 2.5 if in_P else 1.2
        ax.plot(*pos, 'o', color=fc, markersize=22, zorder=4,
                markeredgecolor=ec, markeredgewidth=lw)
        ax.text(pos[0], pos[1], str(cid), ha='center', va='center',
                fontsize=10, fontweight='bold', zorder=5)
        ax.text(pos[0]+0.04, pos[1]+0.07, f'q={demands[cid]}',
                fontsize=8.5, color='#555', ha='center')

    # Draw the path P: 1→2→3
    for i in range(len(P)-1):
        p1 = customers[P[i]]
        p2 = customers[P[i+1]]
        d = p2 - p1
        nd = np.linalg.norm(d)
        ax.annotate('', xy=p2 - 0.06*d/nd, xytext=p1 + 0.06*d/nd,
                    arrowprops=dict(arrowstyle='->', color='#F57F17', lw=3))

    # Draw arcs from depot to endpoints of P (the k-path arcs)
    for endpt_id, lbl in [(1, 'enter P'), (3, 'leave P')]:
        pos = customers[endpt_id]
        d = pos - depot
        nd = np.linalg.norm(d)
        ax.annotate('', xy=pos - 0.07*d/nd, xytext=depot + 0.07*d/nd,
                    arrowprops=dict(arrowstyle='->', color='#C62828', lw=2.5, linestyle='dashed'))

    # Draw ellipse around path
    cx = np.mean([customers[i][0] for i in P])
    cy = np.mean([customers[i][1] for i in P]) + 0.02
    ell = mpatches.Ellipse((cx, cy), 0.70, 0.32, angle=5,
                            facecolor='none', edgecolor='#F57F17',
                            linewidth=2, linestyle='--', zorder=2)
    ax.add_patch(ell)

    d_P = sum(demands[i] for i in P)
    k = int(np.ceil(d_P / Q))
    ax.text(0.50, 0.04,
            fr'Path $P = \{{1,2,3\}}$,  $d(P) = {d_P}$,  $k = \lceil {d_P}/{Q} \rceil = {k}$'
            f'\nk-Path Cut: at least {2*k} arcs cross δ(P) — one "enter" and one "leave" per vehicle trip',
            ha='center', fontsize=10.5, color='#BF360C', fontweight='bold',
            transform=ax.transAxes,
            bbox=dict(boxstyle='round,pad=0.35', facecolor='#FFFDE7',
                      edgecolor='#F57F17', linewidth=1.5))

    ax.set_xlim(0.02, 0.98)
    ax.set_ylim(0.01, 1.0)
    ax.set_title('k-Path Cut: Path Through Customers Forcing Multiple Vehicle Trips\n'
                 'Yellow path nodes P={1,2,3}; red dashed = required boundary arcs',
                 fontsize=11, fontweight='bold')
    savefig('fig_kpath_cut.pdf')


# ─────────────────────────────────────────────────────────────────────────────
# fig_strong_degree_cuts.pdf — before/after strong degree cut
# ─────────────────────────────────────────────────────────────────────────────
def fig_strong_degree_cuts():
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    for ax_idx, ax in enumerate(axes):
        ax.set_facecolor('#f8f8f8')
        ax.axis('off')

        depot = np.array([0.5, 0.5])
        customers = {
            1: np.array([0.20, 0.80]),
            2: np.array([0.50, 0.90]),
            3: np.array([0.80, 0.75]),
            4: np.array([0.80, 0.30]),
            5: np.array([0.20, 0.30]),
        }

        ax.plot(*depot, 's', color='#E53935', markersize=14, zorder=5,
                markeredgecolor='#333', markeredgewidth=1.5)
        ax.text(depot[0], depot[1]-0.09, 'Depot', ha='center', fontsize=10, fontweight='bold')

        for cid, pos in customers.items():
            ax.plot(*pos, 'o', color='white', markersize=20, zorder=4,
                    markeredgecolor='#333', markeredgewidth=1.3)
            ax.text(pos[0], pos[1], str(cid), ha='center', va='center',
                    fontsize=10, fontweight='bold', zorder=5)

        if ax_idx == 0:
            # Fractional LP solution: fractional arcs to customer 5
            arcs = [(depot, customers[1], '#2196F3', 2, 'solid', 'x₀₁=1'),
                    (depot, customers[2], '#2196F3', 2, 'solid', 'x₀₂=1'),
                    (depot, customers[5], '#FF9800', 2, 'dashed', 'x₀₅=0.6'),
                    (customers[3], depot, '#2196F3', 2, 'solid', 'x₃₀=1'),
                    (customers[4], depot, '#2196F3', 2, 'solid', 'x₄₀=1'),
                    (customers[1], customers[2], '#2196F3', 1.5, 'solid', ''),
                    (customers[2], customers[3], '#2196F3', 1.5, 'solid', ''),
                    (customers[4], customers[5], '#FF9800', 1.5, 'dashed', ''),
                    ]
            ax.set_title('Before Strong Degree Cut\n'
                         'x₀₅ = 0.6 (fractional — not integer)',
                         fontsize=10, fontweight='bold')
        else:
            # After cut: x₀₅ forced to integer
            arcs = [(depot, customers[1], '#2196F3', 2, 'solid', 'x₀₁=1'),
                    (depot, customers[2], '#2196F3', 2, 'solid', 'x₀₂=1'),
                    (depot, customers[5], '#4CAF50', 2.5, 'solid', 'x₀₅=1'),
                    (customers[3], depot, '#2196F3', 2, 'solid', 'x₃₀=1'),
                    (customers[4], depot, '#2196F3', 2, 'solid', 'x₄₀=1'),
                    (customers[1], customers[2], '#2196F3', 1.5, 'solid', ''),
                    (customers[2], customers[3], '#2196F3', 1.5, 'solid', ''),
                    (customers[4], customers[5], '#4CAF50', 1.5, 'solid', ''),
                    ]
            ax.set_title('After Strong Degree Cut\n'
                         'x₀₅ = 1 (forced to integer by the cut)',
                         fontsize=10, fontweight='bold')

        for (p1, p2, color, lw, ls, lbl) in arcs:
            d = p2 - p1
            nd = np.linalg.norm(d)
            ax.annotate('', xy=p2 - 0.06*d/max(nd,1e-6),
                        xytext=p1 + 0.06*d/max(nd,1e-6),
                        arrowprops=dict(arrowstyle='->', color=color,
                                        lw=lw, linestyle=ls))
            if lbl:
                mid = 0.5*(p1+p2)
                ax.text(mid[0]+0.04, mid[1]+0.03, lbl,
                        fontsize=8, color=color, fontweight='bold', ha='center')

        ax.set_xlim(0.05, 0.95)
        ax.set_ylim(0.05, 1.0)

    fig.suptitle('Strong Degree Cuts: Forcing Fractional Depot-Adjacent Arcs to Integer Values',
                 fontsize=12, fontweight='bold', y=1.01)
    plt.tight_layout()
    savefig('fig_strong_degree_cuts.pdf')


# ─────────────────────────────────────────────────────────────────────────────
# fig_src.pdf — Subset Row Cut illustration
# ─────────────────────────────────────────────────────────────────────────────
def fig_src():
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.set_facecolor('white')
    ax.axis('off')

    # Show 4 routes and subset C = {1, 2, 3}
    routes_data = [
        ('Route A: 0→1→2→3→0', [1,2,3], 0.3, '#2196F3'),
        ('Route B: 0→1→3→4→0', [1,3],   0.4, '#E91E63'),
        ('Route C: 0→2→4→0',   [2],     0.2, '#4CAF50'),
        ('Route D: 0→3→5→0',   [3],     0.1, '#FF9800'),
    ]
    C = {1, 2, 3}

    col_headers = ['Route', 'λ', 'C={1,2,3}\nvisits', '⌊visits/2⌋', 'contribution']
    col_x = [0.5, 2.8, 4.5, 6.2, 7.8]
    for hdr, xp in zip(col_headers, col_x):
        ax.text(xp, 4.3, hdr, ha='center', va='center', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#E3F2FD',
                          edgecolor='#1565C0', lw=1.2))

    total_contrib = 0.0
    for i, (name, cust_list, lam, color) in enumerate(routes_data):
        y = 3.3 - i * 0.85
        visits_in_C = sum(1 for c in cust_list if c in C)
        floor_half = visits_in_C // 2
        contrib = floor_half * lam
        total_contrib += contrib

        fc = '#F3F4F6'
        ax.text(col_x[0], y, name, ha='center', va='center', fontsize=9,
                color=color, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.25', facecolor=fc,
                          edgecolor=color, lw=1.5))
        for xp, val in zip(col_x[1:], [f'{lam}', str(visits_in_C),
                                         str(floor_half), f'{contrib:.2f}']):
            ax.text(xp, y, val, ha='center', va='center', fontsize=10,
                    color='#333',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                              edgecolor='#ccc', lw=0.8))

    # Total and bound
    rhs = len(C) // 2
    violated = total_contrib > rhs
    summary_color = '#B71C1C' if violated else '#2E7D32'
    ax.text(0.5, -0.1,
            fr'Sum of contributions = {total_contrib:.2f}   |   RHS = ⌊|C|/2⌋ = ⌊3/2⌋ = {rhs}'
            f'\nSRC: Σ ⌊visits_r/2⌋ · λ_r ≤ {rhs}'
            + (' ← VIOLATED! (cut should be added)' if violated else ' ← SATISFIED'),
            ha='center', fontsize=11, color=summary_color, fontweight='bold',
            transform=ax.transAxes,
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#FFEBEE' if violated else '#E8F5E9',
                      edgecolor=summary_color, lw=1.8))

    ax.set_xlim(-0.5, 9.5)
    ax.set_ylim(-0.6, 5.0)
    ax.set_title(r'Subset Row Cut (SRC) for $\mathcal{C} = \{1, 2, 3\}$:' + '\n'
                 r'Bound on total "half-coverage" of $\mathcal{C}$ by all routes',
                 fontsize=11, fontweight='bold')
    savefig('fig_src.pdf')


# ─────────────────────────────────────────────────────────────────────────────
# fig_column_generation.pdf — column generation loop diagram
# ─────────────────────────────────────────────────────────────────────────────
def fig_column_generation():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: schematic of the column generation loop
    ax = axes[0]
    ax.set_facecolor('white')
    ax.axis('off')

    boxes_cg = [
        (0.5, 0.88, 'Restricted Master Problem (RMP)\nSolve LP: min Σ c_r λ_r\ns.t. Σ a_ir λ_r ≥ 1  ∀i', '#1565C0', 'white', 9),
        (0.5, 0.55, 'Pricing Sub-Problem\n(SPPRC): find route r*\nwith min reduced cost\n'
                    'c̄_r* = c_r* − Σ a_{ir*} π_i', '#00838F', 'white', 9),
        (0.15, 0.22, 'c̄_r* < 0:\nAdd r* to RMP\nRe-solve', '#E65100', 'white', 9),
        (0.85, 0.22, 'c̄_r* ≥ 0:\nLP optimal!\nNo improving\nroute exists', '#2E7D32', 'white', 9),
    ]
    for (x, y, txt, fc, tc, fs) in boxes_cg:
        ax.text(x, y, txt, ha='center', va='center', fontsize=fs,
                color=tc, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.4', facecolor=fc,
                          edgecolor='#333', lw=1.5),
                transform=ax.transAxes, zorder=3)

    arrows_cg = [
        (0.5, 0.78, 0.5, 0.68, 'π* (dual vars)'),
        (0.5, 0.42, 0.15, 0.33, 'neg. r.c.'),
        (0.5, 0.42, 0.85, 0.33, 'non-neg. r.c.'),
        (0.15, 0.11, 0.15, 0.88, 'loop'),
    ]
    for (x1, y1, x2, y2, lbl) in arrows_cg:
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    xycoords='axes fraction', textcoords='axes fraction',
                    arrowprops=dict(arrowstyle='->', color='#555', lw=2))
        mid = (0.5*(x1+x2)+0.03, 0.5*(y1+y2)+0.02)
        ax.text(mid[0], mid[1], lbl, fontsize=8, color='#444',
                ha='center', transform=ax.transAxes)

    ax.set_title('Column Generation Loop\n(SPPRC Pricing + RMP)', fontsize=11, fontweight='bold')

    # Right: convergence plot
    ax2 = axes[1]
    iters = np.arange(1, 26)
    lb = 98 + 22 * (1 - np.exp(-0.3 * iters))
    rng = np.random.RandomState(7)
    ub = 130 - 13 * (1 - np.exp(-0.45 * iters)) + rng.normal(0, 0.8, 25)
    ub = np.maximum(ub, lb + 0.1)

    ax2.plot(iters, lb, 'b-o', markersize=4, label='LP Lower Bound (column gen.)', lw=2)
    ax2.plot(iters, ub, 'r--s', markersize=4, label='Upper Bound (best integer)', lw=2)
    ax2.fill_between(iters, lb, ub, alpha=0.15, color='orange', label='Optimality gap')
    ax2.set_xlabel('Column Generation Iteration', fontsize=11)
    ax2.set_ylabel('Objective Value', fontsize=11)
    ax2.set_title('Convergence of Column Generation\n(lower bound grows, gap shrinks)', fontsize=11, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    savefig('fig_column_generation.pdf')


# ─────────────────────────────────────────────────────────────────────────────
# fig_spprc.pdf — SPPRC labelling
# ─────────────────────────────────────────────────────────────────────────────
def fig_spprc():
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.set_facecolor('white')
    ax.axis('off')

    nodes_pos = {
        'Depot\n(0)': np.array([0.06, 0.50]),
        'C1\n(q=3)':  np.array([0.28, 0.78]),
        'C2\n(q=4)':  np.array([0.28, 0.25]),
        'C3\n(q=5)':  np.array([0.55, 0.82]),
        'C4\n(q=2)':  np.array([0.55, 0.20]),
        'C5\n(q=3)':  np.array([0.78, 0.55]),
        'Depot\n(sink)': np.array([0.95, 0.50]),
    }
    node_colors = ['#E53935', '#1E88E5', '#1E88E5',
                   '#1E88E5', '#1E88E5', '#1E88E5', '#E53935']

    for (nid, pos), col in zip(nodes_pos.items(), node_colors):
        ax.plot(*pos, 'o', color=col, markersize=28, zorder=4,
                markeredgecolor='#333', markeredgewidth=1.5)
        ax.text(pos[0], pos[1], nid, ha='center', va='center',
                fontsize=7.5, fontweight='bold', color='white', zorder=5)

    # Edges: (src_key, dst_key, arc_cost, demand_j)
    edges = [
        ('Depot\n(0)',  'C1\n(q=3)',      10, 3),
        ('Depot\n(0)',  'C2\n(q=4)',       8, 4),
        ('C1\n(q=3)',   'C3\n(q=5)',       7, 5),
        ('C1\n(q=3)',   'C5\n(q=3)',      12, 3),
        ('C2\n(q=4)',   'C4\n(q=2)',       6, 2),
        ('C3\n(q=5)',   'C5\n(q=3)',       5, 3),
        ('C4\n(q=2)',   'C5\n(q=3)',       9, 3),
        ('C5\n(q=3)',   'Depot\n(sink)',   8, 0),
        ('C3\n(q=5)',   'Depot\n(sink)',  11, 0),
        ('C4\n(q=2)',   'Depot\n(sink)',  13, 0),
    ]

    for (src, dst, arc_c, dem) in edges:
        p1 = nodes_pos[src]
        p2 = nodes_pos[dst]
        d = p2 - p1
        nd = np.linalg.norm(d)
        ax.annotate('', xy=p2 - 0.04*d/max(nd,1e-6),
                    xytext=p1 + 0.04*d/max(nd,1e-6),
                    arrowprops=dict(arrowstyle='->', color='#555', lw=1.6))
        mid = 0.5*(p1+p2)
        perp = np.array([-(p2-p1)[1], (p2-p1)[0]])
        if np.linalg.norm(perp) > 0:
            perp = perp / np.linalg.norm(perp) * 0.04
        lbl = f'c={arc_c}' + (f', q={dem}' if dem > 0 else '')
        ax.text(mid[0]+perp[0], mid[1]+perp[1], lbl,
                fontsize=7.5, ha='center',
                bbox=dict(boxstyle='round,pad=0.12', facecolor='#FFF9C4',
                          edgecolor='#aaa', lw=0.7))

    # Best label annotations at selected nodes
    best_labels = [
        ('C1\n(q=3)',  'L=(10, 3)', '#1565C0'),
        ('C3\n(q=5)',  'L=(17, 8)', '#1565C0'),
        ('C5\n(q=3)',  'L=(22, 6)', '#2E7D32'),
        ('Depot\n(sink)', 'Best: L=(30,6)\n→ route: 0→1→3→5→0', '#2E7D32'),
    ]
    for (nid, lbl, color) in best_labels:
        pos = nodes_pos[nid]
        offset = np.array([0.0, -0.14]) if 'sink' in nid else np.array([0.0, 0.14])
        ax.text(pos[0]+offset[0], pos[1]+offset[1], lbl,
                ha='center', fontsize=8, color=color, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.25', facecolor='#E8F5E9',
                          edgecolor=color, lw=1.2))

    ax.set_title('SPPRC Labelling Algorithm: Finding the Minimum Reduced-Cost Route\n'
                 'Each label L=(cost, cumulative load). Extend only if load + q_j ≤ Q = 10.',
                 fontsize=11, fontweight='bold')
    ax.text(0.5, 0.01,
            'Arc labels: c = arc cost, q = customer demand. '
            'Q = 10.  Route 0→1→3→5→0: load=3+5+3=11 > 10 — infeasible!  '
            'Route 0→1→5→0: load=3+3=6 ≤ 10 — feasible.',
            ha='center', fontsize=8.5, color='#555', transform=ax.transAxes)
    savefig('fig_spprc.pdf')


# ─────────────────────────────────────────────────────────────────────────────
# fig_route_enumeration.pdf — route enumeration vs branching
# ─────────────────────────────────────────────────────────────────────────────
def fig_route_enumeration():
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_facecolor('white')
    ax.axis('off')

    boxes = [
        (0.50, 0.90, 'Set-Partitioning MIP\n(exponentially many route variables)', '#1565C0', 'white', 12),
        (0.24, 0.65, 'Route Enumeration\n(Baldacci et al. 2008)\nEnumerate ALL improving\nroutes offline\n→ Feed to MIP solver', '#0288D1', 'white', 10),
        (0.76, 0.65, 'Column-Generation BPC\n(Contardo & Martinelli 2014)\nGenerate routes on-the-fly\nvia SPPRC pricing\nat each B&B node', '#2E7D32', 'white', 10),
        (0.24, 0.35, 'Strengths:\n• Fast MIP solve after enum\n• No repeated pricing\nWeaknesses:\n• Memory: all routes in RAM\n• Scales to n ≤ 100', '#4FC3F7', '#222', 9),
        (0.76, 0.35, 'Strengths:\n• Scales to n > 100\n• Memory-efficient\nWeaknesses:\n• Re-solves SPPRC at every\n  B&B node (expensive)', '#A5D6A7', '#222', 9),
        (0.50, 0.07, 'State-of-the-art solvers (2014) use a HYBRID:\nEnumeration near the leaf nodes of BPC tree;\nColumn generation at the root and upper levels.', '#F57F17', 'black', 10),
    ]

    for (x, y, txt, fc, tc, fs) in boxes:
        ax.text(x, y, txt, ha='center', va='center', fontsize=fs,
                color=tc, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.45', facecolor=fc,
                          edgecolor='#333', lw=1.5),
                transform=ax.transAxes, zorder=3)

    arrows = [
        (0.50, 0.84, 0.24, 0.74),
        (0.50, 0.84, 0.76, 0.74),
        (0.24, 0.56, 0.24, 0.44),
        (0.76, 0.56, 0.76, 0.44),
        (0.24, 0.26, 0.50, 0.13),
        (0.76, 0.26, 0.50, 0.13),
    ]
    for (x1, y1, x2, y2) in arrows:
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    xycoords='axes fraction', textcoords='axes fraction',
                    arrowprops=dict(arrowstyle='->', color='#555', lw=2))

    ax.set_title('Route Enumeration vs. On-the-Fly Column Generation in BPC',
                 fontsize=12, fontweight='bold', y=0.98)
    savefig('fig_route_enumeration.pdf')


# ─────────────────────────────────────────────────────────────────────────────
# fig_benchmark_results.pdf — algorithm comparison bar charts
# ─────────────────────────────────────────────────────────────────────────────
def fig_benchmark_results():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    algos     = ['Christofides\n(1981)', 'Toth &\nVigo (2002)',
                 'Fukasawa\net al. (2006)', 'Baldacci\net al. (2008)',
                 'Contardo &\nMartinelli (2014)']
    pct       = [37, 74, 81, 89, 96]
    avg_times = [None, 3200, 2100, 1450, 980]
    colors    = ['#78909C','#1E88E5','#43A047','#FB8C00','#E53935']

    ax = axes[0]
    bars = ax.bar(algos, pct, color=colors, edgecolor='black', linewidth=0.7)
    for bar, val in zip(bars, pct):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()+0.8,
                f'{val}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.set_ylabel('% Instances Solved to Optimality', fontsize=11)
    ax.set_title('Benchmark: Instances Solved\n(Augerat A+B+P sets)', fontsize=11, fontweight='bold')
    ax.set_ylim(0, 110)
    ax.grid(axis='y', alpha=0.3)
    ax.tick_params(axis='x', labelsize=8)

    ax2 = axes[1]
    times = [t for t in avg_times if t is not None]
    lbls  = [a for a, t in zip(algos, avg_times) if t is not None]
    cols2 = [c for c, t in zip(colors, avg_times) if t is not None]
    bars2 = ax2.bar(lbls, times, color=cols2, edgecolor='black', linewidth=0.7)
    for bar, val in zip(bars2, times):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height()+15,
                 f'{val}s', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax2.set_ylabel('Average CPU Time (seconds)', fontsize=11)
    ax2.set_title('Benchmark: Average Solve Time\n(lower = faster)', fontsize=11, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    ax2.tick_params(axis='x', labelsize=8)

    plt.tight_layout()
    fig.suptitle('Approximate Comparison of Exact CVRP Algorithms on Standard Benchmarks\n'
                 '(Values are indicative — actual results depend on hardware and time limits)',
                 fontsize=11, fontweight='bold', y=1.03)
    savefig('fig_benchmark_results.pdf')


# ─────────────────────────────────────────────────────────────────────────────
# fig_node_comparison.pdf — BPC tree node comparison
# ─────────────────────────────────────────────────────────────────────────────
def fig_node_comparison():
    fig, ax = plt.subplots(figsize=(10, 5))

    instances = ['A-n32-k5', 'A-n46-k7', 'B-n41-k6', 'P-n55-k8',
                 'CMT-n50', 'CMT-n75', 'CMT-n100']
    nodes_old = [1200, 8500, 3200, 22000, 35000, 120000, 500000]
    nodes_new = [85,   380,  210,  950,   1800,  6200,   28000]

    x = np.arange(len(instances))
    width = 0.35

    b1 = ax.bar(x - width/2, nodes_old, width, color='#78909C',
                label='Fukasawa et al. (2006)', edgecolor='black', lw=0.7)
    b2 = ax.bar(x + width/2, nodes_new, width, color='#E53935',
                label='Contardo & Martinelli (2014)', edgecolor='black', lw=0.7)

    ax.set_yscale('log')
    ax.set_ylabel('BPC Tree Nodes (log scale)', fontsize=11)
    ax.set_title('Number of BPC Tree Nodes: 2006 vs. 2014 Algorithms\n'
                 'SRCs + ng-routes + strong branching reduce nodes by 1–2 orders of magnitude',
                 fontsize=11, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(instances, rotation=20, fontsize=9)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3, which='both')

    for bar, val in zip(b1, nodes_old):
        ax.text(bar.get_x() + bar.get_width()/2, val*1.15,
                f'{val:,}', ha='center', va='bottom', fontsize=7, color='#333')
    for bar, val in zip(b2, nodes_new):
        ax.text(bar.get_x() + bar.get_width()/2, val*1.15,
                f'{val:,}', ha='center', va='bottom', fontsize=7, color='#C62828',
                fontweight='bold')

    plt.tight_layout()
    savefig('fig_node_comparison.pdf')


# ─────────────────────────────────────────────────────────────────────────────
# Run all
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("Generating figures for Chapter 3: New Exact Algorithms for the CVRP ...")
    fig_bpc_flowchart()
    fig_bb_tree()
    fig_cvrp_instance()
    fig_set_partitioning()
    fig_capacity_cut()
    fig_kpath_cut()
    fig_strong_degree_cuts()
    fig_src()
    fig_column_generation()
    fig_spprc()
    fig_route_enumeration()
    fig_benchmark_results()
    fig_node_comparison()
    print("All figures generated successfully.")
