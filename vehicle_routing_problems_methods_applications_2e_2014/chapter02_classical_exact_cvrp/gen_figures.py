"""
gen_figures.py
Generate all figures for Chapter 2: Classical Exact CVRP Algorithms slides.
Run with: conda run -n py313 python3 gen_figures.py
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.patches import FancyArrowPatch
import numpy as np
import os
import sys

OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(OUTDIR, exist_ok=True)

def savefig(name, dpi=150, tight=True):
    path = os.path.join(OUTDIR, name)
    if tight:
        plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────────────────────────
# Figure 1: CVRP example — depot + 6 customers with two routes
# ─────────────────────────────────────────────────────────────────
def fig_cvrp_example():
    fig, ax = plt.subplots(figsize=(7, 5))

    depot = np.array([3.0, 3.0])
    customers = {
        1: np.array([1.0, 5.0]),
        2: np.array([2.0, 6.5]),
        3: np.array([4.5, 6.5]),
        4: np.array([5.5, 4.5]),
        5: np.array([5.0, 1.5]),
        6: np.array([1.5, 1.5]),
    }
    demands = {1: 3, 2: 4, 3: 3, 4: 5, 5: 4, 6: 3}

    # Route 1: depot -> 1 -> 2 -> 3 -> depot  (total demand=10)
    route1 = [depot, customers[1], customers[2], customers[3], depot]
    # Route 2: depot -> 4 -> 5 -> 6 -> depot  (total demand=12)
    route2 = [depot, customers[4], customers[5], customers[6], depot]

    colors = ['#2196F3', '#FF5722']
    for route, col, lbl in zip([route1, route2], colors, ['Route 1 (Q=10)', 'Route 2 (Q=12)']):
        xs = [p[0] for p in route]
        ys = [p[1] for p in route]
        ax.plot(xs, ys, '-o', color=col, linewidth=2, markersize=6, label=lbl, zorder=2)
        for i in range(len(route) - 1):
            dx = route[i+1][0] - route[i][0]
            dy = route[i+1][1] - route[i][1]
            ax.annotate('', xy=(route[i][0] + 0.65*dx, route[i][1] + 0.65*dy),
                        xytext=(route[i][0] + 0.35*dx, route[i][1] + 0.35*dy),
                        arrowprops=dict(arrowstyle='->', color=col, lw=1.5))

    # Depot
    ax.plot(*depot, 's', color='black', markersize=14, zorder=5)
    ax.text(depot[0], depot[1] - 0.35, 'Depot (0)', ha='center', fontsize=9, fontweight='bold')

    # Customers
    for cid, pos in customers.items():
        ax.plot(*pos, 'o', color='gray', markersize=10, zorder=4)
        ax.text(pos[0], pos[1] + 0.3, f'$c_{cid}$\n(d={demands[cid]})',
                ha='center', fontsize=8.5)

    ax.set_xlim(0, 7.5)
    ax.set_ylim(0, 8)
    ax.set_title('CVRP Example: Depot + 6 Customers, Vehicle Capacity Q = 12', fontsize=11)
    ax.legend(loc='lower right', fontsize=9)
    ax.set_xlabel('x-coordinate')
    ax.set_ylabel('y-coordinate')
    ax.grid(True, alpha=0.3)
    savefig('cvrp_example.pdf')


# ─────────────────────────────────────────────────────────────────
# Figure 2: Branch-and-Bound tree schematic
# ─────────────────────────────────────────────────────────────────
def fig_bnb_tree():
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis('off')

    node_style = dict(boxstyle='round,pad=0.4', facecolor='#E3F2FD', edgecolor='#1565C0', linewidth=1.5)
    pruned_style = dict(boxstyle='round,pad=0.4', facecolor='#FFEBEE', edgecolor='#B71C1C', linewidth=1.5)
    optimal_style = dict(boxstyle='round,pad=0.4', facecolor='#E8F5E9', edgecolor='#1B5E20', linewidth=2)

    nodes = {
        'root': (5, 5.3, 'Root\nLB=100', node_style),
        'L1':   (2.5, 3.8, 'Node A\nLB=115', node_style),
        'R1':   (7.5, 3.8, 'Node B\nLB=108', node_style),
        'LL2':  (1.2, 2.2, 'Node C\nLB=130\n[Pruned: LB>UB]', pruned_style),
        'LR2':  (3.8, 2.2, 'Node D\nLB=118', node_style),
        'RL2':  (6.2, 2.2, 'Node E\nLB=112', node_style),
        'RR2':  (8.8, 2.2, 'Node F\nLB=122\n[Pruned]', pruned_style),
        'LRR3': (3.0, 0.6, 'Leaf\nLB=UB=120', node_style),
        'RLL3': (5.3, 0.6, 'Leaf\nLB=UB=118\n[OPTIMAL]', optimal_style),
        'RLR3': (7.2, 0.6, 'Leaf\nLB=125\n[Pruned]', pruned_style),
    }

    edges = [
        ('root','L1'), ('root','R1'),
        ('L1','LL2'), ('L1','LR2'),
        ('R1','RL2'), ('R1','RR2'),
        ('LR2','LRR3'),
        ('RL2','RLL3'), ('RL2','RLR3'),
    ]

    for (src, dst) in edges:
        x1, y1 = nodes[src][0], nodes[src][1]
        x2, y2 = nodes[dst][0], nodes[dst][1]
        ax.annotate('', xy=(x2, y2+0.18), xytext=(x1, y1-0.18),
                    arrowprops=dict(arrowstyle='->', color='#555', lw=1.2))

    for key, (x, y, label, style) in nodes.items():
        ax.text(x, y, label, ha='center', va='center', fontsize=7.5,
                bbox=style, zorder=3)

    # Edge labels
    ax.text(3.5, 4.65, 'x_ij=0', fontsize=7.5, color='#444', ha='center')
    ax.text(6.5, 4.65, 'x_ij=1', fontsize=7.5, color='#444', ha='center')

    ax.set_title('Branch-and-Bound Search Tree for CVRP\n'
                 '(Blue = active, Red = pruned, Green = optimal leaf)', fontsize=10)

    legend_elements = [
        mpatches.Patch(facecolor='#E3F2FD', edgecolor='#1565C0', label='Active node'),
        mpatches.Patch(facecolor='#FFEBEE', edgecolor='#B71C1C', label='Pruned (LB ≥ UB)'),
        mpatches.Patch(facecolor='#E8F5E9', edgecolor='#1B5E20', label='Optimal solution'),
    ]
    ax.legend(handles=legend_elements, loc='lower left', fontsize=8)
    savefig('bnb_tree.pdf')


# ─────────────────────────────────────────────────────────────────
# Figure 3: Assignment Problem relaxation illustration
# ─────────────────────────────────────────────────────────────────
def fig_ap_relaxation():
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    # Left: fractional AP solution (sub-tours)
    ax = axes[0]
    ax.set_title('AP Relaxation\n(may contain sub-tours)', fontsize=10)
    ax.set_xlim(-0.5, 4.5)
    ax.set_ylim(-0.5, 4.5)
    ax.axis('off')

    positions = {0: (2,4), 1: (0,2.5), 2: (1,0.5), 3: (3,0.5), 4: (4,2.5)}
    labels = {0: '0\n(depot)', 1: '1', 2: '2', 3: '3', 4: '4'}

    subtour_edges = [(1,2),(2,1),(3,4),(4,3)]  # two sub-tours
    depot_edges   = [(0,1),(3,0),(0,4),(2,0)]

    for (i,j) in subtour_edges:
        xi,yi = positions[i]; xj,yj = positions[j]
        ax.annotate('', xy=(xj,yj), xytext=(xi,yi),
                    arrowprops=dict(arrowstyle='->', color='red', lw=1.8,
                                   connectionstyle='arc3,rad=0.15'))
    for (i,j) in depot_edges:
        xi,yi = positions[i]; xj,yj = positions[j]
        ax.annotate('', xy=(xj,yj), xytext=(xi,yi),
                    arrowprops=dict(arrowstyle='->', color='#555', lw=1.2, linestyle='dashed'))

    for nid,(x,y) in positions.items():
        color = '#1565C0' if nid==0 else '#90CAF9'
        ax.plot(x,y,'o',markersize=22, color=color, zorder=4)
        ax.text(x,y, labels[nid], ha='center', va='center', fontsize=8, fontweight='bold')

    ax.text(2, -0.3, 'Red arrows: illegal sub-tours', ha='center', fontsize=8, color='red')

    # Right: Valid CVRP solution
    ax2 = axes[1]
    ax2.set_title('Valid CVRP Solution\n(all routes start/end at depot)', fontsize=10)
    ax2.set_xlim(-0.5, 4.5)
    ax2.set_ylim(-0.5, 4.5)
    ax2.axis('off')

    valid_edges = [(0,1),(1,2),(2,0),(0,3),(3,4),(4,0)]
    for (i,j) in valid_edges:
        xi,yi = positions[i]; xj,yj = positions[j]
        color = '#2196F3' if i==0 or j==0 else '#FF5722'
        ax2.annotate('', xy=(xj,yj), xytext=(xi,yi),
                     arrowprops=dict(arrowstyle='->', color=color, lw=1.8))

    for nid,(x,y) in positions.items():
        color = '#1565C0' if nid==0 else '#90CAF9'
        ax2.plot(x,y,'o',markersize=22, color=color, zorder=4)
        ax2.text(x,y, labels[nid], ha='center', va='center', fontsize=8, fontweight='bold')

    ax2.text(2, -0.3, 'All routes originate from depot 0', ha='center', fontsize=8, color='#2196F3')

    savefig('ap_relaxation.pdf')


# ─────────────────────────────────────────────────────────────────
# Figure 4: Set Partitioning formulation diagram
# ─────────────────────────────────────────────────────────────────
def fig_set_partitioning():
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.axis('off')

    # Show columns (routes) covering customers
    customers = ['c1','c2','c3','c4','c5']
    routes = {
        'r1': (['c1','c2'], 18.0),
        'r2': (['c3','c4'], 22.5),
        'r3': (['c5'],      12.0),
        'r4': (['c1','c3'], 20.0),
        'r5': (['c2','c5'], 19.5),
        'r6': (['c4'],      15.0),
    }

    col_w = 1.1
    row_h = 0.5
    n_r = len(routes)
    n_c = len(customers)

    # Headers
    ax.text(-0.3, n_c*row_h + 0.3, 'Customer', fontsize=9, fontweight='bold', ha='center')
    for j, (rname, (covered, cost)) in enumerate(routes.items()):
        ax.text(0.5 + j*col_w, n_c*row_h + 0.55, rname, fontsize=9, fontweight='bold', ha='center')
        ax.text(0.5 + j*col_w, n_c*row_h + 0.15, f'c={cost}', fontsize=7.5, ha='center', color='#555')

    # Grid
    for i, cust in enumerate(customers):
        y = (n_c - i - 1)*row_h
        ax.text(-0.3, y + 0.15, cust, fontsize=9, ha='center')
        for j, (rname, (covered, cost)) in enumerate(routes.items()):
            x = 0.5 + j*col_w - 0.35
            val = '1' if cust in covered else '0'
            color = '#C8E6C9' if val=='1' else '#FAFAFA'
            rect = mpatches.FancyBboxPatch((x, y+0.02), 0.7, row_h-0.06,
                                           boxstyle='round,pad=0.03',
                                           facecolor=color, edgecolor='#aaa')
            ax.add_patch(rect)
            ax.text(x+0.35, y+0.15, val, ha='center', fontsize=9,
                    color='#1B5E20' if val=='1' else '#aaa')

    ax.text(n_r*col_w/2 + 0.2, -0.4,
            'Green=1 means the route covers that customer.\n'
            'SP selects a minimum-cost subset of routes that covers each customer exactly once.',
            ha='center', fontsize=8.5, color='#333',
            bbox=dict(facecolor='#FFF9C4', edgecolor='#F9A825', boxstyle='round,pad=0.3'))

    ax.set_xlim(-0.8, n_r*col_w + 0.4)
    ax.set_ylim(-0.7, n_c*row_h + 1.0)
    ax.set_title('Set Partitioning Matrix: Routes vs. Customers\n'
                 '(Each customer must be covered by exactly one route)', fontsize=10)
    savefig('set_partitioning.pdf')


# ─────────────────────────────────────────────────────────────────
# Figure 5: Column Generation / LP Relaxation cartoon
# ─────────────────────────────────────────────────────────────────
def fig_column_generation():
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Left: column generation loop
    ax = axes[0]
    ax.axis('off')
    ax.set_title('Column Generation Loop', fontsize=10, fontweight='bold')

    boxes = [
        (0.5, 0.85, 'Restricted Master Problem (RMP)\nSolve LP with current route set', '#E3F2FD', '#1565C0'),
        (0.5, 0.55, 'Extract dual prices $\\pi_i$\nfrom LP optimal solution', '#FFF9C4', '#F57F17'),
        (0.5, 0.27, 'Pricing subproblem (shortest path)\nFind route with negative reduced cost\n'
                    '$\\bar{c}_r = c_r - \\sum_i \\pi_i a_{ir}$', '#E8F5E9', '#1B5E20'),
    ]
    for (x, y, txt, fc, ec) in boxes:
        ax.text(x, y, txt, ha='center', va='center', fontsize=8,
                bbox=dict(facecolor=fc, edgecolor=ec, boxstyle='round,pad=0.4'), transform=ax.transAxes)

    ax.annotate('', xy=(0.5, 0.67), xytext=(0.5, 0.78),
                arrowprops=dict(arrowstyle='->', lw=1.5, color='#333'),
                xycoords='axes fraction', textcoords='axes fraction')
    ax.annotate('', xy=(0.5, 0.39), xytext=(0.5, 0.50),
                arrowprops=dict(arrowstyle='->', lw=1.5, color='#333'),
                xycoords='axes fraction', textcoords='axes fraction')
    ax.annotate('', xy=(0.82, 0.55), xytext=(0.82, 0.27),
                arrowprops=dict(arrowstyle='->', lw=1.5, color='#2196F3', connectionstyle='arc3,rad=-0.3'),
                xycoords='axes fraction', textcoords='axes fraction')
    ax.text(0.91, 0.43, 'Add\nnew col', ha='center', fontsize=7.5, color='#2196F3', transform=ax.transAxes)
    ax.text(0.5, 0.1, 'Stop when no route with $\\bar{c}_r < 0$ exists\n→ LP optimal', ha='center',
            fontsize=8.5, transform=ax.transAxes,
            bbox=dict(facecolor='#FCE4EC', edgecolor='#C62828', boxstyle='round,pad=0.3'))

    # Right: LP bound convergence
    ax2 = axes[1]
    iters = np.arange(1, 12)
    lb = 100 + 20*(1 - np.exp(-0.5*(iters-1)))
    ub = np.full_like(iters, 125.0, dtype=float)
    ax2.plot(iters, lb, 'b-o', label='LP lower bound', linewidth=2)
    ax2.axhline(y=125, color='r', linestyle='--', linewidth=1.8, label='Best integer UB')
    ax2.fill_between(iters, lb, 125, alpha=0.15, color='orange', label='Optimality gap')
    ax2.set_xlabel('Column generation iteration')
    ax2.set_ylabel('Objective value')
    ax2.set_title('LP Bound Convergence\nin Column Generation', fontsize=10)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(95, 130)

    savefig('column_generation.pdf')


# ─────────────────────────────────────────────────────────────────
# Figure 6: Valid inequalities illustration (capacity cuts)
# ─────────────────────────────────────────────────────────────────
def fig_valid_inequalities():
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    # Left: a customer subset S requiring multiple vehicles
    ax = axes[0]
    ax.set_title('Capacity (Rounded) Cut\nfor customer set $S$', fontsize=10)
    ax.set_xlim(-0.5, 5.5)
    ax.set_ylim(-0.5, 5)
    ax.axis('off')

    depot = (2.5, 4.5)
    subset_s = {1:(0.5,2.5), 2:(1.5,1.0), 3:(2.5,0.5), 4:(3.5,1.0), 5:(4.5,2.5)}
    outside  = {6:(0.5,3.8), 7:(4.5,3.8)}
    demands = {1:4, 2:5, 3:3, 4:4, 5:5, 6:2, 7:2}

    # Draw subset S boundary
    ellipse = mpatches.Ellipse((2.5, 1.7), 5.2, 3.0, angle=0,
                                facecolor='#FFF9C4', edgecolor='#F57F17',
                                linewidth=2, linestyle='--', zorder=0)
    ax.add_patch(ellipse)
    ax.text(2.5, -0.3, 'Subset $S$: total demand = 21, $Q$=10\n'
            '$\\Rightarrow$ at least $\\lceil 21/10 \\rceil = 3$ vehicles',
            ha='center', fontsize=8, color='#E65100')

    ax.plot(*depot, 's', color='black', markersize=14, zorder=5)
    ax.text(depot[0], depot[1]-0.35, 'Depot', ha='center', fontsize=8, fontweight='bold')

    for cid,(x,y) in subset_s.items():
        ax.plot(x,y,'o', color='#1565C0', markersize=13, zorder=4)
        ax.text(x, y, f'{cid}\n(d={demands[cid]})', ha='center', va='center', fontsize=7, color='white', fontweight='bold')
    for cid,(x,y) in outside.items():
        ax.plot(x,y,'o', color='#78909C', markersize=13, zorder=4)
        ax.text(x, y, f'{cid}', ha='center', va='center', fontsize=7.5, color='white', fontweight='bold')

    # arrows from depot
    for (x,y) in [subset_s[1], subset_s[5], outside[6], outside[7]]:
        ax.annotate('', xy=(x,y), xytext=depot,
                    arrowprops=dict(arrowstyle='->', color='#555', lw=1.2))

    ax.text(5.2, 4.5, '$S$', fontsize=13, color='#F57F17', fontweight='bold')

    # Right: cut tightening LP feasible region
    ax2 = axes[1]
    ax2.set_title('Cutting Plane Tightens LP\nFeasible Region', fontsize=10)
    ax2.set_xlim(-0.1, 1.1)
    ax2.set_ylim(-0.1, 1.1)

    # Original LP polytope
    lp_x = np.array([0.0, 1.0, 0.8, 0.2, 0.0])
    lp_y = np.array([0.0, 0.0, 0.9, 1.0, 0.0])
    ax2.fill(lp_x, lp_y, alpha=0.2, color='blue', label='LP polytope')
    ax2.plot(lp_x, lp_y, 'b-', linewidth=1.5)

    # Integer hull
    ih_x = np.array([0.0, 1.0, 0.7, 0.2, 0.0])
    ih_y = np.array([0.0, 0.0, 0.7, 0.8, 0.0])
    ax2.fill(ih_x, ih_y, alpha=0.3, color='green', label='Integer hull')
    ax2.plot(ih_x, ih_y, 'g--', linewidth=1.5)

    # LP optimal (fractional)
    ax2.plot(0.85, 0.82, 'r*', markersize=14, label='LP opt (fractional)', zorder=5)
    # Integer optimal
    ax2.plot(0.7, 0.7, 'gD', markersize=10, label='Integer optimum', zorder=5)

    # Cut line
    cut_x = np.array([0.15, 1.05])
    cut_y = 0.82 - 0.6*(cut_x - 0.15)
    ax2.plot(cut_x, cut_y, 'r-', linewidth=2, label='Valid inequality (cut)')

    ax2.legend(fontsize=7.5, loc='lower left')
    ax2.set_xlabel('$x_1$'); ax2.set_ylabel('$x_2$')
    ax2.grid(True, alpha=0.2)

    savefig('valid_inequalities.pdf')


# ─────────────────────────────────────────────────────────────────
# Figure 7: Branch-and-Cut algorithm flowchart
# ─────────────────────────────────────────────────────────────────
def fig_branch_cut_flow():
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.axis('off')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.set_title('Branch-and-Cut Algorithm Flowchart', fontsize=11, fontweight='bold', y=0.97)

    steps = [
        (5, 11.2, 'Initialise: UB = heuristic solution', '#E3F2FD', '#1565C0', 'rect'),
        (5, 9.5,  'Select open node from B&B tree', '#E8EAF6', '#283593', 'rect'),
        (5, 7.8,  'Solve LP relaxation\n(column generation if needed)', '#FFF9C4', '#F57F17', 'rect'),
        (5, 6.1,  'LP infeasible or LB ≥ UB?\n→ Prune node', '#FFEBEE', '#B71C1C', 'diamond'),
        (5, 4.5,  'Separate violated valid inequalities\n(capacity cuts, comb inequalities …)\nAdd cuts → re-solve LP', '#E8F5E9', '#1B5E20', 'rect'),
        (5, 2.8,  'LP solution integer?\n→ Update UB if better', '#FFF9C4', '#F57F17', 'diamond'),
        (5, 1.2,  'Branch: create two child nodes\nby fixing a fractional variable', '#E3F2FD', '#1565C0', 'rect'),
    ]

    for i,(x, y, txt, fc, ec, shape) in enumerate(steps):
        if shape == 'diamond':
            dx, dy = 3.0, 0.55
            pts = np.array([[x, y+dy],[x+dx, y],[x, y-dy],[x-dx, y]])
            patch = mpatches.Polygon(pts, closed=True, facecolor=fc, edgecolor=ec, linewidth=1.5, zorder=2)
            ax.add_patch(patch)
            ax.text(x, y, txt, ha='center', va='center', fontsize=7.8, zorder=3, multialignment='center')
        else:
            rect = mpatches.FancyBboxPatch((x-3, y-0.45), 6, 0.9,
                                           boxstyle='round,pad=0.12',
                                           facecolor=fc, edgecolor=ec, linewidth=1.5, zorder=2)
            ax.add_patch(rect)
            ax.text(x, y, txt, ha='center', va='center', fontsize=7.8, zorder=3, multialignment='center')

        if i < len(steps)-1:
            y_next = steps[i+1][1]
            ax.annotate('', xy=(x, y_next+0.5), xytext=(x, y-0.5),
                        arrowprops=dict(arrowstyle='->', color='#333', lw=1.4), zorder=1)

    # Loop back arrow (from branch back to select node)
    ax.annotate('', xy=(8.5, 9.5), xytext=(8.5, 1.2),
                arrowprops=dict(arrowstyle='->', color='#2196F3', lw=1.5,
                                connectionstyle='arc3,rad=0.0'),
                zorder=1)
    ax.plot([8.5,8.5],[1.2,9.5], color='#2196F3', lw=1.5, zorder=0, linestyle='--')
    ax.text(9.3, 5.5, 'Add\nchildren\nto tree', ha='center', fontsize=7.5, color='#2196F3')

    savefig('branch_cut_flow.pdf')


# ─────────────────────────────────────────────────────────────────
# Figure 8: Comb inequality illustration
# ─────────────────────────────────────────────────────────────────
def fig_comb_inequality():
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.set_title('Comb Inequality: Handle H and Teeth $T_1, T_2, T_3$', fontsize=10)
    ax.set_xlim(-0.5, 8.5)
    ax.set_ylim(-0.5, 6)
    ax.axis('off')

    # Handle H (large ellipse)
    handle = mpatches.Ellipse((4, 3), 8.0, 4.5, angle=0,
                               facecolor='#E3F2FD', edgecolor='#1565C0',
                               linewidth=2.5, zorder=0, alpha=0.5)
    ax.add_patch(handle)
    ax.text(0.4, 5.3, '$H$ (handle)', fontsize=9, color='#1565C0', fontweight='bold')

    # Teeth (3 smaller ellipses partially inside H)
    teeth_params = [(1.5, 2.0, 1.2, 2.0, '#FF5722'), (4.0, 2.0, 1.2, 2.0, '#FF5722'), (6.5, 2.0, 1.2, 2.0, '#FF5722')]
    for i,(tx,ty,w,h,col) in enumerate(teeth_params):
        tooth = mpatches.Ellipse((tx,ty), w, h, facecolor=col, edgecolor=col,
                                  linewidth=2, zorder=1, alpha=0.3)
        ax.add_patch(tooth)
        ax.text(tx, ty-1.3, f'$T_{i+1}$', ha='center', fontsize=9, color=col, fontweight='bold')

    # Customers (nodes)
    h_nodes = [(2.5,3.8),(3.5,4.2),(4.5,3.8),(5.5,4.0),(6.0,3.2),(5.0,2.5)]
    t1_nodes = [(1.0,2.5),(1.5,1.2),(2.0,2.0)]
    t2_nodes = [(3.5,2.0),(4.0,1.0),(4.5,2.2)]
    t3_nodes = [(6.0,2.0),(6.5,1.0),(7.0,2.2)]

    for pos in h_nodes:
        ax.plot(*pos, 'o', color='#1565C0', markersize=9, zorder=3)
    for pos in t1_nodes:
        ax.plot(*pos, 'o', color='#FF5722', markersize=9, zorder=3)
    for pos in t2_nodes:
        ax.plot(*pos, 'o', color='#FF5722', markersize=9, zorder=3)
    for pos in t3_nodes:
        ax.plot(*pos, 'o', color='#FF5722', markersize=9, zorder=3)

    ax.text(4, -0.3,
            r'Comb inequality: $x(\delta(H)) + \sum_{t=1}^{|T|} x(\delta(T_t)) \geq 3|T|+1$',
            ha='center', fontsize=9,
            bbox=dict(facecolor='#FFF9C4', edgecolor='#F57F17', boxstyle='round,pad=0.3'))

    savefig('comb_inequality.pdf')


# ─────────────────────────────────────────────────────────────────
# Figure 9: Benchmark comparison chart (E-instances)
# ─────────────────────────────────────────────────────────────────
def fig_benchmark_comparison():
    fig, ax = plt.subplots(figsize=(9, 4.5))

    algorithms = ['Christofides &\nEilon [27]\nB&B', 'Fisher &\nJaikumar [33]\nSP', 'Laporte\net al. [41]\nB&C', 'Augerat\net al. [5]\nB&C', 'Toth &\nVigo [52]\nB&B', 'Lysgaard\net al. [44]\nB&C']
    max_n      = [50,      100,    100,    135,    100,    135]
    year       = [1969,    1981,   1986,   1995,   2002,   2004]

    colors = ['#78909C','#78909C','#4CAF50','#2196F3','#FF9800','#9C27B0']
    x = np.arange(len(algorithms))
    bars = ax.bar(x, max_n, color=colors, edgecolor='white', linewidth=1.2, width=0.6)

    for bar, yr, n in zip(bars, year, max_n):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1.5,
                f'{yr}\nn≤{n}', ha='center', va='bottom', fontsize=7.5, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(algorithms, fontsize=8.5)
    ax.set_ylabel('Max problem size solved (customers)', fontsize=9)
    ax.set_title('Progress of Exact CVRP Algorithms:\nMaximum Instance Size Solved to Optimality', fontsize=10)
    ax.set_ylim(0, 160)
    ax.grid(True, alpha=0.3, axis='y')

    legend_patches = [
        mpatches.Patch(color='#78909C', label='Branch-and-Bound'),
        mpatches.Patch(color='#4CAF50', label='Early B&C / SP'),
        mpatches.Patch(color='#2196F3', label='Branch-and-Cut (modern)'),
        mpatches.Patch(color='#FF9800', label='Improved B&B'),
        mpatches.Patch(color='#9C27B0', label='State-of-the-art B&C'),
    ]
    ax.legend(handles=legend_patches, fontsize=7.5, loc='upper left')
    savefig('benchmark_comparison.pdf')


# ─────────────────────────────────────────────────────────────────
# Figure 10: Rounded capacity inequality illustration
# ─────────────────────────────────────────────────────────────────
def fig_rounded_capacity():
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Left: subset S with its cut edges (delta(S))
    ax = axes[0]
    ax.set_title('Cut edges $\\delta(S)$ for subset $S$', fontsize=10)
    ax.set_xlim(-1, 8)
    ax.set_ylim(-1, 7)
    ax.axis('off')

    depot_pos = (3.5, 6.0)
    s_nodes  = {1:(1,4), 2:(2,2), 3:(3.5,1), 4:(5,2), 5:(6,4)}
    demands  = {1:3, 2:4, 3:3, 4:4, 5:3}
    Q = 10
    total_d = sum(demands.values())  # 17
    lb_k = int(np.ceil(total_d / Q))  # 2

    # S boundary
    ell = mpatches.Ellipse((3.5, 2.8), 6.8, 4.5, facecolor='#E8F5E9',
                            edgecolor='#1B5E20', linewidth=2, linestyle='--', zorder=0)
    ax.add_patch(ell)
    ax.text(7.5, 4.5, '$S$', fontsize=13, color='#1B5E20', fontweight='bold')

    ax.plot(*depot_pos, 's', color='black', markersize=14, zorder=5)
    ax.text(depot_pos[0], depot_pos[1]+0.3, '0', ha='center', fontsize=9, fontweight='bold')

    for nid,(x,y) in s_nodes.items():
        ax.plot(x,y,'o', color='#1565C0', markersize=13, zorder=4)
        ax.text(x,y, f'{nid}', ha='center', va='center', fontsize=8, color='white', fontweight='bold')
        ax.text(x+0.1, y-0.5, f'd={demands[nid]}', ha='center', fontsize=7, color='#333')

    # Cut edges (depot to S)
    for nid in [1, 5]:
        x,y = s_nodes[nid]
        ax.annotate('', xy=(x,y), xytext=depot_pos,
                    arrowprops=dict(arrowstyle='->', color='#E53935', lw=2.0))
    for nid in [1, 5]:
        x,y = s_nodes[nid]
        ax.annotate('', xy=depot_pos, xytext=(x,y),
                    arrowprops=dict(arrowstyle='->', color='#E53935', lw=2.0,
                                   connectionstyle='arc3,rad=0.2'))

    ax.text(3.5, -0.7, f'$d(S)=\\sum_{{i\\in S}} d_i = {total_d}$, $Q={Q}$\n'
            f'RCI: $x(\\delta(S)) \\geq 2\\lceil {total_d}/{Q} \\rceil = {2*lb_k}$',
            ha='center', fontsize=9,
            bbox=dict(facecolor='#FFF9C4', edgecolor='#F9A825', boxstyle='round,pad=0.3'))

    # Right: framed capacity constraint formula
    ax2 = axes[1]
    ax2.axis('off')
    ax2.set_title('Rounded Capacity Inequality (RCI)', fontsize=10)
    formula = (
        r'For every subset $S \subseteq V \setminus \{0\}$:' '\n\n'
        r'$x(\delta(S)) \geq 2 \left\lceil \dfrac{d(S)}{Q} \right\rceil$' '\n\n'
        'where $x(\\delta(S))$ is the number of\n'
        'edges crossing the boundary of $S$,\n'
        '$d(S) = \\sum_{i \\in S} d_i$ is the total demand\n'
        'of $S$, and $Q$ is the vehicle capacity.\n\n'
        'Intuition: if the total demand of $S$\n'
        'exceeds $kQ$, then at least $k$ vehicles\n'
        'must enter (and leave) $S$,\n'
        'giving at least $2k$ crossing edges.'
    )
    ax2.text(0.5, 0.55, formula, ha='center', va='center', fontsize=9,
             transform=ax2.transAxes,
             bbox=dict(facecolor='#E8F5E9', edgecolor='#1B5E20', boxstyle='round,pad=0.5'))
    savefig('rounded_capacity.pdf')


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("Generating figures for Chapter 2 ...")
    fig_cvrp_example()
    fig_bnb_tree()
    fig_ap_relaxation()
    fig_set_partitioning()
    fig_column_generation()
    fig_valid_inequalities()
    fig_branch_cut_flow()
    fig_comb_inequality()
    fig_benchmark_comparison()
    fig_rounded_capacity()
    print("All figures generated successfully.")
