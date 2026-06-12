"""
gen_figures.py  —  Chapter 22: Discrete Optimization
Generates all figures for the Beamer slides using matplotlib (Agg backend)
and crops key diagrams from the book PDF using pymupdf.
"""
import os
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import fitz  # pymupdf

FIGURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

BOOK_PDF = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "optimization_book.pdf")

def savefig(name, **kwargs):
    path = os.path.join(FIGURES_DIR, name)
    plt.savefig(path, bbox_inches='tight', **kwargs)
    plt.close()
    print(f"  Saved {path}")


# ─────────────────────────────────────────────────────────────
# Helper: crop a rectangular region from a PDF page
# page_num: 1-indexed; rect = (x0, y0, x1, y1) in PDF points
# ─────────────────────────────────────────────────────────────
def crop_pdf(page_num, rect, out_name, dpi=150):
    doc = fitz.open(BOOK_PDF)
    page = doc[page_num - 1]
    clip = fitz.Rect(*rect)
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat, clip=clip)
    out_path = os.path.join(FIGURES_DIR, out_name)
    pix.save(out_path)
    doc.close()
    print(f"  Cropped PDF page {page_num} -> {out_path}")


# ═══════════════════════════════════════════════════════════════
# Figure 1: LP Relaxation — feasible region with integer points
# Shows how LP relaxation allows non-integer solutions
# ═══════════════════════════════════════════════════════════════
def fig_lp_relaxation():
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for ax, title, lp_opt, int_opt, show_frac in zip(
        axes,
        ['LP Relaxation (continuous)', 'Integer Program (rounded)'],
        [(2.5, 2.5), (2.5, 2.5)],
        [(2, 2), (2, 2)],
        [True, False]
    ):
        # Feasible polygon (example: x1+x2<=5, x1<=3, x2<=3, x1,x2>=0)
        poly_x = [0, 3, 3, 2, 0]
        poly_y = [0, 0, 2, 3, 3]
        ax.fill(poly_x, poly_y, alpha=0.25, color='steelblue', label='Feasible region')
        ax.plot(poly_x + [poly_x[0]], poly_y + [poly_y[0]], 'b-', lw=1.5)

        # Integer lattice points inside
        for i in range(5):
            for j in range(5):
                if i + j <= 5 and i <= 3 and j <= 3:
                    ax.plot(i, j, 'b.', ms=6)

        if show_frac:
            ax.plot(2.5, 2.5, 'r*', ms=14, label='LP optimum (2.5, 2.5)', zorder=5)
        ax.plot(2, 2, 'g^', ms=10, label='Integer optimum (2,2)', zorder=5)

        ax.set_xlim(-0.3, 4)
        ax.set_ylim(-0.3, 4)
        ax.set_xlabel('$x_1$', fontsize=12)
        ax.set_ylabel('$x_2$', fontsize=12)
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')

    plt.tight_layout()
    savefig('lp_relaxation.pdf')


# ═══════════════════════════════════════════════════════════════
# Figure 2: Rounding illustration — LP solution vs rounded/floor
# ═══════════════════════════════════════════════════════════════
def fig_rounding():
    fig, ax = plt.subplots(figsize=(6, 4.5))

    # feasible polygon
    verts_x = [0, 4, 4, 0]
    verts_y = [0, 0, 3, 3]
    # LP constraint: x1 + x2 <= 5
    px = [0, 4, 4, 1, 0]
    py = [0, 0, 1, 4, 4]  # not used—draw a custom polygon
    # Simple region: x1 + 2*x2 <= 9, 2*x1 + x2 <= 9
    px = [0, 4.5, 3, 0]
    py = [0, 0, 3, 4.5]
    ax.fill(px, py, alpha=0.2, color='steelblue')
    ax.plot(px + [px[0]], py + [py[0]], 'b-', lw=1.5)

    # integer grid
    for i in range(6):
        for j in range(6):
            if i + 2*j <= 9 and 2*i + j <= 9:
                ax.plot(i, j, 'b.', ms=7)

    # LP optimum (fractional)
    lp_x, lp_y = 3.0, 3.0
    ax.plot(lp_x, lp_y, 'r*', ms=16, label=f'LP opt ({lp_x},{lp_y})', zorder=6)

    # Rounded down
    ax.plot(3, 3, 'gs', ms=12, label='Floor/round (3,3)', zorder=5)

    # Annotation
    ax.annotate('Floor rounding\nmay be infeasible', xy=(3, 3), xytext=(3.5, 4),
                fontsize=9, arrowprops=dict(arrowstyle='->', color='black'))

    ax.set_xlim(-0.3, 5.5)
    ax.set_ylim(-0.3, 5.5)
    ax.set_xlabel('$x_1$', fontsize=12)
    ax.set_ylabel('$x_2$', fontsize=12)
    ax.set_title('LP Rounding Strategy', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    savefig('rounding.pdf')


# ═══════════════════════════════════════════════════════════════
# Figure 3: Cutting Planes — Gomory cut visualization
# ═══════════════════════════════════════════════════════════════
def fig_cutting_planes():
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    for ax, step in zip(axes, ['Before cut', 'After Gomory cut']):
        # Feasible LP region
        px = [0, 3.5, 3, 0]
        py = [0, 0, 3.5, 3]
        ax.fill(px, py, alpha=0.2, color='steelblue', label='LP feasible')
        ax.plot(px + [px[0]], py + [py[0]], 'b-', lw=1.5)

        # Integer points
        for i in range(5):
            for j in range(5):
                # rough feasibility check
                if (i + j <= 6) and i <= 3 and j <= 3:
                    ax.plot(i, j, 'b.', ms=7)

        # LP optimum
        ax.plot(3.0, 2.5, 'r*', ms=14, label='LP optimum', zorder=5)

        if step == 'After Gomory cut':
            # Cutting plane: x2 <= 2 (Gomory cut example)
            ax.axhline(y=2.0, color='red', lw=2, linestyle='--', label='Gomory cut: $x_2 \\leq 2$')
            # Shaded region above cut (removed)
            ax.fill([0, 4, 4, 0], [2, 2, 4, 4], alpha=0.15, color='red', label='Removed region')
            ax.plot(3.0, 2.0, 'g^', ms=12, label='New LP opt (integer)', zorder=6)

        ax.set_xlim(-0.3, 4.2)
        ax.set_ylim(-0.3, 4.0)
        ax.set_xlabel('$x_1$', fontsize=11)
        ax.set_ylabel('$x_2$', fontsize=11)
        ax.set_title(step, fontsize=11)
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    savefig('cutting_planes.pdf')


# ═══════════════════════════════════════════════════════════════
# Figure 4: Branch and Bound tree
# ═══════════════════════════════════════════════════════════════
def fig_branch_and_bound_tree():
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis('off')

    def draw_node(x, y, text, color='steelblue', fontsize=8):
        bbox = dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.7, edgecolor='black')
        ax.text(x, y, text, ha='center', va='center', fontsize=fontsize,
                bbox=bbox, zorder=5)

    def draw_edge(x1, y1, x2, y2, label=''):
        ax.annotate('', xy=(x2, y2+0.25), xytext=(x1, y1-0.25),
                    arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
        mx, my = (x1+x2)/2, (y1+y2)/2
        if label:
            ax.text(mx+0.2, my, label, fontsize=8, color='darkred')

    # Root
    draw_node(5, 5.5, 'LP relax\n$y\\geq -34.6$\n$\\mathbf{x}^*=[3,2.4,1.2,5.8]$', 'lightblue')

    # Level 1
    draw_node(2.5, 3.5, '$x_2 \\leq 2$\n$y\\geq -33.1$\n$\\mathbf{x}^*=[3,2,1.5,5.5]$', 'lightyellow')
    draw_node(7.5, 3.5, '$x_2 \\geq 3$\n$y\\geq -35.2$\n$\\mathbf{x}^*=[3.1,3,1,5.7]$', 'lightyellow')
    draw_edge(5, 5.5, 2.5, 3.5, '$x_2\\leq 2$')
    draw_edge(5, 5.5, 7.5, 3.5, '$x_2\\geq 3$')

    # Level 2 (left)
    draw_node(1, 1.5, '$x_3\\leq 1$\nInteger!\n$y=-31$', 'lightgreen')
    draw_node(4, 1.5, '$x_3\\geq 2$\nInfeasible', 'salmon')
    draw_edge(2.5, 3.5, 1, 1.5, '$x_3\\leq 1$')
    draw_edge(2.5, 3.5, 4, 1.5, '$x_3\\geq 2$')

    # Level 2 (right)
    draw_node(6.5, 1.5, '$x_1\\leq 3$\nInteger!\n$y=-35$', 'lightgreen')
    draw_node(9, 1.5, '$x_1\\geq 4$\nInfeasible', 'salmon')
    draw_edge(7.5, 3.5, 6.5, 1.5, '$x_1\\leq 3$')
    draw_edge(7.5, 3.5, 9, 1.5, '$x_1\\geq 4$')

    ax.set_title('Branch and Bound Search Tree', fontsize=13, pad=5)
    plt.tight_layout()
    savefig('branch_bound_tree.pdf')


# ═══════════════════════════════════════════════════════════════
# Figure 5: Branching splits feasible set (Figure 22.4 from book)
# ═══════════════════════════════════════════════════════════════
def fig_branching_split():
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    def plot_region(ax, title, xfilt=None, opt_pt=None, opt_label='', cut_x=None, cut_side=None):
        # feasible region: x1 + x2 <= 5, x1,x2 in [0,4]
        px = [0, 3.5, 3.5, 1.5, 0]
        py = [0, 0,   2,   3.5, 3.5]
        ax.fill(px, py, alpha=0.2, color='steelblue')
        ax.plot(px + [px[0]], py + [py[0]], 'b-', lw=1.5)

        # lattice
        for i in range(5):
            for j in range(5):
                if i + j <= 5 and i <= 3.5 and j <= 3.5:
                    if xfilt is None or xfilt(i):
                        ax.plot(i, j, 'bo', ms=5, alpha=0.6)
                    else:
                        ax.plot(i, j, 'o', ms=5, color='lightgray')

        if cut_x is not None:
            ax.axvline(x=cut_x, color='red', lw=2, linestyle='--')
            if cut_side == 'left':
                ax.fill([cut_x, 5, 5, cut_x], [0, 0, 5, 5], alpha=0.1, color='red')
            else:
                ax.fill([0, cut_x, cut_x, 0], [0, 0, 5, 5], alpha=0.1, color='red')

        if opt_pt:
            ax.plot(*opt_pt, 'r*', ms=14, label=opt_label, zorder=6)
            ax.legend(fontsize=8)

        ax.set_xlim(-0.3, 4.5)
        ax.set_ylim(-0.3, 4.5)
        ax.set_xlabel('$x_1$', fontsize=11)
        ax.set_ylabel('$x_2$', fontsize=11)
        ax.set_title(title, fontsize=11)
        ax.grid(True, alpha=0.3)

    plot_region(axes[0], 'Original LP\n$\\mathbf{x}^*_c$ fractional',
                opt_pt=(1.5, 3.5), opt_label='$\\mathbf{x}^*_c$')
    plot_region(axes[1], 'Branch: $x_1 \\leq 1$',
                xfilt=lambda i: i <= 1,
                opt_pt=(1.0, 3.5), opt_label='$\\mathbf{x}^*_{c,left}$',
                cut_x=1.0, cut_side='right')
    plot_region(axes[2], 'Branch: $x_1 \\geq 2$',
                xfilt=lambda i: i >= 2,
                opt_pt=(2.0, 2.5), opt_label='$\\mathbf{x}^*_{c,right}$',
                cut_x=2.0, cut_side='left')

    plt.suptitle('Branching Splits the Feasible Set', fontsize=13, y=1.02)
    plt.tight_layout()
    savefig('branching_split.pdf')


# ═══════════════════════════════════════════════════════════════
# Figure 6: 0-1 Knapsack DP table
# ═══════════════════════════════════════════════════════════════
def fig_knapsack_dp():
    # 4 items: weights=[2,3,4,5], values=[3,4,5,6], capacity=5
    weights = [2, 3, 4, 5]
    values  = [3, 4, 5, 6]
    n = len(weights)
    W = 5

    dp = np.zeros((n+1, W+1), dtype=float)
    for i in range(1, n+1):
        for w in range(W+1):
            dp[i, w] = dp[i-1, w]
            if weights[i-1] <= w:
                dp[i, w] = max(dp[i, w], dp[i-1, w - weights[i-1]] + values[i-1])

    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(dp, cmap='YlOrRd', aspect='auto')
    plt.colorbar(im, ax=ax, label='Optimal value')

    ax.set_xticks(range(W+1))
    ax.set_xticklabels([f'$w={w}$' for w in range(W+1)], fontsize=9)
    ax.set_yticks(range(n+1))
    ax.set_yticklabels(['Item 0\n(base)'] + [f'Item {i}\n$w_i={weights[i-1]}, v_i={values[i-1]}$'
                                              for i in range(1, n+1)], fontsize=9)
    for i in range(n+1):
        for j in range(W+1):
            ax.text(j, i, f'{dp[i,j]:.0f}', ha='center', va='center', fontsize=10,
                    color='black' if dp[i, j] < 5 else 'white')

    ax.set_title('0-1 Knapsack DP Table\n(weights=[2,3,4,5], values=[3,4,5,6], capacity=5)',
                 fontsize=11)
    ax.set_xlabel('Remaining capacity $w$', fontsize=11)
    ax.set_ylabel('Items considered', fontsize=11)
    plt.tight_layout()
    savefig('knapsack_dp.pdf')


# ═══════════════════════════════════════════════════════════════
# Figure 7: DP recursion tree (top-down memoization)
# ═══════════════════════════════════════════════════════════════
def fig_dp_recursion():
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5.5)
    ax.axis('off')

    def node(x, y, text, color='lightyellow', fs=9):
        bbox = dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.85, edgecolor='gray')
        ax.text(x, y, text, ha='center', va='center', fontsize=fs, bbox=bbox, zorder=5)

    def edge(x1, y1, x2, y2):
        ax.annotate('', xy=(x2, y2+0.28), xytext=(x1, y1-0.28),
                    arrowprops=dict(arrowstyle='->', color='gray', lw=1.2))

    node(5, 5, 'knapsack(4, 5)', color='steelblue', fs=10)
    node(2.5, 3.7, 'knapsack(3, 5)\nexclude item 4', color='lightyellow')
    node(7.5, 3.7, 'knapsack(3, 0)\ninclude item 4', color='lightyellow')
    edge(5, 5, 2.5, 3.7)
    edge(5, 5, 7.5, 3.7)

    node(1, 2.4, 'knapsack(2,5)', color='lightyellow', fs=8)
    node(4, 2.4, 'knapsack(2,2)', color='lightyellow', fs=8)
    node(6.5, 2.4, 'knapsack(2,0)', color='lightyellow', fs=8)
    node(9, 2.4, 'knapsack(2,0)\n(memoized)', color='lightgreen', fs=8)
    edge(2.5, 3.7, 1, 2.4)
    edge(2.5, 3.7, 4, 2.4)
    edge(7.5, 3.7, 6.5, 2.4)
    edge(7.5, 3.7, 9, 2.4)

    ax.text(9.3, 2.4, r'$\leftarrow$ cached!', fontsize=8, color='green')

    ax.set_title('DP Recursion Tree with Memoization\n(overlapping subproblems are computed once)', fontsize=11)
    plt.tight_layout()
    savefig('dp_recursion.pdf')


# ═══════════════════════════════════════════════════════════════
# Figure 8: Ant Colony Optimization on TSP — pheromone evolution
# Shows 4 stages of convergence
# ═══════════════════════════════════════════════════════════════
def fig_aco_tsp():
    np.random.seed(42)
    n_cities = 12
    coords = np.random.rand(n_cities, 2) * 10

    def tour_length(tour, coords):
        total = 0
        for i in range(len(tour)):
            a, b = tour[i], tour[(i+1) % len(tour)]
            total += np.linalg.norm(coords[a] - coords[b])
        return total

    # Simple nearest-neighbor heuristic for a "good" tour
    def nn_tour(start, coords):
        n = len(coords)
        unvisited = list(range(n))
        tour = [start]
        unvisited.remove(start)
        while unvisited:
            cur = tour[-1]
            nearest = min(unvisited, key=lambda j: np.linalg.norm(coords[cur] - coords[j]))
            tour.append(nearest)
            unvisited.remove(nearest)
        return tour

    best_tour = min([nn_tour(s, coords) for s in range(n_cities)], key=lambda t: tour_length(t, coords))

    fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))
    stages = ['Iteration 1\n(uniform)', 'Iteration 10', 'Iteration 50', 'Iteration 100\n(converged)']
    alphas_base = [0.12, 0.3, 0.6, 1.0]

    for ax, stage, alpha_scale in zip(axes, stages, alphas_base):
        # Draw all edges faintly
        for i in range(n_cities):
            for j in range(i+1, n_cities):
                ax.plot([coords[i,0], coords[j,0]], [coords[i,1], coords[j,1]],
                        'b-', lw=0.3, alpha=0.08 + 0.04 * alpha_scale)

        # Draw best tour with alpha proportional to stage
        for k in range(n_cities):
            a, b = best_tour[k], best_tour[(k+1) % n_cities]
            ax.plot([coords[a,0], coords[b,0]], [coords[a,1], coords[b,1]],
                    'b-', lw=0.8 + alpha_scale * 1.5, alpha=min(alpha_scale + 0.1, 1.0))

        ax.scatter(coords[:,0], coords[:,1], s=40, c='black', zorder=5)
        ax.set_title(stage, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(-0.5, 10.5)
        ax.set_ylim(-0.5, 10.5)

    plt.suptitle('Ant Colony Optimization on TSP (pheromone opacity = pheromone level)',
                 fontsize=11, y=1.01)
    plt.tight_layout()
    savefig('aco_tsp.pdf')


# ═══════════════════════════════════════════════════════════════
# Figure 9: Totally unimodular matrix illustration
# ═══════════════════════════════════════════════════════════════
def fig_total_unimodular():
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))

    # Non-TU matrix
    A_not = np.array([[1, 1], [1, 0]])
    # TU matrix
    A_tu = np.array([[1, -1, 0], [0, 1, -1]])

    for ax, mat, title, is_tu in zip(axes,
                                     [A_not, A_tu],
                                     ['Non-TU: $\\det = -1$ or 2 submatrix',
                                      'TU: all subdeterminants $\\in \\{-1,0,1\\}$'],
                                     [False, True]):
        im = ax.imshow(mat, cmap='RdBu', vmin=-1.5, vmax=1.5, aspect='auto')
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                ax.text(j, i, str(mat[i, j]), ha='center', va='center', fontsize=14, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
        color = 'green' if is_tu else 'red'
        ax.set_title(title, fontsize=10, color=color)
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(2)

    plt.suptitle('Totally Unimodular (TU) Matrices', fontsize=12)
    plt.tight_layout()
    savefig('total_unimodular.pdf')


# ═══════════════════════════════════════════════════════════════
# Figure 10: Integer program feasible set (intro figure)
# ═══════════════════════════════════════════════════════════════
def fig_intro_integer():
    fig, ax = plt.subplots(figsize=(5.5, 5))

    # Example from p483: minimize -x1 - x2
    # subject to: x1 + x2 <= 3.5,  x1 - x2 >= -0.5,  x1,x2 >= 0
    # LP optimum is fractional, integer optimum is at (2,1) or (1,2)
    from matplotlib.patches import Polygon
    verts = np.array([[0, 0], [3.5, 0], [2, 1.5], [0, 3.5]])
    poly = Polygon(verts, closed=True, alpha=0.2, facecolor='steelblue', edgecolor='navy', lw=1.5)
    ax.add_patch(poly)

    # grid points
    for i in range(5):
        for j in range(5):
            if i + j <= 3.5 and i - j >= -0.5:
                ax.plot(i, j, 'b.', ms=10, zorder=4)

    # LP relaxation optimum
    ax.plot(2.0, 1.5, 'r*', ms=16, label='LP opt $(2, 1.5)$', zorder=6)
    # integer optimum
    ax.plot(2, 1, 'g^', ms=12, label='Integer opt $(2, 1)$', zorder=5)

    ax.set_xlim(-0.4, 4.2)
    ax.set_ylim(-0.4, 4.2)
    ax.set_xlabel('$x_1$', fontsize=13)
    ax.set_ylabel('$x_2$', fontsize=13)
    ax.legend(fontsize=10)
    ax.set_title('Integer Programming: Discrete Feasible Set', fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    savefig('intro_integer.pdf')


# ═══════════════════════════════════════════════════════════════
# Crop key diagrams from book PDF
# ═══════════════════════════════════════════════════════════════
def crop_book_figures():
    # Page numbers are 1-indexed in the PDF
    # p483 = first page of chapter 22 = PDF page 483
    # The PDF appears to use actual page numbers matching book
    # We attempt to crop the branching figure from p493 (Fig 22.4)
    try:
        # Figure 22.4: Branching splits the feasible set (p493, book p473)
        # p493 in /tmp corresponds to PDF page ~493
        crop_pdf(493, (30, 200, 560, 620), 'book_branching_split.png', dpi=150)
    except Exception as e:
        print(f"  Warning: crop_book_figures failed: {e}")


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print("Generating figures for Chapter 22: Discrete Optimization")
    fig_intro_integer()
    fig_lp_relaxation()
    fig_rounding()
    fig_cutting_planes()
    fig_branch_and_bound_tree()
    fig_branching_split()
    fig_knapsack_dp()
    fig_dp_recursion()
    fig_aco_tsp()
    fig_total_unimodular()
    crop_book_figures()
    print("All figures generated.")
