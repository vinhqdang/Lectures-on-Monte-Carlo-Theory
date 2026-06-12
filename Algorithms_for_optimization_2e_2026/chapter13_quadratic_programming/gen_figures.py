"""
gen_figures.py — Generate all figures for Chapter 13: Quadratic Programming
Algorithms for Optimization, 2nd ed. (Kochenderfer & Wheeler, 2026)
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import numpy as np
from scipy.optimize import nnls
import os
import fitz  # pymupdf

FIGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
os.makedirs(FIGDIR, exist_ok=True)

def savefig(name):
    plt.savefig(os.path.join(FIGDIR, name + '.pdf'), bbox_inches='tight')
    plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# Fig 1: Problem-transformation chain (diagram)
# ─────────────────────────────────────────────────────────────────────────────
def fig_transform_chain():
    fig, ax = plt.subplots(figsize=(8, 1.8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 2)
    ax.axis('off')

    boxes = [
        (1.0, 'Quadratic\nProgram'),
        (3.5, 'Least-\nSquares'),
        (6.0, 'Least\nDistance'),
        (8.5, 'Nonneg.\nLeast-Squares'),
    ]
    colors = ['#4472C4', '#ED7D31', '#A9D18E', '#FF0000']
    colors = ['#5B9BD5', '#ED7D31', '#70AD47', '#FFC000']

    for x, label in boxes:
        rect = plt.Rectangle((x - 0.9, 0.4), 1.8, 1.2,
                              linewidth=1.5, edgecolor='#2E4057',
                              facecolor='#D6E4F0', zorder=2)
        ax.add_patch(rect)
        ax.text(x, 1.0, label, ha='center', va='center',
                fontsize=9, fontweight='bold', zorder=3)

    for x in [2.1, 4.6, 7.1]:
        ax.annotate('', xy=(x + 0.4, 1.0), xytext=(x, 1.0),
                    arrowprops=dict(arrowstyle='->', color='#2E4057', lw=1.5))

    ax.set_title('Figure 13.1  —  Transformation chain for QP with positive-definite $\\mathbf{Q}$',
                 fontsize=10, pad=4)
    savefig('fig_transform_chain')


# ─────────────────────────────────────────────────────────────────────────────
# Fig 2: Example QP contour plot (Example 13.1)
# ─────────────────────────────────────────────────────────────────────────────
def fig_qp_example():
    """Example 13.1: minimize ||Ax - b|| s.t. 2x1 - x2 >= 2, x1 - 3x2 <= 4"""
    fig, ax = plt.subplots(figsize=(5, 4.5))

    # Objective: minimize ||Ax - b||^2 where A = [[2,1],[-4,3]], b = [1,2]
    A = np.array([[2., 1.], [-4., 3.]])
    b_vec = np.array([1., 2.])

    x1 = np.linspace(-2, 4, 300)
    x2 = np.linspace(-2, 4, 300)
    X1, X2 = np.meshgrid(x1, x2)

    # Objective value
    Z = np.zeros_like(X1)
    for i in range(X1.shape[0]):
        for j in range(X1.shape[1]):
            xv = np.array([X1[i, j], X2[i, j]])
            r = A @ xv - b_vec
            Z[i, j] = np.dot(r, r)

    # Feasible region: 2x1 - x2 >= 2 and x1 - 3x2 <= 4
    feas = (2*X1 - X2 >= 2) & (X1 - 3*X2 <= 4)
    ax.contourf(X1, X2, np.where(feas, 1, np.nan), levels=[0.5, 1.5],
                colors=['#AED6F1'], alpha=0.5)

    levels = [0.5, 2, 5, 10, 20, 40, 80]
    cs = ax.contour(X1, X2, Z, levels=levels,
                    colors=['#2980B9', '#1ABC9C', '#F39C12', '#E74C3C',
                            '#9B59B6', '#1E8BC3', '#16A085'])

    # Constraint boundaries
    ax.plot(x1, 2*x1 - 2, 'k-', lw=1.5, label='$2x_1 - x_2 = 2$')
    ax.plot(x1, (x1 - 4)/3., 'k--', lw=1.5, label='$x_1 - 3x_2 = 4$')

    # Optimal point x* = [1.4, 0.8]
    xs = np.array([1.4, 0.8])
    ax.plot(*xs, 'ro', ms=8, zorder=5)
    ax.annotate(r'$\mathbf{x}^* = [1.4,\,0.8]$', xy=xs,
                xytext=(xs[0] + 0.4, xs[1] + 0.5),
                fontsize=9, arrowprops=dict(arrowstyle='->', color='k'))

    ax.set_xlim(-2, 4)
    ax.set_ylim(-2, 4)
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_title('Example 13.1 — QP with Linear Constraints', fontsize=10)
    ax.legend(fontsize=8, loc='upper left')
    savefig('fig_qp_example')


# ─────────────────────────────────────────────────────────────────────────────
# Fig 3: Four QP solution cases (Fig 13.2)
# ─────────────────────────────────────────────────────────────────────────────
def fig_qp_cases():
    fig, axes = plt.subplots(1, 4, figsize=(10, 2.8))
    titles = ['Internal\nSolution', 'Boundary\nSolution', 'Infeasible', 'Unbounded']

    for ax in axes:
        ax.set_xlim(-1, 3)
        ax.set_ylim(-1, 3)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel('$x_1$', fontsize=8)
        ax.set_ylabel('$x_2$', fontsize=8, rotation=0, labelpad=8)

    # Case 1: Internal solution — feasible region contains the unconstrained minimizer
    ax = axes[0]
    poly = Polygon([[0, 0], [2.5, 0], [2.5, 2.5], [0, 2.5]], closed=True)
    ax.add_patch(plt.Polygon([[0.3, 0.3], [2.5, 0.3], [2.5, 2.5], [0.3, 2.5]],
                              facecolor='#AED6F1', alpha=0.5))
    x1 = np.linspace(-1, 3, 200)
    x2 = np.linspace(-1, 3, 200)
    X1, X2 = np.meshgrid(x1, x2)
    Z = (X1 - 1.2)**2 + 1.5*(X2 - 1.2)**2
    ax.contour(X1, X2, Z, levels=8, colors='gray', alpha=0.7)
    ax.plot(1.2, 1.2, 'b.', ms=10)
    ax.annotate(r'$\mathbf{x}^*$', xy=(1.2, 1.2), xytext=(1.5, 1.5),
                fontsize=9, arrowprops=dict(arrowstyle='->', color='k', lw=0.8))
    ax.set_title(titles[0], fontsize=9)

    # Case 2: Boundary solution
    ax = axes[1]
    ax.add_patch(plt.Polygon([[1.0, -1], [3, -1], [3, 3], [1.0, 3]],
                              facecolor='#AED6F1', alpha=0.5))
    Z = (X1 - 0.5)**2 + (X2 - 1.5)**2
    ax.contour(X1, X2, Z, levels=8, colors='gray', alpha=0.7)
    ax.plot(1.0, 1.5, 'b.', ms=10)
    ax.annotate(r'$\mathbf{x}^*$', xy=(1.0, 1.5), xytext=(1.5, 2.0),
                fontsize=9, arrowprops=dict(arrowstyle='->', color='k', lw=0.8))
    ax.set_title(titles[1], fontsize=9)

    # Case 3: Infeasible — empty feasible region
    ax = axes[2]
    ax.add_patch(plt.Polygon([[2.0, -1], [3, -1], [3, 3], [2.0, 3]],
                              facecolor='#AED6F1', alpha=0.4))
    ax.add_patch(plt.Polygon([[-1, 0], [1.5, 0], [1.5, 3], [-1, 3]],
                              facecolor='#F1948A', alpha=0.4))
    Z = (X1 - 1.5)**2 + (X2 - 1.5)**2
    ax.contour(X1, X2, Z, levels=8, colors='gray', alpha=0.7)
    ax.text(1.5, -0.6, 'No feasible\nregion', ha='center', fontsize=7, color='red')
    ax.set_title(titles[2], fontsize=9)

    # Case 4: Unbounded
    ax = axes[3]
    ax.add_patch(plt.Polygon([[-1, 0.5], [3, 0.5], [3, 3], [-1, 3]],
                              facecolor='#AED6F1', alpha=0.5))
    # Non-PD Q: saddle-like contours
    Z2 = (X1)**2 - (X2)**2
    ax.contour(X1, X2, Z2, levels=10, colors='gray', alpha=0.7)
    ax.text(0.5, 2.3, 'Unbounded\n$\\mathbf{Q}$ not PD', ha='center', fontsize=7, color='red')
    ax.set_title(titles[3], fontsize=9)

    fig.suptitle('Figure 13.2 — QP Solution Cases', fontsize=10, y=1.02)
    plt.tight_layout()
    savefig('fig_qp_cases')


# ─────────────────────────────────────────────────────────────────────────────
# Fig 4: Least distance program (Fig 13.3)
# ─────────────────────────────────────────────────────────────────────────────
def fig_least_distance():
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    G = np.array([[2.0, -2.0], [-1.0, 5.0]])
    h = np.array([2.0, -7.0])

    x1 = np.linspace(-2, 4, 400)
    x2 = np.linspace(-2, 4, 400)
    X1, X2 = np.meshgrid(x1, x2)

    # Feasible: Gx >= h
    feas = np.ones_like(X1, dtype=bool)
    for i in range(len(h)):
        feas &= (G[i, 0]*X1 + G[i, 1]*X2 >= h[i])

    ax.contourf(X1, X2, np.where(feas, 1.0, np.nan),
                levels=[0.5, 1.5], colors=['#AED6F1'], alpha=0.5)

    # Objective contours (circles)
    Z = X1**2 + X2**2
    levels_circ = [0.2, 0.5, 1.0, 1.5, 2.5, 4.0]
    cs = ax.contour(X1, X2, Z, levels=levels_circ,
                    colors=['#1ABC9C', '#27AE60', '#F39C12', '#E67E22',
                            '#E74C3C', '#8E44AD'], alpha=0.8)

    # Constraint lines
    ax.plot(x1, (G[0, 0]*x1 - h[0])/G[0, 1], 'k-', lw=1.5)
    ax.plot(x1, (G[1, 0]*x1 - h[1])/G[1, 1], 'k--', lw=1.5)

    # Optimal x* = [0.5, -0.5]
    xs = np.array([0.5, -0.5])
    ax.plot(*xs, 'ro', ms=8, zorder=5)
    ax.annotate(r'$\mathbf{x}^*$', xy=xs, xytext=(xs[0] + 0.5, xs[1] + 0.5),
                fontsize=10, arrowprops=dict(arrowstyle='->', color='k'))

    ax.set_xlim(-2, 4)
    ax.set_ylim(-2, 4)
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_title('Figure 13.3 — Least Distance Program\n'
                 r'$\mathbf{G}\mathbf{x}\geq\mathbf{h}$, minimize $\|\mathbf{x}\|$', fontsize=9)
    savefig('fig_least_distance')


# ─────────────────────────────────────────────────────────────────────────────
# Fig 5: NNLS algorithmic progression (Example 13.4)
# ─────────────────────────────────────────────────────────────────────────────
def fig_nnls_progression():
    E = np.array([[2.0, 1.0], [-4.0, 3.0]])

    cases = [
        {'f': np.array([-5.0, 0.0]),  'steps': 1, 'label': r'$\mathbf{f}=[-5,0]$, solved immediately'},
        {'f': np.array([-3.0, -12.0]),'steps': 2, 'label': r'$\mathbf{f}=[-3,-12]$, 1 relaxation'},
        {'f': np.array([5.0, 5.0]),   'steps': 3, 'label': r'$\mathbf{f}=[5,5]$, 2 relaxations'},
    ]

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2))

    for ax, case in zip(axes, cases):
        f = case['f']
        x1 = np.linspace(-4, 4, 300)
        x2 = np.linspace(-4, 4, 300)
        X1, X2 = np.meshgrid(x1, x2)

        Z = np.zeros_like(X1)
        for i in range(X1.shape[0]):
            for j in range(X1.shape[1]):
                xv = np.array([X1[i, j], X2[i, j]])
                r = E @ xv - f
                Z[i, j] = np.dot(r, r)

        # Feasible region x >= 0
        ax.contourf(X1, X2, np.where((X1 >= 0) & (X2 >= 0), 1, np.nan),
                    levels=[0.5, 1.5], colors=['#AED6F1'], alpha=0.4)
        ax.contour(X1, X2, Z, levels=10, colors='gray', alpha=0.6, linewidths=0.8)

        # Solution via scipy NNLS
        x_sol, _ = nnls(E, f)
        ax.plot(*x_sol, 'bo', ms=8, zorder=5)
        ax.annotate(r'$\mathbf{x}^*$', xy=x_sol,
                    xytext=(x_sol[0] + 0.5, x_sol[1] + 0.5),
                    fontsize=9, arrowprops=dict(arrowstyle='->', color='k', lw=0.8))
        ax.axhline(0, color='k', lw=0.8)
        ax.axvline(0, color='k', lw=0.8)
        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)
        ax.set_xlabel('$x_1$', fontsize=8)
        ax.set_ylabel('$x_2$', fontsize=8)
        ax.set_title(case['label'], fontsize=8)

    fig.suptitle('Example 13.4 — NNLS Algorithmic Progressions', fontsize=10)
    plt.tight_layout()
    savefig('fig_nnls_progression')


# ─────────────────────────────────────────────────────────────────────────────
# Crop Fig 13.1 (transform chain) from PDF — page 294 (index 293)
# ─────────────────────────────────────────────────────────────────────────────
def crop_fig13_1_from_pdf():
    pdf_path = '/Users/vinhdq1/work/Lectures-on-Monte-Carlo-Theory/Algorithms_for_optimization_2e_2026/optimization_book.pdf'
    doc = fitz.open(pdf_path)
    # Figure 13.1 is on page 294 (index 293), in the right margin
    page = doc[293]
    # Get page dimensions
    rect = page.rect
    # The figure is in the right column, roughly bottom third
    clip = fitz.Rect(rect.width * 0.58, rect.height * 0.55, rect.width, rect.height * 0.80)
    pix = page.get_pixmap(matrix=fitz.Matrix(3, 3), clip=clip)
    pix.save(os.path.join(FIGDIR, 'fig13_1_crop.png'))
    doc.close()


# ─────────────────────────────────────────────────────────────────────────────
# Fig: Example 13.2 — 3D feasible region and reduced 2D problem
# ─────────────────────────────────────────────────────────────────────────────
def fig_example13_2():
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))

    # Left: 3D sketch
    ax3d = fig.add_subplot(1, 2, 1, projection='3d')
    fig.delaxes(axes[0])

    # The feasible region: 0<=x1<=1, 0<=x3<=1, x2=x3
    # We plot the planar feasible region
    X1 = np.linspace(0, 1, 20)
    X3 = np.linspace(0, 1, 20)
    X1g, X3g = np.meshgrid(X1, X3)
    X2g = X3g  # equality x2 = x3

    ax3d.plot_surface(X1g, X2g, X3g, alpha=0.4, color='#3498DB')
    ax3d.set_xlabel('$x_1$', fontsize=8)
    ax3d.set_ylabel('$x_2$', fontsize=8)
    ax3d.set_zlabel('$x_3$', fontsize=8)
    ax3d.set_title('Original 3D problem', fontsize=9)
    ax3d.scatter([0], [0.2], [0.2], color='r', s=50, zorder=5)
    ax3d.text(0, 0.2, 0.2, r'$\mathbf{x}^*$', fontsize=9)

    # Right: 2D reduced problem
    ax = axes[1]
    y1 = np.linspace(-0.5, 1.5, 300)
    y2 = np.linspace(-0.5, 1.5, 300)
    Y1, Y2 = np.meshgrid(y1, y2)

    A2 = np.array([[3.207, 1.793], [-1.414, 1.414]])
    b2 = np.array([1., 2.])

    Z = np.zeros_like(Y1)
    for i in range(Y1.shape[0]):
        for j in range(Y1.shape[1]):
            yv = np.array([Y1[i, j], Y2[i, j]])
            r = A2 @ yv - b2
            Z[i, j] = np.dot(r, r)

    # Feasibility constraint
    C2 = np.array([[0.5, 0.5], [0.707, -0.707], [-0.5, -0.5], [-0.707, 0.707]])
    d2 = np.array([0., 0., -1., -1.])
    feas = np.ones_like(Y1, dtype=bool)
    for i in range(len(d2)):
        feas &= (C2[i, 0]*Y1 + C2[i, 1]*Y2 >= d2[i])

    ax.contourf(Y1, Y2, np.where(feas, 1.0, np.nan),
                levels=[0.5, 1.5], colors=['#AED6F1'], alpha=0.5)
    ax.contour(Y1, Y2, Z, levels=12, colors='gray', alpha=0.7)

    # Optimal in reduced space
    y_sol = np.array([0.566, 0.283])  # approx
    ax.plot(*y_sol, 'ro', ms=8, zorder=5)
    ax.annotate(r'$\mathbf{y}_{2:3}^*$', xy=y_sol,
                xytext=(y_sol[0] + 0.15, y_sol[1] + 0.15), fontsize=9)
    ax.set_xlabel('$y_2$', fontsize=9)
    ax.set_ylabel('$y_3$', fontsize=9)
    ax.set_title('Reduced 2D problem', fontsize=9)
    ax.set_xlim(-0.1, 1.2)
    ax.set_ylim(-0.5, 1.0)

    plt.suptitle('Example 13.2 — Equality Constraint Elimination', fontsize=10)
    plt.tight_layout()
    savefig('fig_example13_2')


# ─────────────────────────────────────────────────────────────────────────────
# Fig: Dual certificate schematic
# ─────────────────────────────────────────────────────────────────────────────
def fig_dual_certificate():
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.axis('off')

    # Three columns: Primal, Dual, Duality Gap
    data = [
        ['Condition', 'Requirement', 'Interpretation'],
        ['Primal Feasibility', r'$\mathbf{C}_{LE}\mathbf{x} \leq \mathbf{d}_{LE}$,  $\mathbf{C}_{EQ}\mathbf{x} = \mathbf{d}_{EQ}$', 'x is in the feasible set'],
        ['Dual Feasibility', r'$\boldsymbol{\mu} \geq \mathbf{0}$', 'Dual variables are non-neg.'],
        ['Complementary\nSlackness', r'$\boldsymbol{\mu}^\top(\mathbf{C}_{LE}\mathbf{x}-\mathbf{d}_{LE})=0$', 'Active iff constraint is tight'],
        ['Zero Duality Gap', r'$f(\mathbf{x}^*) = \mathcal{D}(\boldsymbol{\mu}^*, \boldsymbol{\lambda}^*)$', 'Confirms global optimality'],
    ]

    col_widths = [0.28, 0.44, 0.28]
    col_x = [0.01, 0.30, 0.75]
    row_h = 0.18

    for r, row in enumerate(data):
        for c, cell in enumerate(row):
            y = 1.0 - r * row_h
            weight = 'bold' if r == 0 else 'normal'
            size = 9 if r == 0 else 8
            facecolor = '#2E75B6' if r == 0 else ('#EBF5FB' if r % 2 == 0 else 'white')
            textcolor = 'white' if r == 0 else 'black'
            rect = plt.Rectangle((col_x[c], y - row_h + 0.02), col_widths[c], row_h - 0.02,
                                  facecolor=facecolor, edgecolor='#AAAAAA', lw=0.5,
                                  transform=ax.transAxes)
            ax.add_patch(rect)
            ax.text(col_x[c] + col_widths[c]/2, y - row_h/2 + 0.01, cell,
                    ha='center', va='center', fontsize=size, fontweight=weight,
                    color=textcolor, transform=ax.transAxes)

    ax.set_title('KKT Conditions as Dual Certificate (Sec. 13.6)', fontsize=10)
    savefig('fig_dual_certificate')


# ─────────────────────────────────────────────────────────────────────────────
# Fig: Python code demo — NNLS via scipy
# ─────────────────────────────────────────────────────────────────────────────
def fig_nnls_demo():
    """Demo: solve NNLS and plot result."""
    E = np.array([[2.0, -1.0], [-2.0, 5.0], [2.0, -7.0]])
    f = np.array([0., 0., 1.])
    y_star, residual = nnls(E, f)

    r = E @ y_star - f
    r_n1 = r[-1]
    x_star = -r[:2] / r[-1]

    fig, ax = plt.subplots(figsize=(4.5, 4.5))

    x1 = np.linspace(-2, 3, 400)
    x2 = np.linspace(-2, 3, 400)
    X1, X2 = np.meshgrid(x1, x2)

    G = np.array([[2.0, -2.0], [-1.0, 5.0]])
    h = np.array([2.0, -7.0])

    feas = (G[0, 0]*X1 + G[0, 1]*X2 >= h[0]) & \
           (G[1, 0]*X1 + G[1, 1]*X2 >= h[1])

    ax.contourf(X1, X2, np.where(feas, 1.0, np.nan),
                levels=[0.5, 1.5], colors=['#AED6F1'], alpha=0.5)

    Z = X1**2 + X2**2
    ax.contour(X1, X2, Z, levels=[0.1, 0.3, 0.6, 1.0, 1.5, 2.5, 4.0],
               colors='gray', alpha=0.7)

    ax.plot(x1, (G[0, 0]*x1 - h[0])/G[0, 1], 'k-', lw=1.5)
    ax.plot(x1, (G[1, 0]*x1 - h[1])/G[1, 1], 'k--', lw=1.5)

    ax.plot(*x_star, 'ro', ms=9, zorder=5)
    ax.annotate(r'$\mathbf{x}^*=[0.5,\,-0.5]$', xy=x_star,
                xytext=(x_star[0] + 0.5, x_star[1] + 0.5),
                fontsize=9, arrowprops=dict(arrowstyle='->', color='k'))

    ax.set_xlim(-2, 3)
    ax.set_ylim(-2, 3)
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_title('Example 13.3 — Least Distance via NNLS\n'
                 r'$\mathbf{x}^* = [0.5,\,-0.5]$', fontsize=9)
    savefig('fig_nnls_demo')


# ─────────────────────────────────────────────────────────────────────────────
# Run all
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print('Generating figures...')
    fig_transform_chain()
    print('  fig_transform_chain.pdf')
    fig_qp_example()
    print('  fig_qp_example.pdf')
    fig_qp_cases()
    print('  fig_qp_cases.pdf')
    fig_least_distance()
    print('  fig_least_distance.pdf')
    fig_nnls_progression()
    print('  fig_nnls_progression.pdf')
    fig_example13_2()
    print('  fig_example13_2.pdf')
    fig_dual_certificate()
    print('  fig_dual_certificate.pdf')
    fig_nnls_demo()
    print('  fig_nnls_demo.pdf')
    crop_fig13_1_from_pdf()
    print('  fig13_1_crop.png')
    print('All figures generated.')
