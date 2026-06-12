"""
gen_figures.py  –  Generate all figures for Chapter 12: Linear Programming
Book: Algorithms for Optimization (2nd ed., 2026), Kochenderfer & Wheeler
"""

import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
from scipy.spatial import ConvexHull

FIGURES_DIR = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

def savefig(name):
    path = os.path.join(FIGURES_DIR, name)
    plt.savefig(path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f'  Saved {path}')


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 – Linear objective contours (Fig 12.1 in book)
# ─────────────────────────────────────────────────────────────────────────────
def fig_linear_contours():
    fig, ax = plt.subplots(figsize=(4, 3))
    c = np.array([1.0, 0.5])          # direction of increase
    x1 = np.linspace(-2, 3, 200)
    x2 = np.linspace(-1, 3, 200)
    X1, X2 = np.meshgrid(x1, x2)
    Z = c[0]*X1 + c[1]*X2

    # contour lines
    levels = [-1, 0, 1, 2, 3]
    CS = ax.contour(X1, X2, Z, levels=levels, colors='steelblue', linewidths=1.2)
    ax.clabel(CS, fmt=r'$\mathbf{c}^\top\mathbf{x}=%g$', fontsize=7)

    # arrow for c direction
    ax.annotate('', xy=(0.8, 0.9), xytext=(0.0, 0.5),
                arrowprops=dict(arrowstyle='->', color='darkred', lw=2))
    ax.text(0.85, 0.85, r'$\mathbf{c}$', color='darkred', fontsize=11)

    ax.set_xlabel(r'$x_1$'); ax.set_ylabel(r'$x_2$')
    ax.set_title('Linear Objective Contours', fontsize=10)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    savefig('linear_contours.pdf')


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 – Half-space (Fig 12.2 in book)
# ─────────────────────────────────────────────────────────────────────────────
def fig_halfspace():
    fig, ax = plt.subplots(figsize=(4, 3.5))
    x1 = np.linspace(-2, 4, 200)

    # hyperplane  w^T x = b   with w=[1,1], b=2  =>  x2 = b - x1
    w = np.array([1.0, 1.0])
    b = 2.0
    x2_line = b - x1

    ax.plot(x1, x2_line, 'k-', lw=2, label=r'$\mathbf{w}^\top\mathbf{x}=b$')

    # shade the feasible side  w^T x <= b
    ax.fill_between(x1, x2_line, -2, alpha=0.25, color='steelblue',
                    label=r'$\mathbf{w}^\top\mathbf{x}\leq b$')

    # normal vector w
    mid = np.array([1.0, 1.0])
    ax.annotate('', xy=mid + 0.7*w/np.linalg.norm(w),
                xytext=mid,
                arrowprops=dict(arrowstyle='->', color='darkred', lw=2))
    ax.text(mid[0]+0.75*w[0]/np.linalg.norm(w),
            mid[1]+0.75*w[1]/np.linalg.norm(w),
            r'$\mathbf{w}$', color='darkred', fontsize=11)

    ax.text(2.5, -0.5, r'$\mathbf{w}^\top\mathbf{x}>b$', fontsize=9)
    ax.text(-1.5, 1.5, r'$\mathbf{w}^\top\mathbf{x}<b$', fontsize=9,
            color='steelblue')

    ax.set_xlim(-2, 4); ax.set_ylim(-2, 4)
    ax.set_xlabel(r'$x_1$'); ax.set_ylabel(r'$x_2$')
    ax.set_title('Linear Constraint / Half-space', fontsize=10)
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.3)
    savefig('halfspace.pdf')


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3 – Feasible polytope (intersection of linear constraints) (Fig 12.3)
# ─────────────────────────────────────────────────────────────────────────────
def fig_feasible_polytope():
    """2D feasible region for a simple LP."""
    fig, ax = plt.subplots(figsize=(4, 3.5))
    # Constraints: x1>=0, x2>=0, x1+x2<=4, 2x1+x2<=6
    vertices = np.array([[0, 0], [3, 0], [2, 2], [0, 4]])
    hull = ConvexHull(vertices)
    poly = plt.Polygon(vertices[hull.vertices], closed=True,
                       facecolor='steelblue', alpha=0.35, edgecolor='navy', lw=1.5)
    ax.add_patch(poly)

    ax.text(1.2, 1.2, r'feasible set', fontsize=10, ha='center', va='center')

    # constraint lines
    x1 = np.linspace(-0.5, 4, 200)
    ax.plot(x1, 4 - x1, 'k--', lw=1.2, alpha=0.6, label=r'$x_1+x_2\leq 4$')
    ax.plot(x1, 6 - 2*x1, 'r--', lw=1.2, alpha=0.6, label=r'$2x_1+x_2\leq 6$')
    ax.axhline(0, color='gray', lw=1, alpha=0.5)
    ax.axvline(0, color='gray', lw=1, alpha=0.5)

    # mark vertices
    for v in vertices:
        ax.plot(v[0], v[1], 'ko', ms=5)

    ax.set_xlim(-0.5, 4.5); ax.set_ylim(-0.5, 5)
    ax.set_xlabel(r'$x_1$'); ax.set_ylabel(r'$x_2$')
    ax.set_title('Feasible Polytope', fontsize=10)
    ax.legend(fontsize=7, loc='upper right')
    ax.grid(True, alpha=0.3)
    savefig('feasible_polytope.pdf')


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4 – Four LP outcome cases (Fig 12.4 in book)
# ─────────────────────────────────────────────────────────────────────────────
def fig_lp_cases():
    fig, axes = plt.subplots(1, 4, figsize=(12, 3))
    titles = ['One Solution', 'Unbounded', 'Infinite Solutions', 'No Solution']
    c_dir = np.array([-1.0, -0.6])   # -c direction (descent)

    # define four polygon shapes
    polys = [
        np.array([[0,0],[3,0],[3,2],[1,3],[0,2]]),   # bounded, unique opt at vertex
        np.array([[0,0],[3,0],[3,3],[0,3]]),           # open top-left → unbounded
        np.array([[0,0],[3,0],[2,2],[0,2]]),           # face perpendicular to c → infinite
        np.array([[0,3],[3,3],[3,5],[0,5]]),           # disjoint: no feasible → empty
    ]
    opt_points = [
        np.array([3.0, 0.0]),   # one solution
        None,                    # unbounded
        np.array([1.5, 1.0]),   # somewhere on edge
        None,                    # no solution
    ]

    for ax, poly, title, opt in zip(axes, polys, titles, opt_points):
        hull = ConvexHull(poly)
        p_patch = plt.Polygon(poly[hull.vertices], closed=True,
                              facecolor='steelblue', alpha=0.35,
                              edgecolor='navy', lw=1.5)
        ax.add_patch(p_patch)

        # -c arrow
        cx, cy = 1.5, 1.5
        ax.annotate('', xy=(cx + c_dir[0]*0.7, cy + c_dir[1]*0.7),
                    xytext=(cx, cy),
                    arrowprops=dict(arrowstyle='->', color='darkred', lw=2))
        ax.text(cx + c_dir[0]*0.8, cy + c_dir[1]*0.8,
                r'$-\mathbf{c}$', color='darkred', fontsize=9)

        if opt is not None:
            ax.plot(*opt, 'r*', ms=12, label=r'$\mathbf{x}^*$', zorder=5)
            ax.legend(fontsize=8, loc='upper right')

        ax.set_xlim(-0.5, 4); ax.set_ylim(-0.5, 5.5)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel(r'$x_1$', fontsize=8); ax.set_ylabel(r'$x_2$', fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    savefig('lp_cases.pdf')


# ─────────────────────────────────────────────────────────────────────────────
# Figure 5 – Simplex vertex traversal (Example 12.7)
# ─────────────────────────────────────────────────────────────────────────────
def fig_simplex_traversal():
    """Show 2D LP feasible region and simplex path (Example 12.7)."""
    # LP:  min  3x1 - x2   s.t.  x1 + x2 <= 9, -4x1 + 2x2 <= 2, x1,x2 >= 0
    # Vertices: (0,0), (9,0), (2,5), (0,1)   [from A,b in Example 12.7 equality form]
    # Equality form with slack: x1+x2+s1=9, -4x1+2x2+s2=2
    # Vertices of the feasible polytope (x1,x2):
    verts = np.array([[0, 0], [9, 0], [8, 1], [0, 1]])
    # Actually, let's verify:
    # x1+x2=9, -4x1+2x2=2 => x1=8/3, x2=19/3  -- not integer nice
    # Let me just draw the region directly from the constraints
    from scipy.optimize import linprog
    # Use the actual Example 12.7 data:
    # A = [[1,1,1,0],[-4,2,0,1]], b=[9,2], c=[3,-1,0,0]
    # B={3,4} initially -> x1=x2=0, s1=9, s2=2
    # After pivot q=2: B={2,3} -> x1=0, x2=1, s1=8, s2=0  [x*=(0,1,8,0)]
    # Then x* = [0,1,8,0]

    fig, ax = plt.subplots(figsize=(5, 4))
    # feasible region in (x1,x2) space:
    # x1+x2 <= 9, -4x1+2x2 <= 2, x1>=0, x2>=0
    # corners: (0,0),(9,0),(2.5,6.5),(0,1)
    # check (2.5,6.5): 2.5+6.5=9 ok; -4*2.5+2*6.5=-10+13=3 > 2  NO
    # Let me recompute corners carefully
    # Constraints as <= :  x1+x2<=9, -4x1+2x2<=2 => x2 <= (2+4x1)/2 = 1+2x1
    #                      x1>=0, x2>=0
    # Vertices:
    # (0,0): both tight (x1=0,x2=0) - ok
    # (9,0): x1+x2=9, x2=0 => x1=9.  Check -4*9+0=-36<=2 ok
    # intersection x1+x2=9 and -4x1+2x2=2:
    #   x2=9-x1 => -4x1+2(9-x1)=2 => -6x1+18=2 => x1=8/3, x2=19/3
    # (0,1): x1=0, -4*0+2x2=2 => x2=1. Check 0+1<=9 ok
    corners = np.array([[0,0],[9,0],[8/3,19/3],[0,1]])

    hull = ConvexHull(corners)
    poly = plt.Polygon(corners[hull.vertices], closed=True,
                       facecolor='steelblue', alpha=0.3,
                       edgecolor='navy', lw=1.8)
    ax.add_patch(poly)

    # mark vertices
    vertex_labels = {
        (0,0): r'$\mathcal{B}=\{3,4\}$',
        (9,0): r'$\mathcal{B}=\{1,3\}$',
        (8/3,19/3): r'$\mathcal{B}=\{1,2\}$',
        (0,1): r'$\mathcal{B}=\{2,3\}$  $\mathbf{x}^*$',
    }
    for (x1,x2), label in vertex_labels.items():
        ax.plot(x1, x2, 'ko', ms=7)
        ax.text(x1+0.15, x2+0.1, label, fontsize=8)

    # simplex path: B={3,4} -> B={2,3}
    path = np.array([[0,0],[0,1]])
    ax.plot(path[:,0], path[:,1], 'r-o', lw=2.5, ms=8, label='Simplex path',
            zorder=5)
    ax.annotate('', xy=path[1], xytext=path[0],
                arrowprops=dict(arrowstyle='->', color='red', lw=2))

    # objective direction -c = (-3, +1) normalized
    cx, cy = 4.0, 2.0
    cv = np.array([-3.0, 1.0]); cv = cv / np.linalg.norm(cv)
    ax.annotate('', xy=(cx+cv[0]*1.2, cy+cv[1]*1.2), xytext=(cx,cy),
                arrowprops=dict(arrowstyle='->', color='darkred', lw=2))
    ax.text(cx+cv[0]*1.4, cy+cv[1]*1.4, r'$-\mathbf{c}$',
            color='darkred', fontsize=11)

    ax.set_xlim(-0.5, 10); ax.set_ylim(-0.5, 7.5)
    ax.set_xlabel(r'$x_1$'); ax.set_ylabel(r'$x_2$')
    ax.set_title('Simplex Algorithm: Vertex Traversal\n(Example 12.7)', fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    savefig('simplex_traversal.pdf')


# ─────────────────────────────────────────────────────────────────────────────
# Figure 6 – Example 12.4 (converting standard→equality form)
# ─────────────────────────────────────────────────────────────────────────────
def fig_example124():
    """Feasible region for Example 12.4:
       maximize 5x1+4x2  s.t. 2x1+3x2<=5, 4x1+x2<=11, x1,x2 free (book shows free vars)
       We show as standard form after conversion (x1,x2 free, use x+,x- split shown symbolically)
       For simplicity, show original feasible region with optimal marked.
    """
    fig, ax = plt.subplots(figsize=(4, 4))

    # Feasible region for maximize 5x1+4x2 s.t. 2x1+3x2<=5, 4x1+x2<=11
    # No non-negativity constraints in original (variables are free)
    # But after conversion x+ >= 0. Show the (x1,x2) view
    x1 = np.linspace(-3, 4, 400)

    # constraint 1: 2x1+3x2<=5 => x2 <= (5-2x1)/3
    x2_c1 = (5 - 2*x1) / 3
    # constraint 2: 4x1+x2<=11 => x2 <= 11-4x1
    x2_c2 = 11 - 4*x1

    x2_feas = np.minimum(x2_c1, x2_c2)
    # feasible from below: no lower bound (free variables)
    # shade bounded region loosely
    ax.fill_between(x1, -4, x2_feas, where=(x2_feas > -4),
                    alpha=0.25, color='steelblue', label='Feasible region')
    ax.plot(x1, x2_c1, 'b-', lw=1.5, label=r'$2x_1+3x_2=5$')
    ax.plot(x1, x2_c2, 'r-', lw=1.5, label=r'$4x_1+x_2=11$')

    # optimal: maximize along -c direction; c=[5,4] for max -> min [-5,-4]
    # intersection of both constraints active: 2x1+3x2=5, 4x1+x2=11
    A = np.array([[2,3],[4,1]]); b = np.array([5,11])
    xopt = np.linalg.solve(A, b)
    ax.plot(*xopt, 'r*', ms=14, label=r'$\mathbf{x}^*=(2.5, 0)$... ', zorder=5)
    ax.text(xopt[0]+0.1, xopt[1]+0.2, f'$x^*=({xopt[0]:.2f},{xopt[1]:.2f})$',
            fontsize=8)

    ax.set_xlim(-1, 4); ax.set_ylim(-1, 3)
    ax.set_xlabel(r'$x_1$'); ax.set_ylabel(r'$x_2$')
    ax.set_title('Example 12.4: LP Feasible Region\n(maximize $5x_1+4x_2$)', fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    savefig('example124.pdf')


# ─────────────────────────────────────────────────────────────────────────────
# Figure 7 – Python simplex demo convergence (objective vs iteration)
# ─────────────────────────────────────────────────────────────────────────────
def fig_simplex_convergence():
    """Run a simple LP via scipy and show dual mu_V values and objective decrease."""
    from scipy.optimize import linprog

    # Example 12.7 LP: min 3x1 - x2  s.t.  x1+x2<=9, -4x1+2x2<=2, x>=0
    c = np.array([3.0, -1.0])
    A_ub = np.array([[1, 1], [-4, 2]])
    b_ub = np.array([9.0, 2.0])
    bounds = [(0, None), (0, None)]
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')

    # Manual simplex trace for illustration (equality form)
    # A = [[1,1,1,0],[-4,2,0,1]], b=[9,2], c=[3,-1,0,0]
    A = np.array([[1.,1.,1.,0.],[-4.,2.,0.,1.]])
    b = np.array([9.,2.])
    c_eq = np.array([3.,-1.,0.,0.])

    iterations = []

    def get_vertex(B, A, b):
        b_inds = sorted(B)
        AB = A[:, b_inds]
        xB = np.linalg.solve(AB, b)
        x = np.zeros(A.shape[1])
        x[b_inds] = xB
        return x

    def step_lp(B, A, b, c):
        n = A.shape[1]
        b_inds = sorted(B)
        n_inds = sorted(set(range(n)) - set(b_inds))
        AB = A[:, b_inds]
        AV = A[:, n_inds]
        xB = np.linalg.solve(AB, b)
        cB = c[b_inds]
        cV = c[n_inds]
        lam = np.linalg.solve(AB.T, cB)
        muV = cV - AV.T @ lam
        obj = c @ get_vertex(B, A, b)
        iterations.append({'B': list(B), 'obj': obj, 'muV': muV.copy()})

        if np.all(muV >= 0):
            return B, True  # optimal

        # greedy: pick most negative
        q_local = int(np.argmin(muV))
        q = n_inds[q_local]

        # edge transition: find leaving index
        d = np.linalg.solve(AB, A[:, q])
        ratios = []
        for i, (xb, di) in enumerate(zip(xB, d)):
            if di > 0:
                ratios.append((xb / di, b_inds[i]))
            else:
                ratios.append((np.inf, b_inds[i]))
        _, p = min(ratios)

        B_new = (set(B) - {p}) | {q}
        return B_new, False

    B = {2, 3}   # 0-indexed: columns 2,3 are slack s1,s2
    done = False
    max_iter = 20
    it = 0
    while not done and it < max_iter:
        B, done = step_lp(B, A, b, c_eq)
        it += 1

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))

    # Objective vs iteration
    objs = [r['obj'] for r in iterations]
    axes[0].plot(range(len(objs)), objs, 'bo-', ms=8, lw=2)
    axes[0].set_xlabel('Iteration')
    axes[0].set_ylabel(r'$\mathbf{c}^\top\mathbf{x}$')
    axes[0].set_title('Objective Value per Simplex Iteration')
    axes[0].grid(True, alpha=0.4)
    for i, (it_i, ob) in enumerate(zip(iterations, objs)):
        axes[0].annotate(f"$\\mathcal{{B}}=\\{{{','.join(str(j+1) for j in sorted(it_i['B']))}\\}}$",
                         xy=(i, ob), xytext=(i+0.05, ob+0.3),
                         fontsize=8, color='navy')

    # muV per iteration
    if len(iterations) >= 1:
        muV0 = iterations[0]['muV']
        axes[1].bar(range(len(muV0)), muV0, color=['red' if v < 0 else 'steelblue'
                                                    for v in muV0])
        axes[1].axhline(0, color='k', lw=1)
        axes[1].set_xlabel(r'Non-basic index $q$')
        axes[1].set_ylabel(r'$\mu_\mathcal{V}$')
        axes[1].set_title(r'$\mu_\mathcal{V}$ at Initial Vertex (neg $\Rightarrow$ suboptimal)')
        axes[1].grid(True, alpha=0.4)

    plt.tight_layout()
    savefig('simplex_convergence.pdf')


# ─────────────────────────────────────────────────────────────────────────────
# Figure 8 – Primal-dual relationship
# ─────────────────────────────────────────────────────────────────────────────
def fig_primal_dual():
    """Illustrate weak duality: dual objective <= primal objective."""
    fig, ax = plt.subplots(figsize=(5, 3.5))

    # Conceptual plot: primal feasible region, dual bound
    primal_vals = [5, 3, 2, 1.5, 1.2, 1.0]
    dual_vals   = [-1, 0.3, 0.7, 0.9, 1.0, 1.0]
    iters = list(range(len(primal_vals)))

    ax.plot(iters, primal_vals, 'b-o', lw=2, ms=8, label='Primal objective $p$')
    ax.plot(iters, dual_vals,  'r--s', lw=2, ms=8, label='Dual objective $d$')
    ax.fill_between(iters, dual_vals, primal_vals, alpha=0.15, color='gray',
                    label='Duality gap')
    ax.axhline(1.0, color='green', lw=1.5, ls=':', label='Optimal value $p^*=d^*$')
    ax.text(len(iters)-0.9, 1.05, '$p^*=d^*=1$', color='green', fontsize=9)

    ax.set_xlabel('Iteration / step')
    ax.set_ylabel('Objective value')
    ax.set_title('Strong Duality: Primal vs. Dual Objective', fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    savefig('primal_dual.pdf')


# ─────────────────────────────────────────────────────────────────────────────
# Figure 9 – Crop Fig 12.4 (four cases) from book PDF using PyMuPDF
# ─────────────────────────────────────────────────────────────────────────────
def crop_book_figure(page_index, clip_rect, output_name, dpi=150):
    """Crop a rectangle from a PDF page and save as PDF via PNG."""
    try:
        import fitz
        doc = fitz.open('/Users/vinhdq1/work/Lectures-on-Monte-Carlo-Theory/'
                        'Algorithms_for_optimization_2e_2026/optimization_book.pdf')
        page = doc[page_index]
        mat = fitz.Matrix(dpi/72, dpi/72)
        # clip_rect in PDF points (x0,y0,x1,y1)
        clip = fitz.Rect(*clip_rect)
        pix = page.get_pixmap(matrix=mat, clip=clip)
        png_path = os.path.join(FIGURES_DIR, output_name.replace('.pdf', '.png'))
        pix.save(png_path)
        # Convert PNG to single-page PDF for inclusion in LaTeX
        from PIL import Image
        img = Image.open(png_path)
        pdf_path = os.path.join(FIGURES_DIR, output_name)
        # Save as PDF
        img_rgb = img.convert('RGB')
        img_rgb.save(pdf_path, 'PDF', resolution=dpi)
        print(f'  Cropped PDF page {page_index+1} -> {pdf_path}')
        return True
    except Exception as e:
        print(f'  WARNING: Could not crop book figure: {e}')
        return False


def fig_book_lp_cases():
    """Try to crop Fig 12.4 from book; fall back to matplotlib version."""
    # Page 253 (index 272) contains the four-case figure
    # The figure spans roughly the top portion
    success = crop_book_figure(
        page_index=272,           # book page 253
        clip_rect=(30, 30, 590, 220),   # approximate PDF-point coordinates
        output_name='book_lp_cases.pdf'
    )
    if not success:
        # Just use our matplotlib version
        print('  Using matplotlib fallback for lp_cases')


# ─────────────────────────────────────────────────────────────────────────────
# Run all figure generators
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print('Generating Chapter 12 figures...')
    fig_linear_contours()
    fig_halfspace()
    fig_feasible_polytope()
    fig_lp_cases()
    fig_simplex_traversal()
    fig_example124()
    fig_simplex_convergence()
    fig_primal_dual()
    fig_book_lp_cases()
    print('All figures generated successfully.')
