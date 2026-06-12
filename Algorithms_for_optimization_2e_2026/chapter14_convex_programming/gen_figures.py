"""
gen_figures.py  –  Generate all figures for Chapter 14: Disciplined Convex Programming
Run with: conda run -n py313 python3 gen_figures.py
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os
import fitz  # pymupdf

FIGURES_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

BOOK_PDF = os.path.join(os.path.dirname(__file__), "..", "optimization_book.pdf")

# ─────────────────────────────────────────────────────────────────────────────
# Helper: save figure
# ─────────────────────────────────────────────────────────────────────────────
def savefig(name):
    path = os.path.join(FIGURES_DIR, name)
    plt.savefig(path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"  saved {path}")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 1: Canonical DCP problem structure diagram
# ─────────────────────────────────────────────────────────────────────────────
def fig_dcp_canonical():
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis('off')

    # Title box
    title_box = mpatches.FancyBboxPatch((0.3, 4.5), 9.4, 1.2,
        boxstyle="round,pad=0.15", facecolor='#dce9f5', edgecolor='#2c6aad', lw=2)
    ax.add_patch(title_box)
    ax.text(5.0, 5.1, r'$\underset{\mathbf{x}}{\mathrm{minimize}}\ f_0(\mathbf{x})$   '
            r'subject to   $f_i(\mathbf{x}) \leq 0,\ i=1,\ldots,m$   '
            r'$h_j(\mathbf{x}) = 0,\ j=1,\ldots,p$',
            ha='center', va='center', fontsize=11)

    # Three condition boxes
    colors = ['#e8f4e8', '#fff3cd', '#fde8e8']
    edge_colors = ['#3a7d3a', '#c4900a', '#c0392b']
    labels = [
        r'$f_0,\, f_i$ are convex' + '\n(atoms from library)',
        r'$h_j$ are affine',
        'Expressions are\nproduct-free'
    ]
    xs = [0.5, 3.7, 6.9]
    for x, col, ecol, lbl in zip(xs, colors, edge_colors, labels):
        box = mpatches.FancyBboxPatch((x, 1.5), 2.8, 2.5,
            boxstyle="round,pad=0.1", facecolor=col, edgecolor=ecol, lw=1.5)
        ax.add_patch(box)
        ax.text(x + 1.4, 2.75, lbl, ha='center', va='center', fontsize=9.5)

    # Arrows from title to boxes
    for x in [1.9, 5.1, 8.3]:
        ax.annotate('', xy=(x, 3.95), xytext=(x, 4.55),
                    arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))

    ax.text(5.0, 0.7, 'DCP Requirements', ha='center', fontsize=10,
            color='#555555', style='italic')
    plt.title('Disciplined Convex Programming — Canonical Form', fontsize=12, pad=8)
    savefig('fig_dcp_canonical.pdf')


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2: Atom library table (matplotlib table)
# ─────────────────────────────────────────────────────────────────────────────
def fig_atom_library():
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.axis('off')

    col_labels = ['Function', 'Domain', 'Curvature', 'Monotonicity', 'Range']
    data = [
        [r'$|x|$',              r'$\mathbb{R}$',        'convex',  'nonmonotone',      r'$[0,\infty)$'],
        [r'$x^2$',              r'$\mathbb{R}$',        'convex',  'nonmonotone',      r'$[0,\infty)$'],
        [r'$\max(x_1,x_2)$',   r'$\mathbb{R}^2$',      'convex',  'nondecreasing',    r'$\mathbb{R}$'],
        [r'$\exp(x)$',         r'$\mathbb{R}$',        'convex',  'nondecreasing',    r'$(0,\infty)$'],
        [r'$\log(x)$',         r'$(0,\infty)$',        'concave', 'nondecreasing',    r'$\mathbb{R}$'],
        [r'$\sqrt{x}$',        r'$[0,\infty)$',        'concave', 'nondecreasing',    r'$[0,\infty)$'],
        [r'$\min(x_1,x_2)$',   r'$\mathbb{R}^2$',      'concave', 'nondecreasing',    r'$\mathbb{R}$'],
        [r'$\|\mathbf{x}\|_p$',r'$\mathbb{R}^n$',      'convex',  'nonmonotone',      r'$[0,\infty)$'],
    ]

    table = ax.table(cellText=data, colLabels=col_labels,
                     loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.7)

    # Style header
    for j in range(len(col_labels)):
        table[(0, j)].set_facecolor('#2c6aad')
        table[(0, j)].set_text_props(color='white', fontweight='bold')
    # Alternating row colors
    for i in range(1, len(data) + 1):
        color = '#f0f5fb' if i % 2 == 0 else 'white'
        for j in range(len(col_labels)):
            table[(i, j)].set_facecolor(color)

    plt.title('Atom Library — Common Functions', fontsize=12, pad=14)
    savefig('fig_atom_library.pdf')


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3: Epigraph and hypograph illustration (like Fig 14.3 in book)
# ─────────────────────────────────────────────────────────────────────────────
def fig_epi_hypo():
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))

    x = np.linspace(0, 2, 400)

    # Left: epigraph of x^3 (convex for x>=0)
    ax = axes[0]
    y_epi = x**3
    ax.fill_between(x, y_epi, 6, alpha=0.45, color='steelblue', label=r'$\mathrm{epi}\,f$')
    ax.plot(x, y_epi, 'b-', lw=2.5, label=r'$f(x)=x^3$')
    ax.set_xlim(0, 2); ax.set_ylim(0, 6)
    ax.set_xlabel('$x$', fontsize=12)
    ax.set_ylabel('$y$', fontsize=12)
    ax.set_title(r'Epigraph of $f(x)=x^3$', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Right: hypograph of log(x)
    ax = axes[1]
    x2 = np.linspace(0.05, 2, 400)
    y_hypo = np.log(x2)
    y_floor = -3.0
    ax.fill_between(x2, y_floor, y_hypo, alpha=0.45, color='salmon', label=r'$\mathrm{hypo}\,f$')
    ax.plot(x2, y_hypo, 'r-', lw=2.5, label=r'$f(x)=\log x$')
    ax.set_xlim(0, 2); ax.set_ylim(-3, 1)
    ax.set_xlabel('$x$', fontsize=12)
    ax.set_ylabel('$y$', fontsize=12)
    ax.set_title(r'Hypograph of $f(x)=\log x$', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    savefig('fig_epi_hypo.pdf')


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4: Composition rules illustration — max(eps(x1), x2) heat maps
# ─────────────────────────────────────────────────────────────────────────────
def fig_composition_max():
    x1 = np.linspace(-2, 2, 200)
    x2 = np.linspace(-2, 2, 200)
    X1, X2 = np.meshgrid(x1, x2)

    cases = [
        (r'$\max(x_1, x_2)$',   lambda a, b: np.maximum(a, b),              lambda a: a),
        (r'$\max(\sin(x_1),x_2)$', lambda a, b: np.maximum(np.sin(a), b),    lambda a: np.sin(a)),
        (r'$\max(x_1^2-3, x_2)$', lambda a, b: np.maximum(a**2 - 3, b),      lambda a: a**2 - 3),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(11, 6))

    for col, (title, Z_fn, eps_fn) in enumerate(cases):
        Z = Z_fn(X1, X2)
        # Top: heat map
        ax = axes[0, col]
        im = ax.pcolormesh(X1, X2, Z, cmap='viridis', shading='auto')
        ax.set_title(title, fontsize=9)
        ax.set_xlabel('$x_1$', fontsize=8); ax.set_ylabel('$x_2$', fontsize=8)
        plt.colorbar(im, ax=ax, shrink=0.7)

        # Bottom: 1-D slice x2=0, show eps(x1) in blue, max(...,0) in black
        ax2 = axes[1, col]
        x_line = np.linspace(-2, 2, 300)
        eps_line = eps_fn(x_line)
        max_line = np.maximum(eps_line, 0)
        ax2.plot(x_line, max_line, 'k-', lw=2, label=r'$\max(\epsilon,0)$')
        ax2.plot(x_line, eps_line, 'b-', lw=1.5, alpha=0.7, label=r'$\epsilon(x_1)$')
        ax2.set_xlabel('$x_1$', fontsize=8)
        ax2.legend(fontsize=7)
        ax2.grid(True, alpha=0.3)

    fig.suptitle(r'Composition Rules: $\max(\epsilon(x_1), x_2)$ for Various $\epsilon$',
                 fontsize=11)
    plt.tight_layout()
    savefig('fig_composition_max.pdf')


# ─────────────────────────────────────────────────────────────────────────────
# Figure 5: Expression tree for verification (Fig 14.2 in book)
# ─────────────────────────────────────────────────────────────────────────────
def fig_expression_tree():
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_xlim(0, 10); ax.set_ylim(0, 10)
    ax.axis('off')

    nodes = {
        'root':   (5.0, 9.2, r'$3+2x_1+4x_2+\max(\exp(x_1-\log x_2)+\mathrm{pow}(x_2,2),\,x_1-x_2)$', '#dce9f5'),
        'sum':    (5.0, 7.5, r'$3+2x_1+4x_2+\max(\varepsilon,\varepsilon)$', '#e0f0e0'),
        'arg1':   (2.5, 5.8, r'$x_1 - x_2$', '#fff3cd'),
        'arg2':   (7.5, 5.8, r'$\exp(\varepsilon)+\mathrm{pow}(\varepsilon,\varepsilon)$', '#e0f0e0'),
        'c2_a':   (6.5, 4.1, r'$2$', '#fde8e8'),
        'c2_b':   (8.5, 4.1, r'$x_2$', '#fde8e8'),
        'inner':  (7.0, 2.5, r'$x_1 - \log(\varepsilon)$', '#e0f0e0'),
        'log_arg':(7.0, 0.9, r'$x_2$', '#fff3cd'),
    }

    for key, (x, y, label, color) in nodes.items():
        box = mpatches.FancyBboxPatch((x - 2.2, y - 0.45), 4.4, 0.9,
            boxstyle="round,pad=0.08", facecolor=color, edgecolor='#555', lw=1.2)
        ax.add_patch(box)
        ax.text(x, y, label, ha='center', va='center', fontsize=8.5)

    edges = [
        ('root', 'sum'), ('sum', 'arg1'), ('sum', 'arg2'),
        ('arg2', 'c2_a'), ('arg2', 'c2_b'), ('arg2', 'inner'),
        ('inner', 'log_arg'),
    ]
    for src, dst in edges:
        xs, ys = nodes[src][0], nodes[src][1] - 0.45
        xd, yd = nodes[dst][0], nodes[dst][1] + 0.45
        ax.annotate('', xy=(xd, yd), xytext=(xs, ys),
                    arrowprops=dict(arrowstyle='->', color='#444', lw=1.3))

    plt.title('Expression Tree for DCP Verification (Fig 14.2)', fontsize=11)
    savefig('fig_expression_tree.pdf')


# ─────────────────────────────────────────────────────────────────────────────
# Figure 6: Linearization process diagram
# ─────────────────────────────────────────────────────────────────────────────
def fig_linearization():
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.set_xlim(0, 10); ax.set_ylim(0, 5)
    ax.axis('off')

    # Original problem
    orig = mpatches.FancyBboxPatch((0.2, 1.5), 3.5, 2.0,
        boxstyle="round,pad=0.15", facecolor='#fff3cd', edgecolor='#c4900a', lw=2)
    ax.add_patch(orig)
    ax.text(1.95, 3.0, 'Original DCP', ha='center', fontsize=10, fontweight='bold')
    ax.text(1.95, 2.5, r'$\min\ f(\mathbf{x})$', ha='center', fontsize=10)
    ax.text(1.95, 2.0, r's.t. $g_i(\mathbf{x})\leq 0$', ha='center', fontsize=10)

    # Arrow
    ax.annotate('', xy=(4.5, 2.5), xytext=(3.7, 2.5),
                arrowprops=dict(arrowstyle='->', color='#2c6aad', lw=2.5))
    ax.text(4.1, 2.9, 'Linearize\n(introduce\natom vars)', ha='center', fontsize=8.5, color='#2c6aad')

    # Linearized problem
    lin = mpatches.FancyBboxPatch((4.5, 0.5), 5.2, 4.0,
        boxstyle="round,pad=0.15", facecolor='#e8f4e8', edgecolor='#3a7d3a', lw=2)
    ax.add_patch(lin)
    ax.text(7.1, 4.1, 'Linearized (Canonical) Form', ha='center', fontsize=10, fontweight='bold')
    lines = [
        r'$\min_{{\mathbf{x},\mathbf{v}}}\ d + \sum_i (\mathbf{c}^{(i)})^\top \mathbf{x}^{(i)}$',
        r's.t. $\mathbf{x}^{(i)} \in S^{(i)}$',
        r'$\sum_i \mathbf{A}^{(i)} \mathbf{x}^{(i)} = \mathbf{b}$',
        r'$f_j(\varepsilon_j) \leq v_j$ (convex atoms)',
        r'$f_j(\varepsilon_j) \geq v_j$ (concave atoms)',
    ]
    for k, ln in enumerate(lines):
        ax.text(7.1, 3.6 - k * 0.65, ln, ha='center', va='center', fontsize=9)

    plt.title('Canonicalization: Linearization Step', fontsize=12)
    savefig('fig_linearization.pdf')


# ─────────────────────────────────────────────────────────────────────────────
# Figure 7: Graph implementation of absolute value
# ─────────────────────────────────────────────────────────────────────────────
def fig_abs_graph():
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))

    x = np.linspace(-3, 3, 400)

    ax = axes[0]
    ax.plot(x, np.abs(x), 'b-', lw=2.5, label=r'$|x|$')
    ax.set_title(r'Absolute Value $|x|$', fontsize=11)
    ax.set_xlabel('$x$'); ax.set_ylabel('$y$')
    ax.grid(True, alpha=0.3); ax.legend()
    ax.axvline(0, color='gray', lw=0.8, linestyle='--')

    # LP reformulation illustration
    ax2 = axes[1]
    y_vals = np.linspace(0, 3, 200)
    # feasible region: y >= x, y >= -x  → y >= |x|
    ax2.fill_between(x, np.abs(x), 4, alpha=0.3, color='steelblue',
                     label='Feasible region\n($y\\geq x$ and $y\\geq -x$)')
    ax2.plot(x, np.abs(x), 'b-', lw=2.5)
    ax2.set_xlim(-3, 3); ax2.set_ylim(0, 4)
    ax2.set_title(r'LP Reformulation of $|x|$', fontsize=11)
    ax2.set_xlabel('$x$'); ax2.set_ylabel('$y$')
    ax2.text(0, 0.2, r'$|x| = \min_y y\ $ s.t. $y\geq x,\ y\geq -x$',
             ha='center', fontsize=8.5, color='#333')
    ax2.grid(True, alpha=0.3); ax2.legend(fontsize=8)

    plt.tight_layout()
    savefig('fig_abs_graph.pdf')


# ─────────────────────────────────────────────────────────────────────────────
# Figure 8: Norm ball conversions (L1, L2, Linf)
# ─────────────────────────────────────────────────────────────────────────────
def fig_norm_balls():
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))
    theta = np.linspace(0, 2 * np.pi, 500)
    titles = [r'$L_1$ ball $\|\mathbf{x}\|_1\leq 1$',
              r'$L_2$ ball $\|\mathbf{x}\|_2\leq 1$',
              r'$L_\infty$ ball $\|\mathbf{x}\|_\infty\leq 1$']

    # L1 ball (diamond)
    t = np.linspace(-1, 1, 200)
    axes[0].fill([ 1,  0, -1,  0,  1],
                 [ 0,  1,  0, -1,  0], alpha=0.4, color='steelblue')
    axes[0].plot([ 1,  0, -1,  0,  1],
                 [ 0,  1,  0, -1,  0], 'b-', lw=2)

    # L2 ball (circle)
    axes[1].fill(np.cos(theta), np.sin(theta), alpha=0.4, color='green')
    axes[1].plot(np.cos(theta), np.sin(theta), 'g-', lw=2)

    # Linf ball (square)
    axes[2].fill([ 1,  1, -1, -1,  1],
                 [ 1, -1, -1,  1,  1], alpha=0.4, color='salmon')
    axes[2].plot([ 1,  1, -1, -1,  1],
                 [ 1, -1, -1,  1,  1], 'r-', lw=2)

    for ax, title in zip(axes, titles):
        ax.set_title(title, fontsize=10)
        ax.set_aspect('equal')
        ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.5, 1.5)
        ax.axhline(0, color='gray', lw=0.5); ax.axvline(0, color='gray', lw=0.5)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('$x_1$'); ax.set_ylabel('$x_2$')

    plt.suptitle('Norm Balls in 2D', fontsize=12)
    plt.tight_layout()
    savefig('fig_norm_balls.pdf')


# ─────────────────────────────────────────────────────────────────────────────
# Figure 9: CVXPY workflow diagram
# ─────────────────────────────────────────────────────────────────────────────
def fig_cvxpy_workflow():
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.set_xlim(0, 10); ax.set_ylim(0, 4)
    ax.axis('off')

    steps = [
        ('Define\nVariables\n(cp.Variable)', 1.0),
        ('Build\nObjective\n& Constraints', 3.0),
        ('Create &\nVerify Problem\n(cp.Problem)', 5.0),
        ('Solve\n(problem.solve())', 7.0),
        ('Extract\nResults\n(.value)', 9.0),
    ]
    colors_steps = ['#dce9f5', '#e8f4e8', '#fff3cd', '#fde8e8', '#f0e8fd']
    edge_c = ['#2c6aad', '#3a7d3a', '#c4900a', '#c0392b', '#7d3a7a']

    for (lbl, x), col, ecol in zip(steps, colors_steps, edge_c):
        box = mpatches.FancyBboxPatch((x - 0.8, 0.8), 1.6, 2.4,
            boxstyle="round,pad=0.12", facecolor=col, edgecolor=ecol, lw=1.8)
        ax.add_patch(box)
        ax.text(x, 2.0, lbl, ha='center', va='center', fontsize=8.5)

    # Arrows between steps
    for i in range(len(steps) - 1):
        x1 = steps[i][1] + 0.8
        x2 = steps[i + 1][1] - 0.8
        ax.annotate('', xy=(x2, 2.0), xytext=(x1, 2.0),
                    arrowprops=dict(arrowstyle='->', color='#555', lw=1.8))

    # DCP verification annotation
    ax.annotate('DCP check', xy=(5.0, 0.8), xytext=(5.0, 0.3),
                ha='center', fontsize=8, color='#c4900a',
                arrowprops=dict(arrowstyle='->', color='#c4900a', lw=1.2))

    plt.title('CVXPY Workflow for Disciplined Convex Programming', fontsize=11, pad=8)
    savefig('fig_cvxpy_workflow.pdf')


# ─────────────────────────────────────────────────────────────────────────────
# Figure 10: Crop Fig 14.3 (epigraph/hypograph) from book PDF
# ─────────────────────────────────────────────────────────────────────────────
def crop_book_figure_14_3():
    """Crop Figure 14.3 from book PDF page 313 (0-indexed: page 312)."""
    try:
        doc = fitz.open(BOOK_PDF)
        # Book page 313 = PDF index 312
        page_idx = 312
        page = doc[page_idx]
        # Crop region (right side figure) — approximate coordinates
        # Page size is roughly 612 x 792 pts for letter
        rect = fitz.Rect(420, 470, 612, 650)
        mat = fitz.Matrix(2.0, 2.0)
        clip = page.get_pixmap(matrix=mat, clip=rect)
        out_path = os.path.join(FIGURES_DIR, 'fig_book_epi_hypo.png')
        clip.save(out_path)
        doc.close()
        print(f"  saved {out_path}")
    except Exception as e:
        print(f"  WARNING: could not crop book figure: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 11: Interior point method convergence on a simple convex QP
# ─────────────────────────────────────────────────────────────────────────────
def fig_interior_point():
    """Illustrate interior point trajectory on a 2D convex QP."""
    fig, ax = plt.subplots(figsize=(6, 5))

    # Constraint: x1^2 + x2^2 <= 4  (circle), x1 + x2 <= 2
    theta = np.linspace(0, 2 * np.pi, 400)
    ax.plot(2 * np.cos(theta), 2 * np.sin(theta), 'b-', lw=1.5, alpha=0.5,
            label=r'$x_1^2+x_2^2\leq 4$')

    x_fill = np.linspace(-2, 2, 400)
    ax.fill_between(x_fill, -np.sqrt(np.maximum(0, 4 - x_fill**2)),
                    np.minimum(2 - x_fill, np.sqrt(np.maximum(0, 4 - x_fill**2))),
                    alpha=0.15, color='steelblue')

    # Objective: minimize (x1-1)^2 + (x2-1)^2
    # Optimal solution near the boundary
    opt_x = np.array([1.0, 1.0]) / np.sqrt(2) * 2  # constrained optimum
    opt_x = np.array([np.sqrt(2), np.sqrt(2)])  # exact on circle

    # Simulated barrier method path (starting from center, moving to optimum)
    path = np.array([
        [0.0, 0.0],
        [0.3, 0.3],
        [0.6, 0.6],
        [0.9, 0.9],
        [1.1, 1.1],
        [1.3, 1.3],
        [np.sqrt(2) - 0.05, np.sqrt(2) - 0.05],
        [np.sqrt(2), np.sqrt(2)],
    ])
    ax.plot(path[:, 0], path[:, 1], 'ro-', lw=2, ms=6, label='Interior path')
    ax.plot(*path[-1], 'r*', ms=14, label='Optimum')

    # Contours of objective
    x_g = np.linspace(-2.5, 2.5, 200)
    X1g, X2g = np.meshgrid(x_g, x_g)
    Z = (X1g - 1.5)**2 + (X2g - 1.5)**2
    ax.contour(X1g, X2g, Z, levels=10, colors='gray', alpha=0.4, linewidths=0.8)

    ax.set_xlim(-2.5, 2.5); ax.set_ylim(-2.5, 2.5)
    ax.set_xlabel('$x_1$', fontsize=12); ax.set_ylabel('$x_2$', fontsize=12)
    ax.set_title('Interior Point Method on Convex QP', fontsize=11)
    ax.legend(fontsize=9)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    savefig('fig_interior_point.pdf')


# ─────────────────────────────────────────────────────────────────────────────
# Figure 12: DCP verification two-stage pipeline
# ─────────────────────────────────────────────────────────────────────────────
def fig_verification_pipeline():
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.set_xlim(0, 10); ax.set_ylim(0, 4)
    ax.axis('off')

    stages = [
        ('Input\nProblem', '#dce9f5', '#2c6aad', 1.0),
        ('Stage 1:\nProduct-free\nCheck', '#fff3cd', '#c4900a', 3.2),
        ('Stage 2:\nSign, Composition\n& Top-level Rules', '#e8f4e8', '#3a7d3a', 5.8),
        ('DCP\nVerified', '#fde8e8', '#c0392b', 8.3),
    ]

    for lbl, col, ecol, x in stages:
        box = mpatches.FancyBboxPatch((x - 0.85, 0.8), 1.7, 2.4,
            boxstyle="round,pad=0.12", facecolor=col, edgecolor=ecol, lw=1.8)
        ax.add_patch(box)
        ax.text(x, 2.0, lbl, ha='center', va='center', fontsize=9)

    for i in range(len(stages) - 1):
        x1 = stages[i][3] + 0.85
        x2 = stages[i + 1][3] - 0.85
        ax.annotate('', xy=(x2, 2.0), xytext=(x1, 2.0),
                    arrowprops=dict(arrowstyle='->', color='#555', lw=2))

    ax.text(3.2, 0.4, 'Build expression trees', ha='center', fontsize=8, color='#888')
    ax.text(5.8, 0.4, 'Apply curvature rules', ha='center', fontsize=8, color='#888')

    plt.title('Two-Stage Automatic DCP Verification', fontsize=12, pad=8)
    savefig('fig_verification_pipeline.pdf')


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("Generating figures for Chapter 14: Disciplined Convex Programming")
    fig_dcp_canonical()
    fig_atom_library()
    fig_epi_hypo()
    fig_composition_max()
    fig_expression_tree()
    fig_linearization()
    fig_abs_graph()
    fig_norm_balls()
    fig_cvxpy_workflow()
    crop_book_figure_14_3()
    fig_interior_point()
    fig_verification_pipeline()
    print("Done. All figures saved to:", FIGURES_DIR)
