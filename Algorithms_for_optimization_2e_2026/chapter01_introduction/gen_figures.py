"""
gen_figures.py  –  Generate all figures for Chapter 1: Introduction
Book: Algorithms for Optimization, 2nd ed. (2026), Kochenderfer & Wheeler
Saves all output to figures/ as PDF files.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import fitz  # pymupdf
import os

OUTDIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(OUTDIR, exist_ok=True)

BOOK_PDF = os.path.join(os.path.dirname(__file__), "..", "optimization_book.pdf")

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "figure.dpi": 150,
})


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1: Optimization Process (flowchart)
# ─────────────────────────────────────────────────────────────────────────────
def fig_optimization_process():
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3)
    ax.axis('off')

    boxes = [
        (0.6, 1.5, "Problem\nSpecification"),
        (2.8, 1.5, "Mathematical\nFormulation"),
        (5.0, 1.5, "Optimization\nAlgorithm"),
        (7.2, 1.5, "Evaluate /\nSimulate"),
        (9.0, 1.5, "Best\nDesign"),
    ]
    colors = ['#4e79a7', '#f28e2b', '#59a14f', '#e15759', '#76b7b2']

    for (x, y, label), c in zip(boxes, colors):
        ax.add_patch(mpatches.FancyBboxPatch(
            (x - 0.55, y - 0.5), 1.1, 1.0,
            boxstyle="round,pad=0.05", linewidth=1.5,
            edgecolor='#333333', facecolor=c, alpha=0.85,
            zorder=3))
        ax.text(x, y, label, ha='center', va='center',
                fontsize=9, fontweight='bold', color='white', zorder=4)

    # Arrows
    for i in range(len(boxes) - 1):
        x0 = boxes[i][0] + 0.55
        x1 = boxes[i+1][0] - 0.55
        y = 1.5
        ax.annotate("", xy=(x1, y), xytext=(x0, y),
                     arrowprops=dict(arrowstyle='->', lw=1.8, color='#333333'))

    # Feedback arrow from Evaluate back to Algorithm
    ax.annotate("", xy=(5.0, 0.8), xytext=(7.2, 0.8),
                 arrowprops=dict(arrowstyle='->', lw=1.5, color='gray',
                                 connectionstyle='arc3,rad=0'))
    ax.annotate("", xy=(5.0, 1.0), xytext=(5.0, 0.8),
                 arrowprops=dict(arrowstyle='->', lw=1.5, color='gray'))
    ax.text(6.1, 0.58, "iterate", ha='center', fontsize=9, color='gray',
            style='italic')

    ax.set_title("The Optimization Process", fontsize=13, fontweight='bold', pad=6)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "fig_opt_process.pdf"), bbox_inches='tight')
    plt.close()
    print("Saved fig_opt_process.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2: Feasible Set illustration (2D)
# ─────────────────────────────────────────────────────────────────────────────
def fig_feasible_set():
    fig, ax = plt.subplots(figsize=(5, 4.5))

    theta = np.linspace(0, 2*np.pi, 300)
    # Constraint region: ellipse
    ex = 1.2 * np.cos(theta)
    ey = 0.9 * np.sin(theta) + 0.3
    ax.fill(ex, ey, color='#aec6e8', alpha=0.6, label='Feasible set $\\mathcal{X}$')
    ax.plot(ex, ey, color='#4e79a7', lw=2)

    # Contours of f(x)
    x = np.linspace(-2, 2, 200)
    y = np.linspace(-1.5, 1.8, 200)
    X, Y = np.meshgrid(x, y)
    Z = (X - 0.8)**2 + 2*(Y - 1.1)**2
    cs = ax.contour(X, Y, Z, levels=8, cmap='YlOrRd', linewidths=1.2)

    # Global minimum marker (inside feasible set, constrained)
    ax.plot(0.6, 0.85, 'r*', markersize=14, label='Constrained optimum $\\mathbf{x}^*$', zorder=5)
    # Unconstrained minimum (outside)
    ax.plot(0.8, 1.1, 'bs', markersize=9, label='Unconstrained min', zorder=5)

    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_title('Feasible Set and Objective Contours', fontweight='bold')
    ax.legend(fontsize=9, loc='lower right')
    ax.set_xlim(-2, 2)
    ax.set_ylim(-1.5, 1.8)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "fig_feasible_set.pdf"), bbox_inches='tight')
    plt.close()
    print("Saved fig_feasible_set.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3: Local vs Global Minima (1D)
# ─────────────────────────────────────────────────────────────────────────────
def fig_local_global_minima():
    fig, ax = plt.subplots(figsize=(7, 3.8))
    x = np.linspace(-3.5, 4, 400)
    # A function with local and global minima
    y = 0.25*x**4 - x**3 - x**2 + 4*x + 3
    ax.plot(x, y, 'steelblue', lw=2.5, label='$f(x)$')

    # Find local/global minima numerically
    from scipy.signal import argrelmin
    idx_local = argrelmin(y, order=15)[0]
    for i, idx in enumerate(idx_local):
        xi, yi = x[idx], y[idx]
        if i == np.argmin(y[idx_local]):
            ax.plot(xi, yi, 'r*', markersize=14, zorder=5)
            ax.annotate('Global\nminimum', xy=(xi, yi),
                        xytext=(xi + 0.6, yi + 3),
                        fontsize=9, color='red',
                        arrowprops=dict(arrowstyle='->', color='red'))
        else:
            ax.plot(xi, yi, 'go', markersize=10, zorder=5)
            ax.annotate('Local\nminimum', xy=(xi, yi),
                        xytext=(xi - 2.2, yi + 2.5),
                        fontsize=9, color='green',
                        arrowprops=dict(arrowstyle='->', color='green'))

    ax.set_xlabel('$x$', fontsize=12)
    ax.set_ylabel('$f(x)$', fontsize=12)
    ax.set_title('Local vs. Global Minima', fontweight='bold')
    ax.set_ylim(-8, 20)
    ax.axhline(0, color='gray', lw=0.7, ls='--')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "fig_local_global.pdf"), bbox_inches='tight')
    plt.close()
    print("Saved fig_local_global.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4: Univariate Optimality Conditions
# ─────────────────────────────────────────────────────────────────────────────
def fig_optimality_conditions_1d():
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))
    titles = [
        "Second-order\nbut not first-order",
        "First-order\nand second-order",
        "First-order\nand second-order",
    ]

    # (a) second deriv > 0 but first deriv != 0 at candidate
    x = np.linspace(-1.5, 1.5, 200)
    axes[0].plot(x, x**2 + 0.5*x, 'steelblue', lw=2)
    xstar = -0.25
    axes[0].plot(xstar, xstar**2 + 0.5*xstar, 'go', ms=10, zorder=5)
    axes[0].axvline(xstar, color='gray', ls='--', lw=1)
    axes[0].set_title(titles[0], fontsize=9)
    axes[0].set_xlabel('$x$')
    axes[0].text(xstar + 0.1, -0.18, r"$x^*$", fontsize=11)

    # (b) Strong local min: f'=0, f''>0
    axes[1].plot(x, x**2, 'steelblue', lw=2)
    axes[1].plot(0, 0, 'r*', ms=14, zorder=5)
    axes[1].axvline(0, color='gray', ls='--', lw=1)
    axes[1].set_title(titles[1], fontsize=9)
    axes[1].set_xlabel('$x$')
    axes[1].text(0.08, -0.18, r"$x^*$", fontsize=11)

    # (c) Inflection: f'=0, f''=0 (neither max nor min for x^3)
    axes[2].plot(x, x**3, 'steelblue', lw=2)
    axes[2].plot(0, 0, 'rs', ms=10, zorder=5)
    axes[2].axvline(0, color='gray', ls='--', lw=1)
    axes[2].set_title("First-order only\n(inflection, not min)", fontsize=9)
    axes[2].set_xlabel('$x$')
    axes[2].text(0.08, -0.15, r"$x^*$", fontsize=11)

    for ax in axes:
        ax.set_xlim(-1.5, 1.5)
        ax.axhline(0, color='gray', lw=0.5)

    fig.suptitle("Necessary Conditions for Local Minima (Univariate)",
                 fontweight='bold', fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "fig_optimality_1d.pdf"), bbox_inches='tight')
    plt.close()
    print("Saved fig_optimality_1d.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 5: Example 1.1 — f(x1,x2) = x1^2 - x2^2 (3D surface + contour)
# ─────────────────────────────────────────────────────────────────────────────
def fig_example_surface_contour():
    fig = plt.figure(figsize=(10, 4.5))
    gs = gridspec.GridSpec(1, 2, wspace=0.3)

    x1 = np.linspace(-2.5, 2.5, 100)
    x2 = np.linspace(-2.5, 2.5, 100)
    X1, X2 = np.meshgrid(x1, x2)
    Z = X1**2 - X2**2

    # 3D surface
    ax3d = fig.add_subplot(gs[0], projection='3d')
    ax3d.plot_surface(X1, X2, Z, cmap='viridis', alpha=0.85, linewidth=0)
    ax3d.set_xlabel('$x_1$', labelpad=4)
    ax3d.set_ylabel('$x_2$', labelpad=4)
    ax3d.set_zlabel('$f$', labelpad=4)
    ax3d.set_title('3D Surface:\n$f(x_1,x_2)=x_1^2 - x_2^2$', fontsize=10)

    # 2D contour
    ax2d = fig.add_subplot(gs[1])
    cs = ax2d.contourf(X1, X2, Z, levels=20, cmap='viridis', alpha=0.85)
    ax2d.contour(X1, X2, Z, levels=20, colors='white', linewidths=0.5, alpha=0.5)
    plt.colorbar(cs, ax=ax2d, shrink=0.85)
    ax2d.set_xlabel('$x_1$')
    ax2d.set_ylabel('$x_2$')
    ax2d.set_title('Contour Plot:\n$f(x_1,x_2)=x_1^2 - x_2^2$', fontsize=10)
    ax2d.set_aspect('equal')

    plt.savefig(os.path.join(OUTDIR, "fig_surface_contour.pdf"), bbox_inches='tight')
    plt.close()
    print("Saved fig_surface_contour.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 6: Rosenbrock function + gradient/Hessian example (Example 1.2)
# ─────────────────────────────────────────────────────────────────────────────
def fig_rosenbrock():
    fig, ax = plt.subplots(figsize=(5, 4.5))
    x1 = np.linspace(-2, 2, 300)
    x2 = np.linspace(-1, 3, 300)
    X1, X2 = np.meshgrid(x1, x2)
    Z = (1 - X1)**2 + 5*(X2 - X1**2)**2

    # Log scale for better visibility
    cs = ax.contourf(X1, X2, np.log1p(Z), levels=25, cmap='YlOrRd', alpha=0.9)
    ax.contour(X1, X2, np.log1p(Z), levels=25, colors='k', linewidths=0.4, alpha=0.5)
    plt.colorbar(cs, ax=ax, label='$\\log(1+f)$', shrink=0.9)

    # True minimum at (1,1)
    ax.plot(1, 1, 'b*', ms=14, zorder=5, label='Minimum $(1,1)$')
    # Point from Example 1.2 — same as min in this case
    ax.plot(1, 1, 'r.', ms=10, zorder=6)

    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_title('Rosenbrock Function\n$f(\\mathbf{x}) = (1-x_1)^2 + 5(x_2 - x_1^2)^2$',
                 fontweight='bold', fontsize=10)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "fig_rosenbrock.pdf"), bbox_inches='tight')
    plt.close()
    print("Saved fig_rosenbrock.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 7: Multivariate optimality — hill, saddle, bowl
#   (schematic 3D views as in Figure 1.15 of the book)
# ─────────────────────────────────────────────────────────────────────────────
def fig_multivariate_regions():
    fig = plt.figure(figsize=(12, 3.8))

    x = np.linspace(-1.5, 1.5, 60)
    y = np.linspace(-1.5, 1.5, 60)
    X, Y = np.meshgrid(x, y)

    panels = [
        ("A Hill\n$\\nabla f=0$, $\\nabla^2 f$ neg. def.",
         -(X**2 + Y**2),
         'Reds', 'Not a local min'),
        ("A Saddle\n$\\nabla f=0$, $\\nabla^2 f$ indef.",
         X**2 - Y**2,
         'PuOr', 'Not a local min'),
        ("A Bowl\n$\\nabla f=0$, $\\nabla^2 f$ pos. def.",
         X**2 + Y**2,
         'Blues', 'Local minimum'),
    ]

    for i, (title, Z, cmap, note) in enumerate(panels):
        ax = fig.add_subplot(1, 3, i+1, projection='3d')
        ax.plot_surface(X, Y, Z, cmap=cmap, alpha=0.80, linewidth=0)
        ax.plot([0], [0], [Z[30, 30]], 'r*', ms=12, zorder=5)
        ax.set_title(title, fontsize=9, pad=2)
        ax.set_xlabel('$x_1$', fontsize=8, labelpad=2)
        ax.set_ylabel('$x_2$', fontsize=8, labelpad=2)
        ax.set_zlabel('$f$', fontsize=8, labelpad=2)
        ax.tick_params(labelsize=6)
        ax.text2D(0.5, -0.08, note, transform=ax.transAxes,
                  ha='center', fontsize=9, color='navy', fontweight='bold')

    fig.suptitle("Three Local Regions Where $\\nabla f = 0$ (Figure 1.15)",
                 fontsize=11, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "fig_multivariate_regions.pdf"),
                bbox_inches='tight')
    plt.close()
    print("Saved fig_multivariate_regions.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 8: Book chapter dependency tree (Figure 1.1 approximation)
# ─────────────────────────────────────────────────────────────────────────────
def fig_chapter_tree():
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.axis('off')

    # Node positions: (x, y, label)
    nodes = {
        1:  (5.0, 6.3, "Ch 1\nIntro"),
        2:  (2.0, 5.0, "Ch 2\nDerivatives"),
        3:  (5.0, 5.0, "Ch 3\nBracketing"),
        4:  (8.0, 5.0, "Ch 4\nLocal Descent"),
        5:  (1.0, 3.5, "Ch 5\nFirst-Order"),
        6:  (3.0, 3.5, "Ch 6\nSecond-Order"),
        7:  (5.0, 3.5, "Ch 7\nConjugate"),
        8:  (7.0, 3.5, "Ch 8\nStochastic"),
        9:  (9.0, 3.5, "Ch 9\nNoise"),
        10: (2.0, 2.0, "Ch 10\nConstraints"),
        11: (5.0, 2.0, "Ch 11\nLinear"),
        13: (8.0, 2.0, "Ch 13\nSampling"),
        17: (3.5, 0.8, "Ch 17\nSurrogate"),
        21: (7.0, 0.8, "Ch 21\nMultiobj."),
    }

    edges = [
        (1, 2), (1, 3), (1, 4),
        (2, 5), (2, 6), (4, 7), (4, 8), (4, 9),
        (5, 10), (3, 11), (3, 13),
        (10, 17), (13, 21),
    ]

    colors_map = {
        1: '#4e79a7',
        2: '#f28e2b', 3: '#f28e2b', 4: '#f28e2b',
        5: '#59a14f', 6: '#59a14f', 7: '#59a14f',
        8: '#e15759', 9: '#e15759',
        10: '#76b7b2', 11: '#76b7b2', 13: '#76b7b2',
        17: '#b07aa1', 21: '#b07aa1',
    }

    for (src, dst) in edges:
        x0, y0 = nodes[src][0], nodes[src][1]
        x1, y1 = nodes[dst][0], nodes[dst][1]
        ax.annotate("", xy=(x1, y1 + 0.22), xytext=(x0, y0 - 0.22),
                     arrowprops=dict(arrowstyle='->', lw=1.2, color='#555555'))

    for nid, (x, y, label) in nodes.items():
        c = colors_map.get(nid, '#aaaaaa')
        ax.add_patch(mpatches.FancyBboxPatch(
            (x - 0.55, y - 0.28), 1.1, 0.56,
            boxstyle="round,pad=0.05", linewidth=1.2,
            edgecolor='#333', facecolor=c, alpha=0.8, zorder=3))
        ax.text(x, y, label, ha='center', va='center',
                fontsize=7.5, color='white', fontweight='bold', zorder=4)

    ax.set_title("Chapter Dependency Overview (simplified)", fontsize=11,
                 fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "fig_chapter_tree.pdf"), bbox_inches='tight')
    plt.close()
    print("Saved fig_chapter_tree.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 9: Deep Learning loss surface (stylised)
# ─────────────────────────────────────────────────────────────────────────────
def fig_deep_learning_loss():
    fig, ax = plt.subplots(figsize=(6, 3.5))
    epochs = np.arange(1, 101)
    np.random.seed(42)
    noise = np.random.normal(0, 0.02, 100)
    train_loss = 2.5 * np.exp(-0.04 * epochs) + 0.12 + noise
    val_loss = 2.5 * np.exp(-0.035 * epochs) + 0.18 + 0.3*np.exp(0.005*epochs) + np.random.normal(0, 0.03, 100)

    ax.plot(epochs, train_loss, label='Training loss', color='steelblue', lw=2)
    ax.plot(epochs, val_loss, label='Validation loss', color='tomato', lw=2, ls='--')
    ax.axvline(65, color='gray', ls=':', lw=1.5)
    ax.text(66, 0.7, 'Early\nstopping', fontsize=9, color='gray')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Deep Learning: Optimizing Neural Network Weights', fontweight='bold')
    ax.legend()
    ax.set_ylim(0, 2.8)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "fig_dl_loss.pdf"), bbox_inches='tight')
    plt.close()
    print("Saved fig_dl_loss.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 10: Portfolio optimization — Pareto-style (risk vs return)
# ─────────────────────────────────────────────────────────────────────────────
def fig_portfolio():
    fig, ax = plt.subplots(figsize=(5.5, 4))
    np.random.seed(7)
    n = 300
    risks = np.random.uniform(0.05, 0.4, n)
    returns = 0.6*risks + np.random.normal(0, 0.05, n)
    returns = np.clip(returns, 0.02, 0.35)

    ax.scatter(risks, returns, c='#aec6e8', s=18, alpha=0.6, label='Feasible portfolios')

    # Efficient frontier (convex hull approximation)
    from scipy.spatial import ConvexHull
    pts = np.column_stack([risks, returns])
    hull = ConvexHull(pts)
    hull_pts = pts[hull.vertices]
    # Keep upper-left boundary
    idx = np.argsort(hull_pts[:, 0])
    hull_sorted = hull_pts[idx]
    # Filter: for each risk level keep highest return
    ef_risks = np.linspace(0.06, 0.38, 50)
    ef_returns = []
    for r in ef_risks:
        nearby = returns[np.abs(risks - r) < 0.03]
        ef_returns.append(np.max(nearby) if len(nearby) > 0 else np.nan)
    ef_returns = np.array(ef_returns)
    mask = ~np.isnan(ef_returns)
    ax.plot(ef_risks[mask], ef_returns[mask], 'r-', lw=2.5, label='Efficient frontier')

    ax.set_xlabel('Risk (std dev)')
    ax.set_ylabel('Expected Return')
    ax.set_title('Portfolio Optimization', fontweight='bold')
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "fig_portfolio.pdf"), bbox_inches='tight')
    plt.close()
    print("Saved fig_portfolio.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 11: Crop Figure 1.1 (chapter dependency) from PDF
# ─────────────────────────────────────────────────────────────────────────────
def crop_book_figure(page_num_0indexed, clip_rect, outname):
    """Crop a region from the book PDF and save as PNG then convert to PDF."""
    try:
        doc = fitz.open(BOOK_PDF)
        page = doc[page_num_0indexed]
        # clip_rect: (x0, y0, x1, y1) in PDF points
        mat = fitz.Matrix(2.0, 2.0)  # 2x zoom
        rect = fitz.Rect(*clip_rect)
        pix = page.get_pixmap(matrix=mat, clip=rect)
        png_path = os.path.join(OUTDIR, outname + ".png")
        pix.save(png_path)
        # Convert to PDF via matplotlib
        img = plt.imread(png_path)
        fig2, ax2 = plt.subplots(figsize=(img.shape[1]/150, img.shape[0]/150))
        ax2.imshow(img)
        ax2.axis('off')
        plt.tight_layout(pad=0)
        plt.savefig(os.path.join(OUTDIR, outname + ".pdf"), bbox_inches='tight', dpi=150)
        plt.close()
        doc.close()
        print(f"Cropped {outname}.pdf from PDF page {page_num_0indexed+1}")
    except Exception as e:
        print(f"WARNING: Could not crop from PDF ({e}). Skipping {outname}.")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 12: Taylor expansion illustration
# ─────────────────────────────────────────────────────────────────────────────
def fig_taylor():
    fig, ax = plt.subplots(figsize=(6, 3.8))
    x = np.linspace(-0.8, 2.2, 300)

    def f(x): return np.sin(x) + 0.3*x**2

    xstar = 0.8
    fstar = f(xstar)
    # First derivative approx (numerical)
    h = 1e-5
    fprime = (f(xstar+h) - f(xstar-h))/(2*h)
    fdbl = (f(xstar+h) - 2*f(xstar) + f(xstar-h))/(h**2)

    y_f = f(x)
    y_t1 = fstar + fprime*(x - xstar)                         # 1st-order Taylor
    y_t2 = fstar + fprime*(x - xstar) + 0.5*fdbl*(x-xstar)**2  # 2nd-order Taylor

    ax.plot(x, y_f, 'steelblue', lw=2.5, label='$f(x)$')
    ax.plot(x, y_t1, 'orange', lw=1.8, ls='--', label='1st-order Taylor')
    ax.plot(x, y_t2, 'green', lw=1.8, ls=':', label='2nd-order Taylor')
    ax.plot(xstar, fstar, 'r*', ms=12, zorder=5, label=f'$x^* = {xstar}$')
    ax.axvline(xstar, color='gray', lw=0.8, ls='--')

    ax.set_xlabel('$x$')
    ax.set_ylabel('$f(x)$')
    ax.set_title('Taylor Expansion about $x^*$', fontweight='bold')
    ax.legend(fontsize=9)
    ax.set_ylim(-0.5, 2.5)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "fig_taylor.pdf"), bbox_inches='tight')
    plt.close()
    print("Saved fig_taylor.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Generating figures for Chapter 1 ...")
    fig_optimization_process()
    fig_feasible_set()
    fig_local_global_minima()
    fig_optimality_conditions_1d()
    fig_example_surface_contour()
    fig_rosenbrock()
    fig_multivariate_regions()
    fig_chapter_tree()
    fig_deep_learning_loss()
    fig_portfolio()
    fig_taylor()

    # Crop Figure 1.1 (dependency diagram) from book PDF
    # Page 36 (0-indexed: 35) of the book — adjust if needed
    # The diagram is roughly in the right margin area
    # We'll attempt a crop; if PDF structure differs it falls back gracefully
    crop_book_figure(35, (370, 90, 600, 600), "fig_book_dep_crop")

    print("\nAll figures generated in:", OUTDIR)
