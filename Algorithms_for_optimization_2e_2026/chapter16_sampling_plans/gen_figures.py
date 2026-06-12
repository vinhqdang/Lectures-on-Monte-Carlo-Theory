"""
gen_figures.py  –  Generate all figures for Chapter 16: Sampling Plans
Algorithms for Optimization, 2nd ed., Kochenderfer & Wheeler (2026)

Run via:
  conda run -n py313 python3 gen_figures.py
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from pathlib import Path
import itertools
import fitz  # pymupdf

FIGURES = Path(__file__).parent / "figures"
FIGURES.mkdir(exist_ok=True)

BOOK_PDF = Path(__file__).parent.parent / "optimization_book.pdf"

# ─────────────────────────────────────────────────────────────────────────────
# Helper: save figure
# ─────────────────────────────────────────────────────────────────────────────
def savefig(name, fig=None, dpi=150):
    path = FIGURES / f"{name}.pdf"
    (fig or plt).savefig(path, bbox_inches='tight')
    plt.close('all')
    print(f"  saved {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Crop from PDF using pymupdf
# ─────────────────────────────────────────────────────────────────────────────
def crop_pdf_page(page_number_1indexed, rect_frac, out_name, dpi=200):
    """Crop a rectangular region from a PDF page and save as PDF figure."""
    doc = fitz.open(str(BOOK_PDF))
    page = doc[page_number_1indexed - 1]   # 0-indexed
    pw, ph = page.rect.width, page.rect.height
    x0, y0, x1, y1 = rect_frac
    clip = fitz.Rect(x0 * pw, y0 * ph, x1 * pw, y1 * ph)
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat, clip=clip)
    # save as PNG then embed in a PDF via matplotlib
    import io
    from PIL import Image
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    fig, ax = plt.subplots(figsize=(img.width / dpi * 1.2, img.height / dpi * 1.2))
    ax.imshow(np.array(img))
    ax.axis('off')
    savefig(out_name, fig=fig)
    doc.close()


# ═════════════════════════════════════════════════════════════════════════════
# Figure 1: Full Factorial Grid (2D, 5x4)
# ═════════════════════════════════════════════════════════════════════════════
def fig_full_factorial():
    a = np.array([0.0, 0.0])
    b = np.array([1.0, 1.0])
    m = [6, 5]
    x1 = np.linspace(a[0], b[0], m[0])
    x2 = np.linspace(a[1], b[1], m[1])
    pts = list(itertools.product(x1, x2))
    xs, ys = zip(*pts)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(xs, ys, s=40, color='#1f4e79', zorder=3)

    # Annotations
    ax.annotate('', xy=(x1[-1], x2[-1]), xytext=(x1[-2], x2[-1]),
                arrowprops=dict(arrowstyle='<->', color='gray', lw=1.2))
    ax.text((x1[-1] + x1[-2]) / 2, x2[-1] + 0.06,
            r'$\frac{b_1-a_1}{m_1-1}$', ha='center', fontsize=9)

    ax.annotate('', xy=(x1[-1], x2[-1]), xytext=(x1[-1], x2[-2]),
                arrowprops=dict(arrowstyle='<->', color='gray', lw=1.2))
    ax.text(x1[-1] + 0.07, (x2[-1] + x2[-2]) / 2,
            r'$\frac{b_2-a_2}{m_2-1}$', ha='left', fontsize=9)

    ax.set_xlim(-0.15, 1.25)
    ax.set_ylim(-0.15, 1.25)
    ax.set_xlabel(r'$x_1$', fontsize=11)
    ax.set_ylabel(r'$x_2$', fontsize=11)
    ax.set_xticks([a[0], 0.5, b[0]])
    ax.set_xticklabels([r'$a_1$', r'$x_1$', r'$b_1$'])
    ax.set_yticks([a[1], b[1]])
    ax.set_yticklabels([r'$a_2$', r'$b_2$'])
    ax.set_title('Full Factorial Design (6×5 grid)', fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.4)
    savefig('full_factorial', fig)


# ═════════════════════════════════════════════════════════════════════════════
# Figure 2: Random Sampling vs Uniform Projection (side-by-side, 2D, 20 pts)
# ═════════════════════════════════════════════════════════════════════════════
def fig_random_vs_projection():
    rng = np.random.default_rng(42)
    n = 20

    # Random
    rand_pts = rng.uniform(0, 1, (n, 2))

    # Uniform projection (Latin hypercube style)
    perm1 = rng.permutation(n)
    perm2 = rng.permutation(n)
    proj_pts = np.column_stack([(perm1 + 0.5) / n, (perm2 + 0.5) / n])

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
    titles = ['Random Sampling', 'Uniform Projection Plan (LHS)']
    for ax, pts, title in zip(axes, [rand_pts, proj_pts], titles):
        ax.scatter(pts[:, 0], pts[:, 1], s=35, color='#1f4e79')
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel(r'$x_1$'); ax.set_ylabel(r'$x_2$')
        ax.set_title(title, fontsize=10)
        ax.set_aspect('equal')
        ax.grid(True, linestyle='--', alpha=0.3)

    # Marginal histograms as tick marks
    plt.tight_layout()
    savefig('random_vs_projection', fig)


# ═════════════════════════════════════════════════════════════════════════════
# Figure 3: Stratified Sampling (2D, 4x4 strata)
# ═════════════════════════════════════════════════════════════════════════════
def fig_stratified():
    rng = np.random.default_rng(7)
    k = 4
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))

    # Left: grid strata boundaries + one point per stratum
    ax = axes[0]
    pts = []
    for i in range(k):
        for j in range(k):
            x = (i + rng.uniform()) / k
            y = (j + rng.uniform()) / k
            pts.append((x, y))
    pts = np.array(pts)
    ax.scatter(pts[:, 0], pts[:, 1], s=35, color='#c0392b')
    for t in np.linspace(0, 1, k + 1):
        ax.axhline(t, color='gray', lw=0.7, ls='--')
        ax.axvline(t, color='gray', lw=0.7, ls='--')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel(r'$x_1$'); ax.set_ylabel(r'$x_2$')
    ax.set_title('Stratified Sampling', fontsize=10)
    ax.set_aspect('equal')

    # Right: Latin hypercube - one sample per row & col
    ax = axes[1]
    perm1 = rng.permutation(k * k)
    perm2 = rng.permutation(k * k)
    n = k * k
    lhc = np.column_stack([(perm1 + rng.uniform(size=n)) / n,
                            (perm2 + rng.uniform(size=n)) / n])
    ax.scatter(lhc[:, 0], lhc[:, 1], s=35, color='#1f4e79')
    for t in np.linspace(0, 1, n + 1):
        ax.axhline(t, color='gray', lw=0.3, ls='-', alpha=0.4)
        ax.axvline(t, color='gray', lw=0.3, ls='-', alpha=0.4)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel(r'$x_1$'); ax.set_ylabel(r'$x_2$')
    ax.set_title('Latin Hypercube (Uniform Projection)', fontsize=10)
    ax.set_aspect('equal')

    plt.tight_layout()
    savefig('stratified_sampling', fig)


# ═════════════════════════════════════════════════════════════════════════════
# Figure 4: Discrepancy Example (Example 16.1 from book)
# ═════════════════════════════════════════════════════════════════════════════
def fig_discrepancy_example():
    pts = np.array([[1/5, 1/5], [2/5, 1/5], [1/10, 3/5],
                    [9/10, 3/10], [1/50, 1/50], [3/5, 4/5]])

    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.scatter(pts[:, 0], pts[:, 1], s=55, color='#1f4e79', zorder=5)

    # Blue rectangle (high discrepancy, contains 3 pts)
    blue = patches.Rectangle((1/10, 1/5), 2/5 - 1/10, 3/5 - 1/5,
                              linewidth=1.5, edgecolor='blue',
                              facecolor='blue', alpha=0.25, zorder=2)
    ax.add_patch(blue)
    ax.text(0.08, 0.52, 'Blue: vol=0.12\n3 pts → disc=0.38',
            fontsize=7.5, color='blue')

    # Purple rectangle (even higher discrepancy)
    purple = patches.Rectangle((1/10 + 0.02, 1/5 + 0.02),
                                9/10 - 1/10 - 0.04, 4/5 - 1/5 - 0.04,
                                linewidth=1.5, edgecolor='purple',
                                facecolor='purple', alpha=0.18, zorder=1)
    ax.add_patch(purple)
    ax.text(0.5, 0.14, 'Purple: vol≈0.48, 0 pts inside',
            fontsize=7.5, color='purple')

    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel(r'$x_1$', fontsize=11)
    ax.set_ylabel(r'$x_2$', fontsize=11)
    ax.set_title('Discrepancy Example 16.1', fontsize=11)
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.3)
    savefig('discrepancy_example', fig)


# ═════════════════════════════════════════════════════════════════════════════
# Figure 5: Pairwise distances histogram comparison
# ═════════════════════════════════════════════════════════════════════════════
def fig_pairwise_distances():
    rng = np.random.default_rng(0)
    n = 30

    def lhc(n, rng):
        p1, p2 = rng.permutation(n), rng.permutation(n)
        return np.column_stack([(p1 + 0.5) / n, (p2 + 0.5) / n])

    def pairwise(X):
        dists = []
        for i in range(len(X)):
            for j in range(i + 1, len(X)):
                dists.append(np.linalg.norm(X[i] - X[j]))
        return dists

    rand_pts = rng.uniform(0, 1, (n, 2))
    lhc_pts = lhc(n, rng)

    dr = pairwise(rand_pts)
    dl = pairwise(lhc_pts)

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2), sharey=True)
    for ax, d, title, color in zip(axes, [dr, dl],
                                   ['Random', 'LHS (Uniform Proj.)'],
                                   ['#e74c3c', '#1f4e79']):
        ax.hist(d, bins=15, color=color, alpha=0.75, edgecolor='white')
        ax.set_xlabel('Pairwise distance', fontsize=10)
        ax.set_ylabel('Count')
        ax.set_title(title, fontsize=10)
        ax.axvline(np.min(d), color='black', ls='--', lw=1.2,
                   label=f'min={np.min(d):.3f}')
        ax.legend(fontsize=8)
    plt.tight_layout()
    savefig('pairwise_distances', fig)


# ═════════════════════════════════════════════════════════════════════════════
# Figure 6: Morris-Mitchell Phi_q criterion for UPPs
# ═════════════════════════════════════════════════════════════════════════════
def fig_morris_mitchell():
    rng = np.random.default_rng(123)
    n_plans = 6
    n_pts = 10

    def phi_q(X, q=1):
        dists = []
        for i in range(len(X)):
            for j in range(i + 1, len(X)):
                dists.append(np.linalg.norm(X[i] - X[j]))
        return sum(d ** (-q) for d in dists) ** (1 / q)

    def rand_lhc(n, rng):
        p1, p2 = rng.permutation(n), rng.permutation(n)
        return np.column_stack([(p1 + 0.5) / n, (p2 + 0.5) / n])

    plans = [rand_lhc(n_pts, rng) for _ in range(n_plans)]
    scores = [phi_q(p) for p in plans]
    order = np.argsort(scores)

    fig, axes = plt.subplots(2, 3, figsize=(9, 5.5))
    for ax, idx in zip(axes.flat, order):
        ax.scatter(plans[idx][:, 0], plans[idx][:, 1], s=30, color='#1f4e79')
        ax.set_title(fr'$\Phi_1={scores[idx]:.1f}$', fontsize=9)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.tick_params(labelsize=7)
    plt.suptitle('UPPs sorted by Morris-Mitchell $\\Phi_1$ (best→worst)',
                 fontsize=10)
    plt.tight_layout()
    savefig('morris_mitchell', fig)


# ═════════════════════════════════════════════════════════════════════════════
# Figure 7: Halton sequence tree (van der Corput, base 2)
# ═════════════════════════════════════════════════════════════════════════════
def fig_halton_tree():
    def van_der_corput(n, base=2):
        seq = []
        for i in range(1, n + 1):
            result, f = 0.0, 1.0
            ii = i
            while ii > 0:
                f /= base
                result += f * (ii % base)
                ii = ii // base
            seq.append(result)
        return seq

    vals_b2 = van_der_corput(8, 2)

    fig, ax = plt.subplots(figsize=(7, 2.8))
    levels = [1, 2, 4, 8]
    y_pos = [3, 2, 1, 0]
    for level, y in zip(levels, y_pos):
        pts = van_der_corput(level, 2)
        ax.scatter(pts, [y] * len(pts), s=80, color='#c0392b', zorder=5)
        for p in pts:
            frac_str = ''
            for num, den in [(1, 2), (1, 4), (3, 4), (1, 8),
                              (3, 8), (5, 8), (7, 8), (1, 16)]:
                if abs(p - num / den) < 1e-9:
                    frac_str = f'{num}/{den}'
                    break
            ax.text(p, y + 0.15, frac_str, ha='center', fontsize=7)

    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.5, 3.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f'm={l}' for l in levels])
    ax.set_xlabel('Value', fontsize=10)
    ax.set_title('Van der Corput Sequence (base $b=2$)', fontsize=11)
    ax.axhline(y=0, color='black', lw=0.5, alpha=0.3)
    ax.grid(axis='x', linestyle='--', alpha=0.3)
    savefig('halton_tree', fig)


# ═════════════════════════════════════════════════════════════════════════════
# Figure 8: Halton 2D (base 2 & 3)
# ═════════════════════════════════════════════════════════════════════════════
def fig_halton_2d():
    def halton_1d(n, base):
        seq = []
        for i in range(1, n + 1):
            result, f = 0.0, 1.0
            ii = i
            while ii > 0:
                f /= base
                result += f * (ii % base)
                ii = ii // base
            seq.append(result)
        return np.array(seq)

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2))
    ns = [10, 100, 500]
    for ax, n in zip(axes, ns):
        h2 = halton_1d(n, 2)
        h3 = halton_1d(n, 3)
        colors = plt.cm.viridis(np.linspace(0, 1, n))
        ax.scatter(h2, h3, s=8, c=colors, alpha=0.85)
        ax.set_title(f'Halton n={n} (b=2,3)', fontsize=9)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.set_xlabel(r'$x_1$'); ax.set_ylabel(r'$x_2$')
    plt.tight_layout()
    savefig('halton_2d', fig)


# ═════════════════════════════════════════════════════════════════════════════
# Figure 9: Sobol sequence 2D
# ═════════════════════════════════════════════════════════════════════════════
def fig_sobol_2d():
    try:
        from scipy.stats.qmc import Sobol
        fig, axes = plt.subplots(1, 3, figsize=(10, 3.2))
        ns = [10, 100, 500]
        for ax, n in zip(axes, ns):
            sampler = Sobol(d=2, scramble=False)
            pts = sampler.random(n)
            colors = plt.cm.plasma(np.linspace(0, 1, n))
            ax.scatter(pts[:, 0], pts[:, 1], s=8, c=colors, alpha=0.85)
            ax.set_title(f'Sobol n={n}', fontsize=9)
            ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            ax.set_aspect('equal')
            ax.set_xlabel(r'$x_1$'); ax.set_ylabel(r'$x_2$')
        plt.tight_layout()
        savefig('sobol_2d', fig)
    except ImportError:
        print("  scipy.stats.qmc not available; skipping Sobol figure")


# ═════════════════════════════════════════════════════════════════════════════
# Figure 10: Comparison of methods at n=10, 100, 1000
# ═════════════════════════════════════════════════════════════════════════════
def fig_comparison_methods():
    def halton_1d(n, base):
        seq = []
        for i in range(1, n + 1):
            result, f = 0.0, 1.0
            ii = i
            while ii > 0:
                f /= base
                result += f * (ii % base)
                ii = ii // base
            seq.append(result)
        return np.array(seq)

    def additive_recurrence(m, c=None):
        if c is None:
            phi = (1 + np.sqrt(5)) / 2
            c = phi - 1
        x0 = np.random.rand()
        return np.array([(x0 + k * c) % 1.0 for k in range(1, m + 1)])

    def lhc(n, seed=0):
        rng = np.random.default_rng(seed)
        p1, p2 = rng.permutation(n), rng.permutation(n)
        return np.column_stack([(p1 + 0.5) / n, (p2 + 0.5) / n])

    try:
        from scipy.stats.qmc import Sobol
        has_sobol = True
    except ImportError:
        has_sobol = False

    ns = [10, 100, 1000]
    methods = ['Random', 'Additive Recurrence', 'Halton', 'Sobol', 'LHS']
    n_methods = len(methods)

    fig, axes = plt.subplots(len(ns), n_methods,
                             figsize=(n_methods * 2.2, len(ns) * 2.2))

    rng_main = np.random.default_rng(42)

    for row, n in enumerate(ns):
        rand_pts = rng_main.uniform(0, 1, (n, 2))
        ar1 = additive_recurrence(n)
        ar2 = additive_recurrence(n, c=np.sqrt(2) - 1)
        ar_pts = np.column_stack([ar1, ar2])
        h2 = halton_1d(n, 2)
        h3 = halton_1d(n, 3)
        halton_pts = np.column_stack([h2, h3])
        if has_sobol:
            sampler = Sobol(d=2, scramble=False)
            sobol_pts = sampler.random(n)
        else:
            sobol_pts = rng_main.uniform(0, 1, (n, 2))
        lhc_pts = lhc(n, seed=row)

        all_pts = [rand_pts, ar_pts, halton_pts, sobol_pts, lhc_pts]
        colors_row = plt.cm.viridis(np.linspace(0, 1, n))

        for col, (pts, method) in enumerate(zip(all_pts, methods)):
            ax = axes[row, col]
            ax.scatter(pts[:, 0], pts[:, 1], s=max(1, 6 - row),
                       c=colors_row, alpha=0.8)
            ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            ax.set_aspect('equal')
            ax.tick_params(labelsize=6)
            if row == 0:
                ax.set_title(method, fontsize=8)
            if col == 0:
                ax.set_ylabel(f'n={n}', fontsize=8)

    plt.suptitle('Space-Filling Sampling Methods Comparison', fontsize=11, y=1.01)
    plt.tight_layout()
    savefig('comparison_methods', fig)


# ═════════════════════════════════════════════════════════════════════════════
# Figure 11: Fill Distance illustration
# ═════════════════════════════════════════════════════════════════════════════
def fig_fill_distance():
    rng = np.random.default_rng(5)
    n = 15
    X = rng.uniform(0, 1, (n, 2))

    # For each point on a fine grid, find distance to nearest sample
    gx, gy = np.meshgrid(np.linspace(0, 1, 80), np.linspace(0, 1, 80))
    grid = np.column_stack([gx.ravel(), gy.ravel()])
    dist_to_nearest = np.array([np.min(np.linalg.norm(X - g, axis=1))
                                 for g in grid])
    max_dist = dist_to_nearest.max()
    worst_idx = np.argmax(dist_to_nearest)

    fig, ax = plt.subplots(figsize=(4.5, 4.2))
    heatmap = ax.contourf(gx, gy,
                          dist_to_nearest.reshape(gx.shape),
                          levels=20, cmap='YlOrRd', alpha=0.7)
    plt.colorbar(heatmap, ax=ax, label='Distance to nearest sample')
    ax.scatter(X[:, 0], X[:, 1], s=55, color='#1f4e79', zorder=5,
               label='Sample points')
    ax.scatter(grid[worst_idx, 0], grid[worst_idx, 1], s=120,
               marker='*', color='red', zorder=6,
               label=f'Fill distance = {max_dist:.3f}')
    circle = plt.Circle((grid[worst_idx, 0], grid[worst_idx, 1]),
                         max_dist, fill=False, color='red', lw=1.5,
                         linestyle='--')
    ax.add_patch(circle)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel(r'$x_1$'); ax.set_ylabel(r'$x_2$')
    ax.set_title('Fill Distance (max-min coverage)', fontsize=10)
    ax.legend(fontsize=8, loc='lower right')
    ax.set_aspect('equal')
    savefig('fill_distance', fig)


# ═════════════════════════════════════════════════════════════════════════════
# Figure 12: Greedy vs Exchange subset selection
# ═════════════════════════════════════════════════════════════════════════════
def fig_subset_selection():
    rng = np.random.default_rng(99)
    n_full = 50
    n_sub = 10
    X = rng.uniform(0, 1, (n_full, 2))

    # Greedy: pick first random, then pick point maximizing min distance to S
    def greedy_subset(X, m):
        rng2 = np.random.default_rng(3)
        S = [rng2.integers(len(X))]
        for _ in range(m - 1):
            best, best_d = -1, -np.inf
            for j in range(len(X)):
                if j in S:
                    continue
                d = min(np.linalg.norm(X[j] - X[s]) for s in S)
                if d > best_d:
                    best_d, best = d, j
            S.append(best)
        return S

    idx = greedy_subset(X, n_sub)

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.8))

    for ax, subset_idx, title in zip(axes, [idx, idx],
                                     ['Greedy Subset Selection',
                                      'Space-Filling Subset S ⊂ X']):
        ax.scatter(X[:, 0], X[:, 1], s=25, color='lightblue',
                   label='Full set X', zorder=2)
        sub = np.array([X[i] for i in subset_idx])
        ax.scatter(sub[:, 0], sub[:, 1], s=80, color='#c0392b',
                   label=f'Subset S (m={n_sub})', zorder=5)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.set_xlabel(r'$x_1$'); ax.set_ylabel(r'$x_2$')
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8)
    plt.tight_layout()
    savefig('subset_selection', fig)


# ═════════════════════════════════════════════════════════════════════════════
# Figure 13: Additive recurrence – golden ratio
# ═════════════════════════════════════════════════════════════════════════════
def fig_additive_recurrence():
    phi = (1 + np.sqrt(5)) / 2
    c = phi - 1  # 1/phi  (irrational)

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.0))
    ns = [10, 100, 1000]
    for ax, n in zip(axes, ns):
        x0 = 0.5
        seq = np.array([(x0 + k * c) % 1.0 for k in range(1, n + 1)])
        seq2 = np.array([(x0 + k * (np.sqrt(2) - 1)) % 1.0
                          for k in range(1, n + 1)])
        ax.scatter(seq, seq2, s=max(1, 5 - int(np.log10(n))),
                   color='#1f4e79', alpha=0.8)
        ax.set_title(f'Additive Recurrence n={n}', fontsize=9)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.set_xlabel(r'dim 1 ($c=\phi-1$)')
        ax.set_ylabel(r'dim 2 ($c=\sqrt{2}-1$)')
    plt.tight_layout()
    savefig('additive_recurrence', fig)


# ═════════════════════════════════════════════════════════════════════════════
# Crop key figures from book PDF
# ═════════════════════════════════════════════════════════════════════════════
def crop_book_figures():
    """Crop original figures from book PDF where exact reproduction is needed."""
    try:
        # p370 = book page 350: full factorial figure
        crop_pdf_page(370, (0.03, 0.04, 0.72, 0.52), 'book_full_factorial')
        # p371: uniform projection plan figure
        crop_pdf_page(371, (0.03, 0.04, 0.60, 0.40), 'book_uniform_proj')
        # p372: stratified sampling figure (right side)
        crop_pdf_page(372, (0.45, 0.15, 0.75, 0.60), 'book_stratified')
        # p374: discrepancy rectangles (book fig 16.4)
        crop_pdf_page(374, (0.05, 0.05, 0.72, 0.72), 'book_discrepancy')
        # p376: Morris-Mitchell plans sorted
        crop_pdf_page(376, (0.03, 0.04, 0.73, 0.65), 'book_morris_mitchell')
        # p380: Halton tree (fig 16.11)
        crop_pdf_page(380, (0.55, 0.35, 0.99, 0.70), 'book_halton_tree')
        # p382: comparison figure (fig 16.13)
        crop_pdf_page(382, (0.03, 0.04, 0.75, 0.92), 'book_comparison')
        print("  Book figure crops done.")
    except Exception as e:
        print(f"  Warning: could not crop book figures: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Run all
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("Generating Chapter 16 figures...")
    fig_full_factorial()
    fig_random_vs_projection()
    fig_stratified()
    fig_discrepancy_example()
    fig_pairwise_distances()
    fig_morris_mitchell()
    fig_halton_tree()
    fig_halton_2d()
    fig_sobol_2d()
    fig_comparison_methods()
    fig_fill_distance()
    fig_subset_selection()
    fig_additive_recurrence()
    crop_book_figures()
    print("All figures generated in:", FIGURES)
