"""
gen_figures.py  -  Generate all figures for Chapter 6 Beamer slides.
Figures produced:
  fig_solution_scatter.pdf   - scatter plot of 4 solutions in (c,e) space
  fig_archive_empty.pdf      - empty 20x20 MAP-Elites archive grid
  fig_archive_filled.pdf     - archive with 4 solutions placed
  fig_scaling_formula.pdf    - visual of raw->scaled mapping
  fig_map_elites_flow.pdf    - MAP-Elites algorithm flow diagram
  fig_seeding_effect.pdf     - seeding vs. non-seeding boundary comparison
  fig_parallel_coords.pdf    - illustrative parallel-coordinates plot
  fig_coverage_bar.pdf       - coverage bar chart (seeded vs. unseeded)
  fig_qd_concept.pdf         - quality-diversity concept diagram
  fig_chromosome.pdf         - chromosome gene structure
  fig_book_fig6_1.pdf        - cropped class diagram from PDF (Fig 6.1)
  fig_book_fig6_2.pdf        - cropped seeding figure from PDF (Fig 6.2)
  fig_book_fig6_3.pdf        - cropped parallel coords from PDF (Fig 6.3)
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
os.makedirs(OUT, exist_ok=True)

PDF_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..',
    'Nature Inspired Optimisation for Delivery Problems 2022.pdf'
)


def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f'  saved {path}')


# ---------------------------------------------------------------------------
# 1. Scatter of 4 solutions
# ---------------------------------------------------------------------------
def fig_solution_scatter():
    c = [500, 550, 610, 490]
    e = [1023, 1050, 990, 1120]
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(c, e, color='steelblue', s=80, zorder=3)
    for ci, ei in zip(c, e):
        ax.annotate(f'({ci},{ei})', (ci, ei),
                    textcoords='offset points', xytext=(6, 4), fontsize=8)
    ax.set_xlabel('Cost  $c$', fontsize=11)
    ax.set_ylabel('Emissions  $e$  (CO$_2$)', fontsize=11)
    ax.set_title('Four solutions in the behaviour space', fontsize=11)
    ax.set_xlim(350, 680)
    ax.set_ylim(950, 1150)
    ax.grid(True, linestyle='--', alpha=0.4)
    save(fig, 'fig_solution_scatter.pdf')


# ---------------------------------------------------------------------------
# 2. Empty 20x20 archive grid
# ---------------------------------------------------------------------------
def fig_archive_empty():
    fig, ax = plt.subplots(figsize=(5, 5))
    B = 20
    for x in range(B + 1):
        ax.axvline(x, color='gray', lw=0.4)
    for y in range(B + 1):
        ax.axhline(y, color='gray', lw=0.4)
    ax.set_xlim(0, B)
    ax.set_ylim(0, B)
    ax.set_xticks(range(0, B + 1, 2))
    ax.set_yticks(range(0, B + 1, 2))
    ax.set_xticklabels([str(i) for i in range(0, B + 1, 2)])
    ax.set_yticklabels([str(i) for i in range(0, B + 1, 2)])
    ax.set_xlabel('Scaled cost  $c$  (bin index)', fontsize=11)
    ax.set_ylabel('Scaled emissions  $e$  (bin index)', fontsize=11)
    ax.set_title('Empty MAP-Elites archive (20 x 20 = 400 bins)', fontsize=11)
    save(fig, 'fig_archive_empty.pdf')


# ---------------------------------------------------------------------------
# 3. Archive with 4 solutions placed
# ---------------------------------------------------------------------------
def fig_archive_filled():
    # Book gives: (500,1023)->10,8  (550,1050)->14,11  (610,990)->18,5  (490,1120)->10,18
    buckets_c = [10, 14, 18, 10]
    buckets_e = [8,  11,  5, 18]
    labels    = ['(500,1023)', '(550,1050)', '(610,990)', '(490,1120)']
    colors    = ['crimson', 'darkorange', 'seagreen', 'steelblue']

    fig, ax = plt.subplots(figsize=(5, 5))
    B = 20
    for x in range(B + 1):
        ax.axvline(x, color='gray', lw=0.4)
    for y in range(B + 1):
        ax.axhline(y, color='gray', lw=0.4)
    for bc, be, lbl, col in zip(buckets_c, buckets_e, labels, colors):
        ax.scatter(bc - 0.5, be - 0.5, marker='*', s=180, color=col, zorder=4, label=lbl)
    ax.set_xlim(0, B)
    ax.set_ylim(0, B)
    ax.set_xticks(range(0, B + 1, 2))
    ax.set_yticks(range(0, B + 1, 2))
    ax.set_xticklabels([str(i) for i in range(0, B + 1, 2)])
    ax.set_yticklabels([str(i) for i in range(0, B + 1, 2)])
    ax.set_xlabel('Scaled cost  $c$  (bin)', fontsize=11)
    ax.set_ylabel('Scaled emissions  $e$  (bin)', fontsize=11)
    ax.set_title('Archive with 4 elite solutions placed', fontsize=11)
    ax.legend(fontsize=7, loc='upper left')
    save(fig, 'fig_archive_filled.pdf')


# ---------------------------------------------------------------------------
# 4. Scaling formula worked example
# ---------------------------------------------------------------------------
def fig_scaling_formula():
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.axis('off')
    steps = [
        r"Step 1 — compute range:  $\delta = (max + 1) - min = (650+1) - 350 = 301$",
        r"Step 2 — bin capacity:   $cap = \delta\,/\,b = 301\,/\,20 = 15.05$",
        r"Step 3 — scaled index:   $s = \mathrm{int}\!\left(\dfrac{r - min}{cap} + 1\right) = \mathrm{int}\!\left(\dfrac{500 - 350}{15.05} + 1\right) = \mathrm{int}(10.97) = 10$",
    ]
    for i, step in enumerate(steps):
        ax.text(0.03, 0.88 - i * 0.30, step,
                transform=ax.transAxes, fontsize=10.5, va='top',
                bbox=dict(boxstyle='round,pad=0.35', fc='#f7f7e8', ec='gray', lw=0.8))
    ax.set_title('Worked example: mapping raw value $r=500$ to bin $s=10$\n'
                 '($min=350$, $max=650$, $b=20$ bins)',
                 fontsize=11)
    save(fig, 'fig_scaling_formula.pdf')


# ---------------------------------------------------------------------------
# 5. MAP-Elites flow diagram
# ---------------------------------------------------------------------------
def fig_map_elites_flow():
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.axis('off')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)

    def box(x, y, w, h, text, fc='#d4e6f1', fontsize=9):
        rect = mpatches.FancyBboxPatch(
            (x - w/2, y - h/2), w, h,
            boxstyle='round,pad=0.12',
            facecolor=fc, edgecolor='steelblue', linewidth=1.3)
        ax.add_patch(rect)
        ax.text(x, y, text, ha='center', va='center',
                fontsize=fontsize, multialignment='center')

    def arrow(x1, y1, x2, y2):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color='#333', lw=1.4))

    box(5, 5.4, 7, 0.75, 'INITIALISE: generate init random solutions, add to archive',
        fc='#d5ecd4', fontsize=9)
    box(5, 4.3, 6, 0.7, 'LOOP while evals < totEvals', fc='#fff3cd', fontsize=9)
    box(2.8, 3.2, 4.5, 0.7,
        'xOver branch:\ncrossover two random archive parents', fontsize=8)
    box(7.2, 3.2, 4.5, 0.7,
        'Mutation branch:\ncopy one random archive parent, mutate', fontsize=8)
    box(5, 2.1, 7, 0.7,
        'Evaluate candidate c: compute fitness f and bin key k', fontsize=9)
    box(3, 1.0, 4.2, 0.7,
        'Cell empty?\nPlace c in archive[k]', fc='#d5ecd4', fontsize=8)
    box(7, 1.0, 4.2, 0.7,
        'Cell occupied AND f(c) < f(old)?\nReplace archive[k] with c', fc='#fde8e8', fontsize=8)
    box(5, 0.15, 5, 0.5, 'RETURN archive', fc='#d5ecd4', fontsize=9)

    arrow(5, 5.05, 5, 4.65)
    arrow(5, 3.95, 2.8, 3.55)
    arrow(5, 3.95, 7.2, 3.55)
    arrow(2.8, 2.85, 5, 2.45)
    arrow(7.2, 2.85, 5, 2.45)
    arrow(5, 1.75, 3, 1.35)
    arrow(5, 1.75, 7, 1.35)
    arrow(3, 0.65, 5, 0.40)
    arrow(7, 0.65, 5, 0.40)

    ax.set_title('MAP-Elites Algorithm (Algorithm 14) — control flow', fontsize=12, pad=6)
    save(fig, 'fig_map_elites_flow.pdf')


# ---------------------------------------------------------------------------
# 6. Seeding effect comparison
# ---------------------------------------------------------------------------
def fig_seeding_effect():
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    rng = np.random.default_rng(42)

    for ax, title, seed_factor in zip(
            axes,
            ['Without seeding\n(narrow, potentially misleading bounds)',
             'With seeding\n(wider, realistic bounds from EA population)'],
            [0.3, 1.0]):
        n = 30
        xs = np.linspace(0, 1, n)
        for i in range(n):
            y0 = rng.uniform(0.1, 0.4) * seed_factor
            y1 = rng.uniform(0.6, 0.9) * seed_factor + (1 - seed_factor) * 0.5
            color = plt.cm.Greens(0.4 + 0.5 * rng.random())
            ax.plot([xs[i], xs[i]], [y0, y1], color=color, lw=1.0, alpha=0.8)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel('Characteristic (normalised position)', fontsize=9)
        ax.set_ylabel('Value range covered', fontsize=9)
        ax.set_ylim(0, 1)
        ax.grid(True, linestyle='--', alpha=0.3)

    fig.suptitle('Effect of seeding on archive boundary estimation (Fig. 6.2)', fontsize=11)
    plt.tight_layout()
    save(fig, 'fig_seeding_effect.pdf')


# ---------------------------------------------------------------------------
# 7. Illustrative parallel coordinates
# ---------------------------------------------------------------------------
def fig_parallel_coords():
    dims = ['Fixed\nCost', 'Staff\nCost', 'Run.\nCost', 'Cost/\nDel.', 'Emissions',
            '% Cycle\nDel.', '% Cycle\nDist.', 'Cycles', 'Vans']
    n_dims = len(dims)
    rng = np.random.default_rng(7)
    n_sols = 60
    data = rng.random((n_sols, n_dims))

    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(n_dims)
    for i in range(n_sols):
        color = plt.cm.Greens(0.3 + 0.6 * data[i, 0])
        ax.plot(x, data[i], color=color, alpha=0.45, lw=0.8)

    # Highlight one "good" low-emission solution
    good = np.array([0.2, 0.3, 0.25, 0.15, 0.05, 0.8, 0.75, 0.5, 0.1])
    ax.plot(x, good, color='blue', lw=2.5, label='Example selected solution')

    ax.set_xticks(x)
    ax.set_xticklabels(dims, fontsize=8.5)
    ax.set_ylabel('Normalised value', fontsize=9)
    ax.set_title('Parallel coordinates: archive of elite solutions\n'
                 '(each green line = one elite solution across 9 characteristics)', fontsize=10)
    ax.set_ylim(-0.05, 1.1)
    ax.legend(fontsize=9)
    ax.grid(axis='x', linestyle='--', alpha=0.4)
    save(fig, 'fig_parallel_coords.pdf')


# ---------------------------------------------------------------------------
# 8. Coverage bar chart (seeded vs. unseeded)
# ---------------------------------------------------------------------------
def fig_coverage_bar():
    scenarios = ['A1', 'A1+A2', 'A1+A2+A3', 'A2', 'A3', 'C1', 'C2', 'A4']
    seeded   = [1576.7, 19.4, 15.2, 248.6, 298.6, 137.8, 73.1, 369.7]
    unseeded = [0.0,     0.0,  0.0,   0.0,   0.0,   0.0,  0.0,   0.0]

    fig, ax = plt.subplots(figsize=(9, 4))
    xs = np.arange(len(scenarios))
    w = 0.35
    ax.bar(xs - w/2, seeded,   w, label='Seeded',   color='steelblue', alpha=0.85)
    ax.bar(xs + w/2, unseeded, w, label='Unseeded', color='tomato',    alpha=0.85)
    ax.set_xticks(xs)
    ax.set_xticklabels(scenarios, fontsize=9)
    ax.set_ylabel('Avg. solutions in archive', fontsize=10)
    ax.set_title('Archive coverage: seeded vs. unseeded MAP-Elites\n'
                 '(A problems, Table 6.9 — unseeded archive remains essentially empty)', fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    save(fig, 'fig_coverage_bar.pdf')


# ---------------------------------------------------------------------------
# 9. Quality-Diversity concept: three paradigms
# ---------------------------------------------------------------------------
def fig_qd_concept():
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))

    rng = np.random.default_rng(0)
    pts_x = rng.uniform(1, 9, 40)
    pts_y = rng.uniform(1, 9, 40)

    # Panel A: single-objective
    ax = axes[0]
    ax.scatter(pts_x, pts_y, s=18, c='lightblue', alpha=0.6, zorder=2)
    ax.scatter([4.5], [4.0], s=200, c='red', marker='*', zorder=5, label='Best')
    ax.set_title('Traditional Optimisation\n(one best solution)', fontsize=9)
    ax.set_xlabel('Characteristic 1', fontsize=8)
    ax.set_ylabel('Characteristic 2', fontsize=8)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 10); ax.set_ylim(0, 10)

    # Panel B: Pareto front
    ax = axes[1]
    ax.scatter(pts_x, pts_y, s=18, c='lightblue', alpha=0.6)
    pxs = np.linspace(1.5, 8.5, 30)
    pys = 8.0 / (pxs * 0.25 + 0.5)
    ax.plot(pxs, np.clip(pys, 1, 9), 'r-', lw=2, label='Pareto front')
    ax.set_title('Multi-objective (Pareto)\n(non-dominated front)', fontsize=9)
    ax.set_xlabel('Objective 1 (cost)', fontsize=8)
    ax.set_ylabel('Objective 2 (emissions)', fontsize=8)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 10); ax.set_ylim(0, 10)

    # Panel C: MAP-Elites archive grid
    ax = axes[2]
    grid = rng.random((8, 8))
    mask = rng.random((8, 8)) > 0.6
    grid[mask] = np.nan
    im = ax.imshow(grid, cmap='YlGn', aspect='auto', origin='lower', vmin=0, vmax=1)
    ax.set_title('Quality-Diversity (MAP-Elites)\n(archive of elites covering behaviour space)', fontsize=9)
    ax.set_xlabel('Descriptor 1 (bin)', fontsize=8)
    ax.set_ylabel('Descriptor 2 (bin)', fontsize=8)
    plt.colorbar(im, ax=ax, label='Fitness', fraction=0.046, pad=0.04)

    fig.suptitle('Three paradigms for solution discovery', fontsize=11)
    plt.tight_layout()
    save(fig, 'fig_qd_concept.pdf')


# ---------------------------------------------------------------------------
# 10. Chromosome structure
# ---------------------------------------------------------------------------
def fig_chromosome():
    genes = ['5,0,V', '2,0,B', '4,0,B', '8,0,B', '1,0,B', '7,1,B', '3,0,B', '6,0,B']
    gene_colors = ['#aed6f1'] + ['#a9dfbf'] * 3 + ['#a9dfbf'] + ['#f9e79f'] + ['#a9dfbf'] * 2

    fig, ax = plt.subplots(figsize=(12, 2.0))
    ax.axis('off')
    ax.set_xlim(-0.5, len(genes) + 0.5)
    ax.set_ylim(-0.1, 1.5)

    for i, (gene, col) in enumerate(zip(genes, gene_colors)):
        rect = mpatches.FancyBboxPatch(
            (i + 0.05, 0.3), 0.88, 0.8,
            boxstyle='round,pad=0.06',
            facecolor=col, edgecolor='#555', linewidth=1.2)
        ax.add_patch(rect)
        ax.text(i + 0.49, 0.70, gene, ha='center', va='center',
                fontsize=9.5, fontweight='bold')
        parts = gene.split(',')
        mode_str = 'Van' if parts[2] == 'V' else 'Bike'
        new_str  = 'new route' if parts[1] == '1' else ''
        ax.text(i + 0.49, 0.18, f'C{parts[0]} {mode_str} {new_str}',
                ha='center', va='center', fontsize=7, color='#333')

    ax.text(-0.4, 0.70, 'Gene:', ha='right', va='center', fontsize=9, style='italic')
    ax.text(-0.4, 0.18, '(custID,\nnewRoute,\nmode)', ha='right', va='center',
            fontsize=7, color='#555')
    ax.set_title('Example chromosome for 8 customers — gene = (customerID, newRoute?, mode)',
                 fontsize=10, pad=4)
    save(fig, 'fig_chromosome.pdf')


# ---------------------------------------------------------------------------
# 11-13. Crop figures from the book PDF
# ---------------------------------------------------------------------------
def crop_pdf_figures():
    try:
        import fitz
    except ImportError:
        print('  pymupdf not found - using placeholder figures.')
        for name, text in [
            ('fig_book_fig6_1.pdf', 'Fig. 6.1  Class diagram (supermarket MAP-Elites implementation)'),
            ('fig_book_fig6_2.pdf', 'Fig. 6.2  X and Y characteristics: full range vs. reduced upper bound of Y'),
            ('fig_book_fig6_3.pdf', 'Fig. 6.3  Parallel coordinates archive (B-n67-k10, 1h windows)'),
        ]:
            _placeholder(name, text)
        return

    pdf = fitz.open(PDF_PATH)

    # Fig 6.1 - class diagram  (book page 123, 0-indexed 122)
    _crop_page(pdf, 122, (0.05, 0.05, 0.95, 0.58),
               'fig_book_fig6_1.pdf',
               'Fig. 6.1  Supermarket delivery MAP-Elites implementation class diagram')

    # Fig 6.2 - seeding range figure  (book page 126, 0-indexed 125)
    _crop_page(pdf, 125, (0.04, 0.04, 0.96, 0.54),
               'fig_book_fig6_2.pdf',
               'Fig. 6.2  X and Y characteristics: full range 1-10 (left) vs. reduced upper bound of Y (right)')

    # Fig 6.3 - parallel coordinates  (book page 134, 0-indexed 133)
    _crop_page(pdf, 133, (0.02, 0.01, 0.98, 0.90),
               'fig_book_fig6_3.pdf',
               'Fig. 6.3  Archive as parallel coordinates (B-n67-k10, 1h time windows)')

    pdf.close()


def _crop_page(pdf, page_idx, rect_frac, out_name, caption):
    import fitz
    page = pdf[page_idx]
    pw, ph = page.rect.width, page.rect.height
    x0, y0, x1, y1 = rect_frac
    clip = fitz.Rect(pw * x0, ph * y0, pw * x1, ph * y1)
    pix  = page.get_pixmap(matrix=fitz.Matrix(2.5, 2.5), clip=clip)
    img_path = os.path.join(OUT, out_name.replace('.pdf', '.png'))
    pix.save(img_path)

    img = plt.imread(img_path)
    ratio = img.shape[0] / img.shape[1]
    fig, ax = plt.subplots(figsize=(8, max(ratio * 8, 2)))
    ax.imshow(img)
    ax.axis('off')
    ax.set_title(caption, fontsize=9, pad=4)
    fig.savefig(os.path.join(OUT, out_name), bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f'  saved {os.path.join(OUT, out_name)}')


def _placeholder(name, text):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.axis('off')
    ax.text(0.5, 0.5, text, ha='center', va='center', fontsize=11,
            transform=ax.transAxes,
            bbox=dict(boxstyle='round,pad=0.6', fc='#eef3fb', ec='steelblue', lw=1.5))
    save(fig, name)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    print('Generating figures for Chapter 6 ...')
    fig_solution_scatter()
    fig_archive_empty()
    fig_archive_filled()
    fig_scaling_formula()
    fig_map_elites_flow()
    fig_seeding_effect()
    fig_parallel_coords()
    fig_coverage_bar()
    fig_qd_concept()
    fig_chromosome()
    crop_pdf_figures()
    print('Done.')
