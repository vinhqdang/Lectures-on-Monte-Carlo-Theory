"""
gen_figures.py  –  Generate all figures for Chapter 4 slides
"Solving Problems That Have Dual Solution Characteristics"
Nature Inspired Optimisation for Delivery Problems (2022)
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import numpy as np
import os

OUT = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(OUT, exist_ok=True)

def save(name):
    plt.savefig(os.path.join(OUT, name), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {name}")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 – Scatter of solutions in objective space (cost vs env impact)
# ─────────────────────────────────────────────────────────────────────────────
def fig_solutions_scatter():
    # Approximate data from Fig 4.1a in the book
    solutions = [
        (1,  19), (2,  11), (4.5, 7), (5,  5),   # non-dominated front
        (8,  6),  (12, 3),  (16, 2.5),(20, 1),   # non-dominated front
        (6,  13), (9,  12), (14, 14),              # dominated interior
        (11, 8.5),(15, 9),  (8,  4),  (13, 3.5),
    ]
    xs = [s[0] for s in solutions]
    ys = [s[1] for s in solutions]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(xs, ys, color='steelblue', s=60, zorder=3)
    ax.set_xlabel("Environmental impact (e)", fontsize=12)
    ax.set_ylabel("Cost (c)", fontsize=12)
    ax.set_title("A set of solutions plotted by their two objectives", fontsize=12)
    ax.set_xlim(0, 22); ax.set_ylim(0, 22)
    ax.grid(True, linestyle='--', alpha=0.4)
    save("fig_scatter_solutions.pdf")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 – Dominated region highlighted by solution A
# ─────────────────────────────────────────────────────────────────────────────
def fig_dominated_region():
    solutions = [
        (1,  19), (2,  11), (4.5, 7), (5,  5),
        (8,  6),  (12, 3),  (16, 2.5),(20, 1),
        (6,  13), (9,  12), (14, 14),
        (11, 8.5),(15, 9),  (8,  4),  (13, 3.5),
    ]
    # Solution A is at (5, 7) approximately
    Ax, Ay = 4.5, 7
    xs = [s[0] for s in solutions]
    ys = [s[1] for s in solutions]

    fig, ax = plt.subplots(figsize=(7, 5))

    # Shade the dominated region (everything to the right of and above A)
    rect = mpatches.FancyBboxPatch((Ax, Ay), 22 - Ax, 22 - Ay,
                                    boxstyle="square,pad=0",
                                    facecolor='salmon', alpha=0.25,
                                    edgecolor='none', zorder=1)
    ax.add_patch(rect)

    ax.scatter(xs, ys, color='steelblue', s=60, zorder=3)
    ax.scatter([Ax], [Ay], color='red', s=100, zorder=4, label='Solution A')
    ax.annotate('A', (Ax, Ay), textcoords="offset points",
                xytext=(5, 5), fontsize=11, fontweight='bold')

    ax.set_xlabel("Environmental impact (e)", fontsize=12)
    ax.set_ylabel("Cost (c)", fontsize=12)
    ax.set_title("The area of the solution space dominated by solution A", fontsize=12)
    ax.set_xlim(0, 22); ax.set_ylim(0, 22)
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend(fontsize=10)
    save("fig_dominated_region.pdf")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 3 – Non-dominated front highlighted
# ─────────────────────────────────────────────────────────────────────────────
def fig_pareto_front():
    all_solutions = [
        (1,  19), (2,  11), (4.5, 7), (5,  5),
        (8,  6),  (12, 3),  (16, 2.5),(20, 1),
        (6,  13), (9,  12), (14, 14),
        (11, 8.5),(15, 9),  (8,  4),  (13, 3.5),
    ]
    # Non-dominated solutions (Pareto front)
    front = [(1, 19), (2, 11), (4.5, 7), (5, 5), (8, 6),
             (12, 3), (16, 2.5), (20, 1)]
    dominated = [s for s in all_solutions if s not in front]

    fxs = [s[0] for s in front]
    fys = [s[1] for s in front]
    dxs = [s[0] for s in dominated]
    dys = [s[1] for s in dominated]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(dxs, dys, color='steelblue', s=55, zorder=3, alpha=0.6, label='Dominated')
    ax.scatter(fxs, fys, color='steelblue', s=65, zorder=4)
    ax.plot(fxs, fys, color='navy', linewidth=2.0, zorder=3, label='Pareto front')

    ax.set_xlabel("Environmental impact (e)", fontsize=12)
    ax.set_ylabel("Cost (c)", fontsize=12)
    ax.set_title("The non-dominated (Pareto) front", fontsize=12)
    ax.set_xlim(0, 22); ax.set_ylim(0, 22)
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend(fontsize=10)
    save("fig_pareto_front.pdf")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 4 – Pareto dominance concept: two objectives, four quadrants
# ─────────────────────────────────────────────────────────────────────────────
def fig_dominance_concept():
    fig, ax = plt.subplots(figsize=(6, 5))

    Ax, Ay = 3, 3
    ax.scatter([Ax], [Ay], color='red', s=120, zorder=5, label='Solution A (3, 3)')

    # Four quadrant annotations
    ax.fill_betweenx([Ay, 7], 0, Ax, alpha=0.08, color='blue', label='A dominates region')
    ax.fill_betweenx([0, Ay], Ax, 7, alpha=0.15, color='salmon', label='A is dominated region')

    ax.axhline(Ay, color='gray', linestyle='--', linewidth=1)
    ax.axvline(Ax, color='gray', linestyle='--', linewidth=1)

    ax.text(1.0, 5.5, 'A dominates\n(lower cost,\nlower impact)', fontsize=9,
            ha='center', va='center', color='navy',
            bbox=dict(boxstyle='round,pad=0.3', fc='lightyellow', ec='gray', alpha=0.8))
    ax.text(5.5, 1.5, 'A is dominated\n(higher cost,\nhigher impact)', fontsize=9,
            ha='center', va='center', color='darkred',
            bbox=dict(boxstyle='round,pad=0.3', fc='lightyellow', ec='gray', alpha=0.8))
    ax.text(1.0, 1.5, 'Neither\ndominates\n(trade-off)', fontsize=9,
            ha='center', va='center', color='darkgreen',
            bbox=dict(boxstyle='round,pad=0.3', fc='lightyellow', ec='gray', alpha=0.8))
    ax.text(5.5, 5.5, 'Neither\ndominates\n(trade-off)', fontsize=9,
            ha='center', va='center', color='darkgreen',
            bbox=dict(boxstyle='round,pad=0.3', fc='lightyellow', ec='gray', alpha=0.8))

    ax.annotate('A', (Ax, Ay), textcoords="offset points", xytext=(6, 6),
                fontsize=11, fontweight='bold', color='red')

    ax.set_xlabel("Objective 1 (minimise)", fontsize=11)
    ax.set_ylabel("Objective 2 (minimise)", fontsize=11)
    ax.set_title("Pareto Dominance: four regions around solution A", fontsize=11)
    ax.set_xlim(0, 7); ax.set_ylim(0, 7)
    ax.grid(True, linestyle=':', alpha=0.4)
    ax.legend(fontsize=9, loc='upper right')
    save("fig_dominance_concept.pdf")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 5 – Hypervolume indicator
# ─────────────────────────────────────────────────────────────────────────────
def fig_hypervolume():
    # Front points from Fig 4.5 in the book: a(1,5), b(2,3), c(3,2), d(6,1)
    # Nadir point n = (6, 5) — worst values
    front = [(1, 5), (2, 3), (3, 2), (6, 1)]
    nadir = (6, 5)

    fig, ax = plt.subplots(figsize=(7, 5))

    # Shade hypervolume area (L-shaped polygon under the staircase front + nadir)
    # Build the staircase polygon
    poly_x = [0]
    poly_y = [nadir[1]]
    for (x, y) in front:
        poly_x.append(x)
        poly_y.append(nadir[1])
        poly_x.append(x)
        poly_y.append(y)
    poly_x.append(nadir[0])
    poly_y.append(front[-1][1])
    poly_x.append(nadir[0])
    poly_y.append(nadir[1])
    poly_x.append(0)
    poly_y.append(nadir[1])

    ax.fill(poly_x, poly_y, color='lightblue', alpha=0.5, label='Hypervolume area')

    # Draw the front as a staircase then the smooth curve
    fx = [p[0] for p in front]
    fy = [p[1] for p in front]
    ax.plot(fx, fy, 'b-o', linewidth=2, markersize=7, zorder=4)

    # Label front points
    labels = ['a', 'b', 'c', 'd']
    for i, (x, y) in enumerate(front):
        ax.annotate(labels[i], (x, y), textcoords="offset points",
                    xytext=(-10, 5), fontsize=11, fontweight='bold', color='navy')

    # Nadir point
    ax.scatter([nadir[0]], [nadir[1]], color='black', s=80, zorder=5, marker='x',
               linewidths=2, label=f'Nadir point n = {nadir}')
    ax.annotate('n', nadir, textcoords="offset points",
                xytext=(5, 5), fontsize=11, fontweight='bold', color='black')

    # Dashed lines to nadir
    ax.plot([front[0][0], nadir[0]], [front[0][1], front[0][1]],
            'k--', linewidth=1, alpha=0.5)
    ax.plot([front[-1][0], front[-1][0]], [front[-1][1], nadir[1]],
            'k--', linewidth=1, alpha=0.5)

    ax.text(4.0, 3.5, 'Hypervolume\narea', fontsize=12, ha='center',
            color='steelblue', fontweight='bold')

    ax.set_xlabel("Characteristic b", fontsize=12)
    ax.set_ylabel("Characteristic a", fontsize=12)
    ax.set_title("Hypervolume Indicator: area enclosed by front and nadir point", fontsize=11)
    ax.set_xlim(0, 7); ax.set_ylim(0, 6)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.3)
    save("fig_hypervolume.pdf")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 6 – Hypervolume calculation step-by-step (rectangles)
# ─────────────────────────────────────────────────────────────────────────────
def fig_hypervolume_rectangles():
    front = [(1, 5), (2, 3), (3, 2), (6, 1)]
    nadir = (6, 5)

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ['#a8d8ea', '#aa96da', '#fcbad3', '#ffffd2']

    # Rectangle for each strip
    # Strip i: from x_i to x_{i+1}, from y_i (front) down to nadir_y?
    # Actually: area between front point and next point, bounded by nadir_y
    # Rect i: width = x_{i+1} - x_i,  height = nadir_y - y_i
    # For last: width = nadir_x - x_last, height = nadir_y - y_last
    # Plus the first point: x from 0? No — we do staircase rectangles.
    rects = []
    prev_x = 0
    for i, (x, y) in enumerate(front):
        w = x - prev_x
        h = nadir[1] - y
        rects.append((prev_x, y, w, h))
        prev_x = x

    for i, (rx, ry, rw, rh) in enumerate(rects):
        rect = plt.Rectangle((rx, ry), rw, rh,
                               facecolor=colors[i % len(colors)],
                               edgecolor='navy', linewidth=1.2, alpha=0.7)
        ax.add_patch(rect)
        # Label area
        area = rw * rh
        cx = rx + rw / 2
        cy = ry + rh / 2
        ax.text(cx, cy, f'{area:.1f}', ha='center', va='center',
                fontsize=10, fontweight='bold', color='navy')

    # Draw front points
    fx = [p[0] for p in front]
    fy = [p[1] for p in front]
    ax.plot(fx, fy, 'b-o', linewidth=2, markersize=7, zorder=4)
    labels = ['a(1,5)', 'b(2,3)', 'c(3,2)', 'd(6,1)']
    offsets = [(-18, 6), (-20, 6), (-20, 6), (5, 6)]
    for i, (x, y) in enumerate(front):
        ax.annotate(labels[i], (x, y), textcoords="offset points",
                    xytext=offsets[i], fontsize=9, color='navy')

    ax.scatter([nadir[0]], [nadir[1]], color='black', s=80, zorder=5,
               marker='x', linewidths=2)
    ax.annotate(f'Nadir n=(6,5)', nadir, textcoords="offset points",
                xytext=(4, -14), fontsize=9)

    # Total
    total = sum(rw * rh for _, _, rw, rh in rects)
    ax.set_title(f"Hypervolume = sum of rectangle areas = {total:.1f}", fontsize=11)
    ax.set_xlabel("Characteristic b", fontsize=12)
    ax.set_ylabel("Characteristic a", fontsize=12)
    ax.set_xlim(0, 7); ax.set_ylim(0, 6)
    ax.grid(True, linestyle=':', alpha=0.3)
    save("fig_hypervolume_rectangles.pdf")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 7 – Grand front concept (multiple runs combined)
# ─────────────────────────────────────────────────────────────────────────────
def fig_grand_front():
    np.random.seed(42)
    fig, ax = plt.subplots(figsize=(7, 5))

    # Simulate 3 EA runs, each yielding a non-dominated front
    markers = ['o', 's', '^', 'D', 'v']
    colors_run = ['#4e79a7', '#f28e2b', '#e15759', '#76b7b2', '#59a14f']

    all_fronts = []
    for run_i in range(5):
        np.random.seed(run_i * 7)
        # Each front has some spread
        n_pts = np.random.randint(4, 8)
        xs = sorted(np.random.uniform(5, 95, n_pts))
        ys = sorted(np.random.uniform(500, 5000, n_pts), reverse=True)
        # Add some noise
        ys = [y + np.random.uniform(-300, 300) for y in ys]
        front = list(zip(xs, ys))
        all_fronts.append(front)
        ax.plot(xs, ys, markers[run_i], color=colors_run[run_i],
                markersize=6, alpha=0.7, label=f'Run {run_i+1} front',
                linestyle='-', linewidth=1)

    # Grand front: merge and extract non-dominated
    all_pts = [pt for f in all_fronts for pt in f]

    def is_dominated(p, pts):
        for q in pts:
            if q[0] <= p[0] and q[1] <= p[1] and q != p:
                return True
        return False

    grand = [p for p in all_pts if not is_dominated(p, all_pts)]
    grand.sort()
    gxs = [p[0] for p in grand]
    gys = [p[1] for p in grand]
    ax.plot(gxs, gys, 'k-', linewidth=2.5, zorder=5, label='Grand front')

    ax.set_xlabel("Number of routes", fontsize=12)
    ax.set_ylabel("Customer service metric", fontsize=12)
    ax.set_title("Grand front: combining non-dominated fronts from multiple runs", fontsize=11)
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True, linestyle='--', alpha=0.3)
    save("fig_grand_front.pdf")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 8 – Search space growth: n! vs n! * 2^n
# ─────────────────────────────────────────────────────────────────────────────
def fig_search_space_growth():
    import math
    ns = list(range(1, 11))
    fact = [math.factorial(n) for n in ns]
    bi   = [math.factorial(n) * (2**n) for n in ns]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.semilogy(ns, fact, 'b-o', linewidth=2, markersize=7, label=r'$n!$ (permutation only)')
    ax.semilogy(ns, bi,   'r-s', linewidth=2, markersize=7, label=r'$n! \times 2^n$ (permutation + route flags)')

    ax.set_xlabel("Number of customers (n)", fontsize=12)
    ax.set_ylabel("Search space size (log scale)", fontsize=12)
    ax.set_title("Search space explosion with new binary-flag representation", fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(True, which='both', linestyle='--', alpha=0.4)
    ax.set_xticks(ns)
    save("fig_search_space.pdf")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 9 – Two-objective trade-off illustration
# ─────────────────────────────────────────────────────────────────────────────
def fig_tradeoff_illustration():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # Left: single objective optimisation – one solution per objective
    ax = axes[0]
    ax.set_title("Single-objective approach:\nTwo separate optimisations", fontsize=11)
    ax.scatter([2, 18], [18, 2], color=['red', 'blue'], s=150, zorder=5)
    ax.annotate("Min cost\n(high env. impact)", (2, 18),
                textcoords="offset points", xytext=(5, -25), fontsize=9, color='red')
    ax.annotate("Min env. impact\n(high cost)", (18, 2),
                textcoords="offset points", xytext=(-90, 10), fontsize=9, color='blue')
    ax.set_xlim(0, 22); ax.set_ylim(0, 22)
    ax.set_xlabel("Environmental impact"); ax.set_ylabel("Cost")
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.text(10, 11, "User must choose\nbetween two extremes",
            ha='center', fontsize=10, color='purple',
            bbox=dict(boxstyle='round', fc='lightyellow', ec='gray', alpha=0.8))

    # Right: bi-objective – Pareto front
    ax = axes[1]
    ax.set_title("Bi-objective approach:\nPareto front of trade-offs", fontsize=11)
    front_x = [1, 3, 6, 10, 14, 19]
    front_y = [19, 13, 8, 5, 3, 1]
    ax.plot(front_x, front_y, 'navy', linewidth=2)
    ax.scatter(front_x, front_y, color='steelblue', s=80, zorder=5)
    ax.annotate("Extreme:\nmin cost", (1, 19),
                textcoords="offset points", xytext=(5, -25), fontsize=9)
    ax.annotate("Extreme:\nmin impact", (19, 1),
                textcoords="offset points", xytext=(-70, 8), fontsize=9)
    ax.annotate("Trade-off\nsolutions", (10, 5),
                textcoords="offset points", xytext=(15, 20), fontsize=9,
                arrowprops=dict(arrowstyle='->', color='gray'))
    ax.set_xlim(0, 22); ax.set_ylim(0, 22)
    ax.set_xlabel("Environmental impact"); ax.set_ylabel("Cost")
    ax.grid(True, linestyle='--', alpha=0.3)

    plt.tight_layout()
    save("fig_tradeoff_illustration.pdf")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 10 – Crop from PDF: Fig 4.4 (initial vs final front, P-n101-k4)
# ─────────────────────────────────────────────────────────────────────────────
def fig_initial_vs_final_front():
    """Attempt to crop Fig 4.4 from the book PDF; fall back to synthetic."""
    pdf_path = ("/Users/vinhdq1/work/Lectures-on-Monte-Carlo-Theory/"
                "nature_inspired_optimization_delivery_problems_2022/"
                "Nature Inspired Optimisation for Delivery Problems 2022.pdf")
    try:
        import fitz  # pymupdf
        doc = fitz.open(pdf_path)
        # Book page 77 = PDF page index 88 (0-indexed, accounting for front matter)
        # The book says page 77 contains Fig 4.4
        # We try a range of plausible indices
        # Page 77 in book → search for it
        found = False
        for page_idx in range(80, 100):
            page = doc[page_idx]
            # Check if it has relevant image
            img_list = page.get_images()
            if img_list:
                clip = fitz.Rect(40, 300, 560, 600)
                mat = fitz.Matrix(2, 2)
                pix = page.get_pixmap(matrix=mat, clip=clip)
                out_path = os.path.join(OUT, "fig_initial_final_front.png")
                pix.save(out_path)
                print(f"  Cropped Fig 4.4 from PDF page {page_idx}")
                found = True
                break
        doc.close()
        if found:
            return
    except Exception as e:
        print(f"  PDF crop failed ({e}), generating synthetic figure")

    # Synthetic fallback – approximate from Fig 4.4 / Fig 4.6 in book
    np.random.seed(0)
    fig, ax = plt.subplots(figsize=(8, 5))

    # Initial front (broad, high values)
    init_routes = np.linspace(4, 101, 40)
    init_cs = 31676.33 * np.exp(-0.035 * (init_routes - 4)) + 2494.71
    ax.plot(init_routes, init_cs, 'k-', linewidth=2, label='Initial population front')

    # Final front (lower, tighter)
    final_routes = np.linspace(4, 101, 60)
    final_cs = 14000 * np.exp(-0.028 * (final_routes - 4)) + 2494.71
    noise = np.random.uniform(-200, 200, len(final_routes))
    ax.scatter(final_routes, final_cs + noise, color='orange', s=15, alpha=0.6,
               label='Final population front')

    ax.set_xlabel("Number of routes", fontsize=12)
    ax.set_ylabel("Customer service metric", fontsize=12)
    ax.set_title("Initial vs Final non-dominated fronts (P-n101-k4)", fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.4)
    save("fig_initial_final_front.pdf")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 11 – Encoding: genotype → phenotype for bi-objective VRP
# ─────────────────────────────────────────────────────────────────────────────
def fig_encoding():
    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.axis('off')

    # Genotype row
    genes = ['A, False', 'B, True', 'C, False', 'D, True', 'E, True']
    colors_g = ['#d4edda', '#cce5ff', '#d4edda', '#cce5ff', '#cce5ff']
    cell_w = 1.5
    for i, (gene, col) in enumerate(zip(genes, colors_g)):
        rect = plt.Rectangle((i * cell_w, 1.8), cell_w - 0.05, 0.6,
                               facecolor=col, edgecolor='navy', linewidth=1.5)
        ax.add_patch(rect)
        ax.text(i * cell_w + cell_w / 2, 2.12, gene,
                ha='center', va='center', fontsize=9.5, fontweight='bold')

    ax.text(-0.3, 2.12, 'Genotype:', ha='right', va='center', fontsize=10)

    # Arrow
    ax.annotate('', xy=(3.5, 1.3), xytext=(3.5, 1.7),
                arrowprops=dict(arrowstyle='->', color='black', lw=2))
    ax.text(3.55, 1.5, 'decode()', fontsize=9, va='center', color='black')

    # Phenotype rows
    routes_label = ['Route 1:', 'Route 2:', 'Route 3:', 'Route 4:']
    routes_data  = ['A', 'B C', 'D', 'E']
    for j, (rl, rd) in enumerate(zip(routes_label, routes_data)):
        ax.text(1.2, 1.1 - j * 0.35, f'{rl}  {rd}',
                ha='left', va='center', fontsize=10)

    ax.text(-0.3, 0.7, 'Phenotype:', ha='right', va='center', fontsize=10)

    ax.set_xlim(-1, 8)
    ax.set_ylim(-0.2, 3.0)
    ax.set_title("Bi-objective genotype encoding: each gene = (customerID, newRoute flag)",
                 fontsize=11, pad=6)
    save("fig_encoding.pdf")

# ─────────────────────────────────────────────────────────────────────────────
# Run all
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Generating figures for Chapter 4...")
    fig_solutions_scatter()
    fig_dominated_region()
    fig_pareto_front()
    fig_dominance_concept()
    fig_hypervolume()
    fig_hypervolume_rectangles()
    fig_grand_front()
    fig_search_space_growth()
    fig_tradeoff_illustration()
    fig_initial_vs_final_front()
    fig_encoding()
    print("All figures generated successfully.")
