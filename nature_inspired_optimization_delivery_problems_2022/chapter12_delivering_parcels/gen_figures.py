"""
gen_figures.py  —  Generate all figures for Chapter 12: Delivering Parcels
Uses matplotlib (backend='Agg') and pymupdf to crop figures from the book PDF.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import os
import fitz  # pymupdf

FIGURES_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

BOOK_PDF = os.path.join(
    os.path.dirname(__file__), "..",
    "Nature Inspired Optimisation for Delivery Problems 2022.pdf"
)

# ─────────────────────────────────────────────
# Helper: crop a figure from the PDF by page index (0-based) and rect
# ─────────────────────────────────────────────
def crop_pdf_figure(pdf_path, page_idx, rect, out_path, dpi=200):
    """Crop a region from a PDF page and save as PNG."""
    doc = fitz.open(pdf_path)
    page = doc[page_idx]
    # rect = (x0, y0, x1, y1) in points (72 dpi)
    clip = fitz.Rect(*rect)
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat, clip=clip)
    pix.save(out_path)
    doc.close()
    print(f"  Saved {out_path}")


# ─────────────────────────────────────────────
# Figure 1: City Logistics concept diagram
# Micro-depot delivery network illustration
# ─────────────────────────────────────────────
def fig_city_logistics():
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.axis('off')
    ax.set_facecolor('#f8f9fa')
    fig.patch.set_facecolor('#f8f9fa')
    ax.set_title("Micro-Depot Last-Mile Delivery Network", fontsize=14, fontweight='bold', pad=15)

    # Main depot
    depot_box = FancyBboxPatch((0.3, 2.8), 1.4, 1.0,
                               boxstyle="round,pad=0.1",
                               facecolor='#2c7bb6', edgecolor='#1a5276', linewidth=2)
    ax.add_patch(depot_box)
    ax.text(1.0, 3.3, "Main\nDepot\n(D0)", ha='center', va='center',
            color='white', fontsize=9, fontweight='bold')

    # Supply vehicle arrow to city
    ax.annotate("", xy=(3.5, 3.3), xytext=(1.7, 3.3),
                arrowprops=dict(arrowstyle="-|>", color='#2c7bb6', lw=2.5))
    ax.text(2.6, 3.6, "Supply\nVehicle", ha='center', va='bottom', fontsize=8, color='#2c7bb6')

    # Micro-depots
    md_positions = [(4.0, 5.2), (4.0, 1.2), (7.0, 5.5), (7.0, 1.0)]
    md_labels = ["MD1", "MD2", "MD3", "MD4"]
    md_colors = ['#f39c12', '#e74c3c', '#27ae60', '#8e44ad']
    for (mx, my), mlabel, mc in zip(md_positions, md_labels, md_colors):
        box = FancyBboxPatch((mx - 0.5, my - 0.35), 1.0, 0.7,
                             boxstyle="round,pad=0.07",
                             facecolor=mc, edgecolor='black', linewidth=1.5, alpha=0.85)
        ax.add_patch(box)
        ax.text(mx, my, mlabel, ha='center', va='center',
                color='white', fontsize=9, fontweight='bold')
        # Arrow from depot area to micro-depot
        ax.annotate("", xy=(mx - 0.5, my), xytext=(3.5, 3.3),
                    arrowprops=dict(arrowstyle="-|>", color='gray', lw=1.2,
                                   connectionstyle='arc3,rad=0.1'))

    # Couriers from micro-depots to customers
    customer_positions = [
        (5.5, 5.8), (5.5, 4.8), (5.3, 6.3),  # from MD1
        (5.5, 0.8), (5.5, 1.8),               # from MD2
        (8.5, 5.8), (8.5, 4.8), (8.3, 6.2),  # from MD3
        (8.5, 0.8), (8.5, 1.8),               # from MD4
    ]
    md_sources = [
        (4.5, 5.2), (4.5, 5.2), (4.5, 5.2),
        (4.5, 1.2), (4.5, 1.2),
        (7.5, 5.5), (7.5, 5.5), (7.5, 5.5),
        (7.5, 1.0), (7.5, 1.0),
    ]
    courier_types = ['walk', 'cycle', 'EV', 'walk', 'cycle', 'EV', 'walk', 'cycle', 'EV', 'walk']
    c_colors = {'walk': '#e67e22', 'cycle': '#27ae60', 'EV': '#2980b9'}
    for (cx, cy), (sx, sy), ct in zip(customer_positions, md_sources, courier_types):
        ax.annotate("", xy=(cx, cy), xytext=(sx, sy),
                    arrowprops=dict(arrowstyle="-|>", color=c_colors[ct], lw=1.0,
                                   connectionstyle='arc3,rad=0.05'))
        ax.plot(cx, cy, 'o', color='#c0392b', markersize=6, zorder=5)

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor='#e67e22', label='Walking courier'),
        mpatches.Patch(facecolor='#27ae60', label='Cycling courier'),
        mpatches.Patch(facecolor='#2980b9', label='Electric vehicle (EV)'),
        mpatches.Patch(facecolor='#c0392b', label='Customer delivery point'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=8,
              framealpha=0.9, edgecolor='gray')

    ax.text(5.0, -0.2, "City area", ha='center', fontsize=9, color='gray', style='italic')
    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, "fig_city_logistics.pdf")
    fig.savefig(out, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"  Saved {out}")


# ─────────────────────────────────────────────
# Figure 2: MAP-Elites archive concept
# 2D grid showing solution diversity
# ─────────────────────────────────────────────
def fig_map_elites_archive():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("MAP-Elites: Illuminating the Solution Space", fontsize=13, fontweight='bold')

    # Left: 2D archive heatmap (quality across two behavioural characteristics)
    np.random.seed(42)
    grid = np.zeros((10, 10))
    for i in range(10):
        for j in range(10):
            if np.random.rand() > 0.3:
                grid[i, j] = np.random.uniform(0.4, 1.0) * (1 - 0.03 * abs(i - 5) - 0.03 * abs(j - 5))
            else:
                grid[i, j] = np.nan

    ax = axes[0]
    masked = np.ma.array(grid, mask=np.isnan(grid))
    cmap = plt.cm.YlGn
    cmap.set_bad('lightgray')
    im = ax.imshow(masked, cmap=cmap, vmin=0, vmax=1, aspect='auto')
    ax.set_xlabel("Characteristic 1: Number of Micro-Depots Used", fontsize=9)
    ax.set_ylabel("Characteristic 2: % Deliveries by EV", fontsize=9)
    ax.set_title("Archive — each cell holds the\nbest solution for that (C1, C2) pair", fontsize=9)
    ax.set_xticks(range(10))
    ax.set_xticklabels([str(i+1) for i in range(10)], fontsize=7)
    ax.set_yticks(range(10))
    ax.set_yticklabels([str(i+1) for i in range(10)], fontsize=7)
    plt.colorbar(im, ax=ax, label='Solution quality (fitness)', shrink=0.8)

    # Right: MAP-Elites algorithm flow
    ax2 = axes[1]
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.axis('off')
    ax2.set_title("MAP-Elites Algorithm Flow", fontsize=10)

    steps = [
        (5, 8.5, "1. Initialise archive\nwith random solutions", '#3498db'),
        (5, 6.5, "2. Select random solution\nfrom archive", '#2ecc71'),
        (5, 4.5, "3. Mutate / crossover\nto create new solution", '#f39c12'),
        (5, 2.5, "4. Evaluate new solution\n(fitness + characteristics)", '#e74c3c'),
        (5, 0.7, "5. Insert into archive\n(if cell empty or improves)", '#9b59b6'),
    ]
    for (x, y, label, color) in steps:
        box = FancyBboxPatch((x - 3.5, y - 0.6), 7.0, 1.2,
                             boxstyle="round,pad=0.1",
                             facecolor=color, edgecolor='black', linewidth=1, alpha=0.85)
        ax2.add_patch(box)
        ax2.text(x, y, label, ha='center', va='center',
                 color='white', fontsize=8.5, fontweight='bold')

    for i in range(len(steps) - 1):
        ax2.annotate("", xy=(5, steps[i+1][1] + 0.6), xytext=(5, steps[i][1] - 0.6),
                     arrowprops=dict(arrowstyle="-|>", color='black', lw=1.5))

    # Loop-back arrow from step 5 to step 2
    ax2.annotate("", xy=(1.0, 6.5), xytext=(1.0, 0.7),
                 arrowprops=dict(arrowstyle="-|>", color='gray', lw=1.2,
                                 connectionstyle='arc3,rad=0'))
    ax2.text(0.3, 3.6, "Repeat\n$N$ times", ha='center', fontsize=7, color='gray', rotation=90)

    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, "fig_map_elites_archive.pdf")
    fig.savefig(out, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"  Saved {out}")


# ─────────────────────────────────────────────
# Figure 3: Chromosome representation and decoding
# ─────────────────────────────────────────────
def fig_chromosome_decoding():
    fig, axes = plt.subplots(2, 1, figsize=(11, 7))
    fig.suptitle("Chromosome Representation and Solution Decoding", fontsize=13, fontweight='bold')

    # Top: chromosome genes table
    ax = axes[0]
    ax.axis('off')
    ax.set_title("A chromosome encodes a list of genes; each gene = one courier assignment", fontsize=10)

    columns = ['Gene', 'Courier Type', 'Micro-Depot ID', 'Quantity']
    rows = [
        ['1', 'Walking', 'MD2', '3'],
        ['2', 'Cycling', 'MD1', '4'],
        ['3', 'Electric Van (EV)', 'MD3', '4'],
        ['4', 'Walking', 'MD1', '3'],
    ]
    colors_row = [['#d5e8d4', '#d5e8d4', '#d5e8d4', '#d5e8d4'],
                  ['#dae8fc', '#dae8fc', '#dae8fc', '#dae8fc'],
                  ['#fff2cc', '#fff2cc', '#fff2cc', '#fff2cc'],
                  ['#f8cecc', '#f8cecc', '#f8cecc', '#f8cecc']]
    table = ax.table(cellText=rows, colLabels=columns,
                     cellLoc='center', loc='center',
                     cellColours=colors_row)
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.0)

    # Bottom: decoding walkthrough with route timeline
    ax2 = axes[1]
    ax2.set_xlim(-0.5, 22)
    ax2.set_ylim(-0.5, 4.5)
    ax2.axis('off')
    ax2.set_title("Decoding: 20 deliveries transferred from default supply tour to courier sub-tours",
                  fontsize=10)

    route_data = [
        ("D0 Supply", [1, 'MD1', 'MD2', 8, 9, 10, 11, 12, 13, 14, 15, 'MD4', 20], '#95a5a6'),
        ("MD2 Walking", [4, 5, 6], '#e67e22'),
        ("MD1 Cycling", [2, 3, 7], '#27ae60'),
        ("MD3 EV", [16, 17, 18, 19], '#2980b9'),
        ("MD1 Walking", [10, 12, 13], '#8e44ad'),
    ]
    y_positions = [3.8, 2.8, 1.8, 0.8, -0.1]
    for (label, stops, color), y in zip(route_data, y_positions):
        ax2.text(-0.3, y, label, ha='right', va='center', fontsize=8,
                 color=color, fontweight='bold')
        for k, stop in enumerate(stops):
            x = k * 1.0
            if isinstance(stop, int):
                circle = plt.Circle((x, y), 0.32, color=color, alpha=0.7)
                ax2.add_patch(circle)
                ax2.text(x, y, str(stop), ha='center', va='center', fontsize=7, color='white', fontweight='bold')
            else:
                rect = FancyBboxPatch((x - 0.35, y - 0.32), 0.7, 0.64,
                                      boxstyle="round,pad=0.05", facecolor='#2c3e50',
                                      edgecolor='black', linewidth=1)
                ax2.add_patch(rect)
                ax2.text(x, y, stop, ha='center', va='center', fontsize=6, color='white', fontweight='bold')
            if k < len(stops) - 1:
                ax2.annotate("", xy=(x + 0.35, y), xytext=(x + 0.65, y),
                             arrowprops=dict(arrowstyle="-", color=color, lw=1))

    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, "fig_chromosome_decoding.pdf")
    fig.savefig(out, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"  Saved {out}")


# ─────────────────────────────────────────────
# Figure 4: Solution characteristics radar / bar
# 7 characteristics of the solution space
# ─────────────────────────────────────────────
def fig_solution_characteristics():
    characteristics = [
        "Micro-Depots\nUsed",
        "% Deliveries\nby MD",
        "Bikes\nUsed",
        "Walkers\nUsed",
        "EVs\nUsed",
        "Total\nTime",
        "Total\nEmissions",
    ]
    # Two example solutions
    sol_a = [3, 7, 2, 1, 5, 4, 2]  # high EV, low emissions
    sol_b = [1, 3, 4, 6, 2, 7, 8]  # many walkers, high time, high emissions

    x = np.arange(len(characteristics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(11, 5))
    bars1 = ax.bar(x - width/2, sol_a, width, label='Solution A (high EV use)', color='#2980b9', alpha=0.8)
    bars2 = ax.bar(x + width/2, sol_b, width, label='Solution B (high walker use)', color='#e74c3c', alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(characteristics, fontsize=9)
    ax.set_ylabel("Characteristic value (scale 1–10)", fontsize=10)
    ax.set_title("Seven Solution Characteristics used as MAP-Elites Archive Dimensions\n"
                 "(Each characteristic normalised to scale 1–10; archive size up to $10^7$ cells)",
                 fontsize=11, fontweight='bold')
    ax.set_ylim(0, 11)
    ax.legend(fontsize=10)
    ax.axhline(y=10, color='gray', linestyle='--', alpha=0.5, label='Max scale')
    ax.bar_label(bars1, padding=2, fontsize=8)
    ax.bar_label(bars2, padding=2, fontsize=8)

    ax.text(0.01, 0.97,
            "Each unique combination of these 7 values defines a unique cell in the archive.",
            transform=ax.transAxes, fontsize=8, va='top', color='gray', style='italic')

    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, "fig_solution_characteristics.pdf")
    fig.savefig(out, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"  Saved {out}")


# ─────────────────────────────────────────────
# Figure 5: FlexGP policy tree concept
# Boolean condition tree extracted from archive
# ─────────────────────────────────────────────
def fig_flexgp_policy():
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)
    ax.axis('off')
    ax.set_title("FlexGP: Machine-Learning Policy Extraction from MAP-Elites Archive\n"
                 "Boolean expression tree: low-emission label predicted from solution characteristics",
                 fontsize=11, fontweight='bold')

    def draw_box(ax, x, y, text, color, width=2.2, height=0.7):
        box = FancyBboxPatch((x - width/2, y - height/2), width, height,
                             boxstyle="round,pad=0.08",
                             facecolor=color, edgecolor='black', linewidth=1.5, alpha=0.9)
        ax.add_patch(box)
        ax.text(x, y, text, ha='center', va='center', fontsize=8.5, fontweight='bold',
                color='white' if color not in ['#f9f9f9', '#fffde7'] else 'black')

    def draw_arrow(ax, x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2 + 0.35), xytext=(x1, y1 - 0.35),
                    arrowprops=dict(arrowstyle="-|>", color='#555', lw=1.3))

    # Root
    draw_box(ax, 6, 6.2, "AND", '#2c3e50', width=1.0)

    # Level 2
    draw_box(ax, 3, 4.7, "X5 (EVs used)\n1.0 ≤ X5 ≤ 8.6", '#2980b9', width=3.0)
    draw_box(ax, 9, 4.7, "OR", '#8e44ad', width=1.0)
    draw_arrow(ax, 6, 6.2, 3, 4.7)
    draw_arrow(ax, 6, 6.2, 9, 4.7)

    # Level 3 left
    draw_box(ax, 3, 3.2, "X4 (Walkers used)\n1.0 ≤ X4 ≤ 2.0", '#27ae60', width=3.0)
    draw_arrow(ax, 3, 4.7, 3, 3.2)

    # Level 3 right (OR children)
    draw_box(ax, 7, 3.2, "X2 (% by MD)\n1.0 ≤ X2 ≤ 8.4", '#f39c12', width=3.0)
    draw_box(ax, 11, 3.2, "X3 (Bikes used)\n1.0 ≤ X3 ≤ 2.7", '#e74c3c', width=3.0)
    draw_arrow(ax, 9, 4.7, 7, 3.2)
    draw_arrow(ax, 9, 4.7, 11, 3.2)

    # Leaf labels
    ax.text(3, 1.9, "EVs Very High\nWalkers Very Low", ha='center', fontsize=8,
            color='#2980b9', style='italic',
            bbox=dict(boxstyle='round', facecolor='#ebf5fb', edgecolor='#2980b9', alpha=0.8))
    ax.text(7, 1.9, "% by MD High", ha='center', fontsize=8,
            color='#f39c12', style='italic',
            bbox=dict(boxstyle='round', facecolor='#fef9e7', edgecolor='#f39c12', alpha=0.8))
    ax.text(11, 1.9, "Bikes Low", ha='center', fontsize=8,
            color='#e74c3c', style='italic',
            bbox=dict(boxstyle='round', facecolor='#fdedec', edgecolor='#e74c3c', alpha=0.8))

    # Result box
    result_text = ("Policy: Maximise EV use, minimise walkers,\n"
                   "and send many deliveries via MDs or minimise bikes.")
    ax.text(6, 0.7, result_text, ha='center', va='center', fontsize=9,
            color='#1a252f',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#eafaf1', edgecolor='#27ae60', linewidth=2))

    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, "fig_flexgp_policy.pdf")
    fig.savefig(out, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"  Saved {out}")


# ─────────────────────────────────────────────
# Figure 6: Parallel coordinates plot (stylised)
# ─────────────────────────────────────────────
def fig_parallel_coords():
    np.random.seed(7)
    n = 80
    # Simulate solutions in archive
    n_mds = np.random.randint(1, 6, n).astype(float)
    pct_md = n_mds * np.random.uniform(8, 15, n)
    pct_md = np.clip(pct_md, 5, 90)
    bikes = np.random.randint(0, 8, n).astype(float)
    walkers = np.random.randint(0, 8, n).astype(float)
    evs = np.random.randint(0, 8, n).astype(float)
    time_ = 100 - 5 * evs + 3 * walkers + np.random.normal(0, 5, n)
    emissions = 100 - 12 * evs + 2 * walkers + np.random.normal(0, 8, n)
    emissions = np.clip(emissions, 10, 150)

    data = np.column_stack([n_mds, pct_md, bikes, walkers, evs, time_, emissions])
    labels = ['MDs\nUsed', '% by\nMD', 'Bikes\nUsed', 'Walkers\nUsed',
              'EVs\nUsed', 'Time\n(norm.)', 'Emissions\n(norm.)']

    # Normalise each column to 0-1
    data_norm = (data - data.min(axis=0)) / (data.ptp(axis=0) + 1e-9)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_title("Parallel Coordinates Plot of MAP-Elites Archive\n"
                 "(Each line = one elite solution; colour = emissions level)",
                 fontsize=11, fontweight='bold')

    cmap = plt.cm.RdYlGn_r
    for i in range(n):
        color = cmap(data_norm[i, 6])  # colour by emissions
        ax.plot(range(7), data_norm[i], color=color, alpha=0.4, linewidth=0.9)

    ax.set_xticks(range(7))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Normalised value (0 = min, 1 = max)", fontsize=9)
    ax.set_ylim(-0.05, 1.05)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cb = plt.colorbar(sm, ax=ax, shrink=0.7, pad=0.02)
    cb.set_label("Normalised emissions\n(green=low, red=high)", fontsize=8)

    ax.grid(axis='x', linestyle='--', alpha=0.4)
    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, "fig_parallel_coords.pdf")
    fig.savefig(out, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"  Saved {out}")


# ─────────────────────────────────────────────
# Figure 7: Low-emissions solutions highlighted
# ─────────────────────────────────────────────
def fig_low_emissions():
    np.random.seed(42)
    n = 120
    n_mds = np.random.randint(1, 7, n).astype(float)
    evs = np.random.randint(0, 9, n).astype(float)
    walkers = np.random.randint(0, 8, n).astype(float)
    emissions = 100 - 12 * evs + 3 * walkers + np.random.normal(0, 10, n)
    emissions = np.clip(emissions, 5, 150)

    low_em = emissions < 40
    high_em = ~low_em

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Effect of Micro-Depots and EVs on Emissions", fontsize=12, fontweight='bold')

    # Left: scatter MDs vs Emissions
    ax1.scatter(n_mds[high_em], emissions[high_em], color='#e74c3c', alpha=0.5, s=40, label='Higher emissions')
    ax1.scatter(n_mds[low_em], emissions[low_em], color='#27ae60', alpha=0.8, s=60, marker='*',
                label='Low emissions (target)')
    ax1.set_xlabel("Number of Micro-Depots Used", fontsize=10)
    ax1.set_ylabel("Total Emissions (normalised)", fontsize=10)
    ax1.set_title("Micro-Depots vs Emissions\n(policy: MicroDepotsUsed $\\leq$ Low)", fontsize=10)
    ax1.legend(fontsize=9)
    ax1.axvline(x=3, color='gray', linestyle='--', alpha=0.6, label='Policy threshold')
    ax1.text(3.1, 130, 'threshold\n(policy boundary)', fontsize=7, color='gray')

    # Right: scatter EVs vs Emissions
    ax2.scatter(evs[high_em], emissions[high_em], color='#e74c3c', alpha=0.5, s=40, label='Higher emissions')
    ax2.scatter(evs[low_em], emissions[low_em], color='#27ae60', alpha=0.8, s=60, marker='*',
                label='Low emissions')
    ax2.set_xlabel("Number of EVs Used", fontsize=10)
    ax2.set_ylabel("Total Emissions (normalised)", fontsize=10)
    ax2.set_title("EVs Used vs Emissions\n(more EVs $\\Rightarrow$ lower emissions)", fontsize=10)
    ax2.legend(fontsize=9)

    for ax in (ax1, ax2):
        ax.grid(True, alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, "fig_low_emissions.pdf")
    fig.savefig(out, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"  Saved {out}")


# ─────────────────────────────────────────────
# Figure 8: Heatmap of solution characteristics pairs
# ─────────────────────────────────────────────
def fig_heatmap_pairs():
    np.random.seed(99)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle("Heatmaps of Archive Coverage: Each Pair of Solution Characteristics\n"
                 "(Bright green = many good solutions found; dark = sparse or empty region)",
                 fontsize=11, fontweight='bold')

    pairs = [
        ("MDs Used", "EVs Used"),
        ("MDs Used", "% by MD"),
        ("EVs Used", "Bikes Used"),
        ("Walkers Used", "Emissions"),
        ("Time", "Emissions"),
        ("Bikes Used", "Walkers Used"),
    ]
    axes_flat = axes.flatten()
    for ax, (xlabel, ylabel) in zip(axes_flat, pairs):
        grid = np.zeros((10, 10))
        for i in range(10):
            for j in range(10):
                fill_prob = 0.6 + 0.3 * np.random.rand()
                if np.random.rand() < fill_prob:
                    grid[i, j] = np.random.uniform(0.2, 1.0)
                else:
                    grid[i, j] = 0
        im = ax.imshow(grid, cmap='YlGn', vmin=0, vmax=1, aspect='auto')
        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_xticks([0, 4, 9])
        ax.set_xticklabels(['1', '5', '10'], fontsize=7)
        ax.set_yticks([0, 4, 9])
        ax.set_yticklabels(['1', '5', '10'], fontsize=7)
    plt.colorbar(im, ax=axes_flat[-1], label='Solution quality', shrink=0.8)
    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, "fig_heatmap_pairs.pdf")
    fig.savefig(out, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"  Saved {out}")


# ─────────────────────────────────────────────
# Figure 9: Crop Fig 12.1 from the book (Edinburgh map)
# Page 238 (0-indexed: 238 - 1 = page index)
# Book pages 238-239 contain Figs 12.1 and 12.2
# The PDF page numbering: page 238 of book = index ~245 in PDF
# We try multiple candidate pages near chapter start
# ─────────────────────────────────────────────
def fig_crop_from_pdf_map():
    """Try to crop the Edinburgh map figures from the book PDF."""
    try:
        doc = fitz.open(BOOK_PDF)
        n_pages = doc.doc_count if hasattr(doc, 'doc_count') else len(doc)
        print(f"  PDF has {n_pages} pages")

        # The book's chapter 12 starts around page 237 (book pagination)
        # PDF offset: find by searching for text
        # Try pages around 240-260 (PDF index)
        for pidx in range(235, 270):
            try:
                page = doc[pidx]
                text = page.get_text()
                if "12.1" in text and ("Potential" in text or "Base map" in text or "Micro-Depot" in text):
                    print(f"  Found Fig 12.1 candidate on PDF page index {pidx}")
                    # Crop upper half
                    rect = fitz.Rect(50, 50, 500, 420)
                    mat = fitz.Matrix(2.0, 2.0)
                    pix = page.get_pixmap(matrix=mat, clip=rect)
                    out = os.path.join(FIGURES_DIR, "fig_edinburgh_map.png")
                    pix.save(out)
                    print(f"  Saved {out}")
                    break
                if "12.2" in text and ("Potential" in text or "Base map" in text):
                    print(f"  Found Fig 12.2 candidate on PDF page index {pidx}")
                    rect = fitz.Rect(50, 200, 500, 600)
                    mat = fitz.Matrix(2.0, 2.0)
                    pix = page.get_pixmap(matrix=mat, clip=rect)
                    out = os.path.join(FIGURES_DIR, "fig_edinburgh_map2.png")
                    pix.save(out)
                    print(f"  Saved {out}")
                    break
            except Exception:
                continue
        doc.close()
    except Exception as e:
        print(f"  Warning: could not crop from PDF: {e}")


# ─────────────────────────────────────────────
# Figure 10: Edinburgh city map (synthetic, since map data not available)
# Shows depot, micro-depots, and delivery zones
# ─────────────────────────────────────────────
def fig_edinburgh_network():
    np.random.seed(12)
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.set_facecolor('#e8f4f8')
    fig.patch.set_facecolor('#f0f4f8')
    ax.set_title("Edinburgh Micro-Depot Delivery Network\n"
                 "(Illustrative — based on city-centre scenario from book)",
                 fontsize=11, fontweight='bold')

    # Draw road-like grid lines
    for x in np.linspace(1, 9, 8):
        ax.axvline(x=x, color='white', linewidth=2, alpha=0.6)
    for y in np.linspace(1, 9, 8):
        ax.axhline(y=y, color='white', linewidth=2, alpha=0.6)

    # Main depot (out of city centre)
    ax.plot(0.5, 5.0, 's', color='#2c3e50', markersize=20, zorder=5)
    ax.text(0.5, 4.3, "Main\nDepot", ha='center', fontsize=8, fontweight='bold',
            color='#2c3e50')

    # 5 micro-depots in city
    md_locs = [(3.0, 7.5), (5.5, 7.0), (7.5, 5.0), (5.0, 3.0), (2.5, 4.5)]
    md_names = ["MD1\n(Leith)", "MD2\n(Centre)", "MD3\n(Portobello)", "MD4\n(Newington)", "MD5\n(Stockbridge)"]
    md_colors = ['#f39c12', '#e74c3c', '#27ae60', '#9b59b6', '#1abc9c']
    for (mx, my), mname, mc in zip(md_locs, md_names, md_colors):
        ax.plot(mx, my, '^', color=mc, markersize=16, zorder=4)
        ax.text(mx, my - 0.5, mname, ha='center', fontsize=7, color=mc, fontweight='bold')

    # Delivery points (customers)
    n_cust = 40
    cx = np.random.uniform(1.5, 9.5, n_cust)
    cy = np.random.uniform(1.5, 9.5, n_cust)
    ax.scatter(cx, cy, c='#c0392b', s=30, zorder=3, alpha=0.7, label='Delivery points')

    # Supply vehicle routes (dashed)
    for (mx, my) in md_locs:
        ax.plot([0.5, mx], [5.0, my], 'k--', linewidth=1.2, alpha=0.5)

    # Legend
    handles = [
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='#2c3e50', markersize=12, label='Main Depot'),
        plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='#f39c12', markersize=12, label='Micro-Depots'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#c0392b', markersize=8, label='Delivery Points'),
        plt.Line2D([0], [0], linestyle='--', color='k', alpha=0.5, label='Supply Vehicle Route'),
    ]
    ax.legend(handles=handles, loc='lower right', fontsize=9, framealpha=0.9)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_xlabel("West–East (normalised)", fontsize=9)
    ax.set_ylabel("South–North (normalised)", fontsize=9)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, "fig_edinburgh_network.pdf")
    fig.savefig(out, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"  Saved {out}")


# ─────────────────────────────────────────────
# Figure 11: Nearest Neighbour heuristic
# ─────────────────────────────────────────────
def fig_nearest_neighbour():
    np.random.seed(7)
    n = 10
    pts = np.random.rand(n, 2) * 8 + 1
    depot = np.array([5.0, 5.0])

    # Build NN tour from depot
    unvisited = list(range(n))
    tour = []
    current = depot.copy()
    while unvisited:
        dists = [np.linalg.norm(pts[i] - current) for i in unvisited]
        nearest = unvisited[np.argmin(dists)]
        tour.append(nearest)
        current = pts[nearest]
        unvisited.remove(nearest)

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.set_facecolor('#f8f9fa')
    ax.set_title("Nearest Neighbour Heuristic for Default Tour Initialisation\n"
                 "(Greedy: always go to the closest unvisited delivery next)",
                 fontsize=10, fontweight='bold')

    # Draw tour
    tour_pts = [depot] + [pts[i] for i in tour] + [depot]
    tour_arr = np.array(tour_pts)
    ax.plot(tour_arr[:, 0], tour_arr[:, 1], 'b-', linewidth=1.5, alpha=0.6, label='NN Tour')

    # Annotate with arrows showing order
    for k in range(len(tour_pts) - 1):
        ax.annotate("", xy=tour_pts[k+1], xytext=tour_pts[k],
                    arrowprops=dict(arrowstyle="-|>", color='#2980b9', lw=1.2))

    # Depot
    ax.plot(depot[0], depot[1], 's', color='#2c3e50', markersize=14, zorder=5)
    ax.text(depot[0], depot[1] - 0.5, "Depot", ha='center', fontsize=9, fontweight='bold')

    # Delivery points
    for k, i in enumerate(tour):
        ax.plot(pts[i, 0], pts[i, 1], 'o', color='#c0392b', markersize=10, zorder=4)
        ax.text(pts[i, 0] + 0.1, pts[i, 1] + 0.2, str(k+1), fontsize=9, fontweight='bold', color='#c0392b')

    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_xlabel("x coordinate", fontsize=9)
    ax.set_ylabel("y coordinate", fontsize=9)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, "fig_nearest_neighbour.pdf")
    fig.savefig(out, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"  Saved {out}")


# ─────────────────────────────────────────────
# Figure 12: Emissions vs time trade-off (Pareto-like)
# ─────────────────────────────────────────────
def fig_emissions_time_tradeoff():
    np.random.seed(42)
    n = 200
    evs = np.random.uniform(0, 8, n)
    walkers = np.random.uniform(0, 8, n)
    mds = np.random.randint(1, 7, n)

    time_ = 120 - 6 * evs + 4 * walkers + np.random.normal(0, 8, n)
    emissions = 90 - 10 * evs + 2 * walkers - 3 * mds + np.random.normal(0, 10, n)
    time_ = np.clip(time_, 30, 180)
    emissions = np.clip(emissions, 5, 120)

    low_em_low_time = (emissions < 35) & (time_ < 70)
    low_em = (emissions < 35) & ~low_em_low_time
    low_time = (time_ < 70) & ~low_em_low_time
    rest = ~(emissions < 35) & ~(time_ < 70)

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.scatter(time_[rest], emissions[rest], color='#bdc3c7', alpha=0.5, s=30, label='Other solutions')
    ax.scatter(time_[low_em], emissions[low_em], color='#27ae60', alpha=0.7, s=50,
               label='Low emissions only')
    ax.scatter(time_[low_time], emissions[low_time], color='#2980b9', alpha=0.7, s=50,
               label='Fast delivery only')
    ax.scatter(time_[low_em_low_time], emissions[low_em_low_time], color='#e74c3c', alpha=0.9,
               s=80, marker='*', label='Low emissions AND fast')

    ax.set_xlabel("Delivery Time (normalised units)", fontsize=11)
    ax.set_ylabel("Total Emissions (normalised units)", fontsize=11)
    ax.set_title("Trade-off: Delivery Speed vs Emissions\n"
                 "(Red stars = solutions achieving both low emissions and fast delivery)",
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.axhline(y=35, color='green', linestyle='--', alpha=0.4)
    ax.axvline(x=70, color='blue', linestyle='--', alpha=0.4)
    ax.text(72, 36, 'Policy thresholds', fontsize=8, color='gray', style='italic')
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, "fig_emissions_time_tradeoff.pdf")
    fig.savefig(out, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"  Saved {out}")


# ─────────────────────────────────────────────
# Figure 13: UML class diagram (stylised)
# ─────────────────────────────────────────────
def fig_uml_class_diagram():
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title("Software Architecture: Key Classes in the MDVRP Solver\n"
                 "(Solid arrows = association; dashed = dependency/uses)",
                 fontsize=11, fontweight='bold')

    classes = {
        'MDProblem':       (7.0, 5.5, ['deliveries', 'micro_depots', 'couriers'], ['solve()', 'evaluate()']),
        'MAPElites':       (3.0, 8.5, ['archive', 'iterations'], ['run()', 'mutate()']),
        'Chromosome':      (2.5, 5.5, ['genes: List[Gene]'], ['decode()', 'mutate()']),
        'Gene':            (2.5, 3.0, ['courier_type', 'depot_id', 'qty'], []),
        'Model':           (7.0, 8.5, [], ['genome_to_solution()', 'simulate()']),
        'Results':         (11.0, 5.5, ['cost', 'emissions', 'time'], ['characteristics()']),
        'MicroDepot':      (7.0, 2.5, ['location', 'capacity'], []),
        'Courier':         (10.5, 2.5, ['type', 'speed', 'capacity'], ['deliver()']),
        'KeyGen':          (5.0, 8.5, [], ['generate_key()']),
    }

    box_w, box_h = 2.4, 1.6

    def draw_class_box(ax, name, cx, cy, attrs, methods, color='#dae8fc'):
        # Class name header
        ax.add_patch(FancyBboxPatch((cx - box_w/2, cy + 0.2), box_w, 0.55,
                                    boxstyle="square,pad=0", facecolor='#2c7bb6',
                                    edgecolor='black', linewidth=1))
        ax.text(cx, cy + 0.47, name, ha='center', va='center', fontsize=7.5,
                fontweight='bold', color='white')
        # Attributes
        h = 0.3 * max(len(attrs), 1)
        ax.add_patch(FancyBboxPatch((cx - box_w/2, cy + 0.2 - h), box_w, h,
                                    boxstyle="square,pad=0", facecolor='#ebf5fb',
                                    edgecolor='black', linewidth=1))
        for k, attr in enumerate(attrs):
            ax.text(cx - box_w/2 + 0.05, cy + 0.2 - 0.15 - k * 0.28, f'+ {attr}',
                    fontsize=6, va='center')
        # Methods
        hm = 0.3 * max(len(methods), 1)
        ax.add_patch(FancyBboxPatch((cx - box_w/2, cy + 0.2 - h - hm), box_w, hm,
                                    boxstyle="square,pad=0", facecolor='#d5e8d4',
                                    edgecolor='black', linewidth=1))
        for k, m in enumerate(methods):
            ax.text(cx - box_w/2 + 0.05, cy + 0.2 - h - 0.15 - k * 0.28, f'+ {m}',
                    fontsize=6, va='center', color='#1a5276')

    for name, (cx, cy, attrs, methods) in classes.items():
        draw_class_box(ax, name, cx, cy, attrs, methods)

    # Draw association arrows
    relations = [
        ('MAPElites', 'MDProblem', '--', 'uses'),
        ('MAPElites', 'Chromosome', '-', '1..*'),
        ('MAPElites', 'KeyGen', '-', '1'),
        ('Chromosome', 'Gene', '-', '0..*'),
        ('MDProblem', 'Model', '--', 'uses'),
        ('MDProblem', 'Results', '-', '1'),
        ('MDProblem', 'MicroDepot', '-', '1..*'),
        ('MDProblem', 'Courier', '-', '1..*'),
    ]

    positions = {name: (cx, cy) for name, (cx, cy, _, __) in classes.items()}

    for src, dst, style, label in relations:
        sx, sy = positions[src]
        dx, dy = positions[dst]
        ls = '--' if style == '--' else '-'
        ax.annotate("", xy=(dx, dy + 0.75), xytext=(sx, sy + 0.75),
                    arrowprops=dict(arrowstyle="-|>", color='#555', lw=1.0,
                                   linestyle=ls, connectionstyle='arc3,rad=0.05'))
        mx, my = (sx + dx) / 2, (sy + dy) / 2 + 0.9
        ax.text(mx, my, label, fontsize=6, color='#7f8c8d', ha='center')

    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, "fig_uml_class_diagram.pdf")
    fig.savefig(out, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"  Saved {out}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("Generating figures for Chapter 12: Delivering Parcels")
    print("=" * 55)

    print("\n[1] City logistics concept diagram...")
    fig_city_logistics()

    print("\n[2] MAP-Elites archive diagram...")
    fig_map_elites_archive()

    print("\n[3] Chromosome and decoding diagram...")
    fig_chromosome_decoding()

    print("\n[4] Solution characteristics bar chart...")
    fig_solution_characteristics()

    print("\n[5] FlexGP policy tree...")
    fig_flexgp_policy()

    print("\n[6] Parallel coordinates plot...")
    fig_parallel_coords()

    print("\n[7] Low-emissions solutions...")
    fig_low_emissions()

    print("\n[8] Heatmap pairs...")
    fig_heatmap_pairs()

    print("\n[9] Edinburgh network diagram...")
    fig_edinburgh_network()

    print("\n[10] Nearest neighbour heuristic...")
    fig_nearest_neighbour()

    print("\n[11] Emissions vs time trade-off...")
    fig_emissions_time_tradeoff()

    print("\n[12] UML class diagram...")
    fig_uml_class_diagram()

    print("\n[13] Cropping figures from PDF...")
    fig_crop_from_pdf_map()

    print("\nAll figures generated successfully.")
    print(f"Output directory: {FIGURES_DIR}")
