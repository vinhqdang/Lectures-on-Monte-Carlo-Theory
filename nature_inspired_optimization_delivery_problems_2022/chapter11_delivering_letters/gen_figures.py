"""
gen_figures.py  –  Generate all figures for Chapter 11: Delivering Letters
Requires: matplotlib, numpy, pymupdf (fitz)
Run with: conda run -n py313 python3 gen_figures.py
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import os
import fitz  # PyMuPDF

# Output directory (same as figures/)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(SCRIPT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

BOOK_PDF = os.path.join(
    os.path.dirname(SCRIPT_DIR),
    "Nature Inspired Optimisation for Delivery Problems 2022.pdf"
)

# ─────────────────────────────────────────────────────────────
# Helper: crop a region from the PDF
# ─────────────────────────────────────────────────────────────
def crop_pdf_region(page_num, rect_xywh, out_name, dpi=180):
    """
    page_num  : 1-indexed page number in the PDF
    rect_xywh : (x, y, w, h) in PDF points from top-left of page
    out_name  : output filename (no directory)
    """
    doc = fitz.open(BOOK_PDF)
    page = doc[page_num - 1]
    x, y, w, h = rect_xywh
    clip = fitz.Rect(x, y, x + w, y + h)
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat, clip=clip)
    out_path = os.path.join(FIG_DIR, out_name)
    pix.save(out_path)
    doc.close()
    print(f"  Saved: {out_path}")


# ─────────────────────────────────────────────────────────────
# Figure 1: Urban street layout schematic (replaces Fig 11.1)
# ─────────────────────────────────────────────────────────────
def fig_street_layout():
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_facecolor('#f0f4e8')
    fig.patch.set_facecolor('#f0f4e8')

    # Road network (grey roads)
    road_color = '#b0b0b0'
    road_w = 0.55
    roads_h = [(0, 2, 10), (0, 5, 10)]    # (y_start, y, x_end)
    roads_v = [(2, 0, 7), (5, 0, 7), (8, 0, 7)]

    for (x0, y0, x1, y1) in [(0, 2, 10, 2), (0, 5, 10, 5)]:
        ax.plot([x0, x1], [y0, y1], color=road_color, lw=18, solid_capstyle='round', zorder=1)
    for (x, y0, y1) in [(2, 0, 7), (5, 0, 7), (8, 0, 7)]:
        ax.plot([x, x], [y0, y1], color=road_color, lw=18, solid_capstyle='round', zorder=1)

    # Green blocks (city blocks)
    block_color = '#c8e6c9'
    blocks = [
        (0.3, 2.3, 1.4, 2.4),   # bottom-left
        (2.3, 2.3, 2.4, 2.4),
        (5.3, 2.3, 2.4, 2.4),
        (8.3, 2.3, 1.4, 2.4),
        (0.3, 5.3, 1.4, 1.4),
        (2.3, 5.3, 2.4, 1.4),
        (5.3, 5.3, 2.4, 1.4),
        (8.3, 5.3, 1.4, 1.4),
    ]
    for (x, y, w, h) in blocks:
        rect = plt.Rectangle((x, y), w, h, color=block_color, zorder=2, ec='#558b2f', lw=1)
        ax.add_patch(rect)

    # House dots on streets
    np.random.seed(42)
    house_positions = []
    # Along horizontal road y=2: houses on both sides
    for xh in np.linspace(0.5, 9.5, 20):
        if abs(xh - 2) > 0.3 and abs(xh - 5) > 0.3 and abs(xh - 8) > 0.3:
            house_positions.append((xh, 2.45))
            house_positions.append((xh, 1.55))
    for xh in np.linspace(0.5, 9.5, 20):
        if abs(xh - 2) > 0.3 and abs(xh - 5) > 0.3 and abs(xh - 8) > 0.3:
            house_positions.append((xh, 5.45))
            house_positions.append((xh, 4.55))
    for yh in np.linspace(0.3, 6.7, 14):
        if abs(yh - 2) > 0.3 and abs(yh - 5) > 0.3:
            house_positions.append((2.45, yh))
            house_positions.append((1.55, yh))
            house_positions.append((5.45, yh))
            house_positions.append((4.55, yh))
            house_positions.append((8.45, yh))
            house_positions.append((7.55, yh))

    for (xh, yh) in house_positions:
        ax.plot(xh, yh, 's', color='#e53935', ms=3.5, zorder=4, alpha=0.8)

    # Labels for street sections
    ax.text(1.05, 1.2, "ClunyGdns1\n(one side)", ha='center', fontsize=6.5,
            color='#1a237e', zorder=5, style='italic')
    ax.text(3.5, 1.2, "ClunyDrv1", ha='center', fontsize=6.5, color='#1a237e', zorder=5, style='italic')
    ax.text(6.5, 1.2, "BraidAv1", ha='center', fontsize=6.5, color='#1a237e', zorder=5, style='italic')
    ax.text(1.05, 3.3, "HermitageGdns1", ha='center', fontsize=6, color='#1a237e', zorder=5, style='italic')
    ax.text(3.5, 3.3, "ClunyDrv1\n(both sides)", ha='center', fontsize=6, color='#1a237e', zorder=5, style='italic')

    # Junctions
    for (jx, jy) in [(2, 2), (5, 2), (8, 2), (2, 5), (5, 5), (8, 5)]:
        ax.plot(jx, jy, 'o', color='#f57f17', ms=8, zorder=6, mec='#e65100', mew=1.2)
    ax.text(2, 1.75, 'J1', ha='center', fontsize=7, color='#e65100', fontweight='bold', zorder=7)
    ax.text(5, 1.75, 'J2', ha='center', fontsize=7, color='#e65100', fontweight='bold', zorder=7)
    ax.text(8, 1.75, 'J3', ha='center', fontsize=7, color='#e65100', fontweight='bold', zorder=7)
    ax.text(2, 5.25, 'J4', ha='center', fontsize=7, color='#e65100', fontweight='bold', zorder=7)
    ax.text(5, 5.25, 'J5', ha='center', fontsize=7, color='#e65100', fontweight='bold', zorder=7)
    ax.text(8, 5.25, 'J6', ha='center', fontsize=7, color='#e65100', fontweight='bold', zorder=7)

    ax.set_title("Urban Street Network — Street Sections and Junctions\n"
                 "(Red squares = delivery addresses, orange circles = junctions)",
                 fontsize=10, fontweight='bold', pad=8)

    fig.tight_layout()
    out = os.path.join(FIG_DIR, "fig_street_layout.pdf")
    fig.savefig(out, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")


# ─────────────────────────────────────────────────────────────
# Figure 2: SBR chromosome representation
# ─────────────────────────────────────────────────────────────
def fig_chromosome():
    fig, ax = plt.subplots(figsize=(10, 2.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 2.2)
    ax.axis('off')

    genes = ["ClunyGdns1", "ClunyDrv1", "ClunyDrv1", "BraidAv1",
             "HermitageGdns1", "HermitageGdns1", "BraidAv1", "ClunyGdns1"]
    n = len(genes)
    w = 10.0 / n
    colors = ['#bbdefb', '#c8e6c9', '#c8e6c9', '#ffe0b2',
              '#f8bbd0', '#f8bbd0', '#ffe0b2', '#bbdefb']
    for i, (gene, col) in enumerate(zip(genes, colors)):
        rect = plt.Rectangle((i * w + 0.02, 0.5), w - 0.04, 1.1,
                              facecolor=col, edgecolor='#37474f', lw=1.2)
        ax.add_patch(rect)
        ax.text(i * w + w / 2, 1.05, gene, ha='center', va='center',
                fontsize=7.5, fontweight='bold', color='#1a237e')
        ax.text(i * w + w / 2, 0.28, f"Gene {i+1}", ha='center', va='center',
                fontsize=6.5, color='#555')

    # Highlight paired genes
    for pair_start, pair_col in [(1, '#4caf50'), (4, '#e91e63'), (6, '#ff9800')]:
        for offset in [0, 1]:
            idx = pair_start + offset if pair_col != '#ff9800' else pair_start
            x = (pair_start + offset if pair_col != '#ff9800' else pair_start) * w
            ax.annotate('', xy=(x + w / 2, 1.62),
                        xytext=((pair_start + 1 - offset if pair_col != '#ff9800' else pair_start) * w + w / 2, 1.62),
                        arrowprops=dict(arrowstyle='<->', color=pair_col, lw=1.5))
            break

    ax.annotate('', xy=(1 * w + w / 2, 1.62), xytext=(2 * w + w / 2, 1.62),
                arrowprops=dict(arrowstyle='<->', color='#4caf50', lw=1.5))
    ax.text(1.5 * w + w / 2, 1.80, 'paired\n(both sides)', ha='center', fontsize=6.5, color='#4caf50')

    ax.annotate('', xy=(4 * w + w / 2, 1.62), xytext=(5 * w + w / 2, 1.62),
                arrowprops=dict(arrowstyle='<->', color='#e91e63', lw=1.5))
    ax.text(4.5 * w + w / 2, 1.80, 'paired\n(both sides)', ha='center', fontsize=6.5, color='#e91e63')

    ax.set_title("SBR Chromosome: Street Sections as Genes\n"
                 "(Duplicate genes indicate double-sided street sections)",
                 fontsize=10, fontweight='bold', y=0.98)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "fig_chromosome.pdf")
    fig.savefig(out, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")


# ─────────────────────────────────────────────────────────────
# Figure 3: Delivery patterns (one-sided, both-sided, crossover)
# ─────────────────────────────────────────────────────────────
def fig_delivery_patterns():
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    patterns = [
        ("One Side\n(Single junction)", '#1976d2', [(0.5, 1.0), (1.5, 1.0), (2.5, 1.0),
                                                      (3.5, 1.0), (4.5, 1.0)]),
        ("Both Sides\n(Same junction)", '#388e3c', [(0.5, 1.6), (1.5, 1.6), (2.5, 1.6),
                                                     (3.5, 1.6), (4.5, 1.6),
                                                     (4.5, 0.4), (3.5, 0.4), (2.5, 0.4),
                                                     (1.5, 0.4), (0.5, 0.4)]),
        ("Cross-Over\n(Opposite junction)", '#d32f2f', [(0.5, 1.6), (1.5, 1.6), (2.5, 1.6),
                                                         (3.5, 1.6), (4.5, 1.6),
                                                         (4.5, 0.4), (3.5, 0.4), (2.5, 0.4),
                                                         (1.5, 0.4), (0.5, 0.4)]),
    ]
    for ax, (title, col, path_pts) in zip(axes, patterns):
        ax.set_xlim(-0.2, 5.2)
        ax.set_ylim(-0.2, 2.2)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_facecolor('#fafafa')

        # Street as grey band
        road = plt.Rectangle((-0.1, 0.75), 5.2, 0.5, color='#cfd8dc', zorder=1)
        ax.add_patch(road)

        # Houses top and bottom
        for xh in np.linspace(0.5, 4.5, 5):
            ax.plot(xh, 1.6, 's', color='#e53935', ms=9, zorder=3)
            ax.plot(xh, 0.4, 's', color='#e53935', ms=9, zorder=3)

        # Junctions
        ax.plot(0.0, 1.0, 'o', color='#f57f17', ms=10, zorder=4)
        ax.plot(5.0, 1.0, 'o', color='#f57f17', ms=10, zorder=4)
        ax.text(-0.1, 0.72, 'J_start', fontsize=6.5, ha='center', color='#e65100')
        ax.text(5.1, 0.72, 'J_end', fontsize=6.5, ha='center', color='#e65100')

        # Draw walking path
        if title.startswith("One"):
            xs = [0.0] + [p[0] for p in path_pts] + [5.0]
            ys = [1.0] + [p[1] for p in path_pts] + [1.0]
            ax.plot(xs, ys, '->', color=col, lw=2, zorder=5, ms=6)
        elif title.startswith("Both"):
            # Go up side then back other side
            xs = [0.0, 0.5, 1.5, 2.5, 3.5, 4.5, 5.0, 4.5, 3.5, 2.5, 1.5, 0.5, 0.0]
            ys = [1.0, 1.6, 1.6, 1.6, 1.6, 1.6, 1.0, 0.4, 0.4, 0.4, 0.4, 0.4, 1.0]
            ax.plot(xs, ys, color=col, lw=2, zorder=5)
            ax.annotate('', xy=(xs[1], ys[1]), xytext=(xs[0], ys[0]),
                        arrowprops=dict(arrowstyle='->', color=col, lw=1.5))
        else:
            # Cross-over: start at one junction, finish at opposite
            xs = [0.0, 0.5, 1.5, 2.5, 3.5, 4.5, 4.5, 3.5, 2.5, 1.5, 0.5, 0.0]
            ys = [1.0, 1.6, 1.6, 1.6, 1.6, 1.6, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4]
            ax.plot(xs[:7], ys[:7], color=col, lw=2, zorder=5)
            ax.plot(xs[6:], ys[6:], color=col, lw=2, ls='--', zorder=5)
            ax.annotate('', xy=(xs[1], ys[1]), xytext=(xs[0], ys[0]),
                        arrowprops=dict(arrowstyle='->', color=col, lw=1.5))

        ax.set_title(title, fontsize=9, fontweight='bold', color=col, pad=4)

    fig.suptitle("SBR Delivery Patterns for a Street Section",
                 fontsize=11, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out = os.path.join(FIG_DIR, "fig_delivery_patterns.pdf")
    fig.savefig(out, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")


# ─────────────────────────────────────────────────────────────
# Figure 4: SBR decoding flowchart
# ─────────────────────────────────────────────────────────────
def fig_decoding_flowchart():
    fig, ax = plt.subplots(figsize=(7, 9))
    ax.set_xlim(0, 7)
    ax.set_ylim(0, 9)
    ax.axis('off')

    def box(cx, cy, text, color='#e3f2fd', h=0.55, w=3.5, fontsize=8.5):
        rect = FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                               boxstyle="round,pad=0.08",
                               facecolor=color, edgecolor='#1565c0', lw=1.2)
        ax.add_patch(rect)
        ax.text(cx, cy, text, ha='center', va='center', fontsize=fontsize,
                fontweight='bold', color='#0d47a1', wrap=True)

    def diamond(cx, cy, text, color='#fff9c4', h=0.7, w=3.8):
        diamond_pts = np.array([
            [cx, cy + h/2], [cx + w/2, cy], [cx, cy - h/2], [cx - w/2, cy]
        ])
        poly = plt.Polygon(diamond_pts, closed=True, facecolor=color, edgecolor='#f57f17', lw=1.5)
        ax.add_patch(poly)
        ax.text(cx, cy, text, ha='center', va='center', fontsize=7.5,
                fontweight='bold', color='#e65100')

    def arrow(x1, y1, x2, y2, label='', lcolor='#37474f'):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color=lcolor, lw=1.5))
        if label:
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            ax.text(mx + 0.15, my, label, fontsize=7.5, color='#37474f')

    # Boxes
    box(3.5, 8.5, "START: decoder(genotype, problem)", color='#c8e6c9')
    box(3.5, 7.7, "deliveries=[], geneCount=0")
    diamond(3.5, 6.85, "geneCount < genotype.length?")
    box(3.5, 6.0, "Check if gene[i] == gene[i+1]\n→ doubleSided = True", color='#fff3e0')
    box(3.5, 5.1, "current = gene[geneCount]\nnext = gene[geneCount+1]")
    box(3.5, 4.2, "applyPattern(prevDelivery,\n current, next, doubleSided)")
    box(3.5, 3.3, "dist += walkingDist(lastDel, nextDel)")
    box(3.5, 2.4, "deliveries.append(street)\ngeneCount++")
    box(3.5, 1.55, "dist += walkingDist(lastDel, problem.end)\n× deliveriesLeft")
    box(3.5, 0.7, "RETURN dist", color='#c8e6c9')

    # Arrows
    arrow(3.5, 8.22, 3.5, 7.97)
    arrow(3.5, 7.43, 3.5, 7.22)
    arrow(3.5, 6.5, 3.5, 6.28, label='Yes')
    arrow(3.5, 5.73, 3.5, 5.47)
    arrow(3.5, 4.83, 3.5, 4.57)
    arrow(3.5, 3.93, 3.5, 3.57)
    arrow(3.5, 3.03, 3.5, 2.67)
    arrow(3.5, 2.13, 3.5, 1.82)
    arrow(3.5, 1.28, 3.5, 0.97)

    # No branch from diamond
    ax.annotate('', xy=(6.3, 6.85), xytext=(5.4, 6.85),
                arrowprops=dict(arrowstyle='->', color='#d32f2f', lw=1.5))
    ax.text(5.7, 6.95, 'No', fontsize=8, color='#d32f2f', fontweight='bold')
    box(6.3, 6.85, "→ line 19–22", color='#ffcdd2', w=1.5, fontsize=7)

    ax.set_title("Algorithm 22: SBR Decoding Procedure\n(Flowchart Overview)",
                 fontsize=11, fontweight='bold', pad=10)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "fig_decoding_flowchart.pdf")
    fig.savefig(out, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")


# ─────────────────────────────────────────────────────────────
# Figure 5: Crossover operator illustration
# ─────────────────────────────────────────────────────────────
def fig_crossover():
    fig, axes = plt.subplots(3, 1, figsize=(11, 4.5))
    parents = [
        ("Parent 1 (p1)", ['A','B','B','D','E','D','F','G','A','G','H','H','C','C','F','E']),
        ("Parent 2 (p2)", ['B','G','H','H','B','E','E','A','A','D','G','F','F','C','C','D']),
        ("Child",         ['A','B','B','D','E','D']),  # first few from p1 that are connected
    ]
    colors_p1 = ['#bbdefb'] * 16
    colors_p2 = ['#c8e6c9'] * 16
    colors_ch = ['#bbdefb','#bbdefb','#bbdefb','#bbdefb','#bbdefb','#bbdefb',
                 '#ffe0b2','#ffe0b2','#ffe0b2','#ffe0b2','#ffe0b2']

    child_genes = ['A','B','B','D','E','D','G','H','H','F','C']

    for ax_i, (ax, (label, genes)) in enumerate(zip(axes, parents)):
        if ax_i == 2:
            genes = child_genes
            cols = colors_ch
        elif ax_i == 0:
            cols = colors_p1
        else:
            cols = colors_p2

        n = len(genes)
        w = 10.0 / n
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 1)
        ax.axis('off')
        for i, (g, c) in enumerate(zip(genes, cols)):
            rect = plt.Rectangle((i * w + 0.01, 0.15), w - 0.02, 0.7,
                                  facecolor=c, edgecolor='#455a64', lw=0.8)
            ax.add_patch(rect)
            ax.text(i * w + w / 2, 0.5, g, ha='center', va='center',
                    fontsize=8, fontweight='bold', color='#212121')
        ax.text(-0.3, 0.5, label, ha='right', va='center', fontsize=8.5,
                fontweight='bold', color='#1a237e', transform=ax.transData)

        if ax_i == 2:
            ax.text(6 * (10.0 / 11) + 0.1, 0.88, '← from p2 (connected via junction)',
                    fontsize=7.5, color='#e65100')

    fig.suptitle("SBR Crossover: Child inherits connected street sections from parents",
                 fontsize=10, fontweight='bold', y=1.01)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "fig_crossover.pdf")
    fig.savefig(out, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")


# ─────────────────────────────────────────────────────────────
# Figure 6: Cantor Pairing Function illustration
# ─────────────────────────────────────────────────────────────
def fig_cantor_pairing():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # Left: formula visualization
    ax = axes[0]
    ax.set_xlim(-0.5, 10)
    ax.set_ylim(-0.5, 10)
    ax.set_xlabel('x (first integer)', fontsize=10)
    ax.set_ylabel('y (second integer)', fontsize=10)
    ax.set_title("Cantor Pairing: unique key z for each (x, y)", fontsize=10, fontweight='bold')

    x_grid = np.arange(0, 8)
    y_grid = np.arange(0, 8)
    cmap = plt.cm.YlOrRd
    for x in x_grid:
        for y in y_grid:
            z = int(0.5 * (x + y) * (x + y + 1) + y)
            color = cmap(z / 80.0)
            ax.add_patch(plt.Rectangle((x - 0.45, y - 0.45), 0.9, 0.9, color=color, zorder=2))
            ax.text(x, y, str(z), ha='center', va='center', fontsize=6.5,
                    color='white' if z > 40 else '#212121', fontweight='bold', zorder=3)

    ax.set_xticks(x_grid)
    ax.set_yticks(y_grid)
    ax.grid(False)

    # Right: hash-map cache diagram
    ax2 = axes[1]
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 8)
    ax2.axis('off')
    ax2.set_title("HashMap Cache Architecture", fontsize=10, fontweight='bold')

    # Boxes
    def rbox(ax, x, y, w, h, text, fc='#e3f2fd', ec='#1565c0', fs=9):
        rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                               facecolor=fc, edgecolor=ec, lw=1.2)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=fs, color='#0d47a1')

    rbox(ax2, 0.5, 6.2, 4, 0.8, "Journey ID = cantorPair(hashA, hashB)", fc='#fff9c4', ec='#f57f17', fs=8)
    rbox(ax2, 0.5, 4.8, 4, 0.9, "HashMap<Long, Double>\n(key=journey_id, value=distance)", fc='#e3f2fd', fs=8)
    rbox(ax2, 0.5, 3.3, 4, 0.9, "cache.get(key)\n→ hit? return cached distance", fc='#c8e6c9', ec='#388e3c', fs=8)
    rbox(ax2, 0.5, 1.8, 4, 0.9, "cache miss?\n→ routing API + cache.put(key, dist)", fc='#ffcdd2', ec='#c62828', fs=8)
    rbox(ax2, 0.5, 0.5, 4, 0.8, "return distance", fc='#c8e6c9', ec='#388e3c', fs=8)

    for (y1, y2) in [(6.2, 5.7), (4.8, 4.2), (3.3, 2.7), (1.8, 1.3)]:
        ax2.annotate('', xy=(2.5, y2), xytext=(2.5, y1),
                     arrowprops=dict(arrowstyle='->', color='#37474f', lw=1.5))

    fig.tight_layout()
    out = os.path.join(FIG_DIR, "fig_cantor_pairing.pdf")
    fig.savefig(out, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")


# ─────────────────────────────────────────────────────────────
# Figure 7: Results comparison bar chart
# ─────────────────────────────────────────────────────────────
def fig_results_comparison():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # Distance comparison
    ax1 = axes[0]
    solvers = ['NN\nHeuristic', 'SBR-EA', 'TSP-EA']
    avg_dist = [10.2, 7.7, 20.0]
    min_dist = [10.2, 6.8, 18.1]
    x = np.arange(len(solvers))
    bars1 = ax1.bar(x - 0.2, avg_dist, 0.35, label='Avg. Distance (km)', color=['#64b5f6', '#81c784', '#e57373'])
    bars2 = ax1.bar(x + 0.2, min_dist, 0.35, label='Min. Distance (km)', color=['#1565c0', '#2e7d32', '#c62828'],
                    alpha=0.85)

    ax1.set_xlabel('Solver', fontsize=11)
    ax1.set_ylabel('Route Distance (km)', fontsize=11)
    ax1.set_title('Route Length Comparison\n(Greenbank area, Edinburgh)', fontsize=10, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(solvers, fontsize=10)
    ax1.legend(fontsize=9)
    ax1.set_ylim(0, 25)
    ax1.grid(axis='y', alpha=0.4)

    for bar in bars1:
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                 f'{bar.get_height():.1f}', ha='center', fontsize=8.5, fontweight='bold')
    for bar in bars2:
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                 f'{bar.get_height():.1f}', ha='center', fontsize=8.5, fontweight='bold')

    # Runtime comparison
    ax2 = axes[1]
    avg_time = [20.3, 9.6, 932.4]
    bars3 = ax2.bar(solvers, avg_time, color=['#64b5f6', '#81c784', '#e57373'],
                    edgecolor='#37474f', lw=1)
    ax2.set_xlabel('Solver', fontsize=11)
    ax2.set_ylabel('Average Time (seconds)', fontsize=11)
    ax2.set_title('Computation Time Comparison\n(budget: 100,000 evaluations)', fontsize=10, fontweight='bold')
    ax2.set_yscale('log')
    ax2.grid(axis='y', alpha=0.4)
    for bar in bars3:
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.1,
                 f'{bar.get_height():.1f}s', ha='center', fontsize=9.5, fontweight='bold')

    ax2.axhline(y=100, color='#d32f2f', ls='--', lw=1.5, alpha=0.7)
    ax2.text(2.4, 110, '100s', fontsize=8, color='#d32f2f')

    fig.tight_layout()
    out = os.path.join(FIG_DIR, "fig_results_comparison.pdf")
    fig.savefig(out, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")


# ─────────────────────────────────────────────────────────────
# Figure 8: SBR vs TSP conceptual comparison
# ─────────────────────────────────────────────────────────────
def fig_sbr_vs_tsp():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    for ax, (title, color, n_nodes, note) in zip(axes, [
        ("TSP: 442 individual houses\nSolution space = 442!/2 ≈ 5.5×10⁹⁷⁸",
         '#e57373', 442, "Each node = one house"),
        ("SBR: 48 street sections\nSolution space = 48! ≈ 1.2×10⁶¹",
         '#81c784', 48, "Each node = one street section (10+ houses)")
    ]):
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(title, fontsize=9, fontweight='bold', color='#1a237e', pad=6)
        ax.text(0, -1.18, note, ha='center', fontsize=8, color='#555', style='italic')

        np.random.seed(99 if 'TSP' in title else 42)
        n_display = min(n_nodes, 60)
        theta = np.linspace(0, 2 * np.pi, n_display, endpoint=False)
        r = 0.75 + 0.2 * np.random.randn(n_display)
        xs = r * np.cos(theta)
        ys = r * np.sin(theta)
        ms = 2 if 'TSP' in title else 8
        ax.plot(xs, ys, 'o', color=color, ms=ms, alpha=0.75, zorder=3)

        # A random route
        order = np.random.permutation(n_display)[:20]
        route_x = xs[order]
        route_y = ys[order]
        ax.plot(np.append(route_x, route_x[0]), np.append(route_y, route_y[0]),
                '-', color=color, alpha=0.45, lw=1, zorder=2)

        # Label
        n_label = f"n={n_nodes}" if 'TSP' in title else "groups=48"
        ax.text(0, 0, n_label, ha='center', va='center', fontsize=12,
                fontweight='bold', color=color, alpha=0.5)

    fig.suptitle("SBR Dramatically Reduces the Search Space vs. TSP",
                 fontsize=11, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out = os.path.join(FIG_DIR, "fig_sbr_vs_tsp.pdf")
    fig.savefig(out, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")


# ─────────────────────────────────────────────────────────────
# Figure 9: Crop Fig 11.1 (street map) from PDF
# ─────────────────────────────────────────────────────────────
def fig_crop_street_map():
    """Crop the actual map figure from the book PDF (Fig 11.1, page 231)."""
    try:
        # Page 231 in book = page index 230 (0-indexed); PDF page 231 is page 1-indexed
        # The figure appears on what in the book is labeled page 224 (p231.png)
        # We crop the upper portion of the page
        crop_pdf_region(
            page_num=231,        # 1-indexed PDF page
            rect_xywh=(30, 50, 530, 340),
            out_name="fig_book_street_map.png",
            dpi=200
        )
    except Exception as e:
        print(f"  Warning: could not crop street map from PDF: {e}")


def fig_crop_sbr_decode_map():
    """Crop Fig 11.2 (SBR decoding on map) from book PDF (page 227)."""
    try:
        crop_pdf_region(
            page_num=234,
            rect_xywh=(30, 30, 530, 550),
            out_name="fig_book_sbr_decode.png",
            dpi=180
        )
    except Exception as e:
        print(f"  Warning: could not crop SBR decode map: {e}")


def fig_crop_greenbank_route():
    """Crop Fig 11.5 (best route found by SBR-EA) from book PDF (page 235)."""
    try:
        crop_pdf_region(
            page_num=242,
            rect_xywh=(30, 50, 530, 480),
            out_name="fig_book_greenbank_route.png",
            dpi=180
        )
    except Exception as e:
        print(f"  Warning: could not crop Greenbank route: {e}")


def fig_crop_greenbank_area():
    """Crop Fig 11.4 (Greenbank study area) from book PDF (page 240)."""
    try:
        crop_pdf_region(
            page_num=240,
            rect_xywh=(30, 50, 530, 380),
            out_name="fig_book_greenbank_area.png",
            dpi=180
        )
    except Exception as e:
        print(f"  Warning: could not crop Greenbank area: {e}")


def fig_crop_class_diagram():
    """Crop Fig 11.3 (class diagram) from book PDF (page 238)."""
    try:
        crop_pdf_region(
            page_num=238,
            rect_xywh=(80, 380, 430, 280),
            out_name="fig_book_class_diagram.png",
            dpi=180
        )
    except Exception as e:
        print(f"  Warning: could not crop class diagram: {e}")


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Generating figures for Chapter 11: Delivering Letters...")

    print("\n[1] Street layout schematic")
    fig_street_layout()

    print("\n[2] Chromosome representation")
    fig_chromosome()

    print("\n[3] Delivery patterns")
    fig_delivery_patterns()

    print("\n[4] Decoding flowchart")
    fig_decoding_flowchart()

    print("\n[5] Crossover operator")
    fig_crossover()

    print("\n[6] Cantor pairing / cache")
    fig_cantor_pairing()

    print("\n[7] Results comparison")
    fig_results_comparison()

    print("\n[8] SBR vs TSP")
    fig_sbr_vs_tsp()

    print("\n[9] Crop book figures from PDF")
    fig_crop_street_map()
    fig_crop_sbr_decode_map()
    fig_crop_greenbank_route()
    fig_crop_greenbank_area()
    fig_crop_class_diagram()

    print("\nAll figures generated successfully.")
