"""
gen_figures.py  —  Chapter 7: GeoSpatial Data
Generate all figures needed for chapter07_slides.tex
Uses matplotlib (backend Agg) and pymupdf for PDF crops.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import numpy as np
import os
import math

FIGURES_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

PDF_PATH = os.path.join(
    os.path.dirname(__file__), "..",
    "Nature Inspired Optimisation for Delivery Problems 2022.pdf"
)

# ─────────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────────
def save(fig, name, dpi=150):
    path = os.path.join(FIGURES_DIR, name)
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"  saved {name}")


# ─────────────────────────────────────────────────────────────────
# Figure 1: Raster vs Vector conceptual comparison
# ─────────────────────────────────────────────────────────────────
def fig_raster_vs_vector():
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    fig.suptitle("Raster vs. Vector Graphics", fontsize=13, fontweight='bold')

    # --- Raster panel: pixel grid of a simple shape ---
    ax = axes[0]
    ax.set_title("Raster (pixel grid)", fontsize=11)
    grid = np.zeros((10, 10))
    # Draw a rough diagonal line as pixels
    for i in range(2, 8):
        grid[i, i] = 1
        if i < 9:
            grid[i, i+1] = 0.4
    ax.imshow(grid, cmap='Blues', vmin=0, vmax=1, interpolation='nearest')
    ax.set_xticks(np.arange(-0.5, 10, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 10, 1), minor=True)
    ax.grid(which='minor', color='gray', linewidth=0.5)
    ax.tick_params(which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
    ax.text(5, 9.5, "Zoom in → see pixels", ha='center', va='top', fontsize=8,
            transform=ax.transData, color='gray')

    # --- Vector panel: geometric shapes defined by coordinates ---
    ax2 = axes[1]
    ax2.set_title("Vector (geometric coordinates)", fontsize=11)
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.set_aspect('equal')
    ax2.tick_params(labelbottom=False, labelleft=False)
    # Draw a circle
    circle = plt.Circle((5, 5), 2.5, fill=False, color='steelblue', linewidth=2)
    ax2.add_patch(circle)
    # Draw a polygon (triangle)
    triangle = plt.Polygon([[2, 1], [5, 7], [8, 1]], fill=False,
                            edgecolor='darkorange', linewidth=2)
    ax2.add_patch(triangle)
    # Annotate control points
    for pt, lbl in [([5, 7], 'vertex'), ([2, 1], 'vertex'), ([8, 1], 'vertex')]:
        ax2.plot(*pt, 'o', color='darkorange', markersize=6)
    ax2.text(5, 5, 'center\n(x,y)', ha='center', va='center', fontsize=8, color='steelblue')
    ax2.text(5, 0.3, "Zoom in → stays sharp", ha='center', va='bottom', fontsize=8, color='gray')

    fig.tight_layout()
    save(fig, "raster_vs_vector.pdf")


# ─────────────────────────────────────────────────────────────────
# Figure 2: Coordinate system — lat/lon on a globe sketch
# ─────────────────────────────────────────────────────────────────
def fig_coordinate_system():
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect('equal')
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.axis('off')
    ax.set_title("WGS84: Latitude and Longitude", fontsize=12, fontweight='bold')

    # Globe circle
    globe = plt.Circle((0, 0), 1.0, fill=False, color='steelblue', linewidth=2)
    ax.add_patch(globe)

    # Equator
    ax.plot([-1, 1], [0, 0], color='orange', linewidth=1.5, linestyle='--', label='Equator (lat=0°)')

    # Prime meridian
    ax.plot([0, 0], [-1, 1], color='green', linewidth=1.5, linestyle='--', label='Prime Meridian (lon=0°)')

    # A sample point: lat=40°N, lon=30°E  → (x, y) ≈ (sin30°, sin40°) normalised
    lat_r = math.radians(40)
    lon_r = math.radians(30)
    px = math.cos(lat_r) * math.sin(lon_r)
    py = math.sin(lat_r)
    ax.plot(px, py, 'ro', markersize=10, zorder=5)
    ax.annotate("Point P\n(lat=40°N, lon=30°E)",
                xy=(px, py), xytext=(px + 0.3, py + 0.25),
                arrowprops=dict(arrowstyle='->', color='red'),
                fontsize=9, color='red')

    # latitude arc
    lats = np.linspace(0, lat_r, 60)
    arc_x = [0.35 * math.cos(a - math.pi/2) for a in lats]
    arc_y = [0.35 * math.sin(a - math.pi/2) + 0 for a in lats]
    ax.annotate('', xy=(arc_x[-1], arc_y[-1]+0.02),
                xytext=(0.35, -0.02),
                arrowprops=dict(arrowstyle='->', color='purple'))
    ax.text(0.4, 0.15, 'lat (φ, phi)\n= 40°N', fontsize=8, color='purple')

    # longitude arc
    ax.annotate('', xy=(0.45, -0.05),
                xytext=(0.05, -0.25),
                arrowprops=dict(arrowstyle='->', color='darkgreen'))
    ax.text(0.15, -0.35, 'lon (λ, lambda)\n= 30°E', fontsize=8, color='darkgreen')

    ax.legend(loc='lower left', fontsize=8)
    save(fig, "coordinate_system.pdf")


# ─────────────────────────────────────────────────────────────────
# Figure 3: Haversine formula — worked example diagram
# ─────────────────────────────────────────────────────────────────
def fig_haversine():
    # Show two points on a lat-lon grid with the great-circle arc
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.set_title("Haversine Distance: Edinburgh → London", fontsize=12, fontweight='bold')

    # Edinburgh: lat=55.95, lon=-3.19
    # London:    lat=51.51, lon=-0.13
    lon1, lat1 = -3.19, 55.95
    lon2, lat2 = -0.13, 51.51

    ax.set_xlim(-10, 5)
    ax.set_ylim(48, 60)
    ax.set_xlabel("Longitude (°E)", fontsize=10)
    ax.set_ylabel("Latitude (°N)", fontsize=10)

    # Plot a rough outline of Britain (simplified)
    brit_lon = [-5, -3, 0, 2, 1.5, -0.5, -2, -5, -6, -5]
    brit_lat = [50, 50, 51, 52, 54, 55, 56, 58, 57, 50]
    ax.plot(brit_lon, brit_lat, color='lightgray', linewidth=1.5)

    # Great-circle arc (simplified as a curve)
    n = 30
    lons = np.linspace(lon1, lon2, n)
    lats = np.linspace(lat1, lat2, n)
    ax.plot(lons, lats, 'b-', linewidth=2, label='Great-circle path', zorder=3)

    # Points
    ax.plot(lon1, lat1, 'go', markersize=10, zorder=5)
    ax.plot(lon2, lat2, 'ro', markersize=10, zorder=5)
    ax.annotate("Edinburgh\n(55.95°N, 3.19°W)", xy=(lon1, lat1),
                xytext=(lon1 - 4, lat1 + 1),
                arrowprops=dict(arrowstyle='->', color='green'), fontsize=9, color='green')
    ax.annotate("London\n(51.51°N, 0.13°W)", xy=(lon2, lat2),
                xytext=(lon2 + 0.5, lat2 - 2),
                arrowprops=dict(arrowstyle='->', color='red'), fontsize=9, color='red')

    # Computed distance
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlam/2)**2
    c = 2*math.atan2(math.sqrt(a), math.sqrt(1-a))
    d = R * c

    ax.text(-9, 58.5, f"Haversine distance ≈ {d:.1f} km", fontsize=10,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    ax.legend(fontsize=9)
    ax.grid(True, linestyle='--', alpha=0.4)
    save(fig, "haversine_example.pdf")


# ─────────────────────────────────────────────────────────────────
# Figure 4: Simple undirected graph (Fig 7.5 reproduction)
# ─────────────────────────────────────────────────────────────────
def fig_simple_graph():
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.set_title("A Simple Graph: G = (V, E)", fontsize=12, fontweight='bold')
    ax.set_xlim(-0.5, 3.5)
    ax.set_ylim(-0.5, 3)
    ax.axis('off')

    # Node positions: 1=(0,2), 2=(0,0), 3=(2,1), 4=(3,1)
    pos = {1: (0, 2), 2: (0, 0), 3: (2, 1), 4: (3, 1)}
    # Edges: 1-2, 1-3, 2-3, 3-4
    edges = [(1, 2), (1, 3), (2, 3), (3, 4)]

    for u, v in edges:
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        ax.plot([x1, x2], [y1, y2], 'k-', linewidth=2, zorder=1)

    for node, (x, y) in pos.items():
        circle = plt.Circle((x, y), 0.2, color='steelblue', zorder=3)
        ax.add_patch(circle)
        ax.text(x, y, str(node), ha='center', va='center',
                fontsize=12, color='white', fontweight='bold', zorder=4)

    ax.text(1.5, -0.4,
            "V = {1,2,3,4}   E = {(1,2),(1,3),(2,3),(3,4)}",
            ha='center', fontsize=9, color='gray')
    save(fig, "simple_graph.pdf")


# ─────────────────────────────────────────────────────────────────
# Figure 5: Weighted social network graph (Fig 7.6 reproduction)
# ─────────────────────────────────────────────────────────────────
def fig_weighted_social_graph():
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.set_title("Weighted Graph: Social Network", fontsize=12, fontweight='bold')
    ax.set_xlim(-1, 5)
    ax.set_ylim(-1, 4)
    ax.axis('off')

    pos = {'Jamie': (0, 3), 'Katie': (4, 3), 'Ahmed': (0, 0), 'Millie': (4, 0)}
    weights = {('Jamie', 'Katie'): 5, ('Jamie', 'Ahmed'): 1,
               ('Katie', 'Ahmed'): 6, ('Katie', 'Millie'): 12,
               ('Ahmed', 'Millie'): 5}

    for (u, v), w in weights.items():
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        ax.plot([x1, x2], [y1, y2], 'gray', linewidth=2, zorder=1)
        mx, my = (x1+x2)/2, (y1+y2)/2
        ax.text(mx, my, str(w), fontsize=10, color='darkred',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='none'),
                ha='center', va='center', zorder=3)

    colors = {'Jamie': 'steelblue', 'Katie': 'steelblue',
              'Ahmed': 'steelblue', 'Millie': 'steelblue'}
    for node, (x, y) in pos.items():
        circle = plt.Circle((x, y), 0.35, color=colors[node], zorder=4)
        ax.add_patch(circle)
        ax.text(x, y, node, ha='center', va='center',
                fontsize=9, color='white', fontweight='bold', zorder=5)

    ax.text(2, -0.8,
            "Edge weight = level of activity between users",
            ha='center', fontsize=9, color='gray')
    save(fig, "weighted_social_graph.pdf")


# ─────────────────────────────────────────────────────────────────
# Figure 6: Directed graph (Fig 7.8 reproduction: X→Y, X→Z, Y→Z, Z→Y)
# ─────────────────────────────────────────────────────────────────
def fig_directed_graph():
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.set_title("Directed Graph (Digraph)", fontsize=12, fontweight='bold')
    ax.set_xlim(-0.5, 3.5)
    ax.set_ylim(-0.5, 3.5)
    ax.axis('off')

    pos = {'X': (1, 3), 'Y': (0, 0), 'Z': (3, 1.5)}
    directed_edges = [('X', 'Y'), ('X', 'Z'), ('Y', 'Z'), ('Z', 'Y')]

    def arrow(ax, src, dst, positions, offset=0.0, color='black'):
        x1, y1 = positions[src]
        x2, y2 = positions[dst]
        dx, dy = x2 - x1, y2 - y1
        length = math.sqrt(dx**2 + dy**2)
        # shorten arrow so it doesn't overlap node circles
        shrink = 0.28
        sx = x1 + shrink * dx / length
        sy = y1 + shrink * dy / length
        ex = x2 - shrink * dx / length
        ey = y2 - shrink * dy / length
        # perpendicular offset for parallel edges
        if offset != 0:
            px, py = -dy/length * offset, dx/length * offset
            sx += px; sy += py; ex += px; ey += py
        ax.annotate('', xy=(ex, ey), xytext=(sx, sy),
                    arrowprops=dict(arrowstyle='->', color=color,
                                   lw=2, mutation_scale=15))

    arrow(ax, 'X', 'Y', pos, color='steelblue')
    arrow(ax, 'X', 'Z', pos, color='steelblue')
    arrow(ax, 'Y', 'Z', pos, offset= 0.12, color='darkorange')
    arrow(ax, 'Z', 'Y', pos, offset=-0.12, color='darkorange')

    for node, (x, y) in pos.items():
        circle = plt.Circle((x, y), 0.25, color='steelblue', zorder=4)
        ax.add_patch(circle)
        ax.text(x, y, node, ha='center', va='center',
                fontsize=12, color='white', fontweight='bold', zorder=5)

    ax.text(1.5, -0.4,
            "Arrows show direction: Z→Y and Y→Z both exist\nbut Z→X does NOT exist",
            ha='center', fontsize=9, color='gray')
    save(fig, "directed_graph.pdf")


# ─────────────────────────────────────────────────────────────────
# Figure 7: Road network as directed graph (Fig 7.9: j1–j4, s1–s6)
# ─────────────────────────────────────────────────────────────────
def fig_road_network_graph():
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.set_title("Road Network as a Directed Graph", fontsize=12, fontweight='bold')
    ax.set_xlim(-0.5, 5.5)
    ax.set_ylim(-0.5, 3.5)
    ax.axis('off')

    junctions = {'j1': (0.5, 3), 'j2': (4.5, 3), 'j3': (0.5, 0.5), 'j4': (4.5, 0.5)}

    # Edges: s1 j1→j2, s2 j2→j1 (bidir top), s3 j1→j3 (one-way), s4 j2→j4 (one-way)
    # s5 j4→j2 (one-way), s6 j3→j4 and j4→j3 (bidir bottom)
    edges = [
        ('j1', 'j2', 's1', 0.15, 'steelblue'),
        ('j2', 'j1', 's2', -0.15, 'steelblue'),
        ('j1', 'j3', 's3', 0, 'darkorange'),
        ('j2', 'j4', 's4', 0, 'darkorange'),
        ('j3', 'j4', 's6', 0.1, 'green'),
        ('j4', 'j3', 's6b', -0.1, 'green'),
        ('j4', 'j2', 's5', 0, 'purple'),
    ]

    def road_arrow(ax, src, dst, label, offset, color, positions):
        x1, y1 = positions[src]
        x2, y2 = positions[dst]
        dx, dy = x2 - x1, y2 - y1
        length = math.sqrt(dx**2 + dy**2)
        shrink = 0.25
        sx = x1 + shrink * dx / length
        sy = y1 + shrink * dy / length
        ex = x2 - shrink * dx / length
        ey = y2 - shrink * dy / length
        if offset != 0:
            px, py = -dy/length * offset, dx/length * offset
            sx += px; sy += py; ex += px; ey += py
        ax.annotate('', xy=(ex, ey), xytext=(sx, sy),
                    arrowprops=dict(arrowstyle='->', color=color, lw=1.8,
                                   mutation_scale=14))
        mx, my = (sx+ex)/2, (sy+ey)/2
        if offset != 0:
            px2, py2 = -dy/length * offset, dx/length * offset
            mx += px2; my += py2
        ax.text(mx, my, label, fontsize=8, ha='center', va='center', color=color,
                bbox=dict(boxstyle='round,pad=0.1', facecolor='white', alpha=0.7, edgecolor='none'))

    for src, dst, lbl, off, col in edges:
        road_arrow(ax, src, dst, lbl, off, col, junctions)

    for jname, (x, y) in junctions.items():
        circle = plt.Circle((x, y), 0.22, color='steelblue', zorder=4)
        ax.add_patch(circle)
        ax.text(x, y, jname, ha='center', va='center',
                fontsize=9, color='white', fontweight='bold', zorder=5)

    ax.text(2.5, -0.4,
            "j1–j4 = junctions (nodes);  s1–s6 = road segments (edges)\n"
            "Two edges between j1↔j2 and j3↔j4 for bidirectional roads",
            ha='center', fontsize=8, color='gray')
    save(fig, "road_network_graph.pdf")


# ─────────────────────────────────────────────────────────────────
# Figure 8: Adjacency matrix heatmap for the road network
# ─────────────────────────────────────────────────────────────────
def fig_adjacency_matrix():
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle("Adjacency Matrix for Road Network (j1–j4)", fontsize=12, fontweight='bold')

    # From Fig 7.9/Table 7.1:
    # j1→j2=1, j1→j3=1; j2→j1=1, j2→j4=1; j3→j4=1; j4→j2=1, j4→j3=1
    labels = ['j1', 'j2', 'j3', 'j4']
    A = np.array([
        [0, 1, 1, 0],  # j1
        [1, 0, 0, 1],  # j2
        [0, 0, 0, 1],  # j3
        [0, 1, 1, 0],  # j4
    ], dtype=float)

    ax = axes[0]
    im = ax.imshow(A, cmap='Blues', vmin=0, vmax=1)
    ax.set_xticks(range(4)); ax.set_yticks(range(4))
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlabel("To node", fontsize=10)
    ax.set_ylabel("From node", fontsize=10)
    ax.set_title("Adjacency Matrix (0/1)", fontsize=11)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, int(A[i, j]), ha='center', va='center',
                    fontsize=13, color='white' if A[i,j] > 0.5 else 'black',
                    fontweight='bold')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Weighted version (use small distances as weights)
    W = np.array([
        [-1,  5,  3, -1],
        [ 5, -1, -1,  7],
        [-1, -1, -1,  4],
        [-1,  7,  4, -1],
    ], dtype=float)
    W_disp = np.where(W == -1, 0, W)

    ax2 = axes[1]
    im2 = ax2.imshow(W_disp, cmap='YlOrRd', vmin=0, vmax=10)
    ax2.set_xticks(range(4)); ax2.set_yticks(range(4))
    ax2.set_xticklabels(labels, fontsize=11)
    ax2.set_yticklabels(labels, fontsize=11)
    ax2.set_xlabel("To node", fontsize=10)
    ax2.set_ylabel("From node", fontsize=10)
    ax2.set_title("Weighted Adjacency Matrix\n(distance in km, -1 = no direct link)", fontsize=11)
    for i in range(4):
        for j in range(4):
            val = int(W[i, j])
            txt = str(val) if val != -1 else '−1'
            ax2.text(j, i, txt, ha='center', va='center',
                     fontsize=12, color='black', fontweight='bold')
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    fig.tight_layout()
    save(fig, "adjacency_matrix.pdf")


# ─────────────────────────────────────────────────────────────────
# Figure 9: Crop page images from PDF (book figures 7.1–7.4, 7.7, 7.9, 7.10)
# ─────────────────────────────────────────────────────────────────
def crop_pdf_figures():
    try:
        import fitz  # pymupdf
    except ImportError:
        print("  pymupdf not available, skipping PDF crops")
        return

    if not os.path.exists(PDF_PATH):
        print(f"  PDF not found at {PDF_PATH}, skipping crops")
        return

    doc = fitz.open(PDF_PATH)
    # Pages are 0-indexed; book pages 148–160 correspond to PDF pages ~157–169
    # We'll crop the page images from the book PDF
    # Chapter 7 starts at book page 147 → PDF index ~156
    # Map: (pdf_page_index, crop_rect_as_fraction [x0,y0,x1,y1], output_name)
    crops = [
        # Fig 7.3: Booth's 1889 map — p160 of PDF (book p149 area)
        # We'll crop the whole page and let LaTeX handle sizing
        (159, (0.05, 0.30, 0.95, 0.75), "fig_booth_map_crop.png"),
        # Fig 7.4: OSM aerial shot — book p150 area
        (160, (0.05, 0.05, 0.95, 0.60), "fig_osm_aerial_crop.png"),
        # Fig 7.7: Edinburgh road map + OSM graph  — book p156 area
        (166, (0.05, 0.35, 0.95, 0.95), "fig_edinburgh_map_crop.png"),
        # Fig 7.10: Junction restricted turns  — book p158 area
        (168, (0.30, 0.35, 0.95, 0.85), "fig_junction_turns_crop.png"),
    ]

    for pdf_idx, frac, fname in crops:
        try:
            page = doc[pdf_idx]
            r = page.rect
            clip = fitz.Rect(
                r.x0 + frac[0] * r.width,
                r.y0 + frac[1] * r.height,
                r.x0 + frac[2] * r.width,
                r.y0 + frac[3] * r.height,
            )
            mat = fitz.Matrix(2, 2)  # 2x zoom for clarity
            pix = page.get_pixmap(matrix=mat, clip=clip)
            out_path = os.path.join(FIGURES_DIR, fname)
            pix.save(out_path)
            print(f"  cropped {fname}")
        except Exception as e:
            print(f"  failed to crop {fname}: {e}")

    doc.close()


# ─────────────────────────────────────────────────────────────────
# Figure 10: Haversine formula step-by-step numerical breakdown
# ─────────────────────────────────────────────────────────────────
def fig_haversine_steps():
    """Visual breakdown of the Haversine computation steps."""
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.axis('off')
    ax.set_title("Haversine Formula: Step-by-Step Computation\n"
                 "Edinburgh (55.95°N, 3.19°W) → London (51.51°N, 0.13°W)",
                 fontsize=11, fontweight='bold')

    steps = [
        r"$\Delta\phi$ (delta-phi) $= \phi_2 - \phi_1 = 51.51° - 55.95° = -4.44°$ "
        r"$= -0.0775$ rad",
        r"$\Delta\lambda$ (delta-lambda) $= \lambda_2 - \lambda_1 = -0.13° - (-3.19°) = 3.06°$ "
        r"$= 0.0534$ rad",
        r"$a = \sin^2(\Delta\phi/2) + \cos(\phi_1)\cdot\cos(\phi_2)\cdot\sin^2(\Delta\lambda/2)$",
        r"$a = \sin^2(-0.0388) + \cos(0.977)\cdot\cos(0.899)\cdot\sin^2(0.0267)$",
        r"$a = 0.001505 + 0.5553 \times 0.6248 \times 0.000712 = 0.001505 + 0.000247 = 0.001752$",
        r"$c$ (central angle) $= 2\,\mathrm{atan2}(\sqrt{a},\,\sqrt{1-a}) "
        r"= 2\,\mathrm{atan2}(0.04181,\, 0.9991) = 0.08363$ rad",
        r"$d = R \times c = 6371 \times 0.08363 \approx \mathbf{532.7}$ km",
    ]

    colors = ['#eaf4fb', '#eaf4fb', '#fff3cd', '#fff3cd', '#fff3cd', '#d4edda', '#f8d7da']
    for i, (step, col) in enumerate(zip(steps, colors)):
        y = 0.92 - i * 0.135
        ax.text(0.02, y, f"Step {i+1}:", transform=ax.transAxes,
                fontsize=9, fontweight='bold', va='top')
        ax.text(0.12, y, step, transform=ax.transAxes,
                fontsize=9, va='top',
                bbox=dict(boxstyle='round,pad=0.3', facecolor=col, alpha=0.8, edgecolor='none'))

    save(fig, "haversine_steps.pdf")


# ─────────────────────────────────────────────────────────────────
# Figure 11: Object-oriented vs adjacency-matrix representation comparison
# ─────────────────────────────────────────────────────────────────
def fig_graph_representations():
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    fig.suptitle("Graph Data Structure Representations", fontsize=13, fontweight='bold')

    # ---- Left: OO class diagram ----
    ax = axes[0]
    ax.set_title("Object-Oriented Design", fontsize=11)
    ax.set_xlim(0, 10); ax.set_ylim(0, 8); ax.axis('off')

    def box(ax, x, y, w, h, title, attrs, color='#d0e8f1'):
        rect = mpatches.FancyBboxPatch((x, y), w, h,
                                        boxstyle="round,pad=0.1",
                                        facecolor=color, edgecolor='steelblue', linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h - 0.3, title, ha='center', va='top',
                fontsize=10, fontweight='bold')
        ax.plot([x, x+w], [y+h-0.55, y+h-0.55], 'steelblue', linewidth=1)
        for i, attr in enumerate(attrs):
            ax.text(x + 0.15, y + h - 0.85 - i*0.35, attr, va='top', fontsize=8)

    box(ax, 0.5, 4.5, 2.8, 3.0, "Graph", ["edges: List[Edge]", "nodes: List[Node]"])
    box(ax, 4.0, 4.5, 2.8, 3.0, "Edge",  ["weight: float", "source: Node", "dest: Node"])
    box(ax, 4.0, 0.5, 2.8, 3.0, "Node",  ["node_id: int", "lat: float", "lon: float"])

    # Composition arrows
    ax.annotate('', xy=(4.0, 6.0), xytext=(3.3, 6.0),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    ax.text(3.35, 6.1, '1..*', fontsize=8)
    ax.text(3.9, 6.1, '*', fontsize=8)

    ax.annotate('', xy=(5.4, 4.5), xytext=(5.4, 3.5),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    ax.text(5.5, 4.0, '2', fontsize=8)

    ax.text(5, 0.1, "Scalability issue:\ntens of thousands of objects", ha='center',
            fontsize=8.5, color='red', style='italic')

    # ---- Right: Adjacency matrix ----
    ax2 = axes[1]
    ax2.set_title("Adjacency Matrix (compact, fast)", fontsize=11)
    A = np.array([
        [0, 1, 1, 0],
        [1, 0, 0, 1],
        [0, 0, 0, 1],
        [0, 1, 1, 0],
    ], dtype=float)
    labels = ['j1', 'j2', 'j3', 'j4']
    im = ax2.imshow(A, cmap='Blues', vmin=0, vmax=1)
    ax2.set_xticks(range(4)); ax2.set_yticks(range(4))
    ax2.set_xticklabels(labels, fontsize=12)
    ax2.set_yticklabels(labels, fontsize=12)
    ax2.set_xlabel("To →", fontsize=10); ax2.set_ylabel("From ↓", fontsize=10)
    for i in range(4):
        for j in range(4):
            ax2.text(j, i, int(A[i,j]), ha='center', va='center',
                     fontsize=14, fontweight='bold',
                     color='white' if A[i,j] > 0.5 else 'black')
    ax2.text(1.5, 4.6,
             "O(1) lookup: A[i][j] tells us directly\nif j1 connects to j2",
             ha='center', fontsize=9, color='darkgreen', style='italic',
             transform=ax2.transData)

    fig.tight_layout()
    save(fig, "graph_representations.pdf")


# ─────────────────────────────────────────────────────────────────
# Figure 12: GeoJSON/SVG example code structure (text diagram)
# ─────────────────────────────────────────────────────────────────
def fig_vector_formats():
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.axis('off')
    ax.set_title("Common Vector Data Formats", fontsize=12, fontweight='bold')

    data = [
        ["Format", "Extension", "Human-readable?", "Use case", "Typical size"],
        ["SVG",     ".svg",     "Yes (XML)",        "Web/print maps", "Small–Medium"],
        ["GeoJSON", ".geojson", "Yes (JSON)",        "Web GIS, APIs",  "Medium"],
        ["Shapefile",".shp+",   "No (binary)",       "Desktop GIS",    "Large"],
        ["OSM",     ".osm",     "Yes (XML)",         "OpenStreetMap",  "Very large"],
        ["KML",     ".kml",     "Yes (XML)",         "Google Earth",   "Medium"],
    ]

    col_widths = [0.13, 0.12, 0.18, 0.22, 0.18]
    col_starts = [0.02, 0.16, 0.29, 0.48, 0.71]
    row_colors = ['#2c6e9e'] + ['#eaf4fb', '#d0e8f1'] * 3

    for r, row in enumerate(data):
        facecolor = row_colors[r] if r < len(row_colors) else '#eaf4fb'
        textcolor = 'white' if r == 0 else 'black'
        fw = 'bold' if r == 0 else 'normal'
        y = 0.95 - r * 0.14
        rect = mpatches.FancyBboxPatch((0.01, y - 0.11), 0.97, 0.13,
                                        boxstyle="round,pad=0.005",
                                        facecolor=facecolor,
                                        edgecolor='gray', linewidth=0.5,
                                        transform=ax.transAxes)
        ax.add_patch(rect)
        for c, (cell, xs) in enumerate(zip(row, col_starts)):
            ax.text(xs + 0.005, y - 0.035, cell,
                    transform=ax.transAxes,
                    fontsize=9, va='center', color=textcolor, fontweight=fw)

    save(fig, "vector_formats.pdf")


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Generating figures for Chapter 7: GeoSpatial Data...")
    fig_raster_vs_vector()
    fig_coordinate_system()
    fig_haversine()
    fig_haversine_steps()
    fig_simple_graph()
    fig_weighted_social_graph()
    fig_directed_graph()
    fig_road_network_graph()
    fig_adjacency_matrix()
    fig_graph_representations()
    fig_vector_formats()
    crop_pdf_figures()
    print("Done.")
