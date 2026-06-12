"""
gen_figures.py  — Chapter 9: Linking to Real-world Data Sources and Services
Generates all figures needed for chapter09_slides.tex using matplotlib (Agg backend)
and crops relevant pages from the book PDF via PyMuPDF (fitz).
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import matplotlib.patheffects as pe
import numpy as np

FIGURES_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

PDF_PATH = os.path.join(
    os.path.dirname(__file__), "..",
    "Nature Inspired Optimisation for Delivery Problems 2022.pdf"
)

# ─────────────────────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────────────────────
def save(fig, name):
    path = os.path.join(FIGURES_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  saved {name}")


def crop_pdf_page(page_num, rect_frac, out_name, dpi=180):
    """
    Crop a rectangular region from a PDF page and save as PNG.
    page_num : 0-indexed PDF page number
    rect_frac: (x0_frac, y0_frac, x1_frac, y1_frac) as fractions of page dims
    """
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(PDF_PATH)
        page = doc[page_num]
        pw, ph = page.rect.width, page.rect.height
        x0, y0, x1, y1 = rect_frac
        clip = fitz.Rect(x0 * pw, y0 * ph, x1 * pw, y1 * ph)
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        pix = page.get_pixmap(matrix=mat, clip=clip)
        out_path = os.path.join(FIGURES_DIR, out_name)
        pix.save(out_path)
        doc.close()
        print(f"  cropped PDF page {page_num+1} -> {out_name}")
    except Exception as e:
        print(f"  [WARN] crop_pdf_page failed for {out_name}: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 — Chapter overview / pipeline diagram
# ─────────────────────────────────────────────────────────────────────────────
def fig_pipeline():
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 3)
    ax.axis('off')
    fig.patch.set_facecolor('#f8f8f8')

    steps = [
        ("Address\nStrings", 1.0, "#4e79a7"),
        ("Geocoding\n(Nominatim/\nGoogle)", 3.2, "#f28e2b"),
        ("Routing\nService\n(OSRM/GH)", 5.6, "#59a14f"),
        ("Distance\nMatrix", 7.8, "#e15759"),
        ("Export\n(KML/GPX/\nCSV)", 10.0, "#76b7b2"),
    ]

    for label, cx, color in steps:
        box = FancyBboxPatch((cx - 0.9, 0.6), 1.8, 1.8,
                             boxstyle="round,pad=0.1",
                             facecolor=color, edgecolor='white', linewidth=2,
                             alpha=0.88)
        ax.add_patch(box)
        ax.text(cx, 1.5, label, ha='center', va='center',
                fontsize=9, color='white', fontweight='bold')

    arrow_xs = [(x[1] + 0.9, steps[i+1][1] - 0.9)
                for i, x in enumerate(steps[:-1])]
    for x0, x1 in arrow_xs:
        ax.annotate("", xy=(x1, 1.5), xytext=(x0, 1.5),
                    arrowprops=dict(arrowstyle="-|>", color='#333',
                                   lw=2, mutation_scale=15))

    ax.set_title("Chapter 9 — Real-World Data Integration Pipeline",
                 fontsize=12, fontweight='bold', pad=6)
    save(fig, "fig_pipeline.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 — Geocoding concept: address → (lat, lon)
# ─────────────────────────────────────────────────────────────────────────────
def fig_geocoding_concept():
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis('off')
    fig.patch.set_facecolor('#fafafa')

    # Address box
    ab = FancyBboxPatch((0.3, 1.2), 2.8, 1.6,
                        boxstyle="round,pad=0.15",
                        facecolor="#d0e8f5", edgecolor="#4e79a7", linewidth=2)
    ax.add_patch(ab)
    ax.text(1.7, 2.35, "Address String", ha='center', va='center',
            fontsize=10, fontweight='bold', color='#333')
    ax.text(1.7, 1.85, '"10 Colinton Road,\nEdinburgh"', ha='center',
            va='center', fontsize=9, color='#555', style='italic')

    # Geocoder box
    gb = FancyBboxPatch((3.8, 1.2), 2.4, 1.6,
                        boxstyle="round,pad=0.15",
                        facecolor="#fff3cd", edgecolor="#f28e2b", linewidth=2)
    ax.add_patch(gb)
    ax.text(5.0, 2.2, "Geocoder", ha='center', va='center',
            fontsize=10, fontweight='bold', color='#b35900')
    ax.text(5.0, 1.75, "(Nominatim / Google)", ha='center', va='center',
            fontsize=8.5, color='#666')

    # Coordinate box
    cb = FancyBboxPatch((7.0, 1.2), 2.6, 1.6,
                        boxstyle="round,pad=0.15",
                        facecolor="#d4edda", edgecolor="#59a14f", linewidth=2)
    ax.add_patch(cb)
    ax.text(8.3, 2.35, "Coordinates", ha='center', va='center',
            fontsize=10, fontweight='bold', color='#1a5c2a')
    ax.text(8.3, 1.85, "lat=55.9244\nlon=−3.2096", ha='center',
            va='center', fontsize=9, color='#333')

    # Arrows
    ax.annotate("", xy=(3.8, 2.0), xytext=(3.1, 2.0),
                arrowprops=dict(arrowstyle="-|>", color='#4e79a7', lw=2.5))
    ax.annotate("", xy=(7.0, 2.0), xytext=(6.2, 2.0),
                arrowprops=dict(arrowstyle="-|>", color='#59a14f', lw=2.5))

    ax.set_title("Geocoding: Converting an Address to Geographic Coordinates",
                 fontsize=11, fontweight='bold')
    save(fig, "fig_geocoding_concept.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3 — Forward vs Reverse Geocoding
# ─────────────────────────────────────────────────────────────────────────────
def fig_forward_reverse():
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
    fig.patch.set_facecolor('#fafafa')

    for ax in axes:
        ax.set_xlim(0, 8)
        ax.set_ylim(0, 3.5)
        ax.axis('off')

    # Forward geocoding
    ax = axes[0]
    ax.set_title("Forward Geocoding", fontsize=11, fontweight='bold',
                 color='#4e79a7', pad=4)
    ax.add_patch(FancyBboxPatch((0.3, 1.0), 2.8, 1.4,
                                boxstyle="round,pad=0.1",
                                facecolor="#d0e8f5", edgecolor="#4e79a7", lw=2))
    ax.text(1.7, 1.7, '"Napier University\nMerchiston"', ha='center',
            va='center', fontsize=9, style='italic')
    ax.add_patch(FancyBboxPatch((4.8, 1.0), 2.8, 1.4,
                                boxstyle="round,pad=0.1",
                                facecolor="#d4edda", edgecolor="#59a14f", lw=2))
    ax.text(6.2, 1.7, "55.9329°N\n3.2139°W", ha='center',
            va='center', fontsize=9, fontweight='bold')
    ax.annotate("", xy=(4.8, 1.7), xytext=(3.1, 1.7),
                arrowprops=dict(arrowstyle="-|>", color='#333', lw=2))
    ax.text(3.95, 2.0, "geocode()", ha='center', fontsize=8.5,
            color='#555')

    # Reverse geocoding
    ax = axes[1]
    ax.set_title("Reverse Geocoding", fontsize=11, fontweight='bold',
                 color='#e15759', pad=4)
    ax.add_patch(FancyBboxPatch((0.3, 1.0), 2.8, 1.4,
                                boxstyle="round,pad=0.1",
                                facecolor="#d4edda", edgecolor="#59a14f", lw=2))
    ax.text(1.7, 1.7, "55.9329°N\n3.2139°W", ha='center',
            va='center', fontsize=9, fontweight='bold')
    ax.add_patch(FancyBboxPatch((4.8, 1.0), 2.8, 1.4,
                                boxstyle="round,pad=0.1",
                                facecolor="#ffe0e0", edgecolor="#e15759", lw=2))
    ax.text(6.2, 1.7, '"Napier Univ.\nMerchiston"', ha='center',
            va='center', fontsize=9, style='italic')
    ax.annotate("", xy=(4.8, 1.7), xytext=(3.1, 1.7),
                arrowprops=dict(arrowstyle="-|>", color='#333', lw=2))
    ax.text(3.95, 2.0, "reverseGeocode()", ha='center', fontsize=8.5,
            color='#555')

    fig.suptitle("Two Directions of Geocoding", fontsize=12,
                 fontweight='bold', y=1.02)
    save(fig, "fig_forward_reverse.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4 — Geocoder class hierarchy (UML-style)
# ─────────────────────────────────────────────────────────────────────────────
def fig_geocoder_hierarchy():
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis('off')
    fig.patch.set_facecolor('#fafafa')

    def uml_box(ax, x, y, w, h, title, methods, hdr_color, body_color):
        # header
        ax.add_patch(FancyBboxPatch((x, y + h*0.55), w, h*0.45,
                                   boxstyle="square,pad=0",
                                   facecolor=hdr_color, edgecolor='#555', lw=1.5))
        ax.text(x + w/2, y + h*0.77, title, ha='center', va='center',
                fontsize=9.5, fontweight='bold', color='white')
        # body
        ax.add_patch(FancyBboxPatch((x, y), w, h*0.55,
                                   boxstyle="square,pad=0",
                                   facecolor=body_color, edgecolor='#555', lw=1.5))
        for i, m in enumerate(methods):
            ax.text(x + 0.1, y + h*0.45 - i*0.28, m, ha='left', va='center',
                    fontsize=8, color='#333', family='monospace')

    # Geocoder interface
    uml_box(ax, 3.2, 3.2, 3.6, 1.5, "<<interface>>\nGeocoder",
            ["+ geocode(label) : LatLon",
             "+ reverseGeocode(p) : String"],
            "#4e79a7", "#e8f4fd")

    # NapierLoc
    uml_box(ax, 0.2, 0.8, 3.2, 1.8, "NapierLoc",
            ["- data : List<Entry>",
             "+ geocode(label)",
             "+ reverseGeocode(p)"],
            "#59a14f", "#eafaf0")

    # Nominatim
    uml_box(ax, 3.8, 0.8, 2.8, 1.8, "Nominatim",
            ["- baseURL : String",
             "+ geocode(label)",
             "+ reverseGeocode(p)"],
            "#f28e2b", "#fff8e8")

    # Cache
    uml_box(ax, 6.9, 0.8, 2.8, 1.8, "Cache",
            ["- cache : List<Entry>",
             "- baseCoder : Geocoder",
             "+ geocode(label)"],
            "#e15759", "#fde8e8")

    # Arrows (implements)
    for x_child, y_child in [(1.8, 2.6), (5.2, 2.6), (8.3, 2.6)]:
        ax.annotate("", xy=(x_child, 3.2), xytext=(x_child, y_child),
                    arrowprops=dict(arrowstyle="-|>", color='#555',
                                   lw=1.5, linestyle='dashed'))

    ax.text(5.0, 4.9, "Geocoder Class Hierarchy",
            ha='center', fontsize=12, fontweight='bold')
    save(fig, "fig_geocoder_hierarchy.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 5 — Caching strategy diagram
# ─────────────────────────────────────────────────────────────────────────────
def fig_cache_strategy():
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 4.5)
    ax.axis('off')
    fig.patch.set_facecolor('#fafafa')

    # Request
    ax.add_patch(FancyBboxPatch((0.2, 1.6), 1.8, 1.3,
                                boxstyle="round,pad=0.1",
                                facecolor="#4e79a7", edgecolor='#333', lw=1.5))
    ax.text(1.1, 2.25, "geocode()\nrequest", ha='center', va='center',
            fontsize=8.5, color='white', fontweight='bold')

    # Cache check
    ax.add_patch(FancyBboxPatch((2.5, 1.6), 2.2, 1.3,
                                boxstyle="round,pad=0.1",
                                facecolor="#fff3cd", edgecolor='#f28e2b', lw=2))
    ax.text(3.6, 2.25, "Check\nLocal Cache", ha='center', va='center',
            fontsize=8.5, color='#333', fontweight='bold')

    # Cache hit
    ax.add_patch(FancyBboxPatch((5.3, 2.7), 2.0, 1.0,
                                boxstyle="round,pad=0.1",
                                facecolor="#d4edda", edgecolor='#59a14f', lw=1.5))
    ax.text(6.3, 3.2, "Cache HIT\nReturn cached", ha='center', va='center',
            fontsize=8, color='#1a5c2a', fontweight='bold')

    # Cache miss → API
    ax.add_patch(FancyBboxPatch((5.3, 0.5), 2.0, 1.0,
                                boxstyle="round,pad=0.1",
                                facecolor="#ffe0e0", edgecolor='#e15759', lw=1.5))
    ax.text(6.3, 1.0, "Cache MISS\nCall Web API", ha='center', va='center',
            fontsize=8, color='#8b0000', fontweight='bold')

    # Store & return
    ax.add_patch(FancyBboxPatch((8.3, 0.5), 2.4, 1.0,
                                boxstyle="round,pad=0.1",
                                facecolor="#e8e8f8", edgecolor='#555', lw=1.5))
    ax.text(9.5, 1.0, "Store in cache\n& return result", ha='center',
            va='center', fontsize=8, color='#333', fontweight='bold')

    # Arrows
    ax.annotate("", xy=(2.5, 2.25), xytext=(2.0, 2.25),
                arrowprops=dict(arrowstyle="-|>", color='#333', lw=2))
    ax.annotate("", xy=(5.3, 3.2), xytext=(4.7, 2.9),
                arrowprops=dict(arrowstyle="-|>", color='#59a14f', lw=1.5))
    ax.text(4.7, 3.0, "Yes", fontsize=8, color='#59a14f', fontweight='bold')
    ax.annotate("", xy=(5.3, 1.0), xytext=(4.7, 1.6),
                arrowprops=dict(arrowstyle="-|>", color='#e15759', lw=1.5))
    ax.text(4.7, 1.4, "No", fontsize=8, color='#e15759', fontweight='bold')
    ax.annotate("", xy=(8.3, 1.0), xytext=(7.3, 1.0),
                arrowprops=dict(arrowstyle="-|>", color='#333', lw=1.5))

    ax.set_title("Geocoding Cache Strategy — Reducing Redundant Web API Calls",
                 fontsize=11, fontweight='bold')
    save(fig, "fig_cache_strategy.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 6 — Nominatim XML response structure
# ─────────────────────────────────────────────────────────────────────────────
def fig_nominatim_xml():
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4.5)
    ax.axis('off')
    fig.patch.set_facecolor('#1e1e2e')

    xml_lines = [
        ('<searchresults>', '#76d7c4'),
        ('  <place', '#f9e2af'),
        ('    place_id="282736548"', '#cba6f7'),
        ('    lat="55.9329"', '#a6e3a1'),
        ('    lon="-3.2139"', '#a6e3a1'),
        ('    display_name="Napier University, Merchiston"', '#89dceb'),
        ('    type="university"', '#cba6f7'),
        ('    importance="0.521"/>', '#f9e2af'),
        ('</searchresults>', '#76d7c4'),
    ]

    for i, (line, color) in enumerate(xml_lines):
        ax.text(0.3, 4.0 - i * 0.42, line, ha='left', va='center',
                fontsize=9.5, color=color, family='monospace')

    ax.set_title("Nominatim API: Example XML Response for Address Lookup",
                 fontsize=11, fontweight='bold', color='white')
    fig.patch.set_facecolor('#1e1e2e')
    ax.set_facecolor('#1e1e2e')
    save(fig, "fig_nominatim_xml.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 7 — Routing engine architecture
# ─────────────────────────────────────────────────────────────────────────────
def fig_routing_architecture():
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4.5)
    ax.axis('off')
    fig.patch.set_facecolor('#fafafa')

    # Abstract RoutingEngine
    ax.add_patch(FancyBboxPatch((4.0, 2.8), 4.0, 1.5,
                                boxstyle="round,pad=0.12",
                                facecolor="#4e79a7", edgecolor='#333', lw=2))
    ax.text(6.0, 3.85, "<<abstract>>", ha='center', va='center',
            fontsize=8, color='#cce', style='italic')
    ax.text(6.0, 3.45, "RoutingEngine", ha='center', va='center',
            fontsize=10, fontweight='bold', color='white')
    ax.text(6.0, 3.0, "findRoute(start, end, options)", ha='center',
            va='center', fontsize=8.5, color='#ddf', family='monospace')

    # HomeBrew
    ax.add_patch(FancyBboxPatch((0.5, 0.5), 3.5, 1.8,
                                boxstyle="round,pad=0.12",
                                facecolor="#59a14f", edgecolor='#333', lw=2))
    ax.text(2.25, 1.75, "HomeBrew", ha='center', va='center',
            fontsize=10, fontweight='bold', color='white')
    ax.text(2.25, 1.3, "Uses A* on local\nOSM graph data", ha='center',
            va='center', fontsize=8.5, color='#eee')
    ax.text(2.25, 0.8, "(offline, free)", ha='center', va='center',
            fontsize=8, color='#cfc', style='italic')

    # GraphHopper
    ax.add_patch(FancyBboxPatch((4.25, 0.5), 3.5, 1.8,
                                boxstyle="round,pad=0.12",
                                facecolor="#f28e2b", edgecolor='#333', lw=2))
    ax.text(6.0, 1.75, "GraphHopper", ha='center', va='center',
            fontsize=10, fontweight='bold', color='white')
    ax.text(6.0, 1.3, "Java library +\nOSM data", ha='center',
            va='center', fontsize=8.5, color='#fff')
    ax.text(6.0, 0.8, "(local, open-source)", ha='center', va='center',
            fontsize=8, color='#ffe', style='italic')

    # OSRM / Web API
    ax.add_patch(FancyBboxPatch((8.0, 0.5), 3.5, 1.8,
                                boxstyle="round,pad=0.12",
                                facecolor="#e15759", edgecolor='#333', lw=2))
    ax.text(9.75, 1.75, "OSRM / ORS", ha='center', va='center',
            fontsize=10, fontweight='bold', color='white')
    ax.text(9.75, 1.3, "Web API,\nHTTP requests", ha='center',
            va='center', fontsize=8.5, color='#fff')
    ax.text(9.75, 0.8, "(online, cloud)", ha='center', va='center',
            fontsize=8, color='#fcc', style='italic')

    # Arrows
    for cx in [2.25, 6.0, 9.75]:
        ax.annotate("", xy=(cx, 2.3), xytext=(cx, 2.8) if cx == 6.0 else (cx, 2.3),
                    arrowprops=dict(arrowstyle="-|>", color='#555', lw=1.5,
                                   linestyle='dashed'))
    # connect children to parent
    for cx in [2.25, 6.0, 9.75]:
        ax.plot([cx, cx], [2.3, 2.75], color='#555', lw=1.5,
                linestyle='dashed')
    ax.plot([2.25, 9.75], [2.75, 2.75], color='#555', lw=1.5, linestyle='dashed')
    ax.annotate("", xy=(6.0, 2.8), xytext=(6.0, 2.75),
                arrowprops=dict(arrowstyle="-|>", color='#555', lw=1.5))

    ax.set_title("RoutingEngine Class Hierarchy — Swappable Routing Back-ends",
                 fontsize=11, fontweight='bold')
    save(fig, "fig_routing_architecture.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 8 — Distance matrix concept
# ─────────────────────────────────────────────────────────────────────────────
def fig_distance_matrix():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5),
                             gridspec_kw={'width_ratios': [1, 1.4]})
    fig.patch.set_facecolor('#fafafa')

    # Left: city locations on a mini map
    ax = axes[0]
    ax.set_title("City Locations (schematic)", fontsize=10, fontweight='bold')
    locations = {
        'Edinburgh': (0.45, 0.75),
        'Glasgow':   (0.2,  0.65),
        'Stirling':  (0.35, 0.55),
        'Perth':     (0.55, 0.45),
        'Dundee':    (0.65, 0.35),
    }
    colors = ['#4e79a7', '#f28e2b', '#59a14f', '#e15759', '#76b7b2']
    for (city, (x, y)), col in zip(locations.items(), colors):
        ax.scatter(x, y, s=150, color=col, zorder=5, edgecolors='white', lw=1.5)
        ax.text(x + 0.04, y, city, fontsize=8.5, va='center', color='#333')
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0.2, 1.0)
    ax.set_facecolor('#e8f4fd')
    ax.set_xlabel("Longitude (schematic)", fontsize=8)
    ax.set_ylabel("Latitude (schematic)", fontsize=8)

    # Right: distance matrix heatmap
    ax2 = axes[1]
    cities = list(locations.keys())
    n = len(cities)
    rng = np.random.default_rng(42)
    # Symmetric matrix with realistic values (km)
    base = np.array([
        [0,   74,  56,  60,  93],
        [74,   0,  36,  88, 121],
        [56,  36,   0,  52,  85],
        [60,  88,  52,   0,  34],
        [93, 121,  85,  34,   0],
    ], dtype=float)
    im = ax2.imshow(base, cmap='YlOrRd', aspect='auto')
    ax2.set_xticks(range(n))
    ax2.set_yticks(range(n))
    ax2.set_xticklabels(cities, rotation=35, ha='right', fontsize=8)
    ax2.set_yticklabels(cities, fontsize=8)
    for i in range(n):
        for j in range(n):
            ax2.text(j, i, f"{int(base[i,j])}", ha='center', va='center',
                     fontsize=8.5, color='black' if base[i,j] < 80 else 'white',
                     fontweight='bold')
    ax2.set_title("Distance Matrix (km, road distances)", fontsize=10,
                  fontweight='bold')
    fig.colorbar(im, ax=ax2, label="Distance (km)")

    fig.tight_layout(pad=1.5)
    save(fig, "fig_distance_matrix.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 9 — Export format comparison
# ─────────────────────────────────────────────────────────────────────────────
def fig_export_formats():
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.patch.set_facecolor('#1a1a2e')
    titles = ["KML (Google Earth / Leaflet)",
              "GPX (GPS / Sat Nav)",
              "CSV (Spreadsheet / Database)"]
    contents = [
        [
            ('<Placemark>', '#76d7c4'),
            ('  <name>Start</name>', '#89dceb'),
            ('  <Point>', '#f9e2af'),
            ('    <coord>', '#a6e3a1'),
            ('     -3.21 55.93 0', '#cba6f7'),
            ('    </coord>', '#a6e3a1'),
            ('  </Point>', '#f9e2af'),
            ('</Placemark>', '#76d7c4'),
        ],
        [
            ('<gpx version="1.0">', '#76d7c4'),
            ('  <trk>', '#89dceb'),
            ('   <trkseg>', '#f9e2af'),
            ('    <trkpt', '#a6e3a1'),
            ('     lat="55.93"', '#cba6f7'),
            ('     lon="-3.21"/>', '#cba6f7'),
            ('   </trkseg>', '#f9e2af'),
            ('  </trk>', '#89dceb'),
        ],
        [
            ('label,lat,lon', '#76d7c4'),
            ('Edinburgh,55.93,-3.21', '#a6e3a1'),
            ('Glasgow,55.86,-4.25', '#a6e3a1'),
            ('Stirling,56.12,-3.94', '#a6e3a1'),
            ('Perth,56.40,-3.47', '#a6e3a1'),
            ('Dundee,56.46,-2.97', '#a6e3a1'),
            ('', '#555'),
            ('(plain text table)', '#89dceb'),
        ],
    ]
    for ax, title, lines in zip(axes, titles, contents):
        ax.set_facecolor('#1e1e2e')
        ax.set_xlim(0, 6)
        ax.set_ylim(0, 4.5)
        ax.axis('off')
        ax.set_title(title, fontsize=8.5, fontweight='bold',
                     color='white', pad=4)
        for i, (line, color) in enumerate(lines):
            ax.text(0.15, 4.1 - i * 0.47, line, ha='left', va='center',
                    fontsize=8, color=color, family='monospace')

    fig.suptitle("Export Format Comparison: KML vs GPX vs CSV",
                 fontsize=11, fontweight='bold', color='white', y=1.02)
    save(fig, "fig_export_formats.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 10 — ExportService class diagram
# ─────────────────────────────────────────────────────────────────────────────
def fig_export_service():
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 4.5)
    ax.axis('off')
    fig.patch.set_facecolor('#fafafa')

    def draw_class(ax, x, y, w, h, name, methods, hc, bc):
        ax.add_patch(FancyBboxPatch((x, y + h*0.55), w, h*0.45,
                                   boxstyle="square,pad=0",
                                   facecolor=hc, edgecolor='#555', lw=1.5))
        ax.text(x + w/2, y + h*0.77, name, ha='center', va='center',
                fontsize=9, fontweight='bold', color='white')
        ax.add_patch(FancyBboxPatch((x, y), w, h*0.55,
                                   boxstyle="square,pad=0",
                                   facecolor=bc, edgecolor='#555', lw=1.5))
        for i, m in enumerate(methods):
            ax.text(x + 0.08, y + h*0.48 - i * 0.28, m, ha='left', va='center',
                    fontsize=7.5, family='monospace', color='#333')

    draw_class(ax, 3.5, 2.8, 4.0, 1.5,
               "<<interface>>\nExportService",
               ["addTrack(List<LatLon>)",
                "addWaypoint(loc, caption)",
                "write(path, name)"],
               "#4e79a7", "#e8f4fd")

    draw_class(ax, 0.2, 0.4, 2.5, 2.0, "GPXWriter",
               ["- segments", "- waypoints",
                "+ addTrack()", "+ write()"],
               "#59a14f", "#eafaf0")

    draw_class(ax, 3.0, 0.4, 2.5, 2.0, "KMLWriter",
               ["- placemarks", "- tracks",
                "+ addTrack()", "+ write()"],
               "#f28e2b", "#fff8e8")

    draw_class(ax, 5.8, 0.4, 2.5, 2.0, "CSVWriter",
               ["- rows", "- header",
                "+ addTrack()", "+ write()"],
               "#e15759", "#fde8e8")

    draw_class(ax, 8.4, 0.4, 2.5, 2.0, "ConsoleWriter",
               ["(debug output)",
                "+ addTrack()",
                "+ addWaypoint()", "+ write()"],
               "#76b7b2", "#e8f8fa")

    # Arrows
    for cx in [1.45, 4.25, 7.05, 9.65]:
        ax.plot([cx, cx], [2.4, 2.8], color='#555', lw=1.3, linestyle='dashed')
    ax.plot([1.45, 9.65], [2.4, 2.4], color='#555', lw=1.3, linestyle='dashed')
    ax.annotate("", xy=(5.5, 2.8), xytext=(5.5, 2.4),
                arrowprops=dict(arrowstyle="-|>", color='#555', lw=1.3))

    ax.set_title("ExportService Interface and Concrete Implementations",
                 fontsize=11, fontweight='bold')
    save(fig, "fig_export_service.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 11 — Folium/web map concept
# ─────────────────────────────────────────────────────────────────────────────
def fig_map_visualisation():
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_facecolor('#b8d8f0')
    fig.patch.set_facecolor('#fafafa')

    # Schematic Scotland coastline region
    coast_x = [0.05, 0.08, 0.12, 0.18, 0.25, 0.28, 0.35,
                0.45, 0.55, 0.65, 0.72, 0.78, 0.82, 0.88, 0.92, 0.95]
    coast_y = [0.45, 0.55, 0.65, 0.72, 0.78, 0.82, 0.88,
               0.92, 0.88, 0.80, 0.72, 0.62, 0.52, 0.42, 0.35, 0.25]
    land_x = coast_x + [0.95, 0.05]
    land_y = coast_y + [0.05, 0.05]
    ax.fill(land_x, land_y, color='#c8e6c9', alpha=0.7)
    ax.plot(coast_x, coast_y, color='#555', lw=1)

    # Route points
    pts = [(0.45, 0.62), (0.38, 0.70), (0.55, 0.78),
           (0.62, 0.65), (0.50, 0.55)]
    px, py = zip(*pts)
    ax.plot(px, py, '-o', color='#e15759', lw=3, markersize=10,
            markerfacecolor='#e15759', markeredgecolor='white', markeredgewidth=2,
            zorder=5)
    labels = ['Edinburgh', 'Stirling', 'Perth', 'Dundee', 'Glasgow']
    for (x, y), lbl in zip(pts, labels):
        ax.text(x + 0.025, y + 0.015, lbl, fontsize=8.5,
                fontweight='bold', color='#1a1a2e',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                          edgecolor='#ccc', alpha=0.8))

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Longitude (schematic)", fontsize=9)
    ax.set_ylabel("Latitude (schematic)", fontsize=9)
    ax.set_title("Route Visualisation on OpenStreetMap (schematic)\n"
                 "Rendered using KML/GPX output overlaid on OSM base map",
                 fontsize=10, fontweight='bold')

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='#e15759', lw=3, label='Delivery Route'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#e15759',
               markersize=10, label='Stop / Waypoint'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=8.5)
    save(fig, "fig_map_visualisation.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 12 — Python geocoding workflow (Nominatim via geopy)
# ─────────────────────────────────────────────────────────────────────────────
def fig_python_geocoding_flow():
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis('off')
    fig.patch.set_facecolor('#fafafa')

    steps = [
        (1.0, "Import\ngeopy", "#4e79a7"),
        (3.2, "Create\nNominatim()\ngeolocator", "#f28e2b"),
        (5.4, "Call\ngeolocate(address)", "#59a14f"),
        (7.6, "Read\n.latitude\n.longitude", "#e15759"),
        (9.6, "Use in\noptimiser", "#76b7b2"),
    ]

    for cx, label, color in steps:
        w = 1.6
        ax.add_patch(FancyBboxPatch((cx - w/2, 1.0), w, 1.8,
                                   boxstyle="round,pad=0.1",
                                   facecolor=color, edgecolor='white', lw=2,
                                   alpha=0.88))
        ax.text(cx, 1.9, label, ha='center', va='center',
                fontsize=8.5, color='white', fontweight='bold')

    arrow_xs = [(steps[i][0] + 0.8, steps[i+1][0] - 0.8)
                for i in range(len(steps)-1)]
    for x0, x1 in arrow_xs:
        ax.annotate("", xy=(x1, 1.9), xytext=(x0, 1.9),
                    arrowprops=dict(arrowstyle="-|>", color='#333',
                                   lw=2, mutation_scale=14))

    ax.set_title("Python Geocoding Workflow with geopy + Nominatim",
                 fontsize=11, fontweight='bold')
    save(fig, "fig_python_geocoding_flow.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 13 — Crop Fig 9.2 (GPX visualiser map) from book PDF
# ─────────────────────────────────────────────────────────────────────────────
def fig_crop_gpx_map():
    # Book PDF page 205 (0-indexed = 204), top portion = the OSM map figure
    crop_pdf_page(204, (0.04, 0.02, 0.96, 0.52), "fig_gpx_osm_map.png", dpi=200)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Generating Chapter 9 figures ...")
    fig_pipeline()
    fig_geocoding_concept()
    fig_forward_reverse()
    fig_geocoder_hierarchy()
    fig_cache_strategy()
    fig_nominatim_xml()
    fig_routing_architecture()
    fig_distance_matrix()
    fig_export_formats()
    fig_export_service()
    fig_map_visualisation()
    fig_python_geocoding_flow()
    fig_crop_gpx_map()
    print("All figures generated successfully.")
