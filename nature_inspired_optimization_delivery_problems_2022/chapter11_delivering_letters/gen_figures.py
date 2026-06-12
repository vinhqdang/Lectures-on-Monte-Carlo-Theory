"""
gen_figures.py  --  Chapter 11: Delivering Letters
Generates all figures needed for chapter11_slides.tex.
Uses matplotlib (Agg backend) and pymupdf for PDF crops.
Run with: conda run -n py313 python3 gen_figures.py
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(SCRIPT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

BOOK_PDF = os.path.join(
    os.path.dirname(SCRIPT_DIR),
    "Nature Inspired Optimisation for Delivery Problems 2022.pdf"
)

def save(fig, name):
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {name}")


def crop_pdf(page_num_1indexed, xywh, out_name, dpi=180):
    """Crop a rectangular region from a PDF page. page_num_1indexed is 1-based."""
    try:
        import fitz
        doc = fitz.open(BOOK_PDF)
        page = doc[page_num_1indexed - 1]
        x, y, w, h = xywh
        clip = fitz.Rect(x, y, x + w, y + h)
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        pix = page.get_pixmap(matrix=mat, clip=clip)
        out_path = os.path.join(FIG_DIR, out_name)
        pix.save(out_path)
        doc.close()
        print(f"  saved {out_name}  (PDF crop page {page_num_1indexed})")
    except Exception as e:
        print(f"  WARNING: PDF crop failed for {out_name}: {e}")
        return False
    return True


# ══════════════════════════════════════════════════════════════════════════════
# Book figure crops
# ══════════════════════════════════════════════════════════════════════════════

def crop_all_book_figures():
    # Fig 11.1  --  street map with grouped sections  (book page 224, PDF page 231)
    crop_pdf(231, (30, 50, 530, 320), "fig_book_street_map.png", dpi=200)
    # Fig 11.2  --  SBR decoding steps on map  (book page 227, PDF page 234)
    crop_pdf(234, (30, 30, 530, 560), "fig_book_sbr_decode.png", dpi=180)
    # Fig 11.3  --  class diagram  (book page 231, PDF page 238)
    crop_pdf(238, (70, 370, 450, 290), "fig_book_class_diagram.png", dpi=180)
    # Fig 11.4  --  Greenbank study area  (book page 233, PDF page 240)
    crop_pdf(240, (30, 50, 530, 370), "fig_book_greenbank_area.png", dpi=180)
    # Fig 11.5  --  best SBR-EA route  (book page 235, PDF page 242)
    crop_pdf(242, (30, 30, 530, 500), "fig_book_greenbank_route.png", dpi=180)


# ══════════════════════════════════════════════════════════════════════════════
# FIG 1  --  Urban street network schematic
# ══════════════════════════════════════════════════════════════════════════════
def fig_street_layout():
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.set_xlim(0, 10); ax.set_ylim(0, 7)
    ax.set_aspect('equal'); ax.axis('off')
    ax.set_facecolor('#eef4e8'); fig.patch.set_facecolor('#eef4e8')

    road_c = '#b0b0b0'
    for (x0,y0,x1,y1) in [(0,2,10,2),(0,5,10,5)]:
        ax.plot([x0,x1],[y0,y1], color=road_c, lw=20, solid_capstyle='round', zorder=1)
    for (x,y0,y1) in [(2,0,7),(5,0,7),(8,0,7)]:
        ax.plot([x,x],[y0,y1], color=road_c, lw=20, solid_capstyle='round', zorder=1)

    blocks = [
        (0.3,2.3,1.5,2.4),(2.3,2.3,2.4,2.4),(5.3,2.3,2.4,2.4),(8.3,2.3,1.4,2.4),
        (0.3,5.3,1.5,1.4),(2.3,5.3,2.4,1.4),(5.3,5.3,2.4,1.4),(8.3,5.3,1.4,1.4),
    ]
    for (x,y,w,h) in blocks:
        ax.add_patch(Rectangle((x,y),w,h, color='#c8e6c9', ec='#558b2f', lw=1, zorder=2))

    np.random.seed(42)
    for xh in np.linspace(0.5,9.5,22):
        if all(abs(xh-jx)>0.4 for jx in [2,5,8]):
            ax.plot(xh,2.5,'s',color='#e53935',ms=3,zorder=4,alpha=0.8)
            ax.plot(xh,1.5,'s',color='#e53935',ms=3,zorder=4,alpha=0.8)
            ax.plot(xh,5.5,'s',color='#e53935',ms=3,zorder=4,alpha=0.8)
            ax.plot(xh,4.5,'s',color='#e53935',ms=3,zorder=4,alpha=0.8)
    for yh in np.linspace(0.3,6.7,16):
        if all(abs(yh-jy)>0.4 for jy in [2,5]):
            for xx in [1.5,2.5,4.5,5.5,7.5,8.5]:
                ax.plot(xx,yh,'s',color='#e53935',ms=3,zorder=4,alpha=0.8)

    # Section labels
    labels = [
        (1.05,1.2,"ClunyGdns1"),(3.5,1.2,"ClunyDrv1"),(6.5,1.2,"BraidAv1"),
        (1.05,3.3,"HermitageGdns1"),(3.5,3.3,"ClunyDrv1\n(both sides)"),
        (9.2,3.3,"BraidAv1"),
    ]
    for (lx,ly,lt) in labels:
        ax.text(lx,ly,lt, ha='center', fontsize=6.5, color='#1a237e', style='italic', zorder=5)

    # Junctions
    for (jx,jy),jl in zip([(2,2),(5,2),(8,2),(2,5),(5,5),(8,5)],
                           ['J1','J2','J3','J4','J5','J6']):
        ax.plot(jx,jy,'o',color='#f57f17',ms=9,zorder=6,mec='#e65100',mew=1.2)
        ax.text(jx,jy-0.28,jl, ha='center',fontsize=7,color='#e65100',fontweight='bold',zorder=7)

    ax.set_title("Urban Street Network -- Street Sections and Junctions\n"
                 "(Red squares = delivery addresses, orange circles = junctions)",
                 fontsize=10, fontweight='bold', pad=8)
    save(fig, "fig_street_layout.pdf")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 2  --  SBR chromosome
# ══════════════════════════════════════════════════════════════════════════════
def fig_chromosome():
    fig, ax = plt.subplots(figsize=(12, 2.5))
    ax.set_xlim(0, 12); ax.set_ylim(0, 2.5); ax.axis('off')

    genes = ["ClunyGdns1","ClunyDrv1","ClunyDrv1","BraidAv1",
             "HermitageGdns1","HermitageGdns1","BraidAv1","ClunyGdns1"]
    colors = ['#bbdefb','#c8e6c9','#c8e6c9','#ffe0b2',
              '#f8bbd0','#f8bbd0','#ffe0b2','#bbdefb']
    n = len(genes); w = 12.0/n
    for i,(g,c) in enumerate(zip(genes,colors)):
        rect = FancyBboxPatch((i*w+0.04,0.55), w-0.08, 1.0,
                              boxstyle="round,pad=0.06",
                              facecolor=c, edgecolor='#37474f', lw=1.2)
        ax.add_patch(rect)
        ax.text(i*w+w/2, 1.05, g, ha='center', va='center',
                fontsize=7.5, fontweight='bold', color='#1a237e')
        ax.text(i*w+w/2, 0.35, f"Gene {i+1}", ha='center', fontsize=6.5, color='#555')

    # Pair brackets
    for (i1,i2,col) in [(1,2,'#4caf50'),(4,5,'#e91e63'),(3,6,'#ff9800'),(0,7,'#2196f3')]:
        x1 = i1*w+w/2; x2 = i2*w+w/2
        ax.annotate('', xy=(x2,1.70), xytext=(x1,1.70),
                    arrowprops=dict(arrowstyle='<->', color=col, lw=1.8))
        ax.text((x1+x2)/2, 1.85, 'paired', ha='center', fontsize=6, color=col)

    ax.set_title("SBR Chromosome: each gene = one street section\n"
                 "Duplicate genes (same colour, connected by arrows) = double-sided streets",
                 fontsize=10, fontweight='bold', y=0.97)
    save(fig, "fig_chromosome.pdf")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 3  --  Delivery patterns
# ══════════════════════════════════════════════════════════════════════════════
def fig_delivery_patterns():
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    titles = ["One Side\n(start=end junction)",
              "Both Sides\n(U-turn at end)",
              "Cross-Over\n(start J1, end J2)"]
    colors = ['#1976d2','#388e3c','#d32f2f']

    for ax, title, col in zip(axes, titles, colors):
        ax.set_xlim(-0.5, 5.5); ax.set_ylim(-0.4, 2.4)
        ax.set_aspect('equal'); ax.axis('off')
        ax.set_facecolor('#fafafa')
        ax.add_patch(Rectangle((-0.2,0.65), 5.4, 0.7, color='#cfd8dc', zorder=1))
        for xh in [0.5,1.5,2.5,3.5,4.5]:
            ax.plot(xh, 1.65, 's', color='#e53935', ms=9, zorder=3)
            ax.plot(xh, 0.35, 's', color='#e53935', ms=9, zorder=3)
        ax.plot(0.0,1.0,'o',color='#f57f17',ms=11,zorder=4)
        ax.plot(5.0,1.0,'o',color='#f57f17',ms=11,zorder=4)
        ax.text(-0.05,0.68,'J1',fontsize=7,ha='center',color='#e65100',fontweight='bold')
        ax.text(5.05,0.68,'J2',fontsize=7,ha='center',color='#e65100',fontweight='bold')

        if 'One' in title:
            xs=[0,0.5,1.5,2.5,3.5,4.5,5]
            ys=[1,1.65,1.65,1.65,1.65,1.65,1]
            ax.plot(xs,ys,'-',color=col,lw=2,zorder=5)
            ax.annotate('',xy=(0.5,1.65),xytext=(0,1),
                        arrowprops=dict(arrowstyle='->',color=col,lw=1.8))
        elif 'U-turn' in title or 'Both' in title:
            xs=[0,0.5,1.5,2.5,3.5,4.5,5,4.5,3.5,2.5,1.5,0.5,0]
            ys=[1,1.65,1.65,1.65,1.65,1.65,1,0.35,0.35,0.35,0.35,0.35,1]
            ax.plot(xs,ys,'-',color=col,lw=2,zorder=5)
            ax.annotate('',xy=(0.5,1.65),xytext=(0,1),
                        arrowprops=dict(arrowstyle='->',color=col,lw=1.8))
        else:
            xs=[0,0.5,1.5,2.5,3.5,4.5,5,4.5,3.5,2.5,1.5,0.5,0]
            ys=[1,1.65,1.65,1.65,1.65,1.65,0.35,0.35,0.35,0.35,0.35,0.35,0.35]
            ax.plot(xs,ys,'-',color=col,lw=2,zorder=5)
            ax.annotate('',xy=(0.5,1.65),xytext=(0,1),
                        arrowprops=dict(arrowstyle='->',color=col,lw=1.8))

        ax.set_title(title, fontsize=9.5, fontweight='bold', color=col, pad=4)

    fig.suptitle("SBR Delivery Patterns for a Street Section\n"
                 "Orange circles=junctions, red squares=delivery addresses",
                 fontsize=10, fontweight='bold')
    fig.tight_layout(rect=[0,0,1,0.88])
    save(fig, "fig_delivery_patterns.pdf")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 4  --  SBR decoding flowchart
# ══════════════════════════════════════════════════════════════════════════════
def fig_decoding_flowchart():
    fig, ax = plt.subplots(figsize=(7, 10))
    ax.set_xlim(0,7); ax.set_ylim(0,10); ax.axis('off')

    def box(cx,cy,text,fc='#e3f2fd',w=4.0,h=0.6,fs=8):
        rect = FancyBboxPatch((cx-w/2,cy-h/2),w,h,
                              boxstyle="round,pad=0.08",
                              facecolor=fc, edgecolor='#1565c0', lw=1.2)
        ax.add_patch(rect)
        ax.text(cx,cy,text,ha='center',va='center',fontsize=fs,color='#0d47a1')

    def diamond(cx,cy,text,fc='#fff9c4',w=4.2,h=0.75):
        pts = np.array([[cx,cy+h/2],[cx+w/2,cy],[cx,cy-h/2],[cx-w/2,cy]])
        ax.add_patch(plt.Polygon(pts,closed=True,facecolor=fc,edgecolor='#f57f17',lw=1.5))
        ax.text(cx,cy,text,ha='center',va='center',fontsize=8,fontweight='bold',color='#e65100')

    def arr(x1,y1,x2,y2,lbl='',lc='#37474f'):
        ax.annotate('',xy=(x2,y2),xytext=(x1,y1),
                    arrowprops=dict(arrowstyle='->',color=lc,lw=1.4))
        if lbl:
            ax.text((x1+x2)/2+0.15,(y1+y2)/2,lbl,fontsize=7.5,color=lc)

    box(3.5,9.5,"START: decoder(genotype, problem)",fc='#c8e6c9')
    box(3.5,8.7,"deliveries=[]  geneCount=0  dist=0")
    diamond(3.5,7.75,"geneCount < genotype.length?")
    box(3.5,6.85,"genes[i]==genes[i+1]?\n→ doubleSided=True",fc='#fff3e0')
    box(3.5,5.95,"applyPattern(prevDel, current, next, doubleSided)")
    box(3.5,5.05,"dist += walkingDist(lastDel, nextDel)")
    box(3.5,4.15,"deliveries.append(street)  geneCount++")
    box(3.5,3.25,"dist += walkingDist(lastDel, end) x deliveriesLeft",fc='#e8f5e9')
    box(3.5,2.35,"RETURN dist",fc='#c8e6c9')

    arr(3.5,9.2,3.5,9.0); arr(3.5,8.4,3.5,8.12)
    arr(3.5,7.38,3.5,7.15,'Yes','#388e3c')
    arr(3.5,6.55,3.5,6.25); arr(3.5,5.65,3.5,5.35)
    arr(3.5,4.75,3.5,4.45); arr(3.5,3.85,3.5,3.55)
    arr(3.5,2.95,3.5,2.65)

    # No branch
    ax.annotate('',xy=(6.1,7.75),xytext=(5.6,7.75),
                arrowprops=dict(arrowstyle='->',color='#d32f2f',lw=1.5))
    ax.text(5.75,7.88,'No',fontsize=8,color='#d32f2f',fontweight='bold')
    box(6.55,7.75,"exit loop",fc='#ffcdd2',w=1.3,h=0.45,fs=7.5)
    arr(6.55,7.53,6.55,3.25,'','#d32f2f')
    arr(6.55,3.25,5.5,3.25,'','#d32f2f')

    ax.set_title("Algorithm 22: SBR Decoding Procedure",
                 fontsize=11, fontweight='bold', pad=10)
    save(fig, "fig_decoding_flowchart.pdf")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 5  --  Crossover operator
# ══════════════════════════════════════════════════════════════════════════════
def fig_crossover():
    fig, axes = plt.subplots(3, 1, figsize=(12, 5))
    p1 = ['A','B','B','D','E','D','F','G','A','G','H','H','C','C','F','E']
    p2 = ['B','G','H','H','B','E','E','A','A','D','G','F','F','C','C','D']
    child = ['A','B','B','D','E','D','G','H','H','F','C','C']
    cmap_gene = {'A':'#4472C4','B':'#ED7D31','C':'#FFC000','D':'#70AD47',
                 'E':'#FF4444','F':'#9966CC','G':'#00B0F0','H':'#FF69B4'}

    for ax_i,(ax,(lbl,genes)) in enumerate(zip(axes,[
        ("Parent 1 (p1)",p1),("Parent 2 (p2)",p2),("Child",child)])):
        n=len(genes); w=12.0/n
        ax.set_xlim(0,12); ax.set_ylim(0,1.1); ax.axis('off')
        for i,g in enumerate(genes):
            fc = cmap_gene.get(g,'#aaa')
            alpha = 0.85 if ax_i<2 else (1.0 if i<6 else 0.5)
            rect = FancyBboxPatch((i*w+0.02,0.1),w-0.04,0.78,
                                  boxstyle="round,pad=0.04",
                                  facecolor=fc, edgecolor='#333', lw=0.8, alpha=alpha)
            ax.add_patch(rect)
            ax.text(i*w+w/2,0.49,g,ha='center',va='center',
                    fontsize=8,fontweight='bold',color='white')
        ax.text(-0.25,0.49,lbl,ha='right',va='center',fontsize=8.5,
                fontweight='bold',color='#1a237e',transform=ax.transData)
        if ax_i==2:
            ax.text(6*w+0.1,0.95,'← from p2 (junction-connected)',
                    fontsize=7.5,color='#e65100')
            ax.axvline(6*w,color='#e65100',lw=2,ls='--',ymin=0,ymax=0.9)

    fig.suptitle("SBR Recombination: Child copies connected street sections from both parents",
                 fontsize=10, fontweight='bold')
    fig.tight_layout(rect=[0,0,1,0.94])
    save(fig, "fig_crossover.pdf")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 6  --  Cantor pairing function
# ══════════════════════════════════════════════════════════════════════════════
def fig_cantor_pairing():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    ax = axes[0]
    ax.set_xlim(-0.6,7.5); ax.set_ylim(-0.6,7.5)
    ax.set_xlabel('x (street A hash code)', fontsize=9)
    ax.set_ylabel('y (street B hash code)', fontsize=9)
    ax.set_title("Cantor Pairing: unique key z for each (x, y) pair", fontsize=9.5, fontweight='bold')
    cmap = plt.cm.YlOrRd
    for x in range(7):
        for y in range(7):
            z = int(0.5*(x+y)*(x+y+1)+y)
            c = cmap(min(z/60.0, 1.0))
            ax.add_patch(Rectangle((x-0.45,y-0.45),0.9,0.9, color=c, zorder=2))
            ax.text(x,y,str(z),ha='center',va='center',fontsize=7,
                    color='white' if z>30 else '#212121',fontweight='bold',zorder=3)
    ax.set_xticks(range(7)); ax.set_yticks(range(7))
    ax.grid(False)
    ax.text(3,7.2,r"$z = 0.5\,(x+y)\,(x+y+1)+y$",
            ha='center',fontsize=10,color='#c00',fontweight='bold')

    ax2 = axes[1]
    ax2.set_xlim(0,10); ax2.set_ylim(0,8); ax2.axis('off')
    ax2.set_title("HashMap Cache Architecture\n(avoids redundant routing API calls)", fontsize=9.5, fontweight='bold')

    def rbox(x,y,w,h,text,fc='#e3f2fd',ec='#1565c0',fs=8.5):
        rect = FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.1",
                              facecolor=fc, edgecolor=ec, lw=1.2)
        ax2.add_patch(rect)
        ax2.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=fs,color='#0d47a1')

    rbox(0.5,6.5,5,1.0,"Journey ID = cantorPair(hashA, hashB)",fc='#fff9c4',ec='#f57f17')
    rbox(0.5,5.0,5,1.0,"HashMap<Long, Double>\nkey=journey_id, value=distance_km",fc='#e3f2fd',fs=8)
    rbox(0.5,3.5,5,1.0,"cache.get(key)\nHit? return cached value instantly",fc='#c8e6c9',ec='#388e3c')
    rbox(0.5,2.0,5,1.0,"Miss? call routing API, cache.put(key,dist)",fc='#ffcdd2',ec='#c62828',fs=8)
    rbox(0.5,0.7,5,0.8,"return distance",fc='#c8e6c9',ec='#388e3c')

    for (y1,y2) in [(6.5,6.0),(5.0,4.5),(3.5,3.0),(2.0,1.5)]:
        ax2.annotate('',xy=(3,y2),xytext=(3,y1),
                     arrowprops=dict(arrowstyle='->',color='#37474f',lw=1.5))

    save(fig, "fig_cantor_pairing.pdf")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 7  --  Results comparison
# ══════════════════════════════════════════════════════════════════════════════
def fig_results_comparison():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Distance
    ax1 = axes[0]
    solvers = ['NN\nHeuristic','SBR-EA','TSP-EA']
    avg_d = [10.2, 7.7, 20.0]; min_d = [10.2, 6.8, 18.1]
    x = np.arange(3)
    b1 = ax1.bar(x-0.22, avg_d, 0.38, label='Avg. Distance', color=['#64b5f6','#81c784','#e57373'])
    b2 = ax1.bar(x+0.22, min_d, 0.38, label='Min. Distance', color=['#1565c0','#2e7d32','#c62828'], alpha=0.85)
    ax1.set_xticks(x); ax1.set_xticklabels(solvers, fontsize=10)
    ax1.set_ylabel('Route distance (km)', fontsize=10)
    ax1.set_title('Route Length Comparison\n(Greenbank area, Edinburgh)', fontsize=10, fontweight='bold')
    ax1.legend(fontsize=9); ax1.set_ylim(0,26); ax1.grid(axis='y',alpha=0.35)
    for bar in list(b1)+list(b2):
        ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                 f'{bar.get_height():.1f}', ha='center', fontsize=8.5, fontweight='bold')

    # Time
    ax2 = axes[1]
    times = [20.3, 9.6, 932.4]
    bars = ax2.bar(solvers, times, color=['#64b5f6','#81c784','#e57373'],
                   edgecolor='#37474f', lw=1.0)
    ax2.set_ylabel('Average computation time (s)', fontsize=10)
    ax2.set_title('Computation Time Comparison\n(100,000 evaluations budget)', fontsize=10, fontweight='bold')
    ax2.set_yscale('log'); ax2.grid(axis='y',alpha=0.35)
    for bar,t in zip(bars,times):
        ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()*1.15,
                 f'{t:.1f}s', ha='center', fontsize=9.5, fontweight='bold')
    ax2.text(0.97,0.95,'TSP-EA is ~100x slower\nyet finds worse routes',
             transform=ax2.transAxes, ha='right', va='top', fontsize=8.5,
             bbox=dict(fc='#fff3cd',ec='#c9a227',pad=5))

    fig.tight_layout()
    save(fig, "fig_results_comparison.pdf")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 8  --  SBR vs TSP search-space
# ══════════════════════════════════════════════════════════════════════════════
def fig_sbr_vs_tsp():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    from math import lgamma

    for ax, (label, col, n_pts, note) in zip(axes, [
        ("TSP: 442 individual addresses\nSearch space = 442!/2  5.5e978",
         '#e57373', 60, "Each node = one delivery address"),
        ("SBR: ~48 street sections\nSearch space = 48!  1.2e61",
         '#81c784', 48, "Each node = one street section (10+ addresses)")
    ]):
        ax.set_xlim(-1.3,1.3); ax.set_ylim(-1.3,1.3)
        ax.set_aspect('equal'); ax.axis('off')
        ax.set_title(label, fontsize=9, fontweight='bold', color='#1a237e', pad=6)
        ax.text(0,-1.28,note,ha='center',fontsize=8,color='#555',style='italic')

        np.random.seed(99 if 'TSP' in label else 42)
        theta = np.linspace(0,2*np.pi,n_pts,endpoint=False)
        r = 0.75 + 0.18*np.random.randn(n_pts)
        xs = r*np.cos(theta); ys = r*np.sin(theta)
        ms = 2.5 if 'TSP' in label else 9
        ax.plot(xs,ys,'o',color=col,ms=ms,alpha=0.8,zorder=3)

        order = np.random.permutation(n_pts)
        rx = xs[order]; ry = ys[order]
        ax.plot(np.append(rx,rx[0]),np.append(ry,ry[0]),'-',color=col,alpha=0.35,lw=1,zorder=2)

        n_str = "n=442" if 'TSP' in label else "n=48 sections"
        ax.text(0,0,n_str,ha='center',va='center',fontsize=11,
                fontweight='bold',color=col,alpha=0.45)

    fig.suptitle("SBR Dramatically Reduces the Search Space Compared to TSP",
                 fontsize=11, fontweight='bold')
    fig.tight_layout(rect=[0,0,1,0.92])
    save(fig, "fig_sbr_vs_tsp.pdf")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Generating figures for Chapter 11: Delivering Letters...")

    print("\n[1] Street layout schematic"); fig_street_layout()
    print("\n[2] Chromosome representation"); fig_chromosome()
    print("\n[3] Delivery patterns"); fig_delivery_patterns()
    print("\n[4] Decoding flowchart"); fig_decoding_flowchart()
    print("\n[5] Crossover operator"); fig_crossover()
    print("\n[6] Cantor pairing / cache"); fig_cantor_pairing()
    print("\n[7] Results comparison"); fig_results_comparison()
    print("\n[8] SBR vs TSP"); fig_sbr_vs_tsp()
    print("\n[9] Book figure crops from PDF"); crop_all_book_figures()

    print("\nAll done.")
