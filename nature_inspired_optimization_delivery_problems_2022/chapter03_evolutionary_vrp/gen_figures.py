"""
gen_figures.py  --  Chapter 3: Applying Evolution to Vehicle Routing Problems
Generate all figures needed for chapter03_slides.tex.
Uses matplotlib (Agg) and PyMuPDF for PDF crops.
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe

FIGURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

BOOK_PDF = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..",
    "Nature Inspired Optimisation for Delivery Problems 2022.pdf",
)


def save(name):
    path = os.path.join(FIGURES_DIR, name)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved {name}")


# ---------------------------------------------------------------------------
# Figure 1: Fitness Landscape (conceptual 1-D)
# ---------------------------------------------------------------------------
def fig_fitness_landscape():
    x = np.linspace(0, 10, 600)
    y = (2.0 * np.sin(x) + 0.5 * np.sin(3 * x) + 0.2 * np.sin(5 * x)
         + 0.04 * x ** 1.3)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x, y, "b-", lw=2)
    ax.fill_between(x, y.min() - 0.3, y, alpha=0.12, color="blue")

    idx_min = int(np.argmin(y))
    ax.plot(x[idx_min], y[idx_min], "r*", ms=18, label="Global optimum (lowest distance)", zorder=5)

    local_mask = (x > 4.5) & (x < 7.5)
    idx_local = int(np.argmin(y[local_mask])) + int(np.where(local_mask)[0][0])
    ax.plot(x[idx_local], y[idx_local], "g^", ms=13, label="Local optimum (trap)", zorder=5)

    ax.set_xlabel("Solution space (genotype permutation)", fontsize=12)
    ax.set_ylabel("Fitness  =  total route distance", fontsize=12)
    ax.set_title("Fitness Landscape — Every point is a solution; height = cost", fontsize=13)
    ax.legend(fontsize=11)
    ax.set_xlim(0, 10)
    plt.tight_layout()
    save("fig_fitness_landscape.pdf")


# ---------------------------------------------------------------------------
# Figure 2: EA Cycle flowchart
# ---------------------------------------------------------------------------
def fig_ea_cycle():
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")

    box_style = dict(boxstyle="round,pad=0.4", facecolor="#d0e8ff",
                     edgecolor="#336699", linewidth=1.8)
    kw = dict(ha="center", va="center", fontsize=9, fontweight="bold", bbox=box_style)

    nodes = [
        (5.0, 5.4, "Initialise Population\n(random genotypes)"),
        (5.0, 4.3, "Evaluate Fitness\n(decode & compute distance)"),
        (5.0, 3.2, "evalsBudget > 0?"),
        (5.0, 2.1, "Create Child\n(tournament select, crossover/clone, mutate, evaluate)"),
        (5.0, 1.0, "Replace Weakest if Child Improves"),
    ]
    for (x, y, txt) in nodes:
        ax.text(x, y, txt, **kw)

    # arrows
    arrow_kw = dict(arrowstyle="->", color="#336699", lw=1.6)
    for (_, y0, _), (_, y1, _) in zip(nodes[:-1], nodes[1:]):
        ax.annotate("", xy=(5.0, y1 + 0.32), xytext=(5.0, y0 - 0.32),
                    arrowprops=arrow_kw)

    # "Yes" label
    ax.text(5.15, 2.66, "Yes", fontsize=9, color="green", fontweight="bold")
    # "No" branch -> output
    ax.annotate("", xy=(8.6, 5.4), xytext=(6.4, 3.2),
                arrowprops=dict(arrowstyle="->", color="red", lw=1.5,
                                connectionstyle="arc3,rad=-0.35"))
    ax.text(7.9, 4.5, "No\n(output best)", fontsize=8.5, color="red",
            ha="center", fontweight="bold",
            bbox=dict(boxstyle="round", facecolor="#FDEDEC", edgecolor="red"))
    # loop back arrow
    ax.annotate("", xy=(3.3, 3.2), xytext=(3.3, 1.0),
                arrowprops=dict(arrowstyle="->", color="#777", lw=1.4,
                                connectionstyle="angle,angleA=0,angleB=-90"))
    ax.text(2.2, 2.1, "loop\nback", fontsize=8, color="#777", ha="center")

    ax.set_title("The Evolutionary Algorithm Cycle", fontsize=13, fontweight="bold", pad=8)
    plt.tight_layout()
    save("fig_ea_cycle.pdf")


# ---------------------------------------------------------------------------
# Figure 3: Chromosome encoding (genotype vs. phenotype)
# ---------------------------------------------------------------------------
def fig_chromosome_encoding():
    fig, axes = plt.subplots(2, 1, figsize=(10, 3.8))

    # ── top: genotype ──
    ax = axes[0]
    visits = [3, 7, 1, 9, 4, 2, 8, 5, 6]
    route_colors = ["#4C72B0"] * 3 + ["#55A868"] * 3 + ["#C44E52"] * 3
    ax.set_xlim(-0.2, len(visits))
    ax.set_ylim(0, 1.2)
    ax.axis("off")
    ax.set_title("Genotype: single flat permutation of all customer IDs (the 'grand tour')",
                 fontsize=9, loc="left")
    for i, (v, col) in enumerate(zip(visits, route_colors)):
        rect = mpatches.FancyBboxPatch((i + 0.05, 0.2), 0.9, 0.6,
                                       boxstyle="round,pad=0.05",
                                       facecolor=col, edgecolor="white", lw=1.5)
        ax.add_patch(rect)
        ax.text(i + 0.5, 0.5, str(v), ha="center", va="center",
                color="white", fontsize=12, fontweight="bold")
        ax.text(i + 0.5, 0.1, f"g{i+1}", ha="center", va="center",
                fontsize=7, color="#555")

    # ── bottom: phenotype ──
    ax2 = axes[1]
    routes = [[3, 7, 1], [9, 4, 2], [8, 5, 6]]
    route_cols = ["#4C72B0", "#55A868", "#C44E52"]
    ax2.set_xlim(-0.2, len(visits))
    ax2.set_ylim(0, 1.2)
    ax2.axis("off")
    ax2.set_title("Phenotype: decoded into separate vehicle routes (each route starts and ends at depot D)",
                  fontsize=9, loc="left")
    x = 0
    for r_idx, (route, col) in enumerate(zip(routes, route_cols)):
        ax2.text(x - 0.3 + 0.5, 0.5, "D", ha="center", va="center",
                 fontsize=10, fontweight="bold", color="red")
        for v in route:
            rect = mpatches.FancyBboxPatch((x + 0.05, 0.2), 0.9, 0.6,
                                           boxstyle="round,pad=0.05",
                                           facecolor=col, edgecolor="white", lw=1.5)
            ax2.add_patch(rect)
            ax2.text(x + 0.5, 0.5, str(v), ha="center", va="center",
                     color="white", fontsize=12, fontweight="bold")
            x += 1
        ax2.text(x + 0.2, 0.5, "D", ha="center", va="center",
                 fontsize=10, fontweight="bold", color="red")
        if r_idx < 2:
            ax2.axvline(x + 0.5, color="#999", lw=1.2, ls="--", ymin=0.15, ymax=0.85)
        x += 1
        ax2.text(x - len(route)/2 - 0.3, 1.05, f"Route {r_idx+1}",
                 ha="center", fontsize=8, color=col, fontweight="bold")

    fig.suptitle("CVRP Chromosome: Genotype (permutation) decoded into Phenotype (routes)",
                 fontsize=11, fontweight="bold")
    plt.tight_layout(h_pad=0.5)
    save("fig_chromosome_encoding.pdf")


# ---------------------------------------------------------------------------
# Figure 4: Order-1 Crossover
# ---------------------------------------------------------------------------
def fig_crossover():
    p1 = [3, 7, 1, 9, 4, 2, 8, 5, 6]
    p2 = [9, 2, 6, 1, 7, 5, 3, 4, 8]
    xp = 4
    child = list(p1[:xp])
    for g in p2:
        if g not in child:
            child.append(g)

    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.axis("off")
    ax.set_xlim(-1.2, 10)
    ax.set_ylim(-0.3, 5.0)
    ax.set_title("Order-1 Crossover: copying a segment from P1, then filling from P2",
                 fontsize=12, fontweight="bold")

    def draw_row(genes, y, label, hirange=None, base_col="#d0e8ff"):
        ax.text(-0.5, y + 0.3, label, fontsize=10, fontweight="bold", ha="right")
        for i, g in enumerate(genes):
            fc = "#FFD700" if (hirange and i < hirange) else base_col
            rect = mpatches.FancyBboxPatch((i, y), 0.9, 0.6,
                                           boxstyle="round,pad=0.05",
                                           facecolor=fc, edgecolor="black", lw=1.2)
            ax.add_patch(rect)
            ax.text(i + 0.45, y + 0.3, str(g), ha="center", va="center",
                    fontsize=12, fontweight="bold")
        if hirange:
            ax.annotate("", xy=(hirange, y + 0.75), xytext=(0, y + 0.75),
                        arrowprops=dict(arrowstyle="<->", color="red", lw=2))
            ax.text(hirange / 2, y + 0.88, f"copy first {hirange} genes from P1",
                    ha="center", fontsize=8, color="red")

    draw_row(p1, 3.5, "Parent 1:", hirange=xp, base_col="#d0e8ff")
    draw_row(p2, 2.2, "Parent 2:", base_col="#d4f1c4")
    draw_row(child, 0.6, "Child:", hirange=xp, base_col="#ffe0cc")
    ax.text(4.5, 0.1,
            "Gold cells copied directly from P1.  Remaining cells filled in the order they appear in P2.",
            ha="center", fontsize=9, color="#444")
    plt.tight_layout()
    save("fig_crossover.pdf")


# ---------------------------------------------------------------------------
# Figure 5: Mutation — random gene relocation
# ---------------------------------------------------------------------------
def fig_mutation():
    before = [3, 7, 1, 9, 4, 2, 8, 5, 6]
    after = before.copy()
    ri, ins = 4, 1  # remove index 4 (value=4), re-insert at index 1
    gene = after.pop(ri)
    after.insert(ins, gene)

    fig, ax = plt.subplots(figsize=(9, 3.0))
    ax.axis("off")
    ax.set_xlim(-1.2, 9.5)
    ax.set_ylim(-0.2, 2.8)
    ax.set_title("Mutation: randomly remove one gene and re-insert it at a new position",
                 fontsize=12, fontweight="bold")

    def draw_row(genes, y, label, hi_idx):
        ax.text(-0.5, y + 0.3, label, fontsize=10, fontweight="bold", ha="right")
        for k, g in enumerate(genes):
            fc = "#FF6B6B" if k == hi_idx else "#d0e8ff"
            rect = mpatches.FancyBboxPatch((k, y), 0.9, 0.6,
                                           boxstyle="round,pad=0.05",
                                           facecolor=fc, edgecolor="black", lw=1.2)
            ax.add_patch(rect)
            ax.text(k + 0.45, y + 0.3, str(g), ha="center", va="center",
                    fontsize=12, fontweight="bold")

    draw_row(before, 1.7, "Before:", ri)
    draw_row(after, 0.4, "After:", ins)

    ax.annotate("", xy=(ins + 0.45, 1.55), xytext=(ri + 0.45, 1.55),
                arrowprops=dict(arrowstyle="<->", color="red", lw=2,
                                connectionstyle="arc3,rad=-0.4"))
    ax.text((ri + ins) / 2 + 0.45, 1.2, "move", ha="center", fontsize=9, color="red")
    plt.tight_layout()
    save("fig_mutation.pdf")


# ---------------------------------------------------------------------------
# Figure 6: Tournament selection
# ---------------------------------------------------------------------------
def fig_tournament_selection():
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5.2)

    population = [
        (1.2, 4.2, "Ind A\nfit=450"),
        (1.2, 3.0, "Ind B\nfit=380"),
        (1.2, 1.8, "Ind C\nfit=510"),
        (3.8, 4.2, "Ind D\nfit=290"),
        (3.8, 3.0, "Ind E\nfit=420"),
        (3.8, 1.8, "Ind F\nfit=360"),
    ]
    for x, y, label in population:
        ax.text(x, y, label, ha="center", va="center", fontsize=8.5,
                bbox=dict(boxstyle="round", facecolor="#d0e8ff", edgecolor="black"))

    # Tournament 1: A vs B -> B wins (lower is better)
    for (x0, y0), (x1, y1) in [((1.7, 4.2), (5.2, 3.6)), ((1.7, 3.0), (5.2, 3.2))]:
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="->", color="orange", lw=1.5))
    ax.text(5.3, 3.4, "Tourn. 1", fontsize=8, color="orange")
    ax.text(7.3, 3.4, "WINNER:\nInd B  (fit=380)", fontsize=9, fontweight="bold",
            bbox=dict(boxstyle="round", facecolor="#A9DFBF", edgecolor="green"), color="darkgreen")

    # Tournament 2: D vs F -> D wins
    for (x0, y0), (x1, y1) in [((4.4, 4.2), (5.2, 2.0)), ((4.4, 1.8), (5.2, 1.6))]:
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="->", color="purple", lw=1.5))
    ax.text(5.3, 1.8, "Tourn. 2", fontsize=8, color="purple")
    ax.text(7.3, 1.8, "WINNER:\nInd D  (fit=290)", fontsize=9, fontweight="bold",
            bbox=dict(boxstyle="round", facecolor="#A9DFBF", edgecolor="green"), color="darkgreen")

    ax.text(0.1, 4.9, "Population pool", fontsize=10, fontweight="bold")
    ax.text(7.3, 4.9, "Selected parents", fontsize=10, fontweight="bold")
    ax.set_title("Tournament Selection (size=2): draw 2 at random, keep the one with lower fitness",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    save("fig_tournament_selection.pdf")


# ---------------------------------------------------------------------------
# Figure 7: Crop Fig. 3.1 from book PDF (parent phenotypes + child)
# ---------------------------------------------------------------------------
def fig_crop_parent_child():
    try:
        import fitz
        doc = fitz.open(BOOK_PDF)
        # Fig 3.1 is on book page 45 (p057.png).
        # PDF 0-indexed: page 44 (book pg 45 from cover = PDF page ~56 counting from 0)
        # p057.png = book page 45  => PDF index = 44
        page = doc[44]
        rect = fitz.Rect(150, 30, 580, 440)
        pix = page.get_pixmap(clip=rect, dpi=180)
        out_path = os.path.join(FIGURES_DIR, "fig_parent_child_routes.png")
        pix.save(out_path)
        doc.close()
        print("  saved fig_parent_child_routes.png (PDF crop)")
    except Exception as e:
        print(f"  WARNING: PDF crop failed ({e}); generating placeholder")
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.text(0.5, 0.5,
                "Fig. 3.1 — Two parent VRP solutions (P1, P2)\nand the child created by crossover",
                ha="center", va="center", fontsize=13,
                bbox=dict(boxstyle="round", facecolor="#f0f0f0"))
        ax.axis("off")
        save("fig_parent_child_routes.pdf")


# ---------------------------------------------------------------------------
# Figure 8: Class diagram (UML-style)
# ---------------------------------------------------------------------------
def fig_class_diagram():
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.set_title("VRPea Implementation — Key Classes", fontsize=11, fontweight="bold")

    classes = [
        (5.0, 3.8, "VRPea  (Algorithm)", [
            "POP_SIZE = 500",
            "TOUR_SIZE = 2",
            "XO_RATE = 0.7",
            "evalsBudget = 1,000,000",
            "solve()",
            "InitialisePopulation()",
            "tournamentSelection(k)",
            "tournamentSelectWorst(k)",
        ]),
        (1.8, 1.2, "Individual", [
            "genotype: List[VRPVisit]",
            "phenotype: List[Route]",
            "mutate()",
            "evaluate() -> distance",
            "copy()",
        ]),
        (8.2, 1.2, "CVRPProblem", [
            "capacity: int",
            "getDistance(routes)",
            "getSolution()",
        ]),
    ]
    bw, bh = 3.0, 2.2
    for cx, cy, title, methods in classes:
        rect = mpatches.FancyBboxPatch((cx - bw/2, cy - bh/2), bw, bh,
                                       boxstyle="round,pad=0.1",
                                       facecolor="#EEF4FF", edgecolor="#4C72B0", lw=1.8)
        ax.add_patch(rect)
        ax.text(cx, cy + bh/2 - 0.22, title, ha="center", va="top",
                fontsize=9, fontweight="bold", color="#1a1a5e")
        ax.plot([cx - bw/2 + 0.05, cx + bw/2 - 0.05],
                [cy + bh/2 - 0.46, cy + bh/2 - 0.46], color="#4C72B0", lw=0.8)
        for k, m in enumerate(methods):
            ax.text(cx, cy + bh/2 - 0.65 - k * 0.28, m,
                    ha="center", va="top", fontsize=6.8, color="#333")

    # arrows
    ax.annotate("", xy=(3.1, 2.3), xytext=(3.6, 2.8),
                arrowprops=dict(arrowstyle="<|-", color="#4C72B0", lw=1.3))
    ax.text(2.7, 2.65, "uses", fontsize=8, color="#555", style="italic")
    ax.annotate("", xy=(6.9, 2.3), xytext=(6.4, 2.8),
                arrowprops=dict(arrowstyle="<|-", color="#4C72B0", lw=1.3))
    ax.text(7.0, 2.65, "references", fontsize=8, color="#555", style="italic")
    plt.tight_layout()
    save("fig_class_diagram.pdf")


# ---------------------------------------------------------------------------
# Figure 9: Convergence curves (Fig. 3.3 style)
# ---------------------------------------------------------------------------
def fig_convergence():
    np.random.seed(42)
    t = np.linspace(0, 420, 400)
    base = 480 * np.exp(-t / 75) + 125
    noise = np.cumsum(np.random.randn(400)) * 0.4
    fitness = np.maximum(base + noise, 115)
    best = np.minimum.accumulate(fitness)

    Q, T = 250, 130

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)

    ax = axes[0]
    ax.plot(t, best, "b-", lw=2)
    ax.set_xlabel("Time (s)", fontsize=11)
    ax.set_ylabel("Fitness  =  total distance", fontsize=11)
    ax.set_title("(a) Typical EA run — best individual's fitness drops fast,\nthen levels off as it converges",
                 fontsize=9)
    ax.set_xlim(0, 420); ax.set_ylim(0, 510)
    ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.plot(t, best, "b-", lw=2)
    ax2.axhline(Q, color="orange", ls="--", lw=1.8, label=f"Q = {Q}  (quality threshold)")
    ax2.axvline(T, color="green", ls="--", lw=1.8, label=f"T = {T} s  (patience threshold)")
    mask = t <= T
    ax2.fill_between(t[mask], 0, np.minimum(best[mask], Q),
                     alpha=0.22, color="red", label="Acceptable zone")
    ax2.set_xlabel("Time (s)", fontsize=11)
    ax2.set_title("(b) Same run with thresholds Q and T.\nSolution is acceptable if fitness < Q before T.",
                  fontsize=9)
    ax2.legend(fontsize=9, loc="upper right")
    ax2.set_xlim(0, 420); ax2.set_ylim(0, 510)
    ax2.grid(True, alpha=0.3)
    ax2.text(T + 8, Q + 15, "T", fontsize=12, color="green", fontweight="bold")
    ax2.text(6, Q + 15, "Q", fontsize=12, color="orange", fontweight="bold")

    fig.suptitle("EA Convergence (Fig. 3.3 — reproduced)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    save("fig_convergence.pdf")


# ---------------------------------------------------------------------------
# Figure 10: Parameter sensitivity
# ---------------------------------------------------------------------------
def fig_parameter_sensitivity():
    pop_sizes = [50, 100, 200, 500, 1000, 2000]
    avg_pct = [8.5, 6.2, 4.1, 2.8, 1.9, 2.4]   # conceptual trend

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(pop_sizes, avg_pct, "bo-", lw=2, ms=8)
    ax.axvline(1000, color="red", ls="--", lw=1.6, label="Default pop. size = 1,000")
    ax.set_xlabel("Population size", fontsize=11)
    ax.set_ylabel("Average % above best-known solution", fontsize=11)
    ax.set_title("Effect of Population Size\n(fixed budget = 1,000,000 evaluations)",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xscale("log")
    plt.tight_layout()
    save("fig_parameter_sensitivity.pdf")


# ---------------------------------------------------------------------------
# Figure 11: EA vs Clarke-Wright bar chart
# ---------------------------------------------------------------------------
def fig_ea_vs_cw():
    categories = ["Set A\n(n=31-80)", "Set B\n(n=31-78)", "Set P\n(n=16-101)"]
    ea_wins  = [15, 12, 10]
    cw_wins  = [ 7, 12, 14]
    tied     = [ 2,  2,  2]

    x = np.arange(len(categories))
    w = 0.25
    fig, ax = plt.subplots(figsize=(8, 4.5))
    b1 = ax.bar(x - w, ea_wins, w, label="EA best",            color="#3498DB", edgecolor="black")
    b2 = ax.bar(x,     cw_wins, w, label="Clarke-Wright best", color="#E74C3C", edgecolor="black")
    b3 = ax.bar(x + w, tied,    w, label="Tied",               color="#95A5A6", edgecolor="black")
    ax.set_xticks(x); ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylabel("Number of problem instances", fontsize=11)
    ax.set_title("EA vs Clarke-Wright: Best Solution per Instance\n"
                 "(Augerat 1995 benchmark; EA is competitive on smaller instances)",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)
    for bars in [b1, b2, b3]:
        for rect in bars:
            h = rect.get_height()
            ax.text(rect.get_x() + rect.get_width()/2, h + 0.08,
                    str(int(h)), ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    save("fig_ea_vs_cw.pdf")


# ---------------------------------------------------------------------------
# Figure 12: EA flowchart (Algorithm 8)
# ---------------------------------------------------------------------------
def fig_ea_flowchart():
    fig, ax = plt.subplots(figsize=(6, 9))
    ax.axis("off")
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 9)

    steps = [
        (3, 8.4, "START",                                       "#2ECC71"),
        (3, 7.3, "Initialise Population\n(random genotypes)",   "#AED6F1"),
        (3, 6.2, "Evaluate all individuals",                    "#AED6F1"),
        (3, 5.1, "evalsBudget > 0?",                            "#FAD7A0"),
        (3, 4.0, "Create child\n(select, crossover/clone,\nmutate, evaluate)",
         "#AED6F1"),
        (3, 2.8, "Replace worst if\nchild improves it",         "#AED6F1"),
        (3, 1.7, "Update bestSoFar",                            "#AED6F1"),
        (3, 0.7, "Output bestSoFar",                            "#A9DFBF"),
    ]
    box_kw = dict(ha="center", va="center", fontsize=8.5, fontweight="bold")
    for x, y, txt, col in steps:
        ax.text(x, y, txt, bbox=dict(boxstyle="round,pad=0.35",
                facecolor=col, edgecolor="#336699", lw=1.5), **box_kw)

    ys = [s[1] for s in steps]
    for i in range(len(ys) - 2):
        ax.annotate("", xy=(3, ys[i+1] + 0.33), xytext=(3, ys[i] - 0.33),
                    arrowprops=dict(arrowstyle="->", color="#336699", lw=1.5))

    ax.text(3.12, (ys[3] + ys[4])/2 + 0.05, "Yes", fontsize=9, color="green")
    # No branch
    ax.annotate("", xy=(3, ys[7] + 0.3), xytext=(4.9, ys[3]),
                arrowprops=dict(arrowstyle="->", color="red", lw=1.5,
                                connectionstyle="arc3,rad=-0.35"))
    ax.text(5.1, (ys[3]+ys[7])/2, "No", fontsize=9, color="red")
    # Loop back
    ax.annotate("", xy=(1.2, ys[3]), xytext=(1.2, ys[6] - 0.35),
                arrowprops=dict(arrowstyle="->", color="gray", lw=1.3,
                                connectionstyle="angle,angleA=0,angleB=-90"))
    ax.text(0.5, (ys[3]+ys[6])/2, "loop\nback", fontsize=7.5, color="gray", ha="center")

    ax.set_title("Algorithm 8: Evolutionary Algorithm (VRPea)", fontsize=11, fontweight="bold")
    plt.tight_layout()
    save("fig_ea_flowchart.pdf")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Generating Chapter 3 figures...")
    fig_fitness_landscape()
    fig_ea_cycle()
    fig_chromosome_encoding()
    fig_crossover()
    fig_mutation()
    fig_tournament_selection()
    fig_crop_parent_child()
    fig_class_diagram()
    fig_convergence()
    fig_parameter_sensitivity()
    fig_ea_vs_cw()
    fig_ea_flowchart()
    print("All figures done.")
