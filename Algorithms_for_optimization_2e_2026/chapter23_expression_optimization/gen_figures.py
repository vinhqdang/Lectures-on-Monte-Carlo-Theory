"""
gen_figures.py  –  Generate all figures for Chapter 23: Expression Optimization
Algorithms for Optimization, 2nd ed. (Kochenderfer & Wheeler, 2026)

Run with:
  conda run -n py313 python3 gen_figures.py
"""

import os
import sys
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, Circle, FancyBboxPatch
import matplotlib.patheffects as pe

# ------------------------------------------------------------------
# Output directory (figures/ relative to this file)
# ------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(HERE, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

def savefig(name):
    path = os.path.join(FIG_DIR, name)
    plt.savefig(path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  saved {path}")


# ==================================================================
# Figure 1: Expression tree for x + ln(2)
# (Fig 23.1 / 23.2 in book — parse-tree derivation)
# ==================================================================
def fig_expression_tree():
    fig, axes = plt.subplots(1, 3, figsize=(9, 3.5))

    # Helper to draw a node
    def node(ax, x, y, label, filled=False, color="#4da6ff"):
        if filled:
            c = Circle((x, y), 0.28, color=color, zorder=3)
        else:
            c = Circle((x, y), 0.28, color="white", ec="black", lw=1.5, zorder=3)
        ax.add_patch(c)
        ax.text(x, y, label, ha="center", va="center", fontsize=11,
                fontweight="bold" if filled else "normal", color="white" if filled else "black", zorder=4)

    def arrow(ax, x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2 + 0.28), xytext=(x1, y1 - 0.28),
                    arrowprops=dict(arrowstyle="->", color="black", lw=1.2))

    # --- Panel 1: R -> R + R (R nodes filled blue) ---
    ax = axes[0]
    ax.set_xlim(-0.8, 0.8); ax.set_ylim(-0.3, 1.3); ax.axis("off")
    ax.set_title("Step 1", fontsize=10)
    node(ax, 0, 1.0, "+")
    node(ax, -0.5, 0.3, "R", filled=True)
    node(ax,  0.5, 0.3, "R", filled=True)
    arrow(ax, 0, 1.0, -0.5, 0.3)
    arrow(ax, 0, 1.0,  0.5, 0.3)

    # --- Panel 2: left R -> x, right R -> ln(R) ---
    ax = axes[1]
    ax.set_xlim(-0.9, 0.9); ax.set_ylim(-0.7, 1.3); ax.axis("off")
    ax.set_title("Step 2", fontsize=10)
    node(ax, 0, 1.0, "+")
    node(ax, -0.5, 0.3, "x")
    node(ax,  0.5, 0.3, "ln")
    node(ax,  0.5, -0.4, "R", filled=True)
    arrow(ax, 0, 1.0, -0.5, 0.3)
    arrow(ax, 0, 1.0,  0.5, 0.3)
    arrow(ax,  0.5, 0.3,  0.5, -0.4)

    # --- Panel 3: final tree x + ln(2) ---
    ax = axes[2]
    ax.set_xlim(-0.9, 0.9); ax.set_ylim(-0.7, 1.3); ax.axis("off")
    ax.set_title("Final: x + ln(2)", fontsize=10)
    node(ax, 0, 1.0, "+")
    node(ax, -0.5, 0.3, "x")
    node(ax,  0.5, 0.3, "ln")
    node(ax,  0.5, -0.4, "2")
    arrow(ax, 0, 1.0, -0.5, 0.3)
    arrow(ax, 0, 1.0,  0.5, 0.3)
    arrow(ax,  0.5, 0.3,  0.5, -0.4)

    fig.suptitle("Expression tree derivation for $x + \\ln(2)$", fontsize=12)
    plt.tight_layout()
    savefig("fig_expression_tree.pdf")


# ==================================================================
# Figure 2: Grammar BNF overview diagram
# ==================================================================
def fig_grammar_bnf():
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis("off")

    # Draw a table-like layout
    rows = [
        (r"$\mathbb{R} \mapsto \mathbb{R} + \mathbb{R}$",  "R expands to sum of two R's"),
        (r"$\mathbb{R} \mapsto x$",                         "R becomes terminal: x"),
        (r"$\mathbb{R} \mapsto \ln(\mathbb{R})$",           "R becomes ln applied to R"),
        (r"$\mathbb{R} \mapsto 2$",                         "R becomes terminal: 2"),
    ]

    ax.text(0.05, 0.95, "Example Context-Free Grammar (BNF)", transform=ax.transAxes,
            fontsize=13, fontweight="bold", va="top")

    ax.text(0.05, 0.82, "Production Rule", transform=ax.transAxes,
            fontsize=11, fontweight="bold", va="top", color="#2255aa")
    ax.text(0.5, 0.82, "Interpretation", transform=ax.transAxes,
            fontsize=11, fontweight="bold", va="top", color="#2255aa")

    ax.plot([0.03, 0.97], [0.78, 0.78], color="#aaaaaa", lw=0.8, transform=ax.transAxes)

    for i, (rule, desc) in enumerate(rows):
        y = 0.70 - i * 0.14
        bg = "#eef4ff" if i % 2 == 0 else "white"
        ax.add_patch(FancyBboxPatch((0.03, y - 0.06), 0.94, 0.12,
                                    boxstyle="round,pad=0.01", color=bg,
                                    transform=ax.transAxes, zorder=0))
        ax.text(0.06, y, rule, transform=ax.transAxes,
                fontsize=11, va="center")
        ax.text(0.5, y, desc, transform=ax.transAxes,
                fontsize=10, va="center", color="#444444")

    ax.text(0.05, 0.08, "Nonterminals (types): $\\mathbb{R}$   |   Terminals: $x$, $2$, $+$, $\\ln$",
            transform=ax.transAxes, fontsize=10, color="#555555")
    ax.text(0.05, 0.01, "The symbol $|$ means OR (multiple production rules for one type)",
            transform=ax.transAxes, fontsize=9, color="#777777", style="italic")

    savefig("fig_grammar_bnf.pdf")


# ==================================================================
# Figure 3: Tree crossover diagram
# ==================================================================
def fig_tree_crossover():
    fig, axes = plt.subplots(1, 3, figsize=(10, 4))

    BLUE  = "#4da6ff"
    RED   = "#ff6666"
    GRAY  = "#cccccc"

    def draw_tree(ax, title, nodes_xy, edges, colors, labels):
        ax.set_xlim(-1.0, 1.0); ax.set_ylim(-0.5, 1.2); ax.axis("off")
        ax.set_title(title, fontsize=11, fontweight="bold")
        for (x1, y1), (x2, y2) in edges:
            ax.plot([x1, x2], [y1, y2], "k-", lw=1.2, zorder=1)
        for (x, y), col, lbl in zip(nodes_xy, colors, labels):
            c = Circle((x, y), 0.18, color=col, zorder=2)
            ax.add_patch(c)
            ax.text(x, y, lbl, ha="center", va="center", fontsize=10,
                    color="white", fontweight="bold", zorder=3)

    # Parent A
    nxy_a = [(0,1.0), (-0.5,0.5), (0.5,0.5), (-0.7,0.0), (-0.3,0.0)]
    edg_a = [(nxy_a[0],nxy_a[1]), (nxy_a[0],nxy_a[2]),
              (nxy_a[1],nxy_a[3]), (nxy_a[1],nxy_a[4])]
    col_a = [BLUE, BLUE, BLUE, BLUE, BLUE]
    lbl_a = ["+", "×", "x", "a", "b"]
    draw_tree(axes[0], "Parent A", nxy_a, edg_a, col_a, lbl_a)

    # Parent B — crosspoint subtree highlighted red
    nxy_b = [(0,1.0), (-0.5,0.5), (0.5,0.5), (0.3,0.0), (0.7,0.0)]
    edg_b = [(nxy_b[0],nxy_b[1]), (nxy_b[0],nxy_b[2]),
              (nxy_b[2],nxy_b[3]), (nxy_b[2],nxy_b[4])]
    col_b = [BLUE, BLUE, RED, RED, RED]
    lbl_b = ["ln", "x", "+", "c", "d"]
    draw_tree(axes[1], "Parent B (crosspoint = red)", nxy_b, edg_b, col_b, lbl_b)

    # Child: A with red subtree swapped in
    nxy_c = [(0,1.0), (-0.5,0.5), (0.5,0.5), (-0.7,0.0), (-0.3,0.0),
             (0.3,0.0), (0.7,0.0)]
    edg_c = [(nxy_c[0],nxy_c[1]), (nxy_c[0],nxy_c[2]),
              (nxy_c[1],nxy_c[3]), (nxy_c[1],nxy_c[4]),
              (nxy_c[2],nxy_c[5]), (nxy_c[2],nxy_c[6])]
    col_c = [BLUE, BLUE, RED, BLUE, BLUE, RED, RED]
    lbl_c = ["+", "×", "+", "a", "b", "c", "d"]
    draw_tree(axes[2], "Child (subtree inserted)", nxy_c, edg_c, col_c, lbl_c)

    fig.suptitle("Tree Crossover in Genetic Programming", fontsize=12)
    plt.tight_layout()
    savefig("fig_tree_crossover.pdf")


# ==================================================================
# Figure 4: Tree mutation diagram
# ==================================================================
def fig_tree_mutation():
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    BLUE = "#4da6ff"
    RED  = "#ff6666"

    def draw_tree(ax, title, nodes_xy, edges, colors):
        ax.set_xlim(-1.0, 1.0); ax.set_ylim(-0.8, 1.2); ax.axis("off")
        ax.set_title(title, fontsize=11, fontweight="bold")
        for (x1,y1),(x2,y2) in edges:
            ax.plot([x1,x2],[y1,y2],"k-",lw=1.2,zorder=1)
        for (x,y),col in zip(nodes_xy,colors):
            c = Circle((x,y),0.15,color=col,zorder=2)
            ax.add_patch(c)

    # Before: 7 blue nodes
    na = [(0,1.0),(-0.5,0.55),(0.5,0.55),(-0.75,0.1),(-0.25,0.1),(0.25,0.1),(0.75,0.1)]
    ea = [(na[0],na[1]),(na[0],na[2]),(na[1],na[3]),(na[1],na[4]),(na[2],na[5]),(na[2],na[6])]
    draw_tree(axes[0], "Before mutation", na, ea, [BLUE]*7)

    # After: right subtree replaced by red nodes
    nb = [(0,1.0),(-0.5,0.55),(0.5,0.55),(-0.75,0.1),(-0.25,0.1),
          (0.25,0.1),(0.75,0.1),(0.1,-0.25),(0.4,-0.25),(0.6,-0.25),(0.9,-0.25)]
    eb = [(nb[0],nb[1]),(nb[0],nb[2]),(nb[1],nb[3]),(nb[1],nb[4]),
          (nb[2],nb[5]),(nb[2],nb[6]),
          (nb[5],nb[7]),(nb[5],nb[8]),(nb[6],nb[9]),(nb[6],nb[10])]
    cols_b = [BLUE,BLUE,BLUE,BLUE,BLUE,RED,RED,RED,RED,RED,RED]
    draw_tree(axes[1], "After mutation (new subtree = red)", nb, eb, cols_b)

    fig.suptitle("Tree Mutation: Random Subtree Replaced", fontsize=12)
    plt.tight_layout()
    savefig("fig_tree_mutation.pdf")


# ==================================================================
# Figure 5: Tree permutation diagram
# ==================================================================
def fig_tree_permutation():
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))

    BLUE   = "#4da6ff"
    RED    = "#ff6666"
    PURPLE = "#bb77dd"

    def draw_tree(ax, title, nodes_xy, edges, colors):
        ax.set_xlim(-1.1, 1.1); ax.set_ylim(-0.5, 1.2); ax.axis("off")
        ax.set_title(title, fontsize=11)
        for (x1,y1),(x2,y2) in edges:
            ax.plot([x1,x2],[y1,y2],"k-",lw=1.2,zorder=1)
        for (x,y),col in zip(nodes_xy,colors):
            c = Circle((x,y),0.16,color=col,zorder=2)
            ax.add_patch(c)

    # Shared structure; children of node 1 permuted
    base = [(0,1.0),(-0.55,0.5),(0.55,0.5),
            (-0.85,0.0),(-0.35,0.0),(0.25,0.0),(0.75,0.0)]
    edges = [(base[0],base[1]),(base[0],base[2]),
             (base[1],base[3]),(base[1],base[4]),
             (base[2],base[5]),(base[2],base[6])]
    # Before: children order [blue, red, purple, white]
    cols_before = ["white", BLUE, RED, BLUE, RED, PURPLE, "white"]
    cols_after  = ["white", RED, BLUE, RED, BLUE, PURPLE, "white"]

    for i,(ax,cols,title) in enumerate([(axes[0],cols_before,"Before permutation"),
                                         (axes[1],cols_after, "After permutation")]):
        draw_tree(ax, title, base, edges, cols)

    fig.suptitle("Tree Permutation: Children Randomly Reordered", fontsize=12)
    plt.tight_layout()
    savefig("fig_tree_permutation.pdf")


# ==================================================================
# Figure 6: Grammatical Evolution — integer array decoding
# ==================================================================
def fig_grammatical_evolution():
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.axis("off")

    ax.text(0.5, 0.97, "Grammatical Evolution: Decoding Integer Array to Expression",
            transform=ax.transAxes, ha="center", fontsize=12, fontweight="bold")

    # Grammar rules
    rules_text = [
        r"$\mathbb{R} \mapsto \mathbb{D}\,\mathbb{D}'\,\mathbb{P}\,\mathbb{E}$",
        r"$\mathbb{D}' \mapsto \mathbb{D}\,\mathbb{D}'\;|\;\epsilon$",
        r"$\mathbb{P} \mapsto .\;\mathbb{D}\,\mathbb{D}'\;|\;\epsilon$",
        r"$\mathbb{E} \mapsto \mathbf{E}\,\mathbb{S}\,\mathbb{D}\,\mathbb{D}'\;|\;\epsilon$",
        r"$\mathbb{S} \mapsto +\;|\;-\;|\;\epsilon$",
        r"$\mathbb{D} \mapsto 0|1|2|3|4|5|6|7|8|9$",
    ]
    ax.text(0.02, 0.88, "Grammar:", transform=ax.transAxes, fontsize=10,
            fontweight="bold", color="#333333")
    for i, r in enumerate(rules_text):
        ax.text(0.05, 0.80 - i*0.10, r, transform=ax.transAxes, fontsize=10)

    # Integer design vector
    vec = [205, 52, 4, 27, 10, 59, 6]
    ax.text(0.42, 0.88, "Design vector $\\mathbf{x}$:", transform=ax.transAxes,
            fontsize=10, fontweight="bold", color="#333333")
    for j, v in enumerate(vec):
        bx = 0.44 + j * 0.072
        rect = FancyBboxPatch((bx, 0.76), 0.065, 0.10,
                               boxstyle="round,pad=0.01", facecolor="#ddeeff",
                               edgecolor="#6699cc", transform=ax.transAxes)
        ax.add_patch(rect)
        ax.text(bx + 0.032, 0.81, str(v), transform=ax.transAxes,
                ha="center", va="center", fontsize=10, fontweight="bold")

    # Decoding steps
    steps = [
        ("$\\mathbb{R}$",               "Only 1 rule → $\\mathbb{D}\\mathbb{D}'\\mathbb{P}\\mathbb{E}$"),
        ("$\\mathbb{D}$ (10 options)",   "$205\\,\\mathrm{mod}_1\\,10 = 5$ → digit \\textbf{5}"),
        ("$\\mathbb{D}'$ (2 options)",   "$52\\,\\mathrm{mod}_1\\,2 = 2$ → $\\epsilon$ (empty)"),
        ("$\\mathbb{P}$ (2 options)",    "→ $\\epsilon$ (no decimal part)"),
        ("$\\mathbb{E}$ (2 options)",    "→ $\\mathrm{E}8$ (exponent)"),
        ("Result: $4E+8$",               "= $4\\times10^{8}$"),
    ]
    ax.text(0.42, 0.64, "Decoding steps:", transform=ax.transAxes,
            fontsize=10, fontweight="bold", color="#333333")
    for i, (sym, desc) in enumerate(steps):
        y = 0.56 - i * 0.087
        ax.text(0.44, y, sym, transform=ax.transAxes, fontsize=9,
                color="#225599", fontweight="bold")
        ax.text(0.60, y, desc, transform=ax.transAxes, fontsize=9)

    savefig("fig_grammatical_evolution.pdf")


# ==================================================================
# Figure 7: Probabilistic grammar — rule weights
# ==================================================================
def fig_probabilistic_grammar():
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # --- Left: bar chart of rule weights for type A ---
    ax = axes[0]
    rules_A = [r"$\mathbb{A}\!\mapsto\!a\,\mathbb{A}$",
               r"$\mathbb{A}\!\mapsto\!a\,\mathbb{B}\,a\,\mathbb{A}$",
               r"$\mathbb{A}\!\mapsto\!\epsilon$"]
    weights_A = [1, 3, 2]
    probs_A   = [w / sum(weights_A) for w in weights_A]
    bars = ax.bar(range(3), probs_A, color=["#4da6ff","#ff9944","#66bb66"],
                  edgecolor="black", width=0.5)
    ax.set_xticks(range(3))
    ax.set_xticklabels(rules_A, fontsize=9)
    ax.set_ylabel("Probability", fontsize=10)
    ax.set_ylim(0, 0.65)
    ax.set_title("Production Probabilities for type $\\mathbb{A}$", fontsize=10)
    for bar, p in zip(bars, probs_A):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{p:.2f}", ha="center", fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    # --- Right: annotated probability tree ---
    ax2 = axes[1]
    ax2.set_xlim(-0.2, 1.2); ax2.set_ylim(-0.3, 1.2); ax2.axis("off")
    ax2.set_title("Sample derivation for 'aa' with probability 1/30", fontsize=10)

    def n2(ax, x, y, lbl, col="#4da6ff"):
        c = Circle((x,y),0.10, color=col, zorder=3)
        ax.add_patch(c)
        ax.text(x,y,lbl,ha="center",va="center",fontsize=9,color="white",zorder=4)
    def arr2(ax,x1,y1,x2,y2,lbl=""):
        ax.annotate("", xy=(x2,y2+0.10), xytext=(x1,y1-0.10),
                    arrowprops=dict(arrowstyle="->",color="#555555",lw=1.0))
        mx,my=(x1+x2)/2,(y1+y2)/2
        if lbl:
            ax.text(mx+0.07,my,lbl,fontsize=8,color="#aa3333")

    n2(ax2, 0.5, 1.05, "A")
    n2(ax2, 0.25, 0.65, "B", col="#ff9944")
    n2(ax2, 0.75, 0.65, "A")
    arr2(ax2, 0.5,1.05, 0.25,0.65, "1/2")
    arr2(ax2, 0.5,1.05, 0.75,0.65, "")
    n2(ax2, 0.75, 0.25, "ε", col="#66bb66")
    arr2(ax2, 0.25,0.65, 0.5, 0.25, "1/5")
    arr2(ax2, 0.75,0.65, 0.75,0.25, "1/3")

    ax2.text(0.5, -0.15,
             r"$P = \frac{1}{2}\cdot\frac{1}{5}\cdot\frac{1}{3}=\frac{1}{30}$",
             ha="center", fontsize=11, transform=ax2.transAxes)

    plt.tight_layout()
    savefig("fig_probabilistic_grammar.pdf")


# ==================================================================
# Figure 8: PPT learning — probability mutation illustration
# ==================================================================
def fig_ppt_mutation():
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2))
    np.random.seed(42)
    n_rules = 6
    p0 = np.array([0.05, 0.08, 0.20, 0.10, 0.15, 0.05])
    p0 = p0 / p0.sum()

    # Increase: p[i] += beta*(1 - p[i])  for selected entries
    beta = 0.5
    selected = np.array([True, False, True, True, False, True])
    p1 = p0.copy()
    p1[selected] += beta * (1 - p1[selected])   # not yet normalised

    # Normalize
    p2 = p1 / p1.sum()

    titles = ["Before", "After increase\n($\\beta=0.5$)", "After normalise"]
    data   = [p0, p1, p2]
    colors = ["#4da6ff", "#ff9944", "#66bb66"]

    for ax, d, title, col in zip(axes, data, titles, colors):
        ax.bar(range(n_rules), d, color=col, edgecolor="black", width=0.6)
        ax.set_ylim(0, 1.0)
        ax.set_xticks(range(n_rules))
        ax.set_xticklabels([f"$p_{i+1}$" for i in range(n_rules)], fontsize=9)
        ax.set_ylabel("Probability", fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("PPT Mutation: Probability Vector Update ($\\beta = 0.5$)", fontsize=12)
    plt.tight_layout()
    savefig("fig_ppt_mutation.pdf")


# ==================================================================
# Figure 9: Genetic programming PI approximation tree
# ==================================================================
def fig_pi_approximation_tree():
    """
    Expression tree that evaluates to ~pi = 3.141586
    from book Example 23.5: (3+9/9) * (7*7 + 9*5/3) / 8  ... approximately
    We draw the tree shown in the book.
    """
    fig, ax = plt.subplots(figsize=(5, 6))
    ax.set_xlim(-1, 1); ax.set_ylim(-0.5, 3.8); ax.axis("off")
    ax.set_title("GP tree approximating $\\pi \\approx 3.141586$", fontsize=12)

    BLUE = "#4da6ff"

    nodes = {
        # (label, x, y)
        "r":  ("/",  0.0,  3.4),
        "n1": ("+",  -0.5, 2.7),
        "n2": ("4",   0.5, 2.7),
        "n3": ("+",  -0.8, 2.0),
        "n4": ("×",   -0.2, 2.0),
        "n5": ("3",  -0.95,1.3),
        "n6": ("+",  -0.6, 1.3),
        "n7": ("9",  -0.1, 1.3),
        "n8": ("×",   0.1, 1.3),
        "n9": ("7",  -0.75,0.6),
        "n10":("×",  -0.45,0.6),
        "n11":("7",   0.0, 0.6),
        "n12":("×",   0.2, 0.6),
        "n13":("5",  -0.6,-0.1),
        "n14":("3",  -0.3,-0.1),
        "n15":("9",  0.1,-0.1),
        "n16":("3",  0.35,-0.1),
    }
    edges = [
        ("r","n1"),("r","n2"),
        ("n1","n3"),("n1","n4"),
        ("n3","n5"),("n3","n6"),
        ("n4","n7"),("n4","n8"),
        ("n6","n9"),("n6","n10"),
        ("n8","n11"),("n8","n12"),
        ("n10","n13"),("n10","n14"),
        ("n12","n15"),("n12","n16"),
    ]
    for a,b in edges:
        x1,y1 = nodes[a][1], nodes[a][2]
        x2,y2 = nodes[b][1], nodes[b][2]
        ax.plot([x1,x2],[y1,y2],"k-",lw=1.0,zorder=1)
    for key,(lbl,x,y) in nodes.items():
        c = Circle((x,y),0.14,color=BLUE,zorder=2)
        ax.add_patch(c)
        ax.text(x,y,lbl,ha="center",va="center",fontsize=9,
                color="white",fontweight="bold",zorder=3)

    ax.text(0.5,-0.08,"Evaluates to $\\approx 3.1416$",
            ha="center",transform=ax.transAxes,fontsize=10,color="#333333")
    plt.tight_layout()
    savefig("fig_pi_tree.pdf")


# ==================================================================
# Figure 10: PPT growth diagram (conceptual)
# ==================================================================
def fig_ppt_growth():
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))

    BLUE = "#4da6ff"
    GRAY = "#cccccc"

    def ppt_node(ax, x, y, lbl, col=BLUE):
        c = Circle((x,y),0.12,color=col,zorder=3)
        ax.add_patch(c)
        ax.text(x,y,lbl,ha="center",va="center",fontsize=8,
                color="white",fontweight="bold",zorder=4)

    def edge(ax,x1,y1,x2,y2):
        ax.annotate("",xy=(x2,y2+0.12),xytext=(x1,y1-0.12),
                    arrowprops=dict(arrowstyle="->",color="#555",lw=1.0))

    # After 1st expression
    ax=axes[0]
    ax.set_xlim(-0.2,1.2);ax.set_ylim(-0.5,1.3);ax.axis("off")
    ax.set_title("After sampling 1st expression\n$\\times(\\ln x,\\, 2)$",fontsize=10)
    ppt_node(ax,0.5,1.1,"p1")
    ppt_node(ax,0.2,0.6,"p11")
    ppt_node(ax,0.8,0.6,"p12",col=GRAY)
    ppt_node(ax,0.2,0.1,"p111")
    edge(ax,0.5,1.1,0.2,0.6)
    edge(ax,0.5,1.1,0.8,0.6)
    edge(ax,0.2,0.6,0.2,0.1)
    ax.text(0.05,0.6,"×",fontsize=12,color="#2255aa")
    ax.text(0.05,0.1,"ln",fontsize=10,color="#2255aa")

    # After 2nd expression
    ax=axes[1]
    ax.set_xlim(-0.2,1.2);ax.set_ylim(-0.5,1.3);ax.axis("off")
    ax.set_title("After sampling 2nd expression\n$-(3,\\times(x-2,7))$",fontsize=10)
    ppt_node(ax,0.5,1.1,"p1")
    ppt_node(ax,0.2,0.6,"p11")
    ppt_node(ax,0.8,0.6,"p12")
    ppt_node(ax,0.2,0.1,"p111")
    ppt_node(ax,0.65,0.1,"p121")
    ppt_node(ax,0.95,0.1,"p122")
    for (a,b) in [((0.5,1.1),(0.2,0.6)),((0.5,1.1),(0.8,0.6)),
                  ((0.2,0.6),(0.2,0.1)),
                  ((0.8,0.6),(0.65,0.1)),((0.8,0.6),(0.95,0.1))]:
        edge(ax,a[0],a[1],b[0],b[1])
    ax.text(0.05,1.1,"−",fontsize=14,color="#2255aa")
    ax.text(0.05,0.6,"+",fontsize=12,color="#2255aa")
    ax.text(0.55,0.6,"×",fontsize=12,color="#2255aa")

    fig.suptitle("Probabilistic Prototype Tree Expands During Sampling",fontsize=11)
    plt.tight_layout()
    savefig("fig_ppt_growth.pdf")


# ==================================================================
# Crop key figures from the book PDF using pymupdf
# ==================================================================
def crop_from_pdf():
    try:
        import fitz  # pymupdf
    except ImportError:
        print("  pymupdf not available – skipping PDF crops")
        return

    pdf_path = ("/Users/vinhdq1/work/Lectures-on-Monte-Carlo-Theory/"
                "Algorithms_for_optimization_2e_2026/optimization_book.pdf")
    if not os.path.exists(pdf_path):
        print(f"  PDF not found at {pdf_path} – skipping crops")
        return

    doc = fitz.open(pdf_path)

    # Page numbers are 1-indexed in the book but 0-indexed in pymupdf.
    # Book pages 489-514 correspond to Chapter 23.
    # We'll crop specific figures by their approximate bounding boxes.
    # Format: (page_0idx, x0, y0, x1, y1, output_name)
    crops = [
        # Fig 23.3 – tree crossover diagram (p514 = idx 513)
        (513, 50,  350, 650, 600, "crop_tree_crossover_book.pdf"),
        # Fig 23.5 – tree mutation (p514 = idx 513)
        (513, 50,  610, 650, 850, "crop_tree_mutation_book.pdf"),
        # Fig 23.6 – tree permutation (p515 = idx 514)
        (514, 50,  160, 650, 380, "crop_tree_permutation_book.pdf"),
    ]
    for (pidx, x0, y0, x1, y1, fname) in crops:
        try:
            page = doc[pidx]
            clip = fitz.Rect(x0, y0, x1, y1)
            pix  = page.get_pixmap(clip=clip, dpi=180)
            png_name = fname.replace(".pdf", ".png")
            out  = os.path.join(FIG_DIR, png_name)
            pix.save(out)
            print(f"  cropped {out}")
        except Exception as e:
            print(f"  crop failed for {fname}: {e}")

    doc.close()


# ==================================================================
# Main
# ==================================================================
if __name__ == "__main__":
    print("Generating figures for Chapter 23 …")
    fig_expression_tree()
    fig_grammar_bnf()
    fig_tree_crossover()
    fig_tree_mutation()
    fig_tree_permutation()
    fig_grammatical_evolution()
    fig_probabilistic_grammar()
    fig_ppt_mutation()
    fig_pi_approximation_tree()
    fig_ppt_growth()
    crop_from_pdf()
    print("Done.")
