"""
gen_figures.py
Generate all figures for Chapter 14: "Feature Reinforcement Learning" slides.

Run with:
    conda run -n py313 python3 gen_figures.py

Produces (in ./figures/):
    phi_map_schematic.pdf        -- histories -> states -> MDP, the FRL "big picture"
    example_4state_diagram.pdf   -- Example 14.2.13 aggregation-to-non-MDP diagram
    cost_function_tradeoff.pdf   -- MDL-style U-shaped cost curve (Section 14.3)
    state_bound_comparison.pdf   -- exponential vs. log|A| bound on |S| (Thm 14.2.17 / Eq 14.2.18)
    phi_improve_split_merge.pdf  -- split / merge operations of Algorithm 14.1 (Phi-Improve)
    phimdp_loop_diagram.pdf      -- Algorithm 14.2 (Phi-MDP Agent) interaction loop
    dbn_factorization_schematic.pdf -- Feature Dynamic Bayesian Network factorization (Sec 14.3.3)
    ctm_tree_schematic.pdf       -- binary context tree with a maximizing state set S highlighted
    ctmrl_pipeline_diagram.pdf   -- Algorithm 14.3 (CTMRL) pipeline
    ctw_vs_ctmrl_compute.pdf     -- illustrative compute-time comparison (Section 14.4 discussion)
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
from matplotlib.path import Path

HERE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.join(HERE, "figures")
os.makedirs(FIGDIR, exist_ok=True)

plt.rcParams.update({
    "font.size": 13,
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "legend.fontsize": 11,
    "figure.dpi": 150,
})

COL = {
    "blue": "#1f5fa8",
    "red": "#c0392b",
    "green": "#1e8449",
    "orange": "#d68910",
    "purple": "#6c3483",
    "gray": "#5d6d7e",
    "lgray": "#eaecee",
    "lblue": "#d6e4f0",
    "lgreen": "#d9ecd6",
    "lorange": "#faeacb",
}


def savefig(fig, name):
    path = os.path.join(FIGDIR, name)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


# ─────────────────────────────────────────────────────────────────────────
# Figure 1: phi_map_schematic -- the FRL "big picture"
# ─────────────────────────────────────────────────────────────────────────
def fig_phi_map_schematic():
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis("off")

    # Left cloud: history space H (many long, distinct histories)
    box_h = FancyBboxPatch((0.3, 0.6), 3.0, 3.8, boxstyle="round,pad=0.08,rounding_size=0.12",
                            linewidth=1.6, edgecolor=COL["gray"], facecolor=COL["lgray"])
    ax.add_patch(box_h)
    ax.text(1.8, 4.65, r"History space $\mathcal{H}$", ha="center", fontsize=13, fontweight="bold")
    rng = np.random.default_rng(3)
    hist_ys = [3.9, 3.35, 2.8, 2.25, 1.7, 1.15]
    labels = [r"$a_1o_1r_1a_2o_2r_2\cdots$", r"$a_1o_1'r_1\cdots$", r"$a_1'o_1r_1''\cdots$",
              r"$a_1o_1r_1a_2''\cdots$", r"$a_1'''o_1r_1\cdots$", r"$a_1o_1r_1a_2'\cdots$"]
    for y, lab in zip(hist_ys, labels):
        ax.text(1.75, y, lab, fontsize=8.6, ha="center",
                bbox=dict(boxstyle="round,pad=0.18", fc="white", ec=COL["blue"], lw=0.8))

    # Middle arrow: phi
    arr = FancyArrowPatch((3.5, 2.5), (5.6, 2.5), arrowstyle="-|>", mutation_scale=22,
                           linewidth=2.4, color=COL["red"])
    ax.add_patch(arr)
    ax.text(4.55, 2.95, r"$\phi:\mathcal{H}\to\mathcal{S}$", ha="center", fontsize=13, color=COL["red"], fontweight="bold")
    ax.text(4.55, 2.05, "(feature map)", ha="center", fontsize=9.5, style="italic")

    # Right box: state space S (few states), many-to-one arrows
    box_s = FancyBboxPatch((6.0, 1.3) , 3.4, 3.1, boxstyle="round,pad=0.08,rounding_size=0.12",
                            linewidth=1.6, edgecolor=COL["gray"], facecolor=COL["lblue"])
    ax.add_patch(box_s)
    ax.text(7.7, 4.65, r"State space $\mathcal{S}$", ha="center", fontsize=13, fontweight="bold")
    state_pos = {"s1": (7.0, 3.6), "s2": (8.4, 3.6), "s3": (7.7, 2.1)}
    for name, (x, y) in state_pos.items():
        ax.add_patch(Circle((x, y), 0.42, facecolor="white", edgecolor=COL["blue"], linewidth=1.8, zorder=3))
        ax.text(x, y, "$s$", ha="center", va="center", fontsize=12, zorder=4)

    # many-to-one arrows from histories to the 3 states
    targets = ["s1", "s1", "s2", "s2", "s3", "s3"]
    for y, tgt in zip(hist_ys, targets):
        tx, ty = state_pos[tgt]
        ax.annotate("", xy=(tx - 0.42, ty), xytext=(5.55, y),
                    arrowprops=dict(arrowstyle="->", color=COL["orange"], lw=1.1, alpha=0.85,
                                     connectionstyle="arc3,rad=0.15"))

    ax.text(7.7, 0.75, r"MDP $\overline{\mu}(s'r'|sa)$ solved via Bellman equations",
            ha="center", fontsize=10.5, color=COL["gray"])
    fig.tight_layout()
    savefig(fig, "phi_map_schematic.pdf")


# ─────────────────────────────────────────────────────────────────────────
# Figure 2: example_4state_diagram -- Example 14.2.13
# ─────────────────────────────────────────────────────────────────────────
def fig_example_4state_diagram():
    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    ax.set_xlim(-0.5, 8.5)
    ax.set_ylim(-1.2, 8.7)
    ax.axis("off")
    ax.set_aspect("equal")

    pos = {"00": (1.5, 7.0), "01": (7.0, 7.0), "10": (1.5, 1.0), "11": (7.0, 1.0)}
    rlabel = {
        "00": r"$r'=\frac{\gamma/2}{1+\gamma}$",
        "01": r"$r'=\frac{1+\gamma/2}{1+\gamma}$",
        "10": r"$r'=0$",
        "11": r"$r'=1$",
    }
    for name, (x, y) in pos.items():
        ax.add_patch(Circle((x, y), 0.62, facecolor="white", edgecolor=COL["blue"], linewidth=2.0, zorder=4))
        ax.text(x, y, name, ha="center", va="center", fontsize=15, fontweight="bold", zorder=5)
        dx, dy = (0, 0.95) if y > 4 else (0, -0.95)
        ax.text(x, y + dy, rlabel[name], ha="center", va="center", fontsize=10.5, color=COL["gray"], zorder=5)

    def arrow(a, b, label, rad, color=COL["red"], lpos=0.5, dside=0.18):
        xa, ya = pos[a]; xb, yb = pos[b]
        patch = FancyArrowPatch((xa, ya), (xb, yb), arrowstyle="-|>", mutation_scale=18,
                                 connectionstyle=f"arc3,rad={rad}", linewidth=1.8, color=color,
                                 shrinkA=26, shrinkB=26, zorder=2)
        ax.add_patch(patch)
        xm = xa + lpos * (xb - xa); ym = ya + lpos * (yb - ya)
        # perpendicular offset for label
        vx, vy = (yb - ya), -(xb - xa)
        norm = (vx ** 2 + vy ** 2) ** 0.5
        vx, vy = vx / norm, vy / norm
        ax.text(xm + dside * vx, ym + dside * vy, label, fontsize=11, color=color, zorder=6,
                ha="center", va="center", bbox=dict(boxstyle="round,pad=0.08", fc="white", ec="none", alpha=0.85))

    arrow("00", "01", r"$\frac{1}{2}$", 0.28, color=COL["blue"], lpos=0.35)
    arrow("01", "00", r"$\frac{1}{2}$", 0.28, color=COL["blue"], lpos=0.65)
    arrow("00", "10", r"$\frac{1}{2}$", -0.15, color=COL["green"])
    arrow("01", "11", r"$\frac{1}{2}$", 0.15, color=COL["green"])
    arrow("10", "01", r"$1$", 0.0, color=COL["orange"], lpos=0.42, dside=0.35)
    arrow("11", "00", r"$1$", 0.0, color=COL["purple"], lpos=0.42, dside=-0.35)

    # dashed grouping boxes for s=0 = {00,10}, s=1 = {01,11}
    box0 = FancyBboxPatch((0.3, -0.9), 2.4, 9.2, boxstyle="round,pad=0.1", linewidth=1.4,
                           edgecolor=COL["gray"], facecolor="none", linestyle="--", zorder=1)
    box1 = FancyBboxPatch((5.8, -0.9), 2.4, 9.2, boxstyle="round,pad=0.1", linewidth=1.4,
                           edgecolor=COL["gray"], facecolor="none", linestyle="--", zorder=1)
    ax.add_patch(box0); ax.add_patch(box1)
    ax.text(1.5, -1.5, r"$s=\phi(o)=0$", ha="center", fontsize=12, fontweight="bold", color=COL["gray"])
    ax.text(7.0, -1.5, r"$s=\phi(o)=1$", ha="center", fontsize=12, fontweight="bold", color=COL["gray"])
    ax.set_ylim(-1.8, 8.7)
    ax.set_title("Example 14.2.13: reducing 4 observations to 2 states", fontsize=12.5)
    fig.tight_layout()
    savefig(fig, "example_4state_diagram.pdf")


# ─────────────────────────────────────────────────────────────────────────
# Figure 3: cost_function_tradeoff -- MDL balance
# ─────────────────────────────────────────────────────────────────────────
def fig_cost_function_tradeoff():
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    sizes = np.linspace(1, 10, 400)
    cl_phi = 0.55 * sizes + 0.2          # cost of encoding phi itself: grows with |S|
    cl_fit = 9.0 / sizes + 0.3           # cost of encoding data given phi: shrinks with |S|
    total = cl_phi + cl_fit
    imin = np.argmin(total)

    ax.plot(sizes, cl_phi, color=COL["orange"], lw=2.4, label=r"$CL(\phi)$ (model complexity)")
    ax.plot(sizes, cl_fit, color=COL["blue"], lw=2.4, label=r"$CL(\overline{\mu}\,|\,\overline{h}_{1:t})$ (data fit)")
    ax.plot(sizes, total, color=COL["red"], lw=3.0, label=r"$Cost(\phi|h_{1:t}) = $ sum")
    ax.axvline(sizes[imin], color=COL["gray"], ls=":", lw=1.6)
    ax.scatter([sizes[imin]], [total[imin]], color=COL["red"], zorder=5, s=55)
    ax.annotate("MDL-optimal\ntrade-off", xy=(sizes[imin], total[imin]),
                xytext=(sizes[imin] + 1.4, total[imin] + 2.2),
                arrowprops=dict(arrowstyle="->", color=COL["gray"]), fontsize=10.5)

    ax.text(1.1, cl_fit[3] + 1.0, "many states,\nbandit-like\n(overfits)", fontsize=9, color=COL["gray"])
    ax.text(8.0, cl_phi[-40] + 1.0, "1 state,\ntrivial MDP\n(underfits)", fontsize=9, color=COL["gray"])

    ax.set_xlabel(r"range / size of feature map $\phi$  (i.e. $|\mathcal{S}|$)")
    ax.set_ylabel("code length (bits, illustrative units)")
    ax.set_title("Balancing the cost function (Section 14.3.2)")
    ax.legend(loc="upper center", frameon=False, fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    savefig(fig, "cost_function_tradeoff.pdf")


# ─────────────────────────────────────────────────────────────────────────
# Figure 4: state_bound_comparison -- Thm 14.2.17 vs Eq (14.2.18)
# ─────────────────────────────────────────────────────────────────────────
def fig_state_bound_comparison():
    fig, ax = plt.subplots(figsize=(7.4, 4.7))
    eps = 0.1
    gamma = 0.9
    A = np.arange(2, 17)
    exp_bound = (3.0 / (eps * (1 - gamma) ** 3)) ** A
    log_bound = 17.0 * (np.log(A)) ** 3 / (eps * (1 - gamma) ** 3)

    ax.semilogy(A, exp_bound, "o-", color=COL["red"], lw=2.2, ms=5,
                label=r"Thm 14.2.17: $\left(\frac{3}{\varepsilon'(1-\gamma)^3}\right)^{|\mathcal{A}|}$")
    ax.semilogy(A, log_bound, "s-", color=COL["green"], lw=2.2, ms=5,
                label=r"Eq (14.2.18), binarized: $\frac{17(\log|\mathcal{A}|)^3}{\varepsilon'(1-\gamma)^3}$")
    ax.set_xlabel(r"number of actions $|\mathcal{A}|$")
    ax.set_ylabel(r"bound on required $|\mathcal{S}|$  (log scale)")
    ax.set_title(r"Exponential vs.\ logarithmic dependence on $|\mathcal{A}|$" + "\n" + r"($\varepsilon'=0.1$, $\gamma=0.9$, illustrative)")
    ax.legend(loc="upper left", frameon=False, fontsize=9.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.25, which="both")
    fig.tight_layout()
    savefig(fig, "state_bound_comparison.pdf")


# ─────────────────────────────────────────────────────────────────────────
# Figure 5: phi_improve_split_merge -- Algorithm 14.1 operations
# ─────────────────────────────────────────────────────────────────────────
def fig_phi_improve_split_merge():
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))

    # --- SPLIT ---
    ax = axes[0]
    ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis("off")
    ax.set_title(r"Split: $\{s\}\ \to\ \{os:o\in\mathcal{O}\}$", fontsize=12.5)
    ax.add_patch(Circle((2.2, 3), 0.75, facecolor=COL["lblue"], edgecolor=COL["blue"], lw=2))
    ax.text(2.2, 3, "$s$", ha="center", va="center", fontsize=14)
    arr = FancyArrowPatch((3.1, 3), (5.6, 3), arrowstyle="-|>", mutation_scale=22, lw=2.2, color=COL["red"])
    ax.add_patch(arr)
    for i, (dy, lab) in enumerate(zip([1.6, 0.0, -1.6], [r"$o_1s$", r"$o_2s$", r"$o_3s$"])):
        cx, cy = 7.6, 3 + dy
        ax.add_patch(Circle((cx, cy), 0.62, facecolor=COL["lgreen"], edgecolor=COL["green"], lw=2))
        ax.text(cx, cy, lab, ha="center", va="center", fontsize=11)
    ax.text(5.85, 3.75, "finer states,\nmore expressive", fontsize=9, color=COL["gray"], ha="center")

    # --- MERGE ---
    ax = axes[1]
    ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis("off")
    ax.set_title(r"Merge: $\{os_{>1}:o\in\mathcal{O}\}\ \to\ \{s_{>1}\}$", fontsize=12.5)
    for i, (dy, lab) in enumerate(zip([1.6, 0.0, -1.6], [r"$o_1s_{>1}$", r"$o_2s_{>1}$", r"$o_3s_{>1}$"])):
        cx, cy = 2.2, 3 + dy
        ax.add_patch(Circle((cx, cy), 0.62, facecolor=COL["lorange"], edgecolor=COL["orange"], lw=2))
        ax.text(cx, cy, lab, ha="center", va="center", fontsize=10.5)
    arr = FancyArrowPatch((3.1, 3), (5.6, 3), arrowstyle="-|>", mutation_scale=22, lw=2.2, color=COL["red"])
    ax.add_patch(arr)
    ax.add_patch(Circle((7.6, 3), 0.75, facecolor=COL["lblue"], edgecolor=COL["blue"], lw=2))
    ax.text(7.6, 3, r"$s_{>1}$", ha="center", va="center", fontsize=13)
    ax.text(4.35, 3.75, "coarser states,\nmore learnable", fontsize=9, color=COL["gray"], ha="center")

    fig.tight_layout()
    savefig(fig, "phi_improve_split_merge.pdf")


# ─────────────────────────────────────────────────────────────────────────
# Figure 6: phimdp_loop_diagram -- Algorithm 14.2
# ─────────────────────────────────────────────────────────────────────────
def fig_phimdp_loop_diagram():
    fig, ax = plt.subplots(figsize=(9, 6.4))
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off"); ax.set_aspect("equal")

    boxes = {
        "env":    (5.0, 8.7, "Environment $\\mu$\nemits percept $e_t$"),
        "improve":(1.6, 6.6, "$\\Phi$Improve\n(Algorithm 14.1)\nupdate $\\phi\\to\\phi'$"),
        "hist":   (1.6, 3.6, "Append: $h_{1:t}{:=}h_{<t}a_te_t$\n$s_t{:=}\\phi(h_{1:t})$"),
        "mdp":    (5.0, 1.4, "Build $\\overline{\\mu}$ via (14.3.2)\n(frequency estimate)"),
        "act":    (8.4, 3.6, "$a_t{:=}\\arg\\max_a Q^*_{\\overline{\\mu}}(s_t,a)$"),
    }
    for key, (x, y, txt) in boxes.items():
        w, h = 3.0, 1.7
        fc = COL["lblue"] if key in ("env",) else (COL["lorange"] if key == "improve" else COL["lgray"])
        box = FancyBboxPatch((x - w/2, y - h/2), w, h, boxstyle="round,pad=0.08,rounding_size=0.12",
                              linewidth=1.7, edgecolor=COL["gray"], facecolor=fc, zorder=3)
        ax.add_patch(box)
        ax.text(x, y, txt, ha="center", va="center", fontsize=9.6, zorder=4)

    def arrow(k1, k2, rad=0.0):
        x1, y1, _ = boxes[k1]; x2, y2, _ = boxes[k2]
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=20,
                                      connectionstyle=f"arc3,rad={rad}", lw=2.0, color=COL["red"],
                                      shrinkA=40, shrinkB=40, zorder=2))

    arrow("env", "improve", rad=0.15)
    arrow("improve", "hist", rad=0.0)
    arrow("hist", "mdp", rad=-0.15)
    arrow("mdp", "act", rad=-0.15)
    arrow("act", "env", rad=0.15)

    ax.text(5.0, 0.15, r"loop for $t=1,2,3,\ldots$  (output action $a_t$, receive next percept $e_{t+1}$)",
            ha="center", fontsize=10, color=COL["gray"])
    ax.set_title("Algorithm 14.2: the $\\Phi$MDP agent loop", fontsize=13)
    fig.tight_layout()
    savefig(fig, "phimdp_loop_diagram.pdf")


# ─────────────────────────────────────────────────────────────────────────
# Figure 7: dbn_factorization_schematic -- Section 14.3.3
# ─────────────────────────────────────────────────────────────────────────
def fig_dbn_factorization_schematic():
    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis("off")

    m = 4
    xs = np.linspace(1.2, 8.8, m)
    y_prev, y_next = 4.4, 1.2
    for i, x in enumerate(xs, start=1):
        ax.add_patch(Circle((x, y_prev), 0.5, facecolor=COL["lblue"], edgecolor=COL["blue"], lw=1.8, zorder=3))
        ax.text(x, y_prev, f"$s^{{{i}}}$", ha="center", va="center", fontsize=11, zorder=4)
        ax.add_patch(Circle((x, y_next), 0.5, facecolor=COL["lgreen"], edgecolor=COL["green"], lw=1.8, zorder=3))
        ax.text(x, y_next, f"$s'^{{{i}}}$", ha="center", va="center", fontsize=11, zorder=4)

    ax.text(0.15, y_prev, "time $t{-}1$", fontsize=10, color=COL["gray"], va="center")
    ax.text(0.15, y_next, "time $t$", fontsize=10, color=COL["gray"], va="center")

    # sparse parent sets u^i subset of {s^1..s^m}: define parents for each s'^i
    parents = {1: [1], 2: [1, 2], 3: [2, 3], 4: [3, 4]}
    for i, plist in parents.items():
        xt, yt = xs[i - 1], y_next
        for p in plist:
            xp, yp = xs[p - 1], y_prev
            ax.add_patch(FancyArrowPatch((xp, yp - 0.5), (xt, yt + 0.5), arrowstyle="-|>",
                                          mutation_scale=14, lw=1.4, color=COL["orange"],
                                          connectionstyle="arc3,rad=0.05", zorder=2))

    # action node feeding all next-state nodes
    ax.add_patch(FancyBboxPatch((4.55, 5.35), 0.9, 0.55, boxstyle="round,pad=0.05", lw=1.6,
                                 edgecolor=COL["purple"], facecolor=COL["lorange"], zorder=3))
    ax.text(5.0, 5.62, "$a$", ha="center", va="center", fontsize=11, zorder=4)
    for x in xs:
        ax.add_patch(FancyArrowPatch((5.0, 5.35), (x, y_prev + 0.5), arrowstyle="-|>", mutation_scale=10,
                                      lw=0.9, color=COL["purple"], alpha=0.55,
                                      connectionstyle="arc3,rad=0.05", zorder=1))

    ax.text(5.0, 0.3, r"$\overline{\mu}(s'|sa) = \prod_{i=1}^{m}\overline{\mu}^a(s'^i|u^i)$" +
            "   -- each next feature depends only on its own parent subset $u^i$",
            ha="center", fontsize=10.5, color=COL["gray"])
    ax.set_title(r"$\Phi$DBN: factorized transition structure ($m=4$ binary features)", fontsize=12.5)
    fig.tight_layout()
    savefig(fig, "dbn_factorization_schematic.pdf")


# ─────────────────────────────────────────────────────────────────────────
# Figure 8: ctm_tree_schematic -- binary context tree, maximizing subset S
# ─────────────────────────────────────────────────────────────────────────
def fig_ctm_tree_schematic():
    fig, ax = plt.subplots(figsize=(8.6, 5.6))
    ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis("off")

    # node positions per depth level (depth 0..2)
    levels = {
        0: {"": 5.0},
        1: {"0": 2.7, "1": 7.3},
        2: {"00": 1.4, "01": 4.0, "10": 6.0, "11": 8.6},
    }
    ys = {0: 5.0, 1: 3.2, 2: 1.4}
    highlighted = {"0", "10", "11"}   # illustrative maximizing state set S

    pos = {}
    for d, nodes in levels.items():
        for node, x in nodes.items():
            pos[node] = (x, ys[d])

    edges = [("", "0"), ("", "1"), ("0", "00"), ("0", "01"), ("1", "10"), ("1", "11")]
    for a, b in edges:
        xa, ya = pos[a]; xb, yb = pos[b]
        ax.plot([xa, xb], [ya, yb], color=COL["gray"], lw=1.6, zorder=1)

    for node, (x, y) in pos.items():
        is_hi = node in highlighted
        fc = COL["lgreen"] if is_hi else "white"
        ec = COL["green"] if is_hi else COL["blue"]
        label = r"$\epsilon$" if node == "" else node
        ax.add_patch(Circle((x, y), 0.42, facecolor=fc, edgecolor=ec, lw=2.2, zorder=3))
        ax.text(x, y, label, ha="center", va="center", fontsize=11, zorder=4)

    ax.text(9.7, ys[0], "depth 0", fontsize=9.5, color=COL["gray"], ha="left")
    ax.text(9.7, ys[1], "depth 1", fontsize=9.5, color=COL["gray"], ha="left")
    ax.text(9.7, ys[2], "depth 2 = $D$", fontsize=9.5, color=COL["gray"], ha="left")

    ax.legend(handles=[mpatches.Patch(facecolor=COL["lgreen"], edgecolor=COL["green"],
                                       label=r"maximizing state set $\mathcal{S}^D_{m,\epsilon}$")],
              loc="lower center", frameon=False, fontsize=10, bbox_to_anchor=(0.5, -0.05))
    ax.set_title("Context Tree Maximization: pruning to the maximizing state set", fontsize=12.5)
    fig.tight_layout()
    savefig(fig, "ctm_tree_schematic.pdf")


# ─────────────────────────────────────────────────────────────────────────
# Figure 9: ctmrl_pipeline_diagram -- Algorithm 14.3
# ─────────────────────────────────────────────────────────────────────────
def fig_ctmrl_pipeline_diagram():
    fig, ax = plt.subplots(figsize=(11.5, 4.2))
    ax.set_xlim(0, 12.5); ax.set_ylim(0, 4.5); ax.axis("off")

    steps = [
        "Create $l_e$\nempty CTMs",
        "Update CTMs\non history $h'$",
        "Join into\nContext Tree $\\mathcal{T}$",
        "Frequency estimate\nof MDP $\\overline{\\mu}$",
        "Action-Value\nIteration $\\to \\hat{Q}$",
        "Q-learning\n$\\to$ new $h$",
    ]
    n = len(steps)
    xs = np.linspace(1.1, 11.4, n)
    y = 2.6
    for x, txt in zip(xs, steps):
        box = FancyBboxPatch((x - 0.85, y - 0.75), 1.7, 1.5, boxstyle="round,pad=0.06,rounding_size=0.1",
                              linewidth=1.6, edgecolor=COL["gray"], facecolor=COL["lblue"], zorder=3)
        ax.add_patch(box)
        ax.text(x, y, txt, ha="center", va="center", fontsize=8.8, zorder=4)
    for i in range(n - 1):
        ax.add_patch(FancyArrowPatch((xs[i] + 0.85, y), (xs[i+1] - 0.85, y), arrowstyle="-|>",
                                      mutation_scale=16, lw=1.8, color=COL["red"], zorder=2))

    # feedback loop arrow: "while i<m-1" back to step 2
    ax.add_patch(FancyArrowPatch((xs[-1], y + 0.8), (xs[1], y + 0.8), arrowstyle="-|>",
                                  mutation_scale=16, lw=1.6, color=COL["orange"],
                                  connectionstyle="arc3,rad=-0.25", zorder=2))
    ax.text((xs[1] + xs[-1]) / 2, y + 1.55, r"repeat while $i<m-1$ (grow history, refine $\mathcal{T}$)",
            fontsize=9.5, color=COL["orange"], ha="center")

    ax.add_patch(FancyArrowPatch((xs[-1] + 0.85, y), (12.1, y), arrowstyle="-|>", mutation_scale=16,
                                  lw=1.8, color=COL["green"], zorder=2))
    ax.text(12.2, y, r"$\pi^*$", fontsize=13, color=COL["green"], va="center")

    ax.set_title("Algorithm 14.3: the CTMRL pipeline", fontsize=13)
    fig.tight_layout()
    savefig(fig, "ctmrl_pipeline_diagram.pdf")


# ─────────────────────────────────────────────────────────────────────────
# Figure 10: ctw_vs_ctmrl_compute -- illustrative comparison
# ─────────────────────────────────────────────────────────────────────────
def fig_ctw_vs_ctmrl_compute():
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    names = ["CTMRL", "MC-AIXI-CTW"]
    vals = [1.0, 10.0]
    colors = [COL["green"], COL["red"]]
    bars = ax.bar(names, vals, color=colors, width=0.55, edgecolor="black", linewidth=0.8)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width()/2, v + 0.25, f"{v:.0f}$\\times$", ha="center", fontsize=12)
    ax.set_ylabel("relative compute time\n(illustrative, comparable performance)")
    ax.set_title("Section 14.4: CTMRL reaches similar performance\nwith far less computation")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylim(0, 12)
    fig.tight_layout()
    savefig(fig, "ctw_vs_ctmrl_compute.pdf")


if __name__ == "__main__":
    fig_phi_map_schematic()
    fig_example_4state_diagram()
    fig_cost_function_tradeoff()
    fig_state_bound_comparison()
    fig_phi_improve_split_merge()
    fig_phimdp_loop_diagram()
    fig_dbn_factorization_schematic()
    fig_ctm_tree_schematic()
    fig_ctmrl_pipeline_diagram()
    fig_ctw_vs_ctmrl_compute()
    print("All figures generated in", FIGDIR)
