#!/usr/bin/env python3
"""
gen_figures.py -- Generate all figures for Chapter 1 (Introduction) slides.
Book: An Introduction to Universal Artificial Intelligence (2024), Hutter/Quarel/Catt.

Run with: conda run -n py313 python3 gen_figures.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
import numpy as np
import os

FIGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(FIGDIR, exist_ok=True)

plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
})

# ─────────────────────────────────────────────────────────────────────────
# Figure 1: Cybernetic agent-environment interaction loop
# ─────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 4.2))
ax.set_xlim(0, 10)
ax.set_ylim(0, 5)
ax.axis("off")

agent_box = FancyBboxPatch((0.6, 1.6), 3.0, 1.8, boxstyle="round,pad=0.08",
                            linewidth=2, edgecolor="#1f4e79", facecolor="#cfe2f3")
env_box = FancyBboxPatch((6.4, 1.6), 3.0, 1.8, boxstyle="round,pad=0.08",
                          linewidth=2, edgecolor="#7f6000", facecolor="#fff2cc")
ax.add_patch(agent_box)
ax.add_patch(env_box)
ax.text(2.1, 2.5, "Agent\n" r"policy $\pi$", ha="center", va="center", fontsize=14, fontweight="bold")
ax.text(7.9, 2.5, "Environment\n" r"$\mu$", ha="center", va="center", fontsize=14, fontweight="bold")

# Action arrow: agent -> environment (top)
arrow1 = FancyArrowPatch((3.6, 3.0), (6.4, 3.0), arrowstyle="-|>", mutation_scale=22,
                          linewidth=2.2, color="#1f4e79")
ax.add_patch(arrow1)
ax.text(5.0, 3.25, r"action $a_t$", ha="center", fontsize=12, color="#1f4e79")

# Percept arrow: environment -> agent (bottom)
arrow2 = FancyArrowPatch((6.4, 2.1), (3.6, 2.1), arrowstyle="-|>", mutation_scale=22,
                          linewidth=2.2, color="#7f6000")
ax.add_patch(arrow2)
ax.text(5.0, 1.85, r"percept $e_t=o_t r_t$", ha="center", fontsize=12, color="#7f6000")

ax.text(5.0, 4.5, "Cycle $t = 1, 2, \\ldots, m$", ha="center", fontsize=13, style="italic")
ax.text(5.0, 0.6, "The agent acts, the environment reacts with an observation and a reward,\n"
                  "and the cycle repeats until the lifespan $m$ is reached.",
        ha="center", fontsize=10.5)

plt.tight_layout()
plt.savefig(os.path.join(FIGDIR, "cybernetic_loop.pdf"), bbox_inches="tight")
plt.close(fig)

# ─────────────────────────────────────────────────────────────────────────
# Figure 2: AI capability-level timeline (from the ANI/AGI/ASI/AGSI table)
# ─────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 3.3))

rows = [
    ("ANI",       1980, 1997, "idiot savant", "one"),
    ("Proto-AGI", 2023, 2028, "human",         "many"),
    ("AGI",       2028, 2038, "smart",         "most"),
    ("ASI",       2038, 2042, "genius",        "some/many"),
    ("AGSI",      2042, 2048, "humanity",      "nearly all"),
]
colors = ["#9fc5e8", "#a2c4c9", "#b6d7a8", "#f9cb9c", "#e06666"]

for i, (name, start, end, level, domains) in enumerate(rows):
    y = len(rows) - i
    ax.barh(y, end - start, left=start, height=0.6, color=colors[i], edgecolor="black")
    ax.text(end + 1, y, f"{level}  ({domains} domains)", va="center", fontsize=10.5)
    ax.text(start - 1, y, name, va="center", ha="right", fontsize=11, fontweight="bold")

# UAI/AIXI: theory, available now (not a forecast bar), mark at 2000 with a star
ax.scatter([2000], [0.3], marker="*", s=260, color="#674ea7", zorder=5)
ax.text(2001, 0.3, "UAI / AIXI --- maximal, all domains (theory since 2000)",
        va="center", fontsize=10.5, color="#674ea7")

ax.set_xlim(1975, 2075)
ax.set_ylim(-0.3, len(rows) + 0.8)
ax.set_yticks([])
ax.set_xlabel("Year (achieved / projected)", fontsize=10.5)
ax.set_title("Proliferation of AI terminology: capability levels over time", fontsize=11.5)
ax.tick_params(axis="x", labelsize=9.5)
for spine in ["top", "right", "left"]:
    ax.spines[spine].set_visible(False)
ax.grid(axis="x", linestyle=":", alpha=0.5)

plt.tight_layout()
plt.savefig(os.path.join(FIGDIR, "ai_timeline.pdf"), bbox_inches="tight")
plt.close(fig)

# ─────────────────────────────────────────────────────────────────────────
# Figure 3: The six parts of the book (flow diagram)
# ─────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9.5, 5.6))
ax.set_xlim(0, 10)
ax.set_ylim(0, 13)
ax.axis("off")

parts = [
    ("I", "Background", "Probability, computability,\nKolmogorov complexity"),
    ("II", "Algorithmic Prediction", "Solomonoff induction,\nContext Tree Weighting"),
    ("III", "A Family of Universal Agents", "The cybernetic model,\nAIXI"),
    ("IV", "Approximating Universal Agents", "AIXI-MDP, MC-AIXI-CTW,\nAIXI$tl$"),
    ("V", "Alternative Approaches", "Feature Reinforcement\nLearning"),
    ("VI", "Safety and Discussion", "ASI safety,\nphilosophy of AI"),
]
n = len(parts)
box_h = 1.75
gap = 0.35
total_h = n * box_h + (n - 1) * gap
y0 = (13 - total_h) / 2 + total_h

part_colors = ["#d9d2e9", "#cfe2f3", "#d0e0e3", "#d9ead3", "#fce5cd", "#f4cccc"]

for i, (num, title, desc) in enumerate(parts):
    ytop = y0 - i * (box_h + gap)
    ybot = ytop - box_h
    ycen = (ytop + ybot) / 2
    box = FancyBboxPatch((1.0, ybot), 8.0, box_h, boxstyle="round,pad=0.05",
                          linewidth=1.8, edgecolor="#333333", facecolor=part_colors[i])
    ax.add_patch(box)
    ax.text(1.6, ycen, f"Part {num}", fontsize=12, fontweight="bold", va="center")
    ax.text(4.0, ycen + 0.28, title, fontsize=12.5, fontweight="bold", va="center")
    ax.text(4.0, ycen - 0.35, desc, fontsize=9.5, va="center", color="#333333")
    if i < n - 1:
        arrow = FancyArrowPatch((5.0, ybot - 0.02), (5.0, ybot - gap + 0.05),
                                 arrowstyle="-|>", mutation_scale=16, linewidth=1.6, color="#555555")
        ax.add_patch(arrow)

plt.tight_layout()
plt.savefig(os.path.join(FIGDIR, "book_structure.pdf"), bbox_inches="tight")
plt.close(fig)

# ─────────────────────────────────────────────────────────────────────────
# Figure 4: Occam's razor illustration -- competing continuations of
#           3, 1, 4, 1, 5, 9  weighted by simplicity
# ─────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7.2, 2.9))

hyps = [r"$\pi$ (circle constant)", r"$355/113$", "degree-6\npolynomial fit", "other\n(arbitrary)"]
# Illustrative (not book-stated) prior weights that decrease with description length,
# used only to visualize the *shape* of Occam's-razor weighting, normalized to sum to 1.
weights = [0.55, 0.30, 0.10, 0.05]
colors4 = ["#38761d", "#6aa84f", "#e69138", "#cc4125"]

bars = ax.bar(hyps, weights, color=colors4, edgecolor="black")
for b, w in zip(bars, weights):
    ax.text(b.get_x() + b.get_width() / 2, w + 0.015, f"{w:.2f}", ha="center", fontsize=10)

ax.set_ylabel("Illustrative prior weight", fontsize=10)
ax.set_title("Occam's razor: simpler continuations of  3, 1, 4, 1, 5, 9  get more prior weight",
             fontsize=10.5)
ax.tick_params(axis="both", labelsize=9)
ax.set_ylim(0, 0.7)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(FIGDIR, "occam_weights.pdf"), bbox_inches="tight")
plt.close(fig)

print("All figures written to", FIGDIR)
for f in sorted(os.listdir(FIGDIR)):
    print(" -", f)
