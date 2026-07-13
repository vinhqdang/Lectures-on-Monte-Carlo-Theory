"""
Generate figures for Chapter 14 (Conclusions) slides on:
Bayesian Machine Learning in Quantitative Finance (Mongwe, Mbuvha & Marwala, 2025)

Figures produced (all saved as vector PDF):
  1. fig_ch14_book_map.pdf              -- Map of the book's 14 chapters organized
                                            by theme (background / derivative modeling /
                                            financial management / insurance & investments)
  2. fig_ch14_toolkit.pdf                -- Matrix of the Bayesian ML "toolkit" (MCMC
                                            variants, GPs, normalizing flows, BNNs,
                                            nested sampling) vs. which chapters used them
  3. fig_ch14_derivative_modeling.pdf    -- Contributions & future directions,
                                            derivative modeling theme (mirrors Fig. 14.1)
  4. fig_ch14_financial_management.pdf   -- Contributions & future directions,
                                            financial management theme (mirrors Fig. 14.2)
  5. fig_ch14_insurance_investments.pdf  -- Contributions & future directions,
                                            insurance & investments theme (mirrors Fig. 14.3)
"""

import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "figure.dpi": 150,
})

# ---------------------------------------------------------------------------
# Figure 1: Map of the book -- 14 chapters organized by theme
# ---------------------------------------------------------------------------
themes = [
    {
        "name": "Background to\nBayesian Inference",
        "color": "#4C72B0",
        "chapters": ["Ch 2: Variational Inference & MCMC\nfor training Bayesian models"],
    },
    {
        "name": "Derivative\nModeling",
        "color": "#DD8452",
        "chapters": [
            "Ch 3: SABR model calibration\n(MALA, HMC, Sep. Shadow HMC)",
            "Ch 4: Equity volatility surfaces\n(single/multi-output GPs)",
            "Ch 5: SA equity option prices\n(mixture of normalizing flows)",
        ],
    },
    {
        "name": "Financial\nManagement",
        "color": "#55A868",
        "chapters": [
            "Ch 6: Corporate credit ratings\n(sparse & distributed GPs)",
            "Ch 7: Charged-off loan recovery\n(BLR via MALA, HMC, SSHHMC)",
            "Ch 8: Audit outcome model selection\n(NF harmonic-mean evidence)",
            "Ch 9: Unauthorized expenditure\n(BLR-ARD via MH, MALA, HMC, MHMC)",
        ],
    },
    {
        "name": "Insurance &\nInvestments",
        "color": "#C44E52",
        "chapters": [
            "Ch 10: Motor insurance claims\n(BNN, Laplace approximation)",
            "Ch 11: Nelson-Siegel yield curve\n(HMC, SSHHMC, NUTS)",
            "Ch 12: Yield curve model selection\n(static & dynamic nested sampling)",
            "Ch 13: Bayesian investment analyst\n(BLR-ARD via MCMC, JSE)",
        ],
    },
]

max_chapters = max(len(t["chapters"]) for t in themes)
ch_h = 0.72
ch_gap = 0.14
theme_h = 0.9
theme_y = 6.3
theme_top = theme_y + theme_h
stack_top = theme_y - 0.25
stack_bottom = stack_top - max_chapters * (ch_h + ch_gap)
concl_h = 0.7
concl_y = stack_bottom - 0.45 - concl_h

core_h = 0.8
core_gap = 0.35
core_y = theme_top + core_gap
core_top = core_y + core_h

ch1_h = 0.8
ch1_gap = 0.35
ch1_y = core_top + ch1_gap
ch1_top = ch1_y + ch1_h

fig_height = 9.6
fig, ax = plt.subplots(figsize=(13, fig_height))
ax.set_xlim(0, 13)
ax.set_ylim(concl_y - 0.35, ch1_top + 0.3)
ax.axis("off")

ch1_box = FancyBboxPatch((5.0, ch1_y), 3.0, ch1_h, boxstyle="round,pad=0.05,rounding_size=0.08",
                          linewidth=1.5, edgecolor="black", facecolor="#EAEAEA")
ax.add_patch(ch1_box)
ax.text(6.5, ch1_y + ch1_h / 2, "Chapter 1: Introduction\n(4 open questions)", ha="center",
        va="center", fontsize=10.5, fontweight="bold")

core_box = FancyBboxPatch((5.3, core_y), 2.4, core_h, boxstyle="round,pad=0.05,rounding_size=0.08",
                           linewidth=1.5, edgecolor="black", facecolor="#FFF3CD")
ax.add_patch(core_box)
ax.text(6.5, core_y + core_h / 2, "Bayesian Inference\nFramework (Bayes' Theorem)", ha="center",
        va="center", fontsize=9.5)

ax.add_patch(FancyArrowPatch((6.5, ch1_y), (6.5, core_top), arrowstyle="-|>", mutation_scale=14,
                             linewidth=1.3, color="black"))

n_themes = len(themes)
x_centers = np.linspace(1.6, 11.4, n_themes)
theme_w = 2.6

for xc, theme in zip(x_centers, themes):
    ax.add_patch(FancyArrowPatch((6.5, core_y), (xc, theme_y + theme_h), arrowstyle="-|>",
                                 mutation_scale=12, linewidth=1.1, color="gray",
                                 connectionstyle="arc3,rad=0.0"))
    box = FancyBboxPatch((xc - theme_w / 2, theme_y), theme_w, theme_h,
                          boxstyle="round,pad=0.05,rounding_size=0.08",
                          linewidth=1.5, edgecolor="black", facecolor=theme.get("color"),
                          alpha=0.85)
    ax.add_patch(box)
    ax.text(xc, theme_y + theme_h / 2, theme["name"], ha="center", va="center",
            fontsize=10.5, color="white", fontweight="bold")

    for i, ch_text in enumerate(theme["chapters"]):
        y_top = stack_top - i * (ch_h + ch_gap)
        y_bottom = y_top - ch_h
        ch_box = FancyBboxPatch((xc - theme_w / 2, y_bottom), theme_w, ch_h,
                                 boxstyle="round,pad=0.04,rounding_size=0.06",
                                 linewidth=1.0, edgecolor=theme.get("color"),
                                 facecolor=theme.get("color"), alpha=0.12)
        ax.add_patch(ch_box)
        ax.text(xc, (y_top + y_bottom) / 2, ch_text, ha="center", va="center", fontsize=7.6)
        if i == 0:
            ax.add_patch(FancyArrowPatch((xc, theme_y), (xc, y_top), arrowstyle="-",
                                         mutation_scale=8, linewidth=0.9, color="gray"))

concl_box = FancyBboxPatch((5.0, concl_y), 3.0, concl_h, boxstyle="round,pad=0.05,rounding_size=0.08",
                            linewidth=1.5, edgecolor="black", facecolor="#333333")
ax.add_patch(concl_box)
ax.text(6.5, concl_y + concl_h / 2, "Chapter 14: Conclusions\n(this chapter)", ha="center", va="center",
        fontsize=10.5, fontweight="bold", color="white")
connector_y = concl_y + concl_h + 0.22
for xc in x_centers:
    ax.add_patch(FancyArrowPatch((xc, stack_bottom), (xc, connector_y), arrowstyle="-",
                                 mutation_scale=8, linewidth=0.9, color="gray"))
    ax.add_patch(FancyArrowPatch((xc, connector_y), (6.5, connector_y), arrowstyle="-",
                                 mutation_scale=8, linewidth=0.9, color="gray"))
ax.add_patch(FancyArrowPatch((6.5, connector_y), (6.5, concl_y + concl_h), arrowstyle="-|>",
                             mutation_scale=12, linewidth=1.0, color="gray"))

ax.set_title("The Whole Book, One Page: 14 Chapters, Four Themes", fontsize=14,
             fontweight="bold", pad=10)

fig.tight_layout()
fig.savefig("fig_ch14_book_map.pdf")
plt.close(fig)
print("Wrote fig_ch14_book_map.pdf")

# ---------------------------------------------------------------------------
# Figure 2: Toolkit matrix -- methods vs. chapters that used them
# ---------------------------------------------------------------------------
methods = [
    "Metropolis-Hastings",
    "MALA",
    "Hamiltonian Monte Carlo (HMC)",
    "Separable Shadow HMC",
    "Magnetic HMC",
    "No-U-Turn Sampler (NUTS)",
    "Sparse / Distributed GPs",
    "Normalizing Flows",
    "Bayesian NN + Laplace approx.",
    "Nested Sampling (evidence)",
]

chapters = [f"Ch {c}" for c in range(3, 14)]
# chapter titles for x-axis annotation
chapter_labels = [
    "3\nSABR", "4\nVol.\nsurfaces", "5\nOption\nprices", "6\nCredit\nratings",
    "7\nLoan\nrecovery", "8\nAudit\noutcomes", "9\nUnauth.\nexpend.",
    "10\nMotor\ninsurance", "11\nYield\ncurve", "12\nYield curve\nselection",
    "13\nInvestment\nanalyst",
]

# usage[i, j] = 1 if method i used in chapter j (index 0 -> Chapter 3)
usage = np.zeros((len(methods), len(chapters)))


def mark(method_name, chapter_numbers):
    i = methods.index(method_name)
    for c in chapter_numbers:
        usage[i, c - 3] = 1


mark("Metropolis-Hastings", [9])
mark("MALA", [3, 7, 9])
mark("Hamiltonian Monte Carlo (HMC)", [3, 7, 8, 9, 11])
mark("Separable Shadow HMC", [3, 7, 11])
mark("Magnetic HMC", [9])
mark("No-U-Turn Sampler (NUTS)", [8, 11, 12])
mark("Sparse / Distributed GPs", [4, 6])
mark("Normalizing Flows", [5, 8])
mark("Bayesian NN + Laplace approx.", [10])
mark("Nested Sampling (evidence)", [12])
# Chapter 13's Bayesian investment analyst reuses the MCMC family (BLR-ARD)
mark("MALA", [13])
mark("Hamiltonian Monte Carlo (HMC)", [13])
mark("Separable Shadow HMC", [13])
mark("No-U-Turn Sampler (NUTS)", [13])

fig, ax = plt.subplots(figsize=(12.5, 6.6))
cmap = plt.cm.colors.ListedColormap(["#F2F2F2", "#4C72B0"])
ax.imshow(usage, cmap=cmap, aspect="auto", vmin=0, vmax=1)

ax.set_xticks(range(len(chapters)))
ax.set_xticklabels(chapter_labels, fontsize=8.3)
ax.set_yticks(range(len(methods)))
ax.set_yticklabels(methods, fontsize=9.5)

ax.set_xticks(np.arange(-0.5, len(chapters), 1), minor=True)
ax.set_yticks(np.arange(-0.5, len(methods), 1), minor=True)
ax.grid(which="minor", color="white", linewidth=1.6)
ax.tick_params(which="minor", bottom=False, left=False)

for i in range(len(methods)):
    for j in range(len(chapters)):
        if usage[i, j] == 1:
            ax.text(j, i, "✓", ha="center", va="center", color="white",
                    fontsize=11, fontweight="bold")

ax.set_title("The Bayesian ML Toolkit: Which Chapter Used What", fontsize=13.5,
             fontweight="bold", pad=12)
fig.tight_layout()
fig.savefig("fig_ch14_toolkit.pdf")
plt.close(fig)
print("Wrote fig_ch14_toolkit.pdf")


# ---------------------------------------------------------------------------
# Helper: draw a 4-quadrant "contributions & future directions" theme diagram
# (mirrors the structure of the book's own Figs. 14.1 / 14.2 / 14.3)
# ---------------------------------------------------------------------------
def draw_theme_quadrant(center_label, quadrants, out_name, title):
    """
    quadrants: list of exactly 4 dicts, each with keys:
      'name'  -- box header (the sub-topic)
      'contrib' -- short contribution phrase
      'future'  -- short future-direction phrase
    Layout: two on top, two on bottom, arrows converging to the center label,
    same visual grammar as the book's Figs. 14.1-14.3.
    """
    fig, ax = plt.subplots(figsize=(11, 8.1))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 8.15)
    ax.axis("off")

    header_w, header_h = 4.6, 0.55
    body_h = 1.55
    top_y = 5.7
    bot_y = 0.55
    left_x = 0.35
    right_x = 5.65

    positions = [
        (left_x, top_y), (right_x, top_y),
        (left_x, bot_y), (right_x, bot_y),
    ]

    center_w, center_h = 2.6, 1.0
    center_x = 5.5 - center_w / 2
    center_y = 3.4 - center_h / 2

    for (x0, y0), q in zip(positions, quadrants):
        header = FancyBboxPatch((x0, y0 + body_h), header_w, header_h,
                                 boxstyle="square,pad=0.0",
                                 linewidth=0, facecolor="black")
        ax.add_patch(header)
        ax.text(x0 + header_w / 2, y0 + body_h + header_h / 2, q["name"],
                ha="center", va="center", fontsize=9.3, color="white",
                fontweight="bold")

        body = FancyBboxPatch((x0, y0), header_w, body_h,
                               boxstyle="square,pad=0.0",
                               linewidth=1.0, edgecolor="#999999",
                               facecolor="#FAFAFA")
        ax.add_patch(body)
        ax.plot([x0 + header_w / 2, x0 + header_w / 2], [y0 + 0.08, y0 + body_h - 0.08],
                color="#BBBBBB", lw=0.8)
        ax.text(x0 + header_w * 0.25, y0 + body_h / 2, q["contrib"], ha="center",
                va="center", fontsize=8.0, wrap=True,
                bbox=dict(boxstyle="round,pad=0.3", fc="#E8F0FA", ec="none"))
        ax.text(x0 + header_w * 0.75, y0 + body_h / 2, q["future"], ha="center",
                va="center", fontsize=8.0, wrap=True,
                bbox=dict(boxstyle="round,pad=0.3", fc="#FDEFE3", ec="none"))

        cx, cy = x0 + header_w / 2, y0 + (body_h if y0 == top_y else body_h + header_h)
        target = (center_x + center_w / 2, center_y + (center_h if y0 == bot_y else 0))
        ax.plot([cx, target[0]], [y0 + (0 if y0 == bot_y else body_h + header_h),
                                    center_y + (center_h if y0 == bot_y else 0)],
                color="black", lw=1.0)

    center_box = FancyBboxPatch((center_x, center_y), center_w, center_h,
                                 boxstyle="square,pad=0.0",
                                 linewidth=0, facecolor="black")
    ax.add_patch(center_box)
    ax.text(center_x + center_w / 2, center_y + center_h / 2, center_label,
            ha="center", va="center", fontsize=12.5, color="white",
            fontweight="bold")

    ax.text(0.35, 7.95, "■ contribution   □ future direction", fontsize=8,
            color="#555555")
    ax.set_title(title, fontsize=13.5, fontweight="bold", pad=6)
    fig.tight_layout()
    fig.savefig(out_name)
    plt.close(fig)
    print(f"Wrote {out_name}")


def draw_theme_triad(center_label, boxes, out_name, title):
    """
    boxes: list of exactly 3 dicts with keys 'name', 'contrib', 'future'.
    Layout: two boxes on top converging into the center label, one box
    below the center label -- matching the book's own Fig. 14.1, which
    has only three pillars for the derivative modeling theme.
    """
    fig, ax = plt.subplots(figsize=(11, 7.4))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 7.55)
    ax.axis("off")

    header_w, header_h = 4.6, 0.55
    body_h = 1.55
    top_y = 5.35
    left_x = 0.35
    right_x = 5.65

    center_w, center_h = 2.6, 1.0
    center_x = 5.5 - center_w / 2
    center_y = 3.15 - center_h / 2

    bot_w = 4.6
    bot_h = 1.9
    bot_header_h = 0.5
    bot_x = 5.5 - bot_w / 2
    bot_y = 0.35

    top_positions = [(left_x, top_y), (right_x, top_y)]
    for (x0, y0), q in zip(top_positions, boxes[:2]):
        header = FancyBboxPatch((x0, y0 + body_h), header_w, header_h,
                                 boxstyle="square,pad=0.0",
                                 linewidth=0, facecolor="black")
        ax.add_patch(header)
        ax.text(x0 + header_w / 2, y0 + body_h + header_h / 2, q["name"],
                ha="center", va="center", fontsize=9.3, color="white",
                fontweight="bold")
        body = FancyBboxPatch((x0, y0), header_w, body_h,
                               boxstyle="square,pad=0.0",
                               linewidth=1.0, edgecolor="#999999",
                               facecolor="#FAFAFA")
        ax.add_patch(body)
        ax.plot([x0 + header_w / 2, x0 + header_w / 2], [y0 + 0.08, y0 + body_h - 0.08],
                color="#BBBBBB", lw=0.8)
        ax.text(x0 + header_w * 0.25, y0 + body_h / 2, q["contrib"], ha="center",
                va="center", fontsize=8.0,
                bbox=dict(boxstyle="round,pad=0.3", fc="#E8F0FA", ec="none"))
        ax.text(x0 + header_w * 0.75, y0 + body_h / 2, q["future"], ha="center",
                va="center", fontsize=8.0,
                bbox=dict(boxstyle="round,pad=0.3", fc="#FDEFE3", ec="none"))
        cx = x0 + header_w / 2
        ax.plot([cx, center_x + center_w / 2], [y0, center_y + center_h],
                color="black", lw=1.0)

    q3 = boxes[2]
    header3 = FancyBboxPatch((bot_x, bot_y + bot_h), bot_w, bot_header_h,
                              boxstyle="square,pad=0.0", linewidth=0, facecolor="black")
    ax.add_patch(header3)
    ax.text(bot_x + bot_w / 2, bot_y + bot_h + bot_header_h / 2, q3["name"],
            ha="center", va="center", fontsize=9.3, color="white", fontweight="bold")
    body3 = FancyBboxPatch((bot_x, bot_y), bot_w, bot_h, boxstyle="square,pad=0.0",
                            linewidth=1.0, edgecolor="#999999", facecolor="#FAFAFA")
    ax.add_patch(body3)
    ax.plot([bot_x + bot_w / 2, bot_x + bot_w / 2], [bot_y + 0.08, bot_y + bot_h - 0.08],
            color="#BBBBBB", lw=0.8)
    ax.text(bot_x + bot_w * 0.25, bot_y + bot_h / 2, q3["contrib"], ha="center", va="center",
            fontsize=8.0, bbox=dict(boxstyle="round,pad=0.3", fc="#E8F0FA", ec="none"))
    ax.text(bot_x + bot_w * 0.75, bot_y + bot_h / 2, q3["future"], ha="center", va="center",
            fontsize=8.0, bbox=dict(boxstyle="round,pad=0.3", fc="#FDEFE3", ec="none"))
    ax.plot([bot_x + bot_w / 2, center_x + center_w / 2], [bot_y + bot_h + bot_header_h, center_y],
            color="black", lw=1.0)

    center_box = FancyBboxPatch((center_x, center_y), center_w, center_h,
                                 boxstyle="square,pad=0.0", linewidth=0, facecolor="black")
    ax.add_patch(center_box)
    ax.text(center_x + center_w / 2, center_y + center_h / 2, center_label,
            ha="center", va="center", fontsize=12.5, color="white", fontweight="bold")

    ax.text(0.35, 7.35, "■ contribution   □ future direction", fontsize=8, color="#555555")
    ax.set_title(title, fontsize=13.5, fontweight="bold", pad=6)
    fig.tight_layout()
    fig.savefig(out_name)
    plt.close(fig)
    print(f"Wrote {out_name}")


# --- Derivative modeling (mirrors book Fig. 14.1: three pillars, not four) ---
draw_theme_triad(
    "DERIVATIVE\nMODELING",
    [
        {
            "name": "Bayesian calibration of the SABR model",
            "contrib": "Quantify uncertainty in\nSABR parameters and\npredictions",
            "future": "Extend to more\ncomplex models",
        },
        {
            "name": "Equity volatility surface modeling using GPs",
            "contrib": "Allows more advanced\nmodels to be catered for",
            "future": "Incorporate no-arbitrage\nconditions into loss function",
        },
        {
            "name": "Normalizing flows for option pricing",
            "contrib": "Learn risk-neutral\ndensity from option\nprices",
            "future": "Extend to entire\nvolatility surface",
        },
    ],
    "fig_ch14_derivative_modeling.pdf",
    "Contributions & Future Directions: Derivative Modeling",
)

# --- Financial management (mirrors book Fig. 14.2) ---
draw_theme_quadrant(
    "FINANCIAL\nMANAGEMENT",
    [
        {
            "name": "Sparse and distributed GPs for corporate credit ratings",
            "contrib": "GPs outperform\nbenchmarks",
            "future": "Extend to other\ndistributed GP\nvariations",
        },
        {
            "name": "MCMC and defaulted loan recovery modeling",
            "contrib": "Identified relevant\ninputs for recovery\nmodeling",
            "future": "Extend to model\nabsolute recovery\namounts",
        },
        {
            "name": "Normalizing flows for audit opinion model selection",
            "contrib": "Probabilistically sound\naudit opinion model\nselection approach",
            "future": "Extend to advanced\nmodels such as BNNs",
        },
        {
            "name": "Detecting unauthorized expenditure in municipalities",
            "contrib": "Bayesian approach to\nmodeling unauthorized\nexpenditures",
            "future": "Extend to model\nabsolute unlawful\nexpenditure",
        },
    ],
    "fig_ch14_financial_management.pdf",
    "Contributions & Future Directions: Financial Management",
)

# --- Insurance and investments (mirrors book Fig. 14.3) ---
draw_theme_quadrant(
    "INSURANCE AND\nINVESTMENTS",
    [
        {
            "name": "Bayesian inference of motor claims",
            "contrib": "Identified most relevant\ninputs for motor claims\nmodeling",
            "future": "Extend to modeling\nclaim frequency and\namounts",
        },
        {
            "name": "Bayesian calibration of Nelson & Siegel model",
            "contrib": "Identified the most\nrelevant inputs for\nyield curve modeling",
            "future": "Extend to model the\ndynamic Nelson &\nSiegel model",
        },
        {
            "name": "Yield curve model selection",
            "contrib": "Probabilistically sound\nyield curve model\nselection approach",
            "future": "Extend to advanced\nmodels such as dynamic\nNelson & Siegel",
        },
        {
            "name": "Bayesian investment analyst",
            "contrib": "Introduced a Bayesian\ninvestment analyst\non the JSE",
            "future": "Extend to distributed,\nlocalized industry\nexperts",
        },
    ],
    "fig_ch14_insurance_investments.pdf",
    "Contributions & Future Directions: Insurance and Investments",
)

print("\nAll Chapter 14 figures generated.")
