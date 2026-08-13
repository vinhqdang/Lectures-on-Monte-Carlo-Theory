#!/usr/bin/env python3
"""
gen_figures.py -- Figures for Chapter 16: Philosophy of AI
(An Introduction to Universal Artificial Intelligence, 2024)

Generates:
  1. Book-diagram crops (via PyMuPDF) for figures that are illustrations
     rather than plottable data: teleportation diagrams (16.2, 16.3),
     Chinese room (16.4), Moore's Law (16.5).
  2. Original matplotlib figures illustrating concepts discussed in the
     text: the LH-intelligence weighting scheme (Occam's razor via
     Kolmogorov complexity), the No-Free-Lunch "averaging" phenomenon,
     a schematic of the forward/reverse diffusion process, and a
     convergence plot for the toy AIQ sampling estimator.

Run with:
    conda run -n py313 python3 gen_figures.py
"""

import io
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.join(HERE, "figures")
os.makedirs(FIGDIR, exist_ok=True)

BOOK_PDF = os.path.join(
    os.path.dirname(HERE),
    "An Introduction to Universal Artificial Intelligence 2024.pdf",
)

plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 10,
    "figure.dpi": 150,
})


# ══════════════════════════════════════════════════════════════════════════
# PART 1 -- Crop diagrams from the book PDF (PyMuPDF)
# ══════════════════════════════════════════════════════════════════════════
def crop_book_figures():
    import fitz  # PyMuPDF
    import PIL.Image

    doc = fitz.open(BOOK_PDF)
    ZOOM = 4  # render at 4x for crisp raster crops
    mat = fitz.Matrix(ZOOM, ZOOM)

    jobs = [
        # (pdf_page_index (0-based), (x0,y0,x1,y1) in PDF points, out filename)
        (440, (65.9, 125.3, 183.9, 258.8), "fig16_2_teleport_cutpaste.png"),
        (440, (210.9, 58.3, 453.1, 306.6), "fig16_3_teleport_copydelete.png"),
        (442, (100.0, 58.5, 428.9, 223.5), "fig16_4_chinese_room.png"),
        (446, (60.0, 266.0, 465.0, 536.0), "fig16_5_moore_law.png"),
    ]

    for page_idx, box, fname in jobs:
        page = doc[page_idx]
        pix = page.get_pixmap(matrix=mat, clip=fitz.Rect(*box))
        img = PIL.Image.open(io.BytesIO(pix.tobytes("png")))
        out_path = os.path.join(FIGDIR, fname)
        img.save(out_path)
        print(f"  cropped {fname}  ({img.width}x{img.height})")

    doc.close()


# ══════════════════════════════════════════════════════════════════════════
# PART 2 -- Original matplotlib figures
# ══════════════════════════════════════════════════════════════════════════

def fig_occam_weighting():
    """Illustrates the universal prior 2^-K(nu): weight decays exponentially
    with Kolmogorov complexity K(nu), formalizing Occam's razor / Epicurus'
    principle simultaneously (every K gets positive weight, but simpler
    environments dominate)."""
    K = np.arange(0, 21)
    weight = 2.0 ** (-K)

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.bar(K, weight, color="#3B6FA0", edgecolor="black", linewidth=0.6, width=0.7)
    ax.set_yscale("log")
    ax.set_xlabel(r"Kolmogorov complexity $K(\nu)$ of environment $\nu$  (bits)")
    ax.set_ylabel(r"Universal prior weight  $w^U_\nu = 2^{-K(\nu)}$")
    ax.set_title("Occam's razor + Epicurus' principle in one formula")
    ax.annotate("simple environments\n(short programs)\nget most of the weight",
                xy=(1, 2.0**-1), xytext=(6, 2.0**-2),
                arrowprops=dict(arrowstyle="->", color="black"),
                fontsize=9, ha="left")
    ax.annotate("complex environments\nstill get weight $>0$\n(never ruled out a priori)",
                xy=(18, 2.0**-18), xytext=(8, 2.0**-14),
                arrowprops=dict(arrowstyle="->", color="black"),
                fontsize=9, ha="left")
    ax.grid(alpha=0.25, which="both")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "occam_weighting.pdf"))
    plt.close(fig)


def fig_no_free_lunch():
    """Empirically illustrates the No-Free-Lunch phenomenon: averaged over
    ALL possible reward functions on a small finite search space, two very
    different search algorithms (greedy hill-climb vs. a fixed cyclic
    'random-looking' schedule) achieve identical average performance."""
    rng = np.random.default_rng(0)
    n_points = 8      # tiny search space {0,...,7}
    n_trials = 4000    # number of steps allowed
    n_functions = 3000  # sample of reward functions (bijections)

    def best_value_seen(order, values):
        seen = -np.inf
        out = []
        for idx in order:
            seen = max(seen, values[idx])
            out.append(seen)
        return out

    greedy_curve = np.zeros(n_points)
    cyclic_curve = np.zeros(n_points)

    perms_idx = np.arange(n_points)
    for _ in range(n_functions):
        values = rng.permutation(n_points).astype(float)  # a uniformly random bijection

        # "Greedy": always move towards the neighbour with a higher value seen so far
        # -- but with no gradient info in a random function, it reduces to a fixed
        # exploration order determined by the algorithm's own internal logic.
        greedy_order = np.argsort(-values)[::1]  # algorithm A's fixed traversal rule
        rng.shuffle(greedy_order)  # algorithm's traversal is independent of the *labels*
        cyclic_order = perms_idx.copy()  # algorithm B: always 0,1,2,...,7

        greedy_curve += best_value_seen(greedy_order, values)
        cyclic_curve += best_value_seen(cyclic_order, values)

    greedy_curve /= n_functions
    cyclic_curve /= n_functions

    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    steps = np.arange(1, n_points + 1)
    ax.plot(steps, greedy_curve, "o-", label="Algorithm A (fixed rule 1)", color="#3B6FA0")
    ax.plot(steps, cyclic_curve, "s--", label="Algorithm B (fixed rule 2)", color="#C0574A")
    ax.set_xlabel("Number of points visited so far")
    ax.set_ylabel("Average best value found\n(averaged over all reward functions)")
    ax.set_title("No Free Lunch: averaged over ALL problems, curves coincide")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "no_free_lunch.pdf"))
    plt.close(fig)


def fig_diffusion_schematic():
    """Schematic of the forward noising / reverse denoising diffusion
    process referenced in Section 16.8: a 1-D signal x_0 is progressively
    corrupted by Gaussian noise over T steps into pure noise x_T; the
    generative model learns to reverse this."""
    rng = np.random.default_rng(1)
    T_display = 5
    n = 200
    t_axis = np.linspace(0, 4 * np.pi, n)
    x0 = np.sin(t_axis) + 0.3 * np.sin(3 * t_axis)

    fig, axes = plt.subplots(1, T_display, figsize=(11, 2.4), sharey=True)
    alphas = np.linspace(0, 1, T_display) ** 1.6  # noise fraction schedule
    for i, (ax, a) in enumerate(zip(axes, alphas)):
        xt = np.sqrt(1 - a) * x0 + np.sqrt(a) * rng.normal(size=n)
        ax.plot(t_axis, xt, color="#3B6FA0", linewidth=1.3)
        ax.set_title(rf"$x_{{{i}}}$" if i < T_display - 1 else rf"$x_T$", fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    fig.suptitle(r"Forward diffusion: $x_0 \to x_1 \to \cdots \to x_T$ (signal $\to$ noise)",
                 fontsize=12, y=1.05)
    # arrows between panels
    for i in range(T_display - 1):
        fig.text((i + 1) / T_display - 0.015, 0.5, r"$\rightarrow$", fontsize=16, va="center")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(os.path.join(FIGDIR, "diffusion_schematic.pdf"), bbox_inches="tight")
    plt.close(fig)


def fig_aiq_convergence():
    """Toy illustration of the AIQ sampling estimator: as we sample more
    programs p_i ~ 2^{-l(p)} (approximated here by geometric program-length
    sampling) and average the agent's empirical return, the running average
    converges to the true (unknown) LH-intelligence-style score."""
    rng = np.random.default_rng(42)
    N = 4000
    # Simulate "value" achieved by a fixed agent on the i-th sampled program:
    # shorter programs (simpler environments) give higher expected value,
    # emulating the universal-prior weighting; add trial-to-trial noise.
    lengths = rng.geometric(p=0.15, size=N)  # length of BF-style program
    true_mean_given_len = 1.0 / (1.0 + 0.15 * lengths)
    values = np.clip(true_mean_given_len + rng.normal(scale=0.15, size=N), 0, 1)

    running_avg = np.cumsum(values) / np.arange(1, N + 1)

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.plot(np.arange(1, N + 1), running_avg, color="#3B6FA0", linewidth=1.4)
    ax.axhline(running_avg[-1], color="#C0574A", linestyle="--", linewidth=1,
               label=fr"$\hat\Upsilon(\pi)\approx{running_avg[-1]:.3f}$ after $N={N}$ samples")
    ax.set_xlabel(r"Number of sampled programs $N$")
    ax.set_ylabel(r"Running average $\hat\Upsilon(\pi)=\frac{1}{N}\sum_i \hat V^\pi_{p_i}$")
    ax.set_title("AIQ estimator: empirical average converges as $N$ grows")
    ax.set_xscale("log")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "aiq_convergence.pdf"))
    plt.close(fig)


def fig_lh_intelligence_schematic():
    """Bar chart schematic: LH-intelligence Upsilon(pi) as a weighted sum
    of an agent's value across a handful of representative environments of
    increasing Kolmogorov complexity, contrasting a 'narrow' agent that only
    does well on one complex environment with a 'general' agent that does
    reasonably on all of them."""
    envs = ["$\\nu_1$\n(simple)", "$\\nu_2$", "$\\nu_3$", "$\\nu_4$", "$\\nu_5$\n(complex)"]
    weights = np.array([0.50, 0.25, 0.125, 0.0625, 0.0625])
    weights /= weights.sum()

    general_agent = np.array([0.9, 0.85, 0.75, 0.65, 0.55])
    narrow_agent = np.array([0.05, 0.05, 0.05, 0.05, 0.95])

    upsilon_general = float(np.sum(weights * general_agent))
    upsilon_narrow = float(np.sum(weights * narrow_agent))

    x = np.arange(len(envs))
    width = 0.35

    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    ax.bar(x - width / 2, general_agent, width, label="General agent $\\pi_1$",
           color="#3B6FA0")
    ax.bar(x + width / 2, narrow_agent, width, label="Narrow agent $\\pi_2$",
           color="#C0574A")
    ax.set_xticks(x)
    ax.set_xticklabels(envs)
    ax.set_ylabel(r"Value $V_\nu^\pi(\epsilon)$ in environment $\nu$")
    ax.set_ylim(0, 1.05)
    ax.set_title("LH-intelligence rewards breadth, weighted by simplicity")
    for i, w in enumerate(weights):
        ax.text(i, 1.0, f"$w_{{\\nu_{i+1}}}$={w:.3f}", ha="center", fontsize=8)
    ax.legend(loc="lower left")
    ax.text(0.98, 0.02,
            fr"$\Upsilon(\pi_1)\approx{upsilon_general:.2f}$"
            "\n"
            fr"$\Upsilon(\pi_2)\approx{upsilon_narrow:.2f}$",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=10,
            bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray"))
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "lh_intelligence_schematic.pdf"))
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Cropping book diagrams...")
    crop_book_figures()

    print("Generating original figures...")
    fig_occam_weighting()
    fig_no_free_lunch()
    fig_diffusion_schematic()
    fig_aiq_convergence()
    fig_lh_intelligence_schematic()

    print("Done. Figures written to", FIGDIR)
