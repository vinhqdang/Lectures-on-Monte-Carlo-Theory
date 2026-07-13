#!/usr/bin/env python3
"""
Figure generation for Chapter 6 slides:
  "Sparse and Distributed Gaussian Processes for Modeling Corporate Credit Ratings"

Produces two self-contained, illustrative (TOY, not the book's real data/results)
vector figures used by chapter6_slides.tex:

  (a) fig_softmax_bar.pdf
      Bar chart of softmax probabilities from a toy 3-class multinomial logistic
      regression prediction for one fictional company ("Acme Corp"), using
      made-up financial-ratio features and made-up regression coefficients.

  (b) fig_distributed_gp.pdf
      Schematic of the "distributed GP" / product-of-experts idea: a large toy
      1D dataset is split into chunks, a separate small GP "expert" is fitted to
      each chunk, and the experts' Gaussian predictive distributions are combined
      (product-of-experts) into a single aggregated Gaussian prediction.

Run with: python3 gen_figures.py
Outputs are written as PDF (vector) into the current directory.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# ----------------------------------------------------------------------------
# Palette (validated categorical palette, light-mode slots; see dataviz skill)
# ----------------------------------------------------------------------------
BLUE    = '#2a78d6'   # slot 1
AQUA    = '#1baf7a'   # slot 2
YELLOW  = '#eda100'   # slot 3
GREEN   = '#008300'   # slot 4
VIOLET  = '#4a3aa7'   # slot 5
RED     = '#e34948'   # slot 6
MAGENTA = '#e87ba4'   # slot 7
ORANGE  = '#eb6834'   # slot 8

INK          = '#0b0b0b'
INK_SECOND   = '#52514e'
INK_MUTED    = '#898781'
GRID         = '#e1e0d9'
SURFACE      = '#fcfcfb'

plt.rcParams.update({
    'font.size': 11,
    'axes.edgecolor': INK_MUTED,
    'axes.labelcolor': INK,
    'text.color': INK,
    'xtick.color': INK_SECOND,
    'ytick.color': INK_SECOND,
    'axes.grid': False,
    'figure.facecolor': SURFACE,
    'savefig.facecolor': SURFACE,
})


# ==============================================================================
# Figure (a): softmax bar chart for toy multinomial logistic regression
# ==============================================================================
def make_softmax_figure():
    # Toy company: "Acme Corp" -- made-up standardized financial-ratio features
    # x = [debtRatio_z, profitMargin_z, currentRatio_z]  (z-scored, illustrative)
    x = np.array([1.0, 0.40, -0.60, 1.30])  # [bias=1, debtRatio, profitMargin, currentRatio]

    # Toy learned coefficients (one weight vector per class; "low risk" is the
    # reference/base class fixed at all-zero logit, as is conventional for MLR)
    w_low    = np.array([0.0, 0.0, 0.0, 0.0])     # reference class
    w_medium = np.array([-0.30, 1.10, -0.50, -0.20])
    w_high   = np.array([-0.80, 2.30, -1.40, -0.90])

    z_low    = w_low @ x
    z_medium = w_medium @ x
    z_high   = w_high @ x
    logits = np.array([z_low, z_medium, z_high])

    exp_logits = np.exp(logits)
    probs = exp_logits / exp_logits.sum()

    labels = ['Low risk', 'Medium risk', 'High risk']
    colors = [AQUA, YELLOW, RED]

    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    bars = ax.bar(labels, probs, color=colors, width=0.55,
                   edgecolor='none', zorder=3)

    for rect, p in zip(bars, probs):
        ax.text(rect.get_x() + rect.get_width() / 2, p + 0.02,
                 f'{p:.3f}', ha='center', va='bottom',
                 fontsize=12, color=INK, fontweight='bold')

    ax.set_ylim(0, 1.0)
    ax.set_ylabel('Predicted probability')
    ax.set_title("Multinomial logistic regression:\ntoy prediction for “Acme Corp”",
                  fontsize=12, color=INK)
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_color(INK_MUTED)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)

    fig.tight_layout()
    fig.savefig('fig_softmax_bar.pdf')
    plt.close(fig)
    print('Softmax logits :', np.round(logits, 3))
    print('Softmax probs  :', np.round(probs, 3), ' (sum =', probs.sum(), ')')


# ==============================================================================
# Figure (b): distributed GP / product-of-experts schematic on toy 1D data
# ==============================================================================
def true_function(x):
    return np.sin(1.3 * x) + 0.15 * x


def rbf_kernel(x1, x2, ell=1.0, sf=1.0):
    d2 = (x1[:, None] - x2[None, :]) ** 2
    return sf ** 2 * np.exp(-0.5 * d2 / ell ** 2)


def gp_posterior(x_train, y_train, x_star, ell=1.0, sf=1.0, noise=0.05):
    Kxx = rbf_kernel(x_train, x_train, ell, sf) + noise ** 2 * np.eye(len(x_train))
    Kxs = rbf_kernel(x_train, x_star, ell, sf)
    Kss = rbf_kernel(x_star, x_star, ell, sf)
    Kxx_inv = np.linalg.inv(Kxx)
    mean = Kxs.T @ Kxx_inv @ y_train
    cov = Kss - Kxs.T @ Kxx_inv @ Kxs
    var = np.clip(np.diag(cov), 1e-8, None)
    return mean, var


def make_distributed_gp_figure():
    rng = np.random.default_rng(7)

    # A single big toy dataset, split into M=3 chunks ("experts")
    n_per_expert = 8
    expert_colors = [BLUE, AQUA, ORANGE]
    centers = [-5.0, 0.0, 5.0]
    spread = 1.6

    x_star = np.linspace(-8, 8, 200)
    f_true = true_function(x_star)

    experts = []
    for c, col in zip(centers, expert_colors):
        x_j = np.sort(rng.uniform(c - spread, c + spread, n_per_expert))
        y_j = true_function(x_j) + rng.normal(0, 0.15, n_per_expert)
        mean_j, var_j = gp_posterior(x_j, y_j, x_star, ell=1.4, sf=1.2, noise=0.15)
        experts.append(dict(x=x_j, y=y_j, mean=mean_j, var=var_j, color=col))

    # Generalized product-of-experts combination (Eqs. 6.6-6.7 with beta_j = 1/M,
    # i.e. equal weights -- a simple illustrative special case)
    M = len(experts)
    beta = 1.0 / M
    prec_sum = np.zeros_like(x_star)
    weighted_mean_sum = np.zeros_like(x_star)
    for e in experts:
        prec_j = 1.0 / e['var']
        prec_sum += beta * prec_j
        weighted_mean_sum += beta * prec_j * e['mean']
    poe_var = 1.0 / prec_sum
    poe_mean = poe_var * weighted_mean_sum

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), sharey=True)

    # --- Left panel: the M local experts, each fitted on its own data chunk ---
    ax = axes[0]
    ax.plot(x_star, f_true, color=INK_MUTED, linewidth=1.2, linestyle='--',
             label='true function', zorder=2)
    for j, e in enumerate(experts):
        sd = np.sqrt(e['var'])
        ax.fill_between(x_star, e['mean'] - 1.96 * sd, e['mean'] + 1.96 * sd,
                          color=e['color'], alpha=0.15, linewidth=0)
        ax.plot(x_star, e['mean'], color=e['color'], linewidth=1.8,
                 label=f'expert {j+1} (chunk {j+1})', zorder=3)
        ax.scatter(e['x'], e['y'], color=e['color'], s=22, zorder=4,
                    edgecolor='white', linewidth=0.5)
    ax.set_title('Step 1: fit one small GP\n"expert" per data chunk', fontsize=11.5)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$f(x)$')
    ax.legend(fontsize=7.5, loc='upper left', frameon=False, ncol=1)
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_color(INK_MUTED)
    ax.tick_params(length=0)

    # --- Right panel: combined product-of-experts prediction ---
    ax = axes[1]
    ax.plot(x_star, f_true, color=INK_MUTED, linewidth=1.2, linestyle='--',
             label='true function', zorder=2)
    for j, e in enumerate(experts):
        ax.plot(x_star, e['mean'], color=e['color'], linewidth=0.9,
                 alpha=0.55, zorder=2)
    poe_sd = np.sqrt(poe_var)
    ax.fill_between(x_star, poe_mean - 1.96 * poe_sd, poe_mean + 1.96 * poe_sd,
                      color=VIOLET, alpha=0.18, linewidth=0, label='95\\% CI (combined)')
    ax.plot(x_star, poe_mean, color=VIOLET, linewidth=2.2,
             label='product-of-experts mean', zorder=4)
    ax.set_title('Step 2: combine experts via\nproduct-of-experts rule', fontsize=11.5)
    ax.set_xlabel('$x$')
    ax.legend(fontsize=8, loc='upper left', frameon=False)
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_color(INK_MUTED)
    ax.tick_params(length=0)

    fig.suptitle('Distributed Gaussian processes: split data, fit experts, pool predictions',
                  fontsize=12.5, y=1.02)
    fig.tight_layout()
    fig.savefig('fig_distributed_gp.pdf', bbox_inches='tight')
    plt.close(fig)


# ==============================================================================
# Figure (c): tiny flow diagram: many experts -> combination rule -> one Gaussian
# ==============================================================================
def make_poe_flow_figure():
    fig, ax = plt.subplots(figsize=(9.5, 3.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3.2)
    ax.axis('off')

    expert_colors = [BLUE, AQUA, ORANGE]
    labels = ['Expert 1\n$\\mathcal{N}(m_1,\\sigma_1^2)$',
              'Expert 2\n$\\mathcal{N}(m_2,\\sigma_2^2)$',
              'Expert $M$\n$\\mathcal{N}(m_M,\\sigma_M^2)$']
    xs = [0.9, 0.9, 0.9]
    ys = [2.55, 1.6, 0.55]

    for x, y, col, lab in zip(xs, ys, expert_colors, labels):
        box = FancyBboxPatch((x, y - 0.42), 2.0, 0.84,
                               boxstyle='round,pad=0.05,rounding_size=0.08',
                               linewidth=1.4, edgecolor=col, facecolor='white')
        ax.add_patch(box)
        ax.text(x + 1.0, y, lab, ha='center', va='center', fontsize=9.5, color=INK)

    # dots between expert 2 and expert M
    ax.text(1.9, 1.02, '$\\vdots$', ha='center', va='center', fontsize=14, color=INK_MUTED)

    # combination box
    comb_x, comb_y = 4.5, 1.15
    comb = FancyBboxPatch((comb_x, comb_y), 2.3, 0.9,
                            boxstyle='round,pad=0.05,rounding_size=0.1',
                            linewidth=1.6, edgecolor=VIOLET, facecolor='#4a3aa71a')
    ax.add_patch(comb)
    ax.text(comb_x + 1.15, comb_y + 0.45,
             'Product-of-\nexperts rule\n$p(f_*)\\propto\\prod_j p_j(f_*)^{\\beta_j}$',
             ha='center', va='center', fontsize=9, color=INK)

    for y in ys:
        arr = FancyArrowPatch((2.9, y), (comb_x, comb_y + 0.45), color=INK_MUTED,
                                arrowstyle='-|>', mutation_scale=10, linewidth=1.1,
                                connectionstyle='arc3,rad=0.08')
        ax.add_patch(arr)

    # output box
    out_x, out_y = 7.6, 1.15
    out = FancyBboxPatch((out_x, out_y), 2.1, 0.9,
                           boxstyle='round,pad=0.05,rounding_size=0.1',
                           linewidth=1.6, edgecolor=RED, facecolor='#e3494818')
    ax.add_patch(out)
    ax.text(out_x + 1.05, out_y + 0.45,
             'Combined\nprediction\n$\\mathcal{N}(m_*,\\sigma_*^2)$',
             ha='center', va='center', fontsize=9.3, color=INK)

    arr = FancyArrowPatch((comb_x + 2.3, comb_y + 0.45), (out_x, out_y + 0.45),
                            color=INK_MUTED, arrowstyle='-|>', mutation_scale=12,
                            linewidth=1.3)
    ax.add_patch(arr)

    fig.tight_layout()
    fig.savefig('fig_poe_flow.pdf', bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    make_softmax_figure()
    make_distributed_gp_figure()
    make_poe_flow_figure()
    print('All figures written to current directory as PDF.')
