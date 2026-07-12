#!/usr/bin/env python3
"""
gen_figures.py -- Figures for Chapter 14: Further contrasts between e-values
and p-values -- "Hypothesis Testing with E-Values" by Ramdas & Wang
(arXiv:2410.23614)

Generates (as vector PDFs, into this figures/ directory):
  fig1_duality_curve.pdf   -- the duality map: as the test statistic X moves,
                              the "best" p-value S(X;P) and the "best" e-value
                              1/S(X;P) trace out reciprocal curves, with the
                              concrete e=4 <-> p=0.25 example marked.
  fig2_contrast_panel.pdf  -- a side-by-side summary panel contrasting the
                              structural (conditional expectation / probability)
                              representations of e-values and p-values, and the
                              Conv-vs-Span existence conditions of Section 14.3.

Run with: python3 gen_figures.py
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    'font.size': 11,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'figure.autolayout': True,
})

rng = np.random.default_rng(20240712)

# ─────────────────────────────────────────────────────────────────────────
# Figure 1: The duality curve  sup E = 1 / S(X;P) = 1 / inf P
#   We use X ~ Exponential(1) under P as a simple monotone test statistic,
#   so that S(x;P) = P(X >= x) = exp(-x) is the left-continuous survival
#   function appearing in Theorem 14.2.
# ─────────────────────────────────────────────────────────────────────────
x = np.linspace(0.02, 4.5, 800)
S = np.exp(-x)         # best p-variable: p = S(x;P)  (Proposition 14.1)
E = 1.0 / S            # dual e-variable: e = 1/S(x;P) (Theorem 14.2)

fig, ax1 = plt.subplots(figsize=(8.4, 5.2))
ax2 = ax1.twinx()

l1, = ax1.plot(x, S, color='C3', lw=2.5,
               label=r'$p(x) = S(x;\mathbb{P}) = \mathbb{P}(X\geq x)$')
l2, = ax2.plot(x, E, color='C0', lw=2.5,
               label=r'$e(x) = 1/S(x;\mathbb{P})$  (dual e-value)')

# concrete numerical illustration: p = 0.25  <->  e = 4
x0 = -np.log(0.25)
ax1.plot([x0], [0.25], 'o', color='C3', markersize=9, zorder=5)
ax2.plot([x0], [4.0], 'o', color='C0', markersize=9, zorder=5)
ax1.axvline(x0, color='gray', ls=':', lw=1)
ax1.annotate(r'$x_0=\ln 4\approx 1.386$' '\n' r'$p=0.25,\ e=1/0.25=4$',
             xy=(x0, 0.25), xytext=(x0 + 0.35, 0.55),
             arrowprops=dict(arrowstyle='->', color='gray'), fontsize=9)

ax1.set_xlabel(r'observed test statistic $x$ (larger $=$ more evidence against $H_0$)')
ax1.set_ylabel(r'best p-value $p(x)$', color='C3')
ax2.set_ylabel(r'dual e-value $e(x)=1/p(x)$', color='C0')
ax1.tick_params(axis='y', colors='C3')
ax2.tick_params(axis='y', colors='C0')
ax1.set_ylim(0, 1.05)
ax2.set_ylim(0, 20)
ax1.set_title('Theorem 14.2: the duality between e-values and p-values\n'
              r'$\sup_{E\in\mathcal{E}^X} E = 1/S(X;\mathbb{P})$   ($X\sim\mathrm{Exp}(1)$ under $\mathbb{P}$, illustrative)')
lines = [l1, l2]
ax1.legend(lines, [ln.get_label() for ln in lines], loc='upper center', fontsize=9)
fig.savefig('fig1_duality_curve.pdf')
plt.close(fig)

# ─────────────────────────────────────────────────────────────────────────
# Figure 2: Side-by-side contrast panel (e-values vs. p-values)
#   A "table-as-figure" summarizing Sections 14.1-14.3: the representation
#   results (Thm 14.4 / 14.5) and the existence condition (Thm 14.6).
# ─────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 6.4))
ax.axis('off')

rows = [
    ("Monotone role of $X$",
     r"$E$ increasing in $X$", r"$P$ decreasing in $X$"),
    ("Natural bound (Prop. 14.1)",
     r"$E \leq 1 / S(X;\mathbb{P})$", r"$P \geq S(X;\mathbb{P})$"),
    ("Duality (Thm. 14.2)",
     r"$\sup_{E\in\mathcal{E}^X} E$", r"$=\ 1\ /\ \inf_{P\in\mathcal{U}^X} P$"),
    ("Structural representation",
     r"$E \leq \mathbb{E}^{\mathbb{P}}[\, d\mathbb{Q}/d\mathbb{P} \mid X \,]$" "\n(Thm. 14.4)",
     r"$P \geq \mathbb{P}(T(X')\leq T(X) \mid X)$" "\n(Thm. 14.5)"),
    ("What it \"is\"",
     "a conditional expectation\nof a likelihood ratio",
     "a conditional probability\nof a rarer outcome"),
    ("Existence of exact,\npowered version (Thm. 14.6)",
     r"$\mathrm{Conv}(\mathcal{Q})\cap\mathrm{Span}(\mathcal{P})=\varnothing$",
     r"$\mathrm{Conv}(\mathcal{Q})\cap\mathrm{Span}(\mathcal{P})=\varnothing$"
     "\n(same condition, via duality)"),
]

n = len(rows)
col_x = [0.02, 0.40, 0.71]
col_w = [0.36, 0.29, 0.27]
header_h = 0.10
row_h = (0.92 - header_h) / n

# header
ax.add_patch(plt.Rectangle((col_x[1], 1 - header_h), col_w[1], header_h,
                            transform=ax.transAxes, facecolor='C0', alpha=0.85, edgecolor='none'))
ax.add_patch(plt.Rectangle((col_x[2], 1 - header_h), col_w[2], header_h,
                            transform=ax.transAxes, facecolor='C3', alpha=0.85, edgecolor='none'))
ax.text(col_x[1] + col_w[1] / 2, 1 - header_h / 2, 'e-value', transform=ax.transAxes,
        ha='center', va='center', fontsize=13, color='white', fontweight='bold')
ax.text(col_x[2] + col_w[2] / 2, 1 - header_h / 2, 'p-value', transform=ax.transAxes,
        ha='center', va='center', fontsize=13, color='white', fontweight='bold')

for i, (label, ecell, pcell) in enumerate(rows):
    y_top = 1 - header_h - i * row_h
    y_mid = y_top - row_h / 2
    bg = 'white' if i % 2 == 0 else '#f0f0f0'
    for cx, cw in zip(col_x, col_w):
        ax.add_patch(plt.Rectangle((cx, y_top - row_h), cw, row_h,
                                    transform=ax.transAxes, facecolor=bg,
                                    edgecolor='gray', linewidth=0.6))
    ax.text(col_x[0] + 0.01, y_mid, label, transform=ax.transAxes,
            ha='left', va='center', fontsize=10.5, fontweight='bold')
    ax.text(col_x[1] + col_w[1] / 2, y_mid, ecell, transform=ax.transAxes,
            ha='center', va='center', fontsize=10.5)
    ax.text(col_x[2] + col_w[2] / 2, y_mid, pcell, transform=ax.transAxes,
            ha='center', va='center', fontsize=10.5)

ax.set_title('Chapter 14 at a glance: e-values vs. p-values', fontsize=15, pad=14)
fig.savefig('fig2_contrast_panel.pdf')
plt.close(fig)

print("All figures written to figures/ as PDF.")
