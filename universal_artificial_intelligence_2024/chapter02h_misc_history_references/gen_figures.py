"""
Generate all figures needed for chapter02h_misc_history_references_slides.tex

Book: An Introduction to Universal Artificial Intelligence (2024)
Sections covered: 2.8 Miscellaneous (pp. 108-116), 2.9 History and References (pp. 117-121)

Run with:
    conda run -n py313 python3 gen_figures.py
"""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(HERE, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams.update({
    'font.size': 11,
    'axes.spines.top': False,
    'axes.spines.right': False,
})


# ---------------------------------------------------------------------------
# Figure 1: Pinsker's binary inequality.
# Recreates the spirit of book Figure 2.15: shows that
#   RHS - LHS = [p ln(p/q) + (1-p) ln((1-p)/(1-q))] - 2(p-q)^2 >= 0
# for all 0<=p,q<=1.
# ---------------------------------------------------------------------------
def fig_pinsker():
    eps = 1e-6
    p_fixed = 0.3
    q = np.linspace(eps, 1 - eps, 400)

    def kl_term(p, q):
        # p ln(p/q) + (1-p) ln((1-p)/(1-q)), with 0 ln 0 := 0
        t1 = 0.0 if p == 0 else p * np.log(p / q)
        t2 = 0.0 if p == 1 else (1 - p) * np.log((1 - p) / (1 - q))
        return t1 + t2

    rhs = np.array([kl_term(p_fixed, qq) for qq in q])
    lhs = 2 * (p_fixed - q) ** 2

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6))

    ax = axes[0]
    ax.plot(q, rhs, color='#1a5276', lw=2.2,
             label=r'RHS: $p\ln\frac{p}{q}+(1-p)\ln\frac{1-p}{1-q}$')
    ax.plot(q, lhs, color='#c0392b', lw=2.2, ls='--',
             label=r'LHS: $2(p-q)^2$')
    ax.axvline(p_fixed, color='gray', lw=0.8, ls=':')
    ax.set_xlabel(r'$q$')
    ax.set_ylabel('value')
    ax.set_title(rf'Fixed $p={p_fixed}$: KL term vs. squared distance')
    ax.legend(fontsize=8, loc='upper center')
    ax.set_ylim(-0.05, 1.6)

    ax2 = axes[1]
    gap = rhs - lhs
    ax2.plot(q, gap, color='#117a65', lw=2.2)
    ax2.axhline(0, color='black', lw=0.8)
    ax2.fill_between(q, gap, 0, where=(gap >= 0), color='#117a65', alpha=0.15)
    ax2.set_xlabel(r'$q$')
    ax2.set_ylabel('RHS $-$ LHS')
    ax2.set_title('Gap is always non-negative')

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'pinsker.pdf'), bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2: Distance hierarchy diagram (Corollary 2.8.8).
# KL divergence sits at the top and bounds the Square, Absolute, and
# Hellinger^2 distances from above (each via a specific convex f).
# ---------------------------------------------------------------------------
def fig_distance_hierarchy():
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis('off')

    def box(x, y, w, h, text, color='#eef3fb', edge='#1a5276', fs=11):
        b = FancyBboxPatch((x, y), w, h,
                            boxstyle="round,pad=0.08,rounding_size=0.12",
                            linewidth=1.6, edgecolor=edge, facecolor=color)
        ax.add_patch(b)
        ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fs)

    def arrow(x1, y1, x2, y2, label=None, lx=None, ly=None):
        a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>',
                             mutation_scale=14, linewidth=1.4, color='#333333')
        ax.add_patch(a)
        if label:
            ax.text(lx, ly, label, fontsize=9, ha='center', color='#333333')

    # Top node: KL divergence
    box(3.6, 4.7, 2.8, 1.0,
        r'KL divergence' + '\n' + r'$\sum_i y_i \ln\frac{y_i}{z_i}$',
        color='#d6eaf8', edge='#1a5276', fs=11)

    # Three children: Square, Absolute, Hellinger^2
    box(0.3, 2.6, 2.6, 1.1, r'(S) Square' + '\n' + r'$\sum_i (y_i-z_i)^2$', fs=10)
    box(3.7, 2.6, 2.6, 1.1, r'(A) Absolute' + '\n' + r'$\sum_i |y_i-z_i|$', fs=10)
    box(7.1, 2.6, 2.6, 1.1, r'(H) Hellinger$^2$' + '\n' + r'$\sum_i(\sqrt{y_i}-\sqrt{z_i})^2$', fs=10)

    arrow(4.4, 4.7, 1.7, 3.7)
    arrow(5.0, 4.7, 5.0, 3.7)
    arrow(5.6, 4.7, 8.3, 3.7)

    ax.text(5.0, 1.9,
            r'Each arrow: bounded above by KL, via Thm 2.8.6 with $f(x)=x^2,\,|x|,\,(\sqrt{\cdot})^2$',
            ha='center', fontsize=9.5, color='#555555')

    box(3.4, 0.3, 3.2, 1.0,
        'All bounds proved from\none generalized entropy\ninequality (Thm 2.8.6)',
        color='#fdf2e9', edge='#af601a', fs=9.5)
    arrow(5.0, 2.6, 5.0, 1.3)

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'distance_hierarchy.pdf'), bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3: Numeric verification of Corollary 2.8.8 for one example
# probability vector pair (y, z).
# ---------------------------------------------------------------------------
def fig_entropy_verify():
    y = np.array([0.5, 0.3, 0.2])
    z = np.array([0.3, 0.4, 0.3])

    kl = np.sum(y * np.log(y / z))
    square = np.sum((y - z) ** 2)
    absolute = np.sum(np.abs(y - z))
    hellinger = np.sum((np.sqrt(y) - np.sqrt(z)) ** 2)

    bound_square = kl
    bound_absolute = np.sqrt(2 * kl)
    bound_hellinger = kl

    labels = ['(S) Square', '(A) Absolute', r'(H) Hellinger$^2$']
    actual = [square, absolute, hellinger]
    bound = [bound_square, bound_absolute, bound_hellinger]

    x = np.arange(3)
    width = 0.32

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - width / 2, actual, width, label='actual distance', color='#2874a6')
    ax.bar(x + width / 2, bound, width, label='KL-based upper bound', color='#e59866')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('value')
    ax.set_title(r'$y=(0.5,0.3,0.2)$, $z=(0.3,0.4,0.3)$, $\mathrm{KL}(y\|z)=%.4f$' % kl)
    ax.legend(fontsize=9)
    for i, (a, b) in enumerate(zip(actual, bound)):
        ax.text(i - width / 2, a + 0.01, f'{a:.3f}', ha='center', fontsize=8.5)
        ax.text(i + width / 2, b + 0.01, f'{b:.3f}', ha='center', fontsize=8.5)

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'entropy_verify.pdf'), bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 4: Semimeasure defect / "death" diagram.
# Shows how probability mass on a cylinder set Gamma_x can be strictly
# greater than the sum of its two children's cylinder sets -- the missing
# mass "leaks" to the finite string {x} itself (the source is dying / the
# monotone TM halts after outputting exactly x).
# ---------------------------------------------------------------------------
def fig_semimeasure_leak():
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis('off')

    def box(x, y, w, h, text, color='#eef3fb', edge='#1a5276', fs=10.5):
        b = FancyBboxPatch((x, y), w, h,
                            boxstyle="round,pad=0.06,rounding_size=0.1",
                            linewidth=1.5, edgecolor=edge, facecolor=color)
        ax.add_patch(b)
        ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fs)

    def arrow(x1, y1, x2, y2):
        a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>',
                             mutation_scale=13, linewidth=1.3, color='#333333')
        ax.add_patch(a)

    box(3.9, 3.6, 2.2, 0.9, r'$\Gamma_x$' + '\n' + r'mass $\nu(\Gamma_x)$', fs=10.5)

    box(0.6, 1.6, 2.4, 0.9, r'$\Gamma_{x0}$' + '\n' + r'mass $\nu(\Gamma_{x0})$')
    box(3.9, 1.6, 2.4, 0.9, r'$\Gamma_{x1}$' + '\n' + r'mass $\nu(\Gamma_{x1})$')
    box(7.2, 1.6, 2.4, 0.9, r'$\{x\}$: source dies' + '\n' + r'mass $\tilde\nu(\{x\})$',
        color='#f9ebea', edge='#943126')

    arrow(4.6, 3.6, 2.0, 2.5)
    arrow(5.2, 3.6, 5.1, 2.5)
    arrow(5.7, 3.6, 7.9, 2.5)

    ax.text(5.0, 0.7,
            r'$\nu(\Gamma_x) \;=\; \nu(\Gamma_{x0}) + \nu(\Gamma_{x1}) \;+\; (\mathrm{missing\ mass})$'
            '\n'
            'Extending the alphabet with a death symbol (or extending the sample space)\n'
            r'turns this missing mass into an honest probability $\tilde\nu(\{x\})$ of "the world ending".',
            ha='center', fontsize=9.3, color='#444444')

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'semimeasure_leak.pdf'), bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 5: Solomonoff normalization worked numeric example.
# nu(1^n) = 1/2, nu(0^n) = 2^{-n-1}, 0 elsewhere (as in the book, p.114).
# ---------------------------------------------------------------------------
def fig_solomonoff_normalization():
    ns = np.arange(0, 8)
    nu_1n = np.full_like(ns, 0.5, dtype=float)          # nu(Gamma_{1^n}) = 1/2 for all n>=1
    nu_1n[0] = 1.0                                        # nu(Gamma_epsilon) = 1
    nu_0n = 2.0 ** (-(ns.astype(float)) - 1)              # nu(Gamma_{0^n}) = 2^{-n-1}
    nu_0n[0] = 1.0

    fig, ax = plt.subplots(figsize=(7.2, 4))
    ax.plot(ns, nu_1n, 'o-', color='#1a5276', label=r'$\nu(\Gamma_{1^n})=1/2$ (n' + r'$\geq$' + '1)')
    ax.plot(ns, nu_0n, 's-', color='#c0392b', label=r'$\nu(\Gamma_{0^n})=2^{-n-1}$')
    ax.set_xlabel(r'$n$ (length of string of all 1s / all 0s)')
    ax.set_ylabel('semimeasure mass')
    ax.set_title('A semimeasure that is not a measure: mass strictly decreases')
    ax.legend(fontsize=9)
    ax.set_ylim(-0.05, 1.1)

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'solomonoff_norm.pdf'), bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 6: History timeline for Section 2.9.
# ---------------------------------------------------------------------------
def fig_timeline():
    events = [
        (1654, "Pascal & Fermat:\nprobability calculus"),
        (1763, "Bayes' rule\npublished posthumously"),
        (1812, "Laplace: rule of\nsuccession"),
        (1933, "Kolmogorov's axioms\nof probability"),
        (1936, "Turing machine;\nhalting problem"),
        (1948, "Shannon: information\ntheory, entropy"),
        (1952, "Huffman coding"),
        (1964, "Solomonoff: universal\nprior / induction"),
        (1965, "Kolmogorov complexity"),
        (1966, "Chaitin: algorithmic\ninformation, $\\Omega$"),
        (1974, "Levin: coding theorem,\nsymmetry of information"),
        (2005, "Hutter: universal AI\n(AIXI) monograph"),
    ]

    fig, ax = plt.subplots(figsize=(13, 3.4))
    years = [e[0] for e in events]
    ax.hlines(0, min(years) - 5, max(years) + 5, color='#555555', lw=1.6, zorder=1)

    for i, (year, label) in enumerate(events):
        y_off = 0.9 if i % 2 == 0 else -0.9
        ax.plot([year, year], [0, y_off * 0.55], color='#888888', lw=1, zorder=1)
        ax.scatter([year], [0], color='#1a5276', s=45, zorder=2)
        va = 'bottom' if y_off > 0 else 'top'
        ax.text(year, y_off, f'{year}\n{label}', ha='center', va=va, fontsize=8.6)

    ax.set_ylim(-1.6, 1.6)
    ax.set_xlim(min(years) - 15, max(years) + 15)
    ax.axis('off')
    ax.set_title('Selected milestones referenced in Section 2.9', fontsize=12)

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'timeline.pdf'), bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    fig_pinsker()
    fig_distance_hierarchy()
    fig_entropy_verify()
    fig_semimeasure_leak()
    fig_solomonoff_normalization()
    fig_timeline()
    print("All figures generated in", FIG_DIR)
