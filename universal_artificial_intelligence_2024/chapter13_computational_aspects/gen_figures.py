"""
Generate all figures used in chapter13_computational_aspects_slides.tex
(Chapter 13: Computational Aspects, "An Introduction to Universal
Artificial Intelligence", Hutter, Quarel & Catt, CRC Press, 2024)

Run with:
    conda run -n py313 python3 gen_figures.py
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle
import numpy as np
import os

OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(OUTDIR, exist_ok=True)

plt.rcParams['font.size'] = 11
plt.rcParams['font.family'] = 'serif'


# ---------------------------------------------------------------------------
# Figure : Arithmetic-hierarchy map of AIXI and its variants (Section 13.1)
# Drawn as a "ladder" of stacked bands (bottom = most computable), each band
# listing (as bullet lines) which AIXI-family object attains exactly that
# tightest known level. This avoids any Venn-style label overlap.
# ---------------------------------------------------------------------------
def fig_arith_hierarchy():
    bands = [
        (r'$\Delta_1^0$  (computable)', '#2b3a67', '#dbe4f0',
         [r'nothing: no $\pi^*_\xi$, no $\varepsilon$-optimal $\pi^*_\xi$ for any '
          r'semimeasure $\xi$ (Thm 13.1.7, 13.1.8)']),
        (r'$\Delta_2^0$  (approximable)', '#1f6b2b', '#d7f0d3',
         [r'$\xi_U = M \in \Sigma_1^0 \subset \Delta_2^0$  (Solomonoff mixture)',
          r'$\pi^{\varepsilon}_M$  ($\varepsilon$-optimal AIXI policy, Cor. 13.1.6)',
          r'$\varepsilon$-optimal knowledge-/entropy-seeking policy (Thm 13.1.9)']),
        (r'$\Delta_3^0$', '#8a5a1f', '#f0ded3',
         [r'$\pi^*_M$  (optimal AIXI policy, Cor. 13.1.6)',
          r'BayesExp  (Thm 13.1.10)',
          r'optimal knowledge-/entropy-seeking policy (Thm 13.1.9)']),
        (r'$\Delta_4^0$', '#7a1f1f', '#f3d3d3',
         [r'$\pi^*_M$ for infinite-horizon AIXI (after the $m\to\infty$ limit)']),
    ]

    heights = [0.42 + 0.5*len(items) for _, _, _, items in bands]
    gap = 0.45
    y = 0.3
    centers = []
    for h in heights:
        centers.append(y + h/2)
        y += h + gap
    top = y - gap + 0.65

    fig, ax = plt.subplots(figsize=(9.4, 7.4))
    ax.set_xlim(-0.3, 10.7)
    ax.set_ylim(0, top)
    ax.axis('off')

    for (label, edge, face, items), yc, h in zip(bands, centers, heights):
        box = FancyBboxPatch((0, yc - h/2), 10.4, h,
                              boxstyle="round,pad=0.05,rounding_size=0.14",
                              linewidth=1.9, edgecolor=edge, facecolor=face, zorder=2)
        ax.add_patch(box)
        ax.text(0.25, yc + h/2 - 0.28, label, ha='left', va='top', fontsize=12.5,
                 fontweight='bold', color=edge, zorder=3)
        y0 = yc + h/2 - 0.75
        for item in items:
            ax.text(0.55, y0, u"• " + item, ha='left', va='center', fontsize=9.6,
                     color='#222222', zorder=3)
            y0 -= 0.5

    # upward arrows between consecutive bands
    for i in range(len(bands) - 1):
        y_from = centers[i] + heights[i]/2
        y_to = centers[i+1] - heights[i+1]/2
        ax.annotate("", xy=(10.15, y_to - 0.03), xytext=(10.15, y_from + 0.03),
                    arrowprops=dict(arrowstyle='-|>', lw=1.8, color='#555555'))
    ax.text(10.55, (centers[0] + centers[-1]) / 2, "strictly harder\n(one more limit /\nquantifier)",
            ha='left', va='center', fontsize=8.3, style='italic', color='#555555', rotation=90)

    ax.text(5.2, top - 0.3,
            "Where AIXI and its variants sit in the arithmetic hierarchy (Section 13.1)",
            ha='center', va='top', fontsize=10.8, style='italic', color='#333333')

    plt.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "arith_hierarchy.pdf"), bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure : Finite-horizon value V*,m converging to V* (proof sketch Thm 13.1.1)
# ---------------------------------------------------------------------------
def fig_value_convergence():
    # Matches the worked numeric example in the "Python: Estimating V*_nu"
    # slide: a constant per-step reward r_bar in a geometrically-discounted
    # environment, so that V*_nu(h<t) = r_bar exactly and the finite-depth
    # value V*,m_nu(h<t) rises monotonically towards it, respecting the
    # proven bound 0 <= V*-V*,m <= Gamma_{m+1}/Gamma_t from Lemma 6.7.6.
    gamma = 0.9
    t = 1
    r_bar = 0.7
    m_list = np.arange(1, 41)

    def Gamma(k):
        # Gamma_k = sum_{j=k}^inf gamma^j  (geometric discount normalizer)
        return gamma**k / (1 - gamma)

    Gt = Gamma(t)
    bound = Gamma(m_list + 1) / Gt
    Vstar = r_bar

    # exact finite-depth value: truncate the discounted sum at depth m
    Vm = np.array([sum(gamma**k * r_bar for k in range(t, m + 1)) / Gt for m in m_list])

    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    ax.axhline(Vstar, color='#2b3a67', lw=1.8, ls='--', label=r'$V_\nu^*(h_{<t})$ (true value)')
    ax.plot(m_list, Vm, 'o-', color='#7a1f1f', ms=4.5, lw=1.4,
             label=r'$V_\nu^{*,m}(h_{<t})$ (finite-depth approx.)')
    ax.fill_between(m_list, Vstar - bound, Vstar, color='#dbe4f0', alpha=0.6,
                     label=r'proven band $[V^*-\Gamma_{m+1}/\Gamma_t,\,V^*]$')
    ax.set_xlabel(r'recursion depth $m$  (how many future steps are unrolled)')
    ax.set_ylabel('value')
    ax.set_title('Estimability of ' r'$V_\nu^*$' ': the finite-depth approximation gap is bounded\n'
                 r'by $\Gamma_{m+1}/\Gamma_t \to 0$'
                 '   (numeric illustration, ' r'$\gamma_k=0.9^k$' ', t=1)', fontsize=10)
    ax.legend(fontsize=8.6, loc='lower right')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "value_convergence.pdf"), bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure : AIXItl -- setup phase + interaction loop (Algorithm 13.1)
# ---------------------------------------------------------------------------
def fig_aixitl_loop():
    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6.4)
    ax.axis('off')

    def box(x, y, w, h, text, edge, face, fontsize=9.6):
        b = FancyBboxPatch((x - w/2, y - h/2), w, h, boxstyle="round,pad=0.08,rounding_size=0.14",
                            linewidth=1.8, edgecolor=edge, facecolor=face, zorder=3)
        ax.add_patch(b)
        ax.text(x, y, text, ha='center', va='center', fontsize=fontsize, fontweight='bold',
                color=edge, zorder=4)

    box(2.0, 5.4, 3.7, 1.55,
        r"Setup ($O(l_P^2\,2^{l_P})$ time)" "\ncheck every proof $b\\in\\mathbb{B}^{l_P}$;"
        "\nkeep valid $\\dot{\\pi}$ with $\\ell(\\dot{\\pi})\\leq\\tilde{l}$"
        "\n$\\Rightarrow$ set $\\Pi_{\\mathrm{VA}}$",
        '#2b3a67', '#dbe4f0', fontsize=9.0)

    box(7.6, 5.4, 3.6, 1.55,
        r"Truncate: force $\dot{\pi}$ to output"
        "\n$(a,0)$ if it has not halted"
        "\nby time $\\tilde{t}$ (ensures $t(\\dot{\\pi})\\leq\\tilde{t}$)",
        '#6b3a2b', '#f0ded3', fontsize=9.0)

    ax.annotate("", xy=(5.85, 5.4), xytext=(3.85, 5.4),
                arrowprops=dict(arrowstyle='-|>', lw=1.9, color='black'))

    box(4.85, 3.15, 6.4, 1.35,
        r"For cycle $k=1,2,3,\dots$: run every $\dot{\pi}\in\Pi_{\mathrm{VA}}$ on $h_{<k}$,"
        "\nobtaining $(a_k^{\\dot{\\pi}}, v_k^{\\dot{\\pi}})=\\dot{\\pi}(h_{<k})$",
        '#1f6b2b', '#d7f0d3', fontsize=9.4)

    ax.annotate("", xy=(4.85, 3.85), xytext=(4.85, 4.6),
                arrowprops=dict(arrowstyle='-|>', lw=1.9, color='black'))

    box(4.85, 1.15, 6.4, 1.35,
        r"Pick $\dot{\pi}^*=\arg\max_{\dot{\pi}\in\Pi_{\mathrm{VA}}} v_k^{\dot{\pi}}$;"
        " act $a_k=a_k^{\\dot{\\pi}^*}$; observe $e_k$",
        '#7a1f1f', '#f3d3d3', fontsize=9.4)

    ax.annotate("", xy=(4.85, 1.85), xytext=(4.85, 2.45),
                arrowprops=dict(arrowstyle='-|>', lw=1.9, color='black'))

    ax.annotate("", xy=(8.3, 3.15), xytext=(8.3, 1.15),
                arrowprops=dict(arrowstyle='-', lw=1.6, color='#444444', ls='dashed',
                                 connectionstyle='arc3,rad=-0.6'))
    ax.text(9.55, 2.15, r"loop", fontsize=9, color='#444444', rotation=90, va='center')

    ax.text(4.85, 0.15, r"Per-cycle cost $O(t_{\rm cycle})=O(2^{\tilde{l}}\cdot\tilde{t})$"
                        r" — independent of the number of elapsed cycles $k$",
            ha='center', fontsize=9.0, style='italic', color='#444444')

    plt.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "aixitl_loop.pdf"), bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure : cost of exhaustive proof/program search vs. bound length
# ---------------------------------------------------------------------------
def fig_search_cost():
    lP = np.arange(2, 21)
    setup_cost = lP**2 * 2.0**lP

    ltilde = np.arange(2, 21)
    ttilde = 50
    cycle_cost = 2.0**ltilde * ttilde

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0))

    axes[0].semilogy(lP, setup_cost, 'o-', color='#7a1f1f', ms=4.5)
    axes[0].set_xlabel(r'max proof length $l_P$')
    axes[0].set_ylabel(r'setup time $O(l_P^2\,2^{l_P})$')
    axes[0].set_title('Setup phase cost\n(checking all candidate proofs)', fontsize=10)
    axes[0].grid(alpha=0.3)

    axes[1].semilogy(ltilde, cycle_cost, 's-', color='#2b3a67', ms=4.5)
    axes[1].set_xlabel(r'max policy length $\tilde{l}$')
    axes[1].set_ylabel(r'per-cycle time $O(2^{\tilde{l}}\cdot\tilde{t})$, $\tilde{t}=50$')
    axes[1].set_title('Interaction-loop cost per cycle\n(running every $\\dot{\\pi}\\in\\Pi_{\\mathrm{VA}}$)', fontsize=10)
    axes[1].grid(alpha=0.3)

    fig.suptitle(r'Why AIXI$tl$ is optimal but utterly impractical', fontsize=11)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "search_cost.pdf"), bbox_inches='tight')
    plt.close(fig)


if __name__ == "__main__":
    fig_arith_hierarchy()
    fig_value_convergence()
    fig_aixitl_loop()
    fig_search_cost()
    print("All figures written to", OUTDIR)
