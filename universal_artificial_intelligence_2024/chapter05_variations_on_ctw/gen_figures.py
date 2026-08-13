"""
Figure generator for Chapter 5 -- Variations on CTW
(An Introduction to Universal Artificial Intelligence, Hutter/Quarel/Catt 2024)

Run with:
    conda run -n py313 python3 gen_figures.py

All figures are saved as PDF (or PNG for the cropped book figure) into ./figures/
"""
import os
import io
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

HERE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.join(HERE, 'figures')
os.makedirs(FIGDIR, exist_ok=True)

BOOK_PDF = os.path.join(HERE, '..', 'An Introduction to Universal Artificial Intelligence 2024.pdf')

plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'legend.fontsize': 10,
    'figure.dpi': 150,
})

# ============================================================================
# Figure 0: crop Figure 5.2 (partition trees in B_2) from the book PDF
# ============================================================================
def fig_partition_trees_crop():
    try:
        import fitz
    except ImportError:
        print("pymupdf not available -- skipping book-page crop for partition trees")
        return
    doc = fitz.open(BOOK_PDF)
    page = doc[227]  # book PDF page 228 (1-indexed) = printed page 207, Figure 5.2
    mat = fitz.Matrix(4, 4)
    pix = page.get_pixmap(matrix=mat)
    from PIL import Image
    img = Image.open(io.BytesIO(pix.tobytes('png')))
    crop = img.crop((100, 2080, 1900, 2410))
    crop.save(os.path.join(FIGDIR, 'partition_trees_b2.png'))
    print("Saved partition_trees_b2.png")


# ============================================================================
# CTW engine (sparse, dictionary-based context tree) used for Figure 5.1 redo
# ============================================================================
class CTWModel:
    """A binary Context Tree Weighting model of fixed depth D, updated online.

    Nodes are indexed by a tuple of the k most recent bits (most-recent first),
    0 <= k <= D. Only visited nodes are stored (sparse tree), matching the
    book's O(n|D|) node bound. logPe/logPw are natural logarithms.
    """
    def __init__(self, D):
        self.D = D
        # node -> [a, b, logPe, logPw]
        self.nodes = {(): [0, 0, 0.0, 0.0]}
        self.history = []

    def _get(self, ctx):
        if ctx not in self.nodes:
            self.nodes[ctx] = [0, 0, 0.0, 0.0]
        return self.nodes[ctx]

    def update(self, bit):
        H = self.history
        t1 = len(H)  # number of symbols already observed (0-indexed length)
        L = min(self.D, t1)
        # contexts of length 0..L, most-recent-first tuples
        ctxs = [tuple(reversed(H[t1 - k:t1])) for k in range(L + 1)]
        # --- update counts + Pe bottom-up, deepest first ---
        for k in range(L, -1, -1):
            node = self._get(ctxs[k])
            a, b, logPe, logPw = node
            if bit == 0:
                factor = (a + 0.5) / (a + b + 1)
                a += 1
            else:
                factor = (b + 0.5) / (a + b + 1)
                b += 1
            logPe_new = logPe + math.log(factor)
            if k == self.D:
                logPw_new = logPe_new
            elif k == L:
                # deepest reached this round but not a true depth-D leaf:
                # both children are untouched (default logPw = 0 i.e. Pw = 1)
                logPw_new = math.log(0.5) + np.logaddexp(logPe_new, 0.0)
            else:
                # matching child is ctxs[k+1], already updated this round
                child_new = self.nodes[ctxs[k + 1]][3]
                # sibling child: same context but the OTHER bit at depth k+1
                other_bit = 1 - ctxs[k + 1][0]
                sib_ctx = (other_bit,) + ctxs[k + 1][1:]
                sib_old = self.nodes.get(sib_ctx, [0, 0, 0.0, 0.0])[3]
                logPw_new = math.log(0.5) + np.logaddexp(logPe_new, child_new + sib_old)
            self.nodes[ctxs[k]] = [a, b, logPe_new, logPw_new]
        self.history.append(bit)
        return self.nodes[()][3]  # new log Pw at root


def instantaneous_kl(bits, D):
    """Return array of -ln P(x_t|x_<t) under a depth-D CTW model, for t=1..n."""
    model = CTWModel(D)
    kl = np.zeros(len(bits))
    logPw_prev = 0.0
    for t, bit in enumerate(bits):
        logPw_new = model.update(bit)
        kl[t] = -(logPw_new - logPw_prev)
        logPw_prev = logPw_new
    return kl


def instantaneous_kl_switch(bits, D, block):
    """Switch model: a fresh depth-D CTW model for every contiguous block of
    length `block`; instantaneous KL is measured within each block only."""
    n = len(bits)
    kl = np.zeros(n)
    for start in range(0, n, block):
        seg = bits[start:start + block]
        kl[start:start + len(seg)] = instantaneous_kl(seg, D)
    return kl


def fig_kl_divergence():
    """Recreate Figure 5.1: instantaneous KL divergence for CTW models
    D=1, D=2, D=200 vs. a switching model of four independent depth-1 CTW
    models, on the piecewise 1-Markov sequence of Example 5.2.1."""
    block = 200
    n_blocks = 4
    seq01 = ([0, 1] * (block // 2))
    seq11 = ([1] * block)
    bits = []
    for i in range(n_blocks):
        bits += seq01 if i % 2 == 0 else seq11
    bits = np.array(bits)
    n = len(bits)

    kl_d1 = instantaneous_kl(bits, 1)
    kl_d2 = instantaneous_kl(bits, 2)
    kl_d200 = instantaneous_kl(bits, 200)
    kl_switch = instantaneous_kl_switch(bits, 1, block)

    fig, axes = plt.subplots(2, 2, figsize=(9.5, 6.5), sharex=True)
    t_axis = np.arange(1, n + 1)
    panels = [
        (axes[0, 0], kl_d1, r'CTW $D=1$', '#1f6f8b'),
        (axes[0, 1], kl_d2, r'CTW $D=2$', '#1f6f8b'),
        (axes[1, 0], kl_d200, r'CTW $D=200$', '#1f6f8b'),
        (axes[1, 1], kl_switch, r'Switch (four $D=1$ models)', '#b5442d'),
    ]
    for ax, y, label, color in panels:
        y_plot = np.clip(y, 1e-3, None)
        ax.plot(t_axis, y_plot, color=color, linewidth=1.0, label=label)
        for b in range(block, n, block):
            ax.axvline(b, color='gray', linestyle=':', linewidth=0.8)
        ax.set_yscale('log')
        ax.set_ylim(1e-3, 2)
        ax.legend(loc='upper right', framealpha=0.9)
        ax.grid(alpha=0.25)
    axes[1, 0].set_xlabel('bits')
    axes[1, 1].set_xlabel('bits')
    axes[0, 0].set_ylabel('KL divergence')
    axes[1, 0].set_ylabel('KL divergence')
    fig.suptitle('Instantaneous KL Divergence (Example 5.2.1)')
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(FIGDIR, 'kl_divergence.pdf'))
    plt.close(fig)
    print("Saved kl_divergence.pdf")


# ============================================================================
# Figure: discount schedules for Adaptive CTW (Section 5.1)
# ============================================================================
def fig_discount_schedules():
    t = np.arange(1, 400)
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.plot(t, np.full_like(t, 0.05, dtype=float), label=r'constant, $\gamma=0.05$', lw=1.8)
    for alpha, c in [(0.33, 0.5), (0.5, 1.0), (1.0, 2.0)]:
        gamma = c * t**(-alpha)
        gamma = np.clip(gamma, 0, 1)
        ax.plot(t, gamma, label=rf'seq-length, $c={c}, \alpha={alpha}$', lw=1.5)
    ax.set_xlabel(r'time step $t$')
    ax.set_ylabel(r'discount $\gamma_t$')
    ax.set_title('Discount schedules for Adaptive CTW')
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, 'discount_schedules.pdf'))
    plt.close(fig)
    print("Saved discount_schedules.pdf")


# ============================================================================
# Figure: PTW recursive binary split of [1,n] (Algorithm 5.3 intuition)
# ============================================================================
def fig_ptw_recursion():
    fig, ax = plt.subplots(figsize=(9, 3.6))
    depths = [(0, 0, 16, 1), ]
    levels = [
        [(0, 16)],
        [(0, 8), (8, 16)],
        [(0, 4), (4, 8), (8, 12), (12, 16)],
        [(0, 2), (2, 4), (4, 6), (6, 8), (8, 10), (10, 12), (12, 14), (14, 16)],
    ]
    colors = ['#dbe9f4', '#bcd7ea', '#9cc5e0', '#7bb3d6']
    for row, segs in enumerate(levels):
        y = 3 - row
        for (a, b) in segs:
            rect = patches.Rectangle((a, y - 0.4), b - a, 0.8,
                                      facecolor=colors[row], edgecolor='black', linewidth=1.0)
            ax.add_patch(rect)
            ax.text((a + b) / 2, y, f'[{a+1},{b}]', ha='center', va='center', fontsize=8)
        ax.text(-1.3, y, f'depth {row}', ha='right', va='center', fontsize=9)
    ax.set_xlim(-3, 17)
    ax.set_ylim(-0.6, 3.6)
    ax.axis('off')
    ax.set_title(r'PTW: recursive halving of $x_{1:16}$  ($k=2^{D-1}$ each level)')
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, 'ptw_recursion.pdf'))
    plt.close(fig)
    print("Saved ptw_recursion.pdf")


# ============================================================================
# Figure: FMN hierarchical model pool over a string (Section 5.4 intuition)
# ============================================================================
def fig_fmn_pool():
    fig, ax = plt.subplots(figsize=(9, 3.8))
    levels = [
        [(0, 8, 'depth 0: whole string, model pool $\\mathcal{M}_1$')],
        [(0, 4, ''), (4, 8, '')],
        [(0, 2, ''), (2, 4, ''), (4, 6, ''), (6, 8, '')],
        [(0, 1, ''), (1, 2, ''), (2, 3, ''), (3, 4, ''),
         (4, 5, ''), (5, 6, ''), (6, 7, ''), (7, 8, '')],
    ]
    colors = ['#f4e3d7', '#eccdb4', '#e0af86', '#d3915c']
    for row, segs in enumerate(levels):
        y = 3 - row
        for (a, b, lbl) in segs:
            rect = patches.Rectangle((a, y - 0.4), b - a, 0.8,
                                      facecolor=colors[row], edgecolor='black', linewidth=1.0)
            ax.add_patch(rect)
            ax.text((a + b) / 2, y, rf'$x_{{{a+1}:{b}}}$', ha='center', va='center', fontsize=8)
        if lbl:
            ax.text(9.2, y, lbl, ha='left', va='center', fontsize=8.5)
    ax.text(9.2, 2, r'$\xi(x_{a:b})$ = Bayes mixture over pool $\mathcal{M}_a$', ha='left', va='center', fontsize=8.5)
    ax.text(9.2, 0, r'each leaf: $\rho(\cdot\,|\,x_{a:b})$ (base measure, adaptive)', ha='left', va='center', fontsize=8.5)
    ax.set_xlim(-0.5, 20)
    ax.set_ylim(-0.6, 3.6)
    ax.axis('off')
    ax.set_title('Forget-Me-Not: hierarchical mixture over segmentations of $x_{1:n}$')
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, 'fmn_pool.pdf'))
    plt.close(fig)
    print("Saved fmn_pool.pdf")


if __name__ == '__main__':
    fig_partition_trees_crop()
    fig_kl_divergence()
    fig_discount_schedules()
    fig_ptw_recursion()
    fig_fmn_pool()
    print("All figures generated in", FIGDIR)
