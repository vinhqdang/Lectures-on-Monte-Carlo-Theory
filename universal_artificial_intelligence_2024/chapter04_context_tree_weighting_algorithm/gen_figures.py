"""
Generate all figures for Chapter 4 (The Context Tree Weighting Algorithm) slides.

Two kinds of figures are produced:
  1. Diagrams / real-experiment plots that are hard to reproduce exactly (Markov-chain
     state diagrams, tree diagrams, and the actual experimental accuracy/KL-divergence
     curves from the book) -- these are cropped directly out of the book PDF with
     PyMuPDF at high resolution.
  2. A couple of small illustrative/explanatory plots that we *can* plot cleanly with
     matplotlib (kept simple, used only to support the worked Python examples).

Run with:
    conda run -n py313 python3 gen_figures.py
"""
import os
import fitz  # PyMuPDF
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.join(HERE, 'figures')
os.makedirs(FIGDIR, exist_ok=True)

BOOK_PDF = os.path.join(os.path.dirname(HERE),
                         'An Introduction to Universal Artificial Intelligence 2024.pdf')

# ---------------------------------------------------------------------------
# Part 1: crops from the book PDF (diagrams & real experimental figures)
# ---------------------------------------------------------------------------
doc = fitz.open(BOOK_PDF)

# Each entry: output_name -> (0-indexed PDF page, fitz.Rect in PDF points)
CROPS = {
    'fig4_2_3_letters':    (182, fitz.Rect(60, 298, 465, 432)),
    'fig4_4_markov':        (186, fitz.Rect(193, 60, 340, 168)),
    'fig4_5_kmarkov_exp':   (186, fitz.Rect(60, 230, 466, 382)),
    'fig4_6_context_tree':  (188, fitz.Rect(178, 222, 352, 295)),
    'fig4_7_pst_tree':      (192, fitz.Rect(143, 64.8, 382.9, 144.6)),
    'fig4_8_encodings':     (194, fitz.Rect(55, 58, 466, 280)),
    'fig4_9_pst_good':      (198, fitz.Rect(150, 46, 377, 122)),
    'fig4_10_pst_vs_k':     (201, fitz.Rect(38, 58, 444, 212)),
    'fig4_11a_ctw_init':    (207, fitz.Rect(38, 66, 432, 256)),
    'fig4_11b_ctw_step1':   (207, fitz.Rect(38, 260, 432, 462)),
    'fig4_11c_ctw_step2':   (207, fitz.Rect(38, 460, 432, 662)),
    'fig4_12_ctw_exp':      (211, fitz.Rect(38, 348, 444, 500)),
    'fig4_13a_online_init': (215, fitz.Rect(58, 66, 432, 174)),
    'fig4_13b_online_step1':(215, fitz.Rect(58, 200, 432, 392)),
    'fig4_13c_online_step2':(215, fitz.Rect(38, 420, 432, 610)),
}

for name, (page_idx, rect) in CROPS.items():
    page = doc[page_idx]
    pix = page.get_pixmap(matrix=fitz.Matrix(4, 4), clip=rect)
    outpath = os.path.join(FIGDIR, name + '.png')
    pix.save(outpath)
    print('Saved crop', outpath)

doc.close()

# ---------------------------------------------------------------------------
# Part 2: small illustrative matplotlib plots for the Python worked examples
# ---------------------------------------------------------------------------

plt.rcParams.update({'font.size': 11})


def kt_predict_1(a, b):
    """KT estimator: P_KT(next=1 | a zeros, b ones seen so far)."""
    return (b + 0.5) / (a + b + 1)


# --- Fig: KT estimator convergence on a Bernoulli(theta) source -----------
rng = np.random.default_rng(7)
theta_true = 0.7
n = 400
bits = (rng.random(n) < theta_true).astype(int)
a = 0
b = 0
estimates = []
for x in bits:
    estimates.append(kt_predict_1(a, b))
    if x == 1:
        b += 1
    else:
        a += 1

fig, ax = plt.subplots(figsize=(6.5, 3.6))
ax.plot(estimates, color='#3b6ea5', lw=1.6, label=r'$\mathrm{P}_{\mathrm{KT}}(x_{t+1}=1\,|\,x_{1:t})$')
ax.axhline(theta_true, color='#c0392b', ls='--', lw=1.3, label=r'true $\theta=0.7$')
ax.set_xlabel('number of observed bits $t$')
ax.set_ylabel('estimated probability of next bit $=1$')
ax.set_title('KT estimator converging to the true Bernoulli parameter')
ax.legend(loc='lower right', fontsize=9)
ax.set_ylim(0, 1)
fig.tight_layout()
fig.savefig(os.path.join(FIGDIR, 'py_kt_convergence.pdf'))
plt.close(fig)

# --- Fig: KT probability table heatmap-ish bar illustration (a=0..3,b=0..3) ---
avals = np.arange(0, 6)
bvals = np.arange(0, 6)


def log_gamma(x):
    from math import lgamma
    return lgamma(x)


def p_kt(a, b):
    from math import exp, pi, lgamma
    return (1.0 / pi) * exp(lgamma(a + 0.5) + lgamma(b + 0.5) - lgamma(a + b + 1))


table = np.zeros((len(avals), len(bvals)))
for i, a in enumerate(avals):
    for j, b in enumerate(bvals):
        table[i, j] = p_kt(a, b)

fig, ax = plt.subplots(figsize=(5.2, 4.4))
im = ax.imshow(table, cmap='viridis', origin='upper')
ax.set_xticks(range(len(bvals)))
ax.set_xticklabels(bvals)
ax.set_yticks(range(len(avals)))
ax.set_yticklabels(avals)
ax.set_xlabel('$b$ (number of ones)')
ax.set_ylabel('$a$ (number of zeros)')
ax.set_title(r'$\mathrm{P}_{\mathrm{KT}}(a,b)$')
for i in range(len(avals)):
    for j in range(len(bvals)):
        ax.text(j, i, f'{table[i, j]:.3f}', ha='center', va='center',
                 color='white' if table[i, j] < table.max() * 0.6 else 'black', fontsize=8)
fig.colorbar(im, ax=ax, shrink=0.8)
fig.tight_layout()
fig.savefig(os.path.join(FIGDIR, 'py_kt_table.pdf'))
plt.close(fig)

# --- Fig: CTW mixture weight over 5 depth-2 suffix-set trees, illustrative ---
labels = [r'$\square$\n(empty)', r'$\{0,1\}$', r'$\{1,10,00\}$',
          r'$\{11,01,0\}$', r'$\{11,10,01,00\}$']
weights = [2**-1, 2**-3, 2**-3, 2**-3, 2**-3]
fig, ax = plt.subplots(figsize=(6.5, 3.2))
bars = ax.bar(range(5), weights, color='#4c72b0', edgecolor='black')
ax.set_xticks(range(5))
ax.set_xticklabels(['empty tree', r'$\{0,1\}$', r'$\{1,10,00\}$',
                     r'$\{11,01,0\}$', r'$\{11,10,01,00\}$'], rotation=20, ha='right', fontsize=9)
ax.set_ylabel(r'prior weight $w_S = 2^{-\Gamma_D(S)}$')
ax.set_title(r'CTW prior weight on each suffix set in $C_2$')
for rect, w in zip(bars, weights):
    ax.text(rect.get_x() + rect.get_width() / 2, w + 0.01, f'{w:.3f}', ha='center', fontsize=9)
ax.set_ylim(0, 0.6)
fig.tight_layout()
fig.savefig(os.path.join(FIGDIR, 'py_ctw_prior_weights.pdf'))
plt.close(fig)

print('All figures generated in', FIGDIR)
