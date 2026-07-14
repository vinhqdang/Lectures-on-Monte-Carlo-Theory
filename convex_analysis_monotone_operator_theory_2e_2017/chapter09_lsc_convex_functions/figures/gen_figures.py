#!/usr/bin/env python3
"""
Generate figures for Chapter 9: Lower Semicontinuous Convex Functions
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

# Figure 1: Lower Semicontinuous Convex Function
fig, ax = plt.subplots(figsize=(10, 6))
x = np.linspace(-2, 5, 500)
y = np.where(x < 0, x**2 + 2, np.where(x < 2, -0.5*x + 2, 0.25*x**2 - 1))
ax.plot(x, y, 'b-', linewidth=2.5, label='$f(x)$ (lsc convex)')
ax.plot([2], [0.5], 'bo', markersize=8, label='Point with jump discontinuity')
ax.axhline(y=0, color='k', linewidth=0.5)
ax.axvline(x=0, color='k', linewidth=0.5)
ax.grid(True, alpha=0.3)
ax.set_xlabel('$x$', fontsize=12)
ax.set_ylabel('$f(x)$', fontsize=12)
ax.set_title('Example: Lower Semicontinuous Convex Function', fontsize=13, fontweight='bold')
ax.legend(fontsize=11, loc='upper right')
ax.set_ylim([-2, 6])
plt.tight_layout()
plt.savefig('lsc_convex_function.pdf', dpi=150, bbox_inches='tight')
plt.close()
print("Created: lsc_convex_function.pdf")

# Figure 2: Epigraph Illustration
fig, ax = plt.subplots(figsize=(10, 7))
x = np.linspace(-1, 4, 300)
f = 0.25*x**2 - 0.5*x + 0.5
ax.fill_between(x, f, 5, alpha=0.3, color='lightblue', label='epi $f$')
ax.plot(x, f, 'b-', linewidth=3, label='Graph of $f$')
x_pt = 2
y_pt = 0.25*x_pt**2 - 0.5*x_pt + 0.5
ax.plot([x_pt], [y_pt], 'ro', markersize=8, label='Point $(x, f(x))$')
ax.axvline(x=x_pt, color='gray', linestyle='--', linewidth=1, alpha=0.7)
ax.annotate('$(x, f(x))$', xy=(x_pt, y_pt), xytext=(x_pt+0.5, y_pt-0.3),
            fontsize=11, arrowprops=dict(arrowstyle='->', color='red', lw=1.5))
ax.set_xlim([-1, 4])
ax.set_ylim([-0.5, 5])
ax.set_xlabel('$x$', fontsize=12)
ax.set_ylabel('$y$', fontsize=12)
ax.set_title('Epigraph of a Convex Function: $\\mathrm{epi} f = \\{(x, y) : y \\geq f(x)\\}$',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=11, loc='upper left')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('epigraph.pdf', dpi=150, bbox_inches='tight')
plt.close()
print("Created: epigraph.pdf")

# Figure 3: LSC Envelope Example
fig, ax = plt.subplots(figsize=(10, 7))
x = np.linspace(-2, 3, 500)
f_original = np.where(x < 1, 0.5*x + 1.5, np.where(x < 1.5, 4, 0.3*x**2 + 0.5))
f_lsc = np.where(x < 1, 0.5*x + 1.5, np.where(x < 1.5, 0.5*x + 1.5, 0.3*x**2 + 0.5))
ax.plot(x, f_original, 'r--', linewidth=2.5, label='Original function $f$')
ax.plot(x, f_lsc, 'b-', linewidth=2.5, label='LSC envelope $\\bar{f}$')
x_disc = 1.5
y_disc = 4
ax.plot([x_disc], [y_disc], 'ro', markersize=8)
ax.plot([x_disc], [0.5*x_disc + 1.5], 'bs', markersize=8)
ax.set_xlim([-2, 3])
ax.set_ylim([0, 5])
ax.set_xlabel('$x$', fontsize=12)
ax.set_ylabel('$y$', fontsize=12)
ax.set_title('Lower Semicontinuous Envelope: $\\bar{f} = \\sup\\{g \\in \\Gamma_0(\\mathcal{H}) : g \\leq f\\}$',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=11, loc='upper left')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('lsc_envelope.pdf', dpi=150, bbox_inches='tight')
plt.close()
print("Created: lsc_envelope.pdf")

# Figure 4: Affine Minorant
fig, ax = plt.subplots(figsize=(10, 7))
x = np.linspace(-1, 4, 500)
f = 0.4*x**2 - 0.5*x + 0.8
x0 = 1.5
f_x0 = 0.4*x0**2 - 0.5*x0 + 0.8
slope = 0.8*x0 - 0.5
g = slope * (x - x0) + f_x0
ax.plot(x, f, 'b-', linewidth=3, label='$f(x) = 0.4x^2 - 0.5x + 0.8$')
ax.plot(x, g, 'g--', linewidth=2.5, label=f'Affine minorant: $g(x) = {slope:.1f}(x - {x0}) + f({x0})$')
ax.plot([x0], [f_x0], 'ro', markersize=9, label=f'Point of tangency at $x_0 = {x0}$')
y_max = 4
ax.fill_between(x, f, y_max, where=(f >= g), alpha=0.2, color='blue')
ax.set_xlim([-1, 4])
ax.set_ylim([-0.5, y_max])
ax.set_xlabel('$x$', fontsize=12)
ax.set_ylabel('$y$', fontsize=12)
ax.set_title('Affine Minorization: Every proper lsc convex function has an affine minorant',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=10, loc='upper left')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('affine_minorant.pdf', dpi=150, bbox_inches='tight')
plt.close()
print("Created: affine_minorant.pdf")

# Figure 5: Recession Function
fig, ax = plt.subplots(figsize=(10, 7))
x = np.linspace(-1, 6, 500)
f = 0.5*x**2 - x + 1
f = np.maximum(f, 0)
rec_f = np.where(x > 0.01, x**2, 0)
ax.plot(x, f, 'b-', linewidth=3, label='Proper convex function $f$')
ax.plot(x, rec_f, 'r--', linewidth=2.5, label='Recession function $\\mathrm{rec} f$')
x0_vals = [0.5, 1.5, 2.5]
for x0 in x0_vals:
    y0 = 0.5*x0**2 - x0 + 1
    ax.plot([x0], [y0], 'bo', markersize=6, alpha=0.6)
ax.set_xlim([-1, 6])
ax.set_ylim([-0.5, 8])
ax.set_xlabel('$y$', fontsize=12)
ax.set_ylabel('Value', fontsize=12)
ax.set_title('Recession Function: $\\mathrm{rec} f(y) = \\lim_{\\alpha \\to \\infty} \\frac{f(x+\\alpha y) - f(x)}{\\alpha}$',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=11, loc='upper left')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('recession_function.pdf', dpi=150, bbox_inches='tight')
plt.close()
print("Created: recession_function.pdf")

# Figure 6: Examples of Functions in Gamma_0(R)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
x = np.linspace(-3, 3, 500)
p_vals = [1, 1.5, 2]
for p, color in zip(p_vals, colors[:3]):
    y = np.abs(x)**p
    ax1.plot(x, y, linewidth=2.5, label=f'$|x|^{p}$', color=color)
ax1.set_xlim([-3, 3])
ax1.set_ylim([0, 5])
ax1.set_xlabel('$x$', fontsize=12)
ax1.set_ylabel('$f(x)$', fontsize=12)
ax1.set_title('Strictly Convex Functions in $\\Gamma_0(\\mathbb{R})$: $p$-norms', fontsize=12, fontweight='bold')
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)

x = np.linspace(0.01, 3, 500)
entropy = x * np.log(x) - x
neg_entropy = -x * np.log(x) + x
ax2.plot(x, entropy, linewidth=2.5, label='$x \\ln(x) - x$ (entropy)', color=colors[0])
ax2.plot(x, neg_entropy, linewidth=2.5, label='$-x \\ln(x) + x$ (negative entropy)', color=colors[1])
ax2.plot(x, np.log(x), linewidth=2.5, label='$\\ln(x)$ (logarithm)', color=colors[2])
ax2.set_xlim([0, 3])
ax2.set_ylim([-3, 2])
ax2.set_xlabel('$x$', fontsize=12)
ax2.set_ylabel('$f(x)$', fontsize=12)
ax2.set_title('Strictly Convex Functions in $\\Gamma_0(\\mathbb{R})$: Entropy Functions',
              fontsize=12, fontweight='bold')
ax2.legend(fontsize=11, loc='lower right')
ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('gamma_zero_functions.pdf', dpi=150, bbox_inches='tight')
plt.close()
print("Created: gamma_zero_functions.pdf")

# Figure 7: Jensen's Inequality Visualization
fig, ax = plt.subplots(figsize=(10, 7))
x = np.linspace(0, 5, 500)
f = 0.3*x**2 - x + 1
x1, x2 = 1, 4
y1, y2 = 0.3*x1**2 - x1 + 1, 0.3*x2**2 - x2 + 1
x_mean = (x1 + x2) / 2
y_mean = 0.3*x_mean**2 - x_mean + 1
convex_comb = 0.5*y1 + 0.5*y2
ax.plot(x, f, 'b-', linewidth=3, label='Convex function $f$')
ax.plot([x1, x2], [y1, y2], 'g-', linewidth=2.5, label='Chord connecting points')
ax.plot([x1, x2], [y1, y2], 'go', markersize=10)
ax.plot([x_mean], [y_mean], 'r^', markersize=10, label=f"$f(\\frac{{x_1+x_2}}{{2}})$")
ax.plot([x_mean], [convex_comb], 'bs', markersize=10, label=f"$\\frac{{f(x_1)+f(x_2)}}{{2}}$")
ax.plot([x_mean, x_mean], [y_mean, convex_comb], 'r--', linewidth=2, alpha=0.7)
ax.annotate('', xy=(x_mean+0.15, y_mean), xytext=(x_mean+0.15, convex_comb),
            arrowprops=dict(arrowstyle='<->', color='red', lw=2))
ax.text(x_mean+0.3, (y_mean+convex_comb)/2, "Jensen\ninequality", fontsize=11, color='red', fontweight='bold')
ax.set_xlim([0, 5])
ax.set_ylim([-0.5, 4])
ax.set_xlabel('$x$', fontsize=12)
ax.set_ylabel('$y$', fontsize=12)
ax.set_title("Jensen's Inequality: $f\\left(\\frac{x_1+x_2}{2}\\right) \\leq \\frac{f(x_1)+f(x_2)}{2}$",
             fontsize=13, fontweight='bold')
ax.legend(fontsize=11, loc='upper left')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('jensen_inequality.pdf', dpi=150, bbox_inches='tight')
plt.close()
print("Created: jensen_inequality.pdf")

# Figure 8: KL Divergence and Csiszar divergence
fig, ax = plt.subplots(figsize=(10, 6))
x = np.linspace(0.01, 3, 500)
kl_div = x * np.log(x) - x + 1
ax.plot(x, kl_div, 'b-', linewidth=3, label='KL divergence: $x \\ln(x) - x + 1$')
ax.fill_between(x, kl_div, 0, alpha=0.2, color='blue')
ax.axhline(y=0, color='k', linewidth=0.5)
ax.plot([1], [0], 'ro', markersize=10, label='Minimum at $x=1$, $D(1)=0$')
ax.set_xlim([0, 3])
ax.set_ylim([-0.2, 2])
ax.set_xlabel('$x$', fontsize=12)
ax.set_ylabel('$D(x)$', fontsize=12)
ax.set_title("Kullback-Leibler Divergence as Csiszár $\\phi$-divergence",
             fontsize=13, fontweight='bold')
ax.legend(fontsize=11, loc='upper right')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('kl_divergence.pdf', dpi=150, bbox_inches='tight')
plt.close()
print("Created: kl_divergence.pdf")

print("\nAll figures generated successfully!")
