#!/usr/bin/env python3
"""
Generate figures for Chapter 5b/6: Degree Theory, k-Set Contractions, and Condensing Operators.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch, Circle
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('default')
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'serif'
plt.rcParams['figure.dpi'] = 150

# ============================================================================
# Figure 1: Vector Field with Index Counting
# ============================================================================
def fig_vector_field_index():
    fig, axes = plt.subplots(1, 5, figsize=(14, 3))
    fig.suptitle(r'Index of Vector Field $\Phi = I - A$ at Origin', fontsize=12, fontweight='bold')

    indices = [-2, -1, 0, 1, 2]

    for idx, ax in enumerate(axes):
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

        # Circle
        circle = Circle((0, 0), 1.0, fill=False, edgecolor='black', linewidth=2)
        ax.add_patch(circle)

        # Create vector field based on index
        x = np.linspace(-1.2, 1.2, 12)
        y = np.linspace(-1.2, 1.2, 12)
        X, Y = np.meshgrid(x, y)

        # Different vector field patterns for each index
        ind = indices[idx]
        angle_offset = ind * np.pi / 2
        R = np.sqrt(X**2 + Y**2)

        # Rotate vector field based on index
        U = np.cos(np.arctan2(Y, X) + ind * angle_offset) / (R + 0.5)
        V = np.sin(np.arctan2(Y, X) + ind * angle_offset) / (R + 0.5)

        ax.quiver(X, Y, U, V, R, cmap='viridis', scale=15, width=0.004)

        ax.set_xticks([-1, 0, 1])
        ax.set_yticks([-1, 0, 1])
        ax.set_title(f'Index = {ind}', fontweight='bold')
        ax.set_xlabel(r'$x_1$')
        if idx == 0:
            ax.set_ylabel(r'$x_2$')

    plt.tight_layout()
    plt.savefig('fig_vector_field_index.pdf', bbox_inches='tight', format='pdf')
    print("Created: fig_vector_field_index.pdf")
    plt.close()

# ============================================================================
# Figure 2: Schauder Fixed Point Theorem Illustration
# ============================================================================
def fig_schauder_fpt():
    fig, ax = plt.subplots(figsize=(10, 8))

    # Draw a Banach space
    theta = np.linspace(0, 2*np.pi, 100)
    x_circle = 2*np.cos(theta)
    y_circle = 2*np.sin(theta)

    ax.fill(x_circle, y_circle, color='lightblue', alpha=0.3, label='Closed, Convex, Compact Set K')
    ax.plot(x_circle, y_circle, 'b-', linewidth=2)

    # Add some points on the boundary
    angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
    for ang in angles:
        x, y = 2*np.cos(ang), 2*np.sin(ang)
        ax.plot(x, y, 'bo', markersize=6)

    # Add a continuous mapping illustration
    t = np.linspace(0, 2*np.pi, 100)
    mapped_x = 1.5*np.cos(t + 0.3)
    mapped_y = 1.5*np.sin(t + 0.3)

    ax.fill(mapped_x, mapped_y, color='lightcoral', alpha=0.2, label='Image F(K) ⊂ K')
    ax.plot(mapped_x, mapped_y, 'r--', linewidth=1.5)

    # Mark fixed point
    ax.plot(0.8, 0.5, 'g*', markersize=25, label='Fixed Point x* ∈ K', zorder=5)
    ax.plot(0, 0, 'ko', markersize=8, label='Center')

    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11, loc='upper right')
    ax.set_title(r'Schauder Fixed Point Theorem: $F: K \to K$ continuous $\Rightarrow \exists x^* \in K: F(x^*) = x^*$',
                fontsize=12, fontweight='bold')
    ax.set_xlabel('Banach Space Dimension 1', fontsize=11)
    ax.set_ylabel('Banach Space Dimension 2', fontsize=11)

    plt.tight_layout()
    plt.savefig('fig_schauder_fpt.pdf', bbox_inches='tight', format='pdf')
    print("Created: fig_schauder_fpt.pdf")
    plt.close()

# ============================================================================
# Figure 3: k-Set Contractions and Measure of Noncompactness
# ============================================================================
def fig_k_set_contractions():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(r'k-Set Contractions: $\mu(TA) \leq k\mu(A)$, $k < 1$',
                fontsize=12, fontweight='bold')

    # Left plot: Original set A
    ax = axes[0]
    theta = np.linspace(0, 2*np.pi, 100)

    # Original set
    r_A = 2
    x_A = r_A * np.cos(theta)
    y_A = r_A * np.sin(theta)
    ax.fill(x_A, y_A, color='lightblue', alpha=0.4, label='Set A')
    ax.plot(x_A, y_A, 'b-', linewidth=2)

    # Add diameter visualization
    ax.plot([-r_A, r_A], [0, 0], 'b-', linewidth=2)
    ax.plot([-r_A, -r_A], [-0.1, 0.1], 'b-', linewidth=1.5)
    ax.plot([r_A, r_A], [-0.1, 0.1], 'b-', linewidth=1.5)
    ax.text(0, -0.5, f'diam(A) = 4', ha='center', fontsize=10, fontweight='bold')

    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 1.5)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_title('Original Set A', fontweight='bold')
    ax.legend(fontsize=10)

    # Right plot: Image under T
    ax = axes[1]
    k = 0.6
    r_TA = k * r_A
    x_TA = r_TA * np.cos(theta)
    y_TA = r_TA * np.sin(theta)

    ax.fill(x_TA, y_TA, color='lightcoral', alpha=0.4, label='Image T(A)')
    ax.plot(x_TA, y_TA, 'r-', linewidth=2)

    # Add diameter visualization
    ax.plot([-r_TA, r_TA], [0, 0], 'r-', linewidth=2)
    ax.plot([-r_TA, -r_TA], [-0.1, 0.1], 'r-', linewidth=1.5)
    ax.plot([r_TA, r_TA], [-0.1, 0.1], 'r-', linewidth=1.5)
    ax.text(0, -0.8, f'diam(T(A)) = {2*k*r_A:.1f} = k·diam(A)',
            ha='center', fontsize=10, fontweight='bold', color='red')

    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 1.5)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_title(f'Image under k-Set Contraction (k={k})', fontweight='bold')
    ax.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig('fig_k_set_contractions.pdf', bbox_inches='tight', format='pdf')
    print("Created: fig_k_set_contractions.pdf")
    plt.close()

# ============================================================================
# Figure 4: Measure of Noncompactness Properties
# ============================================================================
def fig_measure_noncompactness():
    fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    fig.suptitle(r'Measure of Noncompactness $\mu$: Key Properties',
                fontsize=13, fontweight='bold')

    # Property 1: Monotonicity
    ax = axes[0, 0]
    x = np.array([1, 2, 3, 4, 5])
    mu_A = np.array([1.0, 1.5, 2.0, 2.3, 2.5])
    mu_B = np.array([0.5, 0.8, 1.2, 1.5, 2.0])

    ax.plot(x, mu_A, 'o-', linewidth=2, markersize=8, label=r'$\mu(A)$: $A \subseteq B$ implies $\mu(A) \leq \mu(B)$')
    ax.plot(x, mu_B, 's--', linewidth=2, markersize=8, label=r'$\mu(B)$')
    ax.fill_between(x, mu_B, mu_A, alpha=0.2, color='gray')
    ax.set_xlabel('Sequence Index n', fontsize=10)
    ax.set_ylabel(r'Measure $\mu$', fontsize=10)
    ax.set_title('Property 1: Monotonicity', fontweight='bold', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Property 2: Regularity
    ax = axes[0, 1]
    x = np.array([1, 2, 3, 4, 5, 6, 7])
    mu_vals = np.array([2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 0.0])

    ax.bar(x[:-1], mu_vals[:-1], color='skyblue', edgecolor='blue', linewidth=1.5,
           label=r'$\mu(A) = 2$ (Non-compact)')
    ax.bar(x[-1:], mu_vals[-1:], color='lightcoral', edgecolor='red', linewidth=1.5,
           label=r'$\mu(A) = 0$ (Compact)')
    ax.set_xlabel('Example', fontsize=10)
    ax.set_ylabel(r'Measure $\mu(A)$', fontsize=10)
    ax.set_title('Property 2: Regularity (μ(A)=0 ⟺ A compact)', fontweight='bold', fontsize=11)
    ax.set_xticks(x)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 2.5])

    # Property 3: Contraction principle
    ax = axes[1, 0]
    k_vals = np.linspace(0, 1, 50)
    iteration_counts = []

    for k in k_vals:
        if k > 0:
            # Estimate iterations needed
            iters = np.ceil(np.log(0.01) / np.log(k)) if k > 0 else 1
        else:
            iters = 1
        iteration_counts.append(iters)

    ax.plot(k_vals, iteration_counts, 'b-', linewidth=2.5)
    ax.axvline(x=1, color='red', linestyle='--', linewidth=2, label='k=1 (not k-contraction)')
    ax.fill_between(k_vals[k_vals < 1], 0, 100, alpha=0.1, color='green', label='k-contraction region')
    ax.set_xlabel(r'Contraction Factor $k$', fontsize=10)
    ax.set_ylabel(r'Est. Iterations to Converge', fontsize=10)
    ax.set_title(r'Property 3: Contraction Principle ($k < 1$)', fontweight='bold', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 100])

    # Property 4: Darbo's theorem
    ax = axes[1, 1]
    x = np.arange(5)
    labels = ['Compact\nK', 'k-Set\nContraction\nk<1', 'Continuous\nF: K→K', 'Measure\nμ(FK)≤kμ(K)', 'Fixed Point\nExists']
    colors = ['lightblue', 'lightgreen', 'lightyellow', 'lightcoral', 'lightcyan']

    ax.bar(x, [1]*5, color=colors, edgecolor='black', linewidth=1.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9, rotation=0)
    ax.set_ylabel('', fontsize=10)
    ax.set_ylim([0, 1.3])
    ax.set_title(r"Darbo's Fixed Point Theorem: Chain of Implications",
                fontweight='bold', fontsize=11)
    ax.set_yticks([])

    # Add arrows
    for i in range(len(x)-1):
        ax.annotate('', xy=(i+0.4, 0.5), xytext=(i+0.6, 0.5),
                   arrowprops=dict(arrowstyle='->', lw=2, color='black'))

    plt.tight_layout()
    plt.savefig('fig_measure_noncompactness.pdf', bbox_inches='tight', format='pdf')
    print("Created: fig_measure_noncompactness.pdf")
    plt.close()

# ============================================================================
# Figure 5: Skrypnik Degree Theory Conditions
# ============================================================================
def fig_skrypnik_degree():
    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3)

    fig.suptitle(r"Skrypnik's Degree Theory: $\Phi = I - A$, $A: \bar{\Omega} \to X^*$",
                fontsize=13, fontweight='bold')

    # Condition 1: Domain
    ax = fig.add_subplot(gs[0, 0])
    theta = np.linspace(0, 2*np.pi, 100)
    x_boundary = 2*np.cos(theta)
    y_boundary = 2*np.sin(theta)

    ax.fill(x_boundary, y_boundary, color='lightblue', alpha=0.3)
    ax.plot(x_boundary, y_boundary, 'b-', linewidth=2, label=r'$\partial\Omega$ (boundary)')
    ax.plot([0], [0], 'g*', markersize=20, label=r'Origin (target)')
    ax.text(0, -1.2, r'$(1)$ $0 \notin \overline{\Omega}$', fontsize=11,
           ha='center', fontweight='bold', bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    ax.set_title('Condition (1): Domain Setup', fontweight='bold')

    # Condition 2: Operator properties
    ax = fig.add_subplot(gs[0, 1])
    ax.axis('off')

    conditions_text = [
        r"Condition (2): Operator $A$ properties",
        r"$\quad\bullet$ Completely continuous on $\overline{\Omega}$",
        r"$\quad\bullet$ Pseudomonotone or demicontinuous",
        r"$\quad\bullet$ $\|Au\|_{X^*} \geq \delta_0 > 0$ for $u \in \partial\Omega$",
        r"",
        r"These ensure $\Phi = I - A$ satisfies",
        r"the $\alpha(\partial\Omega)$ condition"
    ]

    for i, text in enumerate(conditions_text):
        weight = 'bold' if i == 0 else 'normal'
        fontsize = 11 if i == 0 else 10
        ax.text(0.05, 0.95 - i*0.13, text, transform=ax.transAxes,
               fontsize=fontsize, fontweight=weight, family='monospace',
               bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5) if i==0 else None)

    # Conclusion
    ax = fig.add_subplot(gs[1, :])
    ax.axis('off')

    conclusion_text = [
        r"Skrypnik's Theorem: If conditions (1) and (2) hold, then $\deg(A, \overline{\Omega}, 0)$ is well-defined and",
        r"possesses properties similar to finite-dimensional degree. Key properties:",
        r"",
        r"$\quad\bullet$ Existence: $\deg(A, \Omega, 0) \neq 0 \Rightarrow \exists u \in \Omega$ with $Au = u$",
        r"$\quad\bullet$ Homotopy Invariance: Continuous deformation preserves degree (if certain conditions hold)",
        r"$\quad\bullet$ Fixed Point: $\deg(I - A, \Omega, 0) \neq 0 \Rightarrow \exists x \in \Omega: Ax = x$",
    ]

    for i, text in enumerate(conclusion_text):
        fontsize = 11 if i < 2 else 10
        weight = 'bold' if i < 2 else 'normal'
        ax.text(0.02, 0.9 - i*0.12, text, transform=ax.transAxes,
               fontsize=fontsize, fontweight=weight, family='monospace')

    plt.savefig('fig_skrypnik_degree.pdf', bbox_inches='tight', format='pdf')
    print("Created: fig_skrypnik_degree.pdf")
    plt.close()

# ============================================================================
# Figure 6: Condensing Operators Hierarchy
# ============================================================================
def fig_condensing_hierarchy():
    fig, ax = plt.subplots(figsize=(11, 8))

    # Create a hierarchy diagram
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    fig.suptitle('Hierarchy of Operator Classes: Condensing Operators',
                fontsize=13, fontweight='bold', y=0.98)

    # Define boxes and their positions
    boxes = [
        {'pos': (5, 9), 'text': 'Condensing Operators\n(α-condensing)', 'color': 'lightcyan', 'fontsize': 11, 'weight': 'bold'},
        {'pos': (2, 7.2), 'text': 'k-Set Contractions\n(k < 1)', 'color': 'lightblue', 'fontsize': 10},
        {'pos': (5, 7.2), 'text': 'Expansive Measures\nof Noncompactness', 'color': 'lightgreen', 'fontsize': 10},
        {'pos': (8, 7.2), 'text': 'Weakly Continuous\nOperators', 'color': 'lightyellow', 'fontsize': 10},
        {'pos': (1.5, 5), 'text': 'Compact\nOperators', 'color': 'lightcoral', 'fontsize': 9},
        {'pos': (3.5, 5), 'text': 'Nonexpansive\nMappings\n(Lipschitz k=1)', 'color': 'lightcoral', 'fontsize': 9},
        {'pos': (5.5, 5), 'text': 'Completely\nContinuous', 'color': 'lightcoral', 'fontsize': 9},
        {'pos': (7.5, 5), 'text': 'Demicompact\nOperators', 'color': 'lightcoral', 'fontsize': 9},
    ]

    # Draw boxes
    for box in boxes:
        x, y = box['pos']
        width, height = 1.6, 0.8
        rect = patches.FancyBboxPatch((x - width/2, y - height/2), width, height,
                                      boxstyle="round,pad=0.05",
                                      edgecolor='black', facecolor=box['color'],
                                      linewidth=1.5)
        ax.add_patch(rect)

        ax.text(x, y, box['text'], ha='center', va='center',
               fontsize=box.get('fontsize', 10),
               fontweight=box.get('weight', 'normal'))

    # Draw arrows
    arrows = [
        ((5, 8.6), (2, 7.6)),
        ((5, 8.6), (5, 7.6)),
        ((5, 8.6), (8, 7.6)),
        ((2, 6.8), (1.5, 5.4)),
        ((2, 6.8), (3.5, 5.4)),
        ((5, 6.8), (5.5, 5.4)),
        ((8, 6.8), (7.5, 5.4)),
    ]

    for start, end in arrows:
        arrow = FancyArrowPatch(start, end, arrowstyle='->', mutation_scale=20,
                              color='black', linewidth=1.5)
        ax.add_patch(arrow)

    # Add explanatory text
    ax.text(5, 3.5, 'Fixed Point Theorems (Darbo, Sadovskii):',
           ha='center', fontsize=11, fontweight='bold')
    ax.text(5, 2.8, 'For condensing operators T: Ω → Ω on bounded closed convex sets,',
           ha='center', fontsize=10)
    ax.text(5, 2.2, 'if α(TK) < α(K) for any nonempty K ⊂ Ω, then T has a fixed point.',
           ha='center', fontsize=10)

    plt.tight_layout()
    plt.savefig('fig_condensing_hierarchy.pdf', bbox_inches='tight', format='pdf')
    print("Created: fig_condensing_hierarchy.pdf")
    plt.close()

# ============================================================================
# Main execution
# ============================================================================
if __name__ == '__main__':
    print("Generating figures for Chapter 6: Degree Theory and Condensing Operators...")
    print()

    fig_vector_field_index()
    fig_schauder_fpt()
    fig_k_set_contractions()
    fig_measure_noncompactness()
    fig_skrypnik_degree()
    fig_condensing_hierarchy()

    print()
    print("All figures generated successfully!")
