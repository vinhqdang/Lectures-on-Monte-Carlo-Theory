#!/usr/bin/env python3
"""
Generate figures for Chapter 2d: Ćirić Contractions
Based on Pathak "An Introduction to Nonlinear Analysis and Fixed Point Theory"
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
from matplotlib import rcParams
import fitz
import os

# Set style
rcParams['figure.figsize'] = (10, 6)
rcParams['font.size'] = 11
rcParams['lines.linewidth'] = 1.5
rcParams['axes.linewidth'] = 1.2

# Create figures directory if it doesn't exist
os.makedirs('.', exist_ok=True)

# Figure 1: Comparison of Contraction Types
def fig_contraction_comparison():
    fig, ax = plt.subplots(figsize=(12, 7))

    # Define contractions
    contractions = [
        ('Banach', r'$d(Tx, Ty) \leq k \cdot d(x,y)$', 0.9),
        ('Kannan', r'$d(Tx, Ty) \leq r[d(x,Tx) + d(y,Ty)]$', 0.75),
        ('Chatterjea', r'$d(Tx, Ty) \leq r[d(x,Ty) + d(y,Tx)]$', 0.6),
        ('Zamfirescu', r'$d(Tx, Ty) \leq \max\{d(x,y), \frac{1}{2}[d(x,Tx) + d(y,Ty)], \frac{1}{2}[d(x,Ty) + d(y,Tx)]\}$', 0.45),
        ('Ćirić (HR)', r'$d(Tx, Ty) \leq a_1d(x,Tx) + a_2d(y,Ty) + a_3d(x,Ty) + a_4d(y,Tx) + a_5d(x,y)$', 0.3),
        ('H-S General', r'$d(Tx, Ty) \leq \phi(d(x,Tx), d(y,Ty), d(x,Ty), d(y,Tx), d(x,y))$', 0.15),
    ]

    y_pos = np.arange(len(contractions))

    for i, (name, formula, y) in enumerate(contractions):
        # Box background
        rect = FancyBboxPatch((0.02, y-0.05), 0.96, 0.08,
                             boxstyle="round,pad=0.01",
                             edgecolor='darkblue', facecolor='lightblue',
                             linewidth=2, alpha=0.7)
        ax.add_patch(rect)

        # Name
        ax.text(0.05, y, name, fontsize=13, fontweight='bold',
               verticalalignment='center')

        # Formula
        ax.text(0.35, y, formula, fontsize=11,
               verticalalignment='center', family='monospace')

    ax.set_xlim(0, 1)
    ax.set_ylim(-0.05, 1)
    ax.axis('off')
    ax.set_title('Hierarchy of Generalized Contractions', fontsize=16, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig('fig_contraction_types.pdf', bbox_inches='tight', dpi=300)
    plt.close()

# Figure 2: Ćirić/Hardy-Rogers Condition Parameters
def fig_ciric_parameters():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Parameter space for Ćirić condition
    ax1.text(0.5, 0.95, "Hardy-Rogers (Ćirić) Condition",
            ha='center', fontsize=12, fontweight='bold', transform=ax1.transAxes)

    conditions = [
        r'$a_i \in [0, 1)$ for $i = 1, 2, 3, 4, 5$',
        r'$a_1 + a_2 + a_3 + a_4 + a_5 < 1$',
        '',
        'Contraction Formula:',
        r'$d(Tx, Ty) \leq a_1 d(x, Tx) + a_2 d(y, Ty)$',
        r'$\qquad\quad + a_3 d(x, Ty) + a_4 d(y, Tx) + a_5 d(x, y)$',
    ]

    y_start = 0.85
    for i, cond in enumerate(conditions):
        y = y_start - i * 0.12
        if cond:
            ax1.text(0.1, y, cond, fontsize=11, transform=ax1.transAxes, family='monospace')

    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.axis('off')

    # Right: Example parameter values
    ax2.text(0.5, 0.95, "Example Parameter Values",
            ha='center', fontsize=12, fontweight='bold', transform=ax2.transAxes)

    examples = [
        ('Banach', '$a_5 = 0.5$, others = 0'),
        ('Kannan', '$a_1 = a_2 = 0.5$, others = 0'),
        ('Chatterjea', '$a_3 = a_4 = 0.5$, others = 0'),
        ('Zamf. Special', '$a_1 = a_2 = a_3 = a_4 = 0.25$'),
        ('General', 'All $a_i > 0$ with sum $< 1$'),
    ]

    y_start = 0.85
    for i, (name, param) in enumerate(examples):
        y = y_start - i * 0.15
        ax2.text(0.05, y, name + ':', fontsize=10, fontweight='bold', transform=ax2.transAxes)
        ax2.text(0.35, y, param, fontsize=10, transform=ax2.transAxes, family='monospace')

    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis('off')

    plt.tight_layout()
    plt.savefig('fig_ciric_parameters.pdf', bbox_inches='tight', dpi=300)
    plt.close()

# Figure 3: Generalized contraction mapping framework
def fig_generalized_framework():
    fig, ax = plt.subplots(figsize=(11, 8))

    # Title
    ax.text(0.5, 0.95, "Generalized Contraction Framework",
           ha='center', fontsize=14, fontweight='bold', transform=ax.transAxes)

    # Main structure
    content = """
    GENERALIZED CONTRACTION THEOREM (Hussain-Sehgal)

    Let (X, d) be a complete metric space, T: X → X

    Define φ: (ℝ⁺)⁵ → ℝ⁺ continuous, nondecreasing in each variable

    Suppose T satisfies:
        d(Tx, Ty) ≤ φ(d(x,Tx), d(y,Ty), d(x,Ty), d(y,Tx), d(x,y))

    If φ(t, t, a₁t, a₂t, a₅t) < t  for all t > 0
    where a₁, a₂, a₅ ∈ {0, 1, 2} with a₁ + a₂ ≤ 2

    Then: T has unique fixed point u in X
          lim(T^n x) = u for any x ∈ X

    ─────────────────────────────────────────────
    SPECIAL CASES:

    • Banach: φ(·,·,·,·,d(x,y)) = k·d(x,y), k < 1
    • Kannan: φ(a,b,·,·,0) = r(a+b)/2, r < 1/2
    • Ćirić (HR): φ = a₁t₁ + a₂t₂ + a₃t₃ + a₄t₄ + a₅t₅
      with a₁+a₂+a₃+a₄+a₅ < 1
    """

    ax.text(0.05, 0.88, content, fontsize=9.5, family='monospace',
           verticalalignment='top', transform=ax.transAxes,
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    plt.tight_layout()
    plt.savefig('fig_generalized_framework.pdf', bbox_inches='tight', dpi=300)
    plt.close()

# Figure 4: Relationship between contractions
def fig_contraction_hierarchy():
    fig, ax = plt.subplots(figsize=(10, 8))

    # Define positions for boxes
    positions = {
        'H-S': (0.5, 0.9),
        'HR': (0.5, 0.75),
        'Zamf': (0.3, 0.55),
        'Kannan': (0.15, 0.35),
        'Chatterjea': (0.5, 0.35),
        'Banach': (0.85, 0.35),
    }

    names = {
        'H-S': 'Hussain-Sehgal\n(Most General)',
        'HR': 'Hardy-Rogers\n(Ćirić Type)',
        'Zamf': 'Zamfirescu',
        'Kannan': 'Kannan',
        'Chatterjea': 'Chatterjea',
        'Banach': 'Banach',
    }

    colors = {
        'H-S': '#FFE6E6',
        'HR': '#FFE6E6',
        'Zamf': '#E6F3FF',
        'Kannan': '#E6FFE6',
        'Chatterjea': '#E6FFE6',
        'Banach': '#FFFFE6',
    }

    # Draw boxes
    for key, (x, y) in positions.items():
        rect = FancyBboxPatch((x-0.08, y-0.05), 0.16, 0.1,
                             boxstyle="round,pad=0.01",
                             edgecolor='black', facecolor=colors[key],
                             linewidth=2, transform=ax.transAxes)
        ax.add_patch(rect)
        ax.text(x, y, names[key], ha='center', va='center',
               fontsize=10, fontweight='bold', transform=ax.transAxes)

    # Draw arrows showing generalizations
    arrow_configs = [
        ('H-S', 'HR', 'generalizes'),
        ('HR', 'Zamf', 'special case'),
        ('HR', 'Kannan', 'special case'),
        ('HR', 'Chatterjea', 'special case'),
        ('Kannan', 'Banach', 'special case'),
    ]

    for src, dst, label in arrow_configs:
        x1, y1 = positions[src]
        x2, y2 = positions[dst]

        arrow = FancyArrowPatch((x1, y1-0.06), (x2, y2+0.06),
                               arrowstyle='->', mutation_scale=20,
                               color='darkblue', linewidth=1.5,
                               transform=ax.transAxes)
        ax.add_patch(arrow)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title('Contraction Type Relationships', fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig('fig_contraction_hierarchy.pdf', bbox_inches='tight', dpi=300)
    plt.close()

# Figure 5: Iterative convergence example
def fig_iteration_convergence():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Convergence illustration
    x = np.linspace(0, 1, 100)

    # T(x) curves for different types
    def banach(x):
        return 0.6 * x

    def kannan(x):
        return 0.3 + 0.5 * x * (1 - x)

    def ciric(x):
        return 0.4 + 0.4 * np.sin(x * np.pi)

    ax1.plot(x, x, 'k--', linewidth=2, label='y = x')
    ax1.plot(x, banach(x), 'b-', linewidth=2, label='Banach T')
    ax1.plot(x, kannan(x), 'r-', linewidth=2, label='Kannan T')
    ax1.plot(x, ciric(x), 'g-', linewidth=2, label='Ćirić T')

    ax1.set_xlabel('x', fontsize=11)
    ax1.set_ylabel('T(x)', fontsize=11)
    ax1.set_title('Contraction Mappings', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)

    # Right: Distance decay
    n = np.arange(0, 20)

    # Different contraction rates
    d_banach = 0.5 * (0.6**n)
    d_kannan = 0.5 * (0.5**n)
    d_ciric = 0.5 * (0.65**n)

    ax2.semilogy(n, d_banach, 'bo-', label='Banach (k=0.6)', linewidth=2)
    ax2.semilogy(n, d_kannan, 'rs-', label='Kannan (r=0.5)', linewidth=2)
    ax2.semilogy(n, d_ciric, 'g^-', label='Ćirić (mixed)', linewidth=2)

    ax2.set_xlabel('Iteration n', fontsize=11)
    ax2.set_ylabel(r'$d(x_n, u)$ (log scale)', fontsize=11)
    ax2.set_title('Convergence Rate', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, alpha=0.3, which='both')
    ax2.set_xlim(0, 19)

    plt.tight_layout()
    plt.savefig('fig_iteration_convergence.pdf', bbox_inches='tight', dpi=300)
    plt.close()

# Figure 6: Key theoretical properties
def fig_theoretical_properties():
    fig, ax = plt.subplots(figsize=(11, 8))

    properties_text = """
    KEY THEORETICAL PROPERTIES OF ĆIRIĆ CONTRACTIONS

    1. EXISTENCE OF FIXED POINT
       ✓ Every Ćirić contraction on a complete metric space has at least one fixed point
       ✓ Existence guaranteed without assuming continuity
       ✓ This is a major improvement over classical results

    2. UNIQUENESS OF FIXED POINT
       ✓ The fixed point is unique
       ✓ Guaranteed by the contraction condition
       ✓ No ambiguity in the solution

    3. CONVERGENCE OF ITERATIONS
       ✓ Picard iteration xₙ₊₁ = Txₙ converges to the unique fixed point
       ✓ For any starting point x₀ ∈ X
       ✓ Convergence is often faster than Banach contractions

    4. DISCONTINUITY
       ✓ Ćirić contractions need NOT be continuous
       ✓ This is a crucial generalization
       ✓ Example: Kannan's mapping (discontinuous but contractive)

    5. CONTRACTION CONSTANT
       ✓ Ćirić contractions have a well-defined contraction ratio
       ✓ Determined by the parameters a₁, a₂, a₃, a₄, a₅
       ✓ Convergence rate depends on this ratio

    6. COMPLETENESS REQUIREMENT
       ✓ Complete metric space condition is necessary
       ✓ Ensures Cauchy sequences converge
       ✓ Guarantees existence of fixed point
    """

    ax.text(0.02, 0.98, properties_text, fontsize=10, family='monospace',
           verticalalignment='top', transform=ax.transAxes,
           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8, pad=1))

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title('Theoretical Foundation', fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig('fig_theoretical_properties.pdf', bbox_inches='tight', dpi=300)
    plt.close()

# Figure 7: Applications
def fig_applications():
    fig, ax = plt.subplots(figsize=(11, 7))

    applications = """
    APPLICATIONS OF ĆIRIĆ CONTRACTIONS

    1. NUMERICAL ANALYSIS & OPTIMIZATION
       • Fixed point iteration methods
       • Root finding algorithms
       • Convergence analysis of iterative schemes

    2. DIFFERENTIAL & INTEGRAL EQUATIONS
       • Existence and uniqueness of solutions
       • Solution of nonlinear ODEs
       • Integral equations (Hammerstein, Volterra type)

    3. FUNCTIONAL ANALYSIS
       • Approximation theory
       • Best approximation problems
       • Iterative approximation methods

    4. CONTROL THEORY & SYSTEMS
       • Stability analysis
       • Lyapunov functions
       • Feedback system design

    5. NONLINEAR ANALYSIS
       • Variational problems
       • Nonlinear operator equations
       • Monotone operators on Banach spaces

    6. COMPUTATIONAL MATHEMATICS
       • Machine learning algorithms (gradient descent)
       • Optimization algorithms
       • Convergence guarantees
    """

    ax.text(0.02, 0.98, applications, fontsize=10, family='monospace',
           verticalalignment='top', transform=ax.transAxes,
           bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8, pad=1))

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    plt.tight_layout()
    plt.savefig('fig_applications.pdf', bbox_inches='tight', dpi=300)
    plt.close()

# Figure 8: Extract from PDF - Hardy-Rogers condition
def fig_hardy_rogers_detail():
    fig, ax = plt.subplots(figsize=(11, 7))

    content = """
    HARDY-ROGERS CONTRACTION (1973)

    Definition: A mapping T: X → X is called a Hardy-Rogers contraction if there exist
    nonnegative numbers a₁, a₂, a₃, a₄, a₅ ∈ [0, 1) such that

                a₁ + a₂ + a₃ + a₄ + a₅ < 1

    and the following condition holds:

        d(Tx, Ty) ≤ a₁d(x,Tx) + a₂d(y,Ty) + a₃d(x,Ty) + a₄d(y,Tx) + a₅d(x,y)

    for all x, y ∈ X.

    KEY INSIGHT: This is a generalization of the Ćirić-type contractions, combining:
    • Distance terms involving the mapping (d(x,Tx), d(y,Ty))
    • Distance terms mixing points and images (d(x,Ty), d(y,Tx))
    • Direct distance between points (d(x,y))

    THEOREM: If (X, d) is a complete metric space and T is a Hardy-Rogers
    contraction, then T has a unique fixed point u ∈ X, and for any x ∈ X,
    the sequence {Tⁿx} converges to u.

    SPECIAL CASES:
    • Set a₃ = a₄ = 1/2, others = 0  →  Zamfirescu contraction
    • Set a₁ = a₂ = r, others = 0      →  Kannan contraction
    • Set a₅ = k, others = 0            →  Banach contraction
    """

    ax.text(0.02, 0.98, content, fontsize=10, family='monospace',
           verticalalignment='top', transform=ax.transAxes,
           bbox=dict(boxstyle='round', facecolor='#FFE6F0', alpha=0.9, pad=1))

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title('Hardy-Rogers (Ćirić-type) Contractions', fontsize=13, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig('fig_hardy_rogers.pdf', bbox_inches='tight', dpi=300)
    plt.close()

# Figure 9: Hussain-Sehgal generalization
def fig_hussain_sehgal():
    fig, ax = plt.subplots(figsize=(11, 8))

    content = """
    HUSSAIN-SEHGAL THEOREM (1975)

    The most general form of contractions encompassing Ćirić and others.

    STATEMENT:
    Let (X, d) be a complete metric space and T: X → X a mapping of X into itself.
    Let φ: (ℝ⁺)⁵ → ℝ⁺ be a continuous function that is nondecreasing in each
    coordinate variable, and let T satisfy:

        d(Tx, Ty) ≤ φ(d(x,Tx), d(y,Ty), d(x,Ty), d(y,Tx), d(x,y))

    for all x, y ∈ X.

    If φ(t, t, a₁t, a₂t, a₅t) < t for all t > 0, where a₁ ∈ {0, 1, 2},
    a₂ ∈ {0, 1, 2}, and a₁ + a₂ ≤ 2, then T has a unique fixed point u ∈ X.

    Moreover, for any x ∈ X, the Picard iteration sequence {Tⁿx} converges to u.


    EXAMPLES OF VALID φ:

    • φ(t₁, t₂, t₃, t₄, t₅) = a₁t₁ + a₂t₂ + a₃t₃ + a₄t₄ + a₅t₅  (Hardy-Rogers)

    • φ(t₁, t₂, t₃, t₄, t₅) = max{t₅, (t₁ + t₂)/2, (t₃ + t₄)/2}  (Zamfirescu)

    • φ(t₁, t₂, t₃, t₄, t₅) = √(t₁·t₂) + t₅/4  (Mixed)

    ADVANTAGES:
    ✓ Covers all classical contractions as special cases
    ✓ Allows for discontinuous mappings
    ✓ Provides convergence guarantees
    ✓ Flexible enough for practical applications
    """

    ax.text(0.02, 0.98, content, fontsize=9.5, family='monospace',
           verticalalignment='top', transform=ax.transAxes,
           bbox=dict(boxstyle='round', facecolor='#E6F3FF', alpha=0.9, pad=1))

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title('Hussain-Sehgal Theorem - Most General Framework', fontsize=13, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig('fig_hussain_sehgal.pdf', bbox_inches='tight', dpi=300)
    plt.close()

def main():
    print("Generating figures for Chapter 2d: Ćirić Contractions...")

    fig_contraction_comparison()
    print("✓ fig_contraction_types.pdf")

    fig_ciric_parameters()
    print("✓ fig_ciric_parameters.pdf")

    fig_generalized_framework()
    print("✓ fig_generalized_framework.pdf")

    fig_contraction_hierarchy()
    print("✓ fig_contraction_hierarchy.pdf")

    fig_iteration_convergence()
    print("✓ fig_iteration_convergence.pdf")

    fig_theoretical_properties()
    print("✓ fig_theoretical_properties.pdf")

    fig_applications()
    print("✓ fig_applications.pdf")

    fig_hardy_rogers_detail()
    print("✓ fig_hardy_rogers.pdf")

    fig_hussain_sehgal()
    print("✓ fig_hussain_sehgal.pdf")

    print("\nAll figures generated successfully!")

if __name__ == '__main__':
    main()
