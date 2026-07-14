#!/usr/bin/env python3
"""
Generate figures for Chapter 10a: Set-Valued Mappings
Extracts key diagrams and creates illustrations for the Beamer slides.
"""

import fitz
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
from pathlib import Path

# Set up figure directory
FIG_DIR = Path(__file__).parent
PDF_PATH = FIG_DIR.parent.parent / "An Introduction to Nonlinear Analysis and Fixed Point Theory 2018.pdf"

def extract_pdf_pages():
    """Extract relevant pages from the PDF as images."""
    try:
        doc = fitz.open(str(PDF_PATH))

        # Pages for Chapter 10: Set-Valued Mappings and Integral Inclusions
        # These are pages 786-808 in the PDF (0-indexed: 785-807)
        pages_to_extract = list(range(786, 809))  # Chapter 10 content

        for page_num in pages_to_extract:
            if page_num < doc.page_count:
                page = doc[page_num]
                pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5))
                output_path = FIG_DIR / f"ch10_page_{page_num + 1}.pdf"
                pix.save(str(output_path))
                print(f"Extracted page {page_num + 1} as {output_path}")

        doc.close()
    except Exception as e:
        print(f"Warning: Could not extract PDF pages: {e}")

def create_set_valued_mapping_diagram():
    """Create a diagram illustrating set-valued mappings."""
    fig, ax = plt.subplots(figsize=(10, 7))

    # Domain X
    domain_x = FancyBboxPatch((0.5, 2), 2, 3, boxstyle="round,pad=0.1",
                              edgecolor='blue', facecolor='lightblue', linewidth=2)
    ax.add_patch(domain_x)
    ax.text(1.5, 3.5, r'$X$ (Domain)', fontsize=14, ha='center', weight='bold')

    # Codomain Y
    codomain_y = FancyBboxPatch((6, 1.5), 2.5, 4, boxstyle="round,pad=0.1",
                               edgecolor='red', facecolor='lightyellow', linewidth=2)
    ax.add_patch(codomain_y)
    ax.text(7.25, 4.5, r'$\mathcal{P}(Y)$', fontsize=14, ha='center', weight='bold')
    ax.text(7.25, 3.8, r'(Power set of $Y$)', fontsize=11, ha='center', style='italic')

    # Points in domain
    points_x = [1, 1.5, 2]
    for i, px in enumerate(points_x):
        ax.plot(px, 2.5 + i*0.5, 'bo', markersize=10)

    ax.text(0.3, 2.5, r'$x_1$', fontsize=12)
    ax.text(0.3, 3.0, r'$x_2$', fontsize=12)
    ax.text(0.3, 3.5, r'$x_3$', fontsize=12)

    # Image sets in codomain
    circle1 = plt.Circle((7.5, 2.5), 0.4, fill=False, edgecolor='green', linewidth=2)
    circle2 = plt.Circle((7.5, 3.2), 0.5, fill=False, edgecolor='purple', linewidth=2)
    circle3 = plt.Circle((7.5, 4.0), 0.4, fill=False, edgecolor='orange', linewidth=2)
    ax.add_patch(circle1)
    ax.add_patch(circle2)
    ax.add_patch(circle3)

    ax.text(8.3, 2.5, r'$F(x_1)$', fontsize=11)
    ax.text(8.4, 3.2, r'$F(x_2)$', fontsize=11)
    ax.text(8.3, 4.0, r'$F(x_3)$', fontsize=11)

    # Arrows
    arrow1 = FancyArrowPatch((2.5, 2.7), (6.2, 2.5),
                            arrowstyle='->', mutation_scale=30, linewidth=2,
                            color='darkblue')
    arrow2 = FancyArrowPatch((2.5, 3.0), (6.2, 3.2),
                            arrowstyle='->', mutation_scale=30, linewidth=2,
                            color='darkblue')
    arrow3 = FancyArrowPatch((2.5, 3.5), (6.2, 4.0),
                            arrowstyle='->', mutation_scale=30, linewidth=2,
                            color='darkblue')
    ax.add_patch(arrow1)
    ax.add_patch(arrow2)
    ax.add_patch(arrow3)

    # Labels
    ax.text(4.5, 3.5, r'$F: X \to \mathcal{P}(Y)$', fontsize=13, ha='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    ax.set_xlim(-0.5, 9.5)
    ax.set_ylim(0.5, 5.5)
    ax.set_aspect('equal')
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(FIG_DIR / 'set_valued_mapping.pdf', dpi=300, bbox_inches='tight')
    print("Created set_valued_mapping.pdf")
    plt.close()

def create_integral_inclusion_diagram():
    """Create a diagram illustrating integral inclusions."""
    fig, ax = plt.subplots(figsize=(11, 6))

    # Left side: Problem formulation
    box1 = FancyBboxPatch((0.5, 3.5), 3.5, 2.5, boxstyle="round,pad=0.15",
                          edgecolor='darkblue', facecolor='aliceblue', linewidth=2)
    ax.add_patch(box1)

    ax.text(2.25, 5.5, 'Integral Inclusion', fontsize=12, ha='center', weight='bold')
    ax.text(2.25, 5, r'$y(t) \in \int_0^T k(t,s)A(s,y(s))ds$', fontsize=11, ha='center')
    ax.text(2.25, 4.3, r'$a.e.\ t \in [0,T]$', fontsize=10, ha='center', style='italic')

    # Middle: Operator
    box2 = FancyBboxPatch((4.5, 3.5), 3.5, 2.5, boxstyle="round,pad=0.15",
                          edgecolor='darkgreen', facecolor='lightgreen', linewidth=2)
    ax.add_patch(box2)

    ax.text(6.25, 5.5, 'Multivalued Operator', fontsize=12, ha='center', weight='bold')
    ax.text(6.25, 5, r'$A: C \to \mathcal{K}(C)$', fontsize=11, ha='center')
    ax.text(6.25, 4.3, r'(Upper semicontinuous)', fontsize=10, ha='center', style='italic')

    # Right side: Solution
    box3 = FancyBboxPatch((8.5, 3.5), 3.5, 2.5, boxstyle="round,pad=0.15",
                          edgecolor='darkred', facecolor='lightyellow', linewidth=2)
    ax.add_patch(box3)

    ax.text(10.25, 5.5, 'Solution Space', fontsize=12, ha='center', weight='bold')
    ax.text(10.25, 5, r'$y \in L^p[0,T]$', fontsize=11, ha='center')
    ax.text(10.25, 4.3, r'with $\|y\|_p < \alpha$', fontsize=10, ha='center', style='italic')

    # Arrows
    arrow1 = FancyArrowPatch((4.0, 4.8), (4.5, 4.8),
                            arrowstyle='->', mutation_scale=25, linewidth=2.5,
                            color='black')
    arrow2 = FancyArrowPatch((8.0, 4.8), (8.5, 4.8),
                            arrowstyle='->', mutation_scale=25, linewidth=2.5,
                            color='black')
    ax.add_patch(arrow1)
    ax.add_patch(arrow2)

    ax.text(4.25, 5.3, 'Fixed', fontsize=9, ha='center', style='italic')
    ax.text(4.25, 5, 'point', fontsize=9, ha='center', style='italic')

    # Bottom: Key properties
    ax.text(6.25, 2.5, 'Key Properties:', fontsize=11, ha='center', weight='bold')
    properties = [
        r'• Upper semicontinuous mapping',
        r'• Compact-valued',
        r'• Fredholm integral inclusion',
    ]
    for i, prop in enumerate(properties):
        ax.text(6.25, 2.0 - i*0.4, prop, fontsize=9.5, ha='center')

    ax.set_xlim(0, 12.5)
    ax.set_ylim(1, 6.5)
    ax.set_aspect('equal')
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(FIG_DIR / 'integral_inclusion_structure.pdf', dpi=300, bbox_inches='tight')
    print("Created integral_inclusion_structure.pdf")
    plt.close()

def create_multivalued_properties_chart():
    """Create a chart showing properties of multivalued operators."""
    fig, ax = plt.subplots(figsize=(10, 8))

    properties = [
        'Upper\nSemicontinuity',
        'Lower\nSemicontinuity',
        'Measurability',
        'Compactness',
        'Closure',
        'Convexity'
    ]

    # Create a 3x2 grid
    positions = [(0, 2), (1, 2), (0, 1), (1, 1), (0, 0), (1, 0)]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#F7DC6F']

    for i, (prop, pos, color) in enumerate(zip(properties, positions, colors)):
        x, y = pos[0] * 1.8 + 0.5, pos[1] * 1.5 + 0.5

        box = FancyBboxPatch((x - 0.7, y - 0.4), 1.4, 0.8,
                            boxstyle="round,pad=0.08",
                            edgecolor='black', facecolor=color,
                            linewidth=2, alpha=0.7)
        ax.add_patch(box)
        ax.text(x, y, prop, fontsize=11, ha='center', va='center',
               weight='bold', wrap=True)

    # Add definitions box at bottom
    definition_text = (
        'Multivalued Mapping: A function $F: X \\to \\mathcal{P}(Y)$\n'
        'mapping each point $x \\in X$ to a nonempty subset $F(x) \\subseteq Y$'
    )

    ax.text(1.75, -0.5, definition_text, fontsize=10, ha='center', va='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8, pad=0.5))

    ax.set_xlim(-0.2, 3.7)
    ax.set_ylim(-1.2, 3.2)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Properties of Multivalued Operators', fontsize=13, weight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(FIG_DIR / 'multivalued_properties.pdf', dpi=300, bbox_inches='tight')
    print("Created multivalued_properties.pdf")
    plt.close()

def create_nemytskii_operator_diagram():
    """Create a diagram illustrating Nemytskii operators."""
    fig, ax = plt.subplots(figsize=(11, 6))

    # Problem space
    box1 = FancyBboxPatch((0.3, 2.5), 2.5, 2.5, boxstyle="round,pad=0.1",
                          edgecolor='navy', facecolor='lightblue', linewidth=2)
    ax.add_patch(box1)
    ax.text(1.55, 4.6, 'Input Space', fontsize=11, ha='center', weight='bold')
    ax.text(1.55, 4.1, r'$L^p[0,T]$', fontsize=12, ha='center')
    ax.text(1.55, 3.5, r'$u \in L^p[0,T]$', fontsize=10, ha='center', style='italic')

    # Nemytskii operator box
    arrow1 = FancyArrowPatch((2.8, 3.75), (3.7, 3.75),
                            arrowstyle='->', mutation_scale=25, linewidth=2.5,
                            color='darkred')
    ax.add_patch(arrow1)
    ax.text(3.25, 4.1, r'$N_f$', fontsize=11, ha='center', weight='bold')

    # Output space
    box2 = FancyBboxPatch((3.7, 2.5), 2.5, 2.5, boxstyle="round,pad=0.1",
                          edgecolor='darkgreen', facecolor='lightgreen', linewidth=2)
    ax.add_patch(box2)
    ax.text(4.95, 4.6, 'Output Space', fontsize=11, ha='center', weight='bold')
    ax.text(4.95, 4.1, r'$\mathcal{P}(L^p[0,T])$', fontsize=12, ha='center')
    ax.text(4.95, 3.5, r'$y \in N_f(u)$', fontsize=10, ha='center', style='italic')

    # Definition box
    def_box = FancyBboxPatch((0.3, 0.3), 6, 1.8, boxstyle="round,pad=0.1",
                            edgecolor='black', facecolor='lightyellow', linewidth=2)
    ax.add_patch(def_box)

    ax.text(3.3, 1.9, 'Nemytskii Operator', fontsize=11, ha='center', weight='bold')
    ax.text(3.3, 1.4, r'$N_f(u) = \{y \in L^p[0,T] : y(t) \in f(t, u(t))\ \mathrm{a.e.}\ t \in [0,T]\}$',
           fontsize=10.5, ha='center', family='monospace')
    ax.text(3.3, 0.7, 'Maps a single-valued function to a set-valued one',
           fontsize=9.5, ha='center', style='italic')

    ax.set_xlim(-0.2, 6.8)
    ax.set_ylim(0, 5.2)
    ax.set_aspect('equal')
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(FIG_DIR / 'nemytskii_operator.pdf', dpi=300, bbox_inches='tight')
    print("Created nemytskii_operator.pdf")
    plt.close()

def create_fixed_point_existence_diagram():
    """Create a flowchart for fixed point existence."""
    fig, ax = plt.subplots(figsize=(10, 8))

    # Step 1: Problem Statement
    box1 = FancyBboxPatch((2.5, 6.5), 3, 0.8, boxstyle="round,pad=0.1",
                          edgecolor='darkblue', facecolor='lightblue', linewidth=2)
    ax.add_patch(box1)
    ax.text(4, 6.9, '1. Integral Inclusion Problem', fontsize=10, ha='center', weight='bold')

    arrow1 = FancyArrowPatch((4, 6.5), (4, 5.9),
                            arrowstyle='->', mutation_scale=20, linewidth=2, color='black')
    ax.add_patch(arrow1)

    # Step 2: Operator Formulation
    box2 = FancyBboxPatch((2.5, 5.1), 3, 0.8, boxstyle="round,pad=0.1",
                          edgecolor='darkgreen', facecolor='lightgreen', linewidth=2)
    ax.add_patch(box2)
    ax.text(4, 5.5, '2. Define Multivalued Operator $A$', fontsize=10, ha='center', weight='bold')

    arrow2 = FancyArrowPatch((4, 5.1), (4, 4.5),
                            arrowstyle='->', mutation_scale=20, linewidth=2, color='black')
    ax.add_patch(arrow2)

    # Step 3: Verify Properties
    box3 = FancyBboxPatch((2.5, 3.7), 3, 0.8, boxstyle="round,pad=0.1",
                          edgecolor='purple', facecolor='plum', linewidth=2)
    ax.add_patch(box3)
    ax.text(4, 4.1, '3. Verify Semicontinuity & Compactness', fontsize=9.5, ha='center', weight='bold')

    arrow3 = FancyArrowPatch((4, 3.7), (4, 3.1),
                            arrowstyle='->', mutation_scale=20, linewidth=2, color='black')
    ax.add_patch(arrow3)

    # Step 4: Apply FPT
    box4 = FancyBboxPatch((2.5, 2.3), 3, 0.8, boxstyle="round,pad=0.1",
                          edgecolor='darkorange', facecolor='moccasin', linewidth=2)
    ax.add_patch(box4)
    ax.text(4, 2.7, '4. Apply Fixed Point Theorem', fontsize=10, ha='center', weight='bold')

    arrow4 = FancyArrowPatch((4, 2.3), (4, 1.7),
                            arrowstyle='->', mutation_scale=20, linewidth=2, color='black')
    ax.add_patch(arrow4)

    # Step 5: Solution
    box5 = FancyBboxPatch((2.5, 0.9), 3, 0.8, boxstyle="round,pad=0.1",
                          edgecolor='darkred', facecolor='lightcoral', linewidth=2)
    ax.add_patch(box5)
    ax.text(4, 1.3, '5. Existence of Solution $y^* \\in L^p[0,T]$', fontsize=10, ha='center', weight='bold')

    # Key theorems box
    thm_box = FancyBboxPatch((0.2, 0.9), 2, 3.8, boxstyle="round,pad=0.1",
                            edgecolor='black', facecolor='#FFFFCC', linewidth=1.5)
    ax.add_patch(thm_box)
    ax.text(1.2, 4.5, 'Key Theorems', fontsize=10, ha='center', weight='bold')

    theorems = [
        'Theorem 5.167',
        '(Multivalued',
        'fixed point)',
        '',
        'Leray-Schauder',
        'Alternative'
    ]
    for i, thm in enumerate(theorems):
        ax.text(1.2, 4.1 - i*0.35, thm, fontsize=8.5, ha='center')

    ax.set_xlim(-0.2, 5.8)
    ax.set_ylim(0.5, 7.5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Fixed Point Existence for Integral Inclusions',
               fontsize=12, weight='bold', pad=15)

    plt.tight_layout()
    plt.savefig(FIG_DIR / 'fixed_point_existence_flow.pdf', dpi=300, bbox_inches='tight')
    print("Created fixed_point_existence_flow.pdf")
    plt.close()

def main():
    """Generate all figures."""
    print("Generating figures for Chapter 10a: Set-Valued Mappings...")
    print()

    # Try to extract PDF pages
    extract_pdf_pages()
    print()

    # Create diagrams
    create_set_valued_mapping_diagram()
    create_integral_inclusion_diagram()
    create_multivalued_properties_chart()
    create_nemytskii_operator_diagram()
    create_fixed_point_existence_diagram()

    print()
    print("All figures generated successfully!")

if __name__ == '__main__':
    main()
