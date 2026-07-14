#!/usr/bin/env python3
"""
Generate figures for Chapter 3a: Weakly Contractive Mappings
Pathak - An Introduction to Nonlinear Analysis and Fixed Point Theory (2018)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches
from scipy import optimize
import warnings
warnings.filterwarnings('ignore')

# Set up matplotlib
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['grid.linewidth'] = 0.5


def fig_contraction_mapping_principle():
    """Illustration of contraction mapping principle"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Left plot: Function iteration
    x = np.linspace(0, 3, 200)
    k_values = [0.3, 0.6, 0.9]
    colors = ['#2E86AB', '#A23B72', '#F18F01']

    for k, color in zip(k_values, colors):
        y = k * x
        ax1.plot(x, y, color=color, linewidth=2, label=f'$Tx = {k}x$')

    # Identity line
    ax1.plot(x, x, 'k--', linewidth=1.5, label='$y = x$ (fixed point)')
    ax1.set_xlim(0, 3)
    ax1.set_ylim(0, 3)
    ax1.set_xlabel('$x$', fontsize=11)
    ax1.set_ylabel('$Tx$', fontsize=11)
    ax1.set_title('Contraction Mappings: $Tx = kx$', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')

    # Right plot: Iteration convergence
    x0 = 2.5
    iterations = 20
    colors_iter = ['#2E86AB', '#A23B72', '#F18F01']
    k_values = [0.3, 0.6, 0.9]

    for k, color in zip(k_values, colors_iter):
        x_iter = [x0]
        for i in range(iterations):
            x_iter.append(k * x_iter[-1])
        ax2.plot(range(len(x_iter)), x_iter, 'o-', color=color, linewidth=2,
                markersize=4, label=f'$k = {k}$', alpha=0.7)

    ax2.axhline(y=0, color='k', linestyle='--', linewidth=1.5, label='Fixed point $u$')
    ax2.set_xlabel('Iteration $n$', fontsize=11)
    ax2.set_ylabel('$x_n = T^n(x_0)$', fontsize=11)
    ax2.set_title('Convergence of Iterates', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('fig_contraction_mapping_principle.pdf', bbox_inches='tight', dpi=150)
    print("Saved: fig_contraction_mapping_principle.pdf")
    plt.close()


def fig_kannan_chatterjea():
    """Kannan and Chatterjea contractions"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Kannan's contraction: d(Tx, Ty) <= r[d(Tx,y) + d(Ty,x)]
    x = np.linspace(0.01, 2, 100)

    # Illustrate the condition geometrically
    ax1.set_xlim(-0.5, 2.5)
    ax1.set_ylim(-0.5, 2.5)
    ax1.set_aspect('equal')

    # Draw two points
    x_pt = np.array([1.0, 1.5])
    y_pt = np.array([1.8, 0.8])

    ax1.plot(*x_pt, 'o', color='#2E86AB', markersize=10, label='$x$')
    ax1.plot(*y_pt, 'o', color='#A23B72', markersize=10, label='$y$')

    # Kannan contraction involves mixed distances
    ax1.annotate('', xy=y_pt, xytext=x_pt,
                arrowprops=dict(arrowstyle='<->', color='gray', lw=1))
    ax1.text((x_pt[0]+y_pt[0])/2 - 0.3, (x_pt[1]+y_pt[1])/2,
            '$d(x,y)$', fontsize=10, color='gray')

    ax1.set_xlabel('X-coordinate', fontsize=11)
    ax1.set_ylabel('Y-coordinate', fontsize=11)
    ax1.set_title("Kannan's Contraction\n$d(Tx,Ty) \\leq r[d(Tx,y) + d(Ty,x)]$",
                 fontsize=11, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)

    # Chatterjea's contraction
    ax2.set_xlim(-0.5, 2.5)
    ax2.set_ylim(-0.5, 2.5)
    ax2.set_aspect('equal')

    ax2.plot(*x_pt, 'o', color='#2E86AB', markersize=10, label='$x$')
    ax2.plot(*y_pt, 'o', color='#A23B72', markersize=10, label='$y$')

    ax2.annotate('', xy=y_pt, xytext=x_pt,
                arrowprops=dict(arrowstyle='<->', color='gray', lw=1))
    ax2.text((x_pt[0]+y_pt[0])/2 - 0.4, (x_pt[1]+y_pt[1])/2,
            '$d(x,y)$', fontsize=10, color='gray')

    ax2.set_xlabel('X-coordinate', fontsize=11)
    ax2.set_ylabel('Y-coordinate', fontsize=11)
    ax2.set_title("Chatterjea's Contraction\n$d(Tx,Ty) \\leq r[d(Tx,y) + d(Ty,x)]$",
                 fontsize=11, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig('fig_kannan_chatterjea.pdf', bbox_inches='tight', dpi=150)
    print("Saved: fig_kannan_chatterjea.pdf")
    plt.close()


def fig_reich_contraction():
    """Reich contraction and generalized contractions"""
    fig = plt.figure(figsize=(10, 5))

    # Create a 2D parameter space for Reich contraction
    # d(Tx,Ty) <= a*d(x,y) + b*d(x,Tx) + c*d(y,Ty)
    # with a + b + c < 1

    a = np.linspace(0, 1, 100)
    b_max = 1 - a  # Maximum b for given a

    ax = fig.add_subplot(121)

    # Fill the feasible region for Reich contraction
    for i, a_val in enumerate(a[::5]):
        b_range = np.linspace(0, min(1-a_val, 1), 50)
        for b_val in b_range:
            c_range = np.linspace(0, 1-a_val-b_val, 20)
            if len(c_range) > 0:
                ax.scatter([a_val]*len(c_range), [b_val]*len(c_range),
                          c=c_range, s=5, cmap='viridis', alpha=0.6, vmin=0, vmax=1)

    ax.set_xlabel('Parameter $a$', fontsize=11)
    ax.set_ylabel('Parameter $b$', fontsize=11)
    ax.set_title("Reich Contraction Parameter Space\n$a + b + c < 1$",
                fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # Comparison of different contractions
    ax2 = fig.add_subplot(122)

    contraction_types = [
        'Banach\nContraction',
        "Kannan's\nContraction",
        "Chatterjea's\nContraction",
        "Reich\nContraction",
        "Čirić\nContraction"
    ]

    generality_levels = [1, 2, 2, 3, 4]
    colors_bar = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A4C93']

    bars = ax2.barh(contraction_types, generality_levels, color=colors_bar, alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Generality Level', fontsize=11)
    ax2.set_title('Hierarchy of Generalized Contractions', fontsize=11, fontweight='bold')
    ax2.set_xlim(0, 5)
    ax2.grid(True, alpha=0.3, axis='x')

    for i, (bar, val) in enumerate(zip(bars, generality_levels)):
        ax2.text(val + 0.1, i, str(val), va='center', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig('fig_reich_contraction.pdf', bbox_inches='tight', dpi=150)
    print("Saved: fig_reich_contraction.pdf")
    plt.close()


def fig_meir_keeler():
    """Meir-Keeler contraction illustration"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Left: Epsilon-delta diagram
    d_vals = np.linspace(0, 2, 200)
    epsilon = 1.0
    delta = 0.3

    # Meir-Keeler condition
    ax1.axhline(y=epsilon, color='red', linestyle='--', linewidth=2, label=r'$\varepsilon$')
    ax1.axhline(y=epsilon + delta, color='orange', linestyle='--', linewidth=2, label=r'$\varepsilon + \delta$')

    # Fill regions
    ax1.fill_between([0, epsilon], 0, 2.5, alpha=0.1, color='blue', label='$d(x,y) < \\varepsilon$')
    ax1.fill_between([epsilon, epsilon+delta], 0, 2.5, alpha=0.1, color='orange', label='$\\varepsilon \\leq d(x,y) < \\varepsilon+\\delta$')

    ax1.set_xlabel('$d(x,y)$', fontsize=11)
    ax1.set_ylabel('Distance', fontsize=11)
    ax1.set_title("Meir-Keeler Contraction\nEpsilon-Delta Condition", fontsize=11, fontweight='bold')
    ax1.set_xlim(0, 2)
    ax1.set_ylim(0, 2.5)
    ax1.legend(fontsize=9, loc='upper right')
    ax1.grid(True, alpha=0.3)

    # Right: Asymptotic behavior
    x0 = 1.5
    T_iter_mk = [x0]
    T_iter_regular = [x0]

    for i in range(15):
        # Meir-Keeler: contractive but not linear
        x_curr_mk = T_iter_mk[-1]
        if x_curr_mk > 0.05:
            x_new_mk = x_curr_mk * (0.85 - 0.1*np.sin(x_curr_mk))
        else:
            x_new_mk = 0
        T_iter_mk.append(x_new_mk)

        # Regular contraction
        T_iter_regular.append(T_iter_regular[-1] * 0.8)

    ax2.plot(range(len(T_iter_mk)), T_iter_mk, 'o-', color='#2E86AB',
            linewidth=2, markersize=5, label='Meir-Keeler')
    ax2.plot(range(len(T_iter_regular)), T_iter_regular, 's--', color='#A23B72',
            linewidth=2, markersize=5, label='Regular Contraction')
    ax2.set_xlabel('Iteration $n$', fontsize=11)
    ax2.set_ylabel('$x_n$', fontsize=11)
    ax2.set_title('Convergence Comparison', fontsize=11, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')

    plt.tight_layout()
    plt.savefig('fig_meir_keeler.pdf', bbox_inches='tight', dpi=150)
    print("Saved: fig_meir_keeler.pdf")
    plt.close()


def fig_ciric_hardy_rogers():
    """Čirić and Hardy-Rogers contractions"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Čirić: d(Tx,Ty) <= a*d(x,y) + b*d(x,Tx) + c*d(y,Ty) + e*[d(x,Ty) + d(y,Tx)]
    # with a + b + c + 2e < 1

    a = np.linspace(0, 0.4, 50)
    colors = plt.cm.viridis(np.linspace(0, 1, len(a)))

    for i, a_val in enumerate(a[::5]):
        max_remaining = 1 - a_val
        b_range = np.linspace(0, max_remaining/3, 20)
        for b_val in b_range:
            c_val = max_remaining/3
            e_val = max(0, (max_remaining - b_val - c_val)/2)
            if a_val + b_val + c_val + 2*e_val < 1:
                ax1.scatter(a_val, b_val, s=20, color=colors[i], alpha=0.6)

    ax1.set_xlabel('Parameter $a$', fontsize=11)
    ax1.set_ylabel('Parameter $b$', fontsize=11)
    ax1.set_title("Čirić Contraction\n$a + b + c + 2e < 1$", fontsize=11, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 0.4)
    ax1.set_ylim(0, 0.3)

    # Hardy-Rogers: 5 parameters with specific constraint
    # d(Tx,Ty) <= a1*d(x,Tx) + a2*d(y,Ty) + a3*d(x,Ty) + a4*d(y,Tx) + a5*d(x,y)
    # with sum < 1

    param_names = ['$a_1$\n(x-Tx)', '$a_2$\n(y-Ty)', '$a_3$\n(x-Ty)', '$a_4$\n(y-Tx)', '$a_5$\n(x-y)']
    param_values = [0.15, 0.15, 0.15, 0.15, 0.35]
    colors_hr = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A4C93']

    bars = ax2.bar(range(len(param_names)), param_values, color=colors_hr, alpha=0.7, edgecolor='black')
    ax2.axhline(y=1, color='red', linestyle='--', linewidth=2, label='Sum = 1')
    ax2.set_ylabel('Parameter Value', fontsize=11)
    ax2.set_title('Hardy-Rogers Contraction\nParameter Example', fontsize=11, fontweight='bold')
    ax2.set_xticks(range(len(param_names)))
    ax2.set_xticklabels(param_names, fontsize=9)
    ax2.set_ylim(0, 0.5)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.legend(fontsize=10)

    # Add values on bars
    for bar, val in zip(bars, param_values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{val:.2f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig('fig_ciric_hardy_rogers.pdf', bbox_inches='tight', dpi=150)
    print("Saved: fig_ciric_hardy_rogers.pdf")
    plt.close()


def fig_epsilon_chainable_spaces():
    """ε-chainable metric spaces"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Left: ε-chain illustration
    ax1.set_xlim(-0.5, 5.5)
    ax1.set_ylim(-0.5, 3)
    ax1.set_aspect('equal')

    # Points
    x_start = np.array([0.5, 2.5])
    x_end = np.array([4.5, 1.0])

    # ε-chain
    chain_points = [
        x_start,
        np.array([1.5, 2.2]),
        np.array([2.5, 1.8]),
        np.array([3.5, 1.4]),
        x_end
    ]

    chain_x = [p[0] for p in chain_points]
    chain_y = [p[1] for p in chain_points]

    # Draw chain
    ax1.plot(chain_x, chain_y, 'o-', color='#2E86AB', markersize=8, linewidth=2, label='ε-chain')
    ax1.plot([x_start[0], x_end[0]], [x_start[1], x_end[1]], '--', color='gray',
            linewidth=1, alpha=0.5, label='Direct distance')

    # Draw ε-balls around intermediate points
    epsilon = 0.4
    for i, p in enumerate(chain_points[:-1]):
        circle = Circle(chain_points[i], epsilon, fill=False, edgecolor='orange',
                       linestyle=':', linewidth=1, alpha=0.5)
        ax1.add_patch(circle)

    ax1.text(x_start[0]-0.3, x_start[1]+0.2, 'x', fontsize=11, fontweight='bold')
    ax1.text(x_end[0]+0.1, x_end[1]-0.3, 'y', fontsize=11, fontweight='bold')

    ax1.set_xlabel('Dimension 1', fontsize=11)
    ax1.set_ylabel('Dimension 2', fontsize=11)
    ax1.set_title('ε-Chainable Metric Space\nε-chain joining $x$ and $y$', fontsize=11, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Right: Edelstein theorem convergence
    n_vals = np.arange(0, 20)

    # Simulate convergence with different ε-localization
    epsilons = [0.1, 0.2, 0.3]
    colors_edel = ['#2E86AB', '#A23B72', '#F18F01']

    for eps, color in zip(epsilons, colors_edel):
        k = 0.7  # Local contraction constant
        x_iter = [1.0]
        for i in range(19):
            x_iter.append(k * x_iter[-1])
        ax2.plot(n_vals, x_iter, 'o-', color=color, linewidth=2, markersize=4,
                label=f'$\\varepsilon = {eps}$', alpha=0.7)

    ax2.set_xlabel('Iteration $n$', fontsize=11)
    ax2.set_ylabel('$d(x_n, u)$', fontsize=11)
    ax2.set_title('Edelstein Theorem\nε-local Contraction Convergence', fontsize=11, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')

    plt.tight_layout()
    plt.savefig('fig_epsilon_chainable_spaces.pdf', bbox_inches='tight', dpi=150)
    print("Saved: fig_epsilon_chainable_spaces.pdf")
    plt.close()


def fig_bessaga_theorem():
    """Bessaga's converse theorem"""
    fig = plt.figure(figsize=(10, 5))

    # Illustration of equivalence classes
    ax1 = fig.add_subplot(121)
    ax1.set_xlim(-0.5, 5)
    ax1.set_ylim(-0.5, 4)
    ax1.set_aspect('equal')

    # Fixed point
    u = np.array([2.5, 2])
    ax1.plot(*u, 'o', color='red', markersize=12, label='Fixed point $u$', zorder=5)

    # Equivalence classes
    class1 = [np.array([1, 3]), np.array([1.5, 2.5]), np.array([0.5, 1.5])]
    class2 = [np.array([3, 3.5]), np.array([4, 2.5]), np.array([3.5, 1])]
    class3 = [np.array([2, 0.5]), np.array([0.5, 0.5])]

    # Draw equivalence classes with different colors
    for i, pt in enumerate(class1):
        ax1.plot(*pt, 'o', color='#2E86AB', markersize=8, alpha=0.7)
        if i == 0:
            ax1.plot(*pt, 'o', color='#2E86AB', markersize=8,
                    label='Equivalence class 1', alpha=0.7)

    for i, pt in enumerate(class2):
        ax1.plot(*pt, 's', color='#A23B72', markersize=8, alpha=0.7)
        if i == 0:
            ax1.plot(*pt, 's', color='#A23B72', markersize=8,
                    label='Equivalence class 2', alpha=0.7)

    for i, pt in enumerate(class3):
        ax1.plot(*pt, '^', color='#F18F01', markersize=8, alpha=0.7)
        if i == 0:
            ax1.plot(*pt, '^', color='#F18F01', markersize=8,
                    label='Equivalence class 3', alpha=0.7)

    ax1.set_xlabel('X-coordinate', fontsize=11)
    ax1.set_ylabel('Y-coordinate', fontsize=11)
    ax1.set_title("Bessaga's Converse\nEquivalence Classes under $T$", fontsize=11, fontweight='bold')
    ax1.legend(fontsize=9, loc='upper left')
    ax1.grid(True, alpha=0.3)

    # Right: Distance metric construction
    ax2 = fig.add_subplot(122)

    n_vals = np.arange(0, 12)
    lambda_vals = [0.3, 0.6, 0.9]
    colors_bessaga = ['#2E86AB', '#A23B72', '#F18F01']

    for lam, color in zip(lambda_vals, colors_bessaga):
        # d(x, u) = λ^(-n) if T^n(x) = u
        d_vals = [lam**(-n) for n in n_vals]
        ax2.semilogy(n_vals, d_vals, 'o-', color=color, linewidth=2, markersize=5,
                    label=f'$\\lambda = {lam}$', alpha=0.7)

    ax2.set_xlabel('Iterate $n$', fontsize=11)
    ax2.set_ylabel('Distance $d(x, u) = \\lambda^{-n}$', fontsize=11)
    ax2.set_title('Metric Construction in Bessaga Theorem\n$d(x,u) = \\lambda^{-n}$ when $T^n(x)=u$',
                 fontsize=11, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    plt.savefig('fig_bessaga_theorem.pdf', bbox_inches='tight', dpi=150)
    print("Saved: fig_bessaga_theorem.pdf")
    plt.close()


def fig_contraction_types_hierarchy():
    """Hierarchy and relationships of contractive mappings"""
    fig = plt.figure(figsize=(11, 6))
    ax = fig.add_subplot(111)

    # Create hierarchical structure
    contractions = {
        'Banach\nContraction': (0.5, 0.85, '#2E86AB'),
        "Kannan's\nContraction": (-0.2, 0.6, '#A23B72'),
        "Chatterjea's\nContraction": (1.2, 0.6, '#F18F01'),
        'Meir-Keeler\nContraction': (-0.5, 0.3, '#C73E1D'),
        'Reich\nContraction': (0, 0.3, '#6A4C93'),
        'Čirić\nContraction': (0.5, 0.3, '#E63946'),
        'Hardy-Rogers\nContraction': (1, 0.3, '#457B9D'),
        'Asymptotically\nNonexpansive': (1.5, 0.6, '#1D3557'),
    }

    # Draw boxes
    box_height = 0.12
    box_width = 0.18

    for name, (x, y, color) in contractions.items():
        rect = FancyBboxPatch((x - box_width/2, y - box_height/2), box_width, box_height,
                             boxstyle="round,pad=0.01", edgecolor='black', facecolor=color,
                             alpha=0.7, linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x, y, name, ha='center', va='center', fontsize=8, fontweight='bold', color='white')

    # Draw relationships (arrows)
    relationships = [
        ((0.5, 0.79), (-0.2, 0.66)),  # Banach -> Kannan
        ((0.5, 0.79), (1.2, 0.66)),   # Banach -> Chatterjea
        ((-0.2, 0.54), (-0.5, 0.36)), # Kannan -> Meir-Keeler
        ((0, 0.54), (0, 0.36)),       # Kannan -> Reich
        ((1.2, 0.54), (0.5, 0.36)),   # Chatterjea -> Čirić
        ((0.5, 0.79), (0, 0.36)),     # Banach -> Reich
    ]

    for start, end in relationships:
        ax.annotate('', xy=end, xytext=start,
                   arrowprops=dict(arrowstyle='->', lw=1.5, color='gray', alpha=0.5))

    ax.set_xlim(-0.8, 1.8)
    ax.set_ylim(0, 1)
    ax.axis('off')

    ax.text(0.5, 0.95, 'Hierarchy of Contractive Mappings', ha='center', fontsize=13,
           fontweight='bold', transform=ax.transAxes)

    # Add legend
    ax.text(0.02, 0.02, 'Arrows indicate generalization relationships',
           fontsize=9, style='italic', transform=ax.transAxes)

    plt.tight_layout()
    plt.savefig('fig_contraction_types_hierarchy.pdf', bbox_inches='tight', dpi=150)
    print("Saved: fig_contraction_types_hierarchy.pdf")
    plt.close()


def main():
    """Generate all figures"""
    print("Generating figures for Chapter 3a: Weakly Contractive Mappings...")
    print()

    fig_contraction_mapping_principle()
    fig_kannan_chatterjea()
    fig_reich_contraction()
    fig_meir_keeler()
    fig_ciric_hardy_rogers()
    fig_epsilon_chainable_spaces()
    fig_bessaga_theorem()
    fig_contraction_types_hierarchy()

    print()
    print("All figures generated successfully!")


if __name__ == '__main__':
    main()
