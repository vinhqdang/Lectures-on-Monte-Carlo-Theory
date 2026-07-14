#!/usr/bin/env python3
"""
Generate figures for Chapter 8: Convex Functions
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Polygon
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('default')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['font.size'] = 10

def figure_convex_definition():
    """
    Figure 8.1: Convex function illustration
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    # Create a convex function
    x = np.linspace(-3, 3, 200)
    y = 0.3 * x**2 - 1

    ax.plot(x, y, 'b-', linewidth=2.5, label='f(x)')

    # Shade the epigraph
    x_fill = np.linspace(-3, 3, 200)
    y_fill_lower = 0.3 * x_fill**2 - 1
    y_fill_upper = 5
    ax.fill_between(x_fill, y_fill_lower, y_fill_upper, alpha=0.2, color='blue')

    # Add sample points and secant line
    x1, x2 = -2.0, 1.5
    y1 = 0.3 * x1**2 - 1
    y2 = 0.3 * x2**2 - 1

    ax.plot([x1, x2], [y1, y2], 'ro', markersize=8)

    # Draw secant line
    x_secant = np.linspace(x1, x2, 100)
    y_secant = y1 + (y2 - y1) * (x_secant - x1) / (x2 - x1)
    ax.plot(x_secant, y_secant, 'r--', linewidth=2, alpha=0.7, label='Secant line')

    ax.set_xlim(-3.5, 3.5)
    ax.set_ylim(-2, 4)
    ax.set_xlabel('x', fontsize=12, fontweight='bold')
    ax.set_ylabel('f(x)', fontsize=12, fontweight='bold')
    ax.set_title('Convex Function: epi f is a convex set', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.axvline(x=0, color='k', linewidth=0.5)
    ax.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig('convex_definition.pdf', dpi=150, bbox_inches='tight')
    plt.close()
    print("Created: convex_definition.pdf")

def figure_strictly_convex():
    """
    Strictly convex vs convex functions
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Convex (not strictly)
    x = np.linspace(0, 2, 200)
    y = np.abs(x - 1)
    ax1.plot(x, y, 'b-', linewidth=2.5)
    x1, x2 = 0.3, 1.7
    y1, y2 = np.abs(x1 - 1), np.abs(x2 - 1)
    ax1.plot([x1, x2], [y1, y2], 'ro', markersize=8)
    x_secant = np.linspace(x1, x2, 100)
    y_secant = y1 + (y2 - y1) * (x_secant - x1) / (x2 - x1)
    ax1.plot(x_secant, y_secant, 'r--', linewidth=2, alpha=0.7)
    ax1.fill_between(x_secant, y_secant, np.abs(x_secant - 1), alpha=0.2, color='red')
    ax1.set_title('Convex (not strictly)', fontsize=11, fontweight='bold')
    ax1.set_xlabel('x')
    ax1.set_ylabel('f(x)')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-0.1, 1.2)

    # Strictly convex
    x = np.linspace(-2, 2, 200)
    y = x**2
    ax2.plot(x, y, 'g-', linewidth=2.5)
    x1, x2 = -1.2, 1.0
    y1, y2 = x1**2, x2**2
    ax2.plot([x1, x2], [y1, y2], 'ro', markersize=8)
    x_secant = np.linspace(x1, x2, 100)
    y_secant = y1 + (y2 - y1) * (x_secant - x1) / (x2 - x1)
    ax2.plot(x_secant, y_secant, 'r--', linewidth=2, alpha=0.7)
    ax2.fill_between(x_secant, x_secant**2, y_secant, alpha=0.2, color='green')
    ax2.set_title('Strictly Convex', fontsize=11, fontweight='bold')
    ax2.set_xlabel('x')
    ax2.set_ylabel('f(x)')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-0.5, 4)

    plt.tight_layout()
    plt.savefig('strictly_convex.pdf', dpi=150, bbox_inches='tight')
    plt.close()
    print("Created: strictly_convex.pdf")

def figure_norms():
    """
    Convexity of norms
    """
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    # L2 norm (Euclidean)
    ax = axes[0, 0]
    x = np.linspace(-2, 2, 200)
    y = np.abs(x)
    ax.plot(x, y, 'b-', linewidth=2.5, label='L2 norm: ||x||')
    ax.fill_between(x, 0, y, alpha=0.1, color='blue')
    ax.set_title('L2 Norm (Euclidean)', fontsize=11, fontweight='bold')
    ax.set_xlabel('x')
    ax.set_ylabel('||x||')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.axvline(x=0, color='k', linewidth=0.5)

    # Squared norm (strictly convex)
    ax = axes[0, 1]
    x = np.linspace(-2, 2, 200)
    y = x**2
    ax.plot(x, y, 'g-', linewidth=2.5, label='||x||^2')
    ax.fill_between(x, 0, y, alpha=0.1, color='green')
    ax.set_title('Squared L2 Norm (Strictly Convex)', fontsize=11, fontweight='bold')
    ax.set_xlabel('x')
    ax.set_ylabel('||x||^2')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.axvline(x=0, color='k', linewidth=0.5)

    # Level sets of convex function
    ax = axes[1, 0]
    x = np.linspace(-3, 3, 100)
    y = np.linspace(-3, 3, 100)
    X, Y = np.meshgrid(x, y)
    Z = X**2 + Y**2
    contours = ax.contour(X, Y, Z, levels=[0.5, 1, 2, 4], colors='blue')
    ax.clabel(contours, inline=True, fontsize=8)
    ax.set_title('Level Sets of Convex f(x,y) = x² + y²', fontsize=11, fontweight='bold')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.grid(True, alpha=0.3)

    # Minkowski gauge
    ax = axes[1, 1]
    theta = np.linspace(0, 2*np.pi, 100)
    r = 1 + 0.3*np.sin(4*theta)  # Slightly irregular convex set
    x_circle = r * np.cos(theta)
    y_circle = r * np.sin(theta)
    ax.fill(x_circle, y_circle, alpha=0.2, color='purple')
    ax.plot(x_circle, y_circle, 'p-', linewidth=2, color='purple', label='Convex set C')
    ax.plot([0], [0], 'ko', markersize=8)
    ax.text(0.1, 0.1, 'o ∈ C', fontsize=10)
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    ax.set_title('Minkowski Gauge m_C(x)', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.axvline(x=0, color='k', linewidth=0.5)
    ax.legend()

    plt.tight_layout()
    plt.savefig('norms_and_level_sets.pdf', dpi=150, bbox_inches='tight')
    plt.close()
    print("Created: norms_and_level_sets.pdf")

def figure_divergences():
    """
    Various divergence functions
    """
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))

    x = np.linspace(0.1, 3, 200)

    # Kullback-Leibler
    ax = axes[0, 0]
    y = x * np.log(x) - x + 1
    ax.plot(x, y, 'b-', linewidth=2.5)
    ax.fill_between(x, 0, y, alpha=0.1, color='blue')
    ax.set_title('KL Divergence: x ln(x) - x + 1', fontsize=10, fontweight='bold')
    ax.set_xlabel('x')
    ax.set_ylabel('d_KL(x, 1)')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.5, 2)

    # Hellinger
    ax = axes[0, 1]
    y = (np.sqrt(x) - 1)**2
    ax.plot(x, y, 'g-', linewidth=2.5)
    ax.fill_between(x, 0, y, alpha=0.1, color='green')
    ax.set_title('Hellinger: (√x - 1)²', fontsize=10, fontweight='bold')
    ax.set_xlabel('x')
    ax.set_ylabel('d_H(x, 1)')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.5, 2)

    # Pearson (chi-squared)
    ax = axes[0, 2]
    y = (x - 1)**2 / 1
    ax.plot(x, y, 'r-', linewidth=2.5)
    ax.fill_between(x, 0, y, alpha=0.1, color='red')
    ax.set_title('Pearson Chi-squared: (x - 1)²', fontsize=10, fontweight='bold')
    ax.set_xlabel('x')
    ax.set_ylabel('d_P(x, 1)')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.5, 4)

    # Jeffreys
    ax = axes[1, 0]
    y = (x - 1) * np.log(x)
    ax.plot(x, y, 'purple', linewidth=2.5)
    ax.fill_between(x, 0, y, alpha=0.1, color='purple')
    ax.set_title('Jeffreys: (x - 1) ln(x)', fontsize=10, fontweight='bold')
    ax.set_xlabel('x')
    ax.set_ylabel('d_J(x, 1)')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-1, 2)

    # Rényi (α=0.5)
    ax = axes[1, 1]
    alpha_val = 0.5
    y = x**alpha_val
    ax.plot(x, y, 'orange', linewidth=2.5)
    ax.fill_between(x, 0, y, alpha=0.1, color='orange')
    ax.set_title(f'Rényi (α={alpha_val}): x^α', fontsize=10, fontweight='bold')
    ax.set_xlabel('x')
    ax.set_ylabel('d_R(x, 1)')
    ax.grid(True, alpha=0.3)

    # Csiszár divergence
    ax = axes[1, 2]
    phi = lambda t: t * np.log(t) if t > 0 else 0
    y = np.array([phi(xi) for xi in x])
    ax.plot(x, y, 'brown', linewidth=2.5)
    ax.fill_between(x, 0, y, alpha=0.1, color='brown')
    ax.set_title('Csiszár: φ(t) = t ln(t)', fontsize=10, fontweight='bold')
    ax.set_xlabel('x')
    ax.set_ylabel('d_φ(x, 1)')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('divergences.pdf', dpi=150, bbox_inches='tight')
    plt.close()
    print("Created: divergences.pdf")

def figure_huber():
    """
    Huber function
    """
    fig, ax = plt.subplots(figsize=(9, 6))

    rho = 1.0
    x = np.linspace(-3, 3, 300)
    y = np.where(np.abs(x) <= rho,
                 x**2 / 2,
                 rho * np.abs(x) - rho**2 / 2)

    ax.plot(x, y, 'b-', linewidth=2.5, label=f'Huber(ρ={rho})')
    ax.fill_between(x, 0, y, alpha=0.1, color='blue')

    # Mark the transition point
    ax.plot([-rho, rho], [rho**2/2, rho**2/2], 'ro', markersize=8)
    ax.axvline(x=-rho, color='r', linestyle='--', alpha=0.5, linewidth=1)
    ax.axvline(x=rho, color='r', linestyle='--', alpha=0.5, linewidth=1)

    ax.set_xlabel('x', fontsize=12, fontweight='bold')
    ax.set_ylabel('f(x)', fontsize=12, fontweight='bold')
    ax.set_title('Huber Loss Function: Convex and Continuous', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.axvline(x=0, color='k', linewidth=0.5)
    ax.legend(fontsize=11)
    ax.set_xlim(-3, 3)

    plt.tight_layout()
    plt.savefig('huber_function.pdf', dpi=150, bbox_inches='tight')
    plt.close()
    print("Created: huber_function.pdf")

def figure_jensen_inequality():
    """
    Jensen's inequality illustration
    """
    fig, ax = plt.subplots(figsize=(9, 6))

    # Convex function
    x = np.linspace(-2, 2, 200)
    y = x**2

    ax.plot(x, y, 'b-', linewidth=2.5, label='Convex function f')

    # Two points and their convex combination
    x1, x2 = -1.2, 1.5
    alpha = 0.4
    x_mid = alpha * x1 + (1 - alpha) * x2

    y1 = x1**2
    y2 = x2**2
    y_mid = x_mid**2
    y_convex_mid = alpha * y1 + (1 - alpha) * y2

    # Plot points
    ax.plot([x1, x2], [y1, y2], 'ro', markersize=10, label=f'x₁, x₂')
    ax.plot([x_mid], [y_mid], 'gs', markersize=10, label=f'f(αx₁ + (1-α)x₂)')
    ax.plot([x_mid], [y_convex_mid], 'bs', markersize=10, label=f'αf(x₁) + (1-α)f(x₂)')

    # Draw secant line
    x_sec = np.linspace(x1, x2, 100)
    y_sec = y1 + (y2 - y1) * (x_sec - x1) / (x2 - x1)
    ax.plot(x_sec, y_sec, 'r--', linewidth=1.5, alpha=0.7)

    # Draw vertical line showing inequality
    ax.plot([x_mid, x_mid], [y_mid, y_convex_mid], 'k--', linewidth=1.5, alpha=0.7)
    ax.annotate('', xy=(x_mid + 0.2, y_convex_mid), xytext=(x_mid + 0.2, y_mid),
                arrowprops=dict(arrowstyle='<->', color='black', lw=1.5))

    ax.set_xlabel('x', fontsize=12, fontweight='bold')
    ax.set_ylabel('f(x)', fontsize=12, fontweight='bold')
    ax.set_title("Jensen's Inequality: f(αx + (1-α)y) ≤ αf(x) + (1-α)f(y)",
                 fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc='upper left')
    ax.set_xlim(-2, 2)

    plt.tight_layout()
    plt.savefig('jensen_inequality.pdf', dpi=150, bbox_inches='tight')
    plt.close()
    print("Created: jensen_inequality.pdf")

def figure_lipschitz_continuity():
    """
    Lipschitz continuity illustration
    """
    fig, ax = plt.subplots(figsize=(9, 6))

    # Convex function
    x = np.linspace(-2, 3, 200)
    y = np.sqrt(np.maximum(x, 0))  # Convex but not Lipschitz at 0

    ax.plot(x, y, 'b-', linewidth=2.5, label='f(x) = √x (not Lipschitz at 0)')

    # Lipschitz continuous region
    x_lipschitz = np.linspace(0.5, 3, 200)
    y_lipschitz = np.sqrt(x_lipschitz)

    # Draw Lipschitz cone at a point
    x0 = 1.0
    y0 = np.sqrt(x0)
    K = 0.3  # Lipschitz constant

    x_cone = np.linspace(0.5, 2.5, 100)
    y_upper = y0 + K * (x_cone - x0)
    y_lower = y0 - K * (x_cone - x0)

    ax.fill_between(x_cone, y_lower, y_upper, alpha=0.15, color='red', label='Lipschitz cone')
    ax.plot(x_cone, y_upper, 'r--', linewidth=1, alpha=0.7)
    ax.plot(x_cone, y_lower, 'r--', linewidth=1, alpha=0.7)

    ax.plot([x0], [y0], 'ko', markersize=8)
    ax.text(x0 + 0.1, y0 + 0.2, f'x₀', fontsize=10)

    ax.set_xlabel('x', fontsize=12, fontweight='bold')
    ax.set_ylabel('f(x)', fontsize=12, fontweight='bold')
    ax.set_title('Lipschitz Continuity: |f(x) - f(y)| ≤ L||x - y||',
                 fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    ax.set_xlim(-0.5, 3)
    ax.set_ylim(-0.5, 2)

    plt.tight_layout()
    plt.savefig('lipschitz_continuity.pdf', dpi=150, bbox_inches='tight')
    plt.close()
    print("Created: lipschitz_continuity.pdf")

def figure_composition():
    """
    Composition of convex functions
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Inner function (convex)
    x = np.linspace(-2, 2, 200)
    ax = axes[0]
    f_x = x**2
    ax.plot(x, f_x, 'b-', linewidth=2.5, label='f(x) = x²')
    ax.fill_between(x, 0, f_x, alpha=0.1, color='blue')
    ax.set_xlabel('x')
    ax.set_ylabel('f(x)')
    ax.set_title('Step 1: Convex f(x)', fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Outer function (increasing convex)
    ax = axes[1]
    t = np.linspace(0, 4, 200)
    phi_t = np.exp(t) - 1
    ax.plot(t, phi_t, 'g-', linewidth=2.5, label='φ(t) = exp(t) - 1')
    ax.fill_between(t, 0, phi_t, alpha=0.1, color='green')
    ax.set_xlabel('t')
    ax.set_ylabel('φ(t)')
    ax.set_title('Step 2: Increasing Convex φ(t)', fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Composition
    ax = axes[2]
    x = np.linspace(-2, 2, 200)
    composition = np.exp(x**2) - 1
    ax.plot(x, composition, 'r-', linewidth=2.5, label='(φ ∘ f)(x) = exp(x²) - 1')
    ax.fill_between(x, 0, composition, alpha=0.1, color='red')
    ax.set_xlabel('x')
    ax.set_ylabel('(φ ∘ f)(x)')
    ax.set_title('Result: Convex Composition', fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig('composition_convex.pdf', dpi=150, bbox_inches='tight')
    plt.close()
    print("Created: composition_convex.pdf")

if __name__ == '__main__':
    print("Generating figures for Chapter 8: Convex Functions...")
    figure_convex_definition()
    figure_strictly_convex()
    figure_norms()
    figure_divergences()
    figure_huber()
    figure_jensen_inequality()
    figure_lipschitz_continuity()
    figure_composition()
    print("\nAll figures generated successfully!")
