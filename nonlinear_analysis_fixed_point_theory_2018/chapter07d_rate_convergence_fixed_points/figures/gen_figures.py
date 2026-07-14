#!/usr/bin/env python3
"""
Generate figures for Chapter 7d: Variational Methods and Optimization
Converts page images to PDF format for embedding in Beamer slides
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.gridspec import GridSpec

# Set up output directory
output_dir = os.path.dirname(os.path.abspath(__file__))

# Use non-interactive backend
plt.rcParams['backend'] = 'Agg'
plt.rcParams['font.size'] = 10
plt.rcParams['figure.dpi'] = 150

def fig_mountain_pass():
    """Create Mountain Pass Lemma illustration"""
    fig, ax = plt.subplots(figsize=(8, 6))

    # Create a simple landscape for mountain pass visualization
    x = np.linspace(0, 10, 1000)

    # Two valleys with a mountain pass between them
    y1 = 2 + 0.1 * (x - 2)**2  # Valley around x=2
    y2 = 1 + 0.15 * (x - 8)**2  # Valley around x=8
    mountain = 3 * np.exp(-0.3 * (x - 5)**2)  # Mountain at x=5

    landscape = np.minimum(y1, y2) + mountain

    ax.fill_between(x, landscape, 0, alpha=0.3, color='brown', label='Terrain')
    ax.plot(x, landscape, 'k-', linewidth=2)

    # Mark critical points
    ax.plot(2, 2, 'go', markersize=12, label='Valley $u_0$')
    ax.plot(8, 1, 'bo', markersize=12, label='Valley $u_1$')
    ax.plot(5, 3, 'r^', markersize=12, label='Mountain Pass')

    # Add path
    path_x = np.array([2, 5, 8])
    path_y = np.array([2, 3, 1])
    ax.plot(path_x, path_y, 'r--', linewidth=1.5, alpha=0.7)

    ax.set_xlabel('Domain $X$', fontsize=12)
    ax.set_ylabel('Functional $J(u)$', fontsize=12)
    ax.set_title('Mountain Pass Lemma', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 10)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'mountain_pass.pdf'), format='pdf', bbox_inches='tight')
    plt.close()

def fig_variational_principle():
    """Visualize variational principle concept"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    # Left plot: Functional space visualization
    x = np.linspace(-np.pi, np.pi, 1000)
    # Multiple functions converging to minimum
    for k in [0.1, 0.3, 0.5, 0.7, 0.9]:
        y = k * np.sin(x)**2 + (1-k) * np.cos(x)**2
        ax1.plot(x, y, alpha=0.5)

    # Optimal function
    y_opt = np.cos(x)**2
    ax1.plot(x, y_opt, 'r-', linewidth=3, label='Minimizer')
    ax1.fill_between(x, y_opt, alpha=0.2, color='red')

    ax1.set_xlabel('Domain', fontsize=11)
    ax1.set_ylabel('Functional Value', fontsize=11)
    ax1.set_title('Variational Principle:\nFinding Extremals', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Right plot: Euler equation concept
    ax2.text(0.5, 0.8, 'Variational Problem:', ha='center', fontsize=12, fontweight='bold',
            transform=ax2.transAxes)
    ax2.text(0.5, 0.65, r'$\min_{u \in V} J(u) = \int_D F(u, \nabla u) dV$', ha='center', fontsize=11,
            transform=ax2.transAxes, bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))

    ax2.text(0.5, 0.50, 'Leads to Euler Equation:', ha='center', fontsize=12, fontweight='bold',
            transform=ax2.transAxes)
    ax2.text(0.5, 0.35, r'$\frac{\partial F}{\partial u} - \nabla_i \frac{\partial F}{\partial u_i} = 0$', ha='center', fontsize=11,
            transform=ax2.transAxes, bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

    ax2.text(0.5, 0.15, 'Critical Point = Solution', ha='center', fontsize=11,
            transform=ax2.transAxes, style='italic')
    ax2.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'variational_principle.pdf'), format='pdf', bbox_inches='tight')
    plt.close()

def fig_functional_evolution():
    """Show evolution of functional along iterations"""
    fig, ax = plt.subplots(figsize=(8, 6))

    # Simulate convergence of functional
    iterations = np.arange(0, 20)
    # Exponential decay to minimum
    functional_values = 10 * np.exp(-0.3 * iterations) + 1

    ax.semilogy(iterations, functional_values, 'b-o', linewidth=2, markersize=6)
    ax.axhline(y=1, color='r', linestyle='--', linewidth=2, label='Minimum value')
    ax.fill_between(iterations, functional_values, 1, alpha=0.2)

    ax.set_xlabel('Iteration $k$', fontsize=12)
    ax.set_ylabel('Functional Value $J(u_k)$ (log scale)', fontsize=12)
    ax.set_title('Functional Convergence', fontsize=14, fontweight='bold')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(fontsize=11)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'functional_evolution.pdf'), format='pdf', bbox_inches='tight')
    plt.close()

def fig_palais_smale():
    """Illustration of Palais-Smale condition"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    # Left: satisfies PS condition
    x = np.linspace(-3, 3, 1000)
    y = x**2 + 0.1 * np.sin(5*x)
    ax1.plot(x, y, 'b-', linewidth=2)
    ax1.scatter(0, 0, c='red', s=100, marker='*', zorder=5, label='Critical point')
    ax1.fill_between(x, y, alpha=0.2, color='blue')

    ax1.set_xlabel('$u$', fontsize=11)
    ax1.set_ylabel('$J(u)$', fontsize=11)
    ax1.set_title('Satisfies Palais-Smale (PS)', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Right: violates PS condition
    x = np.linspace(-3, 3, 1000)
    y = np.arctan(x**2) + 0.05 * x
    ax2.plot(x, y, 'r-', linewidth=2)
    ax2.fill_between(x, y, alpha=0.2, color='red')

    ax2.set_xlabel('$u$', fontsize=11)
    ax2.set_ylabel('$J(u)$', fontsize=11)
    ax2.set_title('Violates Palais-Smale (PS)', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.text(0, 0.5, 'Sequence converges\nbut no critical point', ha='center',
            transform=ax2.transAxes, fontsize=10, style='italic',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'palais_smale.pdf'), format='pdf', bbox_inches='tight')
    plt.close()

def fig_hamiltonian_phase_space():
    """Phase space diagram for Hamiltonian systems"""
    fig, ax = plt.subplots(figsize=(8, 6))

    # Create phase portrait for a simple Hamiltonian system
    # H(x,p) = p^2/2 + x^2/2
    x = np.linspace(-3, 3, 100)
    p = np.linspace(-3, 3, 100)
    X, P = np.meshgrid(x, p)
    H = P**2/2 + X**2/2

    # Plot level sets of Hamiltonian
    levels = [0.5, 1, 2, 3, 4, 5]
    contour = ax.contour(X, P, H, levels=levels, colors='blue', linewidths=1.5, alpha=0.7)
    ax.clabel(contour, inline=True, fontsize=9)

    # Add direction arrows (Hamiltonian vector field)
    for xi in [-2, -1, 0, 1, 2]:
        for pi in [-2, -1, 0, 1, 2]:
            # Hamiltonian flow: dx/dt = dH/dp = p, dp/dt = -dH/dx = -x
            dx = pi * 0.3
            dp = -xi * 0.3
            ax.arrow(xi, pi, dx, dp, head_width=0.15, head_length=0.1, fc='red', ec='red', alpha=0.6)

    ax.set_xlabel('Position $x$', fontsize=12)
    ax.set_ylabel('Momentum $p$', fontsize=12)
    ax.set_title('Hamiltonian Phase Space: $H(x,p) = p^2/2 + x^2/2$', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'hamiltonian_phase_space.pdf'), format='pdf', bbox_inches='tight')
    plt.close()

def fig_schrodinger_example():
    """Plot Schrödinger equation wave function"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    x = np.linspace(0, 10, 1000)

    # Particle in a box - ground state
    psi_1 = np.sqrt(2/5) * np.sin(np.pi * x / 5)

    # First excited state
    psi_2 = np.sqrt(2/5) * np.sin(2 * np.pi * x / 5)

    # Plot wave functions
    ax1.plot(x, psi_1, 'b-', linewidth=2, label='Ground state $\psi_1$')
    ax1.plot(x, psi_2, 'r--', linewidth=2, label='1st excited state $\psi_2$')
    ax1.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax1.fill_between(x, psi_1, alpha=0.2, color='blue')
    ax1.fill_between(x, psi_2, alpha=0.2, color='red')

    ax1.set_xlabel('Position $x$', fontsize=11)
    ax1.set_ylabel('Wave function $\Psi(x)$', fontsize=11)
    ax1.set_title('Wave Functions: Particle in a Box', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 10)

    # Plot probability densities
    prob_1 = psi_1**2
    prob_2 = psi_2**2

    ax2.fill_between(x, prob_1, alpha=0.5, color='blue', label='$|\psi_1|^2$')
    ax2.fill_between(x, prob_2, alpha=0.5, color='red', label='$|\psi_2|^2$')
    ax2.plot(x, prob_1, 'b-', linewidth=2)
    ax2.plot(x, prob_2, 'r--', linewidth=2)

    ax2.set_xlabel('Position $x$', fontsize=11)
    ax2.set_ylabel('Probability density $|\Psi(x)|^2$', fontsize=11)
    ax2.set_title('Probability Densities', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 10)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'schrodinger_solutions.pdf'), format='pdf', bbox_inches='tight')
    plt.close()

def fig_convergence_analysis():
    """Convergence rates comparison"""
    fig, ax = plt.subplots(figsize=(8, 6))

    n = np.arange(1, 30)

    # Different convergence rates
    linear = 0.9**n
    superlinear = 0.5**n
    quadratic = 0.5**(2*n)

    ax.semilogy(n, linear, 'b-o', linewidth=2, markersize=5, label='Linear convergence')
    ax.semilogy(n, superlinear, 'g-s', linewidth=2, markersize=5, label='Superlinear convergence')
    ax.semilogy(n, quadratic, 'r-^', linewidth=2, markersize=5, label='Quadratic convergence')

    ax.set_xlabel('Iteration $n$', fontsize=12)
    ax.set_ylabel('Error $||e_n||$ (log scale)', fontsize=12)
    ax.set_title('Convergence Rates Comparison', fontsize=14, fontweight='bold')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(fontsize=11)
    ax.set_xlim(0, 30)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'convergence_rates.pdf'), format='pdf', bbox_inches='tight')
    plt.close()

def main():
    """Generate all figures"""
    print("Generating figures for Chapter 7d...")

    fig_mountain_pass()
    print("  - Generated: mountain_pass.pdf")

    fig_variational_principle()
    print("  - Generated: variational_principle.pdf")

    fig_functional_evolution()
    print("  - Generated: functional_evolution.pdf")

    fig_palais_smale()
    print("  - Generated: palais_smale.pdf")

    fig_hamiltonian_phase_space()
    print("  - Generated: hamiltonian_phase_space.pdf")

    fig_schrodinger_example()
    print("  - Generated: schrodinger_solutions.pdf")

    fig_convergence_analysis()
    print("  - Generated: convergence_rates.pdf")

    print("Done! All figures saved to figures/")

if __name__ == '__main__':
    main()
