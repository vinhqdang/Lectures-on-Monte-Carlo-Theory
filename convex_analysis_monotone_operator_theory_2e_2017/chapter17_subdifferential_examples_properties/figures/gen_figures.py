#!/usr/bin/env python3
"""
Generate figures and code examples for Chapter 17: Differentiability of Convex Functions
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# Set up plotting style
rcParams['figure.figsize'] = (10, 6)
rcParams['font.size'] = 10
rcParams['lines.linewidth'] = 2
rcParams['axes.grid'] = True
rcParams['grid.alpha'] = 0.3

def save_pdf(filename):
    """Save figure as PDF"""
    filepath = f"{filename}.pdf"
    plt.savefig(filepath, format='pdf', bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Saved: {filepath}")

def figure_directional_derivative():
    """Figure: Directional derivative concept"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: Directional derivative
    x = np.linspace(-2, 2, 200)
    y = x**2

    ax1.plot(x, y, 'b-', label='$f(x) = x^2$', linewidth=2)

    # Point and direction
    x0 = 0.5
    f_x0 = x0**2
    direction = 1

    # Tangent line showing directional derivative
    alpha_vals = np.array([0.1, 0.2, 0.3, 0.4])
    for alpha in alpha_vals:
        x_new = x0 + alpha * direction
        f_new = x_new**2
        ax1.plot([x0, x_new], [f_x0, f_new], 'r--', alpha=0.3)

    ax1.plot(x0, f_x0, 'ko', markersize=8, label=f'Point $(x_0, f(x_0))$')
    ax1.arrow(x0, f_x0, direction*0.3, 0, head_width=0.05, head_length=0.1, fc='green', ec='green')
    ax1.set_xlabel('$x$')
    ax1.set_ylabel('$f(x)$')
    ax1.set_title('Directional Derivative Concept')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Convex vs non-convex
    x = np.linspace(0, 3, 200)

    # Convex function
    y_convex = np.exp(x)
    ax2.plot(x, y_convex, 'b-', label='Convex: $e^x$', linewidth=2)

    # Tangent line at a point
    x_pt = 1
    y_pt = np.exp(x_pt)
    slope = np.exp(x_pt)
    x_tangent = np.linspace(0.5, 2.5, 100)
    y_tangent = y_pt + slope * (x_tangent - x_pt)
    ax2.plot(x_tangent, y_tangent, 'r--', label='Tangent line', linewidth=2)
    ax2.plot(x_pt, y_pt, 'ko', markersize=8)

    ax2.set_xlabel('$x$')
    ax2.set_ylabel('$f(x)$')
    ax2.set_title('Tangent Line Below Convex Function')
    ax2.set_ylim([0, 10])
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    save_pdf('directional_derivative')

def figure_convexity_characterizations():
    """Figure: Convexity vs gradient monotonicity"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: Convex function and its gradient
    x = np.linspace(-2, 2, 200)
    y = x**2
    gradient = 2*x

    ax1_twin = ax1.twinx()

    line1 = ax1.plot(x, y, 'b-', label='$f(x) = x^2$', linewidth=2)
    line2 = ax1_twin.plot(x, gradient, 'r-', label="$f'(x) = 2x$", linewidth=2)

    ax1.set_xlabel('$x$')
    ax1.set_ylabel('$f(x)$ (blue)', color='b')
    ax1_twin.set_ylabel("$f'(x)$ (red)", color='r')
    ax1.set_title('Convex Function and Monotone Gradient')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1_twin.tick_params(axis='y', labelcolor='r')
    ax1.grid(True, alpha=0.3)

    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left')

    # Plot 2: Strict vs non-strict convexity
    x = np.linspace(-2, 2, 200)
    y_strict = x**2  # Strictly convex
    y_nonstrict = np.abs(x)  # Non-strictly convex (corner at 0)

    ax2.plot(x, y_strict, 'b-', label='Strictly convex: $x^2$', linewidth=2)
    ax2.plot(x, y_nonstrict, 'r-', label='Non-strictly convex: $|x|$', linewidth=2)
    ax2.set_xlabel('$x$')
    ax2.set_ylabel('$f(x)$')
    ax2.set_title('Strict vs Non-Strict Convexity')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    save_pdf('convexity_characterizations')

def figure_subdifferential():
    """Figure: Subdifferential illustration"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: Subdifferential of smooth function (gradient)
    x = np.linspace(-2, 2, 200)
    y = x**2

    ax1.plot(x, y, 'b-', label='$f(x) = x^2$', linewidth=2)

    # Point where we compute subdifferential
    x0 = 0.5
    f_x0 = x0**2
    grad = 2*x0

    # Tangent line (supporting hyperplane in 1D)
    x_tang = np.linspace(-1, 2, 100)
    y_tang = f_x0 + grad * (x_tang - x0)
    ax1.plot(x_tang, y_tang, 'r--', label='Supporting hyperplane', linewidth=2)
    ax1.plot(x0, f_x0, 'ko', markersize=8)
    ax1.text(x0 + 0.1, f_x0 - 0.1, f'$x_0 = {x0}$', fontsize=10)

    ax1.set_xlabel('$x$')
    ax1.set_ylabel('$f(x)$')
    ax1.set_title('Subdifferential of Smooth Function\n$\\partial f(x_0) = \\{\\nabla f(x_0)\\}$')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([-0.5, 4])

    # Plot 2: Subdifferential of non-smooth function
    x = np.linspace(-2, 2, 200)
    y = np.abs(x)  # Absolute value function

    ax2.plot(x, y, 'b-', linewidth=2.5, label='$f(x) = |x|$')

    # Subdifferential at 0
    x0 = 0
    f_x0 = 0

    # All supporting hyperplanes at origin (subgradients in [-1, 1])
    slopes = np.linspace(-1, 1, 5)
    for slope in slopes:
        x_tang = np.linspace(-2, 2, 100)
        y_tang = f_x0 + slope * (x_tang - x0)
        ax2.plot(x_tang, y_tang, 'r-', alpha=0.3, linewidth=1)

    ax2.plot(0, 0, 'ko', markersize=10)
    ax2.text(0.2, 0.2, '$\\partial f(0) = [-1, 1]$', fontsize=11,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax2.set_xlabel('$x$')
    ax2.set_ylabel('$f(x)$')
    ax2.set_title('Subdifferential of Non-Smooth Function')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([-0.5, 2])

    plt.tight_layout()
    save_pdf('subdifferential')

def figure_descent_directions():
    """Figure: Descent directions"""
    fig, ax = plt.subplots(figsize=(10, 8))

    # Create contour plot of a convex function
    x = np.linspace(-3, 3, 200)
    y = np.linspace(-3, 3, 200)
    X, Y = np.meshgrid(x, y)
    Z = X**2 + 2*Y**2  # Elliptic paraboloid

    contours = ax.contour(X, Y, Z, levels=15, colors='blue', alpha=0.6)
    ax.clabel(contours, inline=True, fontsize=8)

    # Current point
    x_curr = np.array([1.5, 1.5])
    ax.plot(x_curr[0], x_curr[1], 'ko', markersize=10, label='Current point')

    # Gradient (direction of steepest ascent)
    grad = np.array([2*x_curr[0], 4*x_curr[1]])
    grad_normalized = grad / np.linalg.norm(grad)

    # Steepest descent direction
    descent = -grad_normalized

    # Descent direction
    ax.arrow(x_curr[0], x_curr[1], descent[0]*0.7, descent[1]*0.7,
             head_width=0.15, head_length=0.15, fc='red', ec='red', linewidth=2,
             label='Steepest descent')

    # Gradient direction (for reference)
    ax.arrow(x_curr[0], x_curr[1], grad_normalized[0]*0.7, grad_normalized[1]*0.7,
             head_width=0.15, head_length=0.15, fc='green', ec='green', linewidth=2,
             alpha=0.5, label='Gradient (ascent)')

    ax.set_xlabel('$x_1$', fontsize=12)
    ax.set_ylabel('$x_2$', fontsize=12)
    ax.set_title('Descent and Gradient Directions\n$f(x_1, x_2) = x_1^2 + 2x_2^2$', fontsize=12)
    ax.set_xlim([-3, 3])
    ax.set_ylim([-3, 3])
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    plt.tight_layout()
    save_pdf('descent_directions')

def figure_gateau_frechet():
    """Figure: Gâteaux vs Fréchet differentiability"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: Fréchet differentiable (smooth) function
    x = np.linspace(-2, 2, 200)
    y = np.sin(x) + 0.5*x**2

    ax1.plot(x, y, 'b-', label='$f(x) = \\sin(x) + 0.5x^2$', linewidth=2.5)

    # Tangent line at a point
    x_pt = 0.5
    y_pt = np.sin(x_pt) + 0.5*x_pt**2
    slope = np.cos(x_pt) + x_pt  # Derivative

    x_tangent = np.linspace(-0.5, 1.5, 100)
    y_tangent = y_pt + slope * (x_tangent - x_pt)

    ax1.plot(x_tangent, y_tangent, 'r--', label='Tangent line', linewidth=2)
    ax1.plot(x_pt, y_pt, 'ko', markersize=8)
    ax1.fill_between(x_tangent, y_tangent,
                     y_pt + slope * (x_tangent - x_pt) - 0.1,
                     alpha=0.1, color='red')

    ax1.text(0.6, -0.5, 'Fréchet differentiable:\n$f$ has a unique linear\napproximation',
             fontsize=10, bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))

    ax1.set_xlabel('$x$', fontsize=11)
    ax1.set_ylabel('$f(x)$', fontsize=11)
    ax1.set_title('Fréchet Differentiability', fontsize=11)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Gâteaux differentiable but not Fréchet
    x = np.linspace(-2, 2, 200)
    # A function that is "pointy" - Gâteaux but not Fréchet differentiable
    y = np.abs(x) + 0.1*x**2

    ax2.plot(x, y, 'b-', linewidth=2.5, label='Gâteaux but not Fréchet')

    # Multiple "tangent" directions
    x_pt = 0
    y_pt = 0
    slopes_different = [-0.8, 0, 0.8]
    colors_diff = ['red', 'green', 'orange']

    for slope, col in zip(slopes_different, colors_diff):
        x_tangent = np.linspace(-1.5, 1.5, 100)
        y_tangent = y_pt + slope * (x_tangent - x_pt)
        ax2.plot(x_tangent, y_tangent, '--', color=col, alpha=0.4, linewidth=1.5)

    ax2.plot(x_pt, y_pt, 'ko', markersize=8)
    ax2.text(0.5, 1.2, 'Gâteaux but not Fréchet:\n$f$ has multiple directional\nderivatives',
             fontsize=10, bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))

    ax2.set_xlabel('$x$', fontsize=11)
    ax2.set_ylabel('$f(x)$', fontsize=11)
    ax2.set_title('Gâteaux Differentiability', fontsize=11)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([-0.5, 3])

    plt.tight_layout()
    save_pdf('gateau_frechet')

def create_python_code_examples():
    """Create Python code example files referenced in slides"""

    # Code 1: Computing subdifferentials
    code1 = '''import numpy as np

def subdifferential_abs_value(x, eps=1e-10):
    """
    Compute subdifferential of f(x) = |x|

    Example: f(x) = |x|
    - For x > 0: ∂f(x) = {1}
    - For x < 0: ∂f(x) = {-1}
    - For x = 0: ∂f(0) = [-1, 1]
    """
    if abs(x) < eps:
        # Return interval [-1, 1] as sample
        return np.array([-1.0, 1.0])
    elif x > 0:
        return np.array([1.0])
    else:
        return np.array([-1.0])

def directional_derivative_convex(f, x, y, eps=1e-8):
    """
    Compute directional derivative f'(x; y) numerically
    using: f'(x; y) = inf_{α>0} [f(x + αy) - f(x)] / α
    """
    alphas = np.logspace(-8, 0, 100)
    diffs = [(f(x + a*y) - f(x))/a for a in alphas]
    return np.min(diffs)

# Example with f(x) = x^2
f = lambda x: x**2
x0 = 1.0
y = 2.0

f_prime = directional_derivative_convex(f, x0, y)
print(f"f'({x0}; {y}) ≈ {f_prime:.6f}")

# Subdifferential examples
print("\\nSubdifferential of |x|:")
for test_x in [-1, -0.5, 0, 0.5, 1]:
    print(f"  ∂f({test_x}) = {subdifferential_abs_value(test_x)}")
'''

    with open('code_subdiff.py', 'w') as f:
        f.write(code1)

    # Code 2: Convex function analysis
    code2 = '''import numpy as np
import matplotlib.pyplot as plt

class ConvexFunction:
    """Analysis of convex functions"""

    def __init__(self, f, grad_f, name="f"):
        """
        f: function
        grad_f: gradient function
        """
        self.f = f
        self.grad_f = grad_f
        self.name = name

    def is_monotone_gradient(self, x_vals):
        """Check if gradient is monotone"""
        grads = [self.grad_f(x) for x in x_vals]
        return all(grads[i] <= grads[i+1]
                  for i in range(len(grads)-1))

    def tangent_line(self, x0, x):
        """Compute tangent line at x0"""
        f_x0 = self.f(x0)
        grad = self.grad_f(x0)
        return f_x0 + grad * (x - x0)

    def verify_convexity(self, x_vals):
        """Verify convexity using first-order condition"""
        for x in x_vals:
            tangent = self.tangent_line(x, x_vals)
            f_vals = np.array([self.f(xi) for xi in x_vals])
            if not np.all(f_vals >= tangent - 1e-10):
                return False
        return True

# Example 1: f(x) = x^2
f1 = lambda x: x**2
grad_f1 = lambda x: 2*x

func1 = ConvexFunction(f1, grad_f1, "f(x) = x²")
x_test = np.linspace(-2, 2, 100)

print(f"Function: {func1.name}")
print(f"Gradient is monotone: {func1.is_monotone_gradient(x_test)}")
print(f"Function is convex: {func1.verify_convexity(x_test)}")

# Example 2: f(x) = e^x
f2 = lambda x: np.exp(x)
grad_f2 = lambda x: np.exp(x)

func2 = ConvexFunction(f2, grad_f2, "f(x) = exp(x)")
print(f"\\nFunction: {func2.name}")
print(f"Gradient is monotone: {func2.is_monotone_gradient(x_test)}")
print(f"Function is convex: {func2.verify_convexity(x_test)}")
'''

    with open('code_convex_analysis.py', 'w') as f:
        f.write(code2)

def main():
    """Generate all figures"""
    print("Generating figures for Chapter 17...")

    figure_directional_derivative()
    figure_convexity_characterizations()
    figure_subdifferential()
    figure_descent_directions()
    figure_gateau_frechet()

    print("\nGenerating Python code examples...")
    create_python_code_examples()

    print("\nAll figures and code examples generated successfully!")

if __name__ == "__main__":
    main()
