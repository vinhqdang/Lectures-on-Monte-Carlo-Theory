import numpy as np

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
print("\nSubdifferential of |x|:")
for test_x in [-1, -0.5, 0, 0.5, 1]:
    print(f"  ∂f({test_x}) = {subdifferential_abs_value(test_x)}")
