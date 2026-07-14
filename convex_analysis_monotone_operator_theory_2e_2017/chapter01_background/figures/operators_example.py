# Example 1: Single-valued operators
def single_valued_operator(x):
    """Linear operator T: R -> R, T(x) = 2x"""
    return 2 * x

# Example 2: Set-valued operator
def set_valued_operator(x):
    """Set-valued operator A: R -> 2^R
    A(x) = [-|x|, |x|] (the closed interval centered at 0)
    """
    return {"min": -abs(x), "max": abs(x)}

# Example 3: Composition of operators
def operator_composition(x, T, A):
    """Compute (A composed with T)(x)
    First apply T, then apply A to the result"""
    tx = T(x)
    return A(tx)

# Test the operators
x_test = 3.0
print("Single-valued operator:")
print(f"  T({x_test}) = {single_valued_operator(x_test)}")

print("\nSet-valued operator:")
A_result = set_valued_operator(x_test)
print(f"  A({x_test}) = [{A_result['min']}, {A_result['max']}]")

print("\nComposition (A ∘ T):")
composition = operator_composition(x_test, single_valued_operator, set_valued_operator)
print(f"  (A ∘ T)({x_test}) = [{composition['min']}, {composition['max']}]")
