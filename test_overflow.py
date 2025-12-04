import numpy as np
import warnings
from datagtgenerator import analytical_solution

# Test with parameters that likely cause overflow
test_cases = [
    (100.0, 0.1, 1000.0, 50.0, -10.0, 100.0),  # Large m, k, t - likely overflow
    (0.01, 10.0, 100.0, 100.0, 50.0, 1000.0),  # High zeta, large t - overdamped case
    (1.0, 5.0, 500.0, 25.0, 75.0, 500.0),      # Another potential overflow case
]

print("Testing cases that might cause overflow:")
for i, (m, zeta, k, x0, v0, t) in enumerate(test_cases):
    c = 2 * zeta * np.sqrt(m * k)
    print(f"\nTest case {i+1}:")
    print(f"  m={m}, zeta={zeta}, k={k}, c={c:.3f}, x0={x0}, v0={v0}, t={t}")
    
    # Capture warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        
        x_t, v_t, a_t = analytical_solution(m, c, k, x0, v0, t)
        
        if w:
            print(f"  *** WARNINGS CAUGHT ***")
            for warning in w:
                print(f"    {warning.category.__name__}: {warning.message}")
                print(f"    File: {warning.filename}, Line: {warning.lineno}")
        
        print(f"  Results: x_t={x_t}, v_t={v_t}, a_t={a_t}")
        print(f"  Finite check: x_t={np.isfinite(x_t)}, v_t={np.isfinite(v_t)}, a_t={np.isfinite(a_t)}")