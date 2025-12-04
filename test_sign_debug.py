"""
Debug sign storage and retrieval
"""

import numpy as np
from Exp_dataset import Exponential_OutputNormalizer

# Create small test data
np.random.seed(42)
test_data = np.array([
    [100.0, -50.0, 25.0],   # Sample 0: [+, -, +]
    [-200.0, 150.0, -75.0],  # Sample 1: [-, +, -]
    [300.0, -250.0, 125.0],  # Sample 2: [+, -, +]
])

print("="*70)
print("Test Data:")
print("="*70)
print(test_data)

# Fit normalizer
normalizer = Exponential_OutputNormalizer(use_log_normalization=True)
normalizer.fit(test_data)

print("\n" + "="*70)
print("Stored Signs:")
print("="*70)
print(f"sign['x']: {normalizer.sign['x']}")
print(f"sign['v']: {normalizer.sign['v']}")
print(f"sign['a']: {normalizer.sign['a']}")

# Normalize
normalized = normalizer.transform(test_data)
print("\n" + "="*70)
print("Normalized:")
print("="*70)
print(normalized)

# Denormalize
reconstructed = normalizer.inverse_transform(normalized)
print("\n" + "="*70)
print("Reconstructed:")
print("="*70)
print(reconstructed)

# Check errors
print("\n" + "="*70)
print("Comparison:")
print("="*70)
for i in range(3):
    print(f"\nSample {i}:")
    print(f"  Original:      {test_data[i]}")
    print(f"  Reconstructed: {reconstructed[i]}")
    print(f"  Difference:    {test_data[i] - reconstructed[i]}")
    print(f"  Rel Error:     {np.abs((test_data[i] - reconstructed[i]) / (test_data[i] + 1e-10))}")
