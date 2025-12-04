"""
Simple test for Exponential_OutputNormalizer normalize-denormalize round-trip
"""

import numpy as np
from Exp_dataset import Exponential_OutputNormalizer

# Load the exponential test data
data_path = r'E:\Ian\PINNexample\exponential_test_data.npz'
data = np.load(data_path)

# Extract the array
array_name = list(data.keys())[0]
data_array = data[array_name]

# Get outputs (targets): [x_t, v_t, a_t]
targets = data_array[:, 3:]  # Outputs: x_t, v_t, a_t

print("="*70)
print("Testing Exponential_OutputNormalizer Round-Trip")
print("="*70)
print(f"\nLoaded data shape: {targets.shape}")
print(f"Data range:")
print(f"  x_t: [{targets[:, 0].min():.2e}, {targets[:, 0].max():.2e}]")
print(f"  v_t: [{targets[:, 1].min():.2e}, {targets[:, 1].max():.2e}]")
print(f"  a_t: [{targets[:, 2].min():.2e}, {targets[:, 2].max():.2e}]")

# Test with log normalization
print("\n" + "="*70)
print("Log Normalization Test")
print("="*70)

normalizer_log = Exponential_OutputNormalizer(use_log_normalization=True)
normalizer_log.fit(targets)

# Normalize
targets_normalized = normalizer_log.transform(targets)
print(f"\nNormalized shape: {targets_normalized.shape}")

# Denormalize
try:
    targets_reconstructed = normalizer_log.inverse_transform(targets_normalized)
    print(f"Reconstructed shape: {targets_reconstructed.shape}")

    # Calculate MSE: sum((target / denormalize(normalize(target)) - 1)**2)
    mse = np.sum(((targets / (targets_reconstructed + 1e-10)) - 1) ** 2)

    print(f"\nMSE (log normalization): {mse:.6e}")
    print(f"  Formula: sum((target / denormalize(normalize(target)) - 1)^2)")

    # Also show max relative error
    max_rel_error = np.max(np.abs((targets - targets_reconstructed) / (targets + 1e-10)))
    print(f"Max relative error: {max_rel_error:.6e}")

except Exception as e:
    print(f"\nError with log normalization: {e}")
    import traceback
    traceback.print_exc()

# Test with standard normalization
print("\n" + "="*70)
print("Standard Normalization Test")
print("="*70)

normalizer_std = Exponential_OutputNormalizer(use_log_normalization=False)
normalizer_std.fit(targets)

# Normalize
targets_normalized_std = normalizer_std.transform(targets)
print(f"\nNormalized shape: {targets_normalized_std.shape}")

# Denormalize
targets_reconstructed_std = normalizer_std.inverse_transform(targets_normalized_std)
print(f"Reconstructed shape: {targets_reconstructed_std.shape}")

# Calculate MSE
mse_std = np.sum(((targets / (targets_reconstructed_std + 1e-10)) - 1) ** 2)

print(f"\nMSE (standard normalization): {mse_std:.6e}")
print(f"  Formula: sum((target / denormalize(normalize(target)) - 1)^2)")

# Also show max relative error
max_rel_error_std = np.max(np.abs((targets - targets_reconstructed_std) / (targets + 1e-10)))
print(f"Max relative error: {max_rel_error_std:.6e}")

print("\n" + "="*70)
