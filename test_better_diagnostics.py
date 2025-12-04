"""
Better diagnostics for output normalizer
"""

import numpy as np
from Exp_dataset import Exponential_OutputNormalizer

# Load data
data = np.load(r'E:\Ian\PINNexample\exponential_test_data.npz')
targets = data[list(data.keys())[0]][:, 3:]

print("Testing with first 1000 samples only...")
targets = targets[:1000]

# Test log normalization
normalizer_log = Exponential_OutputNormalizer(use_log_normalization=True)
normalizer_log.fit(targets)

targets_normalized = normalizer_log.transform(targets)
targets_reconstructed = normalizer_log.inverse_transform(targets_normalized)

# Direct comparison
diff = targets - targets_reconstructed
abs_diff = np.abs(diff)
rel_error = np.abs(diff / (targets + 1e-10))

print("\n" + "="*70)
print("Absolute Difference Statistics:")
print("="*70)
print(f"Mean: {abs_diff.mean():.6e}")
print(f"Max:  {abs_diff.max():.6e}")
print(f"Min:  {abs_diff.min():.6e}")

print("\n" + "="*70)
print("Relative Error Statistics:")
print("="*70)
print(f"Mean: {rel_error.mean():.6e}")
print(f"Max:  {rel_error.max():.6e}")
print(f"Min:  {rel_error.min():.6e}")

# Check for sign mismatches
sign_orig = np.sign(targets)
sign_recon = np.sign(targets_reconstructed)
sign_mismatch = (sign_orig != sign_recon)

print("\n" + "="*70)
print("Sign Mismatches:")
print("="*70)
print(f"Total elements: {sign_mismatch.size}")
print(f"Mismatches: {sign_mismatch.sum()}")
print(f"Percentage: {sign_mismatch.sum() / sign_mismatch.size * 100:.2f}%")

if sign_mismatch.sum() > 0:
    print("\nFirst 5 mismatches:")
    mismatch_indices = np.where(sign_mismatch)
    for i in range(min(5, sign_mismatch.sum())):
        row = mismatch_indices[0][i]
        col = mismatch_indices[1][i]
        feat_name = ['x', 'v', 'a'][col]
        print(f"  Sample {row}, feature '{feat_name}':")
        print(f"    Original: {targets[row, col]:+.4e}")
        print(f"    Reconstructed: {targets_reconstructed[row, col]:+.4e}")
        print(f"    Stored sign: {normalizer_log.sign[feat_name][row]:+.0f}")

# Perfect match check
perfect_match = np.allclose(targets, targets_reconstructed, rtol=1e-10, atol=1e-10)
print(f"\nPerfect match (within 1e-10): {perfect_match}")
