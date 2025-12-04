"""
Example script demonstrating how to use Exponential_OutputNormalizer

This shows how to normalize and denormalize output data [x_t, v_t, a_t]
"""

import numpy as np
import torch
from Exp_dataset import Exponential_OutputNormalizer

def main():
    print("="*60)
    print("Testing Exponential_OutputNormalizer")
    print("="*60)

    # Generate some sample output data
    # For exponential functions, outputs can vary across many orders of magnitude
    np.random.seed(42)
    n_samples = 1000

    # Simulate outputs from exponential functions with various a and b values
    a_vals = np.random.uniform(-5, 5, n_samples)
    b_vals = np.random.uniform(-100, 100, n_samples)
    t_vals = np.random.uniform(0.01, 5, n_samples)

    # Calculate outputs: x(t) = b*exp(a*t), v(t) = b*a*exp(a*t), a(t) = b*a²*exp(a*t)
    x_t = b_vals * np.exp(a_vals * t_vals)
    v_t = b_vals * a_vals * np.exp(a_vals * t_vals)
    a_t = b_vals * a_vals**2 * np.exp(a_vals * t_vals)

    outputs = np.stack([x_t, v_t, a_t], axis=1)

    print(f"\nOriginal output statistics:")
    print(f"  x_t: min={x_t.min():.2e}, max={x_t.max():.2e}, mean={x_t.mean():.2e}")
    print(f"  v_t: min={v_t.min():.2e}, max={v_t.max():.2e}, mean={v_t.mean():.2e}")
    print(f"  a_t: min={a_t.min():.2e}, max={a_t.max():.2e}, mean={a_t.mean():.2e}")

    # Test 1: Log-space normalization (recommended for exponential data)
    print("\n" + "="*60)
    print("Test 1: Log-space Normalization")
    print("="*60)

    normalizer_log = Exponential_OutputNormalizer(use_log_normalization=True)
    normalizer_log.fit(outputs)

    print("\nLog-space normalization parameters:")
    for feat in ['x', 'v', 'a']:
        print(f"  {feat}: log_mean={normalizer_log.log_mean[feat]:.4f}, "
              f"log_std={normalizer_log.log_std[feat]:.4f}")

    # Normalize
    outputs_norm_log = normalizer_log.normalize_outputs(outputs)

    print(f"\nNormalized output statistics (log-space):")
    print(f"  x_norm: min={outputs_norm_log[:, 0].min():.2f}, "
          f"max={outputs_norm_log[:, 0].max():.2f}, "
          f"mean={outputs_norm_log[:, 0].mean():.2f}")
    print(f"  v_norm: min={outputs_norm_log[:, 1].min():.2f}, "
          f"max={outputs_norm_log[:, 1].max():.2f}, "
          f"mean={outputs_norm_log[:, 1].mean():.2f}")
    print(f"  a_norm: min={outputs_norm_log[:, 2].min():.2f}, "
          f"max={outputs_norm_log[:, 2].max():.2f}, "
          f"mean={outputs_norm_log[:, 2].mean():.2f}")

    # Denormalize and check reconstruction
    outputs_reconstructed = normalizer_log.denormalize_outputs(outputs_norm_log)
    reconstruction_error = np.abs(outputs - outputs_reconstructed) / (np.abs(outputs) + 1e-10)

    print(f"\nReconstruction relative error (log-space):")
    print(f"  x: mean={reconstruction_error[:, 0].mean():.2e}, max={reconstruction_error[:, 0].max():.2e}")
    print(f"  v: mean={reconstruction_error[:, 1].mean():.2e}, max={reconstruction_error[:, 1].max():.2e}")
    print(f"  a: mean={reconstruction_error[:, 2].mean():.2e}, max={reconstruction_error[:, 2].max():.2e}")

    # Test 2: Standard normalization
    print("\n" + "="*60)
    print("Test 2: Standard Normalization")
    print("="*60)

    normalizer_std = Exponential_OutputNormalizer(use_log_normalization=False)
    normalizer_std.fit(outputs)

    print("\nStandard normalization parameters:")
    for feat in ['x', 'v', 'a']:
        print(f"  {feat}: mean={normalizer_std.linear_mean[feat]:.4e}, "
              f"std={normalizer_std.linear_std[feat]:.4e}")

    # Normalize
    outputs_norm_std = normalizer_std.normalize_outputs(outputs)

    print(f"\nNormalized output statistics (standard):")
    print(f"  x_norm: min={outputs_norm_std[:, 0].min():.2f}, "
          f"max={outputs_norm_std[:, 0].max():.2f}, "
          f"mean={outputs_norm_std[:, 0].mean():.2e}")
    print(f"  v_norm: min={outputs_norm_std[:, 1].min():.2f}, "
          f"max={outputs_norm_std[:, 1].max():.2f}, "
          f"mean={outputs_norm_std[:, 1].mean():.2e}")
    print(f"  a_norm: min={outputs_norm_std[:, 2].min():.2f}, "
          f"max={outputs_norm_std[:, 2].max():.2f}, "
          f"mean={outputs_norm_std[:, 2].mean():.2e}")

    # Denormalize and check reconstruction
    outputs_reconstructed_std = normalizer_std.denormalize_outputs(outputs_norm_std)
    reconstruction_error_std = np.abs(outputs - outputs_reconstructed_std) / (np.abs(outputs) + 1e-10)

    print(f"\nReconstruction relative error (standard):")
    print(f"  x: mean={reconstruction_error_std[:, 0].mean():.2e}, max={reconstruction_error_std[:, 0].max():.2e}")
    print(f"  v: mean={reconstruction_error_std[:, 1].mean():.2e}, max={reconstruction_error_std[:, 1].max():.2e}")
    print(f"  a: mean={reconstruction_error_std[:, 2].mean():.2e}, max={reconstruction_error_std[:, 2].max():.2e}")

    # Test 3: PyTorch tensor support
    print("\n" + "="*60)
    print("Test 3: PyTorch Tensor Support")
    print("="*60)

    outputs_tensor = torch.FloatTensor(outputs[:10])
    print(f"\nOriginal tensor shape: {outputs_tensor.shape}")
    print(f"Original tensor device: {outputs_tensor.device}")

    outputs_norm_tensor = normalizer_log.normalize_outputs(outputs_tensor)
    print(f"\nNormalized tensor shape: {outputs_norm_tensor.shape}")
    print(f"Normalized tensor type: {type(outputs_norm_tensor)}")

    outputs_denorm_tensor = normalizer_log.denormalize_outputs(outputs_norm_tensor)
    print(f"Denormalized tensor shape: {outputs_denorm_tensor.shape}")

    tensor_error = torch.abs(outputs_tensor - outputs_denorm_tensor) / (torch.abs(outputs_tensor) + 1e-10)
    print(f"\nTensor reconstruction error: mean={tensor_error.mean():.2e}, max={tensor_error.max():.2e}")

    print("\n" + "="*60)
    print("All tests completed successfully!")
    print("="*60)

if __name__ == "__main__":
    main()
