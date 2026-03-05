from PINN_dataset import *
from PINN_modelandloss import *
from datagtgenerator import *
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import numpy as np

def save_checkpoint(model_path, model, inputs_normalizer, outputs_normalizer):
    """
    Save model weights and normalizers together.

    Args:
        model_path (str): Path where model.pt will be saved (e.g., "model_epoch_100.pt")
        model: PyTorch model
        inputs_normalizer: Fitted input normalizer
        outputs_normalizer: Fitted output normalizer

    Creates two files:
        - model_epoch_100.pt (model weights)
        - model_epoch_100_normalizers.pt (normalizer states)
    """
    # Create parent directory if it doesn't exist
    model_dir = os.path.dirname(model_path)
    if model_dir and not os.path.exists(model_dir):
        os.makedirs(model_dir)
        print(f"Created directory: {model_dir}")

    # Save model weights
    torch.save(model.state_dict(), model_path)

    # Save normalizers
    normalizer_path = model_path.replace('.pt', '_normalizers.pt')
    torch.save({
        'inputs_normalizer': inputs_normalizer,
        'outputs_normalizer': outputs_normalizer
    }, normalizer_path)

    print(f"Saved model to: {model_path}")
    print(f"Saved normalizers to: {normalizer_path}")


def load_checkpoint(model_path, model):
    """
    Load model weights and normalizers together.

    Args:
        model_path (str): Path to model.pt file
        model: PyTorch model (must match architecture)

    Returns:
        tuple: (model, inputs_normalizer, outputs_normalizer)
    """
    # Load model weights
    model.load_state_dict(torch.load(model_path))

    # Load normalizers
    normalizer_path = model_path.replace('.pt', '_normalizers.pt')
    normalizers = torch.load(normalizer_path)

    inputs_normalizer = normalizers['inputs_normalizer']
    outputs_normalizer = normalizers['outputs_normalizer']

    print(f"Loaded model from: {model_path}")
    print(f"Loaded normalizers from: {normalizer_path}")

    return model, inputs_normalizer, outputs_normalizer


def check_npz_data_residuals(npz_filepath, use_relative=False, max_samples=None):
    """
    Check residuals directly from raw .npz data (before normalization/processing).
    This helps determine if residual errors are from the data itself or from processing.

    Args:
        npz_filepath: Path to .npz file containing raw data
        use_relative: If True, compute scale-invariant relative residual (default: False)
        max_samples: Maximum number of samples to check (None = check all)

    Returns:
        mean_abs_residual: Mean absolute residual value
    """
    print(f"\n{'='*80}")
    print(f"CHECKING RAW NPZ DATA RESIDUALS")
    print(f"File: {npz_filepath}")
    print(f"{'='*80}")

    # Load raw data
    data = np.load(npz_filepath)

    if isinstance(data, np.lib.npyio.NpzFile):
        # Get the first array in the npz file
        array_name = list(data.keys())[0]
        data_array = data[array_name]
    else:
        data_array = data


    inputs = data_array[:, :6]   # Inputs: m, zeta, k, t, x0, v0
    targets = data_array[:, 6:]  # Outputs: x(t), v(t), a(t)


    print(f"Loaded {len(inputs)} samples from .npz file")
    print(f"Inputs shape: {inputs.shape}")
    print(f"Targets shape: {targets.shape}")

    # Limit samples if requested
    if max_samples is not None and len(inputs) > max_samples:
        print(f"Limiting to first {max_samples} samples for checking")
        inputs = inputs[:max_samples]
        targets = targets[:max_samples]

    # Extract parameters
    m = inputs[:, 0]  # mass
    zeta = inputs[:, 1]  # damping ratio
    k = inputs[:, 2]  # stiffness
    c = 2 * zeta * np.sqrt(m * k)  # damping coefficient

    # Extract target values
    x_t = targets[:, 0]
    v_t = targets[:, 1]
    a_t = targets[:, 2]

    # Physics residual: m*a + c*v + k*x = 0
    eps = 1e-12
    residual = m * a_t + c * v_t + k * x_t

    if use_relative:
        # Scale-invariant relative residual
        scale = np.abs(m * a_t) + eps
        residual = residual / scale

    mean_abs_residual = np.mean(np.abs(residual))

    residual_type = "relative" if use_relative else "absolute"
    print(f"\nRaw NPZ data mean absolute {residual_type} residual: {mean_abs_residual:.6e}")
    print(f"Raw NPZ data {residual_type} residual range: [{residual.min():.6e}, {residual.max():.6e}]")
    print(f"Raw NPZ data {residual_type} residual std: {np.std(residual):.6e}")

    # Statistical analysis
    residual_abs = np.abs(residual)
    percentiles = [50, 90, 95, 99, 99.9]
    print(f"\nResidual percentiles (absolute values):")
    for p in percentiles:
        val = np.percentile(residual_abs, p)
        print(f"  {p:5.1f}th percentile: {val:.6e}")

    # Show TOP 5 largest residuals
    print(f"\n{'='*80}")
    print(f"TOP 5 LARGEST RESIDUALS (Raw NPZ Data):")
    print(f"{'='*80}")

    # Compute both absolute and relative residuals for display
    eps = 1e-12
    residual_abs_values = m * a_t + c * v_t + k * x_t
    scale = np.abs(m * a_t) + eps
    residual_rel_values = residual_abs_values / scale

    if use_relative:
        # Currently showing relative, so find top 5 by relative
        top5_indices = np.argsort(np.abs(residual))[-5:][::-1]
        print(f"[Ranked by RELATIVE residual magnitude]")
    else:
        # Currently showing absolute, so find top 5 by absolute
        top5_indices = np.argsort(np.abs(residual))[-5:][::-1]
        print(f"[Ranked by ABSOLUTE residual magnitude]")

    for rank, idx in enumerate(top5_indices, 1):
        print(f"\n  Rank {rank} (Sample #{idx}):")
        print(f"    Absolute residual: {residual_abs_values[idx]:.6e}")
        print(f"    Relative residual: {residual_rel_values[idx]:.6e}")
        print(f"    Parameters:")
        print(f"      m={inputs[idx, 0]:.6e}, zeta={inputs[idx, 1]:.6e}, k={inputs[idx, 2]:.6e}")
        print(f"      t={inputs[idx, 3]:.6e}, x0={inputs[idx, 4]:.6e}, v0={inputs[idx, 5]:.6e}")
        print(f"    Targets:")
        print(f"      x={x_t[idx]:.6e}, v={v_t[idx]:.6e}, a={a_t[idx]:.6e}")
        print(f"    Physics: m*a + c*v + k*x = {residual_abs_values[idx]:.6e}")
        print(f"    Terms: m*a={m[idx] * a_t[idx]:.6e}, c*v={c[idx] * v_t[idx]:.6e}, k*x={k[idx] * x_t[idx]:.6e}")
        if use_relative:
            print(f"    Scale (m*|a|): {scale[idx]:.6e}")

    # Assessment
    print(f"\n{'='*80}")
    print("RAW NPZ DATA ASSESSMENT:")
    print(f"{'='*80}")
    if mean_abs_residual < 1e-9:
        print("✓ PASSED: Raw data residual is very small (< 1e-9)")
        print("  Data generation appears correct!")
    elif mean_abs_residual < 1e-6:
        print("⚠ WARNING: Raw data residual is small but not negligible (< 1e-6)")
        print("  May indicate numerical precision issues in data generation.")
    else:
        print("✗ FAILED: Raw data residual is large (>= 1e-6)")
        print("  Data may have errors in generation process!")
    print(f"{'='*80}\n")

    return mean_abs_residual

def checktargetres(dataloader, inputs_normalizer, targets_normalizer, device, dtype, use_relative=False):
    """
    Check target residuals to validate VibrationResidualLoss implementation.
    Ground truth targets should satisfy physics equation: m*a + c*v + k*x = 0

    Args:
        dataloader: DataLoader containing (inputs, targets)
        inputs_normalizer: Normalizer for inputs (to get real-space m, zeta, k parameters)
        targets_normalizer: Normalizer for targets (for manual denormalization)
        device: torch device
        dtype: torch dtype
        use_relative: If True, compute scale-invariant relative residual (default: False)

    Returns:
        mean_abs_residual: Mean absolute residual value
    """
    all_residuals = []
    all_residuals_abs = []  # Absolute residuals (always computed)
    all_residuals_rel = []  # Relative residuals (always computed)
    all_inputs_real = []
    all_x_t = []
    all_v_t = []
    all_a_t = []
    all_m = []
    all_c = []
    all_k = []

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device, dtype=dtype)
            targets = targets.to(device, dtype=dtype)

            # Extract parameters m, zeta, k from normalized inputs
            # inputs shape: (batch_size, 6) -> [m, zeta, k, t, x0, v0] (normalized)
            # We need real-space m, zeta, k for the physics equation
            inputs_real = inputs_normalizer.denormalize_inputs(inputs)
            m = inputs_real[:, 0]  # mass parameter
            zeta = inputs_real[:, 1]  # damping ratio
            k = inputs_real[:, 2]  # stiffness parameter
            c = 2 * zeta * torch.sqrt(m * k)  # damping coefficient

            # DIAGNOSTIC: Check data shapes (only print for first batch)
            if len(all_residuals) == 0:
                print(f"\n[DIAGNOSTIC] Data shapes:")
                print(f"  inputs shape: {inputs.shape}")
                print(f"  targets shape: {targets.shape}")
                print(f"  inputs_real shape: {inputs_real.shape}")
                print(f"  Expected targets: (batch_size, 6) = [signs(3), logabs(3)]")

            # Manual denormalization (same as VibrationResidualLoss)
            # targets shape: (batch_size, 6) -> [real_signs (0-2), logabs_values (3-5)]
            real_signs = targets[:, :3]  # (batch_size, 3)
            logabs_normalized = targets[:, 3:]  # (batch_size, 3)

            # Create tensors for log_mean and log_std for [x, v, a]
            log_mean = torch.tensor([
                targets_normalizer.log_mean['x'],
                targets_normalizer.log_mean['v'],
                targets_normalizer.log_mean['a']
            ], device=device, dtype=dtype)  # (3,)

            log_std = torch.tensor([
                targets_normalizer.log_std['x'],
                targets_normalizer.log_std['v'],
                targets_normalizer.log_std['a']
            ], device=device, dtype=dtype)  # (3,)

            # Denormalize logabs values (vectorized, preserves gradients)
            logabs_denorm = logabs_normalized * log_std + log_mean  # (batch_size, 3)

            # Convert to real space: real_value = sign * 10^logabs (vectorized)
            ln10 = torch.tensor(np.log(10.0), device=device, dtype=dtype)
            real_values = real_signs * torch.exp(logabs_denorm * ln10)  # (batch_size, 3)

            # Extract x, v, a predictions
            x_t = real_values[:, 0]
            v_t = real_values[:, 1]
            a_t = real_values[:, 2]

            # DIAGNOSTIC: Compare denormalize_outputs() vs manual denormalization (only for first batch)
            if len(all_residuals) == 0:
                targets_real_auto = targets_normalizer.denormalize_outputs(targets)
                # Convert to torch tensor with same device/dtype
                if isinstance(targets_real_auto, np.ndarray):
                    targets_real_auto = torch.tensor(targets_real_auto, device=device, dtype=dtype)

                # Compare with manual denormalization result
                x_t_auto, v_t_auto, a_t_auto = targets_real_auto[:, 0], targets_real_auto[:, 1], targets_real_auto[:, 2]

                print(f"\n[DIAGNOSTIC] Denormalization comparison (first sample):")
                print(f"  Manual x_t: {x_t[0].item():.6e}, Auto x_t: {x_t_auto[0].item():.6e}, Diff: {abs(x_t[0] - x_t_auto[0]).item():.6e}")
                print(f"  Manual v_t: {v_t[0].item():.6e}, Auto v_t: {v_t_auto[0].item():.6e}, Diff: {abs(v_t[0] - v_t_auto[0]).item():.6e}")
                print(f"  Manual a_t: {a_t[0].item():.6e}, Auto a_t: {a_t_auto[0].item():.6e}, Diff: {abs(a_t[0] - a_t_auto[0]).item():.6e}")

                max_diff_x = torch.max(torch.abs(x_t - x_t_auto))
                max_diff_v = torch.max(torch.abs(v_t - v_t_auto))
                max_diff_a = torch.max(torch.abs(a_t - a_t_auto))
                print(f"  Max diff x_t: {max_diff_x.item():.6e}")
                print(f"  Max diff v_t: {max_diff_v.item():.6e}")
                print(f"  Max diff a_t: {max_diff_a.item():.6e}")

            # Physics residual: m*a + c*v + k*x = 0
            eps = 1e-12
            residual_abs = m * a_t + c * v_t + k * x_t

            # Compute relative residual
            scale = torch.abs(m * a_t) + eps
            residual_rel = residual_abs / scale

            # Store both forms
            all_residuals_abs.append(residual_abs)
            all_residuals_rel.append(residual_rel)

            # Store data for top-5 analysis
            all_inputs_real.append(inputs_real)
            all_x_t.append(x_t)
            all_v_t.append(v_t)
            all_a_t.append(a_t)
            all_m.append(m)
            all_c.append(c)
            all_k.append(k)

            # Choose which residual to use based on flag
            if use_relative:
                all_residuals.append(residual_rel)
            else:
                all_residuals.append(residual_abs)

    # Concatenate all residuals and compute mean absolute value
    all_residuals = torch.cat(all_residuals, dim=0)
    all_residuals_abs_cat = torch.cat(all_residuals_abs, dim=0)
    all_residuals_rel_cat = torch.cat(all_residuals_rel, dim=0)
    all_inputs_real_cat = torch.cat(all_inputs_real, dim=0)
    all_x_t_cat = torch.cat(all_x_t, dim=0)
    all_v_t_cat = torch.cat(all_v_t, dim=0)
    all_a_t_cat = torch.cat(all_a_t, dim=0)
    all_m_cat = torch.cat(all_m, dim=0)
    all_c_cat = torch.cat(all_c, dim=0)
    all_k_cat = torch.cat(all_k, dim=0)

    mean_abs_residual = torch.mean(torch.abs(all_residuals))

    residual_type = "relative" if use_relative else "absolute"
    print(f"Target mean absolute {residual_type} residual: {mean_abs_residual.item():.6e}")
    print(f"Target {residual_type} residual range: [{all_residuals.min().item():.6e}, {all_residuals.max().item():.6e}]")
    print(f"Target {residual_type} residual std: {torch.std(all_residuals).item():.6e}")

    # Statistical percentile analysis
    residual_abs_mag = torch.abs(all_residuals)
    percentiles = [50, 90, 95, 99, 99.9]
    print(f"\nResidual percentiles (absolute values):")
    for p in percentiles:
        # Convert to numpy for percentile calculation
        val = torch.quantile(residual_abs_mag, p/100.0)
        print(f"  {p:5.1f}th percentile: {val.item():.6e}")

    # Show TOP 5 largest residuals in BOTH absolute and relative forms
    print(f"\n{'='*80}")
    print(f"TOP 5 LARGEST RESIDUALS (DataLoader - After Normalization/Denormalization):")
    print(f"{'='*80}")

    # Find top 5 by absolute residual
    abs_residual_magnitudes = torch.abs(all_residuals_abs_cat)
    top5_abs_indices = torch.argsort(abs_residual_magnitudes, descending=True)[:5]

    print(f"\n[A] Ranked by ABSOLUTE residual magnitude:")
    for rank, idx in enumerate(top5_abs_indices, 1):
        idx_val = idx.item()
        print(f"\n  Rank {rank} (Sample #{idx_val}):")
        print(f"    Absolute residual: {all_residuals_abs_cat[idx].item():.6e}")
        print(f"    Relative residual: {all_residuals_rel_cat[idx].item():.6e}")
        print(f"    Parameters:")
        print(f"      m={all_m_cat[idx].item():.6e}, zeta={all_inputs_real_cat[idx, 1].item():.6e}, k={all_k_cat[idx].item():.6e}")
        print(f"      t={all_inputs_real_cat[idx, 3].item():.6e}, x0={all_inputs_real_cat[idx, 4].item():.6e}, v0={all_inputs_real_cat[idx, 5].item():.6e}")
        print(f"    Targets:")
        print(f"      x={all_x_t_cat[idx].item():.6e}, v={all_v_t_cat[idx].item():.6e}, a={all_a_t_cat[idx].item():.6e}")
        print(f"    Physics: m*a + c*v + k*x = {all_residuals_abs_cat[idx].item():.6e}")
        print(f"    Terms: m*a={all_m_cat[idx].item() * all_a_t_cat[idx].item():.6e}, c*v={all_c_cat[idx].item() * all_v_t_cat[idx].item():.6e}, k*x={all_k_cat[idx].item() * all_x_t_cat[idx].item():.6e}")

    # Find top 5 by relative residual
    rel_residual_magnitudes = torch.abs(all_residuals_rel_cat)
    top5_rel_indices = torch.argsort(rel_residual_magnitudes, descending=True)[:5]

    print(f"\n[B] Ranked by RELATIVE residual magnitude:")
    for rank, idx in enumerate(top5_rel_indices, 1):
        idx_val = idx.item()
        print(f"\n  Rank {rank} (Sample #{idx_val}):")
        print(f"    Relative residual: {all_residuals_rel_cat[idx].item():.6e}")
        print(f"    Absolute residual: {all_residuals_abs_cat[idx].item():.6e}")
        print(f"    Parameters:")
        print(f"      m={all_m_cat[idx].item():.6e}, zeta={all_inputs_real_cat[idx, 1].item():.6e}, k={all_k_cat[idx].item():.6e}")
        print(f"      t={all_inputs_real_cat[idx, 3].item():.6e}, x0={all_inputs_real_cat[idx, 4].item():.6e}, v0={all_inputs_real_cat[idx, 5].item():.6e}")
        print(f"    Targets:")
        print(f"      x={all_x_t_cat[idx].item():.6e}, v={all_v_t_cat[idx].item():.6e}, a={all_a_t_cat[idx].item():.6e}")
        print(f"    Physics: m*a + c*v + k*x = {all_residuals_abs_cat[idx].item():.6e}")
        print(f"    Scale (m*|a|): {(all_m_cat[idx].item() * torch.abs(all_a_t_cat[idx]).item()):.6e}")

    print(f"{'='*80}\n")

    return mean_abs_residual.item()

def calculate_calibration_improvement(outputs_before, outputs_after, targets):
    """
    Calculate how much ft_cal improves predictions.

    Args:
        outputs_before: Predictions before calibration (batch, 3)
        outputs_after: Predictions after calibration (batch, 3)
        targets: Ground truth targets (batch, 3)

    Returns:
        closer_rate: Percentage of outputs closer after calibration
        mean_improvement: Average error reduction
    """
    error_before = torch.abs(torch.abs(outputs_before) - torch.abs(targets))
    error_after = torch.abs(torch.abs(outputs_after) - torch.abs(targets))
    improvement = error_before - error_after  # Positive = improvement

    closer_count = (improvement > 0).sum().item()
    total_count = improvement.numel()
    closer_rate = closer_count / total_count * 100
    mean_improvement = improvement.mean().item()

    return closer_rate, mean_improvement

def testdataloaderunchange():
    """Test data loader consistency across Train/Val/Test datasets."""
    device_index = 0
    train_in_64 = True



    # Setup float64 training if requested
    if train_in_64:
        torch.set_default_dtype(torch.float64)
        dtype = torch.float64
        print("Training in float64 (double precision) mode")
    else:
        dtype = torch.float32
        print("Training in float32 (single precision) mode")


    # Data paths
    Train_Val_data_source = r'H:\Postgraudate\Research\Test\SimpleMCKvibration\newwide_scale_trainval_vibration_data.npz'
    Test_data_source = r'H:\Postgraudate\Research\Test\SimpleMCKvibration\newwide_scale_test_vibration_data.npz'
    data_normalize = True

    # Load the dataset
    train_loader, val_loader, _, train_val_inputs_normalizer, train_val_outputs_normalizer, train_val_precision_stats = load_vibration_data_check(
        filepath=Train_Val_data_source,
        batch_size=1024,
        normalize=data_normalize,
        shuffle_train=False,
        dtype=dtype
    )

    test_loader, _, _, test_inputs_normalizer, test_outputs_normalizer, test_precision_stats = load_vibration_data_check(
        filepath=Test_data_source,
        batch_size=1024,
        normalize=data_normalize,
        shuffle_train=False,
        dtype=dtype
    )


    # =========================================================================
    # STATISTICS CHECK: Verify Train/Val/Test data are in the same range
    # =========================================================================
    print("\n" + "="*80)
    print("DATA STATISTICS CHECK: Verifying Train/Val/Test Consistency")
    print("="*80)

    def compute_statistics(dataloader, dataset_name):
        """Compute min, max, mean, std for inputs and targets."""
        all_inputs = []
        all_targets = []

        for inputs, targets in dataloader:
            all_inputs.append(inputs.numpy())
            all_targets.append(targets.numpy())

        all_inputs = np.concatenate(all_inputs, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)

        stats = {
            'name': dataset_name,
            'n_samples': len(all_inputs),
            'inputs': {
                'min': np.min(all_inputs, axis=0),
                'max': np.max(all_inputs, axis=0),
                'mean': np.mean(all_inputs, axis=0),
                'std': np.std(all_inputs, axis=0)
            },
            'targets': {
                'min': np.min(all_targets, axis=0),
                'max': np.max(all_targets, axis=0),
                'mean': np.mean(all_targets, axis=0),
                'std': np.std(all_targets, axis=0)
            }
        }
        return stats

    # Compute statistics for each dataset
    print("\nComputing statistics for Train/Val/Test datasets...")
    train_stats = compute_statistics(train_loader, "Train")
    val_stats = compute_statistics(val_loader, "Val")
    test_stats = compute_statistics(test_loader, "Test")

    # Print sample counts
    print("\n" + "-"*80)
    print("SAMPLE COUNTS:")
    print("-"*80)
    print(f"  Train samples: {train_stats['n_samples']}")
    print(f"  Val samples:   {val_stats['n_samples']}")
    print(f"  Test samples:  {test_stats['n_samples']}")
    print(f"  Total samples: {train_stats['n_samples'] + val_stats['n_samples'] + test_stats['n_samples']}")

    # Print INPUT statistics
    n_input_features = len(train_stats['inputs']['min'])
    input_names = ['m', 'zeta', 'k', 't', 'x0', 'v0'] if n_input_features == 6 else [f'input_{i}' for i in range(n_input_features)]
    print("\n" + "-"*80)
    print("INPUT STATISTICS (Normalized):")
    print("-"*80)

    for idx, name in enumerate(input_names):
        print(f"\n  [{name}]")
        print(f"    {'Dataset':<10} {'Min':>12} {'Max':>12} {'Mean':>12} {'Std':>12}")
        print(f"    {'-'*10} {'-'*12} {'-'*12} {'-'*12} {'-'*12}")
        print(f"    {'Train':<10} {train_stats['inputs']['min'][idx]:12.6e} {train_stats['inputs']['max'][idx]:12.6e} {train_stats['inputs']['mean'][idx]:12.6e} {train_stats['inputs']['std'][idx]:12.6e}")
        print(f"    {'Val':<10} {val_stats['inputs']['min'][idx]:12.6e} {val_stats['inputs']['max'][idx]:12.6e} {val_stats['inputs']['mean'][idx]:12.6e} {val_stats['inputs']['std'][idx]:12.6e}")
        print(f"    {'Test':<10} {test_stats['inputs']['min'][idx]:12.6e} {test_stats['inputs']['max'][idx]:12.6e} {test_stats['inputs']['mean'][idx]:12.6e} {test_stats['inputs']['std'][idx]:12.6e}")

        # Check if ranges are consistent (within 10% tolerance)
        all_mins = [train_stats['inputs']['min'][idx], val_stats['inputs']['min'][idx], test_stats['inputs']['min'][idx]]
        all_maxs = [train_stats['inputs']['max'][idx], val_stats['inputs']['max'][idx], test_stats['inputs']['max'][idx]]
        all_means = [train_stats['inputs']['mean'][idx], val_stats['inputs']['mean'][idx], test_stats['inputs']['mean'][idx]]

        min_range = max(all_mins) - min(all_mins)
        max_range = max(all_maxs) - min(all_maxs)
        mean_range = max(all_means) - min(all_means)

        if min_range > 0.1 or max_range > 0.1 or mean_range > 0.1:
            print(f"    ⚠ WARNING: Significant difference detected in [{name}]!")

    # Print TARGET statistics
    n_target_features = len(train_stats['targets']['min'])
    target_names = ['sign_x', 'sign_v', 'sign_a', 'logabs_x', 'logabs_v', 'logabs_a'] if n_target_features == 6 else [f'target_{i}' for i in range(n_target_features)]
    print("\n" + "-"*80)
    print("TARGET STATISTICS (Normalized):")
    print("-"*80)

    for idx, name in enumerate(target_names):
        print(f"\n  [{name}]")
        print(f"    {'Dataset':<10} {'Min':>12} {'Max':>12} {'Mean':>12} {'Std':>12}")
        print(f"    {'-'*10} {'-'*12} {'-'*12} {'-'*12} {'-'*12}")
        print(f"    {'Train':<10} {train_stats['targets']['min'][idx]:12.6e} {train_stats['targets']['max'][idx]:12.6e} {train_stats['targets']['mean'][idx]:12.6e} {train_stats['targets']['std'][idx]:12.6e}")
        print(f"    {'Val':<10} {val_stats['targets']['min'][idx]:12.6e} {val_stats['targets']['max'][idx]:12.6e} {val_stats['targets']['mean'][idx]:12.6e} {val_stats['targets']['std'][idx]:12.6e}")
        print(f"    {'Test':<10} {test_stats['targets']['min'][idx]:12.6e} {test_stats['targets']['max'][idx]:12.6e} {test_stats['targets']['mean'][idx]:12.6e} {test_stats['targets']['std'][idx]:12.6e}")

        # Check if ranges are consistent (within 10% tolerance for continuous values, exact for signs)
        all_mins = [train_stats['targets']['min'][idx], val_stats['targets']['min'][idx], test_stats['targets']['min'][idx]]
        all_maxs = [train_stats['targets']['max'][idx], val_stats['targets']['max'][idx], test_stats['targets']['max'][idx]]
        all_means = [train_stats['targets']['mean'][idx], val_stats['targets']['mean'][idx], test_stats['targets']['mean'][idx]]

        min_range = max(all_mins) - min(all_mins)
        max_range = max(all_maxs) - min(all_maxs)
        mean_range = max(all_means) - min(all_means)

        # For sign columns (0-2), expect exact match (-1 or 1)
        # For logabs columns (3-5), expect normalized range
        if 'sign' in name:
            if min_range > 0.01 or max_range > 0.01:
                print(f"    ⚠ WARNING: Sign values differ across datasets!")
        else:
            if min_range > 0.2 or max_range > 0.2 or mean_range > 0.2:
                print(f"    ⚠ WARNING: Significant difference detected in [{name}]!")

    # Summary
    print("\n" + "="*80)
    print("SUMMARY:")
    print("="*80)
    print("✓ Statistics computed for Train/Val/Test datasets")
    print("✓ Check for warnings above to ensure data consistency")
    print("="*80 + "\n")

def log_training_results(log_dict, results_folder='./results', filename='training_log.txt', delimiter=', '):
    """
    Log training results to a delimited text file.

    Args:
        log_dict: Dictionary containing training information with keys:
                  - 'epoch': int, current epoch number
                  - 'outputs': numpy array or tensor, last batch predictions [x, v, a]
                  - 'targets': numpy array or tensor, last batch ground truth [x, v, a]
                  - 'train_loss': float, training loss for this epoch
                  - 'val_calibration_rate': float, optional calibration improvement rate
        results_folder: str, folder path to save the log file
        filename: str, name of the log file
        delimiter: str, delimiter to use between values (default: ', ')
                   Examples: ', ' for CSV, ' ' for space-separated, '\t' for tab-separated

    The function creates a delimited file with columns:
    epoch, output_x, output_v, output_a, target_x, target_v, target_a, train_loss, [val_calibration_rate]
    """
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    log_path = os.path.join(results_folder, filename)
    epoch = log_dict['epoch']
    outputs = log_dict['outputs']
    targets = log_dict['targets']
    train_loss = log_dict['train_loss']
    val_calibration_rate = log_dict.get('val_calibration_rate', None)

    if torch.is_tensor(outputs):
        outputs = outputs.detach().cpu().numpy()
    if torch.is_tensor(targets):
        targets = targets.detach().cpu().numpy()

    file_exists = os.path.isfile(log_path)

    with open(log_path, 'a') as f:
        if not file_exists:
            header_fields = ["epoch", "output_x", "output_v", "output_a",
                           "target_x", "target_v", "target_a", "train_loss"]
            if val_calibration_rate is not None:
                header_fields.append("val_calibration_rate")
            f.write(delimiter.join(header_fields) + "\n")

        data_fields = [f"{epoch:6d}", f"{outputs[0]:.6e}", f"{outputs[1]:.6e}", f"{outputs[2]:.6e}",
                      f"{targets[0]:.6e}", f"{targets[1]:.6e}", f"{targets[2]:.6e}", f"{train_loss:.6e}"]
        if val_calibration_rate is not None:
            data_fields.append(f"{val_calibration_rate:.2f}")
        f.write(delimiter.join(data_fields) + "\n")

    return log_path

def prediction_performance(data_path, model_pt_path, model, inputs_normalizer, outputs_normalizer, device, dtype=torch.float32, data_sampling_step=1, figure_folder='./figures'):
    """
    Generate prediction performance scatter plots comparing ground truth vs predictions.

    Args:
        data_path: Path to the test data .npz file
        model_pt_path: Path to the saved model .pt file
        model: The model instance used for training
        inputs_normalizer: The input normalizer instance from training data
        outputs_normalizer: The output normalizer instance from training data
        device: Device to run inference on (CPU or CUDA)
        dtype: torch dtype for inference
        data_sampling_step: Sample every N-th data point (default: 1, use all data)
        figure_folder: Folder path to save the figures (default: './figures')

    Returns:
        None (saves figures to disk)
    """
    print(f"\n{'='*60}")
    print("Generating Prediction Performance Plots")
    print(f"{'='*60}")

    if not os.path.exists(figure_folder):
        os.makedirs(figure_folder)
        print(f"Created folder: {figure_folder}")

    model.load_state_dict(torch.load(model_pt_path))
    model.eval()
    print(f"Loaded model from: {model_pt_path}")

    test_loader, _, _, _, _ = load_vibration_data(
        filepath=data_path,
        batch_size=256,
        normalize=True,
        shuffle_train=False,
        inputs_normalizer=inputs_normalizer,
        outputs_normalizer=outputs_normalizer
    )
    print(f"Loaded test data from: {data_path}")

    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc="Evaluating"):
            inputs = inputs.to(device, dtype=dtype)
            targets = targets.to(device, dtype=dtype)

            # Forward pass - returns (mag_preds, logabs_sign_pred, real_sign_pred, ft_cal) tuple
            mag_preds, logabs_sign_pred, real_sign_pred, ft_cal = model(inputs)

            # Extract log-absolute targets for comparison
            # Targets shape: (batch, 6) -> [real_signs (0-2), logabs_values (3-5)]
            logabs_targets = targets[:, 3:]

            # Apply signs and calibration: outputs = sign * (mag_preds + ft_cal)
            outputs = (logabs_sign_pred * (mag_preds + ft_cal)).detach()

            all_predictions.append(outputs.cpu().numpy())
            all_targets.append(logabs_targets.cpu().numpy())

    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    print(f"Total data points: {len(all_predictions)}")

    # Sample data for plotting to avoid overly dense plots
    if data_sampling_step > 1 and len(all_predictions) > data_sampling_step:
        sampled_indices = np.arange(0, len(all_predictions), data_sampling_step)
        predictions_sampled = all_predictions[sampled_indices]
        targets_sampled = all_targets[sampled_indices]
        print(f"Sampled data points for plotting (step={data_sampling_step}): {len(predictions_sampled)}")
    else:
        predictions_sampled = all_predictions
        targets_sampled = all_targets
        print("Using all data points for plotting.")

    output_names = ['logabs_x', 'logabs_v', 'logabs_a']
    output_titles = [
        'Log-Absolute Position Prediction Performance',
        'Log-Absolute Velocity Prediction Performance',
        'Log-Absolute Acceleration Prediction Performance'
    ]

    for idx, (name, title) in enumerate(zip(output_names, output_titles)):
        plt.figure(figsize=(8, 8))

        ground_truth = targets_sampled[:, idx]
        predictions = predictions_sampled[:, idx]

        plt.scatter(ground_truth, predictions, alpha=0.5, s=20)

        min_val = min(ground_truth.min(), predictions.min())
        max_val = max(ground_truth.max(), predictions.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r-', linewidth=2, label='Perfect Prediction (y=x)')

        plt.grid(True, alpha=0.3)
        plt.xlabel('Ground Truth', fontsize=12)
        plt.ylabel('Prediction', fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.legend()
        plt.axis('equal')

        filename = f"{name}_prediction.png"
        filepath = os.path.join(figure_folder, filename)
        plt.savefig(filepath, dpi=100, bbox_inches='tight')
        print(f"Saved: {filepath}")
        plt.close()

    print(f"{'='*60}")
    print("Prediction performance plots generated successfully!")
    print(f"{'='*60}\n")

def main_ver0():

    device_index = 0

    epochs = 50

    # Setup float64 training if requested (MUST be done BEFORE loading data)
    train_in_64 = True
    if train_in_64:
        torch.set_default_dtype(torch.float64)
        dtype = torch.float64
        print("Training in float64 (double precision) mode")
    else:
        dtype = torch.float32
        print("Training in float32 (single precision) mode")

    # Data paths
    Train_Val_data_source = r'H:\Postgraudate\Research\Test\SimpleMCKvibration\newwide_scale_trainval_vibration_data.npz'
    Test_data_source = r'H:\Postgraudate\Research\Test\SimpleMCKvibration\newwide_scale_test_vibration_data.npz'
    Plot_data_source = r'H:\Postgraudate\Research\Test\SimpleMCKvibration\newwide_scale_test_vibration_data.npz'
    # Load the dataset
    train_loader, val_loader, _, train_val_inputs_normalizer, train_val_targets_normalizer = load_vibration_data(
        filepath= Train_Val_data_source,
        batch_size=512,
        normalize=True,
        shuffle_train=True,
        dtype=dtype
    )

    test_loader, _, _, test_inputs_normalizer, test_targets_normalizer = load_vibration_data(
        filepath= Test_data_source,
        batch_size=512,
        normalize=True,
        shuffle_train=False,
        dtype=dtype,
        inputs_normalizer=train_val_inputs_normalizer,
        outputs_normalizer=train_val_targets_normalizer
    )

    print(f"Data loaders created:")

    # Setup device
    device = torch.device(f'cuda:{device_index}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model_file_name = 'vibration_model_ver1.pt'
    model_and_result_folder = './vibration_ver1'

    # Create the results folder if it doesn't exist
    if not os.path.exists(model_and_result_folder):
        os.makedirs(model_and_result_folder)
        print(f"Created results folder: {model_and_result_folder}")

    model_save_path = os.path.join(model_and_result_folder, model_file_name)

    # Create the Vibration PINN model with multi-network architecture
    model = VibrationPINN_ver3(
        hidden_dims=[16, 32, 64, 64, 32, 16],
        activation='elu',
        use_log_output=False,
        use_finetune=True,
        finetune_hidden_dims=[32, 128, 32],
        finetune_scale=10,
        logabs_sign_network_hidden_dims=[128, 64, 64, 32, 32],
        logabs_sign_network_dropout=0.3,
        real_sign_network_hidden_dims=[64, 64, 32, 32],
        real_sign_network_dropout=0.3
    ).to(device)

    # Configure losses using dict-based interface
    loss_config = {
        "MSE": {
            "weight": 0.8,
            "use_relative": False,
            "use_log": True,
            "sign_bce_weight": 1.0,
            "real_sign_bce_weight": 1.0,
            "ft_cal_weight": 1.0
        },
        "Residual": {"weight": 0.1, "use_relative": True},
        "InitialCondition": {"weight": 0.0, "t_threshold": 1e-8, "use_relative": True},  # Disabled for now
        "Consistency": {
            "weight": 0,
            "t_threshold": 1e-6,
            "use_log": True,
            "Input_grad_outside": True
        }
    }

    loss_fn = PINNLoss(model, loss_config)

    # Create separate optimizers for each network component
    mag_optimizer = torch.optim.Adam(model.network.parameters(), lr=0.005)
    finetune_optimizer = torch.optim.Adam(model.finetune_network.parameters(), lr=0.005)
    sign_optimizer = torch.optim.Adam(
        list(model.logabs_sign_network.parameters()) +
        list(model.real_sign_network.parameters()),
        lr=0.005
    )

    # Create separate schedulers for each optimizer
    mag_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        mag_optimizer, T_max=np.max([epochs//10, 1]), eta_min=1e-12
    )
    finetune_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        finetune_optimizer, T_max=np.max([epochs//10, 1]), eta_min=1e-12
    )
    sign_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        sign_optimizer, T_max=np.max([epochs//10, 1]), eta_min=1e-12
    )

    # Prepare inputs_normalizer for consistency loss
    inputs_normalizer = train_val_inputs_normalizer

    # Validate VibrationResidualLoss on ground truth targets before training
    if loss_fn.has_loss("Residual"):
        print("\n" + "="*80)
        print("VALIDATING VibrationResidualLoss on Ground Truth Targets")
        print("="*80)
        print("Ground truth targets should satisfy physics equation perfectly.")
        print("Expected residual: ~0.0 (ideally < 1e-10)")
        print("-"*80)

        # Get use_relative setting from loss config
        use_relative = loss_config["Residual"].get("use_relative", False)

        # STEP 1: Check raw .npz data BEFORE any processing
        print("\n[STEP 1] Checking RAW .npz data residuals (before normalization)...")
        npz_mean_abs_res = check_npz_data_residuals(
            Train_Val_data_source,
            use_relative=use_relative,
            max_samples=10000  # Check first 10k samples to save time
        )

        # STEP 2: Check processed data from DataLoader (after normalization + denormalization)
        print("\n[STEP 2] Checking DataLoader residuals (after normalization + denormalization)...")
        print("="*80)
        mean_abs_res = checktargetres(
            train_loader,
            train_val_inputs_normalizer,
            train_val_targets_normalizer,
            device,
            dtype,
            use_relative=use_relative
        )

        # STEP 3: Compare and assess
        print("\n" + "="*80)
        print("COMPARISON & ASSESSMENT:")
        print("="*80)
        print(f"Raw .npz data residual:    {npz_mean_abs_res:.6e}")
        print(f"DataLoader residual:       {mean_abs_res:.6e}")
        if npz_mean_abs_res > 0:
            ratio = mean_abs_res / npz_mean_abs_res
            print(f"Ratio (DataLoader/Raw):    {ratio:.4f}x")
            if ratio > 2.0:
                print("⚠ WARNING: DataLoader residual is significantly larger than raw data!")
                print("  This suggests precision loss during normalization/denormalization.")
            elif ratio < 0.5:
                print("✓ GOOD: DataLoader residual is similar or smaller than raw data.")
            else:
                print("✓ ACCEPTABLE: Residuals are comparable (within 2x).")
        print("-"*80)

        if mean_abs_res < 1e-6:
            print("✓ PASSED: Target residual is very small (< 1e-6)")
            print("  VibrationResidualLoss implementation appears correct!")
        elif mean_abs_res < 1e-3:
            print("⚠ WARNING: Target residual is small but not negligible (< 1e-3)")
            print("  Check manual denormalization implementation.")
        else:
            print("✗ FAILED: Target residual is large (>= 1e-3)")
            print("  VibrationResidualLoss implementation may have errors!")
            print("  Please check the denormalization and physics equation.")

        print("="*80 + "\n")

    # Training loop
    # Input data shape: (batch_size, 6) -> [m, zeta, k, t, x0, v0]
    # Target data shape: (batch_size, 6) -> [real_signs(3), logabs_values(3)]
    best_combined_loss = float('inf')
    finetune_activation_epoch = int(epochs * 0.2)  # Activate finetune network after 20% of epochs

    print(f"\nTraining Configuration:")
    print(f"  Phase 1 (epochs 1-{finetune_activation_epoch}): Magnitude + Sign networks")
    print(f"  Phase 2 (epochs {finetune_activation_epoch+1}-{epochs}): All networks (with finetune)")

    best_phase1_loss = float('inf')
    phase1_model_save_path = model_save_path.replace('.pt', '_phase1.pt')

    for epoch in range(epochs):
        # Two-phase training logic
        if epoch == 0:
            # Phase 1 setup
            print(f"\n{'='*60}")
            print("PHASE 1: Training magnitude network + sign networks")
            print("Finetune network: Not optimized (gradients computed but optimizer not stepped)")
            print(f"{'='*60}")

        elif epoch == finetune_activation_epoch:
            # Phase 2 transition: Load best Phase 1 weights
            print(f"\n{'='*60}")
            print(f"PHASE 2 TRANSITION at epoch {epoch+1}/{epochs}")
            print(f"Loading best Phase 1 weights from: {phase1_model_save_path}")
            print(f"Best Phase 1 combined loss: {best_phase1_loss:.4e}")
            print(f"{'='*60}")

            # Load best Phase 1 model
            model.load_state_dict(torch.load(phase1_model_save_path))

            # Reset best loss tracking for Phase 2
            best_combined_loss = float('inf')

            print(f"\n{'='*60}")
            print("PHASE 2: Training ALL networks (magnitude + finetune + sign)")
            print("Magnitude network: CONTINUE TRAINING (jointly with finetune)")
            print("Finetune network: NOW TRAINING (started from zeros)")
            print("Sign networks: CONTINUE TRAINING")
            print(f"{'='*60}")
        print(f"\nEpoch {epoch+1}/{epochs}")
        model.train()
        train_loss = 0.0
        train_loss_components = {}

        # Training progress bar
        train_pbar = tqdm(train_loader, desc=f"Training", leave=False)
        for inputs, targets in train_pbar:
            # Move data to device and convert to proper dtype
            inputs = inputs.to(device, dtype=dtype)
            targets = targets.to(device, dtype=dtype)

            #================test the model=======================
            # if the sum of inputs <0 the target = targets*0-4592.4452
            # if the sum of inputs >=0 the target = targets*0+4592.4452
            # Calculate sum for each row (sample) in the batch
            # row_sums = torch.sum(inputs, dim=1, keepdim=True)

            # # Create a boolean mask where the condition is met
            # # The mask will have shape [batch_size, 1]
            # mask = row_sums < 0

            # # Apply the condition to modify the targets tensor
            # # Where mask is True (row_sum < 0), set target to 4592.4452
            # # Where mask is False (row_sum >= 0), set target to -4592.4452
            # targets = torch.where(mask,
            #                       torch.full_like(targets, 5250.545),
            #                       torch.full_like(targets, -500.2))

            #================================================

            # Zero gradients for all optimizers (ALL networks compute gradients)
            mag_optimizer.zero_grad()
            finetune_optimizer.zero_grad()
            sign_optimizer.zero_grad()

            # Denormalize inputs for loss calculation
            if train_val_inputs_normalizer is not None:
                inputs_real = train_val_inputs_normalizer.denormalize_inputs(inputs).clone()
            else:
                inputs_real = inputs.clone()

            # Build inputs_combined based on which losses are enabled
            inputs_list = [inputs]
            N = inputs.size(0)

            # Generate t=0 samples if InitialCondition loss is enabled
            if loss_fn.has_loss("InitialCondition"):
                inputs_t0_real = inputs_real.clone()
                inputs_t0_real[:, 3] = 0.0  # Set real t=0
                inputs_t0 = torch.tensor(train_val_inputs_normalizer.normalize_inputs(inputs_t0_real.cpu().numpy()), dtype=dtype).to(device)
                inputs_list.append(inputs_t0)

            # Stack all inputs
            inputs_combined = torch.cat(inputs_list, dim=0)

            # Forward pass - enable gradient tracking for inputs if needed for consistency loss
            if loss_fn.has_loss("Consistency"):
                consistency_config = loss_config.get("Consistency", {})
                input_grad_outside = consistency_config.get("Input_grad_outside", False)
                if input_grad_outside:
                    inputs_combined.requires_grad_(True)

            # Forward pass - returns (mag_preds, logabs_sign_pred, real_sign_pred, ft_cal) tuple
            outputs_combined = model(inputs_combined)

            # Split outputs based on what was stacked
            # outputs_combined is now a 4-tuple: (mag_preds, logabs_sign_pred, real_sign_pred, ft_cal)
            # Each element has shape (total_N, 3)
            mag_preds_combined, logabs_sign_pred_combined, real_sign_pred_combined, ft_cal_combined = outputs_combined

            mag_preds = mag_preds_combined[:N]
            logabs_sign_pred = logabs_sign_pred_combined[:N]
            real_sign_pred = real_sign_pred_combined[:N]
            ft_cal = ft_cal_combined[:N]
            outputs = (mag_preds, logabs_sign_pred, real_sign_pred, ft_cal)
            idx = N

            if loss_fn.has_loss("InitialCondition"):
                mag_preds_t0 = mag_preds_combined[idx:idx+N]
                logabs_sign_t0 = logabs_sign_pred_combined[idx:idx+N]
                real_sign_t0 = real_sign_pred_combined[idx:idx+N]
                ft_cal_t0 = ft_cal_combined[idx:idx+N]
                outputs_t0 = (mag_preds_t0, logabs_sign_t0, real_sign_t0, ft_cal_t0)
                inputs_real_t0 = train_val_inputs_normalizer.denormalize_inputs(inputs_t0)
                idx += N

            # Prepare loss arguments
            loss_args = {}
            if loss_fn.has_loss("MSE"):
                # Extract sign probabilities for BCE loss
                logabs_sign_probs = model.logabs_last_sign_probs  # Sigmoid probabilities
                real_sign_probs = model.real_last_sign_probs

                loss_args["MSE"] = (
                    mag_preds, targets,
                    logabs_sign_probs, None, None,  # logabs, unused, unused
                    real_sign_probs, ft_cal,
                    train_val_targets_normalizer  # output normalizer
                )

            if loss_fn.has_loss("Residual"):
                # Phase-aware ft_cal: zero in Phase 1, active in Phase 2
                if epoch < finetune_activation_epoch:
                    ft_cal_for_residual = torch.zeros_like(ft_cal)
                else:
                    ft_cal_for_residual = ft_cal

                # Prepare outputs for residual: [real_signs(3), logabs_values(3)]
                # Apply signs to magnitudes
                logabs_sign = SignWithHardTanh.apply(logabs_sign_pred)  # Hard sign for residual
                signed_mag_preds = logabs_sign * mag_preds  # Element-wise multiplication

                # Add calibration
                outputs_for_residual = torch.cat([
                    real_sign_pred,                        # real signs (columns 0-2)
                    signed_mag_preds + ft_cal_for_residual # calibrated logabs values (columns 3-5)
                ], dim=1)

                loss_args["Residual"] = (
                    outputs_for_residual, targets,
                    inputs_real,
                    train_val_targets_normalizer  # output normalizer
                )

            if loss_fn.has_loss("Consistency"):
                # Only auto-diff consistency loss is supported (following Exp pattern)
                consistency_config = loss_config.get("Consistency", {})
                input_grad_outside = consistency_config.get("Input_grad_outside", False)

                # Determine ft_cal based on phase
                if epoch < finetune_activation_epoch:
                    ft_cal_consistency = torch.zeros_like(ft_cal)  # Phase 1
                else:
                    ft_cal_consistency = ft_cal  # Phase 2

                if input_grad_outside:
                    # MODE 1: Gradients computed in training loop
                    # Filter valid samples (t_real > threshold)
                    t_threshold = consistency_config.get("t_threshold", 1e-6)
                    t_real = inputs_real[:, 3]  # vibration: t is at index 3
                    valid_mask = t_real > t_threshold

                    # Pass FULL inputs_combined (not sliced) - following Exp pattern
                    # The loss function will filter internally using valid_mask
                    loss_args["Consistency"] = (
                        mag_preds, targets, inputs_combined,
                        train_val_inputs_normalizer,   # input normalizer
                        train_val_targets_normalizer,  # output normalizer
                        ft_cal_consistency, valid_mask
                    )
                else:
                    # MODE 2: Gradients computed inside loss function
                    loss_args["Consistency"] = (
                        None, targets, inputs,
                        train_val_inputs_normalizer,
                        train_val_targets_normalizer,
                        ft_cal_consistency, None
                    )

            if loss_fn.has_loss("InitialCondition"):
                loss_args["InitialCondition"] = (outputs_t0, inputs_real_t0)

            # Compute loss
            loss, loss_dict = loss_fn(loss_args)
            loss.backward()

            # Phase-aware optimizer stepping
            if epoch < finetune_activation_epoch:
                # Phase 1: Train magnitude + sign networks only
                mag_optimizer.step()
                sign_optimizer.step()
            else:
                # Phase 2: Train all networks (finetune + sign)
                finetune_optimizer.step()
                sign_optimizer.step()
            train_loss += loss.item() * inputs.size(0)

            # Accumulate loss components
            for key, value in loss_dict.items():
                if key not in train_loss_components:
                    train_loss_components[key] = 0.0
                train_loss_components[key] += value * inputs.size(0)

            # Update progress bar with current loss
            train_pbar.set_postfix({'loss': f'{loss.item():.4e}'})

        # Print last batch predictions vs targets (log-absolute space)
        print("Last batch outputs v.s targets (logabs):")
        print("  mag_preds[-1]:", mag_preds[-1].detach().cpu().numpy())
        print("  targets[-1, 3:]:", targets[-1, 3:].detach().cpu().numpy())
        print("  ft_cal[-1]:", ft_cal[-1].detach().cpu().numpy())

        if loss_fn.has_loss("InitialCondition"):
            sample_m, sample_zeta, sample_k, sample_t, sample_x0, sample_v0 = inputs_real_t0.cpu().numpy()[0]
            # print("Data X0, V0 at t=0:", sample_x0, sample_v0)
            sample_c = 2 * sample_zeta * np.sqrt(sample_m * sample_k)
            ana_sol = analytical_solution(sample_m, sample_c, sample_k, sample_x0, sample_v0, sample_t)
            print("Analytical v.s Denormalized:", "x0:", ana_sol[0], sample_x0, "v0:", ana_sol[1], sample_v0)
        train_loss /= len(train_loader.dataset)

        # Calculate average loss components
        for key in train_loss_components:
            train_loss_components[key] /= len(train_loader.dataset)

        # Construct outputs tensor for logging: [real_signs, signed_logabs]
        logabs_sign = torch.sign(logabs_sign_pred)  # Convert to hard signs
        signed_mag_preds = logabs_sign * mag_preds
        outputs_combined = torch.cat([
            real_sign_pred,           # real signs (columns 0-2)
            signed_mag_preds + ft_cal # calibrated logabs values (columns 3-5)
        ], dim=1)

        # Log training results to file (before validation)
        log_dict = {
            'epoch': epoch + 1,
            'outputs': outputs_combined[-1],  # Last sample from last batch (shape: 6)
            'targets': targets[-1],
            'train_loss': train_loss
        }
        log_training_results(log_dict, results_folder=model_and_result_folder, filename='training_log.txt')

        # Validation loop
        model.eval()
        val_loss = 0.0
        val_loss_components = {}

        # Determine if we need gradients for consistency loss (auto-diff type)
        use_no_grad_val = True
        if loss_fn.has_loss("Consistency") and loss_config["Consistency"].get("type") == "auto":
            use_no_grad_val = False

        # Conditionally use torch.no_grad() based on consistency type
        if use_no_grad_val:
            context_manager = torch.no_grad()
        else:
            context_manager = torch.enable_grad()

        with context_manager:
            # Validation progress bar
            val_pbar = tqdm(val_loader, desc=f"Validation", leave=False)
            for inputs, targets in val_pbar:
                # Move data to device
                inputs, targets = inputs.to(device), targets.to(device)

                # Denormalize inputs for loss calculation
                inputs_real = train_val_inputs_normalizer.denormalize_inputs(inputs).clone()

                # Build inputs_combined based on which losses are enabled
                inputs_list = [inputs]
                N = inputs.size(0)

                # Generate t=0 samples if InitialCondition loss is enabled
                if loss_fn.has_loss("InitialCondition"):
                    inputs_t0_real = inputs_real.clone()
                    inputs_t0_real[:, 3] = 0.0  # Set real t=0
                    inputs_t0 = torch.tensor(train_val_inputs_normalizer.normalize_inputs(inputs_t0_real.cpu().numpy()), dtype=dtype).to(device)
                    inputs_list.append(inputs_t0)

                # Generate perturbed time samples if Consistency loss with finite type is enabled
                if loss_fn.has_loss("Consistency") and loss_config["Consistency"]["type"] == "finite":
                    t_threshold = loss_config["Consistency"]["t_threshold"]

                    inputs_t_minus_minus_real = inputs_real.clone()
                    inputs_t_minus_minus_real[:, 3] = inputs_real[:, 3] - 2 * t_threshold
                    inputs_t_minus_minus = torch.tensor(train_val_inputs_normalizer.normalize_inputs(inputs_t_minus_minus_real.cpu().numpy()), dtype=dtype).to(device)

                    inputs_t_minus_real = inputs_real.clone()
                    inputs_t_minus_real[:, 3] = inputs_real[:, 3] - t_threshold
                    inputs_t_minus = torch.tensor(train_val_inputs_normalizer.normalize_inputs(inputs_t_minus_real.cpu().numpy()), dtype=dtype).to(device)

                    inputs_t_plus_real = inputs_real.clone()
                    inputs_t_plus_real[:, 3] = inputs_real[:, 3] + t_threshold
                    inputs_t_plus = torch.tensor(train_val_inputs_normalizer.normalize_inputs(inputs_t_plus_real.cpu().numpy()), dtype=dtype).to(device)

                    inputs_t_plus_plus_real = inputs_real.clone()
                    inputs_t_plus_plus_real[:, 3] = inputs_real[:, 3] + 2 * t_threshold
                    inputs_t_plus_plus = torch.tensor(train_val_inputs_normalizer.normalize_inputs(inputs_t_plus_plus_real.cpu().numpy()), dtype=dtype).to(device)

                    inputs_list.extend([inputs_t_minus_minus, inputs_t_minus, inputs_t_plus, inputs_t_plus_plus])

                # Stack all inputs
                inputs_combined = torch.cat(inputs_list, dim=0)

                # Forward pass
                outputs_combined = model(inputs_combined)

                # Split outputs based on what was stacked
                outputs = outputs_combined[:N]
                idx = N

                if loss_fn.has_loss("InitialCondition"):
                    outputs_t0 = outputs_combined[idx:idx+N]
                    inputs_real_t0 = train_val_inputs_normalizer.denormalize_inputs(inputs_t0)
                    idx += N

                if loss_fn.has_loss("Consistency") and loss_config["Consistency"]["type"] == "finite":
                    outputs_dt = outputs_combined[idx:idx+4*N]  # 4N samples: [t-2Δt, t-Δt, t+Δt, t+2Δt]
                    idx += 4*N

                # Prepare loss arguments
                loss_args = {}
                if loss_fn.has_loss("MSE"):
                    loss_args["MSE"] = (outputs, targets)
                if loss_fn.has_loss("Residual"):
                    loss_args["Residual"] = (outputs, inputs_real)
                if loss_fn.has_loss("Consistency"):
                    # Check consistency type
                    consistency_type = loss_config["Consistency"]["type"]
                    if consistency_type == "finite":
                        loss_args["Consistency"] = (outputs, outputs_dt, targets)
                    elif consistency_type == "auto":
                        loss_args["Consistency"] = (inputs, inputs_real, norm_params)
                    else:
                        raise ValueError(f"Unknown consistency type: {consistency_type}. Use 'auto' or 'finite'.")
                if loss_fn.has_loss("InitialCondition"):
                    loss_args["InitialCondition"] = (outputs_t0, inputs_real_t0)

                # Compute loss
                loss, loss_dict = loss_fn(loss_args)
                val_loss += loss.item() * inputs.size(0)

                # Accumulate loss components
                for key, value in loss_dict.items():
                    if key not in val_loss_components:
                        val_loss_components[key] = 0.0
                    val_loss_components[key] += value * inputs.size(0)

                # Update progress bar with current loss
                val_pbar.set_postfix({'loss': f'{loss.item():.4e}'})

        val_loss /= len(val_loader.dataset)

        # Calculate average loss components
        for key in val_loss_components:
            val_loss_components[key] /= len(val_loader.dataset)

        # Phase-aware LR scheduler stepping
        if epoch < finetune_activation_epoch:
            mag_scheduler.step()
            sign_scheduler.step()
        else:
            finetune_scheduler.step()
            sign_scheduler.step()

        # Print epoch summary
        print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {train_loss:.4e}, Val Loss: {val_loss:.4e}")

        # Print train loss breakdown with ratios
        print("  Train Loss Breakdown:")
        train_total = train_loss_components.get('total', train_loss)
        for key in sorted(train_loss_components.keys()):
            if key != 'total':
                value = train_loss_components[key]
                ratio = (value / train_total * 100) if train_total > 0 else 0
                print(f"    {key:20s}: {value:.4e} ({ratio:5.2f}%)")

        # Print val loss breakdown with ratios
        print("  Val Loss Breakdown:")
        val_total = val_loss_components.get('total', val_loss)
        for key in sorted(val_loss_components.keys()):
            if key != 'total':
                value = val_loss_components[key]
                ratio = (value / val_total * 100) if val_total > 0 else 0
                print(f"    {key:20s}: {value:.4e} ({ratio:5.2f}%)")

        # Save the model if combined loss (train + val) has improved
        combined_loss = train_loss + val_loss

        if epoch < finetune_activation_epoch:
            # Phase 1: Track best Phase 1 model
            if combined_loss < best_phase1_loss:
                best_phase1_loss = combined_loss
                save_checkpoint(phase1_model_save_path, model, train_val_inputs_normalizer, train_val_targets_normalizer)
                print(f"New best Phase 1 model saved with combined loss: {combined_loss:.4e} (train: {train_loss:.4e}, val: {val_loss:.4e})")
        else:
            # Phase 2: Track best overall model
            if combined_loss < best_combined_loss:
                best_combined_loss = combined_loss
                save_checkpoint(model_save_path, model, train_val_inputs_normalizer, train_val_targets_normalizer)
                print(f"New best model saved with combined loss: {combined_loss:.4e} (train: {train_loss:.4e}, val: {val_loss:.4e})")

    # Testing loop
    print("\nRunning test evaluation on the best model...")
    # Load the best model for testing
    model, loaded_inputs_norm, loaded_outputs_norm = load_checkpoint(model_save_path, model)
    train_val_inputs_normalizer = loaded_inputs_norm
    train_val_targets_normalizer = loaded_outputs_norm
    model.eval()
    test_loss = 0.0

    # Determine if we need gradients for consistency loss (auto-diff type)
    use_no_grad_test = True
    if loss_fn.has_loss("Consistency") and loss_config["Consistency"].get("type") == "auto":
        use_no_grad_test = False

    # Conditionally use torch.no_grad() based on consistency type
    if use_no_grad_test:
        context_manager = torch.no_grad()
    else:
        context_manager = torch.enable_grad()

    with context_manager:
        # Test progress bar
        test_pbar = tqdm(test_loader, desc="Testing", leave=True)
        for inputs, targets in test_pbar:
            # Move data to device
            inputs, targets = inputs.to(device), targets.to(device)

            # Denormalize inputs for loss calculation
            inputs_real = test_inputs_normalizer.denormalize_inputs(inputs).clone()

            # Build inputs_combined based on which losses are enabled
            inputs_list = [inputs]
            N = inputs.size(0)

            # Generate t=0 samples if InitialCondition loss is enabled
            if loss_fn.has_loss("InitialCondition"):
                inputs_t0_real = inputs_real.clone()
                inputs_t0_real[:, 3] = 0.0  # Set real t=0
                inputs_t0 = torch.tensor(test_inputs_normalizer.normalize_inputs(inputs_t0_real.cpu().numpy()), dtype=dtype).to(device)
                inputs_list.append(inputs_t0)

            # Generate perturbed time samples if Consistency loss with finite type is enabled
            if loss_fn.has_loss("Consistency") and loss_config["Consistency"]["type"] == "finite":
                t_threshold = loss_config["Consistency"]["t_threshold"]

                inputs_t_minus_minus_real = inputs_real.clone()
                inputs_t_minus_minus_real[:, 3] = inputs_real[:, 3] - 2 * t_threshold
                inputs_t_minus_minus = torch.tensor(test_inputs_normalizer.normalize_inputs(inputs_t_minus_minus_real.cpu().numpy()), dtype=dtype).to(device)

                inputs_t_minus_real = inputs_real.clone()
                inputs_t_minus_real[:, 3] = inputs_real[:, 3] - t_threshold
                inputs_t_minus = torch.tensor(test_inputs_normalizer.normalize_inputs(inputs_t_minus_real.cpu().numpy()), dtype=dtype).to(device)

                inputs_t_plus_real = inputs_real.clone()
                inputs_t_plus_real[:, 3] = inputs_real[:, 3] + t_threshold
                inputs_t_plus = torch.tensor(test_inputs_normalizer.normalize_inputs(inputs_t_plus_real.cpu().numpy()), dtype=dtype).to(device)

                inputs_t_plus_plus_real = inputs_real.clone()
                inputs_t_plus_plus_real[:, 3] = inputs_real[:, 3] + 2 * t_threshold
                inputs_t_plus_plus = torch.tensor(test_inputs_normalizer.normalize_inputs(inputs_t_plus_plus_real.cpu().numpy()), dtype=dtype).to(device)

                inputs_list.extend([inputs_t_minus_minus, inputs_t_minus, inputs_t_plus, inputs_t_plus_plus])

            # Stack all inputs
            inputs_combined = torch.cat(inputs_list, dim=0)

            # Forward pass
            outputs_combined = model(inputs_combined)

            # Split outputs based on what was stacked
            outputs = outputs_combined[:N]
            idx = N

            if loss_fn.has_loss("InitialCondition"):
                outputs_t0 = outputs_combined[idx:idx+N]
                inputs_real_t0 = test_inputs_normalizer.denormalize_inputs(inputs_t0)
                idx += N

            if loss_fn.has_loss("Consistency") and loss_config["Consistency"]["type"] == "finite":
                outputs_dt = outputs_combined[idx:idx+4*N]  # 4N samples: [t-2Δt, t-Δt, t+Δt, t+2Δt]
                idx += 4*N

            # Prepare norm_params for test (use test_normalizer)
            norm_params_test = {'normalizer': test_normalizer}

            # Prepare loss arguments
            loss_args = {}
            if loss_fn.has_loss("MSE"):
                loss_args["MSE"] = (outputs, targets)
            if loss_fn.has_loss("Residual"):
                loss_args["Residual"] = (outputs, inputs_real)
            if loss_fn.has_loss("Consistency"):
                # Check consistency type
                consistency_type = loss_config["Consistency"]["type"]
                if consistency_type == "finite":
                    loss_args["Consistency"] = (outputs, outputs_dt, targets)
                elif consistency_type == "auto":
                    loss_args["Consistency"] = (inputs, inputs_real, norm_params_test)
                else:
                    raise ValueError(f"Unknown consistency type: {consistency_type}. Use 'auto' or 'finite'.")
            if loss_fn.has_loss("InitialCondition"):
                loss_args["InitialCondition"] = (outputs_t0, inputs_real_t0)

            # Compute loss
            loss, loss_dict = loss_fn(loss_args)
            test_loss += loss.item() * inputs.size(0)

            # Update progress bar with current loss
            test_pbar.set_postfix({'loss': f'{loss.item():.4e}'})

    test_loss /= len(test_loader.dataset)
    print(f"\nTest Loss: {test_loss:.4e}")
    # Generate prediction performance plots
    prediction_performance(
        data_path=Plot_data_source,
        model_pt_path=model_save_path,
        model=model,
        inputs_normalizer=train_val_inputs_normalizer,
        outputs_normalizer=train_val_targets_normalizer,
        device=device,
        dtype=dtype,
        data_sampling_step=100,
        figure_folder=model_and_result_folder
    )

# "、

def main():
    device_index = 0
    train_in_64 = True
    epochs = 500

    # Setup float64 training if requested (MUST be done BEFORE loading data)
    if train_in_64:
        torch.set_default_dtype(torch.float64)
        dtype = torch.float64
        print("Training in float64 (double precision) mode")
    else:
        dtype = torch.float32
        print("Training in float32 (single precision) mode")

    # Data paths
    Train_Val_data_source = r'H:\Postgraudate\Research\Test\SimpleMCKvibration\newwide_scale_trainval_vibration_data.npz'
    Test_data_source = r'H:\Postgraudate\Research\Test\SimpleMCKvibration\newwide_scale_test_vibration_data.npz'
    Plot_data_source = r'H:\Postgraudate\Research\Test\SimpleMCKvibration\newwide_scale_test_vibration_data.npz'
    data_normalize = True

    # Check raw data residuals (before normalization)
    # check_raw_data_residuals(Train_Val_data_source, use_relative=True)
    # check_raw_data_residuals(Test_data_source, use_relative=True)

    # Load the dataset
    train_loader, val_loader, _, train_val_inputs_normalizer, train_val_targets_normalizer = load_vibration_data(
        filepath=Train_Val_data_source,
        batch_size=1024,
        normalize=data_normalize,
        shuffle_train=True,
        dtype=dtype
    )

    test_loader, _, _, test_inputs_normalizer, test_targets_normalizer = load_vibration_data(
        filepath=Test_data_source,
        batch_size=1024,
        normalize=data_normalize,
        shuffle_train=False,
        dtype=dtype,
        inputs_normalizer=train_val_inputs_normalizer,
        outputs_normalizer=train_val_targets_normalizer
    )

    print(f"Data loaders created:")

    # Setup device
    device = torch.device(f'cuda:{device_index}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model_file_name = 'vibration_model_onlymse_mag.pt'
    model_and_result_folder = './vibration_onlymse_mag'
    
    # Create the results folder if it doesn't exist
    if not os.path.exists(model_and_result_folder):
        os.makedirs(model_and_result_folder)
        print(f"Created results folder: {model_and_result_folder}")

    model_save_path = os.path.join(model_and_result_folder, model_file_name)
    # model_save_path = 'exp_test0.pt'#consistency_testOutside_nolog.pt'
    # results_figure_folder = './exp_test0'
    # Create the Exponential PINN model
    model = VibrationPINN_ver3(hidden_dims=[16, 32, 64, 64, 32, 16],
                          activation='elu',
                          use_log_output=False,
                          use_finetune=True,
                          finetune_hidden_dims=[32, 128, 32],
                          finetune_scale=10,
                          logabs_sign_network_hidden_dims=[128, 64, 64, 32, 32],
                          logabs_sign_network_dropout=0.3,
                          real_sign_network_hidden_dims=[128, 64, 64, 32, 32, 16],
                          real_sign_network_dropout=0.3).to(device)


    
    """
    # check for the consistency of the ft_cal implementation
    # first load the best model from previous training
    
    """
    # previous_model_path = r'H:\Postgraudate\Research\Test\SimpleMCKvibration\expwithsign_model_elu_newsignmodel_realtest64_finetunene_consistency_testInside.pt'
    # model.load_state_dict(torch.load(previous_model_path))
    # print(f"Loaded previous model from: {previous_model_path} for ft_cal consistency check.")


    # Configure losses
    loss_config = {
        "MSE": {"weight": 1, "use_relative": False, "use_log": True, "sign_bce_weight": 0, "real_sign_bce_weight": 0, "ft_cal_weight": 0},
        "Residual": {"weight": 0, "use_relative": True},
        "Consistency": {"weight": 0, "t_threshold": 1e-6, "use_log": True, "Input_grad_outside": True}  # Start with weight=0.0 to verify implementation
    }

    loss_fn = PINNLoss(model, loss_config)

    # Create separate optimizers for each network component
    mag_optimizer = torch.optim.Adam(model.network.parameters(), lr=0.005)
    finetune_optimizer = torch.optim.Adam(model.finetune_network.parameters(), lr=0.005)
    sign_optimizer = torch.optim.Adam(
        list(model.logabs_sign_network.parameters()) +
        list(model.real_sign_network.parameters()),
        lr=0.005
    )

    # Create separate schedulers for each optimizer
    mag_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        mag_optimizer, T_max=np.max([epochs//25,1]), eta_min=1e-12
    )
    finetune_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        finetune_optimizer, T_max=np.max([epochs//25,1]), eta_min=1e-12
    )
    sign_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        sign_optimizer, T_max=np.max([epochs//25,1]), eta_min=1e-12
    )

    # Prepare inputs_normalizer for consistency loss
    inputs_normalizer = train_val_inputs_normalizer

    # Validate VibrationResidualLoss on ground truth targets before training
    if loss_fn.has_loss("Residual"):
        print("\n" + "="*80)
        print("VALIDATING VibrationResidualLoss on Ground Truth Targets")
        print("="*80)
        print("Ground truth targets should satisfy physics equation perfectly.")
        print("Expected residual: ~0.0 (ideally < 1e-10)")
        print("-"*80)

        # Get use_relative setting from loss config
        use_relative = loss_config["Residual"].get("use_relative", False)

        mean_abs_res = checktargetres(
            train_loader,
            train_val_inputs_normalizer,
            train_val_targets_normalizer,
            device,
            dtype,
            use_relative=use_relative
        )

        if mean_abs_res < 1e-6:
            print("✓ PASSED: Target residual is very small (< 1e-6)")
            print("  VibrationResidualLoss implementation appears correct!")
        elif mean_abs_res < 1e-3:
            print("⚠ WARNING: Target residual is small but not negligible (< 1e-3)")
            print("  Check manual denormalization implementation.")
        else:
            print("✗ FAILED: Target residual is large (>= 1e-3)")
            print("  VibrationResidualLoss implementation may have errors!")
            print("  Please check the denormalization and physics equation.")

        print("="*80 + "\n")
    



    # Training loop
    # Input data shape: (batch_size, 3) -> [a, b, t]
    # Target data shape: (batch_size, 3) -> [x_t, v_t, a_t]
    best_combined_loss = float('inf')
    finetune_activation_epoch = int(epochs * 0.4)  # Activate finetune network after 40% of epochs



    for epoch in range(epochs):
        # Two-phase training logic
        if epoch == 0:
            # Phase 1 setup
            print(f"\n{'='*60}")
            print("PHASE 1: Training magnitude network + sign networks")
            print("Finetune network: Not optimized (gradients computed but optimizer not stepped)")
            print(f"{'='*60}")

        elif epoch == finetune_activation_epoch:
            # Phase 2 transition: Load best Phase 1 weights
            print(f"\n{'='*60}")
            print(f"PHASE 2 TRANSITION at epoch {epoch+1}/{epochs} (60% threshold)")
            print(f"Loading best Phase 1 weights from: {model_save_path}")
            print(f"{'='*60}")

            # Load best Phase 1 model
            model.load_state_dict(torch.load(model_save_path))

            # Reset best loss tracking for Phase 2
            best_combined_loss = float('inf')

            print(f"\n{'='*60}")
            # print("PHASE 2: Training finetune network + sign networks")
            # print("Magnitude network: Not optimized (gradients computed but optimizer not stepped)")
            print("PHASE 2: Training ALL networks (magnitude + finetune + sign)")
            print("Magnitude network: CONTINUE TRAINING (jointly with finetune)")
            print("Finetune network: NOW TRAINING (started from zeros)")
            print("Sign networks: CONTINUE TRAINING")
            print(f"{'='*60}")


        print(f"\nEpoch {epoch+1}/{epochs}")
        model.train()
        train_loss = 0.0
        train_loss_components = {}

        # Training progress bar
        train_pbar = tqdm(train_loader, desc=f"Training", leave=False)
        for inputs, targets in train_pbar:
            # Move data to device and convert to proper dtype
            inputs = inputs.to(device, dtype=dtype)
            targets = targets.to(device, dtype=dtype)

            # Keep full targets (batch, 6) - [real_signs (0-2), logabs_values (3-5)]
            # No extraction needed anymore

            # Zero gradients for all optimizers (ALL networks compute gradients)
            mag_optimizer.zero_grad()
            finetune_optimizer.zero_grad()
            sign_optimizer.zero_grad()

            # Denormalize inputs for loss calculation (if normalizer exists)
            if train_val_inputs_normalizer is not None:
                inputs_real = train_val_inputs_normalizer.denormalize_inputs(inputs).clone()
            else:
                inputs_real = inputs.clone()


            # Forward pass - returns (mag_preds, logabs_sign_pred, real_sign_pred, ft_cal) tuple
            if loss_fn.has_loss("Consistency"):
                consistency_config = loss_config.get("Consistency", {})
                input_grad_outside = consistency_config.get("Input_grad_outside", False)
                if input_grad_outside:
                    inputs.requires_grad_(True)
    
            mag_preds, logabs_sign_pred, real_sign_pred, ft_cal = model(inputs)

            # Prepare loss arguments
            loss_args = {}
            if loss_fn.has_loss("MSE"):
                # Pass mag_preds, targets (full 6 columns), logabs_sign_probs, real_sign_probs, ft_cal, output_normalizer
                logabs_sign_probs = model.logabs_last_sign_probs
                real_sign_probs = model.real_last_sign_probs
                loss_args["MSE"] = (mag_preds, targets, logabs_sign_probs, None, None, real_sign_probs, ft_cal, train_val_targets_normalizer)
            if loss_fn.has_loss("Residual"):
                # Determine ft_cal based on phase (merged ft_cal_for_residual and ft_cal_for_print)
                if epoch < finetune_activation_epoch:
                    ft_cal_phase = torch.zeros_like(ft_cal)  # Phase 1: no calibration
                else:
                    ft_cal_phase = ft_cal  # Phase 2: with calibration

                # Reconstruct outputs with phase-appropriate ft_cal (NO .detach()!)
                # Use logabs_sign_pred directly (not torch.sign) to keep it trainable
                logabs_values_residual = torch.sign(logabs_sign_pred) * (mag_preds + ft_cal_phase)
                # logabs_values_residual = SignWithHardTanh.apply(logabs_sign_pred) * (mag_preds + ft_cal_phase)
                outputs_for_residual = torch.cat([real_sign_pred, logabs_values_residual], dim=1)

                # Pass outputs, targets, inputs_real, and normalizer to residual loss
                loss_args["Residual"] = (outputs_for_residual, targets, inputs_real, train_val_targets_normalizer)
            if loss_fn.has_loss("Consistency"):
                # Determine ft_cal based on phase (same as Residual loss)
                if epoch < finetune_activation_epoch:
                    ft_cal_consistency = torch.zeros_like(ft_cal)  # Phase 1: no calibration
                else:
                    ft_cal_consistency = ft_cal  # Phase 2: with calibration

                # Check if Input_grad_outside mode is enabled
                consistency_config = loss_config.get("Consistency", {})
                input_grad_outside = consistency_config.get("Input_grad_outside", False)

                if input_grad_outside:
                    # MODE 1: Pass mag_preds and valid_mask
                    # Compute valid_mask for t_real > threshold
                    t_threshold = consistency_config.get("t_threshold", 1e-6)
                    t_normalized = inputs[:, 2]
                    t_mean = train_val_inputs_normalizer.log_mean['t']
                    t_std = train_val_inputs_normalizer.log_std['t']
                    ln10 = torch.log(torch.tensor(10.0, device=device, dtype=dtype))
                    t_real = torch.exp((t_std * t_normalized + t_mean) * ln10)
                    valid_mask = t_real > t_threshold

                    # Combine mag_preds + ft_cal_consistency outside (phase-aware)
                    mag_ft_cal_pred = mag_preds + ft_cal_consistency

                    loss_args["Consistency"] = (mag_preds, targets, inputs, train_val_inputs_normalizer,
                                               train_val_targets_normalizer, ft_cal_consistency, valid_mask)
                else:
                    # MODE 2: Pass None for predictions, consistency loss will call model internally
                    # New signature for MODE 2: (None, targets, inputs, inputs_normalizer, outputs_normalizer, ft_cal, None)
                    loss_args["Consistency"] = (None, targets, inputs, train_val_inputs_normalizer,
                                               train_val_targets_normalizer, ft_cal_consistency, None)
            
            # Reconstruct outputs to match target format: [real_signs (0-2), logabs_values (3-5)]
            # DETACH to prevent gradient blending between magnitude and sign
            signed_logabs = (mag_preds * torch.sign(logabs_sign_pred)).detach()
            outputs = torch.cat([real_sign_pred, signed_logabs], dim=1)
            # Compute loss
            loss, loss_dict = loss_fn(loss_args)
            loss.backward()  # Backward pass computes gradients for ALL networks

            # Step only the optimizers for current phase (selective weight updates)
            if epoch < finetune_activation_epoch:
                # Phase 1: Update magnitude + sign networks only
                mag_optimizer.step()
                sign_optimizer.step()
                # finetune_optimizer does NOT step → finetune_network weights unchanged
            else:
                # Phase 2: Update ALL networks (finetune + sign)
                # mag_optimizer.step()  # Now ALSO update mag network in Phase 2!
                finetune_optimizer.step()
                sign_optimizer.step()

            train_loss += loss.item() * inputs.size(0)

            # Accumulate loss components
            for key, value in loss_dict.items():
                if key not in train_loss_components:
                    train_loss_components[key] = 0.0
                train_loss_components[key] += value * inputs.size(0)

            # Update progress bar with current loss
            train_pbar.set_postfix({'loss': f'{loss.item():.4e}'})

        # Print the last output and last ground truth of the inputs and targets
        logabs_targets = targets[:, 3:]  # Extract for printing
        print("Last batch mag_preds v.s logabs_targets:", (torch.sign(logabs_sign_pred)*mag_preds)[-1].detach().cpu().numpy(), logabs_targets[-1].detach().cpu().numpy())

        # Compute uncalibrated and calibrated real values
        # Uncalibrated (no ft_cal)
        outputs_no_cal = (torch.sign(logabs_sign_pred) * mag_preds).detach()
        pred_normalized_no_cal = torch.cat([real_sign_pred, outputs_no_cal], dim=1).detach()
        real_value_pred_no_cal = train_val_targets_normalizer.denormalize_outputs(pred_normalized_no_cal[-1:].cpu().numpy())[0]

        # Calibrated (with ft_cal - always use full ft_cal for printing)
        outputs_ft_cal = (torch.sign(logabs_sign_pred)*(mag_preds + ft_cal)).detach()
        print("Last batch outputs_ft_cal v.s logabs_targets:", outputs_ft_cal[-1].cpu().numpy(), logabs_targets[-1].detach().cpu().numpy())
        pred_normalized = torch.cat([real_sign_pred, outputs_ft_cal], dim=1).detach()
        real_value_pred_cal = train_val_targets_normalizer.denormalize_outputs(pred_normalized[-1:].cpu().numpy())[0]

        # Ground truth
        real_value_gt = train_val_targets_normalizer.denormalize_outputs(targets.detach().cpu().numpy())[-1]

        # Print all three in one line: uncalibrated, calibrated, ground truth
        print("Last batch - pred_real_value (no cal) v.s pred_real_value (cal) v.s targets_real_value:",
              real_value_pred_no_cal, real_value_pred_cal, real_value_gt)

        # Print residuals if residual loss is active
        if loss_fn.has_loss("Residual"):
            # Extract parameter 'a' from inputs_real
            a_values = inputs_real[:, 0]  # (batch_size,)

            # Use phase-appropriate ft_cal (same as used in residual loss)
            # ft_cal_phase was already computed above when residual loss is active
            # Reconstruct predictions with phase-appropriate ft_cal
            # Use logabs_sign_pred directly (not torch.sign) for consistency
            logabs_values_print = logabs_sign_pred * (mag_preds + ft_cal_phase)
            outputs_for_print = torch.cat([real_sign_pred, logabs_values_print], dim=1)

            # Denormalize (for printing only, gradients already computed in loss)
            pred_real = train_val_targets_normalizer.denormalize_outputs(outputs_for_print.detach().cpu().numpy())
            target_real = train_val_targets_normalizer.denormalize_outputs(targets.detach().cpu().numpy())

            # Calculate physics residuals: (1/(2a))*a_t + 0.5*v_t - a*x_t
            # pred_real shape: (batch_size, 6) -> [x, v, a] are columns 0, 1, 2
            a_values_np = a_values.detach().cpu().numpy()
            pred_residual = (1/(2*a_values_np)) * pred_real[:, 2] + 0.5 * pred_real[:, 1] - a_values_np * pred_real[:, 0]
            target_residual = (1/(2*a_values_np)) * target_real[:, 2] + 0.5 * target_real[:, 1] - a_values_np * target_real[:, 0]

            # Take last batch sample
            pred_res_val = pred_residual[-1]
            target_res_val = target_residual[-1]

            print(f"Last batch pred_residual v.s targets_residual: [{pred_res_val:.6e}] [{target_res_val:.6e}]")

        # Save last batch data for consistency diagnostics (computed after validation)
        # Always save last batch for diagnostics (even if Consistency loss not enabled)
        # if loss_fn.has_loss("Consistency"):
        if True:
            last_batch_inputs = inputs.clone().detach()
            last_batch_targets = targets.clone().detach()

        train_loss /= len(train_loader.dataset)

        # Calculate average loss components
        for key in train_loss_components:
            train_loss_components[key] /= len(train_loader.dataset)

        # Step schedulers (update learning rates for optimizers that stepped this epoch)
        # Must be called AFTER optimizer.step() per PyTorch 1.1.0+ requirements
        if epoch < finetune_activation_epoch:
            # Phase 1: Only step schedulers for optimizers that stepped
            mag_scheduler.step()
            sign_scheduler.step()
            # Don't step finetune_scheduler (finetune_optimizer didn't step)
        else:
            # Phase 2: Step ALL schedulers (all optimizers stepped)
            # mag_scheduler.step()  # Now ALSO step mag scheduler in Phase 2!
            # Phase 2: Only step schedulers for optimizers that stepped
            finetune_scheduler.step()
            sign_scheduler.step()

        # Validation loop
        model.eval()
        val_loss = 0.0
        val_loss_components = {}

        # Initialize calibration metrics
        val_calibration_closer = 0
        val_calibration_total = 0

        # Determine if we need gradients for consistency loss
        # Consistency loss always uses auto-differentiation, so we need gradients
        use_no_grad_val = True
        if loss_fn.has_loss("Consistency"):
            use_no_grad_val = False

        # Conditionally use torch.no_grad() based on consistency type
        if use_no_grad_val:
            context_manager = torch.no_grad()
        else:
            context_manager = torch.enable_grad()

        with context_manager:
            # Validation progress bar
            val_pbar = tqdm(val_loader, desc=f"Validation", leave=False)
            for inputs, targets in val_pbar:
                # Move data to device and convert to proper dtype
                inputs = inputs.to(device, dtype=dtype)
                targets = targets.to(device, dtype=dtype) 

                # Keep full targets (batch, 6) - [real_signs (0-2), logabs_values (3-5)]
                # No extraction needed anymore

                # Denormalize inputs for loss calculation (if normalizer exists)
                if train_val_inputs_normalizer is not None:
                    inputs_real = train_val_inputs_normalizer.denormalize_inputs(inputs).clone()
                else:
                    inputs_real = inputs.clone()

                # Forward pass - returns (mag_preds, logabs_sign_pred, real_sign_pred, ft_cal) tuple
                if loss_fn.has_loss("Consistency"):
                    consistency_config = loss_config.get("Consistency", {})
                    input_grad_outside = consistency_config.get("Input_grad_outside", False)
                    if input_grad_outside:
                        inputs.requires_grad_(True)

                mag_preds, logabs_sign_pred, real_sign_pred, ft_cal = model(inputs)

                # Reconstruct outputs to match target format: [real_signs (0-2), logabs_values (3-5)]
                # DETACH to prevent gradient blending between magnitude and sign
                signed_logabs = (mag_preds * torch.sign(logabs_sign_pred)).detach()
                outputs = torch.cat([real_sign_pred, signed_logabs], dim=1)

                # Prepare loss arguments
                loss_args = {}
                if loss_fn.has_loss("MSE"):
                    # Pass mag_preds, targets (full 6 columns), logabs_sign_probs, real_sign_probs, ft_cal, output_normalizer
                    logabs_sign_probs = model.logabs_last_sign_probs
                    real_sign_probs = model.real_last_sign_probs
                    loss_args["MSE"] = (mag_preds, targets, logabs_sign_probs, None, None, real_sign_probs, ft_cal, train_val_targets_normalizer)
                if loss_fn.has_loss("Residual"):
                    # Determine ft_cal based on phase (merged ft_cal variable)
                    if epoch < finetune_activation_epoch:
                        ft_cal_phase = torch.zeros_like(ft_cal)  # Phase 1: no calibration
                    else:
                        ft_cal_phase = ft_cal  # Phase 2: with calibration

                    # Reconstruct outputs with phase-appropriate ft_cal (NO .detach()!)
                    # Use logabs_sign_pred directly (not torch.sign) to keep it trainable
                    logabs_values_residual = logabs_sign_pred * (mag_preds + ft_cal_phase)
                    outputs_for_residual = torch.cat([real_sign_pred, logabs_values_residual], dim=1)

                    # Pass outputs, targets, inputs_real, and normalizer to residual loss
                    loss_args["Residual"] = (outputs_for_residual, targets, inputs_real, train_val_targets_normalizer)
                if loss_fn.has_loss("Consistency"):
                    # Determine ft_cal based on phase (same as Residual loss)
                    if epoch < finetune_activation_epoch:
                        ft_cal_consistency = torch.zeros_like(ft_cal)  # Phase 1: no calibration
                    else:
                        ft_cal_consistency = ft_cal  # Phase 2: with calibration

                    # Check if Input_grad_outside mode is enabled
                    consistency_config = loss_config.get("Consistency", {})
                    input_grad_outside = consistency_config.get("Input_grad_outside", False)

                    if input_grad_outside:
                        # MODE 1: Pass mag_preds and valid_mask
                        # Note: In validation, inputs doesn't have requires_grad=True, so this mode won't compute gradients
                        # For validation, it's better to use MODE 2 (compute internally)
                        # But if user explicitly requests MODE 1, we respect it
                        t_threshold = consistency_config.get("t_threshold", 1e-6)
                        t_normalized = inputs[:, 2]
                        t_mean = train_val_inputs_normalizer.log_mean['t']
                        t_std = train_val_inputs_normalizer.log_std['t']
                        ln10 = torch.log(torch.tensor(10.0, device=device, dtype=dtype))
                        t_real = torch.exp((t_std * t_normalized + t_mean) * ln10)
                        valid_mask = t_real > t_threshold
                        
                        # Signature: (mag_preds, targets, inputs, inputs_normalizer, outputs_normalizer, ft_cal, valid_mask)
                        loss_args["Consistency"] = (mag_preds, targets, inputs, train_val_inputs_normalizer,
                                                   train_val_targets_normalizer, ft_cal_consistency, valid_mask)
                    else:
                        # MODE 2: Pass None for predictions
                        # Signature: (None, targets, inputs, inputs_normalizer, outputs_normalizer, ft_cal, None)
                        loss_args["Consistency"] = (None, targets, inputs, train_val_inputs_normalizer,
                                                   train_val_targets_normalizer, ft_cal_consistency, None)

                # Compute loss
                loss, loss_dict = loss_fn(loss_args)
                val_loss += loss.item() * inputs.size(0)

                # Accumulate loss components
                for key, value in loss_dict.items():
                    if key not in val_loss_components:
                        val_loss_components[key] = 0.0
                    val_loss_components[key] += value * inputs.size(0)

                # Accumulate calibration metrics
                if model.use_finetune:
                    logabs_targets = targets[:, 3:]
                    # Correct pattern: sign * (mag_preds + ft_cal)
                    outputs_ft_cal = (torch.sign(logabs_sign_pred) * (mag_preds + ft_cal)).detach()
                    outputs_before1 = (torch.sign(logabs_sign_pred) * mag_preds).detach()
                    outputs_before2 = outputs[:, 3:]
                    assert torch.allclose(outputs_before1, outputs_before2), "Outputs before ft_cal mismatch!"
                    error_before = torch.abs(torch.abs(outputs_before2) - torch.abs(logabs_targets))
                    error_after = torch.abs(torch.abs(outputs_ft_cal) - torch.abs(logabs_targets))
                    improvement = error_before - error_after
                    val_calibration_closer += (improvement > 0).sum().item()
                    val_calibration_total += improvement.numel()

                # Update progress bar with current loss
                val_pbar.set_postfix({'loss': f'{loss.item():.4e}'})

        val_loss /= len(val_loader.dataset)

        # Calculate average loss components
        for key in val_loss_components:
            val_loss_components[key] /= len(val_loader.dataset)

        # Compute consistency loss diagnostics (after validation, works for both MODE 1 and MODE 2)
        # Always run consistency diagnostics (even if Consistency loss not enabled)
        # if loss_fn.has_loss("Consistency") and 'last_batch_inputs' in locals():
        #     # Get consistency configuration
        if True:
            # Get consistency configuration (use defaults if not specified)
            consistency_config = loss_config.get("Consistency", {})
            t_threshold = consistency_config.get("t_threshold", 1e-6)

            # Prepare fresh inputs with gradient tracking (completely independent of training)
            diag_inputs = last_batch_inputs.clone().requires_grad_(True)
            diag_targets = last_batch_targets

            # DO A FRESH FORWARD PASS to get predictions connected to diag_inputs
            # This creates a new computational graph separate from training
            with torch.set_grad_enabled(True):
                diag_mag_preds, _, _, diag_ft_cal = model(diag_inputs)

            # Use phase-appropriate ft_cal (same logic as training)
            if epoch < finetune_activation_epoch:
                diag_ft_cal_phase = torch.zeros_like(diag_ft_cal)  # Phase 1
            else:
                diag_ft_cal_phase = diag_ft_cal  # Phase 2

            # Get normalizer stats
            t_mean = train_val_inputs_normalizer.log_mean['t']
            t_std = train_val_inputs_normalizer.log_std['t']
            mean_x = train_val_targets_normalizer.log_mean['x']
            std_x = train_val_targets_normalizer.log_std['x']
            mean_v = train_val_targets_normalizer.log_mean['v']
            std_v = train_val_targets_normalizer.log_std['v']
            mean_a = train_val_targets_normalizer.log_mean['a']
            std_a = train_val_targets_normalizer.log_std['a']
            ln10 = torch.log(torch.tensor(10.0, device=device, dtype=dtype))

            # Compute valid mask
            t_normalized = diag_inputs[:, 2]
            t_real = torch.exp((t_std * t_normalized + t_mean) * ln10)
            valid_mask = t_real > t_threshold

            # Combine mag_preds + ft_cal (now both are from the fresh forward pass)
            mag_ft_cal_pred = torch.abs(diag_mag_preds) + diag_ft_cal_phase
            mag_ft_cal_pred_valid = mag_ft_cal_pred[valid_mask]
            targets_valid = diag_targets[valid_mask]
            t_real_valid = t_real[valid_mask]

            if len(mag_ft_cal_pred_valid) > 0:
                mag_x = mag_ft_cal_pred_valid[:, 0]
                mag_v = mag_ft_cal_pred_valid[:, 1]
                mag_a = mag_ft_cal_pred_valid[:, 2]

                # Compute gradients (fresh graph, doesn't affect training)
                dx_prime_dt_prime = torch.autograd.grad(
                    outputs=mag_x,
                    inputs=diag_inputs,
                    grad_outputs=torch.ones_like(mag_x),
                    create_graph=False,
                    retain_graph=True,
                    allow_unused=True
                )[0]
                if dx_prime_dt_prime is not None:
                    dx_prime_dt_prime = dx_prime_dt_prime[valid_mask, 2]

                    dv_prime_dt_prime = torch.autograd.grad(
                        outputs=mag_v,
                        inputs=diag_inputs,
                        grad_outputs=torch.ones_like(mag_v),
                        create_graph=False,
                        retain_graph=False,  # Last grad call, can free graph
                        allow_unused=True
                    )[0]
                    if dv_prime_dt_prime is not None:
                        dv_prime_dt_prime = dv_prime_dt_prime[valid_mask, 2]

                        # Compute theory values from MODEL predictions
                        logabs_targets = targets_valid[:, 3:]
                        logabs_sign = torch.sign(logabs_targets)

                        x_pred = logabs_sign[:, 0] * mag_x
                        v_pred = logabs_sign[:, 1] * mag_v

                        x_real = torch.exp((std_x * x_pred.detach() + mean_x) * ln10)
                        v_real = torch.exp((std_v * v_pred.detach() + mean_v) * ln10)

                        eps = 1e-12
                        common_factor_v_model = (std_x / t_std) * (x_real / (t_real_valid + eps))
                        v_theory_model = torch.abs(common_factor_v_model * dx_prime_dt_prime.detach())

                        common_factor_a_model = (std_v / t_std) * (v_real / (t_real_valid + eps))
                        a_theory_model = torch.abs(common_factor_a_model * dv_prime_dt_prime.detach())

                        v_theory_model_normalized = (torch.log10(v_theory_model + eps) - mean_v) / std_v
                        a_theory_model_normalized = (torch.log10(a_theory_model + eps) - mean_a) / std_a

                        # Compute theory values from GROUND TRUTH targets (analytical)
                        # Extract target values in normalized log space
                        x_target = targets_valid[:, 3]  # x' normalized
                        v_target = targets_valid[:, 4]  # v' normalized

                        # Denormalize targets to real space
                        x_target_real = torch.exp((std_x * x_target + mean_x) * ln10)
                        v_target_real = torch.exp((std_v * v_target + mean_v) * ln10)

                        # Compute analytical derivatives dx'/dt' and dv'/dt' from ground truth
                        # Using formula from check_dataset_consistency: dx'/dt' = (std_t / std_x) * a * t_real
                        # where a is the exponential rate parameter
                        inputs_valid = diag_inputs[valid_mask]
                        # Denormalize inputs to get real 'a', 'b', 't' values (like in check_dataset_consistency)
                        inputs_valid_np = inputs_valid.detach().cpu().numpy()
                        denorm_inputs = train_val_inputs_normalizer.denormalize_inputs(inputs_valid_np)
                        a_param_real = torch.tensor(denorm_inputs[:, 0], device=device, dtype=dtype)

                        dx_dt_target_analytical = (t_std / std_x) * a_param_real * t_real_valid
                        dv_dt_target_analytical = (t_std / std_v) * a_param_real * t_real_valid

                        # Compute theory from target's derivatives
                        common_factor_v_target = (std_x / t_std) * (x_target_real / (t_real_valid + eps))
                        v_theory_target = torch.abs(common_factor_v_target * dx_dt_target_analytical)

                        common_factor_a_target = (std_v / t_std) * (v_target_real / (t_real_valid + eps))
                        a_theory_target = torch.abs(common_factor_a_target * dv_dt_target_analytical)

                        v_theory_target_normalized = (torch.log10(v_theory_target + eps) - mean_v) / std_v
                        a_theory_target_normalized = (torch.log10(a_theory_target + eps) - mean_a) / std_a

                        # Print diagnostics
                        v_theory_model_log = v_theory_model_normalized.detach().cpu().numpy()
                        a_theory_model_log = a_theory_model_normalized.detach().cpu().numpy()
                        v_theory_target_log = v_theory_target_normalized.detach().cpu().numpy()
                        a_theory_target_log = a_theory_target_normalized.detach().cpu().numpy()
                        v_model_log = mag_v.detach().cpu().numpy()
                        a_model_log = mag_a.detach().cpu().numpy()
                        v_target_log = targets_valid[:, 4].detach().cpu().numpy()
                        a_target_log = targets_valid[:, 5].detach().cpu().numpy()

                        if len(v_model_log) > 0:
                            print(f"Last batch logabs v_model v_theory_model v_target v_theory_target: [{abs(v_model_log[-1]):.8f} {abs(v_theory_model_log[-1]):.8f} {abs(v_target_log[-1]):.8f} {abs(v_theory_target_log[-1]):.8f}]")
                            print(f"Last batch logabs a_model a_theory_model a_target a_theory_target: [{abs(a_model_log[-1]):.8f} {abs(a_theory_model_log[-1]):.8f} {abs(a_target_log[-1]):.8f} {abs(a_theory_target_log[-1]):.8f}]")

        # Calculate calibration rate
        if model.use_finetune and val_calibration_total > 0:
            val_calibration_rate = val_calibration_closer / val_calibration_total * 100
        else:
            val_calibration_rate = None

        # Log training results to file (after validation)
        log_dict = {
            'epoch': epoch + 1,
            'outputs': outputs[-1],  # Last batch last sample
            'targets': logabs_targets[-1],
            'train_loss': train_loss,
            'val_calibration_rate': val_calibration_rate
        }

        # log_training_results(log_dict, results_folder=results_figure_folder, filename='training_explog.txt')
        log_training_results(log_dict, results_folder=model_and_result_folder, filename='training_explog.txt')

        # Print epoch summary
        if val_calibration_rate is not None:
            print(f"Epoch [{epoch+1}/{epochs}] -Model name: {os.path.basename(model_save_path)}  Train Loss: {train_loss:.4e}, Val Loss: {val_loss:.4e}, Calibration Closer rate: {val_calibration_rate:.2f}%")
        else:
            print(f"Epoch [{epoch+1}/{epochs}] -Model name: {os.path.basename(model_save_path)}  Train Loss: {train_loss:.4e}, Val Loss: {val_loss:.4e}")

        # Build train loss breakdown string with grouped loss types
        train_total = train_loss_components.get('total', train_loss)
        train_groups = []

        # Group 1: MSE Loss and its components
        if 'mse_loss' in train_loss_components:
            mse_value = train_loss_components['mse_loss']
            mse_ratio = (mse_value / train_total * 100) if train_total > 0 else 0
            mse_str = f"MSE_loss: {mse_value:.4e} ({mse_ratio:.2f}%)"

            # Add MSE sub-components in brackets
            mse_components = []
            if 'magnitude_loss' in train_loss_components:
                mag_value = train_loss_components['magnitude_loss']
                mag_ratio = (mag_value / mse_value * 100) if mse_value > 0 else 0
                mse_components.append(f"magnitude_loss: {mag_value:.4e} ({mag_ratio:.2f}%)")
            if 'ft_cal_loss' in train_loss_components:
                ft_value = train_loss_components['ft_cal_loss']
                ft_ratio = (ft_value / mse_value * 100) if mse_value > 0 else 0
                mse_components.append(f"ft_cal_loss: {ft_value:.4e} ({ft_ratio:.2f}%)")
            if 'logabs_sign_bce_loss' in train_loss_components:
                logabs_sign_value = train_loss_components['logabs_sign_bce_loss']
                logabs_sign_ratio = (logabs_sign_value / mse_value * 100) if mse_value > 0 else 0
                mse_components.append(f"logabs_sign_bce_loss: {logabs_sign_value:.4e} ({logabs_sign_ratio:.2f}%)")
            if 'real_sign_bce_loss' in train_loss_components:
                real_sign_value = train_loss_components['real_sign_bce_loss']
                real_sign_ratio = (real_sign_value / mse_value * 100) if mse_value > 0 else 0
                mse_components.append(f"real_sign_bce_loss: {real_sign_value:.4e} ({real_sign_ratio:.2f}%)")

            if mse_components:
                mse_str += f" [{' | '.join(mse_components)}]"
            train_groups.append(f"{{{mse_str}}}")

        # Group 2: Residual Loss
        if 'residual_loss' in train_loss_components:
            residual_value = train_loss_components['residual_loss']
            residual_ratio = (residual_value / train_total * 100) if train_total > 0 else 0
            train_groups.append(f"{{Residual_loss: {residual_value:.4e} ({residual_ratio:.2f}%)}}")

        # Group 3: Consistency Loss
        if 'consistency_loss' in train_loss_components:
            consistency_value = train_loss_components['consistency_loss']
            consistency_ratio = (consistency_value / train_total * 100) if train_total > 0 else 0
            train_groups.append(f"{{Consistency_loss: {consistency_value:.4e} ({consistency_ratio:.2f}%)}}")

        # Build val loss breakdown string with grouped loss types
        val_total = val_loss_components.get('total', val_loss)
        val_groups = []

        # Group 1: MSE Loss and its components
        if 'mse_loss' in val_loss_components:
            mse_value = val_loss_components['mse_loss']
            mse_ratio = (mse_value / val_total * 100) if val_total > 0 else 0
            mse_str = f"MSE_loss: {mse_value:.4e} ({mse_ratio:.2f}%)"

            # Add MSE sub-components in brackets
            mse_components = []
            if 'magnitude_loss' in val_loss_components:
                mag_value = val_loss_components['magnitude_loss']
                mag_ratio = (mag_value / mse_value * 100) if mse_value > 0 else 0
                mse_components.append(f"magnitude_loss: {mag_value:.4e} ({mag_ratio:.2f}%)")
            if 'ft_cal_loss' in val_loss_components:
                ft_value = val_loss_components['ft_cal_loss']
                ft_ratio = (ft_value / mse_value * 100) if mse_value > 0 else 0
                mse_components.append(f"ft_cal_loss: {ft_value:.4e} ({ft_ratio:.2f}%)")
            if 'logabs_sign_bce_loss' in val_loss_components:
                logabs_sign_value = val_loss_components['logabs_sign_bce_loss']
                logabs_sign_ratio = (logabs_sign_value / mse_value * 100) if mse_value > 0 else 0
                mse_components.append(f"logabs_sign_bce_loss: {logabs_sign_value:.4e} ({logabs_sign_ratio:.2f}%)")
            if 'real_sign_bce_loss' in val_loss_components:
                real_sign_value = val_loss_components['real_sign_bce_loss']
                real_sign_ratio = (real_sign_value / mse_value * 100) if mse_value > 0 else 0
                mse_components.append(f"real_sign_bce_loss: {real_sign_value:.4e} ({real_sign_ratio:.2f}%)")

            if mse_components:
                mse_str += f" [{' | '.join(mse_components)}]"
            val_groups.append(f"{{{mse_str}}}")

        # Group 2: Residual Loss
        if 'residual_loss' in val_loss_components:
            residual_value = val_loss_components['residual_loss']
            residual_ratio = (residual_value / val_total * 100) if val_total > 0 else 0
            val_groups.append(f"{{Residual_loss: {residual_value:.4e} ({residual_ratio:.2f}%)}}")

        # Group 3: Consistency Loss
        if 'consistency_loss' in val_loss_components:
            consistency_value = val_loss_components['consistency_loss']
            consistency_ratio = (consistency_value / val_total * 100) if val_total > 0 else 0
            val_groups.append(f"{{Consistency_loss: {consistency_value:.4e} ({consistency_ratio:.2f}%)}}")

        # Print both on 2 lines with aligned spacing
        train_breakdown = " ".join(train_groups)
        val_breakdown = " ".join(val_groups)
        print(f"  Train Loss: {train_breakdown}")
        print(f"  Val Loss  : {val_breakdown}")

        # Save the model if combined loss (train + val) has improved
        combined_loss = train_loss + val_loss
        if combined_loss < best_combined_loss:
            best_combined_loss = combined_loss
            save_checkpoint(model_save_path, model, train_val_inputs_normalizer, train_val_targets_normalizer)
            print(f"New best model saved with combined loss: {combined_loss:.4e} (train: {train_loss:.4e}, val: {val_loss:.4e})")

    # Testing loop
    print("\nRunning test evaluation on the best model...")
    # Load the best model for testing
    model, loaded_inputs_norm, loaded_outputs_norm = load_checkpoint(model_save_path, model)
    # Update normalizer references
    train_val_inputs_normalizer = loaded_inputs_norm
    train_val_targets_normalizer = loaded_outputs_norm
    model.eval()
    test_loss = 0.0
    test_loss_components = {}

    # Determine if we need gradients for consistency loss
    # Consistency loss always uses auto-differentiation, so we need gradients
    use_no_grad_test = True
    if loss_fn.has_loss("Consistency"):
        use_no_grad_test = False

    # Conditionally use torch.no_grad() based on consistency type
    if use_no_grad_test:
        context_manager = torch.no_grad()
    else:
        context_manager = torch.enable_grad()

    with context_manager:
        # Test progress bar
        test_pbar = tqdm(test_loader, desc="Testing", leave=True)
        for inputs, targets in test_pbar:
            # Move data to device and convert to proper dtype
            inputs = inputs.to(device, dtype=dtype)
            targets = targets.to(device, dtype=dtype)

            # Denormalize inputs for loss calculation (if normalizer exists)
            if test_inputs_normalizer is not None:
                inputs_real = test_inputs_normalizer.denormalize_inputs(inputs).clone()
            else:
                inputs_real = inputs.clone()

            # Forward pass - returns (mag_preds, logabs_sign_pred, real_sign_pred, ft_cal) tuple
            if loss_fn.has_loss("Consistency"):
                consistency_config = loss_config.get("Consistency", {})
                input_grad_outside = consistency_config.get("Input_grad_outside", False)
                if input_grad_outside:
                    inputs.requires_grad_(True)

            mag_preds, logabs_sign_pred, real_sign_pred, ft_cal = model(inputs)

            # Reconstruct outputs to match target format: [real_signs (0-2), logabs_values (3-5)]
            # DETACH to prevent gradient blending between magnitude and sign
            signed_logabs = (mag_preds * torch.sign(logabs_sign_pred)).detach()
            outputs = torch.cat([real_sign_pred, signed_logabs], dim=1)

            # Prepare loss arguments
            loss_args = {}
            if loss_fn.has_loss("MSE"):
                logabs_sign_probs = model.logabs_last_sign_probs
                real_sign_probs = model.real_last_sign_probs
                loss_args["MSE"] = (mag_preds, targets, logabs_sign_probs, None, None, real_sign_probs, ft_cal, test_targets_normalizer)
            if loss_fn.has_loss("Residual"):
                # Test time: use full ft_cal (Phase 2) as we're testing the fully trained model
                ft_cal_phase = ft_cal

                # Reconstruct outputs with ft_cal (NO .detach()!)
                # Use logabs_sign_pred directly (not torch.sign) to keep it trainable
                logabs_values_residual = logabs_sign_pred * (mag_preds + ft_cal_phase)
                outputs_for_residual = torch.cat([real_sign_pred, logabs_values_residual], dim=1)

                # Pass outputs, targets, inputs_real, and normalizer to residual loss
                loss_args["Residual"] = (outputs_for_residual, targets, inputs_real, test_targets_normalizer)
            if loss_fn.has_loss("Consistency"):
                # Test time: use full ft_cal (Phase 2) - fully trained model
                ft_cal_consistency = ft_cal

                # Check if Input_grad_outside mode is enabled
                consistency_config = loss_config.get("Consistency", {})
                input_grad_outside = consistency_config.get("Input_grad_outside", False)

                if input_grad_outside:
                    # MODE 1: Pass mag_preds and valid_mask
                    # Note: In test, inputs doesn't have requires_grad=True, so this mode won't compute gradients
                    # For test, it's better to use MODE 2 (compute internally)
                    t_threshold = consistency_config.get("t_threshold", 1e-6)
                    t_normalized = inputs[:, 2]
                    t_mean = test_inputs_normalizer.log_mean['t']
                    t_std = test_inputs_normalizer.log_std['t']
                    ln10 = torch.log(torch.tensor(10.0, device=device, dtype=dtype))
                    t_real = torch.exp((t_std * t_normalized + t_mean) * ln10)
                    valid_mask = t_real > t_threshold

                    # Signature: (mag_preds, targets, inputs, inputs_normalizer, outputs_normalizer, ft_cal, valid_mask)
                    loss_args["Consistency"] = (mag_preds, targets, inputs, test_inputs_normalizer,

                                               test_targets_normalizer, ft_cal_consistency, valid_mask)
                else:
                    # MODE 2: Pass None for predictions
                    # Signature: (None, targets, inputs, inputs_normalizer, outputs_normalizer, ft_cal, None)
                    loss_args["Consistency"] = (None, targets, inputs, test_inputs_normalizer,
                                               test_targets_normalizer, ft_cal_consistency, None)

            # Compute loss
            loss, loss_dict = loss_fn(loss_args)
            test_loss += loss.item() * inputs.size(0)

            # Accumulate loss components
            for key, value in loss_dict.items():
                if key not in test_loss_components:
                    test_loss_components[key] = 0.0
                test_loss_components[key] += value * inputs.size(0)

            # Update progress bar with current loss
            test_pbar.set_postfix({'loss': f'{loss.item():.4e}'})

    test_loss /= len(test_loader.dataset)

    # Calculate average loss components
    for key in test_loss_components:
        test_loss_components[key] /= len(test_loader.dataset)

    print(f"\nTest Loss: {test_loss:.4e}")

    # Build test loss breakdown string with grouped loss types
    test_total = test_loss_components.get('total', test_loss)
    test_groups = []

    # Group 1: MSE Loss and its components
    if 'mse_loss' in test_loss_components:
        mse_value = test_loss_components['mse_loss']
        mse_ratio = (mse_value / test_total * 100) if test_total > 0 else 0
        mse_str = f"MSE_loss: {mse_value:.4e} ({mse_ratio:.2f}%)"

        # Add MSE sub-components in brackets
        mse_components = []
        if 'magnitude_loss' in test_loss_components:
            mag_value = test_loss_components['magnitude_loss']
            mag_ratio = (mag_value / mse_value * 100) if mse_value > 0 else 0
            mse_components.append(f"magnitude_loss: {mag_value:.4e} ({mag_ratio:.2f}%)")
        if 'ft_cal_loss' in test_loss_components:
            ft_value = test_loss_components['ft_cal_loss']
            ft_ratio = (ft_value / mse_value * 100) if mse_value > 0 else 0
            mse_components.append(f"ft_cal_loss: {ft_value:.4e} ({ft_ratio:.2f}%)")
        if 'logabs_sign_bce_loss' in test_loss_components:
            logabs_sign_value = test_loss_components['logabs_sign_bce_loss']
            logabs_sign_ratio = (logabs_sign_value / mse_value * 100) if mse_value > 0 else 0
            mse_components.append(f"logabs_sign_bce_loss: {logabs_sign_value:.4e} ({logabs_sign_ratio:.2f}%)")
        if 'real_sign_bce_loss' in test_loss_components:
            real_sign_value = test_loss_components['real_sign_bce_loss']
            real_sign_ratio = (real_sign_value / mse_value * 100) if mse_value > 0 else 0
            mse_components.append(f"real_sign_bce_loss: {real_sign_value:.4e} ({real_sign_ratio:.2f}%)")

        if mse_components:
            mse_str += f" [{' | '.join(mse_components)}]"
        test_groups.append(f"{{{mse_str}}}")

    # Group 2: Residual Loss
    if 'residual_loss' in test_loss_components:
        residual_value = test_loss_components['residual_loss']
        residual_ratio = (residual_value / test_total * 100) if test_total > 0 else 0
        test_groups.append(f"{{Residual_loss: {residual_value:.4e} ({residual_ratio:.2f}%)}}")

    # Group 3: Consistency Loss
    if 'consistency_loss' in test_loss_components:
        consistency_value = test_loss_components['consistency_loss']
        consistency_ratio = (consistency_value / test_total * 100) if test_total > 0 else 0
        test_groups.append(f"{{Consistency_loss: {consistency_value:.4e} ({consistency_ratio:.2f}%)}}")

    # Print test loss breakdown
    test_breakdown = " ".join(test_groups)
    print(f"  Test Loss : {test_breakdown}")

    # =========================================================================
    # VERIFICATION: Re-run Training and Validation with Frozen Parameters
    # This proves the loaded model matches the saved best model
    # =========================================================================
    print("\n" + "="*80)
    print("VERIFICATION: Training and Validation Loss (Just for Proof)")
    print("Running forward pass with frozen parameters to verify loaded model...")
    print("="*80)

    model.eval()

    # Verification on Training Set
    verify_train_loss = 0.0
    verify_train_loss_components = {}

    # Determine if we need gradients for consistency loss
    use_no_grad_verify = True
    if loss_fn.has_loss("Consistency"):
        use_no_grad_verify = False

    if use_no_grad_verify:
        context_manager_verify = torch.no_grad()
    else:
        context_manager_verify = torch.enable_grad()

    with context_manager_verify:
        verify_train_pbar = tqdm(train_loader, desc="Verifying Train", leave=True)
        for inputs, targets in verify_train_pbar:
            inputs = inputs.to(device, dtype=dtype)
            targets = targets.to(device, dtype=dtype)

            if train_val_inputs_normalizer is not None:
                inputs_real = train_val_inputs_normalizer.denormalize_inputs(inputs).clone()
            else:
                inputs_real = inputs.clone()

            if loss_fn.has_loss("Consistency"):
                consistency_config = loss_config.get("Consistency", {})
                input_grad_outside = consistency_config.get("Input_grad_outside", False)
                if input_grad_outside:
                    inputs.requires_grad_(True)

            mag_preds, logabs_sign_pred, real_sign_pred, ft_cal = model(inputs)

            # Prepare loss arguments (use Phase 2 ft_cal - fully trained model)
            loss_args = {}
            if loss_fn.has_loss("MSE"):
                logabs_sign_probs = model.logabs_last_sign_probs
                real_sign_probs = model.real_last_sign_probs
                loss_args["MSE"] = (mag_preds, targets, logabs_sign_probs, None, None, real_sign_probs, ft_cal, train_val_targets_normalizer)
            if loss_fn.has_loss("Residual"):
                ft_cal_phase = ft_cal  # Phase 2: with calibration
                logabs_values_residual = SignWithHardTanh.apply(logabs_sign_pred) * (mag_preds + ft_cal_phase)
                outputs_for_residual = torch.cat([real_sign_pred, logabs_values_residual], dim=1)
                loss_args["Residual"] = (outputs_for_residual, targets, inputs_real, train_val_targets_normalizer)
            if loss_fn.has_loss("Consistency"):
                ft_cal_consistency = ft_cal  # Phase 2: with calibration
                consistency_config = loss_config.get("Consistency", {})
                input_grad_outside = consistency_config.get("Input_grad_outside", False)

                if input_grad_outside:
                    t_threshold = consistency_config.get("t_threshold", 1e-6)
                    t_normalized = inputs[:, 2]
                    t_mean = train_val_inputs_normalizer.log_mean['t']
                    t_std = train_val_inputs_normalizer.log_std['t']
                    ln10 = torch.log(torch.tensor(10.0, device=device, dtype=dtype))
                    t_real = torch.exp((t_std * t_normalized + t_mean) * ln10)
                    valid_mask = t_real > t_threshold
                    loss_args["Consistency"] = (mag_preds, targets, inputs, train_val_inputs_normalizer,
                                               train_val_targets_normalizer, ft_cal_consistency, valid_mask)
                else:
                    loss_args["Consistency"] = (None, targets, inputs, train_val_inputs_normalizer,
                                               train_val_targets_normalizer, ft_cal_consistency, None)

            loss, loss_dict = loss_fn(loss_args)
            verify_train_loss += loss.item() * inputs.size(0)

            for key, value in loss_dict.items():
                if key not in verify_train_loss_components:
                    verify_train_loss_components[key] = 0.0
                verify_train_loss_components[key] += value * inputs.size(0)

            verify_train_pbar.set_postfix({'loss': f'{loss.item():.4e}'})

    verify_train_loss /= len(train_loader.dataset)

    # Calculate average loss components
    for key in verify_train_loss_components:
        verify_train_loss_components[key] /= len(train_loader.dataset)

    # Verification on Validation Set
    verify_val_loss = 0.0
    verify_val_loss_components = {}

    with context_manager_verify:
        verify_val_pbar = tqdm(val_loader, desc="Verifying Val", leave=True)
        for inputs, targets in verify_val_pbar:
            inputs = inputs.to(device, dtype=dtype)
            targets = targets.to(device, dtype=dtype)

            if train_val_inputs_normalizer is not None:
                inputs_real = train_val_inputs_normalizer.denormalize_inputs(inputs).clone()
            else:
                inputs_real = inputs.clone()

            if loss_fn.has_loss("Consistency"):
                consistency_config = loss_config.get("Consistency", {})
                input_grad_outside = consistency_config.get("Input_grad_outside", False)
                if input_grad_outside:
                    inputs.requires_grad_(True)

            mag_preds, logabs_sign_pred, real_sign_pred, ft_cal = model(inputs)

            # Prepare loss arguments (use Phase 2 ft_cal - fully trained model)
            loss_args = {}
            if loss_fn.has_loss("MSE"):
                logabs_sign_probs = model.logabs_last_sign_probs
                real_sign_probs = model.real_last_sign_probs
                loss_args["MSE"] = (mag_preds, targets, logabs_sign_probs, None, None, real_sign_probs, ft_cal, train_val_targets_normalizer)
            if loss_fn.has_loss("Residual"):
                ft_cal_phase = ft_cal  # Phase 2: with calibration
                logabs_values_residual = logabs_sign_pred * (mag_preds + ft_cal_phase)
                outputs_for_residual = torch.cat([real_sign_pred, logabs_values_residual], dim=1)
                loss_args["Residual"] = (outputs_for_residual, targets, inputs_real, train_val_targets_normalizer)
            if loss_fn.has_loss("Consistency"):
                ft_cal_consistency = ft_cal  # Phase 2: with calibration
                consistency_config = loss_config.get("Consistency", {})
                input_grad_outside = consistency_config.get("Input_grad_outside", False)

                if input_grad_outside:
                    t_threshold = consistency_config.get("t_threshold", 1e-6)
                    t_normalized = inputs[:, 2]
                    t_mean = train_val_inputs_normalizer.log_mean['t']
                    t_std = train_val_inputs_normalizer.log_std['t']
                    ln10 = torch.log(torch.tensor(10.0, device=device, dtype=dtype))
                    t_real = torch.exp((t_std * t_normalized + t_mean) * ln10)
                    valid_mask = t_real > t_threshold
                    loss_args["Consistency"] = (mag_preds, targets, inputs, train_val_inputs_normalizer,
                                               train_val_targets_normalizer, ft_cal_consistency, valid_mask)
                else:
                    loss_args["Consistency"] = (None, targets, inputs, train_val_inputs_normalizer,
                                               train_val_targets_normalizer, ft_cal_consistency, None)

            loss, loss_dict = loss_fn(loss_args)
            verify_val_loss += loss.item() * inputs.size(0)

            for key, value in loss_dict.items():
                if key not in verify_val_loss_components:
                    verify_val_loss_components[key] = 0.0
                verify_val_loss_components[key] += value * inputs.size(0)

            verify_val_pbar.set_postfix({'loss': f'{loss.item():.4e}'})

    verify_val_loss /= len(val_loader.dataset)

    # Calculate average loss components
    for key in verify_val_loss_components:
        verify_val_loss_components[key] /= len(val_loader.dataset)

    # Print verification results
    print("\n" + "="*80)
    print("VERIFICATION RESULTS (Just for Proof - Model Loaded from Best Checkpoint)")
    print("="*80)
    print(f"Verified Train Loss: {verify_train_loss:.4e}")
    print(f"Verified Val Loss:   {verify_val_loss:.4e}")
    print(f"Combined Loss:       {verify_train_loss + verify_val_loss:.4e}")

    # Build verify train loss breakdown string with grouped loss types
    verify_train_total = verify_train_loss_components.get('total', verify_train_loss)
    verify_train_groups = []

    # Group 1: MSE Loss and its components
    if 'mse_loss' in verify_train_loss_components:
        mse_value = verify_train_loss_components['mse_loss']
        mse_ratio = (mse_value / verify_train_total * 100) if verify_train_total > 0 else 0
        mse_str = f"MSE_loss: {mse_value:.4e} ({mse_ratio:.2f}%)"

        # Add MSE sub-components in brackets
        mse_components = []
        if 'magnitude_loss' in verify_train_loss_components:
            mag_value = verify_train_loss_components['magnitude_loss']
            mag_ratio = (mag_value / mse_value * 100) if mse_value > 0 else 0
            mse_components.append(f"magnitude_loss: {mag_value:.4e} ({mag_ratio:.2f}%)")
        if 'ft_cal_loss' in verify_train_loss_components:
            ft_value = verify_train_loss_components['ft_cal_loss']
            ft_ratio = (ft_value / mse_value * 100) if mse_value > 0 else 0
            mse_components.append(f"ft_cal_loss: {ft_value:.4e} ({ft_ratio:.2f}%)")
        if 'logabs_sign_bce_loss' in verify_train_loss_components:
            logabs_sign_value = verify_train_loss_components['logabs_sign_bce_loss']
            logabs_sign_ratio = (logabs_sign_value / mse_value * 100) if mse_value > 0 else 0
            mse_components.append(f"logabs_sign_bce_loss: {logabs_sign_value:.4e} ({logabs_sign_ratio:.2f}%)")
        if 'real_sign_bce_loss' in verify_train_loss_components:
            real_sign_value = verify_train_loss_components['real_sign_bce_loss']
            real_sign_ratio = (real_sign_value / mse_value * 100) if mse_value > 0 else 0
            mse_components.append(f"real_sign_bce_loss: {real_sign_value:.4e} ({real_sign_ratio:.2f}%)")

        if mse_components:
            mse_str += f" [{' | '.join(mse_components)}]"
        verify_train_groups.append(f"{{{mse_str}}}")

    # Group 2: Residual Loss
    if 'residual_loss' in verify_train_loss_components:
        residual_value = verify_train_loss_components['residual_loss']
        residual_ratio = (residual_value / verify_train_total * 100) if verify_train_total > 0 else 0
        verify_train_groups.append(f"{{Residual_loss: {residual_value:.4e} ({residual_ratio:.2f}%)}}")

    # Group 3: Consistency Loss
    if 'consistency_loss' in verify_train_loss_components:
        consistency_value = verify_train_loss_components['consistency_loss']
        consistency_ratio = (consistency_value / verify_train_total * 100) if verify_train_total > 0 else 0
        verify_train_groups.append(f"{{Consistency_loss: {consistency_value:.4e} ({consistency_ratio:.2f}%)}}")

    # Build verify val loss breakdown string with grouped loss types
    verify_val_total = verify_val_loss_components.get('total', verify_val_loss)
    verify_val_groups = []

    # Group 1: MSE Loss and its components
    if 'mse_loss' in verify_val_loss_components:
        mse_value = verify_val_loss_components['mse_loss']
        mse_ratio = (mse_value / verify_val_total * 100) if verify_val_total > 0 else 0
        mse_str = f"MSE_loss: {mse_value:.4e} ({mse_ratio:.2f}%)"

        # Add MSE sub-components in brackets
        mse_components = []
        if 'magnitude_loss' in verify_val_loss_components:
            mag_value = verify_val_loss_components['magnitude_loss']
            mag_ratio = (mag_value / mse_value * 100) if mse_value > 0 else 0
            mse_components.append(f"magnitude_loss: {mag_value:.4e} ({mag_ratio:.2f}%)")
        if 'ft_cal_loss' in verify_val_loss_components:
            ft_value = verify_val_loss_components['ft_cal_loss']
            ft_ratio = (ft_value / mse_value * 100) if mse_value > 0 else 0
            mse_components.append(f"ft_cal_loss: {ft_value:.4e} ({ft_ratio:.2f}%)")
        if 'logabs_sign_bce_loss' in verify_val_loss_components:
            logabs_sign_value = verify_val_loss_components['logabs_sign_bce_loss']
            logabs_sign_ratio = (logabs_sign_value / mse_value * 100) if mse_value > 0 else 0
            mse_components.append(f"logabs_sign_bce_loss: {logabs_sign_value:.4e} ({logabs_sign_ratio:.2f}%)")
        if 'real_sign_bce_loss' in verify_val_loss_components:
            real_sign_value = verify_val_loss_components['real_sign_bce_loss']
            real_sign_ratio = (real_sign_value / mse_value * 100) if mse_value > 0 else 0
            mse_components.append(f"real_sign_bce_loss: {real_sign_value:.4e} ({real_sign_ratio:.2f}%)")

        if mse_components:
            mse_str += f" [{' | '.join(mse_components)}]"
        verify_val_groups.append(f"{{{mse_str}}}")

    # Group 2: Residual Loss
    if 'residual_loss' in verify_val_loss_components:
        residual_value = verify_val_loss_components['residual_loss']
        residual_ratio = (residual_value / verify_val_total * 100) if verify_val_total > 0 else 0
        verify_val_groups.append(f"{{Residual_loss: {residual_value:.4e} ({residual_ratio:.2f}%)}}")

    # Group 3: Consistency Loss
    if 'consistency_loss' in verify_val_loss_components:
        consistency_value = verify_val_loss_components['consistency_loss']
        consistency_ratio = (consistency_value / verify_val_total * 100) if verify_val_total > 0 else 0
        verify_val_groups.append(f"{{Consistency_loss: {consistency_value:.4e} ({consistency_ratio:.2f}%)}}")

    # Print both on 2 lines with aligned spacing
    verify_train_breakdown = " ".join(verify_train_groups)
    verify_val_breakdown = " ".join(verify_val_groups)
    print(f"  Verified Train Loss: {verify_train_breakdown}")
    print(f"  Verified Val Loss  : {verify_val_breakdown}")

    print("="*80)
    print("Note: These losses should match the best epoch during training.")
    print("="*80 + "\n")

    # Generate prediction performance plots
    prediction_performance(
        data_path=Plot_data_source,
        model_pt_path=model_save_path,
        model=model,
        inputs_normalizer=train_val_inputs_normalizer,
        outputs_normalizer=train_val_targets_normalizer,
        device=device,
        dtype=dtype,
        data_sampling_step=100,
        figure_folder=model_and_result_folder  # Fixed: use correct parameter name
    )

# """
if __name__ == "__main__":
    main()
    # testdataloaderunchange()
