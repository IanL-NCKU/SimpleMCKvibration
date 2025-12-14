from Exp_dataset_temp import *
from Exp_modelandloss import *
from expdatagenerator import *
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import numpy as np

def checktargetres(dataloader, inputs_normalizer, targets_normalizer, device, dtype, use_relative=False):
    """
    Check target residuals to validate ExponentialResidualLoss implementation.
    Ground truth targets should satisfy physics equation: (1/(2a))*a_t + 0.5*v_t - a*x_t = 0

    Args:
        dataloader: DataLoader containing (inputs, targets)
        inputs_normalizer: Normalizer for inputs (to get real-space 'a' parameter)
        targets_normalizer: Normalizer for targets (for manual denormalization)
        device: torch device
        dtype: torch dtype
        use_relative: If True, compute scale-invariant relative residual (default: False)

    Returns:
        mean_abs_residual: Mean absolute residual value
    """
    all_residuals = []

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device, dtype=dtype)
            targets = targets.to(device, dtype=dtype)

            # Extract parameter 'a' from normalized inputs
            # inputs shape: (batch_size, 3) -> [a, b, t] (normalized)
            # We need real-space 'a' for the physics equation
            inputs_real = inputs_normalizer.denormalize_inputs(inputs)
            a = inputs_real[:, 0]  # exponential rate parameter

            # DIAGNOSTIC: Check data shapes (only print for first batch)
            if len(all_residuals) == 0:
                print(f"\n[DIAGNOSTIC] Data shapes:")
                print(f"  inputs shape: {inputs.shape}")
                print(f"  targets shape: {targets.shape}")
                print(f"  inputs_real shape: {inputs_real.shape}")
                print(f"  Expected targets: (batch_size, 6) = [signs(3), logabs(3)]")

            # Manual denormalization (same as ExponentialResidualLoss)
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

            # Physics residual: (1/(2a))*a_t + 0.5*v_t - a*x_t = 0
            eps = 1e-12
            residual = (1.0 / (2.0 * a + eps)) * a_t + 0.5 * v_t - a * x_t\

            # residual = torch.log(torch.abs((1.0 / (2.0 * a + eps)) * a_t)) - torch.log(torch.abs(0.5 * v_t - a * x_t)) \
            #             + torch.log(torch.abs(a * x_t)) - torch.log(torch.abs((1.0 / (2.0 * a + eps)) * a_t + 0.5 * v_t))\
            #             + torch.log(torch.abs(0.5 * v_t)) - torch.log(torch.abs(a * x_t - (1.0 / (2.0 * a + eps)) * a_t))


            if use_relative:
                # Scale-invariant relative residual (same as ExponentialResidualLoss)
                # Normalize by target's acceleration term: (1/(2a))*a_target
                scale = torch.abs((1.0 / (2.0 * a + eps)) * a_t) + eps
                residual = residual / scale

            all_residuals.append(residual)

    # Concatenate all residuals and compute mean absolute value
    all_residuals = torch.cat(all_residuals, dim=0)
    mean_abs_residual = torch.mean(torch.abs(all_residuals))

    residual_type = "relative" if use_relative else "absolute"
    print(f"Target mean absolute {residual_type} residual: {mean_abs_residual.item():.6e}")
    print(f"Target {residual_type} residual range: [{all_residuals.min().item():.6e}, {all_residuals.max().item():.6e}]")

    return mean_abs_residual.item()

def log_training_results(log_dict, results_folder='./results', filename='training_log.txt', delimiter=', '):
    """Log training results to a delimited text file."""
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


def prediction_performance(data_path, model_pt_path, model, normalizer, device, dtype=torch.float32, data_sampling_step=1, figure_folder='./figures'):
    """Generate prediction performance scatter plots."""
    print(f"\n{'='*60}")
    print("Generating Prediction Performance Plots")
    print(f"{'='*60}")

    if not os.path.exists(figure_folder):
        os.makedirs(figure_folder)
        print(f"Created folder: {figure_folder}")

    model.load_state_dict(torch.load(model_pt_path))
    model.eval()
    print(f"Loaded model from: {model_pt_path}")

    test_loader, _, _, _, _ = load_exponential_data(
        filepath=data_path,
        batch_size=256,
        normalize=True,
        shuffle_train=False
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

            # The model's mag_preds are the predicted log-absolute values.
            # These are directly comparable to the log-absolute part of the targets.
            predictions = mag_preds.detach()

            # Extract log-absolute targets for comparison
            # Targets shape: (batch, 6) -> [real_signs (0-2), logabs_values (3-5)]
            logabs_targets = targets[:, 3:]
            # Apply additive calibration: outputs = signed_mag_preds + ft_cal
            outputs = ( logabs_sign_pred* (mag_preds + ft_cal)).detach()
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
    device_index = 0
    train_in_64 = True
    epochs = 40

    # Data paths
    Train_Val_data_source = r'.\new_exponential_trainval_data.npz'
    Test_data_source = r'.\new_exponential_test_data.npz'
    Plot_data_source = r'.\new_exponential_test_data.npz'
    data_normalize = True
    
    # Check raw data residuals (before normalization)
    check_raw_data_residuals(Train_Val_data_source, use_relative=True)
    # check_raw_data_residuals(Test_data_source, use_relative=True)
    # examinenormalizer(Train_Val_data_source)
    # Load the dataset
    train_loader, val_loader, _, train_val_inputs_normalizer, train_val_targets_normalizer = load_exponential_data(
        filepath=Train_Val_data_source,
        batch_size=1024,
        normalize=data_normalize,
        shuffle_train=False
    )

    test_loader, _, _, test_inputs_normalizer, test_targets_normalizer = load_exponential_data(
        filepath=Test_data_source,
        batch_size=1024,
        normalize=data_normalize,
        shuffle_train=False
    )
    
    

def main():
    device_index = 0
    train_in_64 = True
    epochs = 1000

    # Data paths
    Train_Val_data_source = r'.\new_exponential_trainval_data.npz'
    Test_data_source = r'.\new_exponential_test_data.npz'
    Plot_data_source = r'.\new_exponential_test_data.npz'
    data_normalize = True

    # Check raw data residuals (before normalization)
    check_raw_data_residuals(Train_Val_data_source, use_relative=True)
    # check_raw_data_residuals(Test_data_source, use_relative=True)

    # Load the dataset
    train_loader, val_loader, _, train_val_inputs_normalizer, train_val_targets_normalizer = load_exponential_data(
        filepath=Train_Val_data_source,
        batch_size=1024,
        normalize=data_normalize,
        shuffle_train=True
    )

    test_loader, _, _, test_inputs_normalizer, test_targets_normalizer = load_exponential_data(
        filepath=Test_data_source,
        batch_size=1024,
        normalize=data_normalize,
        shuffle_train=False
    )

    print(f"Data loaders created:")

    # Setup device
    device = torch.device(f'cuda:{device_index}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Setup float64 training if requested
    if train_in_64:
        torch.set_default_dtype(torch.float64)
        dtype = torch.float64
        print("Training in float64 (double precision) mode")
    else:
        dtype = torch.float32
        print("Training in float32 (single precision) mode")

    model_save_path = 'expwithsign_model_elu_newsignmodel_realtest64_finetunene_residualok.pt'
    results_figure_folder = './expwithsign_results_elu_newsignmodel_realtest64_finetunene_residualok'

    # Create the Exponential PINN model
    model = ExponentialPINN_ver3(hidden_dims=[16, 32, 64, 64, 32, 16],
                          activation='elu',
                          use_log_output=False,
                          use_finetune=True,
                          finetune_hidden_dims=[32, 128, 32],
                          finetune_scale=10,
                          logabs_sign_network_hidden_dims=[128, 64, 64, 32, 32],
                          logabs_sign_network_dropout=0.3,
                          real_sign_network_hidden_dims=[64, 64, 32, 32],
                          real_sign_network_dropout=0.3).to(device)

    # model = ExponentialPINN(hidden_dims=[16, 32, 32, 64, 32, 32, 16],
    #                       activation='relu',
    #                       use_log_output=False,
    #                       use_finetune=True,
    #                       finetune_hidden_dims=[16, 32, 64, 32, 16],
    #                       finetune_scale=1,
    #                       use_sign_network=False,
    #                       sign_network_hidden_dims=[16, 32, 32, 16]).to(device)

    # Configure losses
    loss_config = {
        "MSE": {"weight": 0.9, "use_relative": False, "use_log": True, "sign_bce_weight": 1.0, "real_sign_bce_weight": 1.0, "ft_cal_weight": 1.0},
        "Residual": {"weight": 0.1, "use_relative": True},
        "Consistency": {"weight": 0.0, "t_threshold": 1e-5, "type": "auto", "use_relative": True, "use_log": False}
    }

    loss_fn = ExponentialPINNLoss(model, loss_config)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=np.max([epochs//10,1]), eta_min=1e-12)

    # Prepare inputs_normalizer for consistency loss
    inputs_normalizer = train_val_inputs_normalizer

    # Validate ExponentialResidualLoss on ground truth targets before training
    if loss_fn.has_loss("Residual"):
        print("\n" + "="*80)
        print("VALIDATING ExponentialResidualLoss on Ground Truth Targets")
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
            print("  ExponentialResidualLoss implementation appears correct!")
        elif mean_abs_res < 1e-3:
            print("⚠ WARNING: Target residual is small but not negligible (< 1e-3)")
            print("  Check manual denormalization implementation.")
        else:
            print("✗ FAILED: Target residual is large (>= 1e-3)")
            print("  ExponentialResidualLoss implementation may have errors!")
            print("  Please check the denormalization and physics equation.")

        print("="*80 + "\n")

    # Training loop
    # Input data shape: (batch_size, 3) -> [a, b, t]
    # Target data shape: (batch_size, 3) -> [x_t, v_t, a_t]
    best_combined_loss = float('inf')
    finetune_activation_epoch = int(epochs * 0.6)  # Activate finetune network after 60% of epochs

    for epoch in range(epochs):
        # Two-phase training logic
        if epoch == 0:
            # Phase 1 setup: Freeze finetune network, train magnitude + sign networks
            print(f"\n{'='*60}")
            print("PHASE 1: Training magnitude network + sign networks")
            print("Finetune network: FROZEN")
            print(f"{'='*60}")
            model.freeze_finetune_network()
            model.unfreeze_magnitude_network()

        elif epoch == finetune_activation_epoch:
            # Phase 2 transition: Load best Phase 1 weights, freeze magnitude, unfreeze finetune
            print(f"\n{'='*60}")
            print(f"PHASE 2 TRANSITION at epoch {epoch+1}/{epochs} (25% threshold)")
            print(f"Loading best Phase 1 weights from: {model_save_path}")
            print(f"{'='*60}")

            # Load best Phase 1 model
            model.load_state_dict(torch.load(model_save_path))

            # Switch network freeze states
            model.freeze_magnitude_network()
            model.unfreeze_finetune_network()

            # Reset best loss tracking for Phase 2
            best_combined_loss = float('inf')

            print(f"\n{'='*60}")
            print("PHASE 2: Training finetune network + sign networks")
            print("Magnitude network: FROZEN")
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

            optimizer.zero_grad()

            # Denormalize inputs for loss calculation (if normalizer exists)
            if train_val_inputs_normalizer is not None:
                inputs_real = train_val_inputs_normalizer.denormalize_inputs(inputs).clone()
            else:
                inputs_real = inputs.clone()

            # Forward pass - returns (mag_preds, logabs_sign_pred, real_sign_pred, ft_cal) tuple
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
                # Determine ft_cal based on phase (merged ft_cal_for_residual and ft_cal_for_print)
                if epoch < finetune_activation_epoch:
                    ft_cal_phase = torch.zeros_like(ft_cal)  # Phase 1: no calibration
                else:
                    ft_cal_phase = ft_cal  # Phase 2: with calibration

                # Reconstruct outputs with phase-appropriate ft_cal (NO .detach()!)
                # Use logabs_sign_pred directly (not torch.sign) to keep it trainable
                # logabs_values_residual = torch.sign(logabs_sign_pred) * (mag_preds + ft_cal_phase)
                logabs_values_residual = SignWithHardTanh.apply(logabs_sign_pred) * (mag_preds + ft_cal_phase)
                outputs_for_residual = torch.cat([real_sign_pred, logabs_values_residual], dim=1)

                # Pass outputs, targets, inputs_real, and normalizer to residual loss
                loss_args["Residual"] = (outputs_for_residual, targets, inputs_real, train_val_targets_normalizer)
            if loss_fn.has_loss("Consistency"):
                # Auto diff only
                loss_args["Consistency"] = (inputs, inputs_real, inputs_normalizer)

            # Compute loss
            loss, loss_dict = loss_fn(loss_args)
            loss.backward()
            optimizer.step()
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

        train_loss /= len(train_loader.dataset)

        # Calculate average loss components
        for key in train_loss_components:
            train_loss_components[key] /= len(train_loader.dataset)

        # Validation loop
        model.eval()
        val_loss = 0.0
        val_loss_components = {}

        # Initialize calibration metrics
        val_calibration_closer = 0
        val_calibration_total = 0

        # Determine if we need gradients for consistency loss (auto-diff type)
        use_no_grad_val = True
        if loss_fn.has_loss("Consistency") and loss_config["Consistency"]["type"] == "auto":
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
                    # Auto diff only
                    loss_args["Consistency"] = (inputs, inputs_real, inputs_normalizer)

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

        lr_scheduler.step()

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
        log_training_results(log_dict, results_folder=results_figure_folder, filename='training_explog.txt')

        # Print epoch summary
        if val_calibration_rate is not None:
            print(f"Epoch [{epoch+1}/{epochs}] -Model name: {os.path.basename(model_save_path)}  Train Loss: {train_loss:.4e}, Val Loss: {val_loss:.4e}, Calibration Closer rate: {val_calibration_rate:.2f}%")
        else:
            print(f"Epoch [{epoch+1}/{epochs}] -Model name: {os.path.basename(model_save_path)}  Train Loss: {train_loss:.4e}, Val Loss: {val_loss:.4e}")

        # Build train loss breakdown string with MSE components grouped
        train_total = train_loss_components.get('total', train_loss)
        train_parts = []

        # Handle MSE loss and its components specially
        if 'mse_loss' in train_loss_components:
            mse_value = train_loss_components['mse_loss']
            mse_ratio = (mse_value / train_total * 100) if train_total > 0 else 0
            mse_str = f"mse_loss: {mse_value:.4e} ({mse_ratio:.2f}%)"

            # Check if we have magnitude and sign components
            mse_components = []
            if 'magnitude_loss' in train_loss_components:
                mag_value = train_loss_components['magnitude_loss']
                mag_ratio = (mag_value / mse_value * 100) if mse_value > 0 else 0
                mse_components.append(f"magnitude_loss: {mag_value:.4e} ({mag_ratio:.2f}%)")
            if 'sign_bce_loss' in train_loss_components:
                sign_value = train_loss_components['sign_bce_loss']
                sign_ratio = (sign_value / mse_value * 100) if mse_value > 0 else 0
                mse_components.append(f"sign_bce_loss: {sign_value:.4e} ({sign_ratio:.2f}%)")

            # Add MSE with components in brackets if they exist
            if mse_components:
                mse_str += f" [{' | '.join(mse_components)}]"
            train_parts.append(mse_str)

        # Add other losses (excluding mse_loss, magnitude_loss, sign_bce_loss, total)
        for key in sorted(train_loss_components.keys()):
            if key not in ['total', 'mse_loss', 'magnitude_loss', 'sign_bce_loss']:
                value = train_loss_components[key]
                ratio = (value / train_total * 100) if train_total > 0 else 0
                train_parts.append(f"{key}: {value:.4e} ({ratio:.2f}%)")

        # Build val loss breakdown string with MSE components grouped
        val_total = val_loss_components.get('total', val_loss)
        val_parts = []

        # Handle MSE loss and its components specially
        if 'mse_loss' in val_loss_components:
            mse_value = val_loss_components['mse_loss']
            mse_ratio = (mse_value / val_total * 100) if val_total > 0 else 0
            mse_str = f"mse_loss: {mse_value:.4e} ({mse_ratio:.2f}%)"

            # Check if we have magnitude and sign components
            mse_components = []
            if 'magnitude_loss' in val_loss_components:
                mag_value = val_loss_components['magnitude_loss']
                mag_ratio = (mag_value / mse_value * 100) if mse_value > 0 else 0
                mse_components.append(f"magnitude_loss: {mag_value:.4e} ({mag_ratio:.2f}%)")
            if 'sign_bce_loss' in val_loss_components:
                sign_value = val_loss_components['sign_bce_loss']
                sign_ratio = (sign_value / mse_value * 100) if mse_value > 0 else 0
                mse_components.append(f"sign_bce_loss: {sign_value:.4e} ({sign_ratio:.2f}%)")

            # Add MSE with components in brackets if they exist
            if mse_components:
                mse_str += f" [{' | '.join(mse_components)}]"
            val_parts.append(mse_str)

        # Add other losses (excluding mse_loss, magnitude_loss, sign_bce_loss, total)
        for key in sorted(val_loss_components.keys()):
            if key not in ['total', 'mse_loss', 'magnitude_loss', 'sign_bce_loss']:
                value = val_loss_components[key]
                ratio = (value / val_total * 100) if val_total > 0 else 0
                val_parts.append(f"{key}: {value:.4e} ({ratio:.2f}%)")

        # Print both on 2 lines with aligned spacing
        train_breakdown = " | ".join(train_parts)
        val_breakdown = " | ".join(val_parts)
        print(f"  Train Loss: {train_breakdown}")
        print(f"  Val Loss  : {val_breakdown}")

        # Save the model if combined loss (train + val) has improved
        combined_loss = train_loss + val_loss
        if combined_loss < best_combined_loss:
            best_combined_loss = combined_loss
            torch.save(model.state_dict(), model_save_path)
            print(f"New best model saved with combined loss: {combined_loss:.4e} (train: {train_loss:.4e}, val: {val_loss:.4e})")

    # Testing loop
    print("\nRunning test evaluation on the best model...")
    # Load the best model for testing
    model.load_state_dict(torch.load(model_save_path))
    model.eval()
    test_loss = 0.0
    test_loss_components = {}

    # Determine if we need gradients for consistency loss (auto-diff type)
    use_no_grad_test = True
    if loss_fn.has_loss("Consistency") and loss_config["Consistency"]["type"] == "auto":
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
                # Auto diff only
                loss_args["Consistency"] = (inputs, inputs_real, test_inputs_normalizer)

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
    print(f"\nTest Loss: {test_loss:.4e}")

    # Generate prediction performance plots
    prediction_performance(
        data_path=Plot_data_source,
        model_pt_path=model_save_path,
        model=model,
        normalizer=train_val_inputs_normalizer,
        device=device,
        dtype=dtype,
        data_sampling_step=100,
        figure_folder=results_figure_folder
    )

if __name__ == "__main__":
    main()
    # testcudaavailable()
    # testdataloaderunchange()