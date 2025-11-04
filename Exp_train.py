from Exp_dataset import *
from Exp_modelandloss import *
from expdatagenerator import *
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import numpy as np

def log_training_results(log_dict, results_folder='./results', filename='training_log.txt', delimiter=', '):
    """Log training results to a delimited text file."""
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    log_path = os.path.join(results_folder, filename)
    epoch = log_dict['epoch']
    outputs = log_dict['outputs']
    targets = log_dict['targets']
    train_loss = log_dict['train_loss']

    if torch.is_tensor(outputs):
        outputs = outputs.detach().cpu().numpy()
    if torch.is_tensor(targets):
        targets = targets.detach().cpu().numpy()

    file_exists = os.path.isfile(log_path)

    with open(log_path, 'a') as f:
        if not file_exists:
            header_fields = ["epoch", "output_x", "output_v", "output_a",
                           "target_x", "target_v", "target_a", "train_loss"]
            f.write(delimiter.join(header_fields) + "\n")

        data_fields = [f"{epoch}", f"{outputs[0]:.6e}", f"{outputs[1]:.6e}", f"{outputs[2]:.6e}",
                      f"{targets[0]:.6e}", f"{targets[1]:.6e}", f"{targets[2]:.6e}", f"{train_loss:.6e}"]
        f.write(delimiter.join(data_fields) + "\n")

    return log_path


def prediction_performance(data_path, model_pt_path, model, normalizer, device, data_sampling_step=1, figure_folder='./figures'):
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
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            all_predictions.append(outputs.cpu().numpy())
            all_targets.append(targets.numpy())

    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    print(f"Total data points: {len(all_predictions)}")

    sampled_indices = np.arange(0, len(all_predictions), data_sampling_step)
    predictions_sampled = all_predictions[sampled_indices]
    targets_sampled = all_targets[sampled_indices]

    print(f"Sampled data points (step={data_sampling_step}): {len(predictions_sampled)}")

    output_names = ['x', 'v', 'a']
    output_titles = [
        'Position Prediction Performance',
        'Velocity Prediction Performance',
        'Acceleration Prediction Performance'
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


def main():
    device_index = 0
    epochs = 200

    # Data paths
    Train_Val_data_source = r'E:\Ian\PINNexample\exponential_trainval_data.npz'
    Test_data_source = r'E:\Ian\PINNexample\exponential_test_data.npz'
    Plot_data_source = r'E:\Ian\PINNexample\exponential_test_data.npz'
    data_normalize = True
    # Load the dataset
    train_loader, val_loader, _, train_val_inputs_normalizer, train_val_targets_normalizer = load_exponential_data(
        filepath=Train_Val_data_source,
        batch_size=512,
        normalize=data_normalize,
        shuffle_train=True
    )

    test_loader, _, _, test_inputs_normalizer, test_targets_normalizer = load_exponential_data(
        filepath=Test_data_source,
        batch_size=512,
        normalize=data_normalize,
        shuffle_train=False
    )

    print(f"Data loaders created:")

    # Setup device
    device = torch.device(f'cuda:{device_index}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model_save_path = 'exp_model_relu_signmodel.pt'
    results_figure_folder = './exp_results_relu_signmodel'

    # Create the Exponential PINN model
    model = ExponentialPINN(hidden_dims=[16, 32, 32, 64, 32, 32, 16],
                          activation='relu',
                          use_log_output=False,
                          use_finetune=True,
                          finetune_hidden_dims=[16, 32, 64, 32, 16],
                          finetune_scale=1,
                          use_sign_network=True,
                          sign_network_hidden_dims=[16, 32, 32, 16]).to(device)

    # Configure losses
    loss_config = {
        "MSE": {"weight": 1.0, "use_relative": False, "use_log": True},
        "Residual": {"weight": 0.0, "use_relative": True},
        "Consistency": {"weight": 0.0, "t_threshold": 1e-5, "type": "auto", "use_relative": True, "use_log": False}
    }

    loss_fn = ExponentialPINNLoss(model, loss_config)

    # Dual-optimizer setup for separate magnitude and sign training
    # Main network optimizer: updates network + finetune_network with magnitude loss
    main_params = list(model.network.parameters())
    if model.use_finetune:
        main_params += list(model.finetune_network.parameters())
    optimizer_main = torch.optim.Adam(main_params, lr=0.005)

    # Sign network optimizer: updates sign_network with sign loss (only if enabled)
    if model.use_sign_network:
        optimizer_sign = torch.optim.Adam(model.sign_network.parameters(), lr=0.005)
    else:
        optimizer_sign = None

    # Learning rate schedulers for both optimizers
    lr_scheduler_main = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_main, T_max=np.max([epochs//20,1]), eta_min=1e-12)
    if optimizer_sign is not None:
        lr_scheduler_sign = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_sign, T_max=np.max([epochs//20,1]), eta_min=1e-12)
    else:
        lr_scheduler_sign = None

    # Prepare norm_params for consistency loss
    norm_params = {'normalizer': train_val_inputs_normalizer}

    # Training loop
    # Input data shape: (batch_size, 3) -> [a, b, t]
    # Target data shape: (batch_size, 3) -> [x_t, v_t, a_t]
    best_combined_loss = float('inf')
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        model.train()
        train_loss = 0.0
        train_loss_components = {}

        # Training progress bar
        train_pbar = tqdm(train_loader, desc=f"Training", leave=False)
        for inputs, targets in train_pbar:
            # Move data to device
            inputs, targets = inputs.to(device), targets.to(device)

            # Denormalize inputs for loss calculation (if normalizer exists)
            if train_val_inputs_normalizer is not None:
                inputs_real = train_val_inputs_normalizer.denormalize_inputs(inputs).clone()
            else:
                inputs_real = inputs.clone()

            # Build inputs_combined based on which losses are enabled
            inputs_list = [inputs]
            N = inputs.size(0)

            # Generate perturbed time samples if Consistency loss with finite type is enabled
            if loss_fn.has_loss("Consistency") and loss_config["Consistency"]["type"] == "finite":
                t_threshold = loss_config["Consistency"]["t_threshold"]

                inputs_t_minus_minus_real = inputs_real.clone()
                inputs_t_minus_minus_real[:, 2] = inputs_real[:, 2] - 2 * t_threshold
                if train_val_inputs_normalizer is not None:
                    inputs_t_minus_minus = torch.FloatTensor(train_val_inputs_normalizer.normalize_inputs(inputs_t_minus_minus_real.cpu().numpy())).to(device)
                else:
                    inputs_t_minus_minus = inputs_t_minus_minus_real

                inputs_t_minus_real = inputs_real.clone()
                inputs_t_minus_real[:, 2] = inputs_real[:, 2] - t_threshold
                if train_val_inputs_normalizer is not None:
                    inputs_t_minus = torch.FloatTensor(train_val_inputs_normalizer.normalize_inputs(inputs_t_minus_real.cpu().numpy())).to(device)
                else:
                    inputs_t_minus = inputs_t_minus_real

                inputs_t_plus_real = inputs_real.clone()
                inputs_t_plus_real[:, 2] = inputs_real[:, 2] + t_threshold
                if train_val_inputs_normalizer is not None:
                    inputs_t_plus = torch.FloatTensor(train_val_inputs_normalizer.normalize_inputs(inputs_t_plus_real.cpu().numpy())).to(device)
                else:
                    inputs_t_plus = inputs_t_plus_real

                inputs_t_plus_plus_real = inputs_real.clone()
                inputs_t_plus_plus_real[:, 2] = inputs_real[:, 2] + 2 * t_threshold
                if train_val_inputs_normalizer is not None:
                    inputs_t_plus_plus = torch.FloatTensor(train_val_inputs_normalizer.normalize_inputs(inputs_t_plus_plus_real.cpu().numpy())).to(device)
                else:
                    inputs_t_plus_plus = inputs_t_plus_plus_real

                inputs_list.extend([inputs_t_minus_minus, inputs_t_minus, inputs_t_plus, inputs_t_plus_plus])

            # Stack all inputs
            inputs_combined = torch.cat(inputs_list, dim=0)

            # Forward pass
            outputs_combined = model(inputs_combined)

            # Split outputs based on what was stacked
            outputs = outputs_combined[:N]
            idx = N

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

            # Compute loss
            loss, loss_dict = loss_fn(loss_args)

            # Dual-optimizer training: separate magnitude and sign updates
            if 'magnitude_loss_raw' in loss_dict and 'sign_loss_raw' in loss_dict and optimizer_sign is not None:
                # Dual-optimizer mode with TWO SEPARATE FORWARD PASSES
                # This avoids in-place operation conflicts in computational graph

                # ===== PASS 1: Update sign network with sign loss =====
                optimizer_sign.zero_grad()
                sign_loss_weighted = loss_dict['sign_loss_raw'] * loss_fn.loss_components["MSE"].weight
                sign_loss_weighted.backward()
                optimizer_sign.step()

                # ===== PASS 2: Fresh forward pass for main network =====
                # After sign network update, we do a fresh forward pass for main network
                optimizer_main.zero_grad()

                # Fresh forward pass with updated sign network
                outputs_combined_fresh = model(inputs_combined)
                outputs_fresh = outputs_combined_fresh[:N]

                if loss_fn.has_loss("Consistency") and loss_config["Consistency"]["type"] == "finite":
                    outputs_dt_fresh = outputs_combined_fresh[N:N+4*N]
                else:
                    outputs_dt_fresh = None

                # Recompute magnitude loss with fresh outputs
                loss_args_fresh = {}
                if loss_fn.has_loss("MSE"):
                    loss_args_fresh["MSE"] = (outputs_fresh, targets)
                if loss_fn.has_loss("Residual"):
                    loss_args_fresh["Residual"] = (outputs_fresh, inputs_real)
                if loss_fn.has_loss("Consistency"):
                    consistency_type = loss_config["Consistency"]["type"]
                    if consistency_type == "finite":
                        loss_args_fresh["Consistency"] = (outputs_fresh, outputs_dt_fresh, targets)
                    elif consistency_type == "auto":
                        loss_args_fresh["Consistency"] = (inputs, inputs_real, norm_params)

                _, loss_dict_fresh = loss_fn(loss_args_fresh)

                # Build magnitude loss for main network
                magnitude_loss_weighted = loss_dict_fresh['magnitude_loss_raw'] * loss_fn.loss_components["MSE"].weight

                # Add physics losses (they affect main network parameters)
                if loss_fn.has_loss("Residual"):
                    residual_tensor = loss_fn.loss_components["Residual"].compute(
                        outputs_fresh, None, None, None, inputs_real
                    ) * loss_fn.loss_components["Residual"].weight
                    magnitude_loss_weighted = magnitude_loss_weighted + residual_tensor

                if loss_fn.has_loss("Consistency"):
                    consistency_type = loss_config["Consistency"]["type"]
                    if consistency_type == "auto":
                        consistency_tensor = loss_fn.loss_components["Consistency"].compute(
                            None, None, inputs, norm_params, inputs_real
                        ) * loss_fn.loss_components["Consistency"].weight
                    elif consistency_type == "finite":
                        consistency_tensor = loss_fn.loss_components["Consistency"].compute(
                            outputs_fresh, targets, None, None, outputs_dt_fresh
                        ) * loss_fn.loss_components["Consistency"].weight
                    else:
                        consistency_tensor = torch.tensor(0.0, device=outputs_fresh.device)
                    magnitude_loss_weighted = magnitude_loss_weighted + consistency_tensor

                magnitude_loss_weighted.backward()
                optimizer_main.step()
            else:
                # Standard single-optimizer mode (fallback for when use_log=False or no sign network)
                optimizer_main.zero_grad()
                loss.backward()
                optimizer_main.step()

            train_loss += loss.item() * inputs.size(0)

            # Accumulate loss components
            for key, value in loss_dict.items():
                if key not in train_loss_components:
                    train_loss_components[key] = 0.0
                train_loss_components[key] += value * inputs.size(0)

            # Update progress bar with current loss
            train_pbar.set_postfix({'loss': f'{loss.item():.4e}'})

        # Print the last output and last ground truth of the inputs and targets
        print("Last batch outputs v.s targets:", outputs[-1].detach().cpu().numpy(), targets[-1].detach().cpu().numpy())

        train_loss /= len(train_loader.dataset)

        # Calculate average loss components
        for key in train_loss_components:
            train_loss_components[key] /= len(train_loader.dataset)

        # Log training results to file (before validation)
        log_dict = {
            'epoch': epoch + 1,
            'outputs': outputs[-1],  # Last batch last sample
            'targets': targets[-1],
            'train_loss': train_loss
        }
        log_training_results(log_dict, results_folder=results_figure_folder, filename='training_explog.txt')

        # Validation loop
        model.eval()
        val_loss = 0.0
        val_loss_components = {}

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
                # Move data to device
                inputs, targets = inputs.to(device), targets.to(device)

                # Denormalize inputs for loss calculation (if normalizer exists)
                if train_val_inputs_normalizer is not None:
                    inputs_real = train_val_inputs_normalizer.denormalize_inputs(inputs).clone()
                else:
                    inputs_real = inputs.clone()

                # Build inputs_combined based on which losses are enabled
                inputs_list = [inputs]
                N = inputs.size(0)

                # Generate perturbed time samples if Consistency loss with finite type is enabled
                if loss_fn.has_loss("Consistency") and loss_config["Consistency"]["type"] == "finite":
                    t_threshold = loss_config["Consistency"]["t_threshold"]

                    inputs_t_minus_minus_real = inputs_real.clone()
                    inputs_t_minus_minus_real[:, 2] = inputs_real[:, 2] - 2 * t_threshold
                    if train_val_inputs_normalizer is not None:
                        inputs_t_minus_minus = torch.FloatTensor(train_val_inputs_normalizer.normalize_inputs(inputs_t_minus_minus_real.cpu().numpy())).to(device)
                    else:
                        inputs_t_minus_minus = inputs_t_minus_minus_real

                    inputs_t_minus_real = inputs_real.clone()
                    inputs_t_minus_real[:, 2] = inputs_real[:, 2] - t_threshold
                    if train_val_inputs_normalizer is not None:
                        inputs_t_minus = torch.FloatTensor(train_val_inputs_normalizer.normalize_inputs(inputs_t_minus_real.cpu().numpy())).to(device)
                    else:
                        inputs_t_minus = inputs_t_minus_real

                    inputs_t_plus_real = inputs_real.clone()
                    inputs_t_plus_real[:, 2] = inputs_real[:, 2] + t_threshold
                    if train_val_inputs_normalizer is not None:
                        inputs_t_plus = torch.FloatTensor(train_val_inputs_normalizer.normalize_inputs(inputs_t_plus_real.cpu().numpy())).to(device)
                    else:
                        inputs_t_plus = inputs_t_plus_real

                    inputs_t_plus_plus_real = inputs_real.clone()
                    inputs_t_plus_plus_real[:, 2] = inputs_real[:, 2] + 2 * t_threshold
                    if train_val_inputs_normalizer is not None:
                        inputs_t_plus_plus = torch.FloatTensor(train_val_inputs_normalizer.normalize_inputs(inputs_t_plus_plus_real.cpu().numpy())).to(device)
                    else:
                        inputs_t_plus_plus = inputs_t_plus_plus_real

                    inputs_list.extend([inputs_t_minus_minus, inputs_t_minus, inputs_t_plus, inputs_t_plus_plus])

                # Stack all inputs
                inputs_combined = torch.cat(inputs_list, dim=0)

                # Forward pass
                outputs_combined = model(inputs_combined)

                # Split outputs based on what was stacked
                outputs = outputs_combined[:N]
                idx = N

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

        # Update learning rate schedulers for both optimizers
        lr_scheduler_main.step()
        if lr_scheduler_sign is not None:
            lr_scheduler_sign.step()

        # Print epoch summary
        print(f"Epoch [{epoch+1}/{epochs}] -Model name: {os.path.basename(model_save_path)}  Train Loss: {train_loss:.4e}, Val Loss: {val_loss:.4e}")

        # Build train loss breakdown string
        train_total = train_loss_components.get('total', train_loss)
        train_parts = []
        for key in sorted(train_loss_components.keys()):
            if key != 'total':
                value = train_loss_components[key]
                ratio = (value / train_total * 100) if train_total > 0 else 0
                train_parts.append(f"{key}: {value:.4e} ({ratio:.2f}%)")

        # Build val loss breakdown string
        val_total = val_loss_components.get('total', val_loss)
        val_parts = []
        for key in sorted(val_loss_components.keys()):
            if key != 'total':
                value = val_loss_components[key]
                ratio = (value / val_total * 100) if val_total > 0 else 0
                val_parts.append(f"{key}: {value:.4e} ({ratio:.2f}%)")

        # Print both on 2 lines with aligned spacing
        train_breakdown = "  |  ".join(train_parts)
        val_breakdown = "  |  ".join(val_parts)
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
            # Move data to device
            inputs, targets = inputs.to(device), targets.to(device)

            # Denormalize inputs for loss calculation (if normalizer exists)
            if test_inputs_normalizer is not None:
                inputs_real = test_inputs_normalizer.denormalize_inputs(inputs).clone()
            else:
                inputs_real = inputs.clone()

            # Build inputs_combined based on which losses are enabled
            inputs_list = [inputs]
            N = inputs.size(0)

            # Generate perturbed time samples if Consistency loss with finite type is enabled
            if loss_fn.has_loss("Consistency") and loss_config["Consistency"]["type"] == "finite":
                t_threshold = loss_config["Consistency"]["t_threshold"]

                inputs_t_minus_minus_real = inputs_real.clone()
                inputs_t_minus_minus_real[:, 2] = inputs_real[:, 2] - 2 * t_threshold
                if test_inputs_normalizer is not None:
                    inputs_t_minus_minus = torch.FloatTensor(test_inputs_normalizer.normalize_inputs(inputs_t_minus_minus_real.cpu().numpy())).to(device)
                else:
                    inputs_t_minus_minus = inputs_t_minus_minus_real

                inputs_t_minus_real = inputs_real.clone()
                inputs_t_minus_real[:, 2] = inputs_real[:, 2] - t_threshold
                if test_inputs_normalizer is not None:
                    inputs_t_minus = torch.FloatTensor(test_inputs_normalizer.normalize_inputs(inputs_t_minus_real.cpu().numpy())).to(device)
                else:
                    inputs_t_minus = inputs_t_minus_real

                inputs_t_plus_real = inputs_real.clone()
                inputs_t_plus_real[:, 2] = inputs_real[:, 2] + t_threshold
                if test_inputs_normalizer is not None:
                    inputs_t_plus = torch.FloatTensor(test_inputs_normalizer.normalize_inputs(inputs_t_plus_real.cpu().numpy())).to(device)
                else:
                    inputs_t_plus = inputs_t_plus_real

                inputs_t_plus_plus_real = inputs_real.clone()
                inputs_t_plus_plus_real[:, 2] = inputs_real[:, 2] + 2 * t_threshold
                if test_inputs_normalizer is not None:
                    inputs_t_plus_plus = torch.FloatTensor(test_inputs_normalizer.normalize_inputs(inputs_t_plus_plus_real.cpu().numpy())).to(device)
                else:
                    inputs_t_plus_plus = inputs_t_plus_plus_real

                inputs_list.extend([inputs_t_minus_minus, inputs_t_minus, inputs_t_plus, inputs_t_plus_plus])

            # Stack all inputs
            inputs_combined = torch.cat(inputs_list, dim=0)

            # Forward pass
            outputs_combined = model(inputs_combined)

            # Split outputs based on what was stacked
            outputs = outputs_combined[:N]
            idx = N

            if loss_fn.has_loss("Consistency") and loss_config["Consistency"]["type"] == "finite":
                outputs_dt = outputs_combined[idx:idx+4*N]  # 4N samples: [t-2Δt, t-Δt, t+Δt, t+2Δt]
                idx += 4*N

            # Prepare norm_params for test (use test_inputs_normalizer)
            norm_params_test = {'normalizer': test_inputs_normalizer}

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
        normalizer=train_val_inputs_normalizer,
        device=device,
        data_sampling_step=100,
        figure_folder=results_figure_folder
    )

if __name__ == "__main__":
    main()
