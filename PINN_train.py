from PINN_dataset import *
from PINN_modelandloss import *
from datagtgenerator import *
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

def prediction_performance(data_path, model_pt_path, model, normalizer, device, data_sampling_step=1, figure_folder='./figures'):
    """
    Generate prediction performance scatter plots comparing ground truth vs predictions.

    Args:
        data_path: Path to the test data .npz file
        model_pt_path: Path to the saved model .pt file
        model: The model instance used for training
        normalizer: The normalizer instance from training data
        device: Device to run inference on (CPU or CUDA)
        data_sampling_step: Sample every N-th data point (default: 1, use all data)
        figure_folder: Folder path to save the figures (default: './figures')

    Returns:
        None (saves figures to disk)
    """
    print(f"\n{'='*60}")
    print("Generating Prediction Performance Plots")
    print(f"{'='*60}")

    # Create figure folder if it doesn't exist
    if not os.path.exists(figure_folder):
        os.makedirs(figure_folder)
        print(f"Created folder: {figure_folder}")

    # Load model weights
    model.load_state_dict(torch.load(model_pt_path))
    model.eval()
    print(f"Loaded model from: {model_pt_path}")

    # Load test data
    test_loader, _, _, _ = load_vibration_data(
        filepath=data_path,
        batch_size=256,
        normalize=True,
        shuffle_train=False
    )
    print(f"Loaded test data from: {data_path}")

    # Collect all predictions and ground truth
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)

            all_predictions.append(outputs.cpu().numpy())
            all_targets.append(targets.numpy())

    # Concatenate all batches
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    print(f"Total data points: {len(all_predictions)}")

    # Apply sampling
    sampled_indices = np.arange(0, len(all_predictions), data_sampling_step)
    predictions_sampled = all_predictions[sampled_indices]
    targets_sampled = all_targets[sampled_indices]

    print(f"Sampled data points (step={data_sampling_step}): {len(predictions_sampled)}")

    # Define output names and titles
    output_names = ['x', 'v', 'a']
    output_titles = [
        'Position Prediction Performance',
        'Velocity Prediction Performance',
        'Acceleration Prediction Performance'
    ]

    # Generate scatter plots for each output
    for idx, (name, title) in enumerate(zip(output_names, output_titles)):
        plt.figure(figsize=(8, 8))

        ground_truth = targets_sampled[:, idx]
        predictions = predictions_sampled[:, idx]

        # Scatter plot
        plt.scatter(ground_truth, predictions, alpha=0.5, s=20)

        # Perfect prediction line (y=x)
        min_val = min(ground_truth.min(), predictions.min())
        max_val = max(ground_truth.max(), predictions.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r-', linewidth=2, label='Perfect Prediction (y=x)')

        # Add grid
        plt.grid(True, alpha=0.3)

        # Labels and title
        plt.xlabel('Ground Truth', fontsize=12)
        plt.ylabel('Prediction', fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.legend()

        # Make plot square
        plt.axis('equal')

        # Save figure
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
    
    epochs = 100
    Train_Val_data_source = r'E:\Ian\PINNexample\train_val_vibration_data.npz'
    Test_data_source = r'E:\Ian\PINNexample\test_vibration_data.npz'
    Plot_data_source = r'E:\Ian\PINNexample\new_test_vibration_data.npz'
    # Load the dataset 
    train_loader, val_loader, _, train_val_normalizer = load_vibration_data(
        filepath= Train_Val_data_source,
        batch_size=512,
        normalize=True,
        shuffle_train=True
    )

    test_loader, _, _, test_normalizer = load_vibration_data(
        filepath= Test_data_source,
        batch_size=512,
        normalize=True,
        shuffle_train=False
    )

    print(f"Data loaders created:")

    # Setup device
    device = torch.device(f'cuda:{device_index}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model_save_path = 'test_new.pt'#'best_model_tanh_withlog_withfinetune.pt'
    # Create the PINN model with log-space output
    model = VibrationPINN(hidden_dims=[32, 128, 512, 2048, 512, 128, 32], 
                          activation='ELU',
                          use_log_output=False, 
                          use_finetune=True, 
                          finetune_hidden_dims=[128, 128], 
                          finetune_scale= 1).to(device)

    # Configure losses using dict-based interface
    loss_config = {
        "MSE": {"weight": 0.7, "use_relative": True, "use_log": True},
        "Residual": {"weight": 0.1, "use_relative": True},
        "InitialCondition": {"weight": 0.1, "t_threshold": 1e-8, "use_relative": True},
        "Consistency": {"weight": 0.1, "t_threshold": 1e-5, "type": "auto", "use_relative": True, "use_log": False}
    }

    loss_fn = PINNLoss_v2(model, loss_config)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=np.max([epochs//10,1]), eta_min=1e-9)

    # Prepare norm_params for consistency loss
    norm_params = {'normalizer': train_val_normalizer}

    # Training loop
    # Input data shape: (batch_size, 6) -> [m, zeta, k, t, x0, v0]
    # Target data shape: (batch_size, 3) -> [x(t), v(t), a(t)]
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

            optimizer.zero_grad()

            # Denormalize inputs for loss calculation
            inputs_real = train_val_normalizer.denormalize_inputs(inputs).clone()

            # Build inputs_combined based on which losses are enabled
            inputs_list = [inputs]
            N = inputs.size(0)

            # Generate t=0 samples if InitialCondition loss is enabled
            if loss_fn.has_loss("InitialCondition"):
                inputs_t0_real = inputs_real.clone()
                inputs_t0_real[:, 3] = 0.0  # Set real t=0
                inputs_t0 = torch.FloatTensor(train_val_normalizer.normalize_inputs(inputs_t0_real.cpu().numpy())).to(device)
                inputs_list.append(inputs_t0)

            # Generate perturbed time samples if Consistency loss with finite type is enabled
            if loss_fn.has_loss("Consistency") and loss_config["Consistency"]["type"] == "finite":
                t_threshold = loss_config["Consistency"]["t_threshold"]

                inputs_t_minus_minus_real = inputs_real.clone()
                inputs_t_minus_minus_real[:, 3] = inputs_real[:, 3] - 2 * t_threshold
                inputs_t_minus_minus = torch.FloatTensor(train_val_normalizer.normalize_inputs(inputs_t_minus_minus_real.cpu().numpy())).to(device)

                inputs_t_minus_real = inputs_real.clone()
                inputs_t_minus_real[:, 3] = inputs_real[:, 3] - t_threshold
                inputs_t_minus = torch.FloatTensor(train_val_normalizer.normalize_inputs(inputs_t_minus_real.cpu().numpy())).to(device)

                inputs_t_plus_real = inputs_real.clone()
                inputs_t_plus_real[:, 3] = inputs_real[:, 3] + t_threshold
                inputs_t_plus = torch.FloatTensor(train_val_normalizer.normalize_inputs(inputs_t_plus_real.cpu().numpy())).to(device)

                inputs_t_plus_plus_real = inputs_real.clone()
                inputs_t_plus_plus_real[:, 3] = inputs_real[:, 3] + 2 * t_threshold
                inputs_t_plus_plus = torch.FloatTensor(train_val_normalizer.normalize_inputs(inputs_t_plus_plus_real.cpu().numpy())).to(device)

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
                inputs_real_t0 = train_val_normalizer.denormalize_inputs(inputs_t0)
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
        # print the last output and last ground truth of the inputs and targets
        print("Last batch outputs v.s targets:", outputs[-1].detach().cpu().numpy(), targets[-1].detach().cpu().numpy())

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

                # Denormalize inputs for loss calculation
                inputs_real = train_val_normalizer.denormalize_inputs(inputs).clone()

                # Build inputs_combined based on which losses are enabled
                inputs_list = [inputs]
                N = inputs.size(0)

                # Generate t=0 samples if InitialCondition loss is enabled
                if loss_fn.has_loss("InitialCondition"):
                    inputs_t0_real = inputs_real.clone()
                    inputs_t0_real[:, 3] = 0.0  # Set real t=0
                    inputs_t0 = torch.FloatTensor(train_val_normalizer.normalize_inputs(inputs_t0_real.cpu().numpy())).to(device)
                    inputs_list.append(inputs_t0)

                # Generate perturbed time samples if Consistency loss with finite type is enabled
                if loss_fn.has_loss("Consistency") and loss_config["Consistency"]["type"] == "finite":
                    t_threshold = loss_config["Consistency"]["t_threshold"]

                    inputs_t_minus_minus_real = inputs_real.clone()
                    inputs_t_minus_minus_real[:, 3] = inputs_real[:, 3] - 2 * t_threshold
                    inputs_t_minus_minus = torch.FloatTensor(train_val_normalizer.normalize_inputs(inputs_t_minus_minus_real.cpu().numpy())).to(device)

                    inputs_t_minus_real = inputs_real.clone()
                    inputs_t_minus_real[:, 3] = inputs_real[:, 3] - t_threshold
                    inputs_t_minus = torch.FloatTensor(train_val_normalizer.normalize_inputs(inputs_t_minus_real.cpu().numpy())).to(device)

                    inputs_t_plus_real = inputs_real.clone()
                    inputs_t_plus_real[:, 3] = inputs_real[:, 3] + t_threshold
                    inputs_t_plus = torch.FloatTensor(train_val_normalizer.normalize_inputs(inputs_t_plus_real.cpu().numpy())).to(device)

                    inputs_t_plus_plus_real = inputs_real.clone()
                    inputs_t_plus_plus_real[:, 3] = inputs_real[:, 3] + 2 * t_threshold
                    inputs_t_plus_plus = torch.FloatTensor(train_val_normalizer.normalize_inputs(inputs_t_plus_plus_real.cpu().numpy())).to(device)

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
                    inputs_real_t0 = train_val_normalizer.denormalize_inputs(inputs_t0)
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

        lr_scheduler.step()

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

            # Denormalize inputs for loss calculation
            inputs_real = test_normalizer.denormalize_inputs(inputs).clone()

            # Build inputs_combined based on which losses are enabled
            inputs_list = [inputs]
            N = inputs.size(0)

            # Generate t=0 samples if InitialCondition loss is enabled
            if loss_fn.has_loss("InitialCondition"):
                inputs_t0_real = inputs_real.clone()
                inputs_t0_real[:, 3] = 0.0  # Set real t=0
                inputs_t0 = torch.FloatTensor(test_normalizer.normalize_inputs(inputs_t0_real.cpu().numpy())).to(device)
                inputs_list.append(inputs_t0)

            # Generate perturbed time samples if Consistency loss with finite type is enabled
            if loss_fn.has_loss("Consistency") and loss_config["Consistency"]["type"] == "finite":
                t_threshold = loss_config["Consistency"]["t_threshold"]

                inputs_t_minus_minus_real = inputs_real.clone()
                inputs_t_minus_minus_real[:, 3] = inputs_real[:, 3] - 2 * t_threshold
                inputs_t_minus_minus = torch.FloatTensor(test_normalizer.normalize_inputs(inputs_t_minus_minus_real.cpu().numpy())).to(device)

                inputs_t_minus_real = inputs_real.clone()
                inputs_t_minus_real[:, 3] = inputs_real[:, 3] - t_threshold
                inputs_t_minus = torch.FloatTensor(test_normalizer.normalize_inputs(inputs_t_minus_real.cpu().numpy())).to(device)

                inputs_t_plus_real = inputs_real.clone()
                inputs_t_plus_real[:, 3] = inputs_real[:, 3] + t_threshold
                inputs_t_plus = torch.FloatTensor(test_normalizer.normalize_inputs(inputs_t_plus_real.cpu().numpy())).to(device)

                inputs_t_plus_plus_real = inputs_real.clone()
                inputs_t_plus_plus_real[:, 3] = inputs_real[:, 3] + 2 * t_threshold
                inputs_t_plus_plus = torch.FloatTensor(test_normalizer.normalize_inputs(inputs_t_plus_plus_real.cpu().numpy())).to(device)

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
                inputs_real_t0 = test_normalizer.denormalize_inputs(inputs_t0)
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
        normalizer=train_val_normalizer,
        device=device,
        data_sampling_step=100,
        figure_folder='./prediction_figures_new'
    )

if __name__ == "__main__":
    main()

