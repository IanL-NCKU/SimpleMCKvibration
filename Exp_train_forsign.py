"""
Training script for ExponentialSignNN - Sign Classification Network

This network learns to predict signs of [x_t, v_t, a_t] given:
- Input: [a, b, t, (1+0.05*r)*abs(x_t), (1+0.05*r)*abs(v_t), (1+0.05*r)*abs(a_t)]
- Output: sign predictions applied to input magnitudes
- Loss: SignMSE only (no physics constraints)

Note: This file only uses ExponentialSignNN and SignMSELoss.
      ExponentialPINN, ExponentialResidualLoss, and ConsistencyLoss classes
      are NOT used and have been removed from Exp_modelandloss_forsign.py
"""

from Exp_dataset import *
from Exp_modelandloss_forsign import *
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
            # Move data to device
            inputs, targets = inputs.to(device), targets.to(device)

            # Generate random noise for this batch: r ~ Uniform[-1, 1]
            N = inputs.size(0)
            r = torch.rand(N, 3, device=device) * 2 - 1  # Shape: (batch_size, 3)

            # Compute noisy magnitudes: (1 + 0.05*r) * abs(targets)
            noisy_magnitudes = (1 + 0.05 * r) * torch.abs(targets)  # Shape: (batch_size, 3)

            # Modify inputs: concatenate [a, b, t] with noisy magnitudes
            inputs_modified = torch.cat([inputs, noisy_magnitudes], dim=1)  # Shape: (batch_size, 6)

            # Forward pass
            outputs = model(inputs_modified)
            all_predictions.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

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

    model_save_path = 'exp_model_signnn_elu_classres_64_32_16.pt'
    results_figure_folder = './exp_results_signnn_elu_classres_64_32_16'

    # Create the Exponential Sign NN model
    # model = ExponentialSignNN_ver2(hidden_dims=[64, 32, 16],
    #                                activation='relu').to(device)
    model = ExponentialSignNN_ver4(hidden_dims=[64, 32, 16],
                                   activation='elu').to(device)

    # Configure losses - only SignBCE
    loss_config = {
        "SignBCE": {"weight": 1.0}
    }

    # Safety check: Ensure only SignMSE or SignBCE is configured
    allowed_losses = {"SignMSE", "SignBCE"}
    configured_losses = set(loss_config.keys())
    if not configured_losses.issubset(allowed_losses):
        raise ValueError(f"Only SignMSE or SignBCE loss is supported. Found: {configured_losses}")

    loss_fn = ExponentialPINNLoss(model, loss_config)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=np.max([epochs//20,1]), eta_min=1e-9)

    # Training loop for Sign Classification Network
    # Input data shape: (batch_size, 6) -> [a, b, t, noisy_mag_x, noisy_mag_v, noisy_mag_a]
    # Target data shape: (batch_size, 3) -> [(1+0.05*r)*x_t, (1+0.05*r)*v_t, (1+0.05*r)*a_t]
    best_combined_loss = float('inf')
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        model.train()
        train_loss = 0.0
        train_loss_components = {}
        train_sign_accuracy = 0.0
        train_samples = 0

        # Training progress bar
        train_pbar = tqdm(train_loader, desc=f"Training", leave=False)
        for inputs, targets in train_pbar:
            # Move data to device
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()

            # Generate random noise for this batch: r ~ Uniform[-1, 1]
            N = inputs.size(0)
            r = torch.rand(N, 3, device=device) * 2 - 1  # Shape: (batch_size, 3)

            # Compute noisy magnitudes: (1 + 0.05*r) * abs(targets)
            noisy_magnitudes = (1 + 0.05 * r) *torch.abs(targets)  # Shape: (batch_size, 3)

            # Modify inputs: concatenate [a, b, t] with noisy magnitudes
            inputs_modified = torch.cat([inputs, noisy_magnitudes], dim=1)  # Shape: (batch_size, 6)

            # Compute target values: (1 + 0.05*r) * targets (signed values)
            targets_modified = (1 + 0.05 * r) * targets  # Shape: (batch_size, 3)

            # Forward pass
            # outputs = model(inputs_modified)
            outputs = model(inputs)

            # Prepare loss arguments
            loss_args = {}
            if loss_fn.has_loss("SignMSE"):
                loss_args["SignMSE"] = (outputs, targets_modified)
            if loss_fn.has_loss("SignBCE"):
                # Get sigmoid probabilities stored by model
                sigmoid_probs = model.last_sign_probs
                loss_args["SignBCE"] = (sigmoid_probs, targets_modified)

            # Compute loss
            loss, loss_dict = loss_fn(loss_args)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)

            # Compute sign accuracy
            with torch.no_grad():
                pred_signs = torch.sign(outputs)
                target_signs = torch.sign(targets_modified)
                sign_accuracy = (pred_signs == target_signs).float().mean()
                train_sign_accuracy += sign_accuracy.item() * inputs.size(0)
                train_samples += inputs.size(0)

            # Accumulate loss components
            for key, value in loss_dict.items():
                if key not in train_loss_components:
                    train_loss_components[key] = 0.0
                train_loss_components[key] += value * inputs.size(0)

            # Update progress bar with current loss and sign accuracy
            train_pbar.set_postfix({'loss': f'{loss.item():.4e}', 'sign_acc': f'{sign_accuracy.item():.4f}'})

        # Print the last output and last ground truth of the inputs and targets
        print("Last batch - input_mags:", inputs_modified[-1, 3:].detach().cpu().numpy(), "outputs:", outputs[-1].detach().cpu().numpy(), "targets:", targets_modified[-1].detach().cpu().numpy())

        train_loss /= len(train_loader.dataset)
        train_sign_accuracy /= train_samples

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
        val_sign_accuracy = 0.0
        val_samples = 0

        with torch.no_grad():
            # Validation progress bar
            val_pbar = tqdm(val_loader, desc=f"Validation", leave=False)
            for inputs, targets in val_pbar:
                # Move data to device
                inputs, targets = inputs.to(device), targets.to(device)

                # Generate random noise for this batch: r ~ Uniform[-1, 1]
                N = inputs.size(0)
                r = torch.rand(N, 3, device=device) * 2 - 1  # Shape: (batch_size, 3)

                # Compute noisy magnitudes: (1 + 0.05*r) * abs(targets)
                noisy_magnitudes = (1 + 0.05 * r) * torch.abs(targets)  # Shape: (batch_size, 3)

                # Modify inputs: concatenate [a, b, t] with noisy magnitudes
                inputs_modified = torch.cat([inputs, noisy_magnitudes], dim=1)  # Shape: (batch_size, 6)

                # Compute target values: (1 + 0.05*r) * targets (signed values)
                targets_modified = (1 + 0.05 * r) * targets  # Shape: (batch_size, 3)

                # Forward pass
                # outputs = model(inputs_modified)
                outputs = model(inputs)

                # Prepare loss arguments
                loss_args = {}
                if loss_fn.has_loss("SignMSE"):
                    loss_args["SignMSE"] = (outputs, targets_modified)
                if loss_fn.has_loss("SignBCE"):
                    # Get sigmoid probabilities stored by model
                    sigmoid_probs = model.last_sign_probs
                    loss_args["SignBCE"] = (sigmoid_probs, targets_modified)

                # Compute loss
                loss, loss_dict = loss_fn(loss_args)
                val_loss += loss.item() * inputs.size(0)

                # Compute sign accuracy
                pred_signs = torch.sign(outputs)
                target_signs = torch.sign(targets_modified)
                sign_accuracy = (pred_signs == target_signs).float().mean()
                val_sign_accuracy += sign_accuracy.item() * inputs.size(0)
                val_samples += inputs.size(0)

                # Accumulate loss components
                for key, value in loss_dict.items():
                    if key not in val_loss_components:
                        val_loss_components[key] = 0.0
                    val_loss_components[key] += value * inputs.size(0)

                # Update progress bar with current loss and sign accuracy
                val_pbar.set_postfix({'loss': f'{loss.item():.4e}', 'sign_acc': f'{sign_accuracy.item():.4f}'})

        val_loss /= len(val_loader.dataset)

        # Calculate average loss components
        for key in val_loss_components:
            val_loss_components[key] /= len(val_loader.dataset)

        lr_scheduler.step()

        # Print epoch summary
        print(f"Epoch [{epoch+1}/{epochs}] - Model: {os.path.basename(model_save_path)}")
        print(f"  Train: Loss={train_loss:.4e}, Sign Acc={train_sign_accuracy:.4f}")
        print(f"  Val  : Loss={val_loss:.4e}, Sign Acc={val_sign_accuracy:.4f}")

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
    test_sign_accuracy = 0.0
    test_samples = 0

    with torch.no_grad():
        # Test progress bar
        test_pbar = tqdm(test_loader, desc="Testing", leave=True)
        for inputs, targets in test_pbar:
            # Move data to device
            inputs, targets = inputs.to(device), targets.to(device)

            # Generate random noise for this batch: r ~ Uniform[-1, 1]
            N = inputs.size(0)
            r = torch.rand(N, 3, device=device) * 2 - 1  # Shape: (batch_size, 3)

            # Compute noisy magnitudes: (1 + 0.05*r) * abs(targets)
            noisy_magnitudes = (1 + 0.05 * r) * torch.abs(targets)  # Shape: (batch_size, 3)

            # Modify inputs: concatenate [a, b, t] with noisy magnitudes
            inputs_modified = torch.cat([inputs, noisy_magnitudes], dim=1)  # Shape: (batch_size, 6)

            # Compute target values: (1 + 0.05*r) * targets (signed values)
            targets_modified = (1 + 0.05 * r) * targets  # Shape: (batch_size, 3)

            # Forward pass
            # outputs = model(inputs_modified)
            outputs = model(inputs)

            # Prepare loss arguments
            loss_args = {}
            if loss_fn.has_loss("SignMSE"):
                loss_args["SignMSE"] = (outputs, targets_modified)
            if loss_fn.has_loss("SignBCE"):
                # Get sigmoid probabilities stored by model
                sigmoid_probs = model.last_sign_probs
                loss_args["SignBCE"] = (sigmoid_probs, targets_modified)

            # Compute loss
            loss, loss_dict = loss_fn(loss_args)
            test_loss += loss.item() * inputs.size(0)

            # Compute sign accuracy
            pred_signs = torch.sign(outputs)
            target_signs = torch.sign(targets_modified)
            sign_accuracy = (pred_signs == target_signs).float().mean()
            test_sign_accuracy += sign_accuracy.item() * inputs.size(0)
            test_samples += inputs.size(0)

            # Update progress bar with current loss and sign accuracy
            test_pbar.set_postfix({'loss': f'{loss.item():.4e}', 'sign_acc': f'{sign_accuracy.item():.4f}'})

    test_loss /= len(test_loader.dataset)
    test_sign_accuracy /= test_samples
    print(f"\nTest Loss: {test_loss:.4e}, Test Sign Accuracy: {test_sign_accuracy:.4f}")

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
