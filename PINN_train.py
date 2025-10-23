from PINN_dataset import *
from PINN_modelandloss import *
from datagtgenerator import *
import torch

def main():

    device_index = 1
    
    epochs = 500
    Train_Val_data_source = r'E:\Ian\PINNexample\train_val_vibration_data.npz'
    Test_data_source = r'E:\Ian\PINNexample\test_vibration_data.npz'
    # Load the dataset 
    train_loader, val_loader, _, normalizer = load_vibration_data(
        filepath= Train_Val_data_source,
        batch_size=64,
        normalize=True,
        shuffle_train=True
    )

    test_loader, _, _, normalizer = load_vibration_data(
        filepath= Test_data_source,
        batch_size=64,
        normalize=True,
        shuffle_train=False
    )

    print(f"Data loaders created:")

    # Setup device
    device = torch.device(f'cuda:{device_index}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create the PINN model
    model = VibrationPINN().to(device)
    loss_fn = PINNLoss(mse_weight=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs//20)

    # Training loop
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            # Move data to device
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()

            # Generate t=0 samples for initial condition loss
            inputs_t0_real = normalizer.denormalize_inputs(inputs).clone()
            inputs_t0_real[:, 3] = 0.0  # Set real t=0
            inputs_t0 = torch.FloatTensor(normalizer.normalize_inputs(inputs_t0_real.cpu().numpy())).to(device)

            # Stack original and t=0 inputs
            inputs_combined = torch.cat([inputs, inputs_t0], dim=0)
            targets_combined = torch.cat([targets, targets], dim=0)  # t=0 targets same as original

            # Forward pass
            outputs_combined = model(inputs_combined)

            # Denormalize inputs for loss calculation
            invnorm_inputs_combined = normalizer.denormalize_inputs(inputs_combined)

            loss, loss_dict = loss_fn(outputs_combined, targets_combined, invnorm_inputs_combined)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)

        train_loss /= len(train_loader.dataset)

        # Validation loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                # Move data to device
                inputs, targets = inputs.to(device), targets.to(device)

                # Generate t=0 samples for initial condition loss
                inputs_t0_real = normalizer.denormalize_inputs(inputs).clone()
                inputs_t0_real[:, 3] = 0.0  # Set real t=0
                inputs_t0 = torch.FloatTensor(normalizer.normalize_inputs(inputs_t0_real.cpu().numpy())).to(device)

                # Stack original and t=0 inputs
                inputs_combined = torch.cat([inputs, inputs_t0], dim=0)
                targets_combined = torch.cat([targets, targets], dim=0)

                # Forward pass
                outputs_combined = model(inputs_combined)

                # Denormalize inputs for loss calculation
                invnorm_inputs_combined = normalizer.denormalize_inputs(inputs_combined)

                loss, loss_dict = loss_fn(outputs_combined, targets_combined, invnorm_inputs_combined)
                val_loss += loss.item() * inputs.size(0)

        val_loss /= len(val_loader.dataset)

        lr_scheduler.step()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

    # Testing loop
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for inputs, targets in test_loader:
            # Move data to device
            inputs, targets = inputs.to(device), targets.to(device)

            # Generate t=0 samples for initial condition loss
            inputs_t0_real = normalizer.denormalize_inputs(inputs).clone()
            inputs_t0_real[:, 3] = 0.0  # Set real t=0
            inputs_t0 = torch.FloatTensor(normalizer.normalize_inputs(inputs_t0_real.cpu().numpy())).to(device)

            # Stack original and t=0 inputs
            inputs_combined = torch.cat([inputs, inputs_t0], dim=0)
            targets_combined = torch.cat([targets, targets], dim=0)

            # Forward pass
            outputs_combined = model(inputs_combined)

            # Denormalize inputs for loss calculation
            invnorm_inputs_combined = normalizer.denormalize_inputs(inputs_combined)

            loss, loss_dict = loss_fn(outputs_combined, targets_combined, invnorm_inputs_combined)
            test_loss += loss.item() * inputs.size(0)
    test_loss /= len(test_loader.dataset)
    print(f"Test Loss: {test_loss:.6f}")


if __name__ == "__main__":
    main()

