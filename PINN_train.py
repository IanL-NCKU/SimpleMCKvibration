from PINN_dataset import *
from PINN_modelandloss import *
from datagtgenerator import *
import torch
from tqdm import tqdm

def main():

    device_index = 0
    
    epochs = 10
    Train_Val_data_source = r'E:\Ian\PINNexample\train_val_vibration_data.npz'
    Test_data_source = r'E:\Ian\PINNexample\test_vibration_data.npz'
    # Load the dataset 
    train_loader, val_loader, _, train_val_normalizer = load_vibration_data(
        filepath= Train_Val_data_source,
        batch_size=256,
        normalize=True,
        shuffle_train=True
    )

    test_loader, _, _, test_normalizer = load_vibration_data(
        filepath= Test_data_source,
        batch_size=256,
        normalize=True,
        shuffle_train=False
    )

    print(f"Data loaders created:")

    # Setup device
    device = torch.device(f'cuda:{device_index}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create the PINN model with log-space output
    model = VibrationPINN(hidden_dims=[128, 512, 2048, 512, 128], use_log_output=True, use_finetune=False, finetune_hidden_dims=[128, 128]).to(device)

    # Configure losses using dict-based interface
    loss_config = {
        "MSE": {"weight": 0.2},
        "Residual": {"weight": 0.8, "use_relative": True},
        "InitialCondition": {"weight": 0, "t_threshold": 1e-6},
        "Consistency": None#{"weight": 0, "t_threshold": 1e-6}
    }

    loss_fn = PINNLoss_v2(model, loss_config)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Prepare norm_params for consistency loss
    norm_params = {'normalizer': train_val_normalizer}

    # Training loop
    # Input data shape: (batch_size, 6) -> [m, zeta, k, t, x0, v0]
    # Target data shape: (batch_size, 3) -> [x(t), v(t), a(t)]
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        model.train()
        train_loss = 0.0

        # Training progress bar
        train_pbar = tqdm(train_loader, desc=f"Training", leave=False)
        for inputs, targets in train_pbar:
            # Move data to device
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()

            # Generate t=0 samples for initial condition loss
            inputs_real = train_val_normalizer.denormalize_inputs(inputs).clone()
            inputs_t0_real = inputs_real.clone()
            inputs_t0_real[:, 3] = 0.0  # Set real t=0
            inputs_t0 = torch.FloatTensor(train_val_normalizer.normalize_inputs(inputs_t0_real.cpu().numpy())).to(device)

            # Stack original and t=0 inputs
            inputs_combined = torch.cat([inputs, inputs_t0], dim=0)

            # Forward pass
            outputs_combined = model(inputs_combined)
            # print(inputs_combined.size(), outputs_combined.size(), targets.size())
            # break

            # Split outputs into regular and t=0 samples
            outputs = outputs_combined[:inputs.size(0)]
            outputs_t0 = outputs_combined[inputs.size(0):]

            # Denormalize inputs for loss calculation
            inputs_real = train_val_normalizer.denormalize_inputs(inputs)
            inputs_real_t0 = train_val_normalizer.denormalize_inputs(inputs_t0)

            # Prepare loss arguments
            loss_args = {}
            if loss_fn.has_loss("MSE"):
                loss_args["MSE"] = (outputs, targets)
            if loss_fn.has_loss("Residual"):
                loss_args["Residual"] = (outputs, inputs_real)
            if loss_fn.has_loss("Consistency"):
                loss_args["Consistency"] = (inputs, inputs_real, norm_params)
            if loss_fn.has_loss("InitialCondition"):
                loss_args["InitialCondition"] = (outputs_t0, inputs_real_t0)

            # Compute loss
            loss, loss_dict = loss_fn(loss_args)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)

            # Update progress bar with current loss
            train_pbar.set_postfix({'loss': f'{loss.item():.4e}'})


        sample_m, sample_zeta, sample_k, sample_t, sample_x0, sample_v0 = inputs_real_t0.cpu().numpy()[0]
        print("Data X0, V0 at t=0:", sample_x0, sample_v0)
        sample_c = 2 * sample_zeta * np.sqrt(sample_m * sample_k)
        print("Analytical solution at t=0:", analytical_solution(sample_m, sample_c, sample_k, sample_x0, sample_v0, sample_t)[0:2])
        train_loss /= len(train_loader.dataset)

        # Validation loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            # Validation progress bar
            val_pbar = tqdm(val_loader, desc=f"Validation", leave=False)
            for inputs, targets in val_pbar:
                # Move data to device
                inputs, targets = inputs.to(device), targets.to(device)

                # Generate t=0 samples for initial condition loss
                inputs_t0_real = train_val_normalizer.denormalize_inputs(inputs).clone()
                inputs_t0_real[:, 3] = 0.0  # Set real t=0
                inputs_t0 = torch.FloatTensor(train_val_normalizer.normalize_inputs(inputs_t0_real.cpu().numpy())).to(device)

                # Stack original and t=0 inputs
                inputs_combined = torch.cat([inputs, inputs_t0], dim=0)

                # Forward pass
                outputs_combined = model(inputs_combined)

                # Split outputs into regular and t=0 samples
                outputs = outputs_combined[:inputs.size(0)]
                outputs_t0 = outputs_combined[inputs.size(0):]

                # Denormalize inputs for loss calculation
                inputs_real = train_val_normalizer.denormalize_inputs(inputs)
                inputs_real_t0 = train_val_normalizer.denormalize_inputs(inputs_t0)

                # Prepare loss arguments
                loss_args = {}
                if loss_fn.has_loss("MSE"):
                    loss_args["MSE"] = (outputs, targets)
                if loss_fn.has_loss("Residual"):
                    loss_args["Residual"] = (outputs, inputs_real)
                if loss_fn.has_loss("Consistency"):
                    loss_args["Consistency"] = (inputs, inputs_real, norm_params)
                if loss_fn.has_loss("InitialCondition"):
                    loss_args["InitialCondition"] = (outputs_t0, inputs_real_t0)

                # Compute loss
                loss, loss_dict = loss_fn(loss_args)
                val_loss += loss.item() * inputs.size(0)

                # Update progress bar with current loss
                val_pbar.set_postfix({'loss': f'{loss.item():.4e}'})

        val_loss /= len(val_loader.dataset)

        lr_scheduler.step()

        # Print epoch summary
        print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {train_loss:.4e}, Val Loss: {val_loss:.4e}")

    # Testing loop
    print("\nRunning test evaluation...")
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        # Test progress bar
        test_pbar = tqdm(test_loader, desc="Testing", leave=True)
        for inputs, targets in test_pbar:
            # Move data to device
            inputs, targets = inputs.to(device), targets.to(device)

            # Generate t=0 samples for initial condition loss
            inputs_t0_real = test_normalizer.denormalize_inputs(inputs).clone()
            inputs_t0_real[:, 3] = 0.0  # Set real t=0
            inputs_t0 = torch.FloatTensor(test_normalizer.normalize_inputs(inputs_t0_real.cpu().numpy())).to(device)

            # Stack original and t=0 inputs
            inputs_combined = torch.cat([inputs, inputs_t0], dim=0)

            # Forward pass
            outputs_combined = model(inputs_combined)

            # Split outputs into regular and t=0 samples
            outputs = outputs_combined[:inputs.size(0)]
            outputs_t0 = outputs_combined[inputs.size(0):]

            # Denormalize inputs for loss calculation
            inputs_real = test_normalizer.denormalize_inputs(inputs)
            inputs_real_t0 = test_normalizer.denormalize_inputs(inputs_t0)

            # Prepare norm_params for test (use test_normalizer)
            norm_params_test = {'normalizer': test_normalizer}

            # Prepare loss arguments
            loss_args = {}
            if loss_fn.has_loss("MSE"):
                loss_args["MSE"] = (outputs, targets)
            if loss_fn.has_loss("Residual"):
                loss_args["Residual"] = (outputs, inputs_real)
            if loss_fn.has_loss("Consistency"):
                loss_args["Consistency"] = (inputs, inputs_real, norm_params_test)
            if loss_fn.has_loss("InitialCondition"):
                loss_args["InitialCondition"] = (outputs_t0, inputs_real_t0)

            # Compute loss
            loss, loss_dict = loss_fn(loss_args)
            test_loss += loss.item() * inputs.size(0)

            # Update progress bar with current loss
            test_pbar.set_postfix({'loss': f'{loss.item():.4e}'})

    test_loss /= len(test_loader.dataset)
    print(f"\nTest Loss: {test_loss:.4e}")


if __name__ == "__main__":
    main()

