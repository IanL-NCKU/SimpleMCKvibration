import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


class Vibration_DataNormalizer:
    """Normalization for log-uniform and uniform variables in vibration data"""

    def __init__(self):
        self.log_features = ['m', 'zeta', 'k', 't']  # Log-uniform sampled
        self.linear_features = ['x0', 'v0']          # Uniform sampled

        self.log_mean = {}
        self.log_std = {}
        self.linear_mean = {}
        self.linear_std = {}

    def fit(self, data_dict):
        """
        Fit normalization parameters
        data_dict: {'m': array, 'zeta': array, 'k': array, ...}
        """
        # For log-uniform variables: normalize log values
        for feat in self.log_features:
            log_values = np.log10(data_dict[feat])  # Take log
            self.log_mean[feat] = np.mean(log_values)
            self.log_std[feat] = np.std(log_values)

        # For linear variables: standard normalization
        for feat in self.linear_features:
            self.linear_mean[feat] = np.mean(data_dict[feat])
            self.linear_std[feat] = np.std(data_dict[feat])

    def transform(self, data_dict, t_zero_threshold=1e-9):
        """Normalize data

        Args:
            data_dict: Dictionary with feature arrays
            t_zero_threshold: Values below this for 't' are clamped to avoid log10(0) = -inf
        """
        normalized = {}

        # Log-space normalization
        for feat in self.log_features:
            values = data_dict[feat].copy()

            # Special handling for time to avoid log10(0)
            if feat == 't':
                # Clamp very small values (including 0) to small positive value
                values = np.where(values < t_zero_threshold, t_zero_threshold, values)

            log_values = np.log10(values)
            normalized[feat] = (log_values - self.log_mean[feat]) / self.log_std[feat]

        # Standard normalization
        for feat in self.linear_features:
            normalized[feat] = (data_dict[feat] - self.linear_mean[feat]) / self.linear_std[feat]

        return normalized

    def normalize_inputs(self, X):
        """
        Normalize input array [m, zeta, k, t, x0, v0]

        Args:
            X: numpy array of shape (N, 6) with columns [m, zeta, k, t, x0, v0]

        Returns:
            Normalized array of same shape
        """
        data_dict = {
            'm': X[:, 0],
            'zeta': X[:, 1],
            'k': X[:, 2],
            't': X[:, 3],
            'x0': X[:, 4],
            'v0': X[:, 5]
        }
        normalized = self.transform(data_dict)
        # Stack back into array
        return np.stack([
            normalized['m'],
            normalized['zeta'],
            normalized['k'],
            normalized['t'],
            normalized['x0'],
            normalized['v0']
        ], axis=1)

    def inverse_transform(self, normalized_dict, t_zero_threshold=1e-9):
        """Denormalize data

        Args:
            normalized_dict: Dictionary with normalized feature arrays
            t_zero_threshold: If denormalized 't' would be below this, set to exactly 0.0
        """
        original = {}

        # Inverse log-space normalization
        for feat in self.log_features:
            # Special handling for time: check threshold in normalized space
            if feat == 't':
                # Convert threshold from real space to normalized space
                # threshold_real = 1e-9  →  log10(1e-9) = -9  →  normalize
                threshold_log = np.log10(t_zero_threshold)
                threshold_norm = (threshold_log - self.log_mean[feat]) / self.log_std[feat]

                # Add safety margin to ensure clamped values are always caught
                # Values clamped to 1e-10 during normalization should definitely be below this
                threshold_norm_with_margin = threshold_norm + 0.5

                # Map values below threshold directly to 0, compute rest normally
                normalized_values = normalized_dict[feat]
                values = np.where(
                    normalized_values < threshold_norm_with_margin,
                    0.0,  # Map to exactly 0
                    10 ** (normalized_values * self.log_std[feat] + self.log_mean[feat])  # Normal denorm
                )
            else:
                log_values = normalized_dict[feat] * self.log_std[feat] + self.log_mean[feat]
                values = 10 ** log_values

            original[feat] = values

        # Inverse standard normalization
        for feat in self.linear_features:
            original[feat] = normalized_dict[feat] * self.linear_std[feat] + self.linear_mean[feat]

        return original

    def denormalize_inputs(self, X_norm):
        """
        Denormalize input array [m, zeta, k, t, x0, v0]

        Args:
            X_norm: tensor or numpy array of shape (N, 6) with normalized [m, zeta, k, t, x0, v0]

        Returns:
            Denormalized array of same shape and type (tensor or numpy)
        """
        # Check if input is tensor
        is_tensor = torch.is_tensor(X_norm)
        if is_tensor:
            device = X_norm.device
            X_norm = X_norm.detach().cpu().numpy()

        # Create dictionary from array
        normalized_dict = {
            'm': X_norm[:, 0],
            'zeta': X_norm[:, 1],
            'k': X_norm[:, 2],
            't': X_norm[:, 3],
            'x0': X_norm[:, 4],
            'v0': X_norm[:, 5]
        }

        # Denormalize
        original = self.inverse_transform(normalized_dict)

        # Stack back into array
        X_denorm = np.stack([
            original['m'],
            original['zeta'],
            original['k'],
            original['t'],
            original['x0'],
            original['v0']
        ], axis=1)

        # Convert back to tensor if input was tensor
        if is_tensor:
            X_denorm = torch.FloatTensor(X_denorm).to(device)

        return X_denorm


class VibrationDataset(Dataset):
    """PyTorch Dataset for vibration data"""

    def __init__(self, inputs, outputs):
        """
        Args:
            inputs: numpy array of shape (N, 6) - [m, zeta, k, t, x0, v0]
            outputs: numpy array of shape (N, 3) - [x(t), v(t), a(t)]
        """
        self.inputs = torch.FloatTensor(inputs)
        self.outputs = torch.FloatTensor(outputs)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.outputs[idx]


def load_vibration_data(filepath='vibration_data_normalized.npz', batch_size=32, shuffle_train=True, normalize=True):
    """Loads and prepares vibration data from an .npz file.

    This function splits the data into training, validation, and test sets,
    optionally normalizes the inputs, and creates PyTorch DataLoaders.

    The input .npz file is expected to contain a single numpy array where each
    row corresponds to [m, zeta, k, t, x0, v0, x(t), v(t), a(t)].

    Args:
        filepath (str): Path to the .npz data file.
        batch_size (int): Batch size for the DataLoaders.
        shuffle_train (bool): Whether to shuffle the training data.
        normalize (bool): If True, normalizes the input features.

    Returns:
        tuple: A tuple containing:
            - train_loader (DataLoader): DataLoader for the training set.
            - val_loader (DataLoader): DataLoader for the validation set.
            - test_loader (DataLoader): DataLoader for the test set.
            - normalizer (Vibration_DataNormalizer or None): The fitted
              normalizer instance if normalize=True, otherwise None.
    """
    # Load data
    data = np.load(filepath)

    # Extract the array (npz files can contain multiple arrays)
    # Assuming the array key is 'arr_0' or get the first array
    if isinstance(data, np.lib.npyio.NpzFile):
        # Get the first array in the npz file
        array_name = list(data.keys())[0]
        data_array = data[array_name]
    else:
        data_array = data

    input_data = data_array[:, :6]   # Inputs: m, zeta, k, t, x0, v0
    output_data = data_array[:, 6:]  # Outputs: x(t), v(t), a(t)

    # Split into train (70%), val (15%), test (15%)
    X_train, X_temp, y_train, y_temp = train_test_split(
        input_data, output_data, test_size=0.2, random_state=20)

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=20)

    # Normalize inputs if requested
    normalizer = None
    if normalize:
        # Create data dictionary for training set
        train_dict = {
            'm': X_train[:, 0],
            'zeta': X_train[:, 1],
            'k': X_train[:, 2],
            't': X_train[:, 3],
            'x0': X_train[:, 4],
            'v0': X_train[:, 5]
        }

        # Fit normalizer on training data only
        normalizer = Vibration_DataNormalizer()
        normalizer.fit(train_dict)

        # Transform all splits using the normalizer's method
        X_train = normalizer.normalize_inputs(X_train)
        X_val = normalizer.normalize_inputs(X_val)
        X_test = normalizer.normalize_inputs(X_test)

    # Create PyTorch Datasets
    train_dataset = VibrationDataset(X_train, y_train)
    val_dataset = VibrationDataset(X_val, y_val)
    test_dataset = VibrationDataset(X_test, y_test)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=0  # Set to 0 for Windows compatibility
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    return train_loader, val_loader, test_loader, normalizer

