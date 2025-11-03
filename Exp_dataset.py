import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


class Exponential_DataNormalizer:
    """Normalization for exponential data: x(t) = b * exp(a*t)

    Features:
    - 'a': exponential rate [-10, 10] - linear uniform
    - 'b': coefficient [-1000, 1000] - linear uniform
    - 't': time [1e-3, 10] - mixed log-uniform/linear uniform
    """

    def __init__(self):
        self.log_features = ['t']  # Log-uniform sampled (partially)
        self.linear_features = ['a', 'b']  # Linear uniform sampled

        self.log_mean = {}
        self.log_std = {}
        self.linear_mean = {}
        self.linear_std = {}

    def fit(self, data_dict):
        """
        Fit normalization parameters
        data_dict: {'a': array, 'b': array, 't': array}
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

        # Log-space normalization for time
        for feat in self.log_features:
            values = data_dict[feat].copy()

            # Special handling for time to avoid log10(0)
            if feat == 't':
                # Clamp very small values (including 0) to small positive value
                values = np.where(values < t_zero_threshold, t_zero_threshold, values)

            log_values = np.log10(values)
            normalized[feat] = (log_values - self.log_mean[feat]) / self.log_std[feat]

        # Standard normalization for a and b
        for feat in self.linear_features:
            normalized[feat] = (data_dict[feat] - self.linear_mean[feat]) / self.linear_std[feat]

        return normalized

    def normalize_inputs(self, X):
        """
        Normalize input array [a, b, t]

        Args:
            X: numpy array of shape (N, 3) with columns [a, b, t]

        Returns:
            Normalized array of same shape
        """
        data_dict = {
            'a': X[:, 0],
            'b': X[:, 1],
            't': X[:, 2]
        }
        normalized = self.transform(data_dict)
        # Stack back into array
        return np.stack([
            normalized['a'],
            normalized['b'],
            normalized['t']
        ], axis=1)

    def inverse_transform(self, normalized_dict, t_zero_threshold=1e-9):
        """Denormalize data

        Args:
            normalized_dict: Dictionary with normalized feature arrays
            t_zero_threshold: If denormalized 't' would be below this, set to exactly 0.0
        """
        original = {}

        # Inverse log-space normalization for time
        for feat in self.log_features:
            # Special handling for time: check threshold in normalized space
            if feat == 't':
                # Convert threshold from real space to normalized space
                threshold_log = np.log10(t_zero_threshold)
                threshold_norm = (threshold_log - self.log_mean[feat]) / self.log_std[feat]

                # Add safety margin
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

        # Inverse standard normalization for a and b
        for feat in self.linear_features:
            original[feat] = normalized_dict[feat] * self.linear_std[feat] + self.linear_mean[feat]

        return original

    def denormalize_inputs(self, X_norm):
        """
        Denormalize input array [a, b, t]

        Args:
            X_norm: tensor or numpy array of shape (N, 3) with normalized [a, b, t]

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
            'a': X_norm[:, 0],
            'b': X_norm[:, 1],
            't': X_norm[:, 2]
        }

        # Denormalize
        original = self.inverse_transform(normalized_dict)

        # Stack back into array
        X_denorm = np.stack([
            original['a'],
            original['b'],
            original['t']
        ], axis=1)

        # Convert back to tensor if input was tensor
        if is_tensor:
            X_denorm = torch.FloatTensor(X_denorm).to(device)

        return X_denorm


class Exponential_OutputNormalizer:
    """Normalization for exponential output data: [x_t, v_t, a_t]

    Outputs: x_t, v_t (velocity), a_t (acceleration) can span many orders of magnitude.

    Note: Assumes all output values are positive (like absolute magnitudes).
    """

    def __init__(self, use_log_normalization=True):
        """
        Args:
            use_log_normalization: If True, use log-space normalization.
                                  If False, use standard normalization.
        """
        self.use_log_normalization = use_log_normalization

        if self.use_log_normalization:
            self.log_features = ['x', 'v', 'a']  # All outputs use log normalization
            self.log_mean = {}
            self.log_std = {}
        else:
            self.linear_features = ['x', 'v', 'a']
            self.linear_mean = {}
            self.linear_std = {}

    def fit(self, data_dict):
        """
        Fit normalization parameters

        Args:
            data_dict: Dictionary {'x': array, 'v': array, 'a': array}
        """
        if self.use_log_normalization:
            # For log-space normalization
            for feat in self.log_features:
                values = data_dict[feat]
                # Assume all values are positive, add epsilon to avoid log(0)
                log_values = np.log10(np.abs(values) + 1e-10)
                self.log_mean[feat] = np.mean(log_values)
                self.log_std[feat] = np.std(log_values)
        else:
            # For linear normalization
            for feat in self.linear_features:
                self.linear_mean[feat] = np.mean(data_dict[feat])
                self.linear_std[feat] = np.std(data_dict[feat])

    def transform(self, data_dict, eps=1e-10):
        """
        Normalize data

        Args:
            data_dict: Dictionary with feature arrays

        Returns:
            Dictionary with normalized features
        """
        normalized = {}

        if self.use_log_normalization:
            for feat in self.log_features:
                values = data_dict[feat]
                # Assume all values are positive
                log_values = np.log10(np.abs(values) + eps)
                normalized[feat] = (log_values - self.log_mean[feat]) / self.log_std[feat]
        else:
            for feat in self.linear_features:
                normalized[feat] = (data_dict[feat] - self.linear_mean[feat]) / self.linear_std[feat]

        return normalized

    def inverse_transform(self, normalized_dict, eps=1e-10):
        """
        Denormalize data

        Args:
            normalized_dict: Dictionary with normalized feature arrays

        Returns:
            Dictionary with original features
        """
        original = {}

        if self.use_log_normalization:
            for feat in self.log_features:
                log_values = normalized_dict[feat] * self.log_std[feat] + self.log_mean[feat]
                original[feat] = 10 ** log_values
        else:
            for feat in self.linear_features:
                original[feat] = normalized_dict[feat] * self.linear_std[feat] + self.linear_mean[feat]

        return original

    def normalize_outputs(self, Y):
        """
        Normalize output array [x_t, v_t, a_t]

        Args:
            Y: numpy array or tensor of shape (N, 3) with columns [x_t, v_t, a_t]

        Returns:
            Normalized array of same shape and type
        """
        is_tensor = torch.is_tensor(Y)
        if is_tensor:
            device = Y.device
            Y = Y.detach().cpu().numpy()

        # Convert array to dictionary
        data_dict = {
            'x': Y[:, 0],
            'v': Y[:, 1],
            'a': Y[:, 2]
        }

        # Normalize using transform
        normalized_dict = self.transform(data_dict)

        # Convert back to array
        Y_norm = np.stack([
            normalized_dict['x'],
            normalized_dict['v'],
            normalized_dict['a']
        ], axis=1)

        print(f"Min/Max of Y_norm 'x': {np.min(Y_norm[:, 0])}, {np.max(Y_norm[:, 0])}")
        print(f"Min/Max of Y_norm 'v': {np.min(Y_norm[:, 1])}, {np.max(Y_norm[:, 1])}")
        print(f"Min/Max of Y_norm 'a': {np.min(Y_norm[:, 2])}, {np.max(Y_norm[:, 2])}")

        if is_tensor:
            Y_norm = torch.FloatTensor(Y_norm).to(device)

        return Y_norm

    def denormalize_outputs(self, Y_norm):
        """
        Denormalize output array [x_t, v_t, a_t]

        Args:
            Y_norm: tensor or numpy array of shape (N, 3) with normalized [x_t, v_t, a_t]

        Returns:
            Denormalized array of same shape and type
        """
        is_tensor = torch.is_tensor(Y_norm)
        if is_tensor:
            device = Y_norm.device
            Y_norm = Y_norm.detach().cpu().numpy()

        # Convert array to dictionary
        normalized_dict = {
            'x': Y_norm[:, 0],
            'v': Y_norm[:, 1],
            'a': Y_norm[:, 2]
        }

        # Denormalize using inverse_transform
        original_dict = self.inverse_transform(normalized_dict)

        # Convert back to array
        Y = np.stack([
            original_dict['x'],
            original_dict['v'],
            original_dict['a']
        ], axis=1)

        if is_tensor:
            Y = torch.FloatTensor(Y).to(device)

        return Y


class ExponentialDataset(Dataset):
    """PyTorch Dataset for exponential data"""

    def __init__(self, inputs, outputs):
        """
        Args:
            inputs: numpy array of shape (N, 3) - [a, b, t]
            outputs: numpy array of shape (N, 3) - [x_t, v_t, a_t]
        """
        self.inputs = torch.FloatTensor(inputs)
        self.outputs = torch.FloatTensor(outputs)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.outputs[idx]


def load_exponential_data(filepath='exponential_trainval_data.npz', batch_size=32, shuffle_train=True, normalize=True):
    """Loads and prepares exponential data from an .npz file.

    This function splits the data into training, validation, and test sets,
    optionally normalizes both inputs and outputs, and creates PyTorch DataLoaders.

    The input .npz file is expected to contain a single numpy array where each
    row corresponds to [a, b, t, x_t, v_t, a_t].

    Args:
        filepath (str): Path to the .npz data file.
        batch_size (int): Batch size for the DataLoaders.
        shuffle_train (bool): Whether to shuffle the training data.
        normalize (bool): If True, normalizes both input and output features.

    Returns:
        tuple: A tuple containing:
            - train_loader (DataLoader): DataLoader for the training set.
            - val_loader (DataLoader): DataLoader for the validation set.
            - test_loader (DataLoader): DataLoader for the test set.
            - inputs_normalizer (Exponential_DataNormalizer or None): The fitted
              input normalizer instance if normalize=True, otherwise None.
            - outputs_normalizer (Exponential_OutputNormalizer or None): The fitted
              output normalizer instance if normalize=True, otherwise None.
    """
    # Load data
    data = np.load(filepath)

    # Extract the array (npz files can contain multiple arrays)
    if isinstance(data, np.lib.npyio.NpzFile):
        # Get the first array in the npz file
        array_name = list(data.keys())[0]
        data_array = data[array_name]
    else:
        data_array = data

    input_data = data_array[:, :3]   # Inputs: a, b, t
    output_data = data_array[:, 3:]  # Outputs: x_t, v_t, a_t

    # Split into train (80%), val (16%), test (4%)
    X_train, X_temp, y_train, y_temp = train_test_split(
        input_data, output_data, test_size=0.2)

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.2)

    # Normalize inputs and outputs if requested
    inputs_normalizer = None
    outputs_normalizer = None

    if normalize:
        # Create data dictionaries for training set
        train_input_dict = {
            'a': X_train[:, 0],
            'b': X_train[:, 1],
            't': X_train[:, 2]
        }

        train_output_dict = {
            'x': y_train[:, 0],
            'v': y_train[:, 1],
            'a': y_train[:, 2]
        }

        # Fit normalizers on training data only
        inputs_normalizer = Exponential_DataNormalizer()
        inputs_normalizer.fit(train_input_dict)

        outputs_normalizer = Exponential_OutputNormalizer(use_log_normalization=True)
        outputs_normalizer.fit(train_output_dict)

        # Normalize all splits
        X_train = inputs_normalizer.normalize_inputs(X_train)
        X_val = inputs_normalizer.normalize_inputs(X_val)
        X_test = inputs_normalizer.normalize_inputs(X_test)

        y_train = outputs_normalizer.normalize_outputs(y_train)
        y_val = outputs_normalizer.normalize_outputs(y_val)
        y_test = outputs_normalizer.normalize_outputs(y_test)

    # Create PyTorch Datasets
    train_dataset = ExponentialDataset(X_train, y_train)
    val_dataset = ExponentialDataset(X_val, y_val)
    test_dataset = ExponentialDataset(X_test, y_test)

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

    return train_loader, val_loader, test_loader, inputs_normalizer, outputs_normalizer
