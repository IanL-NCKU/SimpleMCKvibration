import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from expdatagenerator import analytical_solution_exp
import matplotlib.pyplot as plt
import os


class Exponential_DataNormalizer:
    """Normalization for exponential data: x(t) = b * exp(a*t)

    Features:
    - 'a': exponential rate [-10, 10] - linear uniform
    - 'b': coefficient [-1000, 1000] - linear uniform
    - 't': time [1e-3, 10] - mixed log-uniform/linear uniform
    """

    def __init__(self, map_range=None):
        """
        Args:
            map_range: Optional list/tuple [lo, hi] (e.g., [-1, 1]).
                       If provided, z-scored values are linearly mapped to this range.
                       If None, standard z-score normalization (unbounded).
        """
        self.log_features = ['t']  # Log-uniform sampled (partially)
        self.linear_features = ['a', 'b']  # Linear uniform sampled

        self.map_range = map_range
        self.log_mean = {}
        self.log_std = {}
        self.linear_mean = {}
        self.linear_std = {}
        self.original_z_min = {}
        self.original_z_max = {}

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

        # Compute z-score bounds for map_range
        if self.map_range is not None:
            # Log features: snap to next integer outward
            for feat in self.log_features:
                z_values = (np.log10(data_dict[feat]) - self.log_mean[feat]) / self.log_std[feat]
                self.original_z_min[feat] = np.floor(np.min(z_values))
                self.original_z_max[feat] = np.ceil(np.max(z_values))
            # Linear features: use exact min/max (no snapping)
            for feat in self.linear_features:
                z_values = (data_dict[feat] - self.linear_mean[feat]) / self.linear_std[feat]
                self.original_z_min[feat] = np.min(z_values)
                self.original_z_max[feat] = np.max(z_values)

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

            sign_values = np.sign(values)
            log_values = np.log10(values)
            normalized[feat] = (log_values - self.log_mean[feat]) / self.log_std[feat]

        # Standard normalization for a and b
        for feat in self.linear_features:
            normalized[feat] = (data_dict[feat] - self.linear_mean[feat]) / self.linear_std[feat]

        # Apply map_range if specified
        if self.map_range is not None:
            map_lo, map_hi = self.map_range
            for feat in list(self.log_features) + list(self.linear_features):
                z_min = self.original_z_min[feat]
                z_max = self.original_z_max[feat]
                normalized[feat] = map_lo + (normalized[feat] - z_min) / (z_max - z_min) * (map_hi - map_lo)

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
        # Reverse map_range if specified
        if self.map_range is not None:
            map_lo, map_hi = self.map_range
            for feat in list(self.log_features) + list(self.linear_features):
                z_min = self.original_z_min[feat]
                z_max = self.original_z_max[feat]
                normalized_dict[feat] = z_min + (normalized_dict[feat] - map_lo) / (map_hi - map_lo) * (z_max - z_min)

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
            original_dtype = X_norm.dtype
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
            X_denorm = torch.tensor(X_denorm, dtype=original_dtype, device=device)

        return X_denorm



class Exponential_OutputNormalizer:
    """Normalization for exponential output data: [x_t, v_t, a_t]

    Outputs: x_t, v_t (velocity), a_t (acceleration) can span many orders of magnitude.

    Note: Assumes all output values are positive (like absolute magnitudes).
    """

    def __init__(self, use_log_normalization=True, map_range=None):
        """
        Args:
            use_log_normalization: If True, use log-space normalization.
                                  If False, use standard normalization.
            map_range: Optional list/tuple [lo, hi] (e.g., [-1, 1]).
                       If provided, z-scored values are linearly mapped to this range.
                       Only applies to continuous values (log-abs or linear), NOT to signs.
                       If None, standard z-score normalization (unbounded).
        """
        self.use_log_normalization = use_log_normalization
        self.map_range = map_range
        self.eps = 1e-12 # Small epsilon for numerical stability
        self.original_z_min = {}
        self.original_z_max = {}
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
                log_values = np.log10(np.abs(values) + self.eps)
                self.log_mean[feat] = np.mean(log_values)
                self.log_std[feat] = np.std(log_values)

            # Compute z-score bounds for map_range
            if self.map_range is not None:
                for feat in self.log_features:
                    log_values = np.log10(np.abs(data_dict[feat]) + self.eps)
                    z_values = (log_values - self.log_mean[feat]) / self.log_std[feat]
                    self.original_z_min[feat] = np.floor(np.min(z_values))
                    self.original_z_max[feat] = np.ceil(np.max(z_values))
        else:
            # For linear normalization
            for feat in self.linear_features:
                self.linear_mean[feat] = np.mean(data_dict[feat])
                self.linear_std[feat] = np.std(data_dict[feat])

            # Compute z-score bounds for map_range (use exact min/max for linear features)
            if self.map_range is not None:
                for feat in self.linear_features:
                    z_values = (data_dict[feat] - self.linear_mean[feat]) / self.linear_std[feat]
                    self.original_z_min[feat] = np.min(z_values)
                    self.original_z_max[feat] = np.max(z_values)

    def transform(self, data_dict):
        """
        Normalize data

        Args:
            data_dict: Dictionary with feature arrays

        Returns:
            If use_log_normalization=True: Tuple of (normalized_dict, sign_dict)
                - normalized_dict: Dictionary with normalized log-absolute features
                - sign_dict: Dictionary with sign arrays
            If use_log_normalization=False: Tuple of (normalized_dict, None)
                - normalized_dict: Dictionary with normalized features
        """
        normalized = {}

        if self.use_log_normalization:
            signs = {}
            for feat in self.log_features:
                values = data_dict[feat]
                # Extract signs
                signs[feat] = np.sign(values)
                # Apply log transform to absolute values
                log_values = np.log10(np.abs(values) + self.eps)
                normalized[feat] = (log_values - self.log_mean[feat]) / self.log_std[feat]

            # Apply map_range to log-abs z-scores (NOT to signs)
            if self.map_range is not None:
                map_lo, map_hi = self.map_range
                for feat in self.log_features:
                    z_min = self.original_z_min[feat]
                    z_max = self.original_z_max[feat]
                    normalized[feat] = map_lo + (normalized[feat] - z_min) / (z_max - z_min) * (map_hi - map_lo)

            return normalized, signs
        else:
            for feat in self.linear_features:
                normalized[feat] = (data_dict[feat] - self.linear_mean[feat]) / self.linear_std[feat]

            # Apply map_range to linear z-scores
            if self.map_range is not None:
                map_lo, map_hi = self.map_range
                for feat in self.linear_features:
                    z_min = self.original_z_min[feat]
                    z_max = self.original_z_max[feat]
                    normalized[feat] = map_lo + (normalized[feat] - z_min) / (z_max - z_min) * (map_hi - map_lo)

            return normalized, None

    def inverse_transform(self, normalized_dict, signs_dict=None):
        """
        Denormalize data

        Args:
            normalized_dict: Dictionary with normalized feature arrays
            signs_dict: Optional dictionary with sign arrays (only used when use_log_normalization=True)

        Returns:
            Dictionary with original features
        """
        # Reverse map_range if specified
        if self.map_range is not None:
            map_lo, map_hi = self.map_range
            features = self.log_features if self.use_log_normalization else self.linear_features
            for feat in features:
                z_min = self.original_z_min[feat]
                z_max = self.original_z_max[feat]
                normalized_dict[feat] = z_min + (normalized_dict[feat] - map_lo) / (map_hi - map_lo) * (z_max - z_min)

        original = {}

        if self.use_log_normalization:
            for feat in self.log_features:
                # Recover log values and then magnitudes
                log_values = normalized_dict[feat] * self.log_std[feat] + self.log_mean[feat]
                mags = 10 ** log_values

                # Apply signs if provided
                if signs_dict is not None:
                    original[feat] = mags * signs_dict[feat]
                else:
                    # Backward compatibility: return unsigned magnitudes
                    original[feat] = mags
        else:
            # Linear normalization - signs are preserved automatically
            for feat in self.linear_features:
                original[feat] = normalized_dict[feat] * self.linear_std[feat] + self.linear_mean[feat]

        return original

    def normalize_outputs(self, Y):
        """
        Normalize output array [x_t, v_t, a_t]

        Args:
            Y: numpy array or tensor of shape (N, 3) with columns [x_t, v_t, a_t]

        Returns:
            If use_log_normalization=True: Array of shape (N, 6) with columns
                [sign_x, sign_v, sign_a, logabs_x, logabs_v, logabs_a]
            If use_log_normalization=False: Array of shape (N, 3) with normalized values
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

        # Compute unnormalized log values first (for printing)
        if self.use_log_normalization:
            # Print normalizer statistics
            print(f"Output normalizer stats:")
            print(f"  x: log_mean={self.log_mean['x']:.6f}, log_std={self.log_std['x']:.6f}")
            print(f"  v: log_mean={self.log_mean['v']:.6f}, log_std={self.log_std['v']:.6f}")
            print(f"  a: log_mean={self.log_mean['a']:.6f}, log_std={self.log_std['a']:.6f}")

            
            abs_x = np.abs(data_dict['x'])
            abs_v = np.abs(data_dict['v'])
            abs_a = np.abs(data_dict['a'])

            log_values_x = np.log10(abs_x + self.eps)
            log_values_v = np.log10(abs_v + self.eps)
            log_values_a = np.log10(abs_a + self.eps)

            # Find minimum absolute values and their corresponding log values
            min_abs_x = np.min(abs_x)
            min_abs_v = np.min(abs_v)
            min_abs_a = np.min(abs_a)

            min_abs_x_idx = np.argmin(abs_x)
            min_abs_v_idx = np.argmin(abs_v)
            min_abs_a_idx = np.argmin(abs_a)

            print(f"Minimum absolute values and their log values:")
            print(f"  x: min(|x|)={min_abs_x:.10e}, corresponding log10(|x|+eps)={log_values_x[min_abs_x_idx]:.6f}")
            print(f"  v: min(|v|)={min_abs_v:.10e}, corresponding log10(|v|+eps)={log_values_v[min_abs_v_idx]:.6f}")
            print(f"  a: min(|a|)={min_abs_a:.10e}, corresponding log10(|a|+eps)={log_values_a[min_abs_a_idx]:.6f}")

            print(f"Min/Max of Y_logabs 'x' (before norm): {np.min(log_values_x):.6f}, {np.max(log_values_x):.6f}")
            print(f"Min/Max of Y_logabs 'v' (before norm): {np.min(log_values_v):.6f}, {np.max(log_values_v):.6f}")
            print(f"Min/Max of Y_logabs 'a' (before norm): {np.min(log_values_a):.6f}, {np.max(log_values_a):.6f}")

        # Normalize using transform
        normalized_dict, sign_dict = self.transform(data_dict)

        if self.use_log_normalization:
            # Create sign array
            signs_array = np.stack([
                sign_dict['x'],
                sign_dict['v'],
                sign_dict['a']
            ], axis=1)

            # Create normalized log-absolute array
            logabs_array = np.stack([
                normalized_dict['x'],
                normalized_dict['v'],
                normalized_dict['a']
            ], axis=1)

            # Concatenate: [sign_x, sign_v, sign_a, logabs_x, logabs_v, logabs_a]
            Y_norm = np.concatenate([signs_array, logabs_array], axis=1)

            print(f"Min/Max of Y_norm logabs 'x' (after norm):  {np.min(Y_norm[:, 3]):.6f}, {np.max(Y_norm[:, 3]):.6f}")
            print(f"Min/Max of Y_norm logabs 'v' (after norm):  {np.min(Y_norm[:, 4]):.6f}, {np.max(Y_norm[:, 4]):.6f}")
            print(f"Min/Max of Y_norm logabs 'a' (after norm):  {np.min(Y_norm[:, 5]):.6f}, {np.max(Y_norm[:, 5]):.6f}")
        else:
            # Linear normalization - just stack the normalized values
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
            Y_norm: tensor or numpy array
                If use_log_normalization=True: shape (N, 6) with [sign_x, sign_v, sign_a, logabs_x, logabs_v, logabs_a]
                                               or shape (N, 3) for backward compatibility
                If use_log_normalization=False: shape (N, 3) with normalized [x_t, v_t, a_t]

        Returns:
            Denormalized array of shape (N, 3) with [x_t, v_t, a_t]
        """
        is_tensor = torch.is_tensor(Y_norm)
        if is_tensor:
            device = Y_norm.device
            original_dtype = Y_norm.dtype
            Y_norm = Y_norm.detach().cpu().numpy()

        if self.use_log_normalization:
            # Check if input has signs included
            if Y_norm.shape[1] == 6:
                # Split into signs and logabs values
                sign_dict = {
                    'x': Y_norm[:, 0],
                    'v': Y_norm[:, 1],
                    'a': Y_norm[:, 2]
                }
                normalized_dict = {
                    'x': Y_norm[:, 3],
                    'v': Y_norm[:, 4],
                    'a': Y_norm[:, 5]
                }
                # Denormalize with signs
                original_dict = self.inverse_transform(normalized_dict, signs_dict=sign_dict)
            else:
                # Backward compatibility: shape (N, 3) without signs
                normalized_dict = {
                    'x': Y_norm[:, 0],
                    'v': Y_norm[:, 1],
                    'a': Y_norm[:, 2]
                }
                # Denormalize without signs (returns unsigned magnitudes)
                original_dict = self.inverse_transform(normalized_dict, signs_dict=None)
        else:
            # Linear normalization
            normalized_dict = {
                'x': Y_norm[:, 0],
                'v': Y_norm[:, 1],
                'a': Y_norm[:, 2]
            }
            # Denormalize (signs preserved automatically)
            original_dict = self.inverse_transform(normalized_dict)

        # Convert back to array
        Y = np.stack([
            original_dict['x'],
            original_dict['v'],
            original_dict['a']
        ], axis=1)

        if is_tensor:
            Y = torch.tensor(Y, dtype=original_dtype, device=device)

        return Y


def generalize_alpha(normalizer, original_alpha, original_beta, feature_name):
    """
    Get effective alpha for a feature, accounting for map_range transformation.

    When map_range is applied, the recovery formula becomes:
        real_value = 10^(beta_eff * t_norm + alpha_eff)

    Derivation (solving the forward mapping for log10(t)):
        alpha_eff = alpha + beta * (feature_min - map_min * D_ori / D_map)

    where:
        feature_min = original_z_min[feature_name]
                      (floor of z-score min for log features, exact min for linear)
        D_ori       = original_z_max - original_z_min  (span of z-score domain)
        D_map       = map_hi - map_lo                  (span of mapped domain)
        map_min     = map_lo

    Args:
        normalizer: Exponential_DataNormalizer or Exponential_OutputNormalizer instance
        original_alpha: Original mean value (log_mean for log features, linear_mean for linear)
        original_beta: Original std value (log_std for log features, linear_std for linear)
        feature_name: Feature name (e.g., 't', 'a', 'b', 'x', 'v', 'a')

    Returns:
        float: Effective alpha (adjusted if map_range is applied, original otherwise)

    Example:
        t_alpha = train_val_inputs_normalizer.log_mean['t']
        t_beta = train_val_inputs_normalizer.log_std['t']
        t_alpha = generalize_alpha(train_val_inputs_normalizer, t_alpha, t_beta, 't')
    """
    if normalizer.map_range is not None:
        feature_min = normalizer.original_z_min[feature_name]
        feature_max = normalizer.original_z_max[feature_name]
        map_lo, map_hi = normalizer.map_range
        D_ori = feature_max - feature_min
        D_map = map_hi - map_lo
        alpha_eff = original_alpha + original_beta * (feature_min - map_lo * D_ori / D_map)
        return alpha_eff
    else:
        return original_alpha


def generalize_beta(normalizer, original_alpha, original_beta, feature_name):
    """
    Get effective beta for a feature, accounting for map_range transformation.

    When map_range is applied, the recovery formula becomes:
        real_value = 10^(beta_eff * t_norm + alpha_eff)

    Derivation (solving the forward mapping for log10(t)):
        beta_eff = beta * D_ori / D_map

    where:
        D_ori = original_z_max - original_z_min  (span of z-score domain)
        D_map = map_hi - map_lo                  (span of mapped domain)

    Args:
        normalizer: Exponential_DataNormalizer or Exponential_OutputNormalizer instance
        original_alpha: Original mean value (not used here, kept for consistent interface)
        original_beta: Original std value (log_std for log features, linear_std for linear)
        feature_name: Feature name (e.g., 't', 'a', 'b', 'x', 'v', 'a')

    Returns:
        float: Effective beta (adjusted if map_range is applied, original otherwise)

    Example:
        t_alpha = train_val_inputs_normalizer.log_mean['t']
        t_beta = train_val_inputs_normalizer.log_std['t']
        t_beta = generalize_beta(train_val_inputs_normalizer, t_alpha, t_beta, 't')
    """
    if normalizer.map_range is not None:
        feature_min = normalizer.original_z_min[feature_name]
        feature_max = normalizer.original_z_max[feature_name]
        map_lo, map_hi = normalizer.map_range
        D_ori = feature_max - feature_min
        D_map = map_hi - map_lo
        beta_eff = original_beta * D_ori / D_map
        return beta_eff
    else:
        return original_beta


class ExponentialDataset(Dataset):
    """PyTorch Dataset for exponential data"""

    def __init__(self, inputs, outputs, dtype=torch.float32):
        """
        Args:
            inputs: numpy array of shape (N, 3) - [a, b, t]
            outputs: numpy array of shape (N, 3) or (N, 6) - [x_t, v_t, a_t] or with signs
            dtype: torch dtype for the tensors (default: torch.float32)
        """
        self.inputs = torch.tensor(inputs, dtype=dtype)
        self.outputs = torch.tensor(outputs, dtype=dtype)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.outputs[idx]


def load_exponential_data(filepath='exponential_trainval_data.npz', batch_size=32, shuffle_train=True, normalize=True, dtype=torch.float32, inputs_normalizer=None, outputs_normalizer=None, inputs_map_range=None, outputs_map_range=None):
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
        dtype (torch.dtype): Torch dtype for the tensors (default: torch.float32).
        inputs_normalizer (Exponential_DataNormalizer or None): Pre-fitted input normalizer.
            If provided, this normalizer will be used instead of creating a new one.
            Useful for applying train/val normalization to test data.
        outputs_normalizer (Exponential_OutputNormalizer or None): Pre-fitted output normalizer.
            If provided, this normalizer will be used instead of creating a new one.
            Useful for applying train/val normalization to test data.
        inputs_map_range (list/tuple or None): Optional [lo, hi] range for input normalization.
            If provided, z-scored inputs are linearly mapped to this range (e.g., [-1, 1]).
            Only used when creating a new normalizer (ignored if inputs_normalizer is provided).
        outputs_map_range (list/tuple or None): Optional [lo, hi] range for output normalization.
            If provided, z-scored outputs are linearly mapped to this range (e.g., [-1, 1]).
            Only used when creating a new normalizer (ignored if outputs_normalizer is provided).

    Returns:
        tuple: A tuple containing:
            - train_loader (DataLoader): DataLoader for the training set.
            - val_loader (DataLoader): DataLoader for the validation set.
            - test_loader (DataLoader): DataLoader for the test set.
            - inputs_normalizer (Exponential_DataNormalizer or None): The input normalizer
              (either provided as parameter or newly fitted). Returns None if normalize=False.
            - outputs_normalizer (Exponential_OutputNormalizer or None): The output normalizer
              (either provided as parameter or newly fitted). Returns None if normalize=False.
    """
    # Load data
    data = np.load(filepath)

    # If input normalizer is provided then error if inputs_map_range is also provided (to avoid confusion)
    if inputs_normalizer is not None and inputs_map_range is not None:
        raise ValueError("Cannot provide both inputs_normalizer and inputs_map_range. Please provide only one to avoid confusion.")
    # If output normalizer is provided then error if outputs_map_range is also provided (to avoid confusion)
    if outputs_normalizer is not None and outputs_map_range is not None:
        raise ValueError("Cannot provide both outputs_normalizer and outputs_map_range. Please provide only one to avoid confusion.")

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
        input_data, output_data, test_size=0.2, random_state=32)

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.2, random_state=32)

    # Normalize inputs and outputs if requested
    if normalize:
        # Check if normalizers were provided as parameters
        if inputs_normalizer is None or outputs_normalizer is None:
            # Create and fit new normalizers on training data only
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

            if inputs_normalizer is None:
                inputs_normalizer = Exponential_DataNormalizer(map_range=inputs_map_range)
                inputs_normalizer.fit(train_input_dict)

            if outputs_normalizer is None:
                outputs_normalizer = Exponential_OutputNormalizer(use_log_normalization=True, map_range=outputs_map_range)
                outputs_normalizer.fit(train_output_dict)

        # Use provided or newly fitted normalizers to normalize all splits
        X_train = inputs_normalizer.normalize_inputs(X_train)
        X_val = inputs_normalizer.normalize_inputs(X_val)
        X_test = inputs_normalizer.normalize_inputs(X_test)

        y_train = outputs_normalizer.normalize_outputs(y_train)
        y_val = outputs_normalizer.normalize_outputs(y_val)
        y_test = outputs_normalizer.normalize_outputs(y_test)
    else:
        # If normalize=False, set normalizers to None
        inputs_normalizer = None
        outputs_normalizer = None

    # Create PyTorch Datasets
    train_dataset = ExponentialDataset(X_train, y_train, dtype=dtype)
    val_dataset = ExponentialDataset(X_val, y_val, dtype=dtype)
    test_dataset = ExponentialDataset(X_test, y_test, dtype=dtype)

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

    results = check_all_datasets(
        train_loader, val_loader, test_loader,
        inputs_normalizer, outputs_normalizer,
        error_threshold=0.10,  # 10% threshold
        verbose=False,  # Set True to see 5 sample errors per dataset
        plot_io_relation=True,
        plot_sample_rate=10
    )

    return train_loader, val_loader, test_loader, inputs_normalizer, outputs_normalizer


def check_dataset_consistency(dataset, inputs_normalizer, outputs_normalizer,
                              dataset_name="Dataset", error_threshold=0.10, verbose=False,
                              plot_io_relation=False, save_dir=None, plot_sample_rate=1,
                              verify_derivatives=False):
    """
    Check if normalized dataset is consistent with analytical solution.

    This function:
    1. Denormalizes the inputs [a, b, t]
    2. Computes analytical solution from denormalized inputs
    3. Compares log10(abs(analytical_solution)) with the normalized targets
    4. Reports percentage of samples with error > threshold
    5. Optionally plots input-output relationships (9 scatter plots: 3 inputs × 3 outputs)
    6. Optionally verifies derivative formulas (analytical vs numerical methods)

    Args:
        dataset: PyTorch Dataset with (inputs, outputs)
        inputs_normalizer: Exponential_DataNormalizer instance
        outputs_normalizer: Exponential_OutputNormalizer instance
        dataset_name: Name for reporting (e.g., "Train", "Val", "Test")
        error_threshold: Relative error threshold (default 0.10 = 10%)
        verbose: If True, show detailed sample errors (default False)
        plot_io_relation: If True, plot input vs output scatter plots (default False)
        save_dir: Directory to save plots. If None, uses './IOrelation' (default None)
        plot_sample_rate: Plot every Nth sample (e.g., 10 means plot 1 out of 10 samples) (default 1)
        verify_derivatives: If True, verify derivative formulas (analytical vs numerical) (default False)

    Returns:
        dict: Statistics including error counts and samples (and derivative verification if enabled)
    """
    print(f"\n{'='*70}")
    print(f"Checking {dataset_name} Dataset ({len(dataset)} samples)")
    print(f"{'='*70}")

    # Get all data from dataset
    n_samples = len(dataset)
    all_inputs = []
    all_targets = []

    # Collect all data
    for i in range(n_samples):
        inputs, targets = dataset[i]
        all_inputs.append(inputs.numpy())
        all_targets.append(targets.numpy())

    all_inputs = np.array(all_inputs)  # Shape: (N, 3)
    all_targets = np.array(all_targets)  # Shape: (N, 3)

    # Step 1: Denormalize inputs to get original [a, b, t]
    denorm_inputs = inputs_normalizer.denormalize_inputs(all_inputs)

    a_values = denorm_inputs[:, 0]
    b_values = denorm_inputs[:, 1]
    t_values = denorm_inputs[:, 2]

    # Step 2: Compute analytical solution
    analytical_outputs = np.zeros((n_samples, 3))
    invalid_count = 0

    for i in range(n_samples):
        a, b, t = a_values[i], b_values[i], t_values[i]
        x_t, v_t, acc_t = analytical_solution_exp(a, b, t)

        if np.isnan(x_t) or np.isnan(v_t) or np.isnan(acc_t):
            invalid_count += 1
            analytical_outputs[i] = [np.nan, np.nan, np.nan]
        else:
            analytical_outputs[i] = [x_t, v_t, acc_t]

    if invalid_count > 0:
        print(f"WARNING: {invalid_count} samples produced NaN values!")

    # Step 3 & 4: Normalize the analytical outputs manually: (log10(abs(x)) - mean) / std
    eps = 1e-10

    # Compute log10(abs(analytical_outputs))
    log_analytical = np.log10(np.abs(analytical_outputs) + eps)

    # Apply z-score normalization: (log_values - mean) / std
    normalized_analytical_array = np.zeros_like(log_analytical)

    feature_names = ['x', 'v', 'a']
    for i, feat in enumerate(feature_names):
        log_values = log_analytical[:, i]
        normalized_analytical_array[:, i] = (log_values - outputs_normalizer.log_mean[feat]) / outputs_normalizer.log_std[feat]

    # Step 5: Compare with actual targets
    # Extract normalized log values from all_targets (columns 3-5)
    # all_targets shape is (N, 6): [sign_x, sign_v, sign_a, logabs_x_norm, logabs_v_norm, logabs_a_norm]
    all_targets_normalized = all_targets[:, 3:6]  # Extract columns 3-5

    # Compute absolute and relative errors
    abs_errors = np.abs(all_targets_normalized - normalized_analytical_array)

    # For relative error, avoid division by zero
    denom = np.abs(normalized_analytical_array) + 1e-10
    rel_errors = abs_errors / denom

    # Count samples exceeding threshold
    high_error_mask = rel_errors > error_threshold
    high_error_count_per_sample = np.any(high_error_mask, axis=1).sum()

    # Statistics per output
    print(f"\n{'Feature':<10} {'Mean Abs':<12} {'Max Abs':<12} {'Mean Rel':<12} {'Count >{:.0%}'.format(error_threshold):<15}")
    print("-" * 70)

    feature_names = ['x_t', 'v_t', 'a_t']

    for i, name in enumerate(feature_names):
        mean_abs = np.nanmean(abs_errors[:, i])
        max_abs = np.nanmax(abs_errors[:, i])
        mean_rel = np.nanmean(rel_errors[:, i])
        high_err_cnt = high_error_mask[:, i].sum()

        print(f"{name:<10} {mean_abs:<12.4e} {max_abs:<12.4e} {mean_rel:<12.4e} {high_err_cnt:<15}")

    print("-" * 70)
    print(f"Total samples with ANY feature error > {error_threshold*100:.0f}%: "
          f"{high_error_count_per_sample} / {n_samples} "
          f"({high_error_count_per_sample/n_samples*100:.2f}%)")

    # ========================================================================
    # DERIVATIVE VERIFICATION (if requested)
    # ========================================================================
    if verify_derivatives:
        print(f"\n{'='*70}")
        print(f"DERIVATIVE FORMULA VERIFICATION - ANALYTICAL VS NUMERICAL")
        print(f"{'='*70}")

        # Extract normalization constants
        std_x = outputs_normalizer.log_std['x']
        std_v = outputs_normalizer.log_std['v']
        std_a = outputs_normalizer.log_std['a']
        std_t = inputs_normalizer.log_std['t']

        mean_x = outputs_normalizer.log_mean['x']
        mean_v = outputs_normalizer.log_mean['v']
        mean_a = outputs_normalizer.log_mean['a']
        mean_t = inputs_normalizer.log_mean['t']

        print(f"\nNormalization constants:")
        print(f"  std_x: {std_x:.6f}, mean_x: {mean_x:.6f}")
        print(f"  std_v: {std_v:.6f}, mean_v: {mean_v:.6f}")
        print(f"  std_a: {std_a:.6f}, mean_a: {mean_a:.6f}")
        print(f"  std_t: {std_t:.6f}, mean_t: {mean_t:.6f}")

        # Initialize error storage lists
        errors_v_analytical = []
        errors_v_numerical = []
        errors_vprime_analytical = []
        errors_vprime_numerical = []

        errors_a_analytical = []
        errors_a_numerical = []
        errors_aprime_analytical = []
        errors_aprime_numerical = []

        # Store predictions for random sample display
        v_real_samples = []
        v_analytical_samples = []
        v_numerical_samples = []
        v_prime_real_samples = []
        v_prime_analytical_samples = []
        v_prime_numerical_samples = []

        a_real_samples = []
        a_analytical_samples = []
        a_numerical_samples = []
        a_prime_real_samples = []
        a_prime_analytical_samples = []
        a_prime_numerical_samples = []

        ln10 = np.log(10.0)
        deriv_invalid_count = 0

        # Loop through each sample
        for i in range(n_samples):
            # Skip if analytical solution was invalid
            if np.isnan(analytical_outputs[i, 0]):
                deriv_invalid_count += 1
                continue

            # Extract data
            a = a_values[i]
            b = b_values[i]
            t = t_values[i]
            x_t = analytical_outputs[i, 0]
            v_t = analytical_outputs[i, 1]
            a_t = analytical_outputs[i, 2]

            # Get normalized ground truth from all_targets
            # Note: all_targets has shape (N, 6) with columns [sign_x, sign_v, sign_a, logabs_x_norm, logabs_v_norm, logabs_a_norm]
            # We need the normalized log absolute values (columns 3-5)
            x_prime = all_targets[i, 3]
            v_prime = all_targets[i, 4]
            a_prime = all_targets[i, 5]

            # Compute t_prime
            t_prime = (np.log10(t) - mean_t) / std_t

            # Method 1: Analytical dx'/dt' and dv'/dt'
            dx_prime_dt_prime_analytical = (std_t / std_x) * a * np.exp((std_t * t_prime + mean_t) * ln10)
            dv_prime_dt_prime_analytical = (std_t / std_v) * a * np.exp((std_t * t_prime + mean_t) * ln10)

            # Method 2: Numerical dx'/dt' and dv'/dt'
            t_prime_low = 0.9999 * t_prime
            t_prime_high = 1.0001 * t_prime

            # Convert back to real time
            t_low = np.exp((std_t * t_prime_low + mean_t) * ln10)
            t_high = np.exp((std_t * t_prime_high + mean_t) * ln10)

            # Get analytical solutions at perturbed times
            x_t_low, v_t_low, _ = analytical_solution_exp(a, b, t_low)
            x_t_high, v_t_high, _ = analytical_solution_exp(a, b, t_high)

            # Check for validity
            if any(np.isnan([x_t_low, x_t_high, v_t_low, v_t_high])) or \
               any(np.isinf([x_t_low, x_t_high, v_t_low, v_t_high])):
                deriv_invalid_count += 1
                continue

            # Normalize x and v at perturbed times
            x_prime_low = (np.log10(np.abs(x_t_low)) - mean_x) / std_x
            x_prime_high = (np.log10(np.abs(x_t_high)) - mean_x) / std_x
            v_prime_low = (np.log10(np.abs(v_t_low)) - mean_v) / std_v
            v_prime_high = (np.log10(np.abs(v_t_high)) - mean_v) / std_v

            # Finite differences (derivative with respect to normalized time t')
            dx_prime_dt_prime_numerical = (x_prime_high - x_prime_low) / (t_prime_high - t_prime_low)
            dv_prime_dt_prime_numerical = (v_prime_high - v_prime_low) / (t_prime_high - t_prime_low)

            # Compute velocities AND accelerations using both methods
            # Velocity predictions (x -> v)
            common_factor_v = (std_x / std_t) * (np.exp((std_x * x_prime + mean_x) * ln10) / t)
            v_analytical_method = np.abs(common_factor_v * dx_prime_dt_prime_analytical)
            v_numerical_method = np.abs(common_factor_v * dx_prime_dt_prime_numerical)

            # Acceleration predictions (v -> a)
            common_factor_a = (std_v / std_t) * (np.exp((std_v * v_prime + mean_v) * ln10) / t)
            a_analytical_method = np.abs(common_factor_a * dv_prime_dt_prime_analytical)
            a_numerical_method = np.abs(common_factor_a * dv_prime_dt_prime_numerical)

            # Ground truths
            v_real_abs = np.abs(v_t)
            a_real_abs = np.abs(a_t)

            # Compute normalized v' and a' from predictions
            v_prime_analytical = (np.log10(v_analytical_method) - mean_v) / std_v
            v_prime_numerical = (np.log10(v_numerical_method) - mean_v) / std_v

            a_prime_analytical = (np.log10(a_analytical_method) - mean_a) / std_a
            a_prime_numerical = (np.log10(a_numerical_method) - mean_a) / std_a

            # Compute errors in real space
            error_v_analytical = v_analytical_method - v_real_abs
            error_v_numerical = v_numerical_method - v_real_abs
            error_a_analytical = a_analytical_method - a_real_abs
            error_a_numerical = a_numerical_method - a_real_abs

            # Errors in normalized space
            error_vprime_analytical = v_prime_analytical - v_prime
            error_vprime_numerical = v_prime_numerical - v_prime
            error_aprime_analytical = a_prime_analytical - a_prime
            error_aprime_numerical = a_prime_numerical - a_prime

            # Store all errors
            errors_v_analytical.append(error_v_analytical/v_real_abs)
            errors_v_numerical.append(error_v_numerical/v_real_abs)
            errors_vprime_analytical.append(error_vprime_analytical/v_prime)
            errors_vprime_numerical.append(error_vprime_numerical/v_prime)

            errors_a_analytical.append(error_a_analytical/a_real_abs)
            errors_a_numerical.append(error_a_numerical/a_real_abs)
            errors_aprime_analytical.append(error_aprime_analytical/a_prime)
            errors_aprime_numerical.append(error_aprime_numerical/a_prime)

            # Store actual values for random sample display
            v_real_samples.append(v_real_abs)
            v_analytical_samples.append(v_analytical_method)
            v_numerical_samples.append(v_numerical_method)
            v_prime_real_samples.append(v_prime)
            v_prime_analytical_samples.append(v_prime_analytical)
            v_prime_numerical_samples.append(v_prime_numerical)

            a_real_samples.append(a_real_abs)
            a_analytical_samples.append(a_analytical_method)
            a_numerical_samples.append(a_numerical_method)
            a_prime_real_samples.append(a_prime)
            a_prime_analytical_samples.append(a_prime_analytical)
            a_prime_numerical_samples.append(a_prime_numerical)

        # Convert lists to arrays
        errors_v_analytical = np.array(errors_v_analytical)
        errors_v_numerical = np.array(errors_v_numerical)
        errors_vprime_analytical = np.array(errors_vprime_analytical)
        errors_vprime_numerical = np.array(errors_vprime_numerical)

        errors_a_analytical = np.array(errors_a_analytical)
        errors_a_numerical = np.array(errors_a_numerical)
        errors_aprime_analytical = np.array(errors_aprime_analytical)
        errors_aprime_numerical = np.array(errors_aprime_numerical)

        valid_deriv_samples = len(errors_v_analytical)

        print(f"\nValid samples: {valid_deriv_samples}")
        print(f"Invalid samples (NaN/Inf): {deriv_invalid_count}")

        # VELOCITY VERIFICATION
        print(f"\n{'-'*70}")
        print(f"VELOCITY VERIFICATION (x -> v)")
        print(f"{'-'*70}")

        print(f"\nReal Space (v in units):")
        print(f"  Analytical dx'/dt':")
        print(f"    Mean error:   {np.mean(errors_v_analytical):.6e}    Median: {np.median(errors_v_analytical):.6e}")
        print(f"    Max error:    {np.max(np.abs(errors_v_analytical)):.6e}    Std:    {np.std(errors_v_analytical):.6e}")

        print(f"\n  Numerical dx'/dt':")
        print(f"    Mean error:   {np.mean(errors_v_numerical):.6e}    Median: {np.median(errors_v_numerical):.6e}")
        print(f"    Max error:    {np.max(np.abs(errors_v_numerical)):.6e}    Std:    {np.std(errors_v_numerical):.6e}")

        print(f"\nNormalized Space (v'):")
        print(f"  Analytical dx'/dt':")
        print(f"    Mean error:   {np.mean(errors_vprime_analytical):.6e}    Median: {np.median(errors_vprime_analytical):.6e}")
        print(f"    Max error:    {np.max(np.abs(errors_vprime_analytical)):.6e}    Std:    {np.std(errors_vprime_analytical):.6e}")

        print(f"\n  Numerical dx'/dt':")
        print(f"    Mean error:   {np.mean(errors_vprime_numerical):.6e}    Median: {np.median(errors_vprime_numerical):.6e}")
        print(f"    Max error:    {np.max(np.abs(errors_vprime_numerical)):.6e}    Std:    {np.std(errors_vprime_numerical):.6e}")

        # ACCELERATION VERIFICATION
        print(f"\n{'-'*70}")
        print(f"ACCELERATION VERIFICATION (v -> a)")
        print(f"{'-'*70}")

        print(f"\nReal Space (a in units):")
        print(f"  Analytical dv'/dt':")
        print(f"    Mean error:   {np.mean(errors_a_analytical):.6e}    Median: {np.median(errors_a_analytical):.6e}")
        print(f"    Max error:    {np.max(np.abs(errors_a_analytical)):.6e}    Std:    {np.std(errors_a_analytical):.6e}")

        print(f"\n  Numerical dv'/dt':")
        print(f"    Mean error:   {np.mean(errors_a_numerical):.6e}    Median: {np.median(errors_a_numerical):.6e}")
        print(f"    Max error:    {np.max(np.abs(errors_a_numerical)):.6e}    Std:    {np.std(errors_a_numerical):.6e}")

        print(f"\nNormalized Space (a'):")
        print(f"  Analytical dv'/dt':")
        print(f"    Mean error:   {np.mean(errors_aprime_analytical):.6e}    Median: {np.median(errors_aprime_analytical):.6e}")
        print(f"    Max error:    {np.max(np.abs(errors_aprime_analytical)):.6e}    Std:    {np.std(errors_aprime_analytical):.6e}")

        print(f"\n  Numerical dv'/dt':")
        print(f"    Mean error:   {np.mean(errors_aprime_numerical):.6e}    Median: {np.median(errors_aprime_numerical):.6e}")
        print(f"    Max error:    {np.max(np.abs(errors_aprime_numerical)):.6e}    Std:    {np.std(errors_aprime_numerical):.6e}")

        # Display random samples
        if valid_deriv_samples >= 5:
            random_indices = np.random.choice(valid_deriv_samples, size=5, replace=False)

            print(f"\nRandom 5 V samples:")
            print(f"Real Abs Space (v in units):")
            for i, idx in enumerate(random_indices, 1):
                print(f"{i}. [{v_real_samples[idx]:.6e}, {v_analytical_samples[idx]:.6e}, {v_numerical_samples[idx]:.6e}]")

            print(f"\nNormalized Space (v'):")
            for i, idx in enumerate(random_indices, 1):
                print(f"{i}. [{v_prime_real_samples[idx]:.6e}, {v_prime_analytical_samples[idx]:.6e}, {v_prime_numerical_samples[idx]:.6e}]")

            print(f"\nRandom 5 A samples:")
            print(f"Real Abs Space (a in units):")
            for i, idx in enumerate(random_indices, 1):
                print(f"{i}. [{a_real_samples[idx]:.6e}, {a_analytical_samples[idx]:.6e}, {a_numerical_samples[idx]:.6e}]")

            print(f"\nNormalized Space (a'):")
            for i, idx in enumerate(random_indices, 1):
                print(f"{i}. [{a_prime_real_samples[idx]:.6e}, {a_prime_analytical_samples[idx]:.6e}, {a_prime_numerical_samples[idx]:.6e}]")

        print(f"\n{'='*70}")

        # Store derivative verification results for return
        deriv_results = {
            'valid_samples': valid_deriv_samples,
            'invalid_samples': deriv_invalid_count,
            'velocity': {
                'analytical': {
                    'v_errors': errors_v_analytical,
                    'vprime_errors': errors_vprime_analytical,
                    'mean_v_error': float(np.mean(errors_v_analytical)),
                    'mean_vprime_error': float(np.mean(errors_vprime_analytical))
                },
                'numerical': {
                    'v_errors': errors_v_numerical,
                    'vprime_errors': errors_vprime_numerical,
                    'mean_v_error': float(np.mean(errors_v_numerical)),
                    'mean_vprime_error': float(np.mean(errors_vprime_numerical))
                }
            },
            'acceleration': {
                'analytical': {
                    'a_errors': errors_a_analytical,
                    'aprime_errors': errors_aprime_analytical,
                    'mean_a_error': float(np.mean(errors_a_analytical)),
                    'mean_aprime_error': float(np.mean(errors_aprime_analytical))
                },
                'numerical': {
                    'a_errors': errors_a_numerical,
                    'aprime_errors': errors_aprime_numerical,
                    'mean_a_error': float(np.mean(errors_a_numerical)),
                    'mean_aprime_error': float(np.mean(errors_aprime_numerical))
                }
            }
        }
    else:
        deriv_results = None

    # Optionally show sample high-error cases
    if verbose and high_error_count_per_sample > 0:
        print(f"\nShowing 5 sample high-error cases:")
        high_err_sample_indices = np.where(np.any(high_error_mask, axis=1))[0][:5]

        for idx in high_err_sample_indices:
            print(f"\n  Sample #{idx}:")
            print(f"    Input (a, b, t): [{a_values[idx]:.3f}, {b_values[idx]:.3f}, {t_values[idx]:.6f}]")
            for j, name in enumerate(feature_names):
                if high_error_mask[idx, j]:
                    print(f"    {name}: target={all_targets_normalized[idx, j]:.4f}, "
                          f"expected={normalized_analytical_array[idx, j]:.4f}, "
                          f"rel_err={rel_errors[idx, j]*100:.2f}%")

    # Plot input-output relationships if requested
    if plot_io_relation:
        if save_dir is None:
            save_dir = './IOrelation'

        # Create directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)

        # Apply sampling
        if plot_sample_rate > 1:
            sample_indices = np.arange(0, n_samples, plot_sample_rate)
            sampled_inputs = denorm_inputs[sample_indices]
            sampled_targets = all_targets_normalized[sample_indices]
            n_plot_samples = len(sample_indices)
        else:
            sampled_inputs = denorm_inputs
            sampled_targets = all_targets_normalized
            n_plot_samples = n_samples

        print(f"\nGenerating input-output relationship plots...")
        print(f"Plotting {n_plot_samples} / {n_samples} samples (sample rate: 1/{plot_sample_rate})")
        print(f"Saving to: {save_dir}")

        input_names = ['a', 'b', 't']
        output_names = ['x_t', 'v_t', 'a_t']

        # Create 9 plots (3 inputs × 3 outputs)
        for i, input_name in enumerate(input_names):
            for j, output_name in enumerate(output_names):
                plt.figure(figsize=(8, 6))

                # Get sampled input and output data
                input_data = sampled_inputs[:, i]
                output_data = sampled_targets[:, j]  # Normalized targets

                # Scatter plot with small alpha for large datasets
                alpha = min(0.5, 5000.0 / n_plot_samples) if n_plot_samples > 0 else 0.5
                plt.scatter(input_data, output_data, alpha=alpha, s=2, c='blue', edgecolors='none')

                plt.xlabel(f'{input_name}', fontsize=12)
                plt.ylabel(f'{output_name} (normalized)', fontsize=12)
                plt.title(f'{dataset_name}: {input_name} vs {output_name}\n({n_plot_samples} samples)', fontsize=14)
                plt.grid(True, alpha=0.3)

                # Save figure
                filename = f'{dataset_name.lower()}_{input_name}_vs_{output_name}.png'
                filepath = os.path.join(save_dir, filename)
                plt.savefig(filepath, dpi=100, bbox_inches='tight')
                plt.close()

        print(f"Saved 9 plots to {save_dir}")

    # Return statistics
    result = {
        'n_samples': n_samples,
        'invalid_count': invalid_count,
        'high_error_count': high_error_count_per_sample,
        'high_error_percentage': high_error_count_per_sample / n_samples * 100,
        'mean_abs_errors': np.nanmean(abs_errors, axis=0),
        'max_abs_errors': np.nanmax(abs_errors, axis=0),
        'mean_rel_errors': np.nanmean(rel_errors, axis=0),
        'high_error_counts_per_feature': high_error_mask.sum(axis=0)
    }

    # Add derivative verification results if enabled
    if verify_derivatives:
        result['derivative_verification'] = deriv_results

    return result


def check_all_datasets(train_loader, val_loader, test_loader,
                      inputs_normalizer, outputs_normalizer,
                      error_threshold=0.10, verbose=False,
                      plot_io_relation=False, save_dir=None, plot_sample_rate=1):
    """
    Check consistency for all datasets (train, val, test).

    Args:
        train_loader: DataLoader for training set
        val_loader: DataLoader for validation set
        test_loader: DataLoader for test set
        inputs_normalizer: Fitted Exponential_DataNormalizer
        outputs_normalizer: Fitted Exponential_OutputNormalizer
        error_threshold: Relative error threshold (default 0.10 = 10%)
        verbose: If True, show detailed sample errors (default False)
        plot_io_relation: If True, plot input vs output scatter plots (default False)
        save_dir: Directory to save plots. If None, uses './IOrelation' (default None)
        plot_sample_rate: Plot every Nth sample (e.g., 10 means plot 1 out of 10 samples) (default 1)

    Returns:
        dict: Statistics for all datasets
    """
    results = {}

    # Check train dataset
    results['train'] = check_dataset_consistency(
        train_loader.dataset,
        inputs_normalizer,
        outputs_normalizer,
        dataset_name="Training",
        error_threshold=error_threshold,
        verbose=verbose,
        plot_io_relation=plot_io_relation,
        save_dir=save_dir,
        plot_sample_rate=plot_sample_rate
    )

    # Check validation dataset
    results['val'] = check_dataset_consistency(
        val_loader.dataset,
        inputs_normalizer,
        outputs_normalizer,
        dataset_name="Validation",
        error_threshold=error_threshold,
        verbose=verbose,
        plot_io_relation=plot_io_relation,
        save_dir=save_dir,
        plot_sample_rate=plot_sample_rate
    )

    # Check test dataset
    results['test'] = check_dataset_consistency(
        test_loader.dataset,
        inputs_normalizer,
        outputs_normalizer,
        dataset_name="Test",
        error_threshold=error_threshold,
        verbose=verbose,
        plot_io_relation=plot_io_relation,
        save_dir=save_dir,
        plot_sample_rate=plot_sample_rate
    )

    # Print overall summary
    print(f"\n{'='*70}")
    print(f"OVERALL SUMMARY (Error threshold: {error_threshold*100:.0f}%)")
    print(f"{'='*70}")
    print(f"{'Dataset':<12} {'Samples':<10} {'High Err':<12} {'% High Err':<12}")
    print("-" * 70)

    for dataset_name in ['train', 'val', 'test']:
        stats = results[dataset_name]
        print(f"{dataset_name.capitalize():<12} {stats['n_samples']:<10} "
              f"{stats['high_error_count']:<12} {stats['high_error_percentage']:<12.2f}%")

    print("="*70)

    return results


def check_raw_data_residuals(data_path, use_relative=False):
    """
    Check physics residuals for raw data (before normalization).
    This validates that the original generated data satisfies the physics equation:
    (1/(2a))*a_t + 0.5*v_t - a*x_t = 0

    Args:
        data_path: Path to the .npz data file
        use_relative: If True, compute scale-invariant relative residual

    Returns:
        dict: Statistics about residuals including mean, std, min, max
    """
    print("\n" + "="*80)
    print("CHECKING RAW DATA PHYSICS RESIDUALS (Before Normalization)")
    print("="*80)
    print(f"Data file: {data_path}")
    print(f"Use relative residual: {use_relative}")
    print("-"*80)

    # Load raw data
    data = np.load(data_path)


    # Extract the array (npz files can contain multiple arrays)
    if isinstance(data, np.lib.npyio.NpzFile):
        # Get the first array in the npz file
        array_name = list(data.keys())[0]
        data_array = data[array_name]
    else:
        data_array = data
    # Extract inputs and targets (raw, unnormalized)
    # inputs: [a, b, t]
    # targets: [x_t, v_t, a_t] (raw real values, not log-space)
    inputs = data_array[:, :3]   # Shape: (n_samples, 3)
    targets = data_array[:, 3:]  # Shape: (n_samples, 3)

    # Extract parameters and target values
    a = inputs[:, 0]  # exponential rate parameter
    x_t = targets[:, 0]
    v_t = targets[:, 1]
    a_t = targets[:, 2]

    # DIAGNOSTIC: Print raw data sample values
    print(f"\n[DIAGNOSTIC] Raw data sample values (first sample):")
    print(f"  a: {a[0]:.6e}, x_t: {x_t[0]:.6e}, v_t: {v_t[0]:.6e}, a_t: {a_t[0]:.6e}")

    # Physics residual: (1/(2a))*a_t + 0.5*v_t - a*x_t = 0
    eps = 1e-10
    residual = (1.0 / (2.0 * a + eps)) * a_t + 0.5 * v_t - a * x_t

    if use_relative:
        # Scale-invariant relative residual
        # Normalize by target's acceleration term: (1/(2a))*a_target
        scale = np.abs((1.0 / (2.0 * a + eps)) * a_t) + eps
        residual = residual / scale
        residual_type = "relative"
    else:
        residual_type = "absolute"

    # Calculate statistics
    mean_abs_residual = np.mean(np.abs(residual))
    std_residual = np.std(residual)
    min_residual = np.min(residual)
    max_residual = np.max(residual)

    # Print results
    print(f"\nRaw data {residual_type} residual statistics:")
    print(f"  Mean absolute residual: {mean_abs_residual:.6e}")
    print(f"  Std residual:           {std_residual:.6e}")
    print(f"  Min residual:           {min_residual:.6e}")
    print(f"  Max residual:           {max_residual:.6e}")
    print(f"  Residual range:         [{min_residual:.6e}, {max_residual:.6e}]")

    # Evaluation
    if use_relative:
        threshold = 1e-3
    else:
        threshold = 1e-6

    if mean_abs_residual < threshold:
        print(f"\n✓ PASSED: Raw data residual is very small (< {threshold:.0e})")
        print("  The original data satisfies the physics equation well!")
    elif mean_abs_residual < threshold * 1000:
        print(f"\n⚠ WARNING: Raw data residual is small but not negligible (< {threshold*1000:.0e})")
        print("  The data may have numerical errors from generation.")
    else:
        print(f"\n✗ FAILED: Raw data residual is large (>= {threshold*1000:.0e})")
        print("  The data generation may have errors!")

    print("="*80 + "\n")

    return {
        'mean_abs_residual': mean_abs_residual,
        'std_residual': std_residual,
        'min_residual': min_residual,
        'max_residual': max_residual,
        'use_relative': use_relative
    }


def examinenormalizer(filepath, tolerance=1e-3, batch_size=1024):
    """
    Examine if normalization→denormalization is perfectly reversible.

    This function tests whether the normalization pipeline can perfectly reconstruct
    the original data after normalize→denormalize operations. Tests both:
    1. Direct numpy array normalization/denormalization
    2. DataLoader-based (tensor format) normalization/denormalization

    Args:
        filepath: Path to .npz data file
        tolerance: Maximum acceptable relative error (default: 1e-6)
        batch_size: Batch size for DataLoader test (default: 1024)

    Returns:
        dict with diagnostic results:
            - 'inputs_max_error': Max relative error for inputs
            - 'outputs_max_error': Max relative error for outputs
            - 'inputs_failed_samples': Number of samples exceeding tolerance
            - 'outputs_failed_samples': Number of samples exceeding tolerance
            - 'dataloader_inputs_failed_samples': Number failing in DataLoader
            - 'dataloader_outputs_failed_samples': Number failing in DataLoader
    """
    print("\n" + "="*80)
    print("EXAMINING NORMALIZER REVERSIBILITY")
    print("="*80)
    print(f"Testing normalization→denormalization pipeline on: {filepath}")
    print(f"Tolerance: {tolerance:.0e}")

    # Step 1: Load raw data
    data = np.load(filepath)
    data_array = data['data']  # Shape: (N, 6) = [a, b, t, x_t, v_t, a_t]

    print(f"Testing all {len(data_array)} samples")

    # Split inputs and outputs
    input_data = data_array[:, :3]   # [a, b, t]
    output_data = data_array[:, 3:]  # [x_t, v_t, a_t]

    # Step 2: Create and fit normalizers (same as load_exponential_data)
    print("\n[FITTING NORMALIZERS]")
    inputs_normalizer = Exponential_DataNormalizer()
    targets_normalizer = Exponential_OutputNormalizer(use_log_normalization=True)

    inputs_normalizer.fit({
        'a': input_data[:, 0],
        'b': input_data[:, 1],
        't': input_data[:, 2]
    })

    targets_normalizer.fit({
        'x': output_data[:, 0],
        'v': output_data[:, 1],
        'a': output_data[:, 2]
    })

    # Print normalization statistics
    print("\n[INPUT NORMALIZER STATS]")
    print(f"  Linear features (a, b):")
    print(f"    a: mean={inputs_normalizer.linear_mean['a']:.6f}, std={inputs_normalizer.linear_std['a']:.6f}")
    print(f"    b: mean={inputs_normalizer.linear_mean['b']:.6f}, std={inputs_normalizer.linear_std['b']:.6f}")
    print(f"  Log features (t):")
    print(f"    t: log_mean={inputs_normalizer.log_mean['t']:.6f}, log_std={inputs_normalizer.log_std['t']:.6f}")

    print("\n[OUTPUT NORMALIZER STATS]")
    print(f"  Log features (x, v, a):")
    print(f"    x: log_mean={targets_normalizer.log_mean['x']:.6f}, log_std={targets_normalizer.log_std['x']:.6f}")
    print(f"    v: log_mean={targets_normalizer.log_mean['v']:.6f}, log_std={targets_normalizer.log_std['v']:.6f}")
    print(f"    a: log_mean={targets_normalizer.log_mean['a']:.6f}, log_std={targets_normalizer.log_std['a']:.6f}")

    # Step 3: Test inputs normalization reversibility
    print("\n[TESTING INPUTS REVERSIBILITY]")
    inputs_dict = {
        'a': input_data[:, 0],
        'b': input_data[:, 1],
        't': input_data[:, 2]
    }
    inputs_normalized = inputs_normalizer.transform(inputs_dict)

    # Convert to array format (same as load_exponential_data)
    inputs_array_norm = np.stack([
        inputs_normalized['a'],
        inputs_normalized['b'],
        inputs_normalized['t']
    ], axis=1)

    # Denormalize back
    inputs_reconstructed = inputs_normalizer.denormalize_inputs(inputs_array_norm)

    # Compute errors
    inputs_error = np.abs(input_data - inputs_reconstructed)
    inputs_rel_error = inputs_error / (np.abs(input_data) + 1e-10)
    inputs_max_error = np.max(inputs_rel_error)
    inputs_failed = np.sum(np.max(inputs_rel_error, axis=1) > tolerance)

    print(f"  Max relative error: {inputs_max_error:.6e}")
    print(f"  Failed samples (>{tolerance:.0e}): {inputs_failed}/{len(input_data)}")

    if inputs_max_error > tolerance:
        print(f"  ✗ FAILED: Inputs normalization is NOT reversible!")
        # Print worst sample
        worst_idx = np.argmax(np.max(inputs_rel_error, axis=1))
        print(f"\n  Worst sample (index {worst_idx}):")
        print(f"    Original:      a={input_data[worst_idx, 0]:.6e}, b={input_data[worst_idx, 1]:.6e}, t={input_data[worst_idx, 2]:.6e}")
        print(f"    Reconstructed: a={inputs_reconstructed[worst_idx, 0]:.6e}, b={inputs_reconstructed[worst_idx, 1]:.6e}, t={inputs_reconstructed[worst_idx, 2]:.6e}")
        print(f"    Relative error: a={inputs_rel_error[worst_idx, 0]:.6e}, b={inputs_rel_error[worst_idx, 1]:.6e}, t={inputs_rel_error[worst_idx, 2]:.6e}")
    else:
        print(f"  ✓ PASSED: Inputs normalization is reversible")

    # Step 4: Test outputs normalization reversibility
    print("\n[TESTING OUTPUTS REVERSIBILITY]")
    outputs_dict = {
        'x': output_data[:, 0],
        'v': output_data[:, 1],
        'a': output_data[:, 2]
    }
    outputs_normalized, signs_dict = targets_normalizer.transform(outputs_dict)

    # Convert to array format (same as load_exponential_data)
    outputs_array_norm = np.stack([
        signs_dict['x'],
        signs_dict['v'],
        signs_dict['a'],
        outputs_normalized['x'],
        outputs_normalized['v'],
        outputs_normalized['a']
    ], axis=1)  # Shape: (N, 6) = [sign_x, sign_v, sign_a, logabs_x, logabs_v, logabs_a]

    # Denormalize back using denormalize_outputs()
    outputs_reconstructed = targets_normalizer.denormalize_outputs(outputs_array_norm)

    # Compute errors
    outputs_error = np.abs(output_data - outputs_reconstructed)
    outputs_rel_error = outputs_error / (np.abs(output_data) + 1e-10)
    outputs_max_error = np.max(outputs_rel_error)
    outputs_failed = np.sum(np.max(outputs_rel_error, axis=1) > tolerance)

    print(f"  Max relative error: {outputs_max_error:.6e}")
    print(f"  Failed samples (>{tolerance:.0e}): {outputs_failed}/{len(output_data)}")

    if outputs_max_error > tolerance:
        print(f"  ✗ FAILED: Outputs normalization is NOT reversible!")
        # Print worst sample
        worst_idx = np.argmax(np.max(outputs_rel_error, axis=1))
        print(f"\n  Worst sample (index {worst_idx}):")
        print(f"    Original:      x_t={output_data[worst_idx, 0]:.6e}, v_t={output_data[worst_idx, 1]:.6e}, a_t={output_data[worst_idx, 2]:.6e}")
        print(f"    Reconstructed: x_t={outputs_reconstructed[worst_idx, 0]:.6e}, v_t={outputs_reconstructed[worst_idx, 1]:.6e}, a_t={outputs_reconstructed[worst_idx, 2]:.6e}")
        print(f"    Relative error: x_t={outputs_rel_error[worst_idx, 0]:.6e}, v_t={outputs_rel_error[worst_idx, 1]:.6e}, a_t={outputs_rel_error[worst_idx, 2]:.6e}")
        print(f"    Signs: x={signs_dict['x'][worst_idx]}, v={signs_dict['v'][worst_idx]}, a={signs_dict['a'][worst_idx]}")
        print(f"    Normalized logabs: x={outputs_normalized['x'][worst_idx]:.6f}, v={outputs_normalized['v'][worst_idx]:.6f}, a={outputs_normalized['a'][worst_idx]:.6f}")
    else:
        print(f"  ✓ PASSED: Outputs normalization is reversible")

    # Step 5: Test with DataLoader (tensor format, no shuffle)
    print("\n[TESTING DATALOADER REVERSIBILITY (TENSOR FORMAT)]")
    print(f"Creating DataLoader with batch_size={batch_size}, shuffle=False")

    # Create dataset from normalized arrays
    dataset = ExponentialDataset(inputs_array_norm, outputs_array_norm)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Collect all denormalized data from dataloader
    all_inputs_from_loader = []
    all_outputs_from_loader = []

    for batch_inputs, batch_outputs in dataloader:
        # Convert to numpy for denormalization
        batch_inputs_np = batch_inputs.numpy()
        batch_outputs_np = batch_outputs.numpy()

        # Denormalize
        batch_inputs_denorm = inputs_normalizer.denormalize_inputs(batch_inputs_np)
        batch_outputs_denorm = targets_normalizer.denormalize_outputs(batch_outputs_np)

        all_inputs_from_loader.append(batch_inputs_denorm)
        all_outputs_from_loader.append(batch_outputs_denorm)

    # Concatenate all batches
    inputs_from_loader = np.concatenate(all_inputs_from_loader, axis=0)
    outputs_from_loader = np.concatenate(all_outputs_from_loader, axis=0)

    # Compare with original raw data
    loader_inputs_error = np.abs(input_data - inputs_from_loader)
    loader_inputs_rel_error = loader_inputs_error / (np.abs(input_data) + 1e-10)
    loader_inputs_max_error = np.max(loader_inputs_rel_error)
    loader_inputs_failed = np.sum(np.max(loader_inputs_rel_error, axis=1) > tolerance)

    loader_outputs_error = np.abs(output_data - outputs_from_loader)
    loader_outputs_rel_error = loader_outputs_error / (np.abs(output_data) + 1e-10)
    loader_outputs_max_error = np.max(loader_outputs_rel_error)
    loader_outputs_failed = np.sum(np.max(loader_outputs_rel_error, axis=1) > tolerance)

    print(f"  Inputs:")
    print(f"    Max relative error: {loader_inputs_max_error:.6e}")
    print(f"    Failed samples (>{tolerance:.0e}): {loader_inputs_failed}/{len(input_data)}")

    print(f"  Outputs:")
    print(f"    Max relative error: {loader_outputs_max_error:.6e}")
    print(f"    Failed samples (>{tolerance:.0e}): {loader_outputs_failed}/{len(output_data)}")

    if loader_inputs_max_error > tolerance or loader_outputs_max_error > tolerance:
        print(f"  ✗ FAILED: DataLoader tensor format introduces errors!")
        if loader_outputs_max_error > tolerance:
            worst_idx = np.argmax(np.max(loader_outputs_rel_error, axis=1))
            print(f"\n  Worst output sample (index {worst_idx}):")
            print(f"    Original:    x_t={output_data[worst_idx, 0]:.6e}, v_t={output_data[worst_idx, 1]:.6e}, a_t={output_data[worst_idx, 2]:.6e}")
            print(f"    From loader: x_t={outputs_from_loader[worst_idx, 0]:.6e}, v_t={outputs_from_loader[worst_idx, 1]:.6e}, a_t={outputs_from_loader[worst_idx, 2]:.6e}")
            print(f"    Rel error:   x_t={loader_outputs_rel_error[worst_idx, 0]:.6e}, v_t={loader_outputs_rel_error[worst_idx, 1]:.6e}, a_t={loader_outputs_rel_error[worst_idx, 2]:.6e}")
    else:
        print(f"  ✓ PASSED: DataLoader tensor format preserves data correctly")

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Direct numpy test:")
    print(f"  Inputs failed:  {inputs_failed}/{len(input_data)} samples")
    print(f"  Outputs failed: {outputs_failed}/{len(output_data)} samples")
    print(f"DataLoader test:")
    print(f"  Inputs failed:  {loader_inputs_failed}/{len(input_data)} samples")
    print(f"  Outputs failed: {loader_outputs_failed}/{len(output_data)} samples")
    print()

    if inputs_max_error < tolerance and outputs_max_error < tolerance:
        if loader_inputs_max_error < tolerance and loader_outputs_max_error < tolerance:
            print("✓ ALL PASSED: Both direct and DataLoader normalization are reversible")
            print("  → Normalizer implementation is correct")
            print("  → Problem must be elsewhere in the training pipeline")
        else:
            print("✗ DATALOADER FAILED: DataLoader introduces errors")
            print("  → Direct normalization works, but DataLoader format has issues")
            print("  → Check tensor conversions or batch processing")
    elif outputs_max_error > tolerance:
        print("✗ OUTPUTS FAILED: Outputs normalization has bugs")
        print("  → Check transform() / inverse_transform() / denormalize_outputs()")
        print("  → Likely issue with sign handling or log-space operations")
    else:
        print("✗ INPUTS FAILED: Inputs normalization has bugs")
        print("  → Check input normalizer implementation")
    print("="*80 + "\n")

    return {
        'inputs_max_error': float(inputs_max_error),
        'outputs_max_error': float(outputs_max_error),
        'inputs_failed_samples': int(inputs_failed),
        'outputs_failed_samples': int(outputs_failed),
        'dataloader_inputs_max_error': float(loader_inputs_max_error),
        'dataloader_outputs_max_error': float(loader_outputs_max_error),
        'dataloader_inputs_failed_samples': int(loader_inputs_failed),
        'dataloader_outputs_failed_samples': int(loader_outputs_failed)
    }
