import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from datagtgenerator import analytical_solution
import matplotlib.pyplot as plt
import os


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
            original_dtype = X_norm.dtype  # Preserve original dtype for float64 support
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
            X_denorm = torch.tensor(X_denorm, dtype=original_dtype, device=device)

        return X_denorm


class Vibration_OutputNormalizer:
    """Normalization for vibration output data: [x(t), v(t), a(t)]

    Outputs can span many orders of magnitude, so log-space normalization
    with sign separation is used by default.
    """

    def __init__(self, use_log_normalization=True):
        """
        Args:
            use_log_normalization: If True, use log-space normalization.
                                  If False, use standard normalization.
        """
        self.use_log_normalization = use_log_normalization
        self.eps = 1e-12  # Small epsilon for numerical stability
        if self.use_log_normalization:
            self.log_features = ['x', 'v', 'a']
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
            for feat in self.log_features:
                values = data_dict[feat]
                log_values = np.log10(np.abs(values) + self.eps)
                self.log_mean[feat] = np.mean(log_values)
                self.log_std[feat] = np.std(log_values)
        else:
            for feat in self.linear_features:
                self.linear_mean[feat] = np.mean(data_dict[feat])
                self.linear_std[feat] = np.std(data_dict[feat])

    def transform(self, data_dict):
        """
        Normalize data

        Args:
            data_dict: Dictionary with feature arrays

        Returns:
            If use_log_normalization=True: Tuple of (normalized_dict, sign_dict)
            If use_log_normalization=False: Tuple of (normalized_dict, None)
        """
        normalized = {}

        if self.use_log_normalization:
            signs = {}
            for feat in self.log_features:
                values = data_dict[feat]
                signs[feat] = np.sign(values)
                log_values = np.log10(np.abs(values) + self.eps)
                normalized[feat] = (log_values - self.log_mean[feat]) / self.log_std[feat]
            return normalized, signs
        else:
            for feat in self.linear_features:
                normalized[feat] = (data_dict[feat] - self.linear_mean[feat]) / self.linear_std[feat]
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
        original = {}

        if self.use_log_normalization:
            for feat in self.log_features:
                log_values = normalized_dict[feat] * self.log_std[feat] + self.log_mean[feat]
                mags = 10 ** log_values

                if signs_dict is not None:
                    original[feat] = mags * signs_dict[feat]
                else:
                    original[feat] = mags
        else:
            for feat in self.linear_features:
                original[feat] = normalized_dict[feat] * self.linear_std[feat] + self.linear_mean[feat]

        return original

    def normalize_outputs(self, Y):
        """
        Normalize output array [x(t), v(t), a(t)]

        Args:
            Y: numpy array or tensor of shape (N, 3) with columns [x, v, a]

        Returns:
            If use_log_normalization=True: Array of shape (N, 6) with columns
                [sign_x, sign_v, sign_a, logabs_x, logabs_v, logabs_a]
            If use_log_normalization=False: Array of shape (N, 3) with normalized values
        """
        is_tensor = torch.is_tensor(Y)
        if is_tensor:
            device = Y.device
            Y = Y.detach().cpu().numpy()

        data_dict = {
            'x': Y[:, 0],
            'v': Y[:, 1],
            'a': Y[:, 2]
        }

        # Compute unnormalized log values first (for printing diagnostics)
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
            signs_array = np.stack([
                sign_dict['x'],
                sign_dict['v'],
                sign_dict['a']
            ], axis=1)

            logabs_array = np.stack([
                normalized_dict['x'],
                normalized_dict['v'],
                normalized_dict['a']
            ], axis=1)

            Y_norm = np.concatenate([signs_array, logabs_array], axis=1)

            print(f"Min/Max of Y_norm logabs 'x' (after norm):  {np.min(Y_norm[:, 3]):.6f}, {np.max(Y_norm[:, 3]):.6f}")
            print(f"Min/Max of Y_norm logabs 'v' (after norm):  {np.min(Y_norm[:, 4]):.6f}, {np.max(Y_norm[:, 4]):.6f}")
            print(f"Min/Max of Y_norm logabs 'a' (after norm):  {np.min(Y_norm[:, 5]):.6f}, {np.max(Y_norm[:, 5]):.6f}")
        else:
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
        Denormalize output array

        Args:
            Y_norm: tensor or numpy array
                If use_log_normalization=True: shape (N, 6) with [sign_x, sign_v, sign_a, logabs_x, logabs_v, logabs_a]
                                               or shape (N, 3) for backward compatibility
                If use_log_normalization=False: shape (N, 3) with normalized [x, v, a]

        Returns:
            Denormalized array of shape (N, 3) with [x, v, a]
        """
        is_tensor = torch.is_tensor(Y_norm)
        if is_tensor:
            device = Y_norm.device
            original_dtype = Y_norm.dtype  # Preserve original dtype for float64 support
            Y_norm = Y_norm.detach().cpu().numpy()

        if self.use_log_normalization:
            if Y_norm.shape[1] == 6:
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
                original_dict = self.inverse_transform(normalized_dict, signs_dict=sign_dict)
            else:
                normalized_dict = {
                    'x': Y_norm[:, 0],
                    'v': Y_norm[:, 1],
                    'a': Y_norm[:, 2]
                }
                original_dict = self.inverse_transform(normalized_dict, signs_dict=None)
        else:
            normalized_dict = {
                'x': Y_norm[:, 0],
                'v': Y_norm[:, 1],
                'a': Y_norm[:, 2]
            }
            original_dict = self.inverse_transform(normalized_dict)

        Y = np.stack([
            original_dict['x'],
            original_dict['v'],
            original_dict['a']
        ], axis=1)

        if is_tensor:
            Y = torch.tensor(Y, dtype=original_dtype, device=device)

        return Y


class VibrationDataset(Dataset):
    """PyTorch Dataset for vibration data"""

    def __init__(self, inputs, outputs, dtype=torch.float32):
        """
        Args:
            inputs: numpy array of shape (N, 6) - [m, zeta, k, t, x0, v0]
            outputs: numpy array of shape (N, 3) - [x(t), v(t), a(t)]
            dtype: torch dtype for the tensors (default: torch.float32)
        """
        self.inputs = torch.tensor(inputs, dtype=dtype)
        self.outputs = torch.tensor(outputs, dtype=dtype)


    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.outputs[idx]


def load_vibration_data(filepath='vibration_data_normalized.npz', batch_size=32, shuffle_train=True, normalize=True, dtype=torch.float32, inputs_normalizer=None, outputs_normalizer=None):
    """Loads and prepares vibration data from an .npz file.

    This function splits the data into training, validation, and test sets,
    optionally normalizes both inputs and outputs, and creates PyTorch DataLoaders.

    The input .npz file is expected to contain a single numpy array where each
    row corresponds to [m, zeta, k, t, x0, v0, x(t), v(t), a(t)].

    Args:
        filepath (str): Path to the .npz data file.
        batch_size (int): Batch size for the DataLoaders.
        shuffle_train (bool): Whether to shuffle the training data.
        normalize (bool): If True, normalizes both input and output features.
        dtype (torch.dtype): Data type for torch tensors (default: torch.float32).
        inputs_normalizer (Vibration_DataNormalizer, optional): Pre-fitted input normalizer
            to use instead of creating a new one. If provided, this normalizer will be used
            directly without fitting. If None (default), a new normalizer will be created
            and fitted on training data.
        outputs_normalizer (Vibration_OutputNormalizer, optional): Pre-fitted output normalizer
            to use instead of creating a new one. If provided, this normalizer will be used
            directly without fitting. If None (default), a new normalizer will be created
            and fitted on training data.

    Returns:
        tuple: A tuple containing:
            - train_loader (DataLoader): DataLoader for the training set.
            - val_loader (DataLoader): DataLoader for the validation set.
            - test_loader (DataLoader): DataLoader for the test set.
            - inputs_normalizer (Vibration_DataNormalizer or None): The fitted
              input normalizer instance if normalize=True, otherwise None.
            - outputs_normalizer (Vibration_OutputNormalizer or None): The fitted
              output normalizer instance if normalize=True, otherwise None.
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
        input_data, output_data, test_size=0.2)#, random_state=20)

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.2)#, random_state=20)

    # Normalize inputs and outputs if requested
    if normalize:
        # Check if normalizers were provided as parameters
        if inputs_normalizer is None or outputs_normalizer is None:
            # Create data dictionaries for training set only if we need to fit new normalizers
            train_input_dict = {
                'm': X_train[:, 0],
                'zeta': X_train[:, 1],
                'k': X_train[:, 2],
                't': X_train[:, 3],
                'x0': X_train[:, 4],
                'v0': X_train[:, 5]
            }

            train_output_dict = {
                'x': y_train[:, 0],
                'v': y_train[:, 1],
                'a': y_train[:, 2]
            }

            # Create and fit new normalizers only if not provided
            if inputs_normalizer is None:
                inputs_normalizer = Vibration_DataNormalizer()
                inputs_normalizer.fit(train_input_dict)

            if outputs_normalizer is None:
                outputs_normalizer = Vibration_OutputNormalizer(use_log_normalization=True)
                outputs_normalizer.fit(train_output_dict)

        # Use normalizers (either provided or newly created)
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
    train_dataset = VibrationDataset(X_train, y_train, dtype=dtype)
    val_dataset = VibrationDataset(X_val, y_val, dtype=dtype)
    test_dataset = VibrationDataset(X_test, y_test, dtype=dtype)

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

    # Check dataset consistency if normalization was applied
    if normalize:
        results = check_all_datasets(
            train_loader, val_loader, test_loader,
            inputs_normalizer, outputs_normalizer,
            error_threshold=0.10,
            verbose=False,
            plot_io_relation=True,
            plot_sample_rate=10
        )

    return train_loader, val_loader, test_loader, inputs_normalizer, outputs_normalizer



def _validate_precision_on_split(input_data_original, output_data_original,
                                 inputs_normalizer, outputs_normalizer,
                                 dataset=None,
                                 tolerance=1e-3, dataset_name="Dataset", verbose=True):
    """Performs three precision validation checks on a single data split.

    This helper function validates that no precision is lost during:
    1. NPZ storage (comparing analytical solution with stored values)
    2. Input normalization/denormalization roundtrip (internal)
    3. Output normalization/denormalization roundtrip (internal)
    4. Dataset-based denormalization checks (if dataset provided)

    All errors are computed relative to analytical solutions from original inputs.

    Args:
        input_data_original: Original (non-normalized) input array [m, zeta, k, t, x0, v0]
        output_data_original: Original (non-normalized) output array [x_t, v_t, a_t]
        inputs_normalizer: Fitted Vibration_DataNormalizer
        outputs_normalizer: Fitted Vibration_OutputNormalizer
        dataset: VibrationDataset object containing normalized data (optional)
        tolerance: Relative error tolerance (default: 1e-3)
        dataset_name: Name for reporting (e.g., "Training", "Validation", "Test")
        verbose: If True, prints detailed results

    Returns:
        precision_stats: dict with validation results
    """
    eps = 1e-12  # For relative error calculations
    n_samples = len(input_data_original)

    # Extract original values
    m_orig = input_data_original[:, 0]
    zeta_orig = input_data_original[:, 1]
    k_orig = input_data_original[:, 2]
    t_orig = input_data_original[:, 3]
    x0_orig = input_data_original[:, 4]
    v0_orig = input_data_original[:, 5]

    x_t_orig = output_data_original[:, 0]
    v_t_orig = output_data_original[:, 1]
    a_t_orig = output_data_original[:, 2]

    # ========== CHECK 1: NPZ Storage Precision ==========
    if verbose:
        print("\n" + "="*80)
        print(f"CHECK 1: NPZ Storage Precision ({dataset_name} Set)")
        print("="*80)

    # Compute analytical solution from npz-stored inputs
    analytical_from_npz = np.zeros((n_samples, 3))
    for i in range(n_samples):
        m, zeta, k, t, x0, v0 = input_data_original[i]
        c = 2 * zeta * np.sqrt(m * k)
        x_t, v_t, a_t = analytical_solution(m, c, k, x0, v0, t)
        analytical_from_npz[i] = [x_t, v_t, a_t]

    # Compare with stored outputs
    # Use relative error: |computed - stored| / |computed|
    # This makes the error relative to the analytical value from original input
    error_x = np.abs(analytical_from_npz[:, 0] - x_t_orig)
    error_v = np.abs(analytical_from_npz[:, 1] - v_t_orig)
    error_a = np.abs(analytical_from_npz[:, 2] - a_t_orig)

    rel_error_x = error_x / (np.abs(analytical_from_npz[:, 0]) + eps)
    rel_error_v = error_v / (np.abs(analytical_from_npz[:, 1]) + eps)
    rel_error_a = error_a / (np.abs(analytical_from_npz[:, 2]) + eps)

    check1_stats = {
        'description': 'NPZ storage precision check',
        'analytical_vs_stored_error': {
            'x_t': {
                'mean': float(np.mean(rel_error_x)),
                'max': float(np.max(rel_error_x)),
                'failed_samples': int(np.sum(rel_error_x > tolerance))
            },
            'v_t': {
                'mean': float(np.mean(rel_error_v)),
                'max': float(np.max(rel_error_v)),
                'failed_samples': int(np.sum(rel_error_v > tolerance))
            },
            'a_t': {
                'mean': float(np.mean(rel_error_a)),
                'max': float(np.max(rel_error_a)),
                'failed_samples': int(np.sum(rel_error_a > tolerance))
            }
        },
        'passed': (np.max(rel_error_x) < tolerance and
                   np.max(rel_error_v) < tolerance and
                   np.max(rel_error_a) < tolerance)
    }

    if verbose:
        print(f"Analytical vs Stored:")
        print(f"  x_t: mean_error={check1_stats['analytical_vs_stored_error']['x_t']['mean']:.6e}, "
              f"max_error={check1_stats['analytical_vs_stored_error']['x_t']['max']:.6e}")
        print(f"  v_t: mean_error={check1_stats['analytical_vs_stored_error']['v_t']['mean']:.6e}, "
              f"max_error={check1_stats['analytical_vs_stored_error']['v_t']['max']:.6e}")
        print(f"  a_t: mean_error={check1_stats['analytical_vs_stored_error']['a_t']['mean']:.6e}, "
              f"max_error={check1_stats['analytical_vs_stored_error']['a_t']['max']:.6e}")
        print(f"\nCheck 1: {'PASSED' if check1_stats['passed'] else 'FAILED'}")

    # ========== CHECK 2: Input Normalization Reversibility ==========
    if verbose:
        print("\n" + "="*80)
        print(f"CHECK 2: Input Normalization Reversibility ({dataset_name} Set)")
        print("="*80)

    # Normalize then denormalize inputs
    input_normalized = inputs_normalizer.normalize_inputs(input_data_original)
    input_reconstructed = inputs_normalizer.denormalize_inputs(input_normalized)

    # Check reconstruction error per feature
    # Use relative error: |reconstructed - original| / |original|
    denorm_errors = {}
    feature_names = ['m', 'zeta', 'k', 't', 'x0', 'v0']
    max_denorm_error = 0

    for idx, feat in enumerate(feature_names):
        abs_error = np.abs(input_reconstructed[:, idx] - input_data_original[:, idx])
        rel_error = abs_error / (np.abs(input_data_original[:, idx]) + eps)
        denorm_errors[feat] = {
            'mean_error': float(np.mean(rel_error)),
            'max_error': float(np.max(rel_error))
        }
        max_denorm_error = max(max_denorm_error, np.max(rel_error))

    # Compute analytical solution from reconstructed inputs
    analytical_from_reconstructed = np.zeros((n_samples, 3))
    for i in range(n_samples):
        m, zeta, k, t, x0, v0 = input_reconstructed[i]
        c = 2 * zeta * np.sqrt(m * k)
        x_t, v_t, a_t = analytical_solution(m, c, k, x0, v0, t)
        analytical_from_reconstructed[i] = [x_t, v_t, a_t]

    # Compare with analytical from original
    # Use relative error: |from_denorm - from_original| / |from_original|
    analytical_from_original = analytical_from_npz  # Reuse from Check 1

    error_x_check2 = np.abs(analytical_from_reconstructed[:, 0] - analytical_from_original[:, 0])
    error_v_check2 = np.abs(analytical_from_reconstructed[:, 1] - analytical_from_original[:, 1])
    error_a_check2 = np.abs(analytical_from_reconstructed[:, 2] - analytical_from_original[:, 2])

    # Relative to the analytical value from original input
    rel_error_x_check2 = error_x_check2 / (np.abs(analytical_from_original[:, 0]) + eps)
    rel_error_v_check2 = error_v_check2 / (np.abs(analytical_from_original[:, 1]) + eps)
    rel_error_a_check2 = error_a_check2 / (np.abs(analytical_from_original[:, 2]) + eps)

    check2_stats = {
        'description': 'Input normalization reversibility',
        'denormalized_vs_original': denorm_errors,
        'analytical_from_denorm_vs_original': {
            'x_t': {
                'mean': float(np.mean(rel_error_x_check2)),
                'max': float(np.max(rel_error_x_check2)),
                'failed_samples': int(np.sum(rel_error_x_check2 > tolerance))
            },
            'v_t': {
                'mean': float(np.mean(rel_error_v_check2)),
                'max': float(np.max(rel_error_v_check2)),
                'failed_samples': int(np.sum(rel_error_v_check2 > tolerance))
            },
            'a_t': {
                'mean': float(np.mean(rel_error_a_check2)),
                'max': float(np.max(rel_error_a_check2)),
                'failed_samples': int(np.sum(rel_error_a_check2 > tolerance))
            }
        },
        'passed': (max_denorm_error < tolerance and
                   np.max(rel_error_x_check2) < tolerance and
                   np.max(rel_error_v_check2) < tolerance and
                   np.max(rel_error_a_check2) < tolerance)
    }

    if verbose:
        print("Input reconstruction errors:")
        for feat in feature_names:
            print(f"  {feat}: mean={denorm_errors[feat]['mean_error']:.6e}, "
                  f"max={denorm_errors[feat]['max_error']:.6e}")

        print("\nAnalytical from denormalized vs original:")
        print(f"  x_t: mean={check2_stats['analytical_from_denorm_vs_original']['x_t']['mean']:.6e}, "
              f"max={check2_stats['analytical_from_denorm_vs_original']['x_t']['max']:.6e}")
        print(f"  v_t: mean={check2_stats['analytical_from_denorm_vs_original']['v_t']['mean']:.6e}, "
              f"max={check2_stats['analytical_from_denorm_vs_original']['v_t']['max']:.6e}")
        print(f"  a_t: mean={check2_stats['analytical_from_denorm_vs_original']['a_t']['mean']:.6e}, "
              f"max={check2_stats['analytical_from_denorm_vs_original']['a_t']['max']:.6e}")
        print(f"\nCheck 2: {'PASSED' if check2_stats['passed'] else 'FAILED'}")

    # ========== CHECK 3: Output Normalization Reversibility ==========
    if verbose:
        print("\n" + "="*80)
        print(f"CHECK 3: Output Normalization Reversibility ({dataset_name} Set)")
        print("="*80)

    # Use analytical solution from original inputs as ground truth
    Y_original = analytical_from_original

    # Normalize then denormalize outputs
    # NOTE: normalize_outputs() prints diagnostic info - we keep this per user preference
    Y_normalized = outputs_normalizer.normalize_outputs(Y_original)
    Y_reconstructed = outputs_normalizer.denormalize_outputs(Y_normalized)

    # Compare reconstructed vs original
    # Use relative error: |reconstructed - original| / |original|
    error_x_check3 = np.abs(Y_reconstructed[:, 0] - Y_original[:, 0])
    error_v_check3 = np.abs(Y_reconstructed[:, 1] - Y_original[:, 1])
    error_a_check3 = np.abs(Y_reconstructed[:, 2] - Y_original[:, 2])

    # Relative to the analytical value from original input
    rel_error_x_check3 = error_x_check3 / (np.abs(Y_original[:, 0]) + eps)
    rel_error_v_check3 = error_v_check3 / (np.abs(Y_original[:, 1]) + eps)
    rel_error_a_check3 = error_a_check3 / (np.abs(Y_original[:, 2]) + eps)

    check3_stats = {
        'description': 'Output normalization reversibility',
        'denormalized_vs_original': {
            'x_t': {
                'mean_error': float(np.mean(rel_error_x_check3)),
                'max_error': float(np.max(rel_error_x_check3)),
                'failed_samples': int(np.sum(rel_error_x_check3 > tolerance))
            },
            'v_t': {
                'mean_error': float(np.mean(rel_error_v_check3)),
                'max_error': float(np.max(rel_error_v_check3)),
                'failed_samples': int(np.sum(rel_error_v_check3 > tolerance))
            },
            'a_t': {
                'mean_error': float(np.mean(rel_error_a_check3)),
                'max_error': float(np.max(rel_error_a_check3)),
                'failed_samples': int(np.sum(rel_error_a_check3 > tolerance))
            }
        },
        'passed': (np.max(rel_error_x_check3) < tolerance and
                   np.max(rel_error_v_check3) < tolerance and
                   np.max(rel_error_a_check3) < tolerance)
    }

    if verbose:
        print("Output reconstruction errors:")
        print(f"  x_t: mean={check3_stats['denormalized_vs_original']['x_t']['mean_error']:.6e}, "
              f"max={check3_stats['denormalized_vs_original']['x_t']['max_error']:.6e}, "
              f"failed={check3_stats['denormalized_vs_original']['x_t']['failed_samples']}")
        print(f"  v_t: mean={check3_stats['denormalized_vs_original']['v_t']['mean_error']:.6e}, "
              f"max={check3_stats['denormalized_vs_original']['v_t']['max_error']:.6e}, "
              f"failed={check3_stats['denormalized_vs_original']['v_t']['failed_samples']}")
        print(f"  a_t: mean={check3_stats['denormalized_vs_original']['a_t']['mean_error']:.6e}, "
              f"max={check3_stats['denormalized_vs_original']['a_t']['max_error']:.6e}, "
              f"failed={check3_stats['denormalized_vs_original']['a_t']['failed_samples']}")
        print(f"\nCheck 3: {'PASSED' if check3_stats['passed'] else 'FAILED'}")

    # ========== CHECK 4: Dataset-based Validation (if dataset provided) ==========
    check4_stats = None
    if dataset is not None:
        if verbose:
            print("\n" + "="*80)
            print(f"CHECK 4: Dataset-based Denormalization ({dataset_name} Set)")
            print("="*80)
            print("Comparing internal normalization vs actual dataset normalized data")

        # Extract normalized data from the dataset (convert torch tensors to numpy)
        X_normalized_ds = dataset.inputs.numpy()  # Normalized inputs from dataset
        Y_normalized_ds = dataset.outputs.numpy()  # Normalized outputs from dataset

        # Denormalize data from dataset
        X_reconstructed_ds = inputs_normalizer.denormalize_inputs(X_normalized_ds)
        Y_reconstructed_ds = outputs_normalizer.denormalize_outputs(Y_normalized_ds)

        # Compare with internal normalization (already computed)
        input_normalized = inputs_normalizer.normalize_inputs(input_data_original)
        Y_normalized_internal = Y_normalized  # From Check 3

        # ===== Part 1: Compare normalized values (dataset vs internal) =====
        error_X_norm = np.abs(X_normalized_ds - input_normalized)
        error_Y_norm = np.abs(Y_normalized_ds - Y_normalized_internal)

        rel_error_X_norm = error_X_norm / (np.abs(input_normalized) + eps)
        rel_error_Y_norm = error_Y_norm / (np.abs(Y_normalized_internal) + eps)

        # ===== Part 2: Compare denormalized inputs with original =====
        error_X_denorm = np.abs(X_reconstructed_ds - input_data_original)
        rel_error_X_denorm = error_X_denorm / (np.abs(input_data_original) + eps)

        # Count samples with large denormalization errors per feature
        denorm_failed_counts = {}
        feature_names = ['m', 'zeta', 'k', 't', 'x0', 'v0']
        for idx, feat in enumerate(feature_names):
            failed_samples = np.sum(rel_error_X_denorm[:, idx] > tolerance)
            denorm_failed_counts[feat] = {
                'failed_samples': int(failed_samples),
                'mean_error': float(np.mean(rel_error_X_denorm[:, idx])),
                'max_error': float(np.max(rel_error_X_denorm[:, idx]))
            }

        # ===== Part 3: Compare denormalized outputs =====
        # Y_original vs Y_reconstructed_internal vs Y_reconstructed_ds
        error_Y_ds_vs_orig = np.abs(Y_reconstructed_ds - Y_original)
        error_Y_internal_vs_ds = np.abs(Y_reconstructed - Y_reconstructed_ds)

        rel_error_Y_ds_vs_orig = error_Y_ds_vs_orig / (np.abs(Y_original) + eps)
        rel_error_Y_internal_vs_ds = error_Y_internal_vs_ds / (np.abs(Y_original) + eps)

        # ===== Part 4: Compute analytical solutions from reconstructed inputs =====
        analytical_from_reconstructed_ds = np.zeros((n_samples, 3))
        for i in range(n_samples):
            m, zeta, k, t, x0, v0 = X_reconstructed_ds[i]
            c = 2 * zeta * np.sqrt(m * k)
            x_t, v_t, a_t = analytical_solution(m, c, k, x0, v0, t)
            analytical_from_reconstructed_ds[i] = [x_t, v_t, a_t]

        # analytical_from_reconstructed_internal is already computed in Check 2
        # Recompute for clarity
        analytical_from_reconstructed_internal = np.zeros((n_samples, 3))
        input_reconstructed_internal = inputs_normalizer.denormalize_inputs(input_normalized)
        for i in range(n_samples):
            m, zeta, k, t, x0, v0 = input_reconstructed_internal[i]
            c = 2 * zeta * np.sqrt(m * k)
            x_t, v_t, a_t = analytical_solution(m, c, k, x0, v0, t)
            analytical_from_reconstructed_internal[i] = [x_t, v_t, a_t]

        # Compare inputs: original vs reconstructed_internal vs reconstructed_ds
        error_X_internal_vs_orig = np.abs(input_reconstructed_internal - input_data_original)
        error_X_ds_vs_orig = np.abs(X_reconstructed_ds - input_data_original)
        error_X_ds_vs_internal = np.abs(X_reconstructed_ds - input_reconstructed_internal)

        rel_error_X_internal_vs_orig = error_X_internal_vs_orig / (np.abs(input_data_original) + eps)
        rel_error_X_ds_vs_orig = error_X_ds_vs_orig / (np.abs(input_data_original) + eps)
        rel_error_X_ds_vs_internal = error_X_ds_vs_internal / (np.abs(input_data_original) + eps)

        # Compare: analytical_from_original vs analytical_from_reconstructed_internal vs analytical_from_reconstructed_ds
        error_analytical_internal_vs_orig = np.abs(analytical_from_reconstructed_internal - analytical_from_original)
        error_analytical_ds_vs_orig = np.abs(analytical_from_reconstructed_ds - analytical_from_original)
        error_analytical_ds_vs_internal = np.abs(analytical_from_reconstructed_ds - analytical_from_reconstructed_internal)

        rel_error_analytical_internal_vs_orig = error_analytical_internal_vs_orig / (np.abs(analytical_from_original) + eps)
        rel_error_analytical_ds_vs_orig = error_analytical_ds_vs_orig / (np.abs(analytical_from_original) + eps)
        rel_error_analytical_ds_vs_internal = error_analytical_ds_vs_internal / (np.abs(analytical_from_original) + eps)

        check4_stats = {
            'description': 'Dataset-based denormalization validation',
            'normalized_data_comparison': {
                'inputs': {
                    'm': {'mean': float(np.mean(rel_error_X_norm[:, 0])), 'max': float(np.max(rel_error_X_norm[:, 0]))},
                    'zeta': {'mean': float(np.mean(rel_error_X_norm[:, 1])), 'max': float(np.max(rel_error_X_norm[:, 1]))},
                    'k': {'mean': float(np.mean(rel_error_X_norm[:, 2])), 'max': float(np.max(rel_error_X_norm[:, 2]))},
                    't': {'mean': float(np.mean(rel_error_X_norm[:, 3])), 'max': float(np.max(rel_error_X_norm[:, 3]))},
                    'x0': {'mean': float(np.mean(rel_error_X_norm[:, 4])), 'max': float(np.max(rel_error_X_norm[:, 4]))},
                    'v0': {'mean': float(np.mean(rel_error_X_norm[:, 5])), 'max': float(np.max(rel_error_X_norm[:, 5]))}
                },
                'outputs': {
                    'x_t': {'mean': float(np.mean(rel_error_Y_norm[:, 0])), 'max': float(np.max(rel_error_Y_norm[:, 0]))},
                    'v_t': {'mean': float(np.mean(rel_error_Y_norm[:, 1])), 'max': float(np.max(rel_error_Y_norm[:, 1]))},
                    'a_t': {'mean': float(np.mean(rel_error_Y_norm[:, 2])), 'max': float(np.max(rel_error_Y_norm[:, 2]))}
                }
            },
            'denormalized_inputs_vs_original': denorm_failed_counts,
            'output_reconstruction_comparison': {
                'Y_orig_vs_Y_reconstructed_ds': {
                    'x_t': {'mean': float(np.mean(rel_error_Y_ds_vs_orig[:, 0])), 'max': float(np.max(rel_error_Y_ds_vs_orig[:, 0])), 'failed_samples': int(np.sum(rel_error_Y_ds_vs_orig[:, 0] > tolerance))},
                    'v_t': {'mean': float(np.mean(rel_error_Y_ds_vs_orig[:, 1])), 'max': float(np.max(rel_error_Y_ds_vs_orig[:, 1])), 'failed_samples': int(np.sum(rel_error_Y_ds_vs_orig[:, 1] > tolerance))},
                    'a_t': {'mean': float(np.mean(rel_error_Y_ds_vs_orig[:, 2])), 'max': float(np.max(rel_error_Y_ds_vs_orig[:, 2])), 'failed_samples': int(np.sum(rel_error_Y_ds_vs_orig[:, 2] > tolerance))}
                },
                'Y_reconstructed_internal_vs_Y_reconstructed_ds': {
                    'x_t': {'mean': float(np.mean(rel_error_Y_internal_vs_ds[:, 0])), 'max': float(np.max(rel_error_Y_internal_vs_ds[:, 0]))},
                    'v_t': {'mean': float(np.mean(rel_error_Y_internal_vs_ds[:, 1])), 'max': float(np.max(rel_error_Y_internal_vs_ds[:, 1]))},
                    'a_t': {'mean': float(np.mean(rel_error_Y_internal_vs_ds[:, 2])), 'max': float(np.max(rel_error_Y_internal_vs_ds[:, 2]))}
                }
            },
            'input_reconstruction_comparison': {
                'input_orig_vs_input_internal': {
                    'm': {'mean': float(np.mean(rel_error_X_internal_vs_orig[:, 0])), 'max': float(np.max(rel_error_X_internal_vs_orig[:, 0])), 'failed_samples': int(np.sum(rel_error_X_internal_vs_orig[:, 0] > tolerance))},
                    'zeta': {'mean': float(np.mean(rel_error_X_internal_vs_orig[:, 1])), 'max': float(np.max(rel_error_X_internal_vs_orig[:, 1])), 'failed_samples': int(np.sum(rel_error_X_internal_vs_orig[:, 1] > tolerance))},
                    'k': {'mean': float(np.mean(rel_error_X_internal_vs_orig[:, 2])), 'max': float(np.max(rel_error_X_internal_vs_orig[:, 2])), 'failed_samples': int(np.sum(rel_error_X_internal_vs_orig[:, 2] > tolerance))},
                    't': {'mean': float(np.mean(rel_error_X_internal_vs_orig[:, 3])), 'max': float(np.max(rel_error_X_internal_vs_orig[:, 3])), 'failed_samples': int(np.sum(rel_error_X_internal_vs_orig[:, 3] > tolerance))},
                    'x0': {'mean': float(np.mean(rel_error_X_internal_vs_orig[:, 4])), 'max': float(np.max(rel_error_X_internal_vs_orig[:, 4])), 'failed_samples': int(np.sum(rel_error_X_internal_vs_orig[:, 4] > tolerance))},
                    'v0': {'mean': float(np.mean(rel_error_X_internal_vs_orig[:, 5])), 'max': float(np.max(rel_error_X_internal_vs_orig[:, 5])), 'failed_samples': int(np.sum(rel_error_X_internal_vs_orig[:, 5] > tolerance))}
                },
                'input_orig_vs_input_ds': {
                    'm': {'mean': float(np.mean(rel_error_X_ds_vs_orig[:, 0])), 'max': float(np.max(rel_error_X_ds_vs_orig[:, 0])), 'failed_samples': int(np.sum(rel_error_X_ds_vs_orig[:, 0] > tolerance))},
                    'zeta': {'mean': float(np.mean(rel_error_X_ds_vs_orig[:, 1])), 'max': float(np.max(rel_error_X_ds_vs_orig[:, 1])), 'failed_samples': int(np.sum(rel_error_X_ds_vs_orig[:, 1] > tolerance))},
                    'k': {'mean': float(np.mean(rel_error_X_ds_vs_orig[:, 2])), 'max': float(np.max(rel_error_X_ds_vs_orig[:, 2])), 'failed_samples': int(np.sum(rel_error_X_ds_vs_orig[:, 2] > tolerance))},
                    't': {'mean': float(np.mean(rel_error_X_ds_vs_orig[:, 3])), 'max': float(np.max(rel_error_X_ds_vs_orig[:, 3])), 'failed_samples': int(np.sum(rel_error_X_ds_vs_orig[:, 3] > tolerance))},
                    'x0': {'mean': float(np.mean(rel_error_X_ds_vs_orig[:, 4])), 'max': float(np.max(rel_error_X_ds_vs_orig[:, 4])), 'failed_samples': int(np.sum(rel_error_X_ds_vs_orig[:, 4] > tolerance))},
                    'v0': {'mean': float(np.mean(rel_error_X_ds_vs_orig[:, 5])), 'max': float(np.max(rel_error_X_ds_vs_orig[:, 5])), 'failed_samples': int(np.sum(rel_error_X_ds_vs_orig[:, 5] > tolerance))}
                },
                'input_internal_vs_input_ds': {
                    'm': {'mean': float(np.mean(rel_error_X_ds_vs_internal[:, 0])), 'max': float(np.max(rel_error_X_ds_vs_internal[:, 0]))},
                    'zeta': {'mean': float(np.mean(rel_error_X_ds_vs_internal[:, 1])), 'max': float(np.max(rel_error_X_ds_vs_internal[:, 1]))},
                    'k': {'mean': float(np.mean(rel_error_X_ds_vs_internal[:, 2])), 'max': float(np.max(rel_error_X_ds_vs_internal[:, 2]))},
                    't': {'mean': float(np.mean(rel_error_X_ds_vs_internal[:, 3])), 'max': float(np.max(rel_error_X_ds_vs_internal[:, 3]))},
                    'x0': {'mean': float(np.mean(rel_error_X_ds_vs_internal[:, 4])), 'max': float(np.max(rel_error_X_ds_vs_internal[:, 4]))},
                    'v0': {'mean': float(np.mean(rel_error_X_ds_vs_internal[:, 5])), 'max': float(np.max(rel_error_X_ds_vs_internal[:, 5]))}
                }
            },
            'analytical_solution_comparison': {
                'analytical_orig_vs_analytical_internal': {
                    'x_t': {'mean': float(np.mean(rel_error_analytical_internal_vs_orig[:, 0])), 'max': float(np.max(rel_error_analytical_internal_vs_orig[:, 0])), 'failed_samples': int(np.sum(rel_error_analytical_internal_vs_orig[:, 0] > tolerance))},
                    'v_t': {'mean': float(np.mean(rel_error_analytical_internal_vs_orig[:, 1])), 'max': float(np.max(rel_error_analytical_internal_vs_orig[:, 1])), 'failed_samples': int(np.sum(rel_error_analytical_internal_vs_orig[:, 1] > tolerance))},
                    'a_t': {'mean': float(np.mean(rel_error_analytical_internal_vs_orig[:, 2])), 'max': float(np.max(rel_error_analytical_internal_vs_orig[:, 2])), 'failed_samples': int(np.sum(rel_error_analytical_internal_vs_orig[:, 2] > tolerance))}
                },
                'analytical_orig_vs_analytical_ds': {
                    'x_t': {'mean': float(np.mean(rel_error_analytical_ds_vs_orig[:, 0])), 'max': float(np.max(rel_error_analytical_ds_vs_orig[:, 0])), 'failed_samples': int(np.sum(rel_error_analytical_ds_vs_orig[:, 0] > tolerance))},
                    'v_t': {'mean': float(np.mean(rel_error_analytical_ds_vs_orig[:, 1])), 'max': float(np.max(rel_error_analytical_ds_vs_orig[:, 1])), 'failed_samples': int(np.sum(rel_error_analytical_ds_vs_orig[:, 1] > tolerance))},
                    'a_t': {'mean': float(np.mean(rel_error_analytical_ds_vs_orig[:, 2])), 'max': float(np.max(rel_error_analytical_ds_vs_orig[:, 2])), 'failed_samples': int(np.sum(rel_error_analytical_ds_vs_orig[:, 2] > tolerance))}
                },
                'analytical_internal_vs_analytical_ds': {
                    'x_t': {'mean': float(np.mean(rel_error_analytical_ds_vs_internal[:, 0])), 'max': float(np.max(rel_error_analytical_ds_vs_internal[:, 0]))},
                    'v_t': {'mean': float(np.mean(rel_error_analytical_ds_vs_internal[:, 1])), 'max': float(np.max(rel_error_analytical_ds_vs_internal[:, 1]))},
                    'a_t': {'mean': float(np.mean(rel_error_analytical_ds_vs_internal[:, 2])), 'max': float(np.max(rel_error_analytical_ds_vs_internal[:, 2]))}
                }
            },
            'passed': (np.max(rel_error_Y_ds_vs_orig) < tolerance and
                      np.max(rel_error_analytical_ds_vs_orig) < tolerance)
        }

        # Add detailed diagnostics for samples with large analytical errors
        # Find samples where v_t analytical error is large (potential v0 issue)
        v_t_analytical_errors = rel_error_analytical_ds_vs_orig[:, 1]  # v_t errors
        large_error_indices = np.where(v_t_analytical_errors > tolerance)[0]

        if len(large_error_indices) > 0:
            # Sample a few for detailed inspection (max 5 examples)
            sample_indices = large_error_indices[:min(5, len(large_error_indices))]
            check4_stats['diagnostic_samples'] = []

            for idx in sample_indices:
                diagnostic = {
                    'sample_idx': int(idx),
                    'original_v0': float(input_data_original[idx, 5]),
                    'denorm_internal_v0': float(input_reconstructed_internal[idx, 5]),
                    'denorm_dataset_v0': float(X_reconstructed_ds[idx, 5]),
                    'v0_error_internal': float(rel_error_X_denorm[:, 5][idx]) if idx < len(rel_error_X_denorm) else 0.0,
                    'normalized_v0_dataset': float(X_normalized_ds[idx, 5]),
                    'normalized_v0_internal': float(input_normalized[idx, 5]),
                    'analytical_v_t_error': float(v_t_analytical_errors[idx])
                }
                check4_stats['diagnostic_samples'].append(diagnostic)

        if verbose:
            print("\n--- Part 1: Normalized Data Comparison (Dataset vs Internal) ---")
            print("Inputs (dataset vs internal normalization):")
            for feat in ['m', 'zeta', 'k', 't', 'x0', 'v0']:
                stats = check4_stats['normalized_data_comparison']['inputs'][feat]
                print(f"  {feat}: mean={stats['mean']:.6e}, max={stats['max']:.6e}")
            print("Outputs (dataset vs internal normalization):")
            for var in ['x_t', 'v_t', 'a_t']:
                stats = check4_stats['normalized_data_comparison']['outputs'][var]
                print(f"  {var}: mean={stats['mean']:.6e}, max={stats['max']:.6e}")

            print("\n--- Part 2: Denormalized Inputs vs Original ---")
            for feat in ['m', 'zeta', 'k', 't', 'x0', 'v0']:
                stats = check4_stats['denormalized_inputs_vs_original'][feat]
                print(f"  {feat}: mean={stats['mean_error']:.6e}, max={stats['max_error']:.6e}, failed={stats['failed_samples']}")

            print("\n--- Part 3: Output Reconstruction Comparison ---")
            print("Y_original vs Y_reconstructed_from_dataset:")
            for var in ['x_t', 'v_t', 'a_t']:
                stats = check4_stats['output_reconstruction_comparison']['Y_orig_vs_Y_reconstructed_ds'][var]
                print(f"  {var}: mean={stats['mean']:.6e}, max={stats['max']:.6e}, failed={stats['failed_samples']}")
            print("Y_reconstructed_internal vs Y_reconstructed_from_dataset:")
            for var in ['x_t', 'v_t', 'a_t']:
                stats = check4_stats['output_reconstruction_comparison']['Y_reconstructed_internal_vs_Y_reconstructed_ds'][var]
                print(f"  {var}: mean={stats['mean']:.6e}, max={stats['max']:.6e}")

            print("\n--- Part 4: Complete Comparison (Inputs + Outputs + Analytical) ---")
            print("\nComparison A: Original vs Internal_denorm")
            print("  Inputs:")
            for feat in ['m', 'zeta', 'k', 't', 'x0', 'v0']:
                stats = check4_stats['input_reconstruction_comparison']['input_orig_vs_input_internal'][feat]
                print(f"    {feat}: mean={stats['mean']:.6e}, max={stats['max']:.6e}, failed={stats['failed_samples']}")
            print("  Analytical outputs:")
            for var in ['x_t', 'v_t', 'a_t']:
                stats = check4_stats['analytical_solution_comparison']['analytical_orig_vs_analytical_internal'][var]
                print(f"    {var}: mean={stats['mean']:.6e}, max={stats['max']:.6e}, failed={stats['failed_samples']}")

            print("\nComparison B: Original vs Dataset_denorm")
            print("  Inputs:")
            for feat in ['m', 'zeta', 'k', 't', 'x0', 'v0']:
                stats = check4_stats['input_reconstruction_comparison']['input_orig_vs_input_ds'][feat]
                print(f"    {feat}: mean={stats['mean']:.6e}, max={stats['max']:.6e}, failed={stats['failed_samples']}")
            print("  Analytical outputs:")
            for var in ['x_t', 'v_t', 'a_t']:
                stats = check4_stats['analytical_solution_comparison']['analytical_orig_vs_analytical_ds'][var]
                print(f"    {var}: mean={stats['mean']:.6e}, max={stats['max']:.6e}, failed={stats['failed_samples']}")

            print("\nComparison C: Internal_denorm vs Dataset_denorm")
            print("  Inputs:")
            for feat in ['m', 'zeta', 'k', 't', 'x0', 'v0']:
                stats = check4_stats['input_reconstruction_comparison']['input_internal_vs_input_ds'][feat]
                print(f"    {feat}: mean={stats['mean']:.6e}, max={stats['max']:.6e}")
            print("  Analytical outputs:")
            for var in ['x_t', 'v_t', 'a_t']:
                stats = check4_stats['analytical_solution_comparison']['analytical_internal_vs_analytical_ds'][var]
                print(f"    {var}: mean={stats['mean']:.6e}, max={stats['max']:.6e}")

            # Print diagnostic samples if available
            if 'diagnostic_samples' in check4_stats and len(check4_stats['diagnostic_samples']) > 0:
                print("\n--- Diagnostic: Sample Cases with Large Analytical Errors ---")
                print(f"Total samples with v_t analytical error > {tolerance}: {len(large_error_indices)}")
                print(f"Showing first {len(check4_stats['diagnostic_samples'])} examples:\n")
                for i, diag in enumerate(check4_stats['diagnostic_samples'], 1):
                    print(f"Example {i} (Sample #{diag['sample_idx']}):")
                    print(f"  Original v0:           {diag['original_v0']:.6e}")
                    print(f"  Denorm internal v0:    {diag['denorm_internal_v0']:.6e}")
                    print(f"  Denorm dataset v0:     {diag['denorm_dataset_v0']:.6e}")
                    print(f"  Normalized v0 (dataset): {diag['normalized_v0_dataset']:.6e}")
                    print(f"  Normalized v0 (internal):{diag['normalized_v0_internal']:.6e}")
                    print(f"  v0 denorm error:       {diag['v0_error_internal']:.6e}")
                    print(f"  Analytical v_t error:  {diag['analytical_v_t_error']:.6e}")
                    print()

            print(f"\nCheck 4: {'PASSED' if check4_stats['passed'] else 'FAILED'}")

    # ========== OVERALL SUMMARY ==========
    if check4_stats is not None:
        overall_passed = (check1_stats['passed'] and
                         check2_stats['passed'] and
                         check3_stats['passed'] and
                         check4_stats['passed'])
    else:
        overall_passed = (check1_stats['passed'] and
                         check2_stats['passed'] and
                         check3_stats['passed'])

    summary = []
    if not check1_stats['passed']:
        summary.append("Check 1 FAILED: NPZ storage has precision issues")
    if not check2_stats['passed']:
        summary.append("Check 2 FAILED: Input normalization loses precision")
    if not check3_stats['passed']:
        summary.append("Check 3 FAILED: Output normalization loses precision")
    if check4_stats is not None and not check4_stats['passed']:
        summary.append("Check 4 FAILED: Dataset-based denormalization has precision issues")

    if overall_passed:
        summary.append("All precision checks PASSED")

    precision_stats = {
        'check1_npz_storage': check1_stats,
        'check2_input_normalization': check2_stats,
        'check3_output_normalization': check3_stats,
        'overall_passed': overall_passed,
        'summary': '\n'.join(summary),
        'tolerance_used': tolerance
    }

    if check4_stats is not None:
        precision_stats['check4_dataset_based'] = check4_stats

    if verbose:
        print("\n" + "="*80)
        print(f"OVERALL PRECISION VALIDATION SUMMARY ({dataset_name} Set)")
        print("="*80)
        print(precision_stats['summary'])
        print("="*80 + "\n")

    return precision_stats


def load_vibration_data_check(filepath='vibration_data_normalized.npz', batch_size=32, shuffle_train=True,
                              normalize=True, precision_check=True, check_tolerance=1e-3, verbose=True, dtype=torch.float32,
                              inputs_normalizer=None, outputs_normalizer=None):
    """Extended version of load_vibration_data() with precision validation.

    This function loads and prepares vibration data with comprehensive precision checks
    to verify that no precision is lost during NPZ storage and normalization operations.

    The function performs three precision validation checks on each dataset (train/val/test):
    1. NPZ storage precision (analytical vs stored)
    2. Input normalization reversibility (norm→denorm roundtrip)
    3. Output normalization reversibility (norm→denorm roundtrip)

    All errors are computed relative to analytical solutions from original inputs,
    providing physically-meaningful precision metrics.

    Args:
        filepath (str): Path to the .npz data file.
        batch_size (int): Batch size for the DataLoaders.
        shuffle_train (bool): Whether to shuffle the training data.
        normalize (bool): If True, normalizes both input and output features.
        precision_check (bool): If True, performs precision validation checks.
        check_tolerance (float): Error tolerance for precision checks (default: 1e-3,
                                 relative to analytical value).
        verbose (bool): If True, prints detailed check results.
        dtype (torch.dtype): Data type for torch tensors (default: torch.float32).
        inputs_normalizer (Vibration_DataNormalizer, optional): Pre-fitted input normalizer
            to use instead of creating a new one. If provided, this normalizer will be used
            directly without fitting. If None (default), a new normalizer will be created
            and fitted on training data.
        outputs_normalizer (Vibration_OutputNormalizer, optional): Pre-fitted output normalizer
            to use instead of creating a new one. If provided, this normalizer will be used
            directly without fitting. If None (default), a new normalizer will be created
            and fitted on training data.

    Returns:
        tuple: A tuple containing:
            - train_loader (DataLoader): DataLoader for the training set.
            - val_loader (DataLoader): DataLoader for the validation set.
            - test_loader (DataLoader): DataLoader for the test set.
            - inputs_normalizer (Vibration_DataNormalizer or None): The fitted
              input normalizer instance if normalize=True, otherwise None.
            - outputs_normalizer (Vibration_OutputNormalizer or None): The fitted
              output normalizer instance if normalize=True, otherwise None.
            - precision_stats (dict or None): Precision validation statistics if
              precision_check=True, otherwise None.
    """
    # 1. Load raw npz data
    data = np.load(filepath)

    # Extract the array (npz files can contain multiple arrays)
    if isinstance(data, np.lib.npyio.NpzFile):
        array_name = list(data.keys())[0]
        data_array = data[array_name]
    else:
        data_array = data

    input_data = data_array[:, :6]   # Inputs: m, zeta, k, t, x0, v0
    output_data = data_array[:, 6:]  # Outputs: x(t), v(t), a(t)

    # 2. Perform train/val/test split (same as load_vibration_data)
    X_train, X_temp, y_train, y_temp = train_test_split(
        input_data, output_data, test_size=0.2)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.2)

    # 3. Store original (non-normalized) splits for precision checking
    X_train_orig, y_train_orig = X_train.copy(), y_train.copy()
    X_val_orig, y_val_orig = X_val.copy(), y_val.copy()
    X_test_orig, y_test_orig = X_test.copy(), y_test.copy()

    # 4. Create and fit normalizers (if normalize=True)
    if normalize:
        # Check if normalizers were provided as parameters
        if inputs_normalizer is None or outputs_normalizer is None:
            # Create data dictionaries for training set only if we need to fit new normalizers
            train_input_dict = {
                'm': X_train[:, 0],
                'zeta': X_train[:, 1],
                'k': X_train[:, 2],
                't': X_train[:, 3],
                'x0': X_train[:, 4],
                'v0': X_train[:, 5]
            }
            train_output_dict = {
                'x': y_train[:, 0],
                'v': y_train[:, 1],
                'a': y_train[:, 2]
            }

            # Create and fit new normalizers only if not provided
            if inputs_normalizer is None:
                inputs_normalizer = Vibration_DataNormalizer()
                inputs_normalizer.fit(train_input_dict)

            if outputs_normalizer is None:
                outputs_normalizer = Vibration_OutputNormalizer(use_log_normalization=True)
                outputs_normalizer.fit(train_output_dict)

        # Use normalizers (either provided or newly created)
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

    # 5. Create datasets and loaders (same as load_vibration_data)
    train_dataset = VibrationDataset(X_train, y_train, dtype=dtype)
    val_dataset = VibrationDataset(X_val, y_val, dtype=dtype)
    test_dataset = VibrationDataset(X_test, y_test, dtype=dtype)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=0
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

    # 6. Perform precision checks AFTER creating loaders (if precision_check=True)
    precision_stats = None
    if precision_check and normalize:
        precision_stats = {
            'train': _validate_precision_on_split(
                X_train_orig, y_train_orig,
                inputs_normalizer, outputs_normalizer,
                dataset=train_dataset,
                tolerance=check_tolerance,
                dataset_name="Training",
                verbose=verbose
            ),
            'val': _validate_precision_on_split(
                X_val_orig, y_val_orig,
                inputs_normalizer, outputs_normalizer,
                dataset=val_dataset,
                tolerance=check_tolerance,
                dataset_name="Validation",
                verbose=verbose
            ),
            'test': _validate_precision_on_split(
                X_test_orig, y_test_orig,
                inputs_normalizer, outputs_normalizer,
                dataset=test_dataset,
                tolerance=check_tolerance,
                dataset_name="Test",
                verbose=verbose
            )
        }

        # 7. Call existing validation function check_all_datasets()
        existing_validation = check_all_datasets(
            train_loader, val_loader, test_loader,
            inputs_normalizer, outputs_normalizer,
            error_threshold=0.10,
            verbose=False,
            plot_io_relation=True,
            plot_sample_rate=10
        )
        precision_stats['existing_validation'] = existing_validation

        # 8. Overall pass/fail
        all_passed = (precision_stats['train']['overall_passed'] and
                     precision_stats['val']['overall_passed'] and
                     precision_stats['test']['overall_passed'])
        precision_stats['all_passed'] = all_passed

    # 9. Return with precision stats
    return train_loader, val_loader, test_loader, inputs_normalizer, outputs_normalizer, precision_stats



def check_dataset_consistency(dataset, inputs_normalizer, outputs_normalizer,
                              dataset_name="Dataset", error_threshold=0.10, verbose=False,
                              plot_io_relation=False, save_dir=None, plot_sample_rate=1,
                              verify_derivatives=False):
    """
    Check if normalized dataset is consistent with analytical solution.

    This function:
    1. Denormalizes the inputs [m, zeta, k, t, x0, v0]
    2. Computes analytical solution from denormalized inputs
    3. Compares log10(abs(analytical_solution)) with the normalized targets
    4. Reports percentage of samples with error > threshold
    5. Optionally plots input-output relationships
    6. Optionally verifies derivative formulas (numerical method only)

    Args:
        dataset: PyTorch Dataset with (inputs, outputs)
        inputs_normalizer: Vibration_DataNormalizer instance
        outputs_normalizer: Vibration_OutputNormalizer instance
        dataset_name: Name for reporting (e.g., "Train", "Val", "Test")
        error_threshold: Relative error threshold (default 0.10 = 10%)
        verbose: If True, show detailed sample errors (default False)
        plot_io_relation: If True, plot input vs output scatter plots (default False)
        save_dir: Directory to save plots. If None, uses './IOrelation' (default None)
        plot_sample_rate: Plot every Nth sample (default 1)
        verify_derivatives: If True, verify derivative formulas using numerical method (default False)

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

    all_inputs = np.array(all_inputs)  # Shape: (N, 6)
    all_targets = np.array(all_targets)  # Shape: (N, 6) if log normalization

    # Step 1: Denormalize inputs to get original [m, zeta, k, t, x0, v0]
    denorm_inputs = inputs_normalizer.denormalize_inputs(all_inputs)

    m_values = denorm_inputs[:, 0]
    zeta_values = denorm_inputs[:, 1]
    k_values = denorm_inputs[:, 2]
    t_values = denorm_inputs[:, 3]
    x0_values = denorm_inputs[:, 4]
    v0_values = denorm_inputs[:, 5]

    # Step 2: Compute analytical solution
    analytical_outputs = np.zeros((n_samples, 3))
    invalid_count = 0

    for i in range(n_samples):
        m, zeta, k, t, x0, v0 = m_values[i], zeta_values[i], k_values[i], t_values[i], x0_values[i], v0_values[i]
        # Compute damping coefficient
        c = 2 * zeta * np.sqrt(m * k)
        x_t, v_t, a_t = analytical_solution(m, c, k, x0, v0, t)

        if np.isnan(x_t) or np.isnan(v_t) or np.isnan(a_t):
            invalid_count += 1
            analytical_outputs[i] = [np.nan, np.nan, np.nan]
        else:
            analytical_outputs[i] = [x_t, v_t, a_t]

    if invalid_count > 0:
        print(f"WARNING: {invalid_count} samples produced NaN values!")

    eps = outputs_normalizer.eps

    # ========================================================================
    # DENORMALIZATION COMPARISON: Compare denormalized targets vs analytical
    # ========================================================================
    print(f"\n{'-'*70}")
    print(f"DENORMALIZATION COMPARISON: Denormalized targets vs Analytical solution")
    print(f"{'-'*70}")

    # Denormalize the targets using output normalizer
    targets_denormalized = outputs_normalizer.denormalize_outputs(all_targets)

    # Extract denormalized x_t, v_t, a_t
    x_t_denorm = targets_denormalized[:, 0]
    v_t_denorm = targets_denormalized[:, 1]
    a_t_denorm = targets_denormalized[:, 2]

    # Extract analytical x_t, v_t, a_t
    x_t_ana = analytical_outputs[:, 0]
    v_t_ana = analytical_outputs[:, 1]
    a_t_ana = analytical_outputs[:, 2]

    # Compute relative errors (avoid division by zero)
    error_threshold_denorm = 0.001  # 0.1%

    # For x_t
    valid_mask_x = ~np.isnan(x_t_ana) & (np.abs(x_t_ana) > eps)
    rel_error_x = np.abs(x_t_denorm - x_t_ana) / (np.abs(x_t_ana) + eps)
    high_error_x = np.sum((rel_error_x > error_threshold_denorm) & valid_mask_x)

    # For v_t
    valid_mask_v = ~np.isnan(v_t_ana) & (np.abs(v_t_ana) > eps)
    rel_error_v = np.abs(v_t_denorm - v_t_ana) / (np.abs(v_t_ana) + eps)
    high_error_v = np.sum((rel_error_v > error_threshold_denorm) & valid_mask_v)

    # For a_t
    valid_mask_a = ~np.isnan(a_t_ana) & (np.abs(a_t_ana) > eps)
    rel_error_a = np.abs(a_t_denorm - a_t_ana) / (np.abs(a_t_ana) + eps)
    high_error_a = np.sum((rel_error_a > error_threshold_denorm) & valid_mask_a)


    valid_count_x = np.sum(valid_mask_x)
    valid_count_v = np.sum(valid_mask_v)
    valid_count_a = np.sum(valid_mask_a)

    print(f"\nSamples with denormalized vs analytical error > {error_threshold_denorm*100:.1f}%:")
    print(f"  x_t: {high_error_x} / {valid_count_x} ({high_error_x/valid_count_x*100:.4f}%)")
    print(f"  v_t: {high_error_v} / {valid_count_v} ({high_error_v/valid_count_v*100:.4f}%)")
    print(f"  a_t: {high_error_a} / {valid_count_a} ({high_error_a/valid_count_a*100:.4f}%)")

    print(f"\nRelative error statistics:")
    print(f"  x_t: mean={np.nanmean(rel_error_x[valid_mask_x]):.6e}, max={np.nanmax(rel_error_x[valid_mask_x]):.6e}")
    print(f"  v_t: mean={np.nanmean(rel_error_v[valid_mask_v]):.6e}, max={np.nanmax(rel_error_v[valid_mask_v]):.6e}")
    print(f"  a_t: mean={np.nanmean(rel_error_a[valid_mask_a]):.6e}, max={np.nanmax(rel_error_a[valid_mask_a]):.6e}")

    # Show 5 worst cases for each
    if high_error_x > 0:
        worst_x_indices = np.argsort(rel_error_x * valid_mask_x)[-5:][::-1]
        print(f"\n  5 worst x_t cases:")
        for idx in worst_x_indices:
            if valid_mask_x[idx] and rel_error_x[idx] > error_threshold_denorm:
                print(f"    idx={idx}: denorm={x_t_denorm[idx]:.6e}, ana={x_t_ana[idx]:.6e}, rel_err={rel_error_x[idx]*100:.4f}%")

    if high_error_v > 0:
        worst_v_indices = np.argsort(rel_error_v * valid_mask_v)[-5:][::-1]
        print(f"\n  5 worst v_t cases:")
        for idx in worst_v_indices:
            if valid_mask_v[idx] and rel_error_v[idx] > error_threshold_denorm:
                print(f"    idx={idx}: denorm={v_t_denorm[idx]:.6e}, ana={v_t_ana[idx]:.6e}, rel_err={rel_error_v[idx]*100:.4f}%")

    if high_error_a > 0:
        worst_a_indices = np.argsort(rel_error_a * valid_mask_a)[-5:][::-1]
        print(f"\n  5 worst a_t cases:")
        for idx in worst_a_indices:
            if valid_mask_a[idx] and rel_error_a[idx] > error_threshold_denorm:
                print(f"    idx={idx}: denorm={a_t_denorm[idx]:.6e}, ana={a_t_ana[idx]:.6e}, rel_err={rel_error_a[idx]*100:.4f}%")

    print(f"{'-'*70}")

    # ========================================================================
    # Continue with original comparison logic
    # ========================================================================

    # Step 3 & 4: Normalize the analytical outputs manually


    # Compute log10(abs(analytical_outputs))
    log_analytical = np.log10(np.abs(analytical_outputs) + eps)

    # Apply z-score normalization
    normalized_analytical_array = np.zeros_like(log_analytical)

    feature_names = ['x', 'v', 'a']
    for i, feat in enumerate(feature_names):
        log_values = log_analytical[:, i]
        normalized_analytical_array[:, i] = (log_values - outputs_normalizer.log_mean[feat]) / outputs_normalizer.log_std[feat]

    # Step 5: Compare with actual targets
    # Extract normalized log values from all_targets (columns 3-5)
    # all_targets shape is (N, 6): [sign_x, sign_v, sign_a, logabs_x_norm, logabs_v_norm, logabs_a_norm]
    all_targets_normalized = all_targets[:, 3:6]

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

    output_names = ['x_t', 'v_t', 'a_t']

    for i, name in enumerate(output_names):
        mean_abs = np.nanmean(abs_errors[:, i])
        max_abs = np.nanmax(abs_errors[:, i])
        mean_rel = np.nanmean(rel_errors[:, i])
        high_err_cnt = high_error_mask[:, i].sum()

        print(f"{name:<10} {mean_abs:<12.4e} {max_abs:<12.4e} {mean_rel:<12.4e} {high_err_cnt:<15}")

    print("-" * 70)
    print(f"Total samples with ANY feature error > {error_threshold*100:.0f}%: "
          f"{high_error_count_per_sample} / {n_samples} "
          f"({high_error_count_per_sample/n_samples*100:.2f}%)")

    # Optionally show sample high-error cases
    if verbose and high_error_count_per_sample > 0:
        print(f"\nShowing 5 sample high-error cases:")
        high_err_sample_indices = np.where(np.any(high_error_mask, axis=1))[0][:5]

        for idx in high_err_sample_indices:
            print(f"\n  Sample #{idx}:")
            print(f"    Input (m, zeta, k, t, x0, v0): [{m_values[idx]:.3f}, {zeta_values[idx]:.3f}, "
                  f"{k_values[idx]:.3f}, {t_values[idx]:.6f}, {x0_values[idx]:.3f}, {v0_values[idx]:.3f}]")
            for j, name in enumerate(output_names):
                if high_error_mask[idx, j]:
                    print(f"    {name}: target={all_targets_normalized[idx, j]:.4f}, "
                          f"expected={normalized_analytical_array[idx, j]:.4f}, "
                          f"rel_err={rel_errors[idx, j]*100:.2f}%")

    # Plot input-output relationships if requested
    if plot_io_relation:
        if save_dir is None:
            save_dir = './IOrelation'

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

        input_names = ['m', 'zeta', 'k', 't', 'x0', 'v0']

        # Create plots (6 inputs × 3 outputs = 18 plots)
        for i, input_name in enumerate(input_names):
            for j, output_name in enumerate(output_names):
                plt.figure(figsize=(8, 6))

                input_data = sampled_inputs[:, i]
                output_data = sampled_targets[:, j]

                alpha = min(0.5, 5000.0 / n_plot_samples) if n_plot_samples > 0 else 0.5
                plt.scatter(input_data, output_data, alpha=alpha, s=2, c='blue', edgecolors='none')

                plt.xlabel(f'{input_name}', fontsize=12)
                plt.ylabel(f'{output_name} (normalized)', fontsize=12)
                plt.title(f'{dataset_name}: {input_name} vs {output_name}\n({n_plot_samples} samples)', fontsize=14)
                plt.grid(True, alpha=0.3)

                filename = f'{dataset_name.lower()}_{input_name}_vs_{output_name}.png'
                filepath = os.path.join(save_dir, filename)
                plt.savefig(filepath, dpi=100, bbox_inches='tight')
                plt.close()

        print(f"Saved 18 plots to {save_dir}")

    # ========================================================================
    # DERIVATIVE VERIFICATION (if requested) - NUMERICAL METHOD ONLY
    # ========================================================================
    deriv_results = None
    if verify_derivatives:
        print(f"\n{'='*70}")
        print(f"DERIVATIVE FORMULA VERIFICATION - NUMERICAL METHOD")
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

        # Initialize error storage lists (numerical only)
        errors_v_numerical = []
        errors_vprime_numerical = []
        errors_a_numerical = []
        errors_aprime_numerical = []

        # Store predictions for random sample display
        v_real_samples = []
        v_numerical_samples = []
        v_prime_real_samples = []
        v_prime_numerical_samples = []

        a_real_samples = []
        a_numerical_samples = []
        a_prime_real_samples = []
        a_prime_numerical_samples = []

        ln10 = np.log(10.0)  # Keep for future conversion to exp form
        deriv_invalid_count = 0

        # Loop through each sample
        for i in range(n_samples):
            # Skip if analytical solution was invalid
            if np.isnan(analytical_outputs[i, 0]):
                deriv_invalid_count += 1
                continue

            # Extract data
            m = m_values[i]
            zeta = zeta_values[i]
            k = k_values[i]
            t = t_values[i]
            x0 = x0_values[i]
            v0 = v0_values[i]
            c = 2 * zeta * np.sqrt(m * k)

            x_t = analytical_outputs[i, 0]
            v_t = analytical_outputs[i, 1]
            a_t = analytical_outputs[i, 2]

            # Get normalized ground truth from all_targets
            x_prime = all_targets[i, 3]
            v_prime = all_targets[i, 4]
            a_prime = all_targets[i, 5]

            # Compute t_prime
            t_prime = (np.log10(t) - mean_t) / std_t

            # Numerical dx'/dt' and dv'/dt' via finite differences
            t_prime_low = 0.9999 * t_prime
            t_prime_high = 1.0001 * t_prime

            # Convert back to real time
            t_low = np.exp((std_t * t_prime_low + mean_t) * ln10)
            t_high = np.exp((std_t * t_prime_high + mean_t) * ln10)

            # Get analytical solutions at perturbed times
            x_t_low, v_t_low, _ = analytical_solution(m, c, k, x0, v0, t_low)
            x_t_high, v_t_high, _ = analytical_solution(m, c, k, x0, v0, t_high)

            # Check for validity
            if any(np.isnan([x_t_low, x_t_high, v_t_low, v_t_high])) or \
               any(np.isinf([x_t_low, x_t_high, v_t_low, v_t_high])):
                deriv_invalid_count += 1
                continue

            # Normalize x and v at perturbed times
            eps_log = outputs_normalizer.eps
            x_prime_low = (np.log10(np.abs(x_t_low) + eps_log) - mean_x) / std_x
            x_prime_high = (np.log10(np.abs(x_t_high) + eps_log) - mean_x) / std_x
            v_prime_low = (np.log10(np.abs(v_t_low) + eps_log) - mean_v) / std_v
            v_prime_high = (np.log10(np.abs(v_t_high) + eps_log) - mean_v) / std_v

            # Finite differences (derivative with respect to normalized time t')
            dx_prime_dt_prime_numerical = (x_prime_high - x_prime_low) / (t_prime_high - t_prime_low)
            dv_prime_dt_prime_numerical = (v_prime_high - v_prime_low) / (t_prime_high - t_prime_low)

            # Velocity predictions (x -> v) using numerical derivative
            common_factor_v = (std_x / std_t) * (np.exp((std_x * x_prime + mean_x) * ln10) / t)
            v_numerical_method = np.abs(common_factor_v * dx_prime_dt_prime_numerical)

            # Acceleration predictions (v -> a) using numerical derivative
            common_factor_a = (std_v / std_t) * (np.exp((std_v * v_prime + mean_v) * ln10) / t)
            a_numerical_method = np.abs(common_factor_a * dv_prime_dt_prime_numerical)

            # Ground truths
            v_real_abs = np.abs(v_t)
            a_real_abs = np.abs(a_t)

            # Compute normalized v' and a' from predictions
            v_prime_numerical = (np.log10(v_numerical_method + eps_log) - mean_v) / std_v
            a_prime_numerical = (np.log10(a_numerical_method + eps_log) - mean_a) / std_a

            # Compute errors in real space
            error_v_numerical = v_numerical_method - v_real_abs
            error_a_numerical = a_numerical_method - a_real_abs

            # Errors in normalized space
            error_vprime_numerical = v_prime_numerical - v_prime
            error_aprime_numerical = a_prime_numerical - a_prime

            # Store all errors (relative)
            errors_v_numerical.append(error_v_numerical / v_real_abs if v_real_abs > 1e-15 else 0)
            errors_vprime_numerical.append(error_vprime_numerical / v_prime if abs(v_prime) > 1e-15 else 0)
            errors_a_numerical.append(error_a_numerical / a_real_abs if a_real_abs > 1e-15 else 0)
            errors_aprime_numerical.append(error_aprime_numerical / a_prime if abs(a_prime) > 1e-15 else 0)

            # Store actual values for random sample display
            v_real_samples.append(v_real_abs)
            v_numerical_samples.append(v_numerical_method)
            v_prime_real_samples.append(v_prime)
            v_prime_numerical_samples.append(v_prime_numerical)

            a_real_samples.append(a_real_abs)
            a_numerical_samples.append(a_numerical_method)
            a_prime_real_samples.append(a_prime)
            a_prime_numerical_samples.append(a_prime_numerical)

        # Convert lists to arrays
        errors_v_numerical = np.array(errors_v_numerical)
        errors_vprime_numerical = np.array(errors_vprime_numerical)
        errors_a_numerical = np.array(errors_a_numerical)
        errors_aprime_numerical = np.array(errors_aprime_numerical)

        valid_deriv_samples = len(errors_v_numerical)

        print(f"\nValid samples: {valid_deriv_samples}")
        print(f"Invalid samples (NaN/Inf): {deriv_invalid_count}")

        # VELOCITY VERIFICATION
        print(f"\n{'-'*70}")
        print(f"VELOCITY VERIFICATION (x -> v) - Numerical dx'/dt'")
        print(f"{'-'*70}")

        print(f"\nReal Space (v in units):")
        print(f"  Mean error:   {np.mean(errors_v_numerical):.6e}    Median: {np.median(errors_v_numerical):.6e}")
        print(f"  Max error:    {np.max(np.abs(errors_v_numerical)):.6e}    Std:    {np.std(errors_v_numerical):.6e}")

        print(f"\nNormalized Space (v'):")
        print(f"  Mean error:   {np.mean(errors_vprime_numerical):.6e}    Median: {np.median(errors_vprime_numerical):.6e}")
        print(f"  Max error:    {np.max(np.abs(errors_vprime_numerical)):.6e}    Std:    {np.std(errors_vprime_numerical):.6e}")

        # ACCELERATION VERIFICATION
        print(f"\n{'-'*70}")
        print(f"ACCELERATION VERIFICATION (v -> a) - Numerical dv'/dt'")
        print(f"{'-'*70}")

        print(f"\nReal Space (a in units):")
        print(f"  Mean error:   {np.mean(errors_a_numerical):.6e}    Median: {np.median(errors_a_numerical):.6e}")
        print(f"  Max error:    {np.max(np.abs(errors_a_numerical)):.6e}    Std:    {np.std(errors_a_numerical):.6e}")

        print(f"\nNormalized Space (a'):")
        print(f"  Mean error:   {np.mean(errors_aprime_numerical):.6e}    Median: {np.median(errors_aprime_numerical):.6e}")
        print(f"  Max error:    {np.max(np.abs(errors_aprime_numerical)):.6e}    Std:    {np.std(errors_aprime_numerical):.6e}")

        # Display random samples
        if valid_deriv_samples >= 5:
            random_indices = np.random.choice(valid_deriv_samples, size=5, replace=False)

            print(f"\nRandom 5 V samples [real, numerical]:")
            print(f"Real Abs Space (v in units):")
            for idx_i, idx in enumerate(random_indices, 1):
                print(f"{idx_i}. [{v_real_samples[idx]:.6e}, {v_numerical_samples[idx]:.6e}]")

            print(f"\nNormalized Space (v'):")
            for idx_i, idx in enumerate(random_indices, 1):
                print(f"{idx_i}. [{v_prime_real_samples[idx]:.6e}, {v_prime_numerical_samples[idx]:.6e}]")

            print(f"\nRandom 5 A samples [real, numerical]:")
            print(f"Real Abs Space (a in units):")
            for idx_i, idx in enumerate(random_indices, 1):
                print(f"{idx_i}. [{a_real_samples[idx]:.6e}, {a_numerical_samples[idx]:.6e}]")

            print(f"\nNormalized Space (a'):")
            for idx_i, idx in enumerate(random_indices, 1):
                print(f"{idx_i}. [{a_prime_real_samples[idx]:.6e}, {a_prime_numerical_samples[idx]:.6e}]")

        print(f"\n{'='*70}")

        # Store derivative verification results
        deriv_results = {
            'valid_samples': valid_deriv_samples,
            'invalid_samples': deriv_invalid_count,
            'velocity': {
                'v_errors': errors_v_numerical,
                'vprime_errors': errors_vprime_numerical,
                'mean_v_error': float(np.mean(errors_v_numerical)),
                'mean_vprime_error': float(np.mean(errors_vprime_numerical))
            },
            'acceleration': {
                'a_errors': errors_a_numerical,
                'aprime_errors': errors_aprime_numerical,
                'mean_a_error': float(np.mean(errors_a_numerical)),
                'mean_aprime_error': float(np.mean(errors_aprime_numerical))
            }
        }

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

    if verify_derivatives:
        result['derivative_verification'] = deriv_results

    return result


def check_all_datasets(train_loader, val_loader, test_loader,
                      inputs_normalizer, outputs_normalizer,
                      error_threshold=0.10, verbose=False,
                      plot_io_relation=False, save_dir=None, plot_sample_rate=1,
                      verify_derivatives=False):
    """
    Check consistency for all datasets (train, val, test).

    Args:
        train_loader: DataLoader for training set
        val_loader: DataLoader for validation set
        test_loader: DataLoader for test set
        inputs_normalizer: Fitted Vibration_DataNormalizer
        outputs_normalizer: Fitted Vibration_OutputNormalizer
        error_threshold: Relative error threshold (default 0.10 = 10%)
        verbose: If True, show detailed sample errors (default False)
        plot_io_relation: If True, plot input vs output scatter plots (default False)
        save_dir: Directory to save plots (default None)
        plot_sample_rate: Plot every Nth sample (default 1)
        verify_derivatives: If True, verify derivative formulas using numerical method (default False)

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
        plot_sample_rate=plot_sample_rate,
        verify_derivatives=verify_derivatives
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
        plot_sample_rate=plot_sample_rate,
        verify_derivatives=verify_derivatives
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
        plot_sample_rate=plot_sample_rate,
        verify_derivatives=verify_derivatives
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
    m * a_t + c * v_t + k * x_t = 0

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

    # Extract the array
    if isinstance(data, np.lib.npyio.NpzFile):
        array_name = list(data.keys())[0]
        data_array = data[array_name]
    else:
        data_array = data

    # Extract inputs and targets
    # inputs: [m, zeta, k, t, x0, v0]
    # targets: [x_t, v_t, a_t]
    inputs = data_array[:, :6]
    targets = data_array[:, 6:]

    m = inputs[:, 0]
    zeta = inputs[:, 1]
    k = inputs[:, 2]
    x_t = targets[:, 0]
    v_t = targets[:, 1]
    a_t = targets[:, 2]

    # Compute damping coefficient
    c = 2 * zeta * np.sqrt(m * k)

    # DIAGNOSTIC: Print raw data sample values
    print(f"\n[DIAGNOSTIC] Raw data sample values (first sample):")
    print(f"  m: {m[0]:.6e}, zeta: {zeta[0]:.6e}, k: {k[0]:.6e}")
    print(f"  c: {c[0]:.6e}, x_t: {x_t[0]:.6e}, v_t: {v_t[0]:.6e}, a_t: {a_t[0]:.6e}")

    # Physics residual: m * a_t + c * v_t + k * x_t = 0
    residual = m * a_t + c * v_t + k * x_t

    if use_relative:
        # Scale-invariant relative residual
        scale = np.abs(m * a_t) + np.abs(c * v_t) + np.abs(k * x_t) + 1e-10
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
    the original data after normalize→denormalize operations.

    Args:
        filepath: Path to .npz data file
        tolerance: Maximum acceptable relative error (default: 1e-3)
        batch_size: Batch size for DataLoader test (default: 1024)

    Returns:
        dict with diagnostic results
    """
    print("\n" + "="*80)
    print("EXAMINING NORMALIZER REVERSIBILITY")
    print("="*80)
    print(f"Testing normalization→denormalization pipeline on: {filepath}")
    print(f"Tolerance: {tolerance:.0e}")

    # Step 1: Load raw data
    data = np.load(filepath)
    if isinstance(data, np.lib.npyio.NpzFile):
        array_name = list(data.keys())[0]
        data_array = data[array_name]
    else:
        data_array = data

    print(f"Testing all {len(data_array)} samples")

    # Split inputs and outputs
    input_data = data_array[:, :6]   # [m, zeta, k, t, x0, v0]
    output_data = data_array[:, 6:]  # [x_t, v_t, a_t]

    # Step 2: Create and fit normalizers
    print("\n[FITTING NORMALIZERS]")
    inputs_normalizer = Vibration_DataNormalizer()
    outputs_normalizer = Vibration_OutputNormalizer(use_log_normalization=True)

    inputs_normalizer.fit({
        'm': input_data[:, 0],
        'zeta': input_data[:, 1],
        'k': input_data[:, 2],
        't': input_data[:, 3],
        'x0': input_data[:, 4],
        'v0': input_data[:, 5]
    })

    outputs_normalizer.fit({
        'x': output_data[:, 0],
        'v': output_data[:, 1],
        'a': output_data[:, 2]
    })

    # Print normalization statistics
    print("\n[INPUT NORMALIZER STATS]")
    print(f"  Log features (m, zeta, k, t):")
    for feat in ['m', 'zeta', 'k', 't']:
        print(f"    {feat}: log_mean={inputs_normalizer.log_mean[feat]:.6f}, log_std={inputs_normalizer.log_std[feat]:.6f}")
    print(f"  Linear features (x0, v0):")
    for feat in ['x0', 'v0']:
        print(f"    {feat}: mean={inputs_normalizer.linear_mean[feat]:.6f}, std={inputs_normalizer.linear_std[feat]:.6f}")

    print("\n[OUTPUT NORMALIZER STATS]")
    print(f"  Log features (x, v, a):")
    for feat in ['x', 'v', 'a']:
        print(f"    {feat}: log_mean={outputs_normalizer.log_mean[feat]:.6f}, log_std={outputs_normalizer.log_std[feat]:.6f}")

    # Step 3: Test inputs normalization reversibility
    print("\n[TESTING INPUTS REVERSIBILITY]")
    inputs_array_norm = inputs_normalizer.normalize_inputs(input_data)
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
        worst_idx = np.argmax(np.max(inputs_rel_error, axis=1))
        print(f"\n  Worst sample (index {worst_idx}):")
        print(f"    Original:      {input_data[worst_idx]}")
        print(f"    Reconstructed: {inputs_reconstructed[worst_idx]}")
    else:
        print(f"  ✓ PASSED: Inputs normalization is reversible")

    # Step 4: Test outputs normalization reversibility
    print("\n[TESTING OUTPUTS REVERSIBILITY]")
    outputs_array_norm = outputs_normalizer.normalize_outputs(output_data)
    outputs_reconstructed = outputs_normalizer.denormalize_outputs(outputs_array_norm)

    # Compute errors
    outputs_error = np.abs(output_data - outputs_reconstructed)
    outputs_rel_error = outputs_error / (np.abs(output_data) + 1e-10)
    outputs_max_error = np.max(outputs_rel_error)
    outputs_failed = np.sum(np.max(outputs_rel_error, axis=1) > tolerance)

    print(f"  Max relative error: {outputs_max_error:.6e}")
    print(f"  Failed samples (>{tolerance:.0e}): {outputs_failed}/{len(output_data)}")

    if outputs_max_error > tolerance:
        print(f"  ✗ FAILED: Outputs normalization is NOT reversible!")
        worst_idx = np.argmax(np.max(outputs_rel_error, axis=1))
        print(f"\n  Worst sample (index {worst_idx}):")
        print(f"    Original:      {output_data[worst_idx]}")
        print(f"    Reconstructed: {outputs_reconstructed[worst_idx]}")
    else:
        print(f"  ✓ PASSED: Outputs normalization is reversible")

    # Step 5: Test with DataLoader (tensor format)
    print("\n[TESTING DATALOADER REVERSIBILITY (TENSOR FORMAT)]")
    print(f"Creating DataLoader with batch_size={batch_size}, shuffle=False")

    dataset = VibrationDataset(inputs_array_norm, outputs_array_norm)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_inputs_from_loader = []
    all_outputs_from_loader = []

    for batch_inputs, batch_outputs in dataloader:
        batch_inputs_np = batch_inputs.numpy()
        batch_outputs_np = batch_outputs.numpy()

        batch_inputs_denorm = inputs_normalizer.denormalize_inputs(batch_inputs_np)
        batch_outputs_denorm = outputs_normalizer.denormalize_outputs(batch_outputs_np)

        all_inputs_from_loader.append(batch_inputs_denorm)
        all_outputs_from_loader.append(batch_outputs_denorm)

    inputs_from_loader = np.concatenate(all_inputs_from_loader, axis=0)
    outputs_from_loader = np.concatenate(all_outputs_from_loader, axis=0)

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
        else:
            print("✗ DATALOADER FAILED: DataLoader introduces errors")
    elif outputs_max_error > tolerance:
        print("✗ OUTPUTS FAILED: Outputs normalization has bugs")
    else:
        print("✗ INPUTS FAILED: Inputs normalization has bugs")
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
