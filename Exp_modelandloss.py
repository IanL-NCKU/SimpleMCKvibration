import torch
import torch.nn as nn
import numpy as np
from abc import ABC, abstractmethod


class ResidualBlock(nn.Module):
    """Residual block with two linear layers and skip connection

    Architecture: x → fc1 → act → BN → dropout → fc2 → BN → dropout → (+shortcut) → act
    """

    def __init__(self, in_dim, mid_dim, out_dim, activation, dropout=0.3):
        super().__init__()

        # First layer: in_dim → mid_dim
        self.fc1 = nn.Linear(in_dim, mid_dim)
        self.act1 = activation()
        self.bn1 = nn.BatchNorm1d(mid_dim)
        self.dropout1 = nn.Dropout(dropout)

        # Second layer: mid_dim → out_dim
        self.fc2 = nn.Linear(mid_dim, out_dim)
        self.bn2 = nn.BatchNorm1d(out_dim)
        self.dropout2 = nn.Dropout(dropout)

        # Shortcut connection: in_dim → out_dim
        if in_dim != out_dim:
            self.shortcut = nn.Linear(in_dim, out_dim)
        else:
            self.shortcut = nn.Identity()

        # Activation after residual addition
        self.act2 = activation()

    def forward(self, x):
        # Main path
        out = self.fc1(x)
        out = self.act1(out)
        out = self.bn1(out)
        out = self.dropout1(out)

        out = self.fc2(out)
        out = self.bn2(out)
        out = self.dropout2(out)

        # Shortcut path
        shortcut = self.shortcut(x)

        # Residual addition
        out = out + shortcut
        out = self.act2(out)

        return out


class ExponentialSignNN_ver3(nn.Module):
    """Binary Classification Network with Residual Connections

    Adds residual blocks to ExponentialSignNN_ver2 architecture.
    - If hidden_dims has <= 2 layers: No residual connections (simple sequential)
    - If hidden_dims has > 2 layers: Group layers in pairs for residual blocks

    Uses sigmoid outputs and BCE loss for sign classification.
    """

    def __init__(self, hidden_dims=[128, 64, 32], activation='relu', dropout=0.3):
        super().__init__()

        # Choose activation function
        if activation == 'tanh':
            act = nn.Tanh
        elif activation == 'elu':
            act = nn.ELU
        elif activation == 'relu':
            act = nn.ReLU
        else:
            act = nn.ReLU

        input_dim = 6  # [a, b, t, noisy_mag_x, noisy_mag_v, noisy_mag_a]

        # Decide whether to use residual blocks
        if len(hidden_dims) <= 2:
            # Simple sequential layers (no residual)
            self.use_residual = False
            layers = []
            for hidden_dim in hidden_dims:
                layers.append(nn.Linear(input_dim, hidden_dim))
                layers.append(act())
                layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.Dropout(dropout))
                input_dim = hidden_dim
            self.shared_layers = nn.Sequential(*layers)
        else:
            # Build residual blocks
            self.use_residual = True
            self.blocks = nn.ModuleList()

            # Pair up hidden_dims for residual blocks
            # Example: [128, 64, 32] → pairs: [(6,128,64)], remaining: [32]
            # Example: [128, 64, 32, 16] → pairs: [(6,128,64), (64,32,16)]
            dims = [input_dim] + hidden_dims
            i = 0
            while i + 2 < len(dims):
                # Create residual block for dims[i] → dims[i+1] → dims[i+2]
                block = ResidualBlock(dims[i], dims[i+1], dims[i+2], act, dropout)
                self.blocks.append(block)
                i += 2

            # Handle remaining layer if odd number
            if i + 1 < len(dims):
                # Add simple layer: dims[i] → dims[i+1]
                remaining_layer = nn.Sequential(
                    nn.Linear(dims[i], dims[i+1]),
                    act(),
                    nn.BatchNorm1d(dims[i+1]),
                    nn.Dropout(dropout)
                )
                self.blocks.append(remaining_layer)
                input_dim = dims[i+1]
            else:
                input_dim = dims[i]

        # Output heads (one for each of the 3 outputs)
        num_outputs = 3
        self.output_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, 16),
                act(),
                nn.Linear(16, 1),
                nn.Sigmoid()
            )
            for _ in range(num_outputs)
        ])

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, 6)
               [a, b, t, (1+0.05*r)*abs(x_t), (1+0.05*r)*abs(v_t), (1+0.05*r)*abs(a_t)]

        Returns:
            predictions: Output tensor of shape (batch_size, 3)
                        Reconstructed signed values: sign_from_probs * input_magnitudes
        """
        # Save original input to extract magnitudes later
        original_input = x

        # Shared feature extraction
        if self.use_residual:
            # Pass through residual blocks
            for block in self.blocks:
                x = block(x)
            shared_features = x
        else:
            # Pass through sequential layers
            shared_features = self.shared_layers(x)

        # Three independent binary classifiers (sigmoid outputs in [0, 1])
        outputs = [head(shared_features) for head in self.output_heads]

        # Concatenate to [batch_size, 3]
        probs = torch.cat(outputs, dim=1)

        # Store probabilities for loss computation
        self.last_sign_probs = probs

        # Convert probabilities to signs: [0, 1] -> [-1, 1]
        signs = 2 * probs - 1

        # Extract input magnitudes from original input (last 3 features)
        input_magnitudes = original_input[:, 3:]  # Shape: (batch_size, 3)


        predictions = signs #* input_magnitudes
        
        return predictions



class ExponentialPINN(nn.Module):
    """Physics-Informed Neural Network for exponential function: x(t) = b * exp(a*t)"""

    def __init__(self, hidden_dims=[64, 128, 128, 64], activation='tanh', use_log_output=False,
                 use_finetune=False, finetune_hidden_dims=[32, 32], finetune_scale=0.1,
                 use_sign_network=False, sign_network_hidden_dims=[32, 32]):
        super().__init__()

        self.use_log_output = use_log_output
        self.use_finetune = use_finetune
        self.finetune_scale = finetune_scale
        self.use_sign_network = use_sign_network

        # Choose activation function
        if activation == 'tanh':
            act = nn.Tanh
        elif activation == 'swish':
            act = nn.SiLU
        elif activation == 'ELU':
            act = nn.ELU
        elif activation == 'relu':
            act = nn.ReLU
        else:
            act = nn.GELU

        # Build main network
        layers = []
        input_dim = 3  # [a, b, t]

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(act())
            input_dim = hidden_dim

        # Final layer output dimension
        if use_log_output:
            layers.append(nn.Linear(input_dim, 6))  # [sign_x, log_x, sign_v, log_v, sign_a, log_a]
        else:
            layers.append(nn.Linear(input_dim, 3))  # [x_t, v_t, a_t]

        # Add softplus if sign network is enabled (to ensure positive outputs)
        # if self.use_sign_network:
        #     layers.append(nn.Softplus())
        layers.append(nn.Softplus())
        self.network = nn.Sequential(*layers)

        # Build fine-tune network (if enabled)
        if self.use_finetune:
            finetune_layers = []
            finetune_input_dim = 6  # [a, b, t, x_base, v_base, a_base]

            for hidden_dim in finetune_hidden_dims:
                finetune_layers.append(nn.Linear(finetune_input_dim, hidden_dim))
                finetune_layers.append(act())
                finetune_input_dim = hidden_dim

            finetune_layers.append(nn.Linear(finetune_input_dim, 3))
            finetune_layers.append(nn.Softplus())

            self.finetune_network = nn.Sequential(*finetune_layers)
        else:
            self.finetune_network = None
        
        # Build sign network (if enabled)
        if self.use_sign_network:
            sign_layers = []
            sign_input_dim = 6  # [a, b, t, x, v, a]

            for hidden_dim in sign_network_hidden_dims:
                sign_layers.append(nn.Linear(sign_input_dim, hidden_dim))
                sign_layers.append(act())
                sign_input_dim = hidden_dim

            sign_layers.append(nn.Linear(sign_input_dim, 3))  # [sign_x, sign_v, sign_a]
            sign_layers.append(nn.Tanh())  # Constrain to [-1, 1]

            self.sign_network = nn.Sequential(*sign_layers)
        else:
            self.sign_network = None

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, 3)
               [a, b, t]

        Returns:
            predictions: Output tensor of shape (batch_size, 3)
                        [x_t, v_t, a_t] in real space
        """
        output = self.network(x)

        # Step 1: Convert network output to real space (base predictions)
        if self.use_log_output:
            # Network outputs: [sign_x, log_x, sign_v, log_v, sign_a, log_a]

            # Extract sign and log magnitude
            sign_x = torch.tanh(output[:, 0:1])
            log_x = output[:, 1:2]
            sign_v = torch.tanh(output[:, 2:3])
            log_v = output[:, 3:4]
            sign_a = torch.tanh(output[:, 4:5])
            log_a = output[:, 5:6]

            # Transform to real space: x = sign * 10^log = sign * exp(log * ln(10))
            ln10 = np.log(10.0)  # Python float, PyTorch handles device automatically
            x_pred_base = sign_x * torch.exp(log_x * ln10)
            v_pred_base = sign_v * torch.exp(log_v * ln10)
            a_pred_base = sign_a * torch.exp(log_a * ln10)
        else:
            # Network outputs directly in real space: [x_t, v_t, a_t]
            x_pred_base = output[:, 0:1]
            v_pred_base = output[:, 1:2]
            a_pred_base = output[:, 2:3]

        # Step 2: Apply fine-tuning (independent of log_output setting)
        if self.use_finetune:
            # Ensure concatenation doesn't break gradient flow
            base_preds = torch.cat([x_pred_base, v_pred_base, a_pred_base], dim=1)
            finetune_input = torch.cat([x, base_preds], dim=1)

            finetune_raw = self.finetune_network(finetune_input)
            finetune_corrections = self.finetune_scale * finetune_raw

            # Apply multiplicative correction
            x_pred = x_pred_base * (1 + finetune_corrections[:, 0:1])
            v_pred = v_pred_base * (1 + finetune_corrections[:, 1:2])
            a_pred = a_pred_base * (1 + finetune_corrections[:, 2:3])
        else:
            x_pred = x_pred_base
            v_pred = v_pred_base
            a_pred = a_pred_base

        # Step 3: Apply sign network (if enabled) - FINAL STEP
        if self.use_sign_network:
            # Prepare input: concatenate [a, b, t] + current predictions
            current_preds = torch.cat([x_pred, v_pred, a_pred], dim=1)
            sign_input = torch.cat([x, current_preds], dim=1)

            # Get sign predictions from network
            sign_output = self.sign_network(sign_input)
            predicted_signs = sign_output  # Keep soft signs in [-1, 1] for gradient flow

            # Apply sign corrections: predictions are already positive (from softplus)
            # So we can directly multiply by predicted signs
            x_pred = x_pred * predicted_signs[:, 0:1]
            v_pred = v_pred * predicted_signs[:, 1:2]
            a_pred = a_pred * predicted_signs[:, 2:3]

        return torch.cat([x_pred, v_pred, a_pred], dim=1)


class ExponentialPINN_ver2(nn.Module):
    """Physics-Informed Neural Network for exponential function: x(t) = b * exp(a*t)"""

    def __init__(self, hidden_dims=[64, 128, 128, 64], activation='tanh', use_log_output=False,
                 use_finetune=False, finetune_hidden_dims=[32, 32], finetune_scale=0.1,
                 sign_network_hidden_dims=[128, 64, 32], sign_network_dropout=0.3):
        super().__init__()

        self.use_log_output = use_log_output
        self.use_finetune = use_finetune
        self.finetune_scale = finetune_scale
        self.last_sign_probs = None

        # Choose activation function
        if activation == 'tanh':
            act = nn.Tanh
        elif activation == 'swish':
            act = nn.SiLU
        elif activation == 'ELU':
            act = nn.ELU
        elif activation == 'relu':
            act = nn.ReLU
        else:
            act = nn.GELU

        # Build main network
        layers = []
        input_dim = 3  # [a, b, t]

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(act())
            input_dim = hidden_dim

        # Final layer output dimension
        if use_log_output:
            layers.append(nn.Linear(input_dim, 6))  # [sign_x, log_x, sign_v, log_v, sign_a, log_a]
        else:
            layers.append(nn.Linear(input_dim, 3))  # [x_t, v_t, a_t]

        # Add softplus to ensure positive outputs (magnitudes)
        layers.append(nn.Softplus())
        self.network = nn.Sequential(*layers)

        # Build fine-tune network (if enabled)
        if self.use_finetune:
            finetune_layers = []
            finetune_input_dim = 6  # [a, b, t, x_base, v_base, a_base]

            for hidden_dim in finetune_hidden_dims:
                finetune_layers.append(nn.Linear(finetune_input_dim, hidden_dim))
                finetune_layers.append(act())
                finetune_input_dim = hidden_dim

            finetune_layers.append(nn.Linear(finetune_input_dim, 3))
            finetune_layers.append(nn.Softplus())

            self.finetune_network = nn.Sequential(*finetune_layers)
        else:
            self.finetune_network = None

        # Build sign network (always enabled with ExponentialSignNN_ver3)
        self.sign_network = ExponentialSignNN_ver3(
            hidden_dims=sign_network_hidden_dims,
            activation=activation,
            dropout=sign_network_dropout
        )

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, 3)
               [a, b, t]

        Returns:
            predictions: Output tensor of shape (batch_size, 3)
                        [x_t, v_t, a_t] in real space
        """
        output = self.network(x)

        # Step 1: Convert network output to real space (base predictions)
        if self.use_log_output:
            # Network outputs: [sign_x, log_x, sign_v, log_v, sign_a, log_a]

            # Extract sign and log magnitude
            sign_x = torch.tanh(output[:, 0:1])
            log_x = output[:, 1:2]
            sign_v = torch.tanh(output[:, 2:3])
            log_v = output[:, 3:4]
            sign_a = torch.tanh(output[:, 4:5])
            log_a = output[:, 5:6]

            # Transform to real space: x = sign * 10^log = sign * exp(log * ln(10))
            ln10 = np.log(10.0)  # Python float, PyTorch handles device automatically
            x_pred_base = sign_x * torch.exp(log_x * ln10)
            v_pred_base = sign_v * torch.exp(log_v * ln10)
            a_pred_base = sign_a * torch.exp(log_a * ln10)
        else:
            # Network outputs directly in real space: [x_t, v_t, a_t]
            x_pred_base = output[:, 0:1]
            v_pred_base = output[:, 1:2]
            a_pred_base = output[:, 2:3]

        # Step 2: Apply fine-tuning (independent of log_output setting)
        if self.use_finetune:
            # Ensure concatenation doesn't break gradient flow
            base_preds = torch.cat([x_pred_base, v_pred_base, a_pred_base], dim=1)
            finetune_input = torch.cat([x, base_preds], dim=1)

            finetune_raw = self.finetune_network(finetune_input)
            finetune_corrections = self.finetune_scale * finetune_raw

            # Apply multiplicative correction
            x_pred = x_pred_base * (1 + finetune_corrections[:, 0:1])
            v_pred = v_pred_base * (1 + finetune_corrections[:, 1:2])
            a_pred = a_pred_base * (1 + finetune_corrections[:, 2:3])
        else:
            x_pred = x_pred_base
            v_pred = v_pred_base
            a_pred = a_pred_base

        # Step 3: Apply sign network - FINAL STEP (always enabled)
        # Prepare input: concatenate [a, b, t] + current magnitude predictions
        current_preds = torch.cat([x_pred, v_pred, a_pred], dim=1)
        sign_input = torch.cat([x, current_preds.detach()], dim=1)  # Shape: [batch, 6]

        # Get signed predictions from ExponentialSignNN_ver3
        signed_output =current_preds *self.sign_network(sign_input)
        # signed_output =current_preds #*self.sign_network(sign_input)
        # signed_output = self.sign_network(sign_input)

        # Store sigmoid probabilities for BCE loss computation
        self.last_sign_probs = self.sign_network.last_sign_probs

        return signed_output



class BaseLossComponent(ABC, nn.Module):
    """Abstract base class for loss components"""

    def __init__(self, weight=1.0, name="base"):
        super().__init__()
        self.weight = weight
        self.name = name
        self.enabled = weight > 0

    @abstractmethod
    def compute(self, predictions, targets, inputs, norm_params=None, inputs_real=None):
        pass

    def forward(self, predictions, targets, inputs, norm_params=None, inputs_real=None):
        if not self.enabled:
            return torch.tensor(0.0, device=predictions.device)
        loss = self.compute(predictions, targets, inputs, norm_params, inputs_real)
        return self.weight * loss

    def __repr__(self):
        status = "✓" if self.enabled else "✗"
        return f"{self.name:20s}: weight={self.weight:.3f} {status}"


class MSELoss(BaseLossComponent):
    """Mean Squared Error loss between predictions and targets"""

    def __init__(self, weight=1.0, use_relative=False, use_log=False, sign_bce_weight=1.0):
        super().__init__(weight=weight, name="MSE Loss")
        self.use_relative = use_relative
        self.use_log = use_log
        # Create SignBCELoss instance for sign loss computation
        self.sign_bce_loss = SignBCELoss(weight=sign_bce_weight) if sign_bce_weight > 0 else None

    def compute(self, predictions, targets, inputs, norm_params=None, inputs_real=None):
        """
        Compute MSE between predictions and targets

        Args:
            predictions: (batch_size, 3) - [x_t, v_t, a_t] predictions in real space
            targets: (batch_size, 3) - [x_t, v_t, a_t] targets in real space

        Returns:
            loss: Scalar tensor
        """
        eps = 1e-10

        if self.use_log:
            # Compute magnitude loss (log-space MSE)
            log_predictions = torch.log(torch.abs(predictions) + eps)
            log_targets = torch.log(torch.abs(targets) + eps)

            if self.use_relative:
                # Relative log-space MSE for magnitudes
                magnitude_loss = torch.mean(((log_predictions - log_targets) ** 2) / (torch.square(log_targets) + eps))
            else:
                # Absolute log-space MSE for magnitudes
                magnitude_loss = torch.mean((log_predictions - log_targets) ** 2)

            # Compute sign BCE loss using SignBCELoss class
            sign_bce_value = 0.0
            if self.sign_bce_loss is not None and inputs is not None:
                # Use SignBCELoss class to compute the loss
                sign_bce_value = self.sign_bce_loss(predictions, targets, inputs, norm_params, inputs_real)

            # Combine magnitude and sign losses
            loss =  sign_bce_value  +magnitude_loss
        else:
            # Standard MSE (not log-space)
            if self.use_relative:
                # Relative MSE: normalized by target magnitude
                loss = torch.mean(((predictions - targets) ** 2) / (torch.square(targets) + eps))
            else:
                # Absolute MSE
                loss = torch.mean((predictions - targets) ** 2)

        return loss


class SignBCELoss(BaseLossComponent):
    """Sign BCE loss - uses binary cross entropy for sign classification"""

    def __init__(self, weight=1.0):
        super().__init__(weight=weight, name="Sign BCE Loss")
        self.criterion = nn.BCELoss()

    def compute(self, predictions, targets, inputs, norm_params=None, inputs_real=None):
        """
        Compute sign BCE between sigmoid probabilities and target signs

        Args:
            predictions: Not used (placeholder for signature compatibility)
            targets: (batch_size, num_outputs) - Target signed values
            inputs: (batch_size, num_outputs) - Sigmoid probabilities from model.last_sign_probs

        Returns:
            loss: Scalar tensor - Sum of BCE losses for all outputs
        """
        # Get sigmoid probabilities (stored by model during forward pass)
        sigmoid_probs = inputs  # Shape: [batch, num_outputs], values in [0, 1]

        # Convert target signs to binary labels: {-1, +1} -> {0, 1}
        # Zeros are treated as positive (label=1.0)
        target_signs = torch.sign(targets).float()  # Shape: [batch, num_outputs], values: -1, 0, +1
        labels = (target_signs >= 0).float()  # Shape: [batch, num_outputs], values: 0 or 1

        # Compute BCE for each output and sum
        num_outputs = sigmoid_probs.shape[1]
        total_bce_loss = sum(
            self.criterion(sigmoid_probs[:, i], labels[:, i])
            for i in range(num_outputs)
        )

        return total_bce_loss


class ExponentialResidualLoss(BaseLossComponent):
    """Physics residual loss for exponential function

    For x(t) = b*exp(a*t), the derivatives are:
    - v(t) = dx/dt = b*a*exp(a*t)
    - a_t = dv/dt = b*a²*exp(a*t)

    The physics equation is:
    (1/(2a))*a_t + 0.5*v_t - a*x_t = 0

    Rearranging:
    residual = (1/(2a))*a_t + 0.5*v_t - a*x_t
    """

    def __init__(self, weight=1.0, use_relative=False):
        super().__init__(weight=weight, name="Exponential Residual Loss")
        self.use_relative = use_relative

    def compute(self, predictions, targets, inputs, norm_params=None, inputs_real=None):
        """
        Enforce exponential physics equation:
        (1/(2a))*a_t + 0.5*v_t - a*x_t = 0

        Args:
            predictions: (batch_size, 3) - [x_t, v_t, a_t] in real space
            inputs_real: (batch_size, 3) - [a, b, t] in real space
        """
        if inputs_real is None:
            raise ValueError("ExponentialResidualLoss requires inputs_real to be provided")

        x_pred = predictions[:, 0]
        v_pred = predictions[:, 1]
        a_pred = predictions[:, 2]

        # Extract parameters
        a = inputs_real[:, 0]  # exponential rate

        eps = 1e-10

        # Physics residual: (1/(2a))*a_t + 0.5*v_t - a*x_t = 0
        residual = (1.0 / (2.0 * a + eps)) * a_pred + 0.5 * v_pred - a * x_pred

        if self.use_relative:
            # Scale-invariant relative residual
            scale = torch.abs(a * x_pred) + eps
            residual = residual / scale

        return torch.mean(residual ** 2)


class ConsistencyLoss_auto_diff(BaseLossComponent):
    """
    Derivative consistency: ensure v=dx/dt, a=dv/dt using automatic differentiation

    Handles log-normalized time transformation:
    t_model = (log10(t_real) - mean) / std

    Chain rule for derivatives:
    dx/dt_real = dx/dt_model * dt_model/dt_real
    where dt_model/dt_real = 1 / (std * t_real * ln(10))
    """

    def __init__(self, weight=1.0, model=None, t_threshold=1e-6, use_log=True):
        super().__init__(weight=weight, name="Consistency Loss (auto)")
        self.model = model
        self.t_threshold = t_threshold
        self.use_log = use_log

    def set_model(self, model):
        """Set the model reference for gradient computation"""
        self.model = model

    def compute(self, predictions, targets, inputs, norm_params=None, inputs_real=None):
        """
        Enforces consistency between position, velocity, and acceleration
        using automatic differentiation with log-normalized time.

        Args:
            inputs: (batch_size, 3) - [a, b, t] NORMALIZED
            norm_params: Dictionary with 'normalizer' key containing normalizer instance
            inputs_real: (batch_size, 3) - [a, b, t] in REAL space

        Returns:
            loss: Scalar tensor measuring consistency error
        """
        if self.model is None:
            raise ValueError("Model not set. Call set_model() before using ConsistencyLoss_auto_diff")

        if inputs_real is None:
            raise ValueError("ConsistencyLoss_auto_diff requires inputs_real to be provided")

        if norm_params is None or 'normalizer' not in norm_params:
            raise ValueError("norm_params with 'normalizer' must be provided for ConsistencyLoss_auto_diff")

        normalizer = norm_params['normalizer']

        # Get t_real from inputs_real
        t_real = inputs_real[:, 2]  # Note: index 2 for exponential data (not 3)

        # Filter out samples where t_real ≈ 0 (to avoid division by zero)
        valid_mask = t_real > self.t_threshold

        if valid_mask.sum() == 0:
            # No valid samples (all t≈0)
            return torch.tensor(0.0, device=inputs.device)

        # Filter to valid samples only
        inputs_valid = inputs[valid_mask]
        t_real_valid = t_real[valid_mask]

        # Get the time std from normalizer (or use 1.0 if no normalization)
        if normalizer is not None:
            t_std = normalizer.log_std['t']  # std of log10(t) values
        else:
            # No normalization: dt_model/dt_real = 1
            # chain_rule_factor = 1 / (t_std * t_real * ln(10))
            # To make chain_rule_factor = 1, we need: t_std * t_real * ln(10) = 1
            # So: t_std = 1 / (t_real * ln(10))
            # But t_std should be a scalar, so we just set it to make chain_rule = 1
            # Actually, easier approach: we'll handle this in chain_rule_factor calculation
            t_std = torch.tensor(1.0, device=inputs.device, dtype=inputs.dtype)

        # Don't use .detach() to preserve gradient linkage
        inputs_with_grad = inputs_valid.clone().requires_grad_(True)

        # Forward pass with gradient tracking
        predictions_with_grad = self.model(inputs_with_grad)

        # Extract predictions (these are in REAL space)
        x_pred = predictions_with_grad[:, 0]
        v_pred = predictions_with_grad[:, 1]
        a_pred = predictions_with_grad[:, 2]

        # Compute dx/dt_model using autograd (gradient w.r.t. t_normalized at index 2)
        dx_dt_model = torch.autograd.grad(
            outputs=x_pred,
            inputs=inputs_with_grad,
            grad_outputs=torch.ones_like(x_pred),
            create_graph=True,
            retain_graph=True,
            allow_unused=True
        )[0][:, 2]  # Take only the gradient w.r.t. t_normalized (index 2)

        # Compute dv/dt_model using autograd
        dv_dt_model = torch.autograd.grad(
            outputs=v_pred,
            inputs=inputs_with_grad,
            grad_outputs=torch.ones_like(v_pred),
            create_graph=True,
            retain_graph=True,
            allow_unused=True
        )[0][:, 2]  # Take only the gradient w.r.t. t_normalized (index 2)

        # Apply chain rule to transform from model domain to real domain
        if normalizer is not None:
            # With normalization: t_model = (log10(t_real) - mean) / std
            # dt_model/dt_real = 1 / (std * t_real * ln(10))
            ln10 = torch.tensor(np.log(10), device=inputs.device, dtype=inputs.dtype)
            chain_rule_factor = 1.0 / (t_std * t_real_valid * ln10)
        else:
            # Without normalization: t_model = t_real, so dt_model/dt_real = 1
            chain_rule_factor = torch.tensor(1.0, device=inputs.device, dtype=inputs.dtype)

        # Transform gradients to real domain
        dx_dt_real = dx_dt_model * chain_rule_factor
        dv_dt_real = dv_dt_model * chain_rule_factor

        # Compute consistency losses
        # v_pred should equal dx/dt_real
        loss_v_consistency = torch.mean((v_pred - dx_dt_real) ** 2)

        # a_pred should equal dv/dt_real
        loss_a_consistency = torch.mean((a_pred - dv_dt_real) ** 2)

        # Total consistency loss
        total_loss = loss_v_consistency + loss_a_consistency

        # Apply log transformation if enabled
        if self.use_log:
            eps = 1e-10
            total_loss = torch.log(total_loss + eps)

        return total_loss


class ConsistencyLoss_finite_diff(BaseLossComponent):
    """
    Derivative consistency using finite differences

    Computes 6 loss components:
    1. v from x (finite diff) vs v_pred
    2. a from v (finite diff) vs a_pred
    3. a from x (finite diff, 2nd derivative) vs a_pred
    4. v from x (finite diff) vs v_target
    5. a from v (finite diff) vs a_target
    6. a from x (finite diff, 2nd derivative) vs a_target
    """

    def __init__(self, weight=1.0, use_relative=False, t_threshold=1e-6,
                 use_log=True,
                 weight_v_x_pred=1.0,
                 weight_a_v_pred=1.0,
                 weight_a_x_pred=1.0,
                 weight_v_x_target=1.0,
                 weight_a_v_target=1.0,
                 weight_a_x_target=1.0):
        super().__init__(weight=weight, name="Consistency Loss (finite)")
        self.use_relative = use_relative
        self.t_threshold = t_threshold
        self.use_log = use_log

        # Individual component weights
        self.weight_v_x_pred = weight_v_x_pred
        self.weight_a_v_pred = weight_a_v_pred
        self.weight_a_x_pred = weight_a_x_pred
        self.weight_v_x_target = weight_v_x_target
        self.weight_a_v_target = weight_a_v_target
        self.weight_a_x_target = weight_a_x_target

    def compute(self, predictions, targets, inputs, norm_params=None, inputs_real=None):
        """
        Compute finite difference consistency loss

        Args:
            predictions: (batch_size, 3) - [x, v, a] at time t
            targets: (batch_size, 3) - [x, v, a] ground truth at time t
            inputs: Not used
            norm_params: Not used
            inputs_real: (4*batch_size, 3) - predictions at perturbed times
                        [t-2Δt, t-Δt, t+Δt, t+2Δt] stacked

        Returns:
            total_loss: Scalar tensor representing the natural log of the total loss.
        """
        if inputs_real is None:
            raise ValueError("ConsistencyLoss_finite_diff requires inputs_real (outputs_dt) to be provided")

        N = predictions.shape[0]
        eps = 1e-10

        # Extract predictions at different time points
        # inputs_real is actually outputs_dt with shape (4N, 3)
        outputs_dt = inputs_real

        x_t = predictions[:, 0:1]
        v_t = predictions[:, 1:2]
        a_t = predictions[:, 2:3]

        x_t_minus_minus = outputs_dt[0:N, 0:1]      # t - 2Δt
        x_t_minus = outputs_dt[N:2*N, 0:1]          # t - Δt
        x_t_plus = outputs_dt[2*N:3*N, 0:1]         # t + Δt
        x_t_plus_plus = outputs_dt[3*N:4*N, 0:1]    # t + 2Δt

        v_t_minus = outputs_dt[N:2*N, 1:2]          # t - Δt
        v_t_plus = outputs_dt[2*N:3*N, 1:2]         # t + Δt

        # Extract targets
        # Debug: ensure targets is a tensor
        if not isinstance(targets, torch.Tensor):
            raise TypeError(f"targets must be a torch.Tensor, got {type(targets)}")

        x_target = targets[:, 0:1]
        v_target = targets[:, 1:2]
        a_target = targets[:, 2:3]

        t_delta = self.t_threshold

        # Compute finite difference derivatives
        # Component 1 & 4: v from x using central difference
        v_fd_from_x = (x_t_plus - x_t_minus) / (2 * t_delta)

        # Component 2 & 5: a from v using central difference
        a_fd_from_v = (v_t_plus - v_t_minus) / (2 * t_delta)

        # Component 3 & 6: a from x using second derivative
        v_t_plus_temp = (x_t_plus_plus - x_t) / (2 * t_delta)
        v_t_minus_temp = (x_t - x_t_minus_minus) / (2 * t_delta)
        a_fd_from_x = (v_t_plus_temp - v_t_minus_temp) / (2 * t_delta)

        # Compute squared differences for all 6 components
        diff_1 = (v_fd_from_x - v_t) ** 2        # v from x vs v_pred
        diff_2 = (a_fd_from_v - a_t) ** 2        # a from v vs a_pred
        diff_3 = (a_fd_from_x - a_t) ** 2        # a from x vs a_pred
        diff_4 = (v_fd_from_x - v_target) ** 2   # v from x vs v_target
        diff_5 = (a_fd_from_v - a_target) ** 2   # a from v vs a_target
        diff_6 = (a_fd_from_x - a_target) ** 2   # a from x vs a_target

        # Apply normalization if use_relative is True
        if self.use_relative:
            diff_1 = diff_1 / (v_target ** 2 + eps)
            diff_2 = diff_2 / (a_target ** 2 + eps)
            diff_3 = diff_3 / (a_target ** 2 + eps)
            diff_4 = diff_4 / (v_target ** 2 + eps)
            diff_5 = diff_5 / (a_target ** 2 + eps)
            diff_6 = diff_6 / (a_target ** 2 + eps)

        # Compute weighted sum of all components
        total_loss = (
            self.weight_v_x_pred * torch.mean(diff_1) +
            self.weight_a_v_pred * torch.mean(diff_2) +
            self.weight_a_x_pred * torch.mean(diff_3) +
            self.weight_v_x_target * torch.mean(diff_4) +
            self.weight_a_v_target * torch.mean(diff_5) +
            self.weight_a_x_target * torch.mean(diff_6)
        )

        # Apply log transformation if enabled
        if self.use_log:
            total_loss = torch.log(total_loss + eps)

        return total_loss


class ExponentialPINNLoss(nn.Module):
    """
    Dictionary-based PINN Loss for exponential data

    Usage:
        # Configuration
        loss_config = {
            "MSE": {"weight": 0.8, "use_relative": True},
            "Residual": {"weight": 0.1, "use_relative": False},
            "Consistency": {"weight": 0.1, "t_threshold": 1e-6, "use_log": True}
        }

        # Create loss function
        loss_fn = ExponentialPINNLoss(model, loss_config)

        # In training loop, prepare arguments for enabled losses
        loss_args = {}
        if loss_fn.has_loss("MSE"):
            loss_args["MSE"] = (outputs, targets)
        if loss_fn.has_loss("Residual"):
            loss_args["Residual"] = (outputs, inputs_real)
        if loss_fn.has_loss("Consistency"):
            loss_args["Consistency"] = (inputs, inputs_real, norm_params)

        # Compute loss
        total_loss, loss_summary = loss_fn(loss_args)
    """

    def __init__(self, model, loss_config):
        super().__init__()

        self.model = model
        self.loss_config = loss_config
        self.loss_components = {}

        # Initialize MSE loss if requested
        if self._should_enable("MSE"):
            config = self.loss_config.get("MSE")
            use_relative = config.get("use_relative", False)
            use_log = config.get("use_log", False)
            weight = config.get("weight", 1.0)
            sign_bce_weight = config.get("sign_bce_weight", 1.0)
            self.loss_components["MSE"] = MSELoss(weight=weight, use_relative=use_relative, use_log=use_log, sign_bce_weight=sign_bce_weight)

        # Initialize Residual loss if requested
        if self._should_enable("Residual"):
            config = self.loss_config.get("Residual")
            use_relative = config.get("use_relative", False)
            weight = config.get("weight", 1.0)
            self.loss_components["Residual"] = ExponentialResidualLoss(weight=weight, use_relative=use_relative)

        # Initialize Consistency loss if requested
        if self._should_enable("Consistency"):
            config = self.loss_config.get("Consistency")
            t_threshold = config.get("t_threshold", 1e-6)
            weight = config.get("weight", 1.0)
            use_relative = config.get("use_relative", False)
            use_log = config.get("use_log", True)
            consistency_type = config.get("type", "auto")  # "auto" or "finite"

            if consistency_type == "auto":
                self.loss_components["Consistency"] = ConsistencyLoss_auto_diff(
                    weight=weight, model=model, t_threshold=t_threshold, use_log=use_log
                )
            elif consistency_type == "finite":
                self.loss_components["Consistency"] = ConsistencyLoss_finite_diff(
                    weight=weight, use_relative=use_relative, t_threshold=t_threshold, use_log=use_log
                )
            else:
                raise ValueError(f"Unknown consistency type: {consistency_type}. Use 'auto' or 'finite'.")

        # Print configuration
        self._print_config()

    def _should_enable(self, loss_name):
        """Check if a loss should be enabled"""
        config = self.loss_config.get(loss_name, None)
        if config is None:
            return False
        if config.get("weight", 0) == 0:
            return False
        return True

    def has_loss(self, loss_name):
        """Check if a loss component is enabled"""
        return loss_name in self.loss_components

    def forward(self, loss_args):
        """
        Compute total loss from arguments dictionary

        Args:
            loss_args: Dictionary with loss arguments
                {
                    "MSE": (outputs, targets),
                    "Residual": (outputs, inputs_real),
                    "Consistency": (inputs, inputs_real, norm_params)
                }

        Returns:
            total_loss: Scalar tensor
            loss_summary: Dictionary with individual loss values
        """
        total_loss = 0.0
        loss_summary = {}

        # MSE Loss
        if "MSE" in self.loss_components and "MSE" in loss_args:
            outputs, targets = loss_args["MSE"]
            # Get sigmoid probabilities from model for sign BCE loss
            sigmoid_probs = self.model.last_sign_probs if hasattr(self.model, 'last_sign_probs') else None
            mse_value = self.loss_components["MSE"](
                outputs, targets, sigmoid_probs, None, None
            )
            total_loss += mse_value
            loss_summary["mse_loss"] = mse_value.item()

        # Residual Loss
        if "Residual" in self.loss_components and "Residual" in loss_args:
            outputs, inputs_real = loss_args["Residual"]
            residual_value = self.loss_components["Residual"](
                outputs, None, None, None, inputs_real
            )
            total_loss += residual_value
            loss_summary["residual_loss"] = residual_value.item()

        # Consistency Loss
        if "Consistency" in self.loss_components and "Consistency" in loss_args:
            # Check if it's auto or finite based on number of arguments
            consistency_args = loss_args["Consistency"]

            if isinstance(self.loss_components["Consistency"], ConsistencyLoss_auto_diff):
                # Auto diff: (inputs, inputs_real, norm_params)
                inputs, inputs_real, norm_params = consistency_args
                consistency_value = self.loss_components["Consistency"](
                    None, None, inputs, norm_params, inputs_real
                )
            elif isinstance(self.loss_components["Consistency"], ConsistencyLoss_finite_diff):
                # Finite diff: (outputs, outputs_dt, targets)
                outputs, outputs_dt, targets = consistency_args
                consistency_value = self.loss_components["Consistency"](
                    outputs, targets, None, None, outputs_dt
                )
            else:
                raise ValueError("Unknown consistency loss type")

            total_loss += consistency_value
            loss_summary["consistency_loss"] = consistency_value.item()

        loss_summary["total"] = total_loss.item()

        return total_loss, loss_summary

    def _print_config(self):
        """Print loss configuration"""
        print(f"\n{'='*60}")
        print("Exponential PINN Loss Configuration:")
        print(f"{'='*60}")

        for loss_name in ["MSE", "Residual", "Consistency"]:
            if loss_name in self.loss_components:
                weight = self.loss_config.get(loss_name).get("weight", 1.0)
                print(f"  ✓ {loss_name:20s}: weight={weight:.3f}")
            else:
                print(f"  ✗ {loss_name:20s}: disabled")

        print(f"{'='*60}\n")
