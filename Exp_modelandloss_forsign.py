"""
Model and Loss Components for Sign Classification Network

This file contains ONLY classes for sign prediction:
- ExponentialSignNN: Network that predicts signs from noisy magnitude inputs
- SignMSELoss: Loss function for sign prediction
- BaseLossComponent: Abstract base class for loss components
- ExponentialPINNLoss: Loss wrapper (only supports SignMSE)

REMOVED CLASSES (use Exp_modelandloss.py instead):
- ExponentialPINN
- MSELoss
- ExponentialResidualLoss
- ConsistencyLoss_auto_diff
- ConsistencyLoss_finite_diff
"""

import torch
import torch.nn as nn
import numpy as np
from abc import ABC, abstractmethod


class ExponentialSignNN(nn.Module):
    """Sign Classification Network for exponential function

    Learns to predict signs of [x_t, v_t, a_t] given:
    - Input: [a, b, t, (1+0.05*r)*abs(x_t), (1+0.05*r)*abs(v_t), (1+0.05*r)*abs(a_t)]
    - Output: sign predictions for [x_t, v_t, a_t]
    """

    def __init__(self, hidden_dims=[64, 128, 128, 64], activation='tanh'):
        super().__init__()

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

        # Build sign prediction network
        layers = []
        input_dim = 6  # [a, b, t, noisy_mag_x, noisy_mag_v, noisy_mag_a]

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            # layers.append(act())
            input_dim = hidden_dim

        # Final layer: output 3 sign values constrained to [-1, 1]
        layers.append(nn.Linear(input_dim, 3))
        layers.append(nn.Tanh())  # Constrain to [-1, 1]

        self.network = nn.Sequential(*layers)

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
                        Reconstructed signed values: sign_predictions * input_magnitudes
                        where input_magnitudes are the last 3 inputs
        """
        # Get sign predictions from network (Tanh output: [-1, 1])
        sign_predictions = self.network(x)  # Shape: (batch_size, 3)

        # Extract input magnitudes (last 3 features)
        input_magnitudes = x[:, 3:]  # Shape: (batch_size, 3)

        # Reconstruct signed values: sign * magnitude
        predictions = sign_predictions * input_magnitudes

        return predictions


class ExponentialSignNN_ver2(nn.Module):
    """Binary Classification Network for sign prediction using multi-head architecture

    Uses sigmoid outputs and BCE loss for sign classification.
    Learns to predict signs of [x_t, v_t, a_t] given:
    - Input: [a, b, t, (1+0.05*r)*abs(x_t), (1+0.05*r)*abs(v_t), (1+0.05*r)*abs(a_t)]
    - Output: 3 binary predictions (positive/negative) for [x_t, v_t, a_t]
    """

    def __init__(self, hidden_dims=[128, 64, 32], activation='relu'):
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

        # Build shared hidden layers
        layers = []
        input_dim = 6  # [a, b, t, noisy_mag_x, noisy_mag_v, noisy_mag_a]

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(act())
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.Dropout(0.3))
            input_dim = hidden_dim

        self.shared_layers = nn.Sequential(*layers)

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
        # Shared feature extraction
        shared_features = self.shared_layers(x)

        # Three independent binary classifiers (sigmoid outputs in [0, 1])
        outputs = [head(shared_features) for head in self.output_heads]

        # Concatenate to [batch_size, 3]
        probs = torch.cat(outputs, dim=1)

        # Store probabilities for loss computation
        self.last_sign_probs = probs

        # Convert probabilities to signs: [0, 1] -> [-1, 1]
        signs = 2 * probs - 1

        # Extract input magnitudes (last 3 features)
        input_magnitudes = x[:, 3:]  # Shape: (batch_size, 3)

        # Reconstruct signed values: sign * magnitude
        predictions = signs * input_magnitudes

        return predictions


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

        # Reconstruct signed values: sign * magnitude
        predictions = signs * input_magnitudes

        return predictions


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


class SignMSELoss(BaseLossComponent):
    """Sign MSE loss - only compares signs of predictions vs targets"""

    def __init__(self, weight=1.0):
        super().__init__(weight=weight, name="Sign MSE Loss")

    def compute(self, predictions, targets, inputs, norm_params=None, inputs_real=None):
        """
        Compute sign MSE between predictions and targets

        Args:
            predictions: (batch_size, 3) - Predicted signed values (sign * magnitude)
            targets: (batch_size, 3) - Target signed values

        Returns:
            loss: Scalar tensor - MSE between normalized signs
        """
        eps = 1e-10

        # Extract signs from targets
        target_signs = torch.sign(targets).float()  # Shape: [batch, 3], values: -1, 0, +1

        # Extract signs from predictions (normalize to [-1, 1])
        pred_signs = predictions / (torch.abs(predictions) + eps)  # Normalize to [-1, 1]

        # Compute sign MSE loss
        sign_mse_loss =torch.sqrt(torch.mean((pred_signs - target_signs) ** 2))

        return sign_mse_loss


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


class ExponentialPINNLoss(nn.Module):
    """
    Loss function wrapper for Sign Classification Network

    Usage:
        # Configuration
        loss_config = {
            "SignMSE": {"weight": 1.0},  # For ExponentialSignNN
            "SignBCE": {"weight": 1.0}   # For ExponentialSignNN_ver2
        }

        # Create loss function
        loss_fn = ExponentialPINNLoss(model, loss_config)

        # In training loop (for SignMSE)
        loss_args = {"SignMSE": (outputs, targets)}
        # OR (for SignBCE)
        loss_args = {"SignBCE": (sigmoid_probs, targets)}

        total_loss, loss_summary = loss_fn(loss_args)
    """

    def __init__(self, model, loss_config):
        super().__init__()

        self.model = model
        self.loss_config = loss_config
        self.loss_components = {}

        # Initialize SignMSE loss if requested
        if self._should_enable("SignMSE"):
            config = self.loss_config.get("SignMSE")
            weight = config.get("weight", 1.0)
            self.loss_components["SignMSE"] = SignMSELoss(weight=weight)

        # Initialize SignBCE loss if requested
        if self._should_enable("SignBCE"):
            config = self.loss_config.get("SignBCE")
            weight = config.get("weight", 1.0)
            self.loss_components["SignBCE"] = SignBCELoss(weight=weight)

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
                    "SignMSE": (outputs, targets),
                    "SignBCE": (sigmoid_probs, targets)
                }

        Returns:
            total_loss: Scalar tensor
            loss_summary: Dictionary with individual loss values
        """
        total_loss = 0.0
        loss_summary = {}

        # SignMSE Loss
        if "SignMSE" in self.loss_components and "SignMSE" in loss_args:
            outputs, targets = loss_args["SignMSE"]
            sign_mse_value = self.loss_components["SignMSE"](
                outputs, targets, None, None, None
            )
            total_loss += sign_mse_value
            loss_summary["sign_mse_loss"] = sign_mse_value.item()

        # SignBCE Loss
        if "SignBCE" in self.loss_components and "SignBCE" in loss_args:
            sigmoid_probs, targets = loss_args["SignBCE"]
            sign_bce_value = self.loss_components["SignBCE"](
                None, targets, sigmoid_probs, None, None
            )
            total_loss += sign_bce_value
            loss_summary["sign_bce_loss"] = sign_bce_value.item()

        loss_summary["total"] = total_loss.item()

        return total_loss, loss_summary

    def _print_config(self):
        """Print loss configuration"""
        print(f"\n{'='*60}")
        print("Sign Classification Network Loss Configuration:")
        print(f"{'='*60}")

        for loss_name in ["SignMSE", "SignBCE"]:
            if loss_name in self.loss_components:
                weight = self.loss_config.get(loss_name).get("weight", 1.0)
                print(f"  ✓ {loss_name:20s}: weight={weight:.3f}")
            else:
                print(f"  ✗ {loss_name:20s}: disabled")

        print(f"{'='*60}\n")
