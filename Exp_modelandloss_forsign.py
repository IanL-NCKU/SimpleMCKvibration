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
            layers.append(act())
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
        sign_mse_loss = torch.mean((pred_signs - target_signs) ** 2)

        return sign_mse_loss



class ExponentialPINNLoss(nn.Module):
    """
    Loss function wrapper for Sign Classification Network

    Usage:
        # Configuration
        loss_config = {
            "SignMSE": {"weight": 1.0}
        }

        # Create loss function
        loss_fn = ExponentialPINNLoss(model, loss_config)

        # In training loop
        loss_args = {"SignMSE": (outputs, targets)}
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
                    "SignMSE": (outputs, targets)
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

        loss_summary["total"] = total_loss.item()

        return total_loss, loss_summary

    def _print_config(self):
        """Print loss configuration"""
        print(f"\n{'='*60}")
        print("Sign Classification Network Loss Configuration:")
        print(f"{'='*60}")

        for loss_name in ["SignMSE"]:
            if loss_name in self.loss_components:
                weight = self.loss_config.get(loss_name).get("weight", 1.0)
                print(f"  ✓ {loss_name:20s}: weight={weight:.3f}")
            else:
                print(f"  ✗ {loss_name:20s}: disabled")

        print(f"{'='*60}\n")
