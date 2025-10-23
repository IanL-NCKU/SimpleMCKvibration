import torch
import torch.nn as nn
import numpy as np
from abc import ABC, abstractmethod

class VibrationPINN(nn.Module):
    """Direct prediction of x, v, a"""
    
    def __init__(self, hidden_dims=[64, 128, 128, 64], activation='tanh'):
        super().__init__()
        
        # Choose activation
        if activation == 'tanh':
            act = nn.Tanh
        elif activation == 'swish':
            act = nn.SiLU
        else:
            act = nn.GELU
        
        # Build network
        layers = []
        input_dim = 6
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(act())
            input_dim = hidden_dim
        
        layers.append(nn.Linear(input_dim, 3))  # Output: [xt, vt, at]
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights (Xavier for Tanh)
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.zeros_(m.bias)
    
    def forward(self, m, c, k, t, x0, v0):
        """
        Inputs can be:
        - Separate tensors: m, c, k, t, x0, v0 each (batch_size,)
        - Stacked tensor: (batch_size, 6)
        """
        if isinstance(m, torch.Tensor) and m.dim() == 1:
            # Combine separate inputs
            x = torch.stack([m, c, k, t, x0, v0], dim=1)
        else:
            x = m  # Already stacked
        
        return self.network(x)  # (batch_size, 3)
    

class BaseLossComponent(ABC, nn.Module):
    """Abstract base class for loss components"""
    
    def __init__(self, weight=1.0, name="base"):
        super().__init__()
        self.weight = weight
        self.name = name
        self.enabled = weight > 0
    
    @abstractmethod
    def compute(self, predictions, targets, inputs, norm_params=None):
        """
        Compute the loss component
        
        Args:
            predictions: (batch_size, 3) - [x, v, a] predictions
            targets: (batch_size, 3) - [x, v, a] targets
            inputs: (batch_size, 6) - [m, zeta, k, t, x0, v0] normalized
            norm_params: Dictionary with normalization parameters
        
        Returns:
            loss: Scalar tensor
        """
        pass
    
    def forward(self, predictions, targets, inputs, norm_params=None):
        """Wrapper that applies weight and checks if enabled"""
        if not self.enabled:
            return torch.tensor(0.0, device=predictions.device)
        
        loss = self.compute(predictions, targets, inputs, norm_params)
        return self.weight * loss
    
    def __repr__(self):
        status = "✓" if self.enabled else "✗"
        return f"{self.name:20s}: weight={self.weight:.3f} {status}"



class ResidualLoss(BaseLossComponent):
    """Physics residual loss: m*a + c*v + k*x = 0"""
    
    def __init__(self, weight=1.0, use_relative=False):
        super().__init__(weight=weight, name="Residual Loss")
        self.use_relative = use_relative
    
    def compute(self, predictions, targets, inputs, norm_params=None):
        """
        Enforce physics equation: m*a + c*v + k*x = 0

        Args:
            predictions: (batch_size, 3) - [x, v, a] in real space
            inputs: (batch_size, 6) - [m, zeta, k, t, x0, v0] in real space (denormalized)
            norm_params: Not used (kept for interface compatibility)
            use_relative: If True, compute residual / |m*a| for scale invariance
        """
        x_pred = predictions[:, 0]
        v_pred = predictions[:, 1]
        a_pred = predictions[:, 2]

        # Extract real-space parameters (already denormalized)
        m = inputs[:, 0]
        zeta = inputs[:, 1]
        k = inputs[:, 2]

        # Compute damping coefficient
        c = 2 * zeta * torch.sqrt(m * k)

        # Physics residual: m*a + c*v + k*x = 0
        residual = m * a_pred + c * v_pred + k * x_pred

        if self.use_relative:
            # Scale-invariant relative residual
            scale = torch.abs(m * a_pred) + 1e-8
            residual = residual / scale

        return torch.mean(residual ** 2)


class InitialConditionLoss(BaseLossComponent):
    """Initial condition loss: x(0)=x0, v(0)=v0

    Expects inputs to include t=0 samples.
    Detects which samples have t=0 and enforces initial conditions.
    """

    def __init__(self, weight=1.0, t_threshold=1e-6):
        super().__init__(weight=weight, name="Initial Cond Loss")
        self.t_threshold = t_threshold

    def compute(self, predictions, targets, inputs, norm_params=None):
        """
        Enforce initial conditions at t=0

        Args:
            predictions: (batch_size, 3) - [x, v, a] predictions in real space
            targets: Not used
            inputs: (batch_size, 6) - [m, zeta, k, t, x0, v0] in real space (denormalized)
                    Must include samples with t=0
            norm_params: Not used (kept for interface compatibility)
        """
        # Find samples where t ≈ 0
        t = inputs[:, 3]  # Real time values
        initial_mask = torch.abs(t) < self.t_threshold

        if initial_mask.sum() == 0:
            # No t=0 samples in this batch
            return torch.tensor(0.0, device=predictions.device)

        # Extract predictions at t=0
        x_pred_t0 = predictions[initial_mask, 0]
        v_pred_t0 = predictions[initial_mask, 1]

        # Extract initial conditions (in real space)
        x0 = inputs[initial_mask, 4]  # Real x0
        v0 = inputs[initial_mask, 5]  # Real v0

        # MSE between predicted at t=0 and initial conditions
        loss_x0 = torch.mean((x_pred_t0 - x0) ** 2)
        loss_v0 = torch.mean((v_pred_t0 - v0) ** 2)

        return loss_x0 + loss_v0


class ConsistencyLoss(BaseLossComponent):
    """Derivative consistency: ensure v=dx/dt, a=dv/dt"""
    
    def __init__(self, weight=1.0, method='finite_diff'):
        super().__init__(weight=weight, name="Consistency Loss")
        self.method = method  # 'finite_diff' or 'autodiff'
    
    def compute(self, predictions, targets, inputs, norm_params=None):
        """
        Ensure derivative consistency
        
        Methods:
        - 'finite_diff': Use finite differences (requires sorted time samples)
        - 'autodiff': Use automatic differentiation (requires special setup)
        
        Note: Finite difference requires multiple samples from same trajectory
        which is difficult in standard mini-batch training.
        This is a simplified placeholder.
        """
        if self.method == 'finite_diff':
            return self._finite_difference_consistency(predictions, inputs)
        elif self.method == 'autodiff':
            return self._autodiff_consistency(predictions, inputs, norm_params)
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def _finite_difference_consistency(self, predictions, inputs):
        """
        Finite difference approximation (simplified)
        
        In practice, this requires trajectory-aware batching
        where consecutive samples in batch are from same system at Δt apart
        """
        # This is a placeholder - proper implementation requires special batching
        # For now, return zero (disabled)
        return torch.tensor(0.0, device=predictions.device)
    
    def _autodiff_consistency(self, predictions, inputs, norm_params):
        """
        Automatic differentiation consistency
        
        Requires model to predict only x(t), then v and a are computed via autodiff
        This requires different model architecture
        """
        # This is a placeholder - requires special model architecture
        return torch.tensor(0.0, device=predictions.device)
    

class PINNLoss(nn.Module):
    """
    Composite PINN Loss with automatic component instantiation

    Usage:
        # Simple interface - set weight=None or weight=0 to disable
        loss_fn = PINNLoss(
            mse_weight=0.2,
            residual_weight=None,      # Won't be created
            initial_weight=0.8,
            consistency_weight=0.0     # Won't be created
        )

        # With component-specific options
        loss_fn = PINNLoss(
            mse_weight=0.5,
            residual_weight=0.3,
            initial_weight=0.2,
            residual_use_relative=True,
            initial_t_threshold=1e-5
        )
    """

    def __init__(self,
                 mse_weight=None,
                 residual_weight=None,
                 initial_weight=None,
                 consistency_weight=None,
                 # Component-specific kwargs
                 residual_use_relative=False,
                 initial_t_threshold=1e-4):
        super().__init__()

        self.components = nn.ModuleList()
        self.component_names = []

        # === MSE Loss ===
        if mse_weight is not None and mse_weight > 0:
            self.mse_weight = mse_weight
            self.mse_fn = nn.MSELoss()
            self.component_names.append('mse_loss')
        else:
            self.mse_weight = None
            self.mse_fn = None

        # === Residual Loss ===
        if residual_weight is not None and residual_weight > 0:
            self.residual_loss = ResidualLoss(
                weight=residual_weight,
                use_relative=residual_use_relative
            )
            self.components.append(self.residual_loss)
            self.component_names.append('residual_loss')
        else:
            self.residual_loss = None

        # === Initial Condition Loss ===
        if initial_weight is not None and initial_weight > 0:
            self.initial_loss = InitialConditionLoss(
                weight=initial_weight,
                t_threshold=initial_t_threshold
            )
            self.components.append(self.initial_loss)
            self.component_names.append('initial_loss')
        else:
            self.initial_loss = None

        # === Consistency Loss ===
        if consistency_weight is not None and consistency_weight > 0:
            self.consistency_loss = ConsistencyLoss(weight=consistency_weight)
            self.components.append(self.consistency_loss)
            self.component_names.append('consistency_loss')
        else:
            self.consistency_loss = None

        # Validate at least one component is active
        has_mse = self.mse_fn is not None
        has_other_components = len(self.components) > 0
        if not (has_mse or has_other_components):
            raise ValueError("At least one loss component must have weight > 0")

        # Print configuration
        self._print_config()
    
    def _print_config(self):
        """Print loss configuration"""
        print(f"\n{'='*60}")
        print("PINN Loss Configuration:")
        print(f"{'='*60}")

        if self.mse_fn is not None:
            print(f"  ✓ MSE Loss           : weight={self.mse_weight:.3f} ✓")
        else:
            print(f"  ✗ MSE Loss           : disabled (weight=0)")

        if hasattr(self, 'residual_loss') and self.residual_loss:
            print(f"  ✓ {self.residual_loss}")
        else:
            print(f"  ✗ Residual Loss      : disabled (weight=0)")

        if hasattr(self, 'initial_loss') and self.initial_loss:
            print(f"  ✓ {self.initial_loss}")
        else:
            print(f"  ✗ Initial Cond Loss  : disabled (weight=0)")

        if hasattr(self, 'consistency_loss') and self.consistency_loss:
            print(f"  ✓ {self.consistency_loss}")
        else:
            print(f"  ✗ Consistency Loss   : disabled (weight=0)")

        print(f"{'='*60}\n")
    
    def forward(self, predictions, targets, inputs, norm_params=None):
        """
        Compute total loss from active components

        Returns:
            total_loss: Scalar tensor
            loss_dict: Dictionary with individual components for logging
        """
        total_loss = 0.0
        loss_dict = {}

        # === MSE Loss ===
        if self.mse_fn is not None:
            mse_value = self.mse_fn(predictions, targets)
            weighted_mse = self.mse_weight * mse_value
            total_loss = total_loss + weighted_mse
            loss_dict['mse_loss'] = weighted_mse.item()

        # === Other Component Losses ===
        for component, name in zip(self.components, self.component_names):
            if name != 'mse_loss':  # MSE already handled above
                loss_value = component(predictions, targets, inputs, norm_params)
                total_loss = total_loss + loss_value
                loss_dict[name] = loss_value.item()

        loss_dict['total'] = total_loss.item()

        return total_loss, loss_dict


