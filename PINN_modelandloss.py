import torch
import torch.nn as nn
import numpy as np
from abc import ABC, abstractmethod

class VibrationPINN(nn.Module):
    """Physics-Informed Neural Network for vibration prediction"""

    def __init__(self, hidden_dims=[64, 128, 128, 64], activation='tanh', use_log_output=False):
        super().__init__()

        self.use_log_output = use_log_output

        # Choose activation
        if activation == 'tanh':
            act = nn.Tanh
        elif activation == 'swish':
            act = nn.SiLU
        else:
            act = nn.GELU

        # Build network
        layers = []
        input_dim = 6  # [m, zeta, k, t, x0, v0]

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(act())
            input_dim = hidden_dim

        # Final layer output dimension depends on whether we use log representation
        if use_log_output:
            # Output 6 values: [sign_x, log_x, sign_v, log_v, sign_a, log_a]
            layers.append(nn.Linear(input_dim, 6))
        else:
            # Output 3 values: [x, v, a] directly
            layers.append(nn.Linear(input_dim, 3))

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
               [m, zeta, k, t, x0, v0]

        Returns:
            predictions: Output tensor of shape (batch_size, 3)
                        [x(t), v(t), a(t)] in real space
        """
        output = self.network(x)

        if self.use_log_output:
            # Network outputs: [sign_x, log_x, sign_v, log_v, sign_a, log_a]
            # Transform to real space: sign * 10^log_magnitude

            # Extract sign and log magnitude for each output
            sign_x = torch.tanh(output[:, 0:1])  # Soft sign in [-1, 1]
            log_x = output[:, 1:2]
            sign_v = torch.tanh(output[:, 2:3])
            log_v = output[:, 3:4]
            sign_a = torch.tanh(output[:, 4:5])
            log_a = output[:, 5:6]

            # Transform to real space: sign * 10^log_magnitude
            x_pred = sign_x * (10 ** log_x)
            v_pred = sign_v * (10 ** log_v)
            a_pred = sign_a * (10 ** log_a)

            return torch.cat([x_pred, v_pred, a_pred], dim=1)
        else:
            # Direct output in real space
            return output  # [batch_size, 3]

    @staticmethod
    def convert_targets_to_log_space(targets, eps=1e-10):
        """
        Convert real-space targets [x, v, a] to log-space [sign_x, log_x, sign_v, log_v, sign_a, log_a]

        Args:
            targets: (batch_size, 3) - [x, v, a] in real space
            eps: Small value to avoid log(0)

        Returns:
            log_targets: (batch_size, 6) - [sign_x, log_x, sign_v, log_v, sign_a, log_a]
        """
        # Extract signs (convert to -1 or +1)
        signs = torch.sign(targets)
        signs = torch.where(signs == 0, torch.ones_like(signs), signs)  # Replace 0 with 1

        # Compute log magnitudes
        magnitudes = torch.abs(targets) + eps
        log_magnitudes = torch.log10(magnitudes)

        # Interleave signs and log magnitudes
        log_targets = torch.stack([
            signs[:, 0], log_magnitudes[:, 0],  # x
            signs[:, 1], log_magnitudes[:, 1],  # v
            signs[:, 2], log_magnitudes[:, 2],  # a
        ], dim=1)

        return log_targets
    

class BaseLossComponent(ABC, nn.Module):
    """Abstract base class for loss components"""
    
    def __init__(self, weight=1.0, name="base"):
        super().__init__()
        self.weight = weight
        self.name = name
        self.enabled = weight > 0
    
    @abstractmethod
    def compute(self, predictions, targets, inputs, norm_params=None, inputs_real=None):
        """
        Compute the loss component

        Args:
            predictions: (batch_size, 3) - [x, v, a] predictions
            targets: (batch_size, 3) - [x, v, a] targets
            inputs: (batch_size, 6) - [m, zeta, k, t, x0, v0] normalized
            norm_params: Dictionary with normalization parameters
            inputs_real: (batch_size, 6) - [m, zeta, k, t, x0, v0] in real space (optional)

        Returns:
            loss: Scalar tensor
        """
        pass

    def forward(self, predictions, targets, inputs, norm_params=None, inputs_real=None):
        """Wrapper that applies weight and checks if enabled"""
        if not self.enabled:
            return torch.tensor(0.0, device=predictions.device)

        loss = self.compute(predictions, targets, inputs, norm_params, inputs_real)
        return self.weight * loss
    
    def __repr__(self):
        status = "✓" if self.enabled else "✗"
        return f"{self.name:20s}: weight={self.weight:.3f} {status}"



class ResidualLoss(BaseLossComponent):
    """Physics residual loss: m*a + c*v + k*x = 0"""
    
    def __init__(self, weight=1.0, use_relative=False):
        super().__init__(weight=weight, name="Residual Loss")
        self.use_relative = use_relative
    
    def compute(self, predictions, targets, inputs, norm_params=None, inputs_real=None):
        """
        Enforce physics equation: m*a + c*v + k*x = 0

        Args:
            predictions: (batch_size, 3) - [x, v, a] in real space
            inputs: (batch_size, 6) - [m, zeta, k, t, x0, v0] normalized (not used here)
            norm_params: Not used (kept for interface compatibility)
            inputs_real: (batch_size, 6) - [m, zeta, k, t, x0, v0] in real space
            use_relative: If True, compute residual / |m*a| for scale invariance
        """
        if inputs_real is None:
            raise ValueError("ResidualLoss requires inputs_real to be provided")

        x_pred = predictions[:, 0]
        v_pred = predictions[:, 1]
        a_pred = predictions[:, 2]

        # Extract real-space parameters
        m = inputs_real[:, 0]
        zeta = inputs_real[:, 1]
        k = inputs_real[:, 2]

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

    def compute(self, predictions, targets, inputs, norm_params=None, inputs_real=None):
        """
        Enforce initial conditions at t=0

        Args:
            predictions: (batch_size, 3) - [x, v, a] predictions in real space
            targets: Not used
            inputs: (batch_size, 6) - [m, zeta, k, t, x0, v0] normalized (not used)
            norm_params: Not used (kept for interface compatibility)
            inputs_real: (batch_size, 6) - [m, zeta, k, t, x0, v0] in real space
                        ALL samples must have t=0
        """
        if inputs_real is None:
            raise ValueError("InitialConditionLoss requires inputs_real to be provided")

        # Verify that all samples have t ≈ 0
        t_real = inputs_real[:, 3]
        if not torch.all(torch.abs(t_real) < self.t_threshold):
            raise ValueError(f"InitialConditionLoss requires all samples to have t≈0, but found max(|t|)={torch.max(torch.abs(t_real)).item()}")

        # Extract predictions at t=0
        x_pred_t0 = predictions[:, 0]
        v_pred_t0 = predictions[:, 1]

        # Extract initial conditions (in real space)
        x0 = inputs_real[:, 4]  # Real x0
        v0 = inputs_real[:, 5]  # Real v0

        # MSE between predicted at t=0 and initial conditions
        loss_x0 = torch.mean((x_pred_t0 - x0) ** 2)
        loss_v0 = torch.mean((v_pred_t0 - v0) ** 2)

        return loss_x0 + loss_v0


class ConsistencyLoss_auto_diff(BaseLossComponent):
    """Derivative consistency: ensure v=dx/dt, a=dv/dt using automatic differentiation

    Handles log-normalized time transformation:
    t_model = (log10(t_real) - mean) / std

    Chain rule for derivatives:
    dx/dt_real = dx/dt_model * dt_model/dt_real
    where dt_model/dt_real = 1 / (std * t_real * ln(10))
    """

    def __init__(self, weight=1.0, model=None, t_threshold=1e-6):
        super().__init__(weight=weight, name="Consistency Loss AD")
        self.model = model
        self.t_threshold = t_threshold  # Filter out t≈0 samples

    def set_model(self, model):
        """Set the model reference for gradient computation"""
        self.model = model

    def compute(self, predictions, targets, inputs, norm_params=None, inputs_real=None):
        """
        Enforces consistency between displacement, velocity, and acceleration
        using automatic differentiation with log-normalized time.

        This loss computes:
        - dx/dt_model using autograd (gradient w.r.t. normalized time)
        - dv/dt_model using autograd (gradient w.r.t. normalized time)

        Then transforms to real domain using chain rule:
        - dx/dt_real = dx/dt_model / (std * t_real * ln(10))
        - dv/dt_real = dv/dt_model / (std * t_real * ln(10))

        Finally enforces:
        - v_pred ≈ dx/dt_real
        - a_pred ≈ dv/dt_real

        Args:
            predictions: Not used (we recompute with gradient tracking)
            targets: Not used
            inputs: (batch_size, 6) - [m, zeta, k, t, x0, v0] NORMALIZED
            norm_params: Dictionary with 'normalizer' key containing Vibration_DataNormalizer instance
            inputs_real: (batch_size, 6) - [m, zeta, k, t, x0, v0] in REAL space
                        Used to get t_real for chain rule

        Returns:
            loss: Scalar tensor measuring consistency error
        """
        if self.model is None:
            raise ValueError("Model not set. Call set_model() before using ConsistencyLoss_auto_diff")

        if inputs_real is None:
            raise ValueError("ConsistencyLoss_auto_diff requires inputs_real to be provided")

        if norm_params is None or 'normalizer' not in norm_params:
            raise ValueError("norm_params with 'normalizer' must be provided for ConsistencyLoss_auto_diff")

        # Get t_real from inputs_real
        t_real = inputs_real[:, 3]

        # Filter out samples where t_real ≈ 0 (to avoid division by zero)
        valid_mask = t_real > self.t_threshold

        if valid_mask.sum() == 0:
            # No valid samples (all t≈0)
            return torch.tensor(0.0, device=predictions.device)

        # Filter to valid samples only
        inputs_valid = inputs[valid_mask]
        t_real_valid = t_real[valid_mask]

        # Get the time std from normalizer
        normalizer = norm_params['normalizer']
        t_std = normalizer.log_std['t']  # std of log10(t) values

        # Enable gradient computation for normalized inputs
        inputs_with_grad = inputs_valid.clone().detach().requires_grad_(True)

        # Forward pass with gradient tracking
        predictions_with_grad = self.model(inputs_with_grad)

        # Extract predictions (these are in REAL space)
        x_pred = predictions_with_grad[:, 0]
        v_pred = predictions_with_grad[:, 1]
        a_pred = predictions_with_grad[:, 2]

        # Compute dx/dt_model using autograd (gradient w.r.t. t_normalized at index 3)
        dx_dt_model = torch.autograd.grad(
            outputs=x_pred,
            inputs=inputs_with_grad,
            grad_outputs=torch.ones_like(x_pred),
            create_graph=True,
            retain_graph=True
        )[0][:, 3]  # Take only the gradient w.r.t. t_normalized (index 3)

        # Compute dv/dt_model using autograd
        dv_dt_model = torch.autograd.grad(
            outputs=v_pred,
            inputs=inputs_with_grad,
            grad_outputs=torch.ones_like(v_pred),
            create_graph=True,
            retain_graph=True
        )[0][:, 3]  # Take only the gradient w.r.t. t_normalized (index 3)

        # Apply chain rule to transform from model domain to real domain
        # Chain rule: dx/dt_real = dx/dt_model * dt_model/dt_real
        #
        # t_model = (log10(t_real) - mean) / std
        # dt_model/dt_real = d/dt_real[(log10(t_real) - mean) / std]
        #                  = (1/std) * d/dt_real[log10(t_real)]
        #                  = (1/std) * (1 / (t_real * ln(10)))
        #
        # Therefore: dx/dt_real = dx/dt_model / (std * t_real * ln(10))

        ln10 = torch.tensor(np.log(10), device=inputs.device, dtype=inputs.dtype)
        chain_rule_factor = 1.0 / (t_std * t_real_valid * ln10)

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

        return total_loss
    



class ConsistencyLoss_finite_diff(BaseLossComponent):
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
            model=my_pinn_model,       # Required for consistency loss
            mse_weight=0.2,
            residual_weight=None,      # Won't be created
            initial_weight=0.8,
            consistency_weight=0.0     # Won't be created
        )

        # With component-specific options
        loss_fn = PINNLoss(
            model=my_pinn_model,
            mse_weight=0.5,
            residual_weight=0.3,
            initial_weight=0.2,
            residual_use_relative=True,
            initial_t_threshold=1e-5
        )
    """

    def __init__(self,
                 model=None,
                 mse_weight=None,
                 residual_weight=None,
                 initial_weight=None,
                 consistency_weight=None,
                 # Component-specific kwargs
                 residual_use_relative=False,
                 initial_t_threshold=1e-4):
        super().__init__()

        self.model = model

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
            if model is None:
                raise ValueError("Model must be provided to PINNLoss when using consistency_weight > 0")
            self.consistency_loss = ConsistencyLoss_auto_diff(weight=consistency_weight, model=model)
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
    
    def forward(self, predictions, targets, inputs, norm_params=None, inputs_real=None):
        """
        Compute total loss from active components

        Args:
            predictions: (batch_size, 3) - [x, v, a] predictions
            targets: (batch_size, 3) - [x, v, a] targets
            inputs: (batch_size, 6) - [m, zeta, k, t, x0, v0] NORMALIZED
            norm_params: Dictionary with normalizer
            inputs_real: (batch_size, 6) - [m, zeta, k, t, x0, v0] in REAL space

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
                loss_value = component(predictions, targets, inputs, norm_params, inputs_real)
                total_loss = total_loss + loss_value
                loss_dict[name] = loss_value.item()

        loss_dict['total'] = total_loss.item()

        return total_loss, loss_dict


