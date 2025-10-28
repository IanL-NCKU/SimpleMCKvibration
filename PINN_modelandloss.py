import torch
import torch.nn as nn
import numpy as np
from abc import ABC, abstractmethod

class VibrationPINN(nn.Module):
    """Physics-Informed Neural Network for vibration prediction"""

    def __init__(self, hidden_dims=[64, 128, 128, 64], activation='tanh', use_log_output=False,
                 use_finetune=False, finetune_hidden_dims=[32, 32], finetune_scale=0.1):
        super().__init__()

        self.use_log_output = use_log_output
        self.use_finetune = use_finetune
        self.finetune_scale = finetune_scale

        # Choose activation function
        if activation == 'tanh':
            act = nn.Tanh
        elif activation == 'swish':
            act = nn.SiLU
        elif activation == 'ELU':
            act = nn.ELU
        else:
            act = nn.GELU

        # Build main network
        layers = []
        input_dim = 6  # [m, zeta, k, t, x0, v0]

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(act())
            input_dim = hidden_dim

        # Final layer output dimension
        if use_log_output:
            layers.append(nn.Linear(input_dim, 6))
        else:
            layers.append(nn.Linear(input_dim, 3))

        self.network = nn.Sequential(*layers)

        # Build fine-tune network (if enabled)
        if self.use_finetune:
            finetune_layers = []
            finetune_input_dim = 9  # [m, zeta, k, t, x0, v0, x_base, v_base, a_base]

            for hidden_dim in finetune_hidden_dims:
                finetune_layers.append(nn.Linear(finetune_input_dim, hidden_dim))
                finetune_layers.append(act())
                finetune_input_dim = hidden_dim

            finetune_layers.append(nn.Linear(finetune_input_dim, 3))
            finetune_layers.append(nn.Tanh())

            self.finetune_network = nn.Sequential(*finetune_layers)
        else:
            self.finetune_network = None

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

            # FIXED: Use np.log(10.0) to avoid device mismatch issues
            # Transform to real space: x = sign * 10^log = sign * exp(log * ln(10))
            ln10 = np.log(10.0)  # Python float, PyTorch handles device automatically
            x_pred_base = sign_x * torch.exp(log_x * ln10)
            v_pred_base = sign_v * torch.exp(log_v * ln10)
            a_pred_base = sign_a * torch.exp(log_a * ln10)
        else:
            # Network outputs directly in real space: [x, v, a]
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

        return torch.cat([x_pred, v_pred, a_pred], dim=1)

    @staticmethod
    def convert_targets_to_log_space(targets, eps=1e-10):
        """
        Convert real-space targets [x, v, a] to log-space
        """
        signs = torch.sign(targets)
        signs = torch.where(signs == 0, torch.ones_like(signs), signs)

        magnitudes = torch.abs(targets) + eps
        log_magnitudes = torch.log10(magnitudes)

        log_targets = torch.stack([
            signs[:, 0], log_magnitudes[:, 0],
            signs[:, 1], log_magnitudes[:, 1],
            signs[:, 2], log_magnitudes[:, 2],
        ], dim=1)

        return log_targets

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

    def __init__(self, weight=1.0, use_relative=False):
        super().__init__(weight=weight, name="MSE Loss")
        self.use_relative = use_relative

    def compute(self, predictions, targets, inputs, norm_params=None, inputs_real=None):
        """
        Compute MSE between predictions and targets

        Args:
            predictions: (batch_size, 3) - [x, v, a] predictions in real space
            targets: (batch_size, 3) - [x, v, a] targets in real space
            inputs: Not used
            norm_params: Not used
            inputs_real: Not used

        Returns:
            loss: Scalar tensor
        """
        eps = 1e-10

        if self.use_relative:
            # Relative MSE: normalized by target magnitude
            loss = torch.mean(((predictions - targets) ** 2) / (torch.square(targets) + eps))
        else:
            # Absolute MSE
            loss = torch.mean((predictions - targets) ** 2)

        return loss


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
            scale = torch.abs(m) + 1e-10
            residual = residual / scale

        return torch.mean(residual ** 2)


class InitialConditionLoss(BaseLossComponent):
    """Initial condition loss: x(0)=x0, v(0)=v0

    Expects inputs to include t=0 samples.
    Detects which samples have t=0 and enforces initial conditions.
    """

    def __init__(self, weight=1.0, t_threshold=1e-6, use_relative=False):
        super().__init__(weight=weight, name="Initial Cond Loss")
        self.t_threshold = t_threshold
        self.use_relative = use_relative

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

        # Add a small epsilon to avoid division by zero, especially if x0 or v0 are zero
        eps = 1e-10

        if self.use_relative:
            # Normalized MSE for initial displacement
            loss_x0 = torch.mean(((x_pred_t0 - x0) ** 2) / (torch.square(x0) + eps))

            # Normalized MSE for initial velocity
            loss_v0 = torch.mean(((v_pred_t0 - v0) ** 2) / (torch.square(v0) + eps))
        else:
            # Absolute MSE for initial displacement
            loss_x0 = torch.mean((x_pred_t0 - x0) ** 2)

            # Absolute MSE for initial velocity
            loss_v0 = torch.mean((v_pred_t0 - v0) ** 2)

        return loss_x0 + loss_v0


class ConsistencyLoss_auto_diff(BaseLossComponent):
    """
    Derivative consistency: ensure v=dx/dt, a=dv/dt using automatic differentiation
    
    FIXED VERSION: Improved automatic differentiation stability + preserved chain rule correction
    
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
            norm_params: Dictionary with 'normalizer' key containing normalizer instance
            inputs_real: (batch_size, 6) - [m, zeta, k, t, x0, v0] in REAL space
        
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
            return torch.tensor(0.0, device=inputs.device)
        
        # Filter to valid samples only
        inputs_valid = inputs[valid_mask]
        t_real_valid = t_real[valid_mask]
        
        # Get the time std from normalizer
        normalizer = norm_params['normalizer']
        t_std = normalizer.log_std['t']  # std of log10(t) values
        
        # Fix 1: Don't use .detach() to preserve gradient linkage
        inputs_with_grad = inputs_valid.clone().requires_grad_(True)
        
        # Forward pass with gradient tracking
        predictions_with_grad = self.model(inputs_with_grad)
        
        # Extract predictions (these are in REAL space)
        x_pred = predictions_with_grad[:, 0]
        v_pred = predictions_with_grad[:, 1]
        a_pred = predictions_with_grad[:, 2]
        
        # Fix 2: Use create_graph=True to preserve higher-order gradients
        # Compute dx/dt_model using autograd (gradient w.r.t. t_normalized at index 3)
        dx_dt_model = torch.autograd.grad(
            outputs=x_pred,
            inputs=inputs_with_grad,
            grad_outputs=torch.ones_like(x_pred),
            create_graph=True,
            retain_graph=True,
            allow_unused=True  # Fix 3: Allow unused inputs
        )[0][:, 3]  # Take only the gradient w.r.t. t_normalized (index 3)
        
        # Compute dv/dt_model using autograd
        dv_dt_model = torch.autograd.grad(
            outputs=v_pred,
            inputs=inputs_with_grad,
            grad_outputs=torch.ones_like(v_pred),
            create_graph=True,
            retain_graph=True,
            allow_unused=True
        )[0][:, 3]  # Take only the gradient w.r.t. t_normalized (index 3)
        
        # IMPORTANT: Preserve the original chain rule correction!
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


class PINNLoss_v2(nn.Module):
    """
    Dictionary-based PINN Loss with flexible argument passing

    Usage:
        # Configuration
        loss_config = {
            "MSE": {"weight": 0.3},
            "Residual": {"weight": 0.2, "use_relative": False},
            "InitialCondition": {"weight": 0.3, "t_threshold": 1e-6},
            "Consistency": {"weight": 0.2, "t_threshold": 1e-6}
        }

        # To disable a loss, use any of:
        # 1. Omit from dict
        # 2. Set to None: "Consistency": None
        # 3. Set weight to 0: "Consistency": {"weight": 0}

        # Create loss function
        loss_fn = PINNLoss_v2(model, loss_config)

        # In training loop, prepare arguments for enabled losses
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
            weight = config.get("weight", 1.0)
            self.loss_components["MSE"] = MSELoss(weight=weight, use_relative=use_relative)

        # Initialize Residual loss if requested
        if self._should_enable("Residual"):
            config = self.loss_config.get("Residual")
            use_relative = config.get("use_relative", False)
            weight = config.get("weight", 1.0)
            self.loss_components["Residual"] = ResidualLoss(weight=weight, use_relative=use_relative)

        # Initialize InitialCondition loss if requested
        if self._should_enable("InitialCondition"):
            config = self.loss_config.get("InitialCondition")
            t_threshold = config.get("t_threshold", 1e-6)
            weight = config.get("weight", 1.0)
            use_relative = config.get("use_relative", False)
            self.loss_components["InitialCondition"] = InitialConditionLoss(weight=weight, t_threshold=t_threshold, use_relative=use_relative)

        # Initialize Consistency loss if requested
        if self._should_enable("Consistency"):
            config = self.loss_config.get("Consistency")
            t_threshold = config.get("t_threshold", 1e-6)
            weight = config.get("weight", 1.0)
            use_relative = config.get("use_relative", False)
            use_log = config.get("use_log", True)  # Default to True
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
        # Use .get() to handle missing keys gracefully
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
                    "Consistency": (inputs, inputs_real, norm_params),
                    "InitialCondition": (outputs_t0, inputs_real_t0)
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
            # MSELoss.forward() handles weighting internally
            mse_value = self.loss_components["MSE"](
                outputs, targets, None, None, None
            )
            total_loss += mse_value
            loss_summary["mse_loss"] = mse_value.item()

        # Residual Loss
        if "Residual" in self.loss_components and "Residual" in loss_args:
            outputs, inputs_real = loss_args["Residual"]
            # ResidualLoss.forward() handles weighting internally
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

        # Initial Condition Loss
        if "InitialCondition" in self.loss_components and "InitialCondition" in loss_args:
            outputs_t0, inputs_real_t0 = loss_args["InitialCondition"]
            # InitialCondition loss doesn't need normalized inputs
            initial_value = self.loss_components["InitialCondition"](
                outputs_t0, None, None, None, inputs_real_t0
            )
            total_loss += initial_value
            loss_summary["initial_loss"] = initial_value.item()

        loss_summary["total"] = total_loss.item()

        return total_loss, loss_summary

    def _print_config(self):
        """Print loss configuration"""
        print(f"\n{'='*60}")
        print("PINN Loss Configuration (Dict-based):")
        print(f"{'='*60}")

        for loss_name in ["MSE", "Residual", "InitialCondition", "Consistency"]:
            if loss_name in self.loss_components:
                weight = self.loss_config.get(loss_name).get("weight", 1.0)
                print(f"  ✓ {loss_name:20s}: weight={weight:.3f}")
            else:
                print(f"  ✗ {loss_name:20s}: disabled")

        print(f"{'='*60}\n")
