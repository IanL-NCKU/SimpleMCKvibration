import torch
import torch.nn as nn
import numpy as np
from abc import ABC, abstractmethod
from PINN_dataset import generalize_alpha, generalize_beta

class SignWithHardTanh(torch.autograd.Function):
    """Custom gradient function for sign operation

    Forward: Uses sign function for binarization
    Backward: Uses HardTanh derivative (rectangular gradient)
    """
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.sign(x)

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        # HardTanh(-1, 1) derivative: 1 between [-1, 1], 0 elsewhere
        grad_input = grad_output.clone()
        grad_input[x.abs() > 1.0] = 0
        return grad_input


class ResidualBlock(nn.Module):
    """Residual block with two linear layers and skip connection

    Architecture depends on use_batchnorm:
        True:  x → fc1 → BN → act → dropout → fc2 → BN → dropout → (+shortcut) → act
        False: x → fc1 → act → BN → dropout → fc2 → BN → dropout → (+shortcut) → act
    """

    def __init__(self, in_dim, mid_dim, out_dim, activation, dropout=0.3, use_batchnorm=False):
        super().__init__()
        self.use_batchnorm = use_batchnorm

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
        if self.use_batchnorm:
            out = self.bn1(out)
            out = self.act1(out)
        else:
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


class VibrationPINN(nn.Module):
    """Physics-Informed Neural Network for vibration prediction"""

    def __init__(self, hidden_dims=[64, 128, 128, 64], activation='tanh', use_internal_sign=False,
                 use_finetune=False, finetune_hidden_dims=[32, 32], finetune_scale=0.1,
                 use_exponential_superposition=False):
        super().__init__()

        self.use_internal_sign = use_internal_sign
        self.use_finetune = use_finetune
        self.finetune_scale = finetune_scale
        self.use_exponential_superposition = use_exponential_superposition

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
        if use_exponential_superposition:
            layers.append(nn.Linear(input_dim, 12))
            # layers.append(act())
        elif use_internal_sign:
            layers.append(nn.Linear(input_dim, 6))
            # layers.append(act())
        else:
            layers.append(nn.Linear(input_dim, 3))
            # layers.append(act())

        self.network = nn.Sequential(*layers)

        # Build fine-tune network (if enabled)
        if self.use_finetune:
            finetune_layers = []
            finetune_input_dim = 9  # [m, zeta, k, t, x0, v0, x_base, v_base, a_base]

            for hidden_dim in finetune_hidden_dims:
                finetune_layers.append(nn.Linear(finetune_input_dim, hidden_dim))
                # finetune_layers.append(act())
                finetune_input_dim = hidden_dim

            finetune_layers.append(nn.Linear(finetune_input_dim, 3))
            # finetune_layers.append(act())

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
        if self.use_exponential_superposition:
            # Network outputs: [m_x1, n_x1, m_x2, n_x2, m_v1, n_v1, m_v2, n_v2, m_a1, n_a1, m_a2, n_a2]
            m_x1 = output[:, 0:1]
            n_x1 = output[:, 1:2]
            m_x2 = output[:, 2:3]
            n_x2 = output[:, 3:4]

            m_v1 = output[:, 4:5]
            n_v1 = output[:, 5:6]
            m_v2 = output[:, 6:7]
            n_v2 = output[:, 7:8]

            m_a1 = output[:, 8:9]
            n_a1 = output[:, 9:10]
            m_a2 = output[:, 10:11]
            n_a2 = output[:, 11:12]

            # Exponential superposition: x_t = m_x1*exp(n_x1) + m_x2*exp(n_x2)
            x_pred_base = m_x1 * torch.exp(n_x1) + m_x2 * torch.exp(n_x2)
            v_pred_base = m_v1 * torch.exp(n_v1) + m_v2 * torch.exp(n_v2)
            a_pred_base = m_a1 * torch.exp(n_a1) + m_a2 * torch.exp(n_a2)

        elif self.use_internal_sign:
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


class VibrationSignNN_ver3(nn.Module):
    """Binary Classification Network with Residual Connections

    Predicts signs from parameters + magnitude inputs (6 features total).
    Uses sigmoid outputs and BCE loss for sign classification.
    """

    def __init__(self, hidden_dims=[128, 64, 32], activation='relu', dropout=0.3, use_batchnorm=False):
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

        input_dim = 9  # [m, zeta, k, t, x0, v0, mag_x, mag_v, mag_a]

        # Decide whether to use residual blocks
        if len(hidden_dims) <= 2:
            # Simple sequential layers (no residual)
            self.use_residual = False
            layers = []
            for hidden_dim in hidden_dims:
                layers.append(nn.Linear(input_dim, hidden_dim))
                if use_batchnorm:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                    layers.append(act())
                else:
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
            dims = [input_dim] + hidden_dims
            i = 0
            while i + 2 < len(dims):
                block = ResidualBlock(dims[i], dims[i+1], dims[i+2], act, dropout, use_batchnorm=use_batchnorm)
                self.blocks.append(block)
                i += 2

            # Handle remaining layer if odd number
            if i + 1 < len(dims):
                if use_batchnorm:
                    remaining_layer = nn.Sequential(
                        nn.Linear(dims[i], dims[i+1]),
                        nn.BatchNorm1d(dims[i+1]),
                        act(),
                        nn.Dropout(dropout)
                    )
                else:
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
        if use_batchnorm:
            self.output_heads = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(input_dim, 16),
                    nn.BatchNorm1d(16),
                    act(),
                    nn.Linear(16, 1),
                    nn.BatchNorm1d(1),
                    nn.Sigmoid()
                )
                for _ in range(num_outputs)
            ])
        else:
            self.output_heads = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(input_dim, 16),
                    act(),
                    nn.BatchNorm1d(16),
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
            x: Input tensor of shape (batch_size, 9)
               [m, zeta, k, t, x0, v0, mag_x, mag_v, mag_a]

        Returns:
            predictions: Output tensor of shape (batch_size, 3)
                        Reconstructed signed values: sign_from_probs
        """
        # Shared feature extraction
        if self.use_residual:
            for block in self.blocks:
                x = block(x)
            shared_features = x
        else:
            shared_features = self.shared_layers(x)

        # Three independent binary classifiers (sigmoid outputs in [0, 1])
        outputs = [head(shared_features) for head in self.output_heads]

        # Concatenate to [batch_size, 3]
        probs = torch.cat(outputs, dim=1)

        # Store probabilities for loss computation
        self.last_sign_probs = probs

        # Convert probabilities to signs: [0, 1] -> [-1, 1]
        signs = 2 * probs - 1

        predictions = signs

        return predictions


class VibrationSignNN_ver4(nn.Module):
    """Binary Classification Network with Residual Connections (params-only version)

    Predicts signs from parameters only (6 features total, no magnitude inputs).
    Uses sigmoid outputs and BCE loss for sign classification.
    """

    def __init__(self, hidden_dims=[128, 64, 32], activation='relu', dropout=0.3, use_batchnorm=False):
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

        input_dim = 6  # [m, zeta, k, t, x0, v0] only

        # Decide whether to use residual blocks
        if len(hidden_dims) <= 2:
            # Simple sequential layers (no residual)
            self.use_residual = False
            layers = []
            for hidden_dim in hidden_dims:
                layers.append(nn.Linear(input_dim, hidden_dim))
                if use_batchnorm:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                    layers.append(act())
                else:
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
            dims = [input_dim] + hidden_dims
            i = 0
            while i + 2 < len(dims):
                block = ResidualBlock(dims[i], dims[i+1], dims[i+2], act, dropout, use_batchnorm=use_batchnorm)
                self.blocks.append(block)
                i += 2

            # Handle remaining layer if odd number
            if i + 1 < len(dims):
                if use_batchnorm:
                    remaining_layer = nn.Sequential(
                        nn.Linear(dims[i], dims[i+1]),
                        nn.BatchNorm1d(dims[i+1]),
                        act(),
                        nn.Dropout(dropout)
                    )
                else:
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
        if use_batchnorm:
            self.output_heads = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(input_dim, 16),
                    nn.BatchNorm1d(16),
                    act(),
                    nn.Linear(16, 1),
                    nn.BatchNorm1d(1),
                    nn.Sigmoid()
                )
                for _ in range(num_outputs)
            ])
        else:
            self.output_heads = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(input_dim, 16),
                    act(),
                    nn.BatchNorm1d(16),
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
               [m, zeta, k, t, x0, v0]

        Returns:
            predictions: Output tensor of shape (batch_size, 3)
                        Sign predictions: values in [-1, 1]
        """
        # Shared feature extraction
        if self.use_residual:
            for block in self.blocks:
                x = block(x)
            shared_features = x
        else:
            shared_features = self.shared_layers(x)

        # Three independent binary classifiers (sigmoid outputs in [0, 1])
        outputs = [head(shared_features) for head in self.output_heads]

        # Concatenate to [batch_size, 3]
        probs = torch.cat(outputs, dim=1)

        # Store probabilities for loss computation
        self.last_sign_probs = probs

        # Convert probabilities to signs: [0, 1] -> [-1, 1]
        signs = 2 * probs - 1

        # Return signs directly (no magnitude multiplication)
        predictions = signs

        return predictions


class VibrationPINN_ver3(nn.Module):
    """Physics-Informed Neural Network with dual sign networks for vibration prediction

    Architecture similar to ExponentialPINN_ver3 but adapted for vibration problem.
    """

    def __init__(self, hidden_dims=[64, 128, 128, 64], activation='tanh', use_internal_sign=False,
                 use_finetune=False, finetune_hidden_dims=[32, 32], finetune_scale=0.1,
                 logabs_sign_network_hidden_dims=[128, 64, 32], logabs_sign_network_dropout=0.3,
                 real_sign_network_hidden_dims=[128, 64, 32], real_sign_network_dropout=0.3,
                 batchnorm=False):
        super().__init__()

        self.use_internal_sign = use_internal_sign
        self.use_finetune = use_finetune
        self.finetune_scale = finetune_scale
        self.logabs_last_sign_probs = None
        self.real_last_sign_probs = None

        # Choose activation function
        if activation == 'tanh':
            act = nn.Tanh
        elif activation == 'swish':
            act = nn.SiLU
        elif activation == 'elu':
            act = nn.ELU
        elif activation == 'ELU':
            act = nn.ELU
        elif activation == 'relu':
            act = nn.ReLU
        else:
            act = nn.GELU

        # Build main network
        layers = []
        input_dim = 6  # [m, zeta, k, t, x0, v0] for vibration

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            if batchnorm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(act())
            input_dim = hidden_dim

        # Final layer output dimension
        if use_internal_sign:
            layers.append(nn.Linear(input_dim, 6))  # [sign_x, log_x, sign_v, log_v, sign_a, log_a]
        else:
            layers.append(nn.Linear(input_dim, 3))  # [x_t, v_t, a_t]

        # Add softplus to ensure positive outputs (magnitudes)
        layers.append(nn.Softplus())
        self.network = nn.Sequential(*layers)

        # Build fine-tune network (if enabled)
        if self.use_finetune:
            finetune_layers = []
            finetune_input_dim = 12  # [m, zeta, k, t, x0, v0, sign_prob_x, sign_prob_v, sign_prob_a, mag_x, mag_v, mag_a]
            finetune_act = nn.Tanh
            for hidden_dim in finetune_hidden_dims:
                finetune_layers.append(nn.Linear(finetune_input_dim, hidden_dim))
                if batchnorm:
                    finetune_layers.append(nn.BatchNorm1d(hidden_dim))
                finetune_layers.append(finetune_act())
                finetune_input_dim = hidden_dim

            finetune_layers.append(nn.Linear(finetune_input_dim, 3))

            self.finetune_network = nn.Sequential(*finetune_layers)
        else:
            self.finetune_network = None

        # Build logabs sign network (predicts signs of log-absolute values)
        self.logabs_sign_network = VibrationSignNN_ver3(
            hidden_dims=logabs_sign_network_hidden_dims,
            activation=activation,
            dropout=logabs_sign_network_dropout,
            use_batchnorm=batchnorm
        )

        # Build real sign network (predicts signs of real values from params only)
        self.real_sign_network = VibrationSignNN_ver4(
            hidden_dims=real_sign_network_hidden_dims,
            activation=activation,
            dropout=real_sign_network_dropout,
            use_batchnorm=batchnorm
        )

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.zeros_(m.bias)

    def freeze_finetune_network(self):
        """Freeze finetune network parameters to prevent optimization"""
        if self.finetune_network is not None:
            for param in self.finetune_network.parameters():
                param.requires_grad = False

    def unfreeze_finetune_network(self):
        """Unfreeze finetune network parameters to enable optimization"""
        if self.finetune_network is not None:
            for param in self.finetune_network.parameters():
                param.requires_grad = True

    def freeze_magnitude_network(self):
        """Freeze magnitude network (main network) parameters to prevent optimization"""
        for param in self.network.parameters():
            param.requires_grad = False

    def unfreeze_magnitude_network(self):
        """Unfreeze magnitude network (main network) parameters to enable optimization"""
        for param in self.network.parameters():
            param.requires_grad = True

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, 6)
               [m, zeta, k, t, x0, v0]

        Returns:
            mag_preds: Magnitude predictions (batch_size, 3)
            logabs_sign_pred: Sign predictions for log-abs values (batch_size, 3)
            real_sign_pred: Sign predictions for real values (batch_size, 3)
            ft_cal: Calibration factors (batch_size, 3) - ones if use_finetune=False
        """
        output = self.network(x)
        ln10 = np.log(10.0)

        # Step 1: Convert network output to magnitude predictions
        if self.use_internal_sign:
            # Network outputs: [sign_x, log_x, sign_v, log_v, sign_a, log_a]
            sign_x = torch.tanh(output[:, 0:1])
            log_x = output[:, 1:2]
            sign_v = torch.tanh(output[:, 2:3])
            log_v = output[:, 3:4]
            sign_a = torch.tanh(output[:, 4:5])
            log_a = output[:, 5:6]

            x_pred_base = sign_x * torch.exp(log_x * ln10)
            v_pred_base = sign_v * torch.exp(log_v * ln10)
            a_pred_base = sign_a * torch.exp(log_a * ln10)
        else:
            # Network outputs directly in real space: [x_t, v_t, a_t]
            x_pred_base = output[:, 0:1]
            v_pred_base = output[:, 1:2]
            a_pred_base = output[:, 2:3]

        # Step 2: Compute base magnitude predictions
        mag_preds = torch.cat([x_pred_base, v_pred_base, a_pred_base], dim=1)

        # Step 3: Apply logabs sign network
        logabs_sign_input = torch.cat([x, mag_preds.detach()], dim=1)  # Shape: [batch, 9]
        logabs_sign_pred = self.logabs_sign_network(logabs_sign_input)
        self.logabs_last_sign_probs = self.logabs_sign_network.last_sign_probs

        # Step 4: Apply real sign network (uses only params as input)
        real_sign_pred = self.real_sign_network(x)  # Shape: [batch, 3]
        self.real_last_sign_probs = self.real_sign_network.last_sign_probs

        # Step 5: Compute calibration factors
        if self.use_finetune:
            finetune_input = torch.cat([x, self.logabs_last_sign_probs.detach(), mag_preds.detach()], dim=1)  # Shape: [batch, 12]

            ft_cal_raw = self.finetune_network(finetune_input)/self.finetune_scale  # Shape: [batch, 3]
            ft_cal = ln10 * torch.tanh(ft_cal_raw)
            # ft_cal = torch.zeros_like(mag_preds)
        else:
            ft_cal = torch.zeros_like(mag_preds)

        # Return magnitude, both sign predictions, and calibration factors separately
        return mag_preds, logabs_sign_pred, real_sign_pred, ft_cal


class SignBCELoss(nn.Module):
    """Sign BCE loss - uses binary cross entropy for sign classification

    Simple helper class used internally by MSELoss for sign prediction.
    Does not inherit from BaseLossComponent as it's not a standalone loss component.
    """

    def __init__(self, weight=1.0):
        super().__init__()
        self.weight = weight
        self.criterion = nn.BCELoss()

    def forward(self, sigmoid_probs, targets):
        """
        Compute sign BCE between sigmoid probabilities and target signs

        Args:
            sigmoid_probs: (batch_size, num_outputs) - Sigmoid probabilities in [0, 1]
            targets: (batch_size, num_outputs) - Target signed values

        Returns:
            loss: Scalar tensor - Sum of BCE losses for all outputs
        """
        # Convert target signs to binary labels: {-1, +1} -> {0, 1}
        # Zeros are treated as positive (label=1.0)
        # Use targets.dtype to match the dtype (float32 or float64)
        target_signs = torch.sign(targets).to(dtype=targets.dtype)  # Shape: [batch, num_outputs], values: -1, 0, +1
        labels = (target_signs >= 0).to(dtype=targets.dtype)  # Shape: [batch, num_outputs], values: 0 or 1

        # Compute BCE for each output and sum
        num_outputs = sigmoid_probs.shape[1]
        total_bce_loss = sum(
            self.criterion(sigmoid_probs[:, i], labels[:, i])
            for i in range(num_outputs)
        )

        return self.weight * total_bce_loss


class BaseLossComponent(ABC, nn.Module):
    """Abstract base class for loss components"""

    def __init__(self, weight=1.0, name="base"):
        super().__init__()
        self.weight = weight
        self.name = name
        self.enabled = weight > 0

    @abstractmethod
    def compute(self, predictions, targets, **kwargs):
        pass

    def forward(self, predictions, targets, **kwargs):
        if not self.enabled:
            return torch.tensor(0.0, device=predictions.device)
        loss = self.compute(predictions, targets, **kwargs)
        return self.weight * loss

    def __repr__(self):
        status = "✓" if self.enabled else "✗"
        return f"{self.name:20s}: weight={self.weight:.3f} {status}"



class MSELoss(BaseLossComponent):
    """Mean Squared Error loss between predictions and targets with dual sign BCE losses and ft_cal loss"""

    def __init__(self, weight=1.0, use_relative=False, use_log=False, sign_bce_weight=1.0, real_sign_bce_weight=1.0, ft_cal_weight=1.0, use_finetune_loss=True):
        super().__init__(weight=weight, name="MSE Loss")
        self.use_relative = use_relative
        self.use_log = use_log
        self.ft_cal_weight = ft_cal_weight
        self.use_finetune_loss = use_finetune_loss
        # Create SignBCELoss instances for both logabs and real sign loss computation
        self.logabs_sign_bce_loss = SignBCELoss(weight=sign_bce_weight) if sign_bce_weight > 0 else None
        self.real_sign_bce_loss = SignBCELoss(weight=real_sign_bce_weight) if real_sign_bce_weight > 0 else None
        # Create L1Loss for ft_cal calibration loss (MAE)
        self.ft_cal_criterion = nn.L1Loss()

    def compute(self, predictions, targets, logabs_sigmoid_probs=None,
                real_sign_probs=None, ft_cal=None, **kwargs):
        """
        Compute MSE between predictions and targets with dual sign BCE losses and ft_cal loss

        Args:
            predictions: (batch_size, 3) - mag_preds (positive magnitudes)
            targets: (batch_size, 6) - complete targets [real_signs (0-2), logabs_values (3-5)]
            logabs_sigmoid_probs: (batch_size, 3) - logabs_sign_probs (sigmoid probabilities for logabs signs)
            real_sign_probs: (batch_size, 3) - real_sign_probs (sigmoid probabilities for real signs)
            ft_cal: (batch_size, 3) - calibration factors from finetune network

        Returns:
            loss: Scalar tensor
        """
        eps = 1e-12

        if self.use_log:
            # Extract logabs targets (columns 3-5) for magnitude loss
            logabs_targets = targets[:, 3:]  # [logabs_x, logabs_v, logabs_a]

            # Compute magnitude loss (log-space MSE)
            # predictions are already mag_preds (positive magnitudes, no abs needed)
            log_predictions = torch.log(torch.abs(predictions) + eps)
            log_targets = torch.log(torch.abs(logabs_targets) + eps)
            #Add only in PINN
            # log_diff = torch.abs(torch.abs(logabs_targets) - torch.abs(predictions))
            if self.use_relative:
                # Relative log-space MSE for magnitudes
                magnitude_loss = torch.mean(((log_predictions - log_targets) ** 2) / (torch.square(log_targets) + eps))
                # magnitude_loss += log_diff.mean()  # Add log difference term for relative loss
            else:
                # Absolute log-space MSE for magnitudes
                magnitude_loss = torch.mean((log_predictions - log_targets) ** 2)
                # magnitude_loss += log_diff.mean()  # Add log difference term for absolute loss
                # MSE on torch.abs(logabs_targets)  and torch.abs(predictions) to directly optimize magnitudes in log-space
                # magnitude_loss = self.ft_cal_criterion(torch.abs(predictions), torch.abs(logabs_targets))
            # Compute logabs sign BCE loss
            logabs_sign_bce_value = 0.0
            if self.logabs_sign_bce_loss is not None and logabs_sigmoid_probs is not None:
                logabs_sign_bce_value = self.logabs_sign_bce_loss(logabs_sigmoid_probs, logabs_targets)

            # Compute real sign BCE loss
            real_sign_bce_value = 0.0
            if self.real_sign_bce_loss is not None and real_sign_probs is not None:
                real_sign_targets = targets[:, :3]  # Extract real signs (columns 0-2)
                real_sign_bce_value = self.real_sign_bce_loss(real_sign_probs, real_sign_targets)

            # Compute ft_cal loss (additive calibration in log-space)
            ft_cal_loss_value = 0.0
            if self.use_finetune_loss and self.ft_cal_weight > 0 and ft_cal is not None:
                # Use torch.sign on logabs_sign_probs (shifted to [-1, 1] range)
                # logabs_sigmoid_probs is in [0, 1], shift to [-0.5, 0.5] then use sign
                logabs_sign = torch.sign(logabs_sigmoid_probs - 0.5)

                # Additive calibration: calibrated = signed_mag_preds + ft_cal
                # signed_mag_preds = (logabs_sign * predictions).detach()
                unsigned_mag_preds = predictions.detach()
                calibrated_preds = unsigned_mag_preds + ft_cal  # Only ft_cal has gradient

                # MAE loss using nn.L1Loss
                ft_cal_loss_value = self.ft_cal_criterion(torch.abs(calibrated_preds), torch.abs(logabs_targets))
                # ft_cal_loss_value = self.ft_cal_criterion(torch.abs(calibrated_preds)/torch.abs(logabs_targets), logabs_targets*0 + 1.0)
                ft_cal_loss_value = self.ft_cal_weight * ft_cal_loss_value

            # Store component losses for detailed reporting
            self.last_magnitude_loss = magnitude_loss.item()
            self.last_logabs_sign_bce_loss = logabs_sign_bce_value.item() if isinstance(logabs_sign_bce_value, torch.Tensor) else logabs_sign_bce_value
            self.last_real_sign_bce_loss = real_sign_bce_value.item() if isinstance(real_sign_bce_value, torch.Tensor) else real_sign_bce_value
            self.last_ft_cal_loss = ft_cal_loss_value.item() if isinstance(ft_cal_loss_value, torch.Tensor) else ft_cal_loss_value

            # Combine all losses: magnitude + logabs sign + real sign + ft_cal
            loss = magnitude_loss + logabs_sign_bce_value + real_sign_bce_value + ft_cal_loss_value
        else:
            # Standard MSE (not log-space)
            if self.use_relative:
                # Relative MSE: normalized by target magnitude
                loss = torch.mean(((predictions - targets) ** 2) / (torch.square(targets) + eps))
            else:
                # Absolute MSE
                loss = torch.mean((predictions - targets) ** 2)

        return loss


class ResidualLoss(BaseLossComponent):
    """Physics residual loss for vibration: m*a + c*v + k*x = 0

    Uses manual denormalization to preserve gradients in log-normalized space.
    """

    def __init__(self, weight=1.0, use_relative=False):
        super().__init__(weight=weight, name="Residual Loss")
        self.use_relative = use_relative

    def compute(self, predictions, targets, inputs_real=None, output_normalizer=None, **kwargs):
        """
        Enforce vibration physics equation: m*a + c*v + k*x = 0

        Args:
            predictions: (batch_size, 6) - [real_signs (0-2), logabs_values (3-5)] in NORMALIZED log-space
            targets: Not used (for signature compatibility)
            inputs_real: (batch_size, 6) - [m, zeta, k, t, x0, v0] in real space
            output_normalizer: Normalizer instance for manual denormalization
        """
        if inputs_real is None:
            raise ValueError("ResidualLoss requires inputs_real to be provided")
        if output_normalizer is None:
            raise ValueError("ResidualLoss requires output_normalizer to be provided")

        # Manual denormalization to preserve gradients
        real_signs = predictions[:, :3]  # (batch_size, 3)
        logabs_normalized = predictions[:, 3:]  # (batch_size, 3)

        # Extract normalizer statistics as tensors (vectorized)
        device = predictions.device
        dtype = predictions.dtype

        # Create tensors for log_mean and log_std for [x, v, a]
        log_mean = torch.tensor([
            output_normalizer.log_mean['x'],
            output_normalizer.log_mean['v'],
            output_normalizer.log_mean['a']
        ], device=device, dtype=dtype)  # (3,)

        log_std = torch.tensor([
            output_normalizer.log_std['x'],
            output_normalizer.log_std['v'],
            output_normalizer.log_std['a']
        ], device=device, dtype=dtype)  # (3,)

        # Denormalize logabs values (vectorized, preserves gradients)
        logabs_denorm = logabs_normalized * log_std + log_mean  # (batch_size, 3)

        # Convert to real space: real_value = sign * 10^logabs (vectorized)
        ln10 = torch.tensor(np.log(10.0), device=device, dtype=dtype)
        # real_values = torch.sign(real_signs) * torch.exp(logabs_denorm * ln10)  # (batch_size, 3)
        real_values = SignWithHardTanh.apply(real_signs) * torch.exp(logabs_denorm * ln10)  # (batch_size, 3)

        # Extract x, v, a predictions
        x_pred = real_values[:, 0]
        v_pred = real_values[:, 1]
        a_pred = real_values[:, 2]

        # Extract real-space parameters
        m = inputs_real[:, 0]
        zeta = inputs_real[:, 1]
        k = inputs_real[:, 2]

        # Compute damping coefficient
        c = 2 * zeta * torch.sqrt(m * k)

        eps = 1e-12

        # Physics residual: m*a + c*v + k*x = 0
        residual = m * a_pred + c * v_pred + k * x_pred

        if self.use_relative:
            # Scale-invariant relative residual
            # Denormalize targets to get a_target for scaling
            target_real_signs = targets[:, :3]
            target_logabs_normalized = targets[:, 3:]
            target_logabs_denorm = target_logabs_normalized * log_std + log_mean
            target_real_values = target_real_signs * torch.exp(target_logabs_denorm * ln10)
            a_target = target_real_values[:, 2].detach()  # Detach target to prevent gradient issues

            scale = torch.abs(m * a_target) + eps
            residual = residual / scale

        # Log-robust loss: mean(log(residual^2 + 1))
        return torch.mean(torch.log(torch.square(residual) + 1))


class InitialConditionLoss(BaseLossComponent):
    """Initial condition loss: x(0)=x0, v(0)=v0

    Self-contained: calls model internally with a surrogate t value instead of
    requiring a pre-built t=0 batch from the training loop.

    use_log=True  (log-normalized t): t=0 is -inf in log space, so a surrogate
                  t_norm = floor(normalize(log10(t_threshold)) - (-log10(t_threshold)))
                  is substituted. Cached at init, never recomputed per batch.

    use_log=False (linearly-normalized t): t=0 normalizes normally to a finite value;
                  same surrogate mechanism, cached at init.

    Both modes compare in real space (Option A): model outputs are manually
    denormalized via cached alpha/beta constants to preserve gradients.
    """

    def __init__(self, weight=1.0, use_log=False, use_relative=True,
                 model=None, inputs_normalizer=None, outputs_normalizer=None,
                 t_threshold=1e-6, eps=1e-12):
        super().__init__(weight=weight, name="Initial Cond Loss")
        self.use_log = use_log
        self.use_relative = use_relative
        self.model = model
        self.eps = eps
        self.t_index  = 3   # [m, zeta, k, t, x0, v0]
        self.x0_index = 4
        self.v0_index = 5

        # ── t_override_norm: cached once, never recomputed per batch ─────
        if use_log:
            t_threshold_log    = np.log10(t_threshold)         # e.g. -6.0
            t_surrogate_offset = -t_threshold_log              # e.g. +6.0 (derived)
            t_raw_norm = (t_threshold_log - inputs_normalizer.log_mean['t']) \
                         / inputs_normalizer.log_std['t']
            if inputs_normalizer.map_range is not None:
                z_min = inputs_normalizer.original_z_min['t']
                z_max = inputs_normalizer.original_z_max['t']
                lo, hi = inputs_normalizer.map_range
                t_raw_norm = lo + (t_raw_norm - z_min) / (z_max - z_min) * (hi - lo)
            self.t_override_norm = float(np.floor(t_raw_norm - t_surrogate_offset))
        else:
            # Linear normalization path
            t_zero_norm = (0.0 - inputs_normalizer.linear_mean['t']) \
                          / inputs_normalizer.linear_std['t']
            if inputs_normalizer.map_range is not None:
                z_min = inputs_normalizer.original_z_min['t']
                z_max = inputs_normalizer.original_z_max['t']
                lo, hi = inputs_normalizer.map_range
                t_zero_norm = lo + (t_zero_norm - z_min) / (z_max - z_min) * (hi - lo)
            self.t_override_norm = float(t_zero_norm)

        print(f"[InitialConditionLoss] t_override_norm = {self.t_override_norm:.4f} "
              f"({'log surrogate' if use_log else 'linear t=0'})")

        # ── output denorm constants: cached, gradient-safe ───────────────
        alpha_x = outputs_normalizer.log_mean['x']
        beta_x  = outputs_normalizer.log_std['x']
        self.alpha_x = float(generalize_alpha(outputs_normalizer, alpha_x, beta_x, 'x'))
        self.beta_x  = float(generalize_beta( outputs_normalizer, alpha_x, beta_x, 'x'))
        alpha_v = outputs_normalizer.log_mean['v']
        beta_v  = outputs_normalizer.log_std['v']
        self.alpha_v = float(generalize_alpha(outputs_normalizer, alpha_v, beta_v, 'v'))
        self.beta_v  = float(generalize_beta( outputs_normalizer, alpha_v, beta_v, 'v'))

    def compute(self, predictions, targets,
                inputs_normalized=None, inputs_real=None, ft_cal=None):
        """
        Args:
            predictions: unused (model called internally)
            targets:     unused
            inputs_normalized: (N, 6) normalized batch inputs — t column is overridden
            inputs_real:       (N, 6) real-space inputs — x0 at col 4, v0 at col 5
            ft_cal:            (N, 3) finetune calibration from current training step
                               (used for phase detection: zeros → Phase 1)
        """
        device = inputs_normalized.device
        dtype  = inputs_normalized.dtype
        ln10   = torch.log(torch.tensor(10.0, device=device, dtype=dtype))

        # ── build IC input batch (override t, detach to avoid cross-graph issues) ──
        ic_inputs = inputs_normalized.clone().detach()
        ic_inputs[:, self.t_index] = self.t_override_norm

        # ── model forward (gradient-enabled so loss trains the model) ────
        mag_preds_ic, logabs_sign_ic, real_sign_ic, ft_cal_ic = self.model(ic_inputs)

        # ── phase-aware ft_cal: zeros in Phase 1 ─────────────────────────
        is_phase1 = ft_cal is None or torch.all(torch.abs(ft_cal) < 1e-12)
        ft_cal_for_ic = torch.zeros_like(ft_cal_ic) if is_phase1 else ft_cal_ic

        # ── signed logabs in normalized log space ─────────────────────────
        signed_logabs = torch.sign(logabs_sign_ic) * (mag_preds_ic + ft_cal_for_ic)

        # ── manual denorm to real space (keeps gradient graph intact) ────
        log_x  = signed_logabs[:, 0] * self.beta_x + self.alpha_x
        log_v  = signed_logabs[:, 1] * self.beta_v + self.alpha_v
        mag_x  = torch.exp(log_x * ln10)
        mag_v  = torch.exp(log_v * ln10)
        x_pred = torch.sign(real_sign_ic[:, 0]) * mag_x
        v_pred = torch.sign(real_sign_ic[:, 1]) * mag_v

        # ── IC ground truth from real-space inputs ────────────────────────
        x0_real = inputs_real[:, self.x0_index]
        v0_real = inputs_real[:, self.v0_index]

        # ── loss ──────────────────────────────────────────────────────────
        if self.use_log:

            #Option1: log-log loss on magnitudes, like residual loss
            # log(r) computed as log-ratio to avoid division; symmetric, scale-invariant
            # Scale-invariant log-ratio loss: log(log10(|pred/target|)² + 1)
            # Matches residual's log-robust pattern; no division, bounded growth
            # log_ratio_x = torch.log10(torch.abs(x_pred) + self.eps) \
            #             - torch.log10(torch.abs(x0_real) + self.eps)
            # log_ratio_v = torch.log10(torch.abs(v_pred) + self.eps) \
            #             - torch.log10(torch.abs(v0_real) + self.eps)
            # loss_x = torch.mean(torch.log(log_ratio_x ** 2 + 1))
            # loss_v = torch.mean(torch.log(log_ratio_v ** 2 + 1))

            # Option2 2*cosh(log|pred/target|) - 2  =  r + 1/r - 2
            log_ratio_x = (torch.log10(torch.abs(x_pred) + self.eps)
                         - torch.log10(torch.abs(x0_real) + self.eps)) * ln10
            log_ratio_v = (torch.log10(torch.abs(v_pred) + self.eps)
                         - torch.log10(torch.abs(v0_real) + self.eps)) * ln10
            loss_x = torch.mean(2 * torch.cosh(log_ratio_x) - 2)
            loss_v = torch.mean(2 * torch.cosh(log_ratio_v) - 2)
        elif self.use_relative:
            loss_x = torch.mean((x_pred - x0_real) ** 2 / (x0_real ** 2 + self.eps))
            loss_v = torch.mean((v_pred - v0_real) ** 2 / (v0_real ** 2 + self.eps))
        else:
            loss_x = torch.mean((x_pred - x0_real) ** 2)
            loss_v = torch.mean((v_pred - v0_real) ** 2)

        return loss_x + loss_v


class ConsistencyLoss_auto_diff(BaseLossComponent):
    """
    Derivative consistency using VERIFIED formulas for log-normalized space training.

    Ensures v=dx/dt and a=dv/dt using autograd on log-normalized model outputs.
    Uses verified transformation formulas adapted from Exp implementation.

    Key features:
    - Model outputs UNSIGNED magnitudes (positive from Softplus) in normalized log-space
    - Computes TARGET derivatives (ground truth) from dataset targets
    - Computes PREDICTION-based derivatives from model's v' and a' outputs
    - Computes AUTOGRAD-based derivatives (spatial gradients ∂x'/∂t')
    - Compares BOTH autograd and prediction-based derivatives against target derivatives
    """

    def __init__(self, weight=1.0, model=None, t_threshold=1e-6, use_log=False, input_grad_outside=False):
        super().__init__(weight=weight, name="Consistency Loss (auto)")
        self.model = model
        self.t_threshold = t_threshold
        self.use_log = use_log
        self.input_grad_outside = input_grad_outside

    def set_model(self, model):
        """Set the model reference for gradient computation"""
        self.model = model

    def compute(self, predictions, targets, inputs=None, inputs_normalizer=None,
                outputs_normalizer=None, ft_cal=None, valid_mask=None, **kwargs):
        """
        Enforces derivative consistency using verified formulas for log-normalized training.

        Args:
            predictions: If input_grad_outside=True, this is mag_preds from training loop
                        If input_grad_outside=False, not used (we compute fresh predictions)
            targets: (batch_size, 6) - [real_sign_x, real_sign_v, real_sign_a, x', v', a']
                     where [:, 0:3] are real space signs (±1)
                     and [:, 3:6] are SIGNED normalized log-space values (can be ±)
            inputs: (batch_size, 6) - [m, zeta, k, t, x0, v0] NORMALIZED
            inputs_normalizer: Normalizer for inputs (contains t normalization params)
            outputs_normalizer: Normalizer for outputs (contains x, v, a normalization params)
            ft_cal: (batch_size, 3) - Finetune calibration outputs [ft_x, ft_v, ft_a] in normalized log-space
            valid_mask: If input_grad_outside=True, boolean mask for valid samples (t_real > threshold)
                       If input_grad_outside=False, not used (computed internally)

        Returns:
            loss: Scalar tensor measuring derivative consistency error
        """
        if inputs_normalizer is None or outputs_normalizer is None:
            raise ValueError("Both inputs_normalizer and outputs_normalizer must be provided")

        device = inputs.device
        dtype = inputs.dtype
        eps = 1e-12

        # Get normalization parameters and adjust for map_range if applied
        alpha_x = outputs_normalizer.log_mean['x']
        beta_x  = outputs_normalizer.log_std['x']
        alpha_x = generalize_alpha(outputs_normalizer, alpha_x, beta_x, 'x')
        beta_x  = generalize_beta( outputs_normalizer, alpha_x, beta_x, 'x')
        alpha_v = outputs_normalizer.log_mean['v']
        beta_v  = outputs_normalizer.log_std['v']
        alpha_v = generalize_alpha(outputs_normalizer, alpha_v, beta_v, 'v')
        beta_v  = generalize_beta( outputs_normalizer, alpha_v, beta_v, 'v')
        alpha_a = outputs_normalizer.log_mean['a']
        beta_a  = outputs_normalizer.log_std['a']
        alpha_a = generalize_alpha(outputs_normalizer, alpha_a, beta_a, 'a')
        beta_a  = generalize_beta( outputs_normalizer, alpha_a, beta_a, 'a')
        t_alpha = inputs_normalizer.log_mean['t']
        t_beta  = inputs_normalizer.log_std['t']
        t_alpha = generalize_alpha(inputs_normalizer, t_alpha, t_beta, 't')
        t_beta  = generalize_beta( inputs_normalizer, t_alpha, t_beta, 't')
        ln10 = torch.log(torch.tensor(10.0, device=device, dtype=dtype))

        # ======================
        # MODE SELECTION: Compute mag_x, mag_v, mag_a and their derivatives
        # ======================
        if self.input_grad_outside:
            # MODE 1: Gradients computed in training loop
            # Use mag_preds from training loop (predictions parameter)
            # Use valid_mask passed from training loop

            if predictions is None:
                raise ValueError("When input_grad_outside=True, predictions (mag_preds) must be provided")
            if ft_cal is None:
                raise ValueError("When input_grad_outside=True, ft_cal must be provided")
            if valid_mask is None:
                raise ValueError("When input_grad_outside=True, valid_mask must be provided")

            # Filter to valid samples
            # Following Exp pattern: use valid_mask to filter, but compute gradients w.r.t. full inputs
            mag_preds_valid = predictions[valid_mask]
            targets_valid = targets[valid_mask]
            ft_cal_valid = ft_cal[valid_mask]

            # For diagnostic purposes - will be used after gradient computation
            N = predictions.shape[0]
            inputs_valid = inputs[:N][valid_mask]

            # Compute t_real for valid samples (vibration: t is at index 3)
            t_normalized_valid = inputs_valid[:, 3]
            t_real_valid = torch.exp((t_beta * t_normalized_valid + t_alpha) * ln10)

            # STEP 1: Get unsigned magnitude predictions and compute derivatives separately
            # Detect phase
            is_phase1 = torch.all(torch.abs(ft_cal_valid) < eps)

            # STEP 1b: Compute derivatives separately, then sum
            # Always compute derivatives of mag_preds components
            dx_mag_dt_full = torch.autograd.grad(
                outputs=mag_preds_valid[:, 0],
                inputs=inputs,
                grad_outputs=torch.ones_like(mag_preds_valid[:, 0]),
                create_graph=True,
                retain_graph=True,
                allow_unused=True
            )[0]

            # Check for None (following Exp pattern)
            if dx_mag_dt_full is None:
                raise RuntimeError(
                    f"Gradient computation failed for dx_mag_dt! "
                    f"Ensure inputs.requires_grad=True is set BEFORE forward pass. "
                    f"inputs.requires_grad={inputs.requires_grad}, "
                    f"mag_preds.requires_grad={mag_preds_valid.requires_grad}, "
                    f"mag_preds.grad_fn={mag_preds_valid.grad_fn}"
                )

            # Extract first N rows (corresponding to predictions), then filter by valid_mask
            dx_mag_dt = dx_mag_dt_full[:N][valid_mask, 3]  # Gradient w.r.t. t_normalized (index 3 for vibration)

            dv_mag_dt_full = torch.autograd.grad(
                outputs=mag_preds_valid[:, 1],
                inputs=inputs,
                grad_outputs=torch.ones_like(mag_preds_valid[:, 1]),
                create_graph=True,
                retain_graph=True,
                allow_unused=True
            )[0]

            # Check for None (following Exp pattern)
            if dv_mag_dt_full is None:
                raise RuntimeError(
                    f"Gradient computation failed for dv_mag_dt! "
                    f"Ensure inputs.requires_grad=True is set BEFORE forward pass. "
                    f"inputs.requires_grad={inputs.requires_grad}, "
                    f"mag_preds.requires_grad={mag_preds_valid.requires_grad}, "
                    f"mag_preds.grad_fn={mag_preds_valid.grad_fn}"
                )

            # Extract first N rows (corresponding to predictions), then filter by valid_mask
            dv_mag_dt = dv_mag_dt_full[:N][valid_mask, 3]  # Gradient w.r.t. t_normalized (index 3 for vibration)

            if is_phase1:
                # Phase 1: Only use mag derivatives (ft_cal is zeros)
                dx_prime_dt_prime = torch.abs(dx_mag_dt)
                dv_prime_dt_prime = torch.abs(dv_mag_dt)
            else:
                # Phase 2: Compute ft_cal derivatives and add them
                dx_ft_dt_full = torch.autograd.grad(
                    outputs=ft_cal_valid[:, 0],
                    inputs=inputs,
                    grad_outputs=torch.ones_like(ft_cal_valid[:, 0]),
                    create_graph=True,
                    retain_graph=True,
                    allow_unused=True
                )[0]

                # Check for None (following Exp pattern)
                if dx_ft_dt_full is None:
                    raise RuntimeError(
                        f"Gradient computation failed for dx_ft_dt! "
                        f"Ensure inputs.requires_grad=True is set BEFORE forward pass. "
                        f"inputs.requires_grad={inputs.requires_grad}, "
                        f"ft_cal.requires_grad={ft_cal_valid.requires_grad}, "
                        f"ft_cal.grad_fn={ft_cal_valid.grad_fn}"
                    )

                # Extract first N rows (corresponding to predictions), then filter by valid_mask
                dx_ft_dt = dx_ft_dt_full[:N][valid_mask, 3]  # Gradient w.r.t. t_normalized (index 3 for vibration)

                dv_ft_dt_full = torch.autograd.grad(
                    outputs=ft_cal_valid[:, 1],
                    inputs=inputs,
                    grad_outputs=torch.ones_like(ft_cal_valid[:, 1]),
                    create_graph=True,
                    retain_graph=True,
                    allow_unused=True
                )[0]

                # Check for None (following Exp pattern)
                if dv_ft_dt_full is None:
                    raise RuntimeError(
                        f"Gradient computation failed for dv_ft_dt! "
                        f"Ensure inputs.requires_grad=True is set BEFORE forward pass. "
                        f"inputs.requires_grad={inputs.requires_grad}, "
                        f"ft_cal.requires_grad={ft_cal_valid.requires_grad}, "
                        f"ft_cal.grad_fn={ft_cal_valid.grad_fn}"
                    )

                # Extract first N rows (corresponding to predictions), then filter by valid_mask
                dv_ft_dt = dv_ft_dt_full[:N][valid_mask, 3]  # Gradient w.r.t. t_normalized (index 3 for vibration)

                # Sum derivatives (linearity: d(f+g)/dt = df/dt + dg/dt)
                dx_prime_dt_prime = torch.abs(dx_mag_dt) + dx_ft_dt
                dv_prime_dt_prime = torch.abs(dv_mag_dt) + dv_ft_dt
        else:
            # MODE 2: Gradients computed inside consistency loss (current approach)
            # Call model internally with inputs_with_grad

            if self.model is None:
                raise ValueError("When input_grad_outside=False, model must be set via set_model()")
            if ft_cal is None:
                raise ValueError("When input_grad_outside=False, ft_cal must be provided")

            # Denormalize inputs to get t_real (vibration: t is at index 3)
            t_normalized = inputs[:, 3]
            t_real = torch.exp((t_beta * t_normalized + t_alpha) * ln10)

            # Filter out samples where t_real ≈ 0 (to avoid division by zero)
            valid_mask = t_real > self.t_threshold

            if valid_mask.sum() == 0:
                return torch.tensor(0.0, device=device)

            # Filter to valid samples
            inputs_valid = inputs[valid_mask]
            targets_valid = targets[valid_mask]
            t_real_valid = t_real[valid_mask]
            ft_cal_valid = ft_cal[valid_mask]

            # Enable gradient tracking on inputs
            inputs_with_grad = inputs_valid.clone().requires_grad_(True)

            # Forward pass to get model outputs
            # Model returns tuple: (mag_preds, logabs_sign_pred, real_sign_pred, ft_cal_from_model)
            mag_preds_internal, _, _, ft_preds_internal = self.model(inputs_with_grad)

            # STEP 1: Get unsigned magnitude predictions and compute derivatives separately
            # Detect phase
            is_phase1 = torch.all(torch.abs(ft_cal_valid) < eps)

            # STEP 1b: Compute derivatives separately, then sum
            # Always compute derivatives of mag_preds_internal components
            dx_mag_dt = torch.autograd.grad(
                outputs=mag_preds_internal[:, 0],
                inputs=inputs_with_grad,
                grad_outputs=torch.ones_like(mag_preds_internal[:, 0]),
                create_graph=True,
                retain_graph=True,
                allow_unused=True
            )[0][:, 3]  # Gradient w.r.t. t_normalized (index 3 for vibration)

            dv_mag_dt = torch.autograd.grad(
                outputs=mag_preds_internal[:, 1],
                inputs=inputs_with_grad,
                grad_outputs=torch.ones_like(mag_preds_internal[:, 1]),
                create_graph=True,
                retain_graph=True,
                allow_unused=True
            )[0][:, 3]  # Gradient w.r.t. t_normalized (index 3 for vibration)

            if is_phase1:
                # Phase 1: Only use mag derivatives (finetune network untrained)
                dx_prime_dt_prime = torch.abs(dx_mag_dt)
                dv_prime_dt_prime = torch.abs(dv_mag_dt)
            else:
                # Phase 2: Compute ft_preds derivatives and add them
                dx_ft_dt = torch.autograd.grad(
                    outputs=ft_preds_internal[:, 0],
                    inputs=inputs_with_grad,
                    grad_outputs=torch.ones_like(ft_preds_internal[:, 0]),
                    create_graph=True,
                    retain_graph=True,
                    allow_unused=True
                )[0][:, 3]  # Gradient w.r.t. t_normalized (index 3 for vibration)

                dv_ft_dt = torch.autograd.grad(
                    outputs=ft_preds_internal[:, 1],
                    inputs=inputs_with_grad,
                    grad_outputs=torch.ones_like(ft_preds_internal[:, 1]),
                    create_graph=True,
                    retain_graph=True,
                    allow_unused=True
                )[0][:, 3]  # Gradient w.r.t. t_normalized (index 3 for vibration)

                # Sum derivatives (linearity: d(f+g)/dt = df/dt + dg/dt)
                dx_prime_dt_prime = torch.abs(dx_mag_dt) + dx_ft_dt
                dv_prime_dt_prime = torch.abs(dv_mag_dt) + dv_ft_dt

        # ==============================================
        # STEP 1: Compute TARGET derivatives (ground truth from dataset)
        # ==============================================
        # Extract targets
        x_target = targets_valid[:, 3]  # x' target (SIGNED normalized log-space)
        v_target = targets_valid[:, 4]  # v' target (SIGNED normalized log-space)
        a_target = targets_valid[:, 5]  # a' target (SIGNED normalized log-space)

        # Denormalize targets to real space
        x_target_real = torch.exp((beta_x * x_target + alpha_x) * ln10)
        v_target_real = torch.exp((beta_v * v_target + alpha_v) * ln10)
        a_target_real = torch.exp((beta_a * a_target + alpha_a) * ln10)

        # Compute common factors from targets (inverse of theory formula)
        common_factor_v_target = (beta_x / t_beta) * (x_target_real / (t_real_valid + eps))
        common_factor_a_target = (beta_v / t_beta) * (v_target_real / (t_real_valid + eps))

        # Invert formula to get target derivatives
        # v_theory = |common_factor * dx'/dt'| → |dx'/dt'| = v / common_factor
        dx_dt_target = torch.abs(v_target_real / (common_factor_v_target + eps))
        dv_dt_target = torch.abs(a_target_real / (common_factor_a_target + eps))

        # ==============================================
        # STEP 2: Compute PREDICTION-based derivatives (from model's v' and a' outputs)
        # ==============================================
        # Extract signs from targets (targets contain SIGNED values)
        # targets[:, 3:6] are already SIGNED normalized log-space values
        logabs_targets = targets_valid[:, 3:]  # [x', v', a'] in normalized log-space (signed)
        logabs_sign = torch.sign(logabs_targets)  # Extract signs from logabs targets

        # Extract predicted v' and a' from model predictions and apply signs
        if self.input_grad_outside:
            # MODE 1: Use mag_preds_valid (unsigned magnitudes)
            v_prime_predict = mag_preds_valid[:, 1] * logabs_sign[:, 1]  # Apply sign to v'
            a_prime_predict = mag_preds_valid[:, 2] * logabs_sign[:, 2]  # Apply sign to a'
        else:
            # MODE 2: Use mag_preds_internal (unsigned magnitudes)
            v_prime_predict = mag_preds_internal[:, 1] * logabs_sign[:, 1]  # Apply sign to v'
            a_prime_predict = mag_preds_internal[:, 2] * logabs_sign[:, 2]  # Apply sign to a'

        # Denormalize predicted v' and a' to real space
        v_predict_real = torch.exp((beta_v * v_prime_predict + alpha_v) * ln10)
        a_predict_real = torch.exp((beta_a * a_prime_predict + alpha_a) * ln10)

        # Compute prediction-based derivatives using theory formula
        # Theory: v_theory = |common_factor_v * dx'/dt'| → |dx'/dt'| = v / common_factor_v
        dx_dt_predict = torch.abs(v_predict_real / (common_factor_v_target + eps))
        dv_dt_predict = torch.abs(a_predict_real / (common_factor_a_target + eps))

        # ==============================================
        # STEP 3: Use AUTOGRAD-based derivatives (from spatial gradients ∂x'/∂t')
        # ==============================================
        # Phase 1: dx_prime_dt_prime = torch.abs(dx_mag_dt)
        # Phase 2: dx_prime_dt_prime = torch.abs(dx_mag_dt) + dx_ft_dt
        model_dx_dt = dx_prime_dt_prime  # Use the phase-aware combined derivatives (from autograd)
        model_dv_dt = dv_prime_dt_prime

        # ======================
        # DERIVATIVE-BASED CONSISTENCY LOSS
        # ======================
        # STEP 9: Compute loss (log-space MSE or regular MSE based on use_log)
        # DUAL CONSISTENCY: Compare BOTH autograd derivatives AND prediction-based derivatives
        if self.use_log:
            # Autograd-based derivative loss (from ∂x'/∂t')
            v_consistency_loss_autograd = torch.mean(torch.log(torch.abs((model_dx_dt - dx_dt_target) / (dx_dt_target + eps)) + 1))
            a_consistency_loss_autograd = torch.mean(torch.log(torch.abs((model_dv_dt - dv_dt_target) / (dv_dt_target + eps)) + 1))

            # Prediction-based derivative loss (from predicted v' and a')
            v_consistency_loss_predict = torch.mean(torch.log(torch.abs((dx_dt_predict - dx_dt_target) / (dx_dt_target + eps)) + 1))
            a_consistency_loss_predict = torch.mean(torch.log(torch.abs((dv_dt_predict - dv_dt_target) / (dv_dt_target + eps)) + 1))

            # Combine both losses
            v_consistency_loss = v_consistency_loss_autograd + v_consistency_loss_predict
            a_consistency_loss = a_consistency_loss_autograd + a_consistency_loss_predict
        else:
            # Relative error MSE - comparing derivatives
            # Autograd-based derivative loss
            v_consistency_loss_autograd = torch.mean(((torch.abs(model_dx_dt) - torch.abs(dx_dt_target)) / torch.abs(model_dx_dt)) ** 2)
            a_consistency_loss_autograd = torch.mean(((torch.abs(model_dv_dt) - torch.abs(dv_dt_target)) / torch.abs(model_dv_dt)) ** 2)

            # Prediction-based derivative loss
            v_consistency_loss_predict = torch.mean(((torch.abs(dx_dt_predict) - torch.abs(dx_dt_target)) / torch.abs(dx_dt_predict)) ** 2)
            a_consistency_loss_predict = torch.mean(((torch.abs(dv_dt_predict) - torch.abs(dv_dt_target)) / torch.abs(dv_dt_predict)) ** 2)

            # Combine both losses
            v_consistency_loss = v_consistency_loss_autograd + v_consistency_loss_predict
            a_consistency_loss = a_consistency_loss_autograd + a_consistency_loss_predict

        # STEP 10: Total consistency loss
        total_loss = v_consistency_loss + a_consistency_loss

        return total_loss




# class ConsistencyLoss_finite_diff(BaseLossComponent):
#     """
#     Derivative consistency using finite differences

#     Computes 6 loss components:
#     1. v from x (finite diff) vs v_pred
#     2. a from v (finite diff) vs a_pred
#     3. a from x (finite diff, 2nd derivative) vs a_pred
#     4. v from x (finite diff) vs v_target
#     5. a from v (finite diff) vs a_target
#     6. a from x (finite diff, 2nd derivative) vs a_target
#     """

#     def __init__(self, weight=1.0, use_relative=False, t_threshold=1e-6,
#                  use_log=True,
#                  weight_v_x_pred=1.0,
#                  weight_a_v_pred=1.0,
#                  weight_a_x_pred=1.0,
#                  weight_v_x_target=1.0,
#                  weight_a_v_target=1.0,
#                  weight_a_x_target=1.0):
#         super().__init__(weight=weight, name="Consistency Loss (finite)")
#         self.use_relative = use_relative
#         self.t_threshold = t_threshold
#         self.use_log = use_log

#         # Individual component weights
#         self.weight_v_x_pred = weight_v_x_pred
#         self.weight_a_v_pred = weight_a_v_pred
#         self.weight_a_x_pred = weight_a_x_pred
#         self.weight_v_x_target = weight_v_x_target
#         self.weight_a_v_target = weight_a_v_target
#         self.weight_a_x_target = weight_a_x_target

#     def compute(self, predictions, targets, inputs, norm_params=None, inputs_real=None):
#         """
#         Compute finite difference consistency loss

#         Args:
#             predictions: (batch_size, 3) - [x, v, a] at time t
#             targets: (batch_size, 3) - [x, v, a] ground truth at time t
#             inputs: Not used
#             norm_params: Not used
#             inputs_real: (4*batch_size, 3) - predictions at perturbed times
#                         [t-2Δt, t-Δt, t+Δt, t+2Δt] stacked

#         Returns:
#             total_loss: Scalar tensor representing the natural log of the total loss.
#         """
#         if inputs_real is None:
#             raise ValueError("ConsistencyLoss_finite_diff requires inputs_real (outputs_dt) to be provided")

#         N = predictions.shape[0]
#         eps = 1e-10

#         # Extract predictions at different time points
#         # inputs_real is actually outputs_dt with shape (4N, 3)
#         outputs_dt = inputs_real

#         x_t = predictions[:, 0:1]
#         v_t = predictions[:, 1:2]
#         a_t = predictions[:, 2:3]

#         x_t_minus_minus = outputs_dt[0:N, 0:1]      # t - 2Δt
#         x_t_minus = outputs_dt[N:2*N, 0:1]          # t - Δt
#         x_t_plus = outputs_dt[2*N:3*N, 0:1]         # t + Δt
#         x_t_plus_plus = outputs_dt[3*N:4*N, 0:1]    # t + 2Δt

#         v_t_minus = outputs_dt[N:2*N, 1:2]          # t - Δt
#         v_t_plus = outputs_dt[2*N:3*N, 1:2]         # t + Δt

#         # Extract targets
#         # Debug: ensure targets is a tensor
#         if not isinstance(targets, torch.Tensor):
#             raise TypeError(f"targets must be a torch.Tensor, got {type(targets)}")

#         x_target = targets[:, 0:1]
#         v_target = targets[:, 1:2]
#         a_target = targets[:, 2:3]

#         t_delta = self.t_threshold

#         # Compute finite difference derivatives
#         # Component 1 & 4: v from x using central difference
#         v_fd_from_x = (x_t_plus - x_t_minus) / (2 * t_delta)

#         # Component 2 & 5: a from v using central difference
#         a_fd_from_v = (v_t_plus - v_t_minus) / (2 * t_delta)

#         # Component 3 & 6: a from x using second derivative
#         v_t_plus_temp = (x_t_plus_plus - x_t) / (2 * t_delta)
#         v_t_minus_temp = (x_t - x_t_minus_minus) / (2 * t_delta)
#         a_fd_from_x = (v_t_plus_temp - v_t_minus_temp) / (2 * t_delta)

#         # Compute squared differences for all 6 components
#         diff_1 = (v_fd_from_x - v_t) ** 2        # v from x vs v_pred
#         diff_2 = (a_fd_from_v - a_t) ** 2        # a from v vs a_pred
#         diff_3 = (a_fd_from_x - a_t) ** 2        # a from x vs a_pred
#         diff_4 = (v_fd_from_x - v_target) ** 2   # v from x vs v_target
#         diff_5 = (a_fd_from_v - a_target) ** 2   # a from v vs a_target
#         diff_6 = (a_fd_from_x - a_target) ** 2   # a from x vs a_target

#         # Apply normalization if use_relative is True
#         if self.use_relative:
#             diff_1 = diff_1 / (v_target ** 2 + eps)
#             diff_2 = diff_2 / (a_target ** 2 + eps)
#             diff_3 = diff_3 / (a_target ** 2 + eps)
#             diff_4 = diff_4 / (v_target ** 2 + eps)
#             diff_5 = diff_5 / (a_target ** 2 + eps)
#             diff_6 = diff_6 / (a_target ** 2 + eps)

#         # Compute weighted sum of all components
#         total_loss = (
#             self.weight_v_x_pred * torch.mean(diff_1) +
#             self.weight_a_v_pred * torch.mean(diff_2) +
#             self.weight_a_x_pred * torch.mean(diff_3) +
#             self.weight_v_x_target * torch.mean(diff_4) +
#             self.weight_a_v_target * torch.mean(diff_5) +
#             self.weight_a_x_target * torch.mean(diff_6)
#         )

#         # Apply log transformation if enabled
#         if self.use_log:
#             total_loss = torch.log(total_loss + eps)

#         return total_loss



class PINNLoss(nn.Module):
    """
    Dictionary-based PINN Loss with flexible argument passing

    Usage:
        # Configuration
        loss_config = {
            "MSE": {"weight": 0.3},
            "Residual": {"weight": 0.2, "use_relative": False},
            "InitialCondition": {"weight": 0.3, "use_log": True, "t_threshold": 1e-6},
            "Consistency": {"weight": 0.2, "t_threshold": 1e-6}
        }

        # To disable a loss, use any of:
        # 1. Omit from dict
        # 2. Set to None: "Consistency": None
        # 3. Set weight to 0: "Consistency": {"weight": 0}

        # Create loss function (pass normalizers for InitialConditionLoss)
        loss_fn = PINNLoss(model, loss_config,
                           inputs_normalizer=train_val_inputs_normalizer,
                           outputs_normalizer=train_val_targets_normalizer)

        # In training loop, prepare arguments for enabled losses
        loss_args = {}
        if loss_fn.has_loss("MSE"):
            loss_args["MSE"] = (outputs, targets)
        if loss_fn.has_loss("Residual"):
            loss_args["Residual"] = (outputs, inputs_real)
        if loss_fn.has_loss("Consistency"):
            loss_args["Consistency"] = (inputs, inputs_real, norm_params)
        if loss_fn.has_loss("InitialCondition"):
            loss_args["InitialCondition"] = (inputs, inputs_real, ft_cal)

        # Compute loss
        total_loss, loss_summary = loss_fn(loss_args)
    """

    def __init__(self, model, loss_config,
                 inputs_normalizer=None, outputs_normalizer=None):
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
            real_sign_bce_weight = config.get("real_sign_bce_weight", 1.0)
            ft_cal_weight = config.get("ft_cal_weight", 1.0)
            # Use model.use_finetune to determine if ft_cal_loss should be computed
            use_finetune_loss = getattr(model, 'use_finetune', False)
            self.loss_components["MSE"] = MSELoss(weight=weight, use_relative=use_relative, use_log=use_log, sign_bce_weight=sign_bce_weight, real_sign_bce_weight=real_sign_bce_weight, ft_cal_weight=ft_cal_weight, use_finetune_loss=use_finetune_loss)

        # Initialize Residual loss if requested
        if self._should_enable("Residual"):
            config = self.loss_config.get("Residual")
            use_relative = config.get("use_relative", False)
            weight = config.get("weight", 1.0)
            self.loss_components["Residual"] = ResidualLoss(weight=weight, use_relative=use_relative)

        # Initialize InitialCondition loss if requested
        if self._should_enable("InitialCondition"):
            config = self.loss_config.get("InitialCondition")
            self.loss_components["InitialCondition"] = InitialConditionLoss(
                weight=config.get("weight", 1.0),
                use_log=config.get("use_log", False),
                use_relative=config.get("use_relative", True),
                model=model,
                inputs_normalizer=inputs_normalizer,
                outputs_normalizer=outputs_normalizer,
                t_threshold=config.get("t_threshold", 1e-6),
                eps=config.get("eps", 1e-12),
            )

        # Initialize Consistency loss if requested
        if self._should_enable("Consistency"):
            config = self.loss_config.get("Consistency")
            t_threshold = config.get("t_threshold", 1e-6)
            weight = config.get("weight", 1.0)
            use_log = config.get("use_log", True)
            input_grad_outside = config.get("Input_grad_outside", False)

            # Only use auto-diff consistency loss (following Exp pattern)
            self.loss_components["Consistency"] = ConsistencyLoss_auto_diff(
                weight=weight, model=model, t_threshold=t_threshold, use_log=use_log,
                input_grad_outside=input_grad_outside
            )

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
            # Unpack 8 arguments: (mag_preds, targets, logabs_sign_probs, _, _, real_sign_probs, ft_cal, output_normalizer)
            mag_preds, targets, logabs_sign_probs, _, _, real_sign_probs, ft_cal, output_normalizer = loss_args["MSE"]
            mse_value = self.loss_components["MSE"](
                mag_preds, targets,
                logabs_sigmoid_probs=logabs_sign_probs,
                real_sign_probs=real_sign_probs,
                ft_cal=ft_cal
            )
            total_loss += mse_value
            loss_summary["mse_loss"] = mse_value.item()

            # Add magnitude and sign loss components for detailed reporting
            if hasattr(self.loss_components["MSE"], 'last_magnitude_loss'):
                loss_summary["magnitude_loss"] = self.loss_components["MSE"].last_magnitude_loss
            if hasattr(self.loss_components["MSE"], 'last_logabs_sign_bce_loss'):
                loss_summary["logabs_sign_bce_loss"] = self.loss_components["MSE"].last_logabs_sign_bce_loss
            if hasattr(self.loss_components["MSE"], 'last_real_sign_bce_loss'):
                loss_summary["real_sign_bce_loss"] = self.loss_components["MSE"].last_real_sign_bce_loss
            if hasattr(self.loss_components["MSE"], 'last_ft_cal_loss'):
                loss_summary["ft_cal_loss"] = self.loss_components["MSE"].last_ft_cal_loss

        # Residual Loss
        if "Residual" in self.loss_components and "Residual" in loss_args:
            outputs_for_residual, targets, inputs_real, output_normalizer = loss_args["Residual"]
            residual_value = self.loss_components["Residual"](
                outputs_for_residual, targets,
                inputs_real=inputs_real,
                output_normalizer=output_normalizer
            )
            total_loss += residual_value
            loss_summary["residual_loss"] = residual_value.item()

        # Consistency Loss
        if "Consistency" in self.loss_components and "Consistency" in loss_args:
            # New signature: (predictions, targets, inputs, inputs_normalizer, outputs_normalizer, ft_cal, valid_mask)
            # predictions = mag_preds (MODE 1) or None (MODE 2)
            # valid_mask = boolean tensor (MODE 1) or None (MODE 2)
            predictions, targets, inputs, inputs_normalizer, outputs_normalizer, ft_cal, valid_mask = loss_args["Consistency"]
            consistency_value = self.loss_components["Consistency"](
                predictions, targets,
                inputs=inputs,
                inputs_normalizer=inputs_normalizer,
                outputs_normalizer=outputs_normalizer,
                ft_cal=ft_cal,
                valid_mask=valid_mask
            )
            total_loss += consistency_value
            loss_summary["consistency_loss"] = consistency_value.item()

        # Initial Condition Loss
        if "InitialCondition" in self.loss_components and "InitialCondition" in loss_args:
            inputs_normalized_ic, inputs_real_ic, ft_cal_ic = loss_args["InitialCondition"]
            initial_value = self.loss_components["InitialCondition"](
                None, None,
                inputs_normalized=inputs_normalized_ic,
                inputs_real=inputs_real_ic,
                ft_cal=ft_cal_ic
            )
            total_loss += initial_value
            loss_summary["initial_condition_loss"] = initial_value.item()

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
                print(f"  [+] {loss_name:20s}: weight={weight:.3f}")
            else:
                print(f"  [-] {loss_name:20s}: disabled")

        print(f"{'='*60}\n")


class PINNLoss_old(nn.Module):
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
            use_log = config.get("use_log", False)
            weight = config.get("weight", 1.0)
            sign_bce_weight = config.get("sign_bce_weight", 1.0)
            real_sign_bce_weight = config.get("real_sign_bce_weight", 1.0)
            ft_cal_weight = config.get("ft_cal_weight", 1.0)
            # Use model.use_finetune to determine if ft_cal_loss should be computed
            use_finetune_loss = getattr(model, 'use_finetune', False)
            self.loss_components["MSE"] = MSELoss(weight=weight, use_relative=use_relative, use_log=use_log, sign_bce_weight=sign_bce_weight, real_sign_bce_weight=real_sign_bce_weight, ft_cal_weight=ft_cal_weight, use_finetune_loss=use_finetune_loss)

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
            use_log = config.get("use_log", True)
            input_grad_outside = config.get("Input_grad_outside", False)

            # Only use auto-diff consistency loss (following Exp pattern)
            self.loss_components["Consistency"] = ConsistencyLoss_auto_diff(
                weight=weight, model=model, t_threshold=t_threshold, use_log=use_log,
                input_grad_outside=input_grad_outside
            )

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
            # Unpack 8 arguments: (mag_preds, targets, logabs_sign_probs, _, _, real_sign_probs, ft_cal, output_normalizer)
            mag_preds, targets, logabs_sign_probs, _, _, real_sign_probs, ft_cal, output_normalizer = loss_args["MSE"]
            mse_value = self.loss_components["MSE"](
                mag_preds, targets,
                logabs_sigmoid_probs=logabs_sign_probs,
                real_sign_probs=real_sign_probs,
                ft_cal=ft_cal
            )
            total_loss += mse_value
            loss_summary["mse_loss"] = mse_value.item()

            # Add magnitude and sign loss components for detailed reporting
            if hasattr(self.loss_components["MSE"], 'last_magnitude_loss'):
                loss_summary["magnitude_loss"] = self.loss_components["MSE"].last_magnitude_loss
            if hasattr(self.loss_components["MSE"], 'last_logabs_sign_bce_loss'):
                loss_summary["logabs_sign_bce_loss"] = self.loss_components["MSE"].last_logabs_sign_bce_loss
            if hasattr(self.loss_components["MSE"], 'last_real_sign_bce_loss'):
                loss_summary["real_sign_bce_loss"] = self.loss_components["MSE"].last_real_sign_bce_loss
            if hasattr(self.loss_components["MSE"], 'last_ft_cal_loss'):
                loss_summary["ft_cal_loss"] = self.loss_components["MSE"].last_ft_cal_loss

        # Residual Loss
        if "Residual" in self.loss_components and "Residual" in loss_args:
            outputs_for_residual, targets, inputs_real, output_normalizer = loss_args["Residual"]
            residual_value = self.loss_components["Residual"](
                outputs_for_residual, targets,
                inputs_real=inputs_real,
                output_normalizer=output_normalizer
            )
            total_loss += residual_value
            loss_summary["residual_loss"] = residual_value.item()

        # Consistency Loss
        if "Consistency" in self.loss_components and "Consistency" in loss_args:
            # New signature: (predictions, targets, inputs, inputs_normalizer, outputs_normalizer, ft_cal, valid_mask)
            # predictions = mag_preds (MODE 1) or None (MODE 2)
            # valid_mask = boolean tensor (MODE 1) or None (MODE 2)
            predictions, targets, inputs, inputs_normalizer, outputs_normalizer, ft_cal, valid_mask = loss_args["Consistency"]
            consistency_value = self.loss_components["Consistency"](
                predictions, targets,
                inputs=inputs,
                inputs_normalizer=inputs_normalizer,
                outputs_normalizer=outputs_normalizer,
                ft_cal=ft_cal,
                valid_mask=valid_mask
            )
            total_loss += consistency_value
            loss_summary["consistency_loss"] = consistency_value.item()

        # Initial Condition Loss
        if "InitialCondition" in self.loss_components and "InitialCondition" in loss_args:
            outputs_t0, inputs_real_t0 = loss_args["InitialCondition"]
            # InitialCondition loss uses simple signature
            initial_value = self.loss_components["InitialCondition"](
                outputs_t0, None,
                inputs_real=inputs_real_t0
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
                print(f"  [+] {loss_name:20s}: weight={weight:.3f}")
            else:
                print(f"  [-] {loss_name:20s}: disabled")

        print(f"{'='*60}\n")
