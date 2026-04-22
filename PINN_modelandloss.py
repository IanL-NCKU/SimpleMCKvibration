import torch
import torch.nn as nn
import numpy as np
from abc import ABC, abstractmethod
from Exp_dataset import generalize_alpha, generalize_beta

class SignWithHardTanh(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        # 1. Forward pass: use sign function for binarization
        ctx.save_for_backward(x)
        return torch.sign(x)

    @staticmethod
    def backward(ctx, grad_output):
        # 2. Backward pass: load x and compute HardTanh derivative (Rectangular gradient)
        x, = ctx.saved_tensors
        
        # HardTanh(-1, 1) derivative: 1 between [-1, 1], 0 elsewhere
        # This is equivalent to (x < 1) & (x > -1)
        grad_input = grad_output.clone()
        grad_input[x.abs() > 1.0] = 0
        
        # Gradient value remains 1
        # return grad_input * 1.0  # Which is just grad_input
        return grad_input


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

    def __init__(self, hidden_dims=[64, 128, 128, 64], activation='tanh', use_internal_sign=False,
                 use_finetune=False, finetune_hidden_dims=[32, 32], finetune_scale=0.1,
                 use_sign_network=False, sign_network_hidden_dims=[32, 32]):
        super().__init__()

        self.use_internal_sign = use_internal_sign
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
        if use_internal_sign:
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
        if self.use_internal_sign:
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


class ExponentialSignNN_ver4(nn.Module):
    """Binary Classification Network with Residual Connections (3-input version)

    Modified version of ExponentialSignNN_ver3 that:
    - Takes only 3 inputs: [a, b, t] (no magnitude inputs)
    - Outputs sign predictions directly (not multiplied by magnitudes)
    - Uses identity tensor instead of input magnitudes

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

        input_dim = 3  # [a, b, t] only

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
            # Example: [128, 64, 32] → pairs: [(3,128,64)], remaining: [32]
            # Example: [128, 64, 32, 16] → pairs: [(3,128,64), (64,32,16)]
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
            x: Input tensor of shape (batch_size, 3)
               [a, b, t]

        Returns:
            predictions: Output tensor of shape (batch_size, 3)
                        Sign predictions: values in [-1, 1]
        """
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

        # Return signs directly (no magnitude multiplication)
        predictions = signs

        return predictions


class ExponentialPINN_ver2(nn.Module):
    """Physics-Informed Neural Network for exponential function: x(t) = b * exp(a*t)"""

    def __init__(self, hidden_dims=[64, 128, 128, 64], activation='tanh', use_internal_sign=False,
                 use_finetune=False, finetune_hidden_dims=[32, 32], finetune_scale=0.1,
                 sign_network_hidden_dims=[128, 64, 32], sign_network_dropout=0.3):
        super().__init__()

        self.use_internal_sign = use_internal_sign
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
        if self.use_internal_sign:
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
        mag_preds = torch.cat([x_pred, v_pred, a_pred], dim=1)
        sign_input = torch.cat([x, mag_preds.detach()], dim=1)  # Shape: [batch, 6]

        # Get sign predictions from ExponentialSignNN_ver3
        sign_pred = self.sign_network(sign_input)

        # Store sigmoid probabilities for BCE loss computation
        self.last_sign_probs = self.sign_network.last_sign_probs

        # Return magnitude and sign predictions separately
        return mag_preds, sign_pred


class ExponentialPINN_ver3(nn.Module):
    """Physics-Informed Neural Network with dual sign networks for both logabs and real values"""

    def __init__(self, hidden_dims=[64, 128, 128, 64], activation='tanh', use_internal_sign=False,
                 use_finetune=False, finetune_hidden_dims=[32, 32], finetune_scale=0.1,
                 logabs_sign_network_hidden_dims=[128, 64, 32], logabs_sign_network_dropout=0.3,
                 real_sign_network_hidden_dims=[128, 64, 32], real_sign_network_dropout=0.3):
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
        input_dim = 3  # [a, b, t]

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
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
            finetune_input_dim = 9  # [a, b, t, sign_prob_x, sign_prob_v, sign_prob_a, mag_x, mag_v, mag_a]
            finetune_act = nn.Tanh
            for hidden_dim in finetune_hidden_dims:
                finetune_layers.append(nn.Linear(finetune_input_dim, hidden_dim))
                finetune_layers.append(finetune_act())
                finetune_input_dim = hidden_dim

            finetune_layers.append(nn.Linear(finetune_input_dim, 3))
            # finetune_layers.append(nn.Tanh())

            self.finetune_network = nn.Sequential(*finetune_layers)
        else:
            self.finetune_network = None

        # Build logabs sign network (predicts signs of log-absolute values)
        self.logabs_sign_network = ExponentialSignNN_ver3(
            hidden_dims=logabs_sign_network_hidden_dims,
            activation=activation,
            dropout=logabs_sign_network_dropout
        )

        # Build real sign network (predicts signs of real values from a, b, t only)
        self.real_sign_network = ExponentialSignNN_ver4(
            hidden_dims=real_sign_network_hidden_dims,
            activation=activation,
            dropout=real_sign_network_dropout
        )

        # Initialize weights
        self.apply(self._init_weights)

        # Freeze finetune network initially (will be unfrozen later in training)
        # if self.use_finetune:
        #     self.freeze_finetune_network()

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
            x: Input tensor of shape (batch_size, 3)
               [a, b, t]

        Returns:
            mag_preds: Magnitude predictions (batch_size, 3)
            logabs_sign_pred: Sign predictions for log-abs values (batch_size, 3)
            real_sign_pred: Sign predictions for real values (batch_size, 3)
            ft_cal: Calibration factors (batch_size, 3) - ones if use_finetune=False
        """
        output = self.network(x)
        ln10 = np.log(10.0)  # Python float, PyTorch handles device automatically
        # Step 1: Convert network output to real space (base predictions)
        if self.use_internal_sign:
            # Network outputs: [sign_x, log_x, sign_v, log_v, sign_a, log_a]

            # Extract sign and log magnitude
            sign_x = torch.tanh(output[:, 0:1])
            log_x = output[:, 1:2]
            sign_v = torch.tanh(output[:, 2:3])
            log_v = output[:, 3:4]
            sign_a = torch.tanh(output[:, 4:5])
            log_a = output[:, 5:6]

            # Transform to real space: x = sign * 10^log = sign * exp(log * ln(10))
            
            x_pred_base = sign_x * torch.exp(log_x * ln10)
            v_pred_base = sign_v * torch.exp(log_v * ln10)
            a_pred_base = sign_a * torch.exp(log_a * ln10)
        else:
            # Network outputs directly in real space: [x_t, v_t, a_t]
            x_pred_base = output[:, 0:1]
            v_pred_base = output[:, 1:2]
            a_pred_base = output[:, 2:3]

        # Step 2: Compute base magnitude predictions
        # mag_preds is just base predictions, NO calibration applied in forward pass
        mag_preds = torch.cat([x_pred_base, v_pred_base, a_pred_base], dim=1)

        # Step 3: Apply logabs sign network
        # Prepare input: concatenate [a, b, t] + current magnitude predictions (detached)
        logabs_sign_input = torch.cat([x, mag_preds.detach()], dim=1)  # Shape: [batch, 6]

        # Get logabs sign predictions from ExponentialSignNN_ver3
        logabs_sign_pred = self.logabs_sign_network(logabs_sign_input)

        # Store sigmoid probabilities for logabs sign BCE loss computation
        self.logabs_last_sign_probs = self.logabs_sign_network.last_sign_probs

        # Step 4: Apply real sign network (uses only a, b, t as input)
        real_sign_pred = self.real_sign_network(x)  # Shape: [batch, 3]

        # Store sigmoid probabilities for real sign BCE loss computation
        self.real_last_sign_probs = self.real_sign_network.last_sign_probs

        # Step 5: Compute calibration factors (independent network with selective detach)
        if self.use_finetune:
            # Selectively detach: keep x attached for consistency loss gradients, detach sign_probs and mag_preds to prevent ft_cal_loss from tuning them
            finetune_input = torch.cat([x, self.logabs_last_sign_probs.detach(), mag_preds.detach()], dim=1)  # Shape: [batch, 9]
            
            ft_cal_raw = self.finetune_network(finetune_input)/self.finetune_scale  # Shape: [batch, 3]
            # use tanh to constrain calibration factors to (-1, 1)
            ft_cal = ln10* torch.tanh(ft_cal_raw)
        else:
            # Return zeros when finetune disabled (identity calibration)
            ft_cal = torch.zeros_like(mag_preds)

        # Return magnitude, both sign predictions, and calibration factors separately
        return mag_preds, logabs_sign_pred, real_sign_pred, ft_cal


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
    """Mean Squared Error loss between predictions and targets"""

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

            if self.use_relative:
                # Relative log-space MSE for magnitudes
                magnitude_loss = torch.mean(((log_predictions - log_targets) ** 2) / (torch.square(log_targets) + eps))
            else:
                # Absolute log-space MSE for magnitudes
                magnitude_loss = torch.mean((log_predictions - log_targets) ** 2)

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

    def compute(self, predictions, targets, inputs_real=None, output_normalizer=None, **kwargs):
        """
        Enforce exponential physics equation:
        (1/(2a))*a_t + 0.5*v_t - a*x_t = 0

        Args:
            predictions: (batch_size, 6) - [real_signs (0-2), logabs_values (3-5)] in NORMALIZED log-space
            targets: Not used (for signature compatibility)
            inputs_real: (batch_size, 3) - [a, b, t] in real space
            output_normalizer: Normalizer instance for manual denormalization
        """
        if inputs_real is None:
            raise ValueError("ExponentialResidualLoss requires inputs_real to be provided")
        if output_normalizer is None:
            raise ValueError("ExponentialResidualLoss requires output_normalizer to be provided")

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
        x_pred = real_values[:, 0]#.detach()  # Detach position to prevent gradient issues
        v_pred = real_values[:, 1]#.detach()  # Detach velocity to prevent gradient issues
        a_pred = real_values[:, 2]#.detach()  # Detach acceleration to prevent gradient issues

        # Extract parameters
        a = inputs_real[:, 0]  # exponential rate

        eps = 1e-12

        # Physics residual: (1/(2a))*a_t + 0.5*v_t - a*x_t = 0
        residual =(1.0 / (2.0 * a + eps)) * a_pred + 0.5 * v_pred - a * x_pred
        # residual = torch.log(torch.abs((1.0 / (2.0 * a + eps)) * a_pred)) - torch.log(torch.abs(0.5 * v_pred - a * x_pred)) \
        #             + torch.log(torch.abs(a * x_pred)) - torch.log(torch.abs((1.0 / (2.0 * a + eps)) * a_pred + 0.5 * v_pred))\
        #             + torch.log(torch.abs(0.5 * v_pred)) - torch.log(torch.abs(a * x_pred - (1.0 / (2.0 * a + eps)) * a_pred))

        if self.use_relative:
            # Scale-invariant relative residual
            # Normalize by target's acceleration term: (1/(2a))*a_target
            # Denormalize targets to get a_target
            target_real_signs = targets[:, :3]
            target_logabs_normalized = targets[:, 3:]
            target_logabs_denorm = target_logabs_normalized * log_std + log_mean
            target_real_values = target_real_signs * torch.exp(target_logabs_denorm * ln10)
            a_target = target_real_values[:, 2].detach()  # Detach target to prevent gradient issues

            scale = torch.abs((1.0 / (2.0 * a + eps)) * a_target) + eps
            residual = residual / scale
        # residual = torch.log(torch.abs(residual) + 1)
        # return torch.mean(torch.abs(residual))
        return torch.mean(torch.log(torch.square(residual) + 1))


class ConsistencyLoss_auto_diff_magver(BaseLossComponent):
    """
    Derivative consistency using VERIFIED formulas for log-normalized space training.

    Ensures v=dx/dt and a=dv/dt using autograd on log-normalized model outputs.
    Uses verified transformation formulas from check_dataset_consistency() in Exp_dataset.py.

    Key features:
    - Model outputs UNSIGNED magnitudes (positive from Softplus) in normalized log-space
    - Applies logabs signs from targets[:, 3:] to create SIGNED normalized log-space predictions
    - Computes derivatives of UNSIGNED magnitudes via autograd in normalized space
    - Applies verified denormalization formulas with SIGNED predictions for theory values
    - Supports both log-space MSE and regular MSE via use_log parameter
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
            inputs: (batch_size, 3) - [a, b, t] NORMALIZED
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

        # Get normalization parameters (needed for both modes)
        beta_x = generalize_beta(outputs_normalizer, outputs_normalizer.log_mean['x'], outputs_normalizer.log_std['x'], 'x')
        beta_v = generalize_beta(outputs_normalizer, outputs_normalizer.log_mean['v'], outputs_normalizer.log_std['v'], 'v')
        beta_a = generalize_beta(outputs_normalizer, outputs_normalizer.log_mean['a'], outputs_normalizer.log_std['a'], 'a')
        alpha_x = generalize_alpha(outputs_normalizer, outputs_normalizer.log_mean['x'], outputs_normalizer.log_std['x'], 'x')
        alpha_v = generalize_alpha(outputs_normalizer, outputs_normalizer.log_mean['v'], outputs_normalizer.log_std['v'], 'v')
        alpha_a = generalize_alpha(outputs_normalizer, outputs_normalizer.log_mean['a'], outputs_normalizer.log_std['a'], 'a')
        t_alpha = generalize_alpha(inputs_normalizer, inputs_normalizer.log_mean['t'], inputs_normalizer.log_std['t'], 't')
        t_beta = generalize_beta(inputs_normalizer, inputs_normalizer.log_mean['t'], inputs_normalizer.log_std['t'], 't')
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
            mag_preds_valid = predictions[valid_mask]
            inputs_valid = inputs[valid_mask]
            targets_valid = targets[valid_mask]
            ft_cal_valid = ft_cal[valid_mask]

            # Compute t_real for valid samples
            t_normalized_valid = inputs_valid[:, 2]
            t_real_valid = torch.exp((t_beta * t_normalized_valid + t_alpha) * ln10)

            # STEP 1: Get unsigned magnitude predictions and compute derivatives separately
            # Detect phase
            is_phase1 = torch.all(torch.abs(ft_cal_valid) < eps)

            # STEP 1a: Compute values (for loss calculation)
            mag_x = mag_preds_valid[:, 0] + ft_cal_valid[:, 0]
            mag_v = mag_preds_valid[:, 1] + ft_cal_valid[:, 1]
            mag_a = mag_preds_valid[:, 2] + ft_cal_valid[:, 2]
            # mag_x = mag_preds_valid[:, 0]
            # mag_v = mag_preds_valid[:, 1]
            # mag_a = mag_preds_valid[:, 2]
            # STEP 1b: Compute derivatives separately, then sum
            # Always compute derivatives of mag_preds components
            dx_mag_dt = torch.autograd.grad(
                outputs=mag_preds_valid[:, 0],
                inputs=inputs,
                grad_outputs=torch.ones_like(mag_preds_valid[:, 0]),
                create_graph=True,
                retain_graph=True,
                allow_unused=True
            )[0][valid_mask, 2]  # Gradient w.r.t. t_normalized, filter by valid_mask

            dv_mag_dt = torch.autograd.grad(
                outputs=mag_preds_valid[:, 1],
                inputs=inputs,
                grad_outputs=torch.ones_like(mag_preds_valid[:, 1]),
                create_graph=True,
                retain_graph=True,
                allow_unused=True
            )[0][valid_mask, 2]  # Gradient w.r.t. t_normalized, filter by valid_mask

            if is_phase1:
                # Phase 1: Only use mag derivatives (ft_cal is zeros)
                dx_prime_dt_prime = torch.abs(dx_mag_dt)
                dv_prime_dt_prime = torch.abs(dv_mag_dt)
            else:
                # Phase 2: Compute ft_cal derivatives and add them
                dx_ft_dt = torch.autograd.grad(
                    outputs=ft_cal_valid[:, 0],
                    inputs=inputs,
                    grad_outputs=torch.ones_like(ft_cal_valid[:, 0]),
                    create_graph=True,
                    retain_graph=True,
                    allow_unused=True
                )[0][valid_mask, 2]  # Gradient w.r.t. t_normalized, filter by valid_mask

                dv_ft_dt = torch.autograd.grad(
                    outputs=ft_cal_valid[:, 1],
                    inputs=inputs,
                    grad_outputs=torch.ones_like(ft_cal_valid[:, 1]),
                    create_graph=True,
                    retain_graph=True,
                    allow_unused=True
                )[0][valid_mask, 2]  # Gradient w.r.t. t_normalized, filter by valid_mask

                # Sum derivatives (linearity: d(f+g)/dt = df/dt + dg/dt)
                dx_prime_dt_prime = torch.abs(dx_mag_dt) + dx_ft_dt
                dv_prime_dt_prime = torch.abs(dv_mag_dt) + dv_ft_dt
                # dx_prime_dt_prime = torch.abs(dx_mag_dt)
                # dv_prime_dt_prime = torch.abs(dv_mag_dt)
        else:
            # MODE 2: Gradients computed inside consistency loss (current approach)
            # Call model internally with inputs_with_grad

            if self.model is None:
                raise ValueError("When input_grad_outside=False, model must be set via set_model()")
            if ft_cal is None:
                raise ValueError("When input_grad_outside=False, ft_cal must be provided")

            # Denormalize inputs to get t_real
            t_normalized = inputs[:, 2]
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

            # STEP 1a: Compute values (for loss calculation)
            mag_x = mag_preds_internal[:, 0] + ft_preds_internal[:, 0]
            mag_v = mag_preds_internal[:, 1] + ft_preds_internal[:, 1]
            mag_a = mag_preds_internal[:, 2] + ft_preds_internal[:, 2]
            # mag_x = mag_preds_internal[:, 0]
            # mag_v = mag_preds_internal[:, 1]
            # mag_a = mag_preds_internal[:, 2]

            # STEP 1b: Compute derivatives separately, then sum
            # Always compute derivatives of mag_preds_internal components
            dx_mag_dt = torch.autograd.grad(
                outputs=mag_preds_internal[:, 0],
                inputs=inputs_with_grad,
                grad_outputs=torch.ones_like(mag_preds_internal[:, 0]),
                create_graph=True,
                retain_graph=True,
                allow_unused=True
            )[0][:, 2]  # Gradient w.r.t. t_normalized (index 2)

            dv_mag_dt = torch.autograd.grad(
                outputs=mag_preds_internal[:, 1],
                inputs=inputs_with_grad,
                grad_outputs=torch.ones_like(mag_preds_internal[:, 1]),
                create_graph=True,
                retain_graph=True,
                allow_unused=True
            )[0][:, 2]  # Gradient w.r.t. t_normalized (index 2)

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
                )[0][:, 2]  # Gradient w.r.t. t_normalized (index 2)

                dv_ft_dt = torch.autograd.grad(
                    outputs=ft_preds_internal[:, 1],
                    inputs=inputs_with_grad,
                    grad_outputs=torch.ones_like(ft_preds_internal[:, 1]),
                    create_graph=True,
                    retain_graph=True,
                    allow_unused=True
                )[0][:, 2]  # Gradient w.r.t. t_normalized (index 2)

                # Sum derivatives (linearity: d(f+g)/dt = df/dt + dg/dt)
                dx_prime_dt_prime = torch.abs(dx_mag_dt) + dx_ft_dt
                dv_prime_dt_prime = torch.abs(dv_mag_dt) + dv_ft_dt
                # dx_prime_dt_prime = torch.abs(dx_mag_dt)
                # dv_prime_dt_prime = torch.abs(dv_mag_dt)
        # ======================
        # COMMON CODE: Apply signs and compute theory values
        # ======================
        # STEP 3: Apply logabs signs (from targets[:, 3:]) → SIGNED normalized log-space predictions
        # targets[:, 3:] contains SIGNED normalized log-space values (can be ±)
        logabs_targets = targets_valid[:, 3:]  # [x', v', a'] in normalized log-space (signed)
        logabs_sign = torch.sign(logabs_targets)  # Extract signs from logabs targets

        x_pred = logabs_sign[:, 0] * mag_x  # Signed normalized log-space
        v_pred = logabs_sign[:, 1] * mag_v  # Signed normalized log-space

        # STEP 4: DETACH derivatives and SIGNED predictions for transformation
        x_pred_detached = x_pred.detach()  # SIGNED normalized log-space
        v_pred_detached = v_pred.detach()  # SIGNED normalized log-space
        dx_dt_detached = dx_prime_dt_prime.detach()
        dv_dt_detached = dv_prime_dt_prime.detach()

        # STEP 5: Denormalize SIGNED predictions to real space
        # This matches dataset: x_prime is signed in normalized space
        x_real = torch.exp((beta_x * x_pred_detached + alpha_x) * ln10)
        v_real = torch.exp((beta_v * v_pred_detached + alpha_v) * ln10)

        # STEP 6: Compute theory values using VERIFIED FORMULA
        common_factor_v = (beta_x / t_beta) * (x_real / (t_real_valid + eps))
        v_theory = torch.abs(common_factor_v * dx_dt_detached)

        common_factor_a = (beta_v / t_beta) * (v_real / (t_real_valid + eps))
        a_theory = torch.abs(common_factor_a * dv_dt_detached)

        # STEP 7: Normalize theory back to log-normalized space
        v_theory_normalized = (torch.log10(v_theory + eps) - alpha_v) / beta_v
        a_theory_normalized = (torch.log10(a_theory + eps) - alpha_a) / beta_a

        # STEP 8: DETACH theory targets (act as ground truth)
        v_theory_normalized = v_theory_normalized.detach()
        a_theory_normalized = a_theory_normalized.detach()

        # STEP 9: Compute loss (log-space MSE or regular MSE based on use_log)
        if self.use_log:
            # Log-space MSE (matching MSELoss format)
            log_mag_v = torch.log(torch.abs(mag_v) + eps)
            log_v_theory = torch.log(torch.abs(v_theory_normalized) + eps)
            v_consistency_loss = torch.mean((log_mag_v - log_v_theory) ** 2)

            log_mag_a = torch.log(torch.abs(mag_a) + eps)
            log_a_theory = torch.log(torch.abs(a_theory_normalized) + eps)
            a_consistency_loss = torch.mean((log_mag_a - log_a_theory) ** 2)
        else:
            # Regular MSE
            v_consistency_loss = torch.mean(((torch.abs(mag_v) - torch.abs(v_theory_normalized))/torch.abs(mag_v)) ** 2)
            a_consistency_loss = torch.mean(((torch.abs(mag_a) - torch.abs(a_theory_normalized))/torch.abs(mag_a)) ** 2)

        # STEP 10: Total consistency loss
        total_loss = v_consistency_loss + a_consistency_loss

        return total_loss
'''


class ConsistencyLoss_auto_diff_derivagrad_ver0(BaseLossComponent):
    """
    REVERSE GRADIENT FLOW: Derivative consistency with trainable theory values.

    Ensures v=dx/dt and a=dv/dt by training DERIVATIVES to produce theory matching GROUND TRUTH TARGETS.
    Uses verified transformation formulas from check_dataset_consistency() in Exp_dataset.py.

    Key features (REVERSE gradient flow with target comparison):
    - Model outputs UNSIGNED magnitudes (positive from Softplus) in normalized log-space
    - Applies logabs signs from targets[:, 3:] to create SIGNED normalized log-space predictions
    - Computes derivatives of UNSIGNED magnitudes via autograd with create_graph=True
    - TARGETS (v_target, a_target) from ground truth → fixed reference values
    - THEORY VALUES (v_theory, a_theory) have GRADIENTS → trainable via derivatives
    - Loss trains derivatives to produce theory that matches ground truth targets
    - Supports both log-space MSE and regular MSE via use_log parameter

    Gradient Flow Direction:
    - ver0: Train predictions → match fixed theory (theory detached)
    - ConsistencyLoss_auto_diff: Train derivatives → produce theory matching fixed predictions
    - THIS VERSION: Train derivatives → produce theory matching ground truth targets
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
            inputs: (batch_size, 3) - [a, b, t] NORMALIZED
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

        # Get normalization parameters (needed for both modes)
        beta_x = generalize_beta(outputs_normalizer, outputs_normalizer.log_mean['x'], outputs_normalizer.log_std['x'], 'x')
        beta_v = generalize_beta(outputs_normalizer, outputs_normalizer.log_mean['v'], outputs_normalizer.log_std['v'], 'v')
        beta_a = generalize_beta(outputs_normalizer, outputs_normalizer.log_mean['a'], outputs_normalizer.log_std['a'], 'a')
        alpha_x = generalize_alpha(outputs_normalizer, outputs_normalizer.log_mean['x'], outputs_normalizer.log_std['x'], 'x')
        alpha_v = generalize_alpha(outputs_normalizer, outputs_normalizer.log_mean['v'], outputs_normalizer.log_std['v'], 'v')
        alpha_a = generalize_alpha(outputs_normalizer, outputs_normalizer.log_mean['a'], outputs_normalizer.log_std['a'], 'a')
        t_alpha = generalize_alpha(inputs_normalizer, inputs_normalizer.log_mean['t'], inputs_normalizer.log_std['t'], 't')
        t_beta = generalize_beta(inputs_normalizer, inputs_normalizer.log_mean['t'], inputs_normalizer.log_std['t'], 't')
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
            mag_preds_valid = predictions[valid_mask]
            inputs_valid = inputs[valid_mask]
            targets_valid = targets[valid_mask]
            ft_cal_valid = ft_cal[valid_mask]

            # Compute t_real for valid samples
            t_normalized_valid = inputs_valid[:, 2]
            t_real_valid = torch.exp((t_beta * t_normalized_valid + t_alpha) * ln10)

            # STEP 1: Get unsigned magnitude predictions and compute derivatives separately
            # Detect phase
            is_phase1 = torch.all(torch.abs(ft_cal_valid) < eps)

            # STEP 1a: Compute values (for loss calculation)
            mag_x = mag_preds_valid[:, 0] + ft_cal_valid[:, 0]
            mag_v = mag_preds_valid[:, 1] + ft_cal_valid[:, 1]
            mag_a = mag_preds_valid[:, 2] + ft_cal_valid[:, 2]

            # STEP 1b: Compute derivatives separately, then sum
            # Always compute derivatives of mag_preds components
            dx_mag_dt = torch.autograd.grad(
                outputs=mag_preds_valid[:, 0],
                inputs=inputs,
                grad_outputs=torch.ones_like(mag_preds_valid[:, 0]),
                create_graph=True,
                retain_graph=True,
                allow_unused=True
            )[0][valid_mask, 2]  # Gradient w.r.t. t_normalized, filter by valid_mask

            dv_mag_dt = torch.autograd.grad(
                outputs=mag_preds_valid[:, 1],
                inputs=inputs,
                grad_outputs=torch.ones_like(mag_preds_valid[:, 1]),
                create_graph=True,
                retain_graph=True,
                allow_unused=True
            )[0][valid_mask, 2]  # Gradient w.r.t. t_normalized, filter by valid_mask

            if is_phase1:
                # Phase 1: Only use mag derivatives (ft_cal is zeros)
                dx_prime_dt_prime = torch.abs(dx_mag_dt)
                dv_prime_dt_prime = torch.abs(dv_mag_dt)
            else:
                # Phase 2: Compute ft_cal derivatives and add them
                dx_ft_dt = torch.autograd.grad(
                    outputs=ft_cal_valid[:, 0],
                    inputs=inputs,
                    grad_outputs=torch.ones_like(ft_cal_valid[:, 0]),
                    create_graph=True,
                    retain_graph=True,
                    allow_unused=True
                )[0][valid_mask, 2]  # Gradient w.r.t. t_normalized, filter by valid_mask

                dv_ft_dt = torch.autograd.grad(
                    outputs=ft_cal_valid[:, 1],
                    inputs=inputs,
                    grad_outputs=torch.ones_like(ft_cal_valid[:, 1]),
                    create_graph=True,
                    retain_graph=True,
                    allow_unused=True
                )[0][valid_mask, 2]  # Gradient w.r.t. t_normalized, filter by valid_mask

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

            # Denormalize inputs to get t_real
            t_normalized = inputs[:, 2]
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

            # STEP 1a: Compute values (for loss calculation)
            mag_x = mag_preds_internal[:, 0] + ft_preds_internal[:, 0]
            mag_v = mag_preds_internal[:, 1] + ft_preds_internal[:, 1]
            mag_a = mag_preds_internal[:, 2] + ft_preds_internal[:, 2]

            # STEP 1b: Compute derivatives separately, then sum
            # Always compute derivatives of mag_preds_internal components
            dx_mag_dt = torch.autograd.grad(
                outputs=mag_preds_internal[:, 0],
                inputs=inputs_with_grad,
                grad_outputs=torch.ones_like(mag_preds_internal[:, 0]),
                create_graph=True,
                retain_graph=True,
                allow_unused=True
            )[0][:, 2]  # Gradient w.r.t. t_normalized (index 2)

            dv_mag_dt = torch.autograd.grad(
                outputs=mag_preds_internal[:, 1],
                inputs=inputs_with_grad,
                grad_outputs=torch.ones_like(mag_preds_internal[:, 1]),
                create_graph=True,
                retain_graph=True,
                allow_unused=True
            )[0][:, 2]  # Gradient w.r.t. t_normalized (index 2)

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
                )[0][:, 2]  # Gradient w.r.t. t_normalized (index 2)

                dv_ft_dt = torch.autograd.grad(
                    outputs=ft_preds_internal[:, 1],
                    inputs=inputs_with_grad,
                    grad_outputs=torch.ones_like(ft_preds_internal[:, 1]),
                    create_graph=True,
                    retain_graph=True,
                    allow_unused=True
                )[0][:, 2]  # Gradient w.r.t. t_normalized (index 2)

                # Sum derivatives (linearity: d(f+g)/dt = df/dt + dg/dt)
                dx_prime_dt_prime = torch.abs(dx_mag_dt) + dx_ft_dt
                dv_prime_dt_prime = torch.abs(dv_mag_dt) + dv_ft_dt

        # ======================
        # COMMON CODE: Apply signs and compute theory values
        # ======================
        # STEP 3: Apply logabs signs (from targets[:, 3:]) → SIGNED normalized log-space predictions
        # targets[:, 3:] contains SIGNED normalized log-space values (can be ±)
        logabs_targets = targets_valid[:, 3:]  # [x', v', a'] in normalized log-space (signed)
        logabs_sign = torch.sign(logabs_targets)  # Extract signs from logabs targets

        x_pred = logabs_sign[:, 0] * mag_x  # Signed normalized log-space
        v_pred = logabs_sign[:, 1] * mag_v  # Signed normalized log-space

        # STEP 4: DETACH SIGNED predictions for transformation (keep derivatives WITH gradients)
        x_pred_detached = x_pred.detach()  # SIGNED normalized log-space
        v_pred_detached = v_pred.detach()  # SIGNED normalized log-space
        dx_dt_with_grad = dx_prime_dt_prime  # Keep gradients for backprop!
        dv_dt_with_grad = dv_prime_dt_prime  # Keep gradients for backprop!

        # STEP 5: Denormalize SIGNED predictions to real space
        # This matches dataset: x_prime is signed in normalized space
        x_real = torch.exp((beta_x * x_pred_detached + alpha_x) * ln10)
        v_real = torch.exp((beta_v * v_pred_detached + alpha_v) * ln10)

        # STEP 6: Compute theory values using VERIFIED FORMULA (WITH gradients for backprop)
        common_factor_v = (beta_x / t_beta) * (x_real / (t_real_valid + eps))
        v_theory = torch.abs(common_factor_v * dx_dt_with_grad)

        common_factor_a = (beta_v / t_beta) * (v_real / (t_real_valid + eps))
        a_theory = torch.abs(common_factor_a * dv_dt_with_grad)

        # STEP 7: Normalize theory back to log-normalized space (KEEP gradients!)
        v_theory_normalized = (torch.log10(v_theory + eps) - alpha_v) / beta_v
        a_theory_normalized = (torch.log10(a_theory + eps) - alpha_a) / beta_a

        # STEP 8: Theory values now TRAINABLE (no detach - gradients flow through derivatives!)

        # Extract ground truth targets (v' and a' in normalized log-space)
        v_target = targets_valid[:, 4]  # v' target (SIGNED normalized log-space)
        a_target = targets_valid[:, 5]  # a' target (SIGNED normalized log-space)

        # STEP 9: Compute loss (log-space MSE or regular MSE based on use_log) - REVERSED gradient flow
        if self.use_log:
            # Log-space MSE - train derivatives to produce theory matching ground truth targets
            log_v_target = torch.log(torch.abs(v_target) + eps)  # Target (ground truth)
            log_v_theory = torch.log(torch.abs(v_theory_normalized) + eps)  # Theory has gradients!
            v_consistency_loss = torch.mean((log_v_theory - log_v_target) ** 2)

            log_a_target = torch.log(torch.abs(a_target) + eps)  # Target (ground truth)
            log_a_theory = torch.log(torch.abs(a_theory_normalized) + eps)  # Theory has gradients!
            a_consistency_loss = torch.mean((log_a_theory - log_a_target) ** 2)
        else:
            # Regular MSE - train derivatives to produce theory matching ground truth targets
            v_consistency_loss = torch.mean(((torch.abs(v_theory_normalized) - torch.abs(v_target))/torch.abs(v_theory_normalized)) ** 2)
            a_consistency_loss = torch.mean(((torch.abs(a_theory_normalized) - torch.abs(a_target))/torch.abs(a_theory_normalized)) ** 2)

        # STEP 10: Total consistency loss
        total_loss = v_consistency_loss + a_consistency_loss

        return total_loss


class ConsistencyLoss_auto_diff_ver1(BaseLossComponent):
    """
    TRIANGULAR CONSISTENCY LOSS: Train predictions with triangular constraint.

    Ensures consistency between THREE values: predictions (mag_v), theory (from derivatives), and targets.
    Uses triangular loss to train predictions considering both theory and targets.

    Key features:
    - Model outputs UNSIGNED magnitudes (positive from Softplus) in normalized log-space
    - Applies logabs signs from targets[:, 3:] to create SIGNED normalized log-space predictions
    - Computes derivatives of UNSIGNED magnitudes via autograd with create_graph=True
    - Derivatives are immediately DETACHED after computation
    - PREDICTIONS (mag_v, mag_a) have GRADIENTS → trainable
    - THEORY VALUES (v_theory, a_theory) are DETACHED → fixed (computed from derivatives)
    - TARGETS (v_target, a_target) from ground truth → fixed reference
    - Loss is triangular: (pred-theory)² + (theory-target)² + (pred-target)²
    - Trains predictions to simultaneously match BOTH theory and target

    Gradient Flow Direction:
    - ver0: Train predictions → match fixed theory only
    - THIS VERSION: Train predictions → match BOTH fixed theory AND targets (triangular)

    Loss Formula:
        L = (mag_v - v_theory)² + (v_theory - v_target)² + (mag_v - v_target)²

    Where only mag_v has gradients. The triangular structure encourages:
    1. Predictions to match theory (derivative consistency)
    2. Theory to be close to targets (theory accuracy check)
    3. Predictions to match targets directly (supervised learning)
    """

    def __init__(self, weight=1.0, model=None, t_threshold=1e-6, use_log=False, input_grad_outside=False):
        super().__init__(weight=weight, name="Consistency Loss (auto ver1)")
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
        Enforces triangular consistency using verified formulas for log-normalized training.

        Args:
            predictions: If input_grad_outside=True, this is mag_preds from training loop
                        If input_grad_outside=False, not used (we compute fresh predictions)
            targets: (batch_size, 6) - [real_sign_x, real_sign_v, real_sign_a, x', v', a']
                     where [:, 0:3] are real space signs (±1)
                     and [:, 3:6] are SIGNED normalized log-space values (can be ±)
            inputs: (batch_size, 3) - [a, b, t] NORMALIZED
            inputs_normalizer: Normalizer for inputs (contains t normalization params)
            outputs_normalizer: Normalizer for outputs (contains x, v, a normalization params)
            ft_cal: (batch_size, 3) - Finetune calibration outputs [ft_x, ft_v, ft_a] in normalized log-space
            valid_mask: If input_grad_outside=True, boolean mask for valid samples (t_real > threshold)
                       If input_grad_outside=False, not used (computed internally)

        Returns:
            loss: Scalar tensor measuring triangular consistency error
        """
        if inputs_normalizer is None or outputs_normalizer is None:
            raise ValueError("Both inputs_normalizer and outputs_normalizer must be provided")

        device = inputs.device
        dtype = inputs.dtype
        eps = 1e-12

        # Get normalization parameters (needed for both modes)
        beta_x = generalize_beta(outputs_normalizer, outputs_normalizer.log_mean['x'], outputs_normalizer.log_std['x'], 'x')
        beta_v = generalize_beta(outputs_normalizer, outputs_normalizer.log_mean['v'], outputs_normalizer.log_std['v'], 'v')
        beta_a = generalize_beta(outputs_normalizer, outputs_normalizer.log_mean['a'], outputs_normalizer.log_std['a'], 'a')
        alpha_x = generalize_alpha(outputs_normalizer, outputs_normalizer.log_mean['x'], outputs_normalizer.log_std['x'], 'x')
        alpha_v = generalize_alpha(outputs_normalizer, outputs_normalizer.log_mean['v'], outputs_normalizer.log_std['v'], 'v')
        alpha_a = generalize_alpha(outputs_normalizer, outputs_normalizer.log_mean['a'], outputs_normalizer.log_std['a'], 'a')
        t_alpha = generalize_alpha(inputs_normalizer, inputs_normalizer.log_mean['t'], inputs_normalizer.log_std['t'], 't')
        t_beta = generalize_beta(inputs_normalizer, inputs_normalizer.log_mean['t'], inputs_normalizer.log_std['t'], 't')
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
            mag_preds_valid = predictions[valid_mask]
            inputs_valid = inputs[valid_mask]
            targets_valid = targets[valid_mask]
            ft_cal_valid = ft_cal[valid_mask]

            # Compute t_real for valid samples
            t_normalized_valid = inputs_valid[:, 2]
            t_real_valid = torch.exp((t_beta * t_normalized_valid + t_alpha) * ln10)

            # STEP 1: Get unsigned magnitude predictions and compute derivatives separately
            # Detect phase
            is_phase1 = torch.all(torch.abs(ft_cal_valid) < eps)

            # STEP 1a: Compute values (for loss calculation)
            mag_x = mag_preds_valid[:, 0] + ft_cal_valid[:, 0]
            mag_v = mag_preds_valid[:, 1] + ft_cal_valid[:, 1]
            mag_a = mag_preds_valid[:, 2] + ft_cal_valid[:, 2]

            # STEP 1b: Compute derivatives separately, then sum
            # Always compute derivatives of mag_preds components
            dx_mag_dt = torch.autograd.grad(
                outputs=mag_preds_valid[:, 0],
                inputs=inputs,
                grad_outputs=torch.ones_like(mag_preds_valid[:, 0]),
                create_graph=True,
                retain_graph=True,
                allow_unused=True
            )[0][valid_mask, 2]  # Gradient w.r.t. t_normalized, filter by valid_mask

            dv_mag_dt = torch.autograd.grad(
                outputs=mag_preds_valid[:, 1],
                inputs=inputs,
                grad_outputs=torch.ones_like(mag_preds_valid[:, 1]),
                create_graph=True,
                retain_graph=True,
                allow_unused=True
            )[0][valid_mask, 2]  # Gradient w.r.t. t_normalized, filter by valid_mask

            if is_phase1:
                # Phase 1: Only use mag derivatives (ft_cal is zeros)
                dx_prime_dt_prime = torch.abs(dx_mag_dt)
                dv_prime_dt_prime = torch.abs(dv_mag_dt)
            else:
                # Phase 2: Compute ft_cal derivatives and add them
                dx_ft_dt = torch.autograd.grad(
                    outputs=ft_cal_valid[:, 0],
                    inputs=inputs,
                    grad_outputs=torch.ones_like(ft_cal_valid[:, 0]),
                    create_graph=True,
                    retain_graph=True,
                    allow_unused=True
                )[0][valid_mask, 2]  # Gradient w.r.t. t_normalized, filter by valid_mask

                dv_ft_dt = torch.autograd.grad(
                    outputs=ft_cal_valid[:, 1],
                    inputs=inputs,
                    grad_outputs=torch.ones_like(ft_cal_valid[:, 1]),
                    create_graph=True,
                    retain_graph=True,
                    allow_unused=True
                )[0][valid_mask, 2]  # Gradient w.r.t. t_normalized, filter by valid_mask

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

            # Denormalize inputs to get t_real
            t_normalized = inputs[:, 2]
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

            # STEP 1a: Compute values (for loss calculation)
            mag_x = mag_preds_internal[:, 0] + ft_preds_internal[:, 0]
            mag_v = mag_preds_internal[:, 1] + ft_preds_internal[:, 1]
            mag_a = mag_preds_internal[:, 2] + ft_preds_internal[:, 2]

            # STEP 1b: Compute derivatives separately, then sum
            # Always compute derivatives of mag_preds_internal components
            dx_mag_dt = torch.autograd.grad(
                outputs=mag_preds_internal[:, 0],
                inputs=inputs_with_grad,
                grad_outputs=torch.ones_like(mag_preds_internal[:, 0]),
                create_graph=True,
                retain_graph=True,
                allow_unused=True
            )[0][:, 2]  # Gradient w.r.t. t_normalized (index 2)

            dv_mag_dt = torch.autograd.grad(
                outputs=mag_preds_internal[:, 1],
                inputs=inputs_with_grad,
                grad_outputs=torch.ones_like(mag_preds_internal[:, 1]),
                create_graph=True,
                retain_graph=True,
                allow_unused=True
            )[0][:, 2]  # Gradient w.r.t. t_normalized (index 2)

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
                )[0][:, 2]  # Gradient w.r.t. t_normalized (index 2)

                dv_ft_dt = torch.autograd.grad(
                    outputs=ft_preds_internal[:, 1],
                    inputs=inputs_with_grad,
                    grad_outputs=torch.ones_like(ft_preds_internal[:, 1]),
                    create_graph=True,
                    retain_graph=True,
                    allow_unused=True
                )[0][:, 2]  # Gradient w.r.t. t_normalized (index 2)

                # Sum derivatives (linearity: d(f+g)/dt = df/dt + dg/dt)
                dx_prime_dt_prime = torch.abs(dx_mag_dt) + dx_ft_dt
                dv_prime_dt_prime = torch.abs(dv_mag_dt) + dv_ft_dt

        # ======================
        # COMMON CODE: Apply signs and compute theory values
        # ======================
        # STEP 3: Apply logabs signs (from targets[:, 3:]) → SIGNED normalized log-space predictions
        # targets[:, 3:] contains SIGNED normalized log-space values (can be ±)
        logabs_targets = targets_valid[:, 3:]  # [x', v', a'] in normalized log-space (signed)
        logabs_sign = torch.sign(logabs_targets)  # Extract signs from logabs targets

        x_pred = logabs_sign[:, 0] * mag_x  # Signed normalized log-space
        v_pred = logabs_sign[:, 1] * mag_v  # Signed normalized log-space

        # STEP 4: Keep predictions WITH gradients, but DETACH derivatives
        x_pred_detached = x_pred.detach()  # Detached for denormalization
        v_pred_detached = v_pred.detach()  # Detached for denormalization
        dx_dt_detached = dx_prime_dt_prime.detach()  # DETACH derivatives - theory won't have gradients
        dv_dt_detached = dv_prime_dt_prime.detach()  # DETACH derivatives - theory won't have gradients

        # STEP 5: Denormalize SIGNED predictions to real space
        # This matches dataset: x_prime is signed in normalized space
        x_real = torch.exp((beta_x * x_pred_detached + alpha_x) * ln10)
        v_real = torch.exp((beta_v * v_pred_detached + alpha_v) * ln10)

        # STEP 6: Compute theory values using VERIFIED FORMULA (derivatives detached - no gradients!)
        common_factor_v = (beta_x / t_beta) * (x_real / (t_real_valid + eps))
        v_theory = torch.abs(common_factor_v * dx_dt_detached)

        common_factor_a = (beta_v / t_beta) * (v_real / (t_real_valid + eps))
        a_theory = torch.abs(common_factor_a * dv_dt_detached)

        # STEP 7: Normalize theory back to log-normalized space (no gradients - derivatives detached)
        v_theory_normalized = (torch.log10(v_theory + eps) - alpha_v) / beta_v
        a_theory_normalized = (torch.log10(a_theory + eps) - alpha_a) / beta_a

        # STEP 8: Extract ground truth targets (v' and a' in normalized log-space)
        v_target = targets_valid[:, 4]  # v' target (SIGNED normalized log-space)
        a_target = targets_valid[:, 5]  # a' target (SIGNED normalized log-space)

        # # STEP 9: Compute TRIANGULAR loss - all three values contribute
        # if self.use_log:
        #     # Log-space MAE - triangular consistency between pred, theory, and target
        #     log_mag_v = torch.log(torch.abs(mag_v) + eps)  # Prediction has gradients!
        #     log_v_theory = torch.log(torch.abs(v_theory_normalized) + eps)  # Theory DETACHED (no gradients)
        #     log_v_target = torch.log(torch.abs(v_target) + eps)  # Target (ground truth)

        #     # Triangular loss: |pred-theory| + |theory-target| + |pred-target|
        #     # Only mag_v has gradients - trains predictions to match both theory and target
        #     v_consistency_loss = torch.mean(
        #     torch.abs(log_mag_v - log_v_theory) +
        #     torch.abs(log_v_theory - log_v_target) +
        #     torch.abs(log_mag_v - log_v_target)
        #     )

        #     log_mag_a = torch.log(torch.abs(mag_a) + eps)  # Prediction has gradients!
        #     log_a_theory = torch.log(torch.abs(a_theory_normalized) + eps)  # Theory DETACHED (no gradients)
        #     log_a_target = torch.log(torch.abs(a_target) + eps)  # Target (ground truth)

        #     # Triangular loss: |pred-theory| + |theory-target| + |pred-target|
        #     # Only mag_a has gradients - trains predictions to match both theory and target
        #     a_consistency_loss = torch.mean(
        #     torch.abs(log_mag_a - log_a_theory) +
        #     torch.abs(log_a_theory - log_a_target) +
        #     torch.abs(log_mag_a - log_a_target)
        #     )
        # else:
        #     # Regular MAE - triangular consistency
        #     v_consistency_loss = torch.mean(
        #     torch.abs((torch.abs(mag_v) - torch.abs(v_theory_normalized)) / torch.abs(mag_v)) +
        #     torch.abs((torch.abs(v_theory_normalized) - torch.abs(v_target)) / torch.abs(v_theory_normalized)) +
        #     torch.abs((torch.abs(mag_v) - torch.abs(v_target)) / torch.abs(mag_v))
        #     )

        #     a_consistency_loss = torch.mean(
        #     torch.abs((torch.abs(mag_a) - torch.abs(a_theory_normalized)) / torch.abs(mag_a)) +
        #     torch.abs((torch.abs(a_theory_normalized) - torch.abs(a_target)) / torch.abs(a_theory_normalized)) +
        #     torch.abs((torch.abs(mag_a) - torch.abs(a_target)) / torch.abs(mag_a))
        #     )

        # STEP 9: Compute TRIANGULAR loss - all three values contribute
        if self.use_log:
            # Log-space MSE - triangular consistency between pred, theory, and target
            log_mag_v = torch.log(torch.abs(mag_v) + eps)  # Prediction has gradients!
            log_v_theory = torch.log(torch.abs(v_theory_normalized) + eps)  # Theory DETACHED (no gradients)
            log_v_target = torch.log(torch.abs(v_target) + eps)  # Target (ground truth)

            # Triangular loss: (pred-theory)² + (theory-target)² + (pred-target)²
            # Only mag_v has gradients - trains predictions to match both theory and target
            v_consistency_loss = torch.mean(
                (log_mag_v - log_v_theory) ** 2 +
                (log_v_theory - log_v_target) ** 2 +
                (log_mag_v - log_v_target) ** 2
            )

            log_mag_a = torch.log(torch.abs(mag_a) + eps)  # Prediction has gradients!
            log_a_theory = torch.log(torch.abs(a_theory_normalized) + eps)  # Theory DETACHED (no gradients)
            log_a_target = torch.log(torch.abs(a_target) + eps)  # Target (ground truth)

            # Triangular loss: (pred-theory)² + (theory-target)² + (pred-target)²
            # Only mag_a has gradients - trains predictions to match both theory and target
            a_consistency_loss = torch.mean(
                (log_mag_a - log_a_theory) ** 2 +
                (log_a_theory - log_a_target) ** 2 +
                (log_mag_a - log_a_target) ** 2
            )
        else:
            # Regular MSE - triangular consistency
            v_consistency_loss = torch.mean(
                ((torch.abs(mag_v) - torch.abs(v_theory_normalized))/torch.abs(mag_v)) ** 2 +
                ((torch.abs(v_theory_normalized) - torch.abs(v_target))/torch.abs(v_theory_normalized)) ** 2 +
                ((torch.abs(mag_v) - torch.abs(v_target))/torch.abs(mag_v)) ** 2
            )

            a_consistency_loss = torch.mean(
                ((torch.abs(mag_a) - torch.abs(a_theory_normalized))/torch.abs(mag_a)) ** 2 +
                ((torch.abs(a_theory_normalized) - torch.abs(a_target))/torch.abs(a_theory_normalized)) ** 2 +
                ((torch.abs(mag_a) - torch.abs(a_target))/torch.abs(mag_a)) ** 2
            )

        # STEP 10: Total consistency loss
        total_loss = v_consistency_loss + a_consistency_loss

        return total_loss

class ConsistencyLoss_auto_diff_ver2(BaseLossComponent):
    """
    TRIANGULAR CONSISTENCY LOSS with GRADIENT RESCALING: Train predictions with triangular constraint and ft derivative control.

    Identical to ver1 but adds gradient rescaling to prevent finetune network from dominating in Phase 2.

    Key features (same as ver1):
    - Model outputs UNSIGNED magnitudes (positive from Softplus) in normalized log-space
    - Applies logabs signs from targets[:, 3:] to create SIGNED normalized log-space predictions
    - Computes derivatives of UNSIGNED magnitudes via autograd with create_graph=True
    - Derivatives are immediately DETACHED after computation
    - PREDICTIONS (mag_v, mag_a) have GRADIENTS → trainable
    - THEORY VALUES (v_theory, a_theory) are DETACHED → fixed (computed from derivatives)
    - TARGETS (v_target, a_target) from ground truth → fixed reference
    - Loss is triangular: (pred-theory)² + (theory-target)² + (pred-target)²

    NEW in ver2:
    - GRADIENT RESCALING: In Phase 2, ft derivatives are clamped to ±(|dx_mag| + |dx_mag|)
    - Prevents finetune network from overwhelming magnitude network
    - Scaling factor is detached to avoid affecting gradient magnitude

    Gradient Flow Direction:
    - Same as ver1: Train predictions → match BOTH fixed theory AND targets (triangular)
    - ver2 adds: Constrained ft derivative contribution in Phase 2

    Loss Formula (same as ver1):
        L = (mag_v - v_theory)² + (v_theory - v_target)² + (mag_v - v_target)²

    Where only mag_v has gradients. The triangular structure encourages:
    1. Predictions to match theory (derivative consistency)
    2. Theory to be close to targets (theory accuracy check)
    3. Predictions to match targets directly (supervised learning)
    """

    def __init__(self, weight=1.0, model=None, t_threshold=1e-6, use_log=False, input_grad_outside=False):
        super().__init__(weight=weight, name="Consistency Loss (auto ver2)")
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
        Enforces triangular consistency using verified formulas for log-normalized training.
        Includes gradient rescaling for ft derivatives in Phase 2.

        Args:
            predictions: If input_grad_outside=True, this is mag_preds from training loop
                        If input_grad_outside=False, not used (we compute fresh predictions)
            targets: (batch_size, 6) - [real_sign_x, real_sign_v, real_sign_a, x', v', a']
                     where [:, 0:3] are real space signs (±1)
                     and [:, 3:6] are SIGNED normalized log-space values (can be ±)
            inputs: (batch_size, 3) - [a, b, t] NORMALIZED
            inputs_normalizer: Normalizer for inputs (contains t normalization params)
            outputs_normalizer: Normalizer for outputs (contains x, v, a normalization params)
            ft_cal: (batch_size, 3) - Finetune calibration outputs [ft_x, ft_v, ft_a] in normalized log-space
            valid_mask: If input_grad_outside=True, boolean mask for valid samples (t_real > threshold)
                       If input_grad_outside=False, not used (computed internally)

        Returns:
            loss: Scalar tensor measuring triangular consistency error
        """
        if inputs_normalizer is None or outputs_normalizer is None:
            raise ValueError("Both inputs_normalizer and outputs_normalizer must be provided")

        device = inputs.device
        dtype = inputs.dtype
        eps = 1e-12

        # Get normalization parameters (needed for both modes)
        beta_x = generalize_beta(outputs_normalizer, outputs_normalizer.log_mean['x'], outputs_normalizer.log_std['x'], 'x')
        beta_v = generalize_beta(outputs_normalizer, outputs_normalizer.log_mean['v'], outputs_normalizer.log_std['v'], 'v')
        beta_a = generalize_beta(outputs_normalizer, outputs_normalizer.log_mean['a'], outputs_normalizer.log_std['a'], 'a')
        alpha_x = generalize_alpha(outputs_normalizer, outputs_normalizer.log_mean['x'], outputs_normalizer.log_std['x'], 'x')
        alpha_v = generalize_alpha(outputs_normalizer, outputs_normalizer.log_mean['v'], outputs_normalizer.log_std['v'], 'v')
        alpha_a = generalize_alpha(outputs_normalizer, outputs_normalizer.log_mean['a'], outputs_normalizer.log_std['a'], 'a')
        t_alpha = generalize_alpha(inputs_normalizer, inputs_normalizer.log_mean['t'], inputs_normalizer.log_std['t'], 't')
        t_beta = generalize_beta(inputs_normalizer, inputs_normalizer.log_mean['t'], inputs_normalizer.log_std['t'], 't')
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
            mag_preds_valid = predictions[valid_mask]
            inputs_valid = inputs[valid_mask]
            targets_valid = targets[valid_mask]
            ft_cal_valid = ft_cal[valid_mask]

            # Compute t_real for valid samples
            t_normalized_valid = inputs_valid[:, 2]
            t_real_valid = torch.exp((t_beta * t_normalized_valid + t_alpha) * ln10)

            # STEP 1: Get unsigned magnitude predictions and compute derivatives separately
            # Detect phase
            is_phase1 = torch.all(torch.abs(ft_cal_valid) < eps)

            # STEP 1a: Compute values (for loss calculation)
            mag_x = mag_preds_valid[:, 0] + ft_cal_valid[:, 0]
            mag_v = mag_preds_valid[:, 1] + ft_cal_valid[:, 1]
            mag_a = mag_preds_valid[:, 2] + ft_cal_valid[:, 2]

            # STEP 1b: Compute derivatives separately, then sum
            # Always compute derivatives of mag_preds components
            dx_mag_dt = torch.autograd.grad(
                outputs=mag_preds_valid[:, 0],
                inputs=inputs,
                grad_outputs=torch.ones_like(mag_preds_valid[:, 0]),
                create_graph=True,
                retain_graph=True,
                allow_unused=True
            )[0][valid_mask, 2]  # Gradient w.r.t. t_normalized, filter by valid_mask

            dv_mag_dt = torch.autograd.grad(
                outputs=mag_preds_valid[:, 1],
                inputs=inputs,
                grad_outputs=torch.ones_like(mag_preds_valid[:, 1]),
                create_graph=True,
                retain_graph=True,
                allow_unused=True
            )[0][valid_mask, 2]  # Gradient w.r.t. t_normalized, filter by valid_mask

            if is_phase1:
                # Phase 1: Only use mag derivatives (ft_cal is zeros)
                dx_prime_dt_prime = torch.abs(dx_mag_dt)
                dv_prime_dt_prime = torch.abs(dv_mag_dt)
            else:
                # Phase 2: Compute ft_cal derivatives and add them WITH RESCALING
                dx_ft_dt = torch.autograd.grad(
                    outputs=ft_cal_valid[:, 0],
                    inputs=inputs,
                    grad_outputs=torch.ones_like(ft_cal_valid[:, 0]),
                    create_graph=True,
                    retain_graph=True,
                    allow_unused=True
                )[0][valid_mask, 2]  # Gradient w.r.t. t_normalized, filter by valid_mask

                dv_ft_dt = torch.autograd.grad(
                    outputs=ft_cal_valid[:, 1],
                    inputs=inputs,
                    grad_outputs=torch.ones_like(ft_cal_valid[:, 1]),
                    create_graph=True,
                    retain_graph=True,
                    allow_unused=True
                )[0][valid_mask, 2]  # Gradient w.r.t. t_normalized, filter by valid_mask

                # RESCALING CONSTRAINT: Prevent ft derivatives from dominating
                # Compute target derivatives first for rescaling bound
                # Extract targets
                x_target = targets_valid[:, 3]  # x' target (SIGNED normalized log-space)
                v_target = targets_valid[:, 4]  # v' target (SIGNED normalized log-space)
                a_target = targets_valid[:, 5]  # a' target (SIGNED normalized log-space)

                # Denormalize targets to real space
                x_target_real = torch.exp((beta_x * x_target + alpha_x) * ln10)
                v_target_real = torch.exp((beta_v * v_target + alpha_v) * ln10)
                a_target_real = torch.exp((beta_a * a_target + alpha_a) * ln10)

                # Compute common factors from targets
                common_factor_v_target = (beta_x / t_beta) * (x_target_real / (t_real_valid + eps))
                common_factor_a_target = (beta_v / t_beta) * (v_target_real / (t_real_valid + eps))

                # Invert to get target derivatives (for bound calculation only)
                dx_dt_target = v_target_real / (common_factor_v_target + eps)
                dv_dt_target = a_target_real / (common_factor_a_target + eps)

                # Calculate scale factor (DETACHED - no gradients affect scaling)
                # Bound: |dx_mag| + |dx_target|
                # Scale: bound / |dx_ft| when |dx_ft| exceeds bound
                with torch.no_grad():
                    bound_x = torch.abs(dx_mag_dt) + torch.abs(dx_dt_target)
                    bound_v = torch.abs(dv_mag_dt) + torch.abs(dv_dt_target)

                    # When |dx_ft| > bound: scale = bound / |dx_ft| to bring it down to bound
                    # When |dx_ft| <= bound: scale = 1.0 (no rescaling)
                    scale_x = torch.clamp(bound_x / (torch.abs(dx_ft_dt) + eps), max=1.0)
                    scale_v = torch.clamp(bound_v / (torch.abs(dv_ft_dt) + eps), max=1.0)

                # Apply scaling (gradients flow through dx_ft_dt, scale is detached)
                dx_ft_dt_stable = dx_ft_dt #* scale_x
                dv_ft_dt_stable = dv_ft_dt #* scale_v

                # Sum derivatives with rescaled ft components
                dx_prime_dt_prime = torch.abs(dx_mag_dt) + dx_ft_dt_stable
                dv_prime_dt_prime = torch.abs(dv_mag_dt) + dv_ft_dt_stable

        else:
            # MODE 2: Gradients computed inside consistency loss (current approach)
            # Call model internally with inputs_with_grad

            if self.model is None:
                raise ValueError("When input_grad_outside=False, model must be set via set_model()")
            if ft_cal is None:
                raise ValueError("When input_grad_outside=False, ft_cal must be provided")

            # Denormalize inputs to get t_real
            t_normalized = inputs[:, 2]
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

            # STEP 1a: Compute values (for loss calculation)
            mag_x = mag_preds_internal[:, 0] + ft_preds_internal[:, 0]
            mag_v = mag_preds_internal[:, 1] + ft_preds_internal[:, 1]
            mag_a = mag_preds_internal[:, 2] + ft_preds_internal[:, 2]

            # STEP 1b: Compute derivatives separately, then sum
            # Always compute derivatives of mag_preds_internal components
            dx_mag_dt = torch.autograd.grad(
                outputs=mag_preds_internal[:, 0],
                inputs=inputs_with_grad,
                grad_outputs=torch.ones_like(mag_preds_internal[:, 0]),
                create_graph=True,
                retain_graph=True,
                allow_unused=True
            )[0][:, 2]  # Gradient w.r.t. t_normalized (index 2)

            dv_mag_dt = torch.autograd.grad(
                outputs=mag_preds_internal[:, 1],
                inputs=inputs_with_grad,
                grad_outputs=torch.ones_like(mag_preds_internal[:, 1]),
                create_graph=True,
                retain_graph=True,
                allow_unused=True
            )[0][:, 2]  # Gradient w.r.t. t_normalized (index 2)

            if is_phase1:
                # Phase 1: Only use mag derivatives (finetune network untrained)
                dx_prime_dt_prime = torch.abs(dx_mag_dt)
                dv_prime_dt_prime = torch.abs(dv_mag_dt)
            else:
                # Phase 2: Compute ft_preds derivatives and add them WITH RESCALING
                dx_ft_dt = torch.autograd.grad(
                    outputs=ft_preds_internal[:, 0],
                    inputs=inputs_with_grad,
                    grad_outputs=torch.ones_like(ft_preds_internal[:, 0]),
                    create_graph=True,
                    retain_graph=True,
                    allow_unused=True
                )[0][:, 2]  # Gradient w.r.t. t_normalized (index 2)

                dv_ft_dt = torch.autograd.grad(
                    outputs=ft_preds_internal[:, 1],
                    inputs=inputs_with_grad,
                    grad_outputs=torch.ones_like(ft_preds_internal[:, 1]),
                    create_graph=True,
                    retain_graph=True,
                    allow_unused=True
                )[0][:, 2]  # Gradient w.r.t. t_normalized (index 2)

                # RESCALING CONSTRAINT: Prevent ft derivatives from dominating
                # Compute target derivatives first for rescaling bound
                # Extract targets
                x_target = targets_valid[:, 3]  # x' target (SIGNED normalized log-space)
                v_target = targets_valid[:, 4]  # v' target (SIGNED normalized log-space)
                a_target = targets_valid[:, 5]  # a' target (SIGNED normalized log-space)

                # Denormalize targets to real space
                x_target_real = torch.exp((beta_x * x_target + alpha_x) * ln10)
                v_target_real = torch.exp((beta_v * v_target + alpha_v) * ln10)
                a_target_real = torch.exp((beta_a * a_target + alpha_a) * ln10)

                # Compute common factors from targets
                common_factor_v_target = (beta_x / t_beta) * (x_target_real / (t_real_valid + eps))
                common_factor_a_target = (beta_v / t_beta) * (v_target_real / (t_real_valid + eps))

                # Invert to get target derivatives (for bound calculation only)
                dx_dt_target = v_target_real / (common_factor_v_target + eps)
                dv_dt_target = a_target_real / (common_factor_a_target + eps)

                # Calculate scale factor (DETACHED - no gradients affect scaling)
                # Bound: |dx_mag| + |dx_target|
                # Scale: bound / |dx_ft| when |dx_ft| exceeds bound
                with torch.no_grad():
                    bound_x = torch.abs(dx_mag_dt) + torch.abs(dx_dt_target)
                    bound_v = torch.abs(dv_mag_dt) + torch.abs(dv_dt_target)

                    # When |dx_ft| > bound: scale = bound / |dx_ft| to bring it down to bound
                    # When |dx_ft| <= bound: scale = 1.0 (no rescaling)
                    scale_x = torch.clamp(bound_x / (torch.abs(dx_ft_dt) + eps), max=1.0)
                    scale_v = torch.clamp(bound_v / (torch.abs(dv_ft_dt) + eps), max=1.0)

                # Apply scaling (gradients flow through dx_ft_dt, scale is detached)
                dx_ft_dt_stable = dx_ft_dt #* scale_x
                dv_ft_dt_stable = dv_ft_dt #* scale_v

                # Sum derivatives with rescaled ft components
                dx_prime_dt_prime = torch.abs(dx_mag_dt) + dx_ft_dt_stable
                dv_prime_dt_prime = torch.abs(dv_mag_dt) + dv_ft_dt_stable

        # ======================
        # COMMON CODE: Apply signs and compute theory values
        # ======================
        # STEP 3: Apply logabs signs (from targets[:, 3:]) → SIGNED normalized log-space predictions
        # targets[:, 3:] contains SIGNED normalized log-space values (can be ±)
        logabs_targets = targets_valid[:, 3:]  # [x', v', a'] in normalized log-space (signed)
        logabs_sign = torch.sign(logabs_targets)  # Extract signs from logabs targets

        x_pred = logabs_sign[:, 0] * mag_x  # Signed normalized log-space
        v_pred = logabs_sign[:, 1] * mag_v  # Signed normalized log-space

        # STEP 4: Keep predictions WITH gradients, but DETACH derivatives
        x_pred_detached = x_pred.detach()  # Detached for denormalization
        v_pred_detached = v_pred.detach()  # Detached for denormalization
        dx_dt_detached = dx_prime_dt_prime.detach()  # DETACH derivatives - theory won't have gradients
        dv_dt_detached = dv_prime_dt_prime.detach()  # DETACH derivatives - theory won't have gradients

        # STEP 5: Denormalize SIGNED predictions to real space
        # This matches dataset: x_prime is signed in normalized space
        x_real = torch.exp((beta_x * x_pred_detached + alpha_x) * ln10)
        v_real = torch.exp((beta_v * v_pred_detached + alpha_v) * ln10)

        # STEP 6: Compute theory values using VERIFIED FORMULA (derivatives detached - no gradients!)
        common_factor_v = (beta_x / t_beta) * (x_real / (t_real_valid + eps))
        v_theory = torch.abs(common_factor_v * dx_dt_detached)

        common_factor_a = (beta_v / t_beta) * (v_real / (t_real_valid + eps))
        a_theory = torch.abs(common_factor_a * dv_dt_detached)

        # STEP 7: Normalize theory back to log-normalized space (no gradients - derivatives detached)
        v_theory_normalized = (torch.log10(v_theory + eps) - alpha_v) / beta_v
        a_theory_normalized = (torch.log10(a_theory + eps) - alpha_a) / beta_a

        # STEP 8: Extract ground truth targets (v' and a' in normalized log-space)
        v_target = targets_valid[:, 4]  # v' target (SIGNED normalized log-space)
        a_target = targets_valid[:, 5]  # a' target (SIGNED normalized log-space)
        
        # STEP 9: Compute TRIANGULAR loss - all three values contribute
        if self.use_log:
            # Log-space MSE - triangular consistency between pred, theory, and target
            log_mag_v = torch.log(torch.abs(mag_v) + eps)  # Prediction has gradients!
            log_v_theory = torch.log(torch.abs(v_theory_normalized) + eps)  # Theory DETACHED (no gradients)
            log_v_target = torch.log(torch.abs(v_target) + eps)  # Target (ground truth)

            # Triangular loss: (pred-theory)² + (theory-target)² + (pred-target)²
            # Only mag_v has gradients - trains predictions to match both theory and target
            v_consistency_loss = torch.mean(
                (log_mag_v - log_v_theory) ** 2 +
                (log_v_theory - log_v_target) ** 2 +
                (log_mag_v - log_v_target) ** 2
            )

            log_mag_a = torch.log(torch.abs(mag_a) + eps)  # Prediction has gradients!
            log_a_theory = torch.log(torch.abs(a_theory_normalized) + eps)  # Theory DETACHED (no gradients)
            log_a_target = torch.log(torch.abs(a_target) + eps)  # Target (ground truth)

            # Triangular loss: (pred-theory)² + (theory-target)² + (pred-target)²
            # Only mag_a has gradients - trains predictions to match both theory and target
            a_consistency_loss = torch.mean(
                (log_mag_a - log_a_theory) ** 2 +
                (log_a_theory - log_a_target) ** 2 +
                (log_mag_a - log_a_target) ** 2
            )
        else:
            # Regular MSE - triangular consistency
            v_consistency_loss = torch.mean(
                ((torch.abs(mag_v) - torch.abs(v_theory_normalized))/torch.abs(mag_v)) ** 2 +
                ((torch.abs(v_theory_normalized) - torch.abs(v_target))/torch.abs(v_theory_normalized)) ** 2 +
                ((torch.abs(mag_v) - torch.abs(v_target))/torch.abs(mag_v)) ** 2
            )

            a_consistency_loss = torch.mean(
                ((torch.abs(mag_a) - torch.abs(a_theory_normalized))/torch.abs(mag_a)) ** 2 +
                ((torch.abs(a_theory_normalized) - torch.abs(a_target))/torch.abs(a_theory_normalized)) ** 2 +
                ((torch.abs(mag_a) - torch.abs(a_target))/torch.abs(mag_a)) ** 2
            )
        
        
        """
        # STEP 9: Compute TRIANGULAR loss - all three values contribute
        if self.use_log:


            # Regular MSE - triangular consistency
            v_consistency_loss = torch.mean(
                torch.log(torch.abs(((torch.abs(mag_v) - torch.abs(v_theory_normalized))/torch.abs(mag_v))+1)) +
                torch.log(torch.abs(((torch.abs(v_theory_normalized) - torch.abs(v_target))/torch.abs(v_theory_normalized))+1)) +
                torch.log(torch.abs(((torch.abs(mag_v) - torch.abs(v_target))/torch.abs(mag_v))+1))
            )

            a_consistency_loss = torch.mean(
                torch.log(torch.abs(((torch.abs(mag_a) - torch.abs(a_theory_normalized))/torch.abs(mag_a))+1)) +
                torch.log(torch.abs(((torch.abs(a_theory_normalized) - torch.abs(a_target))/torch.abs(a_theory_normalized))+1)) +
                torch.log(torch.abs(((torch.abs(mag_a) - torch.abs(a_target))/torch.abs(mag_a))+1))
            )
    
        else:
            # Regular MSE - triangular consistency
            v_consistency_loss = torch.mean(
                ((torch.abs(mag_v) - torch.abs(v_theory_normalized))/torch.abs(mag_v)) ** 2 +
                ((torch.abs(v_theory_normalized) - torch.abs(v_target))/torch.abs(v_theory_normalized)) ** 2 +
                ((torch.abs(mag_v) - torch.abs(v_target))/torch.abs(mag_v)) ** 2
            )

            a_consistency_loss = torch.mean(
                ((torch.abs(mag_a) - torch.abs(a_theory_normalized))/torch.abs(mag_a)) ** 2 +
                ((torch.abs(a_theory_normalized) - torch.abs(a_target))/torch.abs(a_theory_normalized)) ** 2 +
                ((torch.abs(mag_a) - torch.abs(a_target))/torch.abs(mag_a)) ** 2
            )

        """
        # STEP 10: Total consistency loss
        total_loss = v_consistency_loss + a_consistency_loss

        return total_loss
    
'''
class ConsistencyLoss_auto_diff(BaseLossComponent):
    """
    Derivative consistency using VERIFIED formulas for log-normalized space training.

    Ensures v=dx/dt and a=dv/dt using autograd on log-normalized model outputs.
    Uses verified transformation formulas from check_dataset_consistency() in Exp_dataset.py.

    Key features:
    - Model outputs UNSIGNED magnitudes (positive from Softplus) in normalized log-space
    - Applies logabs signs from targets[:, 3:] to create SIGNED normalized log-space predictions
    - Computes derivatives of UNSIGNED magnitudes via autograd in normalized space
    - Applies verified denormalization formulas with SIGNED predictions for theory values
    - Supports both log-space MSE and regular MSE via use_log parameter
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
            inputs: (batch_size, 3) - [a, b, t] NORMALIZED
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

        # Get normalization parameters (needed for both modes)
        beta_x = generalize_beta(outputs_normalizer, outputs_normalizer.log_mean['x'], outputs_normalizer.log_std['x'], 'x')
        beta_v = generalize_beta(outputs_normalizer, outputs_normalizer.log_mean['v'], outputs_normalizer.log_std['v'], 'v')
        beta_a = generalize_beta(outputs_normalizer, outputs_normalizer.log_mean['a'], outputs_normalizer.log_std['a'], 'a')
        alpha_x = generalize_alpha(outputs_normalizer, outputs_normalizer.log_mean['x'], outputs_normalizer.log_std['x'], 'x')
        alpha_v = generalize_alpha(outputs_normalizer, outputs_normalizer.log_mean['v'], outputs_normalizer.log_std['v'], 'v')
        alpha_a = generalize_alpha(outputs_normalizer, outputs_normalizer.log_mean['a'], outputs_normalizer.log_std['a'], 'a')
        t_alpha = generalize_alpha(inputs_normalizer, inputs_normalizer.log_mean['t'], inputs_normalizer.log_std['t'], 't')
        t_beta = generalize_beta(inputs_normalizer, inputs_normalizer.log_mean['t'], inputs_normalizer.log_std['t'], 't')
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
            mag_preds_valid = predictions[valid_mask]
            inputs_valid = inputs[valid_mask]
            targets_valid = targets[valid_mask]
            ft_cal_valid = ft_cal[valid_mask]

            # Compute t_real for valid samples
            t_normalized_valid = inputs_valid[:, 2]
            t_real_valid = torch.exp((t_beta * t_normalized_valid + t_alpha) * ln10)

            # STEP 1: Get unsigned magnitude predictions and compute derivatives separately
            # Detect phase
            is_phase1 = torch.all(torch.abs(ft_cal_valid) < eps)

            # STEP 1b: Compute derivatives separately, then sum
            # Always compute derivatives of mag_preds components
            dx_mag_dt = torch.autograd.grad(
                outputs=mag_preds_valid[:, 0],
                inputs=inputs,
                grad_outputs=torch.ones_like(mag_preds_valid[:, 0]),
                create_graph=True,
                retain_graph=True,
                allow_unused=True
            )[0][valid_mask, 2]  # Gradient w.r.t. t_normalized, filter by valid_mask

            dv_mag_dt = torch.autograd.grad(
                outputs=mag_preds_valid[:, 1],
                inputs=inputs,
                grad_outputs=torch.ones_like(mag_preds_valid[:, 1]),
                create_graph=True,
                retain_graph=True,
                allow_unused=True
            )[0][valid_mask, 2]  # Gradient w.r.t. t_normalized, filter by valid_mask

            if is_phase1:
                # Phase 1: Only use mag derivatives (ft_cal is zeros)
                dx_prime_dt_prime = torch.abs(dx_mag_dt)
                dv_prime_dt_prime = torch.abs(dv_mag_dt)
            else:
                # Phase 2: Compute ft_cal derivatives and add them
                dx_ft_dt = torch.autograd.grad(
                    outputs=ft_cal_valid[:, 0],
                    inputs=inputs,
                    grad_outputs=torch.ones_like(ft_cal_valid[:, 0]),
                    create_graph=True,
                    retain_graph=True,
                    allow_unused=True
                )[0][valid_mask, 2]  # Gradient w.r.t. t_normalized, filter by valid_mask

                dv_ft_dt = torch.autograd.grad(
                    outputs=ft_cal_valid[:, 1],
                    inputs=inputs,
                    grad_outputs=torch.ones_like(ft_cal_valid[:, 1]),
                    create_graph=True,
                    retain_graph=True,
                    allow_unused=True
                )[0][valid_mask, 2]  # Gradient w.r.t. t_normalized, filter by valid_mask

                # Sum derivatives (linearity: d(f+g)/dt = df/dt + dg/dt)
                dx_prime_dt_prime = torch.abs(dx_mag_dt) + dx_ft_dt
                dv_prime_dt_prime = torch.abs(dv_mag_dt) + dv_ft_dt
                # dx_prime_dt_prime = torch.abs(dx_mag_dt)
                # dv_prime_dt_prime = torch.abs(dv_mag_dt)
        else:
            # MODE 2: Gradients computed inside consistency loss (current approach)
            # Call model internally with inputs_with_grad

            if self.model is None:
                raise ValueError("When input_grad_outside=False, model must be set via set_model()")
            if ft_cal is None:
                raise ValueError("When input_grad_outside=False, ft_cal must be provided")

            # Denormalize inputs to get t_real
            t_normalized = inputs[:, 2]
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
            )[0][:, 2]  # Gradient w.r.t. t_normalized (index 2)

            dv_mag_dt = torch.autograd.grad(
                outputs=mag_preds_internal[:, 1],
                inputs=inputs_with_grad,
                grad_outputs=torch.ones_like(mag_preds_internal[:, 1]),
                create_graph=True,
                retain_graph=True,
                allow_unused=True
            )[0][:, 2]  # Gradient w.r.t. t_normalized (index 2)

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
                )[0][:, 2]  # Gradient w.r.t. t_normalized (index 2)

                dv_ft_dt = torch.autograd.grad(
                    outputs=ft_preds_internal[:, 1],
                    inputs=inputs_with_grad,
                    grad_outputs=torch.ones_like(ft_preds_internal[:, 1]),
                    create_graph=True,
                    retain_graph=True,
                    allow_unused=True
                )[0][:, 2]  # Gradient w.r.t. t_normalized (index 2)

                # Sum derivatives (linearity: d(f+g)/dt = df/dt + dg/dt)
                dx_prime_dt_prime = torch.abs(dx_mag_dt) + dx_ft_dt
                dv_prime_dt_prime = torch.abs(dv_mag_dt) + dv_ft_dt
                # dx_prime_dt_prime = torch.abs(dx_mag_dt)
                # dv_prime_dt_prime = torch.abs(dv_mag_dt)

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
            # Pass mag_preds, targets, logabs_sign_probs, and real_sign_probs
            logabs_sign_probs = model.logabs_last_sign_probs
            real_sign_probs = model.real_last_sign_probs
            loss_args["MSE"] = (mag_preds, targets, logabs_sign_probs, None, None, real_sign_probs)
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
            self.loss_components["Residual"] = ExponentialResidualLoss(weight=weight, use_relative=use_relative)

        # Initialize Consistency loss if requested (auto diff only)
        if self._should_enable("Consistency"):
            config = self.loss_config.get("Consistency")
            t_threshold = config.get("t_threshold", 1e-6)
            weight = config.get("weight", 1.0)
            use_log = config.get("use_log", True)
            input_grad_outside = config.get("Input_grad_outside", False)
            self.loss_components["Consistency"] = ConsistencyLoss_auto_diff(
                weight=weight, model=model, t_threshold=t_threshold, use_log=use_log,
                input_grad_outside=input_grad_outside
            )

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
                    "MSE": (mag_preds, targets, logabs_sign_probs, None, None, real_sign_probs, ft_cal, output_normalizer),
                    "Residual": (outputs, targets, inputs_real, output_normalizer),
                    "Consistency": (inputs, inputs_normalizer, outputs_normalizer, targets, ft_cal)
                }

        Returns:
            total_loss: Scalar tensor
            loss_summary: Dictionary with individual loss values
        """
        total_loss = 0.0
        loss_summary = {}

        # MSE Loss
        if "MSE" in self.loss_components and "MSE" in loss_args:
            # Unpack 8 arguments: (mag_preds, targets, logabs_sign_probs, None, None, real_sign_probs, ft_cal, output_normalizer)
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
