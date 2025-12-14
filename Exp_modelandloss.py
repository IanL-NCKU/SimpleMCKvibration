import torch
import torch.nn as nn
import numpy as np
from abc import ABC, abstractmethod

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

    def __init__(self, hidden_dims=[64, 128, 128, 64], activation='tanh', use_log_output=False,
                 use_finetune=False, finetune_hidden_dims=[32, 32], finetune_scale=0.1,
                 logabs_sign_network_hidden_dims=[128, 64, 32], logabs_sign_network_dropout=0.3,
                 real_sign_network_hidden_dims=[128, 64, 32], real_sign_network_dropout=0.3):
        super().__init__()

        self.use_log_output = use_log_output
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

        # Step 5: Compute calibration factors (independent network with detached inputs)
        if self.use_finetune:
            # Use DETACHED inputs: [a, b, t] + sign_probs + mag_preds
            finetune_input = torch.cat([x, self.logabs_last_sign_probs, mag_preds], dim=1).detach()  # Shape: [batch, 9]
            
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
                signed_mag_preds = predictions.detach() 
                calibrated_preds = signed_mag_preds + ft_cal  # Only ft_cal has gradient

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

    def compute(self, predictions, targets, inputs=None, inputs_normalizer=None,
                inputs_real=None, **kwargs):
        """
        Enforces consistency between position, velocity, and acceleration
        using automatic differentiation with log-normalized time.

        Args:
            inputs: (batch_size, 3) - [a, b, t] NORMALIZED
            inputs_normalizer: Normalizer instance containing log-normalization parameters
            inputs_real: (batch_size, 3) - [a, b, t] in REAL space

        Returns:
            loss: Scalar tensor measuring consistency error
        """
        if self.model is None:
            raise ValueError("Model not set. Call set_model() before using ConsistencyLoss_auto_diff")

        if inputs_real is None:
            raise ValueError("ConsistencyLoss_auto_diff requires inputs_real to be provided")

        if inputs_normalizer is None:
            raise ValueError("inputs_normalizer must be provided for ConsistencyLoss_auto_diff")

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
        if inputs_normalizer is not None:
            t_std = inputs_normalizer.log_std['t']  # std of log10(t) values
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
        if inputs_normalizer is not None:
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
            eps = 1e-12
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
            self.loss_components["Consistency"] = ConsistencyLoss_auto_diff(
                weight=weight, model=model, t_threshold=t_threshold, use_log=use_log
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

        # Consistency Loss (auto diff only)
        if "Consistency" in self.loss_components and "Consistency" in loss_args:
            # Auto diff: (inputs, inputs_real, inputs_normalizer)
            inputs, inputs_real, inputs_normalizer = loss_args["Consistency"]
            consistency_value = self.loss_components["Consistency"](
                None, None,
                inputs=inputs,
                inputs_normalizer=inputs_normalizer,
                inputs_real=inputs_real
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
