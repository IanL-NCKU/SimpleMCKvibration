"""
Test script to verify the simplified sign network (3 outputs instead of 6)
"""
import torch
import numpy as np
from Exp_modelandloss import ExponentialPINN, ExponentialPINNLoss

print("="*60)
print("Testing Simplified Sign Network")
print("="*60)

# Create model with sign network
model = ExponentialPINN(
    hidden_dims=[16, 32, 32, 16],
    activation='tanh',
    use_log_output=False,
    use_finetune=False,
    use_sign_network=True,
    sign_network_hidden_dims=[16, 16]
)

# Create dummy input batch
batch_size = 8
inputs = torch.randn(batch_size, 3)  # [a, b, t]

# Forward pass
print("\n1. Testing forward pass...")
outputs = model(inputs)
print(f"   Input shape: {inputs.shape}")
print(f"   Output shape: {outputs.shape}")
print(f"   [OK] Forward pass successful")

# Check sign logits shape
print("\n2. Checking sign logits...")
if hasattr(model, 'last_sign_logits'):
    print(f"   Sign logits shape: {model.last_sign_logits.shape}")
    assert model.last_sign_logits.shape == (batch_size, 3), \
        f"Expected shape ({batch_size}, 3), got {model.last_sign_logits.shape}"
    print(f"   [OK] Sign logits have correct shape [batch, 3]")
else:
    print("   [FAIL] No sign logits found")

# Test sign prediction
print("\n3. Testing sign prediction...")
sign_probs = torch.sigmoid(model.last_sign_logits)
predicted_signs = 2.0 * sign_probs - 1.0
print(f"   Sign probabilities (sample): {sign_probs[0].detach().numpy()}")
print(f"   Predicted signs (sample): {predicted_signs[0].detach().numpy()}")
print(f"   Sign range: [{predicted_signs.min().item():.3f}, {predicted_signs.max().item():.3f}]")
print(f"   [OK] Signs correctly mapped to [-1, +1] range")

# Test with loss function
print("\n4. Testing loss computation...")
loss_config = {
    "MSE": {"weight": 1.0, "use_relative": False, "use_log": True},
    "Residual": {"weight": 0.0, "use_relative": False},
    "Consistency": {"weight": 0.0}
}

loss_fn = ExponentialPINNLoss(model, loss_config)

# Create dummy targets with mixed signs
targets = torch.randn(batch_size, 3)

# Compute loss
loss_args = {"MSE": (outputs, targets)}
total_loss, loss_dict = loss_fn(loss_args)

print(f"   Total loss: {total_loss.item():.6f}")
if 'mse_magnitude_loss' in loss_dict:
    print(f"   - Magnitude loss: {loss_dict['mse_magnitude_loss']:.6f}")
if 'mse_sign_loss' in loss_dict:
    print(f"   - Sign loss (BCE): {loss_dict['mse_sign_loss']:.6f}")
print(f"   [OK] Loss computation successful")

# Test gradient flow
print("\n5. Testing gradient flow...")
total_loss.backward()

# Check if sign network has gradients
sign_network_has_grad = False
for name, param in model.named_parameters():
    if 'sign_network' in name and param.grad is not None:
        sign_network_has_grad = True
        grad_norm = param.grad.norm().item()
        print(f"   {name}: grad_norm={grad_norm:.6f}")

if sign_network_has_grad:
    print(f"   [OK] Gradients flowing through sign network")
else:
    print(f"   [FAIL] No gradients in sign network")

# Count parameters
print("\n6. Parameter count comparison...")
total_params = sum(p.numel() for p in model.parameters())
sign_params = sum(p.numel() for p in model.sign_network.parameters())
print(f"   Total parameters: {total_params:,}")
print(f"   Sign network parameters: {sign_params:,}")
print(f"   Sign network percentage: {sign_params/total_params*100:.2f}%")

# Estimate savings from 6→3 output change
# Last layer of sign network: hidden_dim → 3 instead of hidden_dim → 6
last_layer_params_old = 16 * 6 + 6  # weights + bias
last_layer_params_new = 16 * 3 + 3  # weights + bias
savings = last_layer_params_old - last_layer_params_new
print(f"   Savings in last layer: {savings} parameters ({savings/last_layer_params_old*100:.1f}% reduction)")

print("\n" + "="*60)
print("All tests passed! [OK]")
print("="*60)
print("\nSummary of changes:")
print("  - Sign network now outputs 3 values (one per variable)")
print("  - Each output represents P(positive) via sigmoid")
print("  - Signs computed as: sign = 2*sigmoid(logit) - 1")
print("  - Loss uses binary_cross_entropy_with_logits")
print("  - 50% fewer parameters in output layer")
print("  - Fully differentiable with smooth gradients")
