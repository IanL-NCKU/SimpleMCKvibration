"""Quick test of simplified sign network - no output, just passes/fails"""
import torch
from Exp_modelandloss import ExponentialPINN

# Create model
model = ExponentialPINN(
    hidden_dims=[16, 32, 16],
    activation='tanh',
    use_log_output=False,
    use_finetune=False,
    use_sign_network=True,
    sign_network_hidden_dims=[16]
)

# Test forward pass
inputs = torch.randn(4, 3)
outputs = model(inputs)

# Verify sign logits shape
assert hasattr(model, 'last_sign_logits'), "Missing last_sign_logits"
assert model.last_sign_logits.shape == (4, 3), f"Wrong shape: {model.last_sign_logits.shape}"

# Verify signs are in [-1, 1] range
sign_probs = torch.sigmoid(model.last_sign_logits)
predicted_signs = 2.0 * sign_probs - 1.0
assert predicted_signs.min() >= -1.0 and predicted_signs.max() <= 1.0, "Signs out of range"

# Test backward pass
targets = torch.randn(4, 3)
loss = torch.mean((outputs - targets) ** 2)
loss.backward()

# Check gradients exist
has_grads = any(p.grad is not None for p in model.sign_network.parameters())
assert has_grads, "No gradients in sign network"

print("SUCCESS: All tests passed!")
print(f"  - Sign logits shape: {model.last_sign_logits.shape}")
print(f"  - Sign range: [{predicted_signs.min().item():.3f}, {predicted_signs.max().item():.3f}]")
print(f"  - Gradients: OK")
