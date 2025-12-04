"""Test script for binary classification sign network with cross-entropy loss"""
import torch
import sys
import os
sys.path.append('.')

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

from Exp_modelandloss import ExponentialPINN, ExponentialPINNLoss

def test_sign_classification():
    """Test the binary classification sign network implementation"""
    print("="*60)
    print("Testing Binary Classification Sign Network")
    print("="*60)

    # Create model with sign network enabled
    model = ExponentialPINN(
        hidden_dims=[16, 32, 32, 16],
        activation='relu',
        use_log_output=False,
        use_finetune=True,
        finetune_hidden_dims=[16, 32, 16],
        finetune_scale=1,
        use_sign_network=True,
        sign_network_hidden_dims=[16, 32, 16]
    )

    # Configure losses with log-space MSE
    loss_config = {
        "MSE": {"weight": 1.0, "use_relative": False, "use_log": True},
        "Residual": {"weight": 0.0, "use_relative": False},
        "Consistency": {"weight": 0.0, "t_threshold": 1e-5, "type": "auto", "use_relative": False}
    }

    loss_fn = ExponentialPINNLoss(model, loss_config)

    # Create dummy input data
    batch_size = 8
    # Input: [a, b, t]
    inputs = torch.randn(batch_size, 3)

    # Create targets with mixed signs
    targets = torch.randn(batch_size, 3)
    # Ensure we have both negative and positive values
    targets[0:4, :] = -torch.abs(targets[0:4, :])  # Make first half negative
    targets[4:8, :] = torch.abs(targets[4:8, :])   # Make second half positive

    print("\n1. Testing Forward Pass")
    print("-" * 60)

    # Forward pass
    predictions = model(inputs)

    print(f"Input shape: {inputs.shape}")
    print(f"Target shape: {targets.shape}")
    print(f"Prediction shape: {predictions.shape}")

    # Check if logits were stored
    if hasattr(model, 'last_sign_logits'):
        print(f"✓ Sign logits stored: shape = {model.last_sign_logits.shape}")
        print(f"  Expected shape: [batch_size={batch_size}, 3 variables, 2 classes]")
        assert model.last_sign_logits.shape == (batch_size, 3, 2), "Incorrect logits shape!"
    else:
        print("✗ Error: Sign logits not stored!")
        return False

    print("\n2. Testing Loss Computation")
    print("-" * 60)

    # Prepare loss arguments
    loss_args = {
        "MSE": (predictions, targets)
    }

    # Compute loss
    total_loss, loss_dict = loss_fn(loss_args)

    print(f"Total loss: {total_loss.item():.6f}")
    print(f"Loss components:")
    for key, value in loss_dict.items():
        print(f"  {key}: {value:.6f}")

    # Check if loss is computed correctly
    assert total_loss.item() > 0, "Loss should be positive!"
    assert not torch.isnan(total_loss), "Loss should not be NaN!"
    assert not torch.isinf(total_loss), "Loss should not be infinite!"

    print("\n3. Testing Gradient Flow")
    print("-" * 60)

    # Backward pass
    total_loss.backward()

    # Check gradients
    has_grads = False
    for name, param in model.named_parameters():
        if param.grad is not None:
            has_grads = True
            grad_norm = param.grad.norm().item()
            if 'sign_network' in name:
                print(f"✓ {name}: grad_norm = {grad_norm:.6f}")

    if has_grads:
        print("✓ Gradients computed successfully!")
    else:
        print("✗ No gradients found!")
        return False

    print("\n4. Testing Sign Prediction Behavior")
    print("-" * 60)

    # Examine sign predictions vs targets
    target_signs = torch.sign(targets)
    target_classes = (target_signs >= 0).long()

    # Get predicted probabilities
    sign_probs = torch.softmax(model.last_sign_logits, dim=2)
    pred_classes = torch.argmax(sign_probs, dim=2)

    print(f"Target signs (first sample): {target_signs[0].numpy()}")
    print(f"Target classes (first sample): {target_classes[0].numpy()}")
    print(f"Predicted probabilities (first sample):")
    print(f"  P(negative): {sign_probs[0, :, 0].detach().numpy()}")
    print(f"  P(positive): {sign_probs[0, :, 1].detach().numpy()}")
    print(f"Predicted classes (first sample): {pred_classes[0].numpy()}")

    # Compute soft signs
    soft_signs = sign_probs[:, :, 1] - sign_probs[:, :, 0]
    print(f"Soft signs (first sample): {soft_signs[0].detach().numpy()}")

    print("\n5. Testing Multiple Forward/Backward Passes")
    print("-" * 60)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for step in range(5):
        optimizer.zero_grad()

        # Forward pass
        predictions = model(inputs)

        # Loss computation
        loss_args = {"MSE": (predictions, targets)}
        total_loss, _ = loss_fn(loss_args)

        # Backward pass
        total_loss.backward()
        optimizer.step()

        print(f"  Step {step+1}: Loss = {total_loss.item():.6f}")

    print("\n" + "="*60)
    print("✓ All tests passed successfully!")
    print("="*60)

    return True


if __name__ == "__main__":
    try:
        success = test_sign_classification()
        if success:
            print("\n✓ Binary classification sign network implementation is working correctly!")
            sys.exit(0)
        else:
            print("\n✗ Tests failed!")
            sys.exit(1)
    except Exception as e:
        print(f"\n✗ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
