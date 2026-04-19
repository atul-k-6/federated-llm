from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
import torch
 
def make_private(model, optimizer, data_loader, config):
    """
    Wrap model, optimizer, and loader with Opacus PrivacyEngine.
    Returns: (private_model, private_optimizer, private_loader, privacy_engine)
    """
    noise_multiplier = config.get("noise_multiplier", 0.0)
    max_grad_norm    = config.get("max_grad_norm", 1.0)
    target_epsilon   = config.get("target_epsilon", None)
    target_delta     = config.get("target_delta", 1e-5)
 
    if noise_multiplier == 0.0:
        # No DP: return unchanged
        print("DP disabled (noise_multiplier=0)")
        return model, optimizer, data_loader, None
 
    # Validate model compatibility with Opacus
    # Some layers (BatchNorm) are not compatible; fix them automatically
    errors = ModuleValidator.validate(model, strict=False)
    if errors:
        print(f"Auto-fixing {len(errors)} module compatibility issues...")
        model = ModuleValidator.fix(model)
 
    privacy_engine = PrivacyEngine()
 
    if target_epsilon is not None:
        # Let Opacus calculate the noise_multiplier to achieve target_epsilon
        private_model, private_optimizer, private_loader = \
            privacy_engine.make_private_with_epsilon(
                module=model,
                optimizer=optimizer,
                data_loader=data_loader,
                epochs=config.get("local_epochs", 2),
                target_epsilon=target_epsilon,
                target_delta=target_delta,
                max_grad_norm=max_grad_norm,
            )
        print(f"DP enabled: target_epsilon={target_epsilon}, "
              f"computed sigma={private_optimizer.noise_multiplier:.3f}")
    else:
        # Manually set noise_multiplier
        private_model, private_optimizer, private_loader = \
            privacy_engine.make_private(
                module=model,
                optimizer=optimizer,
                data_loader=data_loader,
                noise_multiplier=noise_multiplier,
                max_grad_norm=max_grad_norm,
            )
        print(f"DP enabled: noise_multiplier={noise_multiplier}")
 
    return private_model, private_optimizer, private_loader, privacy_engine
 
 
def get_epsilon(privacy_engine, delta=1e-5):
    """Get current privacy budget spent so far."""
    if privacy_engine is None:
        return float("inf")  # No DP = infinite budget spent
    return privacy_engine.get_epsilon(delta=delta)
