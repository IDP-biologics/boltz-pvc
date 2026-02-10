"""Vanilla PyTorch checkpoint loader for Boltz models."""

from pathlib import Path
from typing import Optional, Union

import torch
from torch import nn

# Intel Extension for PyTorch (for XPU support)
try:
    import intel_extension_for_pytorch as ipex
    IPEX_AVAILABLE = True
except ImportError:
    IPEX_AVAILABLE = False

from boltz.model.models.boltz1 import Boltz1
from boltz.model.models.boltz2 import Boltz2
from boltz.model.modules.utils import ExponentialMovingAverage


def load_model(
    checkpoint_path: Union[str, Path],
    model_class: str = "boltz1",
    device: str = "cpu",
    use_ema: bool = True,
    use_kernels: bool = False,
    predict_args: Optional[dict] = None,
    **model_kwargs,
) -> nn.Module:
    """
    Load a Boltz model from a checkpoint using vanilla PyTorch.

    Args:
        checkpoint_path: Path to the checkpoint file
        model_class: Either "boltz1" or "boltz2"
        device: Device to load the model on
        use_ema: Whether to load EMA weights if available
        use_kernels: Whether to use cuEquivariance CUDA kernels (default: False for compatibility)
        predict_args: Prediction arguments to pass to the model
        **model_kwargs: Additional arguments to pass to the model constructor

    Returns:
        Loaded model ready for inference
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Load checkpoint
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get model class
    model_cls = Boltz2 if model_class.lower() == "boltz2" else Boltz1
    
    # Extract hyperparameters from checkpoint
    if "hyper_parameters" in checkpoint:
        hparams = checkpoint["hyper_parameters"]
    else:
        hparams = {}
    
    # Merge with provided kwargs
    hparams.update(model_kwargs)

    # Add predict_args if provided
    if predict_args is not None:
        hparams["predict_args"] = predict_args

    # Set use_kernels parameter
    hparams["use_kernels"] = use_kernels

    # Initialize model
    print(f"Initializing {model_class} model...")
    print(f"  use_kernels: {use_kernels}")
    if device == "xpu" and IPEX_AVAILABLE:
        print(f"  Intel Extension for PyTorch: Available (version {ipex.__version__})")
    elif device == "xpu" and not IPEX_AVAILABLE:
        print("  WARNING: Intel Extension for PyTorch not available for XPU device!")
    model = model_cls(**hparams)
    
    # Load state dict
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    
    # Remove any "model." prefix if present
    state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict, strict=False)
    
    # Load EMA weights if requested and available
    if use_ema and "ema" in checkpoint:
        print("Loading EMA weights...")
        ema = ExponentialMovingAverage(
            parameters=model.parameters(),
            decay=hparams.get("ema_decay", 0.999)
        )
        if ema.compatible(checkpoint["ema"]["shadow_params"]):
            ema.load_state_dict(checkpoint["ema"], device=torch.device(device))
            # Apply EMA weights to model
            ema.copy_to(model.parameters())
            print("EMA weights loaded and applied")
        else:
            print("Warning: EMA weights not compatible with model, skipping")
    
    # Move to device and set to eval mode
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded successfully on {device}")
    return model

