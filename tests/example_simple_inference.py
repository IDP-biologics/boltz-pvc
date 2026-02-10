"""
Simple example of running Boltz inference with vanilla PyTorch.

This is a minimal example showing how to use the Boltz models without
PyTorch Lightning or the command-line interface.
"""

import torch
from pathlib import Path

# Intel Extension for PyTorch (for XPU support)
try:
    import intel_extension_for_pytorch as ipex
    IPEX_AVAILABLE = True
except ImportError:
    IPEX_AVAILABLE = False

# Import the vanilla PyTorch inference utilities
from boltz.inference import load_model, BoltzInferenceRunner
from boltz.data.module.inference import BoltzInferenceDataset
from torch.utils.data import DataLoader


def run_simple_inference():
    """
    Simple example of running Boltz inference.
    
    Modify the paths and parameters below to match your setup.
    """
    
    # ========== Configuration ==========
    
    # Path to your checkpoint file
    checkpoint_path = "path/to/your/checkpoint.ckpt"
    
    # Path to your input data (YAML file or directory)
    input_path = "path/to/your/input.yaml"
    
    # Output directory for predictions
    output_dir = "./predictions"
    
    # Model type: "boltz1" or "boltz2"
    model_type = "boltz1"

    # Device: "xpu" (Intel XPU), "cuda" (NVIDIA GPU), or "cpu"
    # Auto-detect available device
    if IPEX_AVAILABLE and hasattr(torch, 'xpu') and torch.xpu.is_available():
        device = "xpu"
        print(f"Using Intel XPU with IPEX version {ipex.__version__}")
    elif torch.cuda.is_available():
        device = "cuda"
        print("Using CUDA GPU")
    else:
        device = "cpu"
        print("Using CPU")
    
    # Inference parameters
    recycling_steps = 3
    sampling_steps = 200
    diffusion_samples = 1

    # Use CUDA kernels (requires cuEquivariance, GPU with compute capability >= 8.0)
    # Set to False for maximum compatibility (works on CPU, older GPUs, without cuEquivariance)
    use_kernels = False

    # ========== Load Model ==========
    
    print(f"Loading {model_type} model from {checkpoint_path}...")
    
    model = load_model(
        checkpoint_path=checkpoint_path,
        model_class=model_type,
        device=device,
        use_ema=True,  # Use EMA weights if available
        use_kernels=use_kernels,  # Use CUDA kernels (requires cuEquivariance)
        predict_args={
            "recycling_steps": recycling_steps,
            "sampling_steps": sampling_steps,
            "diffusion_samples": diffusion_samples,
        }
    )
    
    print(f"Model loaded on {device}")
    
    # ========== Prepare Data ==========
    
    print(f"Loading data from {input_path}...")
    
    # Create dataset
    dataset = BoltzInferenceDataset(
        data_dir=input_path,
    )
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )
    
    print(f"Loaded {len(dataset)} samples")
    
    # ========== Run Inference ==========
    
    print("Running inference...")
    
    # Create inference runner
    runner = BoltzInferenceRunner(
        model=model,
        device=device,
        output_dir=output_dir,
    )
    
    # Run predictions
    predictions = runner.predict(
        dataloader=dataloader,
        recycling_steps=recycling_steps,
        sampling_steps=sampling_steps,
        diffusion_samples=diffusion_samples,
        write_confidence_summary=True,
        write_full_pae=True,
        write_full_pde=False,
    )
    
    # ========== Process Results ==========
    
    print(f"\nInference complete! Generated {len(predictions)} predictions")
    
    # Example: Access prediction results
    for i, pred in enumerate(predictions):
        if pred.get("exception", False):
            print(f"  Prediction {i}: FAILED (out of memory)")
            continue
        
        print(f"  Prediction {i}:")
        print(f"    - Coordinates shape: {pred['coords'].shape}")
        
        if "confidence_score" in pred:
            print(f"    - Confidence score: {pred['confidence_score'].item():.3f}")
        
        if "plddt" in pred:
            print(f"    - pLDDT: {pred['plddt'].mean().item():.3f}")
        
        if "ptm" in pred:
            print(f"    - pTM: {pred['ptm'].item():.3f}")
    
    print(f"\nResults saved to: {output_dir}")
    
    return predictions


if __name__ == "__main__":
    # Run the inference
    predictions = run_simple_inference()
    
    print("\n" + "="*60)
    print("Done! You can now process the predictions as needed.")
    print("="*60)

