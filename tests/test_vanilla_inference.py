"""
Test script for vanilla PyTorch inference with Boltz models.

This script demonstrates how to run Boltz inference without using PyTorch Lightning
or the command-line interface.
"""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# Intel Extension for PyTorch (for XPU support)
try:
    import intel_extension_for_pytorch as ipex
    IPEX_AVAILABLE = True
except ImportError:
    IPEX_AVAILABLE = False

from boltz.data.module.inference import BoltzInferenceDataset
from boltz.data.module.inferencev2 import Boltz2InferenceDataset
from boltz.inference import BoltzInferenceRunner, load_model


def main():
    parser = argparse.ArgumentParser(description="Run Boltz inference with vanilla PyTorch")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input YAML or directory",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./predictions",
        help="Output directory for predictions",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["boltz1", "boltz2"],
        default="boltz1",
        help="Model type to use",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu, auto-detected if not specified)",
    )
    parser.add_argument(
        "--recycling-steps",
        type=int,
        default=3,
        help="Number of recycling steps",
    )
    parser.add_argument(
        "--sampling-steps",
        type=int,
        default=200,
        help="Number of diffusion sampling steps",
    )
    parser.add_argument(
        "--diffusion-samples",
        type=int,
        default=1,
        help="Number of diffusion samples",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for inference",
    )
    parser.add_argument(
        "--use-ema",
        action="store_true",
        default=True,
        help="Use EMA weights if available",
    )
    parser.add_argument(
        "--no-ema",
        action="store_false",
        dest="use_ema",
        help="Don't use EMA weights",
    )
    parser.add_argument(
        "--use-kernels",
        action="store_true",
        default=False,
        help="Use cuEquivariance CUDA kernels (requires GPU with compute capability >= 8.0)",
    )

    args = parser.parse_args()

    # Auto-detect device
    if args.device is None:
        if IPEX_AVAILABLE and hasattr(torch, 'xpu') and torch.xpu.is_available():
            args.device = "xpu"
        elif torch.cuda.is_available():
            args.device = "cuda"
        else:
            args.device = "cpu"

    print("=" * 80)
    print("Boltz Vanilla PyTorch Inference")
    print("=" * 80)
    print(f"Model: {args.model}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Device: {args.device}")
    if args.device == "xpu" and IPEX_AVAILABLE:
        print(f"Intel Extension for PyTorch: {ipex.__version__}")
    elif args.device == "xpu" and not IPEX_AVAILABLE:
        print("WARNING: XPU device requested but Intel Extension for PyTorch not available!")
    print(f"Recycling steps: {args.recycling_steps}")
    print(f"Sampling steps: {args.sampling_steps}")
    print(f"Diffusion samples: {args.diffusion_samples}")
    print(f"Use EMA: {args.use_ema}")
    print(f"Use kernels: {args.use_kernels}")
    print("=" * 80)
    
    # Prepare prediction arguments
    predict_args = {
        "recycling_steps": args.recycling_steps,
        "sampling_steps": args.sampling_steps,
        "diffusion_samples": args.diffusion_samples,
        "write_confidence_summary": True,
        "write_full_pae": True,
        "write_full_pde": False,
    }
    
    # Load model
    print("\n[1/3] Loading model...")
    model = load_model(
        checkpoint_path=args.checkpoint,
        model_class=args.model,
        device=args.device,
        use_ema=args.use_ema,
        use_kernels=args.use_kernels,
        predict_args=predict_args,
    )
    
    # Create dataset and dataloader
    print("\n[2/3] Preparing data...")
    dataset_cls = Boltz2InferenceDataset if args.model == "boltz2" else BoltzInferenceDataset
    dataset = dataset_cls(
        data_dir=args.input,
        # Add other dataset parameters as needed
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,  # Set to 0 for simplicity, increase for performance
    )
    
    # Run inference
    print("\n[3/3] Running inference...")
    runner = BoltzInferenceRunner(
        model=model,
        device=args.device,
        output_dir=args.output,
    )
    
    predictions = runner.predict(
        dataloader=dataloader,
        recycling_steps=args.recycling_steps,
        sampling_steps=args.sampling_steps,
        diffusion_samples=args.diffusion_samples,
    )
    
    print(f"\n✓ Inference complete! Generated {len(predictions)} predictions")
    print(f"  Output directory: {args.output}")
    
    # Print summary
    successful = sum(1 for p in predictions if not p.get("exception", False))
    failed = len(predictions) - successful
    print(f"  Successful: {successful}")
    if failed > 0:
        print(f"  Failed: {failed}")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()

