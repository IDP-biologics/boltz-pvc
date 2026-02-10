"""
Example of integrating Boltz vanilla PyTorch inference into a custom pipeline.

This demonstrates how to:
1. Load a model once and reuse it
2. Process multiple inputs
3. Extract and analyze specific predictions
4. Save results in custom formats
"""

import json
import torch
from pathlib import Path
from typing import List, Dict, Any

# Intel Extension for PyTorch (for XPU support)
try:
    import intel_extension_for_pytorch as ipex
    IPEX_AVAILABLE = True
except ImportError:
    IPEX_AVAILABLE = False

from boltz.inference import load_model, BoltzInferenceRunner
from torch.utils.data import DataLoader


class CustomBoltzPipeline:
    """Custom pipeline for Boltz inference with post-processing."""
    
    def __init__(
        self,
        checkpoint_path: str,
        model_type: str = "boltz1",
        device: str = None,
    ):
        """
        Initialize the pipeline.

        Args:
            checkpoint_path: Path to model checkpoint
            model_type: "boltz1" or "boltz2"
            device: Device to use (auto-detected if None)
        """
        # Auto-detect device if not specified
        if device is None:
            if IPEX_AVAILABLE and hasattr(torch, 'xpu') and torch.xpu.is_available():
                self.device = "xpu"
                print(f"Using Intel XPU with IPEX version {ipex.__version__}")
            elif torch.cuda.is_available():
                self.device = "cuda"
                print("Using CUDA GPU")
            else:
                self.device = "cpu"
                print("Using CPU")
        else:
            self.device = device
        
        # Load model once
        print(f"Loading {model_type} model...")
        self.model = load_model(
            checkpoint_path=checkpoint_path,
            model_class=model_type,
            device=self.device,
            use_ema=True,
        )
        
        # Create runner
        self.runner = BoltzInferenceRunner(
            model=self.model,
            device=self.device,
        )
        
        print(f"Pipeline ready on {self.device}")
    
    def predict_batch(
        self,
        dataloader: DataLoader,
        recycling_steps: int = 3,
        sampling_steps: int = 200,
    ) -> List[Dict[str, Any]]:
        """
        Run predictions on a batch of inputs.
        
        Args:
            dataloader: DataLoader with input data
            recycling_steps: Number of recycling steps
            sampling_steps: Number of sampling steps
            
        Returns:
            List of predictions
        """
        predictions = self.runner.predict(
            dataloader=dataloader,
            recycling_steps=recycling_steps,
            sampling_steps=sampling_steps,
            diffusion_samples=1,
        )
        
        return predictions
    
    def extract_confidence_metrics(
        self,
        predictions: List[Dict[str, Any]]
    ) -> List[Dict[str, float]]:
        """
        Extract confidence metrics from predictions.
        
        Args:
            predictions: List of prediction dictionaries
            
        Returns:
            List of confidence metric dictionaries
        """
        metrics = []
        
        for i, pred in enumerate(predictions):
            if pred.get("exception", False):
                metrics.append({"sample_id": i, "status": "failed"})
                continue
            
            metric_dict = {
                "sample_id": i,
                "status": "success",
            }
            
            # Extract scalar metrics
            if "confidence_score" in pred:
                metric_dict["confidence_score"] = pred["confidence_score"].item()
            
            if "ptm" in pred:
                metric_dict["ptm"] = pred["ptm"].item()
            
            if "iptm" in pred:
                metric_dict["iptm"] = pred["iptm"].item()
            
            # Extract mean pLDDT
            if "plddt" in pred:
                metric_dict["mean_plddt"] = pred["plddt"].mean().item()
                metric_dict["min_plddt"] = pred["plddt"].min().item()
                metric_dict["max_plddt"] = pred["plddt"].max().item()
            
            # Extract complex metrics
            if "complex_plddt" in pred:
                metric_dict["complex_plddt"] = pred["complex_plddt"].item()
            
            if "complex_iplddt" in pred:
                metric_dict["complex_iplddt"] = pred["complex_iplddt"].item()
            
            metrics.append(metric_dict)
        
        return metrics
    
    def save_results(
        self,
        predictions: List[Dict[str, Any]],
        output_dir: Path,
        save_coords: bool = True,
        save_metrics: bool = True,
    ):
        """
        Save predictions to disk.
        
        Args:
            predictions: List of predictions
            output_dir: Directory to save results
            save_coords: Whether to save coordinates
            save_metrics: Whether to save metrics JSON
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Extract and save metrics
        if save_metrics:
            metrics = self.extract_confidence_metrics(predictions)
            metrics_path = output_dir / "metrics.json"
            with open(metrics_path, "w") as f:
                json.dump(metrics, f, indent=2)
            print(f"Saved metrics to {metrics_path}")
        
        # Save coordinates
        if save_coords:
            coords_dir = output_dir / "coordinates"
            coords_dir.mkdir(exist_ok=True)
            
            for i, pred in enumerate(predictions):
                if pred.get("exception", False):
                    continue
                
                coords_path = coords_dir / f"sample_{i:04d}.pt"
                torch.save({
                    "coords": pred["coords"],
                    "masks": pred["masks"],
                }, coords_path)
            
            print(f"Saved coordinates to {coords_dir}")


def main():
    """Example usage of the custom pipeline."""
    
    # Configuration
    checkpoint_path = "path/to/checkpoint.ckpt"
    input_data = "path/to/input.yaml"
    output_dir = "./custom_pipeline_output"
    
    # Initialize pipeline
    pipeline = CustomBoltzPipeline(
        checkpoint_path=checkpoint_path,
        model_type="boltz1",
        device="cuda",
    )
    
    # Prepare your dataloader (example - adjust to your needs)
    # from boltz.data.module.inference import PredictionDataset
    # dataset = PredictionDataset(...)
    # dataloader = DataLoader(dataset, batch_size=1)
    
    # For this example, assume we have a dataloader
    # dataloader = ...
    
    # Run predictions
    print("\nRunning predictions...")
    # predictions = pipeline.predict_batch(
    #     dataloader=dataloader,
    #     recycling_steps=3,
    #     sampling_steps=200,
    # )
    
    # Extract metrics
    # metrics = pipeline.extract_confidence_metrics(predictions)
    
    # Print summary
    # print("\nPrediction Summary:")
    # for metric in metrics:
    #     if metric["status"] == "success":
    #         print(f"  Sample {metric['sample_id']}: "
    #               f"Confidence={metric.get('confidence_score', 'N/A'):.3f}, "
    #               f"pLDDT={metric.get('mean_plddt', 'N/A'):.3f}")
    
    # Save results
    # pipeline.save_results(
    #     predictions=predictions,
    #     output_dir=output_dir,
    #     save_coords=True,
    #     save_metrics=True,
    # )
    
    print("\nPipeline example complete!")
    print("Uncomment the code above and provide your dataloader to run.")


if __name__ == "__main__":
    main()

