"""Vanilla PyTorch inference runner for Boltz models."""

import gc
from pathlib import Path
from typing import Any, Optional, Union

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Intel Extension for PyTorch (for XPU support)
try:
    import intel_extension_for_pytorch as ipex
    IPEX_AVAILABLE = True
except ImportError:
    IPEX_AVAILABLE = False


class BoltzInferenceRunner:
    """Vanilla PyTorch inference runner for Boltz models."""
    
    def __init__(
        self,
        model: nn.Module,
        device: Optional[str] = None,
        output_dir: Optional[Union[str, Path]] = None,
    ):
        """
        Initialize the inference runner.
        
        Args:
            model: The Boltz model to run inference with
            device: Device to run inference on (auto-detected if None)
            output_dir: Directory to save predictions (optional)
        """
        self.model = model
        
        # Auto-detect device if not provided
        if device is None:
            if IPEX_AVAILABLE and hasattr(torch, 'xpu') and torch.xpu.is_available():
                device = "xpu"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        self.device = device

        # Log device info
        if device == "xpu" and IPEX_AVAILABLE:
            print(f"Using Intel XPU with IPEX version {ipex.__version__}")
        elif device == "xpu" and not IPEX_AVAILABLE:
            print("WARNING: XPU device requested but Intel Extension for PyTorch not available!")

        # Move model to device
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Output directory
        self.output_dir = Path(output_dir) if output_dir else None
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def predict(
        self,
        dataloader: DataLoader,
        recycling_steps: int = 3,
        sampling_steps: int = 200,
        diffusion_samples: int = 1,
        write_confidence_summary: bool = True,
        write_full_pae: bool = True,
        write_full_pde: bool = False,
    ) -> list[dict[str, Any]]:
        """
        Run inference on a dataloader.
        
        Args:
            dataloader: DataLoader containing batches to predict
            recycling_steps: Number of recycling steps
            sampling_steps: Number of diffusion sampling steps
            diffusion_samples: Number of diffusion samples
            write_confidence_summary: Whether to include confidence scores
            write_full_pae: Whether to include full PAE matrix
            write_full_pde: Whether to include full PDE matrix
            
        Returns:
            List of prediction dictionaries
        """
        predictions = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc="Running inference")):
                try:
                    # Move batch to device
                    batch = self._transfer_batch_to_device(batch)
                    
                    # Run model
                    out = self.model(
                        batch,
                        recycling_steps=recycling_steps,
                        num_sampling_steps=sampling_steps,
                        diffusion_samples=diffusion_samples,
                        max_parallel_samples=diffusion_samples,
                        run_confidence_sequentially=True,
                    )
                    
                    # Prepare prediction dictionary
                    pred_dict = {"exception": False, "batch_idx": batch_idx}
                    pred_dict["masks"] = batch["atom_pad_mask"].cpu()
                    pred_dict["coords"] = out["sample_atom_coords"].cpu()
                    pred_dict["s"] = out["s"].cpu()
                    pred_dict["z"] = out["z"].cpu()
                    
                    if write_confidence_summary:
                        pred_dict["confidence_score"] = (
                            4 * out["complex_plddt"]
                            + (
                                out["iptm"]
                                if not torch.allclose(
                                    out["iptm"], torch.zeros_like(out["iptm"])
                                )
                                else out["ptm"]
                            )
                        ) / 5
                        
                        for key in [
                            "ptm",
                            "iptm",
                            "ligand_iptm",
                            "protein_iptm",
                            "pair_chains_iptm",
                            "complex_plddt",
                            "complex_iplddt",
                            "complex_pde",
                            "complex_ipde",
                            "plddt",
                        ]:
                            if key in out:
                                pred_dict[key] = out[key].cpu()
                    
                    if write_full_pae and "pae" in out:
                        pred_dict["pae"] = out["pae"].cpu()
                    
                    if write_full_pde and "pde" in out:
                        pred_dict["pde"] = out["pde"].cpu()
                    
                    predictions.append(pred_dict)
                    
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print(f"| WARNING: ran out of memory on batch {batch_idx}, skipping")
                        torch.cuda.empty_cache()
                        gc.collect()
                        predictions.append({"exception": True, "batch_idx": batch_idx})
                    else:
                        raise
        
        return predictions
    
    def _transfer_batch_to_device(self, batch: dict) -> dict:
        """Transfer batch to the target device."""
        device_batch = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                device_batch[key] = value.to(self.device)
            else:
                device_batch[key] = value
        return device_batch

