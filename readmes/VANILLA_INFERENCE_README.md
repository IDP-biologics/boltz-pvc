# Boltz Vanilla PyTorch Inference

This guide explains how to run Boltz inference using vanilla PyTorch without PyTorch Lightning or the command-line interface.

## Overview

The vanilla PyTorch inference implementation provides:
- **No Lightning dependencies** for inference
- **Simple Python API** for programmatic use
- **EMA weight support** for better predictions
- **Memory-efficient** batch processing
- **Easy integration** into custom pipelines

## Quick Start

### 1. Basic Usage

```python
from boltz.inference import load_model, BoltzInferenceRunner
from torch.utils.data import DataLoader

# Load model
model = load_model(
    checkpoint_path="path/to/checkpoint.ckpt",
    model_class="boltz1",  # or "boltz2"
    device="cuda",
    use_ema=True,
)

# Create your dataloader (see Data Preparation below)
dataloader = DataLoader(dataset, batch_size=1)

# Run inference
runner = BoltzInferenceRunner(model=model, device="cuda")
predictions = runner.predict(
    dataloader=dataloader,
    recycling_steps=3,
    sampling_steps=200,
    diffusion_samples=1,
)
```

### 2. Using the Example Scripts

We provide two example scripts:

#### Simple Example (`example_simple_inference.py`)
A minimal, well-commented example for quick testing:

```bash
python example_simple_inference.py
```

Edit the configuration section in the script to match your setup.

#### Full CLI Example (`test_vanilla_inference.py`)
A complete command-line interface with all options:

```bash
python test_vanilla_inference.py \
    --checkpoint path/to/checkpoint.ckpt \
    --input path/to/input.yaml \
    --output ./predictions \
    --model boltz1 \
    --device cuda \
    --recycling-steps 3 \
    --sampling-steps 200 \
    --diffusion-samples 1
```

## API Reference

### `load_model()`

Load a Boltz model from a checkpoint.

**Parameters:**
- `checkpoint_path` (str|Path): Path to checkpoint file
- `model_class` (str): "boltz1" or "boltz2"
- `device` (str): Device to load on ("cuda" or "cpu")
- `use_ema` (bool): Whether to use EMA weights (default: True)
- `use_kernels` (bool): Whether to use cuEquivariance CUDA kernels (default: False)
- `predict_args` (dict, optional): Prediction arguments
- `**model_kwargs`: Additional model arguments

**Returns:** Loaded model ready for inference

**Note on `use_kernels`:**
- `False` (default): Maximum compatibility, works on CPU and all GPUs
- `True`: Faster inference, requires cuEquivariance package and GPU with compute capability >= 8.0

### `BoltzInferenceRunner`

Runner for executing inference on batches.

**Methods:**

#### `__init__(model, device=None, output_dir=None)`
- `model`: The Boltz model
- `device`: Device to use (auto-detected if None)
- `output_dir`: Directory to save predictions (optional)

#### `predict(dataloader, **kwargs)`
Run inference on a dataloader.

**Parameters:**
- `dataloader`: PyTorch DataLoader
- `recycling_steps` (int): Number of recycling steps (default: 3)
- `sampling_steps` (int): Diffusion sampling steps (default: 200)
- `diffusion_samples` (int): Number of samples (default: 1)
- `write_confidence_summary` (bool): Include confidence scores (default: True)
- `write_full_pae` (bool): Include full PAE matrix (default: True)
- `write_full_pde` (bool): Include full PDE matrix (default: False)

**Returns:** List of prediction dictionaries

## Data Preparation

You need to prepare your data using the existing Boltz data modules. Here's how:

```python
from boltz.data.module.inference import PredictionDataset
from boltz.data.parse.manifest import Manifest
from pathlib import Path

# Create manifest from your input
manifest = Manifest.from_yaml("path/to/input.yaml")

# Create dataset
dataset = PredictionDataset(
    manifest=manifest,
    target_dir=Path("path/to/targets"),
    msa_dir=Path("path/to/msas"),
    constraints_dir=None,  # Optional
)

# Create dataloader
from torch.utils.data import DataLoader
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
```

## Prediction Output

Each prediction dictionary contains:

- `exception` (bool): Whether an error occurred
- `batch_idx` (int): Batch index
- `masks` (Tensor): Atom padding masks
- `coords` (Tensor): Predicted coordinates
- `s` (Tensor): Single representation
- `z` (Tensor): Pair representation
- `confidence_score` (Tensor): Overall confidence score
- `ptm` (Tensor): Predicted TM-score
- `iptm` (Tensor): Interface PTM
- `plddt` (Tensor): Per-residue confidence
- `pae` (Tensor): Predicted aligned error (optional)
- `pde` (Tensor): Predicted distance error (optional)

## Differences from Lightning Version

| Feature | Lightning | Vanilla PyTorch |
|---------|-----------|-----------------|
| Model base class | `LightningModule` | `nn.Module` (wrapped) |
| Checkpoint loading | `load_from_checkpoint()` | `load_model()` |
| Inference | `Trainer.predict()` | `runner.predict()` |
| EMA handling | Automatic callbacks | Manual in loader |
| Device placement | Automatic | Manual |
| Progress bars | Built-in | tqdm |

## Notes

- The original model classes still inherit from `LightningModule` for training compatibility
- The vanilla inference wraps these models and handles them as pure PyTorch
- EMA weights are automatically applied during loading if available
- All predictions are moved to CPU to save GPU memory

## Troubleshooting

**Out of Memory Errors:**
- Reduce `diffusion_samples`
- Reduce `batch_size` to 1
- Use CPU if GPU memory is insufficient

**Model Loading Errors:**
- Ensure checkpoint path is correct
- Check that model_class matches the checkpoint
- Verify checkpoint contains required keys

**Import Errors:**
- Ensure you're in the correct environment
- Install required dependencies: `torch`, `tqdm`

