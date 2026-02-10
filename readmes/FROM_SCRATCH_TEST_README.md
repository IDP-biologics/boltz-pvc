# Boltz From-Scratch Test (No Lightning, No CUDA)

This test script demonstrates running Boltz1 model **from scratch** without:
- ❌ PyTorch Lightning
- ❌ CUDA dependencies (cuEquivariance kernels)
- ❌ Checkpoint loading
- ✅ Designed for Intel XPUs and CPU execution

## Quick Start

```bash
python test_boltz_from_scratch.py
```

## What This Script Does

1. **Initializes Boltz1 from scratch** - No checkpoint loading required
   - Creates minimal configuration with default parameters
   - Initializes all model components (embedder, MSA, pairformer, diffusion, confidence)
   - Sets `use_kernels=False` for maximum compatibility

2. **Creates minimal input features** - Simple protein sequence
   - Test sequence: `MADQLTEEQIAEFKEAFSLF` (20 amino acids)
   - Generates synthetic token, atom, and MSA features
   - No real MSA or structure data needed

3. **Runs forward pass** - Vanilla PyTorch inference
   - No Lightning Trainer
   - No CUDA kernels
   - Minimal recycling and sampling steps for speed

## Key Features

### No Checkpoint Loading
Unlike the other inference scripts, this one initializes the model with **random weights**. This is useful for:
- Testing model architecture
- Verifying compatibility with Intel XPUs
- Development and debugging
- Understanding model initialization

### Intel XPU Compatible
- `use_kernels=False` - No cuEquivariance CUDA kernels
- CPU execution by default
- No CUDA-specific operations
- Should work on Aurora Intel XPUs (untested)

### Minimal Dependencies
Only requires:
- PyTorch
- Boltz model code
- No Lightning, no CUDA toolkit

## Configuration

The script uses default Boltz1 configuration from training configs:

```python
# Model dimensions
atom_s = 128
atom_z = 16
token_s = 384
token_z = 128
num_bins = 64

# Architecture
pairformer_blocks = 48
msa_blocks = 4
token_transformer_depth = 24

# Inference settings
use_kernels = False  # No CUDA
recycling_steps = 0  # Fast test
sampling_steps = 10  # Minimal
```

## Customization

### Change the sequence
Edit the `test_sequence` variable in `main()`:

```python
test_sequence = "YOUR_SEQUENCE_HERE"
```

### Change device
Modify the `device` variable:

```python
device = "cpu"  # or "cuda", "xpu", etc.
```

### Adjust inference parameters
Modify the forward pass call:

```python
output = model(
    feats,
    recycling_steps=3,  # More recycling
    num_sampling_steps=200,  # More sampling
    diffusion_samples=5,  # More samples
)
```

## Expected Output

```
================================================================================
Boltz From-Scratch Test (No Lightning, No CUDA)
================================================================================

Device: cpu

[1/4] Creating model configuration...
  - Model dimensions: token_s=384, token_z=128
  - Use kernels: False (CUDA disabled)

[2/4] Initializing Boltz1 model from scratch...
  - Model initialized successfully
  - Total parameters: XXX,XXX,XXX

[3/4] Creating minimal features for sequence: MADQLTEEQIAEFKEAFSLF
  - Sequence length: 20
  - Features created: XX feature tensors

[4/4] Running forward pass...
  ✓ Forward pass completed successfully!
  - Output keys: [...]
    - coords: torch.Size([...])
    - ...

================================================================================
✓ Test completed successfully!
================================================================================
```

## Limitations

⚠️ **This script uses random weights** - The model is not trained, so outputs will be meaningless!

For real predictions, use:
- `example_simple_inference.py` - Loads trained checkpoint
- `test_vanilla_inference.py` - Full CLI with checkpoint loading

## Use Cases

This script is useful for:
- ✅ Testing model architecture
- ✅ Verifying Intel XPU compatibility
- ✅ Development and debugging
- ✅ Understanding model initialization
- ✅ Benchmarking without I/O overhead
- ❌ **NOT for real predictions** (use checkpoint-based scripts)

## Troubleshooting

### Out of Memory
Reduce sequence length or model size:
```python
test_sequence = "MADQLT"  # Shorter sequence
```

### Missing Dependencies
Install Boltz:
```bash
pip install -e .
```

### Import Errors
Make sure you're in the repository root:
```bash
cd /path/to/boltz-pvc
python test_boltz_from_scratch.py
```

## Next Steps

Once this test works, you can:
1. Test on Intel XPU hardware
2. Load real checkpoints with `src/boltz/inference/loader.py`
3. Process real protein sequences with the full pipeline
4. Integrate into your custom workflows

## Related Files

- `src/boltz/inference/loader.py` - Checkpoint loading (vanilla PyTorch)
- `src/boltz/inference/runner.py` - Inference runner (no Lightning)
- `example_simple_inference.py` - Simple inference with checkpoints
- `test_vanilla_inference.py` - Full CLI inference
- `VANILLA_INFERENCE_README.md` - Vanilla PyTorch inference guide
- `CUDA_DEPENDENCIES_ANALYSIS.md` - CUDA dependency analysis

