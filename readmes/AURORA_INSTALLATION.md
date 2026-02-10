# Boltz Installation for Aurora (Intel XPU)

## Overview

This guide explains how to install Boltz on Aurora's HPC environment where PyTorch, NumPy, JAX, and other frameworks are already available in the system modules.

## Prerequisites

Aurora should already have these packages in the frameworks module:
- ✅ `torch>=2.2`
- ✅ `numpy>=1.26`
- ✅ `pandas>=2.2.2`
- ✅ `scipy`
- ✅ `numba`
- ✅ `scikit-learn`
- ✅ JAX (not used by Boltz, but available)

## Installation Options

### Option 1: Minimal Installation (Recommended for Aurora)

Install only the packages **not** provided by Aurora's frameworks module:

```bash
# Load Aurora's frameworks module first
module load frameworks/2024.2  # or appropriate version

# Install Boltz with minimal dependencies
pip install -e . --no-deps
pip install hydra-core==1.3.2 rdkit dm-tree requests types-requests \
    einops einx mashumaro modelcif click pyyaml biopython gemmi \
    chembl_structure_pipeline
```

**What this does:**
- Uses `--no-deps` to prevent pip from installing torch, numpy, etc.
- Only installs Boltz-specific packages not in the frameworks module
- Relies on Aurora's pre-installed PyTorch, NumPy, etc.

---

### Option 2: Install with Framework Dependencies (If Needed)

If Aurora's frameworks module is missing some packages:

```bash
pip install -e .[frameworks]
```

**What this installs:**
- All minimal dependencies (from Option 1)
- Plus: torch, numpy, pandas, scipy, numba, scikit-learn

**Note:** This may conflict with Aurora's module system. Use Option 1 if possible.

---

### Option 3: Full Installation (Development/Training)

For development or training (includes PyTorch Lightning, FairScale, W&B):

```bash
pip install -e .[full]
```

**What this installs:**
- Everything from Option 2
- Plus: pytorch-lightning, fairscale, wandb

**Note:** Not needed for inference! Use Option 1 for inference-only deployment.

---

## Verification

After installation, verify Boltz can import correctly:

```bash
python -c "import boltz; print('Boltz version:', boltz.__version__)"
python -c "from boltz.inference import load_model; print('Inference module OK')"
```

Check that Aurora's frameworks are being used:

```bash
python -c "import torch; print('PyTorch:', torch.__version__, torch.__file__)"
python -c "import numpy; print('NumPy:', numpy.__version__, numpy.__file__)"
```

The file paths should point to Aurora's frameworks module, not your local pip installation.

---

## Running Inference on Aurora

### Quick Test (No Checkpoint)

```bash
# Test model architecture
python tests/test_boltz_from_scratch.py
```

### Real Inference (With Checkpoint)

```python
from boltz.inference import load_model

# Load model on Intel XPU
model = load_model(
    checkpoint_path="path/to/boltz1_conf.ckpt",
    device="xpu",  # Intel XPU
    use_kernels=False,  # No CUDA kernels
)

# Run inference
output = model(features)
```

---

## Dependency Summary

### Always Installed (Minimal)
These are **always** installed because they're Boltz-specific:
- `hydra-core` - Configuration management
- `rdkit` - Chemistry toolkit
- `dm-tree` - Tree utilities
- `requests` - HTTP library
- `einops`, `einx` - Tensor operations
- `mashumaro` - Serialization
- `modelcif` - Model output format
- `click` - CLI framework
- `pyyaml` - YAML parsing
- `biopython` - Bioinformatics tools
- `gemmi` - Crystallography library
- `chembl_structure_pipeline` - Chemistry pipeline

### Optional: Frameworks (Use Aurora's Module)
These should come from Aurora's frameworks module:
- `torch>=2.2`
- `numpy>=1.26,<2.0`
- `pandas>=2.2.2`
- `scipy==1.13.1`
- `numba==0.61.0`
- `scikit-learn==1.6.1`

### Optional: Training (Not Needed for Inference)
Only install if you need to train models:
- `pytorch-lightning==2.5.0`
- `fairscale==0.4.13`
- `wandb==0.18.7`

### Optional: CUDA (Not Needed for Intel XPU)
Only for NVIDIA GPUs:
- `cuequivariance_ops_cu12>=0.5.0`
- `cuequivariance_ops_torch_cu12>=0.5.0`
- `cuequivariance_torch>=0.5.0`

---

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'torch'"

**Solution:** Load Aurora's frameworks module:
```bash
module load frameworks/2024.2
```

### Issue: "ImportError: cannot import name 'load_model'"

**Solution:** Make sure you installed Boltz:
```bash
pip install -e . --no-deps
```

### Issue: Pip tries to install torch/numpy

**Solution:** Use `--no-deps` flag:
```bash
pip install -e . --no-deps
# Then manually install only what's needed
```

### Issue: Version conflicts

**Solution:** Check Aurora's versions and adjust if needed:
```bash
python -c "import torch; print(torch.__version__)"
python -c "import numpy; print(numpy.__version__)"
```

---

## Installation Commands Summary

### For Aurora (Recommended)
```bash
# 1. Load frameworks module
module load frameworks/2024.2

# 2. Install Boltz without dependencies
pip install -e . --no-deps

# 3. Install only Boltz-specific packages
pip install hydra-core==1.3.2 rdkit dm-tree requests types-requests \
    einops einx mashumaro modelcif click pyyaml biopython gemmi \
    chembl_structure_pipeline

# 4. Verify installation
python -c "import boltz; from boltz.inference import load_model; print('OK')"
```

### For Local Development (Full)
```bash
pip install -e .[full]
```

---

## Next Steps

1. **Test installation**: Run `python tests/test_boltz_from_scratch.py`
2. **Download checkpoint**: Get Boltz-1 or Boltz-2 checkpoint
3. **Run inference**: Use `boltz.inference.load_model()` with `device="xpu"`
4. **Check performance**: Benchmark on Intel XPU

---

## Additional Resources

- **Inference Guide**: `readmes/INTEL_XPU_INFERENCE_SUMMARY.md`
- **Dependency Removal**: `readmes/DEPENDENCY_REMOVAL_COMPLETE.md`
- **CUDA Analysis**: `readmes/CUDA_DEPENDENCIES_ANALYSIS.md`
- **FairScale Removal**: `readmes/FAIRSCALE_REMOVAL_SUMMARY.md`

---

**Result:** Boltz is now optimized for Aurora deployment with minimal dependencies! 🚀

