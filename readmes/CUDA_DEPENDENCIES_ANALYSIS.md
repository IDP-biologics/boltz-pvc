# CUDA Dependencies Analysis for Boltz

## Summary

Boltz uses **NVIDIA cuEquivariance** kernels for GPU acceleration, but they are **OPTIONAL** and the codebase has **full fallback implementations** in vanilla PyTorch. The kernels are NOT required for inference.

## CUDA-Dependent Modules

### 1. **cuEquivariance (NVIDIA)**

**Package**: `cuequivariance_torch` (optional dependency)

**Purpose**: Optimized CUDA kernels for triangle attention and triangle multiplication operations

**Prevalence**: Used in 2 key operations:
- Triangle Multiplicative Update (incoming/outgoing)
- Triangle Attention

**Files**:
- `src/boltz/model/layers/triangular_mult.py` - Triangle multiplication kernels
- `src/boltz/model/layers/triangular_attention/primitives.py` - Triangle attention kernels

**Installation**:
```toml
# From pyproject.toml
[project.optional-dependencies]
cuda = [
    "cuequivariance_ops_cu12>=0.5.0",
    "cuequivariance_ops_torch_cu12>=0.5.0",
    "cuequivariance_torch>=0.5.0",
]
```

### 2. **Flash Attention / Triton**

**Status**: ❌ **NOT USED** in the main codebase

**Evidence**: 
- Only appears in `tests/test_kernels.py` for benchmarking
- Not used in production inference or training code
- No imports in model files

## How Kernel Usage is Controlled

### 1. **Global Flag: `use_kernels`**

The codebase uses a `use_kernels` boolean flag that propagates through the model:

```python
# In main.py (line 1321)
model_module = model_cls.load_from_checkpoint(
    checkpoint,
    use_kernels=not no_kernels,  # CLI flag
    ...
)
```

### 2. **Automatic Fallback**

Each layer that can use kernels has a fallback:

```python
# From triangular_mult.py
def forward(self, x: Tensor, mask: Tensor, use_kernels: bool = False) -> Tensor:
    if use_kernels:
        # Use cuEquivariance kernel
        return kernel_triangular_mult(...)
    
    # Vanilla PyTorch fallback
    x = self.norm_in(x)
    x = self.p_in(x) * self.g_in(x).sigmoid()
    # ... rest of vanilla implementation
```

### 3. **GPU Compute Capability Check**

Kernels are automatically disabled on older GPUs:

```python
# From boltz1.py and boltz2.py (setup method)
def setup(self, stage: str) -> None:
    if stage == "predict" and not (
        torch.cuda.is_available()
        and torch.cuda.get_device_properties(torch.device("cuda")).major >= 8.0
    ):
        self.use_kernels = False
```

**Requirement**: CUDA Compute Capability >= 8.0 (Ampere or newer)
- ✅ RTX 30xx series, A100, H100
- ❌ RTX 20xx series, V100, older GPUs

### 4. **Environment Variables**

```python
# From main.py (lines 1105-1108)
for key in ["CUEQ_DEFAULT_CONFIG", "CUEQ_DISABLE_AOT_TUNING"]:
    # Disable kernel tuning by default
    os.environ[key] = os.environ.get(key, "1")
```

## Where Kernels Are Used

### Triangle Multiplication (2 operations)

**Files**: `src/boltz/model/layers/triangular_mult.py`

**Classes**:
- `TriangleMultiplicationOutgoing`
- `TriangleMultiplicationIncoming`

**Kernel Function**: `cuequivariance_torch.primitives.triangle.triangle_multiplicative_update`

**Fallback**: Vanilla PyTorch einsum operations

### Triangle Attention (2 operations)

**Files**: `src/boltz/model/layers/triangular_attention/primitives.py`

**Function**: `kernel_triangular_attn()`

**Kernel Function**: `cuequivariance_torch.primitives.triangle.triangle_attention`

**Fallback**: Standard PyTorch attention implementation

## Propagation Through Model

The `use_kernels` flag flows through the model hierarchy:

```
Model (Boltz1/Boltz2)
  └─> use_kernels parameter
      └─> PairformerModule
          └─> PairformerLayer
              ├─> TriangleMultiplicationOutgoing(use_kernels)
              ├─> TriangleMultiplicationIncoming(use_kernels)
              ├─> TriangleAttentionStartingNode(use_kernels)
              └─> TriangleAttentionEndingNode(use_kernels)
```

## Running Without CUDA Kernels

### Option 1: CLI Flag
```bash
boltz predict input.yaml --no_kernels
```

### Option 2: Don't Install cuEquivariance
If the package isn't installed, kernels are automatically disabled (import will fail gracefully).

### Option 3: Vanilla PyTorch Inference (Our Implementation)
```python
from boltz.inference import load_model

model = load_model(
    checkpoint_path="checkpoint.ckpt",
    use_kernels=False,  # Explicitly disable
)
```

## Performance Impact

**With cuEquivariance kernels** (on supported GPUs):
- ~2-3x faster for triangle operations
- Lower memory usage
- Better throughput

**Without kernels** (vanilla PyTorch):
- Still fully functional
- Slower but works on any GPU/CPU
- Higher memory usage for large sequences

## CPU Inference

✅ **Fully supported** - kernels are automatically disabled on CPU

```python
model = load_model(
    checkpoint_path="checkpoint.ckpt",
    device="cpu",
)
```

## Summary Table

| Component | Required? | Fallback Available? | Notes |
|-----------|-----------|---------------------|-------|
| cuEquivariance | ❌ No | ✅ Yes | Optional acceleration |
| Flash Attention | ❌ No | N/A | Not used in codebase |
| Triton | ❌ No | N/A | Only in tests |
| CUDA | ❌ No | ✅ Yes | CPU inference works |
| PyTorch | ✅ Yes | - | Core requirement |

## Recommendations for Vanilla PyTorch Inference

1. **Default**: Set `use_kernels=False` in the loader
2. **Compatibility**: Works on any hardware (CPU, old GPUs, new GPUs)
3. **Simplicity**: No need to install cuEquivariance packages
4. **Portability**: Easier deployment across different environments

## Updated Loader

I'll update the vanilla PyTorch loader to default to `use_kernels=False` for maximum compatibility.

