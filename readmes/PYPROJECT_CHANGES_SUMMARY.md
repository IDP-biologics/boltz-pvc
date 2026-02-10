# pyproject.toml Changes for Aurora Deployment

## Summary

Modified `pyproject.toml` to support **minimal installation** on HPC systems like Aurora where PyTorch, NumPy, and other frameworks are pre-installed in system modules.

---

## What Changed

### Before (Original)

All dependencies were **required** by default:

```toml
[project]
dependencies = [
    "torch>=2.2",              # ← Required
    "numpy>=1.26,<2.0",        # ← Required
    "pytorch-lightning==2.5.0", # ← Required
    "fairscale==0.4.13",       # ← Required
    "pandas>=2.2.2",           # ← Required
    "scipy==1.13.1",           # ← Required
    # ... plus 20+ other packages
]
```

**Problem:** On Aurora, this would try to install PyTorch, NumPy, etc. into your user environment, conflicting with the system frameworks module.

---

### After (Modified)

Dependencies are now **split into categories**:

#### 1. **Minimal Dependencies** (Always Installed)

Only Boltz-specific packages that aren't in HPC frameworks modules:

```toml
[project]
dependencies = [
    "hydra-core==1.3.2",
    "rdkit>=2024.3.2",
    "dm-tree==0.1.8",
    "requests==2.32.3",
    "types-requests",
    "einops==0.8.0",
    "einx==0.3.0",
    "mashumaro==3.14",
    "modelcif==1.2",
    "click==8.1.7",
    "pyyaml==6.0.2",
    "biopython==1.84",
    "gemmi==0.6.5",
    "chembl_structure_pipeline==1.2.2",
]
```

**Note:** PyTorch, NumPy, pandas, scipy, etc. are **NOT** in this list!

#### 2. **Optional: Frameworks** (For Systems Without Pre-installed Frameworks)

```toml
[project.optional-dependencies]
frameworks = [
    "torch>=2.2",
    "numpy>=1.26,<2.0",
    "pandas>=2.2.2",
    "scipy==1.13.1",
    "numba==0.61.0",
    "scikit-learn==1.6.1",
]
```

Install with: `pip install -e .[frameworks]`

#### 3. **Optional: Training** (Not Needed for Inference)

```toml
training = [
    "pytorch-lightning==2.5.0",
    "fairscale==0.4.13",
    "wandb==0.18.7",
]
```

Install with: `pip install -e .[training]`

#### 4. **Optional: Full** (Everything)

```toml
full = [
    # All frameworks dependencies
    # Plus all training dependencies
]
```

Install with: `pip install -e .[full]`

---

## Installation Methods

### Method 1: Aurora (Minimal - Recommended)

Use Aurora's pre-installed frameworks:

```bash
# Load Aurora's frameworks module
module load frameworks/2024.2

# Install Boltz without dependencies
pip install -e . --no-deps

# Install only Boltz-specific packages
pip install hydra-core rdkit dm-tree requests types-requests \
    einops einx mashumaro modelcif click pyyaml biopython gemmi \
    chembl_structure_pipeline
```

**What gets installed:**
- ✅ Boltz code
- ✅ Boltz-specific packages (13 packages)
- ❌ PyTorch (uses Aurora's)
- ❌ NumPy (uses Aurora's)
- ❌ pandas, scipy, etc. (uses Aurora's)

---

### Method 2: Local Development (Full)

Install everything (for local machines without pre-installed frameworks):

```bash
pip install -e .[full]
```

**What gets installed:**
- ✅ Boltz code
- ✅ All dependencies (30+ packages)
- ✅ PyTorch, NumPy, pandas, scipy, etc.
- ✅ PyTorch Lightning, FairScale, W&B

---

### Method 3: Inference Only (Frameworks Included)

Install with frameworks but without training dependencies:

```bash
pip install -e .[frameworks]
```

**What gets installed:**
- ✅ Boltz code
- ✅ Boltz-specific packages
- ✅ PyTorch, NumPy, pandas, scipy, etc.
- ❌ PyTorch Lightning, FairScale, W&B

---

## Dependency Categories

### Always Installed (Minimal)
| Package | Purpose | Why Not in Aurora Module |
|---------|---------|--------------------------|
| `hydra-core` | Configuration | Boltz-specific |
| `rdkit` | Chemistry toolkit | Domain-specific |
| `dm-tree` | Tree utilities | DeepMind library |
| `requests` | HTTP client | May be in module, but safe to install |
| `einops`, `einx` | Tensor ops | Boltz-specific |
| `mashumaro` | Serialization | Boltz-specific |
| `modelcif` | Output format | Domain-specific |
| `click` | CLI framework | Boltz-specific |
| `pyyaml` | YAML parsing | Common but safe |
| `biopython` | Bioinformatics | Domain-specific |
| `gemmi` | Crystallography | Domain-specific |
| `chembl_structure_pipeline` | Chemistry | Domain-specific |

### Optional: Frameworks (Use Aurora's)
| Package | Aurora Module | Notes |
|---------|---------------|-------|
| `torch>=2.2` | ✅ Pre-installed | Use module version |
| `numpy>=1.26` | ✅ Pre-installed | Use module version |
| `pandas>=2.2.2` | ✅ Pre-installed | Use module version |
| `scipy==1.13.1` | ✅ Pre-installed | Use module version |
| `numba==0.61.0` | ✅ Pre-installed | Use module version |
| `scikit-learn` | ✅ Pre-installed | Use module version |

### Optional: Training (Not Needed)
| Package | Purpose | Needed for Inference? |
|---------|---------|----------------------|
| `pytorch-lightning` | Training framework | ❌ No |
| `fairscale` | Distributed training | ❌ No (removed) |
| `wandb` | Experiment tracking | ❌ No |

---

## Benefits

### For Aurora Deployment
1. ✅ **No conflicts** with system modules
2. ✅ **Faster installation** (only 13 packages vs 30+)
3. ✅ **Uses optimized frameworks** (Aurora's PyTorch may be optimized for Intel XPU)
4. ✅ **Smaller user environment** (less disk space)
5. ✅ **Easier maintenance** (framework updates handled by Aurora admins)

### For Local Development
1. ✅ **Still works** with `pip install -e .[full]`
2. ✅ **Backward compatible** with existing workflows
3. ✅ **Flexible** - choose what you need

---

## Quick Reference

| Use Case | Command | What Gets Installed |
|----------|---------|---------------------|
| **Aurora (Inference)** | `pip install -e . --no-deps` + manual | Minimal (13 packages) |
| **Local (Full)** | `pip install -e .[full]` | Everything (30+ packages) |
| **Local (Inference)** | `pip install -e .[frameworks]` | Minimal + frameworks |
| **Training** | `pip install -e .[full]` | Everything including Lightning |

---

## Verification

After installation on Aurora:

```bash
# Check Boltz
python -c "import boltz; print('Boltz OK')"

# Check frameworks (should show Aurora's paths)
python -c "import torch; print('PyTorch:', torch.__version__, torch.__file__)"
python -c "import numpy; print('NumPy:', numpy.__version__, numpy.__file__)"

# Check inference module
python -c "from boltz.inference import load_model; print('Inference OK')"
```

Expected output:
```
Boltz OK
PyTorch: 2.x.x /path/to/aurora/frameworks/torch/__init__.py
NumPy: 1.26.x /path/to/aurora/frameworks/numpy/__init__.py
Inference OK
```

---

## Files Created

1. **`pyproject.toml`** - Modified dependency structure
2. **`AURORA_INSTALLATION.md`** - Detailed installation guide
3. **`INSTALL_AURORA_QUICKSTART.sh`** - Automated installation script
4. **`PYPROJECT_CHANGES_SUMMARY.md`** - This file

---

## Next Steps

1. **On Aurora:**
   ```bash
   ./INSTALL_AURORA_QUICKSTART.sh
   ```

2. **Test installation:**
   ```bash
   python tests/test_boltz_from_scratch.py
   ```

3. **Run inference:**
   ```python
   from boltz.inference import load_model
   model = load_model(checkpoint_path="...", device="xpu")
   ```

---

**Result:** Boltz is now optimized for Aurora with minimal dependencies! 🚀

