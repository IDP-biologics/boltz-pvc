#!/bin/bash
# Boltz Installation Script for Aurora (Intel XPU)
# This script installs Boltz with minimal dependencies, using Aurora's pre-installed frameworks

set -e  # Exit on error

echo "=========================================="
echo "Boltz Installation for Aurora"
echo "=========================================="
echo ""

# Step 1: Load Aurora's frameworks module
echo "[1/4] Loading Aurora frameworks module..."
module load frameworks || {
    echo "Warning: Could not load frameworks module. Adjust version as needed."
    echo "Available modules:"
    module avail frameworks
}
source /lus/flare/projects/FoundEpidem/avasan/envs/boltz_env/bin/activate
echo ""

# Step 2: Verify PyTorch and NumPy are available
echo "[2/4] Verifying pre-installed frameworks..."
python -c "import torch; print('  ✓ PyTorch:', torch.__version__)" || {
    echo "  ✗ PyTorch not found! Make sure frameworks module is loaded."
    exit 1
}
python -c "import numpy; print('  ✓ NumPy:', numpy.__version__)" || {
    echo "  ✗ NumPy not found! Make sure frameworks module is loaded."
    exit 1
}
echo ""

# Step 3: Install Boltz without dependencies
echo "[3/4] Installing Boltz (without framework dependencies)..."
#uv pip install -e . --no-deps
echo ""

# Step 4: Install Boltz-specific dependencies
echo "[4/4] Installing Boltz-specific packages..."
#pip install \
#    hydra-core==1.3.2 \
#    rdkit \
#    dm-tree \
#    requests \
#    types-requests \
#    einops \
#    einx \
#    mashumaro \
#    modelcif \
#    click \
#    pyyaml \
#    biopython \
#    gemmi \
#    chembl_structure_pipeline
echo ""

# Verification
echo "=========================================="
echo "Verifying Installation"
echo "=========================================="
echo ""

echo "Checking Boltz installation..."
python -c "import boltz; print('  ✓ Boltz imported successfully')" || {
    echo "  ✗ Boltz import failed!"
    exit 1
}

echo "Checking inference module..."
python -c "from boltz.inference import load_model; print('  ✓ Inference module OK')" || {
    echo "  ✗ Inference module import failed!"
    exit 1
}

echo "Checking frameworks (should use Aurora's modules)..."
python -c "import torch; print('  ✓ PyTorch:', torch.__version__, '(from', torch.__file__, ')')"
python -c "import numpy; print('  ✓ NumPy:', numpy.__version__, '(from', numpy.__file__, ')')"
echo ""

echo "=========================================="
echo "✓ Installation Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Test: python tests/test_boltz_from_scratch.py"
echo "  2. Download checkpoint: wget <checkpoint_url>"
echo "  3. Run inference with device='xpu'"
echo ""
echo "See AURORA_INSTALLATION.md for detailed usage instructions."

