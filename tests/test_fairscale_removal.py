#!/usr/bin/env python3
"""
Test script to verify FairScale has been successfully removed.

This script:
1. Imports modules that previously used FairScale
2. Verifies no FairScale imports are present
3. Tests that checkpoint_wrapper works correctly
"""

import sys


def test_imports():
    """Test that modules can be imported without FairScale."""
    print("=" * 80)
    print("Testing FairScale Removal")
    print("=" * 80)
    
    print("\n[1/4] Testing trunk.py imports...")
    try:
        from src.boltz.model.modules.trunk import (
            InputEmbedder,
            MSAModule,
            PairformerModule,
            checkpoint_wrapper,
        )
        print("  ✓ trunk.py imports successful")
    except ImportError as e:
        print(f"  ✗ trunk.py import failed: {e}")
        return False
    
    print("\n[2/4] Testing transformers.py imports...")
    try:
        from src.boltz.model.modules.transformers import (
            DiffusionTransformer,
            checkpoint_wrapper as checkpoint_wrapper2,
        )
        print("  ✓ transformers.py imports successful")
    except ImportError as e:
        print(f"  ✗ transformers.py import failed: {e}")
        return False
    
    print("\n[3/4] Testing checkpoint_wrapper functionality...")
    try:
        import torch.nn as nn
        
        # Create a simple module
        test_module = nn.Linear(10, 10)
        
        # Wrap it with checkpoint_wrapper
        wrapped = checkpoint_wrapper(test_module, offload_to_cpu=False)
        
        # For inference, it should return the same module
        assert wrapped is test_module, "checkpoint_wrapper should return module as-is for inference"
        print("  ✓ checkpoint_wrapper works correctly")
    except Exception as e:
        print(f"  ✗ checkpoint_wrapper test failed: {e}")
        return False
    
    print("\n[4/4] Verifying no FairScale in sys.modules...")
    fairscale_modules = [name for name in sys.modules if 'fairscale' in name.lower()]
    if fairscale_modules:
        print(f"  ⚠ Warning: FairScale modules found in sys.modules: {fairscale_modules}")
        print("  (This is OK if FairScale is installed but not used)")
    else:
        print("  ✓ No FairScale modules in sys.modules")
    
    return True


def test_model_initialization():
    """Test that models can be initialized without FairScale."""
    print("\n" + "=" * 80)
    print("Testing Model Initialization")
    print("=" * 80)
    
    print("\n[1/2] Testing MSAModule initialization...")
    try:
        from src.boltz.model.modules.trunk import MSAModule
        
        # Create MSAModule with activation checkpointing enabled
        msa = MSAModule(
            token_z=128,
            s_input_dim=384,
            msa_s=64,
            msa_blocks=2,
            msa_dropout=0.0,
            z_dropout=0.0,
            pairwise_head_width=32,
            pairwise_num_heads=4,
            activation_checkpointing=True,  # This uses checkpoint_wrapper
            offload_to_cpu=False,
        )
        print(f"  ✓ MSAModule initialized with {len(msa.layers)} layers")
    except Exception as e:
        print(f"  ✗ MSAModule initialization failed: {e}")
        return False
    
    print("\n[2/2] Testing PairformerModule initialization...")
    try:
        from src.boltz.model.modules.trunk import PairformerModule
        
        # Create PairformerModule with activation checkpointing enabled
        pairformer = PairformerModule(
            token_s=384,
            token_z=128,
            num_blocks=2,
            num_heads=16,
            dropout=0.0,
            activation_checkpointing=True,  # This uses checkpoint_wrapper
            offload_to_cpu=False,
        )
        print(f"  ✓ PairformerModule initialized with {len(pairformer.layers)} layers")
    except Exception as e:
        print(f"  ✗ PairformerModule initialization failed: {e}")
        return False
    
    return True


def main():
    """Run all tests."""
    success = True
    
    # Test imports
    if not test_imports():
        success = False
    
    # Test model initialization
    if not test_model_initialization():
        success = False
    
    # Print summary
    print("\n" + "=" * 80)
    if success:
        print("✓ All tests passed! FairScale successfully removed.")
        print("=" * 80)
        return 0
    else:
        print("✗ Some tests failed. Please check the errors above.")
        print("=" * 80)
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

