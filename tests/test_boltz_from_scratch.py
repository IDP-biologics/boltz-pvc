#!/usr/bin/env python3
"""
Test script to run Boltz model from scratch without Lightning or CUDA dependencies.
Designed for Intel XPUs and CPU execution.

This script:
1. Initializes a Boltz1 model from scratch (no checkpoint loading)
2. Creates minimal input features for a simple protein sequence
3. Runs a forward pass without Lightning
4. No CUDA kernels (use_kernels=False)
"""

import torch
import torch.nn as nn
from dataclasses import asdict

# Intel Extension for PyTorch (for XPU support)
try:
    import intel_extension_for_pytorch as ipex
    IPEX_AVAILABLE = True
except ImportError:
    IPEX_AVAILABLE = False

# Import Boltz model and configuration
from src.boltz.model.models.boltz1 import Boltz1
from src.boltz.main import (
    PairformerArgs,
    MSAModuleArgs,
    BoltzDiffusionParams,
)


def create_minimal_config():
    """Create minimal configuration for Boltz1 model."""
    
    # Model dimensions (from training configs)
    atom_s = 128
    atom_z = 16
    token_s = 384
    token_z = 128
    num_bins = 64
    
    # Embedder args
    embedder_args = {
        "atom_encoder_depth": 3,
        "atom_encoder_heads": 4,
    }
    
    # MSA args
    msa_args = asdict(MSAModuleArgs(
        msa_s=64,
        msa_blocks=4,
        msa_dropout=0.0,
        z_dropout=0.0,
        pairwise_head_width=32,
        pairwise_num_heads=4,
        activation_checkpointing=False,
        offload_to_cpu=False,
        subsample_msa=False,
        num_subsampled_msa=1024,
    ))
    
    # Pairformer args
    pairformer_args = asdict(PairformerArgs(
        num_blocks=48,
        num_heads=16,
        dropout=0.0,
        activation_checkpointing=False,
        offload_to_cpu=False,
    ))
    
    # Score model args
    score_model_args = {
        "sigma_data": 16,
        "dim_fourier": 256,
        "atom_encoder_depth": 3,
        "atom_encoder_heads": 4,
        "token_transformer_depth": 24,
        "token_transformer_heads": 16,
        "atom_decoder_depth": 3,
        "atom_decoder_heads": 4,
        "conditioning_transition_layers": 2,
        "activation_checkpointing": False,
        "offload_to_cpu": False,
    }
    
    # Diffusion process args
    diffusion_params = BoltzDiffusionParams()
    diffusion_process_args = {
        "sigma_min": diffusion_params.sigma_min,
        "sigma_max": diffusion_params.sigma_max,
        "sigma_data": diffusion_params.sigma_data,
        "rho": diffusion_params.rho,
        "P_mean": diffusion_params.P_mean,
        "P_std": diffusion_params.P_std,
        "gamma_0": diffusion_params.gamma_0,
        "gamma_min": diffusion_params.gamma_min,
        "noise_scale": diffusion_params.noise_scale,
        "step_scale": diffusion_params.step_scale,
        "coordinate_augmentation": False,  # Disable for inference
        "alignment_reverse_diff": diffusion_params.alignment_reverse_diff,
        "synchronize_sigmas": diffusion_params.synchronize_sigmas,
    }
    
    # Diffusion loss args (not used in inference but required for init)
    diffusion_loss_args = {
        "diffusion_loss_weight": 4.0,
        "distogram_loss_weight": 0.03,
    }
    
    # Confidence model args
    confidence_model_args = {
        "num_dist_bins": 64,
        "max_dist": 22,
        "add_s_to_z_prod": True,
        "add_s_input_to_s": True,
        "use_s_diffusion": True,
        "add_z_input_to_z": True,
        "confidence_args": {
            "num_plddt_bins": 50,
            "num_pde_bins": 64,
            "num_pae_bins": 64,
        },
    }
    
    # Training args (not used in inference but required for init)
    training_args = type('TrainingArgs', (), {
        "recycling_steps": 3,
        "sampling_steps": 20,
        "diffusion_multiplicity": 16,
        "diffusion_samples": 2,
        "confidence_loss_weight": 1e-4,
        "diffusion_loss_weight": 4.0,
        "distogram_loss_weight": 0.03,
        "adam_beta_1": 0.9,
        "adam_beta_2": 0.95,
        "adam_eps": 1e-8,
        "lr_scheduler": "af3",
        "base_lr": 0.0,
        "max_lr": 0.0018,
        "lr_warmup_no_steps": 1000,
        "lr_start_decay_after_n_steps": 50000,
        "lr_decay_every_n_steps": 50000,
        "lr_decay_factor": 0.95,
    })()
    
    # Validation args
    validation_args = type('ValidationArgs', (), {
        "recycling_steps": 3,
        "sampling_steps": 200,
        "diffusion_samples": 5,
        "symmetry_correction": True,
        "run_confidence_sequentially": False,
    })()
    
    return {
        "atom_s": atom_s,
        "atom_z": atom_z,
        "token_s": token_s,
        "token_z": token_z,
        "num_bins": num_bins,
        "training_args": training_args,
        "validation_args": validation_args,
        "embedder_args": embedder_args,
        "msa_args": msa_args,
        "pairformer_args": pairformer_args,
        "score_model_args": score_model_args,
        "diffusion_process_args": diffusion_process_args,
        "diffusion_loss_args": diffusion_loss_args,
        "confidence_model_args": confidence_model_args,
        "atom_feature_dim": 128,
        "confidence_prediction": False,  # Disable for simple test
        "use_kernels": False,  # NO CUDA kernels for Intel XPU compatibility
        "ema": False,  # No EMA for simple test
    }


def create_minimal_features(sequence="MADQLTEEQIAEFKEAFSLF", device="xpu"):
    """
    Create minimal input features for a simple protein sequence.

    Args:
        sequence: Protein sequence (single letter amino acid codes)
        device: Device to create tensors on

    Returns:
        Dictionary of minimal features required for model forward pass
    """
    seq_len = len(sequence)
    batch_size = 1

    # Minimal feature dictionary
    # Note: This is a simplified version - real features would come from featurizer
    feats = {
        # Token features
        "token_single_mask": torch.ones(batch_size, seq_len, dtype=torch.bool, device=device),
        "token_pair_mask": torch.ones(batch_size, seq_len, seq_len, dtype=torch.bool, device=device),
        "token_single_input_feat": torch.randn(batch_size, seq_len, 384, device=device),  # token_s
        "token_pair_input_feat": torch.randn(batch_size, seq_len, seq_len, 128, device=device),  # token_z
        "token_bonds": torch.zeros(batch_size, seq_len, seq_len, 1, device=device),
        "token_residue_type": torch.randint(0, 20, (batch_size, seq_len), device=device),
        "token_asym_id": torch.zeros(batch_size, seq_len, dtype=torch.long, device=device),
        "token_entity_id": torch.zeros(batch_size, seq_len, dtype=torch.long, device=device),
        "token_sym_id": torch.zeros(batch_size, seq_len, dtype=torch.long, device=device),
        "token_entity_type": torch.zeros(batch_size, seq_len, dtype=torch.long, device=device),
        "token_residue_index": torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1),

        # Atom features (simplified - assume ~10 atoms per residue)
        "atom_single_mask": torch.ones(batch_size, seq_len * 10, dtype=torch.bool, device=device),
        "atom_single_input_feat": torch.randn(batch_size, seq_len * 10, 128, device=device),  # atom_s
        "atom_ref_pos": torch.randn(batch_size, seq_len * 10, 3, device=device),
        "atom_ref_mask": torch.ones(batch_size, seq_len * 10, dtype=torch.bool, device=device),
        "atom_ref_space_uid": torch.zeros(batch_size, seq_len * 10, dtype=torch.long, device=device),
        "atom_to_token": torch.arange(seq_len, device=device).repeat_interleave(10).unsqueeze(0).expand(batch_size, -1),

        # MSA features (minimal - just query sequence)
        "msa_feat": torch.randn(batch_size, 1, seq_len, 64, device=device),  # msa_s
        "msa_mask": torch.ones(batch_size, 1, seq_len, dtype=torch.bool, device=device),

        # Pocket/contact features (empty for simple test)
        "pocket_contact_info": torch.zeros(batch_size, seq_len, 4, device=device),
    }

    return feats


def main():
    """Main test function."""
    print("=" * 80)
    print("Boltz From-Scratch Test (No Lightning, No CUDA)")
    print("=" * 80)

    # Set device (CPU for Intel XPU compatibility testing)
    device = "xpu"
    print(f"\nDevice: {device}")
    if device == "xpu":
        if IPEX_AVAILABLE:
            print(f"  - Intel Extension for PyTorch: Available (version {ipex.__version__})")
        else:
            print("  - WARNING: Intel Extension for PyTorch not available!")
            print("  - Install with: pip install intel-extension-for-pytorch")

    # Create model configuration
    print("\n[1/4] Creating model configuration...")
    config = create_minimal_config()
    print(f"  - Model dimensions: token_s={config['token_s']}, token_z={config['token_z']}")
    print(f"  - Use kernels: {config['use_kernels']} (CUDA disabled)")

    # Initialize model
    print("\n[2/4] Initializing Boltz1 model from scratch...")
    model = Boltz1(**config)
    model = model.to(device)
    model.eval()  # Set to evaluation mode
    print(f"  - Model initialized successfully")
    print(f"  - Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create minimal input features
    test_sequence = "MADQLTEEQIAEFKEAFSLF"  # 20 amino acids
    print(f"\n[3/4] Creating minimal features for sequence: {test_sequence}")
    print(f"  - Sequence length: {len(test_sequence)}")
    feats = create_minimal_features(test_sequence, device=device)
    print(f"  - Features created: {len(feats)} feature tensors")

    # Run forward pass
    print("\n[4/4] Running forward pass...")
    with torch.no_grad():
        try:
            output = model(
                feats,
                recycling_steps=0,  # No recycling for simple test
                num_sampling_steps=10,  # Minimal sampling steps
                diffusion_samples=1,
            )
            print("  ✓ Forward pass completed successfully!")
            print(f"  - Output keys: {list(output.keys())}")

            # Print output shapes
            for key, value in output.items():
                if isinstance(value, torch.Tensor):
                    print(f"    - {key}: {value.shape}")

        except Exception as e:
            print(f"  ✗ Forward pass failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    print("\n" + "=" * 80)
    print("✓ Test completed successfully!")
    print("=" * 80)
    return True


if __name__ == "__main__":
    success = main()

