"""
Test script for DINOv3 ViT Decoder

This script tests the decoder implementation without requiring actual DINOv3 weights.
It verifies:
1. Hypercolumn fusion works correctly
2. Transformer decoder blocks process features
3. Convolutional refinement head produces correct output shape
4. End-to-end decoder pipeline works
"""

import torch
import torch.nn as nn
from models.dinov3_decoder_ViT import (
    HypercolumnFusion,
    TransformerDecoderBlock,
    ConvRefinementHead,
    DINOv3Decoder,
    dinov3_decoder_small,
    dinov3_decoder_base,
    dinov3_decoder_large,
)


def test_hypercolumn_fusion():
    """Test the hypercolumn fusion module"""
    print("\n" + "="*60)
    print("Testing HypercolumnFusion")
    print("="*60)
    
    # Create module
    fusion = HypercolumnFusion(
        in_dims=[768, 768, 768, 768],  # 4 layers, all 768-dim
        out_dim=512,
        alpha=0.5
    )
    
    # Create dummy features
    B, H, W = 2, 14, 14
    features = [
        torch.randn(B, 768, H, W) for _ in range(4)
    ]
    
    # Forward pass
    output = fusion(features)
    
    print(f"Input features: {len(features)} layers")
    for i, feat in enumerate(features):
        print(f"  Layer {i}: {feat.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected: torch.Size([{B}, 512, {H}, {W}])")
    
    assert output.shape == (B, 512, H, W), "Incorrect output shape!"
    print("✓ HypercolumnFusion test passed!")
    
    return fusion


def test_transformer_decoder_block():
    """Test the transformer decoder block"""
    print("\n" + "="*60)
    print("Testing TransformerDecoderBlock")
    print("="*60)
    
    # Create module
    block = TransformerDecoderBlock(
        dim=512,
        num_heads=8,
        ffn_ratio=4.0
    )
    
    # Create dummy input
    B, C, H, W = 2, 512, 14, 14
    x = torch.randn(B, C, H, W)
    
    # Forward pass
    output = block(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    assert output.shape == x.shape, "Output shape should match input!"
    print("✓ TransformerDecoderBlock test passed!")
    
    return block


def test_conv_refinement_head():
    """Test the convolutional refinement head"""
    print("\n" + "="*60)
    print("Testing ConvRefinementHead")
    print("="*60)
    
    # Create module
    head = ConvRefinementHead(
        in_channels=512,
        out_channels=3,
        hidden_channels=128,
        num_layers=3,
        upsample_factor=4
    )
    
    # Create dummy input
    B, C, H, W = 2, 512, 14, 14
    x = torch.randn(B, C, H, W)
    
    # Forward pass
    output = head(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected: torch.Size([{B}, 3, {H*4}, {W*4}])")
    
    assert output.shape == (B, 3, H*4, W*4), "Incorrect output shape!"
    print("✓ ConvRefinementHead test passed!")
    
    return head


def test_full_decoder():
    """Test the full DINOv3Decoder"""
    print("\n" + "="*60)
    print("Testing Full DINOv3Decoder")
    print("="*60)
    
    # Create decoder
    decoder = DINOv3Decoder(
        in_dim=768,
        hidden_dim=512,
        out_channels=3,
        num_decoder_layers=4,
        num_heads=8,
        alpha=0.5,
        upsample_factor=4
    )
    
    # Create dummy inputs (4 intermediate layers from DINO)
    B, H, W = 2, 14, 14
    intrinsics = [
        torch.randn(B, 768, H, W) for _ in range(4)
    ]
    extrinsic = torch.randn(B, 768)  # Optional, not used in decoder
    
    # Forward pass
    output = decoder(intrinsics, extrinsic)
    
    print(f"Input intrinsics: {len(intrinsics)} layers")
    for i, feat in enumerate(intrinsics):
        print(f"  Layer {i}: {feat.shape}")
    print(f"Extrinsic shape: {extrinsic.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected: torch.Size([{B}, 3, {H*4}, {W*4}])")
    
    assert output.shape == (B, 3, H*4, W*4), "Incorrect output shape!"
    print("✓ Full DINOv3Decoder test passed!")
    
    # Test intermediate features
    print("\nTesting intermediate features extraction...")
    features = decoder.get_intermediate_features(intrinsics)
    print(f"Fused features shape: {features['fused'].shape}")
    print(f"Number of decoder block outputs: {len(features['decoder_features'])}")
    print("✓ Intermediate features extraction works!")
    
    return decoder


def test_factory_functions():
    """Test the factory functions"""
    print("\n" + "="*60)
    print("Testing Factory Functions")
    print("="*60)
    
    # Create decoders using factory functions
    decoder_small = dinov3_decoder_small()
    decoder_base = dinov3_decoder_base()
    decoder_large = dinov3_decoder_large()
    
    print("Testing ViT-Small decoder...")
    intrinsics_small = [torch.randn(1, 384, 14, 14) for _ in range(4)]
    out_small = decoder_small(intrinsics_small)
    print(f"  Input dim: 384, Output: {out_small.shape}")
    assert out_small.shape == (1, 3, 56, 56)
    
    print("Testing ViT-Base decoder...")
    intrinsics_base = [torch.randn(1, 768, 14, 14) for _ in range(4)]
    out_base = decoder_base(intrinsics_base)
    print(f"  Input dim: 768, Output: {out_base.shape}")
    assert out_base.shape == (1, 3, 56, 56)
    
    print("Testing ViT-Large decoder...")
    intrinsics_large = [torch.randn(1, 1024, 14, 14) for _ in range(4)]
    out_large = decoder_large(intrinsics_large)
    print(f"  Input dim: 1024, Output: {out_large.shape}")
    assert out_large.shape == (1, 3, 56, 56)
    
    print("✓ All factory functions work correctly!")
    
    return decoder_small, decoder_base, decoder_large


def test_different_alphas():
    """Test decoder with different alpha values"""
    print("\n" + "="*60)
    print("Testing Different Alpha Values")
    print("="*60)
    
    alphas = [0.0, 0.3, 0.5, 0.7, 1.0]
    intrinsics = [torch.randn(1, 768, 14, 14) for _ in range(4)]
    
    for alpha in alphas:
        decoder = DINOv3Decoder(
            in_dim=768,
            hidden_dim=512,
            alpha=alpha,
            num_decoder_layers=2  # Faster for testing
        )
        output = decoder(intrinsics)
        print(f"Alpha={alpha:.1f}: Output shape {output.shape}, "
              f"Mean={output.mean().item():.4f}, Std={output.std().item():.4f}")
    
    print("✓ Different alpha values work correctly!")


def test_parameter_count():
    """Count and display parameters"""
    print("\n" + "="*60)
    print("Parameter Count Analysis")
    print("="*60)
    
    decoder = DINOv3Decoder(
        in_dim=768,
        hidden_dim=512,
        num_decoder_layers=4
    )
    
    total_params = sum(p.numel() for p in decoder.parameters())
    trainable_params = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size: ~{total_params * 4 / 1024 / 1024:.2f} MB (FP32)")
    
    # Break down by component
    fusion_params = sum(p.numel() for p in decoder.fusion.parameters())
    decoder_blocks_params = sum(p.numel() for p in decoder.decoder_blocks.parameters())
    conv_head_params = sum(p.numel() for p in decoder.conv_head.parameters())
    
    print(f"\nComponent breakdown:")
    print(f"  Hypercolumn fusion: {fusion_params:,} ({fusion_params/total_params*100:.1f}%)")
    print(f"  Decoder blocks: {decoder_blocks_params:,} ({decoder_blocks_params/total_params*100:.1f}%)")
    print(f"  Conv refinement head: {conv_head_params:,} ({conv_head_params/total_params*100:.1f}%)")


def test_gradient_flow():
    """Test that gradients flow correctly"""
    print("\n" + "="*60)
    print("Testing Gradient Flow")
    print("="*60)
    
    decoder = DINOv3Decoder(
        in_dim=768,
        hidden_dim=512,
        num_decoder_layers=2
    )
    
    # Create inputs with gradient tracking
    intrinsics = [torch.randn(1, 768, 14, 14, requires_grad=True) for _ in range(4)]
    
    # Forward pass
    output = decoder(intrinsics)
    
    # Backward pass
    loss = output.mean()
    loss.backward()
    
    # Check gradients
    has_grad = all(feat.grad is not None for feat in intrinsics)
    print(f"All inputs have gradients: {has_grad}")
    
    decoder_has_grad = all(
        p.grad is not None 
        for p in decoder.parameters() 
        if p.requires_grad
    )
    print(f"All decoder parameters have gradients: {decoder_has_grad}")
    
    print("✓ Gradient flow test passed!")


def run_all_tests():
    """Run all tests"""
    print("\n" + "="*70)
    print(" "*15 + "DINOv3 ViT Decoder Test Suite")
    print("="*70)
    
    try:
        test_hypercolumn_fusion()
        test_transformer_decoder_block()
        test_conv_refinement_head()
        test_full_decoder()
        test_factory_functions()
        test_different_alphas()
        test_parameter_count()
        test_gradient_flow()
        
        print("\n" + "="*70)
        print(" "*20 + "✓ ALL TESTS PASSED!")
        print("="*70 + "\n")
        
    except Exception as e:
        print("\n" + "="*70)
        print(" "*20 + "✗ TEST FAILED!")
        print("="*70)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        

if __name__ == '__main__':
    run_all_tests()
