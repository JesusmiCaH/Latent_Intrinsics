"""
Quick shape debugging for DINOv3 decoder
"""
import torch

# Simulate the shape flow
print("="*60)
print("DINOv3 Decoder Shape Analysis")
print("="*60)

# Input configuration
batch_size = 16
img_size = 224
patch_size = 16
channels = 3

print(f"\nInput Configuration:")
print(f"  Image size: {img_size}×{img_size}")
print(f"  Patch size: {patch_size}×{patch_size}")
print(f"  Batch size: {batch_size}")

# Calculate encoder output size
encoder_spatial_size = img_size // patch_size
print(f"\n1. Encoder Output:")
print(f"  Spatial size: {encoder_spatial_size}×{encoder_spatial_size}")
print(f"  Shape: [{batch_size}, 768, {encoder_spatial_size}, {encoder_spatial_size}]")

# Decoder processing
print(f"\n2. Hypercolumn Fusion:")
print(f"  Input: [{batch_size}, 768, {encoder_spatial_size}, {encoder_spatial_size}]")
print(f"  Output: [{batch_size}, 512, {encoder_spatial_size}, {encoder_spatial_size}]")

print(f"\n3. Transformer Decoder Blocks:")
print(f"  Input: [{batch_size}, 512, {encoder_spatial_size}, {encoder_spatial_size}]")
print(f"  Output: [{batch_size}, 512, {encoder_spatial_size}, {encoder_spatial_size}]")

# Upsampling calculation
print(f"\n4. Convolutional Refinement Head:")
print(f"  Input: [{batch_size}, 512, {encoder_spatial_size}, {encoder_spatial_size}]")

for upsample_factor in [4, 8, 16]:
    output_size = encoder_spatial_size * upsample_factor
    print(f"\n  With upsample_factor={upsample_factor}:")
    print(f"    Output size: {output_size}×{output_size}")
    print(f"    Output shape: [{batch_size}, 3, {output_size}, {output_size}]")
    if output_size == img_size:
        print(f"    ✅ MATCHES INPUT SIZE!")
    else:
        print(f"    ❌ Does NOT match input size ({img_size})")

print(f"\n" + "="*60)
print(f"REQUIRED UPSAMPLE FACTOR: {patch_size}")
print(f"="*60)

# Show the math
print(f"\nMath:")
print(f"  encoder_output_size = img_size / patch_size = {img_size} / {patch_size} = {encoder_spatial_size}")
print(f"  decoder_output_size = encoder_output_size × upsample_factor")
print(f"  To match input: upsample_factor = patch_size = {patch_size}")
print(f"\n  {encoder_spatial_size} × {patch_size} = {encoder_spatial_size * patch_size} ✅")
