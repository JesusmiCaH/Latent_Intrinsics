"""
Demo script for Latent Extrinsic Interpolation
Based on the paper: "Latent Intrinsics Emerge from Training to Relight" (NeurIPS 2024)

This script demonstrates how to interpolate between different extrinsic (lighting) conditions
while keeping the intrinsic (material) properties constant.
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import argparse
import os
from model_utils import plot_relight_img_train
from models import ViT_autoencoder

def prepare_model(chkpt_dir, arch='mae_vit_large_patch16'):
    """Load pre-trained model"""
    model = getattr(ViT_autoencoder, arch)()
    checkpoint = torch.load(chkpt_dir, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    print(f"Model loaded: {msg}")
    return model

def load_and_preprocess_image(image_path, size=(224, 224)):
    """Load and preprocess image for ViT model"""
    # Handle single image or list of images
    if isinstance(image_path, (list, tuple)):
        # Load multiple images and stack them as a batch
        img_tensors = []
        for path in image_path:
            img = Image.open(path).convert('RGB')
            img = img.resize(size)
            img = np.array(img) / 255.0
            
            # Convert to tensor and normalize to [-1, 1] range
            img_tensor = torch.tensor(img).float()
            img_tensor = img_tensor.permute(2, 0, 1)  # HWC -> CHW
            img_tensor = (img_tensor - 0.5) * 2  # Normalize to [-1, 1]
            img_tensors.append(img_tensor)
        
        # Stack into batch
        batch_tensor = torch.stack(img_tensors, dim=0)
        return batch_tensor
    else:
        # Single image (original behavior)
        img = Image.open(image_path).convert('RGB')
        img = img.resize(size)
        img = np.array(img) / 255.0
        
        # Convert to tensor and normalize to [-1, 1] range
        img_tensor = torch.tensor(img).float()
        img_tensor = img_tensor.unsqueeze(0).permute(0, 3, 1, 2)
        img_tensor = (img_tensor - 0.5) * 2  # Normalize to [-1, 1]
        
        return img_tensor

def extrinsic_interpolation(model, img1, img2, img3, num_steps=5, save_path="interpolation_result.png"):
    """
    Perform extrinsic (lighting) interpolation between two images
    This demonstrates the key finding: "Latent extrinsics can be interpolated successfully"
    
    Args:
        model: Pre-trained ViT autoencoder model
        img1: Source image (intrinsic properties will be kept constant)
        img2: First reference image (provides start extrinsic/lighting)
        img3: Second reference image (provides end extrinsic/lighting)
        num_steps: Number of interpolation steps
        save_path: Path to save visualization
    """
    model.eval()
    device = next(model.parameters()).device
    
    # Move images to device
    img1 = img1.to(device)
    img2 = img2.to(device) 
    img3 = img3.to(device)
    
    with torch.no_grad():
        # Extract intrinsic and extrinsic components using the model's encoder
        latent1, _, ids_restore1 = model.forward_encoder(img1, mask_ratio=0)
        intrinsic1, extrinsic1 = latent1[:, 1:, :], latent1[:, :1, :]  # Split latent representation
        
        latent2, _, ids_restore2 = model.forward_encoder(img2, mask_ratio=0)
        intrinsic2, extrinsic2 = latent2[:, 1:, :], latent2[:, :1, :]
        
        latent3, _, ids_restore3 = model.forward_encoder(img3, mask_ratio=0)
        intrinsic3, extrinsic3 = latent3[:, 1:, :], latent3[:, :1, :]
        
        print(f"💡 Extracted latent components:")
        print(f"   Intrinsic shape: {intrinsic1.shape} (material properties)")
        print(f"   Extrinsic shape: {extrinsic1.shape} (lighting conditions)")
        
        # Generate interpolated results - this is the key contribution!
        interpolated_images = []
        alphas = torch.linspace(0, 1, num_steps)
        
        for i, alpha in enumerate(alphas):
            # 🔥 KEY INSIGHT: Linear interpolation in latent extrinsic space works!
            # This shows that the learned extrinsic space has meaningful geometry
            interpolated_extrinsic = (1 - alpha) * extrinsic2 + alpha * extrinsic3
            
            # Reconstruct image: Keep img1's intrinsic + use interpolated extrinsic
            combined_latent = torch.cat([interpolated_extrinsic, intrinsic1], dim=1)
            recon_patches = model.forward_decoder(combined_latent, ids_restore1)
            recon_img = model.unpatchify(recon_patches).float()
            
            interpolated_images.append(recon_img)
            print(f"   Step {i+1}/{num_steps}: α={alpha:.2f} (mixing {1-alpha:.1%} of lighting2 + {alpha:.1%} of lighting3)")
        
        # Also generate some additional demonstrations
        print(f"\n🎨 Generating additional relighting examples...")
        
        # Example 1: Pure relighting (same as paper's relight evaluation)
        relight_with_2 = torch.cat([extrinsic2, intrinsic1], dim=1)
        relight_img_2 = model.unpatchify(model.forward_decoder(relight_with_2, ids_restore1)).float()
        
        relight_with_3 = torch.cat([extrinsic3, intrinsic1], dim=1)
        relight_img_3 = model.unpatchify(model.forward_decoder(relight_with_3, ids_restore1)).float()
        
        # Create comprehensive visualization
        all_images = [img1, img2, img3, relight_img_2, relight_img_3] + interpolated_images
        all_titles = ['Source\n(Material)', 'Lighting A', 'Lighting B', 
                     'Source+LightA', 'Source+LightB'] + [f'α={a:.2f}' for a in alphas]
        
        # Detect batch size from the first image
        batch_size = img1.shape[0]
        save_interpolation_grid(all_images, save_path, titles=all_titles, batch_size=batch_size)
        
        return interpolated_images, (relight_img_2, relight_img_3)

def save_interpolation_grid(images, save_path, titles=None, batch_size=1):
    """Save a grid of images showing the interpolation results"""
    n_images = len(images)
    
    # Convert tensors to numpy arrays - handle batch dimension
    np_images = []
    for img in images:
        # Check if this is a batch of images (batch_size > 1)
        if img.shape[0] > 1:
            C, H, W = img.shape[1], img.shape[2], img.shape[3]
            # Process each image in the batch
            batch_imgs = []
            img = (img.clamp(-1, 1) * 0.5 + 0.5).reshape(2,1, img.shape[1], img.shape[2], img.shape[3])     # 2,1,C,H,W
            img = img.permute(0,3,1,4,2)    # 2,H,1,W,C

            for b in range(img.shape[0]):
                img_np = (img[b].clamp(-1, 1) * 0.5 + 0.5).cpu().numpy().transpose(1, 2, 0)
                img_np = (img_np * 255).astype(np.uint8)
                batch_imgs.append(img_np)
            np_images.append(batch_imgs)
        else:
            # Single image (original behavior)
            img_np = (img[0].clamp(-1, 1) * 0.5 + 0.5).cpu().numpy().transpose(1, 2, 0)
            img_np = (img_np * 255).astype(np.uint8)
            np_images.append([img_np])
    
    # Determine grid layout based on batch size
    batch_size = len(np_images[0])
    
    if batch_size == 2:
        # Arrange as 2 columns when batch_size = 2
        n_cols = 2
        n_rows = n_images
        
        h, w = np_images[0][0].shape[:2]
        title_height = 30 if titles else 0
        spacing = 20
        
        grid_height = n_rows * (h + title_height) + (n_rows - 1) * spacing
        grid_width = n_cols * w + (n_cols - 1) * spacing
        
        # Create white background
        grid = np.ones((grid_height, grid_width, 3), dtype=np.uint8) * 255
        
        # Place images in 2-column grid
        for row in range(n_rows):
            for col in range(n_cols):
                y_offset = row * (h + title_height + spacing)
                x_offset = col * (w + spacing)
                
                img_data = np_images[row][col]
                grid[y_offset + title_height:y_offset + title_height + h, 
                     x_offset:x_offset + w] = img_data
                
                # Add title if provided
                if titles and row * n_cols + col < len(titles):
                    # Note: For simplicity, titles are not rendered as text in this version
                    # You can add PIL ImageDraw text rendering here if needed
                    pass
    else:
        # Original horizontal layout for batch_size = 1 or other cases
        h, w = np_images[0][0].shape[:2]
        title_height = 30 if titles else 0
        grid_height = h + title_height
        grid_width = w * n_images + 20 * (n_images - 1)
        
        # Create white background
        grid = np.ones((grid_height, grid_width, 3), dtype=np.uint8) * 255
        
        # Place images horizontally
        x_offset = 0
        for i, img_list in enumerate(np_images):
            grid[title_height:title_height+h, x_offset:x_offset+w] = img_list[0]
            x_offset += w + 20
    
    # Save the result
    Image.fromarray(grid).save(save_path)
    print(f"Interpolation result saved to: {save_path}")

def main():
    parser = argparse.ArgumentParser(description='Demo: Latent Extrinsic Interpolation')
    parser.add_argument('--model_path', type=str, default='mae_visualize_vit_large_ganloss.pth',
                        help='Path to pre-trained model checkpoint')
    parser.add_argument('--img1', type=str, nargs='+', required=True,
                        help='Source image(s) (intrinsic properties will be preserved). Can be single image or list for batch processing.')
    parser.add_argument('--img2', type=str, nargs='+', required=True, 
                        help='First reference image(s) (start lighting condition). Should match img1 batch size.')
    parser.add_argument('--img3', type=str, nargs='+', required=True,
                        help='Second reference image(s) (end lighting condition). Should match img1 batch size.')
    parser.add_argument('--output', type=str, default='extrinsic_interpolation.png',
                        help='Output path for visualization')
    parser.add_argument('--steps', type=int, default=5,
                        help='Number of interpolation steps')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device id')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print("Loading model...")
    model = prepare_model(args.model_path)
    model = model.to(device)
    
    # Load images
    print("Loading images...")
    # Handle single image or batch of images
    img1_paths = args.img1 if len(args.img1) > 1 else args.img1[0]
    img2_paths = args.img2 if len(args.img2) > 1 else args.img2[0] 
    img3_paths = args.img3 if len(args.img3) > 1 else args.img3[0]
    
    img1 = load_and_preprocess_image(img1_paths)
    img2 = load_and_preprocess_image(img2_paths)  
    img3 = load_and_preprocess_image(img3_paths)
    
    print(f"Image shapes: {img1.shape}")
    print(f"Batch size: {img1.shape[0]}")
    
    # Perform interpolation
    print("Performing extrinsic interpolation...")
    interpolated_images = extrinsic_interpolation(
        model, img1, img2, img3, 
        num_steps=args.steps, 
        save_path=args.output
    )
    
    print(f"✅ Interpolation complete! Results saved to: {args.output}")
    print("\n" + "="*80)
    print("🎨 PAPER CONTRIBUTION DEMONSTRATED:")
    print("="*80)
    print("📖 Quote: \"Latent extrinsics can be interpolated successfully\"")
    print()
    print("🔬 What this demo shows:")
    print("1. 🧬 DISENTANGLEMENT: The model learns separate intrinsic & extrinsic representations")
    print("2. 🌟 INTERPOLATION: Extrinsic latents form a smooth, meaningful space")
    print("3. 🎯 CONTROL: We can smoothly transition between lighting conditions")
    print("4. 📐 GEOMETRY: Linear interpolation in latent space = realistic lighting changes")
    print()
    print("🔍 Key insight:")
    print("   Unlike pixel-space interpolation (which creates artifacts), latent extrinsic")
    print("   interpolation produces physically plausible lighting transitions because")
    print("   the model has learned a structured representation of lighting.")
    print()
    print("💡 This enables applications like:")
    print("   • Interactive relighting with smooth controls")
    print("   • Lighting style transfer")
    print("   • Virtual cinematography")
    print("="*80)

if __name__ == '__main__':
    main()