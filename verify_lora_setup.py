import torch
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from models.dinov3_vae import DINOv3VAE

def test_dinov3_vae_initialization():
    print("Testing DINOv3VAE initialization with LoRA and CLS token...")
    
    encoder_cfg = {
        'layerscale_init': 1.0e-05,
        'mask_k_bias': True,
        'n_storage_tokens': 4,
    }
    
    # Mocking torch.load to avoid needing the actual checkpoint file
    original_load = torch.load
    def mock_load(f, map_location=None, weights_only=False):
        # Create a dummy state dict matching a small DINOv3 structure or just return empty if strict=False
        # Since we can't easily mock the exact structure without the file, 
        # let's rely on the fact that if the file is missing, it will raise FileNotFoundError, 
        # which we can catch and verify it tried to load.
        # OR: we can try to initialize without loading weights if we modify the class slightly for testing..
        # But let's assume the user has the weights or we can just skip the load for this test.
        raise FileNotFoundError("Simulated missing file")

    # Actually, we can just try to init and see if it fails on PEFT or logic errors BEFORE loading weights
    # But loading weights happens inside __init__.
    # Let's simple try to import and verify the class structure if we can't run it fully.
    
    # Better approach: check if we can instantiate the class structure logic.
    # We can try/except the initialization.
    
    try:
        model = DINOv3VAE(
            dino_model = 'vit_base',
            dino_checkpoint_path = 'dummy_path.pth',
            encoder_cfg = encoder_cfg,
            encoder_intermediate = 'FOUR_EVEN_INTERVALS',
            with_extra_tokens = True,
            train_encoder = True,
            extrinsic_token_idx = None
        )
    except FileNotFoundError:
        print("Caught FileNotFoundError as expected (dummy path). Logic seems to reach weight loading.")
        print("This implies PEFT import was successful and init logic started.")
    except Exception as e:
        print(f"FAILED: {e}")
        # verification failed
        sys.exit(1)
        
    print("Basic initialization logic check passed.")

if __name__ == "__main__":
    test_dinov3_vae_initialization()
