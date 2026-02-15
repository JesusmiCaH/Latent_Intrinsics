import ViT_autoencoder
import torch
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt

def prepare_model(chkpt_dir, arch='mae_vit_large_patch16'):
    # build model
    model = getattr(ViT_autoencoder, arch)()
    # load model
    checkpoint = torch.load(chkpt_dir, map_location='cpu')
    print("Checkpoint keys:", checkpoint.keys())
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    print(msg)
    return model

def get_image_for_ViT(image_path):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((224, 224))
    img = np.array(img) / 255.

    assert img.shape == (224, 224, 3)
    print("Image shape:", img.shape)
    # normalize by ImageNet mean and std
    # img = img - img.mean(axis=(0,1), keepdims=True)
    # img = img / img.std(axis=(0,1), keepdims=True)

    return img

def plot_x_y(x, y):
    plt.figure(figsize=(8, 4))
    plt.subplot(1,2,1)
    plt.imshow(x[0].detach().permute(1,2,0).cpu())
    plt.title('Original Image')
    plt.subplot(1,2,2)
    plt.imshow(y[0].detach().permute(1,2,0).cpu())
    plt.title('Reconstructed Image')
    plt.savefig('reconstructed_image.png')


chkpt_dir = 'mae_visualize_vit_large_ganloss.pth'
model_mae = prepare_model(chkpt_dir, 'mae_vit_large_patch16')
print("Yes, model is ready!")

image_path = '/home/cc/Datasets/14n_copyroom6/dir_15_mip2.jpg'
img = get_image_for_ViT(image_path)

x = torch.tensor(img)
x = x.unsqueeze(0).permute(0,3,1,2)
loss, y, mask = model_mae(x.float(), mask_ratio=0)
y = model_mae.unpatchify(y)
print(x.shape, y.shape, mask)

plot_x_y(x,y)

