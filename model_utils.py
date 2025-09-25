import torch, pdb
import torch.nn as nn
import torch.nn.parallel
import torch.distributed as dist
import numpy as np
from PIL import Image
F = torch.nn.functional

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

def compute_logdet(W, eps = 0.2):
    m, p = W.shape  # [B, d]
    W = W.float()
    I = torch.eye(p, device=W.device)
    scalar = p / (m * eps)
    cov = W.T.matmul(W)
    cov = torch.stack(FullGatherLayer.apply(cov)).mean(dim = 0)
    logdet = torch.logdet(I + scalar * cov)
    return logdet / 2.

def intrinsic_loss(intrinsic1_list, intrinsic2_list):
    sim_intrinsic_list = []
    for intrinsic1, intrinsic2 in zip(intrinsic1_list, intrinsic2_list):
        sim_intrinsic_list.append((intrinsic1 * intrinsic2).sum(dim = 1).mean())
    return torch.stack(sim_intrinsic_list).mean()

def intrinsic_loss_ViT(intrinsic1_list, intrinsic2_list):
    sim_intrinsic_list = []
    for intrinsic1, intrinsic2 in zip(intrinsic1_list, intrinsic2_list):
        intrinsic1 = F.normalize(intrinsic1, dim = -1)
        intrinsic2 = F.normalize(intrinsic2, dim = -1)
        sim_intrinsic_list.append((intrinsic1 * intrinsic2).sum(dim = -1).mean())
    return torch.stack(sim_intrinsic_list).mean()

class compute_logdet_loss():
    def __init__(self):
        self.target_val_list = None

    def __call__(self, data_list):
        logdet_list = []
        target_val_list  = []
        with torch.cuda.amp.autocast(enabled = False):
            for idx, data in enumerate(data_list):
                # data: [B, N, C(1024)] (VIT) or [B, C, H, W] (CNN)
                if len(data.shape) == 4:
                    # When using CNN, it should be [B, C, H, W]
                    data = data.permute(0,2,3,1).reshape(-1, data.shape[1])
                elif len(data.shape) == 3:
                    # When using ViT, it should be [B, N, C]
                    data = data.reshape(-1, data.shape[-1])
                logdet = compute_logdet(F.normalize(data, dim = -1))
                # logdet = compute_logdet(data)
                logdet_list.append(logdet)
                if self.target_val_list is None:
                    logdet_target = compute_logdet(F.normalize(torch.randn_like(data), dim = -1)).detach().data
                    target_val_list.append(logdet_target)
            if self.target_val_list is None:
                self.target_val_list = target_val_list

            return torch.stack(logdet_list), torch.stack(self.target_val_list)

@torch.no_grad()
def plot_relight_img_train(model, input_img, ref_img, target_img, save_path):
    model.eval()

    img1 = input_img
    img2 = ref_img
    img3 = target_img

    intrinsic1, extrinsic1 = model(img1, run_encoder = True)
    intrinsic2, extrinsic2 = model(img2, run_encoder = True)
    intrinsic3, extrinsic3 = model(img3, run_encoder = True)

    with torch.no_grad():
        # Reconstruction
        edm_gen_img_e1_i1 = model([intrinsic1, extrinsic1], run_encoder = False)[:25]

    with torch.no_grad():
        # Relighting with extrinsics inferred from the reference
        edm_gen_img_e2_i1 = model([intrinsic1, extrinsic2], run_encoder = False)[:25]

    with torch.no_grad():
        # Relighting with target extrinsic
        edm_gen_img_e3_i1 = model([intrinsic1, extrinsic3], run_encoder = False)[:25]

    def save_img(img_list, name):
        grid_size = 4
        white_space = (np.ones((256*grid_size, 20, 3)).astype(np.float32) * 255).astype(np.uint8)
        np_img_list = []
        print("🍉",len(img_list))
        
        for img in img_list:
            print("img.shape", img.shape)
        for img in img_list:
            img = ((img[:(grid_size**2)].clamp(-1,1) * 0.5 + 0.5).reshape(grid_size, grid_size, 3, 256, 256).permute(0,3, 1, 4, 2).reshape(256*grid_size, 256*grid_size, 3) * 255).cpu().data.numpy().astype(np.uint8)
            np_img_list.append(img)
            np_img_list.append(white_space)
        Image.fromarray(np.concatenate(np_img_list, axis = 1)).save(f'{name}.png')
    save_img([img1, img2, img3, edm_gen_img_e1_i1, edm_gen_img_e2_i1, edm_gen_img_e3_i1], save_path)

@torch.no_grad()
def plot_relight_img_train_ViT(model, input_img, ref_img, target_img, save_path):
    model.eval()

    img1 = input_img
    img2 = ref_img
    img3 = target_img

    latent1, _, ids_restore1 = model.forward_encoder(img1, mask_ratio=0)    # no masking
    intrinsic1, extrinsic1 = latent1[:, 1:, :], latent1[:, :1, :]
    latent2, _, ids_restore2 = model.forward_encoder(img2, mask_ratio=0)    # no masking
    intrinsic2, extrinsic2 = latent2[:, 1:, :], latent2[:, :1, :]
    latent3, _, ids_restore3 = model.forward_encoder(img3, mask_ratio=0)    # no masking
    intrinsic3, extrinsic3 = latent3[:, 1:, :], latent3[:, :1, :]
    
    with torch.no_grad():
        # Reconstruction
        recon_patch_e1_i1 = model.forward_decoder(torch.cat([extrinsic1, intrinsic1], dim=1), ids_restore1).float()
        recon_img_e1_i1 = model.unpatchify(recon_patch_e1_i1).float()
    with torch.no_grad():
        # Relighting with extrinsics inferred from the reference
        recon_patch_e2_i1 = model.forward_decoder(torch.cat([extrinsic2, intrinsic1], dim=1), ids_restore1).float()
        recon_img_e2_i1 = model.unpatchify(recon_patch_e2_i1).float()
    with torch.no_grad():
        # Relighting with target extrinsic
        recon_patch_e3_i1 = model.forward_decoder(torch.cat([extrinsic3, intrinsic1], dim=1), ids_restore1).float()
        recon_img_e3_i1 = model.unpatchify(recon_patch_e3_i1).float()

    print("🚕", img1.shape, recon_img_e1_i1.shape)

    def save_img(img_list, name):
        grid_size = 4
        white_space = (np.ones((224*grid_size, 20, 3)).astype(np.float32) * 255).astype(np.uint8)
        np_img_list = []
        
        for img in img_list:
            print("img.shape", img.shape)
        for img in img_list:
            img = ((img[:(grid_size**2)].clamp(-1,1) * 0.5 + 0.5).reshape(grid_size, grid_size, 3, 224, 224).permute(0, 3, 1, 4, 2).reshape(224*grid_size, 224*grid_size, 3) * 255).cpu().data.numpy().astype(np.uint8)
            np_img_list.append(img)
            np_img_list.append(white_space)

        Image.fromarray(np.concatenate(np_img_list, axis = 1)).save(f'{name}.png')
    save_img([img1, img2, img3, recon_img_e1_i1, recon_img_e2_i1, recon_img_e3_i1], save_path)

class FullGatherLayer(torch.autograd.Function):
    """
    Gather tensors from all process and support backward propagation
    for the gradients across processes.
    """

    @staticmethod
    def forward(ctx, x):
        output = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        all_gradients = torch.stack(grads)
        dist.all_reduce(all_gradients)
        return all_gradients[dist.get_rank()]
