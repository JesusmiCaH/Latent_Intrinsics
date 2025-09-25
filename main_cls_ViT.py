import argparse
import os, json
import random
import shutil
import time, glob, copy
import os
import time
import torch
import argparse
import math
from tqdm import tqdm
import warnings
import numpy as np
import torch, pdb
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.models as models
import matplotlib.pyplot as plt
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import argparse, pdb
import numpy as np
from torch import autograd
from torch.optim import AdamW
from typing import Union, List, Optional, Callable
import pdb
from utils import  AverageMeter, ProgressMeter, init_ema_model, update_ema_model
import builtins
from PIL import Image
import torchvision
import tqdm
from utils import MIT_Dataset, affine_crop_resize, multi_affine_crop_resize, MIT_Dataset_PreLoad
import ViT_autoencoder
import copy
from pytorch_ssim import SSIM as compute_SSIM_loss
from pytorch_losses import gradient_loss
from model_utils import plot_relight_img_train_ViT, compute_logdet_loss, intrinsic_loss_ViT, save_checkpoint

#from sklearn.metrics import average_precision_score
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('-j', '--workers', default=10, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--img_size', default=224, type=int,
                    help='img size')
parser.add_argument('--affine_scale', default=5e-3, type=float)
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--save_freq', default=5, type=int,
                     help='print frequency (default: 10)')
parser.add_argument('--resume', action = 'store_true',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--mae_ckpth', default='mae_visualize_vit_large_ganloss.pth', type=str, metavar='PATH',
                    help='path to mae checkpoint (default: none)')
parser.add_argument('--world-size', default=-1, type=int,
                        help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='env://', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--setting', default='0_0_0', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--local_rank', default=-1, type=int,
                    help='local rank for distributed training')
parser.add_argument('--local-rank', default=-1, type=int,
                    help='local rank for distributed training')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument("--learning_rate", type=float, default=1e-3)
parser.add_argument("--intrinsics_loss_weight", type=float, default=1e-1)
parser.add_argument("--reg_weight", type=float, default=1e-4)
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--weight_decay", type=float, default=0)
# training params
parser.add_argument("--gpus", type=int, default=1)
# datamodule params
parser.add_argument("--data_path", type=str, default=".")
parser.add_argument("--vit_arch", type=str, default='mae_vit_large_patch16')

args = parser.parse_args()


def init_model(args):
    model = getattr(ViT_autoencoder, args.vit_arch)()
    model.cuda(args.gpu)
    checkpoint = torch.load(args.mae_ckpth, map_location='cpu')
    print("Checkpoint keys:", checkpoint.keys())
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    print("🚋", msg)

    # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True,
    # broadcast_buffers=False)

    optimizer = AdamW(model.parameters(),
                lr= args.learning_rate, weight_decay = args.weight_decay)
    return model, optimizer

def main():
    torch.manual_seed(2)
    import os
    #torch.backends.cudnn.benchmark=False
    cudnn.deterministic = True
    args = parser.parse_args()
    #assert args.batch_size % args.batch_iter == 0
    if not os.path.exists('visualize'):
        os.system('mkdir visualize')
    if not os.path.exists('checkpoint'):
        os.system('mkdir checkpoint')
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if "WORLD_SIZE" in os.environ:
        args.world_size = int(os.environ["WORLD_SIZE"])
    args.distributed = args.world_size >= 1
    ngpus_per_node = torch.cuda.device_count()

    print('start')
    if args.distributed:
        if args.local_rank != -1: # for torch.distributed.launch
            args.rank = args.local_rank
            args.gpu = args.local_rank
        elif 'SLURM_PROCID' in os.environ: # for slurm scheduler
            args.rank = int(os.environ['SLURM_PROCID'])
            args.gpu = args.rank % torch.cuda.device_count()
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    args.gpu = args.gpu % torch.cuda.device_count()
    print('world_size', args.world_size)
    print('rank', args.rank)
    # suppress printing if not on master gpu
    if args.rank!=0:
        def print_pass(*args):
            pass
        builtins.print = print_pass

    args.distributed = args.world_size >= 1 or args.multiprocessing_distributed
    main_worker(args.gpu, ngpus_per_node, args)

def main_worker(gpu, ngpus_per_node, args):
    args = copy.deepcopy(args)
    args.cos = True
    save_folder_path = '''checkpoint/intrinsics_loss_weight_{}_reg_weight_{}_lr_{}_batch_size_{}_weight_decay_{}_affine_scale_{}'''.replace('\n',' ').replace(' ','').format(
                        args.intrinsics_loss_weight, args.reg_weight, args.learning_rate, args.batch_size, args.weight_decay, args.affine_scale)
    args.save_folder_path = save_folder_path
    args.is_master = args.rank == 0

    model, optimizer = init_model(args)
    torch.cuda.set_device(args.gpu)

    # optimizer = AdamW(model.parameters(),
    #             lr= args.learning_rate, weight_decay = args.weight_decay)
    scaler = torch.cuda.amp.GradScaler()

    args.start_epoch = 0
    if args.resume:
        args.resume = '{}/last.pth.tar'.format(save_folder_path)
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scaler.load_state_dict(checkpoint['scaler'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
            del checkpoint
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    transform_train = [affine_crop_resize(size = (224, 224), scale = (0.2, 1.0)),
    transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
                    mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
                ),
    ])]

    #train_dataset = MIT_Dataset(args.data_path, transform_train)
    train_dataset = MIT_Dataset_PreLoad(args.data_path, transform_train, total_split = args.world_size, split_id = args.rank)

    print('NUM of training images: {}'.format(len(train_dataset)))
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle = True, drop_last = True)
    else:
        train_sampler = None
    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=False, sampler=train_sampler, drop_last = True, persistent_workers = True)

    if not args.multiprocessing_distributed or (args.multiprocessing_distributed
            and args.is_master):
        if not os.path.exists(save_folder_path):
            os.system('mkdir -p {}'.format(save_folder_path))

    for epoch in range(args.start_epoch, 120):
        if args.distributed and train_sampler is not None:
            train_sampler.set_epoch(epoch)
        loss_pack = train_D(train_loader, model, scaler, optimizer, epoch, args)

        if epoch == 0:
            loss_list, rec_loss_list, sim_intrinsic_list = loss_pack
        else:
            loss_list += loss_pack[0]
            rec_loss_list += loss_pack[1]
            sim_intrinsic_list += loss_pack[2]
        
        print("📈", len(loss_list), len(rec_loss_list), len(sim_intrinsic_list))
        if args.is_master:
            if epoch % 2 == 0 and epoch != 0:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'optimizer' : optimizer.state_dict(),
                    'scaler': scaler.state_dict(),
                }, False, filename = '{}/last.pth.tar'.format(save_folder_path))

                plt.figure(figsize=(20,5))
                plt.subplot(1,3,1)
                plt.plot(loss_list)
                plt.title('total loss')
                plt.subplot(1,3,2)
                plt.plot(rec_loss_list)
                plt.title('reconstruction loss')
                plt.subplot(1,3,3)
                plt.plot(sim_intrinsic_list)
                plt.title('intrinsic similarity')
                plt.savefig('{}/training_curves.png'.format(save_folder_path))
                plt.close('all')
                
            if epoch % 20 == 0 and epoch != 0:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'optimizer' : optimizer.state_dict(),
                    'scaler': scaler.state_dict(),
                },False , filename = '{}/{}.pth.tar'.format(save_folder_path, epoch))

def print_gradients(model):
    max_grad = 0
    max_norm = 0
    for name, p in model.named_parameters():
        if p.grad is not None:
            max_grad = max(max_grad, p.grad.norm())
            max_norm = max(max_norm, p.data.norm(2))
    return max_grad, max_norm

def train_D(train_loader, model, scaler, optimizer, epoch, args):
    loss_name = [
                'loss','logdet', 'light_logdet', 'intrinsic_sim',
                'GPU Mem', 'Time', 'pe', 'ge']
    moco_loss_meter = [AverageMeter(name, ':6.3f') for name in loss_name]
    progress = ProgressMeter(
        len(train_loader),
        moco_loss_meter,
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    t0 = time.time()
    P_mean=-1.2
    P_std=1.2
    sigma_data = 0.5

    loss_list = []
    rec_loss_list = []
    sim_intrinsic_list = []
    logdet_loss_intrinsic = compute_logdet_loss()
    logdet_loss_extrinsic = compute_logdet_loss()
    ssim_loss = compute_SSIM_loss()
    for i, (input_img, ref_img) in enumerate(train_loader):
        input_img = input_img.to(args.gpu)
        ref_img = ref_img.to(args.gpu)

        rnd_normal = torch.randn([input_img.shape[0], 1, 1, 1], device=input_img.device)
        sigma = (rnd_normal * P_std + P_mean).exp()
        if epoch >= 60:
            sigma = sigma * 0

        noisy_input_img = input_img + torch.randn_like(input_img) * sigma   # nchw
        noisy_ref_img = ref_img + torch.randn_like(ref_img) * sigma

        if i >= 70:
            break
            
        with torch.cuda.amp.autocast():
            latent_input, _, ids_restore_input = model.forward_encoder(noisy_input_img, mask_ratio=0)    # no masking
            intrinsic_input, extrinsic_input = latent_input[:, 1:, :], latent_input[:, :1, :]

            latent_ref, _, ids_restore_ref = model.forward_encoder(noisy_ref_img, mask_ratio=0)    # no masking
            intrinsic_ref, extrinsic_ref = latent_ref[:, 1:, :], latent_ref[:, :1, :]

            mask = (torch.rand(input_img.shape[0]) > 0.9).float().to(args.gpu).reshape(-1,1,1).float()    # 10% mask
            
            # Intrinsic mainly from reference image
            intrinsic = mask * intrinsic_input + (1 - mask) * intrinsic_ref # [N, L, D]

            recon_patches = model.forward_decoder(torch.cat([extrinsic_input, intrinsic], dim=1), ids_restore_ref)
            recon_img = model.unpatchify(recon_patches).float()
            # Maybe need some torch.einsum

        # Since logdet_loss function take only LIST as input due to the original design, we should pack our input matrix into a list
        logdet_pred, logdet_target = logdet_loss_intrinsic([intrinsic_input])
        logdet_pred_ext, logdet_target_ext = logdet_loss_extrinsic([extrinsic_input])

        sim_intrinsic = intrinsic_loss_ViT([intrinsic_input], [intrinsic_ref])
        rec_loss = nn.MSELoss()(recon_img,input_img)

        rec_loss = 10 * rec_loss + \
                0.1 * (1 - ssim_loss(recon_img, input_img)) + gradient_loss(recon_img,input_img)
        loss = rec_loss + args.reg_weight * ((logdet_pred - logdet_target) ** 2).mean() + \
                          args.reg_weight * ((logdet_pred_ext - logdet_target_ext) ** 2).mean() + \
                          - args.intrinsics_loss_weight * sim_intrinsic

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        ge, pe = print_gradients(model)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        t1 = time.time()        
        # print("🛠️", loss, sim_intrinsic)

        for val_id, val in enumerate([rec_loss, logdet_pred[:-1].mean(), logdet_pred[-1], sim_intrinsic,
                        torch.cuda.max_memory_allocated() / (1024.0 * 1024.0), t1 - t0, pe, ge
                    ]):
            if not isinstance(val, float) and not isinstance(val, int):
                val = val.item()
            moco_loss_meter[val_id].update(val)
        progress.display(i)
        t0 = time.time()
        torch.cuda.reset_peak_memory_stats()

        loss_list.append(loss.item())
        rec_loss_list.append(rec_loss.item())
        sim_intrinsic_list.append(sim_intrinsic.item())

    if args.gpu == 0 and epoch % 5 == 0:
        target_img = ref_img[torch.randperm(input_img.shape[0]).to(args.gpu)]
        plot_relight_img_train_ViT(model, input_img, ref_img, target_img, args.save_folder_path + '/{:05d}_{:05d}_gen'.format(epoch + 1, i))

    torch.distributed.barrier()

    return loss_list, rec_loss_list, sim_intrinsic_list

if __name__ == '__main__':
    main()
