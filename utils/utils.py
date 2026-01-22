import builtins, pdb
import datetime
import os,glob
import time, tqdm
from collections import defaultdict, deque
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
import torch.utils.data as data
from PIL import Image
import torchvision
import torchvision.transforms as transforms
import cv2
import json
F = torch.nn.functional
import matplotlib.pyplot as plt

class affine_crop_resize(torchvision.transforms.RandomResizedCrop):
    def __init__(self, size, flip = True, **kargs):
        self.size = size
        super(affine_crop_resize, self).__init__(size, **kargs)
        self.flip = flip
    def __call__(self, img_list):
        img = img_list[0]
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        img_h, img_w = img.shape[1:]
        affine = random_crop_resize_affine(j,j+w,i,i+h, img_w, img_h)
        if self.flip and np.random.rand() >= 0.5:
            affine = affine @ flip_affine()
        affine = torch.from_numpy(affine)[:2][None,...]
        out_img = []
        for img in img_list:
            out_img.append(apply_affine(img[None,...], affine, out_size = self.size)[0])
        return out_img

def apply_affine(img, affine, out_size = None, mode = 'bilinear', align_corners = False):
    if out_size is None:
        out_size = img.shape
    elif isinstance(out_size, int):
        out_size = torch.Size([img.shape[0], img.shape[1], out_size, out_size])
    elif isinstance(out_size, tuple):
        out_size = torch.Size([img.shape[0], img.shape[1], out_size[0], out_size[1]])
    grid = F.affine_grid(affine.float(), out_size, align_corners = align_corners)
    out = F.grid_sample(img, grid, mode, align_corners = align_corners)
    return out

def random_crop_resize_affine(x1,x2,y1,y2,width,height):
    affine = np.eye(3)
    affine[0,0] = (x2 - x1) / (width - 1)
    affine[1,1] = (y2 - y1) / (height - 1)
    affine[0,2] = (x1 + x2 - width + 1) / (width - 1)
    affine[1,2] = (y1 + y2 - height + 1) / (height - 1)
    return affine

def flip_affine(hori = True):
    affine = np.eye(3)
    if hori:
        affine[0,0] = -1
        affine[0,2] = 0
    else:
        affine[1,1] = -1
        affine[1,2] = 0
    return affine

class multi_affine_crop_resize(torchvision.transforms.RandomResizedCrop):
    def __init__(self, size, flip = True, **kargs):
        self.size = size
        super(multi_affine_crop_resize, self).__init__(size, **kargs)
        self.flip = flip

    def get_params(self, height, width, scale, ratio, num_of_sample):
        """Get parameters for ``crop`` for a random sized crop.

        Args:
            img (PIL Image or Tensor): Input image.
            scale (list): range of scale of the origin size cropped
            ratio (list): range of aspect ratio of the origin aspect ratio cropped

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
            sized crop.
        """
        area = height * width
        log_ratio = torch.log(torch.tensor(ratio))
        target_area = area * torch.empty(num_of_sample).uniform_(scale[0], scale[1])
        aspect_ratio = torch.exp(torch.empty(num_of_sample).uniform_(log_ratio[0], log_ratio[1]))

        w = torch.round(torch.sqrt(target_area * aspect_ratio)).long()
        h = torch.round(torch.sqrt(target_area / aspect_ratio)).long()
        mask1 = torch.logical_and(w > 0, w <= width)
        mask2 = torch.logical_and(h > 0, h <= height)
        mask = torch.logical_and(mask1, mask2)
        h = h[mask]
        w = w[mask]
        i = (torch.rand(size = (h.shape[0],)) * (height - h + 1)).long()
        j = (torch.rand(size = (h.shape[0],)) * (width- w + 1)).long()
        x1 = j
        x2 = j + w
        y1 = i
        y2 = i + h
        affine = torch.eye(3)[None,...].expand(i.shape[0],-1,-1).clone()
        affine[:,0,0] = (x2 - x1) / (width - 1)
        affine[:,1,1] = (y2 - y1) / (height - 1)
        affine[:,0,2] = (x1 + x2 - width + 1) / (width - 1)
        affine[:,1,2] = (y1 + y2 - height + 1) / (height - 1)
        return affine

    def get_patch_affine(self, img, patch_size, stride):
        img_h, img_w = img.shape[2:]
        patch_size_h = min(patch_size, img_h)
        patch_size_w = min(patch_size, img_w)
        img = torch.stack(torch.meshgrid(torch.arange(img_h), torch.arange(img_w)))[None,...]
        patch_num_h = (img_h - (patch_size_h - 1) - 1 + stride - 1) // stride + 1
        patch_num_w = (img_w - (patch_size_w - 1) - 1 + stride - 1) // stride + 1
        i = torch.arange(patch_num_h) * stride
        j = torch.arange(patch_num_w) * stride
        #print(patch_num_h, patch_num_w, img.shape)
        if (i[-1] + patch_size_h) > img_h:
            i[-1] = img_h - patch_size_h
        if (j[-1] + patch_size_w) > img_w:
            j[-1] = img_w - patch_size_w
        coor = torch.stack(torch.meshgrid(i, j)).reshape(2,-1)
        i = coor[0]
        j = coor[1]
        x1 = j
        x2 = j + patch_size_w
        y1 = i
        y2 = i + patch_size_h
        affine = torch.eye(3)[None,...].expand(i.shape[0],-1,-1).clone()
        affine[:,0,0] = (x2 - x1) / img_w
        affine[:,1,1] = (y2 - y1) / img_h
        affine[:,0,2] = (x1 + x2 - img_w) / img_w
        affine[:,1,2] = (y1 + y2 - img_h) / img_h
        return affine
        #img_patch = apply_affine(img.expand(affine.shape[0],-1,-1,-1).float(), affine = affine[:,:2],out_size = (patch_size, patch_size))
        #img_patch.reshape(patch_num_h, patch_num_w, 2, patch_size, patch_size)
        #print(img_patch.reshape(patch_num_h, patch_num_w, 2, patch_size, patch_size)[-1,0,0])
        #pdb.set_trace()

    def __call__(self, img, num_of_sample = 100):
        assert img.shape[0] == 1
        img_h, img_w = img.shape[2:]
        affine_list = []
        counter = 0
        while True:
            affine = self.get_params(img_h, img_w, self.scale, self.ratio, num_of_sample)
            affine_list.append(affine)
            if counter + affine.shape[0] >= num_of_sample:
                break
            counter += affine.shape[0]
        affine = torch.cat(affine_list)[:num_of_sample]
        flip_affine = torch.eye(3)[None,...].expand(affine.shape[0],-1,-1).clone()
        flip_affine[:,0,0] = (torch.rand(flip_affine.shape[0]) - 0.5).sign()
        affine = affine @ flip_affine
        return affine
        #affine = torch.from_numpy(affine)[:2][None,...]
        #out_img = []
        #for img in img_list:
        #    out_img.append(apply_affine(img[None,...], affine, out_size = self.size)[0])
        #return out_img


class IIW(data.Dataset):
    def __init__(self, root, img_transform = None, split = 'train'):
        #if split == 'train':
        #    img_index = np.load('iiw_train_ids.npy')
        #else:
        #    img_index = np.load('iiw_test_ids.npy')
        with open('val_list.txt') as f:
            val_img_list = [file_name.replace('.png\n','') for file_name in f.readlines()]
        img_index = [file_name.split('/')[-1].replace('.png', '') for file_name in glob.glob(root + '/*.png')]
        train_img_list = list(set(img_index) - set(val_img_list))
        if split == 'train':
            self.img_index = np.array(train_img_list)
        else:
            self.img_index = np.array(val_img_list)
        self.img_index.sort()
        self.root = root
        self.img_transform = img_transform
        self.split = split

    def __len__(self):
        return len(self.img_index)

    def __getitem__(self, index):
        img = Image.open(self.root + '/' + self.img_index[index] + '.png')
        img_h, img_w = np.array(img).shape[:2]
        img = self.img_transform(img)
        return img, np.array([img_h, img_w]), self.img_index[index]

def compute_rank_split(data_list, total_split, split_id):
    num_of_data = len(data_list)
    num_per_split = (num_of_data + (total_split - 1)) // total_split
    split_data_list = data_list[num_per_split * split_id: num_per_split * (split_id + 1)]
    extra_list_num = int(np.ceil(num_per_split)) - len(split_data_list)
    if extra_list_num > 0:
        split_data_list = split_data_list + split_data_list[:extra_list_num]
    return split_data_list


def parallel_load_image(img_path_list, data_filter = '/dir*.jpg'):
    class loader(torch.utils.data.Dataset):
        def __init__(self, img_path_list):
            self.img_path_list = img_path_list

        def __len__(self):
            return len(self.img_path_list)

        def __getitem__(self, index):
            img_list = sorted(glob.glob(self.img_path_list[index] + data_filter))
            images = []
            for img_path in img_list:
                images.append(np.array(Image.open(img_path).convert('RGB')))
            return np.stack(images)
 
    dataset = loader(img_path_list)
    def collate_fn(batch):
        return batch
    data_loader = torch.utils.data.DataLoader(
        dataset, sampler=None,
        batch_size=16,
        num_workers=8,
        pin_memory=False,
        drop_last=False,
        collate_fn=collate_fn
    )
    images_list = []
    
    early_quitter = 0
    for images in tqdm.tqdm(data_loader):
        images_list += images
        # early_quitter += 1
        # if early_quitter >= 5:
        #     break

    return images_list


def init_ema_model(model, ema_model):
    for param_q, param_k in zip(
        model.parameters(), ema_model.parameters()
    ):
        param_k.data.copy_(param_q.data)  # initialize
        param_k.requires_grad = False  # not update by gradient

    for param_q, param_k in zip(
        model.buffers(), ema_model.buffers()
    ):
        param_k.data.copy_(param_q.data)  # initialize

def update_ema_model(model, ema_model, m):
    for param_q, param_k in zip(
        model.parameters(), ema_model.parameters()):
        param_k.data = param_k.data * m + param_q.data * (1.0 - m)

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()
        self.raw_val = []

    def reset(self):
        self.raw_val = []

    def update(self, val):
        self.val = val
        self.raw_val.append(val)

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        data = np.array(self.raw_val)
        return fmtstr.format(name = self.name, val = self.val, avg = data.mean())


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'
