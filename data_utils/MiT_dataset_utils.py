import os,glob
import numpy as np
import torch
import torch.distributed as dist
import torch.utils.data as data
from PIL import Image
import torchvision
import torchvision.transforms as transforms

import json
import torch.nn.functional as F
from utils.utils import compute_rank_split, parallel_load_image

class MIT_Trainset(data.Dataset):
    def __init__(self, root, img_transform, epoch_multiplier=10, total_split=1, split_id=0):
        self.epoch_multiplier = epoch_multiplier
        self.single_img_transform = img_transform[1]
        self.group_img_transform = img_transform[0]
        
        # Get all scene folders
        self.folder_list = sorted(glob.glob(root + '/*'))
        # Distributed split
        self.folder_list = compute_rank_split(self.folder_list, total_split, split_id)
        
        print(f'Online Dataset Init: {len(self.folder_list)} scenes, split {split_id}/{total_split}')

    def __len__(self):
        # Same length calculation as PreLoad
        return len(self.folder_list) * self.epoch_multiplier * 25

    def __getitem__(self, index):
        # Same index logic
        index = index % (len(self.folder_list) * 25)
        folder_idx = index // 25
        folder_path = self.folder_list[folder_idx]
        
        # Select two different random lights
        light_idx1 = np.random.randint(25)
        light_idx2 = (light_idx1 + 1 + np.random.randint(24)) % 25
        
        # Load images on the fly
        # Using specific filename format dir_{}_mip2.jpg which matches the MIT dataset convention
        try:
            img1 = Image.open(os.path.join(folder_path, f'dir_{light_idx1}_mip2.jpg'))
            img2 = Image.open(os.path.join(folder_path, f'dir_{light_idx2}_mip2.jpg'))
        except FileNotFoundError:
            # Fallback if filenames are different (e.g. random glob order in PreLoad)
            # But based on ls check, they form dir_{}_mip2.jpg
            # Just in case, pick random files from directory if specific ones fail?
            # For now, assume format is consistent as validated by existing MIT_Dataset class
            raise

        # Transforms
        img1 = self.single_img_transform(img1)
        img2 = self.single_img_transform(img2)

        if self.group_img_transform is not None:
            img1, img2 = self.group_img_transform([img1, img2])
            
        return img1, img2

class MIT_Valset(data.Dataset):
    def __init__(self, root, img_transform, epoch_multiplier = 10,
                    eval_mode = False):
        self.epoch_multiplier = epoch_multiplier
        img_folder_list = glob.glob(root + '/*')
        img_folder_list.sort()
        self.img_folder_list = img_folder_list
        self.img_idx = np.arange(25).tolist()
        self.eval_mode = eval_mode
        self.eval_pair_light_shift = np.random.randint(1, 25, (len(img_folder_list) * 25))
        self.eval_pair_folder_shift = np.random.randint(1, len(img_folder_list), (len(img_folder_list) * 25))
        self.single_img_transform = img_transform[1]
        self.group_img_transform = img_transform[0]
        print('🍎', len(self.img_folder_list))

    def __len__(self):
        return len(self.img_folder_list) * self.epoch_multiplier * len(self.img_idx)

    def __getitem__(self, index):
        index = index % (len(self.img_folder_list) * len(self.img_idx))
        folder_idx = index // len(self.img_idx)
        folder_path = self.img_folder_list[folder_idx]
        if self.eval_mode:
            ref_folder_path = self.img_folder_list[(folder_idx + self.eval_pair_folder_shift[index]) % len(self.img_folder_list)]
        else:
            ref_folder_path = self.img_folder_list[np.random.randint(len(self.img_folder_list))]

        folder_offset = index % len(self.img_idx)
        if self.eval_mode:
            pair_img_folder_offset = (folder_offset + self.eval_pair_light_shift[index]) % len(self.img_idx)
        else:
            pair_img_folder_offset = np.random.choice(np.where(np.arange(len(self.img_idx)) != folder_offset)[0])
        folder_offset = self.img_idx[folder_offset]
        pair_img_folder_offset = self.img_idx[pair_img_folder_offset]

        img1 = Image.open(f'{folder_path}/dir_{folder_offset}_mip2.jpg')
        img2 = Image.open(f'{folder_path}/dir_{pair_img_folder_offset}_mip2.jpg')
        img3 = Image.open(f'{ref_folder_path}/dir_{pair_img_folder_offset}_mip2.jpg')

        img1 = self.single_img_transform(img1)
        img2 = self.single_img_transform(img2)
        img3 = self.single_img_transform(img3)

        if self.group_img_transform is not None:
            img1, img2, img3 = self.group_img_transform([img1, img2, img3])
        return img1, img2, img3

class MIT_Dataset_show(data.Dataset):
    def __init__(self, root, img_transform, epoch_multiplier = 10,
                    eval_mode = False, return_seg_map = False, seg_transform = None,
                    eval_pair_folder_shift = 5,
                    eval_pair_light_shift = 5,
                    ):
        self.return_seg_map = return_seg_map
        self.seg_transform = seg_transform
        self.epoch_multiplier = epoch_multiplier
        img_folder_list = glob.glob(root + '/*')
        img_folder_list.sort()
        self.img_folder_list = img_folder_list
        #with open('train.txt' if train else 'val.txt', 'w') as f:
        #    for name in sub_img_folder_list:
        #        f.write(name.split('/')[-1] + '\n')
        #pdb.set_trace()
        train_idx = [idx for idx in np.arange(25).tolist()]
        #self.img_idx = train_idx
        #self.img_idx = train_idx if train else test_idx
        self.img_idx = train_idx
        self.eval_mode = eval_mode
        self.eval_pair_light_shift = np.random.randint(1, 25, (len(img_folder_list) * 25))
        self.eval_pair_folder_shift = np.random.randint(1, len(img_folder_list), (len(img_folder_list) * 25))
        self.single_img_transform = img_transform[1]
        self.group_img_transform = img_transform[0]
        print('init')

    def __len__(self):
        return len(self.img_folder_list) * self.epoch_multiplier * len(self.img_idx)

    def get_img(self, img_folder_idx, img_idx):
        folder_path = self.img_folder_list[img_folder_idx]
        img = Image.open(f'{folder_path}/dir_{img_idx}_mip2.jpg')
        img = self.single_img_transform(self.group_img_transform(img))

        json_data = json.load(open(f'{folder_path}/meta.json'))
        mask = np.zeros((1000, 1500))
        bbox = json_data['chrome']['bounding_box']
        bbox = np.array([bbox['x']/ 4, bbox['y'] / 4, bbox['w'] / 4, bbox['h'] / 4]).astype(np.int32)
        mask[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]] = 1
        coor_y, coor_x= np.where(np.array(self.group_img_transform(Image.fromarray(mask))) > 0)
        box_y_min = coor_y.min()
        box_y_max = coor_y.max()
        box_x_min = coor_x.min()
        box_x_max = coor_x.max()
        box_w = box_x_max - box_x_min + 1
        box_h = box_y_max - box_y_min + 1
        box_size = max(box_h, box_w)
        def compute_new_coor(box_size, img_size, coor_min, coor_max):
            aug_size = box_size - (coor_max - coor_min + 1)
            aug_edge = np.ceil(aug_size * 0.5)
            new_coor_min = coor_min - aug_edge
            new_coor_max = coor_max + aug_edge
            if new_coor_min < 0:
                new_coor_max += (-1 * new_coor_min)
                new_coor_min = 0
            if new_coor_max >= img_size:
                new_coor_min -= (new_coor_max - img_size + 1)
                new_coor_max = img_size - 1
            return int(new_coor_min), int(new_coor_max)
        box_x_min, box_x_max = compute_new_coor(box_size, 256, box_x_min, box_x_max)
        box_y_min, box_y_max = compute_new_coor(box_size, 256, box_y_min, box_y_max)
        img_box = img[:,box_y_min:box_y_max + 1, box_x_min:box_x_max+1]
        return img, F.interpolate(img_box[None,:], size = (256, 256), mode = 'bilinear')[0], [box_y_min, box_y_max, box_x_min, box_x_max]

    def __getitem__(self, index):
        index = index % (len(self.img_folder_list) * len(self.img_idx))
        folder_idx = index // len(self.img_idx)
        folder_path = self.img_folder_list[folder_idx]
        #if self.eval_mode:
        #    ref_folder_path = self.img_folder_list[(folder_idx + self.eval_pair_folder_shift[index]) % len(self.img_folder_list)]
        #else:
        ref_folder_idx = (folder_idx + np.random.randint(len(self.img_folder_list) - 1) + 1) % len(self.img_folder_list)
        ref_folder_path = self.img_folder_list[ref_folder_idx]

        folder_offset = index % len(self.img_idx)
        #if self.eval_mode:
        #    pair_img_folder_offset = (folder_offset + self.eval_pair_light_shift[index]) % len(self.img_idx)
        #else:
        pair_img_folder_offset = np.random.choice(np.where(np.arange(len(self.img_idx)) != folder_offset)[0])
        folder_offset = self.img_idx[folder_offset]
        pair_img_folder_offset = self.img_idx[pair_img_folder_offset]

        folder_name = folder_path.split('/')[-1]

        img1 = Image.open(f'{folder_path}/dir_{folder_offset}_mip2.jpg')
        img2 = Image.open(f'{folder_path}/dir_{pair_img_folder_offset}_mip2.jpg')
        img3 = Image.open(f'{ref_folder_path}/dir_{pair_img_folder_offset}_mip2.jpg')

        img1 = self.single_img_transform(self.group_img_transform(img1))
        img2 = self.single_img_transform(self.group_img_transform(img2))
        img3 = self.single_img_transform(self.group_img_transform(img3))

        return img1, img2, img3, folder_offset, pair_img_folder_offset, folder_idx, ref_folder_idx

class MIT_Dataset_Normal(data.Dataset):
    def __init__(self, root,  group_img_transform = None, epoch_multiplier = 1):
        self.group_img_transform = group_img_transform
        self.surface_normal = '/net/projects/willettlab/roxie62/dataset/mit_multiview_normal_omi'
        self.epoch_multiplier = epoch_multiplier
        img_folder_list = glob.glob(root + '/*')
        img_folder_list.sort()
        self.img_folder_list = img_folder_list
        #with open('train.txt' if train else 'val.txt', 'w') as f:
        #    for name in sub_img_folder_list:
        #        f.write(name.split('/')[-1] + '\n')
        #pdb.set_trace()
        train_idx = [idx for idx in np.arange(25).tolist()]
        self.img_idx = train_idx
        #self.img_idx = train_idx if train else test_idx
        print('init')

    def __len__(self):
        return len(self.img_folder_list) * self.epoch_multiplier * len(self.img_idx)

    def __getitem__(self, index):
        index = index % (len(self.img_folder_list) * len(self.img_idx))
        folder_idx = index // len(self.img_idx)
        folder_path = self.img_folder_list[folder_idx]
        folder_offset = index % len(self.img_idx)

        folder_name = folder_path.split('/')[-1]

        img = Image.open(f'{folder_path}/dir_{folder_offset}_mip2.jpg')
        #img = surface_normal_img_transform(img)
        normal = torch.from_numpy(np.array(Image.open(self.surface_normal + f'/{folder_name}.png'))).permute(2,0,1)[None,...].float()
        normal = normal / 255 * 2 - 1
        #normal = np.load(self.surface_normal + f'/{folder_name}.npy')
        #img_pil = img.resize((normal.shape[1], normal.shape[1]))
#
        #fig, ax = plt.subplots(2)
        #ax[0].imshow(img_pil)
        #ax[1].imshow(((normal * 0.5 + 0.5) * 255).astype(np.uint8))
        #plt.savefig('img.png')
        #pdb.set_trace()

        img = np.array(img) * 1.0 / 255 * 2 - 1
        img = torch.from_numpy(img)
        img = img.permute(2,0,1)[None,...].float()
        #normal = torch.from_numpy(normal).permute(2,0,1)[None,...]
        normal = F.normalize(torchvision.transforms.Resize(256)(normal), dim = 1)[0]
        img = torchvision.transforms.CenterCrop(256)(torchvision.transforms.Resize(256)(img))[0]
        if self.group_img_transform is not None:
            img, normal = self.group_img_transform([img, normal])
        #transforms.ToPILImage()(normal * 0.5 + 0.5).save('img1.png')
        #transforms.ToPILImage()(img * 0.5 + 0.5).save('img2.png')
        return img, normal, folder_idx, folder_offset

#train_dataset = MIT_Dataset_Normal('/net/projects/willettlab/roxie62/dataset/mit_multiview_resize', 5, [0,1,2,3], train = True)
#train_dataset[0]

class MIT_Dataset_sequence(data.Dataset):
    def __init__(self, root, img_transform):
        img_folder_list = glob.glob(root + '/*')
        img_folder_list.sort()
        self.img_meta = {'img_root':root, 'img_folder_list':[], 'img_folder_img_num':[], 'img_num_cumsum':[]}
        for img_folder_path in img_folder_list:
            img_list = glob.glob(img_folder_path + '/dir*.jpg')
            self.img_meta['img_folder_list'].append(img_folder_path.split('/')[-1])
            self.img_meta['img_folder_img_num'].append(len(img_list))
        self.img_meta['img_num_cumsum'] = np.cumsum(np.array(self.img_meta['img_folder_img_num']))
        self.img_transform = img_transform

    def __len__(self):
        return self.img_meta['img_num_cumsum'][-1]

    def __getitem__(self, index):
        folder_index = np.searchsorted(self.img_meta['img_num_cumsum'], index, side = 'right')
        if folder_index == 0:
            folder_offset = index
        else:
            folder_offset = index - self.img_meta['img_num_cumsum'][folder_index - 1]
        img_list = glob.glob(self.img_meta['img_root'] + '/' + self.img_meta['img_folder_list'][folder_index] + '/dir*.jpg')
        img_list.sort()
        pair_img_folder_offset = np.random.choice(np.where(np.arange(len(img_list)) != folder_offset)[0])
        img1 = Image.open(img_list[folder_index])
        img2 = Image.open(img_list[pair_img_folder_offset])
        return self.img_transform(img1), self.img_transform(img2)

    def get_img_folder_list(self, folder_index):
        #folder_index = np.searchsorted(self.img_meta['img_num_cumsum'], index, side = 'right')
        img_list = glob.glob(self.img_meta['img_root'] + '/' + self.img_meta['img_folder_list'][folder_index] + '/dir*.jpg')
        img_list.sort()
        img_tensor_list = []
        for img_path in img_list:
            img = Image.open(img_path)
            img_tensor_list.append(self.img_transform(img))
        return img_tensor_list