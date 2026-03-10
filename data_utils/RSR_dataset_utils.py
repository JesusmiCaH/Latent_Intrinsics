import os, glob
import numpy as np
import torch
import torch.distributed as dist
import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms
from collections import defaultdict

from utils.utils import compute_rank_split

class RSR_Trainset(data.Dataset):
    def __init__(self, root, img_transform, epoch_multiplier=10, total_split=1, split_id=0):
        self.epoch_multiplier = epoch_multiplier
        self.single_img_transform = img_transform[1]
        self.group_img_transform = img_transform[0]
        
        # Get all scene folders (e.g., scene_01, scene_02)
        self.folder_list = sorted(glob.glob(os.path.join(root, '*')))
        
        # Exclude random files like zip archives if they exist in the root
        self.folder_list = [f for f in self.folder_list if os.path.isdir(f)]

        # Distributed split
        self.folder_list = compute_rank_split(self.folder_list, total_split, split_id)
        
        # Pre-parse each folder to group images by camera pose
        self.pose_groups = [] # list of lists, where each sublist contains images from the same camera pose
        
        for folder_path in self.folder_list:
            images = sorted(glob.glob(os.path.join(folder_path, '*.jpg')))
            
            # Map camera_pose_id -> list of image paths
            pose_map = defaultdict(list)
            for img_path in images:
                filename = os.path.basename(img_path)
                # Formats like 00001_0001_0_0_... -> index 1 is camera pose
                parts = filename.split('_')
                if len(parts) >= 2:
                    camera_pose_id = parts[1]
                    pose_map[camera_pose_id].append(img_path)
            
            # Only keep poses that have enough images to sample a pair (>=2)
            for pose_id, img_list in pose_map.items():
                if len(img_list) >= 2:
                    self.pose_groups.append(img_list)

        print(f'RSR_Trainset Init: {len(self.folder_list)} scenes, {len(self.pose_groups)} valid camera poses, split {split_id}/{total_split}')

    def __len__(self):
        # We have 576 pose groups. 576 * 10 * 4 = 23040. Wait, each pose group has 36 images.
        # The user said 72 * 288 * 10 = 207360 images.
        # Since every pose group has 36 images, picking a pair for relighting gives a large combination.
        # Let's match the number of iterations to the logic 72 * 288 * 10 = len(self.folder_list) * 288 * self.epoch_multiplier
        return len(self.folder_list) * 288 * self.epoch_multiplier

    def __getitem__(self, index):
        # Map the index to a specific pose group randomly or deterministically
        pose_index = index % len(self.pose_groups)        
        img_list = self.pose_groups[pose_index]
        
        # Select two different random images from the same camera pose
        idx1, idx2 = np.random.choice(len(img_list), 2, replace=False)
        
        # Load images
        img1 = Image.open(img_list[idx1])
        img2 = Image.open(img_list[idx2])

        # Transforms
        if self.single_img_transform is not None:
            img1 = self.single_img_transform(img1)
            img2 = self.single_img_transform(img2)

        if self.group_img_transform is not None:
            img1, img2 = self.group_img_transform([img1, img2])
            
        return img1, img2
