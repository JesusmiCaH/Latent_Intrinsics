import os, glob
import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
from utils.utils import compute_rank_split, parallel_load_image


class JHU_Trainset(data.Dataset):
    def __init__(self, root, img_transform, epoch_multiplier=10, total_split=1, split_id=0):
        self.epoch_multiplier = epoch_multiplier
        self.single_img_transform = img_transform[1]
        self.group_img_transform = img_transform[0]
        
        # Get all scene folders
        self.folder_list = sorted(glob.glob(root + '/*'))
        # Distributed split
        self.folder_list = compute_rank_split(self.folder_list, total_split, split_id)
        
        print(f'JHU Trainset Init: {len(self.folder_list)} scenes, split {split_id}/{total_split}')

    def __len__(self):
        # 25 lights per scene (light1...light25), light0 excluded
        return len(self.folder_list) * self.epoch_multiplier * 25

    def __getitem__(self, index):
        index = index % (len(self.folder_list) * 25)
        folder_idx = index // 25
        folder_path = self.folder_list[folder_idx]
        
        # Select two different random lights from 1 to 25
        # light_idx is 0..24
        # We need to map 0 -> light1, 24 -> light25
        light_idx1 = np.random.randint(25)
        light_idx2 = (light_idx1 + 1 + np.random.randint(24)) % 25
        
        # Map 0-24 to 1-25
        real_light_idx1 = light_idx1 + 1
        real_light_idx2 = light_idx2 + 1
        
        # JHU files are like light1.png ... light25.png
        # Some might be jpg? Check said jpg/png. 
        # Ref says "light0.jpg", others ".png".
        # We'll try png first, then jpg, or glob?
        # Creating path directly is faster than glob
        
        def load_light_img(path, idx):
            # Try png first as observed in ls output
            p = os.path.join(path, f'light{idx}.png')
            if os.path.exists(p):
                return Image.open(p).convert('RGB')
            # Try jpg
            p = os.path.join(path, f'light{idx}.jpg')
            if os.path.exists(p):
                return Image.open(p).convert('RGB')
            raise FileNotFoundError(f"Could not find light{idx} in {path}")

        img1 = load_light_img(folder_path, real_light_idx1)
        img2 = load_light_img(folder_path, real_light_idx2)

        # Transforms
        img1 = self.single_img_transform(img1)
        img2 = self.single_img_transform(img2)

        if self.group_img_transform is not None:
             img1, img2 = self.group_img_transform([img1, img2])
        return img1, img2

class JHU_Valset(data.Dataset):
    def __init__(self, root, img_transform, epoch_multiplier = 10,
                    eval_mode = False):
        self.epoch_multiplier = epoch_multiplier
        self.root = root
        
        # Validate and filter folders
        self.img_folder_list = []
        for folder_path in sorted(glob.glob(root + '/*')):
            if self._has_valid_images(folder_path):
                self.img_folder_list.append(folder_path)
        
        self.img_folder_list.sort() # Ensure sorted
        
        # JHU dataset has 25 valid lights (light1 to light25, light0 excluded)
        # We will assume each folder has these.
        self.num_lights = 25 
        self.img_idx = np.arange(self.num_lights).tolist()
        
        self.eval_mode = eval_mode
        self.eval_pair_light_shift = np.random.randint(1, self.num_lights, (len(self.img_folder_list) * self.num_lights))
        self.eval_pair_folder_shift = np.random.randint(1, len(self.img_folder_list), (len(self.img_folder_list) * self.num_lights))
        
        self.single_img_transform = img_transform[1]
        self.group_img_transform = img_transform[0]
        
        # Cache image paths to avoid re-globbing
        self.cache_images = {} # folder_path -> list of valid image paths
        print('🍎 Valid JHU scenes:', len(self.img_folder_list))

    def _has_valid_images(self, folder_path):
        # Quick check without caching full list if possible, or just use _get_image_paths and discard result?
        # Better to just reuse _get_image_paths logic but we haven't inited cache yet.
        # Let's just do the check.
        all_files = glob.glob(os.path.join(folder_path, '*'))
        for f in all_files:
            fname = os.path.basename(f)
            if 'light0' in fname: continue
            if fname.lower().endswith(('.jpg', '.png', '.jpeg')):
                return True
        print(f"⚠️ Warning: No valid images in {folder_path} (excluding light0)")
        return False

    def _get_image_paths(self, folder_path):
        if folder_path in self.cache_images:
            return self.cache_images[folder_path]
            
        all_files = sorted(glob.glob(os.path.join(folder_path, '*')))
        valid_files = []
        for f in all_files:
            fname = os.path.basename(f)
            # Exclude light0
            if 'light0' in fname:
                continue
            if fname.lower().endswith(('.jpg', '.png', '.jpeg')):
                valid_files.append(f)
        
        # If we expect exactly 25 images, we might want to ensure sorting is correct (light1, light2... light25)
        # Just alphabetical sort might give light1, light10, ...
        # Let's try to sort by number if possible, or just trust stable sorting if filenames are consistent.
        # Assuming format like "light1.jpg", "light10.jpg"
        # We can extract numbers.
        try:
            valid_files.sort(key=lambda f: int(''.join(filter(str.isdigit, os.path.basename(f)))))
        except:
            pass # Fallback to alphabetical if fails
            
        self.cache_images[folder_path] = valid_files
        return valid_files

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

        # Current light index (logical index 0-24)
        folder_offset = index % len(self.img_idx)
        
        if self.eval_mode:
            pair_img_folder_offset = (folder_offset + self.eval_pair_light_shift[index]) % len(self.img_idx)
        else:
            pair_img_folder_offset = np.random.choice(np.where(np.arange(len(self.img_idx)) != folder_offset)[0])
            
        # Get actual file paths
        imgs_current = self._get_image_paths(folder_path)
        imgs_ref = self._get_image_paths(ref_folder_path)
        
        # Should not happen if we filtered correctly, but safety first
        if not imgs_current: 
            # Fallback to ref or error? 
            # If we filtered correctly, this branch is unreachable unless file system changed.
            # Return gray if it happens.
            return torch.zeros(3, 224, 224), torch.zeros(3, 224, 224), torch.zeros(3, 224, 224)
        
        idx1 = folder_offset % len(imgs_current)
        idx2 = pair_img_folder_offset % len(imgs_current)
        idx3 = pair_img_folder_offset % len(imgs_ref) if imgs_ref else 0
        
        img1_path = imgs_current[idx1]
        img2_path = imgs_current[idx2]
        img3_path = imgs_ref[idx3] if imgs_ref else img1_path # Fallback
        
        img1 = Image.open(img1_path).convert('RGB')
        img2 = Image.open(img2_path).convert('RGB')
        img3 = Image.open(img3_path).convert('RGB')

        img1 = self.single_img_transform(img1)
        img2 = self.single_img_transform(img2)
        img3 = self.single_img_transform(img3)

        if self.group_img_transform is not None:
            img1, img2, img3 = self.group_img_transform([img1, img2, img3])
        return img1, img2, img3
