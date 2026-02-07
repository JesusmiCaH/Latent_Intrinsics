import os, glob
import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
from utils.utils import compute_rank_split, parallel_load_image

class JHU_Dataset_PreLoad(data.Dataset):
    def __init__(self, root, img_transform, epoch_multiplier=10, 
                 total_split=4, split_id=0):
        # Scan for scene directories
        scene_list = sorted(glob.glob(root + '/*'))
        
        # Prepare list of image paths to load
        # Each scene has light0-light24. We exclude light0.
        # We need to structure the data so it can be accessed logically.
        # MiT_Dataset_PreLoad seems to flatly load images but organized by scene?
        # Let's see MiT logic:
        # img_list = compute_rank_split(sorted(glob.glob(root + '/*')), total_split, split_id)
        # self.images = parallel_load_image(img_list)
        # It seems MiT root contains folders of scenes, and parallel_load_image loads all images in those folders.
        
        self.scene_list = compute_rank_split(scene_list, total_split, split_id)
        
        # We need a custom data filter for parallel_load_image to exclude light0 if possible, 
        # OR we modify parallel_load_image, OR we just load everything and filter later.
        # But parallel_load_image in utils.py takes a data_filter argument which defaults to '/dir*.jpg'.
        # JHU images are likely named differently. User said "light0 image...". I'll assume naming is like 'light0.jpg' or similar.
        # To be safe, I'll pass a filter that captures all and then filter in the loader?
        # Actually parallel_load_image is in utils.py. Let's look at how to use it.
        # It takes `img_path_list` (list of folders) and `data_filter`.
        # I will handle the exclusion by passing a specific logic or valid wildcards if names allow.
        # If names are light1.jpg, light2.jpg... light24.jpg, I can try to construct a glob? 
        # Or I can just write a custom loader here since I need to exclude light0.
        
        self.images = self.load_jhu_images(self.scene_list)
        
        self.epoch_multiplier = epoch_multiplier
        self.single_img_transform = img_transform[1]
        self.group_img_transform = img_transform[0]
        print('JHU Dataset init', len(self.images), self.scene_list[0] if len(self.scene_list) > 0 else "No scenes")

    def load_jhu_images(self, scene_path_list):
        images_list = []
        import tqdm
        for scene_path in tqdm.tqdm(scene_path_list, desc="Loading JHU Dataset"):
            scene_imgs = []
            # Assuming files are named somewhat standardly, or I list all and filter.
            # User said "light0 image...". Let's assume files contain "light" string.
            # I will list all files and filter out light0.
            all_files = sorted(glob.glob(os.path.join(scene_path, '*')))
            
            # Filter for images and exclude light0
            valid_files = []
            for f in all_files:
                fname = os.path.basename(f)
                if 'light0' in fname:
                    continue
                if fname.lower().endswith(('.jpg', '.png', '.jpeg')):
                    valid_files.append(f)
            
            # Load images
            for img_path in valid_files:
                scene_imgs.append(np.array(Image.open(img_path).convert('RGB')))
            
            if len(scene_imgs) > 0:
                images_list.append(np.stack(scene_imgs))
                
        return images_list

    def __len__(self):
        # Check if we have any images
        if not self.images:
            return 0
        # MiT Logic: return len(self.images) * self.epoch_multiplier * 25
        # JHU has 25 valid lightings per scene (1-25) excluding light0.
        return len(self.images) * self.epoch_multiplier * 25

    def __getitem__(self, index):
        if len(self.images) == 0:
             raise IndexError("Dataset is empty")
             
        # JHU has 25 valid lightings per scene (1-25)
        num_lights = 25 
        
        index = index % (len(self.images) * num_lights)
        folder_idx = index // num_lights
        images = self.images[folder_idx]
        
        # images is a numpy array of shape (N_lights, H, W, C)
        # Ensure we don't go out of bounds if fewer images loaded
        real_num_lights = len(images)
        
        # Pick random index for light
        light_index = np.random.randint(real_num_lights)
        
        img1 = Image.fromarray(images[light_index])
        # Pick another light
        img2 = Image.fromarray(images[(light_index + 1 + np.random.randint(real_num_lights - 1)) % real_num_lights])

        img1 = self.single_img_transform(img1)
        img2 = self.single_img_transform(img2)

        if self.group_img_transform is not None:
             img1, img2 = self.group_img_transform([img1, img2])
        return img1, img2

class JHU_Dataset(data.Dataset):
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
