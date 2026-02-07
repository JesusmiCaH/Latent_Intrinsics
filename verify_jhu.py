import sys
import os
import argparse
from data_utils.JHU_dataset_utils import JHU_Dataset
from torchvision import transforms

def test_jhu_dataset_import():
    print("Testing JHU_Dataset import...")
    try:
        transform = [None, transforms.Compose([transforms.Resize(224), transforms.ToTensor()])]
        # Use a dummy path that exists but is empty of images
        dataset = JHU_Dataset(root="./data/jhu_dataset", img_transform=transform)
        print(f"JHU_Dataset instantiated successfully. Len: {len(dataset)}")
    except Exception as e:
        print(f"Failed to instantiate JHU_Dataset: {e}")
        # It might fail if it expects folders, but my implementation handles empty globs securely?
        # My implementation: sorted(glob.glob(root + '/*'))
        # If root='.', it will find files in current dir.
        # It will look for images inside those "folders" (files treated as folders will fail glob inside).
        # Wait, glob(root + '/*') returns files too. 
        # Inside JHU_Dataset.__init__: self.img_folder_list = sorted(glob.glob(root + '/*'))
        # Inside __getitem__: folder_path = self.img_folder_list[folder_idx]
        # _get_image_paths glob(join(folder_path, '*'))
        # If folder_path is a file, glob might return empty or fail?
        # os.path.join('file', '*') works but returns empty list usually if 'file' exists as file.
        pass

def test_main_argparse():
    print("Testing main_relight_ViT.py arguments...")
    cmd = "python main_relight_ViT.py --dataset jhu --help"
    ret = os.system(cmd)
    if ret == 0:
        print("Argument parsing check passed.")
    else:
        print("Argument parsing check failed.")

if __name__ == "__main__":
    test_jhu_dataset_import()
    # test_main_argparse() 
