from data_utils.JHU_dataset_utils import JHU_Dataset
from torchvision import transforms

def test_jhu_dataset():
    print("Testing JHU_Dataset init with filtering...")
    transform = [None, transforms.Compose([transforms.Resize(224), transforms.ToTensor()])]
    # Use real path
    dataset = JHU_Dataset(root="data/jhu_dataset", img_transform=transform)
    print(f"JHU_Dataset instantiated. Len: {len(dataset)}")

if __name__ == "__main__":
    test_jhu_dataset()
