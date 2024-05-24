from __future__ import print_function

import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from ..registry import DATASETS
from mae_lite.utils import get_root_dir

def list_files(dataset_path):
    print("Listing files in:", dataset_path)
    images = []
    for root, _, files in sorted(os.walk(dataset_path)):
        for name in sorted(files):
            if name.lower().endswith('.tif'):
                images.append(os.path.join(root, name))
    print(f"Found {len(images)} .tif files.")
    return images

@DATASETS.register()
class Fundus(Dataset):
    """Custom dataset class for fundus images in PyTorch."""
    def __init__(self, root=None, transform=None):
        self.img_dir = '/data/fundus'

        self.images = list_files(self.img_dir)
        self.transform = transform if transform else transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image

if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    dataset = Fundus(transform=transform)
    print(f"Number of images: {len(dataset)}")
