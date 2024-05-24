from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import os
from __future__ import print_function
from PIL import Image

import numpy as np
import os
import os.path
import scipy.io

from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.utils import download_url
from torchvision.datasets.utils import extract_archive
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
    """The above class is a custom dataset class for images in PyTorch."""
    def __init__(self):
        self.img_dir = '/cnvrg/fundus'
        self.images = list_files(self.img_dir)
        self.transform =  transforms.Compose([
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






