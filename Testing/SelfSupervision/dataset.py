
from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms


class BirdDataset(Dataset):
    def __init__(self, image_dir, image_size=256):
        self.files = sorted(Path(image_dir).glob("*"))

        self.transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx])

        img = self.transform(img)

        return img