import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np

class Data (Dataset):

    def __init__(self, image_directory, mask_directory, transform=None):
        self.image_directory = image_directory
        self.mask_directory = mask_directory
        self.transform = transform
        self.images = [f for f in os.listdir(image_directory) if f.endswith('.jpg')]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_directory, self.images[idx])
        mask_path = os.path.join(self.mask_directory, self.images[idx].replace(".jpg", "_mask.gif"))
        image = np.array(Image.open(image_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        mask[mask == 255.0] = 1.0

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        return image, mask

