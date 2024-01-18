import os

from PIL import Image
import numpy as np
import torch
import torch.utils.data
from torchvision import transforms as ttf
from torchvision.transforms import functional as F


class Dataset(torch.utils.data.Dataset):
    def __init__(self, img_ids, img_dir, mask_dir, img_ext, mask_ext, num_classes, mode, size):
        """
        Args:
            img_ids (list): Image ids.
            img_dir: Image file directory.
            mask_dir: Mask file directory.
            img_ext (str): Image file extension.
            mask_ext (str): Mask file extension.
            num_classes (int): Number of classes.
            transform (Compose, optional): Compose transforms of albumentations. Defaults to None.
        
        Note:
            Make sure to put the files as the following structure:
            <dataset name>
            ├── images
            |   ├── 0a7e06.jpg
            │   ├── 0aab0a.jpg
            │   ├── 0b1761.jpg
            │   ├── ...
            |
            └── masks
                ├── 0a7e06.png
                ├── 0aab0a.png
                ├── 0b1761.png
                ├── ...
        """
        self.img_ids = img_ids
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_ext = img_ext
        self.mask_ext = mask_ext
        self.num_classes = num_classes

        assert mode in ["train", "test", "val"]
        self.mode = mode
        self.size = size
        self.resize = ttf.Resize(size)
        self.normalize = ttf.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        self.totensor = ttf.ToTensor()
        self.colorjitter = ttf.ColorJitter(0.2, 0.2, 0.2, 0.1)


    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_id = ".".join(img_id.split(".")[:-1])  # remove extension
        img = Image.open(os.path.join(self.img_dir, img_id + self.img_ext)).convert("RGB")
        mask = Image.open(os.path.join(self.mask_dir, img_id.replace("RGB", "gt") + self.img_ext))

        img = self.resize(img)
        mask = self.resize(mask)
        if self.mode == "train":
            # random rotation
            rand_rot = torch.randint(0, 4, (1,)).item()
            if rand_rot:
                img = F.rotate(img, rand_rot * 90.)
                mask = F.rotate(mask, rand_rot * 90.)
            # random hflip
            rand_flip = torch.randint(0, 2, (1,)).item()
            if rand_flip:
                img = F.hflip(img)
                mask = F.hflip(mask)
            # random color jitter
            rand_color = torch.randint(0, 2, (1,)).item()
            if rand_color:
                img = self.colorjitter(img)
        img = self.normalize(self.totensor(img))
        mask = self.totensor(mask)
        
        return img, mask, {'img_id': img_id}
