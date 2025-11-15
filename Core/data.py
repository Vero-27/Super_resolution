# Core/data.py
from pathlib import Path
import random
import math
from PIL import Image
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

def pil_to_cv(img_pil):
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def cv_to_pil(img_cv):
    return Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))

def downsample_bicubic(hr_img_pil: Image.Image, scale: int) -> Image.Image:
    w, h = hr_img_pil.size
    lr = hr_img_pil.resize((w // scale, h // scale), Image.BICUBIC)
    return lr

class SRDataset(Dataset):
    """
    Attend un dossier avec des images en haute résolution.
    On génère les LR on-the-fly par downscaling bicubique.
    Optionnel: petites augmentations pour la robustesse.
    """
    def __init__(self, root_dir, scale=2, patch_size=96, augment=True, ext=(".png",".jpg",".jpeg")):
        self.paths = [p for p in Path(root_dir).rglob("*") if p.suffix.lower() in ext]
        self.scale = scale
        self.patch_size = patch_size
        self.augment = augment

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        hr = Image.open(self.paths[idx]).convert("RGB")
        # assure une taille multiple du scale
        w, h = hr.size
        w -= w % self.scale
        h -= h % self.scale
        hr = hr.crop((0, 0, w, h))
        lr = downsample_bicubic(hr, self.scale)

        # crop patchs alignés
        if self.patch_size:
            ps = self.patch_size
            x = random.randrange(0, lr.width - ps + 1)
            y = random.randrange(0, lr.height - ps + 1)
            lr = lr.crop((x, y, x+ps, y+ps))
            hr = hr.crop((x*self.scale, y*self.scale, (x+ps)*self.scale, (y+ps)*self.scale))

        # augment léger
        if self.augment and random.random() < 0.5:
            lr = TF.hflip(lr); hr = TF.hflip(hr)
        if self.augment and random.random() < 0.5:
            lr = TF.vflip(lr); hr = TF.vflip(hr)

        lr_t = TF.to_tensor(lr)           # [0,1]
        hr_t = TF.to_tensor(hr)

        return lr_t, hr_t
