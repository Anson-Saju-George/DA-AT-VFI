import os
import cv2
import random
import torch
import numpy as np
from torch.utils.data import Dataset

# ------------------------------
# Config
# ------------------------------
IMG_EXT = (".png", ".jpg", ".jpeg")
TSET = [0.25, 0.5, 0.75]

# ------------------------------
# Utils
# ------------------------------
def read_img(path):
    img = cv2.imread(path)
    if img is None:
        raise RuntimeError(f"Failed to read {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype(np.float32) / 255.0

def to_tensor(img):
    return torch.from_numpy(img).permute(2, 0, 1)

def random_crop(imgs, size):
    h, w, _ = imgs[0].shape
    if h < size or w < size:
        raise RuntimeError("Crop size larger than image")
    y = random.randint(0, h - size)
    x = random.randint(0, w - size)
    return [im[y:y+size, x:x+size] for im in imgs]

def sample_t():
    return torch.tensor([random.choice(TSET)], dtype=torch.float32)

# ------------------------------
# Vimeo-90K
# ------------------------------
class Vimeo90K(Dataset):
    def __init__(self, root, split="train", crop=256):
        self.root = root
        self.crop = crop

        list_file = {
            "train": "tri_trainlist.txt",
            "val":   "tri_trainlist.txt",
            "test":  "tri_testlist.txt"
        }[split]

        with open(os.path.join(root, list_file)) as f:
            seqs = f.read().splitlines()

        # 95/5 split for train/val
        if split == "train":
            self.samples = seqs[:int(0.95 * len(seqs))]
        elif split == "val":
            self.samples = seqs[int(0.95 * len(seqs)):]
        else:
            self.samples = seqs

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        p = os.path.join(self.root, "sequences", self.samples[idx])

        I0 = read_img(os.path.join(p, "im1.png"))
        It = read_img(os.path.join(p, "im2.png"))
        I1 = read_img(os.path.join(p, "im3.png"))

        I0, It, I1 = random_crop([I0, It, I1], self.crop)

        return {
            "I0": to_tensor(I0),
            "I1": to_tensor(I1),
            "It": to_tensor(It),
            "t": sample_t()
        }

# ------------------------------
# X4K1000FPS (frames_unified)
# ------------------------------
class X4K1000FPS(Dataset):
    def __init__(self, root, split="train", crop=256, min_frames=3):
        self.crop = crop
        self.samples = []

        base = os.path.join(root, "frames_unified", split)
        if not os.path.exists(base):
            raise RuntimeError(f"Missing: {base}")

        for seq in os.listdir(base):
            p = os.path.join(base, seq)
            if not os.path.isdir(p):
                continue
            frames = [f for f in os.listdir(p) if f.lower().endswith(IMG_EXT)]
            if len(frames) >= min_frames:
                self.samples.append(p)

        if not self.samples:
            raise RuntimeError("X4K: no valid sequences found")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        folder = self.samples[idx]
        frames = sorted([f for f in os.listdir(folder) if f.lower().endswith(IMG_EXT)])

        i0, i1 = sorted(random.sample(range(len(frames)), 2))
        if i1 - i0 < 2:
            i0 = max(0, i0 - 1)
            i1 = min(len(frames) - 1, i1 + 1)

        t_idx = random.randint(i0 + 1, i1 - 1)

        I0 = read_img(os.path.join(folder, frames[i0]))
        It = read_img(os.path.join(folder, frames[t_idx]))
        I1 = read_img(os.path.join(folder, frames[i1]))

        I0, It, I1 = random_crop([I0, It, I1], self.crop)

        return {
            "I0": to_tensor(I0),
            "I1": to_tensor(I1),
            "It": to_tensor(It),
            "t": sample_t()
        }

# ------------------------------
# WMS
# ------------------------------
class WMS(Dataset):
    def __init__(self, root, split="train", crop=256):
        self.crop = crop
        self.samples = []

        for folder in os.listdir(root):
            p = os.path.join(root, folder)
            if not os.path.isdir(p):
                continue

            imgs = sorted(
                [os.path.join(p, f) for f in os.listdir(p) if f.lower().endswith(IMG_EXT)]
            )

            for i in range(len(imgs) - 2):
                h = abs(hash(imgs[i + 1])) % 100
                if split == "train" and h < 90:
                    self.samples.append(imgs[i:i + 3])
                elif split == "val" and h >= 90:
                    self.samples.append(imgs[i:i + 3])
                elif split == "test":
                    self.samples.append(imgs[i:i + 3])

        if not self.samples:
            raise RuntimeError("WMS: no valid triplets found")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        p0, pt, p1 = self.samples[idx]

        I0 = read_img(p0)
        It = read_img(pt)
        I1 = read_img(p1)

        I0, It, I1 = random_crop([I0, It, I1], self.crop)

        return {
            "I0": to_tensor(I0),
            "I1": to_tensor(I1),
            "It": to_tensor(It),
            "t": sample_t()
        }

# ------------------------------
# Factory
# ------------------------------
def get_dataset(name, root, split="train", crop=256):
    name = name.lower()
    if name == "vimeo":
        return Vimeo90K(root, split, crop)
    if name == "x4k":
        return X4K1000FPS(root, split, crop)
    if name == "wms":
        return WMS(root, split, crop)
    raise ValueError(f"Unknown dataset: {name}")
