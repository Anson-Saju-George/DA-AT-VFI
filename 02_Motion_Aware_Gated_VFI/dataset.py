import os
import cv2
import random
import torch
import numpy as np
from torch.utils.data import Dataset

IMG_EXT = (".png", ".jpg", ".jpeg")
TSET = [0.25, 0.5, 0.75]

# -----------------------------
def read_img(path):
    img = cv2.imread(path)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype(np.float32) / 255.0

def to_tensor(img):
    return torch.from_numpy(img).permute(2, 0, 1)

def random_crop(imgs, size):
    h, w, _ = imgs[0].shape
    if h < size or w < size:
        return None
    y = random.randint(0, h - size)
    x = random.randint(0, w - size)
    return [im[y:y+size, x:x+size] for im in imgs]

def sample_t():
    return torch.tensor([random.choice(TSET)], dtype=torch.float32)

# -----------------------------
# Vimeo
# -----------------------------
class Vimeo90K(Dataset):
    def __init__(self, root, crop=256):
        self.root = root
        self.crop = crop

        with open(os.path.join(root, "tri_trainlist.txt")) as f:
            self.samples = f.read().splitlines()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        for _ in range(10):
            p = os.path.join(self.root, "sequences", self.samples[idx])

            I0 = read_img(os.path.join(p, "im1.png"))
            It = read_img(os.path.join(p, "im2.png"))
            I1 = read_img(os.path.join(p, "im3.png"))

            if I0 is None or It is None or I1 is None:
                idx = random.randint(0, len(self.samples) - 1)
                continue

            crop = random_crop([I0, It, I1], self.crop)
            if crop is None:
                idx = random.randint(0, len(self.samples) - 1)
                continue

            I0, It, I1 = crop
            return {
                "I0": to_tensor(I0),
                "I1": to_tensor(I1),
                "It": to_tensor(It),
                "t": sample_t()
            }

        raise RuntimeError("Vimeo failed too many times")

# -----------------------------
# X4K
# -----------------------------
class X4K1000FPS(Dataset):
    def __init__(self, root, crop=256):
        self.crop = crop
        self.samples = []

        base = os.path.join(root, "frames_unified", "train")
        for seq in os.listdir(base):
            p = os.path.join(base, seq)
            if not os.path.isdir(p):
                continue
            frames = [f for f in os.listdir(p) if f.lower().endswith(IMG_EXT)]
            if len(frames) >= 3:
                self.samples.append(p)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        for _ in range(10):
            folder = self.samples[idx]
            frames = sorted([f for f in os.listdir(folder) if f.lower().endswith(IMG_EXT)])

            i0, i1 = sorted(random.sample(range(len(frames)), 2))
            if i1 - i0 < 2:
                continue

            t_idx = random.randint(i0 + 1, i1 - 1)

            I0 = read_img(os.path.join(folder, frames[i0]))
            It = read_img(os.path.join(folder, frames[t_idx]))
            I1 = read_img(os.path.join(folder, frames[i1]))

            if I0 is None or It is None or I1 is None:
                idx = random.randint(0, len(self.samples) - 1)
                continue

            crop = random_crop([I0, It, I1], self.crop)
            if crop is None:
                continue

            I0, It, I1 = crop
            return {
                "I0": to_tensor(I0),
                "I1": to_tensor(I1),
                "It": to_tensor(It),
                "t": sample_t()
            }

        raise RuntimeError("X4K failed too many times")

# -----------------------------
# Factory
# -----------------------------
def get_dataset(name, root, crop=256):
    name = name.lower()
    if name == "vimeo":
        return Vimeo90K(root, crop)
    if name == "x4k":
        return X4K1000FPS(root, crop)
    raise ValueError(name)
