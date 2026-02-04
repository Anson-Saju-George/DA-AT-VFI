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
BAD_LOG = "bad_samples.log"

# ------------------------------
# Utils
# ------------------------------
def log_bad(msg):
    with open(BAD_LOG, "a", encoding="utf-8") as f:
        f.write(msg + "\n")

def read_img(path):
    img = cv2.imread(path)
    if img is None:
        raise RuntimeError(f"Failed to read {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype(np.float32) / 255.0

def to_tensor(img):
    return torch.from_numpy(img).permute(2, 0, 1).contiguous()

def random_crop(imgs, size):
    h, w, _ = imgs[0].shape
    if h < size or w < size:
        raise RuntimeError(f"Crop {size} too large for image {h}x{w}")
    y = random.randint(0, h - size)
    x = random.randint(0, w - size)
    return [im[y:y + size, x:x + size] for im in imgs]

def sample_t():
    return torch.tensor([random.choice(TSET)], dtype=torch.float32)

def safe_get(dataset, idx):
    """Retry logic so bad samples never crash training/eval"""
    for _ in range(10):
        try:
            return dataset._getitem(idx)
        except Exception as e:
            log_bad(str(e))
            idx = random.randint(0, len(dataset) - 1)
    raise RuntimeError("Too many bad samples encountered")

# ------------------------------
# Vimeo-90K
# ------------------------------
class Vimeo90K(Dataset):
    def __init__(self, root, split="train", crop=256):
        self.root = root
        self.crop = crop
        self.samples = []

        list_file = {
            "train": "tri_trainlist.txt",
            "val":   "tri_trainlist.txt",
            "test":  "tri_testlist.txt"
        }[split]

        list_path = os.path.join(root, list_file)
        if not os.path.exists(list_path):
            raise RuntimeError(f"Missing Vimeo list: {list_path}")

        with open(list_path) as f:
            seqs = f.read().splitlines()

        # 95/5 split
        if split == "train":
            seqs = seqs[:int(0.95 * len(seqs))]
        elif split == "val":
            seqs = seqs[int(0.95 * len(seqs)):]

        base = os.path.join(root, "sequences")

        for s in seqs:
            p = os.path.join(base, s)
            f1 = os.path.join(p, "im1.png")
            f2 = os.path.join(p, "im2.png")
            f3 = os.path.join(p, "im3.png")

            if os.path.exists(f1) and os.path.exists(f2) and os.path.exists(f3):
                self.samples.append(p)
            else:
                log_bad(f"[VIMEO MISSING] {p}")

        if not self.samples:
            raise RuntimeError("VIMEO: No valid samples found")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return safe_get(self, idx)

    def _getitem(self, idx):
        p = self.samples[idx]

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
# X4K1000FPS
# ------------------------------
class X4K1000FPS(Dataset):
    def __init__(self, root, split="train", crop=256, min_frames=5):
        self.crop = crop
        self.samples = []

        base = os.path.join(root, "frames_unified", split)
        if not os.path.exists(base):
            raise RuntimeError(f"Missing: {base}")

        for seq in os.listdir(base):
            p = os.path.join(base, seq)
            if not os.path.isdir(p):
                continue

            frames = sorted(
                f for f in os.listdir(p)
                if f.lower().endswith(IMG_EXT)
            )

            if len(frames) >= min_frames:
                self.samples.append((p, frames))

        if not self.samples:
            raise RuntimeError("X4K: No valid sequences found")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return safe_get(self, idx)

    def _getitem(self, idx):
        folder, frames = self.samples[idx]

        if len(frames) < 3:
            raise RuntimeError(f"[X4K SHORT] {folder}")

        i0 = random.randint(0, len(frames) - 3)
        i1 = random.randint(i0 + 2, len(frames) - 1)
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
                os.path.join(p, f)
                for f in os.listdir(p)
                if f.lower().endswith(IMG_EXT)
            )

            for i in range(len(imgs) - 2):
                h = abs(hash(imgs[i + 1])) % 100
                triplet = imgs[i:i + 3]

                if split == "train" and h < 90:
                    self.samples.append(triplet)
                elif split == "val" and h >= 90:
                    self.samples.append(triplet)
                elif split == "test":
                    self.samples.append(triplet)

        if not self.samples:
            raise RuntimeError("WMS: No valid triplets found")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return safe_get(self, idx)

    def _getitem(self, idx):
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
