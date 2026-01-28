
import os
import cv2
import random
import torch
import numpy as np
from torch.utils.data import Dataset

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
        # center crop fallback
        y = max(0, (h - size) // 2)
        x = max(0, (w - size) // 2)
        return [im[y:y+size, x:x+size] for im in imgs]

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
        self.samples = []

        list_file = {
            "train": "tri_trainlist.txt",
            "val": "tri_trainlist.txt",
            "test": "tri_testlist.txt"
        }[split]

        with open(os.path.join(root, list_file)) as f:
            seqs = [l.strip() for l in f if l.strip()]

        for s in seqs:
            p = os.path.join(root, "sequences", s)
            if all(os.path.exists(os.path.join(p, f"im{i}.png")) for i in [1,2,3]):
                self.samples.append(s)

        if not self.samples:
            raise RuntimeError("No valid Vimeo samples found")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        for _ in range(5):  # retry logic
            try:
                s = self.samples[idx]
                p = os.path.join(self.root, "sequences", s)

                I0 = read_img(os.path.join(p, "im1.png"))
                It = read_img(os.path.join(p, "im2.png"))
                I1 = read_img(os.path.join(p, "im3.png"))

                I0, It, I1 = random_crop([I0, It, I1], self.crop)

                return {
                    "I0": to_tensor(I0),
                    "I1": to_tensor(I1),
                    "It": to_tensor(It),
                    "t": sample_t(),
                    "domain": 0  # VIMEO
                }
            except Exception:
                idx = random.randint(0, len(self.samples) - 1)

        raise RuntimeError("Repeated read failure in Vimeo dataset")



# ------------------------------
# X4K1000FPS
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
            "t": sample_t(),
            "domain": 1  # X4K
        }


# ------------------------------
# WMS (Optional)
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
            "t": sample_t(),
            "domain": 2  # WMS
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


