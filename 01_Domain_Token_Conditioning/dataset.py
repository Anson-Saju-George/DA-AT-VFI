import torch
from torch.utils.data import Dataset
from pathlib import Path
import random

class VFIDataset(Dataset):
    def __init__(self, name, root, split, crop, domain_id):
        self.name = name
        self.root = Path(root)
        self.split = split
        self.crop = crop
        self.domain_id = int(domain_id)

        # TODO: replace this with your real file indexing logic
        self.samples = list(self.root.rglob("*.png"))

        if not self.samples:
            raise RuntimeError(f"No samples found in {root}")

    def __len__(self):
        return len(self.samples)

    def _load_image(self, path):
        # TODO: replace with real loader (PIL / cv2 / torchvision)
        # Placeholder: random tensor
        return torch.rand(3, self.crop, self.crop)

    def __getitem__(self, idx):
        # Dummy triplet logic (replace with real frame sampling)
        I0 = self._load_image(self.samples[idx])
        I1 = self._load_image(self.samples[idx])
        It = self._load_image(self.samples[idx])

        # Temporal position
        t = torch.tensor([random.random()], dtype=torch.float32)

        return {
            "I0": I0,
            "I1": I1,
            "It": It,
            "t": t,
            # CONTRACT: domain is ALWAYS int64 scalar
            "domain": torch.tensor(self.domain_id, dtype=torch.long),
        }


def get_dataset(name, root, split, crop, domain_id):
    return VFIDataset(name, root, split, crop, domain_id)
