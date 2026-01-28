from torch.utils.data import DataLoader
from dataset_all import get_dataset
import torch

DATASETS = [
    ("vimeo", r"D:\main-projects\VFI\vimeo_triplet"),
    ("x4k", r"D:\main-projects\VFI\OpenDataLab___X4K1000FPS/raw/X4K1000FPS"),
    ("wms", r"D:\main-projects\WMS_Cleaned"),
]

def run(name, root):
    print(f"\n--- {name.upper()} ---")
    ds = get_dataset(name, root, split="train")
    dl = DataLoader(ds, batch_size=4, shuffle=True)

    ts = []
    for i, batch in enumerate(dl):
        print("I0:", batch["I0"].shape, "t:", batch["t"].T)
        ts.append(batch["t"].flatten())
        if i == 5:
            break

    ts = torch.cat(ts)
    print("t stats -> min:", ts.min().item(), "max:", ts.max().item())
    print("unique t values:", sorted(set(ts.tolist())))

if __name__ == "__main__":
    for name, root in DATASETS:
        run(name, root)
