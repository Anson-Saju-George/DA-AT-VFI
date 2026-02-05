import torch
from torch.utils.data import DataLoader
from dataset_all import get_dataset
from model import DualExpertVFI

DEVICE = "cuda"
CROP = 256

ROOTS = {
    "vimeo": r"D:\main-projects\VFI\vimeo_triplet",
    "wms":   r"D:\main-projects\WMS_Cleaned",
}

model = DualExpertVFI().to(DEVICE)
model.load_state_dict(torch.load("model.pt", map_location=DEVICE))
model.eval()

with torch.no_grad():
    for name in ROOTS:
        ds = get_dataset(name, ROOTS[name], "test", CROP)
        dl = DataLoader(ds, batch_size=1)

        loss = 0
        for b in dl:
            I0 = b["I0"].to(DEVICE)
            I1 = b["I1"].to(DEVICE)
            It = b["It"].to(DEVICE)
            t  = b["t"].to(DEVICE)

            pred, _ = model(I0, I1, t)
            loss += (pred - It).abs().mean().item()

        print(f"{name.upper()} L1: {loss / len(dl):.4f}")
