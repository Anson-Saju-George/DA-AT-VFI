import torch
from torch.utils.data import DataLoader
from dataset import get_dataset
from model import MotionAwareVFI
import math

ROOTS = {
    "vimeo": r"D:\main-projects\VFI\vimeo_triplet",
    # "x4k": r"D:\main-projects\VFI\OpenDataLab___X4K1000FPS\raw\X4K1000FPS"
}

DEVICE = "cuda"
BATCH = 1
CROP = 256

def psnr(a, b):
    mse = ((a - b) ** 2).mean().item()
    if mse == 0:
        return 99
    return 20 * math.log10(1.0 / math.sqrt(mse))

@torch.no_grad()
def eval_dataset(name, root, model):
    ds = get_dataset(name, root, CROP)
    dl = DataLoader(ds, batch_size=1, shuffle=False)

    p_total = 0
    count = 0

    for i, batch in enumerate(dl):
        I0 = batch["I0"].to(DEVICE)
        I1 = batch["I1"].to(DEVICE)
        It = batch["It"].to(DEVICE)
        t = batch["t"].to(DEVICE)

        pred = model(I0, I1, t)
        p_total += psnr(pred, It)
        count += 1

        if i % 1000 == 0:
            print(f"[{i}/{len(dl)}] | PSNR {p_total / count:.2f}")

    return p_total / count

def main():
    model = MotionAwareVFI().to(DEVICE)
    model.load_state_dict(torch.load("model_final.pt"))
    model.eval()

    for name in ROOTS:
        p = eval_dataset(name, ROOTS[name], model)
        print(f"{name.upper()} | PSNR {p:.2f}")

if __name__ == "__main__":
    main()
