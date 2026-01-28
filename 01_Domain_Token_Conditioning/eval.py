import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from dataset import get_dataset
from model import DomainTokenVFI
from pytorch_msssim import ssim

ROOTS = {
    "vimeo": r"D:\\main-projects\\VFI\\vimeo_triplet",
    "x4k":   r"D:\\main-projects\\VFI\\OpenDataLab___X4K1000FPS\\raw\\X4K1000FPS",
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH = 1
CROP = 256


def psnr(a, b):
    mse = torch.mean((a - b) ** 2)
    return 10 * torch.log10(1.0 / mse)


@torch.no_grad()
def eval_dataset(name, root, model):
    ds = get_dataset(name, root, "test", CROP)
    dl = DataLoader(ds, batch_size=BATCH, shuffle=False, num_workers=4)

    psnrs = []
    ssims = []

    for i, batch in enumerate(dl):
        I0 = batch["I0"].to(DEVICE)
        I1 = batch["I1"].to(DEVICE)
        It = batch["It"].to(DEVICE)
        t  = batch["t"].to(DEVICE)
        dom = batch["domain"].to(DEVICE)

        pred = model(I0, I1, t, dom)

        psnrs.append(psnr(pred, It).item())
        ssims.append(ssim(pred, It, data_range=1.0).item())

        if i % 50 == 0:
            print(f"[{i:05d}/{len(dl)}] PSNR {np.mean(psnrs):.2f} | SSIM {np.mean(ssims):.4f}")

    return float(np.mean(psnrs)), float(np.mean(ssims))



def main():
    print("Unified VFI Evaluation")
    model = DomainTokenVFI(num_domains=3).to(DEVICE)

    ckpt = torch.load("model_domain_token.pt", map_location=DEVICE)
    model.load_state_dict(ckpt, strict=False)
    model.eval()

    for name in ROOTS:
        print("\n================", name.upper(), "================")
        p, s = eval_dataset(name, ROOTS[name], model)
        print(f"{name.upper()} | PSNR {p:.3f} | SSIM {s:.4f}")


if __name__ == "__main__":
    main()
