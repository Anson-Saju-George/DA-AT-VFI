import torch
import time
import math
from dataset_all import get_dataset
from model import DiscreteVFI_v3_refine198

ROOT = r"D:\main-projects\VFI\vimeo_triplet"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def psnr(pred, gt):
    mse = torch.mean((pred - gt) ** 2)
    if mse < 1e-10:
        return 100
    return 10 * torch.log10(1.0 / mse)

def main():
    print("=== OVERFIT TEST ===")
    print("Device:", DEVICE)

    ds = get_dataset("vimeo", ROOT, split="train", crop=256)

    sample = ds[0]
    I0 = sample["I0"].unsqueeze(0).to(DEVICE)
    I1 = sample["I1"].unsqueeze(0).to(DEVICE)
    It = sample["It"].unsqueeze(0).to(DEVICE)
    t  = sample["t"].unsqueeze(0).to(DEVICE)

    model = DiscreteVFI_v3_refine198().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)

    print("Training on ONE sample. This should overfit.")

    for step in range(2000):
        opt.zero_grad()
        pred = model(I0, I1, t)
        loss = (pred - It).abs().mean()
        loss.backward()
        opt.step()

        if step % 50 == 0:
            with torch.no_grad():
                p = psnr(pred, It).item()
            print(f"[{step:04d}] loss {loss.item():.6f} | PSNR {p:.2f} dB")

    print("\nPASS CONDITION:")
    print("PSNR >= 40 dB = model + dataset + loss are VALID")
    print("PSNR < 40 dB = SOMETHING IS BROKEN")

if __name__ == "__main__":
    main()
