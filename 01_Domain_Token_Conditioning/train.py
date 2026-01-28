import os
import time
import random
import torch
from torch.utils.data import DataLoader
from dataset import get_dataset
from model import DomainTokenVFI
import torch.optim as optim
from pytorch_msssim import ssim

ROOTS = {
    "vimeo": r"D:\\main-projects\\VFI\\vimeo_triplet",
    "x4k":   r"D:\\main-projects\\VFI\\OpenDataLab___X4K1000FPS\\raw\\X4K1000FPS",
    # "wms": r"D:\\main-projects\\WMS_Cleaned",
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH = 32
WORKERS = 10
STEPS = 50000
LOG_EVERY = 1000
CROP = 256
LR = 5e-5

WEIGHTS = {
    "vimeo": 0.5,
    "x4k": 0.5,
}


def train():
    loaders = {}
    iters = {}

    print("Building datasets...")
    for name in ROOTS:
        ds = get_dataset(name, ROOTS[name], "train", CROP)
        print(f"{name.upper()} samples: {len(ds)}")

        dl = DataLoader(
            ds,
            batch_size=BATCH,
            shuffle=True,
            num_workers=WORKERS,
            pin_memory=True,
            drop_last=True,
            persistent_workers=True
        )
        loaders[name] = dl
        iters[name] = iter(dl)

    print("Building model...")
    model = DomainTokenVFI(num_domains=3).to(DEVICE)
    opt = optim.AdamW(model.parameters(), lr=LR)

    scaler = torch.amp.GradScaler("cuda")

    def next_batch(name):
        nonlocal iters
        try:
            return next(iters[name])
        except StopIteration:
            iters[name] = iter(loaders[name])
            return next(iters[name])

    print("\nUnified training started")
    t_start = time.time()

    for step in range(STEPS):
        domain = random.choices(
            list(WEIGHTS.keys()),
            weights=list(WEIGHTS.values()),
            k=1
        )[0]

        batch = next_batch(domain)

        I0 = batch["I0"].to(DEVICE, non_blocking=True)
        I1 = batch["I1"].to(DEVICE, non_blocking=True)
        It = batch["It"].to(DEVICE, non_blocking=True)
        t  = batch["t"].to(DEVICE, non_blocking=True)
        dom = batch["domain"].to(DEVICE, non_blocking=True)

        opt.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda"):
            pred = model(I0, I1, t, dom)
            l1 = (pred - It).abs().mean()
            s  = 1 - ssim(pred, It, data_range=1.0, size_average=True)
            loss = l1 + 0.5 * s

        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

        if step % LOG_EVERY == 0:
            elapsed = time.time() - t_start
            it_s = (step + 1) / max(elapsed, 1e-6)
            print(
                f"[{step:06d}/{STEPS}] {domain.upper():6} | "
                f"loss {loss.item():.4f} | "
                f"{it_s:.2f} it/s"
            )

    torch.save(model.state_dict(), "model_domain_token.pt")
    print("Training complete. Saved model_domain_token.pt")


if __name__ == "__main__":
    train()
