import torch
import random
import time
from torch.utils.data import DataLoader
from dataset import get_dataset
from model import MotionAwareVFI
from pytorch_msssim import ssim
import torch.optim as optim

ROOTS = {
    "vimeo": r"D:\main-projects\VFI\vimeo_triplet",
    "x4k": r"D:\main-projects\VFI\OpenDataLab___X4K1000FPS\raw\X4K1000FPS"
}

DEVICE = "cuda"
BATCH = 16
CROP = 256
STEPS = 50000
LOG_EVERY = 500
LR = 5e-5
WARMUP = 5000

def train():
    print("Building datasets...")
    loaders = {}
    iters = {}

    for name in ROOTS:
        ds = get_dataset(name, ROOTS[name], CROP)
        dl = DataLoader(
            ds,
            batch_size=BATCH,
            shuffle=True,
            num_workers=8,
            pin_memory=True,
            drop_last=True
        )
        loaders[name] = dl
        iters[name] = iter(dl)
        print(f"{name.upper()} samples: {len(ds)}")

    model = MotionAwareVFI().to(DEVICE)
    opt = optim.AdamW(model.parameters(), lr=LR)

    print("\nTraining started")
    t0 = time.time()

    def next_batch(k):
        nonlocal iters
        try:
            return next(iters[k])
        except StopIteration:
            iters[k] = iter(loaders[k])
            return next(iters[k])

    for step in range(STEPS):
        domain = random.choice(list(ROOTS.keys()))
        batch = next_batch(domain)

        I0 = batch["I0"].to(DEVICE, non_blocking=True)
        I1 = batch["I1"].to(DEVICE, non_blocking=True)
        It = batch["It"].to(DEVICE, non_blocking=True)
        t = batch["t"].to(DEVICE, non_blocking=True)

        pred = model(I0, I1, t)

        l1 = (pred - It).abs().mean()

        if step < WARMUP:
            loss = l1
        else:
            s = 1 - ssim(pred, It, data_range=1.0, size_average=True)
            loss = l1 + 0.5 * s

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % LOG_EVERY == 0:
            it_s = (step + 1) / (time.time() - t0)
            print(f"[{step:05d}/{STEPS:05d}] {domain.upper()} | loss {loss.item():.4f} | {it_s:.2f} it/s")

    torch.save(model.state_dict(), "model_final.pt")
    print("Saved model_final.pt")

if __name__ == "__main__":
    train()
