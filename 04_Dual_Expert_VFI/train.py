import os, time, random, torch
import torch.optim as optim
from torch.utils.data import DataLoader
import time

from dataset_all import get_dataset
from model import DualExpertVFI

DEVICE = "cuda"
BATCH = 16
WORKERS = 8
CROP = 256
STEPS = 50000
LR = 5e-5

ROOTS = {
    "vimeo": r"D:\main-projects\VFI\vimeo_triplet",
    "wms":   r"D:\main-projects\WMS_Cleaned",
}

start_time = time.time()
time_elapsed = time.time() - start_time

def train():
    print("=" * 80)
    print("04_Dual_Expert_VFI TRAINING")
    print("=" * 80)

    loaders, iters = {}, {}
    for name in ROOTS:
        ds = get_dataset(name, ROOTS[name], "train", CROP)
        dl = DataLoader(
            ds, BATCH, shuffle=True,
            num_workers=WORKERS, pin_memory=True, drop_last=True
        )
        loaders[name] = dl
        iters[name] = iter(dl)
        print(f"{name.upper()} samples: {len(ds)}")

    model = DualExpertVFI().to(DEVICE)
    opt = optim.AdamW(model.parameters(), lr=LR)

    time_elapsed = "0h 0m 0s"
    for step in range(STEPS):
        domain = random.choice(list(ROOTS.keys()))

        try:
            batch = next(iters[domain])
        except StopIteration:
            iters[domain] = iter(loaders[domain])
            batch = next(iters[domain])

        I0 = batch["I0"].to(DEVICE)
        I1 = batch["I1"].to(DEVICE)
        It = batch["It"].to(DEVICE)
        t  = batch["t"].to(DEVICE)

        pred, gate = model(I0, I1, t)
        loss = (pred - It).abs().mean()

        opt.zero_grad()
        loss.backward()
        opt.step()

        # Logging
        if step % 1000 == 0:
            time_elapsed = time.strftime(
                "%Hh %Mm %Ss",
                time.gmtime(time.time() - start_time)
            )
            time_remaining = time.strftime(
                "%Hh %Mm %Ss",
                time.gmtime(
                    (time.time() - start_time) / (step + 1) * (STEPS - step - 1)
                )
            )
            g = gate.mean(0).detach().cpu().tolist()
            print(
                f"Step [{step}/{STEPS}] {domain.upper()} | "
                f"Elapsed: {time_elapsed} | "
                f"Remaining: {time_remaining} | "
                f"loss {loss.item():.4f} | "
                f"gate rigid={g[0]:.2f} deform={g[1]:.2f}"
            )


    torch.save(model.state_dict(), "Dual_ExpertVFI_model.pt")
    print("Saved Dual_ExpertVFI_model.pt")

if __name__ == "__main__":
    train()
