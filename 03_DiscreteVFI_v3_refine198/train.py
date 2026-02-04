import os
import time
import random
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset_all import get_dataset
from model import DiscreteVFI_v3_refine198

# -----------------------
# CONFIG
# -----------------------
ROOTS = {
    # "vimeo": r"D:\main-projects\VFI\vimeo_triplet",
    # "x4k":   r"D:\main-projects\VFI\OpenDataLab___X4K1000FPS/raw/X4K1000FPS",
    "wms": r"D:\main-projects\WMS_Cleaned",   # enable later when stable
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH = 16
NUM_WORKERS = 8
STEPS = 60000
LOG_EVERY = 1000
SAVE_EVERY = 2000
LR = 5e-5

CROP = 256
CHECKPOINT = "model_latest.pt"

WEIGHTS = {
    # "vimeo": 0.7,
    # "x4k": 0.3,
}

LOSS_WEIGHT = {
    # "vimeo": 1.0,
    # "x4k": 1.0,
}

USE_AMP = False  # keep false until stable

# -----------------------
# GPU STATS
# -----------------------
try:
    from pynvml import (
        nvmlInit,
        nvmlDeviceGetHandleByIndex,
        nvmlDeviceGetMemoryInfo,
        nvmlDeviceGetUtilizationRates
    )

    nvmlInit()
    GPU_HANDLE = nvmlDeviceGetHandleByIndex(0)
except Exception:
    GPU_HANDLE = None


def gpu_stats():
    if GPU_HANDLE is None:
        return "GPU N/A"
    mem = nvmlDeviceGetMemoryInfo(GPU_HANDLE)
    util = nvmlDeviceGetUtilizationRates(GPU_HANDLE)

    used = mem.used / 1024**3
    total = mem.total / 1024**3
    pct = (used / total) * 100

    return f"VRAM {used:.1f}/{total:.1f}GB ({pct:.0f}%) | GPU {util.gpu}%"


# -----------------------
# TRAIN LOOP
# -----------------------
def train():
    torch.backends.cudnn.benchmark = True

    print("=" * 90)
    print("Unified VFI Training — DiscreteVFI_v3_refine198")
    print("Device:", DEVICE.upper())
    print("=" * 90)

    loaders = {}
    iters = {}

    print("\nBuilding datasets...")
    for name in ROOTS:
        ds = get_dataset(name, ROOTS[name], split="train", crop=CROP)
        dl = DataLoader(
            ds,
            batch_size=BATCH,
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            drop_last=True,
            persistent_workers=NUM_WORKERS > 0
        )
        loaders[name] = dl
        iters[name] = iter(dl)
        print(f"  {name.upper():6} samples: {len(ds)}")

    print("\nBuilding model...")
    model = DiscreteVFI_v3_refine198().to(DEVICE)

    opt = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

    scaler = torch.amp.GradScaler("cuda") if USE_AMP and DEVICE == "cuda" else None

    if os.path.exists(CHECKPOINT):
        print("Resuming from", CHECKPOINT)
        ckpt = torch.load(CHECKPOINT, map_location=DEVICE)
        model.load_state_dict(ckpt["model"], strict=False)
        opt.load_state_dict(ckpt["opt"])
        start_step = ckpt["step"]
    else:
        start_step = 0

    def next_batch(name):
        nonlocal iters
        try:
            return next(iters[name])
        except StopIteration:
            iters[name] = iter(loaders[name])
            return next(iters[name])

    print("\nTraining started")
    print(f"Batch={BATCH} | Workers={NUM_WORKERS} | LR={LR}")
    print("Checkpoint:", CHECKPOINT)
    print("-" * 90)

    t_start = time.time()

    for step in range(start_step, STEPS):
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

        opt.zero_grad(set_to_none=True)

        try:
            if scaler:
                with torch.amp.autocast("cuda"):
                    pred = model(I0, I1, t)
                    l1 = (pred - It).abs().mean()
                    loss = LOSS_WEIGHT[domain] * l1
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                pred = model(I0, I1, t)
                l1 = (pred - It).abs().mean()
                loss = LOSS_WEIGHT[domain] * l1
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
        except RuntimeError as e:
            print("⚠️ Runtime error:", str(e)[:120])
            continue

        if not torch.isfinite(loss):
            print(f"⚠️ Step {step}: Non-finite loss. Skipping.")
            continue

        if step % LOG_EVERY == 0:
            elapsed = time.time() - t_start
            it_s = (step + 1 - start_step) / max(elapsed, 1e-6)
            eta = (STEPS - step) / max(it_s, 1e-6) / 60
            stats = gpu_stats()

            print(
                f"[{step:06d}/{STEPS}] {domain.upper():6} | "
                f"loss {loss.item():.4f} | "
                f"{stats} | "
                f"{it_s:.2f} it/s | ETA {eta:.1f} min | batch {BATCH}"
            )

        if step % SAVE_EVERY == 0 and step > start_step:
            torch.save(
                {
                    "model": model.state_dict(),
                    "opt": opt.state_dict(),
                    "step": step
                },
                CHECKPOINT
            )

    print("\nTraining complete.")
    torch.save(
        {
            "model": model.state_dict(),
            "opt": opt.state_dict(),
            "step": STEPS
        },
        CHECKPOINT
    )
    print("Final model saved to", CHECKPOINT)


# -----------------------
# WINDOWS SAFE ENTRY
# -----------------------
if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    train()
