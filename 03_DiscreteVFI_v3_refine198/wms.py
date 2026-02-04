import os
import time
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset_all import get_dataset
from model import DiscreteVFI_v3_refine198

# -----------------------
# CONFIG
# -----------------------
ROOT = r"D:\main-projects\WMS_Cleaned"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH = 16
NUM_WORKERS = 8
STEPS = 60000
LOG_EVERY = 500
SAVE_EVERY = 2000
LR = 3e-5

CROP = 256
CHECKPOINT_IN = "model_latest.pt"
CHECKPOINT_OUT = "model_wms_finetuned.pt"

USE_AMP = False

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
    print("WMS DOMAIN FINE-TUNING — DiscreteVFI_v3_refine198")
    print("Device:", DEVICE.upper())
    print("Steps:", STEPS)
    print("=" * 90)

    print("\nBuilding WMS dataset...")
    ds = get_dataset("wms", ROOT, split="train", crop=CROP)

    dl = DataLoader(
        ds,
        batch_size=BATCH,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
        persistent_workers=NUM_WORKERS > 0
    )

    it = iter(dl)
    print(f"WMS samples: {len(ds)}")

    print("\nBuilding model...")
    model = DiscreteVFI_v3_refine198().to(DEVICE)

    # LOAD MODEL ONLY — DO NOT LOAD OPTIMIZER
    if os.path.exists(CHECKPOINT_IN):
        print("Loading base model from:", CHECKPOINT_IN)
        ckpt = torch.load(CHECKPOINT_IN, map_location=DEVICE)

        if "model" in ckpt:
            model.load_state_dict(ckpt["model"], strict=False)
        else:
            model.load_state_dict(ckpt, strict=False)
    else:
        print("⚠️ No checkpoint found — training from scratch")

    # Fresh optimizer for domain adaptation
    opt = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

    print("\nTraining started")
    print(f"Batch={BATCH} | Workers={NUM_WORKERS} | LR={LR}")
    print("Output checkpoint:", CHECKPOINT_OUT)
    print("-" * 90)

    t_start = time.time()

    for step in range(1, STEPS + 1):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(dl)
            batch = next(it)

        I0 = batch["I0"].to(DEVICE, non_blocking=True)
        I1 = batch["I1"].to(DEVICE, non_blocking=True)
        It = batch["It"].to(DEVICE, non_blocking=True)
        t  = batch["t"].to(DEVICE, non_blocking=True)

        opt.zero_grad(set_to_none=True)

        try:
            pred = model(I0, I1, t)
            l1 = (pred - It).abs().mean()
            loss = l1

            if not torch.isfinite(loss):
                print(f"⚠️ Step {step}: Non-finite loss — skipping")
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        except RuntimeError as e:
            print("⚠️ Runtime error:", str(e)[:120])
            continue

        if step % LOG_EVERY == 0:
            elapsed = time.time() - t_start
            it_s = step / max(elapsed, 1e-6)
            eta = (STEPS - step) / max(it_s, 1e-6) / 60
            stats = gpu_stats()

            print(
                f"[{step:06d}/{STEPS}] WMS | "
                f"loss {loss.item():.4f} | "
                f"{stats} | "
                f"{it_s:.2f} it/s | ETA {eta:.1f} min"
            )

        if step % SAVE_EVERY == 0:
            torch.save(
                {
                    "model": model.state_dict(),
                    "step": step
                },
                CHECKPOINT_OUT
            )
            print(f"💾 Saved checkpoint at step {step}")

    print("\nTraining complete.")
    torch.save(
        {
            "model": model.state_dict(),
            "step": STEPS
        },
        CHECKPOINT_OUT
    )
    print("Final model saved to", CHECKPOINT_OUT)


# -----------------------
# WINDOWS SAFE ENTRY
# -----------------------
if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    train()
