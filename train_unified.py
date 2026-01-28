import os
import sys
import time
import random
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from pytorch_msssim import ssim

from dataset_all import get_dataset
from model import DiscreteVFI

# ============================================================
# LOGGING (CONSOLE + FILE)
# ============================================================
LOG_FILE = "train.log"

class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, msg):
        for s in self.streams:
            s.write(msg)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()

sys.stdout = Tee(sys.stdout, open(LOG_FILE, "a", buffering=1))
sys.stderr = sys.stdout

# ============================================================
# PATHS
# ============================================================
ROOTS = {
    "vimeo": r"D:\main-projects\VFI\vimeo_triplet",
    "x4k":   r"D:\main-projects\VFI\OpenDataLab___X4K1000FPS\raw\X4K1000FPS",
    "wms":   r"D:\main-projects\WMS_Cleaned",
}

CKPT_PATH = "model_latest.pt"

# ============================================================
# CONFIG
# ============================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH = 16
NUM_WORKERS = 10
TOTAL_STEPS = 50000

LOG_EVERY = 500
SAVE_EVERY = 1000

LR = 5e-5
MAX_GRAD_NORM = 1.0
SSIM_CUTOFF = 5000  # L1 warmup

WEIGHTS = {
    "vimeo": 0.4,
    "x4k":   0.3,
    "wms":   0.3
}

LOSS_WEIGHT = {
    "vimeo": 1.0,
    "x4k":   1.0,
    "wms":   1.5
}

# ============================================================
# GPU MONITOR
# ============================================================
try:
    from pynvml import (
        nvmlInit,
        nvmlDeviceGetHandleByIndex,
        nvmlDeviceGetMemoryInfo,
        nvmlDeviceGetUtilizationRates,
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


# ============================================================
# UTILS
# ============================================================
def sanitize(x):
    if not torch.isfinite(x).all():
        return torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0)
    return x


# ============================================================
# TRAIN
# ============================================================
def train():
    torch.backends.cudnn.benchmark = True

    print("=" * 90)
    print("Unified VFI Training")
    print("Device:", DEVICE.upper())
    print("Log file:", LOG_FILE)
    print("=" * 90)

    print("\nBuilding datasets...")
    loaders = {}
    iters = {}

    for name, root in ROOTS.items():
        ds = get_dataset(name, root, split="train")
        dl = DataLoader(
            ds,
            batch_size=BATCH,
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            drop_last=True,
            persistent_workers=True if NUM_WORKERS > 0 else False
        )
        loaders[name] = dl
        iters[name] = iter(dl)
        print(f"  {name.upper()} samples: {len(ds)}")

    print("\nBuilding model...")
    model = DiscreteVFI().to(DEVICE)
    opt = optim.AdamW(model.parameters(), lr=LR)

    use_amp = False
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)



    # ========================================================
    # RESUME
    # ========================================================
    start_step = 0
    if os.path.exists(CKPT_PATH):
        print("\nResuming from", CKPT_PATH)
        ckpt = torch.load(CKPT_PATH, map_location=DEVICE)

        missing, unexpected = model.load_state_dict(
            ckpt.get("model", {}),
            strict=False
        )
        print(f"Checkpoint loaded | Missing: {len(missing)} | Unexpected: {len(unexpected)}")

        if "opt" in ckpt:
            opt.load_state_dict(ckpt["opt"])

        start_step = ckpt.get("step", 0)
        print("Resumed at step", start_step)

    def next_batch(name):
        nonlocal iters
        try:
            return next(iters[name])
        except StopIteration:
            iters[name] = iter(loaders[name])
            return next(iters[name])

    print("\nUnified training started")
    print(f"Batch={BATCH} | Workers={NUM_WORKERS} | LR={LR}")
    print(f"Warmup steps (L1 only): {SSIM_CUTOFF}")
    print(f"Rolling checkpoint: {CKPT_PATH}")
    print("-" * 90)

    t_start = time.time()

    # ========================================================
    # LOOP
    # ========================================================
    for step in range(start_step, TOTAL_STEPS):

        domain = random.choices(
            list(WEIGHTS.keys()),
            weights=list(WEIGHTS.values()),
            k=1
        )[0]

        batch = next_batch(domain)

        I0 = sanitize(batch["I0"].to(DEVICE, non_blocking=True))
        I1 = sanitize(batch["I1"].to(DEVICE, non_blocking=True))
        It = sanitize(batch["It"].to(DEVICE, non_blocking=True))
        t  = sanitize(batch["t"].to(DEVICE, non_blocking=True)).clamp(0.0, 1.0)

        opt.zero_grad(set_to_none=True)

        use_amp_now = use_amp

        try:
            with torch.amp.autocast("cuda", enabled=use_amp_now):
                pred = model(I0, I1, t)

                if not torch.isfinite(pred).all():
                    print(f"⚠️ Step {step}: NaN in prediction. Skipping batch.")
                    opt.zero_grad(set_to_none=True)
                    torch.cuda.empty_cache()
                    continue

                l1 = (pred - It).abs().mean()

                if step < SSIM_CUTOFF:
                    loss = LOSS_WEIGHT[domain] * l1
                else:
                    s = 1 - ssim(pred, It, data_range=1.0, size_average=True)
                    loss = LOSS_WEIGHT[domain] * (l1 + 0.5 * s)

            if not torch.isfinite(loss):
                print(f"⚠️ Step {step}: Non-finite loss. Skipping.")
                continue

            if use_amp_now:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                opt.step()

        except RuntimeError as e:
            print(f"🔥 Step {step}: Runtime error. Skipping batch.\n{e}")
            torch.cuda.empty_cache()
            continue

        # ====================================================
        # LOG
        # ====================================================
        if step % LOG_EVERY == 0:
            elapsed = time.time() - t_start
            it_s = (step + 1 - start_step) / max(elapsed, 1e-6)
            remaining = TOTAL_STEPS - step
            eta_min = (remaining / max(it_s, 1e-6)) / 60.0

            stats = gpu_stats()

            print(
                f"[{step:06d}/{TOTAL_STEPS}] {domain.upper():6} | "
                f"loss {loss.item():.4f} | "
                f"{stats} | "
                f"{it_s:.2f} it/s | "
                f"ETA {eta_min:.1f} min | "
                f"batch {BATCH}"
            )

        # ====================================================
        # SAVE
        # ====================================================
        if step % SAVE_EVERY == 0 and step > start_step:
            torch.save(
                {
                    "step": step,
                    "model": model.state_dict(),
                    "opt": opt.state_dict(),
                },
                CKPT_PATH
            )

    print("\nTraining complete.")
    torch.save(
        {
            "step": TOTAL_STEPS,
            "model": model.state_dict(),
            "opt": opt.state_dict(),
        },
        CKPT_PATH
    )
    print("Final model saved to", CKPT_PATH)


# ============================================================
# WINDOWS SAFE ENTRY
# ============================================================
if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    train()
