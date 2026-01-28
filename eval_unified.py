import os
import math
import torch
import time
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from pytorch_msssim import ssim

from dataset_all import get_dataset
from model import DiscreteVFI

# -----------------------
# CONFIG
# -----------------------
ROOTS = {
    "vimeo": r"D:\main-projects\VFI\vimeo_triplet",
    "x4k": r"D:\main-projects\VFI\OpenDataLab___X4K1000FPS/raw/X4K1000FPS",
    "wms": r"D:\main-projects\WMS_Cleaned",
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH = 1   # Keep 1 for 2K/4K safety
NUM_WORKERS = 4
MODEL_PATH = "model_latest.pt"

OUT_DIR = "eval_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

LOG_FILE = os.path.join(OUT_DIR, "eval.log")

# -----------------------
# UTILS
# -----------------------
def log(msg):
    print(msg)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(msg + "\n")

def psnr(pred, gt):
    mse = torch.mean((pred - gt) ** 2)
    if mse == 0:
        return 100.0
    return 20 * math.log10(1.0 / math.sqrt(mse.item()))

# -----------------------
# EVAL LOOP
# -----------------------
@torch.no_grad()
def eval_dataset(name, root, model):
    log(f"\n================ {name.upper()} =================")

    ds = get_dataset(name, root, split="test")
    dl = DataLoader(
        ds,
        batch_size=BATCH,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    psnr_total = 0.0
    ssim_total = 0.0
    count = 0

    save_dir = os.path.join(OUT_DIR, name)
    os.makedirs(save_dir, exist_ok=True)

    t0 = time.time()

    for i, batch in enumerate(dl):
        I0 = batch["I0"].to(DEVICE, non_blocking=True)
        I1 = batch["I1"].to(DEVICE, non_blocking=True)
        It = batch["It"].to(DEVICE, non_blocking=True)
        t  = batch["t"].to(DEVICE, non_blocking=True)

        pred = model(I0, I1, t)
        pred = pred.clamp(0, 1)

        p = psnr(pred, It)
        s = ssim(pred, It, data_range=1.0, size_average=True).item()

        psnr_total += p
        ssim_total += s
        count += 1

        # Save samples every 50 frames
        if i % 50 == 0:
            grid = torch.cat([I0, pred, It, I1], dim=0)
            save_image(
                grid,
                os.path.join(save_dir, f"{i:05d}.png"),
                nrow=4
            )

        if i % 20 == 0:
            log(f"[{i:05d}/{len(dl)}] PSNR {p:.2f} | SSIM {s:.4f}")

    dt = time.time() - t0
    log(f"\n{name.upper()} FINAL")
    log(f"Samples: {count}")
    log(f"Avg PSNR: {psnr_total / count:.3f}")
    log(f"Avg SSIM: {ssim_total / count:.4f}")
    log(f"Time: {dt/60:.2f} min")

    return psnr_total / count, ssim_total / count

# -----------------------
# MAIN
# -----------------------
def main():
    log("=" * 80)
    log("Unified VFI Evaluation")
    log(f"Device: {DEVICE}")
    log(f"Model: {MODEL_PATH}")
    log("=" * 80)

    model = DiscreteVFI().to(DEVICE)
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
    state = ckpt["model"] if "model" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)

    log(f"Missing keys: {len(missing)}")
    log(f"Unexpected keys: {len(unexpected)}")

    model.eval()

    results = {}

    for name in ROOTS:
        results[name] = eval_dataset(name, ROOTS[name], model)

    log("\n================ OVERALL SUMMARY ================")
    for k, (p, s) in results.items():
        log(f"{k.upper():6} | PSNR {p:.3f} | SSIM {s:.4f}")

    mean_psnr = sum(r[0] for r in results.values()) / len(results)
    mean_ssim = sum(r[1] for r in results.values()) / len(results)

    log(f"\nMEAN | PSNR {mean_psnr:.3f} | SSIM {mean_ssim:.4f}")
    log("Evaluation complete.")

if __name__ == "__main__":
    main()
