import os
import math
import torch
from torch.utils.data import DataLoader
from pytorch_msssim import ssim

from dataset_all import get_dataset
from model import DiscreteVFI_v3_refine198

# -----------------------
# CONFIG
# -----------------------
ROOTS = {
    # "vimeo": r"D:\main-projects\VFI\vimeo_triplet",
    # "x4k":   r"D:\main-projects\VFI\OpenDataLab___X4K1000FPS/raw/X4K1000FPS",
    "wms": r"D:\main-projects\WMS_Cleaned",
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "model_wms_finetuned.pt"
BATCH = 1
NUM_WORKERS = 14
CROP = 256

# -----------------------
# METRICS
# -----------------------
def psnr(pred, gt):
    mse = torch.mean((pred - gt) ** 2)
    if mse < 1e-10:
        return 100.0
    return 10 * torch.log10(1.0 / mse)


# -----------------------
# EVAL
# -----------------------
@torch.no_grad()
def eval_dataset(name, root, model):
    ds = get_dataset(name, root, split="test", crop=CROP)
    dl = DataLoader(
        ds,
        batch_size=BATCH,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    psnr_sum = 0.0
    ssim_sum = 0.0
    count = 0
    bad = 0

    print(f"\n================ {name.upper()} =================")

    for i, batch in enumerate(dl):
        try:
            I0 = batch["I0"].to(DEVICE)
            I1 = batch["I1"].to(DEVICE)
            It = batch["It"].to(DEVICE)
            t  = batch["t"].to(DEVICE)

            pred = model(I0, I1, t)

            p = psnr(pred, It).item()
            s = ssim(pred, It, data_range=1.0, size_average=True).item()

            psnr_sum += p
            ssim_sum += s
            count += 1

            if i % 1000 == 0:
                print(f"[{i:05d}/{len(dl)}] PSNR {p:.2f} | SSIM {s:.4f}")

        except Exception as e:
            bad += 1
            print("⚠️ Bad sample:", str(e)[:120])

    return {
        "psnr": psnr_sum / max(count, 1),
        "ssim": ssim_sum / max(count, 1),
        "bad": bad
    }


def main():
    print("=" * 80)
    print("Unified VFI Evaluation — DiscreteVFI_v3_refine198")
    print("Device:", DEVICE)
    print("Model:", MODEL_PATH)
    print("=" * 80)

    model = DiscreteVFI_v3_refine198().to(DEVICE)
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE)

    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)

    print("Missing keys:", len(missing))
    print("Unexpected keys:", len(unexpected))

    model.eval()

    results = {}

    for name in ROOTS:
        results[name] = eval_dataset(name, ROOTS[name], model)

    print("\n================ OVERALL SUMMARY ================")
    psnr_mean = 0
    ssim_mean = 0
    n = 0

    for k, v in results.items():
        print(f"{k.upper():6} | PSNR {v['psnr']:.3f} | SSIM {v['ssim']:.4f} | Bad {v['bad']}")
        psnr_mean += v["psnr"]
        ssim_mean += v["ssim"]
        n += 1

    print(f"\nMEAN | PSNR {psnr_mean/n:.3f} | SSIM {ssim_mean/n:.4f}")
    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()
