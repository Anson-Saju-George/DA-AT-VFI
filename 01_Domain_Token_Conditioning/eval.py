import torch
import numpy as np
from torch.utils.data import DataLoader
from dataset import get_dataset
from model import DomainTokenVFI
from pytorch_msssim import ssim

# -------------------------
# CONFIG
# -------------------------

ROOTS = {
    "vimeo": r"D:\main-projects\VFI\vimeo_triplet",
    # "x4k":   r"D:\main-projects\VFI\OpenDataLab___X4K1000FPS\raw\X4K1000FPS",
    "wms":   r"D:\main-projects\WMS_Cleaned",
}

DOMAIN_MAP = {
    "vimeo": 0,
    "wms": 1,
    # "x4k": 2,
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH = 1
CROP = 256

# -------------------------
# METRICS
# -------------------------

def psnr(a, b):
    mse = torch.mean((a - b) ** 2)
    return 10 * torch.log10(1.0 / mse)

# -------------------------
# SAFE CHECKPOINT LOADER
# -------------------------

def load_with_resized_embedding(model, ckpt):
    model_dict = model.state_dict()

    for k, v in ckpt.items():
        if k not in model_dict:
            continue

        # Special case: domain embedding table
        if "embed.weight" in k:
            old_n, dim = v.shape
            new_n, _ = model_dict[k].shape

            print(f"Resizing embedding: {old_n} → {new_n}")

            new_weight = model_dict[k].clone()
            new_weight[:old_n] = v  # copy learned domains
            model_dict[k] = new_weight
        else:
            if model_dict[k].shape == v.shape:
                model_dict[k] = v

    model.load_state_dict(model_dict, strict=True)

# -------------------------
# EVAL
# -------------------------

@torch.no_grad()
def eval_dataset(name, root, model):
    ds = get_dataset(
        name=name,
        root=root,
        split="test",
        crop=CROP,
        domain_id=DOMAIN_MAP[name],
    )

    dl = DataLoader(ds, batch_size=BATCH, shuffle=False, num_workers=4)

    psnrs = []
    ssims = []

    for i, batch in enumerate(dl):
        I0 = batch["I0"].to(DEVICE)
        I1 = batch["I1"].to(DEVICE)
        It = batch["It"].to(DEVICE)
        t  = batch["t"].to(DEVICE)
        dom = batch["domain"].long().to(DEVICE)

        pred = model(I0, I1, t, dom)

        psnrs.append(psnr(pred, It).item())
        ssims.append(ssim(pred, It, data_range=1.0).item())

        if i % 50 == 0:
            print(
                f"[{i:05d}/{len(dl)}] "
                f"PSNR {np.mean(psnrs):.2f} | "
                f"SSIM {np.mean(ssims):.4f}"
            )

    return float(np.mean(psnrs)), float(np.mean(ssims))

# -------------------------
# MAIN
# -------------------------

def main():
    print("Unified VFI Evaluation")

    num_domains = len(DOMAIN_MAP)
    model = DomainTokenVFI(num_domains=num_domains).to(DEVICE)

    ckpt = torch.load("model_domain_token.pt", map_location=DEVICE)

    load_with_resized_embedding(model, ckpt)
    model.eval()

    for name in ROOTS:
        print("\n================", name.upper(), "================")
        p, s = eval_dataset(name, ROOTS[name], model)
        print(f"{name.upper()} | PSNR {p:.3f} | SSIM {s:.4f}")

if __name__ == "__main__":
    main()
