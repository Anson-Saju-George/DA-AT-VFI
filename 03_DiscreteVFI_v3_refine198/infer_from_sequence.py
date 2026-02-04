import torch
from pathlib import Path
from PIL import Image
import torchvision.transforms as T

# ================= CONFIG =================
INPUT_DIR = Path("sample_images")
OUTPUT_DIR = Path("output")
CKPT_PATH = "model_wms_finetuned.pt"

T_VAL = 0.5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# =========================================

from model import DiscreteVFI_v3_refine198

OUTPUT_DIR.mkdir(exist_ok=True)

to_tensor = T.ToTensor()
to_pil = T.ToPILImage()

# ---------- Load model ----------
print("Loading model...")
model = DiscreteVFI_v3_refine198().to(DEVICE)

ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
if isinstance(ckpt, dict) and "model" in ckpt:
    model.load_state_dict(ckpt["model"], strict=True)
else:
    model.load_state_dict(ckpt, strict=True)

model.eval()
print("Model loaded successfully.")

# ---------- Load frames (png + jpg) ----------
frames = sorted(
    list(INPUT_DIR.glob("*.png")) +
    list(INPUT_DIR.glob("*.jpg")) +
    list(INPUT_DIR.glob("*.jpeg"))
)

assert len(frames) >= 2, "Need at least 2 images"

print(f"Found {len(frames)} frames")

# ---------- Inference + Interleaved Save ----------
with torch.no_grad():
    for i in range(len(frames) - 1):
        f0_path = frames[i]
        f1_path = frames[i + 1]

        # Save REAL frame
        real_img = Image.open(f0_path).convert("RGB")
        real_out = OUTPUT_DIR / f"{f0_path.stem}_real.png"
        real_img.save(real_out)

        # Prepare tensors
        I0 = to_tensor(real_img).unsqueeze(0).to(DEVICE)
        I1 = to_tensor(Image.open(f1_path).convert("RGB")).unsqueeze(0).to(DEVICE)
        t  = torch.tensor([[T_VAL]], device=DEVICE)

        # Predict
        pred = model(I0, I1, t)
        pred = pred.clamp(0, 1)

        # Save PRED frame
        pred_name = f"{f0_path.stem}_{f1_path.stem}_pred.png"
        pred_path = OUTPUT_DIR / pred_name
        to_pil(pred.squeeze(0).cpu()).save(pred_path)

        print(f"Saved → {real_out.name} + {pred_name}")

# Save last REAL frame
last_real = Image.open(frames[-1]).convert("RGB")
last_out = OUTPUT_DIR / f"{frames[-1].stem}_real.png"
last_real.save(last_out)
print(f"Saved → {last_out.name}")

print("\nInference complete.")
print(f"Demo-ready output in: {OUTPUT_DIR.resolve()}")
