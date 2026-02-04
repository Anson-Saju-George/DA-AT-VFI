import cv2
import numpy as np
from pathlib import Path

# ================= CONFIG =================
OUTPUT_DIR = Path("output")
ANALYZER_DIR = Path("analyzer")
EPS = 1e-6
# =========================================

ANALYZER_DIR.mkdir(exist_ok=True)

def load_img(p):
    img = cv2.imread(str(p))
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def save_img(p, img):
    img = np.clip(img, 0, 255).astype(np.uint8)
    cv2.imwrite(str(p), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

# Collect real/pred pairs
real_imgs = sorted(OUTPUT_DIR.glob("*_real.png"))

assert len(real_imgs) >= 2, "Need at least 2 real frames"

print(f"Found {len(real_imgs)} real frames")

for i in range(len(real_imgs) - 1):
    real_path = real_imgs[i]
    pred_path = OUTPUT_DIR / f"{real_path.stem.replace('_real','')}_{real_imgs[i+1].stem.replace('_real','')}_pred.png"

    if not pred_path.exists():
        print(f"Skipping missing {pred_path.name}")
        continue

    real = load_img(real_path).astype(np.float32)
    pred = load_img(pred_path).astype(np.float32)

    # ---------- 1. Absolute difference ----------
    diff = np.abs(real - pred)
    diff_gray = diff.mean(axis=2)
    diff_gray = (diff_gray / diff_gray.max() * 255)

    save_img(ANALYZER_DIR / f"{real_path.stem}_diff.png",
             np.stack([diff_gray]*3, axis=-1))

    # ---------- 2. Hue-based change ----------
    mag = diff_gray / 255.0  # [0,1]

    hue = (1.0 - mag) * 120  # green → red
    sat = np.ones_like(hue) * 255
    val = np.ones_like(hue) * 255

    hsv = np.stack([hue, sat, val], axis=-1).astype(np.uint8)
    hue_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    save_img(ANALYZER_DIR / f"{real_path.stem}_hue.png", hue_rgb)

    # ---------- 3. Overlay ----------
    overlay = (0.6 * real + 0.4 * hue_rgb)
    save_img(ANALYZER_DIR / f"{real_path.stem}_overlay.png", overlay)

    print(f"Analyzed → {real_path.stem}")

print("\nAnalysis complete.")
print(f"Results in: {ANALYZER_DIR.resolve()}")
