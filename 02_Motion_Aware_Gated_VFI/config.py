import os

ROOTS = {
    "vimeo": r"D:\main-projects\VFI\vimeo_triplet",
    "x4k":   r"D:\main-projects\VFI\OpenDataLab___X4K1000FPS\raw\X4K1000FPS",
    "wms":   r"D:\main-projects\WMS_Cleaned"
}

DEVICE = "cuda"
CROP = 256
BATCH = 16
NUM_WORKERS = 10

STEPS = 50000
LR = 5e-5
LOG_EVERY = 1000
SAVE_EVERY = 5000

USE_AMP = False   # You already proved AMP destabilizes this
CHECKPOINT = "model_latest.pt"

TSET = [0.25, 0.5, 0.75]
