import os
import cv2
import numpy as np
import multiprocessing as mp
import time
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# ============================================================
# CONFIG
# ============================================================
ROOTS = {
    "vimeo": r"D:\main-projects\VFI\vimeo_triplet\sequences",
    "x4k":   r"D:\main-projects\VFI\OpenDataLab___X4K1000FPS\raw\X4K1000FPS",
    "wms":   r"D:\main-projects\WMS_Cleaned",
}

LOG_FILE = "dataset_sanity_brutal_parallel.log"
MIN_SIZE = 256
EXTS = (".png", ".jpg", ".jpeg")

NUM_PROCESSES = max(1, mp.cpu_count() - 2)
THREADS_PER_PROC = 4

# ============================================================
# SHARED STATE
# ============================================================
PROGRESS = mp.Value("i", 0)
LOG_LOCK = mp.Lock()

# ============================================================
# LOGGING
# ============================================================
def log(msg):
    with LOG_LOCK:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

# ============================================================
# IMAGE UTILS
# ============================================================
def load_img(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype(np.float32) / 255.0

def is_flat(img, eps=1e-6):
    return (img.max() - img.min()) < eps

def check_image(path):
    try:
        img = load_img(path)
        if img is None:
            return f"BAD_IMG | {path} | DECODE_FAIL"

        h, w, _ = img.shape

        if h < MIN_SIZE or w < MIN_SIZE:
            return f"BAD_IMG | {path} | TOO_SMALL {h}x{w}"

        if not np.isfinite(img).all():
            return f"BAD_IMG | {path} | NAN_OR_INF"

        mn, mx = float(img.min()), float(img.max())
        if mn < 0.0 or mx > 1.0:
            return f"BAD_IMG | {path} | RANGE_ERROR min={mn:.4f} max={mx:.4f}"

        if is_flat(img):
            return f"BAD_IMG | {path} | FLAT_IMAGE"

    except Exception as e:
        return f"BAD_IMG | {path} | EXCEPTION {str(e)}"

    return None

# ============================================================
# SEQUENCE CHECK
# ============================================================
def check_sequence(seq_path):
    files = sorted([
        f for f in os.listdir(seq_path)
        if f.lower().endswith(EXTS)
    ])

    if len(files) < 3:
        return [f"BAD_SEQ | {seq_path} | LESS_THAN_3_FRAMES"]

    paths = [os.path.join(seq_path, f) for f in files]

    bad_logs = []
    shapes = []

    # Multi-threaded per-sequence image checks
    with ThreadPoolExecutor(max_workers=THREADS_PER_PROC) as pool:
        for result in pool.map(check_image, paths):
            if result:
                bad_logs.append(result)

    # Shape consistency check (first 3 frames)
    imgs = []
    for p in paths[:3]:
        img = load_img(p)
        if img is not None:
            imgs.append(img)
            shapes.append(img.shape[:2])

    if len(set(shapes)) > 1:
        bad_logs.append(f"BAD_SEQ | {seq_path} | SHAPE_MISMATCH {shapes}")

    # Temporal identity checks
    if len(imgs) == 3:
        if np.allclose(imgs[0], imgs[1], atol=1e-4):
            bad_logs.append(f"BAD_SEQ | {seq_path} | I0_EQUALS_I1")
        if np.allclose(imgs[1], imgs[2], atol=1e-4):
            bad_logs.append(f"BAD_SEQ | {seq_path} | I1_EQUALS_I2")
        if np.allclose(imgs[0], imgs[2], atol=1e-4):
            bad_logs.append(f"BAD_SEQ | {seq_path} | I0_EQUALS_I2")

    return bad_logs

# ============================================================
# WORKER
# ============================================================
def worker(seq_paths, progress, total):
    for seq in seq_paths:
        bads = check_sequence(seq)
        for b in bads:
            log(b)
        with progress.get_lock():
            progress.value += 1

# ============================================================
# SCAN DATASET
# ============================================================
def scan_dataset(name, root):
    print(f"\n==================== {name.upper()} ====================")

    seqs = []
    for r, d, f in os.walk(root):
        imgs = [x for x in f if x.lower().endswith(EXTS)]
        if len(imgs) >= 3:
            seqs.append(r)

    total = len(seqs)
    print(f"Found {total} sequences")
    log(f"\n==================== {name.upper()} ====================")
    log(f"Found {total} sequences")

    global PROGRESS
    PROGRESS.value = 0

    chunks = np.array_split(seqs, NUM_PROCESSES)

    procs = []
    for chunk in chunks:
        p = mp.Process(
            target=worker,
            args=(chunk.tolist(), PROGRESS, total)
        )
        p.start()
        procs.append(p)

    with tqdm(total=total, desc=f"{name.upper()} scan", smoothing=0.1) as bar:
        last = 0
        while any(p.is_alive() for p in procs):
            with PROGRESS.get_lock():
                cur = PROGRESS.value
            bar.update(cur - last)
            last = cur
            time.sleep(0.2)

    for p in procs:
        p.join()

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("============================================")
    print("BRUTAL DATASET SANITY SCAN (PARALLEL MODE)")
    print(f"Processes: {NUM_PROCESSES}")
    print(f"Threads per process: {THREADS_PER_PROC}")
    print("============================================")

    # Clear old log
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write("DATASET SANITY LOG\n")

    for name, root in ROOTS.items():
        scan_dataset(name, root)

    print("\nFULL DATASET SCAN COMPLETE.")
    print(f"Log saved to: {LOG_FILE}")
