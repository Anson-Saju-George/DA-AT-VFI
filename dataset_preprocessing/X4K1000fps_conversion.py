import os
import shutil
import subprocess

ROOT = r"D:\main-projects\VFI\OpenDataLab___X4K1000FPS\raw\X4K1000FPS"
OUT  = os.path.join(ROOT, "frames_unified")

SPLITS = {
    "train": [
        "encoded_train/encoded_train",
        "encoded_train_frames"
    ],
    "val": [
        "val"
    ],
    "test": [
        "Adobe240fps_test_spilt_XVFI"
    ]
}

IMG_EXT = (".png", ".jpg", ".jpeg")
VID_EXT = (".mp4", ".avi", ".mov", ".mkv")


def ensure(p):
    os.makedirs(p, exist_ok=True)


def extract_video(video, out_dir):
    ensure(out_dir)
    cmd = [
        "ffmpeg",
        "-i", video,
        os.path.join(out_dir, "frame_%06d.png")
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def copy_frames(src, out_dir):
    ensure(out_dir)
    for f in sorted(os.listdir(src)):
        if f.lower().endswith(IMG_EXT):
            shutil.copy2(os.path.join(src, f), os.path.join(out_dir, f))


def process_folder(src_root, split):
    out_root = os.path.join(OUT, split)
    ensure(out_root)

    for root, dirs, files in os.walk(src_root):
        seq_name = os.path.relpath(root, src_root).replace("\\", "_").replace("/", "_")
        if seq_name == ".":
            continue

        imgs = [f for f in files if f.lower().endswith(IMG_EXT)]
        vids = [f for f in files if f.lower().endswith(VID_EXT)]

        if not imgs and not vids:
            continue

        out_seq = os.path.join(out_root, seq_name)
        if os.path.exists(out_seq):
            continue

        if imgs:
            copy_frames(root, out_seq)
        for v in vids:
            extract_video(os.path.join(root, v), out_seq)


def main():
    ensure(OUT)

    for split, sources in SPLITS.items():
        print(f"Processing {split}...")
        for src in sources:
            full = os.path.join(ROOT, src)
            if os.path.exists(full):
                process_folder(full, split)


if __name__ == "__main__":
    main()
