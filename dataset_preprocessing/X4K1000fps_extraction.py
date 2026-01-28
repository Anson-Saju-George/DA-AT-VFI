import zipfile
import tarfile
from pathlib import Path
import shutil


ARCHIVE_EXTENSIONS = [".zip", ".tar.gz", ".tgz"]


def is_archive(path: Path):
    name = path.name.lower()
    return any(name.endswith(ext) for ext in ARCHIVE_EXTENSIONS)


def safe_extract_tar(tar_path, extract_to):
    with tarfile.open(tar_path, "r:*") as tar:
        for member in tar.getmembers():
            member_path = extract_to / member.name
            if member_path.exists():
                raise FileExistsError(f"File already exists: {member_path}")
        tar.extractall(path=extract_to)


def safe_extract_zip(zip_path, extract_to):
    with zipfile.ZipFile(zip_path, "r") as zipf:
        for member in zipf.namelist():
            target = extract_to / member
            if target.exists():
                raise FileExistsError(f"File already exists: {target}")
        zipf.extractall(path=extract_to)


def extract_archive(archive_path: Path):
    extract_dir = archive_path.parent / archive_path.stem

    # Handle .tar.gz properly
    if archive_path.name.lower().endswith(".tar.gz"):
        extract_dir = archive_path.parent / archive_path.name[:-7]
    elif archive_path.name.lower().endswith(".tgz"):
        extract_dir = archive_path.parent / archive_path.name[:-4]

    extract_dir.mkdir(exist_ok=True)

    print(f"📦 Extracting: {archive_path}")
    print(f"📁 Into:       {extract_dir}")

    try:
        if archive_path.suffix == ".zip":
            safe_extract_zip(archive_path, extract_dir)
        else:
            safe_extract_tar(archive_path, extract_dir)

    except Exception as e:
        print(f"❌ FAILED: {archive_path}")
        print(f"   Reason: {e}")
        print("   Leaving archive untouched.")
        return

    print(f"✅ Success: {archive_path}")
    print(f"🗑️  Deleting archive...")
    archive_path.unlink()


def recursive_extract(root_dir):
    root = Path(root_dir).resolve()

    if not root.exists():
        raise ValueError(f"Path does not exist: {root}")

    print(f"🔍 Scanning: {root}")

    # Keep looping until no archives remain
    while True:
        archives = [p for p in root.rglob("*") if p.is_file() and is_archive(p)]

        if not archives:
            print("🏁 No more archives found.")
            break

        for archive in archives:
            extract_archive(archive)


if __name__ == "__main__":
    DATASET_PATH = r"D:\main-projects\VFI\OpenDataLab___X4K1000FPS"   # LOCAL PATH TO THE DATASET

    recursive_extract(DATASET_PATH)
    print("🎉 All done!")